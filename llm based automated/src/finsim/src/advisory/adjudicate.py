"""Adjudication: which advice won, and was it compliant?

Three independent channels, so their (dis)agreement is itself measurable:

  1. Community endorsement — net comment votes plus reply engagement on
     each advisor's top-level answer to the question post.
  2. LLM judge (optional) — scores each advice against the asking client's
     profile on suitability, risk alignment, completeness, and compliance.
     Uses any OpenAI-compatible endpoint (DeepSeek works: set
     DEEPSEEK_API_KEY). NB: judge scores are strategy-dependent — rubric
     and scale choices materially affect rankings, so treat the judge as
     one noisy rater, not ground truth.
  3. Client verdict — the asking client's own INTERVIEW answer at the end
     of the run (read from the trace table when available).

Plus a compliance scan: flags advice mentioning prohibited instruments
(crypto, options, leverage, ...) and surfaces non-whitelisted tickers
for human review.
"""

import json
import os
import re
import sqlite3

import pandas as pd

PROHIBITED = [
    "crypto", "bitcoin", "btc", "ethereum", "nft", "meme stock",
    "options", "call option", "put option", "covered call", "leverage",
    "leveraged", "margin", "futures", "forex", "day trad", "penny stock",
    "short sell", "shorting", "0dte",
]

# Vanguard-style whitelist for ticker review (extend per market: CA/US/UK)
TICKER_WHITELIST = {
    "VEQT", "VBAL", "VGRO", "VCNS", "VCIP", "VAB", "VSB", "VFV", "VUN",
    "VXC", "VCN", "VDY", "VRE", "VUT",                      # Vanguard CA
    "VTI", "VOO", "VXUS", "BND", "BNDX", "VT", "VTSAX",     # Vanguard US
    "VWRL", "VWRP", "VGOV", "VUKE", "VEVE", "LS60", "LS80",  # Vanguard UK
    "TFSA", "RRSP", "RESP", "GIC", "ETF", "TDF", "REIT",     # not tickers
}


def _connect(db):
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    return conn


def _uname_expr():
    return "COALESCE(u.user_name, u.name)"


def find_question_post(db_path: str) -> dict:
    """The case's opening post: earliest post by a cli_* account."""
    conn = _connect(db_path)
    row = conn.execute(f"""
        SELECT p.post_id, p.content, {_uname_expr()} AS uname
        FROM post p JOIN user u ON p.user_id = u.user_id
        WHERE {_uname_expr()} LIKE 'cli_%'
        ORDER BY p.created_at LIMIT 1""").fetchone()
    conn.close()
    if not row:
        raise ValueError("No client question post found")
    return dict(row)


def extract_advice(db_path: str) -> pd.DataFrame:
    """All comments on the question post, with per-comment vote counts and
    reply-level engagement, split into advisor advice vs peer input."""
    q = find_question_post(db_path)
    conn = _connect(db_path)
    df = pd.read_sql_query(f"""
        SELECT c.comment_id, {_uname_expr()} AS uname, c.content,
               c.created_at,
               (SELECT COUNT(*) FROM comment_like cl
                WHERE cl.comment_id = c.comment_id) AS likes,
               (SELECT COUNT(*) FROM comment_dislike cd
                WHERE cd.comment_id = c.comment_id) AS dislikes
        FROM comment c JOIN user u ON c.user_id = u.user_id
        WHERE c.post_id = ?
        ORDER BY c.created_at""", conn, params=(q["post_id"],))
    conn.close()
    df["role"] = df["uname"].str.split("_").str[0].map(
        {"adv": "advisor", "cli": "client"}).fillna("other")
    df["net_votes"] = df["likes"] - df["dislikes"]
    # crude reply engagement: later comments on the thread that name the
    # commenter (personas reference each other by name/username)
    df["mentions"] = df.apply(
        lambda r: df[df.created_at > r.created_at]["content"]
        .str.contains(re.escape(r.uname.split("_")[-2])
                      if len(r.uname.split("_")) > 2 else r.uname,
                      case=False, regex=True).sum(), axis=1)
    df["endorsement"] = df["net_votes"] + 0.5 * df["mentions"]
    return df


def compliance_scan(df: pd.DataFrame) -> pd.DataFrame:
    def scan(text):
        t = text.lower()
        hits = [w for w in PROHIBITED if w in t]
        tickers = set(re.findall(r"\b[A-Z]{2,5}\b", text)) - TICKER_WHITELIST
        return pd.Series({"prohibited": hits,
                          "review_tickers": sorted(tickers)})
    out = df.join(df["content"].apply(scan))
    out["compliant"] = out["prohibited"].str.len() == 0
    return out


def judge_advice(df: pd.DataFrame, client_profile: str, question: str,
                 model: str = "deepseek-chat",
                 base_url: str = "https://api.deepseek.com/v1") -> pd.DataFrame:
    """Optional LLM-judge scoring. Requires DEEPSEEK_API_KEY (or pass an
    OpenAI-compatible base_url + OPENAI_API_KEY). Returns df with
    suitability/risk_alignment/completeness/compliance columns (0-10)."""
    from openai import OpenAI  # lazy import; optional dependency
    key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=key, base_url=base_url)
    rubric = (
        "You are evaluating financial advice given to a specific client. "
        "Client profile: {profile}\nClient question: {question}\n"
        "Advice to evaluate: {advice}\n"
        "Score the advice on four criteria, each an integer 0-10: "
        "suitability (fit to this client's circumstances), risk_alignment "
        "(consistency with stated risk tolerance and horizon), completeness "
        "(addresses the actual question, notes key considerations), "
        "compliance (stays within diversified low-cost funds/bonds/cash; "
        "10 = fully compliant). Respond ONLY with a JSON object with those "
        "four keys and integer values. No preamble, no markdown."
    )
    scores = []
    advisor_rows = df[df.role == "advisor"]
    for _, r in advisor_rows.iterrows():
        resp = client.chat.completions.create(
            model=model, max_tokens=100, temperature=0,
            messages=[{"role": "user", "content": rubric.format(
                profile=client_profile, question=question,
                advice=r.content)}])
        try:
            s = json.loads(resp.choices[0].message.content
                           .replace("```json", "").replace("```", "").strip())
        except (json.JSONDecodeError, AttributeError):
            s = {}
        s["comment_id"] = r.comment_id
        scores.append(s)
    sc = pd.DataFrame(scores)
    if not sc.empty:
        crit = [c for c in ["suitability", "risk_alignment",
                            "completeness", "compliance"] if c in sc]
        sc["judge_score"] = sc[crit].mean(axis=1)
    return df.merge(sc, on="comment_id", how="left")


def client_verdict(db_path: str) -> str:
    """The asking client's INTERVIEW answer from the trace, if present."""
    conn = _connect(db_path)
    rows = conn.execute("""
        SELECT info FROM trace WHERE action LIKE '%interview%'
        ORDER BY created_at DESC LIMIT 1""").fetchall()
    conn.close()
    return rows[0]["info"] if rows else ""


def adjudicate(db_path: str, use_judge: bool = False,
               client_profile: str = "", top_n: int = 10) -> pd.DataFrame:
    q = find_question_post(db_path)
    df = compliance_scan(extract_advice(db_path))
    if use_judge:
        df = judge_advice(df, client_profile, q["content"])
    cols = ["uname", "role", "net_votes", "mentions", "endorsement",
            "compliant", "prohibited", "review_tickers"]
    if "judge_score" in df:
        cols.append("judge_score")
    ranked = df.sort_values("endorsement", ascending=False)

    print(f"Question ({q['uname']}): {q['content'][:120]}...\n")
    print(ranked[cols].head(top_n).to_string(index=False))
    verdict = client_verdict(db_path)
    if verdict:
        print(f"\nClient verdict (interview): {verdict[:600]}")
    nc = df[~df.compliant]
    if len(nc):
        print(f"\nCOMPLIANCE FLAGS: {len(nc)} comment(s)")
        for _, r in nc.iterrows():
            print(f"  {r.uname}: {r.prohibited} :: {r.content[:100]}")
    else:
        print("\nCompliance scan: clean (lexicon-level)")
    return ranked


if __name__ == "__main__":
    import sys
    adjudicate(sys.argv[1] if len(sys.argv) > 1 else
               "results/preretirement_drawdown.db",
               use_judge="--judge" in sys.argv)
