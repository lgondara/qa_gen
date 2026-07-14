"""CSV-driven profiles for the advisory network.

Two populations, defined entirely by CSV so experiment populations are
versionable and editable outside code:

  clients.csv:  name, age, sex, risk_tolerance, horizon_years, portfolio,
                goal, income_band, notes
  advisors.csv: name, education, experience_years, specialty, philosophy,
                style

Personas are constructed from attributes. Advisor personas embed the
instrument-universe constraint (diversified low-cost index funds/ETFs,
investment-grade bonds, cash — no crypto, options, leverage, or individual
stock picking) and the case-research protocol (search the platform for
similar prior cases before answering; adapt and credit borrowed ideas).

Username conventions (consumed by analysis/visualization):
  advisors: adv_{specialty_slug}_{i}
  clients:  cli_{risk_tolerance}_{i}
"""

import csv
import json
import random
from pathlib import Path

ADVISOR_CONSTRAINT = (
    "You recommend exclusively from a Vanguard-style universe: broadly "
    "diversified low-cost index funds and ETFs, investment-grade bond funds, "
    "and cash equivalents. You NEVER recommend cryptocurrency, options, "
    "leverage, margin, futures, forex, individual stock picking, or any "
    "speculative instrument, and you gently correct anyone who does. "
    "Before answering a client's question, use the search function to check "
    "whether a similar case was discussed on this platform before; if you "
    "borrow an idea from a prior thread, adapt it to the current client's "
    "circumstances and mention that a similar case informed your thinking. "
    "Engage critically with other advisors: upvote and endorse sound advice, "
    "and respectfully challenge advice you consider unsuitable for this "
    "specific client, explaining your reasoning. Always tie recommendations "
    "to the client's stated risk tolerance, horizon, and goals."
)

CLIENT_BEHAVIOR = (
    "You are not a financial professional. You may reply to advisors with "
    "follow-up questions, and you may comment on other clients' questions "
    "sharing your own experience, always framed as personal experience "
    "rather than advice. You upvote responses you find clear and relevant "
    "to your situation."
)


def _slug(s: str) -> str:
    return "".join(c for c in s.lower().replace(" ", "") if c.isalnum())


def load_population(clients_csv: str, advisors_csv: str,
                    seed: int = 42) -> list:
    """Read both CSVs -> list of profile dicts in the OASIS reddit schema.
    Advisors come first (after the event source, which the runner inserts),
    so advisor/client agent id ranges are contiguous and predictable."""
    rng = random.Random(seed)
    profiles = []

    with open(advisors_csv) as f:
        for i, row in enumerate(csv.DictReader(f)):
            spec = _slug(row["specialty"])
            persona = (
                f"{row['name']} is a financial advisor with "
                f"{row['experience_years']} years of experience, credentialed "
                f"as {row['education']}, specializing in {row['specialty']}. "
                f"Advisory philosophy: {row['philosophy']} "
                f"Communication style: {row['style']}. "
                + ADVISOR_CONSTRAINT
            )
            profiles.append({
                "realname": row["name"],
                "username": f"adv_{spec}_{i}",
                "bio": f"{row['education']} · {row['experience_years']}y · "
                       f"{row['specialty']}",
                "persona": persona,
                "age": 30 + int(row["experience_years"]),
                "gender": rng.choice(["male", "female"]),
                "mbti": rng.choice(["ISTJ", "ENTJ", "INFJ", "ESTJ"]),
                "country": row.get("country", "Canada"),
                "profession": "Financial Advisor",
                "interested_topics": ["Economics", "Business"],
                "_role": "advisor",
            })

    with open(clients_csv) as f:
        for i, row in enumerate(csv.DictReader(f)):
            persona = (
                f"{row['name']} is a {row['age']}-year-old with "
                f"{row['risk_tolerance']} risk tolerance and an investment "
                f"horizon of {row['horizon_years']} years. Current portfolio: "
                f"{row['portfolio']}. Primary goal: {row['goal']}. Income "
                f"band: {row['income_band']}. {row.get('notes', '')} "
                + CLIENT_BEHAVIOR
            )
            profiles.append({
                "realname": row["name"],
                "username": f"cli_{_slug(row['risk_tolerance'])}_{i}",
                "bio": f"{row['risk_tolerance']} investor · "
                       f"{row['horizon_years']}y horizon",
                "persona": persona,
                "age": int(row["age"]),
                "gender": row.get("sex", rng.choice(["male", "female"])),
                "mbti": rng.choice(["ISFJ", "ENFP", "ISTP", "ESFJ",
                                    "INTP", "ESTP"]),
                "country": row.get("country", "Canada"),
                "profession": row.get("profession", "Client"),
                "interested_topics": ["Business"],
                "_role": "client",
            })

    return profiles


def write_profiles_json(profiles: list, path: str) -> Path:
    """Strip internal fields and write the OASIS-schema JSON."""
    clean = [{k: v for k, v in p.items() if not k.startswith("_")}
             for p in profiles]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(clean, indent=2))
    return p


# ---------------------------------------------------------------------------
# Sample data so the pipeline runs out of the box; replace with real CSVs.
# ---------------------------------------------------------------------------

SAMPLE_ADVISORS = [
    ["name", "education", "experience_years", "specialty", "philosophy", "style"],
    ["Sarah Chen", "CFP, CFA", "18", "retirement planning",
     "Sequence-of-returns risk dominates late-career planning; glide paths and cash buffers over market timing.",
     "measured, uses concrete numbers"],
    ["Marcus Webb", "CFP", "9", "portfolio construction",
     "Total-market index funds, aggressive rebalancing discipline, minimize costs and turnover.",
     "direct, occasionally blunt"],
    ["Priya Sharma", "CFA, MBA", "14", "tax efficiency",
     "Asset location matters as much as asset allocation; registered accounts first, tax-loss harvesting where suitable.",
     "detailed, checklist-oriented"],
    ["David Okonkwo", "CFP, ChFC", "22", "behavioral coaching",
     "The investor's behavior gap costs more than fees; the plan must be one the client can hold through a drawdown.",
     "empathetic, asks questions before answering"],
    ["Elena Rossi", "CFP", "6", "estate and family planning",
     "Younger advisor, favors simple all-in-one asset allocation ETFs and automating contributions.",
     "warm, plain-language"],
]

SAMPLE_CLIENTS = [
    ["name", "age", "sex", "risk_tolerance", "horizon_years", "portfolio",
     "goal", "income_band", "notes"],
    ["Robert Fraser", "61", "male", "conservative", "5",
     "70% balanced index fund, 30% bond index fund",
     "retire at 66 without outliving savings", "90-120k",
     "Anxious about a market drop just before retirement."],
    ["Amara Diallo", "29", "female", "aggressive", "30",
     "100% global equity index ETF",
     "financial independence by 50", "60-90k",
     "Contributes monthly; wonders about adding bonds."],
    ["Ken Tanaka", "45", "male", "moderate", "18",
     "60/40 equity/bond index portfolio",
     "children's education in 8y, retirement at 63", "120-160k",
     "Recently inherited a lump sum equal to a year's salary."],
    ["Grace Liu", "34", "female", "moderate", "25",
     "80/20 in a robo account plus employer pension",
     "buy a home in 4 years", "90-120k",
     "Torn between saving the down payment in cash vs. staying invested."],
    ["Tom Beaulieu", "52", "male", "conservative", "12",
     "50/50 dividend index and short-term bonds",
     "bridge income until pension at 60", "60-90k",
     "Distrusts anything he cannot explain to his spouse."],
    ["Nadia Hassan", "38", "female", "aggressive", "22",
     "90/10 equity-heavy index mix",
     "maximize long-run growth", "160k+",
     "High savings rate; sometimes tempted by market timing."],
    ["Bill Ostrowski", "67", "male", "conservative", "20",
     "40/60 equity/bond, drawing 4% annually",
     "sustainable withdrawals plus a bequest", "90-120k",
     "Recently widowed; consolidating accounts."],
    ["Jasmine Park", "26", "female", "moderate", "35",
     "TDF 2065 in workplace plan only",
     "start investing outside the workplace plan", "40-60k",
     "First-generation investor; low confidence, high curiosity."],
]


def ensure_sample_csvs(data_dir: str = "data") -> tuple:
    d = Path(data_dir); d.mkdir(parents=True, exist_ok=True)
    cpath, apath = d / "clients.csv", d / "advisors.csv"
    if not cpath.exists():
        with open(cpath, "w", newline="") as f:
            csv.writer(f).writerows(SAMPLE_CLIENTS)
    if not apath.exists():
        with open(apath, "w", newline="") as f:
            csv.writer(f).writerows(SAMPLE_ADVISORS)
    return str(cpath), str(apath)
