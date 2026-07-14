"""Post-hoc analysis of an OASIS simulation database.

OASIS persists everything to SQLite. The tables of interest here (schema is
stable across recent releases, but run ``inspect_schema`` on your DB if a
query fails after an upgrade):

    user     — agent accounts (user_id, agent_id, user_name, ...)
    post     — posts (post_id, user_id, content, created_at, num_likes,
               num_dislikes, ...)
    comment  — comments (comment_id, post_id, user_id, content, ...)
    like / dislike (and comment_like / comment_dislike) — vote edges
    trace    — full action log per agent (agent_id, action, info, created_at)

The trace table is the most useful for research: it is a complete,
timestamped record of every action every agent took.
"""

import json
import sqlite3

import pandas as pd


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def inspect_schema(db_path: str) -> dict:
    """List all tables and their columns — run this first after any
    camel-oasis version bump."""
    conn = _connect(db_path)
    tables = [r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")]
    schema = {
        t: [c["name"] for c in conn.execute(f"PRAGMA table_info({t})")]
        for t in tables
    }
    conn.close()
    return schema


def load_posts(db_path: str) -> pd.DataFrame:
    conn = _connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT p.post_id, p.user_id,
               COALESCE(u.user_name, u.name) AS user_name,
               p.content, p.num_likes, p.num_dislikes, p.created_at
        FROM post p JOIN user u ON p.user_id = u.user_id
        ORDER BY p.created_at
        """,
        conn,
    )
    conn.close()
    return df


def load_comments(db_path: str) -> pd.DataFrame:
    conn = _connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT c.comment_id, c.post_id, c.user_id,
               COALESCE(u.user_name, u.name) AS user_name,
               c.content, c.created_at
        FROM comment c JOIN user u ON c.user_id = u.user_id
        ORDER BY c.created_at
        """,
        conn,
    )
    conn.close()
    return df


def load_action_trace(db_path: str) -> pd.DataFrame:
    """Full action log — the primary artifact for behavioral analysis."""
    conn = _connect(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM trace ORDER BY created_at", conn
    )
    conn.close()
    return df


def archetype_of(user_name) -> str:
    """Recover archetype from the username convention in personas.py
    (``{first}_{archetype_key}_{i}``), validated against the known key set.
    The event-source account and any nonconforming names fall through to
    'event_source' / 'other'."""
    from .personas import ARCHETYPES
    valid = {a.key for a in ARCHETYPES}
    if not isinstance(user_name, str):
        return "unknown"
    if "market_wire" in user_name:
        return "event_source"
    parts = user_name.split("_")
    key = "_".join(parts[1:-1]) if len(parts) >= 3 else ""
    return key if key in valid else "other"


def action_mix_by_archetype(db_path: str) -> pd.DataFrame:
    """Cross-tab of action types by archetype — the first thing to check:
    do personas actually behave differently, or did they collapse into a
    single behavioral mode?

    Note: per the OASIS schema, ``trace`` keys on ``user_id`` (the platform
    account id), not ``agent_id``; the mapping between the two lives in the
    ``user`` table.
    """
    trace = load_action_trace(db_path)
    conn = _connect(db_path)
    users = pd.read_sql_query(
        "SELECT user_id, COALESCE(user_name, name) AS user_name FROM user",
        conn,
    )
    conn.close()
    trace = trace.merge(users, on="user_id", how="left")
    trace["archetype"] = trace["user_name"].map(archetype_of)
    return (
        trace.pivot_table(index="archetype", columns="action",
                          aggfunc="size", fill_value=0)
    )


def engagement_summary(db_path: str) -> pd.DataFrame:
    """Per-post engagement: score, comment count — sorted by total
    engagement. Injected posts (event source) sit at the top of the causal
    chain, so their trajectories are the treatment-response curves."""
    posts = load_posts(db_path)
    comments = load_comments(db_path)
    n_comments = comments.groupby("post_id").size().rename("n_comments")
    posts = posts.join(n_comments, on="post_id").fillna({"n_comments": 0})
    posts["score"] = posts["num_likes"] - posts["num_dislikes"]
    posts["engagement"] = posts["num_likes"] + posts["num_dislikes"] + posts["n_comments"]
    return posts.sort_values("engagement", ascending=False)


def keyword_propagation(db_path: str, keywords: list) -> pd.DataFrame:
    """Track how many distinct agents mention any of ``keywords`` over time —
    a simple propagation curve for rumor/meme scenarios (e.g. ['Meridian']
    or ['ZVLT'])."""
    posts = load_posts(db_path)[["user_name", "content", "created_at"]]
    comments = load_comments(db_path)[["user_name", "content", "created_at"]]
    text = pd.concat([posts, comments], ignore_index=True)
    pattern = "|".join(keywords)
    hits = text[text["content"].str.contains(pattern, case=False, na=False)].copy()
    hits["archetype"] = hits["user_name"].map(archetype_of)
    return hits.sort_values("created_at")


def report(db_path: str, keywords: list = None) -> None:
    print("=" * 70)
    print(f"Report: {db_path}")
    print("=" * 70)

    posts = engagement_summary(db_path)
    print(f"\nPosts: {len(posts)}   "
          f"Comments: {int(posts['n_comments'].sum())}   "
          f"Total votes: {int((posts['num_likes'] + posts['num_dislikes']).sum())}")

    print("\n-- Top 5 posts by engagement --")
    for _, row in posts.head(5).iterrows():
        print(f"  [{row['score']:+d} | {int(row['n_comments'])}c] "
              f"{row['user_name']}: {row['content'][:80]}")

    print("\n-- Action mix by archetype --")
    print(action_mix_by_archetype(db_path).to_string())

    if keywords:
        prop = keyword_propagation(db_path, keywords)
        print(f"\n-- Propagation of {keywords}: "
              f"{prop['user_name'].nunique()} distinct agents, "
              f"{len(prop)} mentions --")
        print(prop.groupby("archetype").size().to_string())


if __name__ == "__main__":
    import sys
    db = sys.argv[1] if len(sys.argv) > 1 else "results/bank_rumor.db"
    kws = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    print(json.dumps(inspect_schema(db), indent=2))
    report(db, kws)
