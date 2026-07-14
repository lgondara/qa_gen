"""Institutional memory across simulation runs.

Advisors are instructed to search the platform for similar prior cases
before answering. Within one run, SEARCH_POSTS covers everything on the
platform; across runs, this module carries memory forward: it extracts a
compact summary of each prior case (the client question plus the top-voted
advice) from previous run databases and returns them as posts to be seeded
at t=0 by a dedicated "case_archive" account.

Extraction is deliberately verbatim-truncated rather than LLM-summarized,
so archived content is a faithful record and costs no API calls.
"""

import sqlite3


def extract_case_summaries(db_paths: list, max_per_db: int = 3,
                           q_chars: int = 300, a_chars: int = 400) -> list:
    """-> list of archive post strings, one per prior thread."""
    summaries = []
    for db in db_paths:
        try:
            conn = sqlite3.connect(db)
            conn.row_factory = sqlite3.Row
        except sqlite3.Error:
            continue
        # question posts = posts by client accounts (cli_*) with comments
        posts = conn.execute("""
            SELECT p.post_id, p.content,
                   COALESCE(u.user_name, u.name) AS uname
            FROM post p JOIN user u ON p.user_id = u.user_id
            WHERE COALESCE(u.user_name, u.name) LIKE 'cli_%'
        """).fetchall()
        scored = []
        for p in posts:
            n = conn.execute(
                "SELECT COUNT(*) c FROM comment WHERE post_id=?",
                (p["post_id"],)).fetchone()["c"]
            if n:
                scored.append((n, p))
        scored.sort(key=lambda x: -x[0])
        for n, p in scored[:max_per_db]:
            top = conn.execute("""
                SELECT c.content, COALESCE(u.user_name, u.name) AS uname,
                       (SELECT COUNT(*) FROM comment_like cl
                        WHERE cl.comment_id = c.comment_id) AS score
                FROM comment c JOIN user u ON c.user_id = u.user_id
                WHERE c.post_id = ? AND COALESCE(u.user_name, u.name)
                      LIKE 'adv_%'
                ORDER BY score DESC LIMIT 1
            """, (p["post_id"],)).fetchone()
            body = (f"[CASE ARCHIVE] A client previously asked: "
                    f"\"{p['content'][:q_chars]}\"")
            if top:
                body += (f" The most endorsed advice (from {top['uname']}) "
                         f"was: \"{top['content'][:a_chars]}\"")
            summaries.append(body)
        conn.close()
    return summaries
