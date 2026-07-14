"""Scalable simulation dashboard.

The panorama (src/visualize.py) embeds the full dataset in one HTML file
and renders every element in the DOM — fine to a few hundred agents,
unusable at 10^5. This dashboard inverts the architecture: the browser
never receives raw data, only aggregates, pages, and samples computed by
SQL against the run database.

Scaling strategy:
  * On startup: create covering indexes (idempotent), an FTS5 full-text
    index over post content, and a user->group mapping table, so every
    endpoint below is a single indexed query.
  * Timeline: rank-binned event counts per group (fixed number of buckets
    regardless of event count).
  * Network: a group-level supergraph (weighted edges between groups —
    bounded by #groups^2, independent of population) with drill-down to
    the top-N most active individual agents and edges among them only.
  * Feed: server-side pagination + full-text search; threads load on
    demand.

Usage:
    pip install flask
    python -m src.dashboard results/bank_rumor.db --port 8050
"""

import argparse
import json
import re
import sqlite3
import threading

from flask import Flask, jsonify, render_template_string, request

# group extraction mirrors visualize.group_of (kept dependency-free here)
_ARCH = {"retail_momentum", "value_investor", "quant", "permabear",
         "passive_indexer", "fin_journalist", "novice", "advisor"}


def group_of(name):
    if not isinstance(name, str):
        return "unknown"
    if name.startswith("adv_"):
        return "advisor:" + name.split("_")[1]
    if name.startswith("cli_"):
        return "client:" + name.split("_")[1]
    if "market_wire" in name or "case_archive" in name:
        return "event_source"
    p = name.split("_")
    k = "_".join(p[1:-1]) if len(p) >= 3 else ""
    return k if k in _ARCH else "other"


class Store:
    """Single shared connection + one-time index/FTS/group-table setup."""

    def __init__(self, db_path):
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._prepare()

    def _prepare(self):
        c = self.conn
        for stmt in [
            "CREATE INDEX IF NOT EXISTS ix_post_user ON post(user_id)",
            "CREATE INDEX IF NOT EXISTS ix_comment_post ON comment(post_id)",
            "CREATE INDEX IF NOT EXISTS ix_comment_user ON comment(user_id)",
            "CREATE INDEX IF NOT EXISTS ix_like_post ON like(post_id)",
            "CREATE INDEX IF NOT EXISTS ix_trace_user ON trace(user_id)",
            "CREATE INDEX IF NOT EXISTS ix_trace_time ON trace(created_at)",
        ]:
            try:
                c.execute(stmt)
            except sqlite3.OperationalError:
                pass
        # user -> group mapping (one scan of user, then joins are indexed)
        c.execute("DROP TABLE IF EXISTS _dash_group")
        c.execute("CREATE TABLE _dash_group (user_id INTEGER PRIMARY KEY, "
                  "grp TEXT)")
        rows = c.execute(
            "SELECT user_id, COALESCE(user_name, name) n FROM user")
        c.executemany("INSERT INTO _dash_group VALUES (?,?)",
                      [(r["user_id"], group_of(r["n"])) for r in rows])
        c.execute("CREATE INDEX IF NOT EXISTS ix_dash_grp "
                  "ON _dash_group(grp)")
        # full-text search over posts (FTS5 ships with SQLite); 'rebuild'
        # is the canonical index build for external-content tables —
        # plain INSERT...SELECT does not reliably build the token index
        try:
            c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS post_fts USING "
                      "fts5(content, content='post', content_rowid='post_id')")
            c.execute("INSERT INTO post_fts(post_fts) VALUES('rebuild')")
            self.fts = True
        except sqlite3.OperationalError:
            self.fts = False
        c.commit()
        self._cache = {}

    def q(self, sql, params=()):
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, params)]

    def q_cached(self, key, sql, params=()):
        """For aggregates that are static once a run has finished."""
        if key not in self._cache:
            self._cache[key] = self.q(sql, params)
        return self._cache[key]


def create_app(db_path: str) -> Flask:
    app = Flask(__name__)
    store = Store(db_path)

    @app.get("/api/summary")
    def summary():
        s = {}
        for t in ["user", "post", "comment", "like", "dislike", "trace"]:
            try:
                s[t] = store.q(f"SELECT COUNT(*) c FROM '{t}'")[0]["c"]
            except sqlite3.OperationalError:
                s[t] = 0
        s["groups"] = store.q(
            "SELECT grp, COUNT(*) n FROM _dash_group GROUP BY grp "
            "ORDER BY n DESC")
        return jsonify(s)

    @app.get("/api/timeline")
    def timeline():
        bins = min(int(request.args.get("bins", 120)), 500)
        rows = store.q_cached(f"timeline{bins}", """
            WITH e AS (
              SELECT g.grp, ROW_NUMBER() OVER (ORDER BY t.created_at) rn,
                     COUNT(*) OVER () n
              FROM trace t JOIN _dash_group g ON t.user_id = g.user_id
              WHERE t.action NOT IN ('sign_up'))
            SELECT grp, MIN(CAST((rn - 1) * ? / n AS INT), ? - 1) AS bucket,
                   COUNT(*) c
            FROM e GROUP BY grp, bucket""", (bins, bins))
        return jsonify({"bins": bins, "rows": rows})

    @app.get("/api/action_mix")
    def action_mix():
        return jsonify(store.q_cached("mix", """
            SELECT g.grp, t.action, COUNT(*) c
            FROM trace t JOIN _dash_group g ON t.user_id = g.user_id
            WHERE t.action <> 'sign_up'
            GROUP BY g.grp, t.action"""))

    @app.get("/api/network")
    def network():
        mode = request.args.get("mode", "groups")
        if mode == "groups":
            edges = store.q_cached("gedges", """
                SELECT g1.grp f, g2.grp t, COUNT(*) w
                FROM comment c
                JOIN post p ON c.post_id = p.post_id
                JOIN _dash_group g1 ON c.user_id = g1.user_id
                JOIN _dash_group g2 ON p.user_id = g2.user_id
                WHERE g1.grp <> g2.grp GROUP BY g1.grp, g2.grp""")
            nodes = store.q_cached("gnodes", """
                SELECT g.grp id, COUNT(*) value
                FROM trace t JOIN _dash_group g ON t.user_id = g.user_id
                GROUP BY g.grp""")
            return jsonify({"mode": "groups", "nodes": nodes, "edges": edges})
        top = min(int(request.args.get("top", 200)), 1000)
        nodes = store.q("""
            SELECT t.user_id id,
                   COALESCE(u.user_name, u.name) label, g.grp, COUNT(*) value
            FROM trace t
            JOIN user u ON t.user_id = u.user_id
            JOIN _dash_group g ON t.user_id = g.user_id
            GROUP BY t.user_id ORDER BY value DESC LIMIT ?""", (top,))
        ids = [n["id"] for n in nodes]
        marks = ",".join("?" * len(ids))
        edges = store.q(f"""
            SELECT c.user_id f, p.user_id t, COUNT(*) w
            FROM comment c JOIN post p ON c.post_id = p.post_id
            WHERE c.user_id IN ({marks}) AND p.user_id IN ({marks})
              AND c.user_id <> p.user_id
            GROUP BY c.user_id, p.user_id LIMIT 5000""", ids + ids)
        return jsonify({"mode": "agents", "nodes": nodes, "edges": edges})

    @app.get("/api/feed")
    def feed():
        page = max(int(request.args.get("page", 0)), 0)
        per = min(int(request.args.get("per", 20)), 100)
        qtext = request.args.get("q", "").strip()
        base = """
            SELECT p.post_id, p.content, p.num_likes, p.num_dislikes,
                   p.created_at, COALESCE(u.user_name, u.name) uname, g.grp,
                   (SELECT COUNT(*) FROM comment c
                    WHERE c.post_id = p.post_id) n_comments
            FROM post p JOIN user u ON p.user_id = u.user_id
            JOIN _dash_group g ON p.user_id = g.user_id"""
        if qtext and store.fts:
            rows = store.q(base + """
                WHERE p.post_id IN
                  (SELECT rowid FROM post_fts WHERE post_fts MATCH ?)
                ORDER BY p.created_at DESC LIMIT ? OFFSET ?""",
                (qtext, per, page * per))
        elif qtext:
            rows = store.q(base + " WHERE p.content LIKE ? "
                           "ORDER BY p.created_at DESC LIMIT ? OFFSET ?",
                           (f"%{qtext}%", per, page * per))
        else:
            rows = store.q(base + " ORDER BY p.created_at DESC "
                           "LIMIT ? OFFSET ?", (per, page * per))
        return jsonify(rows)

    @app.get("/api/thread/<int:post_id>")
    def thread(post_id):
        page = max(int(request.args.get("page", 0)), 0)
        per = min(int(request.args.get("per", 50)), 200)
        return jsonify(store.q("""
            SELECT c.comment_id, c.content, c.created_at,
                   COALESCE(u.user_name, u.name) uname, g.grp
            FROM comment c JOIN user u ON c.user_id = u.user_id
            JOIN _dash_group g ON c.user_id = g.user_id
            WHERE c.post_id = ? ORDER BY c.created_at
            LIMIT ? OFFSET ?""", (post_id, per, page * per)))

    @app.get("/api/leaderboard")
    def leaderboard():
        """Advisory adjudication (community channel), computed in SQL."""
        q = store.q("""
            SELECT p.post_id FROM post p JOIN _dash_group g
            ON p.user_id = g.user_id
            WHERE g.grp LIKE 'client:%' ORDER BY p.created_at LIMIT 1""")
        if not q:
            return jsonify([])
        return jsonify(store.q("""
            SELECT COALESCE(u.user_name, u.name) uname, g.grp,
                   COUNT(DISTINCT c.comment_id) n_answers,
                   (SELECT COUNT(*) FROM comment_like cl
                    JOIN comment c2 ON cl.comment_id = c2.comment_id
                    WHERE c2.user_id = c.user_id AND c2.post_id = ?)
                   - (SELECT COUNT(*) FROM comment_dislike cd
                      JOIN comment c3 ON cd.comment_id = c3.comment_id
                      WHERE c3.user_id = c.user_id AND c3.post_id = ?)
                   net_votes
            FROM comment c JOIN user u ON c.user_id = u.user_id
            JOIN _dash_group g ON c.user_id = g.user_id
            WHERE c.post_id = ? AND g.grp LIKE 'advisor:%'
            GROUP BY c.user_id ORDER BY net_votes DESC LIMIT 20""",
            (q[0]["post_id"],) * 3))

    @app.get("/")
    def index():
        return render_template_string(PAGE)

    return app


PAGE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>FinSim dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/standalone/umd/vis-network.min.js"></script>
<style>
:root{color-scheme:dark} *{box-sizing:border-box}
body{margin:0;background:#0d1117;color:#e6edf3;font-family:-apple-system,"Segoe UI",Roboto,sans-serif}
header{padding:16px 26px;border-bottom:1px solid #21262d;display:flex;gap:16px;align-items:baseline}
h1{font-size:18px;margin:0} .sub{color:#8b949e;font-size:13px}
main{display:grid;grid-template-columns:1fr 1fr;gap:0}
section{padding:16px 26px;border-bottom:1px solid #21262d;min-width:0}
h2{font-size:12px;text-transform:uppercase;letter-spacing:1px;color:#8b949e;margin:0 0 10px}
.plot{height:300px} #network{height:420px}
#feed{max-height:520px;overflow-y:auto}
.card{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:10px 12px;margin-bottom:10px;font-size:13px}
.meta{color:#8b949e;font-size:12px;margin-bottom:4px}
.badge{padding:1px 7px;border-radius:9px;font-size:11px;font-weight:600;color:#0d1117;margin-right:6px;display:inline-block}
.votes{color:#8b949e;font-size:12px;margin-top:4px;cursor:pointer}
.comment{border-left:2px solid #30363d;margin:8px 0 0 8px;padding:4px 0 2px 10px;color:#c9d1d9}
input,select,button{background:#161b22;color:#e6edf3;border:1px solid #30363d;border-radius:6px;padding:5px 10px;font-size:13px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{color:#8b949e;text-align:left;border-bottom:1px solid #30363d;padding:5px}
td{border-bottom:1px solid #21262d;padding:5px}
.row{display:flex;gap:8px;margin-bottom:10px;align-items:center;flex-wrap:wrap}
</style></head><body>
<header><h1>FinSim dashboard</h1><span class="sub" id="stats">loading…</span></header>
<main>
<section><h2>Cumulative activity by group</h2><div id="traj" class="plot"></div></section>
<section><h2>Action mix by group</h2><div id="mix" class="plot"></div></section>
<section>
  <h2>Interaction network</h2>
  <div class="row">
    <select id="netmode"><option value="groups">group supergraph</option>
      <option value="agents">top agents</option></select>
    <input id="topn" type="number" value="200" min="10" max="1000" style="width:80px">
    <button onclick="loadNet()">reload</button>
  </div>
  <div id="network"></div>
</section>
<section>
  <h2>Feed</h2>
  <div class="row"><input id="q" placeholder="full-text search…" style="flex:1">
    <button onclick="page=0;loadFeed()">search</button>
    <button onclick="page=Math.max(0,page-1);loadFeed()">‹ prev</button>
    <button onclick="page++;loadFeed()">next ›</button>
    <span class="sub" id="pageno"></span></div>
  <div id="feed"></div>
</section>
<section style="grid-column:1/-1" id="lbsec" hidden>
  <h2>Advice leaderboard (community channel)</h2><table id="lb"></table>
</section>
</main>
<script>
const PAL={}; const CYC=["#4dabf7","#38d9a9","#fcc419","#ff6b6b","#9775fa","#f783ac","#63e6be","#e8590c","#74c0fc","#b2f2bb"];
const color=g=>PAL[g]||(PAL[g]=CYC[Object.keys(PAL).length%CYC.length]);
const esc=s=>{const d=document.createElement("div");d.textContent=s||"";return d.innerHTML};
const J=async u=>(await fetch(u)).json();
const dark={paper_bgcolor:"#0d1117",plot_bgcolor:"#0d1117",font:{color:"#8b949e",size:11},
  margin:{t:8,r:8,b:36,l:42},legend:{orientation:"h"}};

(async()=>{
  const s=await J("/api/summary");
  document.getElementById("stats").textContent=
    `${s.user} agents · ${s.post} posts · ${s.comment} comments · ${s.trace} actions`;
  const tl=await J("/api/timeline");
  const groups=[...new Set(tl.rows.map(r=>r.grp))].sort();
  const traces=groups.map(g=>{
    const y=Array(tl.bins).fill(0);
    tl.rows.filter(r=>r.grp===g).forEach(r=>y[r.bucket]=r.c);
    for(let i=1;i<y.length;i++)y[i]+=y[i-1];
    return {x:[...y.keys()],y,name:g,mode:"lines",line:{color:color(g),width:2}}});
  Plotly.newPlot("traj",traces,{...dark,xaxis:{title:"time (binned)",gridcolor:"#21262d"},
    yaxis:{gridcolor:"#21262d"}},{displayModeBar:false});
  const mix=await J("/api/action_mix");
  const acts=[...new Set(mix.map(r=>r.action))].sort();
  const gs=[...new Set(mix.map(r=>r.grp))].sort();
  Plotly.newPlot("mix",acts.map(a=>({x:gs,type:"bar",name:a,
    y:gs.map(g=>(mix.find(r=>r.grp===g&&r.action===a)||{}).c||0)})),
    {...dark,barmode:"stack",xaxis:{tickangle:-30},yaxis:{gridcolor:"#21262d"}},
    {displayModeBar:false});
  loadNet(); loadFeed();
  const lb=await J("/api/leaderboard");
  if(lb.length){document.getElementById("lbsec").hidden=false;
    const cols=Object.keys(lb[0]);
    document.getElementById("lb").innerHTML=
      "<tr>"+cols.map(c=>`<th>${c}</th>`).join("")+"</tr>"+
      lb.map(r=>"<tr>"+cols.map(c=>`<td>${esc(String(r[c]))}</td>`).join("")+"</tr>").join("");}
})();

async function loadNet(){
  const mode=document.getElementById("netmode").value;
  const d=await J(`/api/network?mode=${mode}&top=${document.getElementById("topn").value}`);
  const nodes=new vis.DataSet(d.nodes.map(n=>({id:n.id,label:String(n.label||n.id),
    value:n.value,color:{background:color(n.grp||n.id),border:"#0d1117"},
    shape:(n.grp||n.id).startsWith("advisor:")?"diamond":"dot",
    font:{color:"#8b949e",size:mode==="groups"?13:9}})));
  const edges=new vis.DataSet(d.edges.map((e,i)=>({id:i,from:e.f,to:e.t,
    width:Math.min(1+Math.log1p(e.w),6),color:{color:"#4dabf7",opacity:.45},
    arrows:{to:{enabled:true,scaleFactor:.35}}})));
  new vis.Network(document.getElementById("network"),{nodes,edges},
    {physics:{solver:"forceAtlas2Based",stabilization:{iterations:120}}});
}

let page=0;
async function loadFeed(){
  const q=encodeURIComponent(document.getElementById("q").value);
  const rows=await J(`/api/feed?page=${page}&q=${q}`);
  document.getElementById("pageno").textContent=`page ${page+1}`;
  document.getElementById("feed").innerHTML=rows.map(p=>`
    <div class="card"><div class="meta">
      <span class="badge" style="background:${color(p.grp)}">${esc(p.grp)}</span>
      <b>${esc(p.uname)}</b></div>
      <div>${esc(p.content)}</div>
      <div class="votes" onclick="loadThread(${p.post_id},this)">
        ▲ ${p.num_likes} ▼ ${p.num_dislikes} · ${p.n_comments} comments (click to expand)</div>
      <div id="th${p.post_id}"></div></div>`).join("")||"<div class='card'>no results</div>";
}
async function loadThread(id,el){
  const cs=await J(`/api/thread/${id}`);
  document.getElementById("th"+id).innerHTML=cs.map(c=>`
    <div class="comment"><div class="meta">
      <span class="badge" style="background:${color(c.grp)}">${esc(c.grp)}</span>
      <b>${esc(c.uname)}</b></div>${esc(c.content)}</div>`).join("");
}
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description="FinSim dashboard")
    ap.add_argument("db")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    app = create_app(args.db)
    print(f"Dashboard on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
