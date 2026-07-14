"""Panorama visualization — MiroFish-style replay of a simulation run.

Reads the OASIS SQLite database and emits a single self-contained HTML file
with three synchronized views driven by a time scrubber:

  1. Feed replay   — the simulated platform rendered as threaded post cards,
                     appearing in chronological order as you scrub.
  2. Interaction network — agents as nodes (colored by archetype, sized by
                     activity); edges appear as interactions happen
                     (comment: commenter -> post author; vote: voter -> author).
  3. Trajectories  — cumulative activity per archetype over simulation time,
                     plus the action-mix composition.

Data is embedded as JSON; rendering uses vis-network and Plotly from CDN,
so the file needs an internet connection to open but no local server.

Usage:
    python -m src.visualize results/bank_rumor.db
    # -> results/bank_rumor_panorama.html
"""

import json
import sqlite3
from pathlib import Path
from string import Template

from .analysis import archetype_of

PALETTE = {
    "retail_momentum": "#ff6b6b", "value_investor": "#4dabf7",
    "quant": "#9775fa", "permabear": "#e8590c", "passive_indexer": "#38d9a9",
    "fin_journalist": "#fcc419", "novice": "#f783ac", "advisor": "#63e6be",
    "event_source": "#ced4da", "other": "#868e96", "unknown": "#868e96",
}
_CYCLE = ["#4dabf7", "#38d9a9", "#fcc419", "#ff6b6b", "#9775fa", "#f783ac",
          "#63e6be", "#e8590c", "#74c0fc", "#b2f2bb"]


def group_of(user_name) -> str:
    """Group label for coloring: advisory populations group by specialty
    (adv_*) or risk tolerance (cli_*); legacy populations by archetype."""
    if isinstance(user_name, str):
        if user_name.startswith("adv_"):
            return "advisor:" + user_name.split("_")[1]
        if user_name.startswith("cli_"):
            return "client:" + user_name.split("_")[1]
    return archetype_of(user_name)


def color_for(group: str, palette: dict) -> str:
    if group not in palette:
        palette[group] = _CYCLE[len([k for k in palette
                                     if k not in PALETTE]) % len(_CYCLE)]
    return palette[group]


def _rows(conn, sql, params=()):
    try:
        return [dict(r) for r in conn.execute(sql, params)]
    except sqlite3.OperationalError:
        return []  # table absent in this OASIS version


def build_payload(db_path: str) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    users = _rows(conn, """
        SELECT user_id, COALESCE(user_name, name) AS user_name FROM user""")
    uname = {u["user_id"]: (u["user_name"] or f"user_{u['user_id']}")
             for u in users}
    palette = dict(PALETTE)
    uarch = {uid: group_of(n) for uid, n in uname.items()}

    posts = _rows(conn, """
        SELECT post_id, user_id, content, created_at, num_likes, num_dislikes
        FROM post""")
    comments = _rows(conn, """
        SELECT comment_id, post_id, user_id, content, created_at
        FROM comment""")
    likes = _rows(conn, "SELECT user_id, post_id, created_at FROM like")
    dislikes = _rows(conn, "SELECT user_id, post_id, created_at FROM dislike")
    trace = _rows(conn, "SELECT user_id, action, created_at FROM trace")
    conn.close()

    post_author = {p["post_id"]: p["user_id"] for p in posts}

    # Unified event stream, ordered by created_at (works for both datetime
    # strings and integer sandbox-clock values — SQLite stores what it got).
    events = []
    for p in posts:
        events.append({"kind": "post", "t": p["created_at"], "id": p["post_id"],
                       "user_id": p["user_id"]})
    for c in comments:
        events.append({"kind": "comment", "t": c["created_at"],
                       "id": c["comment_id"], "user_id": c["user_id"],
                       "post_id": c["post_id"]})
    for v in likes:
        events.append({"kind": "like", "t": v["created_at"],
                       "user_id": v["user_id"], "post_id": v["post_id"]})
    for v in dislikes:
        events.append({"kind": "dislike", "t": v["created_at"],
                       "user_id": v["user_id"], "post_id": v["post_id"]})
    events.sort(key=lambda e: (str(e["t"]), e["kind"]))
    for i, e in enumerate(events):
        e["order"] = i

    order_of_post = {e["id"]: e["order"] for e in events if e["kind"] == "post"}
    order_of_comment = {e["id"]: e["order"] for e in events
                        if e["kind"] == "comment"}

    payload_posts = [{
        "post_id": p["post_id"],
        "user": uname.get(p["user_id"], "?"),
        "archetype": uarch.get(p["user_id"], "unknown"),
        "content": p["content"],
        "likes": p["num_likes"], "dislikes": p["num_dislikes"],
        "order": order_of_post.get(p["post_id"], 0),
    } for p in posts]

    payload_comments = [{
        "comment_id": c["comment_id"], "post_id": c["post_id"],
        "user": uname.get(c["user_id"], "?"),
        "archetype": uarch.get(c["user_id"], "unknown"),
        "content": c["content"],
        "order": order_of_comment.get(c["comment_id"], 0),
    } for c in comments]

    # Interaction edges: actor -> content author, stamped with event order.
    edges = []
    for e in events:
        if e["kind"] in ("comment", "like", "dislike"):
            target = post_author.get(e.get("post_id"))
            if target is not None and target != e["user_id"]:
                edges.append({"from": e["user_id"], "to": target,
                              "type": e["kind"], "order": e["order"]})

    # Node activity from the trace (all actions, not just content).
    activity_count = {}
    for t in trace:
        activity_count[t["user_id"]] = activity_count.get(t["user_id"], 0) + 1

    nodes = [{
        "id": uid, "label": uname[uid], "archetype": uarch[uid],
        "color": color_for(uarch[uid], palette),
        "shape": ("diamond" if uarch[uid].startswith("advisor:")
                  else "star" if uarch[uid] == "event_source" else "dot"),
        "value": 1 + activity_count.get(uid, 0),
    } for uid in uname]

    # Cumulative content-activity trajectories per archetype.
    archetypes = sorted({a for a in uarch.values()})
    cum = {a: [0] * (len(events) + 1) for a in archetypes}
    for e in events:
        a = uarch.get(e["user_id"], "unknown")
        for arch in archetypes:
            cum[arch][e["order"] + 1] = cum[arch][e["order"]] + (arch == a)

    # Action mix from the full trace.
    mix = {}
    trace_uarch = {uid: uarch.get(uid, "unknown") for uid in uname}
    for t in trace:
        a = trace_uarch.get(t["user_id"], "unknown")
        mix.setdefault(a, {}).setdefault(t["action"], 0)
        mix[a][t["action"]] += 1

    return {
        "posts": payload_posts, "comments": payload_comments,
        "nodes": nodes, "edges": edges, "n_events": len(events),
        "cumulative": cum, "action_mix": mix, "palette": palette,
        "leaderboard": [],
    }


HTML = Template(r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>$TITLE — Panorama</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/standalone/umd/vis-network.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"></script>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; background: #0d1117; color: #e6edf3;
         font-family: -apple-system, "Segoe UI", Roboto, sans-serif; }
  header { padding: 18px 28px; border-bottom: 1px solid #21262d;
           display: flex; align-items: baseline; gap: 18px; }
  header h1 { font-size: 20px; margin: 0; }
  header .sub { color: #8b949e; font-size: 13px; }
  #controls { padding: 14px 28px; display: flex; align-items: center;
              gap: 14px; border-bottom: 1px solid #21262d;
              position: sticky; top: 0; background: #0d1117; z-index: 10; }
  #scrub { flex: 1; accent-color: #4dabf7; }
  #playBtn { background: #21262d; color: #e6edf3; border: 1px solid #30363d;
             border-radius: 6px; padding: 6px 16px; cursor: pointer;
             font-size: 14px; }
  #playBtn:hover { background: #30363d; }
  #tick { font-variant-numeric: tabular-nums; color: #8b949e;
          min-width: 110px; font-size: 13px; }
  main { display: grid; grid-template-columns: 1fr 1fr; gap: 0; }
  section { padding: 18px 28px; border-bottom: 1px solid #21262d; }
  section h2 { font-size: 13px; text-transform: uppercase;
               letter-spacing: 1px; color: #8b949e; margin: 0 0 12px; }
  #feed { max-height: 560px; overflow-y: auto; border-right: 1px solid #21262d; }
  .card { background: #161b22; border: 1px solid #21262d; border-radius: 10px;
          padding: 12px 14px; margin-bottom: 12px; }
  .card .meta { font-size: 12px; color: #8b949e; margin-bottom: 6px; }
  .badge { display: inline-block; padding: 1px 8px; border-radius: 10px;
           font-size: 11px; font-weight: 600; color: #0d1117; margin-right: 6px; }
  .card .content { font-size: 14px; line-height: 1.45; }
  .votes { font-size: 12px; color: #8b949e; margin-top: 6px; }
  .comment { border-left: 2px solid #30363d; margin: 10px 0 0 10px;
             padding: 6px 0 2px 12px; }
  .comment .content { font-size: 13px; color: #c9d1d9; }
  #network { height: 560px; }
  .charts { grid-column: 1 / -1; display: grid;
            grid-template-columns: 1fr 1fr; gap: 24px; }
  .plot { height: 340px; }
  #legend { padding: 10px 28px; display: flex; flex-wrap: wrap; gap: 10px;
            border-bottom: 1px solid #21262d; }
  #legend span { font-size: 12px; color: #c9d1d9; }
  #legend i { display: inline-block; width: 10px; height: 10px;
              border-radius: 50%; margin-right: 5px; }
</style>
</head>
<body>
<header>
  <h1>$TITLE</h1>
  <span class="sub" id="stats"></span>
</header>
<div id="controls">
  <button id="playBtn">▶ Play</button>
  <input type="range" id="scrub" min="0" value="0">
  <span id="tick"></span>
</div>
<div id="legend"></div>
<main>
  <section><h2>Platform feed</h2><div id="feed"></div></section>
  <section><h2>Interaction network</h2><div id="network"></div></section>
  <section class="charts" id="lbwrap" style="display:none">
    <div style="grid-column:1/-1"><h2>Advice leaderboard</h2>
      <table id="lb" style="width:100%;border-collapse:collapse;font-size:13px"></table>
    </div>
  </section>
  <section class="charts">
    <div><h2>Cumulative activity by archetype</h2><div id="traj" class="plot"></div></div>
    <div><h2>Action mix by archetype</h2><div id="mix" class="plot"></div></div>
  </section>
</main>
<script>
const D = $DATA;

// ---------- header / legend ----------
document.getElementById("stats").textContent =
  `$${D.nodes.length} agents · $${D.posts.length} posts · ` +
  `$${D.comments.length} comments · $${D.n_events} events`;
const archs = Object.keys(D.cumulative).sort();
document.getElementById("legend").innerHTML = archs.map(a =>
  `<span><i style="background:$${D.palette[a] || '#868e96'}"></i>$${a}</span>`
).join("");

// ---------- scrubber ----------
const scrub = document.getElementById("scrub");
scrub.max = D.n_events;
scrub.value = D.n_events;

// ---------- network ----------
const nodes = new vis.DataSet(D.nodes.map(n => ({
  id: n.id, label: n.label, value: n.value, shape: n.shape || "dot",
  color: { background: n.color, border: "#0d1117" },
  font: { color: "#8b949e", size: 10 },
})));
const edgeStyle = { comment: "#4dabf7", like: "#38d9a9", dislike: "#ff6b6b" };
const edges = new vis.DataSet(D.edges.map((e, i) => ({
  id: i, from: e.from, to: e.to, order: e.order, hidden: false,
  color: { color: edgeStyle[e.type], opacity: 0.55 },
  arrows: { to: { enabled: true, scaleFactor: 0.4 } }, width: 1.2,
})));
const net = new vis.Network(document.getElementById("network"),
  { nodes, edges },
  { physics: { solver: "forceAtlas2Based",
               forceAtlas2Based: { gravitationalConstant: -40 } },
    interaction: { hover: true } });

// ---------- feed ----------
function esc(s) { const d = document.createElement("div");
  d.textContent = s || ""; return d.innerHTML; }
function badge(a) {
  return `<span class="badge" style="background:$${D.palette[a] || '#868e96'}">$${a}</span>`;
}
function renderFeed(t) {
  const feed = document.getElementById("feed");
  const visiblePosts = D.posts.filter(p => p.order <= t)
                              .sort((a, b) => b.order - a.order);
  feed.innerHTML = visiblePosts.map(p => {
    const cs = D.comments.filter(c => c.post_id === p.post_id && c.order <= t);
    return `<div class="card">
      <div class="meta">$${badge(p.archetype)}<b>$${esc(p.user)}</b></div>
      <div class="content">$${esc(p.content)}</div>
      <div class="votes">▲ $${p.likes} ▼ $${p.dislikes} · $${cs.length} comments</div>
      $${cs.map(c => `<div class="comment">
          <div class="meta">$${badge(c.archetype)}<b>$${esc(c.user)}</b></div>
          <div class="content">$${esc(c.content)}</div></div>`).join("")}
    </div>`;
  }).join("") || `<div class="card"><div class="content">No activity yet — press play.</div></div>`;
}

// ---------- charts ----------
const dark = { paper_bgcolor: "#0d1117", plot_bgcolor: "#0d1117",
  font: { color: "#8b949e", size: 11 }, margin: { t: 10, r: 10, b: 40, l: 40 },
  legend: { orientation: "h" } };
let trajShapes = [];
function renderTraj(t) {
  const xs = [...Array(D.n_events + 1).keys()];
  const traces = archs.map(a => ({
    x: xs, y: D.cumulative[a], name: a, mode: "lines",
    line: { color: D.palette[a] || "#868e96", width: 2 } }));
  Plotly.react("traj", traces, { ...dark,
    xaxis: { title: "event index", gridcolor: "#21262d" },
    yaxis: { title: "cumulative actions", gridcolor: "#21262d" },
    shapes: [{ type: "line", x0: t, x1: t, y0: 0, y1: 1, yref: "paper",
               line: { color: "#e6edf3", width: 1, dash: "dot" } }] },
    { displayModeBar: false });
}
(function renderMix() {
  const actions = [...new Set(Object.values(D.action_mix)
    .flatMap(m => Object.keys(m)))].sort();
  const traces = actions.map(act => ({
    x: archs, y: archs.map(a => (D.action_mix[a] || {})[act] || 0),
    name: act, type: "bar" }));
  Plotly.newPlot("mix", traces, { ...dark, barmode: "stack",
    xaxis: { tickangle: -30 }, yaxis: { gridcolor: "#21262d" } },
    { displayModeBar: false });
})();

// ---------- leaderboard ----------
if (D.leaderboard && D.leaderboard.length) {
  document.getElementById("lbwrap").style.display = "grid";
  const cols = Object.keys(D.leaderboard[0]);
  document.getElementById("lb").innerHTML =
    "<tr>" + cols.map(c => `<th style="text-align:left;color:#8b949e;
     border-bottom:1px solid #30363d;padding:6px">$${c}</th>`).join("") + "</tr>" +
    D.leaderboard.map(r => "<tr>" + cols.map(c =>
      `<td style="border-bottom:1px solid #21262d;padding:6px">$${r[c]}</td>`)
      .join("") + "</tr>").join("");
}

// ---------- sync ----------
function update(t) {
  document.getElementById("tick").textContent = `event $${t} / $${D.n_events}`;
  renderFeed(t);
  edges.forEach(e => edges.update({ id: e.id, hidden: e.order > t }));
  renderTraj(t);
}
scrub.addEventListener("input", () => update(+scrub.value));

let timer = null;
document.getElementById("playBtn").addEventListener("click", () => {
  if (timer) { clearInterval(timer); timer = null;
    document.getElementById("playBtn").textContent = "▶ Play"; return; }
  if (+scrub.value >= D.n_events) scrub.value = 0;
  document.getElementById("playBtn").textContent = "⏸ Pause";
  timer = setInterval(() => {
    scrub.value = +scrub.value + 1; update(+scrub.value);
    if (+scrub.value >= D.n_events) { clearInterval(timer); timer = null;
      document.getElementById("playBtn").textContent = "▶ Play"; }
  }, 350);
});

update(D.n_events);
</script>
</body>
</html>
""")


def generate(db_path: str, out_path: str = None, leaderboard=None) -> str:
    payload = build_payload(db_path)
    if leaderboard is not None:
        payload["leaderboard"] = leaderboard
    out = out_path or str(Path(db_path).with_suffix("")) + "_panorama.html"
    title = Path(db_path).stem.replace("_", " ")
    Path(out).write_text(HTML.substitute(
        TITLE=title, DATA=json.dumps(payload)))
    print(f"Panorama written to {out} "
          f"({payload['n_events']} events, {len(payload['nodes'])} agents)")
    return out


if __name__ == "__main__":
    import sys
    generate(sys.argv[1] if len(sys.argv) > 1 else "results/bank_rumor.db",
             sys.argv[2] if len(sys.argv) > 2 else None)
