# FinSim: Multi-Agent Social Simulation for Finance Research

**Project documentation** · July 2026 · Built on [OASIS](https://github.com/camel-ai/oasis) (CAMEL-AI)

## 1. Motivation and scope

FinSim is a simulation platform for studying social dynamics in financial contexts using LLM agents on a Reddit-style platform. It supports two modes: an **open social feed** populated by heterogeneous investor archetypes reacting to injected market events, and an **advisory network** in which clients pose questions and credentialed advisor agents deliberate, challenge one another, and are adjudicated. The unifying methodological stance is that injected events are treatments delivered by a controlled source node, everything else is endogenous response, and all platform state persists to SQLite for post-hoc analysis. A companion Monte Carlo fund-panel engine extends hypothesis testing to fund-performance questions (persistence, survivorship) that a discourse simulation cannot address.

Research questions the platform is designed to reach: rumor propagation versus correction reach; belief updating under regime reversals; herding and the amplification of confident voices; advice quality and its adjudication (community votes vs. LLM judge vs. client verdict); the effect of institutional memory on advice; and reception dynamics of index-investing evidence.

## 2. Architecture

```
finsim/
├── run.py                    # open-feed scenarios CLI
├── run_advisory.py           # advisory cases CLI
├── src/
│   ├── personas.py           # 8 investor archetypes, seeded profile generation
│   ├── scenarios.py          # event-schedule scenario library (open feed)
│   ├── models.py             # DeepSeek (default) / OpenAI / vLLM backends
│   ├── simulation.py         # env construction + step loop (open feed)
│   ├── analysis.py           # SQLite post-hoc analysis (schema-verified)
│   ├── visualize.py          # single-file interactive panorama (small runs)
│   ├── dashboard.py          # query-backed Flask dashboard (large runs)
│   ├── fund_panel.py         # Monte Carlo fund-panel hypothesis testing
│   └── advisory/
│       ├── profiles.py       # CSV-driven client/advisor populations
│       ├── network.py        # advisor-client + capped client-client graph
│       ├── cases.py          # advisory case library
│       ├── runner.py         # phased deliberation protocol
│       ├── archive.py        # cross-run institutional memory
│       └── adjudicate.py     # 3-channel adjudication + compliance scan
├── data/                     # profiles, CSVs
└── results/                  # run databases, panoramas
```

**Simulation core.** Agent 0 is always a newswire-style event source that only delivers `ManualAction` injections and never acts autonomously, cleanly separating exogenous treatment from endogenous response. Scenarios (open feed) schedule events at timesteps with organic `LLMAction` steps between; advisory cases run a phased protocol (question → advisor-only answers → open deliberation → client verdict interview). Model backends: DeepSeek `deepseek-chat` by default (V3; reasoning models add cost/latency without benefit for persona decisions), self-hosted vLLM with multi-replica scheduling for larger populations, or OpenAI.

**Populations.** Open feed: eight archetypes (retail momentum, value investor, quant, permabear, passive indexer, financial journalist, novice, advisor), instantiated with seeded stochastic variation. Advisory: clients (age, sex, risk tolerance, horizon, portfolio, goal, income) and advisors (education, experience, specialty, philosophy, style) loaded from CSV; advisor personas embed a hard instrument constraint (Vanguard-style universe — diversified low-cost index funds/ETFs, investment-grade bonds, cash; no crypto/options/leverage/stock-picking) and a search-prior-cases protocol. Username conventions (`{first}_{archetype}_{i}`, `adv_{specialty}_{i}`, `cli_{risk}_{i}`) make group recovery purely lexical in analysis.

**Network.** Clients are assigned experience-weighted primary advisors; client–client ties form a small-world graph hard-capped per client (default 50). Delivery is dual-channel: persona text (reliable — directly conditions LLM behavior) and best-effort platform FOLLOW edges (whether the Reddit recommender uses them is unverified).

**Institutional memory.** Prior-run DBs can seed new runs: each archived case is a verbatim-truncated post (client question + top-voted advice) by a `case_archive` account, findable via the agents' SEARCH_POSTS action.

**Adjudication (advisory).** Three independent channels: community endorsement (net comment votes + reply mentions); optional LLM judge scoring suitability/risk-alignment/completeness/compliance against the client profile (rubric-dependence acknowledged — the judge is one noisy rater, not ground truth); and the asking client's end-of-run INTERVIEW verdict. A compliance scanner flags prohibited-instrument lexicon and surfaces non-whitelisted tickers (Vanguard CA/US/UK whitelist).

**Visualization.** Two tiers. `visualize.py` emits a self-contained HTML panorama (time-scrubbed feed replay, interaction network, trajectories, advice leaderboard) — appropriate to ~10^2–10^3 agents since it embeds all data. `dashboard.py` is the scalable tier: a Flask server where the browser receives only SQL-computed aggregates, pages, and samples — startup builds covering indexes, an FTS5 full-text index (via the canonical `rebuild` command), and a user→group table; the network view is a group-level supergraph (bounded by #groups², independent of population) with drill-down to the top-N most active agents; static aggregates are cached after first computation. Benchmarked at 100k users / 50k posts / 400k comments / 1.5M trace rows: ≤3.5 s first-hit aggregates, 1 ms cached, ~150 ms paginated search, ≤44 KB payloads.

## 3. Verified technical facts

Established against the OASIS repository and live run databases (camel-oasis is pre-1.0; re-verify on upgrade):

- Reddit profile JSON schema: `realname, username, bio, persona, age, gender, mbti, country, profession, interested_topics`.
- `trace` keys on `user_id`, not `agent_id`; the mapping lives in `user`. Standard mapping `user_id = agent_id + 1`.
- `user` has both `user_name` and `name`; the populated column varies — all queries use `COALESCE(user_name, name)`.
- Vote tables: `like`, `dislike`, `comment_like`, `comment_dislike` (analysis degrades gracefully if absent).
- External-content FTS5 indexes must be built with `INSERT INTO fts(fts) VALUES('rebuild')`; `INSERT...SELECT` does not reliably build the token index.

## 4. Experiments and findings to date

**Bank solvency rumor (`bank_rumor`, ~30 agents, DeepSeek).** The seeded rumor drew 62 comments against 25 for the official debunk — the rumor/correction asymmetry the scenario was designed to measure.

**Hormuz oil shock (`hormuz_oil_shock`, 35 agents + source, 5-stage arc mirroring the 2026 conflict).** Findings from staged analysis: (i) strong primacy in attention — engagement on injected posts fell 53→17→3→13→1 comments across stages, with the ceasefire partially re-energizing discussion but the post-resolution reversal re-activating only permabears, suggesting resolution-framing suppresses re-mobilization; (ii) persona fidelity through regime reversal — permabears did not capitulate at the ceasefire (no consensus collapse), retail momentum flip-flopped in character, and quants exhibited archetype-specific timing (activity peaking at resolution, running post-mortems); (iii) homophily and sanctioning — retail momentum gave 41/64 likes in-group; novices gave 22/39 likes to retail momentum content; all 6 quant dislikes targeted retail momentum posts; (iv) content production concentrated in reactive archetypes (only 4 of 8 archetypes authored posts); (v) an emergent advice market around a novice's question. Caveats: single run; recommender exposure confounds the novice-herding claim; the OASIS paper reports LLM agents over-herd relative to humans; positivity bias (142 likes vs. 11 dislikes); one temporal-misattribution artifact (stale stage-3 content posted in stage 5).

**Fund-panel replication (Plagge et al. 2021, Figure 5).** The paper's quintile-transition + survivorship methodology was reimplemented on synthetic panels with controlled ground truth (2,800 funds, two 5-year windows, performance-linked attrition, 50–100 replications per DGP). Results: the merged/liquidated gradient is reproduced by pure luck plus performance-linked closure (Q5 death 32.8% vs. observed 34.4%) — no skill story required; cost heterogeneity at realistic dispersion cannot explain observed winner persistence (Q1→Q1 17.2% vs. observed 27.1%) because idiosyncratic tracking-error noise swamps the cost spread over 5-year windows; persistent factor tilts with zero skill (sd 2%/yr, 0.7 correlation across windows) reproduce the matrix (26.4% / 21.6% / 34.3% vs. observed 27.1% / 20.3% / 34.4%), lowest Monte Carlo distance of all DGPs. Residual misfit in top-quintile deaths (8.0% vs. 15.8%) indicates a non-performance merger channel. Consistency ≠ identification; formal next step is Fama–French (2010)-style bootstrap on the alpha cross-section.

## 5. Known limitations

Single-model populations risk correlated behavior across personas; per-agent model assignment is supported by OASIS and unexploited. Timestamps are wall-clock, not simulation-clock, so stage segmentation relies on injection times. Vote positivity bias limits polarization dynamics. Recommender internals are not logged, confounding exposure and preference. The compliance scan is lexicon-level (catches "bitcoin", not obliquely phrased speculation). Mention-based reply engagement in adjudication is a crude proxy. All findings so far are single-run and qualitative; nothing has error bars yet.

## 6. Roadmap

Near-term: multi-seed replications (5–10 seeds per condition) for the Hormuz decay curve and the rumor/correction asymmetry; fixed-panel INTERVIEW probes to separate silent belief updating from attention exhaustion; the institutional-memory experiment (same case with/without `--history`, judge-score and client-verdict outcomes); recommender-exposure logging to deconfound herding. Medium-term: the coupled fund-panel × social experiment — inject league tables with known luck/tilt/skill ground truth and measure whether population inference tracks the DGP or the ranking; heterogeneous model populations; scale tests at 10^3–10^4 agents on vLLM replicas.
