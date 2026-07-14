# FinSim — Finance Scenario Simulation on OASIS

A simulation platform for studying social dynamics in financial contexts using [OASIS](https://github.com/camel-ai/oasis) (CAMEL-AI). LLM agents with heterogeneous financial personas post, comment, upvote, and downvote on a Reddit-style platform while exogenous market events are injected on a fixed schedule.

## Design

The experimental unit is a **scenario**: a population mix over persona archetypes, a step budget, and a schedule of exogenous events. Events are injected through a dedicated event-source agent (agent 0, a newswire account) that never takes autonomous actions, so treatment (injected events) and response (organic agent behavior) remain cleanly separated. All platform state — posts, comments, votes, and a complete per-agent action trace — is persisted to SQLite by OASIS, which `src/analysis.py` consumes.

Eight archetypes are defined in `src/personas.py`: retail momentum trader, value investor, quant, permabear, passive indexer, financial journalist, novice, and advisor. Each encodes distinct information-processing and engagement dispositions in its persona text (e.g., the novice is explicitly susceptible to herding; the quant demands evidence). Profiles are generated in the exact JSON schema `generate_reddit_agent_graph` expects, with seeded stochastic variation in age, MBTI, and country.

Three scenarios ship in `src/scenarios.py`:

| Scenario | Question | Injections |
|---|---|---|
| `rate_decision` | Sentiment divergence and thread polarization after a surprise 50bp cut | Cut announcement (t=0), hawkish presser (t=4) |
| `bank_rumor` | Rumor propagation vs. correction reach (misinformation dynamics) | Solvency rumor (t=0), official debunk (t=6) |
| `meme_stock` | Herding: do upvotes snowball and do skeptics get drowned out? | Short-squeeze narrative (t=0) |

All tickers and institutions are fictional.

## Setup

```bash
pip install camel-oasis pandas
export DEEPSEEK_API_KEY=...        # default backend (deepseek-chat / V3)
# or: export OPENAI_API_KEY=...    # for --backend openai
```

For a self-hosted backend, serve any chat model with vLLM:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
```

Multiple replicas (e.g., one per GPU) can be passed as multiple URLs; OASIS's model manager schedules agents across them.

## Run

```bash
python run.py --scenario bank_rumor                        # DeepSeek (default)
python run.py --scenario meme_stock --backend vllm \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --vllm-urls http://localhost:8000/v1 http://localhost:8001/v1
```

## Analyze

```bash
python -m src.analysis results/bank_rumor.db Meridian
```

The report prints engagement per post, an action-mix cross-tab by archetype (the first sanity check: personas should behave differently), and a keyword propagation curve identifying which archetypes amplified vs. debunked. `load_action_trace` returns the full timestamped action log for anything deeper (e.g., inter-event response latency, vote-cascade analysis, polarization metrics over comment threads).

## Extending

- **New scenario**: add a `Scenario` to `src/scenarios.py` — population mix, steps, `Event` list.
- **New archetype**: add an `Archetype` to `src/personas.py`; the persona text is the behavioral prior, so be explicit about posting frequency, voting disposition, and credulity.
- **Belief probes**: recent OASIS releases add an `INTERVIEW` action, which lets you query an agent's private beliefs at chosen timesteps without creating platform content — useful for measuring opinion shift (e.g., before/after the debunk in `bank_rumor`). Check your installed version's `ActionType` for availability.
- **Version note**: camel-oasis is pre-1.0 and patch releases can break APIs. If a query in `analysis.py` fails after an upgrade, run `inspect_schema(db_path)` to see the current table layout.

## Project layout

```
finsim/
├── run.py               # CLI entrypoint
├── src/
│   ├── personas.py      # archetype definitions + profile generation
│   ├── scenarios.py     # scenario/event library
│   ├── models.py        # OpenAI / vLLM backends
│   ├── simulation.py    # env construction + step loop
│   └── analysis.py      # SQLite post-hoc analysis
├── data/                # generated agent profiles
└── results/             # simulation DBs
```

## Visualize

```bash
python -m src.visualize results/bank_rumor.db
# -> results/bank_rumor_panorama.html
```

Generates a self-contained interactive panorama (MiroFish-style) with a time scrubber that replays the run: the platform feed with threaded comments, an agent interaction network (nodes colored by archetype, sized by activity; edges appear as comments/votes happen), and cumulative activity trajectories per archetype with an action-mix breakdown. Rendering libraries load from CDN, so opening the file requires an internet connection but no server.

## Advisory network mode

A second simulation mode models a client–advisor advisory community rather than an open social feed. Populations are defined in CSV (`data/clients.csv`: age, sex, risk tolerance, horizon, portfolio, goal, income band, notes; `data/advisors.csv`: education, experience, specialty, philosophy, style) — sample files are generated on first run. Advisor personas embed a hard instrument constraint (diversified low-cost index funds/ETFs, investment-grade bonds, cash; no crypto, options, leverage, or stock picking) and a research protocol instructing them to search the platform for similar prior cases before answering.

Each **case** opens with a designated client posting a question, followed by an advisor-only response phase, then open deliberation where advisors challenge or endorse each other and peer clients contribute experience. The network (each client assigned an experience-weighted primary advisor; client–client small-world graph capped at `--max-connections`, default 50) is delivered through personas (reliable) and best-effort platform FOLLOW edges. Cross-run institutional memory is supported via `--history`: question + top-advice summaries from prior run DBs are seeded as searchable archive posts.

```bash
python run_advisory.py --case preretirement_drawdown
python run_advisory.py --case downturn_panic --history results/preretirement_drawdown.db --judge
python -m src.advisory.adjudicate results/preretirement_drawdown.db
```

Adjudication ranks advice through three independent channels: community endorsement (net comment votes + reply mentions), an optional LLM judge scoring suitability/risk-alignment/completeness/compliance against the asking client's profile (DeepSeek via `DEEPSEEK_API_KEY`; `pip install openai`), and the asking client's own end-of-run interview verdict. A compliance scan flags prohibited-instrument mentions and surfaces non-whitelisted tickers for review. The panorama gains advisor-shaped nodes, specialty/risk-tolerance coloring, and an advice leaderboard.

Cases ship in `src/advisory/cases.py` (pre-retirement drawdown anxiety, inheritance lump sum, drawdown panic with mid-case market events, first-time investor); add new ones as `AdvisoryCase` entries.

## Dashboard (large runs)

The panorama embeds all data in one HTML file and suits runs up to ~10^3 agents. For large populations, `src/dashboard.py` serves a query-backed dashboard where the browser receives only aggregates, pages, and samples:

```bash
pip install flask
python -m src.dashboard results/bank_rumor.db --port 8050
```

Startup builds covering indexes, an FTS5 full-text index, and a user→group table; views include binned activity trajectories, action mix, a group-level supergraph with top-N agent drill-down, a paginated full-text-searchable feed with on-demand threads, and the advisory leaderboard. Benchmarked at 100k users / 1.9M rows: ≤3.5 s first-hit aggregates (1 ms cached), ~150 ms search pages, ≤44 KB payloads. See PROJECT.md for full project documentation.

## Model backends

Any model reachable through CAMEL's ModelFactory can drive agents: hosted APIs (DeepSeek, OpenAI, Anthropic, Gemini), AWS Bedrock (`bedrock` for the OpenAI-compatible gateway via `BEDROCK_API_BASE_URL`/`BEDROCK_API_KEY`, `bedrock-converse` for the native Converse API via standard AWS credentials), local inference (vLLM, Ollama, SGLang), and any OpenAI-compatible endpoint via the `openai-compatible` catch-all (Groq, Together, OpenRouter, Azure, LM Studio). Describe endpoints in a JSON config (see `models.example.json`) and pass `--model-config models.json` to either runner; `count` weights agent assignment across endpoints, and in advisory mode `role: advisor|client` pins populations to different backends (e.g., a frontier model for 5 advisors, a local 3B for thousands of clients).
