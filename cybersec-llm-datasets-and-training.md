# Cybersecurity LLM: Datasets Considered and Training Approach

Consolidated from the Foundation-Sec-8B-Instruct fine-tuning work. Section 1
records the candidate models and datasets surveyed; Section 2 records the
pipeline and training configuration actually adopted.

*Note: benchmark deltas and release details below are as recorded at the time of
the survey (May 2026) and were not re-verified for this document.*

---

## 1. Candidate base models

Open-weight cybersecurity models cluster into four design philosophies. The
choice largely determines what fine-tuning recipe is sensible on top.

### 1.1 Continued pretraining + post-training

| Model | Base | Notes |
|---|---|---|
| **Foundation-Sec-8B** (Cisco) | Llama 3.1 | Continued pretraining on a curated security corpus; reported to match Llama 3.1-70B and GPT-4o-mini on some security-specific tasks |
| **Foundation-Sec-8B-Instruct** | ↑ | Chat-tuned variant. **Selected as our base.** |
| **Foundation-Sec-8B-Reasoning** | ↑ | Native reasoning variant; CTIBench performance competitive with Llama-3.3-70B-Instruct. Candidate distillation teacher. |
| **RedSage-8B** | — | More aggressive recipe: 11.8B-token security pretraining corpus, 266K-sample agentically augmented SFT set, preference alignment |

### 1.2 SFT-only on a strong instruct base

Lily-Cyber, SecurityLLM, SecGPT (Clouditera). Cheaper to train, but
underperform pretraining-augmented models on knowledge-heavy benchmarks. In the
FoundationSec agent-evaluation work these were measured alongside DeepHat,
CyberBase, Cyber-Zero, and FoundationSec on large-scale CTI tasks, with success
rates spanning roughly 30–70%.

### 1.3 Encoder-only (for downstream NLP)

Relevant where the production need is NER, classification, or retrieval rather
than generation.

- **SecureBERT 2.0** — corpus ~13× its predecessor; unifies threat intel,
  technical blogs, incident reports, and OSS code
- **CyBERT**, **SecBERT** — older but usable baselines

### 1.4 Uncensored offensive-security models

**WhiteRabbitNeo** (13B / 33B, since evolved into the proprietary **Deep Hat**).
Genuinely dual-use. Rejected for our setting: in a compliance-adjacent
enterprise context the alignment and audit story outweighs the raw capability
ceiling, and the Cisco/RedSage line is far more defensible.

---

## 2. Datasets considered

### 2.1 Continued-pretraining corpora

| Corpus | License | Notes |
|---|---|---|
| **Primus-Pretraining** | ODC-BY | Continual pretraining reported at +15.9% aggregate |
| **RedSage corpus** | released with model | 11.8B tokens, security-filtered |
| **SecureBERT 2.0 corpus** | — | Encoder-oriented, but the recipe (security seed + filtered web crawl + reasoning QA) is informative |

Prior to early 2025 there was effectively no openly licensed security
pretraining corpus, which is why the older literature is SFT-only.

### 2.2 Instruction / SFT data

| Dataset | Notes |
|---|---|
| **Primus-Instruct** (MIT) | Cleanest choice for legitimate use. **Adopted — 30% of our mix.** |
| **CyberLLMInstruct** | Framed as a safety study; authors evaluate against the OWASP Top 10 and warn about weaponisation via open weights + abundant cyber data + cheap GPUs. Read before any release decision; not used as training data. |
| **RedSage SFT set** | 266K samples, agentically generated, structured around expert workflow simulation rather than static Q&A. Largest released instruction collection surveyed. |
| **Tulu-3 / OpenHermes slice** | General instruction data, to protect general IF. **Adopted — 15%.** |
| Safety/refusal slice | **Adopted — 5%.** See §3.4. |

### 2.3 Structured threat-intelligence sources (upstream raw material)

**VulZoo** was the initial candidate aggregation layer (MITRE CVE, NVD, ZDI,
GitHub Advisory, CISA KEV, CWE, CAPEC, ATT&CK, D3FEND, Rapid7 AttackerKB,
Exploit-DB, plus Bugtraq / Full-Disclosure / OSS-Security / Linux-CVE-Announce
mailing lists). **Rejected** — stale research artifact, last updated March 2025.

Replaced with direct upstream pulls:

| Source | Repo / endpoint | Role |
|---|---|---|
| CVE List v5 | `CVEProject/cvelistV5` | CVE spine, daily releases |
| NVD JSON feeds | `fkie-cad/nvd-json-data-feeds` | Analyst enrichment (CVSS, CPE) |
| ATT&CK STIX | `mitre-attack/attack-stix-data` | Technique catalog |
| CISA KEV | single JSON | Known-exploited flag |
| MITRE CWE | XML | Weakness taxonomy |
| MITRE CAPEC | XML (`data/archive/`, **not** `data/xml/`) | Attack patterns |
| GitHub Advisory DB | `github/advisory-database` | Optional; large, off by default |

**CTI-HAL** (human annotations over CTI text) and CVE-QA derivatives were noted
as useful for extractive downstream tasks but not incorporated.

### 2.4 Reasoning distillation data

**Primus-Reasoning** — cleanest open release; recipe mirrors the math/code
distillation literature (sample chains from a strong teacher, filter on verified
outcomes, distil into the student). The reported +15.8% on CISSP is a reasonable
upper bound for what distillation alone buys on certification-style benchmarks.

### 2.5 Evaluation suites

| Benchmark | Coverage |
|---|---|
| **CyberMetric** (-80/-500/-2000/-10000) | De facto MCQ standard; nine domains incl. IAM, IoT, crypto, network, cloud, pentest, compliance. Smaller subsets human-validated. |
| **CTIBench** | NeurIPS 2024 benchmark of record for threat intelligence |
| **SECURE** | Applied reasoning, ICS focus; six datasets (MAET, CWET, KCV, VOOD, RERT, CPST) |
| **CyberSecEval 2** (Meta) | Code vulnerability + prompt injection |
| **SecEval** / **SecQA** | 2K MCQs across software/app/system/web/crypto/memory/network/pentest; SecQA textbook-derived |
| **RedSage-Bench** | Newest; first to add explicit tool-proficiency evaluation |

---

## 3. Training approach

Six stages, implemented as a single `pipeline.py` with subcommand dispatch plus
`config.yaml`, `setup.sh`, `requirements.txt`, `README.md`. Each stage reads
only its own config section.

### 3.1 Stage 1a — Sync

Incremental pull of the §2.3 sources into `~/data/cybersec` (git fetch+reset for
repos, HTTP refresh for single files). Per-source toggles.

### 3.2 Stage 1b — ETL

Normalize into `data/01_cve_records.jsonl`. Filters: `min_year: 2020`,
`min_cvss: 4.0`.

### 3.3 Stages 2–3 — Synthetic generation, filter, dedup

Creator / Reviewer / Checker loop over CVE records, all **local inference**
(no hosted APIs). Served via vLLM on port 8000:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.92
```

Model strings follow LiteLLM conventions (`openai/<served-name>` for
vLLM/TGI/llama.cpp/LM Studio; `ollama_chat/<name>` for Ollama, smoke tests
only). `use_json_mode: false` if the server lacks `response_format`; the
pipeline auto-falls-back on first error.

**Design note.** Creator and Reviewer should ideally be different model
families. Cost roughly doubles, but single-model failure modes are correlated —
a same-family reviewer systematically misses what the creator gets wrong.

**Serving note.** Ministral-3-8B-Instruct-2512 was tried first and produced
token salad from a tokenizer/format mismatch with vLLM; Qwen2.5-7B-Instruct
worked without special flags. Small models drift on field names in structured
output, so the parser needs field aliasing and score coercion (e.g. `"5/5"` →
`5.0`) or the accept rate silently goes to zero.

**Observed:** 160 accepted / 200 attempted (80%) in smoke-test mode.

### 3.4 Stage 4 — Mixing

Target ≈250K chat-formatted examples — roughly the inflection point above which
marginal returns from more SFT data fall off for an 8B base.

| Source | Share | Rationale |
|---|---|---|
| Synthetic vulnerability pairs | 50% | Task-shaped, with audit trail back to source CVE + Reviewer verdict |
| Primus-Instruct | 30% | Different generator, different question distribution — broadens coverage cheaply |
| General instruction (Tulu-3 / OpenHermes) | 15% | Foundation-Sec-Instruct is *already* domain-shifted; 100% security SFT measurably degrades OOD instruction-following |
| Safety / refusal | 5% | CyberLLMInstruct finding: cyber fine-tuning erodes guardrails even when the training content is itself benign |

All four normalized to one chat schema (system / user / assistant), chat
template pre-applied via `tokenizer.apply_chat_template` (Llama-3.1 template),
shuffled into a single JSONL.

### 3.5 Stage 5 — SFT with LoRA (ms-swift)

```yaml
model: fdtn-ai/Foundation-Sec-8B-Instruct
train_type: lora
lora_rank: 32
lora_alpha: 64
lora_dropout: 0.05
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

dataset: /path/to/mixed_sft.jsonl
max_length: 4096
truncation_strategy: right

per_device_train_batch_size: 4
gradient_accumulation_steps: 8           # effective batch = 32
num_train_epochs: 2
learning_rate: 1.5e-4
lr_scheduler_type: cosine
warmup_ratio: 0.03
weight_decay: 0.01

torch_dtype: bfloat16
gradient_checkpointing: true
attn_implementation: flash_attention_2

save_strategy: steps
save_steps: 500
eval_strategy: steps
eval_steps: 500
```

Deliberate choices:

- **r=32, α=64.** r=16 leaves performance on the table; r≥64 stops paying back.
  The base is already domain-adapted, so a high-capacity adapter isn't needed.
- **Target all linear layers, not just attention.** Including `gate_proj`,
  `up_proj`, `down_proj` materially improves SFT quality on knowledge-heavy data.
- **2 epochs default.** Go to 3 only on continued val-loss improvement — cyber
  instruction data has high lexical overlap (vendor names, CVE patterns) and
  overfits faster than expected.
- **Full FT alternative:** drop LR to 5e-6 and watch IF-Eval for catastrophic
  forgetting.
- **Hardware.** Fits a single A40 (48GB) at 4096 context with FA2 + gradient
  checkpointing + bf16. Wall-clock ≈12–18 h for 250K examples × 2 epochs. On the
  dual-4090 setup, FSDP with the same config works; `cpu_offload: false` since
  optimizer states are LoRA-only. Keep a graceful SDPA fallback — a hard
  flash-attn import crash is a common failure on fresh pods.

### 3.6 Stage 6 — Reasoning distillation (optional)

Same shape as the existing GKD pipeline. Teacher options in increasing cost:

1. **Foundation-Sec-8B-Reasoning** — in-family, free, limited headroom at 8B.
   Recommended starting point.
2. Llama-3.3-70B-Instruct or Qwen2.5-72B-Instruct — strong, free with local GPUs
3. Frontier model via API — best quality, ongoing cost

---

## 4. Open problems flagged

1. **Benchmarks overstate field-readiness.** Across seven cybersecurity LLMs on
   large-scale CTI agent tasks, success rates ranged ~30–70%; most models failed
   the majority of agent-style tasks despite strong MCQ numbers.
2. **Safety/utility tradeoff is unresolved.** CyberLLMInstruct makes the
   dual-use risk concrete.
3. **Tool use and long-context reasoning are poorly covered** by the public
   benchmark suite. RedSage-Bench is a step forward but not widely adopted.

## 5. Downstream target

Recommended first production application: **compliance-aware vulnerability
triage** — best fit with the existing regulatory rule-extraction work, and the
per-example audit trail back to source CVE + Reviewer verdict gives model risk
a real answer to "what's in this thing."
