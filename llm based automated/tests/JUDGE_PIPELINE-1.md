# LLM-as-a-Judge: Dataset Survey, Curation Pipeline, and SFT Protocol

Working record. Covers the public data landscape, the three-stage pipeline built
against it, and the rationale for each design decision — including the ones
deliberately deferred or rejected.

---

## 1. Objective

Fine-tune a small open-weight model (~1–4B) to act as an evaluator that, given an
instruction and one or two candidate responses, emits either a pairwise verdict
(`A` / `B` / `tie`) or a pointwise rubric score. The intended downstream use is
data curation, so calibration and order-invariance matter as much as raw accuracy.

Constraints that shaped everything below:

- **LoRA on a 4B base**, not full fine-tuning of a frontier model.
- **Curation must be separable from training.** Corpus construction, subset
  selection, and SFT are three independent programs with file interfaces between
  them, so a curation change does not require re-running training code and a
  training change does not require re-downloading 11 datasets.
- **Held-out benchmarks must stay uncontaminated**, since the evaluation numbers
  are the point of the exercise.

---

## 2. Data landscape

### 2.1 Training corpora

| Source | HF path | Annotator | Supervision | Scale | License |
|---|---|---|---|---|---|
| JudgeLM-100K | `BAAI/JudgeLM-100K` | GPT-4 | pairwise + critique, scores 1–10 | 100K train / 5K val | CC-BY-NC-4.0 |
| Feedback Collection | `prometheus-eval/Feedback-Collection` | GPT-4 | **pointwise** 1–5, rubric + reference | 1K rubrics, 100K responses | CC-BY-4.0 |
| Preference Collection | `prometheus-eval/Preference-Collection` | GPT-4 | pairwise + rubric + feedback | ~200K pairs | CC-BY-4.0 |
| OffsetBias | `NCSOFT/offsetbias` | mixed | pairwise, bias counter-examples | 8,504 | see card |
| HelpSteer3 (preference) | `nvidia/HelpSteer3` | **human** | pairwise, signed −3…3 margin | ~40K, multilingual | CC-BY-4.0 |
| HelpSteer3 (feedback) | `nvidia/HelpSteer3` | **human** | free-text critique, ordinal 1–5 recoverable | ~40K | CC-BY-4.0 |
| UltraFeedback | `openbmb/UltraFeedback` | GPT-4 | **pointwise** 1–10 × 4 completions | 64K prompts / 256K responses | MIT |
| Arena human prefs | `lmarena-ai/arena-human-preference-100k` | **human** | pairwise, real ties | 100K | see card |
| Skywork Reward | `Skywork/Skywork-Reward-Preference-80K-v0.2` | mixed | chosen/rejected | 80K | see card |
| Tülu 3 mix | `allenai/llama-3.1-tulu-3-8b-preference-mixture` | mixed | chosen/rejected | large | ODC-BY |
| OLMo 2 mix | `allenai/olmo-2-1124-13b-preference-mix` | mixed | chosen/rejected | large | ODC-BY |

### 2.2 Held-out evaluation

Never curated into training; their prompts form the decontamination set.

| Benchmark | Path | Notes |
|---|---|---|
| RewardBench 2 | `allenai/reward-bench-2` | 1,865 cases; 1 chosen + 3 rejected; **Tie subset** |
| RewardBench v1 | `allenai/reward-bench` | superseded — see §2.3 |
| JudgeBench | `ScalerLab/JudgeBench` | splits `gpt`, `claude`; objectively-checkable pairs |
| LLMBar | `princeton-nlp/LLMBar` | Natural, Neighbor, GPTInst, GPTOut, Manual |
| RM-Bench | `THU-KEG/RM-Bench` | style-bias sensitivity |
| PPE | `lmarena-ai/*` | 16K human prefs + correctness split |
| Judge-Bench | github, via `aclanthology.org/2025.acl-short.20` | 20 datasets, mixed human scales |
| SummEval | `mteb/summeval` | 1–5 Likert, classic NLG meta-eval |
| MT-Bench votes | `lmsys/mt_bench_human_judgments` | 3K expert votes, explicit ties |

### 2.3 Contamination map

**RewardBench v1's Chat and Chat-Hard splits are drawn from MT-Bench, AlpacaEval,
and LLMBar.** Several public training mixes ingest those same sources. Training on
such a mix and reporting RewardBench v1 is circular.

Decision: **report RewardBench 2 and JudgeBench as headline numbers**; treat v1 as
diagnostic only. Automated overlap checking is built into stage 1 rather than left
as a manual step.

### 2.4 What the survey settled

Only four sources carry **native graded human labels** — SummEval, HelpSteer2/3,
Feedback Collection, Judge-Bench. Only two carry **explicit human tie labels** —
RewardBench 2's Tie subset and MT-Bench votes. Everything else is binary or
LLM-pseudo-labelled. This constrains any study of scoring granularity or judge
indifference and is the reason the schema below keeps pointwise supervision rather
than collapsing everything to pairs.

---

## 3. Pipeline

```
  Hub ──▶ build_judge_corpus.py ──▶ corpus.jsonl (~6 GB, ~2M records)
                                          │
                                          ▼
                          sample_corpus.py ──▶ pool_60k.jsonl
                                          │
                                          ▼
                             sft_judge.py ──▶ ./judge-sft + judge_eval.json
```

Three programs, file interfaces, no shared state. Each writes a manifest JSON
alongside its output recording filters, seeds, and per-stratum counts, so any
result traces back to an exact data provenance.

| Stage | Program | Responsibility |
|---|---|---|
| 1 | `build_judge_corpus.py` | normalise 11 sources, dedup, decontaminate, tag provenance |
| 2 | `sample_corpus.py` | stratified subset selection from the built corpus |
| 3 | `sft_judge.py` | templating, augmentation, splitting, LoRA SFT, judge eval |

`train_judge.py` from the first iteration is **superseded**. It pulled from the Hub
and trained in one process, which conflated stages 1–3 and had no notion of a
corpus file. Delete it; the `--corpus` flag it lacks is the usual symptom of running
it by mistake.

---

## 4. Unified record schema

```jsonc
{
  "uid":          "sha1(source|instruction|responses)",
  "source":       "helpsteer3_pref",
  "task_type":    "pairwise" | "pointwise",
  "instruction":  "...",
  "responses":    ["a", "b"],      // 2 for pairwise, 1 for pointwise
  "label":        "A" | "B" | "tie" | null,
  "score":        4.0,             // pointwise
  "score_scale":  [1.0, 5.0],
  "margin":       2.0,             // preference strength where the source has one
  "rubric":       "...",
  "reference":    "...",
  "critique":     "...",
  "annotator":    "human" | "llm" | "mixed",
  "license":      "cc-by-4.0",
  "lang":         "english",
  "contaminated": false
}
```

**Why one schema holds both supervision types.** Feedback Collection, UltraFeedback,
and HelpSteer3-feedback are pointwise and graded. Forcing them into pairs discards
the scale information — precisely the signal needed for any study of scoring
granularity. The cost is a `task_type` branch in the renderer; the benefit is that
a graded-only or human-only corpus is a filter expression rather than a rebuild.

**Why `annotator` is first-class.** The human/LLM distinction is the main axis of
interest for curation work. `--annotator human` yields HelpSteer3 + Arena; anything
else is pseudo-labelled and inherits the labelling model's biases.

---

## 5. Design decisions

Each is stated as decision → rationale → how to ablate.

### 5.1 Curation (stage 1)

**D1. Flag contamination, do not drop it by default.**
Overlaps against RewardBench/RewardBench 2/JudgeBench/LLMBar prompts set
`contaminated: true`. Dropping silently loses the ability to *measure* how much
each mix leaks — itself a reportable result. `--drop-contaminated` to remove.

**D2. Dedup on `(source, instruction, responses)`, not on prompt alone.**
The same prompt legitimately recurs with different response pairs. Hashing the
prompt alone would destroy most of Arena and UltraFeedback.

**D3. Recover ordinal scores from HelpSteer3 free-text feedback.**
Critiques are written to open with "The response is {not/slightly/partially/
mostly/perfectly} helpful", so a 1–5 ordinal is extractable by regex. Rows without
the phrase keep `critique` with `score: null` rather than being dropped. This is the
only route to *human-authored* graded scores at scale.

**D4. UltraFeedback emitted in both views.**
`--ultrafeedback-view {pairwise,pointwise,both}`. The pairwise view is best-vs-worst
by `overall_score`. **Open question:** adjacent-rank pairs would be harder and
arguably more informative; best-vs-worst maximises margin and may make the task too
easy. Not yet tested.

**D5. No augmentation at this stage.** Swap augmentation, templating, and splitting
are training-side. Keeping them out means the corpus is a neutral population that
different training runs can sample differently.

### 5.2 Sampling (stage 2)

**D6. Two-pass byte-offset indexing, never load the corpus.**
Pass 1 records offsets, lengths, and interned stratum keys in `array` objects
(~300 MB for 2M records); pass 2 seeks and copies chosen lines. Sorted offsets keep
seeks sequential.

**D7. Stratify on (source × language × task_type × label × length quartile).**
Response length is the dominant confound: a uniform draw from a corpus dominated by
short English chat produces a judge that degrades on exactly the long-form and code
slices the benchmarks probe. Verified balanced — 1259/1249/1246/1246 across
quartiles on a 5K draw.

**D8. Default allocation is `power` (n^0.5), not proportional.**
Measured on a synthetic corpus with a realistic skew:

| | arena | ultrafeedback | judgelm | helpsteer3 | offsetbias |
|---|---|---|---|---|---|
| population | 40.3% | 30.1% | 19.8% | 4.9% | 1.0% |
| `proportional` | 41.1% | 30.8% | 19.3% | 4.4% | 0.6% |
| `power` (default) | 31.6% | 26.7% | 21.7% | 10.2% | 4.2% |
| `balanced` | 20.8% | 20.8% | 20.7% | 19.4% | 11.6% |

`proportional` reduces OffsetBias to ~30 of 5,000 rows. OffsetBias is the only
source carrying deliberate bias counter-examples, so proportional allocation
discards the highest-value-per-row data in the corpus. `balanced` overweights small
sources past what their diversity justifies. Use `proportional` only when the
corpus mix is itself the object of study.

**D9. `--exclude` for disjoint draws.**
Excludes uids from a prior sample, verified disjoint. Enables honest scaling curves
(60K vs 120K) where the second run does not re-see the first's rows.

### 5.3 Training (stage 3)

**D10. Group-aware split, not row-level.**
Split is by `sha1(instruction)` group, so a swapped copy and every pointwise row on
the same instruction land in the same split. Row-level splitting leaks the answer
across the boundary and inflates validation accuracy by several points.

**D11. Position-swap augmentation on by default.**
Every pairwise row is emitted in both orders with the label flipped. Makes the A/B
prior exactly 50/50 by construction and makes order-consistency measurable.
`--no-swap-aug` to ablate.

**D12. `completion_only_loss=True`.**
Gradient flows through the verdict token only. Training on the full sequence turns
a judge run into an expensive language-modelling run over other models' outputs.
Requires `packing=False`.

**D13. Logit-scored evaluation, not free decoding.**
Verdict accuracy scores the fixed label set under the model and takes the argmax.
Free-form decoding conflates *judged badly* with *formatted badly* — a distinction
that matters most for exactly the small models in scope here.

**D14. Report the length-drop rate.**
A pairwise prompt carries two full responses, so `--max-len` drops are heavy and
non-random: long responses are disproportionately the ones judges get wrong. The
rate is printed; raise `--max-len` or filter upstream rather than ignoring it.

---

## 6. Scale reasoning

The full corpus is ~6 GB. With completion-only loss, gradient flows through **one
token per example**. At ~3 KB/record that is ~2M records, so total label
information is ~2M × 1.6 bits ≈ **400 KB** — delivered via ~1.5B forward-pass
tokens into ~50M LoRA parameters. One epoch on a 4B model is ≈3.6e19 FLOPs, roughly
30–40 H100-hours.

Empirical anchors point the same way: OffsetBias trains competitive judges on 8,504
samples; HelpSteer3 has ~40K and produced RMs at 82.4% on RM-Bench and 73.7% on
JudgeBench, ~10 points over the prior best; Skywork-V2's ablation across ~20 large
public mixes finds them clustered in a 67–69 average band regardless of size.

**Conclusion:** LoRA on 4B saturates well before 6 GB. Target **50–60K rows**
(≈110K after swap augmentation, 3–5 GPU-hours), then measure whether 2× moves
validation accuracy before scaling. What scale genuinely buys is *domain coverage*,
which is why the answer is stratified sampling rather than a smaller download.

Keep the full corpus on disk regardless: for curation-strategy work it is the
sampling frame, not the training set.

---

## 7. Evaluation protocol

| Metric | Definition | Why |
|---|---|---|
| Verdict accuracy | argmax over scored label set vs. gold | headline |
| Per-source accuracy | same, grouped by `source` | exposes a judge good only on its dominant source |
| Position consistency | fraction of pairs judged identically under both orders | order-invariance; independent of accuracy |
| Positional bias | P(A) − 0.5 on the order-balanced eval set | should be ≈0 given D11; nonzero means the prior survived |
| Pointwise MAE | mean absolute error on graded rows | calibration on the graded subset |

Headline external numbers: **RewardBench 2** and **JudgeBench**. RewardBench v1 is
diagnostic only (§2.3).

---

## 8. Reproduction

```bash
# 1. build the full corpus (once)
python build_judge_corpus.py --sources all --out corpus.jsonl

# 2. draw a stratified training pool
python sample_corpus.py --corpus corpus.jsonl --n 60000 --out pool_60k.jsonl

# 3. LoRA SFT on a small base
python sft_judge.py --corpus pool_60k.jsonl --model Qwen/Qwen3-4B-Instruct \
    --lora --lora-r 32 --bs 4 --grad-accum 8 --epochs 2 --max-len 3072

# scaling curve: disjoint second draw
python sample_corpus.py --corpus corpus.jsonl --n 60000 \
    --exclude pool_60k.jsonl --out pool_60k_b.jsonl

# ablations
python sft_judge.py --corpus pool_60k.jsonl --no-swap-aug ...   # D11
python sft_judge.py --corpus pool_60k.jsonl --drop-ties ...     # tie handling
python sample_corpus.py --corpus corpus.jsonl --annotator human --n 40000 ...  # human-only
```

Inspect any source's raw schema before trusting its adapter — Hub column names
drift between releases:

```bash
python build_judge_corpus.py --inspect nvidia/HelpSteer3 --inspect-config preference
```

---

## 9. Known gaps and open questions

1. **Adapter fragility.** Field names are pinned to current Hub releases. `--inspect`
   is the intended first move on any failure; fix the adapter, not downstream code.
2. **D4 untested.** Best-vs-worst vs. adjacent-rank pairs for UltraFeedback.
3. **Tie policy untested.** Only HelpSteer3 and Arena carry real human tie signal. A
   judge trained tie-free will be systematically overconfident on indifferent pairs,
   but the size of that effect has not been measured here.
4. **Global vs. per-source length quartiles.** Currently global. Per-source quartiles
   would balance within each source's own length distribution — different, arguably
   better for heterogeneous mixes.
5. **Generative vs. discriminative head.** Only the generative judge is implemented in
   stage 3. A Bradley-Terry scalar head over the same corpus is a natural control and
   is cheaper to evaluate.
6. **Licence heterogeneity.** JudgeLM-100K is CC-BY-NC. Any corpus including it
   inherits a non-commercial restriction; filter by `license` if that matters.
7. **Multilingual coverage** comes almost entirely from HelpSteer3 and Arena; the
   pseudo-labelled sources are English-dominant.
