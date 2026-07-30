"""
sft_judge.py — supervised fine-tuning of a small HF model as an LLM judge.

Consumes the JSONL written by build_judge_corpus.py. Curation happens there;
everything here is training-side: templating, augmentation, splitting, loss.

  python sft_judge.py --corpus corpus.jsonl --model Qwen/Qwen2.5-1.5B-Instruct
  python sft_judge.py --corpus corpus.jsonl --task pairwise --lora --bs 4
  python sft_judge.py --eval-only --adapter ./judge-sft --corpus corpus.jsonl

Design decisions
----------------
1. GROUP-AWARE SPLIT. The train/val split is by prompt group, not by row. A
   swapped copy of a pair, and every pointwise row sharing the same instruction,
   land in the same split. Splitting by row leaks the answer across the boundary
   and inflates validation accuracy by several points.

2. POSITION-SWAP AUGMENTATION. Each pairwise row is emitted in both orders with
   the label flipped. This makes the A/B prior exactly 50/50 by construction and
   makes order-consistency measurable at eval time. On by default; --no-swap-aug
   to ablate it.

3. COMPLETION-ONLY LOSS. Gradient flows through the verdict token only, not the
   two candidate responses. Training on the full sequence turns a judge run into
   a very expensive language-modelling run on other models' outputs.

4. LOGIT-SCORED EVAL, NOT FREE DECODING. Verdict accuracy is computed by scoring
   the fixed label set under the model and taking the argmax, so a small model
   that formats its output badly is not scored as if it judged badly. Free-form
   decoding conflates the two.

5. LENGTH BUDGET. A pairwise judge prompt carries two full responses, so drops
   from --max-len are heavy and non-random: long responses are disproportionately
   the ones judges get wrong. The drop rate is reported; raise --max-len or
   filter upstream rather than ignoring it.

Metrics reported: verdict accuracy, per-source accuracy, position-consistency
(fraction of pairs judged the same way under both orders), positional bias
(P(A) - 0.5 on the order-balanced eval set), and MAE for pointwise rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from typing import Any

import torch

# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #

SYSTEM = (
    "You are an impartial evaluator. Judge only the quality of the responses, "
    "not their length, style, or order of presentation."
)

PAIRWISE_USER = """[Instruction]
{instruction}

[Response A]
{a}

[Response B]
{b}

Which response is better? Answer with exactly one of: A, B, tie."""

POINTWISE_USER = """[Instruction]
{instruction}

[Response]
{r}

[Rubric]
{rubric}

Rate the response on a {lo}-{hi} scale. Answer with the integer only."""

PAIRWISE_LABELS = ["A", "B", "tie"]


# --------------------------------------------------------------------------- #
# Corpus -> training examples
# --------------------------------------------------------------------------- #


def group_id(rec: dict) -> str:
    """Stable id shared by a pair and its swap, and by rows on the same prompt."""
    key = (rec["instruction"] or "")[:2000]
    return hashlib.sha1(key.encode("utf-8", "ignore")).hexdigest()[:16]


def render(rec: dict, swapped: bool = False) -> dict | None:
    if rec["task_type"] == "pairwise":
        a, b = rec["responses"][0], rec["responses"][1]
        label = rec.get("label")
        if label not in PAIRWISE_LABELS:
            return None
        if swapped:
            a, b = b, a
            label = {"A": "B", "B": "A", "tie": "tie"}[label]
        user = PAIRWISE_USER.format(instruction=rec["instruction"], a=a, b=b)
        target = label
    else:
        if rec.get("score") is None:
            return None
        lo, hi = rec.get("score_scale") or [1.0, 5.0]
        user = POINTWISE_USER.format(
            instruction=rec["instruction"], r=rec["responses"][0],
            rubric=rec.get("rubric") or "Overall helpfulness to the user.",
            lo=int(lo), hi=int(hi))
        target = str(int(round(float(rec["score"]))))

    return {
        "prompt": [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": user}],
        "completion": [{"role": "assistant", "content": target}],
        "target": target,
        "task_type": rec["task_type"],
        "source": rec["source"],
        "group": group_id(rec),
        "swapped": swapped,
        "score_scale": rec.get("score_scale"),
    }


def load_corpus(path: str, args) -> list[dict]:
    raw: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("contaminated") and not args.keep_contaminated:
                continue
            if args.task != "both" and rec["task_type"] != args.task:
                continue
            if args.annotator and rec.get("annotator") != args.annotator:
                continue
            if rec.get("label") == "tie" and args.drop_ties:
                continue
            raw.append(rec)

    rows: list[dict] = []
    for rec in raw:
        ex = render(rec, swapped=False)
        if ex:
            rows.append(ex)
        if args.swap_aug and rec["task_type"] == "pairwise":
            ex2 = render(rec, swapped=True)
            if ex2:
                rows.append(ex2)
    return rows


def split_by_group(rows: list[dict], val_frac: float, seed: int
                   ) -> tuple[list[dict], list[dict]]:
    groups = sorted({r["group"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_val = max(1, int(len(groups) * val_frac))
    val_groups = set(groups[:n_val])
    train = [r for r in rows if r["group"] not in val_groups]
    val = [r for r in rows if r["group"] in val_groups]
    rng.shuffle(train)
    return train, val


def filter_by_length(rows: list[dict], tok, max_len: int) -> tuple[list[dict], int]:
    kept, dropped = [], 0
    for r in rows:
        text = tok.apply_chat_template(r["prompt"], tokenize=False,
                                       add_generation_prompt=True)
        if len(tok(text).input_ids) + 8 > max_len:
            dropped += 1
            continue
        kept.append(r)
    return kept, dropped


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


@torch.no_grad()
def score_labels(model, tok, prompt_msgs: list[dict], labels: list[str],
                 device: str) -> list[float]:
    """Mean log-likelihood of each candidate label given the prompt."""
    base = tok.apply_chat_template(prompt_msgs, tokenize=False,
                                   add_generation_prompt=True)
    base_ids = tok(base, return_tensors="pt").input_ids.to(device)
    out = []
    for lab in labels:
        lab_ids = tok(lab, add_special_tokens=False,
                      return_tensors="pt").input_ids.to(device)
        ids = torch.cat([base_ids, lab_ids], dim=1)
        logits = model(ids).logits[:, :-1]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        tgt = ids[:, 1:]
        take = logprobs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        out.append(take[0, -lab_ids.shape[1]:].mean().item())
    return out


def evaluate(model, tok, rows: list[dict], device: str, limit: int | None = None
             ) -> dict[str, Any]:
    model.eval()
    rows = rows[:limit] if limit else rows
    hits, total = 0, 0
    per_source: dict[str, Counter] = defaultdict(Counter)
    a_votes, ab_total = 0, 0
    verdict_by_pair: dict[tuple[str, str], str] = {}
    abs_err, n_point = 0.0, 0

    for r in rows:
        if r["task_type"] == "pairwise":
            labels = PAIRWISE_LABELS
        else:
            lo, hi = r.get("score_scale") or [1, 5]
            labels = [str(i) for i in range(int(lo), int(hi) + 1)]

        scores = score_labels(model, tok, r["prompt"], labels, device)
        pred = labels[int(max(range(len(scores)), key=lambda i: scores[i]))]

        ok = pred == r["target"]
        hits += ok
        total += 1
        per_source[r["source"]]["n"] += 1
        per_source[r["source"]]["hit"] += ok

        if r["task_type"] == "pairwise":
            if pred in ("A", "B"):
                a_votes += pred == "A"
                ab_total += 1
            # canonicalise the verdict back to unswapped orientation
            canon = pred if not r["swapped"] else {"A": "B", "B": "A", "tie": "tie"}[pred]
            verdict_by_pair.setdefault((r["group"], r["target"]), canon)
        else:
            abs_err += abs(float(pred) - float(r["target"]))
            n_point += 1

    consistent = pairs = 0
    seen: dict[str, list[str]] = defaultdict(list)
    for (g, _), v in verdict_by_pair.items():
        seen[g].append(v)
    for g, vs in seen.items():
        if len(vs) >= 2:
            pairs += 1
            consistent += len(set(vs)) == 1

    return {
        "n": total,
        "accuracy": hits / total if total else 0.0,
        "position_bias_P(A)-0.5": (a_votes / ab_total - 0.5) if ab_total else None,
        "position_consistency": consistent / pairs if pairs else None,
        "pointwise_mae": abs_err / n_point if n_point else None,
        "per_source": {s: round(c["hit"] / c["n"], 4) for s, c in per_source.items()},
    }


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                    help="small instruct models that work well here: "
                         "Qwen/Qwen2.5-0.5B-Instruct, Qwen/Qwen2.5-1.5B-Instruct, "
                         "meta-llama/Llama-3.2-1B-Instruct, HuggingFaceTB/SmolLM2-1.7B-Instruct")
    ap.add_argument("--out", default="./judge-sft")
    ap.add_argument("--task", choices=["pairwise", "pointwise", "both"], default="both")
    ap.add_argument("--annotator", choices=["human", "llm", "mixed"], default=None)
    ap.add_argument("--drop-ties", action="store_true")
    ap.add_argument("--keep-contaminated", action="store_true")
    ap.add_argument("--no-swap-aug", dest="swap_aug", action="store_false")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--eval-limit", type=int, default=500)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--adapter", default=None, help="checkpoint to load for --eval-only")
    args = ap.parse_args()

    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    rows = load_corpus(args.corpus, args)
    rows, dropped = filter_by_length(rows, tok, args.max_len)
    train_rows, val_rows = split_by_group(rows, args.val_frac, args.seed)
    print(f"train={len(train_rows)}  val={len(val_rows)}  "
          f"dropped_over_{args.max_len}={dropped} "
          f"({dropped / max(1, dropped + len(rows)):.1%})")
    print("train label mix:", Counter(r["target"] for r in train_rows).most_common(8))
    print("train sources:", Counter(r["source"] for r in train_rows).most_common())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    if args.eval_only:
        model = AutoModelForCausalLM.from_pretrained(args.adapter or args.model,
                                                     dtype=dtype).to(device)
        print(json.dumps(evaluate(model, tok, val_rows, device, args.eval_limit),
                         indent=2))
        return

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.config.use_cache = False

    peft_config = None
    if args.lora:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"])

    from trl import SFTConfig, SFTTrainer

    keep = ["prompt", "completion"]
    train_ds = Dataset.from_list([{k: r[k] for k in keep} for r in train_rows])
    val_ds = Dataset.from_list([{k: r[k] for k in keep} for r in val_rows])

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=(device == "cuda"),
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="epoch",
        max_length=args.max_len,
        completion_only_loss=True,   # loss on the verdict token only
        packing=False,               # packing is incompatible with the above
        gradient_checkpointing=True,
        report_to="none",
        seed=args.seed,
    )

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds,
                         eval_dataset=val_ds, processing_class=tok,
                         peft_config=peft_config)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)

    metrics = evaluate(trainer.model, tok, val_rows, device, args.eval_limit)
    print(json.dumps(metrics, indent=2))
    with open(f"{args.out}/judge_eval.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
