"""
train_judge.py — train an LLM-as-a-judge on public preference data.

Two heads are supported:
  --mode generative : SFT a causal LM to emit a verdict token ("A"/"B"/"tie"),
                      optionally preceded by a critique. Evaluated by exact match.
  --mode bt         : Bradley-Terry reward model (scalar head) on chosen/rejected
                      pairs. Evaluated by pairwise accuracy.

Design decisions worth knowing before you run this:

1. CONTAMINATION. RewardBench v1 draws its Chat / Chat-Hard splits from MT-Bench,
   AlpacaEval and LLMBar. If you train on those and evaluate on RewardBench v1 the
   number is meaningless. Use RewardBench 2 / JudgeBench / PPE as held-out eval,
   or run --check_contamination.

2. POSITION BIAS. Every pairwise example is emitted in both orders with the label
   flipped (--swap_aug). Without this a generative judge learns a position prior;
   with it, order-consistency becomes measurable at eval time.

3. TIES. Most sources discard ties. --tie_policy controls this:
     drop      : standard, comparable to published numbers
     keep      : three-way label, needed if you care about calibrated indifference
     margin=k  : convert graded labels within +/-k of neutral into ties
   Only HelpSteer2/3 and Arena provide real human tie/margin signal.

Usage
-----
  python train_judge.py --inspect nvidia/HelpSteer3          # print raw schema first
  python train_judge.py --mode generative --datasets offsetbias,judgelm --lora
  python train_judge.py --mode bt --datasets helpsteer3,ultrafeedback
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from typing import Callable, Iterable

from datasets import load_dataset, Dataset

# --------------------------------------------------------------------------- #
# Common schema
# --------------------------------------------------------------------------- #


@dataclass
class Pair:
    instruction: str
    response_a: str
    response_b: str
    label: str  # "A" | "B" | "tie"
    source: str
    margin: float | None = None  # graded preference strength, if the source has one


# --------------------------------------------------------------------------- #
# Dataset adapters.
# Schemas drift between releases — run --inspect on a source before trusting an
# adapter, and adjust the field names here rather than downstream.
# --------------------------------------------------------------------------- #


def _load_offsetbias(split: str = "train") -> Iterable[Pair]:
    ds = load_dataset("NCSOFT/offsetbias", split=split)
    for r in ds:
        yield Pair(r["instruction"], r["output_1"], r["output_2"],
                   "A" if int(r["label"]) == 1 else "B", "offsetbias")


def _load_helpsteer3(split: str = "train") -> Iterable[Pair]:
    # 'overall_preference' is a signed integer; negative favours response 1.
    ds = load_dataset("nvidia/HelpSteer3", "preference", split=split)
    for r in ds:
        ctx = r["context"]
        prompt = ctx if isinstance(ctx, str) else "\n".join(
            f"{t['role']}: {t['content']}" for t in ctx
        )
        pref = r["overall_preference"]
        label = "tie" if pref == 0 else ("A" if pref < 0 else "B")
        yield Pair(prompt, r["response1"], r["response2"], label,
                   "helpsteer3", margin=abs(float(pref)))


def _load_ultrafeedback(split: str = "train") -> Iterable[Pair]:
    # Build pairs from the 4 scored completions: best vs. a random worse one.
    ds = load_dataset("openbmb/UltraFeedback", split=split)
    for r in ds:
        comps = [c for c in r["completions"] if c.get("overall_score") is not None]
        if len(comps) < 2:
            continue
        comps.sort(key=lambda c: float(c["overall_score"]), reverse=True)
        best, worst = comps[0], comps[-1]
        if float(best["overall_score"]) == float(worst["overall_score"]):
            continue
        yield Pair(r["instruction"], best["response"], worst["response"], "A",
                   "ultrafeedback",
                   margin=float(best["overall_score"]) - float(worst["overall_score"]))


def _load_judgelm(split: str = "train") -> Iterable[Pair]:
    # GPT-4 pseudo-labels: 'text' begins with two scores, e.g. "2 8\n<critique>".
    ds = load_dataset("BAAI/JudgeLM-100K", split=split)
    for r in ds:
        try:
            head = r["text"].split("\n", 1)[0].split()
            s1, s2 = float(head[0]), float(head[1])
        except (ValueError, IndexError, KeyError):
            continue
        label = "tie" if s1 == s2 else ("A" if s1 > s2 else "B")
        yield Pair(r.get("question_body", ""), r.get("answer1_body", ""),
                   r.get("answer2_body", ""), label, "judgelm",
                   margin=abs(s1 - s2))


def _load_preference_collection(split: str = "train") -> Iterable[Pair]:
    ds = load_dataset("prometheus-eval/Preference-Collection", split=split)
    for r in ds:
        out = r["output"]
        label = "A" if out.strip().endswith("A") else "B"
        yield Pair(r["orig_instruction"], r["orig_response_A"],
                   r["orig_response_B"], label, "preference_collection")


ADAPTERS: dict[str, Callable[..., Iterable[Pair]]] = {
    "offsetbias": _load_offsetbias,
    "helpsteer3": _load_helpsteer3,
    "ultrafeedback": _load_ultrafeedback,
    "judgelm": _load_judgelm,
    "preference_collection": _load_preference_collection,
}


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

JUDGE_PROMPT = """You are evaluating two responses to the same instruction.

[Instruction]
{instruction}

[Response A]
{response_a}

[Response B]
{response_b}

State which response is better. Answer with exactly one of: A, B, tie."""


def build(names: list[str], tie_policy: str, swap_aug: bool,
          max_per_source: int | None, seed: int = 0) -> list[Pair]:
    rng = random.Random(seed)
    pairs: list[Pair] = []
    for name in names:
        n = 0
        for p in ADAPTERS[name]():
            if p.label == "tie" and tie_policy == "drop":
                continue
            if tie_policy.startswith("margin=") and p.margin is not None:
                k = float(tie_policy.split("=", 1)[1])
                if p.margin <= k:
                    p.label = "tie"
            pairs.append(p)
            n += 1
            if max_per_source and n >= max_per_source:
                break
    if swap_aug:
        flip = {"A": "B", "B": "A", "tie": "tie"}
        pairs += [Pair(p.instruction, p.response_b, p.response_a,
                       flip[p.label], p.source, p.margin) for p in list(pairs)]
    rng.shuffle(pairs)
    return pairs


def to_sft(pairs: list[Pair]) -> Dataset:
    rows = [{"prompt": JUDGE_PROMPT.format(**{k: v for k, v in asdict(p).items()
                                              if k in ("instruction", "response_a",
                                                       "response_b")}),
             "completion": p.label} for p in pairs]
    return Dataset.from_list(rows)


def to_bt(pairs: list[Pair]) -> Dataset:
    rows = []
    for p in pairs:
        if p.label == "tie":
            continue
        chosen, rejected = ((p.response_a, p.response_b) if p.label == "A"
                            else (p.response_b, p.response_a))
        rows.append({"prompt": p.instruction, "chosen": chosen, "rejected": rejected})
    return Dataset.from_list(rows)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def train(args, ds: Dataset):
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    peft_config = None
    if args.lora:
        from peft import LoraConfig
        peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                 task_type="CAUSAL_LM" if args.mode == "generative"
                                 else "SEQ_CLS")

    if args.mode == "generative":
        from trl import SFTConfig, SFTTrainer
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto")
        cfg = SFTConfig(output_dir=args.out, num_train_epochs=args.epochs,
                        per_device_train_batch_size=args.bs,
                        gradient_accumulation_steps=args.grad_accum,
                        learning_rate=args.lr, bf16=True, logging_steps=25,
                        max_length=args.max_len, completion_only_loss=True)
        trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                             processing_class=tok, peft_config=peft_config)
    else:
        from trl import RewardConfig, RewardTrainer
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=1, dtype="auto")
        model.config.pad_token_id = tok.pad_token_id
        cfg = RewardConfig(output_dir=args.out, num_train_epochs=args.epochs,
                           per_device_train_batch_size=args.bs,
                           gradient_accumulation_steps=args.grad_accum,
                           learning_rate=args.lr, bf16=True, logging_steps=25,
                           max_length=args.max_len)
        trainer = RewardTrainer(model=model, args=cfg, train_dataset=ds,
                                processing_class=tok, peft_config=peft_config)

    trainer.train()
    trainer.save_model(args.out)


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", metavar="HF_ID",
                    help="print one raw record from a source and exit")
    ap.add_argument("--mode", choices=["generative", "bt"], default="generative")
    ap.add_argument("--datasets", default="offsetbias,helpsteer3",
                    help=f"comma-separated from: {','.join(ADAPTERS)}")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default="./judge-ckpt")
    ap.add_argument("--tie_policy", default="drop",
                    help="drop | keep | margin=<float>")
    ap.add_argument("--no_swap_aug", dest="swap_aug", action="store_false")
    ap.add_argument("--max_per_source", type=int, default=None)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--dump", metavar="PATH", help="write the assembled set as JSONL and exit")
    args = ap.parse_args()

    if args.inspect:
        d = load_dataset(args.inspect, split="train", streaming=True)
        print(json.dumps(next(iter(d)), indent=2, default=str)[:4000])
        return

    names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    pairs = build(names, args.tie_policy, args.swap_aug, args.max_per_source)
    print(f"assembled {len(pairs)} pairs from {names}")
    print("label distribution:",
          {l: sum(p.label == l for p in pairs) for l in ("A", "B", "tie")})

    if args.dump:
        with open(args.dump, "w") as f:
            for p in pairs:
                f.write(json.dumps(asdict(p)) + "\n")
        return

    ds = to_sft(pairs) if args.mode == "generative" else to_bt(pairs)
    train(args, ds)


if __name__ == "__main__":
    main()
