"""
build_judge_corpus.py — curate a single unified corpus for LLM-as-a-judge training.

Curation only. No augmentation, no prompt templating, no train/eval split by ratio.
Those are training-side decisions and belong in a separate program.

What it does
------------
  1. Pulls every public judge/preference source into ONE record schema that holds
     both pairwise and pointwise supervision (see SCHEMA below).
  2. Tags provenance: human vs. LLM annotation, license, language, score scale.
  3. Deduplicates on (instruction, response set).
  4. Decontaminates against the held-out judge benchmarks, since RewardBench v1,
     LLMBar and MT-Bench prompts leak into several training mixes.
  5. Writes JSONL (or Parquet) plus a manifest with per-source counts.

Schema
------
  uid          sha1(source | instruction | responses)
  source       short source key
  task_type    "pairwise" | "pointwise"
  instruction  the prompt being responded to
  responses    [a, b] for pairwise; [r] for pointwise
  label        "A" | "B" | "tie" | None      (pairwise)
  score        float | None                  (pointwise)
  score_scale  [min, max] | None             (pointwise)
  margin       float | None                  preference strength, if the source has one
  rubric       str | None
  reference    str | None
  critique     str | None                    free-text feedback, if present
  annotator    "human" | "llm" | "mixed"
  license      str
  lang         str | None
  contaminated bool                          set by the decontamination pass

Usage
-----
  python build_judge_corpus.py --inspect nvidia/HelpSteer3
  python build_judge_corpus.py --sources all --out corpus.jsonl
  python build_judge_corpus.py --sources helpsteer3,arena --annotator human \\
      --max-per-source 20000 --out human_only.jsonl
  python build_judge_corpus.py --sources all --no-decontaminate --format parquet \\
      --out corpus.parquet

Field names on the Hub drift between dataset releases. Run --inspect on a source
before trusting its adapter, and fix the adapter rather than patching downstream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Iterator

from datasets import load_dataset

# --------------------------------------------------------------------------- #
# Record
# --------------------------------------------------------------------------- #


@dataclass
class Record:
    source: str
    task_type: str
    instruction: str
    responses: list[str]
    annotator: str
    license: str
    label: str | None = None
    score: float | None = None
    score_scale: list[float] | None = None
    margin: float | None = None
    rubric: str | None = None
    reference: str | None = None
    critique: str | None = None
    lang: str | None = None
    contaminated: bool = False
    uid: str = field(default="", init=False)

    def __post_init__(self) -> None:
        key = self.source + "|" + self.instruction + "|" + "|".join(self.responses)
        self.uid = hashlib.sha1(key.encode("utf-8", "ignore")).hexdigest()


def _flatten(ctx: Any) -> str:
    """Turn a message list or plain string into a single prompt string."""
    if isinstance(ctx, str):
        return ctx
    if isinstance(ctx, list):
        parts = []
        for turn in ctx:
            if isinstance(turn, dict):
                parts.append(f"{turn.get('role', '')}: {turn.get('content', '')}".strip())
            else:
                parts.append(str(turn))
        return "\n".join(parts)
    return "" if ctx is None else str(ctx)


def _last_user(ctx: Any) -> str:
    """Prompt text for chosen/rejected message-list formats."""
    if isinstance(ctx, list):
        for turn in reversed(ctx):
            if isinstance(turn, dict) and turn.get("role") == "user":
                return turn.get("content", "")
    return _flatten(ctx)


def _assistant_text(msgs: Any) -> str:
    if isinstance(msgs, list):
        for turn in reversed(msgs):
            if isinstance(turn, dict) and turn.get("role") == "assistant":
                return turn.get("content", "")
    return _flatten(msgs)


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #


def load_judgelm() -> Iterator[Record]:
    """BAAI/JudgeLM-100K — GPT-4 pairwise pseudo-labels with critiques.

    'text' is "<score1> <score2>\\n<critique>", scores on 1-10.
    """
    ds = load_dataset("BAAI/JudgeLM-100K", split="train", streaming=True)
    for r in ds:
        text = r.get("text") or ""
        head, _, critique = text.partition("\n")
        toks = head.split()
        if len(toks) < 2:
            continue
        try:
            s1, s2 = float(toks[0]), float(toks[1])
        except ValueError:
            continue
        yield Record(
            source="judgelm",
            task_type="pairwise",
            instruction=r.get("question_body", "") or "",
            responses=[r.get("answer1_body", "") or "", r.get("answer2_body", "") or ""],
            label="tie" if s1 == s2 else ("A" if s1 > s2 else "B"),
            margin=abs(s1 - s2),
            critique=critique.strip() or None,
            annotator="llm",
            license="cc-by-nc-4.0",
            lang="en",
        )


def load_feedback_collection() -> Iterator[Record]:
    """prometheus-eval/Feedback-Collection — pointwise, 1-5, rubric + reference."""
    ds = load_dataset("prometheus-eval/Feedback-Collection", split="train", streaming=True)
    for r in ds:
        out = r.get("output", "") or ""
        feedback, _, score_str = out.rpartition("[RESULT]")
        try:
            score = float(score_str.strip())
        except ValueError:
            continue
        yield Record(
            source="feedback_collection",
            task_type="pointwise",
            instruction=r.get("orig_instruction", "") or "",
            responses=[r.get("orig_response", "") or ""],
            score=score,
            score_scale=[1.0, 5.0],
            rubric=r.get("orig_criteria"),
            reference=r.get("orig_reference_answer"),
            critique=feedback.strip() or None,
            annotator="llm",
            license="cc-by-4.0",
            lang="en",
        )


def load_preference_collection() -> Iterator[Record]:
    """prometheus-eval/Preference-Collection — pairwise with rubric + feedback."""
    ds = load_dataset("prometheus-eval/Preference-Collection", split="train", streaming=True)
    for r in ds:
        out = (r.get("output") or "").strip()
        feedback, _, verdict = out.rpartition("[RESULT]")
        verdict = verdict.strip().upper()
        if verdict not in ("A", "B"):
            verdict = "A" if out.endswith("A") else ("B" if out.endswith("B") else "")
        if verdict not in ("A", "B"):
            continue
        yield Record(
            source="preference_collection",
            task_type="pairwise",
            instruction=r.get("orig_instruction", "") or "",
            responses=[r.get("orig_response_A", "") or "", r.get("orig_response_B", "") or ""],
            label=verdict,
            rubric=r.get("orig_criteria"),
            reference=r.get("orig_reference_answer"),
            critique=feedback.strip() or None,
            annotator="llm",
            license="cc-by-4.0",
            lang="en",
        )


def load_offsetbias() -> Iterator[Record]:
    """NCSOFT/offsetbias — bias counter-examples, label 1 or 2."""
    ds = load_dataset("NCSOFT/offsetbias", split="train", streaming=True)
    for r in ds:
        try:
            lab = int(r["label"])
        except (KeyError, TypeError, ValueError):
            continue
        yield Record(
            source="offsetbias",
            task_type="pairwise",
            instruction=r.get("instruction", "") or "",
            responses=[r.get("output_1", "") or "", r.get("output_2", "") or ""],
            label="A" if lab == 1 else "B",
            annotator="mixed",
            license="see-dataset-card",
            lang="en",
        )


def load_helpsteer3_preference() -> Iterator[Record]:
    """nvidia/HelpSteer3 (preference) — human, signed -3..3, negative favours r1."""
    ds = load_dataset("nvidia/HelpSteer3", "preference", split="train", streaming=True)
    for r in ds:
        pref = r.get("overall_preference")
        if pref is None:
            continue
        pref = float(pref)
        yield Record(
            source="helpsteer3_pref",
            task_type="pairwise",
            instruction=_flatten(r.get("context")),
            responses=[r.get("response1", "") or "", r.get("response2", "") or ""],
            label="tie" if pref == 0 else ("A" if pref < 0 else "B"),
            margin=abs(pref),
            annotator="human",
            license="cc-by-4.0",
            lang=r.get("language"),
        )


_HELPFULNESS = {"not": 1.0, "slightly": 2.0, "partially": 3.0,
                "mostly": 4.0, "perfectly": 5.0}
_HELP_RE = re.compile(r"the response is (not|slightly|partially|mostly|perfectly) helpful",
                      re.IGNORECASE)


def load_helpsteer3_feedback() -> Iterator[Record]:
    """nvidia/HelpSteer3 (feedback) — free-text critiques per response.

    Feedback is written to begin with 'The response is {not/slightly/partially/
    mostly/perfectly} helpful', so an ordinal 1-5 score is recoverable. Rows where
    the phrase is absent keep the critique with score=None.
    """
    ds = load_dataset("nvidia/HelpSteer3", "feedback", split="train", streaming=True)
    for r in ds:
        prompt = _flatten(r.get("context"))
        for idx in (1, 2):
            resp = r.get(f"response{idx}")
            fb = r.get(f"feedback{idx}")
            if not resp or not fb:
                continue
            texts = fb if isinstance(fb, list) else [fb]
            for t in texts:
                if not isinstance(t, str):
                    continue
                m = _HELP_RE.search(t)
                yield Record(
                    source="helpsteer3_feedback",
                    task_type="pointwise",
                    instruction=prompt,
                    responses=[resp],
                    score=_HELPFULNESS[m.group(1).lower()] if m else None,
                    score_scale=[1.0, 5.0] if m else None,
                    critique=t,
                    annotator="human",
                    license="cc-by-4.0",
                    lang=r.get("language"),
                )


def load_ultrafeedback(view: str = "both") -> Iterator[Record]:
    """openbmb/UltraFeedback — GPT-4 scores 1-10 over 4 completions per prompt.

    view: 'pointwise' (one record per completion), 'pairwise' (best vs worst),
    or 'both'.
    """
    ds = load_dataset("openbmb/UltraFeedback", split="train", streaming=True)
    for r in ds:
        instr = r.get("instruction", "") or ""
        comps = [c for c in (r.get("completions") or [])
                 if c.get("overall_score") is not None and c.get("response")]
        if not comps:
            continue
        if view in ("pointwise", "both"):
            for c in comps:
                yield Record(
                    source="ultrafeedback",
                    task_type="pointwise",
                    instruction=instr,
                    responses=[c["response"]],
                    score=float(c["overall_score"]),
                    score_scale=[1.0, 10.0],
                    critique=c.get("critique"),
                    annotator="llm",
                    license="mit",
                    lang="en",
                )
        if view in ("pairwise", "both") and len(comps) >= 2:
            ranked = sorted(comps, key=lambda c: float(c["overall_score"]), reverse=True)
            best, worst = ranked[0], ranked[-1]
            d = float(best["overall_score"]) - float(worst["overall_score"])
            if d == 0:
                continue
            yield Record(
                source="ultrafeedback",
                task_type="pairwise",
                instruction=instr,
                responses=[best["response"], worst["response"]],
                label="A",
                margin=d,
                annotator="llm",
                license="mit",
                lang="en",
            )


def load_arena() -> Iterator[Record]:
    """lmarena-ai/arena-human-preference-100k — real human votes, ties included.

    Prompt/response columns are JSON-encoded lists in some releases; both forms
    are handled.
    """
    ds = load_dataset("lmarena-ai/arena-human-preference-100k", split="train",
                      streaming=True)

    def _txt(v: Any) -> str:
        if isinstance(v, str) and v.startswith("["):
            try:
                return "\n".join(str(x) for x in json.loads(v))
            except json.JSONDecodeError:
                return v
        return _flatten(v)

    for r in ds:
        if r.get("winner_tie"):
            label = "tie"
        elif r.get("winner_model_a"):
            label = "A"
        elif r.get("winner_model_b"):
            label = "B"
        else:
            w = str(r.get("winner", "")).lower()
            label = {"model_a": "A", "model_b": "B", "tie": "tie"}.get(w, "")
        if label not in ("A", "B", "tie"):
            continue
        yield Record(
            source="arena_100k",
            task_type="pairwise",
            instruction=_txt(r.get("prompt")),
            responses=[_txt(r.get("response_a")), _txt(r.get("response_b"))],
            label=label,
            annotator="human",
            license="see-dataset-card",
            lang=r.get("language"),
        )


def _chosen_rejected(hf_id: str, source: str, annotator: str,
                     lic: str, config: str | None = None) -> Iterator[Record]:
    """Generic adapter for mixes stored as chosen/rejected message lists."""
    ds = load_dataset(hf_id, config, split="train", streaming=True)
    for r in ds:
        chosen, rejected = r.get("chosen"), r.get("rejected")
        if chosen is None or rejected is None:
            continue
        prompt = r.get("prompt") or _last_user(chosen)
        yield Record(
            source=source,
            task_type="pairwise",
            instruction=_flatten(prompt),
            responses=[_assistant_text(chosen), _assistant_text(rejected)],
            label="A",
            annotator=annotator,
            license=lic,
            lang="en",
        )


def load_skywork() -> Iterator[Record]:
    return _chosen_rejected("Skywork/Skywork-Reward-Preference-80K-v0.2",
                            "skywork_80k", "mixed", "see-dataset-card")


def load_tulu3() -> Iterator[Record]:
    return _chosen_rejected("allenai/llama-3.1-tulu-3-8b-preference-mixture",
                            "tulu3_mix", "mixed", "odc-by")


def load_olmo2() -> Iterator[Record]:
    return _chosen_rejected("allenai/olmo-2-1124-13b-preference-mix",
                            "olmo2_mix", "mixed", "odc-by")


ADAPTERS: dict[str, Callable[[], Iterator[Record]]] = {
    "judgelm": load_judgelm,
    "feedback_collection": load_feedback_collection,
    "preference_collection": load_preference_collection,
    "offsetbias": load_offsetbias,
    "helpsteer3": load_helpsteer3_preference,
    "helpsteer3_feedback": load_helpsteer3_feedback,
    "ultrafeedback": load_ultrafeedback,
    "arena": load_arena,
    "skywork": load_skywork,
    "tulu3": load_tulu3,
    "olmo2": load_olmo2,
}

# Held-out judge benchmarks. Never curate these into the training corpus; their
# prompts are used to decontaminate it.
EVAL_SETS: list[tuple[str, str | None, str, str]] = [
    # (hf_id, config, split, prompt_column)
    ("allenai/reward-bench-2", None, "test", "prompt"),
    ("allenai/reward-bench", None, "raw", "prompt"),
    ("ScalerLab/JudgeBench", None, "gpt", "question"),
    ("ScalerLab/JudgeBench", None, "claude", "question"),
    ("princeton-nlp/LLMBar", "Natural", "test", "input"),
]


# --------------------------------------------------------------------------- #
# Dedup + decontamination
# --------------------------------------------------------------------------- #


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def prompt_hash(text: str) -> str:
    return hashlib.sha1(_norm(text)[:512].encode("utf-8", "ignore")).hexdigest()


def build_eval_prompt_hashes(verbose: bool = True) -> set[str]:
    hashes: set[str] = set()
    for hf_id, config, split, col in EVAL_SETS:
        try:
            ds = load_dataset(hf_id, config, split=split)
        except Exception as exc:  # dataset moved, gated, or split renamed
            if verbose:
                print(f"  ! skipped {hf_id}:{split} ({type(exc).__name__})",
                      file=sys.stderr)
            continue
        n = 0
        for r in ds:
            text = r.get(col)
            if isinstance(text, str) and text.strip():
                hashes.add(prompt_hash(text))
                n += 1
        if verbose:
            print(f"  + {hf_id}:{split} -> {n} eval prompts", file=sys.stderr)
    return hashes


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def curate(args) -> tuple[list[dict], dict]:
    names = (list(ADAPTERS) if args.sources.strip() == "all"
             else [s.strip() for s in args.sources.split(",") if s.strip()])
    unknown = [n for n in names if n not in ADAPTERS]
    if unknown:
        raise SystemExit(f"unknown source(s): {unknown}. Known: {list(ADAPTERS)}")

    eval_hashes = set() if args.no_decontaminate else build_eval_prompt_hashes()
    print(f"decontamination set: {len(eval_hashes)} prompt hashes", file=sys.stderr)

    seen: set[str] = set()
    rows: list[dict] = []
    stats: dict[str, Counter] = {}

    for name in names:
        c = stats.setdefault(name, Counter())
        kept = 0
        gen = (ADAPTERS[name]() if name != "ultrafeedback"
               else load_ultrafeedback(args.ultrafeedback_view))
        for rec in gen:
            c["seen"] += 1
            if not rec.instruction.strip() or not all(r.strip() for r in rec.responses):
                c["empty"] += 1
                continue
            if args.annotator and rec.annotator != args.annotator:
                c["annotator_filtered"] += 1
                continue
            if args.task_type and rec.task_type != args.task_type:
                c["task_filtered"] += 1
                continue
            if rec.uid in seen:
                c["dup"] += 1
                continue
            seen.add(rec.uid)
            if eval_hashes and prompt_hash(rec.instruction) in eval_hashes:
                c["contaminated"] += 1
                if args.drop_contaminated:
                    continue
                rec.contaminated = True
            rows.append(asdict(rec))
            c["kept"] += 1
            c[rec.task_type] += 1
            if rec.label:
                c[f"label_{rec.label}"] += 1
            kept += 1
            if args.max_per_source and kept >= args.max_per_source:
                break
        print(f"{name:24s} kept={c['kept']:>8d}  dup={c['dup']:>6d}  "
              f"contam={c['contaminated']:>5d}", file=sys.stderr)

    manifest = {
        "sources": names,
        "total_records": len(rows),
        "ultrafeedback_view": args.ultrafeedback_view,
        "decontaminated": not args.no_decontaminate,
        "dropped_contaminated": args.drop_contaminated,
        "filters": {"annotator": args.annotator, "task_type": args.task_type,
                    "max_per_source": args.max_per_source},
        "per_source": {k: dict(v) for k, v in stats.items()},
    }
    return rows, manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inspect", metavar="HF_ID",
                    help="print one raw record from a Hub dataset and exit")
    ap.add_argument("--inspect-config", default=None)
    ap.add_argument("--sources", default="all",
                    help=f"'all' or comma-separated from: {','.join(ADAPTERS)}")
    ap.add_argument("--out", default="judge_corpus.jsonl")
    ap.add_argument("--format", choices=["jsonl", "parquet"], default="jsonl")
    ap.add_argument("--max-per-source", type=int, default=None)
    ap.add_argument("--annotator", choices=["human", "llm", "mixed"], default=None,
                    help="keep only this annotation provenance")
    ap.add_argument("--task-type", choices=["pairwise", "pointwise"], default=None)
    ap.add_argument("--ultrafeedback-view", choices=["pairwise", "pointwise", "both"],
                    default="both")
    ap.add_argument("--no-decontaminate", action="store_true",
                    help="skip the overlap check against held-out judge benchmarks")
    ap.add_argument("--drop-contaminated", action="store_true",
                    help="drop overlapping rows instead of flagging them")
    args = ap.parse_args()

    if args.inspect:
        ds = load_dataset(args.inspect, args.inspect_config, split="train",
                          streaming=True)
        print(json.dumps(next(iter(ds)), indent=2, default=str)[:6000])
        return

    rows, manifest = curate(args)

    if args.format == "parquet":
        import pandas as pd
        pd.DataFrame(rows).to_parquet(args.out, index=False)
    else:
        with open(args.out, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest_path = args.out.rsplit(".", 1)[0] + ".manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nwrote {len(rows)} records -> {args.out}", file=sys.stderr)
    print(f"manifest -> {manifest_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
