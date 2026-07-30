"""
sample_corpus.py — draw a stratified subset from an existing judge corpus.

Works on the JSONL written by build_judge_corpus.py. Never loads the corpus into
memory: pass 1 indexes byte offsets and stratum keys, pass 2 seeks and copies only
the sampled lines. A 6 GB / ~2M-record corpus indexes in a couple of minutes and
peaks around 300 MB of RAM.

  # 60k rows, balanced across strata, disjoint from nothing
  python sample_corpus.py --corpus corpus.jsonl --n 60000 --out pool_60k.jsonl

  # a second, disjoint draw for a scaling curve
  python sample_corpus.py --corpus corpus.jsonl --n 60000 \\
      --exclude pool_60k.jsonl --out pool_60k_b.jsonl

  # human-annotated pairwise only, strictly balanced
  python sample_corpus.py --corpus corpus.jsonl --n 40000 \\
      --annotator human --task pairwise --allocation balanced --out human_40k.jsonl

Stratification
--------------
Strata are the cross of (source, language, task_type, label, length quartile).
Length quartile is computed over the filtered population, on instruction plus
response characters. It is in the key because response length is the strongest
confound in judge data: a uniform random draw from a corpus dominated by short
English chat yields a judge that degrades on exactly the long-form and code slices
JudgeBench and RM-Bench probe.

Allocation
----------
  proportional  quota proportional to stratum size — preserves the corpus mix,
                which is what you want if the mix itself is the object of study
  balanced      equal quota per stratum, capped at what each stratum has —
                maximum coverage, deliberately distorts the marginal distribution
  power         quota proportional to n^p (default p=0.5) — the usual compromise;
                rare strata are upweighted without erasing the natural mix

Leftover quota from strata smaller than their allocation is redistributed over
the strata that still have rows, so you get the requested n whenever the filtered
population can supply it.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from array import array
from collections import Counter, defaultdict

# --------------------------------------------------------------------------- #
# Pass 1 — index
# --------------------------------------------------------------------------- #


class Interner:
    """Map strings to small ints so the index stays compact."""

    def __init__(self) -> None:
        self.to_id: dict[str, int] = {}
        self.to_str: list[str] = []

    def __call__(self, s: str) -> int:
        i = self.to_id.get(s)
        if i is None:
            i = len(self.to_str)
            self.to_id[s] = i
            self.to_str.append(s)
        return i


def index_corpus(path: str, args) -> dict:
    offsets = array("q")
    lengths = array("i")
    src_i, lang_i, task_i, lab_i = array("H"), array("H"), array("H"), array("H")
    uids: list[str] = []

    src, lang, task, lab = Interner(), Interner(), Interner(), Interner()
    skipped = Counter()

    with open(path, "rb") as f:
        off = 0
        for raw in f:
            here, off = off, off + len(raw)
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                skipped["malformed"] += 1
                continue
            if rec.get("contaminated") and not args.keep_contaminated:
                skipped["contaminated"] += 1
                continue
            if args.task != "both" and rec.get("task_type") != args.task:
                skipped["task"] += 1
                continue
            if args.annotator and rec.get("annotator") != args.annotator:
                skipped["annotator"] += 1
                continue
            if args.sources and rec.get("source") not in args.sources:
                skipped["source"] += 1
                continue
            if rec.get("label") == "tie" and args.drop_ties:
                skipped["tie"] += 1
                continue

            n_chars = len(rec.get("instruction") or "") + sum(
                len(r or "") for r in (rec.get("responses") or []))
            if args.max_chars and n_chars > args.max_chars:
                skipped["too_long"] += 1
                continue

            # pointwise rows have no label; bucket them by score instead
            if rec.get("task_type") == "pointwise":
                s = rec.get("score")
                label = f"score={int(round(float(s)))}" if s is not None else "score=NA"
            else:
                label = rec.get("label") or "NA"

            offsets.append(here)
            lengths.append(n_chars)
            src_i.append(src(rec.get("source") or "unknown"))
            lang_i.append(lang(rec.get("lang") or "unknown"))
            task_i.append(task(rec.get("task_type") or "unknown"))
            lab_i.append(lab(label))
            uids.append(rec.get("uid") or "")

            if len(offsets) % 250_000 == 0:
                print(f"  indexed {len(offsets):,}", file=sys.stderr)

    return {"offsets": offsets, "lengths": lengths, "uids": uids,
            "src_i": src_i, "lang_i": lang_i, "task_i": task_i, "lab_i": lab_i,
            "src": src, "lang": lang, "task": task, "lab": lab,
            "skipped": skipped}


def length_quartiles(lengths: array) -> list[int]:
    s = sorted(lengths)
    n = len(s)
    return [s[int(n * q)] for q in (0.25, 0.50, 0.75)] if n else [0, 0, 0]


def quartile_of(n: int, cuts: list[int]) -> int:
    return 0 if n <= cuts[0] else 1 if n <= cuts[1] else 2 if n <= cuts[2] else 3


# --------------------------------------------------------------------------- #
# Allocation
# --------------------------------------------------------------------------- #


def allocate(sizes: dict[tuple, int], n_target: int, mode: str,
             power: float) -> dict[tuple, int]:
    keys = list(sizes)
    if mode == "proportional":
        w = {k: float(sizes[k]) for k in keys}
    elif mode == "balanced":
        w = {k: 1.0 for k in keys}
    else:
        w = {k: float(sizes[k]) ** power for k in keys}

    quota: dict[tuple, int] = {k: 0 for k in keys}
    remaining = min(n_target, sum(sizes.values()))
    live = set(keys)

    # iterate: assign by weight, clamp at stratum size, redistribute the leftover
    while remaining > 0 and live:
        tot_w = sum(w[k] for k in live)
        if tot_w <= 0:
            break
        assigned = 0
        for k in sorted(live):
            take = int(remaining * w[k] / tot_w)
            take = min(take, sizes[k] - quota[k])
            quota[k] += take
            assigned += take
        if assigned == 0:  # rounding stall — hand out one at a time
            for k in sorted(live, key=lambda k: -(sizes[k] - quota[k])):
                if remaining - assigned <= 0:
                    break
                if sizes[k] - quota[k] > 0:
                    quota[k] += 1
                    assigned += 1
        remaining -= assigned
        live = {k for k in live if sizes[k] - quota[k] > 0}
    return quota


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, required=True, help="target number of rows")
    ap.add_argument("--allocation", choices=["proportional", "balanced", "power"],
                    default="power")
    ap.add_argument("--power", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sources", default=None,
                    help="comma-separated allowlist of source keys")
    ap.add_argument("--task", choices=["pairwise", "pointwise", "both"], default="both")
    ap.add_argument("--annotator", choices=["human", "llm", "mixed"], default=None)
    ap.add_argument("--drop-ties", action="store_true")
    ap.add_argument("--keep-contaminated", action="store_true")
    ap.add_argument("--max-chars", type=int, default=None,
                    help="drop rows whose instruction+responses exceed this")
    ap.add_argument("--exclude", action="append", default=[],
                    help="previously sampled JSONL whose uids to exclude; repeatable")
    ap.add_argument("--no-length-strata", action="store_true")
    args = ap.parse_args()
    args.sources = ({s.strip() for s in args.sources.split(",")}
                    if args.sources else None)

    excluded: set[str] = set()
    for path in args.exclude:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    excluded.add(json.loads(line)["uid"])
                except (json.JSONDecodeError, KeyError):
                    pass
    if excluded:
        print(f"excluding {len(excluded):,} previously sampled uids", file=sys.stderr)

    print("pass 1: indexing", file=sys.stderr)
    ix = index_corpus(args.corpus, args)
    n_pop = len(ix["offsets"])
    print(f"  population after filters: {n_pop:,}", file=sys.stderr)
    print(f"  skipped: {dict(ix['skipped'])}", file=sys.stderr)
    if n_pop == 0:
        raise SystemExit("no rows survived the filters")

    cuts = length_quartiles(ix["lengths"])
    print(f"  length quartile cuts (chars): {cuts}", file=sys.stderr)

    strata: dict[tuple, list[int]] = defaultdict(list)
    for i in range(n_pop):
        if excluded and ix["uids"][i] in excluded:
            continue
        q = 0 if args.no_length_strata else quartile_of(ix["lengths"][i], cuts)
        key = (ix["src"].to_str[ix["src_i"][i]],
               ix["lang"].to_str[ix["lang_i"][i]],
               ix["task"].to_str[ix["task_i"][i]],
               ix["lab"].to_str[ix["lab_i"][i]],
               q)
        strata[key].append(i)

    sizes = {k: len(v) for k, v in strata.items()}
    print(f"  {len(sizes):,} strata, "
          f"{sum(sizes.values()):,} eligible rows", file=sys.stderr)

    quota = allocate(sizes, args.n, args.allocation, args.power)
    rng = random.Random(args.seed)
    chosen: list[int] = []
    for k, idxs in strata.items():
        q = quota.get(k, 0)
        if q:
            chosen.extend(idxs if q >= len(idxs) else rng.sample(idxs, q))
    chosen.sort()  # sequential seeks

    print(f"pass 2: writing {len(chosen):,} rows", file=sys.stderr)
    per_source, per_label, per_q, per_lang = Counter(), Counter(), Counter(), Counter()
    with open(args.corpus, "rb") as src_f, open(args.out, "wb") as out_f:
        for i in chosen:
            src_f.seek(ix["offsets"][i])
            out_f.write(src_f.readline())
            per_source[ix["src"].to_str[ix["src_i"][i]]] += 1
            per_label[ix["lab"].to_str[ix["lab_i"][i]]] += 1
            per_lang[ix["lang"].to_str[ix["lang_i"][i]]] += 1
            per_q[quartile_of(ix["lengths"][i], cuts)] += 1

    manifest = {
        "corpus": args.corpus, "out": args.out,
        "requested": args.n, "drawn": len(chosen),
        "population_after_filters": n_pop, "n_strata": len(sizes),
        "allocation": args.allocation, "power": args.power, "seed": args.seed,
        "length_quartile_cuts_chars": cuts,
        "excluded_uids": len(excluded),
        "filters": {"sources": sorted(args.sources) if args.sources else None,
                    "task": args.task, "annotator": args.annotator,
                    "drop_ties": args.drop_ties, "max_chars": args.max_chars},
        "per_source": dict(per_source.most_common()),
        "per_label": dict(per_label.most_common()),
        "per_language": dict(per_lang.most_common(20)),
        "per_length_quartile": {str(k): v for k, v in sorted(per_q.items())},
    }
    mpath = args.out.rsplit(".", 1)[0] + ".sample.json"
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nsources: {dict(per_source.most_common(10))}", file=sys.stderr)
    print(f"labels:  {dict(per_label.most_common(8))}", file=sys.stderr)
    print(f"length quartiles: {dict(sorted(per_q.items()))}", file=sys.stderr)
    print(f"wrote {args.out}  |  manifest {mpath}", file=sys.stderr)


if __name__ == "__main__":
    main()
