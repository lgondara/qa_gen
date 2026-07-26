"""
Harness for latent-need pseudo-labelling.

Runs the extraction prompt with self-consistency, validates evidence spans
against the target turn, and returns soft labels (agreement fractions) rather
than hard votes. Thresholding is deliberately left to a later, per-label
calibration step against the human gold set.
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

NEED_LABELS = {
    "reassurance", "permission", "control", "comprehension",
    "value_justification", "trust_verification", "burden_transfer",
    "sufficiency", "recognition", "provision_for_others",
}

TRIGGER_LABELS = {
    "retirement", "job_change", "job_loss", "inheritance", "bereavement",
    "marriage", "divorce", "new_dependant", "health_event", "relocation",
    "home_purchase", "business_sale", "none",
}

MAX_NEEDS = 2
EXCLUSIVE_PAIRS = [("control", "burden_transfer")]

SYSTEM_PROMPT = Path("latent_need_extraction_prompt.md").read_text()  # extract the
# fenced system block in practice; kept simple here.

USER_TEMPLATE = """CONTEXT
Conversation summary through turn {t}:
{summary}

Preceding turns:
{preceding_turns}

TARGET TURN (speaker: {speaker}, index: {target_index})
{target_text}"""


def _norm(s: str) -> str:
    """Whitespace-normalised lowercase form, for substring checking only."""
    return re.sub(r"\s+", " ", s).strip().lower()


@dataclass
class ValidationReport:
    span_failures: int = 0
    schema_failures: int = 0
    parse_failures: int = 0
    notes: list = field(default_factory=list)


def validate(sample: dict, target_text: str, report: ValidationReport) -> dict | None:
    """Drop labels that fail span or schema checks. Return None if unrecoverable."""
    hay = _norm(target_text)

    if not isinstance(sample.get("needs"), list):
        report.schema_failures += 1
        return None

    kept = []
    for need in sample["needs"]:
        label = need.get("label")
        span = need.get("evidence_span") or ""
        if label not in NEED_LABELS:
            report.schema_failures += 1
            report.notes.append(f"unknown label: {label!r}")
            continue
        if _norm(span) not in hay:
            report.span_failures += 1
            report.notes.append(f"span not in target: {span!r}")
            continue
        kept.append({"label": label,
                     "evidence_span": span,
                     "confidence": float(need.get("confidence", 0.5))})

    labels = {n["label"] for n in kept}
    for a, b in EXCLUSIVE_PAIRS:
        if a in labels and b in labels:
            # keep the higher-confidence member of the exclusive pair
            worse = min((n for n in kept if n["label"] in (a, b)),
                        key=lambda n: n["confidence"])
            kept.remove(worse)
            report.schema_failures += 1

    if len(kept) > MAX_NEEDS:
        kept = sorted(kept, key=lambda n: -n["confidence"])[:MAX_NEEDS]
        report.schema_failures += 1

    trigger = sample.get("trigger", "none")
    if trigger not in TRIGGER_LABELS:
        trigger = "none"
        report.schema_failures += 1
    if trigger != "none" and _norm(sample.get("trigger_evidence") or "") not in hay:
        trigger = "none"
        report.span_failures += 1

    risk = bool(sample.get("relationship_risk", False))
    if risk and _norm(sample.get("relationship_risk_evidence") or "") not in hay:
        risk = False
        report.span_failures += 1

    return {"needs": kept, "trigger": trigger, "relationship_risk": risk}


def aggregate(samples: list[dict]) -> dict:
    """Self-consistency aggregation. Soft labels, no thresholding."""
    n = len(samples)
    if n == 0:
        return {}

    need_counts = defaultdict(int)
    conf_sums = defaultdict(float)
    spans = defaultdict(list)
    trigger_counts = defaultdict(int)
    risk_count = 0

    for s in samples:
        for need in s["needs"]:
            need_counts[need["label"]] += 1
            conf_sums[need["label"]] += need["confidence"]
            spans[need["label"]].append(need["evidence_span"])
        trigger_counts[s["trigger"]] += 1
        risk_count += int(s["relationship_risk"])

    return {
        # agreement fraction per label: the soft target
        "need_agreement": {k: v / n for k, v in need_counts.items()},
        "need_mean_confidence": {k: conf_sums[k] / need_counts[k] for k in need_counts},
        # modal span per label, for human audit
        "need_spans": {k: max(set(v), key=v.count) for k, v in spans.items()},
        "abstain_fraction": sum(1 for s in samples if not s["needs"]) / n,
        "trigger_agreement": {k: v / n for k, v in trigger_counts.items()},
        "relationship_risk_agreement": risk_count / n,
        "n_valid_samples": n,
    }


def label_turn(client, record: dict, n_samples: int = 5,
               temperature: float = 0.7) -> dict:
    """
    record keys: t, summary, preceding_turns, speaker, target_index, target_text
    Returns soft labels plus a validation report.
    """
    report = ValidationReport()
    valid = []

    for _ in range(n_samples):
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            temperature=temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": USER_TEMPLATE.format(**record)}],
        )
        raw = "".join(b.text for b in resp.content if b.type == "text")
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            report.parse_failures += 1
            continue
        checked = validate(parsed, record["target_text"], report)
        if checked is not None:
            valid.append(checked)

    out = aggregate(valid)
    out["validation"] = vars(report)
    return out


if __name__ == "__main__":
    # Track these three rates across the corpus. A span-failure rate above a few
    # percent means the model is paraphrasing rather than quoting, and the
    # grounding constraint is not doing its job.
    print("span_failures / schema_failures / parse_failures are the quality gates")
