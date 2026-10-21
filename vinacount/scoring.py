"""Contract scoring and label-only scoring.

Contract scoring: a response contributes its support label only if it yields an accepted DetectorAssessment, under
Strict handling (raw response validated without repair) or Guarded handling (the guard is applied first). An invalid
response contributes no label and therefore counts as an error.

Label-only scoring: one lenient parser reads the support label from every raw response and ignores all other schema
fields; an accepted response keeps its accepted label.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

from vinacount.guard import guarded_assessment, strict_assessment

LABELS = ["supported", "weakly_supported", "not_supported", "insufficient_evidence"]
LABEL_KEYS = ("support_level", "decision")
HANDLINGS = ("strict", "guarded", "label_only")


def lenient_label(text: str | None) -> str | None:
    """Return the support label in a raw completion, ignoring every other schema field."""
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            obj = json.loads(match.group(0))
            for key in LABEL_KEYS:
                if obj.get(key) in LABELS:
                    return obj[key]
        except (json.JSONDecodeError, AttributeError):
            pass
    match = re.search(r'"(?:%s)"\s*:\s*"(%s)"' % ("|".join(LABEL_KEYS), "|".join(LABELS)), text)
    return match.group(1) if match else None


def judge_response(text: str | None, packet: dict[str, Any], handling: str) -> dict[str, Any]:
    """Apply one output-handling policy to one raw response."""
    record: dict[str, Any] = {"status": "invalid", "label": None, "severity": None, "guard_actions": [], "error": None}
    if handling == "label_only":
        accepted = judge_response(text, packet, "strict")
        record["label"] = accepted["label"] or lenient_label(text)
        record["status"] = "labelled" if record["label"] else "invalid"
        return record
    if not isinstance(text, str) or not text.strip():
        record["error"] = "missing_response"
        return record
    try:
        if handling == "strict":
            assessment = strict_assessment(text, packet)
        else:
            assessment, record["guard_actions"] = guarded_assessment(text, packet)
    except Exception as error:  # every rejection, including invalid JSON, is an invalid response
        record["error"] = str(error) or error.__class__.__name__
        return record
    record.update(status="accepted", label=assessment["support_level"], severity=assessment["severity"],
                  assessment=assessment)
    return record


def scores(gold: list[str], pred: list[str | None]) -> dict[str, Any]:
    """Accuracy and F1 over attempted packets; a missing prediction counts as wrong."""
    n = len(gold)
    per_class = {}
    for c in LABELS:
        tp = sum(p == c and g == c for g, p in zip(gold, pred))
        fp = sum(p == c and g != c for g, p in zip(gold, pred))
        fn = sum(p != c and g == c for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        per_class[c] = {"gold": tp + fn, "predicted": tp + fp, "precision": prec, "recall": rec, "f1": f1}
    return {
        "n": n,
        "labelled": sum(p is not None for p in pred),
        "valid_rate": sum(p is not None for p in pred) / n,
        "accuracy": sum(g == p for g, p in zip(gold, pred)) / n,
        "macro_f1": sum(v["f1"] for v in per_class.values()) / len(LABELS),
        "weak_f1": per_class["weakly_supported"]["f1"],
        "per_class": per_class,
    }


def confusion(gold: list[str], pred: list[str | None]) -> dict[str, dict[str, int]]:
    cols = LABELS + ["invalid"]
    return {g: {c: sum(1 for x, p in zip(gold, pred) if x == g and (p or "invalid") == c) for c in cols} for g in LABELS}


def bootstrap(gold: list[str], pred: list[str | None], reps: int = 10000, seed: int = 2026) -> dict[str, list[float]]:
    """Packet-level percentile intervals (95%) for accuracy, macro-F1, and weak-support F1."""
    rng = random.Random(seed)
    n = len(gold)
    draws: dict[str, list[float]] = {"accuracy": [], "macro_f1": [], "weak_f1": []}
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        s = scores([gold[i] for i in idx], [pred[i] for i in idx])
        for k in draws:
            draws[k].append(s[k])
    ci = {}
    for k, v in draws.items():
        v.sort()
        ci[k] = [v[int(0.025 * reps)], v[int(0.975 * reps) - 1]]
    return ci


def evaluate(records: list[dict[str, Any]], responses: dict[str, str | None], handling: str) -> tuple[list[str], list[str | None]]:
    """Gold and predicted labels for corpus records; a record without a response counts as invalid."""
    gold = [r["support_level"] for r in records]
    pred = [judge_response(responses.get(r["example_id"]), json.loads(r["messages"][1]["content"]), handling)["label"]
            for r in records]
    return gold, pred
