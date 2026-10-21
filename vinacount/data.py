"""The synthetic detector corpus.

The released file holds 3,778 examples. Each record carries the chat messages (system prompt, DetectorPacket,
retained DetectorAssessment) and its metadata, including the split.

  final      all 3,778 examples (2,926 train / 437 validation / 415 test); trains the released adapter.
  selection  the 3,702 screened examples (2,870 / 417 / 415) used for the paper's configuration comparison. It omits
             the 76 matched-pair examples, uses the v1 system prompt, and names the earnings-cash-flow category
             "earnings_cashflow_quality_risk". These are the only differences.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vinacount.detector import SYSTEM_PROMPTS

SPLITS = ("train", "validation", "test")
CORPORA = ("final", "selection")
_FINAL_CATEGORY, _SELECTION_CATEGORY = "earnings_cashflow_mismatch", "earnings_cashflow_quality_risk"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def load_corpus(path: str | Path, corpus: str = "final") -> list[dict[str, Any]]:
    """Load the released corpus file and return the requested corpus."""
    records = read_jsonl(path)
    for r in records:
        if r["messages"][0]["content"] != SYSTEM_PROMPTS["v2_evidence_bundle"]:
            raise ValueError(f"{r['example_id']}: unexpected system prompt")
    if corpus == "final":
        return records
    if corpus != "selection":
        raise ValueError(f"corpus must be one of {CORPORA}")
    selected = sorted((r for r in records if r["selection_index"] is not None), key=lambda r: r["selection_index"])
    out = []
    for r in selected:
        messages = [{"role": "system", "content": SYSTEM_PROMPTS["v1"]}] + [
            {"role": m["role"], "content": m["content"].replace(_FINAL_CATEGORY, _SELECTION_CATEGORY)}
            for m in r["messages"][1:]
        ]
        category = _SELECTION_CATEGORY if r["risk_category"] == _FINAL_CATEGORY else r["risk_category"]
        out.append({**r, "risk_category": category, "messages": messages})
    return out


def split_records(records: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [r for r in records if r["split"] == split]


def packet_of(record: dict[str, Any]) -> dict[str, Any]:
    return json.loads(record["messages"][1]["content"])


def gold_of(record: dict[str, Any]) -> dict[str, Any]:
    return json.loads(record["messages"][2]["content"])
