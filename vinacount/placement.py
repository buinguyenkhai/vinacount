"""Report placement: a fixed policy maps each accepted label to a report section.

supported              -> grouped findings (grouped by the candidate's primary tool finding)
weakly_supported       -> items for limited review (severity capped at medium)
insufficient_evidence  -> evidence gaps
not_supported          -> kept only in the reviewed-candidate record

Every assessment also enters the reviewed-candidate record, so each reported item can be traced back to its
candidate, tool findings, cited evidence, and assessment. Nothing here can change a label, a reference, or a
candidate; an optional narrative summary is outside this module and receives the placed report as read-only input.
"""

from __future__ import annotations

from typing import Any

_SEVERITY_ORDER = {"high": 3, "medium": 2, "low": 1, "unknown": 0}
_PRIORITY_RANK = {"high": 0, "medium": 1, "low": 2}


def _cap_weak_severity(severity: str) -> str:
    return "medium" if severity == "high" else severity


def place_assessments(candidates: list[dict[str, Any]], assessments: list[dict[str, Any]]) -> dict[str, Any]:
    """Place accepted DetectorAssessments into report sections."""
    candidates_by_id = {c["candidate_id"]: c for c in candidates}

    def trace(a: dict[str, Any]) -> dict[str, Any]:
        c = candidates_by_id[a["candidate_id"]]
        return {
            "assessment_id": a["assessment_id"],
            "candidate_id": a["candidate_id"],
            "risk_category": a["risk_category"],
            "support_level": a["support_level"],
            "severity": a["severity"],
            "tool_refs": list(c["linked_tool_result_ids"]),
            "cited_evidence_refs": [r["ref_id"] for r in a["cited_evidence_refs"]],
            "rationale_short": a["rationale_short"],
        }

    groups: dict[str, list[dict[str, Any]]] = {}
    for a in assessments:
        if a["support_level"] == "supported":
            c = candidates_by_id[a["candidate_id"]]
            key = (c["linked_tool_result_ids"][:1] or [a["risk_category"]])[0]
            groups.setdefault(key, []).append(a)
    grouped_findings = []
    for index, members in enumerate(groups.values(), start=1):
        primary = min(members, key=lambda m: (-_SEVERITY_ORDER[m["severity"]],
                                              _PRIORITY_RANK[candidates_by_id[m["candidate_id"]]["priority"]],
                                              m["assessment_id"]))
        grouped_findings.append({
            "finding_id": f"F-{index:03d}",
            "primary_risk_category": primary["risk_category"],
            "final_severity": max((m["severity"] for m in members), key=_SEVERITY_ORDER.__getitem__),
            "human_review_recommended": True,
            "items": [trace(m) for m in members],
        })
    weak = [{**trace(a), "severity": _cap_weak_severity(a["severity"])}
            for a in assessments if a["support_level"] == "weakly_supported"]
    gaps = [trace(a) for a in assessments if a["support_level"] == "insufficient_evidence"]
    return {
        "grouped_findings": grouped_findings,
        "limited_review_items": weak,
        "evidence_gaps": gaps,
        "reviewed_candidates": [trace(a) for a in assessments],
        "human_review_recommended": bool(grouped_findings or weak),
    }
