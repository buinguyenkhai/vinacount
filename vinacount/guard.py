"""Deterministic contract guard for detector responses.

The guard enforces the DetectorAssessment schema and the candidate identity. It repairs a fixed set of formatting
errors and evidence references that resolve to exactly one packet identifier, then requires every cited reference to
exist in the packet. Responses with violations that cannot be repaired are rejected. The guard never changes the
support or severity label.
"""

from __future__ import annotations

import json
from typing import Any

from vinacount.contract import (
    ALLOWED_EVIDENCE_REF_TYPES,
    ALLOWED_SUPPORT_LEVELS,
    normalize_json_content,
    parse_and_validate_detector_assessment,
    visible_packet_evidence_ids,
)


def parse_assessment_object(text: str) -> dict[str, Any]:
    """Parse a raw completion into a JSON object, unwrapping a {"type": "DetectorAssessment", "data": ...} envelope."""
    parsed = json.loads(normalize_json_content(text))
    if not isinstance(parsed, dict):
        raise ValueError("wrong_top_level_structure")
    if parsed.get("type") == "DetectorAssessment" and isinstance(parsed.get("data"), dict):
        parsed = parsed["data"]
    return parsed


def strict_assessment(text: str, packet: dict[str, Any]) -> dict[str, Any]:
    """Validate a raw completion against the packet without any repair ("Strict" handling)."""
    return parse_and_validate_detector_assessment(text, packet)


def guarded_assessment(text: str, packet: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Apply the guard to a raw completion and validate the result ("Guarded" handling).

    Returns the validated assessment and the list of repair actions; raises ValueError when the response is rejected.
    """
    assessment = parse_assessment_object(text)
    actions = apply_contract_guard(assessment, packet)
    validated = parse_and_validate_detector_assessment(json.dumps(assessment, ensure_ascii=False, sort_keys=True), packet)
    return validated, actions


def apply_contract_guard(assessment: dict[str, Any], packet: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    actions.extend(_repair_legacy_decision_support_level(assessment))
    actions.extend(_repair_missing_identity_fields(assessment, packet))
    actions.extend(_repair_missing_validated_signals(assessment, packet))
    actions.extend(_strip_auxiliary_assessment_fields(assessment))
    actions.extend(_strip_auxiliary_evidence_ref_fields(assessment))
    actions.extend(_strip_auxiliary_signal_fields(assessment))
    actions.extend(_repair_unambiguous_evidence_ref_ids(assessment, packet))
    actions.extend(_repair_unambiguous_evidence_ref_types(assessment, packet))
    actions.extend(_remove_invalid_optional_tool_result_ids(assessment, packet))
    return sorted(set(actions))


def _repair_legacy_decision_support_level(assessment: dict[str, Any]) -> list[str]:
    if "support_level" in assessment:
        return []
    decision = assessment.get("decision")
    if decision not in ALLOWED_SUPPORT_LEVELS:
        return []
    assessment["support_level"] = decision
    return ["repaired_legacy_decision_support_level"]


def _repair_missing_identity_fields(assessment: dict[str, Any], packet: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    if not isinstance(assessment.get("assessment_id"), str) or not assessment.get("assessment_id"):
        packet_id = packet.get("packet_id")
        if isinstance(packet_id, str) and packet_id:
            assessment["assessment_id"] = f"ASSESS_{packet_id}"
            actions.append("repaired_missing_assessment_id")
    for field_name in ("packet_id", "candidate_id", "report_id", "risk_category"):
        if field_name == "risk_category":
            value = packet.get("task", {}).get("risk_category")
        else:
            value = packet.get(field_name)
        if not isinstance(value, str) or not value:
            continue
        if assessment.get(field_name) == value:
            continue
        was_missing = field_name not in assessment
        assessment[field_name] = value
        actions.append(f"repaired_missing_{field_name}" if was_missing else f"repaired_{field_name}_identity")
    return sorted(actions)


def _repair_missing_validated_signals(assessment: dict[str, Any], packet: dict[str, Any]) -> list[str]:
    if "validated_signals" in assessment:
        return []
    support_level = assessment.get("support_level")
    if support_level not in ALLOWED_SUPPORT_LEVELS:
        return []
    cited_evidence_refs = assessment.get("cited_evidence_refs")
    if not isinstance(cited_evidence_refs, list) or not cited_evidence_refs:
        return []
    signal_ids = assessment.get("supporting_signal_ids")
    if not isinstance(signal_ids, list) or not all(isinstance(signal_id, str) for signal_id in signal_ids):
        signal_ids = packet.get("candidate_summary", {}).get("supporting_signal_ids", [])
    visible_signal_ids = set(packet.get("candidate_summary", {}).get("supporting_signal_ids", []))
    visible_signal_ids.update(
        finding.get("signal_id")
        for finding in packet.get("tool_findings", [])
        if isinstance(finding, dict) and isinstance(finding.get("signal_id"), str)
    )
    normalized_signal_ids = [signal_id for signal_id in signal_ids if signal_id in visible_signal_ids]
    if not normalized_signal_ids:
        return []
    status = {
        "supported": "validated",
        "weakly_supported": "partially_validated",
        "not_supported": "rejected",
        "insufficient_evidence": "not_assessable",
    }[support_level]
    assessment["validated_signals"] = [
        {
            "signal_id": signal_id,
            "status": status,
            "support_level": support_level,
            "cited_evidence_refs": list(cited_evidence_refs),
        }
        for signal_id in normalized_signal_ids
    ]
    return ["repaired_missing_validated_signals"]


def _strip_auxiliary_assessment_fields(assessment: dict[str, Any]) -> list[str]:
    allowed_keys = {
        "assessment_id",
        "packet_id",
        "candidate_id",
        "report_id",
        "risk_category",
        "support_level",
        "confidence",
        "severity",
        "validated_signals",
        "cited_evidence_refs",
        "rationale_short",
    }
    extra_keys = sorted(set(assessment) - allowed_keys)
    if not extra_keys:
        return []
    for key in extra_keys:
        assessment.pop(key, None)
    return ["stripped_assessment_aux_fields"]


def _strip_auxiliary_evidence_ref_fields(assessment: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    allowed_ref_keys = {"evidence_ref_type", "ref_id", "role"}
    for evidence_ref in _iter_assessment_evidence_refs(assessment):
        extra_keys = sorted(set(evidence_ref) - allowed_ref_keys)
        if not extra_keys:
            continue
        for key in extra_keys:
            evidence_ref.pop(key, None)
        actions.append("stripped_evidence_ref_aux_fields")
    return sorted(set(actions))


def _strip_auxiliary_signal_fields(assessment: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    allowed_signal_keys = {"signal_id", "status", "support_level", "tool_result_id", "cited_evidence_refs"}
    for signal in _iter_assessment_signals(assessment):
        extra_keys = sorted(set(signal) - allowed_signal_keys)
        if not extra_keys:
            continue
        for key in extra_keys:
            signal.pop(key, None)
        actions.append("stripped_signal_aux_fields")
    return sorted(set(actions))


def _repair_unambiguous_evidence_ref_ids(assessment: dict[str, Any], packet: dict[str, Any]) -> list[str]:
    visible_ids = visible_packet_evidence_ids(packet)
    actions: list[str] = []
    for evidence_ref in _iter_assessment_evidence_refs(assessment):
        ref_id = evidence_ref.get("ref_id")
        if not isinstance(ref_id, str) or ref_id in visible_ids:
            continue
        local_id = ref_id.rsplit(":", 1)[-1]
        matches = sorted(visible_id for visible_id in visible_ids if visible_id.endswith(f":{local_id}"))
        if len(matches) != 1:
            continue
        evidence_ref["ref_id"] = matches[0]
        actions.append("repaired_evidence_ref_id")
    return sorted(set(actions))


def _repair_unambiguous_evidence_ref_types(assessment: dict[str, Any], packet: dict[str, Any]) -> list[str]:
    evidence_type_by_ref_id = _evidence_type_by_ref_id(packet)
    actions: list[str] = []
    for evidence_ref in _iter_assessment_evidence_refs(assessment):
        evidence_ref_type = evidence_ref.get("evidence_ref_type")
        if evidence_ref_type in ALLOWED_EVIDENCE_REF_TYPES:
            continue
        inferred_type = evidence_type_by_ref_id.get(evidence_ref.get("ref_id"))
        if inferred_type is None:
            continue
        evidence_ref["evidence_ref_type"] = inferred_type
        actions.append("repaired_evidence_ref_type")
    return sorted(set(actions))


def _remove_invalid_optional_tool_result_ids(assessment: dict[str, Any], packet: dict[str, Any]) -> list[str]:
    visible_tool_result_ids = {
        finding.get("tool_result_id")
        for finding in packet.get("tool_findings", [])
        if isinstance(finding, dict) and finding.get("tool_result_id")
    }
    actions: list[str] = []
    for signal in _iter_assessment_signals(assessment):
        tool_result_id = signal.get("tool_result_id")
        if not tool_result_id or tool_result_id in visible_tool_result_ids:
            continue
        signal.pop("tool_result_id", None)
        actions.append("removed_invalid_optional_tool_result_id")
    return sorted(set(actions))


def _iter_assessment_signals(assessment: dict[str, Any]) -> list[dict[str, Any]]:
    signals = assessment.get("validated_signals", [])
    if not isinstance(signals, list):
        return []
    return [signal for signal in signals if isinstance(signal, dict)]


def _iter_assessment_evidence_refs(assessment: dict[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    cited_refs = assessment.get("cited_evidence_refs", [])
    if isinstance(cited_refs, list):
        refs.extend(ref for ref in cited_refs if isinstance(ref, dict))
    for signal in _iter_assessment_signals(assessment):
        signal_refs = signal.get("cited_evidence_refs", [])
        if isinstance(signal_refs, list):
            refs.extend(ref for ref in signal_refs if isinstance(ref, dict))
    return refs


def _evidence_type_by_ref_id(packet: dict[str, Any]) -> dict[str, str]:
    evidence_types: dict[str, str] = {}
    for rule in packet.get("rules", []):
        if isinstance(rule, dict) and rule.get("rule_id"):
            evidence_types[rule["rule_id"]] = "rule"
    for finding in packet.get("tool_findings", []):
        if not isinstance(finding, dict):
            continue
        if finding.get("tool_result_id"):
            evidence_types[finding["tool_result_id"]] = "tool_result"
        for ref in finding.get("evidence_refs", []):
            if (
                isinstance(ref, dict)
                and ref.get("ref_id")
                and ref.get("evidence_ref_type") in ALLOWED_EVIDENCE_REF_TYPES
            ):
                evidence_types[ref["ref_id"]] = ref["evidence_ref_type"]
    for row in packet.get("relevant_table_rows", []):
        if not isinstance(row, dict):
            continue
        report_id = row.get("report_id")
        if not report_id:
            continue
        if row.get("row_id"):
            evidence_types[f"{report_id}:{row['row_id']}"] = "table_row"
        if row.get("local_evidence_id"):
            evidence_types[f"{report_id}:{row['local_evidence_id']}"] = "table_row"
        values = row.get("values", {})
        value_iter = values.values() if isinstance(values, dict) else values if isinstance(values, list) else []
        for value in value_iter:
            if isinstance(value, dict) and value.get("cell_id"):
                evidence_types[f"{report_id}:{value['cell_id']}"] = "table_cell"
    for note in packet.get("relevant_notes", []):
        if not isinstance(note, dict):
            continue
        report_id = note.get("report_id")
        if not report_id:
            continue
        if note.get("note_id"):
            evidence_types[f"{report_id}:{note['note_id']}"] = "note"
        if note.get("local_evidence_id"):
            evidence_types[f"{report_id}:{note['local_evidence_id']}"] = "note"
    for span in packet.get("relevant_variance_explanations", []):
        if not isinstance(span, dict):
            continue
        report_id = span.get("report_id")
        if not report_id:
            continue
        if span.get("span_id"):
            evidence_types[f"{report_id}:{span['span_id']}"] = "variance_explanation_span"
        if span.get("local_evidence_id"):
            evidence_types[f"{report_id}:{span['local_evidence_id']}"] = "variance_explanation_span"
    return evidence_types
