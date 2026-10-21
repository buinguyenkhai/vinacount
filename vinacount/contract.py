"""The detector contract: CandidateRisk, ToolFinding, DetectorPacket, and DetectorAssessment.

Validation enforces the packet caps, the assessment schema, the candidate identity, and Refs(A) ⊆ Refs(P): every
evidence reference cited by an assessment must be visible in its packet.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from numbers import Real
from typing import Any, Protocol


DETECTOR_PACKET_CAPS = {
    "relevant_table_rows": 12,
    "relevant_notes": 3,
    "relevant_variance_explanations": 2,
    "tool_findings": 5,
    "rules": 3,
}
ALLOWED_SUPPORT_LEVELS = {"supported", "weakly_supported", "not_supported", "insufficient_evidence"}
ALLOWED_SEVERITIES = {"high", "medium", "low", "unknown"}
ALLOWED_EVIDENCE_REF_TYPES = {
    "table_cell",
    "table_row",
    "note",
    "note_span",
    "variance_explanation_span",
    "accounting_policy_note_span",
    "related_party_note_span",
    "tool_result",
    "rule",
}
ALLOWED_EVIDENCE_REF_ROLES = {
    "supporting",
    "contradicting",
    "refuting",
    "context",
    "missing_required_context",
}
ALLOWED_SIGNAL_STATUSES = {"validated", "partially_validated", "rejected", "not_assessable"}
EARNINGS_CASHFLOW_RISK_CATEGORIES = {
    "earnings_cashflow_mismatch",
    "earnings_cashflow_quality_risk",
}
EVIDENCE_BUNDLE_SEMANTICS = {
    "tool_finding_strength_scope": "trigger_magnitude_only",
    "trigger_inputs_are_not_independent_corroboration": True,
    "independent_corroboration_requires_distinct_signal_or_disclosure": True,
}


@dataclass(frozen=True)
class ToolFinding:
    tool_result_id: str
    report_id: str
    tool_name: str
    risk_category: str
    signal_id: str
    flag: bool
    finding_summary: str
    evidence_refs: list[dict[str, str]]
    metric: dict[str, Any]
    threshold: dict[str, Any]


@dataclass(frozen=True)
class CandidateRisk:
    candidate_id: str
    report_id: str
    risk_category: str
    reason_for_candidate: str
    priority: str
    supporting_signal_ids: list[str]
    linked_tool_result_ids: list[str]
    evidence_refs: list[dict[str, str]]


@dataclass(frozen=True)
class DetectorPacket:
    packet_id: str
    candidate_id: str
    report_id: str
    task: dict[str, Any]
    metadata: dict[str, Any]
    candidate_summary: dict[str, Any]
    relevant_table_rows: list[dict[str, Any]]
    relevant_notes: list[dict[str, Any]]
    relevant_variance_explanations: list[dict[str, Any]]
    tool_findings: list[dict[str, Any]]
    rules: list[dict[str, Any]]
    constraints: dict[str, Any]
    report_set_id: str | None = None


@dataclass(frozen=True)
class DetectorAssessment:
    assessment_id: str
    packet_id: str
    candidate_id: str
    report_id: str
    risk_category: str
    support_level: str
    confidence: float
    severity: str
    validated_signals: list[dict[str, Any]]
    cited_evidence_refs: list[dict[str, str]]
    rationale_short: str


class DetectorAdapter(Protocol):
    def __call__(self, packet: DetectorPacket) -> DetectorAssessment:
        ...


def detector_assessment_json_schema() -> dict[str, Any]:
    evidence_ref_schema = {
        "type": "object",
        "properties": {
            "evidence_ref_type": {"type": "string", "enum": sorted(ALLOWED_EVIDENCE_REF_TYPES)},
            "ref_id": {"type": "string"},
            "role": {"type": "string", "enum": sorted(ALLOWED_EVIDENCE_REF_ROLES)},
        },
        "required": ["evidence_ref_type", "ref_id", "role"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "assessment_id": {"type": "string"},
            "packet_id": {"type": "string"},
            "candidate_id": {"type": "string"},
            "report_id": {"type": "string"},
            "risk_category": {"type": "string"},
            "support_level": {"type": "string", "enum": sorted(ALLOWED_SUPPORT_LEVELS)},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "severity": {"type": "string", "enum": sorted(ALLOWED_SEVERITIES)},
            "validated_signals": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "signal_id": {"type": "string"},
                        "status": {"type": "string", "enum": sorted(ALLOWED_SIGNAL_STATUSES)},
                        "support_level": {"type": "string", "enum": sorted(ALLOWED_SUPPORT_LEVELS)},
                        "tool_result_id": {"type": "string"},
                        "cited_evidence_refs": {"type": "array", "minItems": 1, "items": evidence_ref_schema},
                    },
                    "required": ["signal_id", "status", "support_level", "cited_evidence_refs"],
                    "additionalProperties": False,
                },
            },
            "cited_evidence_refs": {"type": "array", "items": evidence_ref_schema},
            "rationale_short": {"type": "string"},
        },
        "required": [
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
        ],
        "additionalProperties": False,
    }


def parse_and_validate_detector_assessment(content: str, packet: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(normalize_json_content(content))
    except json.JSONDecodeError as error:
        raise ValueError("invalid_json") from error
    if not isinstance(parsed, dict):
        raise ValueError("wrong_top_level_structure")
    if parsed.get("type") == "DetectorAssessment" and isinstance(parsed.get("data"), dict):
        parsed = parsed["data"]
    validate_detector_assessment(parsed, packet)
    return parsed


def normalize_json_content(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```json\n") and stripped.endswith("\n```"):
        return stripped[len("```json\n") : -len("\n```")].strip()
    if stripped.startswith("```\n") and stripped.endswith("\n```"):
        return stripped[len("```\n") : -len("\n```")].strip()
    return content


def validate_detector_packet(packet: Any) -> None:
    packet_data = _contract_dict(packet)
    if not isinstance(packet_data, dict):
        raise ValueError("invalid_detector_packet")
    required_fields = {
        "packet_id",
        "candidate_id",
        "report_id",
        "task",
        "metadata",
        "candidate_summary",
        "relevant_table_rows",
        "relevant_notes",
        "relevant_variance_explanations",
        "tool_findings",
        "rules",
        "constraints",
    }
    if required_fields - packet_data.keys():
        raise ValueError("invalid_detector_packet")
    if not packet_data["candidate_id"]:
        raise ValueError("invalid_detector_packet")
    if packet_data["task"].get("risk_category") in {"no_material_irregularity_signal", "insufficient_evidence"}:
        raise ValueError("invalid_detector_packet")
    if contains_prohibited_detector_visible_payload(packet_data):
        raise ValueError("raw extraction payload or prohibited detector-visible payload")
    for field_name, cap in DETECTOR_PACKET_CAPS.items():
        if len(packet_data[field_name]) > cap:
            raise ValueError(f"{field_name} exceeds detector packet cap")


def validate_detector_assessment(assessment: Any, packet: Any) -> None:
    assessment_data = _contract_dict(assessment)
    packet_data = _contract_dict(packet)
    if not isinstance(assessment_data, dict):
        raise ValueError("schema_mismatch")
    validate_detector_packet(packet_data)
    required_fields = {
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
    if required_fields - assessment_data.keys():
        raise ValueError("schema_mismatch")
    if set(assessment_data.keys()) != required_fields:
        raise ValueError("schema_mismatch")
    if assessment_data["packet_id"] != packet_data["packet_id"]:
        raise ValueError("identity_mismatch")
    if assessment_data["candidate_id"] != packet_data["candidate_id"]:
        raise ValueError("identity_mismatch")
    if assessment_data["report_id"] != packet_data["report_id"]:
        raise ValueError("identity_mismatch")
    if assessment_data["risk_category"] != packet_data["task"]["risk_category"]:
        raise ValueError("risk_category_mismatch")
    if assessment_data["support_level"] not in ALLOWED_SUPPORT_LEVELS:
        raise ValueError("invalid_support_level")
    if assessment_data["severity"] not in ALLOWED_SEVERITIES:
        raise ValueError("invalid_severity")
    confidence = assessment_data["confidence"]
    if not isinstance(confidence, Real) or isinstance(confidence, bool) or confidence < 0 or confidence > 1:
        raise ValueError("invalid_confidence")
    if not isinstance(assessment_data["validated_signals"], list) or not assessment_data["validated_signals"]:
        raise ValueError("schema_mismatch")
    if not isinstance(assessment_data["cited_evidence_refs"], list):
        raise ValueError("schema_mismatch")
    if not isinstance(assessment_data["rationale_short"], str) or not assessment_data["rationale_short"].strip():
        raise ValueError("schema_mismatch")
    if sentence_count(assessment_data["rationale_short"]) > 3:
        raise ValueError("rationale_too_long")
    if contains_prohibited_text(assessment_data):
        raise ValueError("prohibited_risk_language")
    if contains_outside_packet_reasoning(assessment_data):
        raise ValueError("outside_packet_reasoning")
    if contains_hidden_injection_leakage(assessment_data):
        raise ValueError("hidden_injection_leakage")

    visible_ids = visible_packet_evidence_ids(packet_data)
    for evidence_ref in assessment_data["cited_evidence_refs"]:
        validate_evidence_ref(evidence_ref, visible_ids)

    visible_signal_ids = {
        finding.get("signal_id")
        for finding in packet_data.get("tool_findings", [])
        if finding.get("signal_id")
    }
    visible_signal_ids.update(
        signal_id
        for rule in packet_data.get("rules", [])
        for signal_id in rule.get("related_signal_ids", [])
    )
    visible_signal_ids.update(packet_data.get("candidate_summary", {}).get("supporting_signal_ids", []))
    visible_tool_result_ids = {
        finding.get("tool_result_id")
        for finding in packet_data.get("tool_findings", [])
        if finding.get("tool_result_id")
    }
    for signal in assessment_data["validated_signals"]:
        if not isinstance(signal, dict):
            raise ValueError("schema_mismatch")
        required_signal_fields = {"signal_id", "status", "support_level", "cited_evidence_refs"}
        optional_signal_fields = {"tool_result_id"}
        signal_fields = set(signal.keys())
        if not required_signal_fields <= signal_fields or not signal_fields <= required_signal_fields | optional_signal_fields:
            raise ValueError("schema_mismatch")
        if signal["status"] not in ALLOWED_SIGNAL_STATUSES:
            raise ValueError("invalid_signal_status")
        if signal["support_level"] not in ALLOWED_SUPPORT_LEVELS:
            raise ValueError("invalid_support_level")
        if not isinstance(signal["cited_evidence_refs"], list) or not signal["cited_evidence_refs"]:
            raise ValueError("schema_mismatch")
        if signal.get("signal_id") and signal["signal_id"] not in visible_signal_ids:
            raise ValueError("invalid_evidence_ids")
        if signal.get("tool_result_id") and signal["tool_result_id"] not in visible_tool_result_ids:
            raise ValueError("invalid_evidence_ids")
        for evidence_ref in signal.get("cited_evidence_refs", []):
            validate_evidence_ref(evidence_ref, visible_ids)


def validate_evidence_ref(evidence_ref: Any, visible_ids: set[str]) -> None:
    if not isinstance(evidence_ref, dict):
        raise ValueError("schema_mismatch")
    if set(evidence_ref.keys()) != {"evidence_ref_type", "ref_id", "role"}:
        raise ValueError("schema_mismatch")
    if evidence_ref["evidence_ref_type"] not in ALLOWED_EVIDENCE_REF_TYPES:
        raise ValueError("invalid_evidence_ref_type")
    if evidence_ref["role"] not in ALLOWED_EVIDENCE_REF_ROLES:
        raise ValueError("invalid_evidence_ref_role")
    if evidence_ref["ref_id"] not in visible_ids:
        raise ValueError("invalid_evidence_ids")


def visible_packet_evidence_ids(packet: Any) -> set[str]:
    packet_data = _contract_dict(packet)
    ids: set[str] = set()
    ids.update(rule["rule_id"] for rule in packet_data.get("rules", []) if rule.get("rule_id"))
    for finding in packet_data.get("tool_findings", []):
        if finding.get("tool_result_id"):
            ids.add(finding["tool_result_id"])
        for ref in finding.get("evidence_refs", []):
            if isinstance(ref, dict) and ref.get("ref_id"):
                ids.add(ref["ref_id"])
    for row in packet_data.get("relevant_table_rows", []):
        report_id = row.get("report_id")
        for key in ("row_id", "local_evidence_id"):
            if row.get(key) and report_id:
                ids.add(f"{report_id}:{row[key]}")
        values = row.get("values", {})
        value_iter = values.values() if isinstance(values, dict) else values if isinstance(values, list) else []
        for value in value_iter:
            if isinstance(value, dict) and value.get("cell_id") and report_id:
                ids.add(f"{report_id}:{value['cell_id']}")
    for note in packet_data.get("relevant_notes", []):
        report_id = note.get("report_id")
        for key in ("note_id", "local_evidence_id"):
            if note.get(key) and report_id:
                ids.add(f"{report_id}:{note[key]}")
    for span in packet_data.get("relevant_variance_explanations", []):
        report_id = span.get("report_id")
        for key in ("span_id", "local_evidence_id"):
            if span.get(key) and report_id:
                ids.add(f"{report_id}:{span[key]}")
    return ids


def enrich_detector_packet_evidence_roles(packet: Any) -> dict[str, Any]:
    packet_data = deepcopy(_contract_dict(packet))
    if not isinstance(packet_data, dict):
        return packet_data
    task = packet_data.get("task")
    risk_category = task.get("risk_category") if isinstance(task, dict) else None
    if risk_category not in EARNINGS_CASHFLOW_RISK_CATEGORIES:
        return packet_data

    constraints = packet_data.setdefault("constraints", {})
    if isinstance(constraints, dict):
        constraints.setdefault("evidence_bundle_semantics", dict(EVIDENCE_BUNDLE_SEMANTICS))

    tool_findings = packet_data.get("tool_findings")
    if not isinstance(tool_findings, list) or not tool_findings:
        return packet_data

    primary_index = _primary_tool_finding_index(tool_findings)
    primary_finding = tool_findings[primary_index]
    primary_signal_id = primary_finding.get("signal_id") if isinstance(primary_finding, dict) else None
    primary_tool_result_id = primary_finding.get("tool_result_id") if isinstance(primary_finding, dict) else None
    independent_indexes = [
        index
        for index, finding in enumerate(tool_findings)
        if index != primary_index and _is_independent_corroboration_finding(finding, primary_finding)
    ]
    corroboration_refs = [
        ref
        for index in independent_indexes
        for ref in _corroboration_refs_for_finding(tool_findings[index])
    ]

    for index, finding in enumerate(tool_findings):
        if not isinstance(finding, dict):
            continue
        finding.setdefault("trigger_strength", finding.get("strength") or "not_applicable")
        if index == primary_index:
            finding.setdefault("evidence_role", "primary_trigger")
            finding["independent_corroboration_present"] = bool(independent_indexes)
            finding.setdefault("corroboration_evidence_refs", corroboration_refs)
            finding.setdefault("corroborates_signal_ids", [])
            continue
        if index in independent_indexes:
            finding.setdefault("evidence_role", "independent_corroboration")
            finding["independent_corroboration_present"] = True
            finding.setdefault(
                "corroboration_evidence_refs",
                [
                    {
                        "evidence_ref_type": "tool_result",
                        "ref_id": primary_tool_result_id,
                    }
                ]
                if primary_tool_result_id
                else [],
            )
            finding.setdefault(
                "corroborates_signal_ids",
                [primary_signal_id] if primary_signal_id else [],
            )
        else:
            finding.setdefault("evidence_role", "context")
            finding.setdefault("independent_corroboration_present", False)
            finding.setdefault("corroboration_evidence_refs", [])
            finding.setdefault("corroborates_signal_ids", [])
    return packet_data


def _primary_tool_finding_index(tool_findings: list[Any]) -> int:
    for index, finding in enumerate(tool_findings):
        if isinstance(finding, dict) and finding.get("evidence_role") == "primary_trigger":
            return index
    for index, finding in enumerate(tool_findings):
        if isinstance(finding, dict) and finding.get("flag") is True:
            return index
    return 0


def _is_independent_corroboration_finding(finding: Any, primary_finding: Any) -> bool:
    if not isinstance(finding, dict) or finding.get("flag") is not True:
        return False
    if finding.get("evidence_role") == "independent_corroboration":
        return True
    if not isinstance(primary_finding, dict):
        return False
    if finding.get("signal_id") and finding.get("signal_id") == primary_finding.get("signal_id"):
        return False
    if finding.get("tool_result_id") and finding.get("tool_result_id") == primary_finding.get("tool_result_id"):
        return False
    return bool(finding.get("signal_id") or finding.get("tool_result_id"))


def _corroboration_refs_for_finding(finding: Any) -> list[dict[str, str]]:
    if not isinstance(finding, dict):
        return []
    refs: list[dict[str, str]] = []
    if finding.get("tool_result_id"):
        refs.append({"evidence_ref_type": "tool_result", "ref_id": finding["tool_result_id"]})
    for ref in finding.get("evidence_refs", []):
        if isinstance(ref, dict) and ref.get("evidence_ref_type") and ref.get("ref_id"):
            refs.append({"evidence_ref_type": ref["evidence_ref_type"], "ref_id": ref["ref_id"]})
    return refs


def contains_prohibited_detector_visible_payload(value: Any) -> bool:
    prohibited_keys = {
        "raw_ocr_text",
        "full_raw_ocr_text",
        "raw_tables",
        "raw_pdf_coordinates",
        "raw_pdf_path",
        "raw_pdf_text",
        "raw_coordinates",
        "coordinates",
        "bbox",
        "bounding_box",
        "cache_record_id",
        "cache_key",
        "source_file_sha256",
        "normalized_text_hash",
        "table_content_hash",
        "derived_from_report_artifact_id",
        "derived_from_source_document_id",
        "hidden_injection_details",
        "injection_scenario",
        "injection_scenario_id",
        "target_support_level",
        "modified_evidence_ids",
        "original_values",
        "synthetic_values",
        "generation_seed",
        "teacher_model",
        "judge_model",
        "hidden_reasoning",
        "chain_of_thought",
        "hidden_chain_of_thought",
        "omitted_evidence_records",
        "omitted_evidence_ids",
        "omitted_evidence_summaries",
        "dropped_comparison_columns",
        "final_report",
        "final_report_text",
        "final_report_payload",
        "external_context",
        "outside_context",
        "outside_knowledge",
    }
    if isinstance(value, dict):
        return any(
            key in prohibited_keys or contains_prohibited_detector_visible_payload(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(contains_prohibited_detector_visible_payload(item) for item in value)
    if isinstance(value, str):
        normalized = value.lower()
        return any(
            phrase in normalized
            for phrase in [
                "hidden injection",
                "omitted evidence",
                "raw ocr",
                "raw pdf",
                "final report",
            ]
        )
    return False


def contains_hidden_injection_leakage(value: Any) -> bool:
    text = _json_text(value)
    patterns = [
        "hidden injection",
        "injection scenario",
        "target support level",
        "target_support_level",
        "omitted evidence",
        "underlying reportmemory",
        "base report",
        "synthetic scenario",
        "injected report",
        "modified evidence",
    ]
    return any(pattern in text for pattern in patterns)


def contains_outside_packet_reasoning(value: Any) -> bool:
    text = _json_text(value)
    patterns = [
        "outside news",
        "based on news",
        "vietstock",
        "sanction",
        "auditor issued",
        "qualified opinion",
        "later corrected",
        "annual report confirms",
        "known for aggressive accounting",
        "industry conditions",
        "restatement",
        "amendment",
        "replacement filing",
    ]
    return any(pattern in text for pattern in patterns)


def contains_prohibited_text(value: Any) -> bool:
    text = _json_text(value)
    return any(
        prohibited in text
        for prohibited in ["fraud", "manipulat", "conceal", "intent", "illegal", "legal misstatement", "misconduct"]
    )


def sentence_count(text: str) -> int:
    protected = re.sub(r"(?<=\d)\.(?=\d)", "<DECIMAL>", text)
    sentences = re.split(r"[.!?]+", protected)
    return len([sentence for sentence in sentences if sentence.strip()])


def _contract_dict(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    return value


def _json_text(value: Any) -> str:
    return json.dumps(_contract_dict(value), ensure_ascii=False, sort_keys=True).lower()
