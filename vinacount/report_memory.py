"""ReportMemory, the structured store of one filing, and the target/prior-year CompanyReportSet.

Each extracted value and passage keeps its source location. A report set requires the same company, quarter,
report profile, and basis, with the target exactly one year after the prior-year filing.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any


ALLOWED_REPORT_PROFILES = {"standard_corporate", "credit_institution", "securities", "insurance"}
ALLOWED_REPORT_BASES = {"consolidated", "separate", "parent"}
ALLOWED_FILING_STATUSES = {"original", "amended_or_replacement"}
ALLOWED_REPORT_CURRENCIES = {"VND"}
ALLOWED_REPORT_UNITS = {"vnd", "thousand_vnd", "million_vnd", "billion_vnd"}
REQUIRED_MAIN_STATEMENT_TYPES = {"balance_sheet", "income_statement", "cash_flow_statement"}
HAND_AUTHORED_REFERENCE_EXTRACTION_METHODS = {"hand_authored_reference_v1"}
REQUIRED_REPORT_MEMORY_FIELDS = {
    "report_id",
    "metadata",
    "sections",
    "tables",
    "notes",
    "variance_explanations",
    "cell_index",
}
REQUIRED_METADATA_FIELDS = {
    "company_name",
    "period",
    "report_period_type",
    "report_profile",
    "report_basis",
    "business_context_tags",
    "report_assurance_type",
    "currency",
    "unit",
    "filing_status",
    "canonical_source_document_id",
    "source_file",
}
PROHIBITED_CURRENT_FILING_COMPARISON_KEYS = {
    "prior_period_value",
    "prior_year_value",
    "comparison_value",
}


@dataclass(frozen=True)
class ReportMemory:
    raw: dict[str, Any]

    @property
    def report_id(self) -> str:
        return self.raw["report_id"]

    @property
    def metadata(self) -> dict[str, Any]:
        return self.raw["metadata"]

    @property
    def year(self) -> int:
        return _parse_period(self.metadata["period"])[0]

    @property
    def quarter(self) -> int:
        return _parse_period(self.metadata["period"])[1]


@dataclass(frozen=True)
class CompanyReportSet:
    case_id: str
    target: ReportMemory
    prior_year: ReportMemory


@dataclass(frozen=True)
class RealExtractionQualityGateResult:
    status: str
    allows_tool_availability_gating: bool
    records: list[dict[str, Any]]


def validate_report_memory(raw: dict[str, Any]) -> ReportMemory:
    missing_top_level = REQUIRED_REPORT_MEMORY_FIELDS - raw.keys()
    if missing_top_level:
        raise ValueError(f"ReportMemory is missing top-level fields: {sorted(missing_top_level)}")

    metadata = raw["metadata"]
    missing_metadata = REQUIRED_METADATA_FIELDS - metadata.keys()
    if missing_metadata:
        raise ValueError(f"ReportMemory metadata is missing fields: {sorted(missing_metadata)}")

    if metadata["report_period_type"] != "quarterly":
        raise ValueError("ReportMemory only accepts quarterly reports")
    if metadata["report_profile"] not in ALLOWED_REPORT_PROFILES:
        raise ValueError("ReportMemory report_profile must be known and supported")
    if metadata["report_basis"] not in ALLOWED_REPORT_BASES:
        raise ValueError("ReportMemory report_basis must be known and supported")
    if metadata["filing_status"] not in ALLOWED_FILING_STATUSES:
        raise ValueError("ReportMemory filing_status must identify the canonical filing status")
    if not isinstance(metadata["business_context_tags"], list) or not metadata["business_context_tags"]:
        raise ValueError("ReportMemory business_context_tags must be a non-empty list")
    if not metadata["canonical_source_document_id"]:
        raise ValueError("ReportMemory canonical_source_document_id is required")
    if _contains_prohibited_prior_comparison_columns(raw):
        raise ValueError("Tool-facing ReportMemory must exclude prior comparison columns")

    return ReportMemory(raw=raw)


def validate_company_report_set(case_id: str, target: ReportMemory, prior_year: ReportMemory) -> CompanyReportSet:
    if target.metadata["company_name"] != prior_year.metadata["company_name"]:
        raise ValueError("CompanyReportSet requires the same company")
    if target.quarter != prior_year.quarter:
        raise ValueError("CompanyReportSet requires the same quarter in target and prior-year filings")
    if target.year != prior_year.year + 1:
        raise ValueError("CompanyReportSet requires a target/prior-year period relationship")
    if target.metadata["report_profile"] != prior_year.metadata["report_profile"]:
        raise ValueError("CompanyReportSet requires the same report_profile")
    if target.metadata["report_basis"] != prior_year.metadata["report_basis"]:
        raise ValueError("CompanyReportSet requires the same report_basis")

    return CompanyReportSet(case_id=case_id, target=target, prior_year=prior_year)


def validate_real_extraction_quality_gate(report_set: CompanyReportSet) -> RealExtractionQualityGateResult:
    records = []
    for role, report_memory in (("target", report_set.target), ("prior_year", report_set.prior_year)):
        metadata = report_memory.metadata
        if not str(metadata.get("company_name", "")).strip():
            records.append(
                _real_extraction_gate_record(
                    report_set,
                    reason_code="missing_company_identity",
                    reason=f"{role} ReportMemory must identify the company.",
                )
            )
        if not str(metadata.get("period", "")).strip():
            records.append(
                _real_extraction_gate_record(
                    report_set,
                    reason_code="missing_period_identity",
                    reason=f"{role} ReportMemory must identify the period and quarter.",
                )
            )
        if metadata.get("report_profile") == "unknown":
            records.append(
                _real_extraction_gate_record(
                    report_set,
                    reason_code="unknown_report_profile",
                    reason=f"{role} ReportMemory requires an identified report_profile for default analysis.",
                )
            )
        if metadata.get("report_basis") == "unknown":
            records.append(
                _real_extraction_gate_record(
                    report_set,
                    reason_code="unknown_report_basis",
                    reason=f"{role} ReportMemory requires an identified report_basis for default analysis.",
                )
            )
        if metadata.get("currency") not in ALLOWED_REPORT_CURRENCIES or metadata.get("unit") not in ALLOWED_REPORT_UNITS:
            records.append(
                _real_extraction_gate_record(
                    report_set,
                    reason_code="invalid_unit_currency_normalization",
                    reason=f"{role} ReportMemory has invalid currency or unit normalization.",
                )
            )
        if _contains_prohibited_prior_comparison_columns(report_memory.raw):
            records.append(
                _real_extraction_gate_record(
                    report_set,
                    reason_code="current_filing_prior_comparison_column_leakage",
                    reason=f"{role} ReportMemory contains current-filing prior comparison columns.",
                )
            )
        unresolved_amendment_context = _unresolved_operative_amendment_context_reason(report_memory)
        if unresolved_amendment_context:
            records.append(
                _real_extraction_gate_record(
                    report_set,
                    reason_code="amendment_context_corrected_value_resolution_required",
                    reason=f"{role} ReportMemory has unresolved operative amendment context: {unresolved_amendment_context}.",
                )
            )
        for quality_record in report_memory.raw.get("quality_records", []):
            if (
                quality_record.get("record_type") == "note_reference_match_failure"
                and quality_record.get("severity") == "blocking"
            ):
                row_id = quality_record.get("row_id", "unknown row")
                records.append(
                    _real_extraction_gate_record(
                        report_set,
                        reason_code="required_note_reference_match_failed",
                        reason=f"{role} ReportMemory has required Note Reference Match failure for {row_id}.",
                    )
                )
        missing_statements = set()
        if not _is_hand_authored_reference_report_memory(report_memory):
            missing_statements = REQUIRED_MAIN_STATEMENT_TYPES - {
                table.get("table_type") for table in report_memory.raw.get("tables", [])
            }
        if missing_statements:
            records.append(
                _real_extraction_gate_record(
                    report_set,
                    reason_code="missing_required_main_statement",
                    reason=(
                        f"{role} ReportMemory is missing required main statement tables: "
                        f"{', '.join(sorted(missing_statements))}."
                    ),
                )
            )
        for cell in _iter_report_memory_cells(report_memory):
            if not isinstance(cell.get("value"), Real) or isinstance(cell.get("value"), bool):
                records.append(
                    _real_extraction_gate_record(
                        report_set,
                        reason_code="numeric_parse_requirement_failed",
                        reason=f"{role} ReportMemory cell {cell.get('cell_id')} does not contain a parsed number.",
                    )
                )
                break
        invalid_registry_reason = _invalid_cell_index_reason(report_memory)
        if invalid_registry_reason:
            records.append(
                _real_extraction_gate_record(
                    report_set,
                    reason_code="invalid_evidence_registry",
                    reason=f"{role} ReportMemory has invalid evidence registry: {invalid_registry_reason}",
                )
            )
    if records:
        return RealExtractionQualityGateResult(
            status="failed",
            allows_tool_availability_gating=False,
            records=records,
        )
    return RealExtractionQualityGateResult(
        status="passed",
        allows_tool_availability_gating=True,
        records=[],
    )


def _real_extraction_gate_record(
    report_set: CompanyReportSet,
    *,
    reason_code: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "case_id": report_set.case_id,
        "record_type": "recoverable_setup_data_error",
        "reason_code": reason_code,
        "reason": reason,
        "analysis_scope": "default_analysis",
        "target_report_id": report_set.target.report_id,
        "prior_year_report_id": report_set.prior_year.report_id,
    }


def _unresolved_operative_amendment_context_reason(report_memory: ReportMemory) -> str | None:
    attachments = report_memory.metadata.get("amendment_context_attachments", [])
    if not isinstance(attachments, list):
        return "amendment_context_attachments metadata must be a list"
    resolved_attachment_ids = {
        cell["amendment_resolution"]["amendment_context_attachment_id"]
        for cell in _iter_report_memory_cells(report_memory)
        if isinstance(cell.get("amendment_resolution"), dict)
        and cell["amendment_resolution"].get("amendment_context_attachment_id")
    }
    unresolved = []
    for attachment in attachments:
        if not isinstance(attachment, dict):
            return "amendment context attachment metadata must be structured"
        if not attachment.get("affects_operative_values"):
            continue
        attachment_id = attachment.get("source_document_id")
        status = attachment.get("corrected_value_resolution_status")
        if status == "resolved" and attachment_id in resolved_attachment_ids:
            continue
        unresolved.append(str(attachment_id or "unknown_attachment"))
    if unresolved:
        return ", ".join(unresolved)
    return None


def _iter_report_memory_cells(report_memory: ReportMemory):
    for table in report_memory.raw.get("tables", []):
        for row in table.get("rows", []):
            yield from row.get("cells", [])


def _is_hand_authored_reference_report_memory(report_memory: ReportMemory) -> bool:
    return report_memory.metadata.get("extraction_method") in HAND_AUTHORED_REFERENCE_EXTRACTION_METHODS


def _invalid_cell_index_reason(report_memory: ReportMemory) -> str | None:
    indexed_ids = set(report_memory.raw.get("cell_index", {}).keys())
    cell_ids = [cell.get("cell_id") for cell in _iter_report_memory_cells(report_memory)]
    if len(cell_ids) != len(set(cell_ids)):
        return "duplicate cell evidence IDs"
    missing = sorted(set(cell_ids) - indexed_ids)
    if missing:
        return f"missing cell_index entries for {', '.join(missing)}"
    extra = sorted(indexed_ids - set(cell_ids))
    if extra:
        return f"cell_index entries do not point to table cells: {', '.join(extra)}"
    return None


def _parse_period(period: str) -> tuple[int, int]:
    try:
        year_text, quarter_text = period.split("-Q", maxsplit=1)
        return int(year_text), int(quarter_text)
    except ValueError as error:
        raise ValueError(f"Unsupported quarterly period format: {period}") from error


def _contains_prohibited_prior_comparison_columns(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            key in PROHIBITED_CURRENT_FILING_COMPARISON_KEYS
            or _contains_prohibited_prior_comparison_columns(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_prohibited_prior_comparison_columns(child) for child in value)
    return False
