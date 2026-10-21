"""Source selection: validate a filing package, classify its documents, and select the authoritative statement.

The selected target statement and its same-quarter prior-year counterpart are confirmed before extraction.
Supporting attachments may add evidence but cannot replace the primary statement, and no model selects sources.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Protocol


# --- Filing packages ---

ALLOWED_REPORT_BASES = {"consolidated", "separate", "parent"}
ALLOWED_DOCUMENT_TYPES = {
    "main_financial_statement",
    "variance_explanation",
    "variance_explanation_attachment",
    "cover_disclosure_letter",
    "cover_letter",
    "amendment_context_attachment",
    "business_overview_out_of_scope",
    "amended_or_replacement_financial_statement",
}
ALLOWED_SOURCE_KINDS = {"local", "fetched"}

REQUIRED_PACKAGE_FIELDS = {
    "package_id",
    "company",
    "period",
    "report_basis",
    "filing_event",
    "source_documents",
}
REQUIRED_COMPANY_FIELDS = {"name"}
REQUIRED_FILING_EVENT_FIELDS = {"event_id", "filing_status"}
REQUIRED_SOURCE_DOCUMENT_FIELDS = {
    "document_id",
    "document_type",
    "provenance",
    "local_artifact",
    "fingerprint",
    "reported_period_clues",
    "basis_clues",
    "amendment_clues",
}
REQUIRED_PROVENANCE_FIELDS = {"source_kind", "source_name", "source_url", "fetched_at"}
REQUIRED_LOCAL_ARTIFACT_FIELDS = {"artifact_id", "path"}
REQUIRED_FINGERPRINT_FIELDS = {"hash_algorithm", "hash_value"}
REQUIRED_AMENDMENT_CLUE_FIELDS = {"is_amended_or_replacement", "clues"}


@dataclass(frozen=True)
class SourceDocument:
    raw: dict[str, Any]

    @property
    def document_id(self) -> str:
        return self.raw["document_id"]

    @property
    def document_type(self) -> str:
        return self.raw["document_type"]

    @property
    def provenance(self) -> dict[str, Any]:
        return self.raw["provenance"]

    @property
    def reported_period_clues(self) -> list[Any]:
        return self.raw["reported_period_clues"]

    @property
    def basis_clues(self) -> list[Any]:
        return self.raw["basis_clues"]

    @property
    def is_amended_or_replacement(self) -> bool:
        return self.raw["amendment_clues"]["is_amended_or_replacement"]


@dataclass(frozen=True)
class LocalSourceDocument:
    document_id: str
    document_type: str
    artifact_id: str
    path: str
    hash_value: str
    reported_period_clues: list[str]
    basis_clues: list[str]
    hash_algorithm: str = "sha256"
    source_name: str = "local"
    is_amended_or_replacement: bool = False
    amendment_clues: list[str] | None = None


@dataclass(frozen=True)
class FilingPackage:
    raw: dict[str, Any]
    source_documents: list[SourceDocument]

    @property
    def package_id(self) -> str:
        return self.raw["package_id"]

    @property
    def company_name(self) -> str:
        return self.raw["company"]["name"]

    @property
    def period(self) -> str:
        return self.raw["period"]

    @property
    def report_basis(self) -> str:
        return self.raw["report_basis"]

    @property
    def financial_statement_candidates(self) -> list[SourceDocument]:
        return [
            document
            for document in self.source_documents
            if document.document_type
            in {"main_financial_statement", "amended_or_replacement_financial_statement"}
        ]

    @property
    def supporting_documents(self) -> list[SourceDocument]:
        return [
            document
            for document in self.source_documents
            if document.document_type in {
                "variance_explanation",
                "variance_explanation_attachment",
                "cover_disclosure_letter",
                "cover_letter",
                "amendment_context_attachment",
            }
        ]

    @property
    def source_evidence(self) -> list[dict[str, Any]]:
        return self.raw.get("source_evidence", [])


@dataclass(frozen=True)
class CanonicalSourceSelection:
    canonical_source_document_id: str
    filing_status: str
    supporting_document_ids: list[str]
    requested_report_basis: str | None = None
    selected_basis_family: str | None = None
    visible_filing_label_by_document_id: dict[str, str] | None = None
    assurance_type: str | None = None
    superseded_source_document_ids: list[str] | None = None
    amendment_context_attachment_ids: list[str] | None = None
    supporting_document_links: list[dict[str, Any]] | None = None


class CanonicalSourceSelectionError(ValueError):
    pass


def ingest_local_filing_package(
    *,
    package_id: str,
    company: dict[str, Any],
    period: str,
    quarter: str,
    report_basis: str,
    filing_event: dict[str, Any],
    source_documents: list[LocalSourceDocument],
) -> FilingPackage:
    _require_non_empty_string(quarter, "FilingPackage quarter")
    raw = {
        "package_id": package_id,
        "company": company,
        "period": period,
        "quarter": quarter,
        "report_basis": report_basis,
        "filing_event": filing_event,
        "source_documents": [
            _local_source_document_to_raw(document) for document in source_documents
        ],
    }
    return validate_filing_package(raw)


def _local_source_document_to_raw(document: LocalSourceDocument) -> dict[str, Any]:
    return {
        "document_id": document.document_id,
        "document_type": document.document_type,
        "provenance": {
            "source_kind": "local",
            "source_name": document.source_name,
            "source_url": None,
            "fetched_at": None,
        },
        "local_artifact": {
            "artifact_id": document.artifact_id,
            "path": document.path,
        },
        "fingerprint": {
            "hash_algorithm": document.hash_algorithm,
            "hash_value": document.hash_value,
        },
        "reported_period_clues": document.reported_period_clues,
        "basis_clues": document.basis_clues,
        "amendment_clues": {
            "is_amended_or_replacement": document.is_amended_or_replacement,
            "clues": document.amendment_clues or [],
        },
    }


def select_canonical_source(package: FilingPackage) -> CanonicalSourceSelection:
    financial_statement_candidates = _exact_financial_statement_candidates(package)
    if not financial_statement_candidates:
        raise CanonicalSourceSelectionError(_missing_candidate_reason(package))

    amended_candidates = [
        document
        for document in financial_statement_candidates
        if document.document_type == "amended_or_replacement_financial_statement"
        and document.is_amended_or_replacement
    ]
    original_candidates = [
        document
        for document in financial_statement_candidates
        if document.document_type == "main_financial_statement"
    ]

    if amended_candidates:
        if len(amended_candidates) != 1:
            raise CanonicalSourceSelectionError("ambiguous_financial_statement_candidates")
        canonical_document = amended_candidates[0]
        filing_status = "amended_or_replacement"
    else:
        if len(original_candidates) != 1:
            raise CanonicalSourceSelectionError("ambiguous_financial_statement_candidates")
        canonical_document = original_candidates[0]
        filing_status = "original"

    return CanonicalSourceSelection(
        canonical_source_document_id=canonical_document.document_id,
        filing_status=filing_status,
        supporting_document_ids=[document.document_id for document in package.supporting_documents],
        requested_report_basis=package.report_basis,
    )


def _exact_financial_statement_candidates(package: FilingPackage) -> list[SourceDocument]:
    return [
        document
        for document in package.financial_statement_candidates
        if package.period in document.reported_period_clues
        and package.report_basis in document.basis_clues
    ]


def _missing_candidate_reason(package: FilingPackage) -> str:
    if not package.financial_statement_candidates:
        return "missing_financial_statement"
    if any(package.period in document.reported_period_clues for document in package.financial_statement_candidates):
        return "basis_mismatch"
    return "period_mismatch"


def validate_filing_package(raw: dict[str, Any]) -> FilingPackage:
    _require_fields(raw, REQUIRED_PACKAGE_FIELDS, "FilingPackage")

    company = raw["company"]
    if not isinstance(company, dict):
        raise ValueError("FilingPackage company must be an object")
    _require_fields(company, REQUIRED_COMPANY_FIELDS, "FilingPackage company")
    _require_non_empty_string(company["name"], "FilingPackage company.name")

    _require_non_empty_string(raw["package_id"], "FilingPackage package_id")
    _require_non_empty_string(raw["period"], "FilingPackage period")
    if raw["report_basis"] not in ALLOWED_REPORT_BASES:
        raise ValueError("FilingPackage report_basis must be consolidated, separate, or parent")

    filing_event = raw["filing_event"]
    if not isinstance(filing_event, dict):
        raise ValueError("FilingPackage filing_event must be an object")
    _require_fields(filing_event, REQUIRED_FILING_EVENT_FIELDS, "FilingPackage filing_event")
    _require_non_empty_string(filing_event["event_id"], "FilingPackage filing_event.event_id")
    _require_non_empty_string(filing_event["filing_status"], "FilingPackage filing_event.filing_status")

    source_documents = raw["source_documents"]
    if not isinstance(source_documents, list) or not source_documents:
        raise ValueError("FilingPackage source_documents must be a non-empty list")

    validated_documents = [_validate_source_document(document) for document in source_documents]
    document_ids = [document.document_id for document in validated_documents]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("FilingPackage source_documents contain a duplicate source document ID")
    return FilingPackage(raw=raw, source_documents=validated_documents)


def _validate_source_document(raw: dict[str, Any]) -> SourceDocument:
    if not isinstance(raw, dict):
        raise ValueError("FilingPackage source document must be an object")
    _require_fields(raw, REQUIRED_SOURCE_DOCUMENT_FIELDS, "FilingPackage source document")
    _require_non_empty_string(raw["document_id"], "FilingPackage source document document_id")
    if raw["document_type"] not in ALLOWED_DOCUMENT_TYPES:
        raise ValueError("FilingPackage source document has unsupported document_type")

    provenance = raw["provenance"]
    if not isinstance(provenance, dict):
        raise ValueError("FilingPackage source document provenance must be an object")
    _require_fields(provenance, REQUIRED_PROVENANCE_FIELDS, "FilingPackage source document provenance")
    if provenance["source_kind"] not in ALLOWED_SOURCE_KINDS:
        raise ValueError("FilingPackage source document provenance source_kind must be local or fetched")
    _require_non_empty_string(provenance["source_name"], "FilingPackage source document provenance.source_name")
    if provenance["source_kind"] == "fetched":
        _require_non_empty_string(provenance["source_url"], "FilingPackage fetched provenance source_url")
        _require_non_empty_string(provenance["fetched_at"], "FilingPackage fetched provenance fetched_at")

    local_artifact = raw["local_artifact"]
    if not isinstance(local_artifact, dict):
        raise ValueError("FilingPackage source document local_artifact must be an object")
    _require_fields(local_artifact, REQUIRED_LOCAL_ARTIFACT_FIELDS, "FilingPackage source document local_artifact")
    _require_non_empty_string(local_artifact["artifact_id"], "FilingPackage source document local_artifact.artifact_id")
    _require_non_empty_string(local_artifact["path"], "FilingPackage source document local_artifact.path")

    fingerprint = raw["fingerprint"]
    if not isinstance(fingerprint, dict):
        raise ValueError("FilingPackage source document fingerprint must be an object")
    _require_fields(fingerprint, REQUIRED_FINGERPRINT_FIELDS, "FilingPackage source document fingerprint")
    _require_non_empty_string(fingerprint["hash_algorithm"], "FilingPackage source document fingerprint.hash_algorithm")
    _require_non_empty_string(fingerprint["hash_value"], "FilingPackage source document fingerprint.hash_value")

    if not isinstance(raw["reported_period_clues"], list):
        raise ValueError("FilingPackage source document reported_period_clues must be a list")
    if not isinstance(raw["basis_clues"], list):
        raise ValueError("FilingPackage source document basis_clues must be a list")

    amendment_clues = raw["amendment_clues"]
    if not isinstance(amendment_clues, dict):
        raise ValueError("FilingPackage source document amendment_clues must be an object")
    _require_fields(amendment_clues, REQUIRED_AMENDMENT_CLUE_FIELDS, "FilingPackage source document amendment_clues")
    if not isinstance(amendment_clues["is_amended_or_replacement"], bool):
        raise ValueError("FilingPackage source document amendment flag must be boolean")
    if not isinstance(amendment_clues["clues"], list):
        raise ValueError("FilingPackage source document amendment clues must be a list")

    return SourceDocument(raw=raw)


def _require_fields(raw: dict[str, Any], required_fields: set[str], label: str) -> None:
    missing = required_fields - raw.keys()
    if missing:
        raise ValueError(f"{label} is missing fields: {sorted(missing)}")


def _require_non_empty_string(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required")



# --- Document classification ---

ALLOWED_ROLES = {
    "financial_statement",
    "amendment_context_attachment",
    "cover_disclosure_letter",
    "cover_or_disclosure",
    "variance_explanation",
    "business_overview_out_of_scope",
    "business_overview",
    "out_of_scope",
    "unknown",
}
ALLOWED_LANGUAGES = {"vi", "en", "mixed", "unknown"}
ALLOWED_BASIS_FAMILIES = {"consolidated", "separate", "parent", "non_consolidated", "unknown"}
MIN_CLASSIFICATION_CONFIDENCE = 0.5
MAX_SNIPPET_CHARS = 160


@dataclass(frozen=True)
class SourceDocumentClassifierInput:
    vietstock_entry_metadata: dict[str, Any]
    source_document_metadata: dict[str, Any]
    filename: str
    first_few_page_evidence: dict[str, Any]


@dataclass(frozen=True)
class SourceDocumentClassification:
    source_document_id: str
    state: str
    document_role: str
    language: str
    basis_family: str
    visible_filing_label: str | None
    company_evidence: list[str]
    period_evidence: list[str]
    quarter_evidence: list[str]
    filing_status_hints: list[str]
    amendment_hints: list[str]
    assurance_hints: list[str]
    is_full_financial_statement: bool
    is_supporting_document: bool
    confidence: float
    visible_evidence_snippets: list[str]
    validation_errors: list[str]
    is_canonical_source_selection: bool = False
    approves_basis_switch: bool = False
    repairs_extraction: bool = False
    is_source_authority: bool = False


@dataclass(frozen=True)
class SourceDocumentClassificationResult:
    status: str
    classifications: list[SourceDocumentClassification]


class SourceDocumentClassifier(Protocol):
    def classify(self, classifier_input: SourceDocumentClassifierInput) -> dict[str, Any] | SourceDocumentClassification:
        ...


class DeterministicSourceDocumentClassifier:
    def classify(self, classifier_input: SourceDocumentClassifierInput) -> dict[str, Any]:
        evidence = classifier_input.first_few_page_evidence
        metadata = classifier_input.source_document_metadata
        text = _normalize(
            " ".join(
                [
                    classifier_input.filename,
                    str(classifier_input.vietstock_entry_metadata.get("title") or ""),
                    " ".join(evidence.get("visible_title_snippets") or []),
                    " ".join(evidence.get("role_clues") or []),
                ]
            )
        )
        role = _document_role(text, evidence)
        if role in {"out_of_scope", "cover_or_disclosure"} and metadata.get("document_type") == "main_financial_statement":
            role = "financial_statement"
        basis_family = _first_allowed(evidence.get("basis_clues") or metadata.get("basis_clues") or [], ALLOWED_BASIS_FAMILIES)
        language = _language(evidence, text)
        confidence = _confidence(role, evidence, text)
        snippets = _short_snippets(evidence.get("visible_title_snippets") or [])
        if not snippets and role == "financial_statement":
            snippets = _short_snippets([classifier_input.vietstock_entry_metadata.get("title")])
        company_evidence = _short_snippets(evidence.get("company_clues") or [])
        ticker = classifier_input.vietstock_entry_metadata.get("ticker")
        if isinstance(ticker, str) and ticker.strip():
            ticker_evidence = ticker.strip().upper()
            if ticker_evidence not in company_evidence:
                company_evidence.append(ticker_evidence)
        period_evidence = _short_snippets(evidence.get("period_clues") or [])
        if role == "financial_statement":
            for clue in _short_snippets(metadata.get("reported_period_clues") or []):
                if clue not in period_evidence:
                    period_evidence.append(clue)

        return {
            "source_document_id": metadata["source_document_id"],
            "document_role": role,
            "language": language,
            "basis_family": basis_family,
            "visible_filing_label": snippets[0] if snippets else None,
            "company_evidence": company_evidence,
            "period_evidence": period_evidence,
            "quarter_evidence": [clue for clue in period_evidence if "Q" in str(clue).upper()][:3],
            "filing_status_hints": _filing_status_hints(text),
            "amendment_hints": _amendment_hints(text),
            "assurance_hints": _assurance_hints(text),
            "is_full_financial_statement": role == "financial_statement",
            "is_supporting_document": role in {
                "amendment_context_attachment",
                "cover_disclosure_letter",
                "cover_or_disclosure",
                "variance_explanation",
            },
            "confidence": confidence,
            "visible_evidence_snippets": snippets,
        }


def classify_filing_package_source_documents(
    package: Any,
    *,
    classifier: SourceDocumentClassifier,
) -> SourceDocumentClassificationResult:
    raw = package.raw if hasattr(package, "raw") else package
    evidence_by_id = {
        evidence.get("source_document_id"): evidence
        for evidence in raw.get("source_evidence", [])
        if isinstance(evidence, dict)
    }
    classifications = []
    for document in raw.get("source_documents", []):
        classifier_input = build_source_document_classifier_input(raw, document, evidence_by_id.get(document.get("document_id"), {}))
        try:
            output = classifier.classify(classifier_input)
        except Exception as exc:
            output = {"source_document_id": document.get("document_id"), "_validation_error": str(exc)}
        classifications.append(validate_source_document_classification(output))
    status = "completed" if all(item.state == "classified" for item in classifications) else "completed_with_inconclusive_classifications"
    return SourceDocumentClassificationResult(status=status, classifications=classifications)


def build_source_document_classifier_input(
    raw_package: dict[str, Any],
    document: dict[str, Any],
    evidence: dict[str, Any],
) -> SourceDocumentClassifierInput:
    source_document_id = str(document.get("document_id") or "")
    filename = _direct_or_member_filename(document)
    return SourceDocumentClassifierInput(
        vietstock_entry_metadata={
            "file_id": (document.get("vietstock_entry") or {}).get("file_id"),
            "title": (document.get("vietstock_entry") or {}).get("title"),
            "url": (document.get("vietstock_entry") or {}).get("url"),
            "ticker": (raw_package.get("company") or {}).get("ticker"),
            "company_name": (raw_package.get("company") or {}).get("name"),
            "period": raw_package.get("period"),
            "quarter": raw_package.get("quarter"),
            "report_basis": raw_package.get("report_basis"),
        },
        source_document_metadata={
            "source_document_id": source_document_id,
            "document_type": document.get("document_type"),
            "reported_period_clues": list(document.get("reported_period_clues") or []),
            "basis_clues": list(document.get("basis_clues") or []),
            "amendment_clues": {
                "is_amended_or_replacement": bool((document.get("amendment_clues") or {}).get("is_amended_or_replacement")),
                "clues": list((document.get("amendment_clues") or {}).get("clues") or []),
            },
            "file_type": document.get("file_type"),
        },
        filename=filename,
        first_few_page_evidence=_bounded_evidence(evidence, source_document_id=source_document_id),
    )


def validate_source_document_classification(raw: dict[str, Any] | SourceDocumentClassification) -> SourceDocumentClassification:
    if isinstance(raw, SourceDocumentClassification):
        return raw
    errors = []
    source_document_id = raw.get("source_document_id") if isinstance(raw, dict) else None
    if not isinstance(raw, dict):
        errors.append("classification output must be an object")
        raw = {}
    role = raw.get("document_role")
    language = raw.get("language")
    basis_family = raw.get("basis_family")
    confidence = raw.get("confidence")
    if not isinstance(source_document_id, str) or not source_document_id:
        errors.append("source_document_id is required")
        source_document_id = "unknown"
    if role not in ALLOWED_ROLES:
        errors.append("document_role is invalid")
        role = "unknown"
    if language not in ALLOWED_LANGUAGES:
        errors.append("language is invalid")
        language = "unknown"
    if basis_family not in ALLOWED_BASIS_FAMILIES:
        errors.append("basis_family is invalid")
        basis_family = "unknown"
    if not isinstance(confidence, (int, float)):
        errors.append("confidence is required")
        confidence = 0.0
    snippets = _short_snippets(raw.get("visible_evidence_snippets") if isinstance(raw.get("visible_evidence_snippets"), list) else [])
    state = "classified"
    if errors or float(confidence) < MIN_CLASSIFICATION_CONFIDENCE:
        state = "inconclusive"
    return SourceDocumentClassification(
        source_document_id=source_document_id,
        state=state,
        document_role=role,
        language=language,
        basis_family=basis_family,
        visible_filing_label=_short_optional(raw.get("visible_filing_label")),
        company_evidence=_short_string_list(raw.get("company_evidence")),
        period_evidence=_short_string_list(raw.get("period_evidence")),
        quarter_evidence=_short_string_list(raw.get("quarter_evidence")),
        filing_status_hints=_short_string_list(raw.get("filing_status_hints")),
        amendment_hints=_short_string_list(raw.get("amendment_hints")),
        assurance_hints=_short_string_list(raw.get("assurance_hints")),
        is_full_financial_statement=bool(raw.get("is_full_financial_statement")) and state == "classified",
        is_supporting_document=bool(raw.get("is_supporting_document")) and state == "classified",
        confidence=float(confidence),
        visible_evidence_snippets=snippets,
        validation_errors=errors,
    )


def _bounded_evidence(evidence: dict[str, Any], *, source_document_id: str) -> dict[str, Any]:
    return {
        "source_document_id": source_document_id,
        "status": evidence.get("status"),
        "inspected_page_span": evidence.get("inspected_page_span"),
        "visible_title_snippets": _short_snippets(evidence.get("visible_title_snippets") or []),
        "company_clues": _short_snippets(evidence.get("company_clues") or []),
        "period_clues": _short_snippets(evidence.get("period_clues") or []),
        "basis_clues": list(evidence.get("basis_clues") or []),
        "language_clues": list(evidence.get("language_clues") or []),
        "role_clues": list(evidence.get("role_clues") or []),
        "confidence": evidence.get("confidence"),
        "warnings": _short_snippets(evidence.get("warnings") or []),
    }


def _direct_or_member_filename(document: dict[str, Any]) -> str:
    member = document.get("vietstock_member") or {}
    if member.get("filename"):
        return str(member["filename"])
    entry_url = (document.get("vietstock_entry") or {}).get("url")
    return str(entry_url or "")


def _document_role(text: str, evidence: dict[str, Any]) -> str:
    role_clues = set(evidence.get("role_clues") or [])
    if "business_overview" in role_clues or "tong quan tinh hinh kinh doanh" in text:
        return "business_overview"
    if "disclosure" in role_clues or "cong bo thong tin" in text or "information disclosure" in text:
        return "cover_or_disclosure"
    if ("thuyet minh" in text or "dieu chinh" in text) and not _proves_full_replacement(text):
        return "amendment_context_attachment"
    if "variance_explanation" in role_clues or "giai trinh" in text or "explanation" in text:
        return "variance_explanation"
    if "financial_statement" in role_clues or "bao cao tai chinh" in text or "financial statement" in text or "bctc" in text:
        return "financial_statement"
    return "out_of_scope"


def _proves_full_replacement(text: str) -> bool:
    return "thay the toan bo" in text or "full replacement" in text


def _language(evidence: dict[str, Any], text: str) -> str:
    clues = evidence.get("language_clues") or []
    if "en" in clues and "vi" in clues:
        return "mixed"
    if "en" in clues:
        return "en"
    if "vi" in clues:
        return "vi"
    if "financial statements" in text:
        return "en"
    if "bao cao" in text or "cong bo" in text or "bctc" in text:
        return "vi"
    return "unknown"


def _confidence(role: str, evidence: dict[str, Any], text: str) -> float:
    metadata_supports_financial_statement = role == "financial_statement" and (
        "bao cao tai chinh" in text or "bctc" in text
    )
    if evidence.get("status") != "completed":
        if metadata_supports_financial_statement:
            return 0.6
        return 0.0
    evidence_confidence = evidence.get("confidence")
    if evidence_confidence == "none":
        if metadata_supports_financial_statement:
            return 0.6
        return 0.0
    if evidence_confidence == "low":
        return 0.45
    if role in {"financial_statement", "cover_or_disclosure", "business_overview", "amendment_context_attachment", "variance_explanation"}:
        return 0.86
    return 0.6


def _filing_status_hints(text: str) -> list[str]:
    if any(marker in text for marker in ["dieu chinh", "bo sung", "sua doi", "thay the", "adjusted", "amended"]):
        return ["amendment"]
    return ["original"]


def _amendment_hints(text: str) -> list[str]:
    return [marker for marker in ["dieu chinh", "bo sung", "sua doi", "thay the", "adjusted", "amended"] if marker in text][:4]


def _assurance_hints(text: str) -> list[str]:
    hints = []
    if "soat xet" in text or "reviewed" in text:
        hints.append("reviewed")
    if "kiem toan" in text or "audited" in text:
        hints.append("audited")
    return hints


def _first_allowed(values: list[Any], allowed: set[str]) -> str:
    for value in values:
        if isinstance(value, str) and value in allowed:
            return value
    return "unknown"


def _short_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return _short_snippets(value)


def _short_optional(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return _sanitize_snippet(value)


def _short_snippets(values: list[Any]) -> list[str]:
    snippets = []
    for value in values:
        if not isinstance(value, str):
            continue
        snippet = _sanitize_snippet(value)
        if snippet and snippet not in snippets:
            snippets.append(snippet)
    return snippets[:6]


def _sanitize_snippet(value: str) -> str:
    cleaned = re.sub(r"\b(?:x|y)=\d+(?:\.\d+)?\b", "", value)
    cleaned = re.sub(r"\bpage_number\s*=\s*\d+\b", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:MAX_SNIPPET_CHARS].strip()


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_value = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", ascii_value).lower()



# --- Canonical source selection ---

NON_CONSOLIDATED_BASIS_FAMILIES = {"parent", "separate", "non_consolidated"}
SUPPORTING_DOCUMENT_ROLES = {
    "amendment_context_attachment",
    "cover_disclosure_letter",
    "cover_or_disclosure",
    "variance_explanation",
}


@dataclass(frozen=True)
class CanonicalSourceSelectionRecord:
    severity: str
    code: str
    message: str
    source_document_id: str | None = None


@dataclass(frozen=True)
class CanonicalSourceSelectionResult:
    status: str
    code: str | None
    selection: CanonicalSourceSelection | None
    records: list[CanonicalSourceSelectionRecord]
    ignored_document_ids: list[str]


def select_canonical_source_from_classifications(
    package: FilingPackage,
    classifications: list[SourceDocumentClassification],
) -> CanonicalSourceSelectionResult:
    classifications_by_id = {item.source_document_id: item for item in classifications}
    candidate_ids = []
    supporting_ids = []
    ignored_ids = []
    records: list[CanonicalSourceSelectionRecord] = []

    for document in package.source_documents:
        classification = classifications_by_id.get(document.document_id)
        if classification is None:
            ignored_ids.append(document.document_id)
            records.append(
                CanonicalSourceSelectionRecord(
                    severity="recoverable_data_error",
                    code="missing_classifier_record",
                    message="Source document has no classifier record.",
                    source_document_id=document.document_id,
                )
            )
            continue
        if classification.state != "classified":
            ignored_ids.append(document.document_id)
            records.append(
                CanonicalSourceSelectionRecord(
                    severity="recoverable_data_error",
                    code="inconclusive_classifier_record",
                    message="Inconclusive classifier record cannot become canonical.",
                    source_document_id=document.document_id,
                )
            )
            continue
        if classification.is_supporting_document or classification.document_role in SUPPORTING_DOCUMENT_ROLES:
            if _is_english_supporting_duplicate(package, classification, classifications):
                ignored_ids.append(document.document_id)
                records.append(
                    CanonicalSourceSelectionRecord(
                        severity="info",
                        code="english_supporting_duplicate_ignored",
                        message="English supporting duplicate is outside normal source selection because a matching Vietnamese supporting document exists.",
                        source_document_id=document.document_id,
                    )
                )
                continue
            supporting_ids.append(document.document_id)
            continue
        if _is_requested_vietnamese_main_fs(package, classification):
            candidate_ids.append(document.document_id)
        else:
            ignored_ids.append(document.document_id)
            if _is_english_requested_basis_main_fs(package, classification):
                records.append(
                    CanonicalSourceSelectionRecord(
                        severity="info",
                        code="english_financial_statement_duplicate_ignored",
                        message="English financial statement is outside normal Vietnamese canonical source selection.",
                        source_document_id=document.document_id,
                    )
                )

    if (
        len(candidate_ids) == 1
        and _is_searchable_version(package, candidate_ids[0])
        and not _identity_confirms_requested_filing(package, classifications_by_id[candidate_ids[0]])
    ):
        return CanonicalSourceSelectionResult(
            status="recoverable_data_error",
            code="searchable_identity_not_confirmed",
            selection=None,
            records=[
                *records,
                CanonicalSourceSelectionRecord(
                    severity="recoverable_data_error",
                    code="searchable_identity_not_confirmed",
                    message="Only in-scope Vietnamese financial statement is searchable, but identity checks did not confirm the requested filing.",
                    source_document_id=candidate_ids[0],
                ),
            ],
            ignored_document_ids=ignored_ids,
        )

    inconclusive_reviewed_ids = _inconclusive_reviewed_full_fs_identity_ids(package, classifications, candidate_ids)
    if inconclusive_reviewed_ids:
        return CanonicalSourceSelectionResult(
            status="recoverable_data_error",
            code="inconclusive_reviewed_full_fs_identity",
            selection=None,
            records=[
                *records,
                CanonicalSourceSelectionRecord(
                    severity="recoverable_data_error",
                    code="inconclusive_reviewed_full_fs_identity",
                    message="A possible reviewed full financial statement has inconclusive identity and may supersede the unreviewed candidate.",
                    source_document_id=inconclusive_reviewed_ids[0],
                ),
            ],
            ignored_document_ids=ignored_ids,
        )

    preferred = _select_preferred_candidate(package, candidate_ids, classifications_by_id)
    if preferred is not None:
        canonical_id, reason_code, reason_message, superseded_ids = preferred
        filing_status = _filing_status(classifications_by_id[canonical_id])
        amendment_context_attachment_ids = _linked_amendment_context_attachment_ids(
            package,
            canonical_id,
            supporting_ids,
            classifications_by_id,
        )
        supporting_document_links = _supporting_document_links(
            package,
            canonical_id,
            supporting_ids,
            classifications_by_id,
        )
        selection = CanonicalSourceSelection(
            canonical_source_document_id=canonical_id,
            filing_status=filing_status,
            supporting_document_ids=supporting_ids,
            requested_report_basis=package.report_basis,
            selected_basis_family=classifications_by_id[canonical_id].basis_family,
            visible_filing_label_by_document_id={
                item.source_document_id: item.visible_filing_label
                for item in classifications
                if item.visible_filing_label
            },
            assurance_type=_assurance_type(classifications_by_id[canonical_id]),
            superseded_source_document_ids=superseded_ids,
            amendment_context_attachment_ids=amendment_context_attachment_ids,
            supporting_document_links=supporting_document_links,
        )
        correction_records = [
            CanonicalSourceSelectionRecord(
                severity="recoverable_data_error",
                code="corrected_value_resolution_required",
                message="Amendment context attachment may affect operative values; corrected-value resolution is required before tools and detector packets.",
                source_document_id=attachment_id,
            )
            for attachment_id in amendment_context_attachment_ids
        ]
        return CanonicalSourceSelectionResult(
            status="selected",
            code=None,
            selection=selection,
            records=[
                *records,
                *correction_records,
                CanonicalSourceSelectionRecord(
                    severity="info",
                    code=reason_code,
                    message=reason_message,
                    source_document_id=canonical_id,
                ),
            ],
            ignored_document_ids=ignored_ids,
        )

    reviewed_order_blocker = _reviewed_supersession_order_blocker(package, candidate_ids, classifications_by_id)
    if reviewed_order_blocker is not None:
        return CanonicalSourceSelectionResult(
            status="hitl_needed",
            code="ambiguous_reviewed_supersession_order",
            selection=None,
            records=[
                *records,
                CanonicalSourceSelectionRecord(
                    severity="hitl_needed",
                    code="ambiguous_reviewed_supersession_order",
                    message="Reviewed and unreviewed same-identity full financial statements cannot be ordered by source provenance.",
                    source_document_id=reviewed_order_blocker,
                ),
            ],
            ignored_document_ids=ignored_ids,
        )

    if len(candidate_ids) > 1:
        return CanonicalSourceSelectionResult(
            status="hitl_needed",
            code="ambiguous_financial_statement_candidates",
            selection=None,
            records=[
                *records,
                CanonicalSourceSelectionRecord(
                    severity="hitl_needed",
                    code="ambiguous_financial_statement_candidates",
                    message="Multiple requested-basis Vietnamese financial statements cannot be ordered by locked policy.",
                ),
            ],
            ignored_document_ids=ignored_ids,
        )

    return CanonicalSourceSelectionResult(
        status="recoverable_data_error",
        code="missing_requested_basis_vietnamese_financial_statement",
        selection=None,
        records=[
            *records,
            CanonicalSourceSelectionRecord(
                severity="recoverable_data_error",
                code="missing_requested_basis_vietnamese_financial_statement",
                message="No classified requested-basis Vietnamese main financial statement is selectable.",
            ),
        ],
        ignored_document_ids=ignored_ids,
    )


def _linked_amendment_context_attachment_ids(
    package: FilingPackage,
    canonical_id: str,
    supporting_ids: list[str],
    classifications_by_id: dict[str, SourceDocumentClassification],
) -> list[str]:
    canonical = classifications_by_id[canonical_id]
    linked_ids = []
    for supporting_id in supporting_ids:
        supporting = classifications_by_id[supporting_id]
        if (
            supporting.document_role == "amendment_context_attachment"
            and _same_supporting_identity(package, canonical, supporting)
        ):
            linked_ids.append(supporting_id)
    return linked_ids


def _supporting_document_links(
    package: FilingPackage,
    canonical_id: str,
    supporting_ids: list[str],
    classifications_by_id: dict[str, SourceDocumentClassification],
) -> list[dict[str, str | bool | None]]:
    links = []
    canonical = classifications_by_id[canonical_id]
    for supporting_id in supporting_ids:
        attachment = classifications_by_id[supporting_id]
        if not _same_supporting_identity(package, canonical, attachment):
            continue
        requires_corrected_value_resolution = attachment.document_role == "amendment_context_attachment"
        links.append(
            {
                "source_document_id": supporting_id,
                "document_role": attachment.document_role,
                "visible_filing_label": attachment.visible_filing_label,
                "source_provenance": _bounded_source_provenance(package, supporting_id),
                "linked_canonical_source_document_id": canonical_id,
                "link_basis": "clear_company_period_quarter_basis_language_identity",
                "requires_corrected_value_resolution": requires_corrected_value_resolution,
                "corrected_value_resolution_status": "not_implemented" if requires_corrected_value_resolution else None,
            }
        )
    return links


def _bounded_source_provenance(package: FilingPackage, source_document_id: str) -> dict[str, str | None]:
    for document in package.source_documents:
        if document.document_id == source_document_id:
            provenance = document.provenance
            return {
                "source_kind": provenance.get("source_kind"),
                "source_name": provenance.get("source_name"),
            }
    return {"source_kind": None, "source_name": None}


def _select_preferred_candidate(
    package: FilingPackage,
    candidate_ids: list[str],
    classifications_by_id: dict[str, SourceDocumentClassification],
) -> tuple[str, str, str, list[str]] | None:
    if len(candidate_ids) == 1:
        return (
            candidate_ids[0],
            "canonical_source_selected",
            "Selected one requested-basis Vietnamese financial statement.",
            [],
        )

    amended_supersession = _select_amended_supersession(package, candidate_ids, classifications_by_id)
    if amended_supersession is not None:
        return amended_supersession

    reviewed_supersession = _select_reviewed_supersession(package, candidate_ids, classifications_by_id)
    if reviewed_supersession is not None:
        return reviewed_supersession

    searchable_ids = [candidate_id for candidate_id in candidate_ids if _is_searchable_version(package, candidate_id)]
    default_ids = [candidate_id for candidate_id in candidate_ids if candidate_id not in searchable_ids]
    confirmed_searchable_ids = [
        candidate_id
        for candidate_id in searchable_ids
        if _identity_confirms_requested_filing(package, classifications_by_id[candidate_id])
    ]
    if len(confirmed_searchable_ids) == 1 and len(default_ids) <= 1:
        return (
            confirmed_searchable_ids[0],
            "searchable_version_selected",
            "Selected confirmed searchable filing version over default official source.",
            [],
        )
    if len(searchable_ids) == 1 and len(default_ids) == 1:
        return (
            default_ids[0],
            "searchable_identity_not_confirmed_fallback",
            "Searchable filing version identity was not confirmed; selected default official source.",
            [],
        )
    return None


def _select_amended_supersession(
    package: FilingPackage,
    candidate_ids: list[str],
    classifications_by_id: dict[str, SourceDocumentClassification],
) -> tuple[str, str, str, list[str]] | None:
    amended_ids = [
        candidate_id
        for candidate_id in candidate_ids
        if _filing_status(classifications_by_id[candidate_id]) == "amended_or_replacement"
    ]
    original_ids = [
        candidate_id
        for candidate_id in candidate_ids
        if _filing_status(classifications_by_id[candidate_id]) == "original"
    ]
    if len(amended_ids) != 1 or not original_ids:
        return None
    amended_id = amended_ids[0]
    amended = classifications_by_id[amended_id]
    if not all(_same_requested_identity(package, amended, classifications_by_id[other_id]) for other_id in original_ids):
        return None
    return (
        amended_id,
        "amended_full_fs_supersedes_original",
        "Selected amended/replacement full financial statement over original same-identity statement.",
        original_ids,
    )


def _select_reviewed_supersession(
    package: FilingPackage,
    candidate_ids: list[str],
    classifications_by_id: dict[str, SourceDocumentClassification],
) -> tuple[str, str, str, list[str]] | None:
    reviewed_ids = [
        candidate_id
        for candidate_id in candidate_ids
        if _assurance_type(classifications_by_id[candidate_id]) == "reviewed"
    ]
    unreviewed_ids = [
        candidate_id
        for candidate_id in candidate_ids
        if _assurance_type(classifications_by_id[candidate_id]) != "reviewed"
    ]
    if len(reviewed_ids) != 1 or not unreviewed_ids:
        return None
    reviewed_id = reviewed_ids[0]
    reviewed = classifications_by_id[reviewed_id]
    if not all(_same_requested_identity(package, reviewed, classifications_by_id[other_id]) for other_id in unreviewed_ids):
        return None
    if not all(_document_fetched_after(package, reviewed_id, other_id) for other_id in unreviewed_ids):
        return None
    return (
        reviewed_id,
        "reviewed_full_fs_supersedes_unreviewed",
        "Selected later reviewed full financial statement over earlier unreviewed same-identity statement.",
        unreviewed_ids,
    )


def _reviewed_supersession_order_blocker(
    package: FilingPackage,
    candidate_ids: list[str],
    classifications_by_id: dict[str, SourceDocumentClassification],
) -> str | None:
    reviewed_ids = [
        candidate_id
        for candidate_id in candidate_ids
        if _assurance_type(classifications_by_id[candidate_id]) == "reviewed"
    ]
    unreviewed_ids = [
        candidate_id
        for candidate_id in candidate_ids
        if _assurance_type(classifications_by_id[candidate_id]) != "reviewed"
    ]
    if len(reviewed_ids) != 1 or not unreviewed_ids:
        return None
    reviewed_id = reviewed_ids[0]
    reviewed = classifications_by_id[reviewed_id]
    if not all(_same_requested_identity(package, reviewed, classifications_by_id[other_id]) for other_id in unreviewed_ids):
        return None
    if any(not _document_fetched_after(package, reviewed_id, other_id) for other_id in unreviewed_ids):
        return reviewed_id
    return None


def _inconclusive_reviewed_full_fs_identity_ids(
    package: FilingPackage,
    classifications: list[SourceDocumentClassification],
    candidate_ids: list[str],
) -> list[str]:
    if not candidate_ids:
        return []
    return [
        item.source_document_id
        for item in classifications
        if item.source_document_id not in candidate_ids
        and item.state != "classified"
        and item.document_role == "financial_statement"
        and item.language == "vi"
        and _basis_matches(package.report_basis, item.basis_family)
        and _assurance_type(item) == "reviewed"
    ]


def _is_requested_vietnamese_main_fs(
    package: FilingPackage,
    classification: SourceDocumentClassification,
) -> bool:
    return (
        classification.document_role == "financial_statement"
        and classification.is_full_financial_statement
        and classification.language in {"vi", "mixed"}
        and _basis_matches(package.report_basis, classification.basis_family)
    )


def _is_english_requested_basis_main_fs(
    package: FilingPackage,
    classification: SourceDocumentClassification,
) -> bool:
    return (
        classification.document_role == "financial_statement"
        and classification.is_full_financial_statement
        and classification.language == "en"
        and _basis_matches(package.report_basis, classification.basis_family)
    )


def _is_english_supporting_duplicate(
    package: FilingPackage,
    classification: SourceDocumentClassification,
    classifications: list[SourceDocumentClassification],
) -> bool:
    return (
        classification.language == "en"
        and (
            any(_is_requested_vietnamese_main_fs(package, other) for other in classifications)
            or any(
                other.source_document_id != classification.source_document_id
                and other.state == "classified"
                and other.is_supporting_document
                and other.document_role == classification.document_role
                and other.language == "vi"
                and _basis_matches(classification.basis_family, other.basis_family)
                for other in classifications
            )
        )
    )


def _basis_matches(requested_basis: str, classified_basis: str) -> bool:
    if requested_basis == classified_basis:
        return True
    if requested_basis in NON_CONSOLIDATED_BASIS_FAMILIES:
        return classified_basis in NON_CONSOLIDATED_BASIS_FAMILIES
    return False


def _is_searchable_version(package: FilingPackage, source_document_id: str) -> bool:
    for document in package.source_documents:
        if document.document_id != source_document_id:
            continue
        filename = _normalize_search_text((document.raw.get("vietstock_member") or {}).get("filename") or "")
        return "tra cuu" in filename or "searchable" in filename
    return False


def _identity_confirms_requested_filing(
    package: FilingPackage,
    classification: SourceDocumentClassification,
) -> bool:
    company_tokens = [
        (package.raw.get("company") or {}).get("ticker"),
        package.company_name,
    ]
    return (
        _any_evidence_matches(classification.company_evidence, company_tokens)
        and _any_evidence_matches(classification.period_evidence, [package.period])
        and _any_evidence_matches(classification.quarter_evidence, [package.raw.get("quarter")])
        and _basis_matches(package.report_basis, classification.basis_family)
        and classification.document_role == "financial_statement"
        and classification.is_full_financial_statement
        and _filing_status(classification) == (package.raw.get("filing_event") or {}).get("filing_status", "original")
    )


def _same_requested_identity(
    package: FilingPackage,
    left: SourceDocumentClassification,
    right: SourceDocumentClassification,
) -> bool:
    company_tokens = [(package.raw.get("company") or {}).get("ticker"), package.company_name]
    return (
        _any_evidence_matches(left.company_evidence, company_tokens)
        and _any_evidence_matches(right.company_evidence, company_tokens)
        and _any_evidence_matches(left.period_evidence, [package.period])
        and _any_evidence_matches(right.period_evidence, [package.period])
        and _any_evidence_matches(left.quarter_evidence, [package.raw.get("quarter")])
        and _any_evidence_matches(right.quarter_evidence, [package.raw.get("quarter")])
        and left.language == right.language == "vi"
        and _basis_matches(left.basis_family, right.basis_family)
        and _basis_matches(package.report_basis, left.basis_family)
        and left.document_role == right.document_role == "financial_statement"
        and left.is_full_financial_statement
        and right.is_full_financial_statement
    )


def _same_supporting_identity(
    package: FilingPackage,
    canonical: SourceDocumentClassification,
    supporting: SourceDocumentClassification,
) -> bool:
    company_tokens = [(package.raw.get("company") or {}).get("ticker"), package.company_name]
    return (
        _any_evidence_matches(canonical.company_evidence, company_tokens)
        and _any_evidence_matches(supporting.company_evidence, company_tokens)
        and _any_evidence_matches(canonical.period_evidence, [package.period])
        and _any_evidence_matches(supporting.period_evidence, [package.period])
        and _any_evidence_matches(canonical.quarter_evidence, [package.raw.get("quarter")])
        and _any_evidence_matches(supporting.quarter_evidence, [package.raw.get("quarter")])
        and canonical.language == supporting.language == "vi"
        and _basis_matches(package.report_basis, canonical.basis_family)
        and _basis_matches(package.report_basis, supporting.basis_family)
    )


def _document_fetched_after(package: FilingPackage, later_id: str, earlier_id: str) -> bool:
    fetched_at_by_id = {
        document.document_id: (document.provenance.get("fetched_at") or "")
        for document in package.source_documents
    }
    return fetched_at_by_id.get(later_id, "") > fetched_at_by_id.get(earlier_id, "")


def _any_evidence_matches(evidence_values: list[str], expected_values: list[str | None]) -> bool:
    evidence = " ".join(_normalize_search_text(value) for value in evidence_values)
    return any(
        expected and _normalize_search_text(expected) in evidence
        for expected in expected_values
    )


def _normalize_search_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value))
    ascii_value = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(re.sub(r"[^a-zA-Z0-9]+", " ", ascii_value).lower().split())


def _filing_status(classification: SourceDocumentClassification) -> str:
    if "amended_or_replacement" in classification.filing_status_hints:
        return "amended_or_replacement"
    if "amendment" in classification.filing_status_hints:
        return "amended_or_replacement"
    return "original"


def _assurance_type(classification: SourceDocumentClassification) -> str:
    normalized_hints = {_normalize_search_text(hint) for hint in classification.assurance_hints}
    if "reviewed" in normalized_hints or "soat xet" in normalized_hints or "soat_xet" in classification.assurance_hints:
        return "reviewed"
    if "audited" in normalized_hints or "kiem toan" in normalized_hints:
        return "audited"
    return "unaudited"
