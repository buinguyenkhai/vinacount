"""Deterministic checks, CandidateRisk construction, and DetectorPacket assembly.

Each check reads a CompanyReportSet (target filing and same-quarter prior-year filing) and emits a ToolFinding with
its calculation, threshold, flag, and source references. A check abstains with an unflagged finding when an input is
missing, the report profile does not apply, or a denominator is zero. Flagged findings form CandidateRisks; each
candidate's references select the rows, notes, explanations, findings, and rules that enter its DetectorPacket.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from vinacount.contract import (
    ALLOWED_SUPPORT_LEVELS,
    CandidateRisk,
    DetectorPacket,
    ToolFinding,
    enrich_detector_packet_evidence_roles,
    validate_detector_packet,
)
from vinacount.report_memory import CompanyReportSet, ReportMemory


def run_checks_to_packets(report_set: CompanyReportSet) -> dict[str, Any]:
    """Run every deterministic check, build CandidateRisks from flagged findings, and assemble one packet each."""
    findings = run_deterministic_tool_checks(report_set)
    candidates = build_candidate_risks(report_set, findings)
    packets = build_detector_packets(report_set, candidates, findings)
    return {
        "report_set_id": report_set.case_id,
        "tool_findings": [_to_dict(finding) for finding in findings],
        "candidate_risks": [_to_dict(candidate) for candidate in candidates],
        "detector_packets": [_to_dict(packet) for packet in packets],
    }


def run_deterministic_tool_checks(report_set: CompanyReportSet) -> list[ToolFinding]:
    return [
        _build_revenue_growth_finding(report_set),
        _build_receivables_vs_revenue_finding(report_set),
        _build_earnings_cashflow_mismatch_finding(report_set),
        _build_securities_fvtpl_cashflow_bridge_finding(report_set),
        _build_insurance_premium_receivables_gwp_finding(report_set),
        _build_credit_institution_loan_quality_finding(report_set),
        _build_related_party_disclosure_finding(report_set),
        _build_accounting_policy_disclosure_finding(report_set),
        _build_variance_explanation_disclosure_finding(report_set),
    ]


def build_candidate_risks(report_set: CompanyReportSet, findings: list[ToolFinding]) -> list[CandidateRisk]:
    flagged = {finding.signal_id: finding for finding in findings if finding.flag and finding.evidence_refs}
    candidates = []
    required_signals = {"revenue_growth_high", "receivables_growth_outpaces_revenue"}
    if required_signals <= flagged.keys():
        linked = [flagged["revenue_growth_high"], flagged["receivables_growth_outpaces_revenue"]]
        candidates.append(
            CandidateRisk(
                candidate_id=f"CAND_{report_set.target.report_id}_REV_REC_001",
                report_id=report_set.target.report_id,
                risk_category="revenue_income_recognition_risk",
                reason_for_candidate=(
                    "Revenue increased significantly while receivables grew faster than revenue, "
                    "requiring review of revenue quality risk."
                ),
                priority="high",
                supporting_signal_ids=[finding.signal_id for finding in linked],
                linked_tool_result_ids=[finding.tool_result_id for finding in linked],
                evidence_refs=_dedupe_evidence_refs(
                    [
                        {"evidence_ref_type": "tool_result", "ref_id": finding.tool_result_id, "role": "supporting"}
                        for finding in linked
                    ]
                    + [ref for finding in linked for ref in finding.evidence_refs]
                ),
            )
        )
    cashflow_signal = flagged.get("positive_profit_negative_cfo") or flagged.get("cfo_net_income_gap_high")
    if cashflow_signal is not None:
        candidates.append(
            CandidateRisk(
                candidate_id=f"CAND_{report_set.target.report_id}_CASH_001",
                report_id=report_set.target.report_id,
                risk_category="earnings_cashflow_mismatch",
                reason_for_candidate=(
                    "The company reported profit with weak operating cash-flow support, requiring review "
                    "of earnings quality and cash realization."
                ),
                priority="high",
                supporting_signal_ids=[cashflow_signal.signal_id],
                linked_tool_result_ids=[cashflow_signal.tool_result_id],
                evidence_refs=_dedupe_evidence_refs(
                    [{"evidence_ref_type": "tool_result", "ref_id": cashflow_signal.tool_result_id, "role": "supporting"}]
                    + cashflow_signal.evidence_refs
                ),
            )
        )
    securities_cashflow_signal = flagged.get("securities_fvtpl_profit_bridge_high")
    if securities_cashflow_signal is not None:
        candidates.append(
            CandidateRisk(
                candidate_id=f"CAND_{report_set.target.report_id}_SEC_FVTPL_CASH_001",
                report_id=report_set.target.report_id,
                risk_category="earnings_cashflow_mismatch",
                reason_for_candidate=(
                    "Securities-company profit includes material non-cash FVTPL gains with weak cash-flow support, "
                    "requiring review of the earnings cash-flow bridge."
                ),
                priority="high",
                supporting_signal_ids=[securities_cashflow_signal.signal_id],
                linked_tool_result_ids=[securities_cashflow_signal.tool_result_id],
                evidence_refs=_dedupe_evidence_refs(
                    [
                        {
                            "evidence_ref_type": "tool_result",
                            "ref_id": securities_cashflow_signal.tool_result_id,
                            "role": "supporting",
                        }
                    ]
                    + securities_cashflow_signal.evidence_refs
                ),
            )
        )
    insurance_receivables_signal = flagged.get("insurance_premium_receivables_outpace_gwp")
    if insurance_receivables_signal is not None:
        candidates.append(
            CandidateRisk(
                candidate_id=f"CAND_{report_set.target.report_id}_INS_PREM_REC_001",
                report_id=report_set.target.report_id,
                risk_category="receivables_credit_quality_risk",
                reason_for_candidate=(
                    "Premium receivables grew materially faster than gross written premium, "
                    "requiring review of insurance receivable quality."
                ),
                priority="high",
                supporting_signal_ids=[insurance_receivables_signal.signal_id],
                linked_tool_result_ids=[insurance_receivables_signal.tool_result_id],
                evidence_refs=_dedupe_evidence_refs(
                    [
                        {
                            "evidence_ref_type": "tool_result",
                            "ref_id": insurance_receivables_signal.tool_result_id,
                            "role": "supporting",
                        }
                    ]
                    + insurance_receivables_signal.evidence_refs
                ),
            )
        )
    bank_loan_quality_signal = flagged.get("credit_institution_npl_ratio_rising_without_coverage_improvement")
    if bank_loan_quality_signal is not None:
        candidates.append(
            CandidateRisk(
                candidate_id=f"CAND_{report_set.target.report_id}_BANK_LOAN_QUALITY_001",
                report_id=report_set.target.report_id,
                risk_category="receivables_credit_quality_risk",
                reason_for_candidate=(
                    "Credit-institution loan quality evidence shows a rising non-performing-loan ratio without "
                    "improved provision coverage, requiring review of credit quality."
                ),
                priority="high",
                supporting_signal_ids=[bank_loan_quality_signal.signal_id],
                linked_tool_result_ids=[bank_loan_quality_signal.tool_result_id],
                evidence_refs=_dedupe_evidence_refs(
                    [
                        {
                            "evidence_ref_type": "tool_result",
                            "ref_id": bank_loan_quality_signal.tool_result_id,
                            "role": "supporting",
                        }
                    ]
                    + bank_loan_quality_signal.evidence_refs
                ),
            )
        )
    related_party_signal = flagged.get("related_party_disclosure_present")
    if related_party_signal is not None:
        candidates.append(
            CandidateRisk(
                candidate_id=f"CAND_{report_set.target.report_id}_RPT_DISC_001",
                report_id=report_set.target.report_id,
                risk_category="related_party_disclosure_risk",
                reason_for_candidate=(
                    "Source-bound related-party disclosure evidence is present and requires bounded detector review "
                    "of whether the disclosure supports a related-party risk signal."
                ),
                priority="medium",
                supporting_signal_ids=[related_party_signal.signal_id],
                linked_tool_result_ids=[related_party_signal.tool_result_id],
                evidence_refs=_dedupe_evidence_refs(
                    [
                        {
                            "evidence_ref_type": "tool_result",
                            "ref_id": related_party_signal.tool_result_id,
                            "role": "supporting",
                        }
                    ]
                    + related_party_signal.evidence_refs
                ),
            )
        )
    accounting_policy_signal = flagged.get("accounting_policy_disclosure_present")
    if accounting_policy_signal is not None:
        candidates.append(
            CandidateRisk(
                candidate_id=f"CAND_{report_set.target.report_id}_POLICY_DISC_001",
                report_id=report_set.target.report_id,
                risk_category="disclosure_inconsistency_or_obfuscation",
                reason_for_candidate=(
                    "Source-bound accounting-policy disclosure evidence is present and requires bounded detector "
                    "review for disclosure consistency or obfuscation."
                ),
                priority="medium",
                supporting_signal_ids=[accounting_policy_signal.signal_id],
                linked_tool_result_ids=[accounting_policy_signal.tool_result_id],
                evidence_refs=_dedupe_evidence_refs(
                    [
                        {
                            "evidence_ref_type": "tool_result",
                            "ref_id": accounting_policy_signal.tool_result_id,
                            "role": "supporting",
                        }
                    ]
                    + accounting_policy_signal.evidence_refs
                ),
            )
        )
    variance_signal = flagged.get("variance_explanation_present")
    if variance_signal is not None:
        candidates.append(
            CandidateRisk(
                candidate_id=f"CAND_{report_set.target.report_id}_VAR_DISC_001",
                report_id=report_set.target.report_id,
                risk_category="disclosure_inconsistency_or_obfuscation",
                reason_for_candidate=(
                    "Source-bound variance explanation evidence is present and requires bounded detector review "
                    "for disclosure consistency or explanation quality."
                ),
                priority="medium",
                supporting_signal_ids=[variance_signal.signal_id],
                linked_tool_result_ids=[variance_signal.tool_result_id],
                evidence_refs=_dedupe_evidence_refs(
                    [
                        {
                            "evidence_ref_type": "tool_result",
                            "ref_id": variance_signal.tool_result_id,
                            "role": "supporting",
                        }
                    ]
                    + variance_signal.evidence_refs
                ),
            )
        )
    return candidates


def build_detector_packets(
    report_set: CompanyReportSet,
    candidates: list[CandidateRisk],
    findings: list[ToolFinding],
) -> list[DetectorPacket]:
    findings_by_id = {finding.tool_result_id: finding for finding in findings}
    packets = []
    for index, candidate in enumerate(candidates, start=1):
        if not candidate.evidence_refs:
            raise ValueError("CandidateRisk requires evidence refs before DetectorPacket construction")
        linked_findings = [findings_by_id[tool_result_id] for tool_result_id in candidate.linked_tool_result_ids]
        packet = DetectorPacket(
            packet_id=f"PACKET_{candidate.report_id}_{index:03d}",
            candidate_id=candidate.candidate_id,
            report_id=candidate.report_id,
            task={
                "risk_category": candidate.risk_category,
                "question": "Does the provided evidence support the candidate risk signal?",
                "expected_output": "Return a structured DetectorAssessment using risk-signal language only.",
            },
            metadata={
                "company_name": report_set.target.metadata["company_name"],
                "period": report_set.target.metadata["period"],
                "report_profile": report_set.target.metadata["report_profile"],
                "report_basis": report_set.target.metadata["report_basis"],
                "currency": report_set.target.metadata["currency"],
                "unit": report_set.target.metadata["unit"],
                "language": "vi",
            },
            candidate_summary={
                "reason_for_candidate": candidate.reason_for_candidate,
                "priority": candidate.priority,
                "supporting_signal_ids": candidate.supporting_signal_ids,
            },
            relevant_table_rows=_packet_table_rows(report_set, candidate),
            relevant_notes=_packet_notes(report_set, candidate),
            relevant_variance_explanations=_packet_variance_explanations(report_set, candidate),
            tool_findings=[_packet_tool_finding(finding) for finding in linked_findings],
            rules=_packet_rules(candidate),
            constraints={
                "allowed_decisions": sorted(ALLOWED_SUPPORT_LEVELS),
                "evidence_must_reference_provided_ids": True,
                "do_not_claim_fraud": True,
                "max_rationale_sentences": 3,
            },
            report_set_id=report_set.case_id,
        )
        packet = DetectorPacket(**enrich_detector_packet_evidence_roles(packet))
        validate_detector_packet(packet)
        packets.append(packet)
    return packets


def _build_revenue_growth_finding(report_set: CompanyReportSet) -> ToolFinding:
    if report_set.target.metadata.get("report_profile") != "standard_corporate":
        return _standard_corporate_abstention_finding(
            report_set,
            tool_result_id=f"TOOL_{report_set.target.report_id}_REV_GROWTH_001",
            tool_name="standard_corporate_revenue_growth_tool",
            risk_category="revenue_income_recognition_risk",
            signal_id="standard_corporate_revenue_wrong_profile",
            finding_summary="Standard-corporate revenue growth check is not applicable to this report profile.",
            metric_name="revenue_growth_pct",
            threshold={"threshold_type": "greater_than", "value": 20.0, "unit": "percent"},
        )
    current = _require_account_cell(report_set.target, "revenue")
    comparison = _require_account_cell(report_set.prior_year, "revenue")
    growth_pct = _percentage_growth(current["value"], comparison["value"])
    flag = growth_pct > 20.0
    return ToolFinding(
        tool_result_id=f"TOOL_{report_set.target.report_id}_REV_GROWTH_001",
        report_id=report_set.target.report_id,
        tool_name="standard_corporate_revenue_growth_tool",
        risk_category="revenue_income_recognition_risk",
        signal_id="revenue_growth_high" if flag else "revenue_growth_not_high",
        flag=flag,
        finding_summary=f"Revenue increased by {growth_pct:.2f}% year over year.",
        evidence_refs=[
            _cell_evidence_ref(report_set.target.report_id, current["cell_id"]),
            _cell_evidence_ref(report_set.prior_year.report_id, comparison["cell_id"]),
        ],
        metric={"metric_name": "revenue_growth_pct", "value": growth_pct, "unit": "percent"},
        threshold={"threshold_type": "greater_than", "value": 20.0, "unit": "percent"},
    )


def _build_receivables_vs_revenue_finding(report_set: CompanyReportSet) -> ToolFinding:
    if report_set.target.metadata.get("report_profile") != "standard_corporate":
        return _standard_corporate_abstention_finding(
            report_set,
            tool_result_id=f"TOOL_{report_set.target.report_id}_REC_VS_REV_001",
            tool_name="standard_corporate_receivables_vs_revenue_growth_tool",
            risk_category="revenue_income_recognition_risk",
            signal_id="standard_corporate_receivables_vs_revenue_wrong_profile",
            finding_summary="Standard-corporate receivables versus revenue check is not applicable to this report profile.",
            metric_name="receivables_growth_minus_revenue_growth_pct_points",
            threshold={"threshold_type": "greater_than", "value": 10.0, "unit": "percentage_points"},
        )
    current_receivables = _require_account_cell(report_set.target, "trade_receivables")
    comparison_receivables = _require_account_cell(report_set.prior_year, "trade_receivables")
    current_revenue = _require_account_cell(report_set.target, "revenue")
    comparison_revenue = _require_account_cell(report_set.prior_year, "revenue")
    receivables_growth = _percentage_growth(current_receivables["value"], comparison_receivables["value"])
    revenue_growth = _percentage_growth(current_revenue["value"], comparison_revenue["value"])
    growth_gap = round(receivables_growth - revenue_growth, 2)
    flag = growth_gap > 10.0
    return ToolFinding(
        tool_result_id=f"TOOL_{report_set.target.report_id}_REC_VS_REV_001",
        report_id=report_set.target.report_id,
        tool_name="standard_corporate_receivables_vs_revenue_growth_tool",
        risk_category="revenue_income_recognition_risk",
        signal_id="receivables_growth_outpaces_revenue" if flag else "receivables_growth_not_outpacing_revenue",
        flag=flag,
        finding_summary=f"Receivables growth exceeded revenue growth by {growth_gap:.2f} percentage points.",
        evidence_refs=[
            _cell_evidence_ref(report_set.target.report_id, current_receivables["cell_id"]),
            _cell_evidence_ref(report_set.prior_year.report_id, comparison_receivables["cell_id"]),
            _cell_evidence_ref(report_set.target.report_id, current_revenue["cell_id"]),
            _cell_evidence_ref(report_set.prior_year.report_id, comparison_revenue["cell_id"]),
        ],
        metric={"metric_name": "receivables_growth_minus_revenue_growth_pct_points", "value": growth_gap, "unit": "percentage_points"},
        threshold={"threshold_type": "greater_than", "value": 10.0, "unit": "percentage_points"},
    )


def _standard_corporate_abstention_finding(
    report_set: CompanyReportSet,
    *,
    tool_result_id: str,
    tool_name: str,
    risk_category: str,
    signal_id: str,
    finding_summary: str,
    metric_name: str,
    threshold: dict[str, Any],
) -> ToolFinding:
    return ToolFinding(
        tool_result_id=tool_result_id,
        report_id=report_set.target.report_id,
        tool_name=tool_name,
        risk_category=risk_category,
        signal_id=signal_id,
        flag=False,
        finding_summary=finding_summary,
        evidence_refs=[],
        metric={"metric_name": metric_name, "value": None, "unit": threshold["unit"]},
        threshold=threshold,
    )


def _build_earnings_cashflow_mismatch_finding(report_set: CompanyReportSet) -> ToolFinding:
    if report_set.target.metadata.get("report_profile") != "standard_corporate":
        return _cashflow_abstention_finding(
            report_set,
            signal_id="earnings_cashflow_wrong_profile",
            finding_summary="Earnings cash-flow mismatch check is not applicable to this report profile.",
        )
    try:
        profit = _require_account_cell(report_set.target, "profit_after_tax")
        operating_cash_flow = _require_account_cell(report_set.target, "operating_cash_flow")
    except ValueError:
        return _cashflow_abstention_finding(
            report_set,
            signal_id="earnings_cashflow_missing_evidence",
            finding_summary="Earnings cash-flow mismatch check skipped because profit or operating cash flow evidence is missing.",
        )
    if not _cell_period_matches_report(report_set.target, profit) or not _cell_period_matches_report(
        report_set.target,
        operating_cash_flow,
    ):
        return ToolFinding(
            tool_result_id=f"TOOL_{report_set.target.report_id}_CASH_001",
            report_id=report_set.target.report_id,
            tool_name="standard_corporate_earnings_cashflow_mismatch_tool",
            risk_category="earnings_cashflow_mismatch",
            signal_id="earnings_cashflow_wrong_period",
            flag=False,
            finding_summary="Earnings cash-flow mismatch check skipped because profit or operating cash flow evidence is outside the target period.",
            evidence_refs=[
                _cell_evidence_ref(report_set.target.report_id, profit["cell_id"]),
                _cell_evidence_ref(report_set.target.report_id, operating_cash_flow["cell_id"]),
            ],
            metric={"metric_name": "operating_cash_flow_to_abs_profit", "value": None, "unit": "ratio"},
            threshold={"threshold_type": "less_than", "value": 0.7, "unit": "ratio"},
        )
    profit_value = profit["value"]
    ocf_value = operating_cash_flow["value"]
    if profit_value == 0:
        return ToolFinding(
            tool_result_id=f"TOOL_{report_set.target.report_id}_CASH_001",
            report_id=report_set.target.report_id,
            tool_name="standard_corporate_earnings_cashflow_mismatch_tool",
            risk_category="earnings_cashflow_mismatch",
            signal_id="earnings_cashflow_zero_denominator",
            flag=False,
            finding_summary="Earnings cash-flow mismatch check skipped because profit after tax is zero.",
            evidence_refs=[
                _cell_evidence_ref(report_set.target.report_id, profit["cell_id"]),
                _cell_evidence_ref(report_set.target.report_id, operating_cash_flow["cell_id"]),
            ],
            metric={"metric_name": "operating_cash_flow_to_abs_profit", "value": None, "unit": "ratio"},
            threshold={"threshold_type": "less_than", "value": 0.7, "unit": "ratio"},
        )
    ratio = round(ocf_value / abs(profit_value), 4)
    positive_profit_negative_cfo = profit_value > 0 and ocf_value < 0
    flag = ratio < 0.7 or positive_profit_negative_cfo
    signal_id = (
        "positive_profit_negative_cfo"
        if positive_profit_negative_cfo
        else "cfo_net_income_gap_high"
        if flag
        else "earnings_cashflow_support_not_weak"
    )
    return ToolFinding(
        tool_result_id=f"TOOL_{report_set.target.report_id}_CASH_001",
        report_id=report_set.target.report_id,
        tool_name="standard_corporate_earnings_cashflow_mismatch_tool",
        risk_category="earnings_cashflow_mismatch",
        signal_id=signal_id,
        flag=flag,
        finding_summary=(
            f"Operating cash flow was {ratio:.2f} times absolute profit after tax; "
            "positive profit with negative operating cash flow was observed."
            if positive_profit_negative_cfo
            else f"Operating cash flow was {ratio:.2f} times absolute profit after tax."
        ),
        evidence_refs=[
            _cell_evidence_ref(report_set.target.report_id, profit["cell_id"]),
            _cell_evidence_ref(report_set.target.report_id, operating_cash_flow["cell_id"]),
        ],
        metric={"metric_name": "operating_cash_flow_to_abs_profit", "value": ratio, "unit": "ratio"},
        threshold={"threshold_type": "less_than", "value": 0.7, "unit": "ratio"},
    )


def _build_securities_fvtpl_cashflow_bridge_finding(report_set: CompanyReportSet) -> ToolFinding:
    if report_set.target.metadata.get("report_profile") != "securities":
        return _securities_fvtpl_abstention_finding(
            report_set,
            signal_id="securities_fvtpl_wrong_profile",
            finding_summary="Securities FVTPL cash-flow bridge check is not applicable to this report profile.",
        )
    try:
        profit = _require_account_cell(report_set.target, "profit_after_tax")
        fvtpl_gain = _require_account_cell(report_set.target, "fvtpl_unrealized_gain")
        operating_cash_flow = _require_account_cell(report_set.target, "operating_cash_flow")
    except ValueError:
        return _securities_fvtpl_abstention_finding(
            report_set,
            signal_id="securities_fvtpl_missing_evidence",
            finding_summary="Securities FVTPL cash-flow bridge check skipped because required evidence is missing.",
        )
    if not all(
        _cell_period_matches_report(report_set.target, cell)
        for cell in [profit, fvtpl_gain, operating_cash_flow]
    ):
        return ToolFinding(
            tool_result_id=f"TOOL_{report_set.target.report_id}_SEC_FVTPL_CASH_001",
            report_id=report_set.target.report_id,
            tool_name="ctck_earnings_cash_bridge_tool",
            risk_category="earnings_cashflow_mismatch",
            signal_id="securities_fvtpl_wrong_period",
            flag=False,
            finding_summary="Securities FVTPL cash-flow bridge check skipped because required evidence is outside the target period.",
            evidence_refs=[
                _cell_evidence_ref(report_set.target.report_id, profit["cell_id"]),
                _cell_evidence_ref(report_set.target.report_id, fvtpl_gain["cell_id"]),
                _cell_evidence_ref(report_set.target.report_id, operating_cash_flow["cell_id"]),
            ],
            metric={"metric_name": "fvtpl_unrealized_gain_to_abs_profit", "value": None, "unit": "ratio"},
            threshold={"threshold_type": "greater_than", "value": 0.3, "unit": "ratio"},
        )
    profit_value = profit["value"]
    if profit_value == 0:
        return ToolFinding(
            tool_result_id=f"TOOL_{report_set.target.report_id}_SEC_FVTPL_CASH_001",
            report_id=report_set.target.report_id,
            tool_name="ctck_earnings_cash_bridge_tool",
            risk_category="earnings_cashflow_mismatch",
            signal_id="securities_fvtpl_zero_denominator",
            flag=False,
            finding_summary="Securities FVTPL cash-flow bridge check skipped because profit after tax is zero.",
            evidence_refs=[
                _cell_evidence_ref(report_set.target.report_id, profit["cell_id"]),
                _cell_evidence_ref(report_set.target.report_id, fvtpl_gain["cell_id"]),
                _cell_evidence_ref(report_set.target.report_id, operating_cash_flow["cell_id"]),
            ],
            metric={"metric_name": "fvtpl_unrealized_gain_to_abs_profit", "value": None, "unit": "ratio"},
            threshold={"threshold_type": "greater_than", "value": 0.3, "unit": "ratio"},
        )
    ratio = round(abs(fvtpl_gain["value"]) / abs(profit_value), 4)
    weak_cash_support = operating_cash_flow["value"] < 0 or operating_cash_flow["value"] / abs(profit_value) < 0.7
    flag = ratio > 0.3 and weak_cash_support
    return ToolFinding(
        tool_result_id=f"TOOL_{report_set.target.report_id}_SEC_FVTPL_CASH_001",
        report_id=report_set.target.report_id,
        tool_name="ctck_earnings_cash_bridge_tool",
        risk_category="earnings_cashflow_mismatch",
        signal_id="securities_fvtpl_profit_bridge_high" if flag else "securities_fvtpl_cash_bridge_not_high",
        flag=flag,
        finding_summary=(
            f"Unrealized FVTPL gains were {ratio:.2f} times absolute profit after tax, "
            "with weak operating cash-flow support."
        ),
        evidence_refs=[
            _cell_evidence_ref(report_set.target.report_id, profit["cell_id"]),
            _cell_evidence_ref(report_set.target.report_id, fvtpl_gain["cell_id"]),
            _cell_evidence_ref(report_set.target.report_id, operating_cash_flow["cell_id"]),
        ],
        metric={"metric_name": "fvtpl_unrealized_gain_to_abs_profit", "value": ratio, "unit": "ratio"},
        threshold={"threshold_type": "greater_than", "value": 0.3, "unit": "ratio"},
    )


def _securities_fvtpl_abstention_finding(
    report_set: CompanyReportSet,
    *,
    signal_id: str,
    finding_summary: str,
) -> ToolFinding:
    return ToolFinding(
        tool_result_id=f"TOOL_{report_set.target.report_id}_SEC_FVTPL_CASH_001",
        report_id=report_set.target.report_id,
        tool_name="ctck_earnings_cash_bridge_tool",
        risk_category="earnings_cashflow_mismatch",
        signal_id=signal_id,
        flag=False,
        finding_summary=finding_summary,
        evidence_refs=[],
        metric={"metric_name": "fvtpl_unrealized_gain_to_abs_profit", "value": None, "unit": "ratio"},
        threshold={"threshold_type": "greater_than", "value": 0.3, "unit": "ratio"},
    )


def _build_insurance_premium_receivables_gwp_finding(report_set: CompanyReportSet) -> ToolFinding:
    if report_set.target.metadata.get("report_profile") != "insurance":
        return _insurance_premium_receivables_abstention_finding(
            report_set,
            signal_id="insurance_premium_receivables_wrong_profile",
            finding_summary="Insurance premium receivables check is not applicable to this report profile.",
        )
    try:
        current_receivables = _require_account_cell(report_set.target, "premium_receivables")
        comparison_receivables = _require_account_cell(report_set.prior_year, "premium_receivables")
        current_gwp = _require_account_cell(report_set.target, "gross_written_premium")
        comparison_gwp = _require_account_cell(report_set.prior_year, "gross_written_premium")
    except ValueError:
        return _insurance_premium_receivables_abstention_finding(
            report_set,
            signal_id="insurance_premium_receivables_missing_evidence",
            finding_summary="Insurance premium receivables check skipped because required evidence is missing.",
        )
    cells = [current_receivables, comparison_receivables, current_gwp, comparison_gwp]
    reports = [report_set.target, report_set.prior_year, report_set.target, report_set.prior_year]
    if not all(_cell_period_matches_report(report, cell) for report, cell in zip(reports, cells, strict=True)):
        return ToolFinding(
            tool_result_id=f"TOOL_{report_set.target.report_id}_INS_PREM_REC_001",
            report_id=report_set.target.report_id,
            tool_name="insurance_premium_receivable_reserve_coherence_tool",
            risk_category="receivables_credit_quality_risk",
            signal_id="insurance_premium_receivables_wrong_period",
            flag=False,
            finding_summary="Insurance premium receivables check skipped because required evidence is outside the selected periods.",
            evidence_refs=[
                _cell_evidence_ref(report_set.target.report_id, current_receivables["cell_id"]),
                _cell_evidence_ref(report_set.prior_year.report_id, comparison_receivables["cell_id"]),
                _cell_evidence_ref(report_set.target.report_id, current_gwp["cell_id"]),
                _cell_evidence_ref(report_set.prior_year.report_id, comparison_gwp["cell_id"]),
            ],
            metric={
                "metric_name": "premium_receivables_growth_minus_gwp_growth_pct_points",
                "value": None,
                "unit": "percentage_points",
            },
            threshold={"threshold_type": "greater_than", "value": 20.0, "unit": "percentage_points"},
        )
    try:
        receivables_growth = _percentage_growth(current_receivables["value"], comparison_receivables["value"])
        gwp_growth = _percentage_growth(current_gwp["value"], comparison_gwp["value"])
    except ValueError:
        return ToolFinding(
            tool_result_id=f"TOOL_{report_set.target.report_id}_INS_PREM_REC_001",
            report_id=report_set.target.report_id,
            tool_name="insurance_premium_receivable_reserve_coherence_tool",
            risk_category="receivables_credit_quality_risk",
            signal_id="insurance_premium_receivables_zero_comparison",
            flag=False,
            finding_summary="Insurance premium receivables check skipped because a prior-year comparison value is zero.",
            evidence_refs=[
                _cell_evidence_ref(report_set.target.report_id, current_receivables["cell_id"]),
                _cell_evidence_ref(report_set.prior_year.report_id, comparison_receivables["cell_id"]),
                _cell_evidence_ref(report_set.target.report_id, current_gwp["cell_id"]),
                _cell_evidence_ref(report_set.prior_year.report_id, comparison_gwp["cell_id"]),
            ],
            metric={
                "metric_name": "premium_receivables_growth_minus_gwp_growth_pct_points",
                "value": None,
                "unit": "percentage_points",
            },
            threshold={"threshold_type": "greater_than", "value": 20.0, "unit": "percentage_points"},
        )
    growth_gap = round(receivables_growth - gwp_growth, 2)
    flag = growth_gap > 20.0
    return ToolFinding(
        tool_result_id=f"TOOL_{report_set.target.report_id}_INS_PREM_REC_001",
        report_id=report_set.target.report_id,
        tool_name="insurance_premium_receivable_reserve_coherence_tool",
        risk_category="receivables_credit_quality_risk",
        signal_id="insurance_premium_receivables_outpace_gwp" if flag else "insurance_premium_receivables_not_outpacing_gwp",
        flag=flag,
        finding_summary=(
            f"Premium receivables growth exceeded gross written premium growth by {growth_gap:.2f} percentage points."
        ),
        evidence_refs=[
            _cell_evidence_ref(report_set.target.report_id, current_receivables["cell_id"]),
            _cell_evidence_ref(report_set.prior_year.report_id, comparison_receivables["cell_id"]),
            _cell_evidence_ref(report_set.target.report_id, current_gwp["cell_id"]),
            _cell_evidence_ref(report_set.prior_year.report_id, comparison_gwp["cell_id"]),
        ],
        metric={
            "metric_name": "premium_receivables_growth_minus_gwp_growth_pct_points",
            "value": growth_gap,
            "unit": "percentage_points",
        },
        threshold={"threshold_type": "greater_than", "value": 20.0, "unit": "percentage_points"},
    )


def _insurance_premium_receivables_abstention_finding(
    report_set: CompanyReportSet,
    *,
    signal_id: str,
    finding_summary: str,
) -> ToolFinding:
    return ToolFinding(
        tool_result_id=f"TOOL_{report_set.target.report_id}_INS_PREM_REC_001",
        report_id=report_set.target.report_id,
        tool_name="insurance_premium_receivable_reserve_coherence_tool",
        risk_category="receivables_credit_quality_risk",
        signal_id=signal_id,
        flag=False,
        finding_summary=finding_summary,
        evidence_refs=[],
        metric={
            "metric_name": "premium_receivables_growth_minus_gwp_growth_pct_points",
            "value": None,
            "unit": "percentage_points",
        },
        threshold={"threshold_type": "greater_than", "value": 20.0, "unit": "percentage_points"},
    )


def _build_credit_institution_loan_quality_finding(report_set: CompanyReportSet) -> ToolFinding:
    if report_set.target.metadata.get("report_profile") != "credit_institution":
        return _credit_institution_loan_quality_abstention_finding(
            report_set,
            signal_id="credit_institution_loan_quality_wrong_profile",
            finding_summary="Credit-institution loan quality check is not applicable to this report profile.",
        )
    try:
        current_loans = _require_account_cell(report_set.target, "customer_loans")
        comparison_loans = _require_account_cell(report_set.prior_year, "customer_loans")
        current_npl = _require_account_cell(report_set.target, "nonperforming_loans")
        comparison_npl = _require_account_cell(report_set.prior_year, "nonperforming_loans")
        current_provisions = _require_account_cell(report_set.target, "loan_loss_provisions")
        comparison_provisions = _require_account_cell(report_set.prior_year, "loan_loss_provisions")
    except ValueError:
        return _credit_institution_loan_quality_abstention_finding(
            report_set,
            signal_id="credit_institution_loan_quality_missing_evidence",
            finding_summary=(
                "Credit-institution loan quality check skipped because loan exposure, non-performing loan, "
                "or loan-loss provision evidence is missing."
            ),
        )
    cells = [
        current_loans,
        comparison_loans,
        current_npl,
        comparison_npl,
        current_provisions,
        comparison_provisions,
    ]
    reports = [
        report_set.target,
        report_set.prior_year,
        report_set.target,
        report_set.prior_year,
        report_set.target,
        report_set.prior_year,
    ]
    evidence_refs = [
        _cell_evidence_ref(report.report_id, cell["cell_id"])
        for report, cell in zip(reports, cells, strict=True)
    ]
    if not all(_cell_period_matches_report(report, cell) for report, cell in zip(reports, cells, strict=True)):
        return ToolFinding(
            tool_result_id=f"TOOL_{report_set.target.report_id}_BANK_LOAN_QUALITY_001",
            report_id=report_set.target.report_id,
            tool_name="credit_institution_loan_quality_tool",
            risk_category="receivables_credit_quality_risk",
            signal_id="credit_institution_loan_quality_wrong_period",
            flag=False,
            finding_summary="Credit-institution loan quality check skipped because required evidence is outside the selected periods.",
            evidence_refs=evidence_refs,
            metric={
                "metric_name": "npl_ratio_change_pct_points",
                "value": None,
                "unit": "percentage_points",
                "provision_coverage_change_pct_points": None,
            },
            threshold={"threshold_type": "greater_than", "value": 1.0, "unit": "percentage_points"},
        )
    if (
        current_loans["value"] == 0
        or comparison_loans["value"] == 0
        or current_npl["value"] == 0
        or comparison_npl["value"] == 0
    ):
        return ToolFinding(
            tool_result_id=f"TOOL_{report_set.target.report_id}_BANK_LOAN_QUALITY_001",
            report_id=report_set.target.report_id,
            tool_name="credit_institution_loan_quality_tool",
            risk_category="receivables_credit_quality_risk",
            signal_id="credit_institution_loan_quality_zero_denominator",
            flag=False,
            finding_summary="Credit-institution loan quality check skipped because a loan or NPL denominator is zero.",
            evidence_refs=evidence_refs,
            metric={
                "metric_name": "npl_ratio_change_pct_points",
                "value": None,
                "unit": "percentage_points",
                "provision_coverage_change_pct_points": None,
            },
            threshold={"threshold_type": "greater_than", "value": 1.0, "unit": "percentage_points"},
        )
    current_npl_ratio = current_npl["value"] / current_loans["value"] * 100
    comparison_npl_ratio = comparison_npl["value"] / comparison_loans["value"] * 100
    npl_ratio_change = round(current_npl_ratio - comparison_npl_ratio, 2)
    current_coverage = current_provisions["value"] / current_npl["value"] * 100
    comparison_coverage = comparison_provisions["value"] / comparison_npl["value"] * 100
    coverage_change = round(current_coverage - comparison_coverage, 2)
    flag = npl_ratio_change > 1.0 and coverage_change <= 0
    return ToolFinding(
        tool_result_id=f"TOOL_{report_set.target.report_id}_BANK_LOAN_QUALITY_001",
        report_id=report_set.target.report_id,
        tool_name="credit_institution_loan_quality_tool",
        risk_category="receivables_credit_quality_risk",
        signal_id=(
            "credit_institution_npl_ratio_rising_without_coverage_improvement"
            if flag
            else "credit_institution_loan_quality_not_flagged"
        ),
        flag=flag,
        finding_summary=(
            f"NPL ratio changed by {npl_ratio_change:.2f} percentage points while provision coverage changed by "
            f"{coverage_change:.2f} percentage points."
        ),
        evidence_refs=evidence_refs,
        metric={
            "metric_name": "npl_ratio_change_pct_points",
            "value": npl_ratio_change,
            "unit": "percentage_points",
            "provision_coverage_change_pct_points": coverage_change,
        },
        threshold={"threshold_type": "greater_than", "value": 1.0, "unit": "percentage_points"},
    )


def _credit_institution_loan_quality_abstention_finding(
    report_set: CompanyReportSet,
    *,
    signal_id: str,
    finding_summary: str,
) -> ToolFinding:
    return ToolFinding(
        tool_result_id=f"TOOL_{report_set.target.report_id}_BANK_LOAN_QUALITY_001",
        report_id=report_set.target.report_id,
        tool_name="credit_institution_loan_quality_tool",
        risk_category="receivables_credit_quality_risk",
        signal_id=signal_id,
        flag=False,
        finding_summary=finding_summary,
        evidence_refs=[],
        metric={
            "metric_name": "npl_ratio_change_pct_points",
            "value": None,
            "unit": "percentage_points",
            "provision_coverage_change_pct_points": None,
        },
        threshold={"threshold_type": "greater_than", "value": 1.0, "unit": "percentage_points"},
    )


def _build_related_party_disclosure_finding(report_set: CompanyReportSet) -> ToolFinding:
    notes = [
        note
        for note in report_set.target.raw.get("notes", [])
        if note.get("note_type") == "related_party_note"
        and _is_source_bound_text_evidence(note, local_id_key="note_id")
    ]
    if not notes:
        return ToolFinding(
            tool_result_id=f"TOOL_{report_set.target.report_id}_RPT_DISC_001",
            report_id=report_set.target.report_id,
            tool_name="related_party_exposure_tool",
            risk_category="related_party_disclosure_risk",
            signal_id="related_party_disclosure_missing_or_unsupported",
            flag=False,
            finding_summary="Related-party disclosure check skipped because source-bound related-party evidence is missing or unsupported.",
            evidence_refs=[],
            metric={"metric_name": "source_bound_related_party_note_count", "value": 0, "unit": "count"},
            threshold={"threshold_type": "greater_than", "value": 0, "unit": "count"},
        )
    note = notes[0]
    return ToolFinding(
        tool_result_id=f"TOOL_{report_set.target.report_id}_RPT_DISC_001",
        report_id=report_set.target.report_id,
        tool_name="related_party_exposure_tool",
        risk_category="related_party_disclosure_risk",
        signal_id="related_party_disclosure_present",
        flag=True,
        finding_summary="Source-bound related-party disclosure evidence is available for detector review.",
        evidence_refs=[
            _text_evidence_ref(
                report_set.target.report_id,
                note["note_id"],
                evidence_ref_type="related_party_note_span",
            )
        ],
        metric={"metric_name": "source_bound_related_party_note_count", "value": len(notes), "unit": "count"},
        threshold={"threshold_type": "greater_than", "value": 0, "unit": "count"},
    )


def _build_accounting_policy_disclosure_finding(report_set: CompanyReportSet) -> ToolFinding:
    notes = [
        note
        for note in report_set.target.raw.get("notes", [])
        if note.get("note_type") in {"accounting_policy_change", "generic_accounting_policy"}
        and _is_source_bound_text_evidence(note, local_id_key="note_id")
    ]
    if not notes:
        return ToolFinding(
            tool_result_id=f"TOOL_{report_set.target.report_id}_POLICY_DISC_001",
            report_id=report_set.target.report_id,
            tool_name="accounting_policy_change_tool",
            risk_category="disclosure_inconsistency_or_obfuscation",
            signal_id="accounting_policy_disclosure_missing_or_unsupported",
            flag=False,
            finding_summary="Accounting-policy disclosure check skipped because source-bound policy evidence is missing or unsupported.",
            evidence_refs=[],
            metric={"metric_name": "source_bound_accounting_policy_note_count", "value": 0, "unit": "count"},
            threshold={"threshold_type": "greater_than", "value": 0, "unit": "count"},
        )
    note = notes[0]
    return ToolFinding(
        tool_result_id=f"TOOL_{report_set.target.report_id}_POLICY_DISC_001",
        report_id=report_set.target.report_id,
        tool_name="accounting_policy_change_tool",
        risk_category="disclosure_inconsistency_or_obfuscation",
        signal_id="accounting_policy_disclosure_present",
        flag=True,
        finding_summary="Source-bound accounting-policy disclosure evidence is available for detector review.",
        evidence_refs=[
            _text_evidence_ref(
                report_set.target.report_id,
                note["note_id"],
                evidence_ref_type="accounting_policy_note_span",
            )
        ],
        metric={"metric_name": "source_bound_accounting_policy_note_count", "value": len(notes), "unit": "count"},
        threshold={"threshold_type": "greater_than", "value": 0, "unit": "count"},
    )


def _build_variance_explanation_disclosure_finding(report_set: CompanyReportSet) -> ToolFinding:
    explanations = [
        explanation
        for explanation in report_set.target.raw.get("variance_explanations", [])
        if _is_source_bound_text_evidence(explanation, local_id_key="span_id")
    ]
    if not explanations:
        return ToolFinding(
            tool_result_id=f"TOOL_{report_set.target.report_id}_VAR_DISC_001",
            report_id=report_set.target.report_id,
            tool_name="variance_explanation_quality_tool",
            risk_category="disclosure_inconsistency_or_obfuscation",
            signal_id="variance_explanation_missing_or_unsupported",
            flag=False,
            finding_summary="Variance explanation check skipped because source-bound explanation evidence is missing or unsupported.",
            evidence_refs=[],
            metric={"metric_name": "source_bound_variance_explanation_count", "value": 0, "unit": "count"},
            threshold={"threshold_type": "greater_than", "value": 0, "unit": "count"},
        )
    explanation = explanations[0]
    return ToolFinding(
        tool_result_id=f"TOOL_{report_set.target.report_id}_VAR_DISC_001",
        report_id=report_set.target.report_id,
        tool_name="variance_explanation_quality_tool",
        risk_category="disclosure_inconsistency_or_obfuscation",
        signal_id="variance_explanation_present",
        flag=True,
        finding_summary="Source-bound variance explanation evidence is available for detector review.",
        evidence_refs=[
            _text_evidence_ref(
                report_set.target.report_id,
                explanation["span_id"],
                evidence_ref_type="variance_explanation_span",
            )
        ],
        metric={"metric_name": "source_bound_variance_explanation_count", "value": len(explanations), "unit": "count"},
        threshold={"threshold_type": "greater_than", "value": 0, "unit": "count"},
    )


def _cashflow_abstention_finding(
    report_set: CompanyReportSet,
    *,
    signal_id: str,
    finding_summary: str,
) -> ToolFinding:
    return ToolFinding(
        tool_result_id=f"TOOL_{report_set.target.report_id}_CASH_001",
        report_id=report_set.target.report_id,
        tool_name="standard_corporate_earnings_cashflow_mismatch_tool",
        risk_category="earnings_cashflow_mismatch",
        signal_id=signal_id,
        flag=False,
        finding_summary=finding_summary,
        evidence_refs=[],
        metric={"metric_name": "operating_cash_flow_to_abs_profit", "value": None, "unit": "ratio"},
        threshold={"threshold_type": "less_than", "value": 0.7, "unit": "ratio"},
    )


def _packet_tool_finding(finding: ToolFinding) -> dict[str, Any]:
    return {
        "tool_result_id": finding.tool_result_id,
        "tool_name": finding.tool_name,
        "risk_category": finding.risk_category,
        "signal_id": finding.signal_id,
        "flag": finding.flag,
        "finding_summary": finding.finding_summary,
        "evidence_refs": finding.evidence_refs,
        "metric": finding.metric,
        "threshold": finding.threshold,
    }


def _packet_rules(candidate: CandidateRisk) -> list[dict[str, Any]]:
    if candidate.supporting_signal_ids == ["variance_explanation_present"]:
        return [
            {
                "rule_id": "RULE_VARIANCE_EXPLANATION_DISCLOSURE",
                "related_signal_ids": candidate.supporting_signal_ids,
                "risk_category": candidate.risk_category,
                "description": (
                    "Create a disclosure consistency review candidate only when source-bound variance explanation "
                    "evidence is present in ReportMemory."
                ),
            }
        ]
    if candidate.supporting_signal_ids == ["accounting_policy_disclosure_present"]:
        return [
            {
                "rule_id": "RULE_ACCOUNTING_POLICY_DISCLOSURE",
                "related_signal_ids": candidate.supporting_signal_ids,
                "risk_category": candidate.risk_category,
                "description": (
                    "Create a disclosure consistency review candidate only when source-bound accounting-policy "
                    "note evidence is present in ReportMemory."
                ),
            }
        ]
    if candidate.supporting_signal_ids == ["related_party_disclosure_present"]:
        return [
            {
                "rule_id": "RULE_RELATED_PARTY_DISCLOSURE",
                "related_signal_ids": candidate.supporting_signal_ids,
                "risk_category": candidate.risk_category,
                "description": (
                    "Create a related-party disclosure review candidate only when source-bound related-party "
                    "note evidence is present in ReportMemory."
                ),
            }
        ]
    if candidate.supporting_signal_ids == ["insurance_premium_receivables_outpace_gwp"]:
        return [
            {
                "rule_id": "RULE_INSURANCE_PREMIUM_RECEIVABLES_GWP",
                "related_signal_ids": candidate.supporting_signal_ids,
                "risk_category": candidate.risk_category,
                "description": (
                    "For insurance reports, review is warranted when premium receivables materially outpace "
                    "gross written premium in the same-quarter prior-year comparison."
                ),
            }
        ]
    if candidate.supporting_signal_ids == ["credit_institution_npl_ratio_rising_without_coverage_improvement"]:
        return [
            {
                "rule_id": "RULE_CREDIT_INSTITUTION_LOAN_QUALITY",
                "related_signal_ids": candidate.supporting_signal_ids,
                "risk_category": candidate.risk_category,
                "description": (
                    "For credit-institution reports, review is warranted when source-bound loan exposure, "
                    "non-performing-loan, and provision evidence show a rising NPL ratio without improved coverage."
                ),
            }
        ]
    if candidate.supporting_signal_ids == ["securities_fvtpl_profit_bridge_high"]:
        return [
            {
                "rule_id": "RULE_SECURITIES_FVTPL_CASHFLOW_BRIDGE",
                "related_signal_ids": candidate.supporting_signal_ids,
                "risk_category": candidate.risk_category,
                "description": (
                    "For securities reports, review is warranted when unrealized FVTPL gains are material "
                    "relative to profit and packet-visible cash-flow evidence is weak."
                ),
            }
        ]
    if candidate.risk_category == "earnings_cashflow_mismatch":
        return [
            {
                "rule_id": "RULE_EARNINGS_CASHFLOW_MISMATCH",
                "related_signal_ids": candidate.supporting_signal_ids,
                "risk_category": candidate.risk_category,
                "description": (
                    "Review is warranted when operating cash flow is below 0.70 times absolute profit "
                    "or profit is positive while operating cash flow is negative."
                ),
            }
        ]
    return [
        {
            "rule_id": "RULE_REVENUE_RECEIVABLES_GROWTH",
            "related_signal_ids": candidate.supporting_signal_ids,
            "risk_category": candidate.risk_category,
            "description": "Revenue growth and faster receivables growth may support a revenue quality risk signal.",
        }
    ]


def _packet_table_rows(report_set: CompanyReportSet, candidate: CandidateRisk) -> list[dict[str, Any]]:
    cell_refs = [ref for ref in candidate.evidence_refs if ref["evidence_ref_type"] == "table_cell"]
    rows = []
    for ref in cell_refs:
        report = _report_for_ref(report_set, ref)
        cell = _cell_by_id(report, ref["local_evidence_id"])
        row = _row_by_cell_id(report, ref["local_evidence_id"])
        rows.append(
            {
                "row_id": row["row_id"],
                "report_id": report.report_id,
                "standard_account": row.get("standard_account"),
                "label": row.get("label"),
                "values": {"current": {"value": cell["value"], "cell_id": cell["cell_id"]}},
            }
        )
    return rows


def _packet_notes(report_set: CompanyReportSet, candidate: CandidateRisk) -> list[dict[str, Any]]:
    note_refs = [
        ref
        for ref in candidate.evidence_refs
        if ref["evidence_ref_type"] in {"note", "note_span", "accounting_policy_note_span", "related_party_note_span"}
    ]
    notes = []
    for ref in note_refs:
        report = _report_for_ref(report_set, ref)
        note = _note_by_id(report, ref["local_evidence_id"])
        notes.append(
            {
                "note_id": note["note_id"],
                "report_id": report.report_id,
                "note_type": note.get("note_type"),
                "text": note["text"],
                "source_document_id": note["source_document_id"],
                "evidence_ref_type": ref["evidence_ref_type"],
            }
        )
    return notes


def _packet_variance_explanations(report_set: CompanyReportSet, candidate: CandidateRisk) -> list[dict[str, Any]]:
    explanation_refs = [
        ref
        for ref in candidate.evidence_refs
        if ref["evidence_ref_type"] == "variance_explanation_span"
    ]
    explanations = []
    for ref in explanation_refs:
        report = _report_for_ref(report_set, ref)
        explanation = _variance_explanation_by_id(report, ref["local_evidence_id"])
        explanations.append(
            {
                "span_id": explanation["span_id"],
                "report_id": report.report_id,
                "text": explanation["text"],
                "source_document_id": explanation["source_document_id"],
                "related_metric": explanation.get("related_metric"),
                "evidence_ref_type": ref["evidence_ref_type"],
            }
        )
    return explanations


def _require_account_cell(report_memory: ReportMemory, account: str) -> dict[str, Any]:
    for table in report_memory.raw.get("tables", []):
        for row in table.get("rows", []):
            if row.get("standard_account") == account or _row_matches_account_code(row, account, table):
                cells = row.get("cells", [])
                if cells:
                    return cells[0]
    raise ValueError(f"Missing required account evidence for {account}")


def _report_for_ref(report_set: CompanyReportSet, ref: dict[str, str]) -> ReportMemory:
    if ref["report_id"] == report_set.target.report_id:
        return report_set.target
    if ref["report_id"] == report_set.prior_year.report_id:
        return report_set.prior_year
    raise ValueError(f"Evidence ref report_id is outside the report set: {ref['report_id']}")


def _row_matches_account_code(row: dict[str, Any], account: str, table: dict[str, Any]) -> bool:
    if account == "revenue":
        return row.get("account_code") == "01"
    if account == "trade_receivables":
        return row.get("account_code") == "131"
    if account == "profit_after_tax":
        return row.get("account_code") == "60"
    if account == "operating_cash_flow":
        return row.get("account_code") == "20" and table.get("table_type") == "cash_flow_statement"
    return False


def _row_by_cell_id(report_memory: ReportMemory, cell_id: str) -> dict[str, Any]:
    for table in report_memory.raw.get("tables", []):
        for row in table.get("rows", []):
            if any(cell.get("cell_id") == cell_id for cell in row.get("cells", [])):
                return row
    raise ValueError(f"Missing row for cell evidence {cell_id}")


def _cell_by_id(report_memory: ReportMemory, cell_id: str) -> dict[str, Any]:
    for table in report_memory.raw.get("tables", []):
        for row in table.get("rows", []):
            for cell in row.get("cells", []):
                if cell.get("cell_id") == cell_id:
                    return cell
    raise ValueError(f"Missing cell evidence {cell_id}")


def _note_by_id(report_memory: ReportMemory, note_id: str) -> dict[str, Any]:
    for note in report_memory.raw.get("notes", []):
        if note.get("note_id") == note_id:
            return note
    raise ValueError(f"Missing note evidence {note_id}")


def _variance_explanation_by_id(report_memory: ReportMemory, span_id: str) -> dict[str, Any]:
    for explanation in report_memory.raw.get("variance_explanations", []):
        if explanation.get("span_id") == span_id:
            return explanation
    raise ValueError(f"Missing variance explanation evidence {span_id}")


def _cell_period_matches_report(report_memory: ReportMemory, cell: dict[str, Any]) -> bool:
    return cell.get("period") == report_memory.metadata.get("period")


def _percentage_growth(current_value: float, comparison_value: float) -> float:
    if comparison_value == 0:
        raise ValueError("Cannot compute growth against a zero comparison value")
    return round((current_value - comparison_value) / abs(comparison_value) * 100, 2)


def _cell_evidence_ref(report_id: str, cell_id: str) -> dict[str, str]:
    return {
        "evidence_ref_type": "table_cell",
        "ref_id": f"{report_id}:{cell_id}",
        "report_id": report_id,
        "local_evidence_id": cell_id,
        "role": "supporting",
    }


def _text_evidence_ref(report_id: str, local_evidence_id: str, *, evidence_ref_type: str) -> dict[str, str]:
    return {
        "evidence_ref_type": evidence_ref_type,
        "ref_id": f"{report_id}:{local_evidence_id}",
        "report_id": report_id,
        "local_evidence_id": local_evidence_id,
        "role": "supporting",
    }


def _is_source_bound_text_evidence(evidence: dict[str, Any], *, local_id_key: str) -> bool:
    provenance = evidence.get("evidence_provenance")
    return (
        bool(evidence.get(local_id_key))
        and bool(str(evidence.get("text", "")).strip())
        and bool(evidence.get("source_document_id"))
        and isinstance(provenance, dict)
        and provenance.get("source_document_id") == evidence.get("source_document_id")
    )


def _dedupe_evidence_refs(evidence_refs: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    deduped = []
    for ref in evidence_refs:
        key = (ref["evidence_ref_type"], ref["ref_id"], ref["role"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    return deduped


def _to_dict(value: Any) -> dict[str, Any]:
    return asdict(value)
