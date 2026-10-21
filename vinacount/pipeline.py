"""Review one filing pair: checks -> candidates -> packets -> detector -> guard -> report placement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vinacount.checks import run_checks_to_packets
from vinacount.detector import packet_messages
from vinacount.guard import guarded_assessment
from vinacount.placement import place_assessments
from vinacount.report_memory import CompanyReportSet, validate_company_report_set, validate_report_memory


def load_filing_pair(path: str | Path) -> CompanyReportSet:
    """Read {"case_id", "target", "prior_year"} where target and prior_year are ReportMemory objects."""
    pair = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_company_report_set(pair["case_id"], validate_report_memory(pair["target"]),
                                       validate_report_memory(pair["prior_year"]))


def review(report_set: CompanyReportSet, detector: Any) -> dict[str, Any]:
    """Run the full workflow. Only the detector is a language model; every other step is deterministic.

    Returns the tool findings, candidates, packets, raw detector responses, the guard outcome for each packet,
    and the placed report. A response the guard rejects is reported but never placed.
    """
    trace = run_checks_to_packets(report_set)
    prompts = [packet_messages(p) for p in trace["detector_packets"]]
    responses = detector.generate(prompts) if prompts else []
    accepted, outcomes = [], []
    for messages, text in zip(prompts, responses):
        packet = json.loads(messages[1]["content"])
        try:
            assessment, actions = guarded_assessment(text, packet)
        except Exception as error:  # any contract violation the guard cannot repair
            outcomes.append({"packet_id": packet["packet_id"], "status": "rejected", "error": str(error)})
            continue
        accepted.append(assessment)
        outcomes.append({"packet_id": packet["packet_id"], "status": "accepted", "guard_actions": actions,
                         "assessment": assessment})
    return {**trace, "responses": responses, "guard": outcomes,
            "report": place_assessments(trace["candidate_risks"], accepted)}
