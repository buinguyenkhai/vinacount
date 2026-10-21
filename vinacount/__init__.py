"""Vinacount: evidence-bounded language-model judgment for accounting review.

Only one step uses a language model: the detector judges whether a fixed packet of filing evidence supports a
deterministically constructed risk candidate. Every other step is deterministic.

    sources        select the authoritative filing pair
    report_memory  structured filings and the target/prior-year pair
    checks         ToolFindings, CandidateRisks, and DetectorPackets
    detector       the language-model detector and the Rules-Only baseline
    guard          validate each response against its packet; never changes a label
    placement      map accepted labels to report sections
    pipeline       review(): all of the above for one filing pair
"""

from vinacount.detector import make_detector
from vinacount.pipeline import load_filing_pair, review

__all__ = ["load_filing_pair", "make_detector", "review"]
