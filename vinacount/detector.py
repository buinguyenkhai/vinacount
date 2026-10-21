"""The detector D_theta: prompts, the three inference backends, and the Rules-Only baseline.

The detector receives one DetectorPacket and returns one DetectorAssessment as JSON text. Every backend returns raw
text; acceptance is decided afterwards by the contract (Strict) or by the guard (Guarded), see vinacount/guard.py.

Every detector has generate(prompts) -> list[str], where each prompt is [system message, user message].

  TransformersDetector  local inference with transformers + PEFT (the released LoRA adapter, or the base model)
  VllmDetector          a vLLM server that serves the adapter
  HostedDetector        an OpenAI-compatible chat API (used for the DeepSeek-V4 baselines in the paper)
  RulesDetector         labels from the deterministic findings alone, without a language model
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from vinacount.contract import (
    DetectorPacket,
    detector_assessment_json_schema,
    enrich_detector_packet_evidence_roles,
    validate_detector_packet,
)

# System prompts exactly as stored in the training corpus. The released adapter uses "v2_evidence_bundle";
# "v1" was used for the configuration comparison in the paper.
SYSTEM_PROMPT_V1 = (
    "You are a specialized accounting irregularity risk-signal detector for Vietnamese financial reports.\n\n"
    "Your task is to assess whether the provided DetectorPacket supports the given CandidateRisk category.\n\n"
    "Return only a valid DetectorAssessment JSON object.\n\n"
    "Use only evidence provided in the packet.\n"
    "Reference only provided evidence IDs.\n"
    "Do not use outside knowledge.\n"
    "Do not use hidden injection metadata.\n"
    "Do not claim fraud, manipulation, intent, concealment, or legal misstatement.\n"
    "Use risk-signal language only.\n"
    "Keep rationale_short to 1–3 sentences."
)
SYSTEM_PROMPT_V2_EVIDENCE_BUNDLE = (
    f"{SYSTEM_PROMPT_V1}\n\n"
    "Calibrate support from the complete visible evidence bundle, not from trigger magnitude alone.\n"
    "A tool finding's strength describes the magnitude of that trigger; it does not by itself establish "
    "that the candidate is fully supported.\n"
    "Use supported when the packet contains direct, coherent, and sufficient evidence, normally through "
    "multiple aligned signals, one signal plus independent corroboration, or a category-specific decisive "
    "item identified by a visible rule.\n"
    "Use weakly_supported when a meaningful signal is present but isolated, single-source, incomplete, "
    "mixed, or missing an important corroborating component. A strong isolated signal may therefore be "
    "weakly_supported.\n"
    "Use not_supported when visible evidence contradicts or fails to support the candidate. Use "
    "insufficient_evidence when the packet lacks the evidence needed to assess it."
)
SYSTEM_PROMPTS = {"v1": SYSTEM_PROMPT_V1, "v2_evidence_bundle": SYSTEM_PROMPT_V2_EVIDENCE_BUNDLE}


def canonical_packet(packet: DetectorPacket | dict[str, Any]) -> dict[str, Any]:
    """Keep the DetectorPacket fields, add evidence roles for earnings-cash-flow packets, and validate the caps."""
    data = packet if isinstance(packet, dict) else {k: getattr(packet, k) for k in DetectorPacket.__dataclass_fields__}
    canonical = {k: data[k] for k in DetectorPacket.__dataclass_fields__ if data.get(k) is not None}
    canonical = enrich_detector_packet_evidence_roles(canonical)
    validate_detector_packet(canonical)
    return canonical


def packet_json(packet: dict[str, Any]) -> str:
    """Serialize a packet the way the training corpus does: sorted keys, compact separators, UTF-8 text."""
    return json.dumps(packet, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def packet_messages(packet: DetectorPacket | dict[str, Any], prompt_version: str = "v2_evidence_bundle") -> list[dict[str, str]]:
    """The system and user messages for a new packet (corpus records already carry these two messages)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[prompt_version]},
        {"role": "user", "content": packet_json(canonical_packet(packet))},
    ]


def render_qwen_no_think(messages: list[dict[str, str]]) -> str:
    """Qwen chat prompt with thinking disabled, for completion endpoints such as vLLM /v1/completions."""
    system, user = (m["content"].strip() for m in messages)
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, str]], **kwargs: Any) -> Any:
    # The Qwen3.5 processor template expects content blocks; plain strings are wrapped without changing the text.
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError as error:
        if "string indices must be integers" not in str(error):
            raise
        blocks = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in messages]
        return tokenizer.apply_chat_template(blocks, **kwargs)


class TransformersDetector:
    """Greedy decoding with thinking disabled, as in the offline evaluator used for every synthetic result."""

    def __init__(self, base_model: str, adapter: str | None, *, revision: str | None = None, device: str = "auto",
                 max_new_tokens: int = 1024, batch_size: int = 1) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch, self.device = torch, torch.device(device)
        self.max_new_tokens, self.batch_size = max_new_tokens, batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(adapter or base_model, revision=None if adapter else revision)
        model = AutoModelForCausalLM.from_pretrained(base_model, revision=revision, dtype=torch.bfloat16)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
        self.model = model.to(self.device).eval()

    def generate(self, prompts: list[list[dict[str, str]]]) -> list[str]:
        out: list[str] = []
        for start in range(0, len(prompts), self.batch_size):
            out.extend(self._generate_batch(prompts[start : start + self.batch_size]))
        return out

    def _generate_batch(self, batch: list[list[dict[str, str]]]) -> list[str]:
        tok, torch = self.tokenizer, self.torch
        texts = [_apply_chat_template(tok, m, tokenize=False, add_generation_prompt=True, enable_thinking=False) for m in batch]
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        inputs = {k: v.to(self.device) for k, v in tok(texts, return_tensors="pt", padding=True).items()}
        width = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        return [tok.decode(row[width:], skip_special_tokens=True).strip() for row in ids]


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float, attempts: int = 3) -> dict[str, Any]:
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json", **headers}, method="POST")
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as error:
            if error.code not in {408, 409, 425, 429, 500, 502, 503, 504} or attempt == attempts - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == attempts - 1:
                raise
        time.sleep(2**attempt)
    raise RuntimeError("unreachable")


class VllmDetector:
    """A vLLM server started with --enable-lora; the adapter is addressed by its served name."""

    def __init__(self, base_url: str, model: str, *, max_tokens: int = 1024, timeout: float = 180.0) -> None:
        self.base_url, self.model, self.max_tokens, self.timeout = base_url.rstrip("/"), model, max_tokens, timeout

    def generate(self, prompts: list[list[dict[str, str]]]) -> list[str]:
        out = []
        for messages in prompts:
            payload = {"model": self.model, "prompt": render_qwen_no_think(messages), "temperature": 0.0,
                       "max_tokens": self.max_tokens}
            out.append(_post_json(f"{self.base_url}/completions", payload, {}, self.timeout)["choices"][0]["text"])
        return out


# Hosted baselines: the same packet plus an explicit output contract, temperature 0, JSON mode, thinking disabled.
_HOSTED_BASE_PROMPT = (
    "You are a specialized accounting irregularity risk-signal detector for Vietnamese financial reports. "
    "Return a single JSON object with these top-level keys: assessment_id, packet_id, candidate_id, "
    "report_id, risk_category, support_level, confidence, severity, validated_signals, "
    "cited_evidence_refs, rationale_short. "
    "Do not wrap the object in assessment, type, data, or any other envelope. "
    "Do not use decision, decision_rationale, signal_ids, or evidence_ids_used fields. "
    "Use only evidence provided in the DetectorPacket. "
    "Use exact ref_id strings from DetectorPacket.tool_findings[].evidence_refs, DetectorPacket.rules[].rule_id, "
    "DetectorPacket.tool_findings[].tool_result_id, and canonical packet evidence IDs formed as "
    "report_id:cell_id, report_id:row_id, report_id:note_id, or report_id:span_id from visible packet rows, "
    "notes, and variance explanations. "
    "Do not use local_evidence_id, row_id, note_id, span_id, or cell_id by itself. "
    "Use only these evidence_ref_type values: table_cell, table_row, note, note_span, "
    "variance_explanation_span, accounting_policy_note_span, related_party_note_span, tool_result, rule. "
    "Use only these evidence roles: supporting, contradicting, refuting, context, missing_required_context. "
    "Use only these validated_signals status values: validated, partially_validated, rejected, not_assessable. "
    "Confidence must be a number between 0 and 1, never a string label. "
    "Do not use outside knowledge. "
    "Do not use hidden metadata. "
    "Do not claim fraud, manipulation, intent, concealment, or legal misstatement. "
    "Use risk-signal language only. "
    "Keep rationale_short to 1-3 sentences."
)
_HOSTED_EVIDENCE_GUARD = (
    " For every cited_evidence_refs item, ref_id must be exactly one of: "
    "DetectorPacket.tool_findings[].evidence_refs[].ref_id, DetectorPacket.rules[].rule_id, "
    "DetectorPacket.tool_findings[].tool_result_id, or a canonical report-prefixed evidence ID from visible "
    "packet rows, notes, or variance explanations. Never put signal_id, candidate_summary, "
    "metric_name, row_id, note_id, span_id, local_evidence_id, cell_id, report_id, or tool_name by itself into ref_id. "
    "Use fewer cited_evidence_refs rather than inventing refs."
)


HOSTED_SYSTEM_PROMPT = _HOSTED_BASE_PROMPT + _HOSTED_EVIDENCE_GUARD


def hosted_output_contract(packet: dict[str, Any]) -> dict[str, Any]:
    schema = detector_assessment_json_schema()
    signal_schema = schema["properties"]["validated_signals"]["items"]
    evidence_ref_schema = schema["properties"]["cited_evidence_refs"]["items"]
    return {
        "return_only": "one DetectorAssessment JSON object; no markdown, array, or wrapper object",
        "top_level_keys_exact": list(schema["required"]),
        "additional_top_level_keys_allowed": False,
        "copy_identity_fields_exactly": {
            "packet_id": packet["packet_id"],
            "candidate_id": packet["candidate_id"],
            "report_id": packet["report_id"],
            "risk_category": packet["task"]["risk_category"],
        },
        "allowed_values": {
            "support_level": schema["properties"]["support_level"]["enum"],
            "severity": schema["properties"]["severity"]["enum"],
            "validated_signals.status": signal_schema["properties"]["status"]["enum"],
            "evidence_ref.evidence_ref_type": evidence_ref_schema["properties"]["evidence_ref_type"]["enum"],
            "evidence_ref.role": evidence_ref_schema["properties"]["role"]["enum"],
        },
        "confidence": "number only, between 0 and 1; do not use string labels such as high, medium, or low",
        "validated_signals_item": {
            "required_keys": list(signal_schema["required"]),
            "optional_keys": ["tool_result_id"],
            "additional_keys_allowed": False,
        },
        "evidence_ref_item": {
            "keys_exact": list(evidence_ref_schema["required"]),
            "ref_id_rule": "Use only IDs visible in the detector_packet.",
        },
        "rationale_short": "1-3 concise sentences using risk-signal language only",
    }


class HostedDetector:
    """An OpenAI-compatible chat endpoint (DeepSeek by default). The API key is read from an environment variable."""

    def __init__(self, model: str, *, base_url: str = "https://api.deepseek.com", api_key_env: str = "DEEPSEEK_API_KEY",
                 timeout: float = 120.0) -> None:
        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"set {api_key_env} to call {model}")
        self.model, self.url, self.timeout = model, f"{base_url.rstrip('/')}/chat/completions", timeout
        self.headers = {"Authorization": f"Bearer {key}"}

    def generate(self, prompts: list[list[dict[str, str]]]) -> list[str]:
        out = []
        for messages in prompts:
            packet = json.loads(messages[1]["content"])
            user = {"detector_packet": packet, "detector_assessment_output_contract": hosted_output_contract(packet)}
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": HOSTED_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(user, ensure_ascii=False, sort_keys=True)},
                ],
                "temperature": 0.0,
                "thinking": {"type": "disabled"},
                "response_format": {"type": "json_object"},
            }
            out.append(_post_json(self.url, payload, self.headers, self.timeout)["choices"][0]["message"]["content"])
        return out


# Rules-Only baseline: labels from the deterministic findings alone.


def rules_only_assessment(packet: dict[str, Any]) -> dict[str, Any]:
    support_level = _rule_support_level(packet)
    cited_evidence_refs = _rule_cited_evidence_refs(packet, support_level=support_level)
    signal_id = _primary_signal_id(packet)
    return {
        "assessment_id": f"RULE_ASSESS_{packet['packet_id']}",
        "packet_id": packet["packet_id"],
        "candidate_id": packet["candidate_id"],
        "report_id": packet["report_id"],
        "risk_category": packet["task"]["risk_category"],
        "support_level": support_level,
        "confidence": _rule_confidence(support_level),
        "severity": _rule_severity(packet, support_level=support_level),
        "validated_signals": [
            {
                "signal_id": signal_id,
                "status": _signal_status(support_level),
                "support_level": support_level,
                "cited_evidence_refs": cited_evidence_refs,
            }
        ],
        "cited_evidence_refs": cited_evidence_refs,
        "rationale_short": _rule_rationale(support_level),
    }


def _rule_support_level(packet: dict[str, Any]) -> str:
    if not _candidate_visible_evidence_refs(packet):
        return "insufficient_evidence"
    if any(finding.get("flag") and finding.get("evidence_refs") for finding in packet.get("tool_findings", [])):
        return "supported"
    if packet.get("tool_findings"):
        return "not_supported"
    return "weakly_supported"


def _rule_cited_evidence_refs(packet: dict[str, Any], *, support_level: str) -> list[dict[str, str]]:
    if support_level == "insufficient_evidence":
        rule_id = packet["rules"][0]["rule_id"]
        return [{"evidence_ref_type": "rule", "ref_id": rule_id, "role": "missing_required_context"}]
    for finding in packet.get("tool_findings", []):
        for evidence_ref in finding.get("evidence_refs", []):
            cited_ref = dict(evidence_ref)
            cited_ref.setdefault("role", "supporting")
            return [cited_ref]
    cited_ref = dict(_candidate_visible_evidence_refs(packet)[0])
    cited_ref.setdefault("role", "supporting")
    return [cited_ref]


def _candidate_visible_evidence_refs(packet: dict[str, Any]) -> list[dict[str, str]]:
    candidate_summary = packet.get("candidate_summary", {})
    if "evidence_refs" in candidate_summary:
        refs = candidate_summary.get("evidence_refs", [])
    else:
        refs = _packet_visible_evidence_refs(packet)
    return [ref for ref in refs if isinstance(ref, dict) and ref.get("ref_id")]


def _packet_visible_evidence_refs(packet: dict[str, Any]) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    for finding in packet.get("tool_findings", []):
        evidence_refs = finding.get("evidence_refs", [])
        if isinstance(evidence_refs, list):
            refs.extend(ref for ref in evidence_refs if isinstance(ref, dict))

    for row in packet.get("relevant_table_rows", []):
        report_id = row.get("report_id")
        if not report_id:
            continue
        values = row.get("values", {})
        value_iter = values.values() if isinstance(values, dict) else values if isinstance(values, list) else []
        for value in value_iter:
            if isinstance(value, dict) and value.get("cell_id"):
                refs.append(
                    {
                        "evidence_ref_type": "table_cell",
                        "ref_id": f"{report_id}:{value['cell_id']}",
                        "role": "supporting",
                    }
                )
        row_id = row.get("row_id") or row.get("local_evidence_id")
        if row_id:
            refs.append(
                {
                    "evidence_ref_type": "table_row",
                    "ref_id": f"{report_id}:{row_id}",
                    "role": "supporting",
                }
            )

    for note in packet.get("relevant_notes", []):
        report_id = note.get("report_id")
        note_id = note.get("note_id") or note.get("local_evidence_id")
        if not report_id or not note_id:
            continue
        note_type = note.get("note_type")
        evidence_ref_type = note.get("evidence_ref_type") or {
            "accounting_policy_change": "accounting_policy_note_span",
            "generic_accounting_policy": "accounting_policy_note_span",
            "related_party_note": "related_party_note_span",
        }.get(note_type, "note")
        refs.append(
            {
                "evidence_ref_type": evidence_ref_type,
                "ref_id": f"{report_id}:{note_id}",
                "role": "supporting",
            }
        )

    for span in packet.get("relevant_variance_explanations", []):
        report_id = span.get("report_id")
        span_id = span.get("span_id") or span.get("local_evidence_id")
        if report_id and span_id:
            refs.append(
                {
                    "evidence_ref_type": "variance_explanation_span",
                    "ref_id": f"{report_id}:{span_id}",
                    "role": "supporting",
                }
            )
    return refs


def _primary_signal_id(packet: dict[str, Any]) -> str:
    for finding in packet.get("tool_findings", []):
        if finding.get("signal_id"):
            return finding["signal_id"]
    for signal_id in packet.get("candidate_summary", {}).get("supporting_signal_ids", []):
        if signal_id:
            return signal_id
    return packet["rules"][0]["related_signal_ids"][0]


def _rule_severity(packet: dict[str, Any], *, support_level: str) -> str:
    if support_level in {"not_supported", "insufficient_evidence"}:
        return "unknown"
    priority = packet.get("candidate_summary", {}).get("priority")
    if priority in {"high", "medium", "low"}:
        return priority
    return "medium"


def _rule_confidence(support_level: str) -> float:
    if support_level == "supported":
        return 0.65
    if support_level == "weakly_supported":
        return 0.45
    return 0.5


def _signal_status(support_level: str) -> str:
    return {
        "supported": "validated",
        "weakly_supported": "partially_validated",
        "not_supported": "rejected",
        "insufficient_evidence": "not_assessable",
    }[support_level]


def _rule_rationale(support_level: str) -> str:
    return {
        "supported": "Rule baseline found a flagged tool signal with visible packet evidence.",
        "weakly_supported": "Rule baseline found visible evidence but no deterministic confirming signal.",
        "not_supported": "Rule baseline found visible evidence but no flagged supporting signal.",
        "insufficient_evidence": "Rule baseline found no visible candidate evidence to assess the risk.",
    }[support_level]


class RulesDetector:
    """The Rules-Only baseline behind the common detector interface."""

    def generate(self, prompts: list[list[dict[str, str]]]) -> list[str]:
        return [json.dumps(rules_only_assessment(json.loads(m[1]["content"]))) for m in prompts]


def make_detector(kind: str = "lora", *, adapter: str | None = None, device: str = "auto", batch_size: int = 1,
                  vllm_url: str = "http://127.0.0.1:8000/v1", vllm_model: str = "vinacount-detector",
                  api_model: str = "deepseek-v4-pro", api_base_url: str = "https://api.deepseek.com",
                  api_key_env: str = "DEEPSEEK_API_KEY"):
    """Build a detector: lora (released adapter unless `adapter` is given), base, rules, vllm, or api."""
    from vinacount import config

    if kind == "rules":
        return RulesDetector()
    if kind == "vllm":
        return VllmDetector(vllm_url, vllm_model, max_tokens=config.MAX_NEW_TOKENS)
    if kind == "api":
        return HostedDetector(api_model, base_url=api_base_url, api_key_env=api_key_env)
    if kind not in ("lora", "base"):
        raise ValueError(f"unknown detector: {kind}")
    if kind == "lora" and adapter is None:
        from huggingface_hub import snapshot_download

        adapter = snapshot_download(repo_id=config.ADAPTER_REPO, revision=config.ADAPTER_REVISION)
    return TransformersDetector(config.BASE_MODEL, adapter if kind == "lora" else None,
                                revision=config.BASE_MODEL_REVISION, device=device,
                                max_new_tokens=config.MAX_NEW_TOKENS, batch_size=batch_size)
