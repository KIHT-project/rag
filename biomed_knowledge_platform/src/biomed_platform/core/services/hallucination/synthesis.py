from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from biomed_platform.common.logging import get_logger
from biomed_platform.core.domains.llm import LlmChatMessage
from biomed_platform.core.domains.retrieval import ChunkCandidate
from biomed_platform.core.domains.synthesis import SynthesisResult, SynthesisOutput
from biomed_platform.core.ports.llm import LlmCallError, LlmClientPort

log = get_logger(__name__)

DEFAULT_MAX_CONTEXT_CHARS = 24000
DEFAULT_NUM_PREDICT = 1024

LOG_INVALID_JSON_MAX_CHARS = 1200

SUMMARY_MAX_CHARS = 2500
SNIPPET_MAX_CHARS = 350
RATIONALE_MAX_CHARS = 600

MAX_RISK_FACTORS = 7
MAX_CITATIONS = 10

SUMMARY_HARD_CAP = 900
SNIPPET_HARD_CAP = 260
RATIONALE_HARD_CAP = 420

RISK_FACTOR_FALLBACK_CITATIONS = 2

MIN_TOP_LEVEL_CITATIONS = 3

SYNTHESIS_JSON_ONLY_PROMPT_TEMPLATE = f"""You are a biomedical evidence grounded answer generator.

Goal
Produce a structured answer to the user question using ONLY the information in the provided chunk contexts.

Hard rules
1. Output MUST be a single JSON object and nothing else.
2. No markdown, no code fences, no commentary.
3. Do not invent facts.
4. If you cite anything, cite only allowed chunk_ids.
5. If the chunks do not support an answer, say so in answer.summary, keep risk_factors empty, and include a limitation.
6. Acronyms rule (MANDATORY)
   1. On first mention, write the full term followed by the acronym in parentheses.
   2. After first mention, use the acronym only.
   3. Do not introduce acronyms without definition.
   4. Do not redefine acronyms.
7. Each risk_factors item SHOULD include at least one citation chunk_id. If uncertain, cite the most relevant chunk that mentions it.

Quality requirements
Summary requirements
1. Write a compact evidence grounded synthesis, not a one liner.
2. Include: population or setting, intervention or exposure, key outcome direction, and 1 to 2 key limitations.
3. Do not include numbers unless they exist in the chunk text.
4. Max {SUMMARY_HARD_CAP} characters.

Risk factor rationale requirements
1. Rationale explains why this is a relevant risk factor in this specific question context.
2. Must reference what the chunks say, not generic medical knowledge.
3. Max {RATIONALE_HARD_CAP} characters per rationale.

Output constraints
1. Max {MAX_RISK_FACTORS} risk_factors.
2. Max {MAX_CITATIONS} citations.
3. Each citation.snippet must be a short excerpt from the chunk, max {SNIPPET_HARD_CAP} characters.

Allowed chunk_ids CSV
{{allowed_chunk_ids_csv}}

Chunk contexts
{{context}}

User question
{{question}}

Required JSON shape example
{{schema_json}}

Output JSON now.
"""


def _truncate(text: str, max_chars: int) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    if max_chars <= 0:
        return ""
    if len(s) <= max_chars:
        return s
    return s[: max(0, max_chars - 3)].rstrip() + "..."


def _to_request_error(message: str) -> RequestValidationError:
    return RequestValidationError(
        [{"type": "value_error", "loc": ("body", "llm_output"), "msg": message, "input": None}]
    )


def _schema_json() -> str:
    obj = {
        "answer": {
            "summary": "string",
            "risk_factors": [
                {
                    "rank": 1,
                    "normalized_name": "string",
                    "aliases": ["string"],
                    "confidence": 0.5,
                    "rationale": "string",
                    "citations": ["chunk_id"],
                }
            ],
            "limitations": ["string"],
        },
        "citations": [
            {"chunk_id": "chunk_id", "doi": "", "title": "", "year": 0, "snippet": "string"}
        ],
    }
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _build_context(chunks: Sequence[ChunkCandidate], max_chars: int) -> tuple[str, set[str]]:
    limit_raw = int(max_chars)
    limit = limit_raw if limit_raw > 0 else DEFAULT_MAX_CONTEXT_CHARS

    out: list[str] = []
    used = 0
    allowed: set[str] = set()

    for c in chunks:
        chunk_id = (c.chunk_id or "").strip()
        chunk_text = (c.chunk_text or "").strip()
        if not chunk_id or not chunk_text:
            continue

        prefix = f"\n<<<CHUNK {chunk_id}>>>\n"
        suffix = "\n<<<END_CHUNK>>>\n"

        remaining = limit - used
        if remaining <= 0:
            break

        overhead = len(prefix) + len(suffix)
        if remaining <= overhead:
            break

        max_text_len = remaining - overhead

        if len(chunk_text) <= max_text_len:
            block = prefix + chunk_text + suffix
            out.append(block)
            allowed.add(chunk_id)
            used += len(block)
            continue

        if out:
            continue

        truncated_text = _truncate(chunk_text, max_text_len)
        if not truncated_text:
            continue

        block = prefix + truncated_text + suffix
        out.append(block)
        allowed.add(chunk_id)
        used += len(block)

    return "".join(out).strip(), allowed


def _extract_json_slice(text: str) -> str | None:
    s = (text or "").strip()
    if not s:
        return None

    if s.startswith("{") and s.endswith("}"):
        return s

    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end <= start:
        return None

    return s[start : end + 1]


def _try_load_json_object(text: str) -> dict[str, Any] | None:
    raw = _extract_json_slice(text)
    if not raw:
        return None
    try:
        obj = json.loads(raw)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def _as_str(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return int(float(s))
        except Exception:
            return None
    return None


def _as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            return None
    return None


def _as_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    if isinstance(value, list):
        out: list[str] = []
        for it in value:
            s = _as_str(it)
            if s:
                out.append(s)
        return out
    return []


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _normalize_risk_factor(item: object) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None

    rationale_cap = min(RATIONALE_MAX_CHARS, RATIONALE_HARD_CAP)

    return {
        "rank": int(_as_int(item.get("rank")) or 0),
        "normalized_name": _as_str(item.get("normalized_name")) or _as_str(item.get("name")),
        "aliases": _dedupe_keep_order(_as_str_list(item.get("aliases"))),
        "confidence": float(_as_float(item.get("confidence")) or 0.0),
        "rationale": _truncate(_as_str(item.get("rationale")), rationale_cap),
        "citations": _as_str_list(item.get("citations")),
    }


def _normalize_answer(answer_in: object) -> dict[str, Any]:
    src = answer_in if isinstance(answer_in, dict) else {}

    rf_raw = src.get("risk_factors")
    risk_factors: list[dict[str, Any]] = []
    if isinstance(rf_raw, list):
        for it in rf_raw:
            rf = _normalize_risk_factor(it)
            if rf is not None:
                risk_factors.append(rf)

    return {
        "summary": _as_str(src.get("summary")),
        "risk_factors": risk_factors,
        "limitations": _as_str_list(src.get("limitations")),
    }


def _normalize_citation(item: object) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None

    chunk_id = _as_str(item.get("chunk_id"))
    if not chunk_id:
        return None

    year = _as_int(item.get("year"))
    snippet_cap = min(SNIPPET_MAX_CHARS, SNIPPET_HARD_CAP)

    return {
        "chunk_id": chunk_id,
        "doi": _as_str(item.get("doi")),
        "title": _as_str(item.get("title")),
        "year": int(year) if year is not None else 0,
        "snippet": _truncate(_as_str(item.get("snippet")), snippet_cap),
    }


def _normalize_citations(citations_in: object) -> list[dict[str, Any]]:
    if not isinstance(citations_in, list):
        return []
    out: list[dict[str, Any]] = []
    for it in citations_in:
        c = _normalize_citation(it)
        if c is not None:
            out.append(c)
    return out


def _normalize_payload(data: object) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    return {
        "answer": _normalize_answer(data.get("answer")),
        "citations": _normalize_citations(data.get("citations")),
    }


def _collect_risk_factor_chunk_ids(answer: dict[str, Any]) -> list[str]:
    rf_raw = answer.get("risk_factors")
    rf_list = rf_raw if isinstance(rf_raw, list) else []
    out: list[str] = []
    for rf in rf_list:
        if not isinstance(rf, dict):
            continue
        out.extend(_as_str_list(rf.get("citations")))
    return out


def _collect_top_level_chunk_ids(citations: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for c in citations:
        if not isinstance(c, dict):
            continue
        out.append(_as_str(c.get("chunk_id")))
    return out


def _filter_chunk_ids_to_allowed(chunk_ids: list[str], allowed: set[str]) -> list[str]:
    out: list[str] = []
    for cid in chunk_ids:
        if not cid:
            continue
        if cid not in allowed:
            continue
        out.append(cid)
    return _dedupe_keep_order(out)


def _build_chunk_meta(selected_chunks: Sequence[ChunkCandidate]) -> dict[str, ChunkCandidate]:
    out: dict[str, ChunkCandidate] = {}
    for c in selected_chunks:
        cid = (c.chunk_id or "").strip()
        if not cid:
            continue
        out[cid] = c
    return out


def _ensure_top_level_citations(
    normalized: dict[str, Any],
    *,
    allowed: set[str],
    chunk_meta: Mapping[str, ChunkCandidate],
) -> list[dict[str, Any]]:
    citations_raw = normalized.get("citations")
    citations_list = citations_raw if isinstance(citations_raw, list) else []

    answer_raw = normalized.get("answer")
    answer = answer_raw if isinstance(answer_raw, dict) else {}

    top_ids = _filter_chunk_ids_to_allowed(_collect_top_level_chunk_ids(citations_list), allowed)
    rf_ids = _filter_chunk_ids_to_allowed(_collect_risk_factor_chunk_ids(answer), allowed)

    all_ids = _dedupe_keep_order(top_ids + rf_ids)
    if not all_ids:
        return []

    by_id: dict[str, dict[str, Any]] = {}
    for c in citations_list:
        if not isinstance(c, dict):
            continue
        cid = _as_str(c.get("chunk_id"))
        if not cid or cid not in allowed:
            continue
        by_id[cid] = c

    snippet_cap = min(SNIPPET_MAX_CHARS, SNIPPET_HARD_CAP)

    out: list[dict[str, Any]] = []
    for cid in all_ids:
        existing = by_id.get(cid)
        meta = chunk_meta.get(cid)

        doi = _as_str((existing or {}).get("doi")) or _as_str(getattr(meta, "doi", ""))
        title = _as_str((existing or {}).get("title")) or _as_str(getattr(meta, "title", ""))

        year_val = _as_int((existing or {}).get("year"))
        if year_val is None:
            year_val = _as_int(getattr(meta, "year", None))

        snippet = _as_str((existing or {}).get("snippet"))
        if not snippet and meta is not None:
            snippet = _as_str(getattr(meta, "chunk_text", ""))
        snippet = _truncate(snippet, snippet_cap)

        out.append(
            {
                "chunk_id": cid,
                "doi": doi,
                "title": title,
                "year": int(year_val) if year_val is not None else 0,
                "snippet": snippet,
            }
        )

        if len(out) >= MAX_CITATIONS:
            break

    return out


def _count_unique_top_level_citations(citations: list[dict[str, Any]]) -> int:
    seen: set[str] = set()
    for c in citations:
        if not isinstance(c, dict):
            continue
        cid = _as_str(c.get("chunk_id"))
        if cid:
            seen.add(cid)
    return len(seen)


def _cap_risk_factors(answer: dict[str, Any]) -> dict[str, Any]:
    rf_raw = answer.get("risk_factors")
    rf_list = rf_raw if isinstance(rf_raw, list) else []
    if len(rf_list) > MAX_RISK_FACTORS:
        answer["risk_factors"] = rf_list[:MAX_RISK_FACTORS]
    return answer


def _backfill_risk_factor_citations(
    answer: dict[str, Any],
    *,
    allowed: set[str],
    fallback_ids: list[str],
) -> tuple[dict[str, Any], bool]:
    rf_raw = answer.get("risk_factors")
    rf_list = rf_raw if isinstance(rf_raw, list) else []

    fallback = _filter_chunk_ids_to_allowed(fallback_ids, allowed)
    if not fallback:
        fallback = list(sorted(allowed))[:RISK_FACTOR_FALLBACK_CITATIONS]

    any_backfilled = False
    kept: list[dict[str, Any]] = []

    rationale_cap = min(RATIONALE_MAX_CHARS, RATIONALE_HARD_CAP)

    for rf in rf_list:
        if not isinstance(rf, dict):
            continue

        rf_out = dict(rf)

        cids = _filter_chunk_ids_to_allowed(_as_str_list(rf_out.get("citations")), allowed)
        if not cids:
            cids = fallback[:RISK_FACTOR_FALLBACK_CITATIONS]
            if cids:
                any_backfilled = True

        rf_out["citations"] = cids
        rf_out["aliases"] = _dedupe_keep_order(_as_str_list(rf_out.get("aliases")))
        rf_out["rationale"] = _truncate(_as_str(rf_out.get("rationale")), rationale_cap)

        kept.append(rf_out)
        if len(kept) >= MAX_RISK_FACTORS:
            break

    out_answer = dict(answer)
    out_answer["risk_factors"] = kept
    return out_answer, any_backfilled


def _append_limitations(answer: dict[str, Any], limitations: list[str]) -> dict[str, Any]:
    lim = _as_str_list(answer.get("limitations"))
    lim.extend([x for x in limitations if x])
    answer["limitations"] = _dedupe_keep_order(lim)
    return answer


def _ensure_summary(answer: dict[str, Any], fallback_summary: str) -> dict[str, Any]:
    cap = min(SUMMARY_MAX_CHARS, SUMMARY_HARD_CAP)
    existing = _as_str(answer.get("summary"))
    value = existing or fallback_summary
    answer["summary"] = _truncate(value, cap)
    return answer


def _build_prompt(question: str, context: str, allowed: set[str]) -> str:
    return SYNTHESIS_JSON_ONLY_PROMPT_TEMPLATE.format(
        question=question,
        context=context,
        allowed_chunk_ids_csv=",".join(sorted(allowed)),
        schema_json=_schema_json(),
    )


def _build_retry_prompt(base_prompt: str, reason: str) -> str:
    return (
        "Your previous output was invalid.\n"
        f"Reason: {reason}\n"
        "Return a single JSON object only.\n"
        "If you provide citations, use only allowed chunk_ids.\n\n" + base_prompt
    )


def _build_llm_options(llm_options: Mapping[str, Any] | None) -> dict[str, Any]:
    options: dict[str, Any] = {
        "temperature": 0,
        "num_predict": DEFAULT_NUM_PREDICT,
        "format": "json",
    }
    if llm_options:
        options.update(dict(llm_options))
    return options


async def _call_llm(
    llm: LlmClientPort, model_id: str, prompt: str, options: Mapping[str, Any]
) -> str:
    try:
        return await llm.chat(
            model_id=model_id,
            messages=[LlmChatMessage(role="user", content=prompt)],
            options=options,
        )
    except LlmCallError as e:
        log.error(
            "Synthesis LLM call failed | model_id=%s | retryable=%s | details=%s",
            model_id,
            bool(e.retryable),
            e.details,
        )
        raise


@dataclass(frozen=True)
class _ParseResult:
    ok: bool
    output: SynthesisOutput | None
    grounded: bool
    code: str


def _build_fallback_output(*, reason: str) -> SynthesisOutput:
    payload = {
        "answer": {
            "summary": "No evidence grounded answer could be produced.",
            "risk_factors": [],
            "limitations": [reason],
        },
        "citations": [],
    }
    return SynthesisOutput.model_validate(payload)


def _parse_and_validate(
    *,
    raw: str,
    allowed: set[str],
    selected_chunks: Sequence[ChunkCandidate],
    model_id: str,
) -> _ParseResult:
    cleaned = (raw or "").strip()
    if not cleaned:
        log.warning("Synthesis empty output | model_id=%s", model_id)
        return _ParseResult(ok=False, output=None, grounded=False, code="llm_empty_output")

    excerpt = _truncate(cleaned, LOG_INVALID_JSON_MAX_CHARS)

    obj = _try_load_json_object(cleaned)
    if obj is None:
        log.warning("Synthesis invalid JSON | model_id=%s | excerpt=%s", model_id, excerpt)
        return _ParseResult(ok=False, output=None, grounded=False, code="invalid_json")

    normalized = _normalize_payload(obj)
    if normalized is None:
        log.warning("Synthesis invalid root type | model_id=%s | excerpt=%s", model_id, excerpt)
        return _ParseResult(ok=False, output=None, grounded=False, code="invalid_root")

    chunk_meta = _build_chunk_meta(selected_chunks)

    citations = _ensure_top_level_citations(normalized, allowed=allowed, chunk_meta=chunk_meta)
    grounded = bool(citations)
    normalized["citations"] = citations

    answer_raw = normalized.get("answer")
    answer = answer_raw if isinstance(answer_raw, dict) else {}

    answer = _cap_risk_factors(answer)

    fallback_ids = [c.get("chunk_id", "") for c in citations if isinstance(c, dict)]
    answer, did_backfill = _backfill_risk_factor_citations(
        answer, allowed=allowed, fallback_ids=fallback_ids
    )

    if did_backfill:
        answer = _append_limitations(
            answer, ["Some risk factor citations were assigned from the overall cited evidence."]
        )

        normalized["answer"] = answer
        citations = _ensure_top_level_citations(normalized, allowed=allowed, chunk_meta=chunk_meta)
        grounded = bool(citations)
        normalized["citations"] = citations

    unique_top_level = _count_unique_top_level_citations(citations)
    if grounded and unique_top_level < MIN_TOP_LEVEL_CITATIONS:
        answer = _append_limitations(
            answer,
            [
                f"Evidence base is limited, only {unique_top_level} unique citations were available for the synthesis."
            ],
        )

    answer = _ensure_summary(answer, "No evidence grounded answer could be produced.")
    if not grounded:
        answer = _append_limitations(
            answer, ["No citations were produced by the model, answer is not evidence grounded."]
        )

    normalized["answer"] = answer

    try:
        out = SynthesisOutput.model_validate(normalized)
        return _ParseResult(ok=True, output=out, grounded=grounded, code="")
    except (ValidationError, Exception):
        log.warning(
            "Synthesis schema validate failed | model_id=%s | excerpt=%s", model_id, excerpt
        )
        return _ParseResult(ok=False, output=None, grounded=False, code="invalid_schema_or_types")


async def synthesize_answer(
    *,
    llm: LlmClientPort,
    model_id: str,
    question: str,
    selected_chunks: Sequence[ChunkCandidate],
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    max_json_retries: int = 1,
    llm_options: Mapping[str, Any] | None = None,
) -> SynthesisResult:
    q = (question or "").strip()
    if not q:
        raise _to_request_error("question must be non empty")

    context, allowed = _build_context(selected_chunks, max_context_chars)
    if not context or not allowed:
        raise _to_request_error("no_usable_chunk_text")

    base_prompt = _build_prompt(question=q, context=context, allowed=allowed)
    options = _build_llm_options(llm_options)

    attempts = max(0, int(max_json_retries)) + 1
    last_code = "unknown"

    for attempt_no in range(1, attempts + 1):
        prompt = (
            base_prompt if attempt_no == 1 else _build_retry_prompt(base_prompt, reason=last_code)
        )

        log.info(
            "Synthesis started | model_id=%s | attempt=%s/%s | chunks=%s | context_chars=%s",
            model_id,
            attempt_no,
            attempts,
            len(allowed),
            len(context),
        )

        raw = await _call_llm(llm, model_id, prompt, options)
        parsed = _parse_and_validate(
            raw=raw,
            allowed=allowed,
            selected_chunks=selected_chunks,
            model_id=model_id,
        )

        if parsed.ok and parsed.output is not None:
            out = parsed.output
            log.info(
                "Synthesis completed | model_id=%s | citations=%s | risk_factors=%s | grounded=%s",
                model_id,
                len(out.citations),
                len(out.answer.risk_factors),
                bool(parsed.grounded),
            )
            return SynthesisResult(answer=out.answer, citations=out.citations)

        last_code = parsed.code or "unknown"
        log.warning(
            "Synthesis attempt failed | model_id=%s | attempt=%s/%s | code=%s",
            model_id,
            attempt_no,
            attempts,
            last_code,
        )

    log.warning("Synthesis retries exhausted | model_id=%s | last_code=%s", model_id, last_code)
    out = _build_fallback_output(reason=last_code)
    return SynthesisResult(answer=out.answer, citations=out.citations)
