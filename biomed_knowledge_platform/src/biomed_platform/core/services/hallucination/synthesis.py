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
DEFAULT_NUM_PREDICT = 512

LOG_INVALID_JSON_MAX_CHARS = 1200

SYNTHESIS_JSON_ONLY_PROMPT_TEMPLATE = """You are a biomedical evidence grounded answer generator.

Goal
Produce a structured answer to the user question using ONLY the information in the provided chunk contexts.

Hard rules
1. Output MUST be a single JSON object and nothing else.
2. Output MUST start with '{{' and end with '}}'.
3. No markdown, no code fences, no extra text, no leading or trailing whitespace.
4. Do not invent facts. If the chunks do not support an answer, say so in answer.summary and keep risk_factors empty.
5. Every citation chunk_id MUST be one of the allowed chunk_ids listed below.
6. Do not include duplicate chunk_ids in any citations list.
7. Keep text concise. Prefer short rationales tied to evidence.
8. Acronyms rule (MANDATORY):
   - On first mention, write the full term followed by the acronym in parentheses.
   - After first mention, use the acronym only.
   - Do not introduce acronyms without definition.
   - Do not redefine acronyms.

Allowed chunk_ids CSV
{allowed_chunk_ids_csv}

Chunk contexts
{context}

User question
{question}

Required JSON shape example
{schema_json}

Output now as strict JSON only.
"""


def _truncate(text: str, max_chars: int) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3].rstrip() + "..."


def _safe_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        v = value.strip()
        if v.isdigit():
            try:
                return int(v)
            except Exception:
                return None
    return None


def _chunk_header(*, c: ChunkCandidate) -> str:
    year = c.year if isinstance(c.year, int) else _safe_int(c.year)
    source_type = c.source_type.value if c.source_type is not None else ""
    section = (c.section or "").strip()
    title = (c.title or "").strip()
    doi = (c.doi or "").strip()

    parts = [
        f"chunk_id={c.chunk_id}",
        f"doc_id={c.doc_id}",
        f"doi={doi}",
        f"title={title}",
        f"year={year if year is not None else ''}",
        f"section={section}",
        f"source_type={source_type}",
    ]
    return " | ".join(parts).strip()


def _truncate_chunk_text(*, text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    t = (text or "").strip()
    if not t:
        return ""
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rstrip()


def _block_overhead(*, header: str) -> tuple[str, str, int]:
    block_prefix = f"\n<<<CHUNK {header}>>>\n"
    block_suffix = "\n<<<END_CHUNK>>>\n"
    return block_prefix, block_suffix, len(block_prefix) + len(block_suffix)


def _remaining_capacity(*, used: int, max_chars: int, overhead: int) -> int:
    remaining = max_chars - used - overhead
    return remaining if remaining > 0 else 0


def _should_stop(*, used: int, max_chars: int) -> bool:
    return used >= max_chars


def _build_context(*, chunks: Sequence[ChunkCandidate], max_chars: int) -> tuple[str, set[str]]:
    limit = int(max_chars)
    if limit <= 0:
        limit = DEFAULT_MAX_CONTEXT_CHARS

    out: list[str] = []
    included_chunk_ids: set[str] = set()
    used = 0

    for c in chunks:
        chunk_id = (c.chunk_id or "").strip()
        text = (c.chunk_text or "").strip()
        if not chunk_id or not text:
            continue

        header = _chunk_header(c=c)
        block_prefix, block_suffix, overhead = _block_overhead(header=header)
        if used + overhead >= limit:
            break

        remaining_for_text = _remaining_capacity(used=used, max_chars=limit, overhead=overhead)
        chunk_text = _truncate_chunk_text(text=text, max_chars=remaining_for_text)
        if not chunk_text:
            break

        out.append(block_prefix)
        out.append(chunk_text)
        out.append(block_suffix)

        included_chunk_ids.add(chunk_id)
        used += overhead + len(chunk_text)

        if _should_stop(used=used, max_chars=limit):
            break

    context = "".join(out).strip()
    if context and len(context) > limit:
        context = context[:limit].rstrip()

    if not context:
        return "", set()

    return context, included_chunk_ids


def _schema_json() -> str:
    example = {
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
            {
                "chunk_id": "chunk_id",
                "doi": "string or empty",
                "title": "string or empty",
                "year": 0,
                "snippet": "string",
            }
        ],
    }
    return json.dumps(example, ensure_ascii=False, separators=(",", ":"))


def _extract_referenced_chunk_ids(parsed: SynthesisOutput) -> set[str]:
    ids: set[str] = set()

    for rf in parsed.answer.risk_factors:
        for cid in rf.citations:
            if isinstance(cid, str) and cid.strip():
                ids.add(cid.strip())

    for c in parsed.citations:
        cid = (c.chunk_id or "").strip()
        if cid:
            ids.add(cid)

    return ids


def _validate_chunk_ids(*, parsed: SynthesisOutput, allowed_chunk_ids: set[str]) -> None:
    referenced = _extract_referenced_chunk_ids(parsed)
    missing = sorted([cid for cid in referenced if cid not in allowed_chunk_ids])
    if missing:
        raise ValueError(f"unknown_chunk_ids_referenced: {missing}")


def _to_request_validation_error(*, message: str) -> RequestValidationError:
    return RequestValidationError(
        [
            {
                "type": "value_error",
                "loc": ("body", "llm_output"),
                "msg": message,
                "input": None,
            }
        ]
    )


def _normalize_synthesis_payload(data: object) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None

    answer = data.get("answer")
    if isinstance(answer, dict):
        limitations = answer.get("limitations")

        if limitations is None:
            answer["limitations"] = []
        elif isinstance(limitations, str):
            s = limitations.strip()
            answer["limitations"] = [] if not s else [s]
        elif isinstance(limitations, list):
            cleaned: list[str] = []
            for item in limitations:
                if isinstance(item, str):
                    t = item.strip()
                    if t:
                        cleaned.append(t)
            answer["limitations"] = cleaned
        else:
            answer["limitations"] = []

    return data


def _extract_json_object(text: str) -> str | None:
    s = (text or "").strip()
    if not s:
        return None
    if s.startswith("{") and s.endswith("}"):
        return s
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return s[start : end + 1]


def _validate_inputs(
    *, question: str, selected_chunks: Sequence[ChunkCandidate]
) -> tuple[str, list[ChunkCandidate]]:
    q = (question or "").strip()
    if not q:
        raise _to_request_validation_error(message="question must be non empty")

    chunks = list(selected_chunks or [])
    has_any_chunk_id = any(isinstance(c.chunk_id, str) and c.chunk_id.strip() for c in chunks)
    if not has_any_chunk_id:
        raise _to_request_validation_error(
            message="selected_chunks must include at least one chunk_id"
        )

    return q, chunks


def _build_prompt(
    *,
    question: str,
    chunks: Sequence[ChunkCandidate],
    max_context_chars: int,
) -> tuple[str, str, set[str]]:
    context, included_chunk_ids = _build_context(chunks=chunks, max_chars=max_context_chars)
    if not context or not included_chunk_ids:
        raise _to_request_validation_error(
            message="selected_chunks must include non empty chunk_text"
        )

    allowed_chunk_ids_csv = ",".join(sorted(included_chunk_ids))
    schema_json = _schema_json()

    base_prompt = SYNTHESIS_JSON_ONLY_PROMPT_TEMPLATE.format(
        question=question,
        context=context,
        schema_json=schema_json,
        allowed_chunk_ids_csv=allowed_chunk_ids_csv,
    )
    return base_prompt, context, included_chunk_ids


def _build_llm_options(*, llm_options: Mapping[str, Any] | None) -> dict[str, Any]:
    options: dict[str, Any] = {
        "temperature": 0,
        "num_predict": DEFAULT_NUM_PREDICT,
        "format": "json",
    }
    if llm_options:
        options.update(dict(llm_options))
    return options


def _retry_prompt(*, base_prompt: str, attempt_idx: int) -> str:
    if attempt_idx <= 0:
        return base_prompt
    return (
        "Previous output was invalid. Return STRICT JSON only, starting with { and ending with }.\n\n"
        + base_prompt
    )


async def _call_llm(
    *,
    llm: LlmClientPort,
    model_id: str,
    prompt: str,
    options: Mapping[str, Any],
) -> str:
    try:
        return await llm.chat(
            model_id=model_id,
            messages=[LlmChatMessage(role="user", content=prompt)],
            options=options,
        )
    except LlmCallError as e:
        log.error(
            f"Synthesis LLM call failed | model_id={model_id} | retryable={e.retryable} | details={e.details}"
        )
        raise


@dataclass(frozen=True)
class _ParseOk:
    parsed: SynthesisOutput


@dataclass(frozen=True)
class _ParseErr:
    code: str
    excerpt: str


def _parse_and_validate_output(
    *,
    raw: str,
    allowed_chunk_ids: set[str],
    model_id: str,
    attempt_no: int,
    attempts: int,
) -> _ParseOk | _ParseErr:
    cleaned = (raw or "").strip()
    if not cleaned:
        log.warning(
            f"Synthesis LLM returned empty output | model_id={model_id} | attempt={attempt_no}/{attempts}"
        )
        return _ParseErr(code="llm_returned_empty_output", excerpt="")

    extracted = _extract_json_object(cleaned)
    if not extracted:
        excerpt = _truncate(cleaned, LOG_INVALID_JSON_MAX_CHARS)
        log.warning(
            f"Synthesis output missing JSON object | model_id={model_id} | attempt={attempt_no}/{attempts} | raw_excerpt={excerpt}"
        )
        return _ParseErr(code="invalid_json: no_object", excerpt=excerpt)

    try:
        data = json.loads(extracted)
    except Exception as e:
        excerpt = _truncate(extracted, LOG_INVALID_JSON_MAX_CHARS)
        log.warning(
            f"Synthesis output invalid JSON | model_id={model_id} | attempt={attempt_no}/{attempts} | error_type={type(e).__name__} | json_excerpt={excerpt}"
        )
        return _ParseErr(code=f"invalid_json: {type(e).__name__}", excerpt=excerpt)

    normalized = _normalize_synthesis_payload(data)
    if normalized is None:
        excerpt = _truncate(extracted, LOG_INVALID_JSON_MAX_CHARS)
        log.warning(
            f"Synthesis output invalid root type | model_id={model_id} | attempt={attempt_no}/{attempts} | json_excerpt={excerpt}"
        )
        return _ParseErr(code="invalid_json: root_not_object", excerpt=excerpt)

    try:
        parsed = SynthesisOutput.model_validate(normalized)
        _validate_chunk_ids(parsed=parsed, allowed_chunk_ids=allowed_chunk_ids)
    except (ValidationError, ValueError) as e:
        excerpt = _truncate(extracted, LOG_INVALID_JSON_MAX_CHARS)
        log.warning(
            f"Synthesis output failed validation | model_id={model_id} | attempt={attempt_no}/{attempts} | error_type={type(e).__name__} | json_excerpt={excerpt}"
        )
        return _ParseErr(code=f"invalid_schema_or_citations: {type(e).__name__}", excerpt=excerpt)

    return _ParseOk(parsed=parsed)


def _attempts(max_json_retries: int) -> int:
    return max(0, int(max_json_retries)) + 1


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
    q, chunks = _validate_inputs(question=question, selected_chunks=selected_chunks)
    base_prompt, context, allowed_chunk_ids = _build_prompt(
        question=q,
        chunks=chunks,
        max_context_chars=max_context_chars,
    )
    options = _build_llm_options(llm_options=llm_options)

    attempts = _attempts(max_json_retries)
    last_error: str | None = None

    for attempt_idx in range(attempts):
        attempt_no = attempt_idx + 1
        prompt = _retry_prompt(base_prompt=base_prompt, attempt_idx=attempt_idx)

        log.info(
            f"Synthesis LLM call started | model_id={model_id} | "
            f"question_length={len(q)} | chunks={len(chunks)} | "
            f"context_chars={len(context)} | attempt={attempt_no}/{attempts}"
        )

        raw = await _call_llm(llm=llm, model_id=model_id, prompt=prompt, options=options)
        parsed = _parse_and_validate_output(
            raw=raw,
            allowed_chunk_ids=allowed_chunk_ids,
            model_id=model_id,
            attempt_no=attempt_no,
            attempts=attempts,
        )

        if isinstance(parsed, _ParseOk):
            out = parsed.parsed
            log.info(
                f"Synthesis completed | model_id={model_id} | "
                f"risk_factors={len(out.answer.risk_factors)} | citations={len(out.citations)}"
            )
            return SynthesisResult(answer=out.answer, citations=out.citations)

        last_error = parsed.code

    raise _to_request_validation_error(message=last_error or "invalid_llm_output")
