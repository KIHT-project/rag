from __future__ import annotations

from dataclasses import dataclass, field
import os
import re

from biomed_platform.common.logging import get_logger
from biomed_platform.core.domains.ingestion import TextChunk
from biomed_platform.core.ports.ingestion import Chunker

log = get_logger(__name__)

_EXCLUDE_TITLES_ENV = "RAG_SECTION_EXCLUDE_TITLES"
_DEFAULT_EXCLUDE_TITLES = (
    "references",
    "bibliography",
    "acknowledgments",
    "acknowledgements",
    "supplementary",
    "supplementary materials",
    "appendix",
    "funding",
    "conflict of interest",
    "conflicts of interest",
    "disclosures",
)

_KNOWN_SECTION_TITLES = (
    "abstract",
    "background",
    "introduction",
    "methods",
    "materials and methods",
    "methodology",
    "patients and methods",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "objective",
    "objectives",
    "aim",
    "aims",
)

_PARA_SPLIT_RE = re.compile(r"\n\s*\n+")
_LABEL_LINE_RE = re.compile(r"^[A-Z][A-Z0-9 /\-]{2,}:\s+\S")
_INLINE_LABEL_RE = re.compile(r"^(?P<label>[A-Za-z][A-Za-z0-9 /\-]{2,})\s*:\s*(?P<rest>\S.*)$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\\s+")


@dataclass(frozen=True, slots=True)
class SimpleCharChunker(Chunker):
    chunk_size: int = 1200
    overlap: int = 150

    def chunk(self, *, text: str) -> list[TextChunk]:
        log.debug(
            "Chunking request received, raw_text_length=%s, chunk_size=%s, overlap=%s",
            len(text) if text else 0,
            self.chunk_size,
            self.overlap,
        )

        t = (text or "").strip()
        if not t:
            log.info("Empty or whitespace only text received, returning no chunks")
            return []

        size = max(1, int(self.chunk_size))
        ov = max(0, int(self.overlap))

        if ov >= size:
            adjusted_ov = max(0, size // 4)
            log.warning(
                "Overlap >= chunk_size, adjusting overlap, original_overlap=%s,"
                "adjusted_overlap=%s",
                ov,
                adjusted_ov,
            )
            ov = adjusted_ov

        out: list[TextChunk] = []
        start = 0
        idx = 0
        text_len = len(t)

        while start < text_len:
            end = min(text_len, start + size)

            out.append(
                TextChunk(
                    index=idx,
                    text=t[start:end],
                    start=start,
                    end=end,
                )
            )

            log.debug(
                "Created chunk, index=%s, start=%s, end=%s, chunk_length=%s",
                idx,
                start,
                end,
                end - start,
            )

            idx += 1

            if end == text_len:
                log.debug("Reached end of text at index=%s", idx - 1)
                break

            start = max(0, end - ov)

        log.info(
            "Chunking completed, total_chunks=%s, text_length=%s",
            len(out),
            text_len,
        )

        return out


def _normalize_title(title: str) -> str:
    return " ".join(title.lower().strip().split())


def _format_section_title(title: str) -> str:
    normalized = _normalize_title(title)
    if not normalized:
        return title.strip()

    small = {"and", "or", "of", "in", "on", "for", "with", "to", "the", "a", "an"}
    words = []
    for word in normalized.split():
        if word in small:
            words.append(word)
        else:
            words.append(word.capitalize())
    formatted = " ".join(words)
    return formatted[:1].upper() + formatted[1:] if formatted else title.strip()


def _extract_inline_label(block: str) -> tuple[str, str] | None:
    match = _INLINE_LABEL_RE.match(block)
    if not match:
        return None
    label = match.group("label").strip()
    if _normalize_title(label) not in _KNOWN_SECTION_TITLES:
        return None
    rest = match.group("rest").strip()
    return _format_section_title(label), rest


def _load_exclude_titles(override: tuple[str, ...]) -> tuple[str, ...]:
    if override:
        return tuple(_normalize_title(t) for t in override if t.strip())

    raw = os.getenv(_EXCLUDE_TITLES_ENV, "").strip()
    if raw:
        return tuple(_normalize_title(t) for t in raw.split(",") if t.strip())

    return tuple(_normalize_title(t) for t in _DEFAULT_EXCLUDE_TITLES)


def _split_paragraphs(text: str) -> list[str]:
    blocks = [b.strip() for b in _PARA_SPLIT_RE.split(text) if b.strip()]
    if len(blocks) > 1:
        return blocks

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) > 1 and any(_LABEL_LINE_RE.match(line) for line in lines):
        return lines

    return blocks if blocks else lines


def _passes_title_case_rules(block: str) -> bool:
    stripped = block.strip()
    if not stripped:
        return False
    return stripped[0].isupper() or stripped.isupper()


def _passes_title_length_rules(block: str, *, max_len: int) -> bool:
    if " " not in block and len(block) > min(max_len, 60):
        return False
    return len(block) <= max_len


def _passes_title_punctuation_rules(block: str) -> bool:
    if ":" in block:
        return False
    return block[-1] not in ".!?;"


def _passes_title_word_rules(block: str, *, max_words: int) -> bool:
    return len(block.split()) <= max_words


def _passes_title_alpha_ratio(block: str) -> bool:
    alpha_chars = sum(1 for c in block if c.isalpha())
    return alpha_chars / max(1, len(block)) >= 0.6


def _is_section_title(block: str, *, max_len: int, max_words: int) -> bool:
    if not block or "\n" in block:
        return False

    normalized = _normalize_title(block)
    if normalized in _KNOWN_SECTION_TITLES:
        return True

    if not _passes_title_case_rules(block):
        return False
    if not _passes_title_length_rules(block, max_len=max_len):
        return False
    if not _passes_title_punctuation_rules(block):
        return False
    if not _passes_title_word_rules(block, max_words=max_words):
        return False
    if not _passes_title_alpha_ratio(block):
        return False

    return True


def _is_major_title(title: str) -> bool:
    return _normalize_title(title) in _KNOWN_SECTION_TITLES


def _split_long_text(text: str, *, max_len: int, overlap: int) -> list[str]:
    if max_len <= 0:
        return [text]

    ov = max(0, overlap)
    if ov >= max_len:
        ov = max_len // 4

    chunks: list[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(text_len, start + max_len)
        chunks.append(text[start:end])
        if end == text_len:
            break
        start = max(0, end - ov)
    return chunks


def _split_sentence_by_words(sentence: str, *, max_len: int, overlap: int) -> list[str]:
    words = sentence.split()
    chunks: list[str] = []
    wcur: list[str] = []
    wlen = 0

    for word in words:
        if len(word) > max_len:
            if wcur:
                chunks.append(" ".join(wcur).strip())
                wcur = []
                wlen = 0
            chunks.extend(_split_long_text(word, max_len=max_len, overlap=overlap))
            continue

        add = len(word) if not wcur else len(word) + 1
        if wcur and (wlen + add) > max_len:
            chunks.append(" ".join(wcur).strip())
            wcur = [word]
            wlen = len(word)
        else:
            wcur.append(word)
            wlen += add

    if wcur:
        chunks.append(" ".join(wcur).strip())
    return chunks


def _pack_sentences(sentences: list[str], *, max_len: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0

    def flush() -> None:
        nonlocal cur, cur_len
        if not cur:
            return
        chunks.append(" ".join(cur).strip())
        cur = []
        cur_len = 0

    for sentence in sentences:
        if len(sentence) > max_len:
            flush()
            chunks.extend(_split_sentence_by_words(sentence, max_len=max_len, overlap=overlap))
            continue

        add = len(sentence) if not cur else len(sentence) + 1
        if cur and (cur_len + add) > max_len:
            flush()
            cur = [sentence]
            cur_len = len(sentence)
        else:
            cur.append(sentence)
            cur_len += add

    flush()
    return chunks


def _apply_overlap(chunks: list[str], *, max_len: int, overlap: int) -> list[str]:
    ov = max(0, int(overlap))
    if ov <= 0 or len(chunks) <= 1:
        return chunks

    out: list[str] = []
    prev_tail = ""
    for i, chunk in enumerate(chunks):
        if i == 0:
            out.append(chunk)
        else:
            if prev_tail and len(prev_tail) + 1 + len(chunk) <= max_len:
                out.append((prev_tail + " " + chunk).strip())
            else:
                out.append(chunk)

        tail = chunk
        if len(tail) > ov:
            tail = tail[-ov:]
            tail = tail.split(maxsplit=1)[-1] if " " in tail else tail
        prev_tail = tail.strip()

    return out


def _append_section(
    sections: list[tuple[str | None, str | None, list[str]]],
    major: str | None,
    sub: str | None,
    paras: list[str],
) -> None:
    if major is not None or sub is not None or paras:
        sections.append((major, sub, paras))


def _build_sections_from_blocks(
    blocks: list[str],
    *,
    max_title_length: int,
    max_title_words: int,
) -> list[tuple[str | None, str | None, list[str]]]:
    sections: list[tuple[str | None, str | None, list[str]]] = []
    current_major: str | None = None
    current_sub: str | None = None
    current_paras: list[str] = []

    for block in blocks:
        inline = _extract_inline_label(block)
        if inline:
            _append_section(sections, current_major, current_sub, current_paras)
            label, rest = inline
            current_major = label
            current_sub = None
            current_paras = [rest] if rest else []
            continue

        if _is_section_title(block, max_len=max_title_length, max_words=max_title_words):
            _append_section(sections, current_major, current_sub, current_paras)
            title = block.strip()
            if _is_major_title(title):
                current_major = title
                current_sub = None
            else:
                if current_major is None:
                    current_major = title
                    current_sub = None
                else:
                    current_sub = title
            current_paras = []
            continue

        current_paras.append(block)

    _append_section(sections, current_major, current_sub, current_paras)
    return sections


def _split_long_paragraph(text: str, *, max_len: int, overlap: int) -> list[str]:
    """
    Prefer splitting on sentence boundaries (and then words) before falling back to raw
    char slicing.

    This avoids chunks that start mid-sentence, common in PMC full text where paragraph-length
    sentences appear due to figure captions and inline citations.
    """

    t = (text or "").strip()
    if not t:
        return []
    if max_len <= 0 or len(t) <= max_len:
        return [t]

    # Sentence-based packing.
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(t) if s.strip()]
    if len(sentences) > 1:
        chunks = _pack_sentences(sentences, max_len=max_len, overlap=overlap)
        return _apply_overlap(chunks, max_len=max_len, overlap=overlap)

    # Fall back to simple char slicing.
    pieces = _split_long_text(t, max_len=max_len, overlap=overlap)
    return [piece.strip() for piece in pieces if piece.strip()]


@dataclass(frozen=True, slots=True)
class SectionAwareChunker(Chunker):
    chunk_size: int = 1200
    overlap: int = 150
    max_title_length: int = 120
    max_title_words: int = 12
    exclude_titles: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "exclude_titles", _load_exclude_titles(self.exclude_titles))

    def _is_excluded(self, title: str) -> bool:
        normalized = _normalize_title(title)
        return any(term in normalized for term in self.exclude_titles)

    def _is_excluded_section(self, section: str | None, subsection: str | None) -> bool:
        return (section and self._is_excluded(section)) or (
            subsection and self._is_excluded(subsection)
        )

    def _chunk_section(self, *, paragraphs: list[str]) -> list[str]:
        if not paragraphs:
            return []

        max_body = max(1, int(self.chunk_size))

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for paragraph in paragraphs:
            if not paragraph:
                continue

            if len(paragraph) > max_body:
                if current:
                    chunks.append("\n\n".join(current))
                    current = []
                    current_len = 0
                for piece in _split_long_paragraph(
                    paragraph, max_len=max_body, overlap=int(self.overlap)
                ):
                    if piece:
                        chunks.append(piece)
                continue

            add_len = len(paragraph) if not current else len(paragraph) + 2
            if current and (current_len + add_len) > max_body:
                chunks.append("\n\n".join(current))
                current = [paragraph]
                current_len = len(paragraph)
            else:
                current.append(paragraph)
                current_len += add_len

        if current:
            chunks.append("\n\n".join(current))

        return chunks

    def chunk(self, *, text: str) -> list[TextChunk]:
        log.debug(
            "Section-aware chunking request received, raw_text_length=%s, chunk_size=%s",
            len(text) if text else 0,
            self.chunk_size,
        )

        t = (text or "").strip()
        if not t:
            log.info("Empty or whitespace only text received, returning no chunks")
            return []

        blocks = _split_paragraphs(t)
        if not blocks:
            return []

        sections = _build_sections_from_blocks(
            blocks,
            max_title_length=self.max_title_length,
            max_title_words=self.max_title_words,
        )

        out: list[TextChunk] = []
        idx = 0
        cursor = 0

        for section, subsection, paras in sections:
            if self._is_excluded_section(section, subsection):
                log.debug(
                    "Skipping excluded section, section=%s, subsection=%s", section, subsection
                )
                continue

            for chunk_text in self._chunk_section(paragraphs=paras):
                out.append(
                    TextChunk(
                        index=idx,
                        text=chunk_text,
                        start=cursor,
                        end=cursor + len(chunk_text),
                        section=section,
                        subsection=subsection,
                    )
                )
                cursor += len(chunk_text)
                idx += 1

        log.info(
            "Section-aware chunking completed, total_chunks=%s, text_length=%s",
            len(out),
            len(t),
        )

        return out
