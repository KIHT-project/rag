from __future__ import annotations

import pytest

from biomed_platform.core.services.ingestion.chunking import (
    SectionAwareChunker,
    SimpleCharChunker,
    _split_long_text,
)

DEFAULT_EXCLUDES = (
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


@pytest.fixture(autouse=True)
def _clear_section_excludes(monkeypatch):
    monkeypatch.delenv("RAG_SECTION_EXCLUDE_TITLES", raising=False)


def test_chunker_returns_no_chunks_for_empty_text() -> None:
    # Given a chunker
    chunker = SectionAwareChunker(chunk_size=5, overlap=1, exclude_titles=DEFAULT_EXCLUDES)

    # When text is empty
    chunks = chunker.chunk(text="  \n  ")

    # Then no chunks are returned
    assert chunks == []


def test_chunker_creates_overlapping_chunks_and_adjusts_overlap() -> None:
    # Given a chunker where overlap is greater than chunk size
    chunker = SimpleCharChunker(chunk_size=4, overlap=10)

    # When chunking a short text
    chunks = chunker.chunk(text="abcdefgh")

    # Then chunks cover the full text and indices are sequential
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].text
    assert chunks[-1].end == len("abcdefgh")
    assert "".join([c.text for c in chunks]).replace(" ", "") != ""


def test_section_chunker_includes_titles_and_splits_sections() -> None:
    text = "Introduction\n\nIntro paragraph.\n\nMethods\n\nMethod paragraph."
    chunker = SectionAwareChunker(chunk_size=200, overlap=0, exclude_titles=DEFAULT_EXCLUDES)

    chunks = chunker.chunk(text=text)

    assert len(chunks) == 2
    assert chunks[0].section == "Introduction"
    assert chunks[0].text.startswith("Intro paragraph.")
    assert chunks[1].section == "Methods"
    assert "Method paragraph." in chunks[1].text


def test_section_chunker_excludes_references_section(monkeypatch) -> None:
    text = "Introduction\n\nIntro paragraph.\n\nReferences\n\nRef1\n\nRef2"
    chunker = SectionAwareChunker(chunk_size=200, overlap=0, exclude_titles=DEFAULT_EXCLUDES)

    chunks = chunker.chunk(text=text)

    assert len(chunks) == 1
    assert chunks[0].section == "Introduction"
    assert "References" not in chunks[0].text
    assert "Ref1" not in chunks[0].text

    monkeypatch.setenv("RAG_SECTION_EXCLUDE_TITLES", "methods")
    chunker2 = SectionAwareChunker(chunk_size=200, overlap=0)
    chunks2 = chunker2.chunk(text="Methods\n\nHidden.\n\nResults\n\nShown.")
    assert len(chunks2) == 1
    assert chunks2[0].section == "Results"
    assert chunks2[0].text.startswith("Shown.")


def test_section_chunker_splits_long_section_with_title_prefix() -> None:
    text = "Results\n\n" + ("A" * 120) + "\n\n" + ("B" * 120)
    chunker = SectionAwareChunker(chunk_size=80, overlap=0, exclude_titles=DEFAULT_EXCLUDES)

    chunks = chunker.chunk(text=text)

    assert len(chunks) >= 2
    assert all(chunk.section == "Results" for chunk in chunks)


def test_section_chunker_uses_label_lines_when_no_blank_lines() -> None:
    text = "BACKGROUND: A.\nMETHODS: B.\nRESULTS: C."
    chunker = SectionAwareChunker(chunk_size=500, overlap=0, exclude_titles=DEFAULT_EXCLUDES)

    chunks = chunker.chunk(text=text)

    assert len(chunks) == 3
    assert [c.section for c in chunks] == ["Background", "Methods", "Results"]
    assert chunks[0].text == "A."
    assert chunks[1].text == "B."
    assert chunks[2].text == "C."


def test_section_chunker_skips_title_only_section() -> None:
    chunker = SectionAwareChunker(chunk_size=50, overlap=0, exclude_titles=DEFAULT_EXCLUDES)
    chunks = chunker.chunk(text="Introduction")
    assert chunks == []


def test_section_chunker_treats_lowercase_sentence_as_paragraph() -> None:
    chunker = SectionAwareChunker(chunk_size=50, overlap=0, exclude_titles=DEFAULT_EXCLUDES)
    chunks = chunker.chunk(text="shared token alpha")
    assert len(chunks) == 1
    assert chunks[0].text == "shared token alpha"


def test_split_long_text_handles_overlap_adjustment() -> None:
    pieces = _split_long_text("abcdefghij", max_len=3, overlap=10)
    assert pieces[0] == "abc"
    assert "".join(pieces) == "abcdefghij"


def test_split_long_text_handles_non_positive_max_len() -> None:
    pieces = _split_long_text("abc", max_len=0, overlap=1)
    assert pieces == ["abc"]
