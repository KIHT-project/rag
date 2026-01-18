from __future__ import annotations

from biomed_platform.core.services.ingestion.chunking import SimpleCharChunker


def test_chunker_returns_no_chunks_for_empty_text() -> None:
    # Given a chunker
    chunker = SimpleCharChunker(chunk_size=5, overlap=1)

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
