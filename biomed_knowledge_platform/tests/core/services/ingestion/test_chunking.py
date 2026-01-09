# tests/core/services/ingestion/test_simple_char_chunker.py
from __future__ import annotations

from typing import Any

import pytest

from biomed_platform.core.services.ingestion.chunking import SimpleCharChunker
from biomed_platform.core.services.ingestion_ports import TextChunk


class TestSimpleCharChunker:
    def test_given_none_text_when_chunk_then_returns_empty_list(self) -> None:
        # Given
        chunker = SimpleCharChunker(chunk_size=10, overlap=2)

        # When
        got = chunker.chunk(text=None)  # type: ignore[arg-type]

        # Then
        assert got == []

    def test_given_empty_string_when_chunk_then_returns_empty_list(self) -> None:
        # Given
        chunker = SimpleCharChunker(chunk_size=10, overlap=2)

        # When
        got = chunker.chunk(text="")

        # Then
        assert got == []

    def test_given_whitespace_only_when_chunk_then_returns_empty_list(self) -> None:
        # Given
        chunker = SimpleCharChunker(chunk_size=10, overlap=2)

        # When
        got = chunker.chunk(text="   \n\t  ")

        # Then
        assert got == []

    def test_given_short_text_when_chunk_then_returns_single_chunk_with_trimmed_text(self) -> None:
        # Given
        chunker = SimpleCharChunker(chunk_size=100, overlap=10)

        # When
        got = chunker.chunk(text="  hello world  ")

        # Then
        assert got == [
            TextChunk(index=0, text="hello world", start=0, end=len("hello world"))
        ]

    def test_given_exact_chunk_size_text_when_chunk_then_returns_single_full_chunk(self) -> None:
        # Given
        chunker = SimpleCharChunker(chunk_size=5, overlap=2)
        text = "abcde"

        # When
        got = chunker.chunk(text=text)

        # Then
        assert got == [TextChunk(index=0, text="abcde", start=0, end=5)]

    def test_given_text_longer_than_chunk_size_when_chunk_then_splits_into_multiple_chunks_with_overlap(self) -> None:
        # Given
        chunker = SimpleCharChunker(chunk_size=5, overlap=2)
        text = "abcdefghij"  # len 10

        # When
        got = chunker.chunk(text=text)

        # Then
        assert got == [
            TextChunk(index=0, text="abcde", start=0, end=5),
            TextChunk(index=1, text="defgh", start=3, end=8),
            TextChunk(index=2, text="ghij", start=6, end=10),
        ]

    def test_given_zero_overlap_when_chunk_then_chunks_do_not_overlap(self) -> None:
        # Given
        chunker = SimpleCharChunker(chunk_size=4, overlap=0)
        text = "abcdefghij"  # len 10

        # When
        got = chunker.chunk(text=text)

        # Then
        assert got == [
            TextChunk(index=0, text="abcd", start=0, end=4),
            TextChunk(index=1, text="efgh", start=4, end=8),
            TextChunk(index=2, text="ij", start=8, end=10),
        ]

    def test_given_negative_overlap_when_chunk_then_overlap_is_clamped_to_zero(self) -> None:
        # Given
        chunker = SimpleCharChunker(chunk_size=4, overlap=-123)
        text = "abcdefghij"  # len 10

        # When
        got = chunker.chunk(text=text)

        # Then
        assert got == [
            TextChunk(index=0, text="abcd", start=0, end=4),
            TextChunk(index=1, text="efgh", start=4, end=8),
            TextChunk(index=2, text="ij", start=8, end=10),
        ]

    def test_given_non_positive_chunk_size_when_chunk_then_chunk_size_is_clamped_to_one(self) -> None:
        # Given
        chunker = SimpleCharChunker(chunk_size=0, overlap=10)
        text = "abc"

        # When
        got = chunker.chunk(text=text)

        # Then
        assert got == [
            TextChunk(index=0, text="a", start=0, end=1),
            TextChunk(index=1, text="b", start=1, end=2),
            TextChunk(index=2, text="c", start=2, end=3),
        ]

    def test_given_overlap_greater_or_equal_to_chunk_size_when_chunk_then_overlap_is_adjusted(self) -> None:
        # Given
        chunker = SimpleCharChunker(chunk_size=8, overlap=999)
        text = "abcdefghijklmnop"  # len 16

        # When
        got = chunker.chunk(text=text)

        # Then
        # overlap adjusted to size // 4 => 2
        assert got == [
            TextChunk(index=0, text="abcdefgh", start=0, end=8),
            TextChunk(index=1, text="ghijklmn", start=6, end=14),
            TextChunk(index=2, text="mnop", start=12, end=16),
        ]

    def test_given_long_text_when_chunk_then_indexes_are_sequential_and_ranges_are_consistent(self) -> None:
        # Given
        chunker = SimpleCharChunker(chunk_size=7, overlap=3)
        text = "abcdefghijklmnopqrstuvwxyz"  # len 26

        # When
        got = chunker.chunk(text=text)

        # Then
        assert got
        assert [c.index for c in got] == list(range(len(got)))

        for c in got:
            assert 0 <= c.start <= c.end <= len(text)
            assert c.text == text[c.start : c.end]
            assert len(c.text) == c.end - c.start

        assert got[-1].end == len(text)

    def test_given_text_with_leading_and_trailing_whitespace_when_chunk_then_chunk_offsets_match_trimmed_text(self) -> None:
        # Given
        chunker = SimpleCharChunker(chunk_size=3, overlap=1)
        raw = "   abcd   "
        trimmed = "abcd"

        # When
        got = chunker.chunk(text=raw)

        # Then
        assert got == [
            TextChunk(index=0, text="abc", start=0, end=3),
            TextChunk(index=1, text="cd", start=2, end=4),
        ]
        assert "".join([c.text for c in got]).replace("c", "c")  # no-op, sanity
        for c in got:
            assert c.text == trimmed[c.start : c.end]


class TestSimpleCharChunkerLogging:
    def test_given_overlap_ge_chunk_size_when_chunk_then_logs_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Given
        calls: dict[str, list[tuple[tuple[Any, ...], dict[str, Any]]]] = {"warning": []}

        class _Logger:
            def debug(self, *args: Any, **kwargs: Any) -> None:
                return

            def info(self, *args: Any, **kwargs: Any) -> None:
                return

            def warning(self, *args: Any, **kwargs: Any) -> None:
                calls["warning"].append((args, kwargs))

        # Patch module level logger used by SimpleCharChunker
        import biomed_platform.core.services.ingestion.chunking as chunking_mod

        monkeypatch.setattr(chunking_mod, "log", _Logger())

        chunker = SimpleCharChunker(chunk_size=8, overlap=8)

        # When
        _ = chunker.chunk(text="abcdefghijklmnop")

        # Then
        assert calls["warning"]
        msg_template = calls["warning"][0][0][0]
        assert "Overlap >= chunk_size, adjusting overlap" in msg_template
