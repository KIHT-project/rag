from __future__ import annotations

from dataclasses import dataclass

from biomed_platform.common.logging import get_logger
from biomed_platform.core.domains.ingestion import TextChunk
from biomed_platform.core.ports.ingestion import Chunker

log = get_logger(__name__)


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
