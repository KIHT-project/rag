from __future__ import annotations

import os
import sys
from typing import Iterable

from qdrant_client import QdrantClient


def _sanitize_model_id(model_id: str) -> str:
    return model_id.strip().replace("/", "_").replace(" ", "_")


def _normalize_prefix(prefix: str) -> str:
    p = (prefix or "").strip()
    if not p:
        p = "docs_"
    if not p.endswith("_"):
        p = f"{p}_"
    return p


def _iter_target_collections(
    all_names: Iterable[str],
    *,
    prefix: str,
    embedding_model_id: str | None,
) -> list[str]:
    names = sorted(set(n for n in all_names if isinstance(n, str) and n))

    if embedding_model_id:
        exact = f"{prefix}{_sanitize_model_id(embedding_model_id)}"
        if exact in names:
            return [exact]

    return [n for n in names if n.startswith(prefix)]


def main() -> int:
    url = os.getenv("QDRANT_URL", "http://localhost:6335").rstrip("/")

    prefix = _normalize_prefix(os.getenv("QDRANT_COLLECTION_PREFIX", "docs_"))

    embedding_model_id = (
        os.getenv("EMBEDDING_MODEL_ID")
        or os.getenv("QDRANT_EMBEDDING_MODEL_ID")
        or None
    )

    try:
        client = QdrantClient(url=url, timeout=5.0)

        collections = client.get_collections().collections
        all_names = [c.name for c in collections]

        targets = _iter_target_collections(
            all_names,
            prefix=prefix,
            embedding_model_id=embedding_model_id,
        )

        for name in targets:
            try:
                client.delete_collection(collection_name=name)
            except Exception as exc:
                print(
                    f"[clear_qdrant] failed deleting collection={name}, error={exc}",
                    file=sys.stderr,
                )

        return 0

    except Exception as e:
        print(f"[clear_qdrant] skipping cleanup: {e}", file=sys.stderr)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
