from __future__ import annotations

import os
import sys
from qdrant_client import QdrantClient


def main() -> int:
    url = os.getenv("QDRANT_URL", "http://localhost:6335").rstrip("/")
    collection = os.getenv("QDRANT_COLLECTION", "biomed_docs")

    try:
        client = QdrantClient(url=url, timeout=5)
        collections = {c.name for c in client.get_collections().collections}
        if collection in collections:
            client.delete_collection(collection_name=collection)
        return 0
    except Exception as e:
        print(f"[clear_qdrant] skipping cleanup: {e}", file=sys.stderr)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
