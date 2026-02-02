from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def build_audit_sample(
    *,
    rag_no_hyde_jsonl: Path,
    rag_hyde_jsonl: Path,
    llm_only_jsonl: Path,
    out_jsonl: Path,
    sample_size: int,
    seed: int,
) -> Path:
    def load_map(p: Path) -> dict[str, dict[str, Any]]:
        m = {}
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                m[str(r["query_id"])] = r
        return m

    a = load_map(rag_no_hyde_jsonl)
    b = load_map(rag_hyde_jsonl)
    c = load_map(llm_only_jsonl)

    qids = sorted(set(a.keys()) & set(b.keys()) & set(c.keys()))
    rnd = random.Random(seed)
    rnd.shuffle(qids)
    pick = qids[: min(sample_size, len(qids))]

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as out:
        for qid in pick:
            out.write(
                json.dumps(
                    {
                        "query_id": qid,
                        "question": a[qid].get("question"),
                        "contexts": a[qid].get("contexts") or [],
                        "rag_no_hyde": a[qid],
                        "rag_hyde": b[qid],
                        "llm_only": c[qid],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return out_jsonl
