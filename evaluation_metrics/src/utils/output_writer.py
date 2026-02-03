from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def write_outputs(df: pd.DataFrame, out_csv: str | Path) -> None:
    """
    Write a dataframe to csv plus json plus jsonl next to it.

    out_csv should be a path ending with .csv
    """
    out_csv_path = Path(out_csv)
    if out_csv_path.suffix.lower() != ".csv":
        raise ValueError(f"out_csv must end with .csv, got {out_csv_path}")

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # CSV
    df.to_csv(out_csv_path, index=False)

    # Replace NaN with None for JSON outputs
    df_clean = df.replace({np.nan: None})

    # JSON array
    out_json = out_csv_path.with_suffix(".json")
    records: list[dict[str, Any]] = df_clean.to_dict(orient="records")
    out_json.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    # JSONL
    out_jsonl = out_csv_path.with_suffix(".jsonl")
    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")
