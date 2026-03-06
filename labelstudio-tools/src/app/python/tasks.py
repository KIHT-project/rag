import pandas as pd
from typing import List, Dict, Any
from config import logger

def validate_row(row: pd.Series):
    if pd.isna(row["PMID"]) or str(row["PMID"]).strip() == "":
        return "PMID missing"
    return None

def transform_to_tasks(df) -> List[Dict[str, Any]]:
    logger.info("Transforming rows to Label Studio tasks")
    tasks = []
    skipped = 0
    for idx, r in df.iterrows():
        problem = validate_row(r)
        if problem:
            skipped += 1
            logger.warning("Skipping row %s due to %s", idx, problem)
            continue

        pmid = str(r["PMID"]).strip()
        title = "" if pd.isna(r["ArticleTitle"]) else str(r["ArticleTitle"]).strip()
        abstract = "" if pd.isna(r["Abstract"]) else str(r["Abstract"]).strip()

        data = {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
        }
        tasks.append({"data": data})
    logger.info("Prepared %d tasks, skipped %d invalid rows", len(tasks), skipped)
    if not tasks:
        raise RuntimeError("No valid tasks to import")
    return tasks
