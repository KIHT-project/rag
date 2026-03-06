import math
import pandas as pd
from config import logger, OPTIONAL_BOOL_COLS, REQUIRED_COLS

def to_bool(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return False
    return str(v).strip().lower() in {"x", "yes", "true", "1", "y"}

def load_dataframe(path: str, sheet: str):
    logger.info("Reading Excel file")
    try:
        df = pd.read_excel(path, sheet_name=sheet, header=0)
    except Exception as e:
        raise RuntimeError(f"Error reading Excel: {e}") from e

    if df.empty:
        raise RuntimeError("Excel sheet is empty")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    for c in OPTIONAL_BOOL_COLS:
        if c in df.columns:
            df[c] = df[c].apply(to_bool)

    logger.debug("Dataframe shape after load: %s", df.shape)
    return df
