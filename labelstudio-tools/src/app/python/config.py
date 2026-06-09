import os
import logging
from dotenv import load_dotenv, find_dotenv

ENV_PATH = find_dotenv(usecwd=True)
load_dotenv(ENV_PATH)
PROJECT_ROOT = os.path.dirname(ENV_PATH) if ENV_PATH else os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("excel-to-ls")

def env_bool(name: str, default: str = "false") -> bool:
    val = (os.getenv(name, default) or "").strip().lower()
    return val in {"1", "true", "yes", "y", "on", "x"}

def env_int_or_none(name: str, default: str = "10"):
    raw = (os.getenv(name, default) or "").strip().lower()
    if raw in {"", "none", "null", "0"}:
        return None
    try:
        v = int(raw)
        return v if v > 0 else None
    except ValueError:
        return None

def _csv_env(name: str, default_list):
    raw = (os.getenv(name, "") or "").strip()
    if not raw:
        return default_list
    return [x.strip() for x in raw.split(",") if x.strip()]

LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL", "https://label.app.thrombus.eu/").rstrip("/")
API_KEY = (os.getenv("LABEL_STUDIO_API_KEY") or "").strip()
PROJECT_ID = int(os.getenv("PROJECT_ID", "3"))

EXCEL_FILE = os.getenv(
    "EXCEL_FILE",
    os.path.join(PROJECT_ROOT, "src", "app", "resources", "T5.3_RiskFactor_Results_2025_09_10_v02.xlsx"),
)
SHEET_NAME = os.getenv("SHEET_NAME", "After Automatic Exclusion")

FILTER_EXCLUDED = env_bool("FILTER_EXCLUDED", "false")
FILTER_NOT_RELATED_TO_VTE = env_bool("FILTER_NOT_RELATED_TO_VTE", "false")
FILTER_REQUIRE_RISK_FACTORS = env_bool("FILTER_REQUIRE_RISK_FACTORS", "false")

ROWS = env_int_or_none("ROWS", "10")
SAMPLE_AFTER_FILTER = env_bool("SAMPLE_AFTER_FILTER", "true")

IMPORT_RETRIES = int(os.getenv("IMPORT_RETRIES", "3"))
IMPORT_BACKOFF = float(os.getenv("IMPORT_BACKOFF", "1.5"))

REQUIRED_COLS = _csv_env("REQUIRED_COLS_CSV", ["PMID", "ArticleTitle", "Abstract"])
OPTIONAL_BOOL_COLS = _csv_env("OPTIONAL_BOOL_COLS_CSV", [
    "Excluded",
    "Not related to venous thrombosis",
    "Reporting on risk factors",
    "Narrative Review",
    "Systematic Review",
    "Meta Analysis",
])
CATEGORY_COLS = _csv_env("CATEGORY_COLS_CSV", [
    "Category 1", "Category 2", "Category 3", "Category 4", "Category 5",
])

logging.getLogger("excel-to-ls").info("Loaded .env from: %s", ENV_PATH or "<none>")
