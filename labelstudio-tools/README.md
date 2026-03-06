# Excel to Label Studio Import

Import rows from an Excel file into a Label Studio project using the Label Studio SDK.

## Prerequisites
- Python 3.11.9
- Access to a running Label Studio instance
- Your Label Studio API key

## Installation
Set up a virtual environment and install the required dependencies:

```shell
pyenv install 3.11.9
pyenv local 3.11.9
app --version
app -m venv .venv

# macos:
source .venv/bin/activate

# windows:
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\activate.bat

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Environment Variables

All configuration is handled through a `.env` file at the project root. Below are the supported variables:

| Variable                       | Default                                                                                                                | Description                                                                                                                   |
|--------------------------------|------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| `LABEL_STUDIO_URL`             | `https://label.app.thrombus.eu/`                                                                                       | Base URL of your Label Studio instance (must include scheme, e.g. `https://`).                                                |
| `LABEL_STUDIO_API_KEY`         | *(none)*                                                                                                               | Personal access token for Label Studio. Required for authentication.                                                          |
| `PROJECT_ID`                   | `3`                                                                                                                    | Numeric ID of the Label Studio project where tasks will be imported.                                                          |
| `EXCEL_FILE`                   | `src/app/resources/T5.3_RiskFactor_Results_2025_09_10_v02.xlsx`                                                        | Path to the Excel file with PubMed data. Can be absolute or relative.                                                         |
| `SHEET_NAME`                   | `After Automatic Exclusion`                                                                                            | Excel sheet name to read.                                                                                                     |
| `FILTER_EXCLUDED`              | `false`                                                                                                                | If `true`, drop rows where the `Excluded` column is set (e.g. `X`, `Yes`, `1`).                                               |
| `FILTER_NOT_RELATED_TO_VTE`    | `false`                                                                                                                | If `true`, drop rows marked as not related to venous thrombosis.                                                              |
| `FILTER_REQUIRE_RISK_FACTORS`  | `false`                                                                                                                | If `true`, keep only rows reporting on risk factors.                                                                          |
| `ROWS`                         | `0`                                                                                                                    | Limit the number of rows imported for testing. Use `0`, `none`, or unset to import all rows.                                  |
| `SAMPLE_AFTER_FILTER`          | `true`                                                                                                                 | If `true`, apply filters first and then sample rows. Guarantees valid tasks.                                                  |
| `IMPORT_RETRIES`               | `3`                                                                                                                    | How many times to retry a failed batch import before giving up.                                                               |
| `IMPORT_BACKOFF`               | `1.5`                                                                                                                  | Backoff factor for retries. Delay increases exponentially.                                                                    |
| `LOG_LEVEL`                    | `INFO`                                                                                                                 | Logging verbosity. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`.                                                              |
| `REQUIRED_COLS_CSV`            | `PMID,ArticleTitle,Abstract`                                                                                           | Override for required Excel column headers.                                                                                   |
| `OPTIONAL_BOOL_COLS_CSV`       | `Excluded,Not related to venous thrombosis,Reporting on risk factors,Narrative Review,Systematic Review,Meta Analysis` | List of boolean-like columns to normalize.                                                                                    |
| `CATEGORY_COLS_CSV`            | `Category 1,Category 2,Category 3,Category 4,Category 5`                                                               | List of category columns to carry into task JSON.                                                                             |
| `LABEL_STUDIO_EXPORT_DIR`      | `ls_data`                                                                                                              | Directory where exported and enriched Label Studio task JSON files are written.                                               |
| `LABEL_STUDIO_EXPORT_FILENAME` | `tasks.json`                                                                                                           | Filename for the enriched Label Studio task export.                                                                           |
| `LS_FAIL_ON_MISSING_DOI`       | `False`                                                                                                                | If `true`, abort when any PMID cannot be resolved to a DOI. If `false`, unresolved tasks are excluded from the clean dataset. |
| `RAG_API_BASE_URL`             | `http://localhost:8000`                                                                                                | Base URL of the RAG ingestion API.                                                                                            |
| `RAG_TASKS_CLEAN_JSON_PATH`    | `ls_data/tasks_clean.json`                                                                                             | Path to the clean task dataset used for RAG ingestion. Only tasks with complete metadata are included.                        |
| `RAG_BATCH_SIZE`               | `25`                                                                                                                   | Number of documents per ingestion request.                                                                                    |
| `RAG_TIMEOUT`                  | `60`                                                                                                                   | Timeout in seconds for RAG ingestion requests.                                                                                |
| `RAG_MAX_RETRIES`              | `12`                                                                                                                   | Maximum retries for RAG ingestion on 429 or 5xx responses.                                                                    |
| `RAG_BACKOFF_BASE_S`           | `1.0`                                                                                                                  | Base backoff duration in seconds for RAG ingestion retries.                                                                   |
| `RAG_BACKOFF_MAX_S`            | `60.0`                                                                                                                 | Maximum backoff duration in seconds.                                                                                          |
| `RAG_POST_SUCCESS_SLEEP_S`     | `2.0`                                                                                                                  | Sleep duration after each successful ingestion batch to avoid rate limiting.                                                  |
