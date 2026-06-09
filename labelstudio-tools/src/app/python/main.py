from config import (
    PROJECT_ROOT, EXCEL_FILE, SHEET_NAME,
    ROWS, SAMPLE_AFTER_FILTER, logger
)
from data_loader import load_dataframe
from filters import apply_filters
from tasks import transform_to_tasks
from uploader import upload_tasks

def main():
    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Excel path: %s", EXCEL_FILE)

    df = load_dataframe(EXCEL_FILE, SHEET_NAME)

    if SAMPLE_AFTER_FILTER:
        df = apply_filters(df)
        if ROWS:
            logger.info("Sampling first %d rows AFTER filters", ROWS)
            df = df.head(ROWS)
    else:
        if ROWS:
            logger.info("Sampling first %d rows BEFORE filters", ROWS)
            df = df.head(ROWS)
        df = apply_filters(df)

    tasks = transform_to_tasks(df)
    upload_tasks(tasks)

if __name__ == "__main__":
    try:
        main()
    except (ValueError, PermissionError, RuntimeError) as e:
        logger.error("Fatal error: %s", e)
        raise
    except Exception as e:
        logger.exception("Unexpected fatal error")
        raise
