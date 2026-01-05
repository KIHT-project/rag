from __future__ import annotations

from fastapi import FastAPI

from biomed_platform.core.logging import configure_logging, get_logger
from biomed_platform.core.settings import load_settings


def create_app() -> FastAPI:
    settings = load_settings()
    configure_logging(settings.logging_path)

    log = get_logger(__name__)

    api_cfg = settings.require("api")

    app = FastAPI(
        title=api_cfg.get("title", "Biomedical Knowledge Platform"),
        description=api_cfg.get("description"),
        version=api_cfg.get("version"),
    )

    app.state.settings = settings
    log.info(
        "API %s | version=%s | Description: %s",
        api_cfg.get("title"),
        api_cfg.get("version"),
        api_cfg.get("description"),
    )

    return app


app = create_app()
