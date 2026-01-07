from __future__ import annotations

from fastapi import FastAPI

from biomed_platform.api.endpoints.system import router as system_router
from biomed_platform.common.middleware.request_context import RequestContextMiddleware, AccessLogMiddleware
from biomed_platform.api.router import router as v1_router
from biomed_platform.common.logging import configure_logging, get_logger
from biomed_platform.common.settings import load_settings


def create_app() -> FastAPI:
    settings = load_settings()
    configure_logging(settings.logging_path)

    log = get_logger(__name__)
    api_cfg = settings.require_api()

    app = FastAPI(
        title=api_cfg.get("title"),
        description=api_cfg.get("description"),
        version=api_cfg.get("version"),
    )

    app.state.settings = settings
    app.include_router(system_router)
    app.include_router(v1_router)

    log.info(
        "API %s | version=%s | Description: %s",
        api_cfg.get("title"),
        api_cfg.get("version"),
        api_cfg.get("description"),
    )

    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(RequestContextMiddleware)

    return app


app = create_app()
