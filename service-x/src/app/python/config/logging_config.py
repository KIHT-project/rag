import os
import logging
import logging.config
import contextvars
from functools import wraps
from uuid import uuid4
from contextlib import contextmanager
from typing import Callable, Generator
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
from fastapi import Response
from dotenv import load_dotenv

mdc_context: contextvars.ContextVar[dict] = contextvars.ContextVar("mdc_context", default={})

def set_mdc(**kwargs):
    ctx = mdc_context.get().copy()
    ctx.update(kwargs)
    mdc_context.set(ctx)

def clear_mdc():
    mdc_context.set({})

def get_mdc_value(key: str, default=None):
    return mdc_context.get().get(key, default)

@contextmanager
def mdc_context_scope(**mdc_values) -> Generator:
    set_mdc(**mdc_values)
    try:
        yield
    finally:
        clear_mdc()

def with_mdc_context(**mdc_values):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with mdc_context_scope(**mdc_values):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def apply_mdc_from_request(request) -> str:
    business_id = request.headers.get("X-Business-ID", str(uuid4()))
    user_id = request.headers.get("X-User-ID")
    operation = request.url.path

    set_mdc(
        business_id=business_id,
        user_id=user_id,
        service=os.getenv("SERVICE_NAME", "service-x"),
        operation=operation,
    )

    return business_id  # for response headers, if needed


class MDCMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        try:
            business_id = apply_mdc_from_request(request)
            response = await call_next(request)
            response.headers["X-Business-ID"] = business_id
            return response
        finally:
            clear_mdc()


class MDCLogRecordFactory:
    def __init__(self, base_factory):
        self.base_factory = base_factory

    def __call__(self, *args, **kwargs):
        record = self.base_factory(*args, **kwargs)
        mdc = mdc_context.get()
        for key, value in mdc.items():
            setattr(record, key, value)
        return record

def setup_logging():
    load_dotenv()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    logging.setLogRecordFactory(MDCLogRecordFactory(logging.getLogRecordFactory()))

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": (
                    "%(asctime)s %(levelname)s %(name)s %(module)s %(funcName)s %(lineno)d "
                    "%(message)s %(business_id)s %(service)s %(operation)s %(user_id)s"
                )
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "json",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "level": log_level,
            "handlers": ["console"],
        },
    }

    logging.config.dictConfig(config)
