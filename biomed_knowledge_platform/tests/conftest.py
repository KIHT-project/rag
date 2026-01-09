from __future__ import annotations

import pytest

from biomed_platform.common.middleware.trace import request_id_ctx


@pytest.fixture(autouse=True)
def clear_request_id_ctx() -> None:
    token = request_id_ctx.set(None)
    try:
        yield
    finally:
        request_id_ctx.reset(token)
