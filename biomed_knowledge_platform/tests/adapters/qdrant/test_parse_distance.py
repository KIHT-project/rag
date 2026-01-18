from __future__ import annotations

import pytest

qdrant_client = pytest.importorskip("qdrant_client")

from biomed_platform.adapters.qdrant.vector_index import parse_distance
from biomed_platform.core.errors.errors import SystemError


def test_parse_distance_accepts_known_values_case_insensitive() -> None:
    # Given supported distance values
    # When parsing them
    d1 = parse_distance("cosine")
    d2 = parse_distance(" DOT ")
    d3 = parse_distance("euclid")

    # Then no error is raised
    assert d1 is not None
    assert d2 is not None
    assert d3 is not None


def test_parse_distance_raises_system_error_on_invalid_value() -> None:
    # Given an invalid distance
    # When parsing it
    # Then a SystemError is raised
    with pytest.raises(SystemError) as exc:
        parse_distance("bad")

    assert exc.value.code == "invalid_qdrant_distance"
