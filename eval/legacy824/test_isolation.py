"""
Tests for eval/legacy824/isolation.py (Task 39, scenario 11): isolation
configuration refuses port 6333 and any endpoint matching current
Qdrant configuration. Pure, dependency-free -- no network call.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.legacy824.isolation import (  # noqa: E402
    UnsafeQdrantEndpointError,
    validate_isolated_endpoint,
)


def test_port_6333_refused_regardless_of_host():
    with pytest.raises(UnsafeQdrantEndpointError):
        validate_isolated_endpoint("some-other-host", 6333)
    with pytest.raises(UnsafeQdrantEndpointError):
        validate_isolated_endpoint("localhost", 6333)


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "LOCALHOST", " localhost "])
def test_known_current_endpoint_refused(host):
    with pytest.raises(UnsafeQdrantEndpointError):
        validate_isolated_endpoint(host, 6333)


def test_distinct_isolated_endpoint_accepted():
    validate_isolated_endpoint("localhost", 17333)  # must not raise
