"""
eval/legacy824/isolation.py

Task 39: refuses to let the historical VectorStore ever connect to the
current project's Qdrant instance or any existing evidence collection.
This is a pure, dependency-free validation module -- it makes no
network call itself; it only rejects an endpoint configuration before
anything else is allowed to use it.
"""

from __future__ import annotations

# The current project's production/default Qdrant port. Every official
# and legacy collection (including the current project's own
# "isco_occupations" -- confirmed present with a real point count as of
# Task 39's pre-work snapshot) lives behind this port. The historical
# VectorStore defaults to this exact port when QDRANT_PORT is unset, so
# refusing it here is the primary safeguard against an accidental
# production connection.
FORBIDDEN_PORT = 6333

# Host/port pairs already confirmed to be the current project's live
# Qdrant endpoint (see the Task 39 final report's pre-work point-count
# snapshot). Any of these is refused outright, regardless of the
# configured port value, in case a future default changes.
FORBIDDEN_ENDPOINTS = {
    ("localhost", 6333),
    ("127.0.0.1", 6333),
}


class UnsafeQdrantEndpointError(ValueError):
    """Raised when an isolation-required caller is configured to point
    at the current project's live Qdrant endpoint."""


def validate_isolated_endpoint(host: str, port: int) -> None:
    """
    Raises UnsafeQdrantEndpointError if (host, port) matches the
    forbidden port or a forbidden host/port pair. Callers that need an
    isolated Qdrant endpoint must call this BEFORE constructing any
    client or setting QDRANT_HOST/QDRANT_PORT for the historical code.
    """
    if port == FORBIDDEN_PORT:
        raise UnsafeQdrantEndpointError(
            f"port {port} is the current project's production Qdrant port; "
            "Task 39 requires a distinct, isolated port."
        )
    normalized_host = (host or "").strip().lower()
    for forbidden_host, forbidden_port in FORBIDDEN_ENDPOINTS:
        if normalized_host == forbidden_host and port == forbidden_port:
            raise UnsafeQdrantEndpointError(
                f"{host}:{port} matches a known current-project Qdrant endpoint; "
                "Task 39 requires a distinct, isolated endpoint."
            )
