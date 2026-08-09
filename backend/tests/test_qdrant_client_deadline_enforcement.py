"""
Task 34 — required hermetic proof of a genuine CLIENT-SIDE Qdrant
deadline, exercised through the real, installed qdrant-client REST
transport against a local, intentionally delayed HTTP server.

This file deliberately does NOT use a MagicMock/FakeQdrantClient in
place of the transport (unlike every other test file in this suite) --
the point is to prove that a client constructed with a specific
`timeout=` value genuinely bounds an in-flight request's wall time,
using only the real `qdrant_client.QdrantClient` + `httpx` stack. The
server is a plain stdlib `http.server.ThreadingHTTPServer` bound to a
DYNAMICALLY ALLOCATED localhost port (never 6333, never a live Qdrant
service) -- fully hermetic: no external network access, no real Qdrant
collection, no embedding model, no LLM/API call anywhere in this file.

Every server and client is closed in `finally`/fixture teardown so the
full suite cannot hang or leak a background thread/socket.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from backend.rag.hierarchical_store import _QdrantClientDeadlinePool
from backend.rag.hierarchy_engine import HierarchyBeamSearchEngine, StageConfig, _is_retryable_exception

TWO_STAGE = [
    StageConfig(name="root", collection="col_root", weight=0.4),
    StageConfig(name="leaf", collection="col_leaf", weight=0.6),
]

_SUCCESS_BODY = {
    "result": {
        "points": [
            {"id": 1, "version": 0, "score": 0.9, "payload": {"code": "2512", "label_en": "x", "label_ar": ""}},
        ]
    },
    "status": "ok",
    "time": 0.001,
}


class _ScriptedDelayHandler(BaseHTTPRequestHandler):
    """Each POST consumes the next scripted behaviour from the class-level
    `script` list (a queue of (delay_seconds, status_code) tuples); once
    exhausted, repeats the last entry. `calls` records one entry per
    request received, appended BEFORE the delay so a test can observe
    that the server-side handler was actually invoked."""

    script: list[tuple[float, int]] = [(0.0, 200)]
    calls: list[float] = []  # wall-clock time.monotonic() at request receipt

    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        pass  # silence default stderr request logging

    def do_POST(self):
        type(self).calls.append(time.monotonic())
        length = int(self.headers.get("Content-Length", 0))
        if length:
            self.rfile.read(length)
        idx = min(len(type(self).calls) - 1, len(type(self).script) - 1)
        delay_seconds, status_code = type(self).script[idx]
        if delay_seconds > 0:
            time.sleep(delay_seconds)
        if status_code != 200:
            body = json.dumps({"status": {"error": "simulated non-timeout failure"}}).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = json.dumps(_SUCCESS_BODY).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def delayed_server():
    """A real local HTTP server on a dynamically allocated port (never
    6333). Configure per-test behaviour via `server.handler_class.script`
    before issuing requests. Always shut down and closed in teardown."""
    handler_class = type(
        f"_ScriptedDelayHandler_{id(object())}", (_ScriptedDelayHandler,), {"script": [(0.0, 200)], "calls": []},
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_class)
    port = server.server_address[1]
    assert port != 6333
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, handler_class, port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _make_engine(port, query_timeout_seconds, max_query_attempts=1, retry_backoff_seconds=0.0, use_pool=True):
    primary_client = QdrantClient(
        host="127.0.0.1", port=port, timeout=query_timeout_seconds, check_compatibility=False,
    )
    client_for_deadline = None
    pool = None
    if use_pool:
        pool = _QdrantClientDeadlinePool(
            host="127.0.0.1", port=port,
            primary_client=primary_client, primary_deadline_seconds=int(query_timeout_seconds),
        )
        client_for_deadline = pool.get
    engine = HierarchyBeamSearchEngine(
        primary_client, TWO_STAGE, 0.70,
        max_query_attempts=max_query_attempts, retry_backoff_seconds=retry_backoff_seconds,
        query_timeout_seconds=query_timeout_seconds, client_for_deadline=client_for_deadline,
    )
    return engine, primary_client, pool


def _close(primary_client, pool):
    if pool is not None:
        pool.close_all()
    primary_client.close()


# ---------------------------------------------------------------------------
# 1/2. A delayed endpoint exceeding the configured deadline raises a
#      timeout-shaped exception within bounded wall time, through the
#      real production query path, correctly classified as retryable
# ---------------------------------------------------------------------------

def test_delayed_response_raises_within_bounded_wall_time(delayed_server):
    server, handler_class, port = delayed_server
    handler_class.script = [(3.0, 200)]  # server sleeps 3s; client deadline is far shorter
    engine, primary_client, pool = _make_engine(port, query_timeout_seconds=1)
    try:
        telemetry: dict = {}
        t0 = time.perf_counter()
        hits = engine._query(collection="col_root", query_vec=[0.0], limit=3, query_telemetry=telemetry)
        elapsed_s = time.perf_counter() - t0
        assert hits == []
        # Generous CI tolerance, but must clearly be bounded by the ~1s
        # client deadline, not the deliberately much longer 3s server delay.
        assert elapsed_s < 2.5, f"expected a bounded wait near 1s, observed {elapsed_s:.2f}s (server delay was 3s)"
        assert telemetry["outcome"] == "exception"  # max_query_attempts=1 -> no retry configured
        assert telemetry["retryable"] is True  # _is_retryable_exception() classified it correctly
        assert telemetry["exception_type"] == "ResponseHandlingException"
    finally:
        _close(primary_client, pool)


def test_raised_exception_classified_retryable_only_for_wrapped_httpx_timeout(delayed_server):
    """Direct proof of _is_retryable_exception() against the REAL
    exception raised by the real transport (requirement 2), not a
    synthetic stand-in."""
    server, handler_class, port = delayed_server
    handler_class.script = [(2.0, 200)]
    client = QdrantClient(host="127.0.0.1", port=port, timeout=1, check_compatibility=False)
    try:
        with pytest.raises(ResponseHandlingException) as exc_info:
            client.query_points(collection_name="col_root", query=[0.0], limit=3, with_payload=True)
        exc = exc_info.value
        assert isinstance(exc.source, httpx.TimeoutException)
        assert _is_retryable_exception(exc) is True
    finally:
        client.close()


# ---------------------------------------------------------------------------
# 3. With a shared stage budget, the attempt deadline is no longer than
#    the remaining stage time -- a delayed first attempt returns control
#    before the stage could silently overrun indefinitely
# ---------------------------------------------------------------------------

def test_stage_budget_caps_client_side_deadline_below_configured_default(delayed_server):
    server, handler_class, port = delayed_server
    handler_class.script = [(5.0, 200)]  # server sleeps 5s
    # Engine's OWN default query_timeout_seconds is generous (10s), but
    # the shared stage budget only leaves ~1s remaining -- the pool must
    # hand back a client whose OWN deadline is capped to that ~1s, not
    # the engine's 10s default.
    engine, primary_client, pool = _make_engine(port, query_timeout_seconds=10)
    try:
        stage_deadline = time.monotonic() + 1.0
        telemetry: dict = {}
        t0 = time.perf_counter()
        hits = engine._query(
            collection="col_root", query_vec=[0.0], limit=3,
            stage_deadline_monotonic=stage_deadline, query_telemetry=telemetry,
        )
        elapsed_s = time.perf_counter() - t0
        assert hits == []
        assert elapsed_s < 3.0, (
            f"expected the stage budget (~1s remaining) to bound this attempt, "
            f"not the engine's 10s default or the server's 5s delay; observed {elapsed_s:.2f}s"
        )
        assert telemetry["outcome"] == "exception"
        assert telemetry["retryable"] is True
    finally:
        _close(primary_client, pool)


# ---------------------------------------------------------------------------
# 4. A retryable timeout with sufficient remaining stage budget is
#    retried exactly as before, and succeeds genuinely on the retry
# ---------------------------------------------------------------------------

def test_retry_succeeds_when_stage_budget_remains(delayed_server):
    server, handler_class, port = delayed_server
    # First attempt: server sleeps past the 1s client deadline -> timeout.
    # Second attempt: server responds immediately -> genuine success.
    handler_class.script = [(2.0, 200), (0.0, 200)]
    engine, primary_client, pool = _make_engine(port, query_timeout_seconds=1, max_query_attempts=2, retry_backoff_seconds=0.05)
    try:
        stage_deadline = time.monotonic() + 30.0  # plenty of budget for a retry
        telemetry: dict = {}
        hits = engine._query(
            collection="col_root", query_vec=[0.0], limit=3,
            stage_deadline_monotonic=stage_deadline, query_telemetry=telemetry,
        )
        assert len(hits) == 1
        assert hits[0].payload["code"] == "2512"  # genuine, real, parsed result -- not fabricated
        assert telemetry["outcome"] == "success_after_retry"
        assert telemetry["attempts"] == 2
        assert len(handler_class.calls) == 2  # the server really was hit twice
    finally:
        _close(primary_client, pool)


# ---------------------------------------------------------------------------
# 5. When no stage budget remains after a timeout, no retry starts and
#    stage_budget_exhausted remains intact -- proven against the REAL
#    delayed server, not a fake
# ---------------------------------------------------------------------------

def test_no_retry_when_stage_budget_exhausted_after_real_timeout(delayed_server):
    server, handler_class, port = delayed_server
    handler_class.script = [(3.0, 200), (0.0, 200)]  # 2nd entry would succeed IF ever reached
    # The engine's OWN default query_timeout_seconds (5s) is deliberately
    # generous; the STAGE budget (~0.8s) is the binding constraint on the
    # first attempt's own client-side deadline, so that attempt's own
    # timeout consumes approximately the ENTIRE remaining budget, leaving
    # ~0 for a second attempt.
    engine, primary_client, pool = _make_engine(port, query_timeout_seconds=5, max_query_attempts=3, retry_backoff_seconds=0.05)
    try:
        stage_deadline = time.monotonic() + 0.8
        telemetry: dict = {}
        hits = engine._query(
            collection="col_root", query_vec=[0.0], limit=3,
            stage_deadline_monotonic=stage_deadline, query_telemetry=telemetry,
        )
        assert hits == []
        assert telemetry["outcome"] == "stage_budget_exhausted"
        assert len(handler_class.calls) == 1  # never retried -- the 2nd scripted success was never reached
    finally:
        _close(primary_client, pool)


# ---------------------------------------------------------------------------
# 6. A successful immediate response preserves existing result parsing,
#    candidate selection, and telemetry -- through the real transport
# ---------------------------------------------------------------------------

def test_successful_immediate_response_parses_correctly(delayed_server):
    server, handler_class, port = delayed_server
    handler_class.script = [(0.0, 200)]
    engine, primary_client, pool = _make_engine(port, query_timeout_seconds=5)
    try:
        telemetry: dict = {}
        hits = engine._query(collection="col_root", query_vec=[0.0], limit=3, query_telemetry=telemetry)
        assert len(hits) == 1
        assert hits[0].payload == {"code": "2512", "label_en": "x", "label_ar": ""}
        assert float(hits[0].score) == 0.9
        assert telemetry["outcome"] == "success"
        assert telemetry["attempts"] == 1
    finally:
        _close(primary_client, pool)


# ---------------------------------------------------------------------------
# 7. Non-timeout responses (HTTP validation/auth/server failures) remain
#    non-retryable and fail closed -- through the real transport
# ---------------------------------------------------------------------------

def test_http_error_response_is_non_retryable(delayed_server):
    server, handler_class, port = delayed_server
    handler_class.script = [(0.0, 400), (0.0, 200)]  # 2nd entry would succeed IF ever reached
    engine, primary_client, pool = _make_engine(port, query_timeout_seconds=5, max_query_attempts=3)
    try:
        telemetry: dict = {}
        hits = engine._query(collection="col_root", query_vec=[0.0], limit=3, query_telemetry=telemetry)
        assert hits == []
        assert telemetry["outcome"] == "exception"
        assert telemetry["retryable"] is False
        assert telemetry["exception_type"] == "UnexpectedResponse"
        assert len(handler_class.calls) == 1  # never retried
    finally:
        _close(primary_client, pool)


def test_unexpected_response_directly_not_retryable(delayed_server):
    server, handler_class, port = delayed_server
    handler_class.script = [(0.0, 401)]
    client = QdrantClient(host="127.0.0.1", port=port, timeout=5, check_compatibility=False)
    try:
        with pytest.raises(UnexpectedResponse) as exc_info:
            client.query_points(collection_name="col_root", query=[0.0], limit=3, with_payload=True)
        assert _is_retryable_exception(exc_info.value) is False
    finally:
        client.close()


# ---------------------------------------------------------------------------
# _QdrantClientDeadlinePool: bounded, cleanly closed, reuses the primary
# client for the default deadline
# ---------------------------------------------------------------------------

def test_pool_returns_primary_client_for_default_deadline(delayed_server):
    server, handler_class, port = delayed_server
    primary_client = QdrantClient(host="127.0.0.1", port=port, timeout=8, check_compatibility=False)
    try:
        pool = _QdrantClientDeadlinePool(host="127.0.0.1", port=port, primary_client=primary_client, primary_deadline_seconds=8)
        assert pool.get(8) is primary_client
        assert pool.get(7.9) is primary_client  # ceil(7.9) == 8
    finally:
        primary_client.close()


def test_pool_constructs_and_caches_distinct_client_for_smaller_deadline(delayed_server):
    server, handler_class, port = delayed_server
    primary_client = QdrantClient(host="127.0.0.1", port=port, timeout=8, check_compatibility=False)
    try:
        pool = _QdrantClientDeadlinePool(host="127.0.0.1", port=port, primary_client=primary_client, primary_deadline_seconds=8)
        c1 = pool.get(3)
        c2 = pool.get(3)
        assert c1 is c2  # cached, not reconstructed
        assert c1 is not primary_client
    finally:
        primary_client.close()
        pool.close_all()


def test_pool_bounded_size_evicts_least_recently_used(delayed_server):
    server, handler_class, port = delayed_server
    primary_client = QdrantClient(host="127.0.0.1", port=port, timeout=8, check_compatibility=False)
    try:
        pool = _QdrantClientDeadlinePool(
            host="127.0.0.1", port=port, primary_client=primary_client, primary_deadline_seconds=8, max_size=2,
        )
        c1 = pool.get(1)
        c2 = pool.get(2)
        c3 = pool.get(3)  # should evict c1 (least-recently-used)
        assert len(pool._pool) == 2
        assert 1 not in pool._pool
        assert 2 in pool._pool and 3 in pool._pool
        c1.close()  # already evicted+closed internally; closing again must not raise
    finally:
        primary_client.close()
        pool.close_all()


def test_pool_close_all_is_idempotent(delayed_server):
    server, handler_class, port = delayed_server
    primary_client = QdrantClient(host="127.0.0.1", port=port, timeout=8, check_compatibility=False)
    try:
        pool = _QdrantClientDeadlinePool(host="127.0.0.1", port=port, primary_client=primary_client, primary_deadline_seconds=8)
        pool.get(3)
        pool.close_all()
        pool.close_all()  # must not raise
        assert pool._pool == {}
    finally:
        primary_client.close()
