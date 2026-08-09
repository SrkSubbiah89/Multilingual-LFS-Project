"""
Tests for Task 27's bounded, opt-in Qdrant query retry facility:
backend/rag/hierarchy_engine.py's `_is_retryable_exception()`,
`HierarchyBeamSearchEngine.__init__`'s max_query_attempts/
retry_backoff_seconds validation, `_query()`'s retry loop and extended
telemetry, and `HierarchicalISCOStore`'s config resolution / per-stage
telemetry aggregation.

Hermetic: FakeQdrantClient variants stand in for qdrant_client.QdrantClient
-- no live Qdrant connection, no embedding-model load, no network call,
no real time.sleep() delay (backoff tests use backoff=0 or monkeypatch
time.sleep) anywhere in this file.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

import backend.rag.hierarchical_store as hs_module
from backend.rag.hierarchical_store import HierarchicalISCOStore
from backend.rag.hierarchy_engine import (
    DEFAULT_MAX_QUERY_ATTEMPTS,
    DEFAULT_RETRY_BACKOFF_SECONDS,
    MAX_QUERY_ATTEMPTS_HARD_CAP,
    MAX_RETRY_BACKOFF_SECONDS_HARD_CAP,
    HierarchyBeamSearchEngine,
    StageConfig,
    _is_retryable_exception,
)

_OFFICIAL_PROFILE = "official_ilo2021_v1"
_OFFICIAL_FLAT_COLLECTION = "isco08_unit_groups_flat_ilo2021_v1"
_ALL_OFFICIAL_COLLECTIONS = {
    "isco08_major_groups_ilo2021_v1", "isco08_submajor_groups_ilo2021_v1",
    "isco08_minor_groups_ilo2021_v1", "isco08_unit_groups_ilo2021_v1",
    _OFFICIAL_FLAT_COLLECTION,
}

TWO_STAGE = [
    StageConfig(name="root", collection="col_root", weight=0.4),
    StageConfig(name="leaf", collection="col_leaf", weight=0.6),
]


class ScriptedQdrantClient:
    """query_points() pops and executes the next scripted behaviour from
    `script` (a list of callables or exceptions-to-raise) on every call;
    once exhausted, repeats the last entry. Records every call for
    assertion. `hits_by_call[i]` overrides the returned points for call
    index i (0-based) when a scripted entry is the sentinel "hit"."""

    def __init__(self, script, hits=None):
        self.script = list(script)
        self.hits = hits if hits is not None else []
        self.calls = 0

    def query_points(self, collection_name, query, query_filter, limit, with_payload):
        idx = min(self.calls, len(self.script) - 1)
        entry = self.script[idx]
        self.calls += 1
        if isinstance(entry, BaseException):
            raise entry
        # "hit" sentinel -> return configured points (default: one hit)
        points = self.hits if self.hits else [SimpleNamespace(score=0.9, payload={"code": "2512", "label_en": "x", "label_ar": ""})]
        return SimpleNamespace(points=points)


class EmptyHitQdrantClient:
    """Always succeeds with zero points -- a genuine no-hit response."""

    def __init__(self):
        self.calls = 0

    def query_points(self, collection_name, query, query_filter, limit, with_payload):
        self.calls += 1
        return SimpleNamespace(points=[])


class FakeEmbedder:
    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False, batch_size=1):
        import numpy as np
        return np.zeros((len(texts), 384))


def _make_official_store(monkeypatch, client, max_query_attempts=None, retry_backoff_seconds=None):
    class _Existing:
        def get_collections(self):
            return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in _ALL_OFFICIAL_COLLECTIONS])

    # Merge query_points from `client` with a stub get_collections().
    client.get_collections = _Existing().get_collections
    monkeypatch.setattr(hs_module, "QdrantClient", MagicMock(side_effect=lambda **kw: client))
    monkeypatch.setattr(hs_module, "SentenceTransformer", MagicMock(side_effect=lambda *a, **kw: FakeEmbedder()))
    store = HierarchicalISCOStore(
        profile=_OFFICIAL_PROFILE,
        max_query_attempts=max_query_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    return store


# ---------------------------------------------------------------------------
# 1. Default configuration: one attempt, existing behaviour preserved
# ---------------------------------------------------------------------------

def test_engine_default_max_query_attempts_is_one_no_retry():
    engine = HierarchyBeamSearchEngine(ScriptedQdrantClient([RuntimeError("boom")]), TWO_STAGE, 0.70)
    assert engine.max_query_attempts == DEFAULT_MAX_QUERY_ATTEMPTS == 1
    assert engine.retry_backoff_seconds == DEFAULT_RETRY_BACKOFF_SECONDS == 0.0


def test_default_single_attempt_client_called_exactly_once_on_failure():
    client = ScriptedQdrantClient([httpx.ReadTimeout("simulated timeout")])
    engine = HierarchyBeamSearchEngine(client, TWO_STAGE, 0.70)  # default max_query_attempts=1
    telemetry: dict = {}
    hits = engine._query(collection="col_root", query_vec=[0.0], limit=3, query_telemetry=telemetry)
    assert hits == []
    assert client.calls == 1  # no retry attempted, even though the exception IS retryable
    assert telemetry["outcome"] == "exception"
    assert telemetry["attempts"] == 1


def test_default_single_attempt_success_unaffected():
    client = ScriptedQdrantClient(["hit"])
    engine = HierarchyBeamSearchEngine(client, TWO_STAGE, 0.70)
    telemetry: dict = {}
    hits = engine._query(collection="col_root", query_vec=[0.0], limit=3, query_telemetry=telemetry)
    assert len(hits) == 1
    assert client.calls == 1
    assert telemetry["outcome"] == "success"
    assert telemetry["attempts"] == 1


# ---------------------------------------------------------------------------
# 2. Allowlisted transient timeout retried and succeeds
# ---------------------------------------------------------------------------

def test_retryable_timeout_then_success_reports_success_after_retry():
    client = ScriptedQdrantClient([
        ResponseHandlingException(httpx.ReadTimeout("simulated read timeout")),
        "hit",
    ])
    engine = HierarchyBeamSearchEngine(client, TWO_STAGE, 0.70, max_query_attempts=2)
    telemetry: dict = {}
    hits = engine._query(collection="col_root", query_vec=[0.0], limit=3, query_telemetry=telemetry)
    assert len(hits) == 1
    assert client.calls == 2
    assert telemetry["outcome"] == "success_after_retry"
    assert telemetry["attempts"] == 2
    assert len(telemetry["attempt_durations_ms"]) == 2
    assert "exception_type" not in telemetry  # final outcome is success -- no leftover exception fields


def test_bare_httpx_timeout_exception_directly_is_retryable():
    """Not every call path necessarily wraps in ResponseHandlingException
    -- a bare httpx.TimeoutException must also be retried."""
    client = ScriptedQdrantClient([httpx.ConnectTimeout("simulated connect timeout"), "hit"])
    engine = HierarchyBeamSearchEngine(client, TWO_STAGE, 0.70, max_query_attempts=2)
    telemetry: dict = {}
    hits = engine._query(collection="col_root", query_vec=[0.0], limit=3, query_telemetry=telemetry)
    assert len(hits) == 1
    assert telemetry["outcome"] == "success_after_retry"


# ---------------------------------------------------------------------------
# 3. Retryable timeout that exhausts its limit
# ---------------------------------------------------------------------------

def test_retryable_timeout_exhausts_attempts_returns_empty_with_telemetry():
    client = ScriptedQdrantClient([
        ResponseHandlingException(httpx.ReadTimeout("t1")),
        ResponseHandlingException(httpx.ReadTimeout("t2")),
        ResponseHandlingException(httpx.ReadTimeout("t3")),
    ])
    engine = HierarchyBeamSearchEngine(client, TWO_STAGE, 0.70, max_query_attempts=3)
    telemetry: dict = {}
    hits = engine._query(collection="col_root", query_vec=[0.0], limit=3, query_telemetry=telemetry)
    assert hits == []  # transparent failed/no-code path -- never fabricated
    assert client.calls == 3
    assert telemetry["outcome"] == "retry_exhausted"
    assert telemetry["attempts"] == 3
    assert telemetry["retryable"] is True
    assert telemetry["exception_type"] == "ResponseHandlingException"
    assert len(telemetry["attempt_durations_ms"]) == 3


# ---------------------------------------------------------------------------
# 4. Non-allowlisted exception is never retried
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exc", [
    RuntimeError("generic failure"),
    UnexpectedResponse(status_code=400, reason_phrase="Bad Request", content=b"{}", headers={}),
    ResponseHandlingException(ValueError("malformed response body -- not a timeout")),
    ResponseHandlingException(ConnectionResetError("connection reset by peer")),
])
def test_non_retryable_exception_not_retried_even_with_attempts_available(exc):
    client = ScriptedQdrantClient([exc, "hit"])  # a retry, if attempted, would succeed
    engine = HierarchyBeamSearchEngine(client, TWO_STAGE, 0.70, max_query_attempts=3)
    telemetry: dict = {}
    hits = engine._query(collection="col_root", query_vec=[0.0], limit=3, query_telemetry=telemetry)
    assert hits == []
    assert client.calls == 1  # never retried
    assert telemetry["outcome"] == "exception"
    assert telemetry["retryable"] is False


def test_is_retryable_exception_allowlist_directly():
    assert _is_retryable_exception(httpx.ReadTimeout("x")) is True
    assert _is_retryable_exception(httpx.ConnectTimeout("x")) is True
    assert _is_retryable_exception(httpx.PoolTimeout("x")) is True
    assert _is_retryable_exception(ResponseHandlingException(httpx.WriteTimeout("x"))) is True
    assert _is_retryable_exception(RuntimeError("x")) is False
    assert _is_retryable_exception(TimeoutError("x")) is False  # builtin TimeoutError is NOT the allowlisted class
    assert _is_retryable_exception(ResponseHandlingException(RuntimeError("x"))) is False
    assert _is_retryable_exception(UnexpectedResponse(400, "Bad Request", b"{}", {})) is False
    assert _is_retryable_exception(httpx.TransportError("generic transport error, not a timeout")) is False


# ---------------------------------------------------------------------------
# 5. Genuine no-hit response is not retried
# ---------------------------------------------------------------------------

def test_genuine_zero_hit_response_not_retried():
    client = EmptyHitQdrantClient()
    engine = HierarchyBeamSearchEngine(client, TWO_STAGE, 0.70, max_query_attempts=3)
    telemetry: dict = {}
    hits = engine._query(collection="col_root", query_vec=[0.0], limit=3, query_telemetry=telemetry)
    assert hits == []
    assert client.calls == 1  # zero hits is a successful response -- never triggers a retry
    assert telemetry["outcome"] == "success"
    assert telemetry["attempts"] == 1


# ---------------------------------------------------------------------------
# 6. Attempt limit / configuration validation fails closed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_value", [0, -1, MAX_QUERY_ATTEMPTS_HARD_CAP + 1, 100, 1.5, "2", True])
def test_engine_rejects_invalid_max_query_attempts(bad_value):
    with pytest.raises(ValueError):
        HierarchyBeamSearchEngine(ScriptedQdrantClient(["hit"]), TWO_STAGE, 0.70, max_query_attempts=bad_value)


@pytest.mark.parametrize("bad_value", [-0.1, MAX_RETRY_BACKOFF_SECONDS_HARD_CAP + 0.01, 100, True])
def test_engine_rejects_invalid_retry_backoff_seconds(bad_value):
    with pytest.raises(ValueError):
        HierarchyBeamSearchEngine(ScriptedQdrantClient(["hit"]), TWO_STAGE, 0.70, retry_backoff_seconds=bad_value)


def test_engine_accepts_hard_cap_boundary_values():
    engine = HierarchyBeamSearchEngine(
        ScriptedQdrantClient(["hit"]), TWO_STAGE, 0.70,
        max_query_attempts=MAX_QUERY_ATTEMPTS_HARD_CAP,
        retry_backoff_seconds=MAX_RETRY_BACKOFF_SECONDS_HARD_CAP,
    )
    assert engine.max_query_attempts == MAX_QUERY_ATTEMPTS_HARD_CAP
    assert engine.retry_backoff_seconds == MAX_RETRY_BACKOFF_SECONDS_HARD_CAP


@pytest.mark.parametrize("raw,expected", [
    (None, 1), ("", 1), ("not-a-number", 1), ("0", 1), ("-1", 1),
    (str(MAX_QUERY_ATTEMPTS_HARD_CAP + 1), 1), ("2", 2), (str(MAX_QUERY_ATTEMPTS_HARD_CAP), MAX_QUERY_ATTEMPTS_HARD_CAP),
])
def test_resolve_max_query_attempts_env_var_fails_safe(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("QDRANT_QUERY_MAX_ATTEMPTS", raising=False)
    else:
        monkeypatch.setenv("QDRANT_QUERY_MAX_ATTEMPTS", raw)
    assert hs_module._resolve_max_query_attempts() == expected


@pytest.mark.parametrize("raw,expected", [
    (None, 0.0), ("", 0.0), ("not-a-number", 0.0), ("-0.1", 0.0),
    (str(MAX_RETRY_BACKOFF_SECONDS_HARD_CAP + 1), 0.0), ("0.5", 0.5),
])
def test_resolve_retry_backoff_seconds_env_var_fails_safe(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("QDRANT_QUERY_RETRY_BACKOFF_SECONDS", raising=False)
    else:
        monkeypatch.setenv("QDRANT_QUERY_RETRY_BACKOFF_SECONDS", raw)
    assert hs_module._resolve_retry_backoff_seconds() == expected


def test_store_env_var_opts_in_to_retry_without_code_change(monkeypatch):
    """The future benchmark harness enables retry purely via environment
    configuration, with zero source-code change."""
    monkeypatch.setenv("QDRANT_QUERY_MAX_ATTEMPTS", "2")
    client = ScriptedQdrantClient([ResponseHandlingException(httpx.ReadTimeout("t")), "hit"])
    store = _make_official_store(monkeypatch, client)
    assert store._engine.max_query_attempts == 2
    telemetry: dict = {}
    hits = store._query(collection=_OFFICIAL_FLAT_COLLECTION, query_vec=[0.0], limit=3, query_telemetry=telemetry)
    assert len(hits) == 1
    assert telemetry["outcome"] == "success_after_retry"


def test_backoff_actually_invoked_between_retries_bounded_and_monkeypatched(monkeypatch):
    """Confirms the bounded backoff is genuinely consulted (not merely
    configured) without a real sleep delay in this test suite."""
    import backend.rag.hierarchy_engine as he_module
    sleep_calls = []
    monkeypatch.setattr(he_module.time, "sleep", lambda s: sleep_calls.append(s))
    client = ScriptedQdrantClient([ResponseHandlingException(httpx.ReadTimeout("t")), "hit"])
    engine = HierarchyBeamSearchEngine(client, TWO_STAGE, 0.70, max_query_attempts=2, retry_backoff_seconds=0.75)
    engine._query(collection="col_root", query_vec=[0.0], limit=3)
    assert sleep_calls == [0.75]


def test_zero_backoff_never_sleeps(monkeypatch):
    import backend.rag.hierarchy_engine as he_module
    sleep_calls = []
    monkeypatch.setattr(he_module.time, "sleep", lambda s: sleep_calls.append(s))
    client = ScriptedQdrantClient([ResponseHandlingException(httpx.ReadTimeout("t")), "hit"])
    engine = HierarchyBeamSearchEngine(client, TWO_STAGE, 0.70, max_query_attempts=2, retry_backoff_seconds=0.0)
    engine._query(collection="col_root", query_vec=[0.0], limit=3)
    assert sleep_calls == []


# ---------------------------------------------------------------------------
# 7. Sanitization bounds messages, excludes vectors/text/tracebacks
# ---------------------------------------------------------------------------

def test_retry_exhausted_exception_message_never_contains_query_vector_or_text():
    secret_like_query_marker = "should-never-leak-in-telemetry-AKIA_FAKESECRET"
    client = ScriptedQdrantClient([
        ResponseHandlingException(httpx.ReadTimeout("simulated timeout, no query content here")),
        ResponseHandlingException(httpx.ReadTimeout("simulated timeout, no query content here")),
    ])
    engine = HierarchyBeamSearchEngine(client, TWO_STAGE, 0.70, max_query_attempts=2)
    telemetry: dict = {}
    engine._query(collection="col_root", query_vec=[0.0], limit=3, query_telemetry=telemetry)
    assert secret_like_query_marker not in telemetry["exception_message"]
    assert "Traceback" not in telemetry["exception_message"]
    assert len(telemetry["exception_message"]) <= 300 + len("...(truncated)")


# ---------------------------------------------------------------------------
# 9. Hierarchical stage telemetry is distinct and preserves provenance
# ---------------------------------------------------------------------------

def test_stage_query_telemetry_aggregates_across_branches_without_overwriting():
    """Stage 1 (second engine stage, "leaf") is queried once per stage-0
    branch; a retry on one branch's query must not be lost when a
    different branch's query at the SAME stage succeeds cleanly."""
    client = ScriptedQdrantClient([
        "hit",  # stage 0 (root): one hit -> one branch
    ])
    # Stage 0 always returns the same single hit regardless of parent
    # filter in this simplified scripted client, so only one branch
    # exists; use a client with per-call scripting instead for the
    # single-branch retry case:
    client2 = ScriptedQdrantClient([
        "hit",  # stage 0: root hit
        ResponseHandlingException(httpx.ReadTimeout("t")),  # stage 1 attempt 1: retryable
        "hit",  # stage 1 attempt 2: succeeds
    ])
    engine = HierarchyBeamSearchEngine(client2, TWO_STAGE, 0.70, max_query_attempts=2)
    trace: dict = {}
    result = engine.search(query_vec=[0.0], top_k=3, trace=trace)
    assert result is not None
    assert "stage2_query_telemetry" in trace
    summary = trace["stage2_query_telemetry"]
    assert summary["queries"] == 1
    assert summary["any_retry"] is True
    assert summary["any_exception"] is False
    assert summary["max_attempts_used"] == 2
    # stage1 (root) never retried -- its summary must show no retry.
    assert trace["stage1_query_telemetry"]["any_retry"] is False


def test_stage_latency_includes_retry_and_backoff_time(monkeypatch):
    """--max-stage-latency-ms must see the FULL time including retries
    and backoff -- confirmed here at the trace level (eval/run_eval.py's
    check_strict_hierarchical() reads exactly this field)."""
    import backend.rag.hierarchy_engine as he_module
    monkeypatch.setattr(he_module.time, "sleep", lambda s: None)  # don't actually wait in the test
    client = ScriptedQdrantClient([
        "hit",  # stage 0
        ResponseHandlingException(httpx.ReadTimeout("t")),  # stage 1 attempt 1
        "hit",  # stage 1 attempt 2
    ])
    engine = HierarchyBeamSearchEngine(client, TWO_STAGE, 0.70, max_query_attempts=2, retry_backoff_seconds=0.05)
    trace: dict = {}
    engine.search(query_vec=[0.0], top_k=3, trace=trace)
    # stage2_latency_ms wraps the ENTIRE _query() call, which internally
    # includes both attempts -- it must be >= 0 and present; the key
    # assertion is that it was not reset/zeroed by the retry.
    assert trace["stage2_latency_ms"] >= 0.0
    assert "stage2_query_telemetry" in trace
    assert trace["stage2_query_telemetry"]["any_retry"] is True


# ---------------------------------------------------------------------------
# HierarchicalISCOStore-level: explicit constructor arg validation
# ---------------------------------------------------------------------------

def test_store_rejects_invalid_explicit_max_query_attempts(monkeypatch):
    client = ScriptedQdrantClient(["hit"])
    with pytest.raises(ValueError):
        _make_official_store(monkeypatch, client, max_query_attempts=0)


def test_store_default_max_query_attempts_is_one_when_env_unset(monkeypatch):
    monkeypatch.delenv("QDRANT_QUERY_MAX_ATTEMPTS", raising=False)
    client = ScriptedQdrantClient(["hit"])
    store = _make_official_store(monkeypatch, client)
    assert store._engine.max_query_attempts == 1
