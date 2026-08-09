"""
Tests for Task 31's reliable per-request timeout propagation and strict
stage-level deadline budget enforcement:
backend/rag/hierarchy_engine.py's `_query()`/`search()`'s new
`stage_deadline_monotonic`/`max_stage_latency_ms` parameters, and
`HierarchyBeamSearchEngine.__init__`'s new `query_timeout_seconds` param.

Hermetic: FakeQdrantClient variants (recording every `timeout=` kwarg
received) stand in for qdrant_client.QdrantClient; `time.monotonic`/
`time.sleep` are monkeypatched where a controllable clock is needed. No
live Qdrant connection, no embedding-model load, no network call, no
real sleep delay anywhere in this file.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from qdrant_client.http.exceptions import ResponseHandlingException

import backend.rag.hierarchical_store as hs_module
from backend.rag.hierarchical_store import HierarchicalISCOStore
from backend.rag.hierarchy_engine import HierarchyBeamSearchEngine, StageConfig

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
FOUR_STAGE = [
    StageConfig(name="major", collection="col_major", weight=0.10),
    StageConfig(name="submajor", collection="col_submajor", weight=0.20),
    StageConfig(name="minor", collection="col_minor", weight=0.20),
    StageConfig(name="unit", collection="col_unit", weight=0.50),
]


class RecordingQdrantClient:
    """table: dict[(collection, parent_code_or_None)] -> list[(code, label_en, label_ar, score)].
    Records every call's (collection, parent_code, limit, timeout).
    `script` (per-collection list of callables/exceptions/"hit") lets a
    test script per-call behaviour for a specific collection, consumed
    in order; once exhausted, falls back to `table`."""

    def __init__(self, table=None, existing_collections=None, script=None):
        self.table = table or {}
        self.existing_collections = set(existing_collections or _ALL_OFFICIAL_COLLECTIONS)
        self.calls: list = []
        self.timeouts: list = []
        self._script = dict(script or {})  # collection -> list of behaviours (consumed in order)

    def get_collections(self):
        return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in self.existing_collections])

    def query_points(self, collection_name, query, query_filter, limit, with_payload, timeout=None):
        parent_code = None
        if query_filter is not None:
            parent_code = query_filter.must[0].match.value
        self.calls.append((collection_name, parent_code, limit))
        self.timeouts.append(timeout)

        behaviours = self._script.get(collection_name)
        if behaviours:
            entry = behaviours.pop(0)
            if isinstance(entry, BaseException):
                raise entry
            if entry == "hit":
                return SimpleNamespace(points=[
                    SimpleNamespace(score=0.9, payload={"code": "2512", "label_en": "x", "label_ar": ""})
                ])

        rows = self.table.get((collection_name, parent_code), [])
        points = [
            SimpleNamespace(score=score, payload={"code": code, "label_en": label_en, "label_ar": label_ar})
            for code, label_en, label_ar, score in rows[:limit]
        ]
        return SimpleNamespace(points=points)


class FakeEmbedder:
    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False, batch_size=1):
        import numpy as np
        return np.zeros((len(texts), 384))


def _make_official_store(monkeypatch, client, **engine_kwargs):
    monkeypatch.setattr(hs_module, "QdrantClient", MagicMock(side_effect=lambda **kw: client))
    monkeypatch.setattr(hs_module, "SentenceTransformer", MagicMock(side_effect=lambda *a, **kw: FakeEmbedder()))
    store = HierarchicalISCOStore(profile=_OFFICIAL_PROFILE, **engine_kwargs)
    return store


class _MonotonicClock:
    """A controllable fake time.monotonic() -- each call advances by a
    caller-scripted amount (default 0.0, i.e. instantaneous unless the
    test explicitly advances it)."""

    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


# ---------------------------------------------------------------------------
# 1. Explicit per-request timeout passed at the real query call boundary
# ---------------------------------------------------------------------------

def test_configured_query_timeout_passed_to_query_points():
    client = RecordingQdrantClient(table={("col_root", None): [("2512", "x", "", 0.9)]})
    engine = HierarchyBeamSearchEngine(client, TWO_STAGE, 0.70, query_timeout_seconds=8)
    engine._query(collection="col_root", query_vec=[0.0], limit=3)
    assert client.timeouts == [8]


def test_no_query_timeout_configured_omits_timeout_kwarg():
    client = RecordingQdrantClient(table={("col_root", None): [("2512", "x", "", 0.9)]})
    engine = HierarchyBeamSearchEngine(client, TWO_STAGE, 0.70)  # query_timeout_seconds=None
    engine._query(collection="col_root", query_vec=[0.0], limit=3)
    assert client.timeouts == [None]


def test_store_propagates_its_own_resolved_timeout_to_the_engine(monkeypatch):
    client = RecordingQdrantClient(table={(_OFFICIAL_FLAT_COLLECTION, None): [("2512", "x", "", 0.9)]})
    store = _make_official_store(monkeypatch, client, timeout_seconds=8)
    assert store._engine.query_timeout_seconds == 8.0
    store.search_flat_only("software developer", top_k=5)
    assert client.timeouts == [8]


# ---------------------------------------------------------------------------
# 2. Timeout is capped by remaining stage budget
# ---------------------------------------------------------------------------

def test_effective_timeout_capped_by_remaining_stage_budget(monkeypatch):
    clock = _MonotonicClock()
    monkeypatch.setattr("backend.rag.hierarchy_engine.time.monotonic", clock)
    client = RecordingQdrantClient(table={("col_root", None): [("2512", "x", "", 0.9)]})
    engine = HierarchyBeamSearchEngine(client, TWO_STAGE, 0.70, query_timeout_seconds=8)
    # A 3-second-remaining deadline is tighter than the configured 8s.
    deadline = clock.now + 3.0
    engine._query(collection="col_root", query_vec=[0.0], limit=3, stage_deadline_monotonic=deadline)
    assert client.timeouts == [3]  # math.ceil(3.0) == 3, capped below the configured 8


def test_effective_timeout_uses_configured_value_when_budget_is_looser(monkeypatch):
    clock = _MonotonicClock()
    monkeypatch.setattr("backend.rag.hierarchy_engine.time.monotonic", clock)
    client = RecordingQdrantClient(table={("col_root", None): [("2512", "x", "", 0.9)]})
    engine = HierarchyBeamSearchEngine(client, TWO_STAGE, 0.70, query_timeout_seconds=8)
    deadline = clock.now + 100.0  # much looser than the configured 8s
    engine._query(collection="col_root", query_vec=[0.0], limit=3, stage_deadline_monotonic=deadline)
    assert client.timeouts == [8]


# ---------------------------------------------------------------------------
# 3. Default/non-strict calls retain pre-task no-stage-budget behaviour
# ---------------------------------------------------------------------------

def test_search_without_max_stage_latency_ms_never_establishes_a_deadline(monkeypatch):
    monkeypatch.setattr("backend.rag.hierarchy_engine.time.monotonic", lambda: (_ for _ in ()).throw(
        AssertionError("time.monotonic() must never be called when no stage budget is requested")
    ))
    table = {
        ("col_root", None): [("r1", "Root 1", "", 0.9)],
        ("col_leaf", "r1"): [("2512", "Software Developers", "", 0.85)],
    }
    client = RecordingQdrantClient(table=table)
    engine = HierarchyBeamSearchEngine(client, TWO_STAGE, 0.70)
    result = engine.search(query_vec=[0.0], top_k=3)  # no max_stage_latency_ms
    assert result is not None
    assert result.code == "2512"
    assert client.timeouts == [None, None]


def test_default_engine_construction_accepts_omitted_query_timeout():
    """HierarchyBeamSearchEngine(client, stages, threshold) -- the exact
    pre-Task-31 call signature -- must still work unchanged."""
    client = RecordingQdrantClient(table={("col_root", None): [("2512", "x", "", 0.9)]})
    engine = HierarchyBeamSearchEngine(client, TWO_STAGE, 0.70)
    assert engine.query_timeout_seconds is None


# ---------------------------------------------------------------------------
# 6/12. Configuration validation for query_timeout_seconds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_value", [0, -1, -0.5, "8", True])
def test_engine_rejects_invalid_query_timeout_seconds(bad_value):
    client = RecordingQdrantClient()
    with pytest.raises(ValueError):
        HierarchyBeamSearchEngine(client, TWO_STAGE, 0.70, query_timeout_seconds=bad_value)


def test_engine_accepts_none_query_timeout_seconds():
    client = RecordingQdrantClient()
    engine = HierarchyBeamSearchEngine(client, TWO_STAGE, 0.70, query_timeout_seconds=None)
    assert engine.query_timeout_seconds is None


# ---------------------------------------------------------------------------
# 4. A stage with multiple branch queries stops before starting another
#    branch once its shared budget is exhausted
# ---------------------------------------------------------------------------

def test_second_branch_query_skipped_once_stage_budget_exhausted(monkeypatch):
    clock = _MonotonicClock()
    monkeypatch.setattr("backend.rag.hierarchy_engine.time.monotonic", clock)
    # Stage 0 returns 2 candidates -> stage 1 (final) explores 2 branches.
    table = {
        ("col_root", None): [("r1", "Root 1", "", 0.9), ("r2", "Root 2", "", 0.8)],
        ("col_leaf", "r1"): [("2512", "x", "", 0.85)],
        ("col_leaf", "r2"): [("2513", "y", "", 0.80)],
    }
    client = RecordingQdrantClient(table=table)
    engine = HierarchyBeamSearchEngine(client, TWO_STAGE, 0.70)

    # Call order: #1 = stage 0's own query (root); #2 = stage 1's FIRST
    # branch query (leaf, parent=r1); #3 = stage 1's SECOND branch query
    # (leaf, parent=r2) -- which must never happen. The stage-1 deadline
    # is established once, before the branch loop begins (after call #1
    # returns) -- advance the clock only after call #2 completes, so the
    # already-established deadline is in the past by the time call #3
    # would check its own remaining budget.
    real_query = engine._query
    call_count = {"n": 0}

    def _tracking_query(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            result = real_query(*args, **kwargs)
            clock.advance(1000.0)  # blow the stage-1 deadline before branch 2 starts
            return result
        return real_query(*args, **kwargs)

    monkeypatch.setattr(engine, "_query", _tracking_query)
    trace: dict = {}
    result = engine.search(query_vec=[0.0], top_k=3, beam=2, max_stage_latency_ms=1.0, trace=trace)
    # Branch 1's genuine, already-obtained hit is not discarded just
    # because a LATER sibling branch's budget ran out -- but the second
    # branch's query was genuinely skipped (no fabricated candidate for
    # it), and the exhaustion is disclosed in the stage telemetry.
    assert result is not None
    assert result.code == "2512"
    stage2_calls = [c for c in client.calls if c[0] == "col_leaf"]
    assert len(stage2_calls) == 1  # branch 2's query was never issued
    assert trace["stage2_query_telemetry"]["stage_budget_exhausted"] is True
    assert trace["stage2_query_telemetry"]["queries"] == 2  # both branch attempts recorded (one real, one exhausted)


# ---------------------------------------------------------------------------
# 5/6. Retry and backoff consume the same stage budget; a retry is not
#      started when insufficient budget remains
# ---------------------------------------------------------------------------

def test_retry_not_started_when_stage_budget_exhausted_after_failure(monkeypatch):
    clock = _MonotonicClock()
    sleeps: list = []
    monkeypatch.setattr("backend.rag.hierarchy_engine.time.monotonic", clock)
    monkeypatch.setattr("backend.rag.hierarchy_engine.time.sleep", lambda s: sleeps.append(s))

    def _raise_then_blow_clock(*a, **kw):
        clock.advance(1000.0)  # simulate a slow failing attempt that blows the deadline
        raise ResponseHandlingException(httpx.ReadTimeout("t"))

    client = RecordingQdrantClient()
    client.query_points = _raise_then_blow_clock  # type: ignore[method-assign]
    engine = HierarchyBeamSearchEngine(client, TWO_STAGE, 0.70, max_query_attempts=3, retry_backoff_seconds=0.5)
    deadline = clock.now + 5.0
    telemetry: dict = {}
    hits = engine._query(collection="col_root", query_vec=[0.0], limit=3, stage_deadline_monotonic=deadline, query_telemetry=telemetry)
    assert hits == []
    assert sleeps == []  # never slept -- budget was already gone
    assert telemetry["outcome"] == "stage_budget_exhausted"
    assert telemetry["attempts"] == 1  # exactly one attempt was made, no retry


def test_backoff_capped_by_remaining_stage_budget(monkeypatch):
    """The backoff sleep itself must never exceed the remaining budget."""
    clock = _MonotonicClock()
    sleeps: list = []
    monkeypatch.setattr("backend.rag.hierarchy_engine.time.monotonic", clock)
    monkeypatch.setattr("backend.rag.hierarchy_engine.time.sleep", lambda s: sleeps.append(s))

    call_n = {"n": 0}

    def _fail_then_succeed(*a, **kw):
        call_n["n"] += 1
        if call_n["n"] == 1:
            clock.advance(4.7)  # leaves 0.3s of a 5.0s budget remaining
            raise ResponseHandlingException(httpx.ReadTimeout("t"))
        return SimpleNamespace(points=[SimpleNamespace(score=0.9, payload={"code": "2512", "label_en": "x", "label_ar": ""})])

    client = RecordingQdrantClient()
    client.query_points = _fail_then_succeed  # type: ignore[method-assign]
    engine = HierarchyBeamSearchEngine(client, TWO_STAGE, 0.70, max_query_attempts=2, retry_backoff_seconds=2.0)
    deadline = clock.now + 5.0
    telemetry: dict = {}
    hits = engine._query(collection="col_root", query_vec=[0.0], limit=3, stage_deadline_monotonic=deadline, query_telemetry=telemetry)
    assert len(hits) == 1  # succeeded on the retry, within budget
    assert telemetry["outcome"] == "success_after_retry"
    assert len(sleeps) == 1
    assert sleeps[0] <= 0.30001  # capped to (approximately) the remaining budget, not the full 2.0s configured backoff


# ---------------------------------------------------------------------------
# 7/9/10/11. stage_budget_exhausted never fabricates; success-after-retry
#            stays genuine; non-retryable fails immediately; zero-hit
#            success is not retried -- all under an active budget
# ---------------------------------------------------------------------------

def test_stage_budget_exhausted_at_entry_returns_no_fabricated_hits(monkeypatch):
    clock = _MonotonicClock()
    monkeypatch.setattr("backend.rag.hierarchy_engine.time.monotonic", clock)
    client = RecordingQdrantClient(table={("col_root", None): [("2512", "x", "", 0.9)]})
    engine = HierarchyBeamSearchEngine(client, TWO_STAGE, 0.70)
    deadline = clock.now - 1.0  # already expired
    telemetry: dict = {}
    hits = engine._query(collection="col_root", query_vec=[0.0], limit=3, stage_deadline_monotonic=deadline, query_telemetry=telemetry)
    assert hits == []
    assert client.calls == []  # query_points() was never called at all
    assert telemetry["outcome"] == "stage_budget_exhausted"
    assert telemetry["attempts"] == 0


def test_success_after_retry_within_budget_has_real_final_evidence(monkeypatch):
    clock = _MonotonicClock()
    monkeypatch.setattr("backend.rag.hierarchy_engine.time.monotonic", clock)
    monkeypatch.setattr("backend.rag.hierarchy_engine.time.sleep", lambda s: None)
    client = RecordingQdrantClient(script={"col_root": [ResponseHandlingException(httpx.ReadTimeout("t")), "hit"]})
    engine = HierarchyBeamSearchEngine(client, TWO_STAGE, 0.70, max_query_attempts=2, retry_backoff_seconds=0.1)
    deadline = clock.now + 30.0
    telemetry: dict = {}
    hits = engine._query(collection="col_root", query_vec=[0.0], limit=3, stage_deadline_monotonic=deadline, query_telemetry=telemetry)
    assert len(hits) == 1
    assert hits[0].payload["code"] == "2512"
    assert telemetry["outcome"] == "success_after_retry"
    assert telemetry["attempts"] == 2


def test_non_retryable_error_fails_immediately_even_with_budget_active(monkeypatch):
    clock = _MonotonicClock()
    monkeypatch.setattr("backend.rag.hierarchy_engine.time.monotonic", clock)
    client = RecordingQdrantClient(script={"col_root": [RuntimeError("generic"), "hit"]})
    engine = HierarchyBeamSearchEngine(client, TWO_STAGE, 0.70, max_query_attempts=3)
    deadline = clock.now + 30.0
    telemetry: dict = {}
    hits = engine._query(collection="col_root", query_vec=[0.0], limit=3, stage_deadline_monotonic=deadline, query_telemetry=telemetry)
    assert hits == []
    assert client.calls == [("col_root", None, 3)]  # exactly one call -- never retried
    assert telemetry["outcome"] == "exception"


def test_genuine_zero_hit_not_retried_with_budget_active(monkeypatch):
    clock = _MonotonicClock()
    monkeypatch.setattr("backend.rag.hierarchy_engine.time.monotonic", clock)
    client = RecordingQdrantClient(table={})  # empty -> zero hits, no exception
    engine = HierarchyBeamSearchEngine(client, TWO_STAGE, 0.70, max_query_attempts=3)
    deadline = clock.now + 30.0
    telemetry: dict = {}
    hits = engine._query(collection="col_root", query_vec=[0.0], limit=3, stage_deadline_monotonic=deadline, query_telemetry=telemetry)
    assert hits == []
    assert len(client.calls) == 1
    assert telemetry["outcome"] == "success"


# ---------------------------------------------------------------------------
# 13. Telemetry is bounded, sanitized, additive, parseable
# ---------------------------------------------------------------------------

def test_stage_query_telemetry_includes_new_task31_fields(monkeypatch):
    clock = _MonotonicClock()
    monkeypatch.setattr("backend.rag.hierarchy_engine.time.monotonic", clock)
    table = {
        ("col_root", None): [("r1", "Root 1", "", 0.9)],
        ("col_leaf", "r1"): [("2512", "x", "", 0.85)],
    }
    client = RecordingQdrantClient(table=table)
    engine = HierarchyBeamSearchEngine(client, TWO_STAGE, 0.70, query_timeout_seconds=8)
    trace: dict = {}
    result = engine.search(query_vec=[0.0], top_k=3, max_stage_latency_ms=5000.0, trace=trace)
    assert result is not None
    for key in ("stage1_query_telemetry", "stage2_query_telemetry"):
        summary = trace[key]
        assert summary["stage_budget_exhausted"] is False
        assert summary["configured_query_timeout_seconds"] == 8
        assert summary["initial_stage_budget_ms"] == 5000.0
        assert isinstance(summary["queries_detail"], list) and len(summary["queries_detail"]) == 1
        detail = summary["queries_detail"][0]
        assert detail["outcome"] == "success"
        assert detail["attempts"] == 1
        assert isinstance(detail["remaining_stage_budget_ms_at_entry"], float)


def test_stage_budget_fields_absent_when_no_budget_requested():
    client = RecordingQdrantClient(table={("col_root", None): [("2512", "x", "", 0.9)]})
    engine = HierarchyBeamSearchEngine(client, TWO_STAGE, 0.70)
    trace: dict = {}
    engine._query(collection="col_root", query_vec=[0.0], limit=3)  # no trace usage here; use search() instead
    trace2: dict = {}
    engine.search(query_vec=[0.0], top_k=3, trace=trace2)  # no max_stage_latency_ms
    summary = trace2["stage1_query_telemetry"]
    assert summary["initial_stage_budget_ms"] is None
    assert summary["stage_budget_exhausted"] is False
