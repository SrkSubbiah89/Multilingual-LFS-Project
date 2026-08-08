"""
Tests for Task 25's additive flat-query telemetry:
backend/rag/hierarchy_engine.py::HierarchyBeamSearchEngine._query()'s
optional `query_telemetry` param and `_sanitize_exception_message()`, and
backend/rag/hierarchical_store.py::HierarchicalISCOStore._flat_search()'s
threading of that telemetry into `trace`.

Hermetic: FakeQdrantClient/_RaisingQdrantClient/FakeEmbedder stand in for
qdrant_client.QdrantClient/SentenceTransformer -- no live Qdrant connection,
no embedding-model load, no network call anywhere in this file. Reuses the
same fixture pattern as test_hierarchical_store.py.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import backend.rag.hierarchical_store as hs_module
from backend.rag.hierarchical_store import HierarchicalISCOStore
from backend.rag.hierarchy_engine import HierarchyBeamSearchEngine, StageConfig, _sanitize_exception_message

_OFFICIAL_PROFILE = "official_ilo2021_v1"
_OFFICIAL_FLAT_COLLECTION = "isco08_unit_groups_flat_ilo2021_v1"
_ALL_OFFICIAL_COLLECTIONS = {
    "isco08_major_groups_ilo2021_v1", "isco08_submajor_groups_ilo2021_v1",
    "isco08_minor_groups_ilo2021_v1", "isco08_unit_groups_ilo2021_v1",
    _OFFICIAL_FLAT_COLLECTION,
}
_ALL_LEGACY_COLLECTIONS = {
    "isco08_major_groups", "isco08_submajor_groups",
    "isco08_minor_groups", "isco08_unit_groups", "isco_occupations",
}


class FakeQdrantClient:
    """table: dict[(collection, parent_code_or_None)] -> list[(code, label_en, label_ar, score)]."""

    def __init__(self, table, existing_collections):
        self.table = table
        self.existing_collections = set(existing_collections)
        self.calls = []

    def get_collections(self):
        return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in self.existing_collections])

    def query_points(self, collection_name, query, query_filter, limit, with_payload):
        parent_code = None
        if query_filter is not None:
            parent_code = query_filter.must[0].match.value
        self.calls.append((collection_name, parent_code, limit))
        rows = self.table.get((collection_name, parent_code), [])
        points = [
            SimpleNamespace(score=score, payload={"code": code, "label_en": label_en, "label_ar": label_ar})
            for code, label_en, label_ar, score in rows[:limit]
        ]
        return SimpleNamespace(points=points)


class _RaisingQdrantClient(FakeQdrantClient):
    """query_points() always raises -- simulates a timeout or any other
    Qdrant request failure, same pattern as test_hierarchical_store.py."""

    def __init__(self, table, existing_collections, exc):
        super().__init__(table, existing_collections)
        self._exc = exc

    def query_points(self, *args, **kwargs):
        raise self._exc


class FakeEmbedder:
    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False, batch_size=1):
        import numpy as np
        return np.zeros((len(texts), 384))


def _make_official_store(monkeypatch, table, client_cls=FakeQdrantClient, exc=None):
    if exc is not None:
        client = client_cls(table, _ALL_OFFICIAL_COLLECTIONS, exc)
    else:
        client = client_cls(table, _ALL_OFFICIAL_COLLECTIONS)
    monkeypatch.setattr(hs_module, "QdrantClient", MagicMock(side_effect=lambda **kw: client))
    monkeypatch.setattr(hs_module, "SentenceTransformer", MagicMock(side_effect=lambda *a, **kw: FakeEmbedder()))
    store = HierarchicalISCOStore(profile=_OFFICIAL_PROFILE)
    return store, client


def _make_legacy_store(monkeypatch, table, client_cls=FakeQdrantClient, exc=None):
    if exc is not None:
        client = client_cls(table, _ALL_LEGACY_COLLECTIONS, exc)
    else:
        client = client_cls(table, _ALL_LEGACY_COLLECTIONS)
    monkeypatch.setattr(hs_module, "QdrantClient", MagicMock(side_effect=lambda **kw: client))
    monkeypatch.setattr(hs_module, "SentenceTransformer", MagicMock(side_effect=lambda *a, **kw: FakeEmbedder()))
    store = HierarchicalISCOStore()  # legacy profile (default)
    return store, client


# ---------------------------------------------------------------------------
# 1. Successful zero-point Qdrant response records no exception telemetry
# ---------------------------------------------------------------------------

def test_flat_search_zero_hits_records_success_outcome_no_exception(monkeypatch):
    store, client = _make_official_store(monkeypatch, table={})  # empty table -> zero hits
    trace: dict = {}
    result = store.search_flat_only("some unclassifiable text", top_k=5, trace=trace)

    assert result.code == ""  # existing unavailable/empty-result behaviour preserved
    assert trace["flat_query_outcome"] == "success"
    assert isinstance(trace["flat_query_duration_ms"], float)
    assert trace["flat_query_duration_ms"] >= 0.0
    assert trace["flat_query_exception_type"] == ""
    assert trace["flat_query_exception_message"] == ""


# ---------------------------------------------------------------------------
# 2/3. Raised exceptions (generic + timeout-shaped) are captured distinctly
# ---------------------------------------------------------------------------

def test_flat_search_generic_exception_records_exception_outcome(monkeypatch):
    store, client = _make_official_store(
        monkeypatch, table={}, client_cls=_RaisingQdrantClient,
        exc=ConnectionError("simulated transport failure"),
    )
    trace: dict = {}
    result = store.search_flat_only("some job title", top_k=5, trace=trace)

    # Classifier decision is unchanged: still the existing explicit empty result.
    assert result.code == ""
    assert trace["flat_query_outcome"] == "exception"
    assert trace["flat_query_exception_type"] == "ConnectionError"
    assert "simulated transport failure" in trace["flat_query_exception_message"]
    assert isinstance(trace["flat_query_duration_ms"], float)


def test_flat_search_timeout_shaped_exception_records_exception_outcome(monkeypatch):
    store, client = _make_official_store(
        monkeypatch, table={}, client_cls=_RaisingQdrantClient,
        exc=TimeoutError("simulated Qdrant request timeout"),
    )
    trace: dict = {}
    result = store.search_flat_only("some job title", top_k=5, trace=trace)

    assert result.code == ""
    assert trace["flat_query_outcome"] == "exception"
    assert trace["flat_query_exception_type"] == "TimeoutError"
    assert "timeout" in trace["flat_query_exception_message"].lower()


# ---------------------------------------------------------------------------
# 4. Flat-query duration telemetry is populated on both success and hit paths
# ---------------------------------------------------------------------------

def test_flat_search_successful_hit_still_records_duration(monkeypatch):
    table = {(_OFFICIAL_FLAT_COLLECTION, None): [("2512", "Software Developers", "", 0.91)]}
    store, client = _make_official_store(monkeypatch, table)
    trace: dict = {}
    result = store.search_flat_only("software developer", top_k=5, trace=trace)

    assert result.code == "2512"
    assert trace["flat_query_outcome"] == "success"
    assert isinstance(trace["flat_query_duration_ms"], float)
    assert trace["flat_query_duration_ms"] >= 0.0
    # Existing stage1..4 trace-bucket behaviour (Task 21) is unaffected.
    assert trace["stage4"][0]["code"] == "2512"


# ---------------------------------------------------------------------------
# 6. Successful official flat retrieval remains a valid four-digit result
# ---------------------------------------------------------------------------

def test_official_flat_search_still_returns_valid_four_digit_code(monkeypatch):
    table = {(_OFFICIAL_FLAT_COLLECTION, None): [("2512", "Software Developers", "", 0.91)]}
    store, client = _make_official_store(monkeypatch, table)
    result = store.search_flat_only("software developer", top_k=5)
    assert result.code == "2512"
    assert result.fallback_used is True  # unchanged existing semantics


def test_official_flat_search_never_fabricates_a_code_on_exception(monkeypatch):
    store, client = _make_official_store(
        monkeypatch, table={}, client_cls=_RaisingQdrantClient,
        exc=TimeoutError("timed out"),
    )
    result = store.search_flat_only("software developer", top_k=5)
    assert result.code == ""
    assert result.hitl_required is True


# ---------------------------------------------------------------------------
# 8. Legacy-profile compatibility remains unchanged
# ---------------------------------------------------------------------------

def test_legacy_profile_flat_search_still_records_telemetry_but_is_otherwise_unchanged(monkeypatch):
    """The legacy profile's HierarchicalISCOStore._flat_search() (used
    when e.g. all four hierarchical collections are absent) shares the
    same _flat_search() implementation as the official profile, so it
    also gains the new telemetry fields -- but its existing return value/
    behaviour is byte-identical to before this task. (The SEPARATE legacy
    VectorStore-based flat path used by ISCOClassifier._classify_flat()
    for --isco-catalogue-profile legacy --system flat is untouched by
    this task entirely -- see backend/agents/isco_classifier.py, not
    exercised by this store-level test file.)"""
    table = {("isco_occupations", None): [("2512", "Software Developers", "", 0.88)]}
    store, client = _make_legacy_store(monkeypatch, table)
    trace: dict = {}
    result = store._flat_search([0.0], top_k=5, trace=trace)

    assert result.code == "2512"
    assert result.fallback_used is True
    assert trace["flat_query_outcome"] == "success"


# ---------------------------------------------------------------------------
# 9. No raw query text or secret-like text appears in telemetry
# ---------------------------------------------------------------------------

def test_sanitize_exception_message_bounds_length_and_flattens_newlines():
    raw = ("line one\nline two\twith tab\n" + ("x" * 500))
    sanitized = _sanitize_exception_message(raw)
    assert "\n" not in sanitized
    assert "\t" not in sanitized
    assert len(sanitized) <= 300 + len("...(truncated)")
    assert sanitized.endswith("...(truncated)")


def test_sanitize_exception_message_short_message_passthrough_flattened():
    assert _sanitize_exception_message("timed out") == "timed out"


def test_flat_search_exception_message_never_contains_query_vector_or_text(monkeypatch):
    """The exception message is derived only from str(exc) -- confirm the
    query text/vector passed into search_flat_only() never leaks into the
    recorded telemetry even when it happens to share characters with the
    (unrelated) simulated exception text."""
    secret_like_query = "confidential-job-title-should-never-leak-AKIA_FAKESECRET123"
    store, client = _make_official_store(
        monkeypatch, table={}, client_cls=_RaisingQdrantClient,
        exc=TimeoutError("simulated Qdrant request timeout"),
    )
    trace: dict = {}
    store.search_flat_only(secret_like_query, top_k=5, trace=trace)
    assert secret_like_query not in trace["flat_query_exception_message"]
    assert "Traceback" not in trace["flat_query_exception_message"]


# ---------------------------------------------------------------------------
# Engine-level: query_telemetry defaults to None (fully backward compatible)
# ---------------------------------------------------------------------------

def test_query_telemetry_param_optional_default_none_unchanged_behaviour():
    stages = [StageConfig(name="a", collection="col_a", weight=0.5), StageConfig(name="b", collection="col_b", weight=0.5)]
    engine = HierarchyBeamSearchEngine(FakeQdrantClient({}, set()), stages, 0.70)
    # No query_telemetry passed -- must not raise, returns [] as before.
    assert engine._query(collection="col_a", query_vec=[0.0], limit=3) == []


def test_query_telemetry_populated_on_success():
    stages = [StageConfig(name="a", collection="col_a", weight=0.5), StageConfig(name="b", collection="col_b", weight=0.5)]
    table = {("col_a", None): [("x1", "X1", "", 0.5)]}
    engine = HierarchyBeamSearchEngine(FakeQdrantClient(table, set()), stages, 0.70)
    telemetry: dict = {}
    hits = engine._query(collection="col_a", query_vec=[0.0], limit=3, query_telemetry=telemetry)
    assert len(hits) == 1
    assert telemetry["outcome"] == "success"
    assert "exception_type" not in telemetry


def test_query_telemetry_populated_on_exception():
    stages = [StageConfig(name="a", collection="col_a", weight=0.5), StageConfig(name="b", collection="col_b", weight=0.5)]

    class _Raising(FakeQdrantClient):
        def query_points(self, *a, **kw):
            raise RuntimeError("boom")

    engine = HierarchyBeamSearchEngine(_Raising({}, set()), stages, 0.70)
    telemetry: dict = {}
    hits = engine._query(collection="col_a", query_vec=[0.0], limit=3, query_telemetry=telemetry)
    assert hits == []
    assert telemetry["outcome"] == "exception"
    assert telemetry["exception_type"] == "RuntimeError"
    assert telemetry["exception_message"] == "boom"
