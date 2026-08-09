"""
Tests for backend/rag/hierarchical_store.py -- Task 13's keyword-anchor
retry recovery and bounded Qdrant request timeout.

Hermetic: FakeQdrantClient/FakeEmbedder stand in for
qdrant_client.QdrantClient/SentenceTransformer, monkeypatched at module
level before HierarchicalISCOStore() construction -- no live Qdrant
connection, no embedding-model load, no network call anywhere in this
file.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import backend.rag.hierarchical_store as hs_module
from backend.rag.hierarchical_store import (
    QDRANT_DEFAULT_TIMEOUT_SECONDS,
    HierarchicalISCOStore,
    _resolve_qdrant_timeout_seconds,
)

_ALL_ISCO_COLLECTIONS = {
    "isco08_major_groups", "isco08_submajor_groups",
    "isco08_minor_groups", "isco08_unit_groups", "isco_occupations",
}


class FakeQdrantClient:
    """table: dict[(collection, parent_code_or_None)] -> list[(code, label_en, label_ar, score)]."""

    def __init__(self, table, existing_collections):
        self.table = table
        self.existing_collections = set(existing_collections)
        self.calls = []  # (collection_name, parent_code, limit) per call, in order

    def get_collections(self):
        return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in self.existing_collections])

    def query_points(self, collection_name, query, query_filter, limit, with_payload, timeout=None):
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
    """query_points() always raises -- simulates a bounded-timeout firing
    or any other Qdrant request failure."""

    def query_points(self, *args, **kwargs):
        raise TimeoutError("simulated Qdrant request timeout")


class FakeEmbedder:
    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False, batch_size=1):
        import numpy as np
        return np.zeros((len(texts), 384))


def _make_store(monkeypatch, table, existing=None, client_cls=FakeQdrantClient, timeout_seconds=None):
    existing = existing if existing is not None else set(_ALL_ISCO_COLLECTIONS)
    client = client_cls(table, existing)
    monkeypatch.setattr(hs_module, "QdrantClient", MagicMock(side_effect=lambda **kw: client))
    monkeypatch.setattr(hs_module, "SentenceTransformer", MagicMock(side_effect=lambda *a, **kw: FakeEmbedder()))
    store = HierarchicalISCOStore(timeout_seconds=timeout_seconds)
    return store, client


# ---------------------------------------------------------------------------
# 1. Successful keyword seed -- one engine search, prior behaviour preserved
# ---------------------------------------------------------------------------

_SUCCESS_TABLE = {
    ("isco08_submajor_groups", "2"): [("25", "ICT Professionals", "", 0.85)],
    ("isco08_minor_groups", "25"): [("251", "Software Developers", "", 0.83)],
    ("isco08_unit_groups", "251"): [("2512", "Software Developers", "", 0.80)],
}


def test_successful_keyword_seed_performs_one_engine_search(monkeypatch):
    store, client = _make_store(monkeypatch, _SUCCESS_TABLE)
    trace: dict = {}
    result = store._hierarchical_search([0.0], top_k=5, major_hint="2", trace=trace)

    assert result is not None
    assert result.code == "2512"
    assert result.fallback_used is False
    assert client.calls == [
        ("isco08_submajor_groups", "2", 2),
        ("isco08_minor_groups", "25", 2),
        ("isco08_unit_groups", "251", 5),
    ]
    assert trace["keyword_anchor_retry"] is False
    assert trace["keyword_anchor_original_hint"] == "2"
    assert trace["stage1_source"] == "keyword_map"


def test_no_major_hint_never_sets_keyword_anchor_metadata(monkeypatch):
    """Ordinary semantic stage-1 (no keyword hint at all) must not carry
    any keyword_anchor_* trace key -- those are only meaningful when an
    anchor was actually attempted."""
    table = {
        ("isco08_major_groups", None): [("2", "Professionals", "", 0.80)],
        ("isco08_submajor_groups", "2"): [("25", "ICT Professionals", "", 0.85)],
        ("isco08_minor_groups", "25"): [("251", "Software Developers", "", 0.83)],
        ("isco08_unit_groups", "251"): [("2512", "Software Developers", "", 0.80)],
    }
    store, client = _make_store(monkeypatch, table)
    trace: dict = {}
    result = store._hierarchical_search([0.0], top_k=5, major_hint="", trace=trace)
    assert result is not None
    assert "keyword_anchor_retry" not in trace
    assert "keyword_anchor_original_hint" not in trace


# ---------------------------------------------------------------------------
# 2 & 3. A failed seed retries exactly once, unseeded, and the trace shows
# only the winning attempt's stage evidence
# ---------------------------------------------------------------------------

_RECOVERY_TABLE = {
    ("isco08_submajor_groups", "5"): [],  # seeded attempt anchored at WRONG major "5" -> dead end
    ("isco08_major_groups", None): [("2", "Professionals", "", 0.80)],  # unseeded retry finds "2"
    ("isco08_submajor_groups", "2"): [("25", "ICT Professionals", "", 0.85)],
    ("isco08_minor_groups", "25"): [("251", "Software Developers", "", 0.83)],
    ("isco08_unit_groups", "251"): [("2512", "Software Developers", "", 0.80)],
}


def test_failed_seed_retries_once_unseeded_and_recovers(monkeypatch):
    store, client = _make_store(monkeypatch, _RECOVERY_TABLE)
    trace: dict = {}
    result = store._hierarchical_search([0.0], top_k=5, major_hint="5", trace=trace)

    assert result is not None
    assert result.code == "2512"
    assert result.fallback_used is False
    assert trace["keyword_anchor_retry"] is True
    assert trace["keyword_anchor_original_hint"] == "5"
    # Exactly one retry: the seeded attempt's single dead-end query, then
    # the full unseeded 4-stage chain -- never repeated further.
    assert client.calls == [
        ("isco08_submajor_groups", "5", 2),
        ("isco08_major_groups", None, 2),
        ("isco08_submajor_groups", "2", 2),
        ("isco08_minor_groups", "25", 2),
        ("isco08_unit_groups", "251", 5),
    ]


def test_retry_trace_shows_only_winning_stage_evidence(monkeypatch):
    store, client = _make_store(monkeypatch, _RECOVERY_TABLE)
    trace: dict = {}
    store._hierarchical_search([0.0], top_k=5, major_hint="5", trace=trace)

    # stage1 must contain ONLY the retry's genuine semantic hit ("2"),
    # never the failed seed's own single-candidate entry ("5") -- the
    # failed attempt's throwaway trace must never be merged in.
    assert trace["stage1"] == [{"code": "2", "label_en": "Professionals", "score": 0.8}]
    assert trace["stage1_source"] == "semantic_retrieval"  # the genuine final path, not "keyword_map"
    codes_seen = {c["code"] for c in trace["stage1"]}
    assert "5" not in codes_seen


# ---------------------------------------------------------------------------
# 4. Both seeded and unseeded paths fail -> explicit flat fallback
# ---------------------------------------------------------------------------

_BOTH_FAIL_TABLE = {
    ("isco08_submajor_groups", "5"): [],       # seeded: dead end
    ("isco08_major_groups", None): [("2", "Professionals", "", 0.80)],
    ("isco08_submajor_groups", "2"): [],       # unseeded retry ALSO dead-ends
    ("isco_occupations", None): [("2512", "Software Developers", "", 0.75)],
}


def test_both_seeded_and_unseeded_fail_falls_back_to_flat_explicitly(monkeypatch):
    store, client = _make_store(monkeypatch, _BOTH_FAIL_TABLE)
    trace: dict = {}
    result = store.search("software developer", major_hint="5", trace=trace)

    assert result.fallback_used is True
    assert result.code == "2512"
    # Flat fallback's documented duplicate-candidate trace shape -- still
    # explicitly distinguishable from a genuine hierarchical result via
    # fallback_used=True, never silently relabelled as hierarchical.
    assert trace["stage1"] == trace["stage2"] == trace["stage3"] == trace["stage4"]
    assert trace["stage1"][0]["code"] == "2512"
    # The failed keyword-anchor attempt is still honestly recorded, even
    # though the overall result fell all the way through to flat.
    assert trace["keyword_anchor_retry"] is True
    assert trace["keyword_anchor_original_hint"] == "5"


# ---------------------------------------------------------------------------
# 5. Qdrant timeout: default, valid override, invalid override fallback,
# and client construction -- no live connection anywhere
# ---------------------------------------------------------------------------

def test_timeout_default_is_30_seconds():
    assert QDRANT_DEFAULT_TIMEOUT_SECONDS == 30


def test_timeout_resolves_to_default_when_env_absent(monkeypatch):
    monkeypatch.delenv("QDRANT_TIMEOUT_SECONDS", raising=False)
    assert _resolve_qdrant_timeout_seconds() == QDRANT_DEFAULT_TIMEOUT_SECONDS


def test_timeout_valid_env_override(monkeypatch):
    monkeypatch.setenv("QDRANT_TIMEOUT_SECONDS", "45")
    assert _resolve_qdrant_timeout_seconds() == 45


@pytest.mark.parametrize("raw", ["not_a_number", "", "   ", "0", "-5", "3.5"])
def test_timeout_invalid_env_falls_back_to_default(monkeypatch, raw):
    monkeypatch.setenv("QDRANT_TIMEOUT_SECONDS", raw)
    assert _resolve_qdrant_timeout_seconds() == QDRANT_DEFAULT_TIMEOUT_SECONDS


def test_client_construction_passes_resolved_default_timeout(monkeypatch):
    monkeypatch.delenv("QDRANT_TIMEOUT_SECONDS", raising=False)
    captured = {}
    client = FakeQdrantClient({}, set(_ALL_ISCO_COLLECTIONS))
    monkeypatch.setattr(hs_module, "QdrantClient", MagicMock(side_effect=lambda **kw: captured.update(kw) or client))
    monkeypatch.setattr(hs_module, "SentenceTransformer", MagicMock(side_effect=lambda *a, **kw: FakeEmbedder()))
    HierarchicalISCOStore()
    assert captured["timeout"] == QDRANT_DEFAULT_TIMEOUT_SECONDS


def test_client_construction_honours_explicit_timeout_seconds_param(monkeypatch):
    captured = {}
    client = FakeQdrantClient({}, set(_ALL_ISCO_COLLECTIONS))
    monkeypatch.setattr(hs_module, "QdrantClient", MagicMock(side_effect=lambda **kw: captured.update(kw) or client))
    monkeypatch.setattr(hs_module, "SentenceTransformer", MagicMock(side_effect=lambda *a, **kw: FakeEmbedder()))
    HierarchicalISCOStore(timeout_seconds=5)
    assert captured["timeout"] == 5


def test_client_construction_honours_valid_env_override(monkeypatch):
    monkeypatch.setenv("QDRANT_TIMEOUT_SECONDS", "12")
    captured = {}
    client = FakeQdrantClient({}, set(_ALL_ISCO_COLLECTIONS))
    monkeypatch.setattr(hs_module, "QdrantClient", MagicMock(side_effect=lambda **kw: captured.update(kw) or client))
    monkeypatch.setattr(hs_module, "SentenceTransformer", MagicMock(side_effect=lambda *a, **kw: FakeEmbedder()))
    HierarchicalISCOStore()
    assert captured["timeout"] == 12


# ---------------------------------------------------------------------------
# 6. Timeout/query exceptions never appear as hierarchical success
# ---------------------------------------------------------------------------

def test_query_exception_never_appears_as_hierarchical_success(monkeypatch):
    """A Qdrant query exception (e.g. a bounded timeout firing) must
    degrade to zero hits at that stage -- never raise out of search(),
    and never fabricate a hierarchical success."""
    store, client = _make_store(monkeypatch, {}, client_cls=_RaisingQdrantClient)
    result = store.search("software developer")
    assert result.code == ""              # never a fabricated success
    assert result.fallback_used is True   # honestly labelled unavailable, not mislabelled hierarchical
