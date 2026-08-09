"""
Tests for backend/rag/hierarchical_store.py's Task 21 profile support
(official_ilo2021_v1 vs legacy). Hermetic: FakeQdrantClient/FakeEmbedder
stand in for qdrant_client.QdrantClient/SentenceTransformer -- no live
Qdrant connection, no embedding-model load, no network call anywhere in
this file. No WISCO artifact is read.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import backend.rag.hierarchical_store as hs_module
from backend.rag.hierarchical_store import (
    LEGACY_PROFILE,
    OFFICIAL_PROFILE_ILO2021_V1,
    HierarchicalISCOStore,
    UnknownISCOCatalogueProfileError,
)

_OFFICIAL_COLLECTIONS = {
    "isco08_major_groups_ilo2021_v1", "isco08_submajor_groups_ilo2021_v1",
    "isco08_minor_groups_ilo2021_v1", "isco08_unit_groups_ilo2021_v1",
    "isco08_unit_groups_flat_ilo2021_v1",
}
_LEGACY_COLLECTIONS = {
    "isco08_major_groups", "isco08_submajor_groups",
    "isco08_minor_groups", "isco08_unit_groups", "isco_occupations",
}


class FakeQdrantClient:
    def __init__(self, table, existing_collections):
        self.table = table
        self.existing_collections = set(existing_collections)
        self.calls = []

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


class FakeEmbedder:
    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False, batch_size=1):
        import numpy as np
        return np.zeros((len(texts), 384))


def _make_store(monkeypatch, table, existing, profile):
    client = FakeQdrantClient(table, existing)
    monkeypatch.setattr(hs_module, "QdrantClient", MagicMock(side_effect=lambda **kw: client))
    monkeypatch.setattr(hs_module, "SentenceTransformer", MagicMock(side_effect=lambda *a, **kw: FakeEmbedder()))
    store = HierarchicalISCOStore(profile=profile)
    return store, client


# ---------------------------------------------------------------------------
# Unknown profile fails closed
# ---------------------------------------------------------------------------

def test_unknown_profile_raises(monkeypatch):
    client = FakeQdrantClient({}, _LEGACY_COLLECTIONS)
    monkeypatch.setattr(hs_module, "QdrantClient", MagicMock(side_effect=lambda **kw: client))
    monkeypatch.setattr(hs_module, "SentenceTransformer", MagicMock(side_effect=lambda *a, **kw: FakeEmbedder()))
    with pytest.raises(UnknownISCOCatalogueProfileError, match="not a known ISCO-08 catalogue profile"):
        HierarchicalISCOStore(profile="not_a_real_profile")


# ---------------------------------------------------------------------------
# 13. Official hierarchical profile selects only versioned official
#     collection names
# ---------------------------------------------------------------------------

_OFFICIAL_SUCCESS_TABLE = {
    ("isco08_submajor_groups_ilo2021_v1", "2"): [("25", "ICT Professionals", "", 0.85)],
    ("isco08_minor_groups_ilo2021_v1", "25"): [("251", "Software Developers", "", 0.83)],
    ("isco08_unit_groups_ilo2021_v1", "251"): [("2512", "Software Developers", "", 0.80)],
}


def test_official_hierarchical_profile_selects_only_official_collections(monkeypatch):
    store, client = _make_store(monkeypatch, _OFFICIAL_SUCCESS_TABLE, _OFFICIAL_COLLECTIONS, OFFICIAL_PROFILE_ILO2021_V1)
    assert store.profile == OFFICIAL_PROFILE_ILO2021_V1
    assert store._hierarchical_ready is True

    trace: dict = {}
    result = store._hierarchical_search([0.0], top_k=5, major_hint="2", trace=trace)

    assert result is not None
    assert result.code == "2512"
    assert result.fallback_used is False
    queried_collections = {c for c, _p, _l in client.calls}
    assert queried_collections <= _OFFICIAL_COLLECTIONS
    assert not (queried_collections & _LEGACY_COLLECTIONS)


def test_legacy_profile_default_unaffected(monkeypatch):
    """Compatibility guard: omitting `profile=` (the default) must select
    exactly the pre-Task-21 legacy collection names."""
    store, _client = _make_store(monkeypatch, {}, _LEGACY_COLLECTIONS, LEGACY_PROFILE)
    assert store.profile == LEGACY_PROFILE
    assert store._col_flat == "isco_occupations"
    assert store._engine.stages[-1].collection == "isco08_unit_groups"


# ---------------------------------------------------------------------------
# 14. Official flat profile selects only the versioned four-digit
#     collection
# ---------------------------------------------------------------------------

def test_official_flat_fallback_uses_only_official_flat_collection(monkeypatch):
    # Hierarchical stages return nothing at all -> falls through to flat.
    table = {
        ("isco08_unit_groups_flat_ilo2021_v1", None): [("2512", "Software Developers", "", 0.9)],
    }
    store, client = _make_store(monkeypatch, table, _OFFICIAL_COLLECTIONS, OFFICIAL_PROFILE_ILO2021_V1)
    result = store._flat_search([0.0], top_k=5)
    assert result.code == "2512"
    assert result.fallback_used is True
    queried_collections = {c for c, _p, _l in client.calls}
    assert queried_collections == {"isco08_unit_groups_flat_ilo2021_v1"}
    assert "isco_occupations" not in queried_collections


# ---------------------------------------------------------------------------
# 15. Official flat rejects a coarse candidate code rather than
#     returning it
# ---------------------------------------------------------------------------

def test_official_flat_rejects_non_four_digit_code(monkeypatch):
    table = {
        ("isco08_unit_groups_flat_ilo2021_v1", None): [("25", "ICT Professionals (coarse)", "", 0.9)],
    }
    store, _client = _make_store(monkeypatch, table, _OFFICIAL_COLLECTIONS, OFFICIAL_PROFILE_ILO2021_V1)
    result = store._flat_search([0.0], top_k=5)
    assert result.code == ""
    assert result.fallback_used is True  # _empty_result() sentinel


def test_legacy_flat_still_allows_coarse_code(monkeypatch):
    """Compatibility guard: the legacy profile's documented coarse-code
    behaviour (FLAT_BASELINE_COVERAGE_AUDIT.md) must be completely
    unaffected by the official-profile 4-digit check."""
    table = {
        ("isco_occupations", None): [("25", "ICT Professionals (coarse)", "", 0.9)],
    }
    store, _client = _make_store(monkeypatch, table, _LEGACY_COLLECTIONS, LEGACY_PROFILE)
    result = store._flat_search([0.0], top_k=5)
    assert result.code == "25"
    assert result.fallback_used is True


# ---------------------------------------------------------------------------
# 17. Official profile unavailable/error is explicit, never silently legacy
# ---------------------------------------------------------------------------

def test_official_profile_total_unavailability_is_explicit(monkeypatch):
    """Neither official hierarchical nor official flat collections exist
    -- must return the empty sentinel, never silently query/return a
    legacy collection result."""
    store, client = _make_store(monkeypatch, {}, set(), OFFICIAL_PROFILE_ILO2021_V1)
    assert store._hierarchical_ready is False
    assert store._flat_ready is False

    result = store.search("some job title")
    assert result.code == ""
    assert result.fallback_used is True
    queried_collections = {c for c, _p, _l in client.calls}
    assert not (queried_collections & _LEGACY_COLLECTIONS)
