"""
Tests for the label_en/title_en payload-key fix (found during Task 43's
live-reranker work): the official ISCO-08 profile's Qdrant collections
write `title_en` (never `label_en`), but every raw-payload read in
hierarchical_store.py/hierarchy_engine.py previously checked only
`label_en`, so official-profile results always carried a blank English
label. Fixed via `hierarchy_engine.extract_label_en()`.

Hermetic: FakeQdrantClient/FakeEmbedder stand in for
qdrant_client.QdrantClient/SentenceTransformer -- no live Qdrant
connection, no embedding-model load, no network call anywhere in this
file.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import backend.rag.hierarchical_store as hs_module
from backend.rag.hierarchical_store import HierarchicalISCOStore, OFFICIAL_PROFILE_ILO2021_V1
from backend.rag.hierarchy_engine import extract_label_en

_ALL_OFFICIAL_COLLECTIONS = {
    "isco08_major_groups_ilo2021_v1", "isco08_submajor_groups_ilo2021_v1",
    "isco08_minor_groups_ilo2021_v1", "isco08_unit_groups_ilo2021_v1",
    "isco08_unit_groups_flat_ilo2021_v1",
}


# ---------------------------------------------------------------------------
# extract_label_en -- unit level
# ---------------------------------------------------------------------------

def test_extract_label_en_prefers_label_en_when_present():
    """Legacy collections write label_en -- must be preserved exactly."""
    assert extract_label_en({"label_en": "Software Developers", "title_en": "should not be used"}) == "Software Developers"


def test_extract_label_en_falls_back_to_title_en_when_label_en_absent():
    """Official collections write title_en, never label_en."""
    assert extract_label_en({"title_en": "Software Developers"}) == "Software Developers"


def test_extract_label_en_falls_back_to_title_en_when_label_en_empty_string():
    assert extract_label_en({"label_en": "", "title_en": "Software Developers"}) == "Software Developers"


def test_extract_label_en_returns_empty_string_when_neither_key_present():
    assert extract_label_en({"code": "2512"}) == ""


# ---------------------------------------------------------------------------
# End-to-end: official-profile flat search now resolves real titles
# ---------------------------------------------------------------------------

class _OfficialProfileFakeQdrantClient:
    """Mirrors the official collection builder's real payload shape --
    `title_en`, never `label_en`/`label_ar` -- unlike the legacy-profile
    fixtures elsewhere, which include `label_en`/`label_ar` directly."""

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
            SimpleNamespace(score=score, payload={"code": code, "title_en": title_en})
            for code, title_en, score in rows[:limit]
        ]
        return SimpleNamespace(points=points)


class _FakeEmbedder:
    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False, batch_size=1):
        import numpy as np
        return np.zeros((len(texts), 384))


def _make_official_store(monkeypatch, table):
    client = _OfficialProfileFakeQdrantClient(table, _ALL_OFFICIAL_COLLECTIONS)
    monkeypatch.setattr(hs_module, "QdrantClient", MagicMock(side_effect=lambda **kw: client))
    monkeypatch.setattr(hs_module, "SentenceTransformer", MagicMock(side_effect=lambda *a, **kw: _FakeEmbedder()))
    store = HierarchicalISCOStore(profile=OFFICIAL_PROFILE_ILO2021_V1)
    return store, client


def test_official_flat_search_now_resolves_real_title_en_not_blank(monkeypatch):
    """Before the fix, this would have asserted label_en == '' (the
    actual, buggy behavior). After the fix, it resolves the real
    official title instead."""
    table = {
        ("isco08_unit_groups_flat_ilo2021_v1", None): [
            ("2512", "Software Developers", 0.91),
            ("2513", "Web Developers", 0.85),
            ("2511", "Systems Analysts", 0.80),
        ],
    }
    store, _client = _make_official_store(monkeypatch, table)
    result = store.search_flat_only("software engineer", top_k=3)
    assert result.code == "2512"
    assert result.label_en == "Software Developers"  # not ""
    assert result.top_candidates[0].label_en == "Software Developers"
    assert result.top_candidates[1].label_en == "Web Developers"


def test_official_flat_search_label_ar_stays_honestly_blank(monkeypatch):
    """The official catalogue has no Arabic title field at all -- unlike
    label_en, there is nothing to fall back to, and label_ar must stay
    an honest empty string rather than being fabricated."""
    table = {
        ("isco08_unit_groups_flat_ilo2021_v1", None): [("2512", "Software Developers", 0.91)],
    }
    store, _client = _make_official_store(monkeypatch, table)
    result = store.search_flat_only("software engineer", top_k=1)
    assert result.label_ar == ""
