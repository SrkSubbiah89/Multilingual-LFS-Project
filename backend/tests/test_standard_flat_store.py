"""
Tests for backend/rag/standard_hierarchical_store.py's StandardFlatStore
addition (2026-08-25) -- single-collection, direct (no parent-chain) flat
retrieval for ISIC Rev.4 classes / ISCED-F 2013 detailed fields. Added to
give these two standards the architecturally-identical counterpart to
ISCO-08's own BEST-TESTED configuration (flat retrieval + e5-large), since
ISCO-08's own hierarchical retrieval measurably underperformed its flat
retrieval (see CLAUDE.md's 10.35% vs 21.19% finding).

Fully offline -- fake Qdrant client + fake embedder, no live Qdrant, no
real model download.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from backend.rag.standard_hierarchical_store import (
    ISCEDF_FLAT_COLLECTIONS_BY_PROFILE,
    ISIC_FLAT_COLLECTIONS_BY_PROFILE,
    PROFILE_MODEL_CONFIG,
    StandardFlatStore,
    get_isic_flat_store,
    get_iscedf_flat_store,
)


class _FakeQdrantClient:
    def __init__(self, table, existing_collections):
        self.table = table
        self.existing_collections = set(existing_collections)
        self.calls = []

    def get_collections(self):
        return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in self.existing_collections])

    def query_points(self, collection_name, query, limit, with_payload):
        self.calls.append((collection_name, limit))
        rows = self.table.get(collection_name, [])
        points = [
            SimpleNamespace(score=score, payload={"code": code, "label_en": label_en, "label_ar": label_ar})
            for code, label_en, label_ar, score in rows[:limit]
        ]
        return SimpleNamespace(points=points)


class _FakeEmbedder:
    def __init__(self, dim=1024):
        self.dim = dim

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False, batch_size=1):
        return np.zeros((len(texts), self.dim))


_COLLECTION = "isic_rev4_classes_flat_e5large"
_HIT_TABLE = {
    _COLLECTION: [
        ("0111", "Growing of cereals", "", 0.83),
        ("0130", "Growing of vegetables", "", 0.61),
    ],
}


class TestCollectionNames:
    def test_isic_flat_collections_distinct_from_hierarchical(self):
        from backend.rag.standard_hierarchical_store import ISIC_COLLECTIONS_BY_PROFILE
        hierarchical_names = set(ISIC_COLLECTIONS_BY_PROFILE["e5_small"].values()) | set(
            ISIC_COLLECTIONS_BY_PROFILE["e5_large"].values()
        )
        flat_names = set(ISIC_FLAT_COLLECTIONS_BY_PROFILE.values())
        assert hierarchical_names.isdisjoint(flat_names)

    def test_iscedf_flat_collections_distinct_from_hierarchical(self):
        from backend.rag.standard_hierarchical_store import ISCEDF_COLLECTIONS_BY_PROFILE
        hierarchical_names = set(ISCEDF_COLLECTIONS_BY_PROFILE["e5_small"].values()) | set(
            ISCEDF_COLLECTIONS_BY_PROFILE["e5_large"].values()
        )
        flat_names = set(ISCEDF_FLAT_COLLECTIONS_BY_PROFILE.values())
        assert hierarchical_names.isdisjoint(flat_names)

    def test_all_three_profiles_present_for_both_standards(self):
        """"enriched_e5large" (added 2026-08-25, real official-source
        enrichment -- see backend/rag/official_source_enrichment.py) joins
        e5_small/e5_large."""
        assert set(ISIC_FLAT_COLLECTIONS_BY_PROFILE) == {"e5_small", "e5_large", "enriched_e5large"}
        assert set(ISCEDF_FLAT_COLLECTIONS_BY_PROFILE) == {"e5_small", "e5_large", "enriched_e5large"}


class TestStandardFlatStoreSearch:
    def test_ready_true_and_direct_query_no_parent_filter(self):
        client = _FakeQdrantClient(_HIT_TABLE, {_COLLECTION})
        store = StandardFlatStore(
            standard="ISIC Rev.4", collection=_COLLECTION, client=client, embedder=_FakeEmbedder(),
        )
        assert store.ready is True

        result = store.search("cereal farming", top_k=5)

        assert result.ready is True
        assert result.unavailable_reason == ""
        assert result.code == "0111"
        assert result.confidence == 0.83
        assert result.hierarchy_path == ["0111"]
        assert result.stage_confidences == {"flat": 0.83}
        assert len(result.top_candidates) == 2
        # Exactly one direct call -- no collection_name/parent chaining.
        assert client.calls == [(_COLLECTION, 5)]

    def test_not_ready_when_collection_missing(self):
        client = _FakeQdrantClient({}, existing_collections=set())
        store = StandardFlatStore(standard="ISIC Rev.4", collection=_COLLECTION, client=client, embedder=_FakeEmbedder())
        assert store.ready is False

        result = store.search("anything")
        assert result.ready is False
        assert result.code == ""
        assert _COLLECTION in result.unavailable_reason
        assert client.calls == []  # never attempted a query

    def test_ready_but_zero_hits_is_explicit_not_fabricated(self):
        client = _FakeQdrantClient({}, existing_collections={_COLLECTION})
        store = StandardFlatStore(standard="ISIC Rev.4", collection=_COLLECTION, client=client, embedder=_FakeEmbedder())

        result = store.search("nothing matches this")
        assert result.ready is True
        assert result.code == ""
        assert "no candidates" in result.unavailable_reason.lower()

    def test_empty_text_is_explicit_not_a_query(self):
        client = _FakeQdrantClient(_HIT_TABLE, {_COLLECTION})
        store = StandardFlatStore(standard="ISIC Rev.4", collection=_COLLECTION, client=client, embedder=_FakeEmbedder())

        result = store.search("   ")
        assert result.ready is True
        assert result.unavailable_reason == "empty input text"
        assert client.calls == []

    def test_hitl_required_reflects_threshold(self):
        low_conf_table = {_COLLECTION: [("0130", "Growing of vegetables", "", 0.40)]}
        client = _FakeQdrantClient(low_conf_table, {_COLLECTION})
        store = StandardFlatStore(
            standard="ISIC Rev.4", collection=_COLLECTION, hitl_threshold=0.70,
            client=client, embedder=_FakeEmbedder(),
        )
        result = store.search("something")
        assert result.confidence == 0.40
        assert result.hitl_required is True


class TestFactories:
    def test_get_isic_flat_store_resolves_e5_large_collection_and_model(self):
        stages_collection = ISIC_FLAT_COLLECTIONS_BY_PROFILE["e5_large"]
        client = _FakeQdrantClient({}, {stages_collection})
        store = get_isic_flat_store(client=client, profile="e5_large")
        assert store.collection == stages_collection
        assert store._model_name == PROFILE_MODEL_CONFIG["e5_large"][0]

    def test_get_iscedf_flat_store_resolves_e5_large_collection_and_model(self):
        collection = ISCEDF_FLAT_COLLECTIONS_BY_PROFILE["e5_large"]
        client = _FakeQdrantClient({}, {collection})
        store = get_iscedf_flat_store(client=client, profile="e5_large")
        assert store.collection == collection
        assert store._model_name == PROFILE_MODEL_CONFIG["e5_large"][0]

    def test_injected_client_never_touches_production_singleton(self):
        collection = ISIC_FLAT_COLLECTIONS_BY_PROFILE["e5_small"]
        client_a = _FakeQdrantClient({}, {collection})
        client_b = _FakeQdrantClient({}, {collection})
        store_a = get_isic_flat_store(client=client_a, profile="e5_small")
        store_b = get_isic_flat_store(client=client_b, profile="e5_small")
        assert store_a is not store_b
