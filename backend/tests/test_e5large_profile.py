"""
Tests for the E5LARGE_PROFILE addition (2026-08-24): a new, opt-in ISCO-08
catalogue profile that reuses the exact same verified official records but
embeds them with intfloat/multilingual-e5-large (1024-dim) instead of the
project's long-standing -small default (384-dim).

Covers:
- backend/rag/official_isco08_catalogue.py: PROFILE_COLLECTION_NAMES /
  PROFILE_EMBEDDING_CONFIG / embedding_config_for_profile() -- confirms
  every pre-existing profile is byte-identical to before, and the new
  profile resolves correctly.
- backend/rag/hierarchical_store.py: HierarchicalISCOStore resolves the
  right embedding model per profile.

Fully offline -- no live Qdrant, no real model download (SentenceTransformer
construction is monkeypatched).
"""

from unittest.mock import MagicMock, patch

import pytest

from backend.rag.official_isco08_catalogue import (
    DEFAULT_PROFILE,
    E5LARGE_PROFILE,
    PROFILE_COLLECTION_NAMES,
    embedding_config_for_profile,
)


class TestEmbeddingConfigForProfile:
    def test_default_profile_unchanged(self):
        assert embedding_config_for_profile(DEFAULT_PROFILE) == ("intfloat/multilingual-e5-small", 384)

    def test_legacy_profile_unchanged(self):
        assert embedding_config_for_profile("legacy") == ("intfloat/multilingual-e5-small", 384)

    def test_unknown_profile_falls_back_to_default_embedding(self):
        """A profile with no PROFILE_EMBEDDING_CONFIG entry -- including any
        future profile added without updating that dict -- gets the
        project's long-standing default rather than raising or returning
        something undefined."""
        assert embedding_config_for_profile("some_future_profile") == ("intfloat/multilingual-e5-small", 384)

    def test_e5large_profile_resolves_to_e5_large(self):
        assert embedding_config_for_profile(E5LARGE_PROFILE) == ("intfloat/multilingual-e5-large", 1024)


class TestE5LargeProfileCollectionNames:
    def test_registered_with_distinct_names_from_default_profile(self):
        assert E5LARGE_PROFILE in PROFILE_COLLECTION_NAMES
        default_names = set(PROFILE_COLLECTION_NAMES[DEFAULT_PROFILE].values())
        e5large_names = set(PROFILE_COLLECTION_NAMES[E5LARGE_PROFILE].values())
        assert default_names.isdisjoint(e5large_names)

    def test_has_all_five_roles(self):
        names = PROFILE_COLLECTION_NAMES[E5LARGE_PROFILE]
        assert set(names) == {"major", "submajor", "minor", "unit", "flat"}

    def test_collection_names_end_with_e5large_suffix(self):
        for name in PROFILE_COLLECTION_NAMES[E5LARGE_PROFILE].values():
            assert name.endswith("_e5large")


class TestHierarchicalStorePerProfileEmbedding:
    """HierarchicalISCOStore.__init__ resolves the embedding model from
    embedding_config_for_profile(profile) -- confirms existing profiles are
    unaffected and the new profile gets the larger model, without touching
    live Qdrant or downloading a real model."""

    def _make_store(self, profile, monkeypatch):
        from backend.rag.hierarchical_store import HierarchicalISCOStore

        fake_client = MagicMock()
        fake_client.get_collections.return_value = MagicMock(collections=[])
        monkeypatch.setattr(
            "backend.rag.hierarchical_store.QdrantClient",
            lambda **kw: fake_client,
        )
        fake_model = MagicMock()
        with patch("backend.rag.hierarchical_store.SentenceTransformer", return_value=fake_model) as ctor:
            store = HierarchicalISCOStore(profile=profile)
        return store, ctor

    def test_legacy_profile_uses_e5_small(self, monkeypatch):
        from backend.rag.hierarchical_store import LEGACY_PROFILE, MODEL_NAME
        store, ctor = self._make_store(LEGACY_PROFILE, monkeypatch)
        ctor.assert_called_once_with(MODEL_NAME)
        assert store.embedding_model_identity == MODEL_NAME
        assert store.embedding_vector_dim == 384

    def test_official_profile_uses_e5_small(self, monkeypatch):
        from backend.rag.hierarchical_store import OFFICIAL_PROFILE_ILO2021_V1, MODEL_NAME
        store, ctor = self._make_store(OFFICIAL_PROFILE_ILO2021_V1, monkeypatch)
        ctor.assert_called_once_with(MODEL_NAME)
        assert store.embedding_model_identity == MODEL_NAME

    def test_e5large_profile_uses_e5_large(self, monkeypatch):
        store, ctor = self._make_store(E5LARGE_PROFILE, monkeypatch)
        ctor.assert_called_once_with("intfloat/multilingual-e5-large")
        assert store.embedding_model_identity == "intfloat/multilingual-e5-large"
        assert store.embedding_vector_dim == 1024
