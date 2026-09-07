"""
Tests for the ISIC/ISCED-F e5-large profile addition (2026-08-25): the same
additive pattern already used for ISCO-08's E5LARGE_PROFILE
(backend/tests/test_e5large_profile.py) -- a second, opt-in embedding
profile that reuses the exact same derived hierarchy nodes but embeds them
with intfloat/multilingual-e5-large (1024-dim) instead of the project's
long-standing -small default (384-dim), in separately-named Qdrant
collections. Confirms the pre-existing default profile is byte-identical to
before this change, and the new profile resolves correctly.

Fully offline -- no live Qdrant, no real model download (SentenceTransformer
construction is monkeypatched where a store is constructed at all).
"""

from unittest.mock import MagicMock

import pytest

from backend.rag.standard_hierarchical_store import (
    ISCEDF_COLLECTIONS,
    ISCEDF_COLLECTIONS_BY_PROFILE,
    ISIC_COLLECTIONS,
    ISIC_COLLECTIONS_BY_PROFILE,
    PROFILE_MODEL_CONFIG,
    isic_stages,
    iscedf_stages,
)


class TestProfileModelConfig:
    def test_e5_small_is_the_existing_default_model(self):
        assert PROFILE_MODEL_CONFIG["e5_small"] == ("intfloat/multilingual-e5-small", 384)

    def test_e5_large_resolves_to_the_larger_model(self):
        assert PROFILE_MODEL_CONFIG["e5_large"] == ("intfloat/multilingual-e5-large", 1024)


class TestBackCompatAliasesUnchanged:
    """ISIC_COLLECTIONS / ISCEDF_COLLECTIONS are read directly by
    backend/tests/test_build_standard_hierarchical_collections.py and by
    build_standard_hierarchical_collections.py's own dry_run() default --
    they must still equal exactly the e5_small profile's names."""

    def test_isic_collections_alias_matches_e5_small_profile(self):
        assert ISIC_COLLECTIONS == ISIC_COLLECTIONS_BY_PROFILE["e5_small"]

    def test_iscedf_collections_alias_matches_e5_small_profile(self):
        assert ISCEDF_COLLECTIONS == ISCEDF_COLLECTIONS_BY_PROFILE["e5_small"]

    def test_isic_default_names_unchanged(self):
        assert ISIC_COLLECTIONS == {
            "sections": "isic_rev4_sections",
            "divisions": "isic_rev4_divisions",
            "groups": "isic_rev4_groups",
            "classes": "isic_rev4_classes",
        }

    def test_iscedf_default_names_unchanged(self):
        assert ISCEDF_COLLECTIONS == {
            "broad_fields": "iscedf2013_broad_fields",
            "narrow_fields": "iscedf2013_narrow_fields",
            "detailed_fields": "iscedf2013_detailed_fields",
        }


class TestE5LargeCollectionNames:
    def test_isic_e5large_names_distinct_from_default(self):
        default_names = set(ISIC_COLLECTIONS_BY_PROFILE["e5_small"].values())
        e5large_names = set(ISIC_COLLECTIONS_BY_PROFILE["e5_large"].values())
        assert default_names.isdisjoint(e5large_names)

    def test_iscedf_e5large_names_distinct_from_default(self):
        default_names = set(ISCEDF_COLLECTIONS_BY_PROFILE["e5_small"].values())
        e5large_names = set(ISCEDF_COLLECTIONS_BY_PROFILE["e5_large"].values())
        assert default_names.isdisjoint(e5large_names)

    def test_isic_e5large_names_end_with_suffix(self):
        for name in ISIC_COLLECTIONS_BY_PROFILE["e5_large"].values():
            assert name.endswith("_e5large")

    def test_iscedf_e5large_names_end_with_suffix(self):
        for name in ISCEDF_COLLECTIONS_BY_PROFILE["e5_large"].values():
            assert name.endswith("_e5large")

    def test_both_profiles_have_same_level_keys(self):
        assert set(ISIC_COLLECTIONS_BY_PROFILE["e5_small"]) == set(ISIC_COLLECTIONS_BY_PROFILE["e5_large"])
        assert set(ISCEDF_COLLECTIONS_BY_PROFILE["e5_small"]) == set(ISCEDF_COLLECTIONS_BY_PROFILE["e5_large"])


class TestStagesPerProfile:
    def test_isic_stages_default_profile_matches_default_collections(self):
        stages = isic_stages()
        assert [s.collection for s in stages] == [
            ISIC_COLLECTIONS["sections"], ISIC_COLLECTIONS["divisions"],
            ISIC_COLLECTIONS["groups"], ISIC_COLLECTIONS["classes"],
        ]

    def test_isic_stages_e5_large_profile_uses_suffixed_collections(self):
        stages = isic_stages("e5_large")
        for s in stages:
            assert s.collection.endswith("_e5large")

    def test_iscedf_stages_default_profile_matches_default_collections(self):
        stages = iscedf_stages()
        assert [s.collection for s in stages] == [
            ISCEDF_COLLECTIONS["broad_fields"], ISCEDF_COLLECTIONS["narrow_fields"],
            ISCEDF_COLLECTIONS["detailed_fields"],
        ]

    def test_iscedf_stages_e5_large_profile_uses_suffixed_collections(self):
        stages = iscedf_stages("e5_large")
        for s in stages:
            assert s.collection.endswith("_e5large")

    def test_stage_weights_identical_across_profiles(self):
        """Only collection names/embedding model change per profile -- the
        (undocumented-as-tuned) stage weights are a property of the
        hierarchy shape, not the embedding profile, and must stay identical
        so a later accuracy comparison isolates the embedding-model effect."""
        default_weights = [s.weight for s in isic_stages("e5_small")]
        large_weights = [s.weight for s in isic_stages("e5_large")]
        assert default_weights == large_weights

    def test_isic_stages_rejects_enriched_e5large_with_clear_error_not_keyerror(self):
        """Real, reproduced bug (code review, 2026-08-27): enriched_e5large
        is a valid PROFILE_MODEL_CONFIG key but has no hierarchical
        collection entries (only flat ones) -- must fail with a clear,
        actionable ValueError, never a bare KeyError."""
        with pytest.raises(ValueError, match="no hierarchical collections"):
            isic_stages("enriched_e5large")

    def test_iscedf_stages_rejects_enriched_e5large_with_clear_error_not_keyerror(self):
        with pytest.raises(ValueError, match="no hierarchical collections"):
            iscedf_stages("enriched_e5large")


class TestStoreConstructionPerProfile:
    """StandardHierarchicalStore.__init__ resolves the embedding model from
    the injected model_name -- confirms the default is unaffected and the
    new profile wires the larger model, without touching live Qdrant or
    downloading a real model."""

    def _fake_client(self, known_collections):
        client = MagicMock()
        client.get_collections.return_value = MagicMock(
            collections=[MagicMock(name=n) for n in known_collections]
        )
        # MagicMock(name=...) does not set the .name attribute the way a
        # real object would -- set it explicitly.
        for m, n in zip(client.get_collections.return_value.collections, known_collections):
            m.name = n
        return client

    def test_default_store_uses_e5_small_model_name(self):
        from backend.rag.standard_hierarchical_store import StandardHierarchicalStore
        stages = isic_stages("e5_small")
        client = self._fake_client([s.collection for s in stages])
        store = StandardHierarchicalStore(standard="ISIC Rev.4", stages=stages, client=client)
        assert store._model_name == "intfloat/multilingual-e5-small"

    def test_e5_large_store_uses_e5_large_model_name(self):
        from backend.rag.standard_hierarchical_store import StandardHierarchicalStore
        stages = isic_stages("e5_large")
        client = self._fake_client([s.collection for s in stages])
        store = StandardHierarchicalStore(
            standard="ISIC Rev.4", stages=stages, client=client,
            model_name="intfloat/multilingual-e5-large",
        )
        assert store._model_name == "intfloat/multilingual-e5-large"

    def test_e5_large_store_ready_false_when_suffixed_collections_absent(self):
        """Confirms the two profiles' collections are genuinely independent
        namespaces: a client that only knows about the e5_small collections
        reports NOT ready for the e5_large stage set."""
        from backend.rag.standard_hierarchical_store import StandardHierarchicalStore
        small_stages = isic_stages("e5_small")
        client = self._fake_client([s.collection for s in small_stages])
        large_stages = isic_stages("e5_large")
        store = StandardHierarchicalStore(standard="ISIC Rev.4", stages=large_stages, client=client)
        assert store.ready is False
        assert "isic_rev4_sections_e5large" in store._unavailable_reason


class TestGetStoreFactoriesAcceptProfile:
    def test_get_isic_store_with_injected_client_honours_profile(self):
        from backend.rag.standard_hierarchical_store import get_isic_hierarchical_store
        stages = isic_stages("e5_large")
        client = MagicMock()
        client.get_collections.return_value = MagicMock(
            collections=[MagicMock(name=s.collection) for s in stages]
        )
        for m, s in zip(client.get_collections.return_value.collections, stages):
            m.name = s.collection
        store = get_isic_hierarchical_store(client=client, profile="e5_large")
        assert store._model_name == "intfloat/multilingual-e5-large"
        assert store.stages[0].collection.endswith("_e5large")

    def test_get_iscedf_store_with_injected_client_honours_profile(self):
        from backend.rag.standard_hierarchical_store import get_iscedf_hierarchical_store
        stages = iscedf_stages("e5_large")
        client = MagicMock()
        client.get_collections.return_value = MagicMock(
            collections=[MagicMock(name=s.collection) for s in stages]
        )
        for m, s in zip(client.get_collections.return_value.collections, stages):
            m.name = s.collection
        store = get_iscedf_hierarchical_store(client=client, profile="e5_large")
        assert store._model_name == "intfloat/multilingual-e5-large"
        assert store.stages[0].collection.endswith("_e5large")
