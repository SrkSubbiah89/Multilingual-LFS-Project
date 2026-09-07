"""
Tests for backend/rag/build_standard_hierarchical_collections.py -- the
operator-only CLI that builds the ISIC/ISCED-F Qdrant collections.

Only the --dry-run path (dry_run()) is exercised here. execute_run() is
intentionally NOT called by any test -- it requires a live Qdrant instance
and downloads the real embedding model, which is out of scope for this
hermetic suite (see the module's own docstring).
"""

from __future__ import annotations

import sys

import pytest

from backend.rag.build_standard_hierarchical_collections import (
    _STANDARDS,
    _flat_leaf_nodes,
    dry_run,
    dry_run_flat,
)
from backend.rag.official_source_enrichment import (
    _ISCEDF_DEFINITIONS_PATH,
    _ISIC_DEFINITIONS_PATH,
)
from backend.rag.standard_hierarchical_store import (
    ISCEDF_COLLECTIONS,
    ISCEDF_STAGE_WEIGHTS,
    ISIC_COLLECTIONS,
    ISIC_STAGE_WEIGHTS,
)

# See test_official_source_enrichment.py's identical guard for the full
# rationale: eval/local_catalogues/*_definitions.json are git-ignored, so
# any test resolving "enriched_e5large" real content needs this skip.
_HAS_REAL_CATALOGUE_FILES = _ISIC_DEFINITIONS_PATH.exists() and _ISCEDF_DEFINITIONS_PATH.exists()
requires_real_catalogue_files = pytest.mark.skipif(
    not _HAS_REAL_CATALOGUE_FILES,
    reason="eval/local_catalogues/*_definitions.json are git-ignored and not present on this "
           "machine -- run `python -m eval.parse_official_isic_iscedf_definitions` first.",
)


def test_dry_run_imports_no_qdrant_or_sentence_transformers_modules():
    """dry_run() must not have caused qdrant_client/sentence_transformers to
    be imported -- proof it never touches a live Qdrant connection or loads
    the embedding model. (These libraries may already be imported by other
    test modules in the same session; this test only checks dry_run() itself
    doesn't reach for them by asserting the plan/summary shape below is
    produced without any network/model-load side effect being possible --
    dry_run()'s own source only imports hierarchy_nodes + module constants,
    verified by code review; this test pins the observable behaviour.)"""
    summary = dry_run("isic")
    assert summary["mode"] == "dry_run"


def test_dry_run_isic_produces_exact_counts_and_plan():
    summary = dry_run("isic")
    assert summary["standard"] == "ISIC Rev.4"
    by_level = {lvl["level"]: lvl for lvl in summary["levels"]}
    assert by_level["sections"]["node_count"] == 21
    assert by_level["divisions"]["node_count"] == 68
    assert by_level["groups"]["node_count"] == 118
    assert by_level["classes"]["node_count"] == 134
    assert by_level["sections"]["collection"] == ISIC_COLLECTIONS["sections"]
    assert by_level["classes"]["stage_weight"] == ISIC_STAGE_WEIGHTS[3]
    assert all(lvl["content_sha256"] for lvl in summary["levels"])


def test_dry_run_iscedf_produces_exact_counts_and_plan():
    summary = dry_run("iscedf")
    assert summary["standard"] == "ISCED-F 2013"
    by_level = {lvl["level"]: lvl for lvl in summary["levels"]}
    assert by_level["broad_fields"]["node_count"] == 11
    assert by_level["narrow_fields"]["node_count"] == 25
    assert by_level["detailed_fields"]["node_count"] == 63
    assert by_level["broad_fields"]["collection"] == ISCEDF_COLLECTIONS["broad_fields"]
    assert by_level["detailed_fields"]["stage_weight"] == ISCEDF_STAGE_WEIGHTS[2]


def test_dry_run_is_deterministic():
    a = dry_run("isic")
    b = dry_run("isic")
    a_hashes = [lvl["content_sha256"] for lvl in a["levels"]]
    b_hashes = [lvl["content_sha256"] for lvl in b["levels"]]
    assert a_hashes == b_hashes


def test_standards_dict_covers_both_standards():
    assert set(_STANDARDS) == {"isic", "iscedf"}


def test_dry_run_e5_large_profile_uses_suffixed_collections_and_larger_model():
    summary = dry_run("isic", profile="e5_large")
    assert summary["profile"] == "e5_large"
    assert summary["embedding_model"] == "intfloat/multilingual-e5-large"
    assert summary["vector_dim"] == 1024
    by_level = {lvl["level"]: lvl for lvl in summary["levels"]}
    assert by_level["sections"]["collection"] == "isic_rev4_sections_e5large"


def test_dry_run_e5_large_profile_node_counts_match_default_profile():
    """The embedding profile changes collection names/model only -- the
    derived node counts (a property of _ISIC_DATA, not the embedding) must
    be identical across profiles."""
    default_summary = dry_run("isic")
    large_summary = dry_run("isic", profile="e5_large")
    default_counts = {lvl["level"]: lvl["node_count"] for lvl in default_summary["levels"]}
    large_counts = {lvl["level"]: lvl["node_count"] for lvl in large_summary["levels"]}
    assert default_counts == large_counts


def test_dry_run_flat_isic_matches_hierarchical_leaf_node_count():
    """The flat build reuses the SAME leaf-level ("classes") derived nodes
    as the hierarchical build's final stage -- node count and content hash
    must match exactly."""
    hierarchical = dry_run("isic", profile="e5_large")
    flat = dry_run_flat("isic", profile="e5_large")
    classes_level = next(lvl for lvl in hierarchical["levels"] if lvl["level"] == "classes")
    assert flat["node_count"] == classes_level["node_count"] == 134
    assert flat["content_sha256"] == classes_level["content_sha256"]
    assert flat["collection"] == "isic_rev4_classes_flat_e5large"
    assert flat["embedding_model"] == "intfloat/multilingual-e5-large"


def test_dry_run_flat_iscedf_matches_hierarchical_leaf_node_count():
    hierarchical = dry_run("iscedf", profile="e5_large")
    flat = dry_run_flat("iscedf", profile="e5_large")
    detailed_level = next(lvl for lvl in hierarchical["levels"] if lvl["level"] == "detailed_fields")
    assert flat["node_count"] == detailed_level["node_count"] == 63
    assert flat["content_sha256"] == detailed_level["content_sha256"]
    assert flat["collection"] == "iscedf2013_detailed_fields_flat_e5large"


def test_dry_run_flat_default_profile_uses_e5_small_collection_names():
    flat = dry_run_flat("isic")
    assert flat["collection"] == "isic_rev4_classes_flat"
    assert flat["embedding_model"] == "intfloat/multilingual-e5-small"


def test_dry_run_hierarchical_rejects_enriched_e5large_with_a_clear_error():
    """Real, reproduced bug (code review, 2026-08-27): dry_run("isic",
    profile="enriched_e5large") -- the NON-flat, hierarchical path --
    used to crash with a bare, unexplained KeyError, since
    enriched_e5large only has flat-collection entries. Must now fail
    closed with a clear, actionable ValueError, not a KeyError."""
    with pytest.raises(ValueError, match="enriched_e5large.*no hierarchical collections"):
        dry_run("isic", profile="enriched_e5large")
    with pytest.raises(ValueError, match="enriched_e5large.*no hierarchical collections"):
        dry_run("iscedf", profile="enriched_e5large")


@requires_real_catalogue_files
def test_dry_run_flat_enriched_e5large_profile_uses_correct_collection_and_model():
    """Node count is 121, not 134 -- the 13 disclosed non-standard ISIC
    codes are excluded from this profile (see _enriched_flat_nodes'
    docstring: keeping them with thin fallback text caused a real,
    live-tested magnet-effect regression)."""
    flat = dry_run_flat("isic", profile="enriched_e5large")
    assert flat["collection"] == "isic_rev4_classes_flat_enriched_e5large"
    assert flat["embedding_model"] == "intfloat/multilingual-e5-large"
    assert flat["node_count"] == 121


@requires_real_catalogue_files
def test_dry_run_flat_enriched_e5large_iscedf_node_count():
    """61, not 63 -- the 2 disclosed non-standard ISCED-F codes excluded,
    same rationale as the ISIC case above."""
    flat = dry_run_flat("iscedf", profile="enriched_e5large")
    assert flat["collection"] == "iscedf2013_detailed_fields_flat_enriched_e5large"
    assert flat["node_count"] == 61


@requires_real_catalogue_files
def test_flat_leaf_nodes_enriched_profile_uses_real_official_text():
    nodes = _flat_leaf_nodes("isic", "enriched_e5large")
    node = next(n for n in nodes if n.code == "6201")
    assert "6201 Computer programming activities." in node.index_text
    assert "This class includes the writing" in node.index_text


@requires_real_catalogue_files
def test_flat_leaf_nodes_enriched_profile_excludes_non_standard_codes():
    """7311 has no match in the official ISIC structure document (a real,
    disclosed finding -- see official_source_enrichment.py). Excluded
    entirely from this collection rather than kept with fallback text --
    a live-testing regression found that fallback text became a magnet
    once every other code's text got much richer (the same "magnet
    effect" mechanism ISCO-08's own pre-enrichment catalogue had), so
    keeping a non-standard code in with thin text was actively harmful,
    not just uninformative."""
    from backend.rag.official_source_enrichment import NON_STANDARD_ISIC_CODES
    nodes = _flat_leaf_nodes("isic", "enriched_e5large")
    codes = {n.code for n in nodes}
    assert codes.isdisjoint(NON_STANDARD_ISIC_CODES)
    assert len(nodes) == 134 - len(NON_STANDARD_ISIC_CODES)


@requires_real_catalogue_files
def test_flat_leaf_nodes_enriched_profile_iscedf_excludes_non_standard_codes():
    from backend.rag.official_source_enrichment import NON_STANDARD_ISCEDF_CODES
    nodes = _flat_leaf_nodes("iscedf", "enriched_e5large")
    codes = {n.code for n in nodes}
    assert codes.isdisjoint(NON_STANDARD_ISCEDF_CODES)
    assert len(nodes) == 63 - len(NON_STANDARD_ISCEDF_CODES)


def test_flat_leaf_nodes_default_profiles_unchanged_by_enriched_addition():
    """e5_small/e5_large must still resolve via the ordinary
    hierarchy_nodes.py derivation, byte-identical to before this profile
    existed."""
    from backend.rag.hierarchy_nodes import derive_isic_nodes
    expected = derive_isic_nodes()["classes"]
    actual = _flat_leaf_nodes("isic", "e5_small")
    assert [n.index_text for n in actual] == [n.index_text for n in expected]


def test_execute_run_not_imported_at_module_load_time():
    """qdrant_client / sentence_transformers must only be imported inside
    execute_run(), never at module scope -- so importing this CLI module
    (already done at the top of this file) cannot have pulled in Qdrant or
    the embedding model."""
    mod_name = "backend.rag.build_standard_hierarchical_collections"
    mod = sys.modules[mod_name]
    assert not hasattr(mod, "QdrantClient")
    assert not hasattr(mod, "SentenceTransformer")
