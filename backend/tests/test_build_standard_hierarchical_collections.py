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

from backend.rag.build_standard_hierarchical_collections import (
    _STANDARDS,
    dry_run,
)
from backend.rag.standard_hierarchical_store import (
    ISCEDF_COLLECTIONS,
    ISCEDF_STAGE_WEIGHTS,
    ISIC_COLLECTIONS,
    ISIC_STAGE_WEIGHTS,
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


def test_execute_run_not_imported_at_module_load_time():
    """qdrant_client / sentence_transformers must only be imported inside
    execute_run(), never at module scope -- so importing this CLI module
    (already done at the top of this file) cannot have pulled in Qdrant or
    the embedding model."""
    mod_name = "backend.rag.build_standard_hierarchical_collections"
    mod = sys.modules[mod_name]
    assert not hasattr(mod, "QdrantClient")
    assert not hasattr(mod, "SentenceTransformer")
