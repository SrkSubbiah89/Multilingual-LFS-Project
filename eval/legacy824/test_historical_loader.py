"""
Tests for eval/legacy824/historical_loader.py (Task 39): proves the
worktree-scoped loader restores sys.modules afterward (no permanent
contamination of the literal `backend`/`backend.rag`/`backend.llm`
names), and that the loaded classifier is registered only under a
private module name, never under `backend.agents.isco_classifier`.

Skipped gracefully if the Task 39 detached worktree is not present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.legacy824.historical_loader import load_historical_isco_classifier  # noqa: E402

WORKTREE_PATH = Path("C:/task39_legacy824_worktree")

pytestmark = pytest.mark.skipif(
    not WORKTREE_PATH.exists(), reason="Task 39 detached worktree not present in this environment"
)


def test_loader_does_not_leave_backend_bound_in_sys_modules():
    had_backend_before = "backend" in sys.modules
    had_backend_rag_before = "backend.rag" in sys.modules
    had_backend_llm_before = "backend.llm" in sys.modules

    load_historical_isco_classifier(WORKTREE_PATH)

    assert ("backend" in sys.modules) == had_backend_before
    assert ("backend.rag" in sys.modules) == had_backend_rag_before
    assert ("backend.llm" in sys.modules) == had_backend_llm_before


def test_loaded_classifier_registered_only_under_private_name():
    """The loaded historical module is registered under the private
    name, and never overwrites or aliases whatever this repository's
    own `backend.agents.isco_classifier` (current tree) may already be
    bound to in sys.modules -- which is legitimately present when this
    test runs alongside the full suite, since many unrelated test files
    import the current classifier directly. This test proves no
    interference, not that the current-tree name is absent."""
    current_tree_module_before = sys.modules.get("backend.agents.isco_classifier")

    loaded = load_historical_isco_classifier(WORKTREE_PATH)

    assert "_legacy824_isco_classifier" in sys.modules
    assert sys.modules["_legacy824_isco_classifier"] is loaded.isco_classifier
    assert sys.modules.get("backend.agents.isco_classifier") is current_tree_module_before
    if current_tree_module_before is not None:
        assert loaded.isco_classifier is not current_tree_module_before


def test_loaded_classifier_has_expected_public_names():
    loaded = load_historical_isco_classifier(WORKTREE_PATH)
    assert hasattr(loaded.isco_classifier, "ISCOClassifier")
    assert hasattr(loaded.isco_classifier, "ISCOClassification")
    assert hasattr(loaded.vector_store, "get_vector_store")
    assert hasattr(loaded.vector_store, "COLLECTION_NAME")
    assert loaded.vector_store.COLLECTION_NAME == "isco_occupations"
    assert loaded.vector_store.MODEL_NAME == "intfloat/multilingual-e5-large"
    assert hasattr(loaded.llm_client, "get_llm")
    assert loaded.llm_client.MODEL_CRITICAL == "anthropic/claude-3-5-sonnet-20241022"
    assert loaded.llm_client._TEMP_CRITICAL == 0.0
