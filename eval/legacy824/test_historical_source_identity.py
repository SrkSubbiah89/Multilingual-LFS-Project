"""
Tests for eval/legacy824/historical_source_identity.py (Task 39).

Fully hermetic for scenario 1: the git-show function is injected as a
fake, so no real git/subprocess call happens. Scenario 13 (worktree
byte-identity) is the one deliberate exception -- it reads real files
from the real detached worktree and calls real `git show`, since that
is exactly what it must prove; it is skipped gracefully if the
worktree is not present in the current environment (e.g. a fresh
clone that has not run Task 39's setup step).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.legacy824 import LEGACY_SHA  # noqa: E402
from eval.legacy824.historical_source_identity import (  # noqa: E402
    EXPECTED_SNIPPETS,
    ISCO_CLASSIFIER_PATH,
    LLM_CLIENT_PATH,
    VECTOR_STORE_PATH,
    verify_historical_source_identity,
)

WORKTREE_PATH = Path("C:/task39_legacy824_worktree")

_REAL_SOURCE = {
    ISCO_CLASSIFIER_PATH: (
        "_HIGH_CONFIDENCE_THRESHOLD = 0.92\n"
        "candidates = self._store.search(job_title, top_k=top_k)\n"
        "if candidates[0].confidence >= _HIGH_CONFIDENCE_THRESHOLD:\n"
        'method="semantic",\n'
        'method="llm_ranked",\n'
        "self._llm  = get_llm(TaskType.CRITICAL)\n"
        '{"selected_code": "<isco_code>", "reasoning": "<one sentence>"}\n'
        "if selected_code in code_map:\n"
        "Fallback to top semantic match (LLM response could not be parsed).\n"
        "top_k: int = 5,\n"
    ),
    VECTOR_STORE_PATH: (
        'COLLECTION_NAME = "isco_occupations"\n'
        'MODEL_NAME      = "intfloat/multilingual-e5-large"\n'
        "TOP_K_DEFAULT   = 5\n"
    ),
    LLM_CLIENT_PATH: (
        'MODEL_CRITICAL = "anthropic/claude-3-5-sonnet-20241022"\n'
        "_TEMP_CRITICAL = 0.0\n"
    ),
}


def _fake_git_show_matching(sha: str, path: str) -> str:
    return _REAL_SOURCE[path]


def test_matching_source_passes_every_check():
    report = verify_historical_source_identity(git_show_fn=_fake_git_show_matching)
    assert report["all_ok"] is True
    for path, entry in report["per_file"].items():
        assert entry["ok"] is True, (path, entry["missing_snippets"])


@pytest.mark.parametrize(
    "path,removed_snippet",
    [
        (ISCO_CLASSIFIER_PATH, "_HIGH_CONFIDENCE_THRESHOLD = 0.92"),
        (ISCO_CLASSIFIER_PATH, "top_k: int = 5,"),
        (ISCO_CLASSIFIER_PATH, 'method="llm_ranked",'),
        (ISCO_CLASSIFIER_PATH, "Fallback to top semantic match (LLM response could not be parsed)."),
        (VECTOR_STORE_PATH, 'MODEL_NAME      = "intfloat/multilingual-e5-large"'),
        (VECTOR_STORE_PATH, "TOP_K_DEFAULT   = 5"),
        (LLM_CLIENT_PATH, 'MODEL_CRITICAL = "anthropic/claude-3-5-sonnet-20241022"'),
        (LLM_CLIENT_PATH, "_TEMP_CRITICAL = 0.0"),
    ],
)
def test_a_changed_or_removed_value_is_rejected(path, removed_snippet):
    """Scenario 1: a threshold/top-k/model-route/temperature/method-label/
    fallback-behavior change in the historical source must be caught."""
    tampered_source = {**_REAL_SOURCE, path: _REAL_SOURCE[path].replace(removed_snippet, "")}

    def fake_git_show(sha: str, p: str) -> str:
        return tampered_source[p]

    report = verify_historical_source_identity(git_show_fn=fake_git_show)
    assert report["all_ok"] is False
    assert report["per_file"][path]["ok"] is False
    assert removed_snippet in report["per_file"][path]["missing_snippets"]


def test_git_show_error_is_reported_not_raised():
    def failing_git_show(sha: str, path: str) -> str:
        raise subprocess.CalledProcessError(1, ["git", "show"])

    report = verify_historical_source_identity(git_show_fn=failing_git_show)
    assert report["all_ok"] is False
    for entry in report["per_file"].values():
        assert entry["ok"] is False
        assert "error" in entry


@pytest.mark.skipif(not WORKTREE_PATH.exists(), reason="Task 39 detached worktree not present in this environment")
def test_worktree_source_byte_identical_to_git_show():
    """Scenario 13: the detached worktree's own files must be unchanged
    relative to LEGACY_SHA -- proves the worktree was never edited.

    Uses `git diff --quiet <sha> -- <path>` (run inside the worktree)
    rather than a raw byte comparison against `git show`: on Windows,
    a worktree checkout applies core.autocrlf line-ending normalization
    (LF -> CRLF) while `git show` returns the raw committed blob (LF),
    so a naive byte comparison reports a false difference that is pure
    checkout-encoding noise, not a real edit. `git diff` applies the
    same normalization to both sides before comparing, so it cannot be
    fooled by that noise the way a manual byte comparison can -- see
    the identical fix already applied in Task 34.1's own check for the
    same reason."""
    for rel_path in EXPECTED_SNIPPETS:
        worktree_file = WORKTREE_PATH / rel_path
        assert worktree_file.exists(), f"missing in worktree: {rel_path}"

        result = subprocess.run(
            ["git", "diff", "--quiet", LEGACY_SHA, "--", rel_path],
            cwd=str(WORKTREE_PATH),
        )
        assert result.returncode == 0, f"{rel_path}: worktree differs from {LEGACY_SHA} (git diff exit {result.returncode})"
