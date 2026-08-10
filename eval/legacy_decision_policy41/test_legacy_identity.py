"""
Tests for eval/legacy_decision_policy41/legacy_identity.py (Task 41).

The negative-case tests are fully hermetic (synthetic source text only).
`test_real_legacy_sources_pass_identity_check` is the one place this task
legitimately runs `git show LEGACY_SHA:<path>` for real, exactly as the
task instructs ("Create static identity checks against `git show
LEGACY_SHA:...`") -- a plain read of already-fetched repository history,
not a live network/Qdrant/LLM/Docker operation.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.legacy_decision_policy41.legacy_identity import (  # noqa: E402
    ISCO_CLASSIFIER_PATH,
    LEGACY_SHA,
    LLM_CLIENT_PATH,
    LegacyIdentityMismatch,
    fetch_legacy_sources,
    verify_legacy_identity,
)

_GOOD_ISCO_SOURCE = """
def classify(self, job_title: str, context: str = "", top_k: int = 5) -> ISCOClassification:
    ...
_HIGH_CONFIDENCE_THRESHOLD = 0.92
if candidates[0].confidence >= _HIGH_CONFIDENCE_THRESHOLD:
    return ISCOClassification(..., method="semantic")
return ISCOClassification(..., method="llm_ranked")
return (candidates[0], "Fallback to top semantic match (LLM response could not be parsed).")
if selected_code in code_map:
    return code_map[selected_code], reasoning
"""

_GOOD_LLM_CLIENT_SOURCE = """
CRITICAL = "critical"
MODEL_CRITICAL = "anthropic/claude-3-5-sonnet-20241022"
_TEMP_CRITICAL = 0.0
"""


def _good_sources():
    return {
        ISCO_CLASSIFIER_PATH: _GOOD_ISCO_SOURCE,
        LLM_CLIENT_PATH: _GOOD_LLM_CLIENT_SOURCE,
    }


def test_identity_check_passes_on_matching_synthetic_source():
    verify_legacy_identity(_good_sources())  # must not raise


def test_identity_check_rejects_altered_threshold():
    sources = _good_sources()
    sources[ISCO_CLASSIFIER_PATH] = sources[ISCO_CLASSIFIER_PATH].replace(
        "_HIGH_CONFIDENCE_THRESHOLD = 0.92", "_HIGH_CONFIDENCE_THRESHOLD = 0.85"
    )
    with pytest.raises(LegacyIdentityMismatch, match="0.92"):
        verify_legacy_identity(sources)


def test_identity_check_rejects_altered_candidate_count():
    sources = _good_sources()
    sources[ISCO_CLASSIFIER_PATH] = sources[ISCO_CLASSIFIER_PATH].replace(
        "top_k: int = 5", "top_k: int = 3"
    )
    with pytest.raises(LegacyIdentityMismatch, match="top_k"):
        verify_legacy_identity(sources)


def test_identity_check_rejects_altered_model_route():
    sources = _good_sources()
    sources[LLM_CLIENT_PATH] = sources[LLM_CLIENT_PATH].replace(
        "anthropic/claude-3-5-sonnet-20241022", "anthropic/claude-3-opus-20240229"
    )
    with pytest.raises(LegacyIdentityMismatch, match="claude-3-5-sonnet"):
        verify_legacy_identity(sources)


def test_identity_check_rejects_altered_temperature():
    sources = _good_sources()
    sources[LLM_CLIENT_PATH] = sources[LLM_CLIENT_PATH].replace(
        "_TEMP_CRITICAL = 0.0", "_TEMP_CRITICAL = 0.2"
    )
    with pytest.raises(LegacyIdentityMismatch, match="_TEMP_CRITICAL"):
        verify_legacy_identity(sources)


def test_identity_check_rejects_altered_fallback_condition():
    sources = _good_sources()
    sources[ISCO_CLASSIFIER_PATH] = sources[ISCO_CLASSIFIER_PATH].replace(
        "Fallback to top semantic match (LLM response could not be parsed).",
        "Falling back to first candidate.",
    )
    with pytest.raises(LegacyIdentityMismatch, match="Fallback to top semantic match"):
        verify_legacy_identity(sources)


def test_identity_check_reports_all_missing_snippets_not_just_first():
    sources = {ISCO_CLASSIFIER_PATH: "", LLM_CLIENT_PATH: ""}
    with pytest.raises(LegacyIdentityMismatch) as exc_info:
        verify_legacy_identity(sources)
    message = str(exc_info.value)
    assert "top_k: int = 5" in message
    assert "MODEL_CRITICAL" in message


# ---------------------------------------------------------------------------
# Real check: the actual historical source, fetched via a real (but
# read-only, no-network, no-live-service) `git show`.
# ---------------------------------------------------------------------------

def _real_git_show(commit: str, path: str) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=repo_root, capture_output=True, check=True,
    )
    return result.stdout.decode("utf-8")


def test_real_legacy_sources_pass_identity_check():
    sources = fetch_legacy_sources(_real_git_show)
    verify_legacy_identity(sources)  # must not raise -- proves LEGACY_SHA is unchanged


def test_legacy_sha_constant_matches_task_specification():
    assert LEGACY_SHA == "824fcf235ae2f8787706cf479a07620519c914de"
