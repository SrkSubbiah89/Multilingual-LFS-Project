"""
eval/legacy_decision_policy41/legacy_identity.py

Static identity checks against the real historical source at LEGACY_SHA,
proving the decision-policy facts this compatibility study (policy.py)
relies on have not silently changed. Mirrors Task 39's
`eval/legacy824/historical_source_identity.py` injectable-git-show
pattern: callers supply the already-fetched source text (or an
injectable `git_show(commit, path) -> str` function), so this module
never runs git itself and stays trivially hermetic to test.
"""

from __future__ import annotations

from typing import Callable

LEGACY_SHA = "824fcf235ae2f8787706cf479a07620519c914de"

ISCO_CLASSIFIER_PATH = "backend/agents/isco_classifier.py"
LLM_CLIENT_PATH = "backend/llm/llm_client.py"

# Each snippet is an exact, literal substring of the real file at
# LEGACY_SHA (verified via `git show LEGACY_SHA:<path>` -- see the Task 41
# final report for the exact commands and full quoted source).
EXPECTED_SNIPPETS: dict[str, list[str]] = {
    ISCO_CLASSIFIER_PATH: [
        "top_k: int = 5",
        "_HIGH_CONFIDENCE_THRESHOLD = 0.92",
        "candidates[0].confidence >= _HIGH_CONFIDENCE_THRESHOLD",
        'method="semantic"',
        'method="llm_ranked"',
        "Fallback to top semantic match (LLM response could not be parsed).",
        "if selected_code in code_map:",
    ],
    LLM_CLIENT_PATH: [
        'MODEL_CRITICAL = "anthropic/claude-3-5-sonnet-20241022"',
        "_TEMP_CRITICAL = 0.0",
        'CRITICAL = "critical"',
    ],
}


class LegacyIdentityMismatch(AssertionError):
    """Raised when the real historical source no longer contains an
    expected snippet -- i.e. a fact this compatibility study claims to
    preserve is no longer what LEGACY_SHA actually says."""


def verify_legacy_identity(source_by_path: dict[str, str]) -> None:
    """
    *source_by_path* maps each path in EXPECTED_SNIPPETS to its real
    `git show LEGACY_SHA:<path>` text. Raises LegacyIdentityMismatch
    (fail-closed) naming every missing snippet across every path;
    returns None silently if every expected snippet is present.
    """
    missing: list[str] = []
    for path, snippets in EXPECTED_SNIPPETS.items():
        text = source_by_path.get(path)
        if text is None:
            missing.append(f"{path}: source not supplied")
            continue
        for snippet in snippets:
            if snippet not in text:
                missing.append(f"{path}: missing {snippet!r}")
    if missing:
        raise LegacyIdentityMismatch("; ".join(missing))


def fetch_legacy_sources(git_show: Callable[[str, str], str]) -> dict[str, str]:
    """*git_show* is a callable (commit, path) -> str, e.g. a thin wrapper
    around `git show <commit>:<path>`. Returns the real source text for
    every path this module checks, keyed exactly as EXPECTED_SNIPPETS."""
    return {path: git_show(LEGACY_SHA, path) for path in EXPECTED_SNIPPETS}
