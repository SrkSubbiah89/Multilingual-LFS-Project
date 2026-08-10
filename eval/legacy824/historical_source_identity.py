"""
eval/legacy824/historical_source_identity.py

Task 39: verifies, by direct source inspection, that the historical
commit still contains every literal policy value the exact-policy table
requires -- fails closed on any drift (a changed threshold, top-k,
model route, temperature, prompt field, parser condition, method
label, or fallback behavior) rather than silently trusting a cached
assumption.

The git-show function is injectable so hermetic tests can supply a
deliberately mismatched historical source string and prove this module
correctly rejects it -- no live git/subprocess call is made in tests.
"""

from __future__ import annotations

import subprocess
from typing import Callable, Optional

from . import LEGACY_SHA

# Relative paths (within the repo) of the three files the task requires
# be inspected directly.
ISCO_CLASSIFIER_PATH = "backend/agents/isco_classifier.py"
VECTOR_STORE_PATH = "backend/rag/vector_store.py"
LLM_CLIENT_PATH = "backend/llm/llm_client.py"

# Every literal snippet that must be present, keyed by which file it is
# expected in. Each snippet is checked via plain substring containment
# (not a fuzzy match) -- deliberately brittle, so a real behavioral
# change is caught rather than paraphrased away.
EXPECTED_SNIPPETS: dict[str, list[str]] = {
    ISCO_CLASSIFIER_PATH: [
        "_HIGH_CONFIDENCE_THRESHOLD = 0.92",
        "candidates = self._store.search(job_title, top_k=top_k)",
        "if candidates[0].confidence >= _HIGH_CONFIDENCE_THRESHOLD:",
        'method="semantic",',
        'method="llm_ranked",',
        "self._llm  = get_llm(TaskType.CRITICAL)",
        '{"selected_code": "<isco_code>", "reasoning": "<one sentence>"}',
        "if selected_code in code_map:",
        "Fallback to top semantic match (LLM response could not be parsed).",
        "top_k: int = 5,",
    ],
    VECTOR_STORE_PATH: [
        'COLLECTION_NAME = "isco_occupations"',
        'MODEL_NAME      = "intfloat/multilingual-e5-large"',
        "TOP_K_DEFAULT   = 5",
    ],
    LLM_CLIENT_PATH: [
        'MODEL_CRITICAL = "anthropic/claude-3-5-sonnet-20241022"',
        "_TEMP_CRITICAL = 0.0",
    ],
}

GitShowFn = Callable[[str, str], str]


def default_git_show(sha: str, path: str) -> str:
    """Real `git show <sha>:<path>` -- the production implementation.
    Never called by any hermetic test (those inject a fake function)."""
    result = subprocess.run(
        ["git", "show", f"{sha}:{path}"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def verify_historical_source_identity(
    git_show_fn: Optional[GitShowFn] = None,
    sha: str = LEGACY_SHA,
) -> dict:
    """
    Checks every expected snippet against the literal historical source
    at *sha*. Returns a report dict; never raises on a mismatch (the
    caller decides whether a mismatch is fatal) -- this function itself
    always returns a complete report so a caller can render a full gate
    table regardless of outcome.
    """
    show = git_show_fn or default_git_show
    per_file: dict[str, dict] = {}
    all_ok = True

    for path, snippets in EXPECTED_SNIPPETS.items():
        try:
            source = show(sha, path)
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            per_file[path] = {"ok": False, "error": str(exc), "missing_snippets": list(snippets)}
            all_ok = False
            continue

        missing = [s for s in snippets if s not in source]
        ok = not missing
        all_ok = all_ok and ok
        per_file[path] = {
            "ok": ok,
            "missing_snippets": missing,
            "source_sha256_len": len(source),
        }

    return {"sha": sha, "all_ok": all_ok, "per_file": per_file}
