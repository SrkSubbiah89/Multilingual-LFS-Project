"""
eval/legacy_decision_policy41/live_reranker.py

Task 43: a real Anthropic reranker callable for Task 41's decision policy
(`policy.py`, used unmodified), plus a live-run driver that actually
invokes it against WISCO dev cases that Task 42 found were below the
0.92 fast-path threshold.

Preserved from history: model `claude-3-5-sonnet-20241022`, temperature
`0.0`, and the exact prompt text `policy.build_prompt_text()` already
builds and tests -- nothing about the decision content changes.

Disclosed, deliberate difference from the historical code: the original
`isco_classifier.py` wrapped this call in a CrewAI `Agent`/`Crew`. This
module calls the Anthropic API directly via the `anthropic` SDK instead,
skipping that agent-framework wrapper (which only adds role/goal/
backstory token overhead around the same underlying model call, not
decision logic) -- disclosed here and in the final report, not silently
substituted.

This module is intentionally kept separate from `policy.py` (which must
have zero LLM/network import) and from `flat_retrieval_adapter.py`
(Task 42's module, which is tested to have zero LLM/network import at
module scope) -- it is the one place in this package that is allowed to
import `anthropic` and make a real network call.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import anthropic

from eval.legacy_decision_policy41.flat_retrieval_adapter import (
    DevRow,
    fetch_five_official_flat_candidates,
)
from eval.legacy_decision_policy41.policy import (
    HISTORICAL_MODEL,
    HISTORICAL_TEMPERATURE,
    METHOD_LLM_RANKED,
    METHOD_SEMANTIC,
    classify_with_policy,
)

_ANTHROPIC_MODEL_ID = "claude-3-5-sonnet-20241022"  # matches HISTORICAL_MODEL's suffix after "anthropic/"
_MAX_TOKENS = 200
_RETRYABLE_EXCEPTIONS = (
    anthropic.RateLimitError,
    anthropic.APITimeoutError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
)
_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 2.0

assert HISTORICAL_MODEL == f"anthropic/{_ANTHROPIC_MODEL_ID}"
assert HISTORICAL_TEMPERATURE == 0.0


_FATAL_EXCEPTION_TYPES = (anthropic.AuthenticationError, anthropic.PermissionDeniedError)
_INSUFFICIENT_CREDIT_MARKER = "credit balance is too low"


class FatalRerankerError(Exception):
    """
    A reranker call failed for a reason retrying cannot fix and that is
    NOT "the LLM responded but the output was malformed" (auth failure,
    permission failure, or insufficient account credit). This must never
    be allowed to look like a normal historical-fallback row: Task 41's
    `classify_with_policy` catches *any* exception from the reranker
    callable and folds it into the same semantic-top fallback used for a
    genuinely malformed response -- correct behavior for a malformed
    response, but wrong for "the call never happened at all". Discovered
    2026-08-10/11: an exhausted Anthropic credit balance silently produced
    thousands of `method="llm_ranked"` rows that were, in fact, plain
    semantic-top predictions with a fabricated-looking label -- the flat
    heldout run's predictions matched Task 36's non-reranked baseline
    prediction for 100% of 18,747 cases, which is how this was caught.

    `make_anthropic_reranker`'s closure records this exception into a
    caller-supplied `fatal_tracker` dict (since it cannot itself abort a
    multi-row run -- `classify_with_policy` will always catch what it
    raises) so `run_dev_with_live_reranker` can check after every row and
    abort the entire run loudly the moment one occurs, rather than
    silently completing with mislabeled results.
    """


def _is_fatal(exc: Exception) -> bool:
    if isinstance(exc, _FATAL_EXCEPTION_TYPES):
        return True
    if isinstance(exc, anthropic.BadRequestError) and _INSUFFICIENT_CREDIT_MARKER in str(exc).lower():
        return True
    return False


def make_anthropic_reranker(client: "anthropic.Anthropic", fatal_tracker: Optional[dict] = None):
    """
    Returns a `reranker(prompt_text: str) -> str` callable bound to
    *client*, suitable for `policy.classify_with_policy(..., reranker=...)`.
    Retries up to `_MAX_ATTEMPTS` times, only on transient errors (rate
    limit, timeout, connection, transient 5xx) -- never on auth/permission/
    bad-request errors, which retrying cannot fix.

    If *fatal_tracker* (a plain dict) is supplied, any fatal, non-retryable
    error (see `_is_fatal`) is recorded into it as `fatal_tracker["error"]`
    before being re-raised, so a caller driving many rows (e.g.
    `run_dev_with_live_reranker`) can detect it and abort the whole run --
    see `FatalRerankerError`'s docstring for why this exists.
    """

    def reranker(prompt_text: str) -> str:
        last_exc: Optional[Exception] = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = client.messages.create(
                    model=_ANTHROPIC_MODEL_ID,
                    max_tokens=_MAX_TOKENS,
                    temperature=HISTORICAL_TEMPERATURE,
                    messages=[{"role": "user", "content": prompt_text}],
                )
                return "".join(block.text for block in response.content if block.type == "text")
            except _RETRYABLE_EXCEPTIONS as exc:
                last_exc = exc
                if attempt < _MAX_ATTEMPTS:
                    time.sleep(_RETRY_BACKOFF_SECONDS * attempt)
            except Exception as exc:
                if fatal_tracker is not None and _is_fatal(exc):
                    fatal_tracker["error"] = exc
                raise
        raise RuntimeError(f"Anthropic call failed after {_MAX_ATTEMPTS} attempts") from last_exc

    return reranker


@dataclass
class LiveDevRowResult:
    case_id: str
    predicted_code: str
    gold_code: str
    method: str  # "semantic" | "llm_ranked"
    correct: bool


@dataclass
class LiveDevRunReport:
    rows: list[LiveDevRowResult] = field(default_factory=list)
    n_total: int = 0
    n_correct: int = 0
    n_semantic: int = 0
    n_llm_ranked: int = 0
    n_semantic_correct: int = 0
    n_llm_ranked_correct: int = 0
    run_manifest: dict = field(default_factory=dict)


def run_dev_with_live_reranker(
    store,
    dev_rows: list[DevRow],
    reranker,
    limit: Optional[int] = None,
    run_manifest: Optional[dict] = None,
    progress_jsonl_path: Optional[Path] = None,
    candidate_fetcher=fetch_five_official_flat_candidates,
    fatal_tracker: Optional[dict] = None,
) -> LiveDevRunReport:
    """
    Runs the historical decision policy against *dev_rows* with a REAL
    reranker -- every row gets a complete prediction (no
    PENDING_RERANK_NO_LLM_CALLED outcome is possible here). If
    *progress_jsonl_path* is given, each row's result is appended to it
    immediately after computing (one JSON object per line) so a mid-run
    failure never loses already-paid-for results.

    *fatal_tracker*, if supplied, must be the SAME dict passed to
    `make_anthropic_reranker(client, fatal_tracker=...)` that built
    *reranker*. After every row, if the reranker recorded a fatal,
    non-retryable failure (auth/permission/insufficient-credit -- see
    `FatalRerankerError`), this function raises `FatalRerankerError`
    immediately and does NOT record that row's (meaningless, silently-
    fallen-back) result -- this is the fix for the exact failure mode
    that produced thousands of mislabeled `"llm_ranked"` rows on
    2026-08-10/11 when the Anthropic account ran out of credit.

    Any OTHER exception also stops the run immediately (fail closed) --
    already appended progress-file rows are preserved on disk, but this
    function itself never returns a partial report silently; it raises.
    """
    rows = dev_rows[:limit] if limit is not None else dev_rows
    report = LiveDevRunReport(run_manifest=run_manifest or {})
    progress_file = open(progress_jsonl_path, "a", encoding="utf-8") if progress_jsonl_path else None

    try:
        for row in rows:
            candidates = candidate_fetcher(store, row.input_text)
            result = classify_with_policy(
                job_title=row.input_text,
                candidates=candidates,
                lang=row.input_language,
                reranker=reranker,
            )
            if fatal_tracker is not None and fatal_tracker.get("error") is not None:
                raise FatalRerankerError(
                    f"aborting run: fatal reranker error on case {row.case_id!r} "
                    f"({report.n_total} rows already completed): {fatal_tracker['error']}"
                )
            correct = result.primary.code == row.gold_isco_4digit
            row_result = LiveDevRowResult(
                case_id=row.case_id, predicted_code=result.primary.code,
                gold_code=row.gold_isco_4digit, method=result.method, correct=correct,
            )
            report.rows.append(row_result)
            report.n_total += 1
            report.n_correct += int(correct)
            if result.method == METHOD_SEMANTIC:
                report.n_semantic += 1
                report.n_semantic_correct += int(correct)
            elif result.method == METHOD_LLM_RANKED:
                report.n_llm_ranked += 1
                report.n_llm_ranked_correct += int(correct)

            if progress_file:
                progress_file.write(json.dumps({
                    "case_id": row_result.case_id, "predicted_code": row_result.predicted_code,
                    "gold_code": row_result.gold_code, "method": row_result.method,
                    "correct": row_result.correct,
                }, ensure_ascii=False) + "\n")
                progress_file.flush()
    finally:
        if progress_file:
            progress_file.close()

    return report
