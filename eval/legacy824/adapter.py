"""
eval/legacy824/adapter.py

Task 39: the smallest adapter that maps WISCO v2 development-split rows
into one-row-at-a-time calls to the unmodified historical
ISCOClassifier (LEGACY_SHA = 824fcf235ae2f8787706cf479a07620519c914de),
and records full per-row evidence for a fail-closed descriptive-only
result.

This module never alters candidate ordering, confidence values,
threshold, top-k, prompt text, model route, temperature, JSON parser,
candidate validation, fallback policy, or method label. It never calls
a hierarchy engine, the current ISCOClassifier, run_eval.py, the
official catalogue loader, ISIC, ISCED, or SRE. It never reads a
heldout row (eval/legacy824/dataset_gate.py refuses that outright
before any classification is attempted).

No LLM retry, evaluator retry, or provider fallback exists anywhere in
this module: any exception, or any occurrence of the historical
malformed/out-of-candidate-response fallback marker, stops the whole
run immediately (DevelopmentRunStopped) with the partial result
preserved -- never rerun, never silently continued.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .historical_loader import load_historical_isco_classifier
from .isolation import validate_isolated_endpoint

HIGH_CONFIDENCE_THRESHOLD = 0.92
TOP_K = 5
FALLBACK_REASONING = "Fallback to top semantic match (LLM response could not be parsed)."


@dataclass
class RowResult:
    case_id: str
    language: str
    input_text: str
    gold_code: str
    pred_code: Optional[str] = None
    method: Optional[str] = None
    top_semantic_score: Optional[float] = None
    candidate_codes: list = field(default_factory=list)
    llm_fired: bool = False
    reasoning: Optional[str] = None
    elapsed_ms: Optional[float] = None
    correct: Optional[bool] = None
    status: str = "pending"  # "success" | "stopped_exception" | "stopped_fallback_triggered"
    error: Optional[str] = None


class DevelopmentRunStopped(RuntimeError):
    """Raised to signal the run must stop immediately -- no retry, no
    continuation, per Task 39's explicit no-retry policy. The partial
    result list up to and including the failing row is still available
    to the caller via the exception's own row-result accumulation."""


def build_isolated_classifier(worktree_path: Path, qdrant_host: str, qdrant_port: int):
    """
    Validates the isolated endpoint, points the historical VectorStore
    at it via environment variables only (QDRANT_HOST/QDRANT_PORT --
    never via source modification), loads the historical modules
    verbatim from *worktree_path*, and returns
    (ISCOClassifier_instance, loaded_modules, call_counter).

    call_counter is a plain {"total": int} dict incremented by an
    observational-only Crew subclass that counts real
    Crew.kickoff() invocations without changing what kickoff() does or
    returns -- used only to independently verify "exactly one LLM call
    per reranked row" as runtime evidence, not merely an assumption
    from reading the source.
    """
    validate_isolated_endpoint(qdrant_host, qdrant_port)
    os.environ["QDRANT_HOST"] = qdrant_host
    os.environ["QDRANT_PORT"] = str(qdrant_port)

    loaded = load_historical_isco_classifier(worktree_path)

    call_counter = {"total": 0}
    real_crew_cls = loaded.isco_classifier.Crew

    class _CountingCrew(real_crew_cls):  # type: ignore[misc, valid-type]
        def kickoff(self, *args, **kwargs):
            call_counter["total"] += 1
            return super().kickoff(*args, **kwargs)

    loaded.isco_classifier.Crew = _CountingCrew

    clf = loaded.isco_classifier.ISCOClassifier()
    return clf, loaded, call_counter


def classify_dev_rows(clf, rows: list[dict], call_counter: dict, timer=None) -> list[RowResult]:
    """
    Runs clf.classify() once per row, in order, stopping immediately
    (raising DevelopmentRunStopped) on any exception or on the
    historical malformed-response fallback marker. *timer* defaults to
    time.perf_counter and is injectable only so hermetic tests can
    supply a deterministic clock.
    """
    if timer is None:
        import time
        timer = time.perf_counter

    results: list[RowResult] = []
    calls_before_row = 0

    for row in rows:
        r = RowResult(
            case_id=row["case_id"], language=row["input_language"],
            input_text=row["input_text"], gold_code=row["gold_isco_4digit"].strip(),
        )
        t0 = timer()
        try:
            classification = clf.classify(job_title=row["input_text"], context="", top_k=TOP_K)
        except Exception as exc:  # noqa: BLE001 - preserved verbatim, run stops
            r.status = "stopped_exception"
            r.error = f"{type(exc).__name__}: {exc}"
            r.elapsed_ms = round((timer() - t0) * 1000, 3)
            results.append(r)
            raise DevelopmentRunStopped(f"row {row['case_id']}: {r.error}") from exc

        r.elapsed_ms = round((timer() - t0) * 1000, 3)
        r.pred_code = classification.primary.code
        r.method = classification.method
        r.candidate_codes = [c.code for c in classification.candidates]
        r.top_semantic_score = (
            classification.candidates[0].confidence if classification.candidates else None
        )
        r.reasoning = classification.reasoning
        r.llm_fired = classification.method == "llm_ranked"

        calls_after_row = call_counter["total"]
        row_call_count = calls_after_row - calls_before_row
        calls_before_row = calls_after_row

        if classification.reasoning == FALLBACK_REASONING:
            r.status = "stopped_fallback_triggered"
            r.error = "historical malformed/out-of-candidate LLM response fallback triggered"
            results.append(r)
            raise DevelopmentRunStopped(f"row {row['case_id']}: {r.error}")

        if r.llm_fired and row_call_count != 1:
            r.status = "stopped_exception"
            r.error = f"expected exactly 1 LLM call for a reranked row, observed {row_call_count}"
            results.append(r)
            raise DevelopmentRunStopped(f"row {row['case_id']}: {r.error}")
        if not r.llm_fired and row_call_count != 0:
            r.status = "stopped_exception"
            r.error = f"expected 0 LLM calls for a direct-semantic row, observed {row_call_count}"
            results.append(r)
            raise DevelopmentRunStopped(f"row {row['case_id']}: {r.error}")

        if r.pred_code not in r.candidate_codes:
            r.status = "stopped_exception"
            r.error = "predicted code is not among the five supplied candidates"
            results.append(r)
            raise DevelopmentRunStopped(f"row {row['case_id']}: {r.error}")

        r.correct = r.pred_code == r.gold_code
        r.status = "success"
        results.append(r)

    return results


def write_results_csv(results: list[RowResult], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(results[0]).keys()) if results else []
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            row = asdict(r)
            row["candidate_codes"] = json.dumps(row["candidate_codes"])
            w.writerow(row)


def compute_descriptive_stats(results: list[RowResult]) -> dict:
    """
    Descriptive-only summary. Only meaningful when every row in
    *results* has status == "success" -- callers must check
    `all(r.status == "success" for r in results)` before treating this
    as a valid development-run result.
    """
    n = len(results)
    correct = sum(1 for r in results if r.correct)
    semantic_rows = [r for r in results if not r.llm_fired]
    llm_rows = [r for r in results if r.llm_fired]
    semantic_correct = sum(1 for r in semantic_rows if r.correct)
    llm_correct = sum(1 for r in llm_rows if r.correct)
    llm_differs_from_semantic_top = sum(
        1 for r in llm_rows if r.candidate_codes and r.pred_code != r.candidate_codes[0]
    )
    method_counts: dict[str, int] = {}
    for r in results:
        method_counts[r.method or "unknown"] = method_counts.get(r.method or "unknown", 0) + 1
    latencies = sorted(r.elapsed_ms for r in results if r.elapsed_ms is not None)

    def _pctl(p: float) -> Optional[float]:
        if not latencies:
            return None
        import math
        idx = max(0, min(len(latencies) - 1, math.ceil(p * len(latencies)) - 1))
        return latencies[idx]

    return {
        "n": n,
        "correct": correct,
        "accuracy": (correct / n) if n else None,
        "semantic_only_n": len(semantic_rows),
        "semantic_only_correct": semantic_correct,
        "semantic_only_accuracy": (semantic_correct / len(semantic_rows)) if semantic_rows else None,
        "conditional_llm_n": len(llm_rows),
        "conditional_llm_correct": llm_correct,
        "conditional_llm_accuracy": (llm_correct / len(llm_rows)) if llm_rows else None,
        "llm_decisions_differ_from_semantic_top_count": llm_differs_from_semantic_top,
        "llm_decisions_differ_from_semantic_top_pct": (
            (llm_differs_from_semantic_top / len(llm_rows) * 100) if llm_rows else None
        ),
        "method_label_counts": method_counts,
        "latency_median_ms": statistics.median(latencies) if latencies else None,
        "latency_p95_ms": _pctl(0.95),
        "no_error_confirmed": all(r.status == "success" for r in results),
    }
