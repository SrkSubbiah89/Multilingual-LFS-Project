"""
Tests for eval/legacy824/adapter.py (Task 39, scenarios 2, 8 [adapter
side], 12, 14).

Fully hermetic: the historical classifier is replaced by a small fake
object exposing only the same duck-typed call contract
(`classify(job_title, context, top_k) -> object with .primary.code,
.method, .candidates, .reasoning`) -- no real historical loader,
Qdrant, embedding model, or Anthropic call happens anywhere in this
file. Real classifier decision-logic behavior is covered separately in
test_historical_classifier_behavior.py.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.legacy824.adapter import (  # noqa: E402
    FALLBACK_REASONING,
    DevelopmentRunStopped,
    RowResult,
    classify_dev_rows,
    compute_descriptive_stats,
    write_results_csv,
)


@dataclass
class _FakeCandidate:
    code: str
    confidence: float


@dataclass
class _FakeClassification:
    primary: _FakeCandidate
    method: str
    candidates: list = field(default_factory=list)
    reasoning: str = ""


class _RecordingFakeClassifier:
    """Records every classify() call's exact arguments, and returns
    canned results (or raises) from a script, one entry per call.

    If constructed with a call_counter dict, increments it by 1 whenever
    the scripted outcome's method is "llm_ranked" -- simulating the real
    _CountingCrew wrapper's observational instrumentation, so tests that
    exercise the adapter's own call-count cross-check don't need to
    manage the counter by hand."""

    def __init__(self, script, call_counter=None):
        self.script = script
        self.calls = []
        self.call_counter = call_counter

    def classify(self, job_title, context="", top_k=5):
        self.calls.append({"job_title": job_title, "context": context, "top_k": top_k})
        idx = len(self.calls) - 1
        outcome = self.script[idx]
        if isinstance(outcome, Exception):
            raise outcome
        if self.call_counter is not None and getattr(outcome, "method", None) == "llm_ranked":
            self.call_counter["total"] += 1
        return outcome


def _rows(n):
    return [
        {"case_id": f"WISCO-DEV-{i}", "input_text": f"job title {i}", "input_language": "en",
         "gold_isco_4digit": "2512"}
        for i in range(n)
    ]


def _semantic_result(code="2512", conf=0.95):
    return _FakeClassification(
        primary=_FakeCandidate(code, conf), method="semantic",
        candidates=[_FakeCandidate(code, conf)] + [_FakeCandidate(f"999{i}", conf - 0.1 * i) for i in range(1, 5)],
        reasoning=f"Unambiguous semantic match (score {conf:.2%}).",
    )


def _llm_ranked_result(code="2511", candidates_codes=("2512", "2511", "2519", "2521", "2522")):
    cands = [_FakeCandidate(c, 0.9 - 0.05 * i) for i, c in enumerate(candidates_codes)]
    return _FakeClassification(primary=_FakeCandidate(code, cands[1].confidence), method="llm_ranked",
                                candidates=cands, reasoning="Selected by LLM classifier.")


# ---------------------------------------------------------------------------
# Scenario 2: adapter calls the classifier once per row, unchanged text, top_k=5
# ---------------------------------------------------------------------------

def test_adapter_calls_classifier_once_per_row_with_unchanged_text_and_top_k_5():
    rows = _rows(3)
    clf = _RecordingFakeClassifier(script=[_semantic_result() for _ in range(3)])
    call_counter = {"total": 0}
    results = classify_dev_rows(clf, rows, call_counter)

    assert len(results) == 3
    assert len(clf.calls) == 3
    for i, call in enumerate(clf.calls):
        assert call["job_title"] == rows[i]["input_text"]
        assert call["top_k"] == 5
        assert call["context"] == ""


# ---------------------------------------------------------------------------
# Scenario 8 (adapter side): a live exception or fallback marker stops the run
# ---------------------------------------------------------------------------

def test_adapter_stops_on_classifier_exception_preserving_partial_results():
    rows = _rows(3)
    clf = _RecordingFakeClassifier(script=[_semantic_result(), RuntimeError("simulated API failure")])
    call_counter = {"total": 0}
    with pytest.raises(DevelopmentRunStopped):
        classify_dev_rows(clf, rows, call_counter)
    assert len(clf.calls) == 2  # never attempted row 3 -- no retry, no continuation


def test_adapter_stops_on_fallback_marker():
    rows = _rows(2)
    fallback_result = _FakeClassification(
        primary=_FakeCandidate("2512", 0.85), method="llm_ranked",
        candidates=[_FakeCandidate(c, 0.8) for c in ("2512", "2511", "2519", "2521", "2522")],
        reasoning=FALLBACK_REASONING,
    )
    clf = _RecordingFakeClassifier(script=[fallback_result])
    call_counter = {"total": 1}  # simulate the LLM call already having fired for this row
    with pytest.raises(DevelopmentRunStopped):
        classify_dev_rows(clf, rows, call_counter)
    assert len(clf.calls) == 1


def test_adapter_stops_when_call_count_does_not_match_expected():
    """A stage-level double-call (or zero-call) is treated as an anomaly,
    not silently accepted."""
    rows = _rows(1)
    clf = _RecordingFakeClassifier(script=[_semantic_result()])
    call_counter = {"total": 1}  # a semantic (no-LLM) row must have 0 calls, not 1
    with pytest.raises(DevelopmentRunStopped):
        classify_dev_rows(clf, rows, call_counter)


def test_successful_run_produces_correct_and_status_fields():
    rows = _rows(2)
    clf = _RecordingFakeClassifier(script=[_semantic_result(code="2512"), _semantic_result(code="9999")])
    call_counter = {"total": 0}
    results = classify_dev_rows(clf, rows, call_counter)
    assert all(r.status == "success" for r in results)
    assert results[0].correct is True   # gold=2512, pred=2512
    assert results[1].correct is False  # gold=2512, pred=9999


# ---------------------------------------------------------------------------
# Scenario 14: output schema retains sufficient per-row raw evidence
# ---------------------------------------------------------------------------

def test_output_schema_has_sufficient_audit_fields(tmp_path):
    rows = _rows(2)
    call_counter = {"total": 0}
    clf = _RecordingFakeClassifier(script=[_semantic_result(), _llm_ranked_result()], call_counter=call_counter)
    results = classify_dev_rows(clf, rows, call_counter)
    out = tmp_path / "results.csv"
    write_results_csv(results, out)

    import csv
    with out.open(encoding="utf-8", newline="") as f:
        written = list(csv.DictReader(f))
    required_fields = {
        "case_id", "language", "input_text", "gold_code", "pred_code", "method",
        "top_semantic_score", "candidate_codes", "llm_fired", "reasoning",
        "elapsed_ms", "correct", "status", "error",
    }
    assert required_fields.issubset(written[0].keys())
    assert written[1]["llm_fired"] == "True"
    assert "2511" in written[1]["candidate_codes"]


def test_descriptive_stats_only_meaningful_when_all_rows_succeeded():
    rows = _rows(2)
    call_counter = {"total": 0}
    clf = _RecordingFakeClassifier(script=[_semantic_result(code="2512"), _llm_ranked_result(code="2511")], call_counter=call_counter)
    results = classify_dev_rows(clf, rows, call_counter)
    stats = compute_descriptive_stats(results)
    assert stats["no_error_confirmed"] is True
    assert stats["n"] == 2
    assert stats["semantic_only_n"] == 1
    assert stats["conditional_llm_n"] == 1


# ---------------------------------------------------------------------------
# Scenario 12: adapter does not import forbidden current-tree components
# ---------------------------------------------------------------------------

_FORBIDDEN_IMPORT_ROOTS = {
    "eval.run_eval", "backend.agents.isco_classifier", "backend.rag.hierarchy_engine",
    "backend.rag.hierarchical_store", "backend.rag.official_isco08_catalogue",
    "backend.agents.isic_classifier", "backend.agents.isced_classifier",
    "backend.agents.semantic_relation",
}


def test_adapter_module_imports_no_forbidden_current_tree_component():
    adapter_dir = Path(__file__).resolve().parent
    for py_file in adapter_dir.glob("*.py"):
        if py_file.name.startswith("test_"):
            continue
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in _FORBIDDEN_IMPORT_ROOTS, f"{py_file.name} imports {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module not in _FORBIDDEN_IMPORT_ROOTS, f"{py_file.name} imports from {node.module}"
                for forbidden in _FORBIDDEN_IMPORT_ROOTS:
                    assert not node.module.startswith(forbidden + "."), (
                        f"{py_file.name} imports from {node.module} (forbidden prefix {forbidden})"
                    )
