"""
Tests for eval/sre_eval_format.py -- Section G of the Conference I
Reviewer #2 response (SRE precision/recall/FPR/FNR/reviewer-workload
evaluation format).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sre_eval_format as sef  # noqa: E402


def test_empty_labels_returns_no_labels_supplied():
    m = sef.evaluate_sre_labels([])
    assert m.status == "no_labels_supplied"
    assert m.precision is None
    assert m.recall is None
    assert m.fpr is None
    assert m.fnr is None
    assert m.n_labeled == 0


def test_perfect_classifier_precision_recall_one():
    cases = [(True, True), (True, True), (False, False), (False, False)]
    m = sef.evaluate_sre_labels(cases)
    assert m.status == "measured"
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.fpr == 0.0
    assert m.fnr == 0.0
    assert m.n_labeled == 4


def test_all_false_positives():
    cases = [(True, False), (True, False)]
    m = sef.evaluate_sre_labels(cases)
    assert m.precision == 0.0
    assert m.fpr == 1.0
    assert m.recall is None  # no actual positives at all -- undefined, not fabricated as 0


def test_mixed_confusion_matrix():
    # tp=1, fp=1, fn=1, tn=1
    cases = [(True, True), (True, False), (False, True), (False, False)]
    m = sef.evaluate_sre_labels(cases)
    assert m.precision == 0.5
    assert m.recall == 0.5
    assert m.fpr == 0.5
    assert m.fnr == 0.5


def test_reviewer_workload_delta_computed_when_both_rates_given():
    m = sef.evaluate_sre_labels([(True, True)], baseline_escalation_rate=0.20, sre_escalation_rate=0.35)
    assert m.reviewer_workload_delta == 0.15


def test_reviewer_workload_delta_none_when_only_one_rate_given():
    m = sef.evaluate_sre_labels([(True, True)], baseline_escalation_rate=0.20)
    assert m.reviewer_workload_delta is None


def test_reviewer_workload_delta_none_without_labels():
    m = sef.evaluate_sre_labels([], baseline_escalation_rate=0.2, sre_escalation_rate=0.3)
    # no_labels_supplied path returns early -- workload delta not computed
    # even if rates were passed, since the whole metrics object is "no data"
    assert m.status == "no_labels_supplied"
    assert m.reviewer_workload_delta is None
