"""
eval/sre_eval_format.py

Evaluation format for SemanticRelationEngine (SRE) precision/recall/FPR/FNR
and reviewer-workload delta (Conference I Reviewer #2 response, Section G).

The SRE flags cross-standard (ISCO/ISIC/ISCED) incoherence deterministically
(see backend/agents/semantic_relation.py) but its PRECISION/RECALL as an
incoherence detector has never been measured -- that requires a labelled
set of cases where a human has judged whether the case is genuinely
incoherent (ground truth), which does not exist in this repo. Per the
accuracy boundary for this whole effort: every metric field here is None
with status="no_labels_supplied" unless a labelled fixture is actually
passed in -- never a placeholder number.

Usage
-----
    from eval.sre_eval_format import SREEvalMetrics, evaluate_sre_labels
    metrics = evaluate_sre_labels(labelled_cases)  # [] -> no_labels_supplied
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SREEvalMetrics:
    precision: Optional[float]
    recall: Optional[float]
    fpr: Optional[float]  # false positive rate
    fnr: Optional[float]  # false negative rate
    reviewer_workload_delta: Optional[float]  # change in escalation rate vs. a no-SRE baseline, if provided
    n_labeled: int
    status: str  # "no_labels_supplied" | "measured"


def _empty_metrics(n_labeled: int = 0) -> SREEvalMetrics:
    return SREEvalMetrics(
        precision=None, recall=None, fpr=None, fnr=None,
        reviewer_workload_delta=None, n_labeled=n_labeled,
        status="no_labels_supplied",
    )


def evaluate_sre_labels(
    labelled_cases: list[tuple[bool, bool]],
    baseline_escalation_rate: Optional[float] = None,
    sre_escalation_rate: Optional[float] = None,
) -> SREEvalMetrics:
    """
    labelled_cases : list of (sre_flagged_incoherent, human_judged_incoherent)
        pairs. Empty list (the default in every run today, since no
        human-labelled incoherence set exists) -> status="no_labels_supplied",
        every metric None.
    baseline_escalation_rate / sre_escalation_rate : optional, both required
        together to compute reviewer_workload_delta (e.g. from two
        eval.manifest.ExperimentRunManifest.hitl_escalation_rate values, one
        with SRE severity contributing to escalation and one without --
        see eval/ablation_runner.py's no-SRE/with-SRE configs). If only one
        or neither is supplied, reviewer_workload_delta stays None.
    """
    if not labelled_cases:
        return _empty_metrics(0)

    tp = sum(1 for pred, gold in labelled_cases if pred and gold)
    fp = sum(1 for pred, gold in labelled_cases if pred and not gold)
    fn = sum(1 for pred, gold in labelled_cases if not pred and gold)
    tn = sum(1 for pred, gold in labelled_cases if not pred and not gold)

    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else None
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else None
    fpr = round(fp / (fp + tn), 4) if (fp + tn) > 0 else None
    fnr = round(fn / (fn + tp), 4) if (fn + tp) > 0 else None

    workload_delta = None
    if baseline_escalation_rate is not None and sre_escalation_rate is not None:
        workload_delta = round(sre_escalation_rate - baseline_escalation_rate, 4)

    return SREEvalMetrics(
        precision=precision, recall=recall, fpr=fpr, fnr=fnr,
        reviewer_workload_delta=workload_delta, n_labeled=len(labelled_cases),
        status="measured",
    )
