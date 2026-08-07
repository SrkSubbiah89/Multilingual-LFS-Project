"""
Regression tests for the SRE-to-ISIC/ISCED coupling bugfix (Conference I
Reviewer #2 response, Step 5.1). Pre-fix, eval/run_eval.py's run_one_case()
gated ISIC/ISCED classification on sre_enabled, so `--sre off` silently
produced blank ISIC/ISCED predictions instead of merely skipping the SRE
coherence check. See Documentation/Conference_I_Reviewer_2/
SRE_COUPLING_BUGFIX.md for the full root-cause analysis and
eval/local_runs/step5_synthetic_integration_20260807/FINDING_sre_isic_isced_coupling_bug.md
for the original discovery record.

This file is scoped narrowly to the fix itself; eval/test_run_eval_b2.py
already covers the general sre_enabled=True/False call-count assertions
(updated in this same change) -- this file adds the evidence-safety and
isolation guarantees the Step 5.1 task specification requires explicitly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_eval  # noqa: E402


def _make_clf_result(code="2512", confidence=0.9, method="hierarchical_llm"):
    return SimpleNamespace(
        primary=SimpleNamespace(code=code, title_en="Software Developers", title_ar="", confidence=confidence),
        method=method, hitl_required=False, reasoning="Selected by LLM classifier.",
    )


def _make_fake_isco_clf(clf_result=None, trace_updates=None):
    clf_result = clf_result or _make_clf_result()
    trace_updates = trace_updates or {}
    clf = MagicMock()

    def _classify(job_title, language, top_k, use_llm, trace):
        trace.update(trace_updates)
        return clf_result

    clf.classify.side_effect = _classify
    return clf


def _make_isic_result():
    return SimpleNamespace(
        section="J", section_title="ICT", division_code="62", division_title="",
        group_code="620", group_title="", class_code="6201", class_title="",
        confidence=0.9, method="keyword", alternatives=[], raw_text="",
    )


def _make_isced_result():
    return SimpleNamespace(
        level=6, level_title="Bachelor's", broad_code="06", broad_title="",
        narrow_code="061", narrow_title="", detailed_code="0613", detailed_title="",
        confidence=0.8, method="keyword", raw_text="",
    )


def _make_sre_coherence(score=0.95, violations=None):
    return SimpleNamespace(score=score, violations=violations or [])


def _base_kwargs(sre_enabled: bool, industry_text="software company", education_text="bachelor of science"):
    isic_clf = MagicMock()
    isic_clf.classify.return_value = _make_isic_result()
    isced_clf = MagicMock()
    isced_clf.classify.return_value = _make_isced_result()
    sre = MagicMock()
    sre.analyse.return_value = _make_sre_coherence()
    return dict(
        sre=sre, isic_clf=isic_clf, isced_clf=isced_clf,
        row_index=0, case_id="c1", input_text="software developer",
        input_language="en", gold_isco_4digit="2512", gold_isic="J", gold_isced="6",
        config_hash="abc123", industry_text=industry_text, education_text=education_text,
        sre_enabled=sre_enabled,
    ), isic_clf, isced_clf, sre


# ---------------------------------------------------------------------------
# 1. sre=False: ISCO/ISIC/ISCED/ISCED-F all execute; SRE explicitly disabled
# ---------------------------------------------------------------------------

def test_sre_false_isco_classification_executed():
    clf = _make_fake_isco_clf()
    kwargs, _, _, _ = _base_kwargs(sre_enabled=False)
    r = run_eval.run_one_case(clf, **kwargs)
    clf.classify.assert_called_once()
    assert r.pred_isco_4digit == "2512"


def test_sre_false_isic_classification_executed():
    clf = _make_fake_isco_clf()
    kwargs, isic_clf, _, _ = _base_kwargs(sre_enabled=False)
    r = run_eval.run_one_case(clf, **kwargs)
    isic_clf.classify.assert_called_once_with("software company")
    assert r.pred_isic_section == "J"
    assert r.pred_isic_division == "62"
    assert r.pred_isic_group == "620"
    assert r.pred_isic_class == "6201"


def test_sre_false_isced_and_iscedf_classification_executed():
    clf = _make_fake_isco_clf()
    kwargs, _, isced_clf, _ = _base_kwargs(sre_enabled=False)
    r = run_eval.run_one_case(clf, **kwargs)
    isced_clf.classify.assert_called_once_with("bachelor of science")
    assert r.pred_isced_level == "6"
    assert r.pred_isced_broad == "06"
    assert r.pred_isced_narrow == "061"
    assert r.pred_isced_detailed == "0613"


def test_sre_false_sre_output_is_explicit_disabled_status_not_blank():
    clf = _make_fake_isco_clf()
    kwargs, _, _, sre = _base_kwargs(sre_enabled=False)
    r = run_eval.run_one_case(clf, **kwargs)
    sre.analyse.assert_not_called()
    assert r.sre_status == "disabled_by_configuration"
    assert r.sre_status_reason == "semantic_relation_engine_disabled_by_configuration"
    assert r.sre_coherence_score is None  # never a fabricated 0.0
    assert r.sre_severity == ""  # never "NONE" (that would claim SRE ran and found nothing)


def test_sre_false_no_blank_classification_due_to_sre_disabled():
    """The core regression: none of the base classifier outputs may be
    blank/default merely because SRE is off."""
    clf = _make_fake_isco_clf()
    kwargs, _, _, _ = _base_kwargs(sre_enabled=False)
    r = run_eval.run_one_case(clf, **kwargs)
    assert r.pred_isco_4digit != ""
    assert r.pred_isic_section != ""
    assert r.pred_isced_level != ""


# ---------------------------------------------------------------------------
# 2. sre=True: same base classifiers run; SRE receives base outputs; SRE
# fields populated only when the engine actually evaluates the case
# ---------------------------------------------------------------------------

def test_sre_true_same_base_classifiers_run_as_sre_false():
    clf_a = _make_fake_isco_clf()
    kwargs_a, isic_a, isced_a, _ = _base_kwargs(sre_enabled=True)
    run_eval.run_one_case(clf_a, **kwargs_a)

    clf_b = _make_fake_isco_clf()
    kwargs_b, isic_b, isced_b, _ = _base_kwargs(sre_enabled=False)
    run_eval.run_one_case(clf_b, **kwargs_b)

    isic_a.classify.assert_called_once_with("software company")
    isic_b.classify.assert_called_once_with("software company")
    isced_a.classify.assert_called_once_with("bachelor of science")
    isced_b.classify.assert_called_once_with("bachelor of science")


def test_sre_true_receives_predicted_isco_isic_isced_outputs():
    clf = _make_fake_isco_clf(clf_result=_make_clf_result(code="2512"))
    kwargs, _, _, sre = _base_kwargs(sre_enabled=True)
    run_eval.run_one_case(clf, **kwargs)
    sre.analyse.assert_called_once()
    _, call_kwargs = sre.analyse.call_args
    assert call_kwargs["isco_code"] == "2512"
    assert call_kwargs["isic_section"] == "J"
    assert call_kwargs["isced_level"] == 6


def test_sre_true_fields_populated_when_evaluated():
    clf = _make_fake_isco_clf()
    kwargs, _, _, _ = _base_kwargs(sre_enabled=True)
    r = run_eval.run_one_case(clf, **kwargs)
    assert r.sre_status == "evaluated"
    assert r.sre_coherence_score == 0.95
    assert r.sre_severity == "NONE"  # evaluated, zero violations -- NOT blank


def test_sre_true_but_no_industry_or_education_text_skips_everything_consistently():
    """SRE=on cannot force ISIC/ISCED/SRE to run without the underlying
    free-text inputs -- this row was never in scope, independent of --sre."""
    clf = _make_fake_isco_clf()
    kwargs, isic_clf, isced_clf, sre = _base_kwargs(sre_enabled=True, industry_text="", education_text="")
    r = run_eval.run_one_case(clf, **kwargs)
    isic_clf.classify.assert_not_called()
    isced_clf.classify.assert_not_called()
    sre.analyse.assert_not_called()
    assert r.sre_status == "not_applicable"
    assert r.sre_status_reason == "industry_text and/or education_text not supplied on this row"


# ---------------------------------------------------------------------------
# 3. Isolation
# ---------------------------------------------------------------------------

def test_toggling_sre_does_not_change_classifier_method_selection():
    clf_a = _make_fake_isco_clf(clf_result=_make_clf_result(method="hierarchical_llm"))
    kwargs_a, _, _, _ = _base_kwargs(sre_enabled=True)
    r_a = run_eval.run_one_case(clf_a, **kwargs_a)

    clf_b = _make_fake_isco_clf(clf_result=_make_clf_result(method="hierarchical_llm"))
    kwargs_b, _, _, _ = _base_kwargs(sre_enabled=False)
    r_b = run_eval.run_one_case(clf_b, **kwargs_b)

    assert r_a.pred_method == r_b.pred_method == "hierarchical_llm"


def test_toggling_sre_does_not_change_reranker_invocation():
    """sre_enabled must never reach clf.classify()'s use_llm/trace args --
    toggling SRE is orthogonal to retrieval/reranking configuration."""
    clf_a = _make_fake_isco_clf()
    kwargs_a, _, _, _ = _base_kwargs(sre_enabled=True)
    run_eval.run_one_case(clf_a, **kwargs_a, use_llm_reranker=True)
    _, call_kwargs_a = clf_a.classify.call_args

    clf_b = _make_fake_isco_clf()
    kwargs_b, _, _, _ = _base_kwargs(sre_enabled=False)
    run_eval.run_one_case(clf_b, **kwargs_b, use_llm_reranker=True)
    _, call_kwargs_b = clf_b.classify.call_args

    assert call_kwargs_a["use_llm"] == call_kwargs_b["use_llm"] is True


def test_toggling_sre_does_not_suppress_isic_or_isced_result_generation():
    clf = _make_fake_isco_clf()
    kwargs, isic_clf, isced_clf, _ = _base_kwargs(sre_enabled=False)
    r = run_eval.run_one_case(clf, **kwargs)
    assert isic_clf.classify.called
    assert isced_clf.classify.called
    assert r.pred_isic_section and r.pred_isced_level


# ---------------------------------------------------------------------------
# 4. Evidence safety
# ---------------------------------------------------------------------------

def test_no_sre_outputs_cannot_report_severity_as_measured_zero():
    """sre_severity must be '' (not evaluated), never 'NONE' (which means
    'evaluated, zero violations') when SRE is disabled -- these are
    different claims and must not be conflated."""
    clf = _make_fake_isco_clf()
    kwargs, _, _, _ = _base_kwargs(sre_enabled=False)
    r = run_eval.run_one_case(clf, **kwargs)
    assert r.sre_severity != "NONE"
    assert r.sre_severity == ""


def test_disabled_sre_distinguishable_from_no_violation_found():
    clf_disabled = _make_fake_isco_clf()
    kwargs_disabled, _, _, _ = _base_kwargs(sre_enabled=False)
    r_disabled = run_eval.run_one_case(clf_disabled, **kwargs_disabled)

    clf_evaluated = _make_fake_isco_clf()
    kwargs_evaluated, _, _, sre_evaluated = _base_kwargs(sre_enabled=True)
    sre_evaluated.analyse.return_value = _make_sre_coherence(score=1.0, violations=[])
    r_evaluated = run_eval.run_one_case(clf_evaluated, **kwargs_evaluated)

    assert r_disabled.sre_status == "disabled_by_configuration"
    assert r_evaluated.sre_status == "evaluated"
    assert r_disabled.sre_status != r_evaluated.sre_status
    assert r_disabled.sre_severity != r_evaluated.sre_severity  # "" vs "NONE"


def test_ablation_runner_manifest_records_sre_configuration(tmp_path, monkeypatch):
    """Run-level manifest must record the SRE on/off configuration
    correctly (eval/ablation_runner.py::run_config()'s retrieval_params)."""
    import csv
    from types import SimpleNamespace as SNS
    import ablation_runner as ar

    monkeypatch.setattr(ar, "RAW_RUNS_DIR", tmp_path / "raw_runs")
    monkeypatch.setattr(ar, "DEV_SELECTION_DIR", tmp_path / "dev_selection")

    csv_path = tmp_path / "raw_runs" / "case_result.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case_id", "end_to_end_latency_ms", "escalation_triggered", "reranker_fired", "estimated_cost_usd", "peak_memory_mb"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow({"case_id": "c0", "end_to_end_latency_ms": "10.0", "escalation_triggered": "False", "reranker_fired": "True", "estimated_cost_usd": "0.0", "peak_memory_mb": ""})

    test_set = tmp_path / "test_set.csv"
    test_set.write_text("case_id,input_text\nc0,x\n", encoding="utf-8")

    def runner(argv, capture_output, text):
        return SNS(returncode=0, stdout=f"Wrote 1 row(s) to {csv_path}", stderr="")

    _, manifest_no_sre = ar.run_config("no_sre", test_set, "heldout", reranker_model="m", subprocess_runner=runner)
    _, manifest_with_sre = ar.run_config("with_sre", test_set, "heldout", reranker_model="m", subprocess_runner=runner)

    assert manifest_no_sre.retrieval_params["sre"] == "off"
    assert manifest_with_sre.retrieval_params["sre"] == "on"


# ---------------------------------------------------------------------------
# 5. Backward compatibility
# ---------------------------------------------------------------------------

def test_default_sre_enabled_omitted_still_runs_isic_isced_and_sre():
    """Omitting sre_enabled entirely (existing callers predating this
    parameter) must behave exactly as sre_enabled=True."""
    clf = _make_fake_isco_clf()
    kwargs, isic_clf, isced_clf, sre = _base_kwargs(sre_enabled=True)
    del kwargs["sre_enabled"]
    r = run_eval.run_one_case(clf, **kwargs)
    isic_clf.classify.assert_called_once()
    isced_clf.classify.assert_called_once()
    sre.analyse.assert_called_once()
    assert r.sre_status == "evaluated"


def test_case_result_new_sre_status_fields_default_harmless():
    """A CaseResult built without going through run_one_case (e.g. a
    dry-run row) must still expose sre_status/sre_status_reason with
    harmless defaults -- purely additive fields, no existing caller breaks."""
    r = run_eval.CaseResult(case_id="x", input_text="software developer", input_language="en")
    assert r.sre_status == "not_applicable"
    assert r.sre_status_reason == ""
