"""
Tests for the B2 additions to eval/run_eval.py: capture_pool_metadata
threading, the new CaseResult bookkeeping fields, and the
invalid_output_flag/timed_out_flag derivation in run_one_case().

Scope is deliberately B2-only -- this file does not re-test B0/B1 behaviour
(stage1-4 candidate columns, reranker_output, escalation, SRE, etc.), which
predates this change and is unaffected by it (see run_eval.py's module
docstring and the capture_pool_metadata docstrings in hierarchical_store.py
/ isco_classifier.py for why). What IS tested here is that every new field
defaults to a harmless, B0/B1-identical value when the new params are
omitted, and is populated correctly when they're supplied.

run_eval.py is not a package (no eval/__init__.py) -- imported the same way
run_eval.py itself reaches into backend/, via an explicit sys.path insert,
so this file can live next to the module it tests without a package init.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_eval  # noqa: E402


def make_clf_result(code="2512", title_en="Software Developers", confidence=0.9,
                     method="hierarchical", hitl_required=False, reasoning="Selected by LLM classifier."):
    return SimpleNamespace(
        primary=SimpleNamespace(code=code, title_en=title_en, title_ar="", confidence=confidence),
        method=method,
        hitl_required=hitl_required,
        reasoning=reasoning,
    )


def make_fake_clf(clf_result=None, trace_updates=None):
    """A duck-typed stand-in for ISCOClassifier/_BM25Adapter -- run_one_case()
    only ever calls .classify(job_title=, language=, top_k=, use_llm=, trace=)."""
    clf_result = clf_result or make_clf_result()
    trace_updates = trace_updates or {}

    clf = MagicMock()

    def _classify(job_title, language, top_k, use_llm, trace, max_stage_latency_ms=None):
        trace.update(trace_updates)
        return clf_result

    clf.classify.side_effect = _classify
    return clf


def base_kwargs():
    return dict(
        sre=MagicMock(), isic_clf=MagicMock(), isced_clf=MagicMock(),
        row_index=0, case_id="c1", input_text="software developer",
        input_language="en", gold_isco_4digit="2512", gold_isic="", gold_isced="",
        config_hash="abc123",
    )


# ---------------------------------------------------------------------------
# CaseResult defaults -- harmless for any existing B0/B1 row
# ---------------------------------------------------------------------------

def test_case_result_b2_fields_default_harmless():
    r = run_eval.CaseResult(case_id="c1", input_text="x", input_language="en")
    assert r.capture_pool_metadata_enabled is False
    assert r.stage4_pool_enriched == "[]"
    assert r.run_id == ""
    assert r.git_commit == ""
    assert r.seed is None
    assert r.input_order_position is None
    assert r.retry_count == 0
    assert r.invalid_output_flag is False
    assert r.timed_out_flag is False
    assert r.peak_memory_mb is None


# ---------------------------------------------------------------------------
# build_system() threads capture_pool_metadata through, defaults False
# ---------------------------------------------------------------------------

def test_build_system_passes_capture_pool_metadata_default_false(monkeypatch):
    captured = {}
    fake_cls = MagicMock(side_effect=lambda **kw: captured.update(kw))
    monkeypatch.setattr(run_eval, "ISCOClassifier", fake_cls)
    run_eval.build_system("hierarchical", reranker_model="ollama/llama3.2:1b")
    assert captured["capture_pool_metadata"] is False


def test_build_system_passes_capture_pool_metadata_true(monkeypatch):
    captured = {}
    fake_cls = MagicMock(side_effect=lambda **kw: captured.update(kw))
    monkeypatch.setattr(run_eval, "ISCOClassifier", fake_cls)
    run_eval.build_system("hierarchical", reranker_model="ollama/llama3.2:1b", capture_pool_metadata=True)
    assert captured["capture_pool_metadata"] is True


# ---------------------------------------------------------------------------
# run_one_case(): new params populate the new fields, default omitted
# ---------------------------------------------------------------------------

def test_run_one_case_b2_params_default_to_harmless_values():
    clf = make_fake_clf()
    r = run_eval.run_one_case(clf, **base_kwargs())
    assert r.capture_pool_metadata_enabled is False
    assert r.run_id == ""
    assert r.git_commit == ""
    assert r.seed is None
    assert r.input_order_position is None
    assert r.stage4_pool_enriched == "[]"


def test_run_one_case_b2_params_populate_when_passed():
    clf = make_fake_clf()
    r = run_eval.run_one_case(
        clf, **base_kwargs(),
        capture_pool_metadata_enabled=True, run_id="run-42", git_commit="abcdef1",
        seed=7, input_order_position=3,
    )
    assert r.capture_pool_metadata_enabled is True
    assert r.run_id == "run-42"
    assert r.git_commit == "abcdef1"
    assert r.seed == 7
    assert r.input_order_position == 3


def test_run_one_case_captures_stage4_pool_enriched_when_present():
    enriched = [{"code": "2512", "branch_id": "2/25/251", "source_rank": 1,
                 "raw_score": 0.8, "normalized_score": 1.0, "path": ["2", "25", "251", "2512"]}]
    clf = make_fake_clf(trace_updates={"stage4_pool_enriched": enriched})
    r = run_eval.run_one_case(clf, **base_kwargs(), capture_pool_metadata_enabled=True)
    assert r.stage4_pool_enriched != "[]"
    import json
    assert json.loads(r.stage4_pool_enriched) == enriched


def test_run_one_case_stage4_pool_untouched_by_enriched_capture():
    """B1's existing stage4_pool / gold_rank_in_pool must come from
    trace['stage4_pool'] only -- never from the new enriched trace key,
    even when both are present in the same trace."""
    pool = [{"code": "2512", "label_en": "", "score": 0.8}]
    enriched = [{"code": "9999", "branch_id": "x", "source_rank": 1,
                 "raw_score": 0.1, "normalized_score": 0.1, "path": []}]
    clf = make_fake_clf(trace_updates={"stage4_pool": pool, "stage4_pool_enriched": enriched})
    r = run_eval.run_one_case(clf, **base_kwargs(), capture_pool_metadata_enabled=True)
    import json
    assert json.loads(r.stage4_pool) == pool
    assert r.gold_rank_in_pool == 1  # gold_isco_4digit="2512" is pool[0], from stage4_pool not enriched


# ---------------------------------------------------------------------------
# invalid_output_flag / timed_out_flag derivation
# ---------------------------------------------------------------------------

def test_invalid_output_flag_true_when_reasoning_matches_parse_failure_string():
    clf_result = make_clf_result(reasoning="Fallback to top semantic match (LLM response could not be parsed).")
    clf = make_fake_clf(clf_result=clf_result)
    r = run_eval.run_one_case(clf, **base_kwargs())
    assert r.invalid_output_flag is True
    assert r.timed_out_flag is False


def test_invalid_output_flag_false_on_normal_selection():
    clf = make_fake_clf(clf_result=make_clf_result(reasoning="Selected by LLM classifier."))
    r = run_eval.run_one_case(clf, **base_kwargs())
    assert r.invalid_output_flag is False


def test_timed_out_flag_true_on_timeout_signature_reranker_error():
    clf = make_fake_clf(trace_updates={
        "reranker_error": "Timeout: Connection timed out after 120.0 seconds",
    })
    r = run_eval.run_one_case(clf, **base_kwargs())
    assert r.timed_out_flag is True
    assert r.degraded is True


def test_timed_out_flag_false_on_non_timeout_reranker_error():
    clf = make_fake_clf(trace_updates={"reranker_error": "ConnectionRefusedError: refused"})
    r = run_eval.run_one_case(clf, **base_kwargs())
    assert r.timed_out_flag is False
    assert r.degraded is True  # still degraded -- just not a timeout specifically


def test_retry_count_always_zero_for_standard_reranker():
    clf = make_fake_clf()
    r = run_eval.run_one_case(clf, **base_kwargs())
    assert r.retry_count == 0


# ---------------------------------------------------------------------------
# sre_enabled (Conference I Reviewer #2, Section E ablation support)
# ---------------------------------------------------------------------------

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


def _make_sre_coherence():
    return SimpleNamespace(score=0.95, violations=[])


def _kwargs_with_sre_inputs(sre_enabled):
    kwargs = base_kwargs()
    isic_clf = MagicMock()
    isic_clf.classify.return_value = _make_isic_result()
    isced_clf = MagicMock()
    isced_clf.classify.return_value = _make_isced_result()
    sre = MagicMock()
    sre.analyse.return_value = _make_sre_coherence()
    kwargs.update(
        isic_clf=isic_clf, isced_clf=isced_clf, sre=sre,
        industry_text="software company", education_text="bachelor of science",
        sre_enabled=sre_enabled,
    )
    return kwargs, isic_clf, isced_clf, sre


def test_sre_enabled_default_true_runs_isic_isced_and_sre():
    clf = make_fake_clf()
    kwargs, isic_clf, isced_clf, sre = _kwargs_with_sre_inputs(sre_enabled=True)
    r = run_eval.run_one_case(clf, **kwargs)
    isic_clf.classify.assert_called_once()
    isced_clf.classify.assert_called_once()
    sre.analyse.assert_called_once()
    assert r.pred_isic_section == "J"
    assert r.pred_isic_division == "62"
    assert r.pred_isic_class == "6201"
    assert r.pred_isced_broad == "06"
    assert r.pred_isced_detailed == "0613"
    assert r.sre_coherence_score == 0.95
    assert r.sre_status == "evaluated"
    assert r.sre_status_reason == ""


def test_sre_disabled_still_runs_isic_isced_but_skips_sre():
    """Conference I Reviewer #2, Step 5.1 (SRE-to-ISIC/ISCED coupling
    bugfix). --sre off must skip ONLY the SRE coherence check -- ISIC/ISCED
    classification must still run normally. Pre-fix, this test asserted the
    opposite (isic_clf.classify.assert_not_called()), which was the bug
    itself locked in as "expected" behaviour; see
    Documentation/Conference_I_Reviewer_2/SRE_COUPLING_BUGFIX.md."""
    clf = make_fake_clf()
    kwargs, isic_clf, isced_clf, sre = _kwargs_with_sre_inputs(sre_enabled=False)
    r = run_eval.run_one_case(clf, **kwargs)
    isic_clf.classify.assert_called_once()
    isced_clf.classify.assert_called_once()
    sre.analyse.assert_not_called()
    assert r.pred_isic_section == "J"
    assert r.pred_isic_division == "62"
    assert r.pred_isced_broad == "06"
    assert r.sre_coherence_score is None
    assert r.sre_severity == ""
    assert "sre_severity" not in r.escalation_reason
    assert r.sre_status == "disabled_by_configuration"
    assert r.sre_status_reason == "semantic_relation_engine_disabled_by_configuration"


def test_sre_disabled_default_omitted_param_behaves_as_enabled():
    """Omitting sre_enabled entirely must be identical to sre_enabled=True
    (the parameter's default) -- byte-for-byte the same as before this
    parameter existed."""
    clf = make_fake_clf()
    kwargs, isic_clf, isced_clf, sre = _kwargs_with_sre_inputs(sre_enabled=True)
    del kwargs["sre_enabled"]
    r = run_eval.run_one_case(clf, **kwargs)
    isic_clf.classify.assert_called_once()


# ---------------------------------------------------------------------------
# use_llm_reranker (Conference I Reviewer #2, Section E ablation support)
# ---------------------------------------------------------------------------

def test_use_llm_reranker_default_true_passes_use_llm_true_to_classifier():
    clf = make_fake_clf()
    run_eval.run_one_case(clf, **base_kwargs())
    _, kwargs = clf.classify.call_args
    assert kwargs["use_llm"] is True


def test_use_llm_reranker_false_passes_use_llm_false_to_classifier():
    clf = make_fake_clf()
    run_eval.run_one_case(clf, **base_kwargs(), use_llm_reranker=False)
    _, kwargs = clf.classify.call_args
    assert kwargs["use_llm"] is False


def test_use_llm_reranker_omitted_defaults_to_true():
    """Omitting use_llm_reranker entirely must be identical to
    use_llm_reranker=True -- byte-for-byte the same as before this
    parameter existed (it replaced a hardcoded use_llm=True call)."""
    clf = make_fake_clf()
    run_eval.run_one_case(clf, **base_kwargs())
    _, kwargs = clf.classify.call_args
    assert kwargs["use_llm"] is True


# ---------------------------------------------------------------------------
# Exact classifier-input regression test (schema-strengthening task, item 3):
# respondent_text/input_text must reach ISCOClassifier.classify() completely
# unchanged -- no normalization, translation, concatenation, rewriting, or
# truncation -- and the dev-set (eval/dev_set_v1.csv) mapping must be
# IDENTICAL to the mapping B0/B1 already use for
# eval/test_set_smoke20.csv/eval/test_set_full130.csv's own input_text
# column. No production code is touched by this test unless it proves an
# existing inconsistency (none was found).
# ---------------------------------------------------------------------------

# Deliberately adversarial text: mixed case (would change under
# lowercasing), internal double/irregular whitespace (would change under
# whitespace-collapsing), leading/trailing whitespace (would change under
# stripping), and >200 chars (would change under truncation). Any
# transformation applied before classify() would corrupt at least one of
# these properties, making it detectable by the assertions below.
_TRICKY_INPUT_TEXT = "  Software   DEVELOPER  \t building MOBILE apps" + (" extra words" * 20)


def test_run_one_case_passes_input_text_to_classify_verbatim():
    """run_eval.run_one_case() must call clf.classify(job_title=input_text)
    with NO transformation applied to input_text anywhere along the way."""
    captured = {}

    def _classify(job_title, language, top_k, use_llm, trace, max_stage_latency_ms=None):
        captured["job_title"] = job_title
        return make_clf_result()

    clf = MagicMock()
    clf.classify.side_effect = _classify

    kwargs = base_kwargs()
    kwargs["input_text"] = _TRICKY_INPUT_TEXT
    run_eval.run_one_case(clf, **kwargs)

    assert captured["job_title"] == _TRICKY_INPUT_TEXT  # byte-for-byte identical
    assert captured["job_title"] is _TRICKY_INPUT_TEXT or captured["job_title"] == _TRICKY_INPUT_TEXT
    # explicitly rule out the transformations a "helpful" normalization step
    # might apply, so this test would actually fail if one were added:
    assert captured["job_title"] != _TRICKY_INPUT_TEXT.strip()  # not stripped
    assert captured["job_title"] != _TRICKY_INPUT_TEXT.lower()  # not lowercased
    assert captured["job_title"] != " ".join(_TRICKY_INPUT_TEXT.split())  # whitespace not collapsed
    assert len(captured["job_title"]) == len(_TRICKY_INPUT_TEXT)  # not truncated


def test_dev_set_and_full130_style_calls_produce_identical_job_title():
    """Proves the dev-set input mapping is IDENTICAL to the frozen B1
    evaluation mapping: by the time run_one_case() is called, there is no
    code-path distinction between "this row came from eval/dev_set_v1.csv's
    respondent_text column" and "this row came from eval/test_set_full130
    .csv's input_text column" -- both become run_one_case(input_text=...),
    and this test proves both produce the exact same job_title at the
    classify() call site for the exact same underlying string."""
    captured = []

    def _classify(job_title, language, top_k, use_llm, trace, max_stage_latency_ms=None):
        captured.append(job_title)
        return make_clf_result()

    clf = MagicMock()
    clf.classify.side_effect = _classify

    # Path A: what dev_sweep.py's _load_dev_rows_as_test_set() produces from
    # a dev_set_v1.csv row (respondent_text -> input_text), fed to run_one_case.
    kwargs_a = base_kwargs()
    kwargs_a["case_id"] = "dev_case"
    kwargs_a["input_text"] = _TRICKY_INPUT_TEXT
    run_eval.run_one_case(clf, **kwargs_a)

    # Path B: what run_eval.py's own CSV loader produces from a
    # test_set_full130.csv row (input_text column, read directly in
    # run_eval.main() -- see row["input_text"] there), fed to run_one_case
    # with the exact same underlying string.
    kwargs_b = base_kwargs()
    kwargs_b["case_id"] = "full130_case"
    kwargs_b["input_text"] = _TRICKY_INPUT_TEXT
    run_eval.run_one_case(clf, **kwargs_b)

    assert captured[0] == captured[1] == _TRICKY_INPUT_TEXT


def test_dev_sweep_csv_to_classify_chain_is_verbatim_end_to_end(tmp_path):
    """Full chain: dev_set_v1.csv's respondent_text column -> dev_sweep.
    _load_dev_rows_as_test_set() -> run_eval.run_one_case()'s input_text
    param -> clf.classify()'s job_title param. Reads the real
    dev_sweep.py (not a reimplementation) so a future edit to the mapping
    would be caught here, not just in dev_sweep's own unit tests."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import dev_sweep as ds

    p = tmp_path / "dev.csv"
    p.write_text(
        "case_id,language,respondent_text,gold_isco_code,gold_label_source,"
        "annotator_or_adjudication_reference,dataset_split\n"
        f'dev001,en,"{_TRICKY_INPUT_TEXT}",7512,human_coder_single,AB,dev_v1\n',
        encoding="utf-8",
    )
    dev_rows = ds._load_dev_rows_as_test_set(p)
    assert dev_rows[0]["input_text"] == _TRICKY_INPUT_TEXT  # unchanged by the CSV->dict mapping

    captured = {}

    def _classify(job_title, language, top_k, use_llm, trace, max_stage_latency_ms=None):
        captured["job_title"] = job_title
        return make_clf_result()

    clf = MagicMock()
    clf.classify.side_effect = _classify

    row = dev_rows[0]
    run_eval.run_one_case(
        clf, sre=MagicMock(), isic_clf=MagicMock(), isced_clf=MagicMock(), row_index=0,
        case_id=row["case_id"], input_text=row["input_text"], input_language=row["input_language"],
        gold_isco_4digit=row["gold_isco_4digit"], gold_isic="", gold_isced="", config_hash="x",
    )
    assert captured["job_title"] == _TRICKY_INPUT_TEXT  # verbatim, end to end
