"""
Tests for eval/dev_sweep.py's pure functions -- compute_metrics(),
check_eligibility(), select_k(), render_markdown_report() -- implementing
the B2 operational eligibility rule (see dev_sweep.py's module docstring).
No live LLM/Qdrant calls; case results are plain SimpleNamespace stand-ins
exposing the same attributes as run_eval.CaseResult.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dev_sweep as ds  # noqa: E402


def case(gold="2512", pred="2512", gold_rank_in_pool=1, latency=100.0,
         invalid_output_flag=False, timed_out_flag=False, error=""):
    return SimpleNamespace(
        gold_isco_4digit=gold, pred_isco_4digit=pred, gold_rank_in_pool=gold_rank_in_pool,
        end_to_end_latency_ms=latency, invalid_output_flag=invalid_output_flag,
        timed_out_flag=timed_out_flag, error=error,
    )


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

def test_compute_metrics_empty_case_list():
    m = ds.compute_metrics([], k=5)
    assert m.n_cases == 0
    assert m.candidate_recall_at_k == 0.0
    assert m.top1_accuracy_all_case == 0.0


def test_compute_metrics_empty_with_expected_cases_is_incomplete():
    m = ds.compute_metrics([], k=5, n_expected_cases=30)
    assert m.incomplete_run is True


def test_compute_metrics_candidate_recall_not_k_dependent_rerun():
    cases = [case(gold_rank_in_pool=3), case(gold_rank_in_pool=7), case(gold_rank_in_pool=None)]
    m5 = ds.compute_metrics(cases, k=5)
    assert m5.candidate_recall_at_k == 1 / 3  # only rank=3 is <=5
    m10 = ds.compute_metrics(cases, k=10)
    assert m10.candidate_recall_at_k == 2 / 3  # rank=3 and rank=7 both <=10


def test_compute_metrics_top1_all_case_accuracy():
    cases = [case(gold="2512", pred="2512"), case(gold="2511", pred="9999")]
    m = ds.compute_metrics(cases, k=5)
    assert m.top1_accuracy_all_case == 0.5


def test_compute_metrics_blank_prediction_never_counts_as_a_hit():
    cases = [case(gold="2512", pred="")]
    m = ds.compute_metrics(cases, k=5)
    assert m.top1_accuracy_all_case == 0.0


def test_compute_metrics_timeout_or_invalid_case_counts_as_incorrect_even_if_pred_matches_gold():
    """The fallback-on-failure prediction could coincidentally equal gold --
    all-case Top-1 must still count it as incorrect (rule #2)."""
    cases = [case(gold="2512", pred="2512", timed_out_flag=True)]
    m = ds.compute_metrics(cases, k=5)
    assert m.top1_accuracy_all_case == 0.0


def test_compute_metrics_conditional_accuracy_excludes_unrecovered_cases_from_denominator():
    cases = [
        case(gold="2512", pred="2512"),                     # successful hit
        case(gold="2511", pred="2511", timed_out_flag=True),  # excluded from conditional
        case(gold="2519", pred="0000", invalid_output_flag=True),  # excluded from conditional
    ]
    m = ds.compute_metrics(cases, k=5)
    assert m.n_successful_cases == 1
    assert m.top1_accuracy_conditional == 1.0  # 1/1, the timed-out/invalid cases don't count
    assert m.top1_accuracy_all_case == 1 / 3   # denom is still all 3 cases


def test_compute_metrics_conditional_accuracy_zero_when_no_successful_cases():
    cases = [case(timed_out_flag=True), case(invalid_output_flag=True)]
    m = ds.compute_metrics(cases, k=5)
    assert m.n_successful_cases == 0
    assert m.top1_accuracy_conditional == 0.0


def test_compute_metrics_raw_counts_and_rates_both_recorded():
    cases = [
        case(invalid_output_flag=True), case(timed_out_flag=True),
        case(invalid_output_flag=False, timed_out_flag=False), case(),
    ]
    m = ds.compute_metrics(cases, k=5)
    assert m.n_invalid_outputs == 1
    assert m.n_timeouts == 1
    assert m.invalid_output_rate == 0.25
    assert m.timeout_rate == 0.25


def test_compute_metrics_latency_median_p95_max():
    cases = [case(latency=v) for v in [100, 200, 300, 400, 500]]
    m = ds.compute_metrics(cases, k=5)
    assert m.median_latency_ms == 300.0
    assert m.max_latency_ms == 500.0


def test_compute_metrics_n_hard_failures_only_counts_hard_failures():
    cases = [case(pred="", error="ValueError: boom"), case(pred="2512", error="")]
    m = ds.compute_metrics(cases, k=5)
    assert m.n_hard_failures == 1


def test_compute_metrics_crashed_and_corrupted_flags_pass_through():
    m = ds.compute_metrics([case()], k=5, crashed=True, crash_message="MemoryError: oom",
                            corrupted_output=True, peak_memory_mb=9999.0)
    assert m.crashed is True
    assert m.crash_message == "MemoryError: oom"
    assert m.corrupted_output is True
    assert m.peak_memory_mb == 9999.0


# ---------------------------------------------------------------------------
# check_eligibility -- rule #1-4
# ---------------------------------------------------------------------------

def kresult(k, recall=0.8, top1=0.7, n_cases=30, n_expected_cases=30, n_hard_failures=0,
            incomplete_run=False, crashed=False, crash_message="", corrupted_output=False,
            n_timeouts=0, n_invalid_outputs=0, median_latency_ms=100.0, p95_latency_ms=150.0,
            max_latency_ms=200.0, peak_memory_mb=500.0):
    n_successful = n_cases - n_timeouts - n_invalid_outputs
    return ds.KSweepResult(
        k=k, n_expected_cases=n_expected_cases, n_cases=n_cases, n_hard_failures=n_hard_failures,
        incomplete_run=incomplete_run, crashed=crashed, crash_message=crash_message,
        corrupted_output=corrupted_output, n_timeouts=n_timeouts, n_invalid_outputs=n_invalid_outputs,
        timeout_rate=n_timeouts / n_cases if n_cases else 0.0,
        invalid_output_rate=n_invalid_outputs / n_cases if n_cases else 0.0,
        candidate_recall_at_k=recall, n_successful_cases=n_successful,
        top1_accuracy_all_case=top1,
        top1_accuracy_conditional=top1,
        median_latency_ms=median_latency_ms, p95_latency_ms=p95_latency_ms,
        max_latency_ms=max_latency_ms, mean_latency_ms=median_latency_ms,
        peak_memory_mb=peak_memory_mb,
    )


def test_eligibility_passes_reference_against_itself():
    ref = kresult(5)
    ok, reasons, factor = ds.check_eligibility(ref, ref)
    assert ok is True
    assert reasons == []
    assert factor == 1.0


def test_eligibility_rejects_crashed():
    ref = kresult(5)
    candidate = kresult(10, crashed=True, crash_message="MemoryError: oom")
    ok, reasons, _ = ds.check_eligibility(candidate, ref)
    assert ok is False
    assert any("crash" in r for r in reasons)


def test_eligibility_rejects_incomplete_run():
    ref = kresult(5)
    candidate = kresult(10, n_cases=20, n_expected_cases=30, incomplete_run=True)
    ok, reasons, _ = ds.check_eligibility(candidate, ref)
    assert ok is False
    assert any("incomplete run" in r for r in reasons)


def test_eligibility_rejects_corrupted_output():
    ref = kresult(5)
    candidate = kresult(10, corrupted_output=True)
    ok, reasons, _ = ds.check_eligibility(candidate, ref)
    assert ok is False
    assert any("read-back" in r for r in reasons)


def test_eligibility_rejects_hard_failures():
    ref = kresult(5)
    candidate = kresult(10, n_hard_failures=2)
    ok, reasons, _ = ds.check_eligibility(candidate, ref)
    assert ok is False
    assert any("hard case failure" in r for r in reasons)


def test_eligibility_allows_one_additional_timeout_or_invalid_vs_reference():
    ref = kresult(5, n_timeouts=0, n_invalid_outputs=0)
    candidate = kresult(10, n_timeouts=1, n_invalid_outputs=0)  # delta = 1, allowed by default
    ok, reasons, _ = ds.check_eligibility(candidate, ref, max_additional_timeout_or_invalid=1)
    assert ok is True


def test_eligibility_rejects_two_additional_timeout_or_invalid_vs_reference():
    ref = kresult(5, n_timeouts=0, n_invalid_outputs=0)
    candidate = kresult(10, n_timeouts=1, n_invalid_outputs=1)  # delta = 2, exceeds default of 1
    ok, reasons, _ = ds.check_eligibility(candidate, ref, max_additional_timeout_or_invalid=1)
    assert ok is False
    assert any("timeout+invalid_output count" in r for r in reasons)


def test_eligibility_rejects_p95_latency_over_ceiling():
    ref = kresult(5, p95_latency_ms=100.0)
    candidate = kresult(10, p95_latency_ms=200.0)  # 2.0x, exceeds default 1.5x ceiling
    ok, reasons, factor = ds.check_eligibility(candidate, ref, p95_ceiling_factor=1.5)
    assert ok is False
    assert factor == 2.0
    assert any("P95 latency" in r for r in reasons)


def test_eligibility_allows_p95_latency_within_ceiling():
    ref = kresult(5, p95_latency_ms=100.0)
    candidate = kresult(10, p95_latency_ms=140.0)  # 1.4x, within default 1.5x ceiling
    ok, reasons, factor = ds.check_eligibility(candidate, ref, p95_ceiling_factor=1.5)
    assert ok is True
    assert factor == 1.4


def test_eligibility_memory_budget_skipped_when_not_set():
    ref = kresult(5, peak_memory_mb=9000.0)
    candidate = kresult(10, peak_memory_mb=9000.0)
    ok, reasons, _ = ds.check_eligibility(candidate, ref, memory_budget_mb=None)
    assert ok is True  # no budget set -> not rejected, however high


def test_eligibility_rejects_over_memory_budget():
    ref = kresult(5)
    candidate = kresult(10, peak_memory_mb=5000.0)
    ok, reasons, _ = ds.check_eligibility(candidate, ref, memory_budget_mb=4000.0)
    assert ok is False
    assert any("peak_memory_mb" in r for r in reasons)


def test_eligibility_missing_peak_memory_does_not_falsely_reject():
    ref = kresult(5, peak_memory_mb=None)
    candidate = kresult(10, peak_memory_mb=None)
    ok, reasons, _ = ds.check_eligibility(candidate, ref, memory_budget_mb=100.0)
    assert ok is True  # can't reject on a measurement that wasn't taken (e.g. no psutil)


# ---------------------------------------------------------------------------
# select_k -- eligibility-gated 5-step rule
# ---------------------------------------------------------------------------

def test_select_k_empty_results():
    decision = ds.select_k([])
    assert decision.chosen_k is None


def test_select_k_missing_reference_returns_no_selection():
    results = [kresult(8), kresult(10)]  # no K=5 (default reference)
    decision = ds.select_k(results, reference_k=5)
    assert decision.chosen_k is None
    assert any("not among the supplied results" in r for r in decision.rationale)


def test_select_k_picks_max_recall_among_eligible_when_not_close():
    results = [kresult(5, recall=0.80, top1=0.90), kresult(10, recall=0.95, top1=0.50)]
    decision = ds.select_k(results, reference_k=5, recall_close_threshold=0.02)
    assert decision.chosen_k == 10


def test_select_k_breaks_close_recall_tie_by_all_case_top1():
    results = [kresult(5, recall=0.90, top1=0.60), kresult(10, recall=0.91, top1=0.80)]
    decision = ds.select_k(results, reference_k=5, recall_close_threshold=0.02)
    assert decision.chosen_k == 10


def test_select_k_prefers_smaller_k_on_exact_tie():
    results = [
        kresult(5, recall=0.90, top1=0.70), kresult(10, recall=0.90, top1=0.70),
        kresult(15, recall=0.90, top1=0.70),
    ]
    decision = ds.select_k(results, reference_k=5, recall_close_threshold=0.02)
    assert decision.chosen_k == 5


def test_select_k_excludes_ineligible_k_from_consideration_even_with_best_numbers():
    results = [
        kresult(5, recall=0.80, top1=0.70),   # reference
        kresult(10, recall=0.99, top1=0.99, crashed=True, crash_message="oom"),  # best numbers, ineligible
        kresult(15, recall=0.85, top1=0.75),
    ]
    decision = ds.select_k(results, reference_k=5)
    assert 10 in decision.rejected
    assert decision.chosen_k == 15


def test_select_k_all_ineligible_returns_none_with_explanation():
    results = [kresult(5, crashed=True, crash_message="oom")]
    decision = ds.select_k(results, reference_k=5)
    assert decision.chosen_k is None
    assert 5 in decision.rejected
    assert any("no k can be selected" in r.lower() for r in decision.rationale)


def test_select_k_never_needs_full130_data():
    """Sanity/documentation check: select_k's signature has no parameter
    that could carry eval/test_set_full130.csv data -- the held-out set
    structurally cannot influence this decision."""
    import inspect
    params = inspect.signature(ds.select_k).parameters
    assert "full130" not in " ".join(params).lower()
    assert "held_out" not in " ".join(params).lower()


# ---------------------------------------------------------------------------
# render_markdown_report -- smoke test on shape, not exact text
# ---------------------------------------------------------------------------

def test_render_markdown_report_contains_chosen_k_and_table():
    results = [kresult(5, recall=0.8, top1=0.7), kresult(10, recall=0.9, top1=0.75)]
    decision = ds.select_k(results, reference_k=5)
    report = ds.render_markdown_report(
        decision, results, "eval/dev_set_v1.csv", "ollama/llama3.2:1b", 30, "2026-01-01T00:00:00Z",
    )
    assert "# B2 dev-set K-sweep report" in report
    assert f"Chosen K = {decision.chosen_k}" in report
    assert "| 5 |" in report
    assert "| 10 |" in report


def test_render_markdown_report_no_selection_case():
    results = [kresult(5, crashed=True, crash_message="oom")]
    decision = ds.select_k(results, reference_k=5)
    report = ds.render_markdown_report(
        decision, results, "eval/dev_set_v1.csv", "ollama/llama3.2:1b", 10, "2026-01-01T00:00:00Z",
    )
    assert "No K selected" in report


def test_render_markdown_report_notes_memory_budget_not_set():
    results = [kresult(5)]
    decision = ds.select_k(results, reference_k=5, memory_budget_mb=None)
    report = ds.render_markdown_report(
        decision, results, "eval/dev_set_v1.csv", "ollama/llama3.2:1b", 10, "2026-01-01T00:00:00Z",
        memory_budget_mb=None,
    )
    assert "not set -- check skipped" in report


def test_render_markdown_report_includes_determinism_and_baseline_provenance():
    results = [kresult(5, recall=0.8, top1=0.7)]
    decision = ds.select_k(results, reference_k=5)
    report = ds.render_markdown_report(
        decision, results, "eval/dev_set_v1.csv", "ollama/llama3.2:latest", 10, "2026-01-01T00:00:00Z",
        baseline_config_path="eval/configs/b1_frozen.json",
        baseline={"beam": 3, "stage1_mode": "leaf_vote", "keyword_map_enabled": False,
                  "llm_temperature": 0.0, "timeout_s": 120, "prompt_fingerprint": "abc123",
                  "_source_csv": "eval/results/raw_runs/x.csv", "_source_top1_accuracy": "54/130"},
        git_commit="deadbeef", seed=42, dev_row_order=["c1", "c2"], k_execution_order=[5],
    )
    assert "eval/configs/b1_frozen.json" in report
    assert "deadbeef" in report
    assert "Seed: 42" in report
    assert "c1, c2" in report
    assert "[5]" in report


# ---------------------------------------------------------------------------
# Baseline config: load, shape validation, and codebase cross-check
# ---------------------------------------------------------------------------

FAKE_MODEL_DIGEST = "fake0000digest0000for0000hermetic0000tests0000not0000live0000ollama0000query"


def valid_baseline(**overrides):
    """A hypothetical CURRENT, sweep-ready baseline -- implementation_
    fingerprint is computed live, so this always matches the live codebase
    by construction. Represents what a fresh B1 re-freeze would look like;
    distinct from the real shipped eval/configs/b1_frozen.json, which is
    deliberately historical/stale (Conference I Reviewer #2, Task 04) --
    see test_the_actual_shipped_b1_frozen_json_is_correctly_quarantined()."""
    fingerprint = ds.compute_composite_fingerprint()
    baseline = {
        "reranker_model": "ollama/llama3.2:latest",
        "beam": 3,
        "stage1_mode": "leaf_vote",
        "keyword_map_enabled": False,
        "branch_collapse": False,
        "llm_temperature": ds.HARDCODED_LLM_TEMPERATURE,
        "timeout_s": ds._OLLAMA_INFERENCE_TIMEOUT,
        "beam_evidence": {"status": "confirmed", "source": "test fixture", "detail": "n/a"},
        "ollama_model_identity": {"tag": "llama3.2:latest", "digest": FAKE_MODEL_DIGEST},
        "implementation_fingerprint": fingerprint,
        "baseline_validity": {
            "status": "current_verified_ready",
            "b2_sweep_permitted": True,
            "reason": "test fixture -- represents a hypothetical fresh re-freeze",
            "permitted_use": "test fixture only",
            "re_freeze_requires": "n/a -- already current",
        },
    }
    baseline.update(overrides)
    return baseline


@pytest.fixture(autouse=False)
def fake_ollama_identity(monkeypatch):
    """Hermetic stand-in for resolve_ollama_model_identity() -- tests must
    not depend on Ollama actually running/having this exact model pulled.
    Matches valid_baseline()'s ollama_model_identity fixture value."""
    monkeypatch.setattr(
        ds, "resolve_ollama_model_identity",
        lambda tag: {"status": "confirmed", "tag": tag, "digest": FAKE_MODEL_DIGEST,
                     "parameter_size": "3.2B", "quantization_level": "Q4_K_M",
                     "family": "llama", "size_bytes": 123},
    )


def test_load_baseline_config_reads_json(tmp_path):
    p = tmp_path / "baseline.json"
    p.write_text('{"reranker_model": "ollama/llama3.2:latest", "beam": 3}', encoding="utf-8")
    cfg = ds.load_baseline_config(p)
    assert cfg["reranker_model"] == "ollama/llama3.2:latest"
    assert cfg["beam"] == 3


def test_validate_baseline_shape_valid_config_has_no_errors():
    assert ds.validate_baseline_shape(valid_baseline()) == []


def test_validate_baseline_shape_reports_missing_fields():
    errors = ds.validate_baseline_shape({"reranker_model": "ollama/llama3.2:latest"})
    assert errors
    assert "missing required field" in errors[0]


def test_validate_baseline_shape_rejects_bad_reranker_model_prefix():
    errors = ds.validate_baseline_shape(valid_baseline(reranker_model="llama3.2:latest"))
    assert any("reranker_model" in e for e in errors)


def test_validate_baseline_shape_rejects_non_positive_beam():
    errors = ds.validate_baseline_shape(valid_baseline(beam=0))
    assert any("beam" in e for e in errors)


def test_validate_baseline_shape_rejects_bad_stage1_mode():
    errors = ds.validate_baseline_shape(valid_baseline(stage1_mode="bogus"))
    assert any("stage1_mode" in e for e in errors)


def test_validate_baseline_shape_rejects_bad_beam_evidence_status():
    errors = ds.validate_baseline_shape(valid_baseline(beam_evidence={"status": "maybe"}))
    assert any("beam_evidence" in e for e in errors)


def test_validate_baseline_shape_rejects_incomplete_model_identity():
    errors = ds.validate_baseline_shape(valid_baseline(ollama_model_identity={"tag": "x"}))  # no digest
    assert any("ollama_model_identity" in e for e in errors)


def test_validate_baseline_shape_rejects_empty_fingerprint_components():
    errors = ds.validate_baseline_shape(
        valid_baseline(implementation_fingerprint={"components": {}, "composite_sha256": "abc"})
    )
    assert any("implementation_fingerprint" in e for e in errors)


# ---------------------------------------------------------------------------
# baseline_validity shape (Conference I Reviewer #2, Task 04 -- B1 baseline
# quarantine after the hierarchy-engine refactor)
# ---------------------------------------------------------------------------

def test_validate_baseline_shape_rejects_missing_baseline_validity():
    baseline = valid_baseline()
    del baseline["baseline_validity"]
    errors = ds.validate_baseline_shape(baseline)
    assert any("missing required field" in e and "baseline_validity" in e for e in errors)


def test_validate_baseline_shape_rejects_bad_baseline_validity_status():
    errors = ds.validate_baseline_shape(
        valid_baseline(baseline_validity={
            "status": "totally_fine_trust_me", "b2_sweep_permitted": True,
            "reason": "x", "permitted_use": "x", "re_freeze_requires": "x",
        })
    )
    assert any("baseline_validity.status" in e for e in errors)


def test_validate_baseline_shape_rejects_non_bool_b2_sweep_permitted():
    errors = ds.validate_baseline_shape(
        valid_baseline(baseline_validity={
            "status": "historical_stale_requires_rerun", "b2_sweep_permitted": "false",
            "reason": "x", "permitted_use": "x", "re_freeze_requires": "x",
        })
    )
    assert any("b2_sweep_permitted" in e for e in errors)


def test_validate_baseline_shape_rejects_stale_status_with_sweep_permitted_true():
    """The exact internal-inconsistency the task calls out by name: a
    baseline cannot claim to be historical/stale AND simultaneously permit
    a B2 sweep."""
    errors = ds.validate_baseline_shape(
        valid_baseline(baseline_validity={
            "status": "historical_stale_requires_rerun", "b2_sweep_permitted": True,
            "reason": "x", "permitted_use": "x", "re_freeze_requires": "x",
        })
    )
    assert any("internally inconsistent" in e for e in errors)


def test_validate_baseline_shape_rejects_blank_baseline_validity_reason():
    errors = ds.validate_baseline_shape(
        valid_baseline(baseline_validity={
            "status": "current_verified_ready", "b2_sweep_permitted": True,
            "reason": "", "permitted_use": "x", "re_freeze_requires": "x",
        })
    )
    assert any("baseline_validity.reason" in e for e in errors)


def test_validate_baseline_shape_accepts_well_formed_current_baseline_validity():
    assert ds.validate_baseline_shape(valid_baseline()) == []


# ---------------------------------------------------------------------------
# Composite implementation fingerprint (task C)
# ---------------------------------------------------------------------------

def test_compute_composite_fingerprint_is_stable_and_deterministic():
    a = ds.compute_composite_fingerprint()
    b = ds.compute_composite_fingerprint()
    assert a == b
    assert len(a["composite_sha256"]) == 64
    assert set(a["components"]) == set(ds.IMPLEMENTATION_FINGERPRINT_TARGETS)


def test_compute_composite_fingerprint_component_names_are_exactly_the_documented_three():
    fp = ds.compute_composite_fingerprint()
    assert set(fp["components"].keys()) == {
        "HierarchicalISCOStore._hierarchical_search",
        "ISCOClassifier._llm_select_from_candidates",
        "ISCOClassifier._parse_llm_response",
    }


def test_compute_composite_fingerprint_composite_depends_on_every_component():
    """Changing what a single component hashes to (simulated by monkeypatching
    IMPLEMENTATION_FINGERPRINT_TARGETS) must change the composite -- proves
    the composite isn't silently derived from just one of the three."""
    baseline_fp = ds.compute_composite_fingerprint()

    class _Dummy:
        def dummy(self):
            return "different source text entirely"

    import types as _types
    patched = dict(ds.IMPLEMENTATION_FINGERPRINT_TARGETS)
    patched["ISCOClassifier._parse_llm_response"] = lambda: _Dummy.dummy
    old = ds.IMPLEMENTATION_FINGERPRINT_TARGETS
    ds.IMPLEMENTATION_FINGERPRINT_TARGETS = patched
    try:
        changed_fp = ds.compute_composite_fingerprint()
    finally:
        ds.IMPLEMENTATION_FINGERPRINT_TARGETS = old
    assert changed_fp["composite_sha256"] != baseline_fp["composite_sha256"]


def test_compute_live_prompt_fingerprint_still_works_as_legacy_wrapper():
    a = ds.compute_live_prompt_fingerprint()
    assert isinstance(a, str) and len(a) == 16


# ---------------------------------------------------------------------------
# assert_baseline_matches_codebase (task B/C additions)
# ---------------------------------------------------------------------------

def test_assert_baseline_matches_codebase_passes_for_valid_baseline(fake_ollama_identity):
    ds.assert_baseline_matches_codebase(valid_baseline())  # must not raise


def test_assert_baseline_matches_codebase_rejects_branch_collapse_true(fake_ollama_identity):
    with pytest.raises(ds.BaselineMismatchError, match="branch_collapse"):
        ds.assert_baseline_matches_codebase(valid_baseline(branch_collapse=True))


def test_assert_baseline_matches_codebase_rejects_wrong_temperature(fake_ollama_identity):
    with pytest.raises(ds.BaselineMismatchError, match="llm_temperature"):
        ds.assert_baseline_matches_codebase(valid_baseline(llm_temperature=0.7))


def test_assert_baseline_matches_codebase_rejects_wrong_timeout(fake_ollama_identity):
    with pytest.raises(ds.BaselineMismatchError, match="timeout_s"):
        ds.assert_baseline_matches_codebase(valid_baseline(timeout_s=30))


def test_assert_baseline_matches_codebase_rejects_stale_composite_fingerprint(fake_ollama_identity):
    stale = {"components": {"a": "0000000000000000"}, "composite_sha256": "0" * 64}
    with pytest.raises(ds.BaselineMismatchError, match="implementation_fingerprint"):
        ds.assert_baseline_matches_codebase(valid_baseline(implementation_fingerprint=stale))


def test_assert_baseline_matches_codebase_rejects_stale_baseline_validity(fake_ollama_identity):
    """Even a baseline whose fingerprint/temperature/timeout/branch_collapse
    would all otherwise PASS must still be rejected if baseline_validity
    itself says the sweep is not permitted -- this check is independent of,
    and does not require, a fingerprint mismatch to fire."""
    stale = valid_baseline(baseline_validity={
        "status": "historical_stale_requires_rerun", "b2_sweep_permitted": False,
        "reason": "test: pretend this fresh-fingerprint baseline is actually stale",
        "permitted_use": "test fixture only", "re_freeze_requires": "n/a",
    })
    with pytest.raises(ds.BaselineMismatchError, match="baseline_validity"):
        ds.assert_baseline_matches_codebase(stale)


def test_assert_baseline_matches_codebase_rejects_model_identity_mismatch(fake_ollama_identity):
    with pytest.raises(ds.BaselineMismatchError, match="ollama_model_identity"):
        ds.assert_baseline_matches_codebase(
            valid_baseline(ollama_model_identity={"tag": "llama3.2:latest", "digest": "wrong-digest"})
        )


def test_assert_baseline_matches_codebase_fails_closed_when_ollama_unreachable(monkeypatch):
    monkeypatch.setattr(
        ds, "resolve_ollama_model_identity",
        lambda tag: {"status": "unavailable", "tag": tag, "reason": "connection refused"},
    )
    with pytest.raises(ds.BaselineMismatchError, match="could not confirm"):
        ds.assert_baseline_matches_codebase(valid_baseline())


def test_assert_baseline_matches_codebase_reports_all_mismatches_at_once(fake_ollama_identity):
    baseline = valid_baseline(branch_collapse=True, llm_temperature=0.7, timeout_s=1)
    try:
        ds.assert_baseline_matches_codebase(baseline)
        assert False, "expected BaselineMismatchError"
    except ds.BaselineMismatchError as exc:
        msg = str(exc)
        assert "branch_collapse" in msg
        assert "llm_temperature" in msg
        assert "timeout_s" in msg


def test_the_actual_shipped_b1_frozen_json_is_correctly_quarantined(fake_ollama_identity):
    """Conference I Reviewer #2, Task 04 (B1 baseline quarantine); Task 04.1
    (hermetic test isolation). The real eval/configs/b1_frozen.json this
    repo ships is INTENTIONALLY historical and stale as of the hierarchy-
    engine refactor (backend/rag/hierarchical_store.py's
    _hierarchical_search now delegates to backend/rag/hierarchy_engine.py,
    changing its source and therefore its implementation-fingerprint hash)
    -- this is an expected, permanent consequence of that refactor, not a
    newly-measured regression, and not something to "fix" by re-running B1
    in this test. This test asserts the file is well-formed AND correctly
    self-reports as not sweep-ready, exactly the safety state the
    repository must be in until a real B1 re-freeze happens (separate
    explicit approval required -- see
    Documentation/AI_HANDOFF/CLAUDE_B1_BASELINE_STATUS_REPORT.md).

    Uses fake_ollama_identity (Task 04.1): this test's purpose is to prove
    the fail-closed BASELINE_VALIDITY/fingerprint quarantine state, which
    has nothing to do with Ollama model identity -- it does not need, and
    must not require, a live local Ollama with this exact model pulled, so
    it no longer makes a real /api/tags request. check_ollama_model_
    identity() itself is NOT weakened, removed, or bypassed anywhere in
    production -- a real B2 run (eval/dev_sweep.py's own main(), invoked
    outside of tests) still calls the real resolve_ollama_model_identity()
    and still fails closed exactly as before if Ollama is unreachable or
    the digest doesn't match; only THIS unit test substitutes a
    deterministic identity fixture, the same substitution every other
    assert_baseline_matches_codebase() test in this file already uses."""
    path = Path(__file__).resolve().parent / "configs" / "b1_frozen.json"
    baseline = ds.load_baseline_config(path)

    # 1. The shipped JSON passes shape/metadata validation -- it is
    # well-formed, not malformed; "stale" and "malformed" are different
    # things, and only the latter is a shape error.
    assert ds.validate_baseline_shape(baseline) == []

    # 2/3. It correctly, explicitly self-reports as historical/stale and
    # not permitted to seed a sweep -- never silently "upgraded".
    assert baseline["baseline_validity"]["status"] == "historical_stale_requires_rerun"
    assert baseline["baseline_validity"]["b2_sweep_permitted"] is False

    # 4/5. assert_baseline_matches_codebase() still raises BaselineMismatchError
    # (fail-closed, not bypassed), and the message clearly identifies BOTH
    # the baseline_validity self-report AND the underlying implementation-
    # fingerprint mismatch that caused it -- a reader is never left
    # guessing why the sweep is blocked, and it is never silently permitted.
    with pytest.raises(ds.BaselineMismatchError) as exc_info:
        ds.assert_baseline_matches_codebase(baseline)
    msg = str(exc_info.value)
    assert "baseline_validity" in msg
    assert "historical/stale" in msg
    assert "implementation_fingerprint" in msg

    assert baseline["reranker_model"] == "ollama/llama3.2:latest"  # NOT ...1b, see file's _notes
    assert baseline["beam_evidence"]["status"] == "inferred"  # honest, not silently upgraded
    assert baseline["_source_top1_accuracy"] == "54/130"  # historical result, unchanged


# ---------------------------------------------------------------------------
# Beam-provenance gate (task A)
# ---------------------------------------------------------------------------

def test_check_beam_evidence_confirmed_passes_without_override():
    ok, msg = ds.check_beam_evidence(valid_baseline(), confirm_inferred_beam=None)
    assert ok is True


def test_check_beam_evidence_inferred_fails_without_override():
    baseline = valid_baseline(beam_evidence={"status": "inferred", "source": "x", "detail": "y"})
    ok, msg = ds.check_beam_evidence(baseline, confirm_inferred_beam=None)
    assert ok is False
    assert "confirm-inferred-beam" in msg


def test_check_beam_evidence_inferred_passes_with_correct_override():
    baseline = valid_baseline(beam=3, beam_evidence={"status": "inferred", "source": "x", "detail": "y"})
    ok, msg = ds.check_beam_evidence(baseline, confirm_inferred_beam=3)
    assert ok is True
    assert "OVERRIDE" in msg


def test_check_beam_evidence_inferred_fails_with_wrong_override_value():
    baseline = valid_baseline(beam=3, beam_evidence={"status": "inferred", "source": "x", "detail": "y"})
    ok, msg = ds.check_beam_evidence(baseline, confirm_inferred_beam=5)
    assert ok is False
    assert "does not match" in msg


# ---------------------------------------------------------------------------
# Determinism: dev-row shuffle and K execution order
# ---------------------------------------------------------------------------

def test_deterministic_shuffle_same_seed_same_order():
    items = list(range(20))
    a = ds.deterministic_shuffle(items, seed=42)
    b = ds.deterministic_shuffle(items, seed=42)
    assert a == b


def test_deterministic_shuffle_different_seed_different_order():
    items = list(range(20))
    a = ds.deterministic_shuffle(items, seed=42)
    b = ds.deterministic_shuffle(items, seed=43)
    assert a != b


def test_deterministic_shuffle_is_a_permutation_not_a_mutation_or_loss():
    items = list(range(20))
    shuffled = ds.deterministic_shuffle(items, seed=1)
    assert sorted(shuffled) == items
    assert items == list(range(20))  # original list untouched


def test_deterministic_k_order_same_seed_same_order():
    k_values = [5, 8, 10, 15, 20]
    a = ds.deterministic_k_order(k_values, seed=42)
    b = ds.deterministic_k_order(k_values, seed=42)
    assert a == b
    assert sorted(a) == k_values


def test_deterministic_k_order_differs_from_dev_row_shuffle_stream():
    """K-order uses seed+1, not seed, specifically so it doesn't move in
    lockstep with the dev-row shuffle for the same --seed value."""
    k_values = [5, 8, 10, 15, 20, 25, 30, 35]
    k_order = ds.deterministic_k_order(k_values, seed=42)
    row_shuffle_of_same_values = ds.deterministic_shuffle(k_values, seed=42)
    assert k_order != row_shuffle_of_same_values


# ---------------------------------------------------------------------------
# Provenance: git commit resolution and per-K config hash
# ---------------------------------------------------------------------------

def test_resolve_git_commit_full_returns_a_real_hash_in_this_repo():
    commit = ds.resolve_git_commit_full()
    assert isinstance(commit, str)
    assert len(commit) == 40  # full git rev-parse HEAD, not --short
    assert all(c in "0123456789abcdef" for c in commit)


def test_compute_k_config_hash_differs_across_k():
    h5 = ds.compute_k_config_hash(5, "ollama/llama3.2:latest", False, 3, "leaf_vote", "label")
    h10 = ds.compute_k_config_hash(10, "ollama/llama3.2:latest", False, 3, "leaf_vote", "label")
    assert h5 != h10
    assert h5 and h10  # both non-empty


def test_compute_k_config_hash_deterministic_for_same_inputs():
    a = ds.compute_k_config_hash(5, "ollama/llama3.2:latest", False, 3, "leaf_vote", "label")
    b = ds.compute_k_config_hash(5, "ollama/llama3.2:latest", False, 3, "leaf_vote", "label")
    assert a == b


def test_compute_k_config_hash_matches_run_eval_config_hash_mechanism():
    """Sanity check that this isn't a reimplementation that could drift from
    run_eval._config_hash() -- it must BE that function, called with a
    branch_collapse=False, hierarchical-system stand-in args object.

    sre/use_llm_reranker="on": these two run_eval.py CLI flags (Conference I
    Reviewer #2 Section E ablation support) postdate this test and
    compute_k_config_hash() -- both now included, matching B2's true
    always-on-default K-sweep behaviour (see compute_k_config_hash()'s own
    docstring in eval/dev_sweep.py)."""
    from types import SimpleNamespace as SNS
    fake_args = SNS(system="hierarchical", beam=3, stage1_mode="leaf_vote",
                     reranker_candidates=5, branch_collapse=False, config="label",
                     sre="on", use_llm_reranker="on")
    expected = ds.run_eval._config_hash(fake_args, "ollama/llama3.2:latest", False)
    actual = ds.compute_k_config_hash(5, "ollama/llama3.2:latest", False, 3, "leaf_vote", "label")
    assert actual == expected


# ---------------------------------------------------------------------------
# Provenance (task D): sha256_of_file, git-tree-clean, environment snapshot
# ---------------------------------------------------------------------------

def test_sha256_of_file_matches_hashlib_directly(tmp_path):
    p = tmp_path / "x.txt"
    p.write_bytes(b"hello world")
    import hashlib as _hashlib
    assert ds.sha256_of_file(p) == _hashlib.sha256(b"hello world").hexdigest()


def test_sha256_of_file_returns_none_for_missing_file(tmp_path):
    assert ds.sha256_of_file(tmp_path / "does_not_exist.txt") is None


def test_check_git_tree_clean_returns_bool_and_string():
    clean, output = ds.check_git_tree_clean()
    assert isinstance(clean, bool)
    assert isinstance(output, str)
    assert clean == (output == "")


def test_check_git_tree_clean_detects_dirty_when_git_status_has_output(monkeypatch):
    class _FakeResult:
        returncode = 0
        stdout = " M eval/dev_sweep.py\n"
        stderr = ""
    monkeypatch.setattr(ds.subprocess, "run", lambda *a, **k: _FakeResult())
    clean, output = ds.check_git_tree_clean()
    assert clean is False
    assert "eval/dev_sweep.py" in output


def test_check_git_tree_clean_fails_closed_when_git_unavailable(monkeypatch):
    def _raise(*a, **k):
        raise FileNotFoundError("git not found")
    monkeypatch.setattr(ds.subprocess, "run", _raise)
    clean, output = ds.check_git_tree_clean()
    assert clean is False  # unknown state -> NOT treated as clean
    assert "FileNotFoundError" in output


def test_resolve_environment_provenance_returns_all_documented_keys():
    env = ds.resolve_environment_provenance()
    for key in (
        "current_b2_git_commit", "git_tree_clean", "git_tree_dirty_files",
        "python_version", "requirements_path", "requirements_sha256",
        "ollama_version", "qdrant_version", "os", "cpu", "ram_gb", "gpu",
    ):
        assert key in env


def test_resolve_environment_provenance_survives_ollama_and_qdrant_being_unreachable(monkeypatch):
    """Neither Ollama nor Qdrant being reachable must not crash provenance
    resolution -- ollama_version/qdrant_version just come back None."""
    def _raise(*a, **k):
        raise OSError("connection refused")
    monkeypatch.setattr(ds.urllib.request, "urlopen", _raise)
    env = ds.resolve_environment_provenance()  # must not raise
    assert env["ollama_version"] is None
    assert env["qdrant_version"] is None


def test_resolve_qdrant_version_returns_none_on_connection_error(monkeypatch):
    def _raise(*a, **k):
        raise OSError("connection refused")
    monkeypatch.setattr(ds.urllib.request, "urlopen", _raise)
    assert ds.resolve_qdrant_version() is None


def test_resolve_ollama_version_returns_none_on_connection_error(monkeypatch):
    def _raise(*a, **k):
        raise OSError("connection refused")
    monkeypatch.setattr(ds.urllib.request, "urlopen", _raise)
    assert ds.resolve_ollama_version() is None


def test_resolve_ollama_model_identity_returns_unavailable_on_connection_error(monkeypatch):
    def _raise(*a, **k):
        raise OSError("connection refused")
    monkeypatch.setattr(ds.urllib.request, "urlopen", _raise)
    result = ds.resolve_ollama_model_identity("llama3.2:latest")
    assert result["status"] == "unavailable"


def test_resolve_ollama_model_identity_returns_unavailable_when_tag_not_found(monkeypatch):
    import io as _io

    class _FakeResp:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self):
            return b'{"models": [{"name": "other-model:latest", "digest": "abc"}]}'
    monkeypatch.setattr(ds.urllib.request, "urlopen", lambda *a, **k: _FakeResp())
    result = ds.resolve_ollama_model_identity("llama3.2:latest")
    assert result["status"] == "unavailable"


# ---------------------------------------------------------------------------
# _load_dev_rows_as_test_set -- the canonical dev_set_v1.csv schema
# translation layer (case_id/language/respondent_text/gold_isco_code ->
# run_eval's case_id/input_language/input_text/gold_isco_4digit). See
# eval/dev_set_schema.md for the canonical schema this depends on.
# ---------------------------------------------------------------------------

def test_load_dev_rows_as_test_set_maps_canonical_columns(tmp_path):
    p = tmp_path / "dev.csv"
    p.write_text(
        "case_id,language,respondent_text,gold_isco_code,gold_label_source,"
        "annotator_or_adjudication_reference,dataset_split\n"
        "dev001,en,baker,7512,human_coder_single,AB,dev_v1\n",
        encoding="utf-8",
    )
    rows = ds._load_dev_rows_as_test_set(p)
    assert rows == [{
        "case_id": "dev001", "input_text": "baker", "input_language": "en",
        "gold_isco_4digit": "7512",
    }]


def test_load_dev_rows_as_test_set_ignores_extra_provenance_columns(tmp_path):
    """gold_label_source/annotator_or_adjudication_reference/dataset_split
    are read from the file (DictReader sees them) but must never leak into
    the translated row dev_sweep.py actually runs -- they're provenance
    metadata for humans/validate_dev_set.py, not classifier input."""
    p = tmp_path / "dev.csv"
    p.write_text(
        "case_id,language,respondent_text,gold_isco_code,gold_label_source,"
        "annotator_or_adjudication_reference,dataset_split\n"
        "dev001,en,baker,7512,human_coder_single,AB,dev_v1\n",
        encoding="utf-8",
    )
    rows = ds._load_dev_rows_as_test_set(p)
    assert set(rows[0].keys()) == {"case_id", "input_text", "input_language", "gold_isco_4digit"}


def test_load_dev_rows_as_test_set_multiple_rows_preserve_order(tmp_path):
    p = tmp_path / "dev.csv"
    p.write_text(
        "case_id,language,respondent_text,gold_isco_code,gold_label_source,"
        "annotator_or_adjudication_reference,dataset_split\n"
        "dev001,en,baker,7512,human_coder_single,AB,dev_v1\n"
        "dev002,ar,طباخ,7512,human_coder_single,CD,dev_v1\n",
        encoding="utf-8",
    )
    rows = ds._load_dev_rows_as_test_set(p)
    assert [r["case_id"] for r in rows] == ["dev001", "dev002"]
    assert rows[1]["input_text"] == "طباخ"
