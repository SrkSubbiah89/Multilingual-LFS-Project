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
