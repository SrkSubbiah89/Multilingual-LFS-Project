"""
eval/dev_sweep.py

B2 dev-set K-sweep: runs eval/dev_set_v1.csv through the SAME B1 pooling
pipeline (backend.agents.isco_classifier.ISCOClassifier, branch_collapse
hardcoded False -- see module docstring below) once per candidate value of
K (--reranker-candidates), applies the pre-specified B2 OPERATIONAL
ELIGIBILITY RULE (not a statistical stability test -- a deterministic,
temperature=0 pipeline has no run-to-run variance for a stability test to
measure), and writes a Markdown report recording every eligibility check
and the resulting decision.

This script NEVER touches eval/test_set_full130.csv -- only --dev-set. It
exists specifically so K can be chosen without looking at the held-out
confirmation set. See eval/dev_set_schema.md for why eval/
test_set_smoke20.csv is also excluded (reused for earlier experiments).

Experiment-separation constraints (do not relax without re-reading the B2
spec this script was built against):
  - branch_collapse is hardcoded False. B2 varies ONLY K; it does not
    combine K with the pre-fix branch-collapse behaviour (that's B0, a
    separate, already-frozen baseline).
  - The reranker model, prompt, parser, and fallback-on-failure behaviour
    are exactly ISCOClassifier's existing _llm_select_from_candidates() /
    _parse_llm_response() -- the same ones B0/B1 use. This script does NOT
    wire in backend.agents.isco_reranker_strict.StrictReranker (that is
    B3-Reliability, a separate future ablation) and does NOT change sort
    policy (that is B3-Sort, also separate and not activated here).
  - capture_pool_metadata is always True for sweep runs (this script's only
    reason to exist is to look at the resulting metadata) but that flag is
    observational-only and does not change what gets classified or how --
    see hierarchical_store.py's capture_pool_metadata docstring.

B2 operational eligibility rule
---------------------------------------------------------------------------
Reference configuration: K=5 (--reference-k) is the B1-compatible reference
run on the same dev set, hardware, model, prompt, timeout, input-ordering
policy, and logging settings as every candidate K -- all of that is already
guaranteed here because every K in the sweep runs in the same process,
against the same dev_rows, through the same build_system()/run_one_case()
call path, differing only in --reranker-candidates.

A candidate K is operationally eligible only if ALL of the following hold
(see check_eligibility()):

1. Hard failures -- no OOM, process crash, corrupted output file, or
   incomplete run. Any one of these rejects K outright, unconditionally.
   Operationalised as: KSweepResult.crashed (an exception escaped the
   sweep loop for this K -- see run_sweep_for_k()'s try/except), .
   incomplete_run (fewer cases completed than were in the dev set),
   .corrupted_output (the raw per-K CSV failed a read-back row-count
   check), or n_hard_failures > 0 (a per-case exception with no
   prediction produced, e.g. a Qdrant/network error unrelated to the
   reranker).

2. Timeout and invalid-output events -- raw counts AND rates are recorded
   (KSweepResult.n_timeouts / n_invalid_outputs / timeout_rate /
   invalid_output_rate). K is rejected if
   (n_timeouts + n_invalid_outputs) exceeds the reference K's
   (n_timeouts + n_invalid_outputs) by more than
   --max-additional-timeout-or-invalid (default 1). A timed-out or
   invalid-output case ALWAYS counts as incorrect in
   top1_accuracy_all_case (denominator = every case in the dev set), even
   though isco_classifier.py's existing fallback-on-failure behaviour
   still produces *some* prediction for that case (which could coincidentally
   equal gold) -- crediting a fallback guess as correct would misstate
   reliability. top1_accuracy_conditional is reported separately, computed
   only over cases that neither timed out nor produced invalid output
   (denominator = n_successful_cases), so "how good is the reranker when it
   actually runs" and "how reliable is the whole K configuration" don't get
   conflated into one number.

3. Latency -- median, P95, and max end_to_end_latency_ms are recorded per
   K, with the first --warmup-cases dev-set calls (default 2) run and
   discarded before the measured pass, per case, so cold-start latency
   (model/connection warm-up) doesn't distort the percentiles. K is
   rejected if its P95 latency exceeds --p95-ceiling-factor (default 1.5)
   times the reference K's P95 latency; the actual factor is always
   recorded in the report regardless of pass/fail.

4. Memory -- peak_memory_mb is sampled (via psutil, if installed; None
   with a logged reason if not) as the maximum RSS observed while that K's
   cases ran. K is rejected if peak_memory_mb exceeds --memory-budget-mb
   (no default -- if omitted, this specific check is skipped and the
   report says so explicitly rather than silently passing). Swap/thrashing
   is not independently detectable from this process without OS-level
   tooling this harness doesn't have -- a MemoryError raised during the
   run is caught and treated as `crashed` (see #1); anything short of an
   actual MemoryError is not claimed to be detected here, and the report
   says so.

5. Selection after eligibility -- among OPERATIONALLY ELIGIBLE K values
   only: maximise Candidate Recall@K; if recall is equivalent or nearly
   equivalent (within --recall-close-threshold, default 0.02), choose the
   best top1_accuracy_all_case (not conditional -- see #2 for why); if
   still tied, select the smaller K. eval/test_set_full130.csv is never
   read by this script, so it cannot factor into this decision even by
   accident.

Usage
-----
    python eval/validate_dev_set.py --dev-set eval/dev_set_v1.csv   # first!
    python eval/dev_sweep.py --dev-set eval/dev_set_v1.csv \
        --reranker-model ollama/llama3.2:1b --k-values 5,8,10,15,20
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_eval  # noqa: E402

try:
    import psutil
    _HAVE_PSUTIL = True
except ImportError:
    _HAVE_PSUTIL = False

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "dev_sweep"

DEFAULT_K_VALUES = [5, 8, 10, 15, 20]
DEFAULT_REFERENCE_K = 5
DEFAULT_RECALL_CLOSE_THRESHOLD = 0.02
DEFAULT_P95_CEILING_FACTOR = 1.5
DEFAULT_MAX_ADDITIONAL_TIMEOUT_OR_INVALID = 1
DEFAULT_WARMUP_CASES = 2


@dataclass
class KSweepResult:
    k: int
    n_expected_cases: int
    n_cases: int                        # cases actually completed (post-warmup)
    n_hard_failures: int                # per-case exception, no prediction produced
    incomplete_run: bool                # n_cases < n_expected_cases
    crashed: bool                       # process-level exception escaped the sweep loop
    crash_message: str
    corrupted_output: bool              # raw per-K CSV failed a read-back sanity check
    n_timeouts: int
    n_invalid_outputs: int
    timeout_rate: float
    invalid_output_rate: float
    candidate_recall_at_k: float
    n_successful_cases: int             # n_cases - n_timeouts - n_invalid_outputs
    top1_accuracy_all_case: float       # denom n_cases; timeout/invalid always count as incorrect
    top1_accuracy_conditional: float    # denom n_successful_cases; timeout/invalid cases excluded
    median_latency_ms: float
    p95_latency_ms: float
    max_latency_ms: float
    mean_latency_ms: float
    peak_memory_mb: Optional[float]


@dataclass
class SelectionDecision:
    chosen_k: Optional[int]
    reference_k: int
    rejected: dict = field(default_factory=dict)     # k -> "; "-joined reason string
    rationale: list = field(default_factory=list)     # ordered human-readable steps


# ---------------------------------------------------------------------------
# Pure metric computation -- operates on plain objects/dicts exposing the
# same attributes as run_eval.CaseResult (gold_isco_4digit, pred_isco_4digit,
# gold_rank_in_pool, end_to_end_latency_ms, invalid_output_flag,
# timed_out_flag, error). Unit-testable without any live LLM/Qdrant call.
# ---------------------------------------------------------------------------

def _percentile(values: list, pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(pct / 100 * (len(s) - 1))))
    return s[idx]


def compute_metrics(
    case_results: list,
    k: int,
    n_expected_cases: Optional[int] = None,
    crashed: bool = False,
    crash_message: str = "",
    corrupted_output: bool = False,
    peak_memory_mb: Optional[float] = None,
) -> KSweepResult:
    n_cases = len(case_results)
    n_expected_cases = n_expected_cases if n_expected_cases is not None else n_cases

    if n_cases == 0:
        return KSweepResult(
            k=k, n_expected_cases=n_expected_cases, n_cases=0, n_hard_failures=0,
            incomplete_run=n_expected_cases > 0, crashed=crashed, crash_message=crash_message,
            corrupted_output=corrupted_output, n_timeouts=0, n_invalid_outputs=0,
            timeout_rate=0.0, invalid_output_rate=0.0, candidate_recall_at_k=0.0,
            n_successful_cases=0, top1_accuracy_all_case=0.0, top1_accuracy_conditional=0.0,
            median_latency_ms=0.0, p95_latency_ms=0.0, max_latency_ms=0.0, mean_latency_ms=0.0,
            peak_memory_mb=peak_memory_mb,
        )

    n_hard_failures = sum(
        1 for r in case_results if getattr(r, "error", "") and not getattr(r, "pred_isco_4digit", "")
    )
    n_recall_hits = sum(
        1 for r in case_results
        if getattr(r, "gold_rank_in_pool", None) is not None and r.gold_rank_in_pool <= k
    )
    n_timeouts = sum(1 for r in case_results if getattr(r, "timed_out_flag", False))
    # invalid_output_flag and timed_out_flag are mutually exclusive by construction in
    # run_eval.run_one_case(): timed_out_flag only arises from a reranker_error (kickoff()
    # itself raised); invalid_output_flag only arises from a successful kickoff() whose
    # response could not be parsed. A case cannot be both.
    n_invalid = sum(1 for r in case_results if getattr(r, "invalid_output_flag", False))
    n_unrecovered = n_timeouts + n_invalid
    n_successful = n_cases - n_unrecovered

    n_top1_hits_all_case = sum(
        1 for r in case_results
        if not getattr(r, "timed_out_flag", False) and not getattr(r, "invalid_output_flag", False)
        and getattr(r, "pred_isco_4digit", "") and r.pred_isco_4digit == getattr(r, "gold_isco_4digit", "")
    )
    latencies = [
        r.end_to_end_latency_ms for r in case_results
        if getattr(r, "end_to_end_latency_ms", None) is not None
    ]

    return KSweepResult(
        k=k,
        n_expected_cases=n_expected_cases,
        n_cases=n_cases,
        n_hard_failures=n_hard_failures,
        incomplete_run=n_cases < n_expected_cases,
        crashed=crashed,
        crash_message=crash_message,
        corrupted_output=corrupted_output,
        n_timeouts=n_timeouts,
        n_invalid_outputs=n_invalid,
        timeout_rate=n_timeouts / n_cases,
        invalid_output_rate=n_invalid / n_cases,
        candidate_recall_at_k=n_recall_hits / n_cases,
        n_successful_cases=n_successful,
        top1_accuracy_all_case=n_top1_hits_all_case / n_cases,
        top1_accuracy_conditional=(n_top1_hits_all_case / n_successful) if n_successful else 0.0,
        median_latency_ms=_percentile(latencies, 50),
        p95_latency_ms=_percentile(latencies, 95),
        max_latency_ms=max(latencies) if latencies else 0.0,
        mean_latency_ms=statistics.fmean(latencies) if latencies else 0.0,
        peak_memory_mb=peak_memory_mb,
    )


def check_eligibility(
    candidate: KSweepResult,
    reference: KSweepResult,
    p95_ceiling_factor: float = DEFAULT_P95_CEILING_FACTOR,
    max_additional_timeout_or_invalid: int = DEFAULT_MAX_ADDITIONAL_TIMEOUT_OR_INVALID,
    memory_budget_mb: Optional[float] = None,
) -> tuple:
    """Returns (eligible: bool, reasons: list[str], latency_factor: Optional[float]).
    reasons is empty iff eligible. latency_factor (candidate P95 / reference P95) is
    always returned for reporting, even when eligible, since the report must show the
    actual factor regardless of pass/fail (see rule #3)."""
    reasons = []

    # 1. Hard failures -- unconditional, no threshold.
    if candidate.crashed:
        reasons.append(f"process crash during sweep: {candidate.crash_message or '(no message captured)'}")
    if candidate.incomplete_run:
        reasons.append(
            f"incomplete run: completed {candidate.n_cases}/{candidate.n_expected_cases} case(s)"
        )
    if candidate.corrupted_output:
        reasons.append("raw output CSV failed its read-back sanity check")
    if candidate.n_hard_failures > 0:
        reasons.append(
            f"{candidate.n_hard_failures} hard case failure(s) (exception during classify(), "
            f"no prediction produced)"
        )

    # 2. Timeout + invalid-output delta vs. reference.
    candidate_unrecovered = candidate.n_timeouts + candidate.n_invalid_outputs
    reference_unrecovered = reference.n_timeouts + reference.n_invalid_outputs
    delta = candidate_unrecovered - reference_unrecovered
    if delta > max_additional_timeout_or_invalid:
        reasons.append(
            f"timeout+invalid_output count {candidate_unrecovered} (timeouts={candidate.n_timeouts}, "
            f"invalid={candidate.n_invalid_outputs}) exceeds reference K={reference.k}'s count "
            f"{reference_unrecovered} by {delta}, more than the allowed "
            f"{max_additional_timeout_or_invalid}"
        )

    # 3. P95 latency vs. reference.
    latency_factor = (
        candidate.p95_latency_ms / reference.p95_latency_ms if reference.p95_latency_ms > 0 else None
    )
    if latency_factor is not None and latency_factor > p95_ceiling_factor:
        reasons.append(
            f"P95 latency {candidate.p95_latency_ms:.1f}ms is {latency_factor:.2f}x reference "
            f"K={reference.k}'s P95 ({reference.p95_latency_ms:.1f}ms), exceeding the ceiling "
            f"{p95_ceiling_factor}x"
        )

    # 4. Memory budget (only checked if a budget was actually provided).
    if memory_budget_mb is not None:
        if candidate.peak_memory_mb is not None and candidate.peak_memory_mb > memory_budget_mb:
            reasons.append(
                f"peak_memory_mb={candidate.peak_memory_mb:.1f} exceeds "
                f"memory_budget_mb={memory_budget_mb}"
            )

    return (len(reasons) == 0, reasons, latency_factor)


def select_k(
    results: list,
    reference_k: int = DEFAULT_REFERENCE_K,
    recall_close_threshold: float = DEFAULT_RECALL_CLOSE_THRESHOLD,
    p95_ceiling_factor: float = DEFAULT_P95_CEILING_FACTOR,
    max_additional_timeout_or_invalid: int = DEFAULT_MAX_ADDITIONAL_TIMEOUT_OR_INVALID,
    memory_budget_mb: Optional[float] = None,
) -> SelectionDecision:
    """Pure selection logic over a list of KSweepResult, implementing the B2
    operational eligibility rule (module docstring). Deterministic given the
    same inputs and thresholds -- never looks at anything beyond the
    KSweepResult list passed in (in particular, never at
    eval/test_set_full130.csv)."""
    decision = SelectionDecision(chosen_k=None, reference_k=reference_k)
    if not results:
        decision.rationale.append("No K results supplied -- nothing to select.")
        return decision

    by_k = {r.k: r for r in results}
    if reference_k not in by_k:
        decision.rationale.append(
            f"Reference K={reference_k} is not among the supplied results "
            f"({sorted(by_k.keys())}) -- cannot apply the eligibility rule "
            f"without a reference. Include --reference-k in --k-values."
        )
        return decision
    reference = by_k[reference_k]

    if memory_budget_mb is None:
        decision.rationale.append(
            "No --memory-budget-mb was set -- the memory-budget eligibility check "
            "was NOT applied to any K (peak_memory_mb is still recorded for every K)."
        )
    if not _HAVE_PSUTIL:
        decision.rationale.append(
            "psutil is not installed -- peak_memory_mb could not be measured for any "
            "K (recorded as None); the memory-budget check could not be evaluated "
            "even if --memory-budget-mb was set."
        )

    eligible = []
    for r in results:
        ok, reasons, latency_factor = check_eligibility(
            r, reference, p95_ceiling_factor, max_additional_timeout_or_invalid, memory_budget_mb,
        )
        factor_note = f" (P95 latency factor vs. reference: {latency_factor:.2f}x)" if latency_factor is not None else ""
        if ok:
            eligible.append(r)
            decision.rationale.append(f"K={r.k}: eligible.{factor_note}")
        else:
            decision.rejected[r.k] = "; ".join(reasons)
            decision.rationale.append(f"K={r.k}: REJECTED -- {'; '.join(reasons)}{factor_note}")

    if not eligible:
        decision.rationale.append(
            "All K values were rejected by the operational eligibility rule -- no K "
            "can be selected from this sweep. Investigate the rejection reasons above "
            "(and re-run the reference K itself if it also failed) rather than relaxing "
            "thresholds silently."
        )
        return decision

    max_recall = max(r.candidate_recall_at_k for r in eligible)
    decision.rationale.append(
        f"Step 5a: max Candidate Recall@K among eligible K values = {max_recall:.4f}."
    )

    close = [r for r in eligible if (max_recall - r.candidate_recall_at_k) <= recall_close_threshold]
    decision.rationale.append(
        f"Step 5b: K values within recall_close_threshold={recall_close_threshold} of the max: "
        f"{sorted(r.k for r in close)}."
    )

    max_top1 = max(r.top1_accuracy_all_case for r in close)
    top1_tied = [r for r in close if abs(r.top1_accuracy_all_case - max_top1) < 1e-9]
    decision.rationale.append(
        f"Step 5c: among those, max all-case Top-1 accuracy = {max_top1:.4f}, achieved by K="
        f"{sorted(r.k for r in top1_tied)}."
    )

    chosen = min(top1_tied, key=lambda r: r.k)
    decision.rationale.append(f"Step 5d: smallest-K tie-break -> chosen K={chosen.k}.")
    decision.chosen_k = chosen.k
    return decision


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def render_markdown_report(
    decision: SelectionDecision,
    results: list,
    dev_set_path: str,
    reranker_model: str,
    n_dev_cases: int,
    generated_at: str,
    p95_ceiling_factor: float = DEFAULT_P95_CEILING_FACTOR,
    max_additional_timeout_or_invalid: int = DEFAULT_MAX_ADDITIONAL_TIMEOUT_OR_INVALID,
    memory_budget_mb: Optional[float] = None,
) -> str:
    lines = [
        "# B2 dev-set K-sweep report -- operational eligibility rule",
        "",
        f"- Dev set: `{dev_set_path}` ({n_dev_cases} cases)",
        f"- Reranker model: `{reranker_model}`",
        f"- Pooling: B1 (branch_collapse=False, hardcoded -- see module docstring)",
        f"- Reference K: {decision.reference_k}",
        f"- P95 latency ceiling: {p95_ceiling_factor}x reference",
        f"- Max additional timeout+invalid_output events vs. reference: {max_additional_timeout_or_invalid}",
        f"- Memory budget: {memory_budget_mb if memory_budget_mb is not None else '(not set -- check skipped)'} MB",
        f"- Generated: {generated_at}",
        "",
        "## Per-K metrics",
        "",
        "| K | n | hard_fail | incomplete | crashed | corrupted | timeouts | invalid | "
        "Recall@K | Top-1 (all-case) | Top-1 (conditional) | median (ms) | P95 (ms) | max (ms) | peak mem (MB) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda r: r.k):
        mem = f"{r.peak_memory_mb:.1f}" if r.peak_memory_mb is not None else "n/a"
        lines.append(
            f"| {r.k} | {r.n_cases}/{r.n_expected_cases} | {r.n_hard_failures} | "
            f"{'yes' if r.incomplete_run else 'no'} | {'yes' if r.crashed else 'no'} | "
            f"{'yes' if r.corrupted_output else 'no'} | {r.n_timeouts} ({r.timeout_rate:.3f}) | "
            f"{r.n_invalid_outputs} ({r.invalid_output_rate:.3f}) | {r.candidate_recall_at_k:.4f} | "
            f"{r.top1_accuracy_all_case:.4f} | {r.top1_accuracy_conditional:.4f} | "
            f"{r.median_latency_ms:.1f} | {r.p95_latency_ms:.1f} | {r.max_latency_ms:.1f} | {mem} |"
        )

    lines += ["", "## Eligibility and selection rationale", ""]
    for step in decision.rationale:
        lines.append(f"- {step}")

    lines += ["", "## Decision", ""]
    if decision.chosen_k is not None:
        lines.append(
            f"**Chosen K = {decision.chosen_k}**, frozen for the single B2 "
            f"confirmation run on `eval/test_set_full130.csv`. Per the final-test "
            f"rule, K was selected without looking at that set, and that set will "
            f"be run exactly once with this K."
        )
    else:
        lines.append(
            "**No K selected.** See the rejection reasons above -- every candidate "
            "K failed the operational eligibility rule, the reference K was missing "
            "from the sweep, or no results were supplied."
        )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI: actually runs the sweep (NOT executed by this session -- see the B2
# spec's explicit "do not run the K-sweep... yet" instruction; infra only).
# ---------------------------------------------------------------------------

def _load_dev_rows_as_test_set(dev_set_path: Path) -> list[dict]:
    """Translate dev_set_v1.csv's schema (eval/dev_set_schema.md) into the
    column names run_eval.run_one_case() expects (input_text,
    input_language, gold_isco_4digit) -- kept as an explicit, visible
    mapping rather than silently aliasing column names, since the two
    schemas exist for different audiences (human annotators vs. the
    harness) and should stay allowed to diverge further later."""
    with open(dev_set_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        {
            "case_id": row["case_id"],
            "input_text": row["respondent_text"],
            "input_language": row["language"],
            "gold_isco_4digit": row["gold_isco_code"],
        }
        for row in rows
    ]


def _peak_rss_mb() -> Optional[float]:
    if not _HAVE_PSUTIL:
        return None
    try:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:  # noqa: BLE001 - best-effort sampling, never fatal to the sweep
        return None


def run_sweep_for_k(dev_rows: list[dict], k: int, reranker_model: str, beam: int,
                     stage1_mode: str, disable_keyword_map: bool,
                     warmup_cases: int = DEFAULT_WARMUP_CASES) -> tuple:
    """Runs the dev set once through B1's pooling pipeline with
    --reranker-candidates=k. branch_collapse is hardcoded False -- see
    module docstring's experiment-separation constraints.

    Returns (case_results, crashed, crash_message, peak_memory_mb). Catches
    exceptions at the sweep level (not just per-case, which run_one_case()
    already handles) so a crash on one K doesn't take down the whole sweep
    -- the crash is recorded and that K is rejected by check_eligibility()
    rather than losing every other K's results too."""
    crashed = False
    crash_message = ""
    peak_mb = _peak_rss_mb()
    case_results = []

    try:
        clf = run_eval.build_system(
            "hierarchical", reranker_model=reranker_model,
            disable_keyword_map=disable_keyword_map, beam=beam, stage1_mode=stage1_mode,
            reranker_candidates=k, branch_collapse=False, capture_pool_metadata=True,
        )
        isic_clf = run_eval.ISICClassifier()
        isced_clf = run_eval.ISCEDClassifier()
        sre = run_eval.SemanticRelationEngine(use_llm=False)
        cfg_hash = f"devsweep_k{k}"

        # Warm-up phase: discarded entirely, never enters case_results, so it
        # cannot affect recall/top1/latency stats -- see rule #3.
        for i in range(min(warmup_cases, len(dev_rows))):
            row = dev_rows[i]
            run_eval.run_one_case(
                clf, sre, isic_clf, isced_clf, row_index=i,
                case_id=f"warmup_{row['case_id']}", input_text=row["input_text"],
                input_language=row["input_language"], gold_isco_4digit=row["gold_isco_4digit"],
                gold_isic="", gold_isced="", config_hash=cfg_hash, system="hierarchical",
                keyword_map_enabled=not disable_keyword_map, branch_collapse_enabled=False,
                capture_pool_metadata_enabled=True, run_id=f"devsweep_k{k}_warmup", seed=run_eval.RANDOM_SEED,
            )
            sample = _peak_rss_mb()
            if sample is not None:
                peak_mb = sample if peak_mb is None else max(peak_mb, sample)

        for i, row in enumerate(dev_rows):
            r = run_eval.run_one_case(
                clf, sre, isic_clf, isced_clf, row_index=i,
                case_id=row["case_id"], input_text=row["input_text"],
                input_language=row["input_language"], gold_isco_4digit=row["gold_isco_4digit"],
                gold_isic="", gold_isced="", config_hash=cfg_hash, system="hierarchical",
                keyword_map_enabled=not disable_keyword_map, branch_collapse_enabled=False,
                capture_pool_metadata_enabled=True, run_id=f"devsweep_k{k}",
                git_commit="", seed=run_eval.RANDOM_SEED, input_order_position=i,
            )
            case_results.append(r)
            sample = _peak_rss_mb()
            if sample is not None:
                peak_mb = sample if peak_mb is None else max(peak_mb, sample)
    except Exception as exc:  # noqa: BLE001 - see docstring: a crash on one K must not kill the sweep
        crashed = True
        crash_message = f"{type(exc).__name__}: {exc}"

    return case_results, crashed, crash_message, peak_mb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-set", required=True, type=Path)
    parser.add_argument("--reranker-model", required=True, type=str)
    parser.add_argument("--k-values", type=str, default=",".join(str(k) for k in DEFAULT_K_VALUES))
    parser.add_argument("--reference-k", type=int, default=DEFAULT_REFERENCE_K)
    parser.add_argument("--beam", type=int, default=2)
    parser.add_argument("--stage1-mode", choices=["description", "leaf_vote"], default="description")
    parser.add_argument("--disable-keyword-map", action="store_true")
    parser.add_argument("--warmup-cases", type=int, default=DEFAULT_WARMUP_CASES)
    parser.add_argument("--recall-close-threshold", type=float, default=DEFAULT_RECALL_CLOSE_THRESHOLD)
    parser.add_argument("--p95-ceiling-factor", type=float, default=DEFAULT_P95_CEILING_FACTOR)
    parser.add_argument("--max-additional-timeout-or-invalid", type=int,
                         default=DEFAULT_MAX_ADDITIONAL_TIMEOUT_OR_INVALID)
    parser.add_argument("--memory-budget-mb", type=float, default=None,
                         help="If omitted, the memory-budget eligibility check is skipped "
                              "(peak_memory_mb is still recorded for every K).")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    if not args.dev_set.exists():
        parser.error(f"Dev set not found: {args.dev_set}")
    k_values = [int(k.strip()) for k in args.k_values.split(",") if k.strip()]
    if not k_values:
        parser.error("--k-values produced an empty list")
    if args.reference_k not in k_values:
        parser.error(
            f"--reference-k={args.reference_k} must be included in --k-values={k_values} "
            f"-- the eligibility rule compares every K against this reference run."
        )
    if not _HAVE_PSUTIL:
        print("WARNING: psutil is not installed -- peak_memory_mb will be None for every K "
              "and the memory-budget eligibility check cannot be evaluated.", file=sys.stderr)

    dev_rows = _load_dev_rows_as_test_set(args.dev_set)
    print(f"Loaded {len(dev_rows)} dev case(s) from {args.dev_set}")
    print(f"Sweeping K={k_values} (reference K={args.reference_k}) with reranker_model={args.reranker_model}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    all_metrics = []
    for k in k_values:
        print(f"\n=== K={k} ===")
        case_results, crashed, crash_message, peak_mb = run_sweep_for_k(
            dev_rows, k, args.reranker_model, args.beam, args.stage1_mode,
            args.disable_keyword_map, warmup_cases=args.warmup_cases,
        )
        if crashed:
            print(f"K={k}: CRASHED -- {crash_message}")

        raw_csv_path = args.output_dir / f"{timestamp}_devsweep_k{k}_raw.csv"
        corrupted_output = False
        fieldnames = list(run_eval.CaseResult.__dataclass_fields__.keys())
        try:
            with open(raw_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in case_results:
                    writer.writerow(r.__dict__)
            with open(raw_csv_path, newline="", encoding="utf-8") as f:
                written_rows = list(csv.DictReader(f))
            if len(written_rows) != len(case_results):
                corrupted_output = True
        except Exception as exc:  # noqa: BLE001
            corrupted_output = True
            print(f"K={k}: output write/read-back failed: {exc}")

        metrics = compute_metrics(
            case_results, k, n_expected_cases=len(dev_rows), crashed=crashed,
            crash_message=crash_message, corrupted_output=corrupted_output, peak_memory_mb=peak_mb,
        )
        all_metrics.append(metrics)
        print(
            f"K={k}: recall@K={metrics.candidate_recall_at_k:.4f} "
            f"top1_all_case={metrics.top1_accuracy_all_case:.4f} "
            f"top1_conditional={metrics.top1_accuracy_conditional:.4f} "
            f"timeouts={metrics.n_timeouts} invalid={metrics.n_invalid_outputs} "
            f"P95={metrics.p95_latency_ms:.1f}ms peak_mem={metrics.peak_memory_mb}"
        )

    decision = select_k(
        all_metrics,
        reference_k=args.reference_k,
        recall_close_threshold=args.recall_close_threshold,
        p95_ceiling_factor=args.p95_ceiling_factor,
        max_additional_timeout_or_invalid=args.max_additional_timeout_or_invalid,
        memory_budget_mb=args.memory_budget_mb,
    )

    report_md = render_markdown_report(
        decision, all_metrics, str(args.dev_set), args.reranker_model,
        len(dev_rows), datetime.now(timezone.utc).isoformat(),
        p95_ceiling_factor=args.p95_ceiling_factor,
        max_additional_timeout_or_invalid=args.max_additional_timeout_or_invalid,
        memory_budget_mb=args.memory_budget_mb,
    )
    report_path = args.output_dir / f"{timestamp}_devsweep_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\nWrote report to {report_path}")
    print(f"Chosen K: {decision.chosen_k}")


if __name__ == "__main__":
    main()
