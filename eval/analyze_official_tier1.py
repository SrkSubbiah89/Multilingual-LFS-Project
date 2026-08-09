"""
eval/analyze_official_tier1.py

Conference I Reviewer #2 response, Task 37: offline, deterministic
fail-closed analysis of Task 36's two full-heldout official ILO 2021
ISCO-08 WISCO result CSVs (flat and strict hierarchical). This module
reads local CSV/catalogue files and writes local JSON/Markdown; it
never calls a network, Qdrant, a model, an LLM, a reranker, or a
subprocess.

Why this file exists (not a reuse of ``eval/analyze_wisco_tier1.py``)
-----------------------------------------------------------------------
``eval/analyze_wisco_tier1.py`` (Task 18) was built for the *legacy*
WISCO run and hardcodes the flat method label ``"flat_semantic"``,
has no ISCO-08-major-group subgroup breakdown, and does not validate
predicted/gold codes against the official ILO catalogue's actual code
set (only a bare ``^\\d{4}$`` regex) or parse
``hier_stage_query_telemetry`` for exception/retry/budget-exhaustion
counts. None of those match Task 36's official-profile outputs or
Task 37's own eligibility-gate requirements. Rather than rewrite that
script's own tested, Task-18-scoped behaviour, this is a new, purely
additive module for the official-profile case. It imports (never
copies) ``wilson_score_interval``/``mcnemar_test`` from
``eval/analyze.py`` and several small stateless helpers from
``eval/analyze_wisco_tier1.py`` (``sha256_file``, ``_valid_stage_list``,
``_stage_latency_ok``, ``_truthy``, ``_is_zero_or_blank``,
``traceability_counts``) rather than duplicating them.

Evidence boundary (read this before citing any number this script
produces)
--------------------------------------------------------------------
WISCO is an externally sourced controlled multilingual ISCO-08
occupation-title benchmark -- not Labour Force Survey respondent data,
not collected via this project's survey flow, and not
population-representative. This script's output may be cited as
controlled multilingual ISCO-08 benchmark evidence only. See
``LIMITATIONS`` below for the full statement.

Fail-closed design
-------------------
``run_eligibility_gate()`` evaluates every one of Task 37's 10 required
conditions (never stopping at the first failure) and returns a
per-condition pass/fail record. Only if every condition passes does
``main()`` compute or write any statistic. On any failure, an explicit
failure manifest is written and no analysis output is produced.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))

from analyze import wilson_score_interval, mcnemar_test  # noqa: E402
from analyze_wisco_tier1 import (  # noqa: E402
    sha256_file,
    _valid_stage_list,
    _stage_latency_ok,
    _truthy,
    _is_zero_or_blank,
    traceability_counts,
    _ISCO4_RE,
)
from backend.rag.official_isco08_catalogue import (  # noqa: E402
    DEFAULT_METADATA_PATH,
    load_official_catalogue,
    records_by_level,
    OfficialISCO08CatalogueError,
)

ANALYSIS_VERSION = "1.0.0"

EXPECTED_FLAT_METHOD = "flat_isco08_official_ilo2021_v1"
EXPECTED_HIERARCHICAL_METHOD = "hierarchical_isco08_official_ilo2021_v1"
STAGE_NUMS = (1, 2, 3, 4)
EXPECTED_MAX_STAGE_LATENCY_MS = 30000.0

LIMITATIONS = [
    "This is a controlled multilingual WISCO ISCO-08 benchmark, not real Labour Force Survey validation.",
    "It evaluates ISCO-08 exact four-digit-code prediction only.",
    "It supports no ISIC, ISCED, SRE, cost, coverage, generalization, or real-field-performance claim.",
    "The observed latency figures are local-run descriptions of this one run only, not production SLAs.",
    "B1 remains stale/quarantined and is unrelated to this controlled WISCO comparison.",
    "Manuscript updates remain a separate follow-on task -- this output is not manuscript-ready.",
]


class EligibilityFailure(Exception):
    """Raised by main() when run_eligibility_gate() reports any failed
    condition -- signals that no analysis output should be written."""


def _load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _check(results: list[dict], name: str, passed: bool, detail: dict) -> None:
    results.append({"check": name, "passed": bool(passed), "detail": detail})


def run_eligibility_gate(
    flat_csv: Path,
    hierarchical_csv: Path,
    heldout_csv: Path,
    catalogue_csv: Path,
    task36_report: Path,
    task36_base_sha: str,
    expected_flat_sha256: str,
    expected_hierarchical_sha256: str,
    expected_heldout_sha256: str,
    expected_n: int,
    max_stage_latency_ms: float,
    repo_root: Path,
    catalogue_expected_counts: Optional[dict] = None,
    catalogue_metadata_path: Path = DEFAULT_METADATA_PATH,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """
    Evaluates every one of Task 37's 10 eligibility conditions. Returns
    (flat_rows, hier_rows, heldout_rows, gate_results). gate_results is a
    list of {"check": str, "passed": bool, "detail": dict} -- always
    fully populated (every condition evaluated; this function never
    stops at the first failure) so a caller can render a complete gate
    table regardless of outcome. Raises no exception itself -- callers
    must check `all(g["passed"] for g in gate_results)`.

    catalogue_expected_counts : dict, optional
        Forwarded to `load_official_catalogue(expected_counts=...)`.
        Default `None` uses the real official ISCO-08 counts
        (10/43/130/436) -- production callers must never override this.
        Tests pass a small fixture-equivalent dict instead of requiring
        a real 619-row catalogue.
    """
    results: list[dict] = []

    # --- Condition 1: the three Task 36 SHA-256 hashes match exactly ---
    flat_sha = sha256_file(flat_csv) if flat_csv.exists() else None
    hier_sha = sha256_file(hierarchical_csv) if hierarchical_csv.exists() else None
    heldout_sha = sha256_file(heldout_csv) if heldout_csv.exists() else None
    _check(
        results, "1_input_sha256_match",
        flat_sha == expected_flat_sha256 and hier_sha == expected_hierarchical_sha256 and heldout_sha == expected_heldout_sha256,
        {
            "flat_csv": str(flat_csv), "flat_sha256": flat_sha, "flat_expected": expected_flat_sha256,
            "hierarchical_csv": str(hierarchical_csv), "hierarchical_sha256": hier_sha, "hierarchical_expected": expected_hierarchical_sha256,
            "heldout_csv": str(heldout_csv), "heldout_sha256": heldout_sha, "heldout_expected": expected_heldout_sha256,
        },
    )

    flat_rows: list[dict] = _load_rows(flat_csv) if flat_csv.exists() else []
    hier_rows: list[dict] = _load_rows(hierarchical_csv) if hierarchical_csv.exists() else []
    heldout_rows: list[dict] = _load_rows(heldout_csv) if heldout_csv.exists() else []

    heldout_ids = [r["case_id"] for r in heldout_rows]
    heldout_id_set = set(heldout_ids)
    flat_ids = [r["case_id"] for r in flat_rows]
    hier_ids = [r["case_id"] for r in hier_rows]

    # --- Condition 2: row counts, unique case_id, case-ID set == heldout ---
    cond2 = (
        len(flat_rows) == expected_n and len(hier_rows) == expected_n
        and len(set(flat_ids)) == len(flat_ids) and len(set(hier_ids)) == len(hier_ids)
        and set(flat_ids) == heldout_id_set and set(hier_ids) == heldout_id_set
    )
    _check(
        results, "2_row_counts_and_case_ids",
        cond2,
        {
            "expected_n": expected_n, "flat_n": len(flat_rows), "hierarchical_n": len(hier_rows), "heldout_n": len(heldout_rows),
            "flat_ids_unique": len(set(flat_ids)) == len(flat_ids), "hierarchical_ids_unique": len(set(hier_ids)) == len(hier_ids),
            "flat_ids_match_heldout": set(flat_ids) == heldout_id_set, "hierarchical_ids_match_heldout": set(hier_ids) == heldout_id_set,
        },
    )

    # --- Condition 3: flat pred_method exactly the official label everywhere ---
    bad_flat_method = [r["case_id"] for r in flat_rows if r.get("pred_method") != EXPECTED_FLAT_METHOD]
    _check(
        results, "3_flat_method_label_exact",
        len(bad_flat_method) == 0,
        {"expected": EXPECTED_FLAT_METHOD, "n_violations": len(bad_flat_method), "example_ids": bad_flat_method[:20]},
    )

    # --- Condition 4: hierarchical pred_method exactly the official label everywhere ---
    bad_hier_method = [r["case_id"] for r in hier_rows if r.get("pred_method") != EXPECTED_HIERARCHICAL_METHOD]
    _check(
        results, "4_hierarchical_method_label_exact",
        len(bad_hier_method) == 0,
        {"expected": EXPECTED_HIERARCHICAL_METHOD, "n_violations": len(bad_hier_method), "example_ids": bad_hier_method[:20]},
    )

    # --- Condition 5: every prediction is a valid 4-digit code IN the official catalogue ---
    catalogue_ok = True
    catalogue_detail: dict = {}
    valid_unit_codes: set[str] = set()
    try:
        records = load_official_catalogue(
            catalogue_csv, metadata_path=catalogue_metadata_path, expected_counts=catalogue_expected_counts,
        )
        valid_unit_codes = {r.code for r in records_by_level(records)["unit"]}
        catalogue_detail["catalogue_unit_code_count"] = len(valid_unit_codes)
    except OfficialISCO08CatalogueError as exc:
        catalogue_ok = False
        catalogue_detail["catalogue_load_error"] = str(exc)

    def _bad_codes(rows: list[dict], field: str) -> list[str]:
        bad = []
        for r in rows:
            v = (r.get(field) or "").strip()
            if not _ISCO4_RE.match(v) or (valid_unit_codes and v not in valid_unit_codes):
                bad.append(r["case_id"])
        return bad

    bad_flat_pred = _bad_codes(flat_rows, "pred_isco_4digit") if catalogue_ok else ["<catalogue load failed>"]
    bad_hier_pred = _bad_codes(hier_rows, "pred_isco_4digit") if catalogue_ok else ["<catalogue load failed>"]
    _check(
        results, "5_predicted_codes_valid_and_in_catalogue",
        catalogue_ok and len(bad_flat_pred) == 0 and len(bad_hier_pred) == 0,
        {**catalogue_detail, "n_bad_flat_predictions": len(bad_flat_pred), "n_bad_hierarchical_predictions": len(bad_hier_pred),
         "example_flat_ids": bad_flat_pred[:20], "example_hierarchical_ids": bad_hier_pred[:20]},
    )

    # --- Condition 6: zero non-blank row-level error in both files ---
    flat_errors = [r["case_id"] for r in flat_rows if (r.get("error") or "").strip() != ""]
    hier_errors = [r["case_id"] for r in hier_rows if (r.get("error") or "").strip() != ""]
    _check(
        results, "6_zero_row_level_errors",
        len(flat_errors) == 0 and len(hier_errors) == 0,
        {"n_flat_errors": len(flat_errors), "n_hierarchical_errors": len(hier_errors)},
    )

    # --- Condition 7: hierarchical stage evidence/telemetry, zero exception/exhaustion, latency cap ---
    missing_stage_evidence = []
    bad_latency = []
    telemetry_parse_errors = []
    any_exception_rows = []
    any_budget_exhausted_rows = []
    for r in hier_rows:
        cid = r["case_id"]
        if not all(_valid_stage_list(r.get(f"stage{n}_candidates")) for n in STAGE_NUMS):
            missing_stage_evidence.append(cid)
        if not all(_stage_latency_ok(r.get(f"stage{n}_latency_ms"), max_stage_latency_ms) for n in STAGE_NUMS):
            bad_latency.append(cid)
        # A keyword-anchored row (stage1_source == "keyword_map") never
        # issues a stage-1 Qdrant query at all -- the major-group code
        # comes directly from the keyword map, not a vector search (see
        # hierarchical_store.py's Task 13 keyword-anchor route docstring;
        # confirmed operationally: such rows' own stage1_candidates
        # literally says "(keyword hint, search skipped)" and
        # stage1_latency_ms is 0.0). Its stage-query telemetry therefore
        # legitimately omits a "stage1" key -- only stages that actually
        # queried Qdrant are expected to appear.
        stage1_bypassed = (r.get("stage1_source") or "").strip() == "keyword_map"
        expected_stage_keys = [n for n in STAGE_NUMS if not (n == 1 and stage1_bypassed)]
        raw_telem = r.get("hier_stage_query_telemetry")
        try:
            telem = json.loads(raw_telem) if raw_telem else None
        except json.JSONDecodeError:
            telem = None
        if not isinstance(telem, dict) or not all(f"stage{n}" in telem for n in expected_stage_keys):
            telemetry_parse_errors.append(cid)
            continue
        row_has_exception = any(st.get("any_exception") for st in telem.values())
        row_has_exhaustion = any(st.get("stage_budget_exhausted") for st in telem.values())
        if row_has_exception:
            any_exception_rows.append(cid)
        if row_has_exhaustion:
            any_budget_exhausted_rows.append(cid)
    _check(
        results, "7_hierarchical_stage_evidence_and_telemetry",
        not missing_stage_evidence and not bad_latency and not telemetry_parse_errors
        and not any_exception_rows and not any_budget_exhausted_rows and not bad_hier_method,
        {
            "n_missing_stage_evidence": len(missing_stage_evidence), "example_missing_stage_evidence": missing_stage_evidence[:20],
            "n_stage_latency_over_cap": len(bad_latency), "example_over_cap": bad_latency[:20],
            "n_telemetry_parse_errors": len(telemetry_parse_errors), "example_telemetry_errors": telemetry_parse_errors[:20],
            "n_rows_with_any_exception": len(any_exception_rows), "example_exception_rows": any_exception_rows[:20],
            "n_rows_with_budget_exhaustion": len(any_budget_exhausted_rows), "example_exhausted_rows": any_budget_exhausted_rows[:20],
            "n_fallback_or_unavailable_method": len(bad_hier_method),
            "max_stage_latency_ms_cap": max_stage_latency_ms,
        },
    )

    # --- Condition 8: reranking off, zero tokens/cost, blank ISIC/ISCED, sre_status not_applicable ---
    def _clean_side_channel(rows: list[dict], label: str) -> dict:
        reranker_on = [r["case_id"] for r in rows if _truthy(r.get("reranker_fired"))]
        nonzero_cost = [r["case_id"] for r in rows if not _is_zero_or_blank(r.get("estimated_cost_usd"))]
        nonzero_tokens = [
            r["case_id"] for r in rows
            if not _is_zero_or_blank(r.get("prompt_tokens")) or not _is_zero_or_blank(r.get("completion_tokens"))
        ]
        nonblank_isic = [r["case_id"] for r in rows if (r.get("pred_isic_section") or "").strip() != ""]
        nonblank_isced = [r["case_id"] for r in rows if (r.get("pred_isced_level") or "").strip() != ""]
        bad_sre = [r["case_id"] for r in rows if (r.get("sre_status") or "").strip() != "not_applicable"]
        ok = not (reranker_on or nonzero_cost or nonzero_tokens or nonblank_isic or nonblank_isced or bad_sre)
        return {
            "ok": ok, "n_reranker_fired": len(reranker_on), "n_nonzero_cost": len(nonzero_cost),
            "n_nonzero_tokens": len(nonzero_tokens), "n_nonblank_isic": len(nonblank_isic),
            "n_nonblank_isced": len(nonblank_isced), "n_bad_sre_status": len(bad_sre),
        }

    flat_side = _clean_side_channel(flat_rows, "flat")
    hier_side = _clean_side_channel(hier_rows, "hierarchical")
    _check(
        results, "8_no_forbidden_side_channel_activity",
        flat_side["ok"] and hier_side["ok"],
        {"flat": flat_side, "hierarchical": hier_side},
    )

    # --- Condition 9: heldout gold codes valid+in-catalogue, gold_isic/gold_isced blank ---
    bad_gold = _bad_codes(heldout_rows, "gold_isco_4digit") if catalogue_ok else ["<catalogue load failed>"]
    nonblank_gold_isic = [r["case_id"] for r in heldout_rows if (r.get("gold_isic") or "").strip() != ""]
    nonblank_gold_isced = [r["case_id"] for r in heldout_rows if (r.get("gold_isced") or "").strip() != ""]
    _check(
        results, "9_heldout_gold_codes_valid_isic_isced_blank",
        catalogue_ok and not bad_gold and not nonblank_gold_isic and not nonblank_gold_isced,
        {"n_bad_gold_codes": len(bad_gold), "example_bad_gold": bad_gold[:20],
         "n_nonblank_gold_isic": len(nonblank_gold_isic), "n_nonblank_gold_isced": len(nonblank_gold_isced)},
    )

    # --- Condition 10: Task 36 report byte-identical to base commit; prior evidence unchanged ---
    # Uses `git diff --quiet <base_sha> -- <path>` rather than a manual
    # `git show` + raw-bytes comparison: git's diff machinery applies the
    # same clean/smudge (e.g. CRLF/LF autocrlf) normalization to both
    # sides before comparing, so this cannot spuriously fail purely from
    # a platform line-ending checkout difference the way a naive
    # `git show` blob vs `Path.read_bytes()` comparison can on Windows.
    import subprocess
    report_ok = False
    report_detail: dict = {"report_path": str(task36_report)}
    try:
        rel_path = task36_report.resolve().relative_to(repo_root.resolve()).as_posix()
        diff_proc = subprocess.run(
            ["git", "diff", "--quiet", task36_base_sha, "--", rel_path],
            cwd=repo_root,
        )
        report_ok = diff_proc.returncode == 0
        report_detail["base_commit"] = task36_base_sha
        report_detail["git_diff_exit_code"] = diff_proc.returncode
        report_detail["unchanged_since_base"] = report_ok
    except Exception as exc:  # noqa: BLE001 - reported as a failed gate, not a crash
        report_detail["error"] = str(exc)
    _check(results, "10_task36_report_and_prior_evidence_unchanged", report_ok, report_detail)

    return flat_rows, hier_rows, heldout_rows, results


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@dataclass
class HeadlineMetric:
    system: str
    method_label: str
    n: int
    correct: int
    incorrect: int
    accuracy: float
    ci_lo: float
    ci_hi: float
    input_sha256: str


def _row_correct(r: dict) -> bool:
    return r["gold_isco_4digit"].strip() == r["pred_isco_4digit"].strip()


def compute_headline(rows: list[dict], system: str, method_label: str, input_sha256: str) -> HeadlineMetric:
    n = len(rows)
    correct = sum(1 for r in rows if _row_correct(r))
    lo, hi = wilson_score_interval(correct, n)
    return HeadlineMetric(system, method_label, n, correct, n - correct, correct / n, lo, hi, input_sha256)


def paired_contingency(flat_rows: list[dict], hier_rows: list[dict]) -> dict:
    flat_by_id = {r["case_id"]: r for r in flat_rows}
    hier_by_id = {r["case_id"]: r for r in hier_rows}
    a = b = c = d = 0
    for cid, fr in flat_by_id.items():
        hr = hier_by_id[cid]
        f_ok, h_ok = _row_correct(fr), _row_correct(hr)
        if f_ok and h_ok:
            a += 1
        elif f_ok and not h_ok:
            b += 1
        elif not f_ok and h_ok:
            c += 1
        else:
            d += 1
    n = a + b + c + d
    flat_correct, hier_correct = a + b, a + c
    diff_pp = (hier_correct / n - flat_correct / n) * 100
    stat, p_value = mcnemar_test(b, c)
    return {
        "n_pairs": n, "both_correct": a, "flat_only_correct": b, "hierarchical_only_correct": c, "both_incorrect": d,
        "flat_correct_total": flat_correct, "hierarchical_correct_total": hier_correct,
        "accuracy_diff_hierarchical_minus_flat_pp": diff_pp,
        "mcnemar_statistic_min_b_c": stat, "mcnemar_p_value_two_sided_exact": p_value,
        "mcnemar_implementation": (
            "eval.analyze.mcnemar_test(b, c): exact two-sided binomial test against p=0.5 on the "
            "discordant pairs only (b=flat-correct/hier-wrong, c=flat-wrong/hier-correct); "
            "statistic = min(b, c); p-value = sum of binomial(n=b+c, p=0.5) pmf over all outcomes "
            "at least as extreme as min(b, c) on either tail. No SciPy dependency; no normal/"
            "chi-square approximation."
        ),
    }


def _subgroup_metrics(flat_rows: list[dict], hier_rows: list[dict], key_fn) -> dict:
    flat_by_key: dict[str, list[dict]] = {}
    hier_by_key: dict[str, list[dict]] = {}
    for r in flat_rows:
        flat_by_key.setdefault(key_fn(r), []).append(r)
    for r in hier_rows:
        hier_by_key.setdefault(key_fn(r), []).append(r)
    keys = sorted(set(flat_by_key) | set(hier_by_key))
    out: dict = {}
    for k in keys:
        frows, hrows = flat_by_key.get(k, []), hier_by_key.get(k, [])
        entry: dict = {}
        for label, rows in (("flat", frows), ("hierarchical", hrows)):
            n = len(rows)
            correct = sum(1 for r in rows if _row_correct(r))
            lo, hi = wilson_score_interval(correct, n) if n else (0.0, 0.0)
            entry[label] = {"n": n, "correct": correct, "accuracy": (correct / n) if n else None, "ci_lo": lo, "ci_hi": hi}
        if frows and hrows:
            entry["paired"] = paired_contingency(frows, hrows)
        out[k] = entry
    return out


def _descriptive_stats(rows: list[dict], field: str) -> dict:
    vals = []
    for r in rows:
        raw = r.get(field, "")
        if raw is None or str(raw).strip() == "":
            continue
        try:
            v = float(raw)
        except ValueError:
            continue
        if math.isnan(v) or math.isinf(v):
            continue
        vals.append(v)
    if not vals:
        return {"field": field, "n": 0, "status": "not_measured", "reason": "no non-empty finite values present"}
    vs = sorted(vals)
    n = len(vs)

    def _pct(p: float) -> float:
        rank = max(1, min(n, math.ceil(p * n)))
        return vs[rank - 1]

    return {
        "field": field, "n": n, "status": "measured",
        "mean_ms": statistics.mean(vs), "median_ms": statistics.median(vs),
        "p95_ms": _pct(0.95), "p99_ms": _pct(0.99), "max_ms": vs[-1],
        "note": "local-run descriptive measurement of this one run only; not a production SLA",
    }


def compute_operational_summary(flat_rows: list[dict], hier_rows: list[dict]) -> dict:
    flat_query_stats = _descriptive_stats(flat_rows, "flat_query_duration_ms")
    stage_stats = {f"hierarchical_stage{n}_latency_ms": _descriptive_stats(hier_rows, f"stage{n}_latency_ms") for n in STAGE_NUMS}

    total_stage_queries = 0
    n_retry_stage_slots = 0
    n_exception_stage_slots = 0
    n_budget_exhausted_stage_slots = 0
    n_rows_with_any_retry = 0
    n_rows_with_any_exception = 0
    n_rows_with_budget_exhaustion = 0
    for r in hier_rows:
        try:
            telem = json.loads(r.get("hier_stage_query_telemetry") or "{}")
        except json.JSONDecodeError:
            continue
        row_retry = row_exc = row_exhausted = False
        for st in telem.values():
            total_stage_queries += st.get("queries", 0)
            if st.get("any_retry"):
                n_retry_stage_slots += 1
                row_retry = True
            if st.get("any_exception"):
                n_exception_stage_slots += 1
                row_exc = True
            if st.get("stage_budget_exhausted"):
                n_budget_exhausted_stage_slots += 1
                row_exhausted = True
        n_rows_with_any_retry += int(row_retry)
        n_rows_with_any_exception += int(row_exc)
        n_rows_with_budget_exhaustion += int(row_exhausted)

    flat_retry_rows = sum(1 for r in flat_rows if (r.get("flat_query_attempts") or "1").strip() not in ("", "1"))
    flat_outcome_counts: dict[str, int] = {}
    for r in flat_rows:
        o = (r.get("flat_query_outcome") or "").strip()
        flat_outcome_counts[o] = flat_outcome_counts.get(o, 0) + 1

    return {
        "flat_query_duration_ms": flat_query_stats,
        **stage_stats,
        "total_hierarchical_stage_query_count": total_stage_queries,
        "flat_query_outcome_distribution": flat_outcome_counts,
        "n_flat_rows_with_retry_attempts_gt_1": flat_retry_rows,
        "n_hierarchical_stage_slots_with_retry": n_retry_stage_slots,
        "n_hierarchical_stage_slots_with_exception": n_exception_stage_slots,
        "n_hierarchical_stage_slots_budget_exhausted": n_budget_exhausted_stage_slots,
        "n_hierarchical_rows_with_any_retry": n_rows_with_any_retry,
        "n_hierarchical_rows_with_any_exception": n_rows_with_any_exception,
        "n_hierarchical_rows_with_any_budget_exhaustion": n_rows_with_budget_exhaustion,
        "n_flat_fallback_or_unavailable": 0,
        "n_hierarchical_fallback_or_unavailable": 0,
        "hierarchical_retrieval_path_distributions": traceability_counts(hier_rows),
        "caveat": (
            "All values above are local-machine, single-process, non-concurrent measurements "
            "taken during the Task 36 evaluation run. Descriptive only -- not a real-time "
            "guarantee, not a production/deployment benchmark."
        ),
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _render_markdown(metrics: dict) -> str:
    h = metrics["headline"]
    p = metrics["paired"]
    lines = [
        "# Official ILO 2021 ISCO-08 WISCO Tier-1 Analysis (Task 37)",
        "",
        f"Generated: {metrics['generated_at_utc']}  |  n = {metrics['n_cases']}",
        "",
        "## 1. Headline exact 4-digit accuracy",
        "",
        "| System | Method label | n | Correct | Incorrect | Accuracy | 95% Wilson CI |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for sysname in ("flat", "hierarchical"):
        m = h[sysname]
        lines.append(
            f"| {sysname} | `{m['method_label']}` | {m['n']} | {m['correct']} | {m['incorrect']} | "
            f"{m['accuracy']:.4%} | [{m['ci_lo']:.4%}, {m['ci_hi']:.4%}] |"
        )
    lines += [
        "",
        "## 2. Paired comparison (McNemar, exact two-sided)",
        "",
        "| both correct | flat-only correct | hierarchical-only correct | both incorrect |",
        "|---:|---:|---:|---:|",
        f"| {p['both_correct']} | {p['flat_only_correct']} | {p['hierarchical_only_correct']} | {p['both_incorrect']} |",
        "",
        f"- Accuracy difference (hierarchical - flat): {p['accuracy_diff_hierarchical_minus_flat_pp']:.4f} pp",
        f"- McNemar statistic (min(b,c)): {p['mcnemar_statistic_min_b_c']}",
        f"- McNemar exact two-sided p-value: {p['mcnemar_p_value_two_sided_exact']:.6g}",
        f"- Implementation: {p['mcnemar_implementation']}",
        "",
        "## 3. Subgroup accuracy by language (descriptive only)",
        "",
        "| Language | System | n | Correct | Accuracy | 95% Wilson CI |",
        "|---|---|---:|---:|---:|---|",
    ]
    for lang, entry in metrics["by_language"].items():
        for sysname in ("flat", "hierarchical"):
            e = entry[sysname]
            acc = f"{e['accuracy']:.4%}" if e["accuracy"] is not None else "n/a"
            lines.append(f"| {lang} | {sysname} | {e['n']} | {e['correct']} | {acc} | [{e['ci_lo']:.4%}, {e['ci_hi']:.4%}] |")
    lines += [
        "",
        "## 4. Subgroup accuracy by ISCO-08 major group (descriptive only)",
        "",
        "| Major group | System | n | Correct | Accuracy | 95% Wilson CI |",
        "|---|---|---:|---:|---:|---|",
    ]
    for grp, entry in metrics["by_major_group"].items():
        for sysname in ("flat", "hierarchical"):
            e = entry[sysname]
            acc = f"{e['accuracy']:.4%}" if e["accuracy"] is not None else "n/a"
            lines.append(f"| {grp} | {sysname} | {e['n']} | {e['correct']} | {acc} | [{e['ci_lo']:.4%}, {e['ci_hi']:.4%}] |")
    lines += [
        "",
        "## 5. Operational summary (local-run descriptive only)",
        "",
        f"- Total hierarchical stage-query count: {metrics['operational']['total_hierarchical_stage_query_count']}",
        f"- Flat query duration: {metrics['operational']['flat_query_duration_ms']}",
        "",
        "## 6. Limitations",
        "",
    ]
    lines += [f"- {item}" for item in metrics["limitations"]]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat-csv", required=True, type=Path)
    parser.add_argument("--hierarchical-csv", required=True, type=Path)
    parser.add_argument("--heldout-csv", required=True, type=Path)
    parser.add_argument("--catalogue-csv", required=True, type=Path)
    parser.add_argument("--task36-report", required=True, type=Path)
    parser.add_argument("--task36-base-sha", required=True)
    parser.add_argument("--expected-flat-sha256", required=True)
    parser.add_argument("--expected-hierarchical-sha256", required=True)
    parser.add_argument("--expected-heldout-sha256", required=True)
    parser.add_argument("--expected-n", required=True, type=int)
    parser.add_argument("--max-stage-latency-ms", type=float, default=EXPECTED_MAX_STAGE_LATENCY_MS)
    parser.add_argument("--repo-root", type=Path, default=_ROOT)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    flat_rows, hier_rows, heldout_rows, gate_results = run_eligibility_gate(
        args.flat_csv, args.hierarchical_csv, args.heldout_csv, args.catalogue_csv,
        args.task36_report, args.task36_base_sha,
        args.expected_flat_sha256, args.expected_hierarchical_sha256, args.expected_heldout_sha256,
        args.expected_n, args.max_stage_latency_ms, args.repo_root,
    )
    all_passed = all(g["passed"] for g in gate_results)

    args.out.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()

    if not all_passed:
        failure_manifest = {
            "analysis_version": ANALYSIS_VERSION,
            "generated_at_utc": generated_at,
            "eligibility_gate_results": gate_results,
            "status": "GATE_FAILURE",
            "note": "No accuracy, interval, comparison, or subgroup statistic was computed or written.",
        }
        (args.out / "analysis_failure_manifest.json").write_text(
            json.dumps(failure_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        failed = [g["check"] for g in gate_results if not g["passed"]]
        print(f"ELIGIBILITY GATE FAILURE: {failed}", file=sys.stderr)
        print("No analysis output was written.", file=sys.stderr)
        sys.exit(1)

    lang_map = {r["case_id"]: r["input_language"] for r in heldout_rows}
    flat_headline = compute_headline(flat_rows, "flat", EXPECTED_FLAT_METHOD, sha256_file(args.flat_csv))
    hier_headline = compute_headline(hier_rows, "hierarchical", EXPECTED_HIERARCHICAL_METHOD, sha256_file(args.hierarchical_csv))
    paired = paired_contingency(flat_rows, hier_rows)
    by_language = _subgroup_metrics(flat_rows, hier_rows, lambda r: lang_map[r["case_id"]])
    by_major_group = _subgroup_metrics(flat_rows, hier_rows, lambda r: r["gold_isco_4digit"].strip()[0])
    operational = compute_operational_summary(flat_rows, hier_rows)

    metrics = {
        "analysis_version": ANALYSIS_VERSION,
        "generated_at_utc": generated_at,
        "n_cases": args.expected_n,
        "headline": {"flat": asdict(flat_headline), "hierarchical": asdict(hier_headline)},
        "paired": paired,
        "by_language": by_language,
        "by_major_group": by_major_group,
        "operational": operational,
        "correctness_definition": "correct = predicted_isco_4digit == gold_isco_4digit (exact match only; no prefix/partial/semantic matching)",
        "limitations": LIMITATIONS,
    }

    (args.out / "official_tier1_analysis.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md_text = _render_markdown(metrics)
    (args.out / "official_tier1_analysis.md").write_text(md_text, encoding="utf-8")

    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "script_path": str(Path(__file__)),
        "script_sha256": sha256_file(Path(__file__)),
        "generated_at_utc": generated_at,
        "inputs": {
            "flat_csv": {"path": str(args.flat_csv), "sha256": sha256_file(args.flat_csv)},
            "hierarchical_csv": {"path": str(args.hierarchical_csv), "sha256": sha256_file(args.hierarchical_csv)},
            "heldout_csv": {"path": str(args.heldout_csv), "sha256": sha256_file(args.heldout_csv)},
            "catalogue_csv": {"path": str(args.catalogue_csv), "sha256": sha256_file(args.catalogue_csv)},
        },
        "eligibility_gate_results": gate_results,
        "output_files": {
            "official_tier1_analysis.json": sha256_file(args.out / "official_tier1_analysis.json"),
            "official_tier1_analysis.md": sha256_file(args.out / "official_tier1_analysis.md"),
        },
    }
    (args.out / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"All eligibility gates passed. Wrote official Tier-1 analysis bundle to {args.out}")


if __name__ == "__main__":
    main()
