"""
eval/analyze_wisco_tier1.py

Conference I Reviewer #2 response, Task 18: the first authorized
accuracy/statistics analysis of Task 17's two raw, full-heldout WISCO v2
result CSVs (model-free flat retrieval and model-free strict
hierarchical retrieval). This module is deterministic and fully
offline -- it reads three local CSVs and writes local JSON/Markdown; it
never calls a network, Qdrant, a model, an LLM, or a subprocess.

Evidence boundary (read this before citing any number this script
produces)
--------------------------------------------------------------------
WISCO is an **externally sourced controlled multilingual ISCO-08
occupation-title benchmark** -- a curated list of job titles in 5
languages mapped to ISCO-08 codes by the WISCO project's own publishers.
It is **not** Labour Force Survey respondent data, not collected via
this project's survey flow, and not population-representative. Results
produced by this script may be cited as controlled multilingual ISCO-08
benchmark evidence only. They must never be represented as:
  - real Labour Force Survey validation;
  - population-representative performance;
  - ISIC, ISCED, or SRE performance (this run is ISCO-08-only by
    construction -- see the reference-CSV gate below);
  - reranker or LLM performance (reranking was off for both systems);
  - real-time or production performance (latency figures here are
    local-machine descriptive measurements under the Task 17
    environment, not a deployment guarantee);
  - an all-language, all-country, or all-occupation guarantee.

Fail-closed design
-------------------
Every raw-input invariant listed in Task 18's brief is checked BEFORE
any statistic is computed. On the first violation, the script prints a
human-readable error naming the violated invariant (and affected case
IDs where applicable) to stderr and exits nonzero. No partial output
bundle is ever written on a gate failure -- either every gate passes and
the full bundle (provenance.json, wisco_tier1_metrics.json,
wisco_tier1_metrics.md) is written, or nothing is written.

Reused statistics
------------------
wilson_score_interval() and mcnemar_test() are imported unmodified from
eval/analyze.py (Section D) rather than re-implemented, per Task 18's
instruction to reuse the project's existing definitions where practical.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from analyze import wilson_score_interval, mcnemar_test  # noqa: E402

ANALYSIS_VERSION = "1.0.0"

# Fixed list of the 24 Task 12 silent-fallback case IDs, verbatim from
# Task 15/17/18's shared brief -- must appear in the hierarchical raw
# output and individually pass every strict-hierarchy check.
KNOWN_RISK_IDS = [
    "WISCO-2310570000000-en", "WISCO-4312002200018-hi", "WISCO-6113040000000-en",
    "WISCO-6113040000000-ar", "WISCO-6113990000000-ar", "WISCO-6121002200018-en",
    "WISCO-6121070000000-en", "WISCO-6129000900018-en", "WISCO-6210000300018-ar",
    "WISCO-6221050000000-en", "WISCO-6221050000000-ar", "WISCO-6222010000000-en",
    "WISCO-6222010000000-ar", "WISCO-6222020000000-en", "WISCO-6222020000000-ar",
    "WISCO-6223000200018-en", "WISCO-6223000200018-ar", "WISCO-6224000500018-ar",
    "WISCO-6224000700018-ar", "WISCO-6330010000000-ar", "WISCO-6340000100016-en",
    "WISCO-9213010000000-ar", "WISCO-9216010000000-en", "WISCO-8160001500018-ur",
]

_ISCO4_RE = re.compile(r"^\d{4}$")
_EXPECTED_EVALUATION_STATUS = "measured"
_FLAT_METHOD_LABEL = "flat_semantic"
_STAGE_NUMS = (1, 2, 3, 4)
_MAX_EXAMPLES_IN_ERROR = 20


class GateFailure(Exception):
    """Raised to fail closed on any raw-input validation violation. The
    message names the violated invariant and, where applicable, the
    affected case IDs -- never a bare 'validation failed'."""


def _fail(msg: str) -> None:
    raise GateFailure(msg)


def _fail_with_ids(msg: str, ids: list[str]) -> None:
    examples = ids[:_MAX_EXAMPLES_IN_ERROR]
    more = f" (+{len(ids) - _MAX_EXAMPLES_IN_ERROR} more)" if len(ids) > _MAX_EXAMPLES_IN_ERROR else ""
    raise GateFailure(f"{msg} -- {len(ids)} affected case_id(s): {examples}{more}")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _require_columns(rows: list[dict], columns: list[str], csv_label: str) -> None:
    if not rows:
        _fail(f"{csv_label}: CSV has zero data rows")
    present = set(rows[0].keys())
    missing = [c for c in columns if c not in present]
    if missing:
        _fail(f"{csv_label}: required column(s) missing: {missing}")


def _truthy(raw: Optional[str]) -> bool:
    return (raw or "").strip().lower() in ("true", "1", "yes")


def _is_zero_or_blank(raw: Optional[str]) -> bool:
    raw = (raw or "").strip()
    if raw == "":
        return True
    try:
        return float(raw) == 0.0
    except ValueError:
        return False


def _valid_stage_list(raw: Optional[str]) -> bool:
    if not raw or raw == "null":
        return False
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, list) and len(parsed) > 0


def _stage_latency_ok(raw: Optional[str], max_ms: float) -> bool:
    if raw is None or str(raw).strip() == "":
        return False
    try:
        v = float(raw)
    except ValueError:
        return False
    if math.isnan(v) or math.isinf(v):
        return False
    return v <= max_ms


# ---------------------------------------------------------------------------
# Raw-input provenance and integrity gates
# ---------------------------------------------------------------------------

def validate_reference(reference_csv: Path, expected_sha256: str, expected_n: int) -> tuple[list[dict], dict]:
    if not reference_csv.exists():
        _fail(f"reference CSV not found: {reference_csv}")
    actual_sha = sha256_file(reference_csv)
    if actual_sha != expected_sha256:
        _fail(
            f"reference CSV sha256 mismatch: expected {expected_sha256}, "
            f"got {actual_sha} (path={reference_csv})"
        )
    rows = _load_rows(reference_csv)
    _require_columns(rows, ["case_id", "input_language", "gold_isic", "gold_isced"], "reference CSV")
    if len(rows) != expected_n:
        _fail(f"reference CSV row count mismatch: expected {expected_n}, got {len(rows)} (path={reference_csv})")

    ids = [r["case_id"] for r in rows]
    if len(set(ids)) != len(ids):
        seen, dupes = set(), []
        for i in ids:
            if i in seen:
                dupes.append(i)
            seen.add(i)
        _fail_with_ids("reference CSV has duplicate case_id values", dupes)

    nonblank_isic = [r["case_id"] for r in rows if (r.get("gold_isic") or "").strip()]
    if nonblank_isic:
        _fail_with_ids("reference CSV has nonblank gold_isic (expected all-blank for this ISCO-only benchmark)", nonblank_isic)
    nonblank_isced = [r["case_id"] for r in rows if (r.get("gold_isced") or "").strip()]
    if nonblank_isced:
        _fail_with_ids("reference CSV has nonblank gold_isced (expected all-blank for this ISCO-only benchmark)", nonblank_isced)

    gate_record = {
        "path": str(reference_csv),
        "sha256": actual_sha,
        "expected_sha256": expected_sha256,
        "sha256_match": True,
        "n_rows": len(rows),
        "expected_n": expected_n,
        "n_rows_match": True,
        "ids_unique": True,
        "n_nonblank_gold_isic": 0,
        "n_nonblank_gold_isced": 0,
        "status": "pass",
    }
    return rows, gate_record


_RESULT_CORE_COLUMNS = [
    "case_id", "gold_isco_4digit", "pred_isco_4digit", "pred_method", "error",
    "reranker_fired", "reranker_model", "estimated_cost_usd", "prompt_tokens",
    "completion_tokens",
]
_HIERARCHICAL_EXTRA_COLUMNS = [
    "stage1_candidates", "stage2_candidates", "stage3_candidates", "stage4_candidates",
    "stage1_latency_ms", "stage2_latency_ms", "stage3_latency_ms", "stage4_latency_ms",
]


def validate_result_csv(
    csv_path: Path,
    reference_ids: list[str],
    expected_n: int,
    system: str,
    hierarchical_prefix: str,
    max_stage_latency_ms: float,
) -> tuple[list[dict], dict]:
    label = f"{system} result CSV"
    if not csv_path.exists():
        _fail(f"{label} not found: {csv_path}")
    actual_sha = sha256_file(csv_path)
    rows = _load_rows(csv_path)

    required = list(_RESULT_CORE_COLUMNS)
    if system == "hierarchical":
        required += _HIERARCHICAL_EXTRA_COLUMNS
    _require_columns(rows, required, label)

    if len(rows) != expected_n:
        _fail(f"{label} row count mismatch: expected {expected_n}, got {len(rows)} (path={csv_path})")

    ids = [r["case_id"] for r in rows]
    if len(set(ids)) != len(ids):
        seen, dupes = set(), []
        for i in ids:
            if i in seen:
                dupes.append(i)
            seen.add(i)
        _fail_with_ids(f"{label} has duplicate case_id values", dupes)

    if ids != reference_ids:
        if set(ids) != set(reference_ids):
            missing = [i for i in reference_ids if i not in set(ids)]
            extra = [i for i in ids if i not in set(reference_ids)]
            if missing:
                _fail_with_ids(f"{label} is missing case_id(s) present in the reference CSV", missing)
            _fail_with_ids(f"{label} has case_id(s) not present in the reference CSV", extra)
        _fail(f"{label}: case_id order does not exactly match the canonical reference CSV order")

    # Gold/pred 4-digit ISCO code validity (also the source of the derived
    # 1/2/3-digit prefixes used everywhere else in this script -- Task 18
    # explicitly requires deriving prefixes from the 4-digit code rather
    # than trusting a separate, potentially-mismatched auxiliary column).
    bad_gold = [r["case_id"] for r in rows if not _ISCO4_RE.match((r.get("gold_isco_4digit") or "").strip())]
    if bad_gold:
        _fail_with_ids(f"{label} has missing/invalid gold_isco_4digit (must be exactly 4 digits)", bad_gold)
    bad_pred = [r["case_id"] for r in rows if not _ISCO4_RE.match((r.get("pred_isco_4digit") or "").strip())]
    if bad_pred:
        _fail_with_ids(f"{label} has missing/invalid pred_isco_4digit (must be exactly 4 digits)", bad_pred)

    non_blank_error = [r["case_id"] for r in rows if (r.get("error") or "").strip() != ""]
    if non_blank_error:
        _fail_with_ids(f"{label} has nonblank row-level error field", non_blank_error)

    if "evaluation_status" in rows[0]:
        bad_status = [
            r["case_id"] for r in rows
            if (r.get("evaluation_status") or "").strip() != _EXPECTED_EVALUATION_STATUS
        ]
        if bad_status:
            _fail_with_ids(
                f"{label} has evaluation_status != {_EXPECTED_EVALUATION_STATUS!r} "
                f"(dry-run or synthetic rows are not eligible for analysis)",
                bad_status,
            )

    if "sre_status" in rows[0]:
        bad_sre = [r["case_id"] for r in rows if (r.get("sre_status") or "").strip() != "not_applicable"]
        if bad_sre:
            _fail_with_ids(
                f"{label}: sre_status is not consistently 'not_applicable' "
                f"(this run is ISCO-08-only; ISIC/ISCED/SRE must never have been constructed)",
                bad_sre,
            )

    reranker_fired_ids = [r["case_id"] for r in rows if _truthy(r.get("reranker_fired"))]
    if reranker_fired_ids:
        _fail_with_ids(f"{label}: reranker_fired is true (reranking must be off for a model-free run)", reranker_fired_ids)
    nonblank_reranker_model = [r["case_id"] for r in rows if (r.get("reranker_model") or "").strip() not in ("",)]
    if nonblank_reranker_model:
        _fail_with_ids(f"{label}: reranker_model is nonblank (reranking must be off)", nonblank_reranker_model)
    nonzero_cost = [r["case_id"] for r in rows if not _is_zero_or_blank(r.get("estimated_cost_usd"))]
    if nonzero_cost:
        _fail_with_ids(f"{label}: nonzero estimated_cost_usd (no LLM/API call is authorized for this run)", nonzero_cost)
    nonzero_tokens = [
        r["case_id"] for r in rows
        if not _is_zero_or_blank(r.get("prompt_tokens")) or not _is_zero_or_blank(r.get("completion_tokens"))
    ]
    if nonzero_tokens:
        _fail_with_ids(f"{label}: nonzero prompt_tokens/completion_tokens (no LLM call is authorized for this run)", nonzero_tokens)

    if system == "flat":
        bad_method = [r["case_id"] for r in rows if r.get("pred_method") != _FLAT_METHOD_LABEL]
        if bad_method:
            _fail_with_ids(f"{label}: pred_method is not {_FLAT_METHOD_LABEL!r} for every row", bad_method)
    elif system == "hierarchical":
        bad_method = [r["case_id"] for r in rows if not (r.get("pred_method") or "").startswith(hierarchical_prefix)]
        if bad_method:
            _fail_with_ids(
                f"{label}: pred_method does not start with {hierarchical_prefix!r} for every row "
                f"(flat fallback or a missing method would appear here)",
                bad_method,
            )
        for stage_num in _STAGE_NUMS:
            col = f"stage{stage_num}_candidates"
            bad_stage = [r["case_id"] for r in rows if not _valid_stage_list(r.get(col))]
            if bad_stage:
                _fail_with_ids(f"{label}: {col} is missing/empty/invalid JSON for these rows", bad_stage)
        for stage_num in _STAGE_NUMS:
            col = f"stage{stage_num}_latency_ms"
            bad_latency = [r["case_id"] for r in rows if not _stage_latency_ok(r.get(col), max_stage_latency_ms)]
            if bad_latency:
                _fail_with_ids(
                    f"{label}: {col} is missing/non-numeric/exceeds --max-stage-latency-ms={max_stage_latency_ms}",
                    bad_latency,
                )

        by_id = {r["case_id"]: r for r in rows}
        missing_risk = [rid for rid in KNOWN_RISK_IDS if rid not in by_id]
        if missing_risk:
            _fail_with_ids("hierarchical result CSV is missing known-risk case_id(s)", missing_risk)
        failing_risk = []
        for rid in KNOWN_RISK_IDS:
            r = by_id[rid]
            ok = (
                (r.get("pred_method") or "").startswith(hierarchical_prefix)
                and all(_valid_stage_list(r.get(f"stage{n}_candidates")) for n in _STAGE_NUMS)
                and all(_stage_latency_ok(r.get(f"stage{n}_latency_ms"), max_stage_latency_ms) for n in _STAGE_NUMS)
                and (r.get("error") or "").strip() == ""
            )
            if not ok:
                failing_risk.append(rid)
        if failing_risk:
            _fail_with_ids("known-risk case_id(s) failed the strict hierarchical check", failing_risk)
    else:
        _fail(f"unknown system {system!r} (must be 'flat' or 'hierarchical')")

    gate_record = {
        "path": str(csv_path),
        "sha256": actual_sha,
        "n_rows": len(rows),
        "expected_n": expected_n,
        "n_rows_match": True,
        "ids_unique": True,
        "ids_match_reference_order": True,
        "n_nonblank_error": 0,
        "evaluation_status_checked": "evaluation_status" in rows[0],
        "sre_status_checked": "sre_status" in rows[0],
        "reranker_off_confirmed": True,
        "zero_cost_and_tokens_confirmed": True,
        "status": "pass",
    }
    if system == "hierarchical":
        gate_record["all_stage_evidence_valid"] = True
        gate_record["all_stage_latencies_within_cap"] = True
        gate_record["known_risk_ids_present_and_passing"] = len(KNOWN_RISK_IDS)
    return rows, gate_record


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@dataclass
class DigitAccuracy:
    metric_name: str
    n: int
    correct: int
    accuracy: float
    ci_lo: float
    ci_hi: float


def _row_correct_4digit(r: dict) -> bool:
    return r["gold_isco_4digit"].strip() == r["pred_isco_4digit"].strip()


def compute_digit_accuracy(rows: list[dict], digit: int, metric_name: str) -> DigitAccuracy:
    n = len(rows)
    correct = sum(
        1 for r in rows
        if r["gold_isco_4digit"].strip()[:digit] == r["pred_isco_4digit"].strip()[:digit]
    )
    lo, hi = wilson_score_interval(correct, n)
    return DigitAccuracy(metric_name, n, correct, round(correct / n, 4), round(lo, 4), round(hi, 4))


def compute_overall_accuracy(rows: list[dict]) -> list[DigitAccuracy]:
    return [compute_digit_accuracy(rows, d, f"isco_top1_{d}digit") for d in (1, 2, 3, 4)]


def compute_language_accuracy(rows: list[dict], lang_map: dict[str, str]) -> tuple[dict[str, DigitAccuracy], float]:
    by_lang: dict[str, list[dict]] = {}
    for r in rows:
        lang = lang_map[r["case_id"]]
        by_lang.setdefault(lang, []).append(r)
    results = {
        lang: compute_digit_accuracy(lrows, 4, f"isco_top1_4digit_{lang}")
        for lang, lrows in sorted(by_lang.items())
    }
    macro_avg = round(statistics.mean(m.accuracy for m in results.values()), 4)
    return results, macro_avg


def paired_comparison_4digit(flat_rows: list[dict], hier_rows: list[dict]) -> dict:
    flat_by_id = {r["case_id"]: r for r in flat_rows}
    hier_by_id = {r["case_id"]: r for r in hier_rows}
    a = b = c = d = 0
    for cid, fr in flat_by_id.items():
        hr = hier_by_id[cid]
        f_ok = _row_correct_4digit(fr)
        h_ok = _row_correct_4digit(hr)
        if f_ok and h_ok:
            a += 1
        elif f_ok and not h_ok:
            b += 1
        elif not f_ok and h_ok:
            c += 1
        else:
            d += 1
    n = a + b + c + d
    flat_correct = a + b
    hier_correct = a + c
    diff_pp = round((hier_correct / n - flat_correct / n) * 100, 4)
    stat, p_value = mcnemar_test(b, c)
    return {
        "n_pairs": n,
        "flat_correct": flat_correct,
        "hierarchical_correct": hier_correct,
        "both_correct": a,
        "flat_correct_hierarchical_wrong_b": b,
        "flat_wrong_hierarchical_correct_c": c,
        "both_wrong": d,
        "n_discordant_pairs": b + c,
        "accuracy_diff_hierarchical_minus_flat_pp": diff_pp,
        "mcnemar_statistic_min_b_c": stat,
        "mcnemar_p_value_two_sided_exact": p_value,
        "mcnemar_null_hypothesis": (
            "The two systems (flat, hierarchical) have equal marginal probability "
            "of a correct four-digit ISCO-08 prediction on these paired WISCO cases. "
            "This is a statistical statement about these paired controlled-benchmark "
            "predictions only -- not a causal claim and not a real-world performance claim."
        ),
    }


def latency_descriptives(rows: list[dict], field_name: str) -> dict:
    vals = []
    for r in rows:
        raw = r.get(field_name, "")
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
        return {
            "field": field_name, "n": 0, "mean_ms": None, "median_ms": None,
            "p95_ms": None, "max_ms": None, "status": "not_measured",
            "reason": "no non-empty finite values present",
        }
    vals_sorted = sorted(vals)
    n = len(vals_sorted)
    rank = max(1, min(n, math.ceil(0.95 * n)))
    return {
        "field": field_name,
        "n": n,
        "mean_ms": round(statistics.mean(vals_sorted), 3),
        "median_ms": round(statistics.median(vals_sorted), 3),
        "p95_ms": round(vals_sorted[rank - 1], 3),
        "max_ms": round(vals_sorted[-1], 3),
        "status": "measured",
        "note": "local-machine descriptive measurement under the Task 17 environment; not a real-time or deployment guarantee",
    }


def traceability_counts(rows: list[dict]) -> dict:
    has_retry_col = bool(rows) and "keyword_anchor_retry_used" in rows[0]
    has_stage1_source_col = bool(rows) and "stage1_source" in rows[0]
    result: dict = {
        "keyword_anchor_retry_used_field_present": has_retry_col,
        "stage1_source_field_present": has_stage1_source_col,
    }
    if has_retry_col:
        crosstab = {
            "retry_used_and_correct": 0, "retry_used_and_incorrect": 0,
            "retry_not_used_and_correct": 0, "retry_not_used_and_incorrect": 0,
        }
        n_retry = 0
        for r in rows:
            retried = _truthy(r.get("keyword_anchor_retry_used"))
            correct = _row_correct_4digit(r)
            if retried:
                n_retry += 1
                crosstab["retry_used_and_correct" if correct else "retry_used_and_incorrect"] += 1
            else:
                crosstab["retry_not_used_and_correct" if correct else "retry_not_used_and_incorrect"] += 1
        result["n_keyword_anchor_retry_used_true"] = n_retry
        result["retry_used_vs_correct_crosstab"] = crosstab
        result["diagnostic_note"] = (
            "This is a behavioral traceability count, not accuracy evidence for a "
            "separate retrieval method -- retried and non-retried rows are both "
            "scored by the same single hierarchical accuracy metric above."
        )
    if has_stage1_source_col:
        counts: dict[str, int] = {}
        for r in rows:
            src = r.get("stage1_source", "")
            counts[src] = counts.get(src, 0) + 1
        result["stage1_source_counts"] = counts
    return result


def read_qdrant_provenance(flat_csv: Path, hierarchical_csv: Path) -> dict:
    """Reads Task 17's already-recorded before/after Qdrant point-count
    evidence as provenance only. Makes zero Qdrant calls itself."""
    root_flat = flat_csv.resolve().parent.parent
    root_hier = hierarchical_csv.resolve().parent.parent
    result: dict = {
        "note": "Read from Task 17's recorded evidence only -- no live Qdrant call was made in this task.",
    }
    if root_flat != root_hier:
        result["warning"] = "flat and hierarchical CSVs do not share a common Task 17 output root; provenance not read"
        return result
    before = root_flat / "qdrant_point_counts_before.txt"
    after = root_flat / "qdrant_point_counts_after.txt"
    result["before_counts_path"] = str(before)
    result["after_counts_path"] = str(after)
    result["before_counts_text"] = before.read_text(encoding="utf-8") if before.exists() else None
    result["after_counts_text"] = after.read_text(encoding="utf-8") if after.exists() else None
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat-csv", required=True, type=Path)
    parser.add_argument("--hierarchical-csv", required=True, type=Path)
    parser.add_argument("--reference-csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--expected-reference-sha256", required=True)
    parser.add_argument("--expected-n", required=True, type=int)
    parser.add_argument("--expected-hierarchical-method-prefix", default="hierarchical_")
    parser.add_argument("--max-stage-latency-ms", required=True, type=float)
    args = parser.parse_args()

    try:
        reference_rows, ref_gate = validate_reference(
            args.reference_csv, args.expected_reference_sha256, args.expected_n
        )
        reference_ids = [r["case_id"] for r in reference_rows]
        flat_rows, flat_gate = validate_result_csv(
            args.flat_csv, reference_ids, args.expected_n, "flat",
            args.expected_hierarchical_method_prefix, args.max_stage_latency_ms,
        )
        hier_rows, hier_gate = validate_result_csv(
            args.hierarchical_csv, reference_ids, args.expected_n, "hierarchical",
            args.expected_hierarchical_method_prefix, args.max_stage_latency_ms,
        )
    except GateFailure as exc:
        print(f"GATE FAILURE: {exc}", file=sys.stderr)
        print("No analysis output was written.", file=sys.stderr)
        sys.exit(1)

    lang_map = {r["case_id"]: r["input_language"] for r in reference_rows}

    overall = {
        "flat": [asdict(m) for m in compute_overall_accuracy(flat_rows)],
        "hierarchical": [asdict(m) for m in compute_overall_accuracy(hier_rows)],
    }
    flat_lang, flat_macro = compute_language_accuracy(flat_rows, lang_map)
    hier_lang, hier_macro = compute_language_accuracy(hier_rows, lang_map)
    by_language = {
        "flat": {lang: asdict(m) for lang, m in flat_lang.items()},
        "hierarchical": {lang: asdict(m) for lang, m in hier_lang.items()},
        "flat_macro_average_4digit": flat_macro,
        "hierarchical_macro_average_4digit": hier_macro,
        "macro_average_definition": (
            "Unweighted arithmetic mean of the per-language 4-digit accuracy "
            "proportions (5 languages, each weighted equally regardless of n). "
            "Descriptive only -- no pooled confidence interval is reported for "
            "a macro-average of proportions computed over different sample sizes."
        ),
    }
    paired = paired_comparison_4digit(flat_rows, hier_rows)

    latency = {
        "flat_end_to_end_ms": latency_descriptives(flat_rows, "end_to_end_latency_ms"),
        "hierarchical_end_to_end_ms": latency_descriptives(hier_rows, "end_to_end_latency_ms"),
        "hierarchical_stage1_ms": latency_descriptives(hier_rows, "stage1_latency_ms"),
        "hierarchical_stage2_ms": latency_descriptives(hier_rows, "stage2_latency_ms"),
        "hierarchical_stage3_ms": latency_descriptives(hier_rows, "stage3_latency_ms"),
        "hierarchical_stage4_ms": latency_descriptives(hier_rows, "stage4_latency_ms"),
        "caveat": (
            "All latency values above are local-machine, single-process, "
            "non-concurrent measurements taken during the Task 17 evaluation run. "
            "They are descriptive only: not a real-time guarantee, not a "
            "production/deployment benchmark, and not adjusted for hardware, "
            "concurrency, or network conditions."
        ),
    }

    traceability = {
        "flat": traceability_counts(flat_rows),
        "hierarchical": traceability_counts(hier_rows),
    }

    qdrant_provenance = read_qdrant_provenance(args.flat_csv, args.hierarchical_csv)

    limitations = [
        "WISCO is externally sourced occupation-title reference data, not real LFS respondent data.",
        "Results cover ISCO-08 only -- not ISIC, ISCED, SRE, reranking, LLM performance, or system-wide survey validation.",
        "Performance/latency values are descriptive local-machine measurements, not real-time or production guarantees.",
        "The full 18,747-row results use a leakage-audited heldout split and model-free retrieval, but are still not evidence of national representativeness or field deployment.",
        "The historical B1 baseline remains quarantined and unrelated to this controlled WISCO comparison.",
    ]

    script_hash = sha256_file(Path(__file__))
    provenance = {
        "analysis_version": ANALYSIS_VERSION,
        "script_path": str(Path(__file__)),
        "script_sha256": script_hash,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "reference_csv": ref_gate,
            "flat_csv": flat_gate,
            "hierarchical_csv": hier_gate,
        },
        "known_risk_ids": KNOWN_RISK_IDS,
        "max_stage_latency_ms_cap": args.max_stage_latency_ms,
        "expected_hierarchical_method_prefix": args.expected_hierarchical_method_prefix,
        "qdrant_provenance": qdrant_provenance,
        "evidence_boundary": (
            "WISCO is an externally sourced controlled multilingual ISCO-08 "
            "occupation-title benchmark, not real Labour Force Survey respondent "
            "data. Results are controlled multilingual ISCO-08 benchmark evidence "
            "only -- not real-LFS validation, population-representative evidence, "
            "ISIC/ISCED/SRE evidence, reranker/LLM performance, or a real-time/"
            "production performance guarantee."
        ),
        "limitations": limitations,
    }

    metrics = {
        "analysis_version": ANALYSIS_VERSION,
        "generated_at_utc": provenance["generated_at_utc"],
        "n_cases": args.expected_n,
        "overall_accuracy": overall,
        "by_language_4digit_accuracy": by_language,
        "paired_4digit_comparison": paired,
        "latency_descriptives_ms": latency,
        "diagnostic_traceability": traceability,
        "evidence_boundary": provenance["evidence_boundary"],
        "limitations": limitations,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out / "wisco_tier1_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out / "wisco_tier1_metrics.md").write_text(_render_markdown(metrics), encoding="utf-8")

    print(f"All raw-input gates passed. Wrote WISCO Tier 1 analysis bundle to {args.out}")


def _render_markdown(metrics: dict) -> str:
    lines = [
        "# WISCO Tier 1 Controlled Evaluation -- Measured Results",
        "",
        "## 1. Scope and evidence boundary",
        "",
        metrics["evidence_boundary"],
        "",
        "## 2. Overall ISCO-08 accuracy (exact match, derived from 4-digit codes)",
        "",
        "| System | Digits | n | Correct | Accuracy | 95% Wilson CI |",
        "|---|---|---|---|---|---|",
    ]
    for system in ("flat", "hierarchical"):
        for m in metrics["overall_accuracy"][system]:
            lines.append(
                f"| {system} | {m['metric_name']} | {m['n']} | {m['correct']} | "
                f"{m['accuracy']:.4f} | [{m['ci_lo']:.4f}, {m['ci_hi']:.4f}] |"
            )
    lines += ["", "## 3. 4-digit accuracy by WISCO language", "", "| System | Language | n | Correct | Accuracy | 95% Wilson CI |", "|---|---|---|---|---|---|"]
    by_lang = metrics["by_language_4digit_accuracy"]
    for system in ("flat", "hierarchical"):
        for lang, m in sorted(by_lang[system].items()):
            lines.append(
                f"| {system} | {lang} | {m['n']} | {m['correct']} | "
                f"{m['accuracy']:.4f} | [{m['ci_lo']:.4f}, {m['ci_hi']:.4f}] |"
            )
    lines += [
        "",
        f"Unweighted macro-average (5 languages, no pooled CI) -- flat: "
        f"{by_lang['flat_macro_average_4digit']:.4f}, hierarchical: "
        f"{by_lang['hierarchical_macro_average_4digit']:.4f}.",
        "",
        by_lang["macro_average_definition"],
        "",
        "## 4. Paired 4-digit comparison (flat vs. hierarchical)",
        "",
    ]
    p = metrics["paired_4digit_comparison"]
    lines += [
        f"- n pairs: {p['n_pairs']}",
        f"- flat correct: {p['flat_correct']}; hierarchical correct: {p['hierarchical_correct']}",
        f"- both correct: {p['both_correct']}; both wrong: {p['both_wrong']}",
        f"- flat correct, hierarchical wrong (b): {p['flat_correct_hierarchical_wrong_b']}",
        f"- flat wrong, hierarchical correct (c): {p['flat_wrong_hierarchical_correct_c']}",
        f"- discordant pairs (b+c): {p['n_discordant_pairs']}",
        f"- accuracy difference (hierarchical - flat): {p['accuracy_diff_hierarchical_minus_flat_pp']:.4f} percentage points",
        f"- McNemar exact two-sided p-value: {p['mcnemar_p_value_two_sided_exact']:.6g} (statistic = min(b,c) = {p['mcnemar_statistic_min_b_c']})",
        "",
        p["mcnemar_null_hypothesis"],
        "",
        "## 5. Latency descriptives (local-machine, non-real-time)",
        "",
        "| Field | n | Mean (ms) | Median (ms) | p95 (ms) | Max (ms) |",
        "|---|---|---|---|---|---|",
    ]
    lat = metrics["latency_descriptives_ms"]
    for key in (
        "flat_end_to_end_ms", "hierarchical_end_to_end_ms",
        "hierarchical_stage1_ms", "hierarchical_stage2_ms",
        "hierarchical_stage3_ms", "hierarchical_stage4_ms",
    ):
        d = lat[key]
        if d["status"] == "measured":
            lines.append(f"| {key} | {d['n']} | {d['mean_ms']} | {d['median_ms']} | {d['p95_ms']} | {d['max_ms']} |")
        else:
            lines.append(f"| {key} | 0 | — | — | — | — (not measured: {d['reason']}) |")
    lines += ["", lat["caveat"], "", "## 6. Diagnostic traceability", ""]
    for system in ("flat", "hierarchical"):
        t = metrics["diagnostic_traceability"][system]
        lines.append(f"**{system}**: {json.dumps(t, ensure_ascii=False)}")
        lines.append("")
    lines += ["## 7. Limitations and prohibited inferences", ""]
    for item in metrics["limitations"]:
        lines.append(f"- {item}")
    lines += [
        "",
        "These results must never be described as real-LFS validation, "
        "population-representative performance, ISIC/ISCED/SRE evidence, "
        "reranker/LLM performance, or a real-time/production guarantee.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
