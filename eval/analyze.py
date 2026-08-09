"""
eval/analyze.py

Accuracy analysis for eval/run_eval.py's CaseResult CSVs (Conference I
Reviewer #2 response, Section D). This is the file run_eval.py's module
docstring has forward-referenced since the B2 work, but which never
existed on this branch until now -- no accuracy scoring of any kind
existed for ISCO, ISIC, or ISCED before this module, despite gold_isic/
gold_isced already being captured per row.

Scope, honestly stated
-----------------------
ISCO: top-1 exact-match accuracy at 1/2/3/4-digit granularity (gold_isco_
      Ndigit vs pred_isco_Ndigit, both already populated per row) plus a
      "top-3 in pre-rerank pool" metric from CaseResult.gold_rank_in_pool
      (<=3), which measures something DIFFERENT from top-3 of the FINAL
      ranked prediction -- CaseResult does not store a ranked top-3 of
      final predictions, so this module does not claim to compute that.
ISIC: exact-match accuracy at section/division/group/class, using the
      pred_isic_* columns (division/group/class added to CaseResult in
      this same change -- see run_eval.py's CaseResult docstring).
ISCED: exact-match accuracy at level (0-8) and broad/narrow/detailed field.

If a CSV lacks a required column (e.g. an older run predating the
division/group/class columns), the corresponding metric is reported as
None with a reason -- never silently computed as 0% or skipped without
explanation.

Wilson score interval and McNemar's test are hand-rolled (no scipy
dependency in this project) -- see wilson_score_interval() and
mcnemar_test() below.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Statistics (hand-rolled, no scipy)
# ---------------------------------------------------------------------------

def wilson_score_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% (default z=1.96) Wilson score confidence interval for a binomial
    proportion. Standard closed-form formula -- see e.g. Wilson (1927) or
    any statistics reference for binomial CIs. Returns (lo, hi) in [0, 1].
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    if not (0 <= successes <= n):
        raise ValueError("successes must be in [0, n]")
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = p + z ** 2 / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n)
    lo = (centre - margin) / denom
    hi = (centre + margin) / denom
    return max(0.0, lo), min(1.0, hi)


def mcnemar_test(b: int, c: int) -> tuple[float, float]:
    """
    Exact two-sided McNemar's test for paired binary outcomes (e.g. "system
    A correct, system B wrong" = b cases; "system A wrong, system B
    correct" = c cases -- concordant cases where both are right or both are
    wrong are irrelevant to the test and not passed in).

    Uses the exact binomial form (appropriate for small b+c, standard
    practice -- avoids the continuity-corrected chi-square approximation's
    known inaccuracy when b+c is small): statistic is min(b, c); p-value is
    the two-sided exact binomial test against p=0.5.

    Returns (statistic, p_value). statistic here is min(b, c), following
    the common exact-test convention (not a chi-square statistic).

    Task 37: `math.comb(n, k) * p**k * (1-p)**(n-k)` raises
    `OverflowError` once `math.comb(n, k)` exceeds what a Python float
    can represent (~1.8e308) -- which a full 18,747-row paired WISCO
    comparison can reach: it takes only a few hundred discordant pairs
    (b+c) to overflow, and thousands are routine at this scale. Below
    that threshold, this still uses the original exact-integer
    combinatorics (bit-for-bit identical to every prior caller/test,
    including exact-fraction cases like b=c=1 that land on a power of
    two). Only on `OverflowError` does it fall back to an equivalent
    LOG-SPACE computation (`math.lgamma`-based `log(C(n, k))`, summed
    then exponentiated) -- mathematically identical, numerically stable
    at any n, but not guaranteed bit-exact (routine float error, ~1e-15)
    since it no longer benefits from exact power-of-two fractions the
    way the direct form incidentally does for tiny n.
    """
    if b < 0 or c < 0:
        raise ValueError("b and c must be >= 0")
    n = b + c
    if n == 0:
        return 0.0, 1.0
    k = min(b, c)

    def _binom_pmf_exact(k_: int, n_: int) -> float:
        return math.comb(n_, k_) * (0.5 ** k_) * (0.5 ** (n_ - k_))

    def _binom_pmf_log(k_: int, n_: int) -> float:
        # log(C(n_, k_) * 0.5**n_) -- p=0.5 is fixed for McNemar's exact test.
        log_choose = math.lgamma(n_ + 1) - math.lgamma(k_ + 1) - math.lgamma(n_ - k_ + 1)
        return math.exp(log_choose - n_ * math.log(2))

    # Two-sided exact p-value: sum the probability of every outcome at
    # least as extreme as the observed split, under the null p=0.5.
    try:
        p_value = sum(_binom_pmf_exact(i, n) for i in range(0, k + 1))
        p_value += sum(_binom_pmf_exact(i, n) for i in range(n - k, n + 1) if i > k)
    except OverflowError:
        p_value = sum(_binom_pmf_log(i, n) for i in range(0, k + 1))
        p_value += sum(_binom_pmf_log(i, n) for i in range(n - k, n + 1) if i > k)
    p_value = min(1.0, p_value)
    return float(k), p_value


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_case_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Accuracy metrics
# ---------------------------------------------------------------------------

@dataclass
class AccuracyMetric:
    metric_name: str
    n: int
    successes: Optional[int]
    accuracy: Optional[float]
    ci_lo: Optional[float]
    ci_hi: Optional[float]
    status: str  # "measured" | "not_measured"
    reason: Optional[str] = None


def _exact_match_metric(rows: list[dict], gold_col: str, pred_col: str, metric_name: str) -> AccuracyMetric:
    if not rows or gold_col not in rows[0] or pred_col not in rows[0]:
        return AccuracyMetric(
            metric_name=metric_name, n=0, successes=None, accuracy=None,
            ci_lo=None, ci_hi=None, status="not_measured",
            reason=f"column '{gold_col}' or '{pred_col}' not present in this CSV",
        )
    scored = [
        (r[gold_col].strip(), r[pred_col].strip()) for r in rows
        if r.get(gold_col, "").strip()
    ]
    if not scored:
        return AccuracyMetric(
            metric_name=metric_name, n=0, successes=None, accuracy=None,
            ci_lo=None, ci_hi=None, status="not_measured",
            reason=f"no row has a non-empty '{gold_col}' value",
        )
    successes = sum(1 for gold, pred in scored if gold == pred)
    n = len(scored)
    lo, hi = wilson_score_interval(successes, n)
    return AccuracyMetric(
        metric_name=metric_name, n=n, successes=successes,
        accuracy=round(successes / n, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
        status="measured",
    )


def isco_accuracy(rows: list[dict]) -> list[AccuracyMetric]:
    metrics = [
        _exact_match_metric(rows, f"gold_isco_{d}digit", f"pred_isco_{d}digit", f"isco_top1_{d}digit")
        for d in (1, 2, 3, 4)
    ]
    # "top-3 in pre-rerank pool" -- a real, different metric from top-3 of
    # the final prediction (see module docstring); uses the existing
    # gold_rank_in_pool column.
    if rows and "gold_rank_in_pool" in rows[0]:
        ranked = [r for r in rows if (r.get("gold_rank_in_pool") or "").strip()]
        if ranked:
            n = len(ranked)
            successes = sum(1 for r in ranked if int(r["gold_rank_in_pool"]) <= 3)
            lo, hi = wilson_score_interval(successes, n)
            metrics.append(AccuracyMetric(
                metric_name="isco_top3_prererank_pool", n=n, successes=successes,
                accuracy=round(successes / n, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                status="measured",
            ))
        else:
            metrics.append(AccuracyMetric(
                metric_name="isco_top3_prererank_pool", n=0, successes=None, accuracy=None,
                ci_lo=None, ci_hi=None, status="not_measured",
                reason="no row has a non-empty gold_rank_in_pool value",
            ))
    else:
        metrics.append(AccuracyMetric(
            metric_name="isco_top3_prererank_pool", n=0, successes=None, accuracy=None,
            ci_lo=None, ci_hi=None, status="not_measured",
            reason="column 'gold_rank_in_pool' not present in this CSV",
        ))
    return metrics


def isic_accuracy(rows: list[dict]) -> list[AccuracyMetric]:
    return [
        _exact_match_metric(rows, "gold_isic", "pred_isic_section", "isic_section"),
        _exact_match_metric(rows, "gold_isic_division", "pred_isic_division", "isic_division"),
        _exact_match_metric(rows, "gold_isic_group", "pred_isic_group", "isic_group"),
        _exact_match_metric(rows, "gold_isic_class", "pred_isic_class", "isic_class"),
    ]


def isced_accuracy(rows: list[dict]) -> list[AccuracyMetric]:
    return [
        _exact_match_metric(rows, "gold_isced", "pred_isced_level", "isced_level"),
        _exact_match_metric(rows, "gold_isced_broad", "pred_isced_broad", "iscedf_broad"),
        _exact_match_metric(rows, "gold_isced_narrow", "pred_isced_narrow", "iscedf_narrow"),
        _exact_match_metric(rows, "gold_isced_detailed", "pred_isced_detailed", "iscedf_detailed"),
    ]


def analyze_csv(csv_path: Path) -> dict[str, list[AccuracyMetric]]:
    rows = load_case_rows(csv_path)
    return {
        "isco": isco_accuracy(rows),
        "isic": isic_accuracy(rows),
        "isced": isced_accuracy(rows),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_json(results: dict[str, list[AccuracyMetric]], path: Path) -> None:
    payload = {group: [asdict(m) for m in metrics] for group, metrics in results.items()}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_markdown(results: dict[str, list[AccuracyMetric]], path: Path) -> None:
    lines = ["# Accuracy Analysis", "", "| Group | Metric | n | Accuracy | 95% CI | Status |", "|---|---|---|---|---|---|"]
    for group, metrics in results.items():
        for m in metrics:
            acc = f"{m.accuracy:.4f}" if m.accuracy is not None else "—"
            ci = f"[{m.ci_lo:.4f}, {m.ci_hi:.4f}]" if m.ci_lo is not None else "—"
            status = m.status if m.status == "measured" else f"not measured ({m.reason})"
            lines.append(f"| {group} | {m.metric_name} | {m.n} | {acc} | {ci} | {status} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-csv", required=True, type=Path, help="A CaseResult CSV from eval/run_eval.py")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if not args.case_csv.exists():
        parser.error(f"CSV not found: {args.case_csv}")

    args.out.mkdir(parents=True, exist_ok=True)
    results = analyze_csv(args.case_csv)
    write_json(results, args.out / f"{args.case_csv.stem}_accuracy.json")
    write_markdown(results, args.out / f"{args.case_csv.stem}_accuracy.md")
    print(f"Wrote accuracy analysis for {args.case_csv} to {args.out}")


if __name__ == "__main__":
    main()
