"""
eval/wisco_tier1_topk_kappa_offline_analysis.py

Offline analysis filling three gaps in the canonical published WISCO
Tier-1 result (Documentation/Conference_I_Reviewer_2/
OFFICIAL_WISCO_TIER1_CONTROLLED_RESULTS.md, Tasks 36/37/37.1): that
document has aggregate-only Wilson CIs, no top-3 accuracy, and no
Cohen's kappa. This script computes all three, per language and
aggregate, from the SAME already-collected raw per-case CSVs -- no new
live classification, no re-run, byte-identical inputs to the published
result.

Top-3: `gold_rank_in_pool`/`stage4_pool` (the B2 pool-capture columns)
are entirely empty in this run -- that instrumentation flag wasn't
enabled for Task 36. Top-3 is instead computed from `stage4_candidates`,
a separate, always-populated column holding the real top-5 ranked
candidate list (confirmed non-empty across all 18,747 rows in both
files) -- verified directly, not assumed.

Cohen's kappa: computed at the 1-digit major-group level
(gold_isco_1digit vs pred_isco_1digit), matching this project's existing
convention in eval/legacy_thesis_ch6/evaluate.py (sklearn.cohen_kappa_score).
Kappa's CI is bootstrapped (2000 resamples, seeded) since it has no
closed-form interval the way a proportion does.

Usage
-----
    python eval/wisco_tier1_topk_kappa_offline_analysis.py \
        --flat-csv <path> --hierarchical-csv <path> \
        --out Documentation/Conference_I_Reviewer_2/generated/wisco_tier1_topk_kappa.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import wilson_score_interval  # noqa: E402

from sklearn.metrics import cohen_kappa_score  # noqa: E402

_SEED = 42
_BOOTSTRAP_N = 2000


def _load_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _top3_hit(row: dict) -> bool:
    """True if gold is among the 3 highest-*score* distinct codes in
    stage4_candidates. IMPORTANT: list position is NOT score order for
    every system -- flat's stage4_candidates (5 entries) happens to
    already be pre-sorted by score descending, but hierarchical's
    (20-30+ entries, pooled across multiple beam-search branches) is
    grouped by branch traversal order, NOT globally sorted. Verified
    directly: for one hierarchical case, the correct top-1 prediction
    (matching pred_isco_4digit) had the single highest score in the
    pool but sat at list position 14 of 26. An earlier version of this
    function assumed position order and produced a logically impossible
    top3_correct < top1_correct on the hierarchical Hindi subset --
    caught by that monotonicity violation, not assumed correct."""
    gold = row["gold_isco_4digit"]
    raw = row.get("stage4_candidates", "").strip()
    if not raw or raw == "[]":
        return False
    try:
        candidates = json.loads(raw)
    except json.JSONDecodeError:
        return False
    best_score_by_code: dict[str, float] = {}
    for c in candidates:
        code = c.get("code", "")
        score = c.get("score", float("-inf"))
        if code and (code not in best_score_by_code or score > best_score_by_code[code]):
            best_score_by_code[code] = score
    ranked = sorted(best_score_by_code.items(), key=lambda kv: kv[1], reverse=True)
    top3_codes = [code for code, _ in ranked[:3]]
    return gold in top3_codes


def _bootstrap_kappa_ci(true_majors: list[str], pred_majors: list[str], n_resamples: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(true_majors)
    if n < 2:
        return (0.0, 0.0)
    samples = []
    idx_range = range(n)
    for _ in range(n_resamples):
        idxs = [rng.randrange(n) for _ in idx_range]
        t = [true_majors[i] for i in idxs]
        p = [pred_majors[i] for i in idxs]
        if len(set(t)) < 2:
            continue
        try:
            samples.append(cohen_kappa_score(t, p))
        except Exception:
            continue
    if not samples:
        return (0.0, 0.0)
    samples.sort()
    lo = samples[int(0.025 * len(samples))]
    hi = samples[min(len(samples) - 1, int(0.975 * len(samples)))]
    return (round(lo, 4), round(hi, 4))


def _metrics_for_group(rows: list[dict]) -> dict:
    n = len(rows)
    top1_hits = sum(1 for r in rows if r["pred_isco_4digit"] == r["gold_isco_4digit"])
    top3_hits = sum(1 for r in rows if _top3_hit(r))
    # top-k accuracy must be monotonically non-decreasing in k -- a
    # violation here means _top3_hit's ranking logic is wrong, not that
    # the model genuinely does worse at top-3 than top-1. Caught exactly
    # this way once already during development; kept as a permanent
    # regression guard rather than trusting the arithmetic silently.
    assert top3_hits >= top1_hits, (
        f"top3_hits ({top3_hits}) < top1_hits ({top1_hits}) -- impossible, "
        f"top-3 must include every top-1 hit; _top3_hit's ranking logic is wrong"
    )
    true_majors = [r["gold_isco_1digit"] for r in rows]
    pred_majors = [r["pred_isco_1digit"] for r in rows]

    if n >= 2 and len(set(true_majors)) > 1:
        kappa = float(cohen_kappa_score(true_majors, pred_majors))
    elif n >= 2:
        kappa = 1.0 if pred_majors == true_majors else 0.0
    else:
        kappa = 0.0

    top1_lo, top1_hi = wilson_score_interval(top1_hits, n) if n > 0 else (0.0, 0.0)
    top3_lo, top3_hi = wilson_score_interval(top3_hits, n) if n > 0 else (0.0, 0.0)
    kappa_lo, kappa_hi = _bootstrap_kappa_ci(true_majors, pred_majors, _BOOTSTRAP_N, _SEED)

    return {
        "n": n,
        "top1_correct": top1_hits,
        "top1_accuracy_pct": round(100 * top1_hits / n, 4) if n else None,
        "top1_wilson_95ci_pct": [round(100 * top1_lo, 4), round(100 * top1_hi, 4)],
        "top3_correct": top3_hits,
        "top3_accuracy_pct": round(100 * top3_hits / n, 4) if n else None,
        "top3_wilson_95ci_pct": [round(100 * top3_lo, 4), round(100 * top3_hi, 4)],
        "cohen_kappa_major_group": round(kappa, 4),
        "cohen_kappa_95ci_bootstrap": [kappa_lo, kappa_hi],
        "kappa_bootstrap_n": _BOOTSTRAP_N,
        "kappa_bootstrap_seed": _SEED,
    }


def analyze(rows: list[dict]) -> dict:
    by_lang = defaultdict(list)
    for r in rows:
        by_lang[r["input_language"]].append(r)

    result = {"aggregate": _metrics_for_group(rows), "by_language": {}}
    for lang in sorted(by_lang):
        result["by_language"][lang] = _metrics_for_group(by_lang[lang])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flat-csv", required=True)
    parser.add_argument("--hierarchical-csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    flat_rows = _load_rows(Path(args.flat_csv))
    hier_rows = _load_rows(Path(args.hierarchical_csv))

    assert len(flat_rows) == 18747, f"expected 18747 flat rows, got {len(flat_rows)}"
    assert len(hier_rows) == 18747, f"expected 18747 hierarchical rows, got {len(hier_rows)}"

    out = {
        "source_note": (
            "Offline analysis of Task 36's already-published raw CSVs "
            "(byte-identical inputs to OFFICIAL_WISCO_TIER1_CONTROLLED_RESULTS.md). "
            "top1 recomputed here for cross-check against the published 21.1927%/10.3537%."
        ),
        "flat": analyze(flat_rows),
        "hierarchical": analyze(hier_rows),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    print()
    print("=== AGGREGATE cross-check against published numbers ===")
    print(f"flat top1: {out['flat']['aggregate']['top1_accuracy_pct']}% (published: 21.1927%)")
    print(f"hierarchical top1: {out['hierarchical']['aggregate']['top1_accuracy_pct']}% (published: 10.3537%)")


if __name__ == "__main__":
    main()
