"""
eval/select_wisco_reranking_subset.py

Conference I Reviewer #2 response, Step 7A, Phase D.2: deterministic,
stratified selection of a fixed subset of the (group-aware, leakage-audited)
WISCO heldout split for the "paired reranking" evaluation tier -- the tier
that needs LLM reranking and is therefore too slow/costly to run over the
full ~18.7k heldout records (see
Documentation/Conference_I_Reviewer_2/WISCO_LEAKAGE_AUDIT_AND_RUN_PLAN.md
Phase E for the time/resource estimate this subset size is based on).

Selection is stratified by (language, ISCO-08 major group) and made
BEFORE any system result is viewed -- selection depends only on
benchmark_id (a stable identifier) and a fixed seed, never on any
prediction, confidence, or accuracy figure. This is DATA SELECTION ONLY:
no classifier/LLM/network call is made by this script.

Usage
-----
    python eval/select_wisco_reranking_subset.py \\
        --records eval/local_benchmarks/wisco_isco08_v2_group_split/records.json \\
        --split heldout --target-size 500 --seed 42 \\
        --out eval/local_benchmarks/wisco_isco08_v2_group_split/reranking_subset_500.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


def _rank_key(seed: int, benchmark_id: str) -> str:
    return hashlib.sha256(f"{seed}:{benchmark_id}".encode("utf-8")).hexdigest()


def select_stratified_subset(records: list[dict], split: str, target_size: int, seed: int) -> list[dict]:
    pool = [r for r in records if r["split"] == split and r.get("gold_code")]
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in pool:
        major = r["gold_code"][0]
        strata[(r["language"], major)].append(r)

    for members in strata.values():
        members.sort(key=lambda r: _rank_key(seed, r["benchmark_id"]))

    total = len(pool)
    if total == 0:
        return []

    # Largest-remainder proportional allocation -- deterministic, no
    # randomness beyond the seeded rank ordering already applied above.
    raw_allocations = {k: (len(v) / total) * target_size for k, v in strata.items()}
    allocations = {k: int(v) for k, v in raw_allocations.items()}
    remaining = target_size - sum(allocations.values())
    remainders = sorted(strata.keys(), key=lambda k: -(raw_allocations[k] - allocations[k]))
    for k in remainders[:max(0, remaining)]:
        allocations[k] += 1

    selected: list[dict] = []
    for k, members in strata.items():
        n = min(allocations.get(k, 0), len(members))
        selected.extend(members[:n])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--split", default="heldout")
    parser.add_argument("--target-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.records.read_text(encoding="utf-8"))
    selected = select_stratified_subset(data["records"], args.split, args.target_size, args.seed)

    lang_counts = Counter(r["language"] for r in selected)
    major_counts = Counter(r["gold_code"][0] for r in selected)

    out = {
        "selection_method": "stratified by (language, ISCO-08 major group), deterministic sha256(seed:benchmark_id) ranking, largest-remainder proportional allocation",
        "seed": args.seed,
        "source_split": args.split,
        "target_size": args.target_size,
        "actual_size": len(selected),
        "language_distribution": dict(lang_counts),
        "isco_major_group_distribution": dict(sorted(major_counts.items())),
        "benchmark_ids": sorted(r["benchmark_id"] for r in selected),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Selected {len(selected)} records -> {args.out}")
    print(f"language_distribution={dict(lang_counts)}")


if __name__ == "__main__":
    main()
