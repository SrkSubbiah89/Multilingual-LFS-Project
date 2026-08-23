"""
eval/build_wisco_isco_benchmark_v3_dev_validation_split.py

Thesis benchmark-design work (2026-08-24). Adds a genuine 3-way split
(dev / validation / heldout) on top of v2's audited, group-safe 2-way
split (eval/build_wisco_isco_benchmark_v2_group_split.py), so that
config selection (flat vs. hierarchical, e5-small vs. e5-large, reranker
thresholds, ...) has a real validation set instead of being decided by
eyeballing dev directly or, worse, by repeated looks at heldout.

Design, and why it's built this way
------------------------------------
- **heldout is untouched, by construction, not by promise.** This script
  imports v2's `build_groups()` and `_split_for_group()` verbatim (not
  reimplemented) to compute the exact same group->dev/heldout assignment
  v2 already produced and had leakage-audited. It NEVER recomputes or
  second-guesses that assignment for groups landing in heldout. A test
  in this file's companion test suite asserts the resulting heldout
  benchmark_id set is byte-identical to v2's own heldout set -- not
  approximately equal, identical -- as the load-bearing correctness
  check for this whole script.
- **Only dev is further split.** Every group v2 assigned to "dev" gets a
  SECOND, independent, fixed-seed hash decision (different seed/salt
  than the dev-vs-heldout decision, so the two splits are not
  correlated) sorting it into "dev" or "validation". Group membership is
  still respected -- a group's language variants never split across dev
  and validation, for the same leakage reason v2 established for
  dev/heldout.
- **Ratio: ~70% dev / ~30% validation of the existing ~2,013-case dev
  pool** (not the template-suggested 60/20/20 of the total -- this
  project's dev/heldout ratio (~10%/90%) is already fixed by v2's own
  hash and changing it would invalidate the already-used, already-
  leakage-audited heldout set). 70/30 was chosen, not defaulted to,
  to keep enough cases in dev for exploratory work while giving
  validation real statistical power (~30% of 2,013 is still ~600 cases
  -- far larger than most single comparisons run in this project so far,
  e.g. the 63-case reranker experiments).

Usage
-----
    python eval/build_wisco_isco_benchmark_v3_dev_validation_split.py \\
        --out-root eval/local_benchmarks/wisco_isco08_v3_dev_validation_split
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from controlled_benchmark_schema import (  # noqa: E402
    ADJUDICATION_UNKNOWN,
    BenchmarkRecord,
    DATA_SOURCE_EXTERNAL_AUTHORITATIVE,
    DOUBLE_CODING_UNKNOWN,
    INDEPENDENT_LABEL_STATUS_INDEPENDENT,
    ISCO08,
    LABEL_SOURCE_EXTERNAL_PUBLISHER,
    SPLIT_DEV,
    SPLIT_HELDOUT,
    SPLIT_VALIDATION,
)
from dataset_card_schema import DEFAULT_DATASET_LABEL  # noqa: E402
from validate_controlled_benchmark import validate_benchmark_package  # noqa: E402
from audit_wisco_benchmark_leakage import normalize_text, run_full_audit  # noqa: E402
from build_wisco_isco_benchmark import (  # noqa: E402
    CLASSIFICATION_VERSION,
    LABELER,
    TARGET_LANGUAGES,
    WISCO_PARSED_PATH,
)
from build_wisco_isco_benchmark_v2_group_split import (  # noqa: E402
    FIXED_SEED,
    build_groups,
    _split_for_group,
    detect_gold_code_text_conflicts,
    GOLD_CODE_TEXT_CONFLICT_REASON_TEMPLATE,
)

REPO_ROOT = _HERE.parent

# Independent of FIXED_SEED (v2's dev-vs-heldout decision) so the two
# splits are not correlated with each other.
VALIDATION_SUB_SPLIT_SEED = 4224
VALIDATION_FRACTION_DENOMINATOR = 10   # mod-10 thresholds below
VALIDATION_FRACTION_THRESHOLD = 3      # digest % 10 < 3 -> validation (~30%)


def _sub_split_for_group(group_root: str, member_keys: list[str]) -> str:
    """Only ever called for groups v2 already assigned to SPLIT_DEV.
    Independent fixed-seed hash -> 'dev' (~70%) or 'validation' (~30%)."""
    payload = f"{VALIDATION_SUB_SPLIT_SEED}:{','.join(sorted(member_keys))}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return SPLIT_VALIDATION if int(digest[:8], 16) % VALIDATION_FRACTION_DENOMINATOR < VALIDATION_FRACTION_THRESHOLD else SPLIT_DEV


def build_records(wisco_records: list[dict]) -> tuple[list[BenchmarkRecord], dict]:
    key_to_group, n_merges = build_groups(wisco_records)

    group_members: dict[str, list[str]] = defaultdict(list)
    for k, g in key_to_group.items():
        group_members[g].append(k)

    # Stage 1: reproduce v2's dev/heldout decision exactly, verbatim.
    primary_split = {g: _split_for_group(g, members) for g, members in group_members.items()}

    # Stage 2: sub-split ONLY groups v2 put in dev; heldout groups pass through untouched.
    final_split: dict[str, str] = {}
    for g, members in group_members.items():
        if primary_split[g] == SPLIT_HELDOUT:
            final_split[g] = SPLIT_HELDOUT
        else:
            final_split[g] = _sub_split_for_group(g, members)

    records: list[BenchmarkRecord] = []
    for occ in wisco_records:
        key = str(occ["key"])
        split = final_split[key_to_group[key]]
        gold_code = occ.get("isco08_unit", "")
        gold_title = occ.get("master_label_en", "")
        titles = occ.get("titles", {})
        for lang in TARGET_LANGUAGES:
            text = titles.get(lang)
            if not text:
                continue
            rec = BenchmarkRecord(
                benchmark_id=f"WISCO-{key}-{lang}",
                task=ISCO08,
                data_source_type=DATA_SOURCE_EXTERNAL_AUTHORITATIVE,
                dataset_label=DEFAULT_DATASET_LABEL,
                language=lang,
                input_text=text,
                context_fields=None,
                gold_code=gold_code,
                gold_code_title=gold_title,
                classification_standard="ISCO-08",
                classification_version=CLASSIFICATION_VERSION,
                hierarchy_level="unit_group_4digit",
                label_source_type=LABEL_SOURCE_EXTERNAL_PUBLISHER,
                labeler_identifier_or_role=LABELER,
                independent_label_status=INDEPENDENT_LABEL_STATUS_INDEPENDENT,
                double_coding_status=DOUBLE_CODING_UNKNOWN,
                adjudication_status=ADJUDICATION_UNKNOWN,
                ambiguity_flag=False,
                exclusion_reason=None,
                split=split,
            )
            records.append(rec)

    conflicts = detect_gold_code_text_conflicts(records)
    for rec in records:
        conflict = conflicts.get(rec.benchmark_id)
        if conflict is None:
            continue
        other_keys, other_codes = conflict
        rec.ambiguity_flag = True
        rec.exclusion_reason = GOLD_CODE_TEXT_CONFLICT_REASON_TEMPLATE.format(
            n_other=len(other_keys),
            other_keys=", ".join(other_keys),
            other_codes=", ".join(other_codes),
        )

    for rec in records:
        payload = rec.model_dump_json(exclude={"record_hash"}, exclude_none=False).encode("utf-8")
        rec.record_hash = hashlib.sha256(payload).hexdigest()

    n_dev_groups = sum(1 for g in group_members if primary_split[g] == SPLIT_DEV)
    n_validation_groups = sum(1 for g in group_members if final_split[g] == SPLIT_VALIDATION)
    meta = {
        "fixed_seed_dev_vs_heldout": FIXED_SEED,
        "fixed_seed_dev_vs_validation": VALIDATION_SUB_SPLIT_SEED,
        "n_source_keys": len(key_to_group),
        "n_groups": len(group_members),
        "n_merges_from_duplicate_text": n_merges,
        "n_dev_pool_groups_before_sub_split": n_dev_groups,
        "n_validation_groups": n_validation_groups,
        "n_records_flagged_gold_code_text_conflict": len(conflicts),
    }
    return records, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--wisco-parsed", type=Path, default=WISCO_PARSED_PATH)
    parser.add_argument(
        "--v2-records", type=Path,
        default=REPO_ROOT / "eval" / "local_benchmarks" / "wisco_isco08_v2_group_split" / "records.json",
        help="v2's own output, used only for the byte-identical-heldout consistency check.",
    )
    args = parser.parse_args()

    if not args.wisco_parsed.exists():
        parser.error(f"WISCO parsed file not found: {args.wisco_parsed}")

    wisco_records = json.loads(args.wisco_parsed.read_text(encoding="utf-8"))
    records, group_meta = build_records(wisco_records)

    args.out_root.mkdir(parents=True, exist_ok=True)

    records_payload = [json.loads(r.model_dump_json()) for r in records]
    dataset_hash = hashlib.sha256(
        json.dumps(records_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    validation_report = validate_benchmark_package(records, dataset_hash=dataset_hash)
    leakage_report = run_full_audit(records_payload, expected_dataset_hash=dataset_hash)

    # Load-bearing correctness check: heldout must be byte-identical to v2's.
    heldout_ids_v3 = {r.benchmark_id for r in records if r.split == SPLIT_HELDOUT}
    heldout_consistency = {"checked": False, "identical_to_v2": None, "n_v3": len(heldout_ids_v3)}
    if args.v2_records.exists():
        v2_data = json.loads(args.v2_records.read_text(encoding="utf-8"))
        heldout_ids_v2 = {r["benchmark_id"] for r in v2_data["records"] if r["split"] == SPLIT_HELDOUT}
        heldout_consistency = {
            "checked": True,
            "identical_to_v2": heldout_ids_v3 == heldout_ids_v2,
            "n_v3": len(heldout_ids_v3),
            "n_v2": len(heldout_ids_v2),
            "symmetric_difference_count": len(heldout_ids_v3 ^ heldout_ids_v2),
        }

    (args.out_root / "records.json").write_text(
        json.dumps({"records": records_payload}, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    (args.out_root / "dataset_hash.txt").write_text(dataset_hash + "\n", encoding="utf-8")

    dev_ids = [r.benchmark_id for r in records if r.split == SPLIT_DEV]
    validation_ids = [r.benchmark_id for r in records if r.split == SPLIT_VALIDATION]
    heldout_ids = sorted(heldout_ids_v3)
    split_manifest = {
        "dataset_id": "WISCO-ISCO08-BENCHMARK-v3-dev-validation-split",
        "split_hash": hashlib.sha256(
            (",".join(sorted(dev_ids)) + "|" + ",".join(sorted(validation_ids)) + "|" + ",".join(heldout_ids)).encode("utf-8")
        ).hexdigest(),
        "splits": {
            "dev": {"split_id": f"WISCO-ISCO08-V3-DEV-{len(dev_ids)}", "purpose": "exploratory parameter selection", "n_cases": len(dev_ids), "frozen": False},
            "validation": {"split_id": f"WISCO-ISCO08-V3-VALIDATION-{len(validation_ids)}", "purpose": "final config selection before one confirmatory heldout reading", "n_cases": len(validation_ids), "frozen": False},
            "heldout": {"split_id": f"WISCO-ISCO08-V2-HELDOUT-{len(heldout_ids)}", "purpose": "frozen confirmation set -- identical to v2, never resplit", "n_cases": len(heldout_ids), "frozen": True},
        },
        "heldout_consistency_check_vs_v2": heldout_consistency,
        "leakage_check_performed": True,
        "leakage_check_method": (
            "group-aware, identical grouping to v2 (union-find over WISCO occupation "
            "keys sharing byte-identical normalized title text in any target language). "
            "dev-vs-heldout decision reused verbatim from v2 (fixed seed 42). Only groups "
            "v2 assigned to dev are further sub-split into dev/validation via an "
            f"independent fixed seed ({VALIDATION_SUB_SPLIT_SEED}); heldout groups are "
            "never touched by the sub-split."
        ),
    }
    (args.out_root / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    lang_counts = Counter(r.language for r in records)
    split_counts = Counter(r.split for r in records)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "supersedes_for_split_purposes": "nothing -- this is additive; v2's own dev/heldout files are untouched and remain valid",
        "reason_for_v3": (
            "Add a genuine validation split for config selection (flat vs. hierarchical, "
            "e5-small vs. e5-large, reranker thresholds, ...), separate from the frozen "
            "heldout confirmation set -- see Documentation/PROJECT_FLOW_AND_STATUS.md section 11.1."
        ),
        "source_file": str(args.wisco_parsed.relative_to(REPO_ROOT)),
        "source_file_sha256": hashlib.sha256(args.wisco_parsed.read_bytes()).hexdigest(),
        "n_source_occupations": len(wisco_records),
        "n_benchmark_records": len(records),
        "language_counts": dict(lang_counts),
        "split_counts": dict(split_counts),
        "dataset_hash": dataset_hash,
        "grouping": group_meta,
        "heldout_consistency_check_vs_v2": heldout_consistency,
        "validation_report": {"ok": validation_report.ok, "errors": validation_report.errors, "warnings": validation_report.warnings},
        "leakage_audit_summary": {
            "leakage_found": leakage_report["leakage_found"],
            "overall_ok_no_leakage": leakage_report["overall_ok_no_leakage"],
            "n_leaking_source_keys": leakage_report["source_family_split_check"]["n_leaking_source_keys"],
            "n_exact_duplicate_groups_cross_split": leakage_report["text_duplicate_check"]["n_exact_duplicate_groups_cross_split"],
        },
    }
    (args.out_root / "build_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(records)} benchmark records ({dict(split_counts)}) to {args.out_root}")
    print(f"dataset_hash={dataset_hash}")
    print(f"group_meta={group_meta}")
    print(f"heldout_consistency_check_vs_v2={heldout_consistency}")
    print(f"validation: ok={validation_report.ok} errors={len(validation_report.errors)}")
    print(f"leakage audit: leakage_found={leakage_report['leakage_found']} overall_ok={leakage_report['overall_ok_no_leakage']}")


if __name__ == "__main__":
    main()
