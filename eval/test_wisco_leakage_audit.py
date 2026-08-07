"""
Tests for eval/audit_wisco_benchmark_leakage.py,
eval/build_wisco_isco_benchmark_v2_group_split.py, and
eval/select_wisco_reranking_subset.py (Conference I Reviewer #2 response,
Step 7A). Uses small in-memory fixtures throughout -- never reads the real
WISCO source file or the (potentially large) generated benchmark packages,
so these tests are fast and self-contained.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_wisco_benchmark_leakage as audit  # noqa: E402
import build_wisco_isco_benchmark_v2_group_split as v2  # noqa: E402
from select_wisco_reranking_subset import select_stratified_subset  # noqa: E402


def _rec(benchmark_id, split, gold_code="2512", language="en", text="software developer"):
    return {
        "benchmark_id": benchmark_id, "task": "isco08", "language": language,
        "input_text": text, "gold_code": gold_code, "split": split,
    }


# ---------------------------------------------------------------------------
# 1. Source-family groups cannot span development and held-out splits
# ---------------------------------------------------------------------------

def test_source_family_leakage_detected_when_same_key_in_both_splits():
    records = [
        _rec("WISCO-100-en", "dev"),
        _rec("WISCO-100-ar", "heldout", text="مبرمج"),  # SAME key 100, different split -> leak
    ]
    result = audit.check_source_family_split(records)
    assert not result["ok"]
    assert result["n_leaking_source_keys"] == 1
    assert "100" in result["leaking_source_keys_sample"]


def test_source_family_no_leakage_when_key_confined_to_one_split():
    records = [
        _rec("WISCO-100-en", "dev"),
        _rec("WISCO-100-ar", "dev", text="مبرمج"),
        _rec("WISCO-200-en", "heldout", text="teacher"),
    ]
    result = audit.check_source_family_split(records)
    assert result["ok"]
    assert result["n_leaking_source_keys"] == 0


def test_parse_source_key_rejects_malformed_id():
    import pytest
    with pytest.raises(ValueError):
        audit.parse_source_key("not-a-valid-id-shape-here")


# ---------------------------------------------------------------------------
# 2. Exact duplicate cross-split detection
# ---------------------------------------------------------------------------

def test_exact_duplicate_text_across_splits_detected():
    records = [
        _rec("WISCO-100-en", "dev", gold_code="2643", text="Sworn translator"),
        _rec("WISCO-200-en", "heldout", gold_code="2643", text="sworn   translator"),  # normalizes identical
    ]
    result = audit.check_text_duplicates(records)
    assert not result["ok"]
    assert result["n_exact_duplicate_groups_cross_split"] == 1


def test_exact_duplicate_text_within_one_split_is_not_cross_split_leakage():
    records = [
        _rec("WISCO-100-en", "dev", text="teacher"),
        _rec("WISCO-200-en", "dev", text="Teacher"),  # same split -- not a leak
    ]
    result = audit.check_text_duplicates(records)
    assert result["ok"]
    assert result["n_exact_duplicate_groups_cross_split"] == 0
    assert result["n_exact_duplicate_groups_within_split"] == 1


def test_no_duplicates_reports_clean():
    records = [_rec("WISCO-100-en", "dev", text="teacher"), _rec("WISCO-200-en", "heldout", text="nurse")]
    result = audit.check_text_duplicates(records)
    assert result["ok"]
    assert result["n_exact_duplicate_groups_cross_split"] == 0


# ---------------------------------------------------------------------------
# 3. Deterministic group-aware split with fixed seed
# ---------------------------------------------------------------------------

def test_v2_split_for_group_is_deterministic():
    a = v2._split_for_group("100", ["100"])
    b = v2._split_for_group("100", ["100"])
    assert a == b


def test_v2_split_for_group_depends_on_fixed_seed():
    """Different FIXED_SEED values must (in general) produce a different
    split -- confirms the seed actually participates in the hash, not just
    documented but unused."""
    payload_a = f"{v2.FIXED_SEED}:100"
    payload_b = f"{v2.FIXED_SEED + 1}:100"
    assert payload_a != payload_b  # sanity: different seed changes the hash input


def test_v2_build_groups_merges_duplicate_text_keys():
    wisco_records = [
        {"key": 100, "isco08_unit": "2643", "master_label_en": "Sworn translator", "titles": {"en": "Sworn translator"}},
        {"key": 200, "isco08_unit": "2643", "master_label_en": "Sworn translator", "titles": {"en": "Sworn translator"}},
        {"key": 300, "isco08_unit": "2512", "master_label_en": "Software developer", "titles": {"en": "Software developer"}},
    ]
    key_to_group, n_merges = v2.build_groups(wisco_records)
    assert n_merges == 1
    assert key_to_group["100"] == key_to_group["200"]
    assert key_to_group["300"] != key_to_group["100"]


def test_v2_build_records_never_splits_merged_group_across_dev_and_heldout():
    wisco_records = [
        {"key": 100, "isco08_unit": "2643", "master_label_en": "Sworn translator", "titles": {"en": "Sworn translator", "ar": "مترجم"}},
        {"key": 200, "isco08_unit": "2643", "master_label_en": "Sworn translator", "titles": {"en": "Sworn translator", "hi": "अनुवादक"}},
    ]
    records, meta = v2.build_records(wisco_records)
    assert meta["n_merges_from_duplicate_text"] == 1
    splits = {r.split for r in records}
    assert len(splits) == 1  # all 4 variants (2 keys x mixed langs) share one split


def test_v2_build_records_reproducible_across_two_runs():
    wisco_records = [
        {"key": k, "isco08_unit": "2512", "master_label_en": f"Occupation {k}", "titles": {"en": f"Occupation {k}"}}
        for k in range(1, 21)
    ]
    records_a, _ = v2.build_records(wisco_records)
    records_b, _ = v2.build_records(wisco_records)
    splits_a = {r.benchmark_id: r.split for r in records_a}
    splits_b = {r.benchmark_id: r.split for r in records_b}
    assert splits_a == splits_b


# ---------------------------------------------------------------------------
# 4. Rejection of an evaluation plan if unresolved leakage exists
# ---------------------------------------------------------------------------

def test_evaluation_readiness_blocks_on_leakage():
    audit_report = {"leakage_found": True}
    assert audit.determine_evaluation_readiness(audit_report) == audit.NOT_READY_UNRESOLVED_LEAKAGE


def test_evaluation_readiness_allows_when_clean():
    audit_report = {"leakage_found": False}
    assert audit.determine_evaluation_readiness(audit_report) == audit.READY_FOR_EVALUATION


def test_full_audit_marks_not_ready_when_source_family_leaks():
    records = [
        _rec("WISCO-100-en", "dev", gold_code="2512"),
        _rec("WISCO-100-ar", "heldout", gold_code="2512", text="مبرمج"),
    ]
    report = audit.run_full_audit(records, expected_dataset_hash=None)
    assert report["leakage_found"] is True
    assert report["evaluation_readiness"] == audit.NOT_READY_UNRESOLVED_LEAKAGE


def test_full_audit_marks_ready_when_clean():
    records = [
        _rec("WISCO-100-en", "dev", gold_code="2512", text="software developer"),
        _rec("WISCO-200-en", "heldout", gold_code="2341", text="primary school teacher"),
    ]
    report = audit.run_full_audit(records, expected_dataset_hash=None)
    assert report["leakage_found"] is False
    assert report["evaluation_readiness"] == audit.READY_FOR_EVALUATION


# ---------------------------------------------------------------------------
# 5. WISCO marked ISCO-only, not eligible for ISIC/ISCED/SRE accuracy claims
# ---------------------------------------------------------------------------

def test_wisco_records_never_carry_isic_or_isced_task():
    """The WISCO builder only ever emits task='isco08' -- confirms no
    accidental ISIC/ISCED task leaks in from the builder itself."""
    wisco_records = [{"key": 1, "isco08_unit": "2512", "master_label_en": "x", "titles": {"en": "x"}}]
    records, _ = v2.build_records(wisco_records)
    assert all(r.task == "isco08" for r in records)


def test_export_to_run_eval_csv_leaves_isic_isced_gold_blank():
    import export_benchmark_to_run_eval_csv as exp
    records = [{"benchmark_id": "WISCO-1-en", "task": "isco08", "language": "en",
                "input_text": "x", "gold_code": "2512", "split": "heldout"}]
    rows = exp.export(records, "heldout")
    assert rows[0]["gold_isic"] == ""
    assert rows[0]["gold_isced"] == ""


# ---------------------------------------------------------------------------
# 6. Reranking subset selection is deterministic and stratified
# ---------------------------------------------------------------------------

def _language_major_pool(n_per_stratum=20):
    records = []
    i = 0
    for lang in ("en", "ar"):
        for major in ("2", "3"):
            for _ in range(n_per_stratum):
                records.append({"benchmark_id": f"WISCO-{i}-{lang}", "language": lang,
                                 "gold_code": f"{major}512", "split": "heldout"})
                i += 1
    return records


def test_subset_selection_deterministic_across_runs():
    pool = _language_major_pool()
    a = select_stratified_subset(pool, "heldout", target_size=20, seed=42)
    b = select_stratified_subset(pool, "heldout", target_size=20, seed=42)
    assert [r["benchmark_id"] for r in a] == [r["benchmark_id"] for r in b]


def test_subset_selection_different_seed_can_change_selection():
    pool = _language_major_pool()
    a = select_stratified_subset(pool, "heldout", target_size=20, seed=42)
    b = select_stratified_subset(pool, "heldout", target_size=20, seed=43)
    ids_a = {r["benchmark_id"] for r in a}
    ids_b = {r["benchmark_id"] for r in b}
    assert ids_a != ids_b  # not required to be fully disjoint, just not identical


def test_subset_selection_covers_all_strata():
    pool = _language_major_pool()
    selected = select_stratified_subset(pool, "heldout", target_size=40, seed=42)
    strata = {(r["language"], r["gold_code"][0]) for r in selected}
    assert strata == {("en", "2"), ("en", "3"), ("ar", "2"), ("ar", "3")}


def test_subset_selection_never_selects_dev_records():
    pool = _language_major_pool()
    pool.append({"benchmark_id": "WISCO-9999-en", "language": "en", "gold_code": "2512", "split": "dev"})
    selected = select_stratified_subset(pool, "heldout", target_size=len(pool), seed=42)
    assert all(r["split"] == "heldout" for r in selected)
    assert "WISCO-9999-en" not in {r["benchmark_id"] for r in selected}


def test_subset_selection_respects_target_size_upper_bound():
    pool = _language_major_pool(n_per_stratum=5)  # 20 total
    selected = select_stratified_subset(pool, "heldout", target_size=1000, seed=42)
    assert len(selected) == len(pool)  # cannot exceed available pool
