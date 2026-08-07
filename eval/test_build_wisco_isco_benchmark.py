"""
Tests for eval/build_wisco_isco_benchmark.py and
eval/export_benchmark_to_run_eval_csv.py (Conference I Reviewer #2 response,
Step 6, Phase D/G). Uses small in-memory fixtures -- never reads the real
WISCO source file, so these tests are fast and don't depend on
backend/evaluation/wisco/ being present in a stripped-down checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_wisco_isco_benchmark as bwb  # noqa: E402
import export_benchmark_to_run_eval_csv as exp  # noqa: E402
from controlled_benchmark_schema import SPLIT_DEV, SPLIT_HELDOUT  # noqa: E402


def _fake_occ(key, unit="2512", label="Software developer", langs=None):
    langs = langs or {"en": "Software developer", "ar": "مبرمج"}
    return {"key": key, "isco08_unit": unit, "master_label_en": label, "titles": langs}


# ---------------------------------------------------------------------------
# _split_for_key: deterministic, per-occupation (not per-language)
# ---------------------------------------------------------------------------

def test_split_for_key_deterministic():
    assert bwb._split_for_key(12345) == bwb._split_for_key(12345)


def test_split_for_key_only_dev_or_heldout():
    for k in range(50):
        assert bwb._split_for_key(k) in (SPLIT_DEV, SPLIT_HELDOUT)


def test_split_for_key_produces_both_splits_across_a_range():
    splits = {bwb._split_for_key(k) for k in range(200)}
    assert splits == {SPLIT_DEV, SPLIT_HELDOUT}


# ---------------------------------------------------------------------------
# build_records: one record per (occupation, language); same split per occ
# ---------------------------------------------------------------------------

def test_build_records_one_per_occupation_language_pair():
    occs = [_fake_occ(1, langs={"en": "Teacher", "ar": "معلم"})]
    records = bwb.build_records(occs)
    assert len(records) == 2
    assert {r.language for r in records} == {"en", "ar"}


def test_build_records_same_occupation_shares_split_across_languages():
    occs = [_fake_occ(42, langs={"en": "Nurse", "ar": "ممرضة", "ur": "نرس", "hi": "नर्स", "tl": "Nars"})]
    records = bwb.build_records(occs)
    assert len({r.split for r in records}) == 1


def test_build_records_missing_language_skipped_not_fabricated():
    occs = [_fake_occ(7, langs={"en": "Chef"})]  # only English present
    records = bwb.build_records(occs)
    assert len(records) == 1
    assert records[0].language == "en"


def test_build_records_gold_code_and_title_from_source():
    occs = [_fake_occ(9, unit="2512", label="Software developer", langs={"en": "Software developer"})]
    records = bwb.build_records(occs)
    assert records[0].gold_code == "2512"
    assert records[0].gold_code_title == "Software developer"


def test_build_records_never_labelled_approved_real_lfs():
    occs = [_fake_occ(1)]
    records = bwb.build_records(occs)
    assert all(r.dataset_label == "synthetic_or_operationally_realistic" for r in records)


def test_build_records_record_hash_populated():
    occs = [_fake_occ(1)]
    records = bwb.build_records(occs)
    assert all(r.record_hash for r in records)


# ---------------------------------------------------------------------------
# export_benchmark_to_run_eval_csv.export(): reformatting only
# ---------------------------------------------------------------------------

def _fake_benchmark_record(benchmark_id="X-en", split="heldout", task="isco08", gold="2512"):
    return {
        "benchmark_id": benchmark_id, "task": task, "language": "en",
        "input_text": "Software developer", "gold_code": gold, "split": split,
    }


def test_export_filters_by_split():
    records = [_fake_benchmark_record(split="dev"), _fake_benchmark_record(benchmark_id="Y-en", split="heldout")]
    rows = exp.export(records, "heldout")
    assert len(rows) == 1
    assert rows[0]["case_id"] == "Y-en"


def test_export_leaves_isic_isced_blank():
    records = [_fake_benchmark_record()]
    rows = exp.export(records, "heldout")
    assert rows[0]["gold_isic"] == ""
    assert rows[0]["gold_isced"] == ""


def test_export_skips_non_isco_task():
    records = [_fake_benchmark_record(task="isic_rev4")]
    rows = exp.export(records, "heldout")
    assert rows == []
