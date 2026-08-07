"""
Tests for eval/validate_controlled_benchmark.py and
eval/controlled_benchmark_schema.py (Conference I Reviewer #2 response,
Step 6, Phase F). Uses eval/fixtures/controlled_benchmark_synthetic_example/
synthetic_benchmark_records.json (6 records: 5 multilingual valid + 1
ambiguous/excluded) as the base "eligible" fixture, mutated per test to
exercise each rejection path required by the task specification.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from controlled_benchmark_schema import BenchmarkRecord  # noqa: E402
from validate_controlled_benchmark import validate_benchmark_package  # noqa: E402
from dataset_card_schema import APPROVED_REAL_LFS_VALIDATION  # noqa: E402

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "controlled_benchmark_synthetic_example" / "synthetic_benchmark_records.json"


def load_fixture_records() -> list[dict]:
    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return data["records"]


def make_records(dicts: list[dict]) -> list[BenchmarkRecord]:
    return [BenchmarkRecord(**d) for d in dicts]


# ---------------------------------------------------------------------------
# 1. Eligible benchmark
# ---------------------------------------------------------------------------

def test_eligible_benchmark_passes():
    records = make_records(load_fixture_records())
    report = validate_benchmark_package(records, dataset_hash="deadbeef")
    assert report.ok, report.errors


def test_eligible_benchmark_covers_five_languages():
    records = make_records(load_fixture_records())
    languages = {r.language for r in records}
    assert languages == {"en", "ar", "ur", "hi", "tl"}


# ---------------------------------------------------------------------------
# 2. Missing label provenance
# ---------------------------------------------------------------------------

def test_missing_label_provenance_unknown_source_rejected():
    dicts = load_fixture_records()
    dicts[0]["label_source_type"] = "unknown"
    records = make_records(dicts)
    report = validate_benchmark_package(records, dataset_hash="deadbeef")
    assert not report.ok
    assert any("provenance is absent" in e for e in report.errors)


def test_missing_label_provenance_blank_labeler_rejected():
    dicts = load_fixture_records()
    dicts[0]["labeler_identifier_or_role"] = ""
    records = make_records(dicts)
    report = validate_benchmark_package(records, dataset_hash="deadbeef")
    assert not report.ok
    assert any("labeler_identifier_or_role is blank" in e for e in report.errors)


# ---------------------------------------------------------------------------
# 3. Missing split discipline
# ---------------------------------------------------------------------------

def test_dev_heldout_overlap_rejected():
    dicts = load_fixture_records()
    # Force the dev record and a heldout record to share a benchmark_id.
    dicts[0]["split"] = "dev"
    dicts[2]["split"] = "heldout"
    dicts[2]["benchmark_id"] = dicts[0]["benchmark_id"]
    records = make_records(dicts)
    report = validate_benchmark_package(records, dataset_hash="deadbeef")
    assert not report.ok
    assert any("overlap" in e for e in report.errors)


def test_duplicate_benchmark_id_rejected():
    dicts = load_fixture_records()
    dicts[1]["benchmark_id"] = dicts[0]["benchmark_id"]
    records = make_records(dicts)
    report = validate_benchmark_package(records, dataset_hash="deadbeef")
    assert not report.ok
    assert any("duplicate benchmark_id" in e for e in report.errors)


def test_missing_dataset_hash_rejected():
    records = make_records(load_fixture_records())
    report = validate_benchmark_package(records, dataset_hash=None)
    assert not report.ok
    assert any("dataset_hash is missing" in e for e in report.errors)


def test_missing_record_hash_rejected():
    dicts = load_fixture_records()
    dicts[0]["record_hash"] = ""
    records = make_records(dicts)
    report = validate_benchmark_package(records, dataset_hash="deadbeef")
    assert not report.ok
    assert any("record_hash is missing" in e for e in report.errors)


# ---------------------------------------------------------------------------
# 4. Self-generated labels
# ---------------------------------------------------------------------------

def test_system_self_generated_labels_rejected():
    dicts = load_fixture_records()
    dicts[0]["label_source_type"] = "system_self_generated"
    records = make_records(dicts)
    report = validate_benchmark_package(records, dataset_hash="deadbeef")
    assert not report.ok
    assert any("system under evaluation cannot be its own gold-label source" in e for e in report.errors)


def test_not_independent_label_status_rejected():
    dicts = load_fixture_records()
    dicts[0]["independent_label_status"] = "not_independent"
    records = make_records(dicts)
    report = validate_benchmark_package(records, dataset_hash="deadbeef")
    assert not report.ok
    assert any("not produced independently" in e for e in report.errors)


# ---------------------------------------------------------------------------
# 5. Ambiguous labels
# ---------------------------------------------------------------------------

def test_ambiguous_record_without_exclusion_reason_rejected():
    dicts = load_fixture_records()
    ambiguous = next(d for d in dicts if d["ambiguity_flag"])
    ambiguous["exclusion_reason"] = None
    records = make_records(dicts)
    report = validate_benchmark_package(records, dataset_hash="deadbeef")
    assert not report.ok
    assert any("exclusion_reason is blank" in e and "ambiguous" in e.lower() for e in report.errors)


def test_ambiguous_record_with_exclusion_reason_and_blank_gold_is_accepted():
    """The fixture's ambiguous case has a blank gold_code but a populated
    exclusion_reason -- this must NOT be rejected as 'gold labels absent',
    because forcing a false gold code would be worse than excluding it."""
    records = make_records(load_fixture_records())
    report = validate_benchmark_package(records, dataset_hash="deadbeef")
    assert not any("gold_code is blank" in e for e in report.errors)


def test_non_ambiguous_blank_gold_code_rejected():
    dicts = load_fixture_records()
    dicts[0]["gold_code"] = ""
    dicts[0]["ambiguity_flag"] = False
    records = make_records(dicts)
    report = validate_benchmark_package(records, dataset_hash="deadbeef")
    assert not report.ok
    assert any("gold_code is blank" in e for e in report.errors)


# ---------------------------------------------------------------------------
# 6. Sensitive-data rejection (a benchmark can never claim real-LFS status)
# ---------------------------------------------------------------------------

def test_approved_real_lfs_label_on_benchmark_always_rejected():
    dicts = load_fixture_records()
    dicts[0]["dataset_label"] = APPROVED_REAL_LFS_VALIDATION
    records = make_records(dicts)
    report = validate_benchmark_package(records, dataset_hash="deadbeef", dataset_card=None)
    assert not report.ok
    assert any("must never claim dataset_label=approved_real_lfs_validation" in e for e in report.errors)


def test_error_messages_never_echo_input_text_or_gold_code():
    """A validation report must be safe to print/log without leaking
    dataset content -- only IDs, field names, and counts."""
    dicts = load_fixture_records()
    dicts[0]["gold_code"] = ""
    dicts[0]["ambiguity_flag"] = False
    records = make_records(dicts)
    report = validate_benchmark_package(records, dataset_hash="deadbeef")
    all_text = " ".join(report.errors + report.warnings)
    for r in records:
        assert r.input_text not in all_text
        if r.gold_code:
            assert r.gold_code not in all_text or r.gold_code in ("",)


# ---------------------------------------------------------------------------
# 7. Multilingual benchmark records (positive coverage, redundant with §1
# but scoped exactly to the task's required test list)
# ---------------------------------------------------------------------------

def test_multilingual_records_each_pass_individually():
    """Each language's record(s), validated alone, must never fail on
    provenance/label-eligibility grounds -- only a single-split subset's
    inevitable 'no dev/heldout coverage' warning is expected."""
    records = make_records(load_fixture_records())
    for lang in ("en", "ar", "ur", "hi", "tl"):
        subset = [r for r in records if r.language == lang]
        assert subset, f"fixture missing a record for language {lang!r}"
        report = validate_benchmark_package(subset, dataset_hash="deadbeef")
        assert report.ok, (lang, report.errors)
