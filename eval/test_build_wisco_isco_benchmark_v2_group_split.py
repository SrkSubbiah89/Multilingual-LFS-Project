"""
Tests for eval/build_wisco_isco_benchmark_v2_group_split.py's gold-code
text-conflict detection (added after a manual audit found that WISCO
source keys 3240000400018 and 5164140000000 both carry the literal English
text "Veterinary assistant" but disagree on gold_code -- see
Documentation/Conference_I_Reviewer_2/WISCO_GOLD_LABEL_AMBIGUITY_AUDIT.md).

Uses small in-memory fixtures -- never reads the real WISCO source file.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_wisco_isco_benchmark_v2_group_split as bv2  # noqa: E402
from validate_controlled_benchmark import validate_benchmark_package  # noqa: E402


def _fake_occ(key, unit="2512", label="Software developer", langs=None):
    langs = langs if langs is not None else {"en": "Software developer", "ar": "مبرمج"}
    return {"key": key, "isco08_unit": unit, "master_label_en": label, "titles": langs}


# ---------------------------------------------------------------------------
# detect_gold_code_text_conflicts
# ---------------------------------------------------------------------------

def test_no_conflict_when_all_texts_distinct():
    wisco = [_fake_occ("K1", unit="2512"), _fake_occ("K2", unit="2513", langs={"en": "Data scientist"})]
    records, _meta = bv2.build_records(wisco)
    conflicts = bv2.detect_gold_code_text_conflicts(records)
    assert conflicts == {}
    assert all(not r.ambiguity_flag for r in records)


def test_no_conflict_when_duplicate_text_agrees_on_gold_code():
    """Same text, same code, two different keys -- a genuine repeated title,
    not an ambiguity. Must NOT be flagged."""
    wisco = [
        _fake_occ("K1", unit="2512", langs={"en": "Software developer"}),
        _fake_occ("K2", unit="2512", langs={"en": "Software developer"}),
    ]
    records, meta = bv2.build_records(wisco)
    assert all(not r.ambiguity_flag for r in records)
    assert meta["n_records_flagged_gold_code_text_conflict"] == 0


def test_conflict_flagged_when_duplicate_text_disagrees_on_gold_code():
    wisco = [
        _fake_occ("3240000400018", unit="3240", langs={"en": "Veterinary assistant", "ar": "مساعد بيطري"}),
        _fake_occ("5164140000000", unit="5164", langs={"en": "Veterinary assistant", "ar": "المساعدات البيطرية"}),
    ]
    records, meta = bv2.build_records(wisco)
    by_id = {r.benchmark_id: r for r in records}

    flagged = by_id["WISCO-3240000400018-en"]
    other_flagged = by_id["WISCO-5164140000000-en"]
    assert flagged.ambiguity_flag is True
    assert other_flagged.ambiguity_flag is True
    assert flagged.exclusion_reason and "gold-code text conflict" in flagged.exclusion_reason
    assert "5164140000000" in flagged.exclusion_reason
    assert other_flagged.exclusion_reason and "3240000400018" in other_flagged.exclusion_reason

    # gold_code is preserved verbatim (never blanked/fabricated)
    assert flagged.gold_code == "3240"
    assert other_flagged.gold_code == "5164"

    assert meta["n_records_flagged_gold_code_text_conflict"] == 2


def test_conflict_flag_only_applies_to_the_colliding_language_not_other_languages():
    """Only 'en' collides here; 'ar' texts for the two keys are distinct and
    must remain unflagged."""
    wisco = [
        _fake_occ("3240000400018", unit="3240", langs={"en": "Veterinary assistant", "ar": "مساعد بيطري"}),
        _fake_occ("5164140000000", unit="5164", langs={"en": "Veterinary assistant", "ar": "المساعدات البيطرية"}),
    ]
    records, _meta = bv2.build_records(wisco)
    by_id = {r.benchmark_id: r for r in records}

    assert by_id["WISCO-3240000400018-en"].ambiguity_flag is True
    assert by_id["WISCO-5164140000000-en"].ambiguity_flag is True
    assert by_id["WISCO-3240000400018-ar"].ambiguity_flag is False
    assert by_id["WISCO-5164140000000-ar"].ambiguity_flag is False
    assert by_id["WISCO-3240000400018-ar"].exclusion_reason is None
    assert by_id["WISCO-5164140000000-ar"].exclusion_reason is None


def test_three_way_conflict_lists_both_other_keys():
    wisco = [
        _fake_occ("K1", unit="1111", langs={"en": "Ambiguous title"}),
        _fake_occ("K2", unit="2222", langs={"en": "Ambiguous title"}),
        _fake_occ("K3", unit="3333", langs={"en": "Ambiguous title"}),
    ]
    records, meta = bv2.build_records(wisco)
    by_id = {r.benchmark_id: r for r in records}
    for k in ("K1", "K2", "K3"):
        rec = by_id[f"WISCO-{k}-en"]
        assert rec.ambiguity_flag is True
        others = {"K1", "K2", "K3"} - {k}
        for other in others:
            assert other in rec.exclusion_reason
    assert meta["n_records_flagged_gold_code_text_conflict"] == 3


def test_conflict_detection_case_and_whitespace_insensitive():
    """Mirrors audit_wisco_benchmark_leakage.normalize_text's own
    whitespace-collapse + lowercase behavior used for split-safety
    grouping -- gold-code conflict detection must use the identical
    normalization so it never misses a collision the leakage audit
    would have caught."""
    wisco = [
        _fake_occ("K1", unit="1111", langs={"en": "  Veterinary   Assistant "}),
        _fake_occ("K2", unit="2222", langs={"en": "veterinary assistant"}),
    ]
    records, _meta = bv2.build_records(wisco)
    by_id = {r.benchmark_id: r for r in records}
    assert by_id["WISCO-K1-en"].ambiguity_flag is True
    assert by_id["WISCO-K2-en"].ambiguity_flag is True


# ---------------------------------------------------------------------------
# record_hash is computed AFTER flags are finalized
# ---------------------------------------------------------------------------

def test_record_hash_reflects_final_ambiguity_flag_state():
    wisco = [
        _fake_occ("3240000400018", unit="3240", langs={"en": "Veterinary assistant"}),
        _fake_occ("5164140000000", unit="5164", langs={"en": "Veterinary assistant"}),
    ]
    records, _meta = bv2.build_records(wisco)
    for rec in records:
        assert rec.record_hash  # non-empty
        # Recompute the hash the same way build_records does and confirm it
        # matches -- proves the hash was taken from the post-flag state, not
        # a stale pre-flag snapshot.
        import hashlib
        payload = rec.model_dump_json(exclude={"record_hash"}, exclude_none=False).encode("utf-8")
        assert rec.record_hash == hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Schema validator accepts the flagged output
# ---------------------------------------------------------------------------

def test_flagged_records_pass_schema_validation():
    wisco = [
        _fake_occ("3240000400018", unit="3240", langs={"en": "Veterinary assistant"}),
        _fake_occ("5164140000000", unit="5164", langs={"en": "Veterinary assistant"}),
        _fake_occ("K3", unit="2512", langs={"en": "Software developer"}),
    ]
    records, _meta = bv2.build_records(wisco)
    dataset_hash = "0" * 64  # placeholder -- not what's under test here
    report = validate_benchmark_package(records, dataset_hash=dataset_hash)
    assert report.ok, report.errors


# ---------------------------------------------------------------------------
# The real, already-discovered conflict is exactly one pair (regression
# guard -- if this ever grows, it means a NEW gold-code conflict has
# appeared in the real dataset and must be investigated, not silently
# absorbed).
# ---------------------------------------------------------------------------

def test_real_wisco_v2_dataset_has_exactly_one_known_conflict_pair():
    import json
    records_path = (
        Path(__file__).resolve().parent
        / "local_benchmarks" / "wisco_isco08_v2_group_split" / "records.json"
    )
    if not records_path.exists():
        import pytest
        pytest.skip("real WISCO v2 local benchmark not present in this checkout")
    payload = json.loads(records_path.read_text(encoding="utf-8"))
    flagged = [r for r in payload["records"] if r["ambiguity_flag"]]
    flagged_ids = sorted(r["benchmark_id"] for r in flagged)
    assert flagged_ids == ["WISCO-3240000400018-en", "WISCO-5164140000000-en"]
    for r in flagged:
        assert r["exclusion_reason"]
        assert r["gold_code"]  # preserved, never blanked
