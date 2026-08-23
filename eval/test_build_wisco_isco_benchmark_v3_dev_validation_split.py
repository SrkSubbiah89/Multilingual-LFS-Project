"""
Tests for eval/build_wisco_isco_benchmark_v3_dev_validation_split.py --
the genuine 3-way (dev/validation/heldout) split built on top of v2's
audited 2-way (dev/heldout) split.

The single most important property this file checks: heldout is
UNTOUCHED. Every group v2 assigns to heldout must land in v3's heldout
too, byte-identical, never resplit into validation. Uses small in-memory
fixtures throughout -- never reads the real WISCO source file.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_wisco_isco_benchmark_v2_group_split as bv2  # noqa: E402
import build_wisco_isco_benchmark_v3_dev_validation_split as bv3  # noqa: E402
from controlled_benchmark_schema import SPLIT_DEV, SPLIT_HELDOUT, SPLIT_VALIDATION  # noqa: E402


def _fake_occ(key, unit="2512", label="Software developer", langs=None):
    langs = langs if langs is not None else {"en": f"Occupation {key}", "ar": f"مهنة {key}"}
    return {"key": key, "isco08_unit": unit, "master_label_en": label, "titles": langs}


def _synthetic_wisco(n: int) -> list[dict]:
    """n distinct, unrelated occupations (no shared text -- no cross-key
    group merging), so each is its own singleton group."""
    return [_fake_occ(str(1000 + i), unit=f"{2500 + (i % 50)}") for i in range(n)]


# ---------------------------------------------------------------------------
# The load-bearing property: heldout is untouched by the sub-split
# ---------------------------------------------------------------------------

def test_heldout_membership_identical_to_v2():
    wisco = _synthetic_wisco(500)
    v2_records, _ = bv2.build_records(wisco)
    v3_records, _ = bv3.build_records(wisco)

    v2_heldout_ids = {r.benchmark_id for r in v2_records if r.split == SPLIT_HELDOUT}
    v3_heldout_ids = {r.benchmark_id for r in v3_records if r.split == SPLIT_HELDOUT}

    assert v2_heldout_ids == v3_heldout_ids
    assert len(v2_heldout_ids) > 0  # sanity: the fixture actually produced some heldout records


def test_no_heldout_group_ever_becomes_validation():
    wisco = _synthetic_wisco(500)
    v3_records, _ = bv3.build_records(wisco)
    assert not any(r.split == SPLIT_VALIDATION for r in v3_records if _is_originally_heldout(wisco, r))


def _is_originally_heldout(wisco, rec) -> bool:
    key_to_group, _ = bv2.build_groups(wisco)
    group_members: dict = {}
    for k, g in key_to_group.items():
        group_members.setdefault(g, []).append(k)
    # rec.benchmark_id looks like "WISCO-{key}-{lang}"
    key = rec.benchmark_id.split("-")[1]
    group = key_to_group[key]
    return bv2._split_for_group(group, group_members[group]) == SPLIT_HELDOUT


# ---------------------------------------------------------------------------
# Only the dev pool is sub-split, and only into dev/validation
# ---------------------------------------------------------------------------

def test_every_record_is_dev_validation_or_heldout():
    wisco = _synthetic_wisco(500)
    v3_records, _ = bv3.build_records(wisco)
    assert all(r.split in (SPLIT_DEV, SPLIT_VALIDATION, SPLIT_HELDOUT) for r in v3_records)


def test_validation_split_is_nonempty_at_reasonable_scale():
    wisco = _synthetic_wisco(2000)
    v3_records, meta = bv3.build_records(wisco)
    validation_records = [r for r in v3_records if r.split == SPLIT_VALIDATION]
    dev_records = [r for r in v3_records if r.split == SPLIT_DEV]
    assert len(validation_records) > 0
    assert len(dev_records) > 0
    # Roughly 70/30 dev/validation of the dev pool -- loose bounds, this is
    # a hash-based split on a small-ish sample, not a promise of exactness.
    dev_pool = len(validation_records) + len(dev_records)
    validation_fraction = len(validation_records) / dev_pool
    assert 0.15 < validation_fraction < 0.45


# ---------------------------------------------------------------------------
# Group integrity: a group's members never split across dev and validation
# ---------------------------------------------------------------------------

def test_group_never_splits_across_dev_and_validation():
    """Two source keys sharing identical text in one language get merged
    into one group by build_groups() -- both their language records must
    land in the SAME split (dev or validation), never split apart."""
    wisco = [
        _fake_occ("K1", unit="2512", langs={"en": "Same Title Here", "ar": "عنوان أ"}),
        _fake_occ("K2", unit="2512", langs={"en": "Same Title Here", "ar": "عنوان ب"}),
    ] + _synthetic_wisco(300)  # padding so both dev and validation are populated

    v3_records, meta = bv3.build_records(wisco)
    assert meta["n_merges_from_duplicate_text"] >= 1

    k1_en = next(r for r in v3_records if r.benchmark_id == "WISCO-K1-en")
    k2_en = next(r for r in v3_records if r.benchmark_id == "WISCO-K2-en")
    # If either landed in heldout, they must both be heldout (same group);
    # if in the dev pool, they must be in the SAME dev-pool sub-split.
    assert k1_en.split == k2_en.split


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_deterministic_across_two_runs():
    wisco = _synthetic_wisco(300)
    records_a, _ = bv3.build_records(wisco)
    records_b, _ = bv3.build_records(wisco)
    splits_a = {r.benchmark_id: r.split for r in records_a}
    splits_b = {r.benchmark_id: r.split for r in records_b}
    assert splits_a == splits_b


# ---------------------------------------------------------------------------
# Sub-split seed is independent of v2's dev/heldout seed
# ---------------------------------------------------------------------------

def test_validation_sub_split_uses_a_different_seed_than_dev_vs_heldout():
    assert bv3.VALIDATION_SUB_SPLIT_SEED != bv2.FIXED_SEED
