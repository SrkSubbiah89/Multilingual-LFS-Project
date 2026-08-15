"""
Tests for backend/rag/load_full_isco.py's static ISCO-08 catalogue tables.

Covers two fixes:

1. The subsistence-farming sub-major mis-numbering bug found in
   Documentation/Phase_2/Week_1/module_a_week1_report.md Sec.5.3: codes
   6161-6164 were filed under sub-major group 61 even though no minor
   group 614/615/616 is defined at all -- fixed by renumbering to
   6310-6340 (same labels, same content -- only the code changed).

2. 2026-08-12: the full remaining cross-check against the primary ILO
   ISCO-08 source (isco.ilo.org/en/isco-08, official structure CSV
   export -- not WISCO, not a guess). This resolved all 15 remaining
   project-only / 10 official-only codes Module A Week 1's WISCO-only
   comparison had flagged as out of scope, plus 6 further codes that
   comparison could not catch at all: several official, structurally
   valid codes (9510, 9520, 9611, 9612, 9613, 9621, and the 6121/6122/
   6123 trio) carried another occupation's label entirely -- a genuine
   content-assignment bug invisible to a code-existence check. Full
   before/after accounting: 441 -> 436 entries, zero duplicates. The
   sole disclosed, intentionally-NOT-fixed exception is major group 0
   (Armed Forces): this file keeps 4-digit "0110"/"0210"/"0310" rather
   than ISCO-08's own bare-3-digit convention ("110"/"210"/"310"), since
   changing code string length here would ripple into every other
   4-digit-code assumption in this codebase (CSV joins, WISCO gold-code
   comparison, `_digits()`) -- a coordinated decision, not a unilateral
   rename. See this file's own comments at the fixed blocks for the
   full reasoning behind each individual change.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.rag.load_full_isco import _MINOR, _SUBMAJOR, _UNIT  # noqa: E402

# 2026-08-12: after the primary-source cross-check fix, every unit code's
# 3-digit minor-group prefix resolves to a defined minor group -- the
# structural-orphan class of bug (this file's own earlier, narrower check)
# no longer has any known instances. Kept as an explicit empty allowlist
# (rather than deleting the regression guard) so a *newly introduced*
# orphan is still caught immediately, and so this history is visible
# rather than silently disappearing.
_KNOWN_STRUCTURALLY_ORPHANED_CODES: frozenset[str] = frozenset()

# Spot-check values for every code touched by the 2026-08-12 primary-source
# fix, each traceable to isco.ilo.org's official ISCO-08 structure export
# (fetched and cross-checked directly, not guessed). Codes removed
# entirely (1347, 9131, 9132, 9141, 9152) are covered by
# test_non_standard_codes_removed() below instead.
_FIXED_CODE_LABELS = {
    "6121": "Livestock and Dairy Producers",
    "6122": "Poultry Producers",
    "6123": "Apiarists and Sericulturists",
    "6210": "Forestry and Related Workers",
    "6221": "Aquaculture Workers",
    "6222": "Inland and Coastal Waters Fishery Workers",
    "6223": "Deep-sea Fishery Workers",
    "6224": "Hunters and Trappers",
    "7119": "Building Frame and Related Trades Workers Not Elsewhere Classified",
    "9510": "Street and Related Services Workers",
    "9520": "Street Vendors (excluding Food)",
    "9611": "Garbage and Recycling Collectors",
    "9612": "Refuse Sorters",
    "9613": "Sweepers and Related Labourers",
    "9621": "Messengers, Package Deliverers and Luggage Porters",
    "9622": "Odd-job Persons",
    "9623": "Meter Readers and Vending-machine Collectors",
    "9624": "Water and Firewood Collectors",
    "9629": "Elementary Workers Not Elsewhere Classified",
}


def test_subsistence_farming_codes_no_longer_present():
    unit_codes = {code for code, _label in _UNIT}
    assert "6161" not in unit_codes
    assert "6162" not in unit_codes
    assert "6163" not in unit_codes
    assert "6164" not in unit_codes


def test_subsistence_farming_codes_correctly_renumbered_same_labels():
    unit_by_code = dict(_UNIT)
    assert unit_by_code["6310"] == "Subsistence Crop Farmers"
    assert unit_by_code["6320"] == "Subsistence Livestock Farmers"
    assert unit_by_code["6330"] == "Subsistence Mixed Crop and Livestock Farmers"
    assert unit_by_code["6340"] == "Subsistence Fishers, Hunters, Trappers and Gatherers"


def test_subsistence_farming_unit_codes_have_defined_minor_parents():
    minor_codes = {code for code, _label, _ar in _MINOR}
    for code in ("631", "632", "633", "634"):
        assert code in minor_codes, f"minor group {code} must be defined"
    for unit_code in ("6310", "6320", "6330", "6340"):
        assert unit_code[:3] in minor_codes


def test_submajor_63_and_its_minors_exist():
    submajor_codes = {code for code, _label, _ar in _SUBMAJOR}
    assert "63" in submajor_codes
    minor_codes = {code for code, _label, _ar in _MINOR}
    assert {"631", "632", "633", "634"} <= minor_codes


def test_no_unit_code_structurally_orphaned_except_the_known_set():
    """Every unit code's 3-digit minor-group prefix must match a defined
    minor group -- the known-orphan allowlist is now empty (see module
    docstring). Regression guard: if a *new* structurally orphaned code
    is introduced, this test catches it rather than passing silently."""
    minor_codes = {code for code, _label, _ar in _MINOR}
    orphans = {code for code, _label in _UNIT if code[:3] not in minor_codes}
    assert orphans == _KNOWN_STRUCTURALLY_ORPHANED_CODES, (
        f"structurally orphaned unit codes changed: new={orphans - _KNOWN_STRUCTURALLY_ORPHANED_CODES}, "
        f"resolved={_KNOWN_STRUCTURALLY_ORPHANED_CODES - orphans} -- if a code was genuinely "
        f"resolved, update _KNOWN_STRUCTURALLY_ORPHANED_CODES here to match, with a citation "
        f"to a verified official ISCO-08 source, not a guess"
    )


def test_unit_group_count_matches_official_436():
    """2026-08-12 primary-source fix: 441 -> 436 entries, matching the
    real ISCO-08 unit-group count exactly (isco.ilo.org official
    structure export). The 3 disclosed Armed Forces exceptions (see
    module docstring) are a code-format question, not a count question --
    they were always counted as 3 unit groups, before and after this fix."""
    assert len(_UNIT) == 436


def test_unit_group_codes_have_no_duplicates():
    codes = [code for code, _label in _UNIT]
    assert len(codes) == len(set(codes))


def test_non_standard_codes_removed():
    """1347, 9131, 9132, 9141, 9152 were confirmed non-standard against
    the primary ILO ISCO-08 source with no salvageable replacement code
    (either genuinely absorbed into an existing correct code, or -- for
    9132/9152 -- no official ISCO-08 unit group exists for that content
    at all). Removed outright, not renumbered."""
    unit_codes = {code for code, _label in _UNIT}
    for code in ("1347", "9131", "9132", "9141", "9152"):
        assert code not in unit_codes, f"{code} was confirmed non-standard and should have been removed"


def test_primary_source_fixed_codes_have_official_labels():
    """Every code touched by the 2026-08-12 primary-source cross-check
    carries the exact official ISCO-08 label -- both newly-added codes
    and codes whose existing label was replaced (the 9510-9629 block,
    where the code itself was always valid but was carrying an entirely
    different occupation's label)."""
    unit_by_code = dict(_UNIT)
    for code, expected_label in _FIXED_CODE_LABELS.items():
        assert code in unit_by_code, f"{code} should exist after the primary-source fix"
        assert unit_by_code[code] == expected_label, (
            f"{code}: expected official label {expected_label!r}, got {unit_by_code[code]!r}"
        )


def test_6124_and_6141_6142_6150_6141_no_longer_present():
    """The specific non-standard codes replaced by the 612x re-shift and
    the 62x forestry/aquaculture fix must not linger as leftover
    duplicates alongside their corrected replacements."""
    unit_codes = {code for code, _label in _UNIT}
    for code in ("6124", "6141", "6142", "6150", "7116", "9151", "9153", "9161", "9162", "9420"):
        assert code not in unit_codes, f"{code} should have been renumbered/removed, not left in place"
