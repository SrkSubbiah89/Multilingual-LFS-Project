"""
Tests for backend/rag/load_full_isco.py's static ISCO-08 catalogue tables.

Covers the fix for the subsistence-farming sub-major mis-numbering bug
found in Documentation/Phase_2/Week_1/module_a_week1_report.md Sec.5.3:
codes 6161-6164 ("Subsistence Crop/Livestock/Mixed/Fishers Farmers") were
filed under sub-major group 61 even though this file's own _SUBMAJOR/
_MINOR tables never define a minor group 614/615/616 at all -- the
correct sub-major (63, "Subsistence Farmers, Fishers, Hunters and
Gatherers") and its minor groups (631-634) were already present, just
missing their unit-group children. Fixed by renumbering to 6310-6340
(same labels, same content -- only the code changed).

This file also adds a general parent-consistency check so this exact bug
class (a unit code whose 3-digit minor-group prefix has no defined
parent) cannot silently recur for the 4 now-fixed codes, while being
honest that Module A's report found 15 further project-only codes total
(and 10 official-only codes this system doesn't have at all) that are
NOT fixed here -- those need a full cross-check against the official
ISCO-08 structure document, not a guess (see that report Sec.5.1/5.2 and
its own stated scope boundary). Of those 15, only 10 are structurally
orphaned (no minor-group parent at all) and thus verifiable by this
file's simpler check; the other 5 have a defined parent but are still
not confirmed-correct ISCO-08 codes -- see the constant's own docstring
below for exactly which is which.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.rag.load_full_isco import _MINOR, _SUBMAJOR, _UNIT  # noqa: E402

# Module A Week 1 found 19 project-only codes overall (i.e. codes this
# system's knowledge base has that the official 436-code ISCO-08 unit-
# group list does not) by direct comparison against the official
# standard -- a strictly stronger check than the structural one below.
# This file's own test instead checks a narrower, purely structural
# condition (does a unit code's 3-digit prefix match ANY minor group
# this same file defines, right or wrong) -- it can only catch a
# *subset* of those 19 (a unit with literally no minor-group parent at
# all), not codes whose parent exists but the unit itself still isn't a
# real ISCO-08 code. Of the 15 project-only codes remaining after this
# file's 4-code subsistence-farming fix, exactly this structural subset
# is caught here; the other 5 (1347, 6124, 7116, 9131, 9132) have a
# defined-but-possibly-still-wrong minor-group parent and are not
# distinguishable by this check -- they remain part of Module A's
# broader, not-yet-fixed 15-code finding, just not verifiable by this
# particular test.
_KNOWN_STRUCTURALLY_ORPHANED_CODES = frozenset({
    "6141", "6142", "6150", "9141", "9151", "9152", "9153", "9161", "9162", "9420",
})


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
    minor group, EXCEPT the 10 codes already known (see module docstring)
    to have no minor-group parent at all in this file's own tables. This
    is a regression guard: if a *new* structurally orphaned code appears
    (or if someone believes they've resolved one of the 10 without
    updating this allowlist), this test will catch it rather than
    passing silently."""
    minor_codes = {code for code, _label, _ar in _MINOR}
    orphans = {code for code, _label in _UNIT if code[:3] not in minor_codes}
    assert orphans == _KNOWN_STRUCTURALLY_ORPHANED_CODES, (
        f"structurally orphaned unit codes changed: new={orphans - _KNOWN_STRUCTURALLY_ORPHANED_CODES}, "
        f"resolved={_KNOWN_STRUCTURALLY_ORPHANED_CODES - orphans} -- if a code was genuinely "
        f"resolved, update _KNOWN_STRUCTURALLY_ORPHANED_CODES here to match, with a citation "
        f"to a verified official ISCO-08 source, not a guess"
    )
