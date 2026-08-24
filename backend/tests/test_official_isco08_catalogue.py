"""
Tests for backend/rag/official_isco08_catalogue.py (Task 21).

Fully hermetic: every catalogue/metadata fixture is small and hand-built
in-process. No official ILO workbook, normalized official catalogue, or
WISCO artifact is read anywhere in this file.
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import backend.rag.official_isco08_catalogue as oic  # noqa: E402


_FIELDS = ["level", "code", "parent_code", "label"]

_SMALL_ROWS = [
    ("major", "1", "", "Managers"),
    ("submajor", "11", "1", "Chief Executives"),
    ("submajor", "12", "1", "Administrative Managers"),
    ("minor", "111", "11", "Legislators and Senior Officials"),
    ("minor", "121", "12", "Business Services Managers"),
    ("unit", "1111", "111", "Legislators"),
    ("unit", "1112", "111", "Senior Government Officials"),
    ("unit", "1211", "121", "Finance Managers"),
]
_SMALL_EXPECTED_COUNTS = {"major": 1, "submajor": 2, "minor": 2, "unit": 3}


def _write_csv(path: Path, rows: list[tuple]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(_FIELDS)
        for r in rows:
            w.writerow(r)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_metadata(path: Path, catalogue_sha256: str, counts: dict) -> Path:
    data = {"isco08": {"normalized_catalogue_sha256": catalogue_sha256, "verified_counts": counts}}
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _valid_fixture(tmp_path: Path):
    csv_path = _write_csv(tmp_path / "cat.csv", _SMALL_ROWS)
    meta_path = _write_metadata(tmp_path / "meta.yaml", _sha256(csv_path), _SMALL_EXPECTED_COUNTS)
    return csv_path, meta_path


# ---------------------------------------------------------------------------
# 1. Valid synthetic load, parameterized expected counts
# ---------------------------------------------------------------------------

def test_valid_synthetic_catalogue_loads(tmp_path):
    csv_path, meta_path = _valid_fixture(tmp_path)
    records = oic.load_official_catalogue(
        csv_path, meta_path, profile="test_profile", expected_counts=_SMALL_EXPECTED_COUNTS,
    )
    assert len(records) == 8
    by_level = oic.records_by_level(records)
    assert {lv: len(rs) for lv, rs in by_level.items()} == _SMALL_EXPECTED_COUNTS
    r = next(r for r in records if r.code == "1111")
    assert r.title_en == "Legislators"
    assert r.parent_code == "111"
    assert r.profile == "test_profile"
    assert r.embedding_text == "1111 Legislators"
    assert r.source_catalogue_sha256 == _sha256(csv_path)


# ---------------------------------------------------------------------------
# 2. Metadata missing / malformed
# ---------------------------------------------------------------------------

def test_missing_metadata_file_rejected(tmp_path):
    csv_path, _ = _valid_fixture(tmp_path)
    with pytest.raises(oic.OfficialISCO08CatalogueError, match="not found"):
        oic.load_official_catalogue(csv_path, tmp_path / "nope.yaml", expected_counts=_SMALL_EXPECTED_COUNTS)


def test_malformed_metadata_missing_isco08_key_rejected(tmp_path):
    csv_path, _ = _valid_fixture(tmp_path)
    meta_path = tmp_path / "bad_meta.yaml"
    meta_path.write_text(yaml.safe_dump({"not_isco08": {}}), encoding="utf-8")
    with pytest.raises(oic.OfficialISCO08CatalogueError, match="isco08"):
        oic.load_official_catalogue(csv_path, meta_path, expected_counts=_SMALL_EXPECTED_COUNTS)


def test_malformed_metadata_missing_required_keys_rejected(tmp_path):
    csv_path, _ = _valid_fixture(tmp_path)
    meta_path = tmp_path / "bad_meta2.yaml"
    meta_path.write_text(yaml.safe_dump({"isco08": {"normalized_catalogue_sha256": "abc"}}), encoding="utf-8")
    with pytest.raises(oic.OfficialISCO08CatalogueError, match="missing required key"):
        oic.load_official_catalogue(csv_path, meta_path, expected_counts=_SMALL_EXPECTED_COUNTS)


def test_malformed_metadata_bad_verified_counts_shape_rejected(tmp_path):
    csv_path, _ = _valid_fixture(tmp_path)
    meta_path = tmp_path / "bad_meta3.yaml"
    meta_path.write_text(
        yaml.safe_dump({"isco08": {"normalized_catalogue_sha256": _sha256(csv_path), "verified_counts": {"major": 1}}}),
        encoding="utf-8",
    )
    with pytest.raises(oic.OfficialISCO08CatalogueError, match="verified_counts"):
        oic.load_official_catalogue(csv_path, meta_path, expected_counts=_SMALL_EXPECTED_COUNTS)


# ---------------------------------------------------------------------------
# 3. Input hash mismatch
# ---------------------------------------------------------------------------

def test_hash_mismatch_rejected(tmp_path):
    csv_path, _ = _valid_fixture(tmp_path)
    meta_path = _write_metadata(tmp_path / "meta_wronghash.yaml", "0" * 64, _SMALL_EXPECTED_COUNTS)
    with pytest.raises(oic.OfficialISCO08CatalogueError, match="sha256 mismatch"):
        oic.load_official_catalogue(csv_path, meta_path, expected_counts=_SMALL_EXPECTED_COUNTS)


# ---------------------------------------------------------------------------
# 4. Missing source (catalogue) file
# ---------------------------------------------------------------------------

def test_missing_catalogue_file_rejected(tmp_path):
    _, meta_path = _valid_fixture(tmp_path)
    with pytest.raises(oic.OfficialISCO08CatalogueError, match="not found"):
        oic.load_official_catalogue(tmp_path / "nope.csv", meta_path, expected_counts=_SMALL_EXPECTED_COUNTS)


# ---------------------------------------------------------------------------
# 5. Wrong counts
# ---------------------------------------------------------------------------

def test_wrong_expected_counts_rejected(tmp_path):
    csv_path, meta_path = _valid_fixture(tmp_path)
    wrong = dict(_SMALL_EXPECTED_COUNTS)
    wrong["unit"] = 999
    with pytest.raises(oic.OfficialISCO08CatalogueError, match="expected 999"):
        oic.load_official_catalogue(csv_path, meta_path, expected_counts=wrong)


def test_metadata_count_mismatch_rejected(tmp_path):
    csv_path = _write_csv(tmp_path / "cat2.csv", _SMALL_ROWS)
    wrong_meta_counts = dict(_SMALL_EXPECTED_COUNTS)
    wrong_meta_counts["minor"] = 999
    meta_path = _write_metadata(tmp_path / "meta4.yaml", _sha256(csv_path), wrong_meta_counts)
    with pytest.raises(oic.OfficialISCO08CatalogueError, match="verified_counts.minor"):
        oic.load_official_catalogue(csv_path, meta_path, expected_counts=_SMALL_EXPECTED_COUNTS)


# ---------------------------------------------------------------------------
# 6. Malformed, duplicate, non-four-digit unit codes
# ---------------------------------------------------------------------------

def test_malformed_code_rejected(tmp_path):
    rows = list(_SMALL_ROWS)
    rows[0] = ("major", "X", "", "Managers")
    csv_path = _write_csv(tmp_path / "cat_malformed.csv", rows)
    meta_path = _write_metadata(tmp_path / "meta5.yaml", _sha256(csv_path), _SMALL_EXPECTED_COUNTS)
    with pytest.raises(oic.OfficialISCO08CatalogueError, match="non-numeric code"):
        oic.load_official_catalogue(csv_path, meta_path, expected_counts=_SMALL_EXPECTED_COUNTS)


def test_duplicate_code_rejected(tmp_path):
    rows = list(_SMALL_ROWS) + [("unit", "1111", "111", "Duplicate Legislators")]
    csv_path = _write_csv(tmp_path / "cat_dup.csv", rows)
    meta_path = _write_metadata(tmp_path / "meta6.yaml", _sha256(csv_path), _SMALL_EXPECTED_COUNTS)
    with pytest.raises(oic.OfficialISCO08CatalogueError, match="duplicate code"):
        oic.load_official_catalogue(csv_path, meta_path, expected_counts=_SMALL_EXPECTED_COUNTS)


def test_non_four_digit_unit_code_rejected(tmp_path):
    rows = list(_SMALL_ROWS)
    rows[5] = ("unit", "111", "111", "Legislators")  # 3 digits at unit level
    csv_path = _write_csv(tmp_path / "cat_shortunit.csv", rows)
    meta_path = _write_metadata(tmp_path / "meta7.yaml", _sha256(csv_path), _SMALL_EXPECTED_COUNTS)
    with pytest.raises(oic.OfficialISCO08CatalogueError, match="expected 4"):
        oic.load_official_catalogue(csv_path, meta_path, expected_counts=_SMALL_EXPECTED_COUNTS)


# ---------------------------------------------------------------------------
# 7. Invalid parent links
# ---------------------------------------------------------------------------

def test_orphan_parent_rejected(tmp_path):
    rows = [
        ("major", "1", "", "Managers"),
        ("submajor", "11", "1", "Chief Executives"),
        ("unit", "9999", "999", "Orphan"),  # parent "999" never appears
    ]
    csv_path = _write_csv(tmp_path / "cat_orphan.csv", rows)
    counts = {"major": 1, "submajor": 1, "minor": 0, "unit": 1}
    meta_path = _write_metadata(tmp_path / "meta8.yaml", _sha256(csv_path), counts)
    with pytest.raises(oic.OfficialISCO08CatalogueError, match="has not appeared"):
        oic.load_official_catalogue(csv_path, meta_path, expected_counts=counts)


def test_unexpected_parent_for_top_level_rejected(tmp_path):
    rows = list(_SMALL_ROWS)
    rows[0] = ("major", "1", "9", "Managers")  # top-level with a parent_code
    csv_path = _write_csv(tmp_path / "cat_topparent.csv", rows)
    meta_path = _write_metadata(tmp_path / "meta9.yaml", _sha256(csv_path), _SMALL_EXPECTED_COUNTS)
    with pytest.raises(oic.OfficialISCO08CatalogueError, match="must be blank"):
        oic.load_official_catalogue(csv_path, meta_path, expected_counts=_SMALL_EXPECTED_COUNTS)


# ---------------------------------------------------------------------------
# 8. Nonblank title requirement
# ---------------------------------------------------------------------------

def test_blank_title_rejected(tmp_path):
    rows = list(_SMALL_ROWS)
    rows[0] = ("major", "1", "", "")
    csv_path = _write_csv(tmp_path / "cat_blanktitle.csv", rows)
    meta_path = _write_metadata(tmp_path / "meta10.yaml", _sha256(csv_path), _SMALL_EXPECTED_COUNTS)
    with pytest.raises(oic.OfficialISCO08CatalogueError, match="blank title"):
        oic.load_official_catalogue(csv_path, meta_path, expected_counts=_SMALL_EXPECTED_COUNTS)


# ---------------------------------------------------------------------------
# 9. Loader never imports/reads WISCO
# ---------------------------------------------------------------------------

def test_loader_source_has_no_wisco_reference():
    """The module docstring explains, in prose, that WISCO is never read
    -- that mention (and the illustrative path it names) is expected.
    What must never appear is an actual import statement or hard-coded
    path literal pulling in WISCO data."""
    import ast

    source = Path(oic.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "wisco" not in alias.name.lower(), f"unexpected import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module and "wisco" in node.module.lower()), f"unexpected import from: {node.module}"
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Skip the module docstring itself (first statement of the
            # module body) -- it explains non-use in prose, which is fine.
            pass
    # Independently: no string literal outside the docstring may name a
    # WISCO path -- checked by removing the (first) module docstring text
    # and re-scanning what remains.
    docstring = ast.get_docstring(tree) or ""
    body_without_docstring = source.replace(docstring, "", 1)
    for token in ("backend.evaluation.wisco", "backend/evaluation/wisco", "eval.legacy_thesis_ch6.wisco", "eval/legacy_thesis_ch6/wisco", "local_benchmarks", "wisco_raw_parsed"):
        assert token not in body_without_docstring, f"unexpected WISCO reference outside docstring: {token!r}"


def test_loader_reads_no_file_besides_the_two_given_paths(tmp_path, monkeypatch):
    """Guard against a future regression that adds a hidden extra file
    read (e.g. a hard-coded WISCO path) -- patches Path.exists so any
    unexpected path probe fails loudly instead of silently succeeding."""
    csv_path, meta_path = _valid_fixture(tmp_path)
    allowed = {csv_path.resolve(), meta_path.resolve()}
    real_read_bytes = Path.read_bytes
    real_read_text = Path.read_text

    def _guarded_read_bytes(self, *a, **kw):
        assert self.resolve() in allowed, f"unexpected file read: {self}"
        return real_read_bytes(self, *a, **kw)

    def _guarded_read_text(self, *a, **kw):
        assert self.resolve() in allowed, f"unexpected file read: {self}"
        return real_read_text(self, *a, **kw)

    monkeypatch.setattr(Path, "read_bytes", _guarded_read_bytes)
    monkeypatch.setattr(Path, "read_text", _guarded_read_text)
    oic.load_official_catalogue(csv_path, meta_path, expected_counts=_SMALL_EXPECTED_COUNTS)
