"""
Tests for backend/rag/build_official_isco08_collections.py (Task 21).

Fully hermetic: small synthetic catalogue/metadata fixtures only. No
Qdrant, SentenceTransformer, network, or WISCO resource is ever touched
-- several tests below assert that directly by monkeypatching
QdrantClient/SentenceTransformer to raise if constructed.
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import backend.rag.build_official_isco08_collections as builder  # noqa: E402
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[tuple]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(_FIELDS)
        for r in rows:
            w.writerow(r)
    return path


def _write_metadata(path: Path, catalogue_sha256: str, counts: dict) -> Path:
    path.write_text(
        yaml.safe_dump({"isco08": {"normalized_catalogue_sha256": catalogue_sha256, "verified_counts": counts}}),
        encoding="utf-8",
    )
    return path


def _valid_fixture(tmp_path: Path):
    csv_path = _write_csv(tmp_path / "cat.csv", _SMALL_ROWS)
    meta_path = _write_metadata(tmp_path / "meta.yaml", _sha256(csv_path), _SMALL_EXPECTED_COUNTS)
    return csv_path, meta_path


# ---------------------------------------------------------------------------
# 10. Dry-run plan has level-specific counts and versioned names
# ---------------------------------------------------------------------------

def test_dry_run_plan_counts_and_names(tmp_path):
    csv_path, meta_path = _valid_fixture(tmp_path)
    plan = builder.build_plan(csv_path, meta_path, profile=oic.DEFAULT_PROFILE, expected_counts=_SMALL_EXPECTED_COUNTS)

    by_level = {e.level: e for e in plan.entries if e.name != "isco08_unit_groups_flat_ilo2021_v1"}
    assert by_level["major"].record_count == 1
    assert by_level["submajor"].record_count == 2
    assert by_level["minor"].record_count == 2
    assert by_level["unit"].record_count == 3

    names = {e.name for e in plan.entries}
    assert names == {
        "isco08_major_groups_ilo2021_v1",
        "isco08_submajor_groups_ilo2021_v1",
        "isco08_minor_groups_ilo2021_v1",
        "isco08_unit_groups_ilo2021_v1",
        "isco08_unit_groups_flat_ilo2021_v1",
    }
    flat_entry = next(e for e in plan.entries if e.name == "isco08_unit_groups_flat_ilo2021_v1")
    assert flat_entry.record_count == 3  # same as unit-level count
    assert flat_entry.level == "unit"

    plan_dict = builder.plan_to_dict(plan)
    assert plan_dict["profile"] == oic.DEFAULT_PROFILE
    assert len(plan_dict["entries"]) == 5


def test_unknown_profile_rejected(tmp_path):
    csv_path, meta_path = _valid_fixture(tmp_path)
    with pytest.raises(builder.UnknownOfficialProfileError, match="no registered collection-name mapping"):
        builder.build_plan(csv_path, meta_path, profile="not_a_real_profile", expected_counts=_SMALL_EXPECTED_COUNTS)


# ---------------------------------------------------------------------------
# 11. Dry-run never constructs Qdrant/embedder
# ---------------------------------------------------------------------------

def test_dry_run_never_constructs_qdrant_or_embedder(tmp_path, monkeypatch):
    csv_path, meta_path = _valid_fixture(tmp_path)

    def _boom(*args, **kwargs):
        raise AssertionError("Qdrant/embedder must never be constructed during a Task 21 dry-run")

    monkeypatch.setattr("qdrant_client.QdrantClient", _boom)
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _boom)

    plan = builder.build_plan(csv_path, meta_path, profile=oic.DEFAULT_PROFILE, expected_counts=_SMALL_EXPECTED_COUNTS)
    assert len(plan.entries) == 5


def test_builder_module_does_not_import_qdrant_or_embedder():
    """Stronger than the runtime monkeypatch above: the module has no
    import path to Qdrant/SentenceTransformer at all, so a dry-run
    literally cannot reach either, regardless of code path. (The module
    docstring mentions "Qdrant" in prose, explaining that it is never
    instantiated -- that's expected and checked separately by AST
    inspection of actual import statements, not a substring scan.)"""
    import ast

    source = Path(builder.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    banned = ("qdrant", "sentence_transformers")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not any(b in alias.name.lower() for b in banned), f"unexpected import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").lower()
            assert not any(b in mod for b in banned), f"unexpected import from: {node.module}"


def test_cli_execute_flag_refused_and_raises_before_any_planning(tmp_path, monkeypatch, capsys):
    csv_path, meta_path = _valid_fixture(tmp_path)

    def _boom(*a, **kw):
        raise AssertionError("build_plan must never be called when --execute is passed")

    monkeypatch.setattr(builder, "build_plan", _boom)
    argv = ["build_official_isco08_collections.py", "--catalogue", str(csv_path),
            "--metadata", str(meta_path), "--execute"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc_info:
        builder.main()
    assert exc_info.value.code != 0
    out = capsys.readouterr().out
    assert "REFUSED" in out


# ---------------------------------------------------------------------------
# 12. Builder rejects invalid source before planning
# ---------------------------------------------------------------------------

def test_builder_rejects_invalid_source_before_planning(tmp_path):
    rows = list(_SMALL_ROWS)
    rows[0] = ("major", "1", "", "")  # blank title -> loader must reject
    csv_path = _write_csv(tmp_path / "cat_bad.csv", rows)
    meta_path = _write_metadata(tmp_path / "meta_bad.yaml", _sha256(csv_path), _SMALL_EXPECTED_COUNTS)

    with pytest.raises(oic.OfficialISCO08CatalogueError, match="blank title"):
        builder.build_plan(csv_path, meta_path, profile=oic.DEFAULT_PROFILE, expected_counts=_SMALL_EXPECTED_COUNTS)


def test_cli_reports_validation_failure_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    rows = list(_SMALL_ROWS)
    rows[0] = ("major", "1", "", "")
    csv_path = _write_csv(tmp_path / "cat_bad2.csv", rows)
    meta_path = _write_metadata(tmp_path / "meta_bad2.yaml", _sha256(csv_path), _SMALL_EXPECTED_COUNTS)

    argv = ["build_official_isco08_collections.py", "--catalogue", str(csv_path), "--metadata", str(meta_path)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc_info:
        builder.main()
    assert exc_info.value.code != 0
    out = capsys.readouterr().out
    assert "CATALOGUE/PROFILE VALIDATION FAILURE" in out
