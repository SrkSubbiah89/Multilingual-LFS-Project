"""
Tests for eval/validate_real_lfs_governance.py -- Section F of the
Conference I Reviewer #2 response (real LFS validation intake support,
reviewer comment 3), expanded in the Step 3 real-LFS-intake hardening pass.

Covers the 10 required test scenarios from that pass's task G:
  1. Complete approved real-LFS manifest passes validation.
  2. Missing ethics/governance approval fails.
  3. Missing data-owner information fails.
  4. Missing de-identification confirmation fails.
  5. Missing frozen test split fails.
  6. Missing independent-label details fails.
  7. Synthetic fixture passes only as synthetic_or_operationally_realistic.
  8. Synthetic fixture cannot be relabelled as approved real LFS.
  9. Raw data paths under Git-tracked directories are rejected.
  10. Generated reports carry the correct validation label
      (covered in eval/test_manifest.py / eval/test_ablation_runner.py,
      which exercise the label end-to-end on real manifest objects).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import validate_real_lfs_governance as vrg  # noqa: E402
from dataset_card_schema import (  # noqa: E402
    APPROVED_REAL_LFS_VALIDATION,
    DATASET_LABELS,
    DatasetCard,
    INVALID_INCOMPLETE_GOVERNANCE,
    SYNTHETIC_OR_OPERATIONALLY_REALISTIC,
)

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_MINIMAL_SYNTHETIC_PATH = _FIXTURES_DIR / "synthetic_lfs_dataset_card_example.json"
_COMPLETE_FIXTURE_PATH = _FIXTURES_DIR / "complete_approved_real_lfs_dataset_card_example.json"
_INTAKE_PACKAGE_CARD_PATH = _FIXTURES_DIR / "synthetic_lfs_intake_package" / "dataset_card.json"


def load_minimal_synthetic_card() -> DatasetCard:
    return DatasetCard.model_validate_json(_MINIMAL_SYNTHETIC_PATH.read_text(encoding="utf-8"))


def load_complete_card() -> DatasetCard:
    return DatasetCard.model_validate_json(_COMPLETE_FIXTURE_PATH.read_text(encoding="utf-8"))


def load_intake_package_card() -> DatasetCard:
    return DatasetCard.model_validate_json(_INTAKE_PACKAGE_CARD_PATH.read_text(encoding="utf-8"))


def make_manifest(**overrides):
    m = {"dataset_label": APPROVED_REAL_LFS_VALIDATION, "split_name": "heldout_v1"}
    m.update(overrides)
    return m


# ---------------------------------------------------------------------------
# Closed label vocabulary
# ---------------------------------------------------------------------------

def test_dataset_labels_is_exactly_three_values():
    assert DATASET_LABELS == {
        SYNTHETIC_OR_OPERATIONALLY_REALISTIC,
        APPROVED_REAL_LFS_VALIDATION,
        INVALID_INCOMPLETE_GOVERNANCE,
    }


# ---------------------------------------------------------------------------
# Non-approved labels: informational only, never blocked
# ---------------------------------------------------------------------------

def test_synthetic_label_is_never_blocked_even_without_card():
    report = vrg.validate({"dataset_label": SYNTHETIC_OR_OPERATIONALLY_REALISTIC, "split_name": "dev_v1"}, None)
    assert report.ok
    assert report.warnings


def test_invalid_incomplete_governance_label_is_never_blocked():
    """This label is the OUTCOME of a failed approved-real attempt (see
    eval/manifest.py's on_governance_failure='downgrade') -- validate()
    itself never re-blocks a manifest already carrying it."""
    report = vrg.validate({"dataset_label": INVALID_INCOMPLETE_GOVERNANCE, "split_name": "heldout_v1"}, None)
    assert report.ok


def test_missing_label_is_never_blocked():
    report = vrg.validate({}, None)
    assert report.ok


# ---------------------------------------------------------------------------
# approved_real_lfs_validation: card presence / split_name
# ---------------------------------------------------------------------------

def test_approved_real_lfs_without_card_is_blocked():
    report = vrg.validate(make_manifest(), None)
    assert not report.ok
    assert any("no DatasetCard was supplied" in e for e in report.errors)


def test_approved_real_lfs_without_split_name_is_blocked():
    card = load_complete_card()
    report = vrg.validate(make_manifest(split_name=""), card)
    assert not report.ok
    assert any("split_name" in e for e in report.errors)


# ---------------------------------------------------------------------------
# G.1 -- Complete approved real-LFS manifest passes validation.
# ---------------------------------------------------------------------------

def test_complete_card_passes_approved_real_lfs_validation():
    card = load_complete_card()
    report = vrg.validate(make_manifest(), card)
    assert report.ok, report.errors


# ---------------------------------------------------------------------------
# G.2 -- Missing ethics/governance approval fails.
# ---------------------------------------------------------------------------

def test_missing_ethics_approval_fails():
    card = load_complete_card()
    data = card.model_dump()
    data["ethics_approval_ref"] = None
    report = vrg.validate(make_manifest(), DatasetCard.model_validate(data))
    assert not report.ok
    assert any("ethics_approval_ref" in e for e in report.errors)


def test_missing_data_sharing_agreement_ref_fails():
    card = load_complete_card()
    data = card.model_dump()
    data["data_sharing_agreement_ref"] = ""
    report = vrg.validate(make_manifest(), DatasetCard.model_validate(data))
    assert not report.ok
    assert any("data_sharing_agreement_ref" in e for e in report.errors)


# ---------------------------------------------------------------------------
# G.3 -- Missing data-owner information fails.
# ---------------------------------------------------------------------------

def test_missing_data_owner_or_custodian_fails():
    card = load_complete_card()
    data = card.model_dump()
    data["data_owner_or_custodian"] = None
    report = vrg.validate(make_manifest(), DatasetCard.model_validate(data))
    assert not report.ok
    assert any("data_owner_or_custodian" in e for e in report.errors)


# ---------------------------------------------------------------------------
# G.4 -- Missing de-identification confirmation fails.
# ---------------------------------------------------------------------------

def test_missing_direct_identifiers_removed_confirmation_fails():
    card = load_complete_card()
    data = card.model_dump()
    data["direct_identifiers_removed_confirmed"] = None
    report = vrg.validate(make_manifest(), DatasetCard.model_validate(data))
    assert not report.ok
    assert any("direct_identifiers_removed_confirmed" in e for e in report.errors)


def test_direct_identifiers_removed_confirmation_false_fails():
    card = load_complete_card()
    data = card.model_dump()
    data["direct_identifiers_removed_confirmed"] = False
    report = vrg.validate(make_manifest(), DatasetCard.model_validate(data))
    assert not report.ok
    assert any("direct_identifiers_removed_confirmed" in e for e in report.errors)


def test_missing_deidentification_method_fails():
    card = load_complete_card()
    data = card.model_dump()
    data["deidentification_method"] = ""
    report = vrg.validate(make_manifest(), DatasetCard.model_validate(data))
    assert not report.ok
    assert any("deidentification_method" in e for e in report.errors)


# ---------------------------------------------------------------------------
# G.5 -- Missing frozen test split fails.
# ---------------------------------------------------------------------------

def test_missing_frozen_heldout_test_status_fails():
    card = load_complete_card()
    data = card.model_dump()
    data["frozen_heldout_test_status"] = None
    report = vrg.validate(make_manifest(), DatasetCard.model_validate(data))
    assert not report.ok
    assert any("frozen_heldout_test_status" in e for e in report.errors)


def test_missing_split_hash_fails():
    card = load_complete_card()
    data = card.model_dump()
    data["split_hash"] = ""
    report = vrg.validate(make_manifest(), DatasetCard.model_validate(data))
    assert not report.ok
    assert any("split_hash" in e for e in report.errors)


# ---------------------------------------------------------------------------
# G.6 -- Missing independent-label details fails.
# ---------------------------------------------------------------------------

def test_single_coded_fails_hard_not_just_a_warning():
    """Upgraded from a warning (pre-Step-3 behaviour) to a hard error --
    single-coded datasets are no longer accepted evidence for approved
    real LFS validation."""
    card = load_complete_card()
    data = card.model_dump()
    data["double_coded"] = False
    report = vrg.validate(make_manifest(), DatasetCard.model_validate(data))
    assert not report.ok
    assert any("double_coded" in e for e in report.errors)


def test_missing_adjudication_process_when_double_coded_fails():
    card = load_complete_card()
    data = card.model_dump()
    data["adjudication_process"] = None
    report = vrg.validate(make_manifest(), DatasetCard.model_validate(data))
    assert not report.ok
    assert any("adjudication_process" in e for e in report.errors)


def test_inter_annotator_agreement_null_is_a_warning_not_a_block():
    """Explicitly nullable per the task spec -- absence warns, never blocks."""
    card = load_complete_card()
    data = card.model_dump()
    data["inter_annotator_agreement"] = None
    report = vrg.validate(make_manifest(), DatasetCard.model_validate(data))
    assert report.ok
    assert any("inter_annotator_agreement" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# G.7 / G.8 -- Synthetic fixtures pass only as
# synthetic_or_operationally_realistic, never as approved real LFS.
# ---------------------------------------------------------------------------

def test_minimal_synthetic_fixture_passes_as_synthetic_label():
    card = load_minimal_synthetic_card()
    report = vrg.validate({"dataset_label": SYNTHETIC_OR_OPERATIONALLY_REALISTIC, "split_name": "dev_v1"}, card)
    assert report.ok


def test_minimal_synthetic_fixture_cannot_be_relabelled_as_approved_real_lfs():
    card = load_minimal_synthetic_card()
    report = vrg.validate(make_manifest(), card)
    assert not report.ok
    assert any("synthetic" in e.lower() for e in report.errors)


def test_synthetic_intake_package_cannot_be_relabelled_as_approved_real_lfs():
    """eval/fixtures/synthetic_lfs_intake_package/dataset_card.json has
    EVERY field filled in (unlike the minimal fixture) -- proving the
    content-aware safeguard, not mere incompleteness, is what blocks it."""
    card = load_intake_package_card()
    report = vrg.validate(make_manifest(), card)
    assert not report.ok
    assert any("synthetic" in e.lower() for e in report.errors)


@pytest.mark.parametrize("marker", ["SYNTHETIC data", "this is FAKE data", "dummy data only", "not real respondents"])
def test_any_synthetic_marker_word_blocks_regardless_of_field(marker):
    card = load_complete_card()
    data = card.model_dump()
    data["permitted_research_purpose"] = marker  # an otherwise-unrelated field
    report = vrg.validate(make_manifest(), DatasetCard.model_validate(data))
    assert not report.ok
    assert any("permitted_research_purpose" in e for e in report.errors)


def test_find_synthetic_marker_returns_none_for_complete_card():
    assert vrg._find_synthetic_marker(load_complete_card()) is None


# ---------------------------------------------------------------------------
# G.9 -- Raw data paths under Git-tracked directories are rejected.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field_name", vrg.PATH_LIKE_FIELDS)
def test_in_repo_path_in_path_like_field_is_rejected(field_name):
    card = load_complete_card()
    data = card.model_dump()
    data[field_name] = "eval/local_catalogues/real_lfs_data.csv"
    report = vrg.validate(make_manifest(), DatasetCard.model_validate(data))
    assert not report.ok
    assert any(field_name in e and "inside this Git repository" in e for e in report.errors)


def test_absolute_in_repo_path_is_rejected():
    card = load_complete_card()
    data = card.model_dump()
    repo_root = vrg._REPO_ROOT
    data["secure_storage_location_description"] = str(repo_root / "eval" / "results" / "real_data.csv")
    report = vrg.validate(make_manifest(), DatasetCard.model_validate(data))
    assert not report.ok
    assert any("secure_storage_location_description" in e for e in report.errors)


def test_prose_description_is_not_flagged_as_a_path():
    """Regression guard: ordinary prose (colons, the occasional slash) must
    never trip the in-repo-path check -- only tokens that actually look
    like a path (drive letter, absolute path, ./ or ../ prefix, or a
    known repo-top-level-dir prefix) are ever resolved."""
    card = load_complete_card()
    data = card.model_dump()
    data["secure_storage_location_description"] = (
        "Example fixture text: encrypted institutional drive, access restricted to named PI, and/or backup."
    )
    report = vrg.validate(make_manifest(), DatasetCard.model_validate(data))
    assert report.ok, report.errors


def test_path_outside_repo_is_not_flagged():
    card = load_complete_card()
    data = card.model_dump()
    data["secure_storage_location_description"] = "D:\\secure_external_drive\\lfs_data.csv"
    report = vrg.validate(make_manifest(), DatasetCard.model_validate(data))
    assert report.ok, report.errors


class TestPathPointsInsideRepoUnit:
    def test_plain_prose_false(self):
        assert vrg._path_points_inside_repo("Example fixture text: see the accompanying manifest.") is False

    def test_empty_false(self):
        assert vrg._path_points_inside_repo("") is False

    def test_repo_relative_true(self):
        assert vrg._path_points_inside_repo("eval/local_catalogues/real_data.csv") is True

    def test_windows_repo_absolute_true(self):
        inside = str(vrg._REPO_ROOT / "eval" / "foo.csv")
        assert vrg._path_points_inside_repo(inside) is True

    def test_external_absolute_false(self):
        assert vrg._path_points_inside_repo("/mnt/external_secure_drive/lfs.csv") is False


# ---------------------------------------------------------------------------
# Manuscript wording guard
# ---------------------------------------------------------------------------

def test_check_manuscript_wording_flags_banned_phrase_for_non_approved_label():
    hits = vrg.check_manuscript_wording(
        "This system was validated on real LFS data.", SYNTHETIC_OR_OPERATIONALLY_REALISTIC,
    )
    assert hits


def test_check_manuscript_wording_allows_banned_phrase_for_approved_label():
    hits = vrg.check_manuscript_wording(
        "This system was validated on real LFS data.", APPROVED_REAL_LFS_VALIDATION,
    )
    assert hits == []


def test_check_manuscript_wording_safe_text_never_flagged():
    hits = vrg.check_manuscript_wording(
        "This system was evaluated on a synthetic test set.", SYNTHETIC_OR_OPERATIONALLY_REALISTIC,
    )
    assert hits == []


# ---------------------------------------------------------------------------
# DatasetCard schema itself
# ---------------------------------------------------------------------------

def test_minimal_synthetic_fixture_parses_as_valid_dataset_card():
    card = load_minimal_synthetic_card()
    assert card.source.startswith("SYNTHETIC")
    assert card.consent_obtained is True


def test_dataset_card_requires_core_fields():
    with pytest.raises(Exception):
        DatasetCard.model_validate({})


def test_dataset_card_new_fields_are_optional_minimal_card_still_constructs():
    """Backward compatibility: every field added in the Step 3 pass is
    Optional -- a card with only the ORIGINAL required fields must still
    construct without error."""
    card = load_minimal_synthetic_card()
    assert card.dataset_id is None
    assert card.direct_identifiers_removed_confirmed is None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_passes_on_complete_fixture(tmp_path, monkeypatch, capsys):
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(json.dumps(make_manifest()) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "validate_real_lfs_governance.py",
        "--manifest", str(manifest_path),
        "--dataset-card", str(_COMPLETE_FIXTURE_PATH),
    ])
    vrg.main()  # must not raise / SystemExit(0) implicitly by not calling sys.exit
    out = capsys.readouterr().out
    assert "PASS" in out


def test_cli_fails_without_dataset_card(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(json.dumps(make_manifest()) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "validate_real_lfs_governance.py",
        "--manifest", str(manifest_path),
    ])
    with pytest.raises(SystemExit) as exc_info:
        vrg.main()
    assert exc_info.value.code == 1


def test_cli_passes_on_non_approved_label_without_card(tmp_path, monkeypatch, capsys):
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(json.dumps({"dataset_label": SYNTHETIC_OR_OPERATIONALLY_REALISTIC, "split_name": "dev_v1"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "validate_real_lfs_governance.py",
        "--manifest", str(manifest_path),
    ])
    vrg.main()
    out = capsys.readouterr().out
    assert "PASS" in out


def test_cli_rejects_unrecognised_label(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(json.dumps({"dataset_label": "real_world", "split_name": "dev_v1"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "validate_real_lfs_governance.py",
        "--manifest", str(manifest_path),
    ])
    with pytest.raises(SystemExit):
        vrg.main()


def test_cli_error_output_never_contains_respondent_data_marker(tmp_path, monkeypatch, capsys):
    """The failure output identifies missing FIELDS, never any respondent
    data -- since DatasetCard never holds respondent records, this is
    verified by construction, but this test pins the observable contract:
    error strings always start with 'DatasetCard.<field_name>'."""
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(json.dumps(make_manifest()) + "\n", encoding="utf-8")
    incomplete_card_path = tmp_path / "incomplete_card.json"
    data = load_complete_card().model_dump()
    data["ethics_approval_ref"] = None
    incomplete_card_path.write_text(json.dumps(data), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", [
        "validate_real_lfs_governance.py",
        "--manifest", str(manifest_path),
        "--dataset-card", str(incomplete_card_path),
    ])
    with pytest.raises(SystemExit):
        vrg.main()
    out = capsys.readouterr().out
    error_lines = [line for line in out.splitlines() if line.strip().startswith("ERROR:")]
    assert error_lines
    for line in error_lines:
        assert "DatasetCard." in line
