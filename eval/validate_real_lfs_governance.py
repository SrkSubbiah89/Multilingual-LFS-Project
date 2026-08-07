"""
eval/validate_real_lfs_governance.py

Blocking governance validator for datasets/runs labelled
"approved_real_lfs_validation" (Conference I Reviewer #2 response,
Section F; expanded in the Step 3 real-LFS-intake hardening pass). Directly
answers reviewer comment 3 (the manuscript lacks validation on real Labour
Force Survey data) by making it structurally impossible to label a run
"approved real LFS validation" without also supplying complete
data-governance documentation across five groups: dataset identity,
governance/permissions, classification-label provenance, evaluation-split
discipline, and privacy posture.

NO real respondent data is added anywhere by this module -- it only
validates governance metadata (a DatasetCard) against a run manifest. Every
error message below names a FIELD, never a VALUE -- this module cannot leak
respondent data because DatasetCard never contains any (see
dataset_card_schema.py's module docstring). See
Documentation/Conference_I_Reviewer_2/REAL_LFS_VALIDATION_DATASET_CARD_TEMPLATE.md
for the human-facing template,
Documentation/Conference_I_Reviewer_2/REAL_LFS_DATA_INTAKE_CHECKLIST.md for
the step-by-step custodian checklist, and eval/fixtures/ for SYNTHETIC
examples used only by tests.

Closed label vocabulary
--------------------------
See dataset_card_schema.py's DATASET_LABELS. This module only performs
governance checks for the label APPROVED_REAL_LFS_VALIDATION; any other
value in DATASET_LABELS is accepted with no governance requirement (a
warning only); any string NOT in DATASET_LABELS is rejected outright by
this module's callers (eval/manifest.py, eval/ablation_runner.py) before
this validator is even reached -- there is no way to set an unrecognised
label and have it silently pass through.

Content-aware safeguards (Step 3 additions)
----------------------------------------------
Presence-only checks are not enough: a fixture card could have every
REQUIRED field filled in with placeholder/synthetic text and still "pass"
a naive presence check. Two additional checks close this gap:

  - `_find_synthetic_marker()`: scans every string field for a marker word
    (SYNTHETIC, FAKE, DUMMY, "not real", "no real respondent", ...) and
    blocks approved_real_lfs_validation outright if found, regardless of
    how many other fields are filled in.
  - `_path_points_inside_repo()`: applied to the fields that describe
    where the real dataset lives (secure_storage_location_description,
    split_manifest_ref, parameter_selection_dataset_ref) -- blocks if any
    of them resolves to a path inside this Git repository, so a submission
    can never claim its real data lives in a Git-tracked location.

Reuses eval/validate_dev_set.py's pattern: a pure ``validate()`` function
returning a report with ``.ok``/errors/warnings (no filesystem/CLI
concerns), plus a thin CLI wrapper that exits non-zero on any hard error.
Called from two places (belt-and-braces): eval/manifest.py's
``build_manifest()`` calls this at manifest-creation time (fail fast, or
downgrade to invalid_incomplete_governance -- see that module); this
module's own CLI is also runnable standalone against an already-written
manifest file (fail late, for iterative governance-document review).

Usage
-----
    python eval/validate_real_lfs_governance.py --manifest path/to/manifest.jsonl --dataset-card path/to/card.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset_card_schema import (  # noqa: E402
    APPROVED_REAL_LFS_VALIDATION,
    DATASET_LABELS,
    DatasetCard,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Backward-compat alias -- the old label string this module used before the
# Step 3 rename. Kept only as a name so any stray reference resolves to the
# new canonical constant rather than the retired string.
REAL_LFS_LABEL = APPROVED_REAL_LFS_VALIDATION

# ---------------------------------------------------------------------------
# Required-field lists, grouped exactly per the Step 3 field-group spec.
# Every field below is Optional on DatasetCard itself (dataset_card_schema.py)
# -- required-ness lives HERE, scoped to approved_real_lfs_validation only,
# so synthetic/dev-selection cards never need to fill any of this in.
# ---------------------------------------------------------------------------

# Fields required (non-blank) at the DatasetCard level regardless of label
# -- unchanged from before this pass (these were already Pydantic-required).
REQUIRED_CARD_FIELDS = [
    "source", "region", "wave", "languages", "collection_start_date",
    "collection_end_date", "labelling_process", "coder_qualifications",
    "classification_standard_version",
]

REQUIRED_FOR_APPROVED_REAL_LFS_IDENTITY = [
    "dataset_id", "data_owner_or_custodian", "survey_programme_name",
    "dataset_version", "dataset_hash",
]
REQUIRED_FOR_APPROVED_REAL_LFS_GOVERNANCE = [
    "data_sharing_agreement_ref", "ethics_approval_ref", "lawful_basis_description",
    "permitted_research_purpose", "retention_period", "access_control_description",
    "deidentification_method", "deidentification_status", "prohibited_disclosure_rules",
]
REQUIRED_FOR_APPROVED_REAL_LFS_LABELS = [
    "isco08_version", "isic_rev4_version", "isced2011_version", "iscedf2013_version",
    "label_source", "labeler_type",
]
REQUIRED_FOR_APPROVED_REAL_LFS_SPLIT = [
    "split_manifest_ref", "split_hash", "frozen_heldout_test_status",
    "parameter_selection_dataset_ref", "test_set_access_restriction",
    "missing_data_policy", "exclusion_criteria",
]
REQUIRED_FOR_APPROVED_REAL_LFS_PRIVACY = [
    "quasi_identifier_risk_note", "free_text_redaction_method",
    "secure_storage_location_description",
]

# All scalar/string fields above, flattened, for convenience (used by the
# generic blank-check loop).
REQUIRED_FOR_APPROVED_REAL_LFS = (
    REQUIRED_FOR_APPROVED_REAL_LFS_IDENTITY
    + REQUIRED_FOR_APPROVED_REAL_LFS_GOVERNANCE
    + REQUIRED_FOR_APPROVED_REAL_LFS_LABELS
    + REQUIRED_FOR_APPROVED_REAL_LFS_SPLIT
    + REQUIRED_FOR_APPROVED_REAL_LFS_PRIVACY
)

# Booleans that must be explicitly True (not just non-None) for
# approved_real_lfs_validation.
REQUIRED_TRUE_FOR_APPROVED_REAL_LFS = [
    "direct_identifiers_removed_confirmed",
    "raw_records_excluded_from_git_confirmed",
]

# Path-shaped fields checked against "does this resolve inside this Git repo".
PATH_LIKE_FIELDS = [
    "secure_storage_location_description", "split_manifest_ref", "parameter_selection_dataset_ref",
]

# Retained name (pre-Step-3): the two fields that were already
# required-for-real-data before this pass.
REQUIRED_FOR_REAL_DATA = ["ethics_approval_ref", "deidentification_method"]

_SYNTHETIC_MARKERS = (
    "synthetic", "not real", "no real respondent", "fake data", "dummy data",
    "invented for test", "test/demo purposes",
)

BANNED_MANUSCRIPT_PHRASES = [
    "validated on real lfs data",
    "validated on real-world data",
    "real-world validation",
    "validated using real respondent data",
]


@dataclass
class ValidationReport:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


class GovernanceError(ValueError):
    """Raised when a run/manifest claims approved_real_lfs_validation
    without passing this module's validate(). Subclasses ValueError so
    existing ``pytest.raises(ValueError, ...)`` call sites keep working."""


def _find_synthetic_marker(card: DatasetCard) -> Optional[str]:
    """Scans every string field for a marker word indicating this card
    describes synthetic/placeholder data. Returns a field-name-only
    message (never the actual field value) or None."""
    for field_name, value in card.model_dump().items():
        if isinstance(value, str):
            lowered = value.lower()
            for marker in _SYNTHETIC_MARKERS:
                if marker in lowered:
                    return (
                        f"DatasetCard.{field_name} contains a synthetic/placeholder marker "
                        f"({marker!r}) -- this card describes non-real data and can never be "
                        f"labelled {APPROVED_REAL_LFS_VALIDATION!r}, regardless of which other "
                        f"fields are filled in."
                    )
    return None


_DRIVE_OR_ABS_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]|^[\\/]|^\.{1,2}[\\/]")
_REPO_TOP_LEVEL_DIR_RE = re.compile(r"^(eval|backend|Documentation|frontend)[\\/]")


def _path_points_inside_repo(value: str) -> bool:
    """True if any whitespace-separated TOKEN in `value`, interpreted as a
    filesystem path, resolves inside this Git repository.

    Tokenised (rather than "does the whole string contain a slash or
    colon anywhere") specifically because ordinary prose is full of
    colons and the occasional slash ("Example fixture text: ...",
    "and/or") -- a naive substring check on ':' or '/' false-positives on
    completely innocent sentences. Only a token that itself LOOKS like a
    path (drive letter, absolute path, relative './'/'../' prefix, or a
    known repo-top-level-directory prefix like 'eval/') is ever resolved
    and checked against the repo root.
    """
    if not value:
        return False
    for raw_token in value.split():
        token = raw_token.strip("\"'.,;:()[]")
        if not token:
            continue
        if not (_DRIVE_OR_ABS_PATH_RE.search(token) or _REPO_TOP_LEVEL_DIR_RE.search(token)):
            continue
        try:
            candidate = Path(token)
            resolved = (candidate if candidate.is_absolute() else (_REPO_ROOT / token)).resolve()
            if resolved == _REPO_ROOT or _REPO_ROOT in resolved.parents:
                return True
        except Exception:  # noqa: BLE001 - malformed path token, not a repo hit
            continue
    return False


def check_manuscript_wording(text: str, label: str) -> list[str]:
    """Returns the list of banned manuscript phrases found in `text`
    (case-insensitive) that are NOT permitted unless `label` is
    APPROVED_REAL_LFS_VALIDATION. Empty list = safe. See
    REAL_LFS_DATA_INTAKE_CHECKLIST.md's manuscript-wording section."""
    if label == APPROVED_REAL_LFS_VALIDATION:
        return []
    lowered = text.lower()
    return [phrase for phrase in BANNED_MANUSCRIPT_PHRASES if phrase in lowered]


def validate(manifest: dict, card: Optional[DatasetCard]) -> ValidationReport:
    """
    manifest : dict with at least 'dataset_label' and 'split_name' keys
        (matches eval.manifest.ExperimentRunManifest's field names -- a
        plain dict, not the dataclass itself, so this module has no
        import-time dependency on eval.manifest; eval.manifest imports
        THIS module, not the other way around).
    card : the DatasetCard for this dataset, or None if not supplied.

    Only blocks when dataset_label == APPROVED_REAL_LFS_VALIDATION. Any
    other value in DATASET_LABELS is out of scope for this validator --
    governance documentation is not required to label a run as anything
    other than approved real LFS validation. A dataset_label NOT in
    DATASET_LABELS is a caller-side bug (eval/manifest.py and
    eval/ablation_runner.py reject it before this function is ever
    called) -- this function does not re-validate the label's closedness.
    """
    report = ValidationReport()
    label = (manifest or {}).get("dataset_label", "")

    if label != APPROVED_REAL_LFS_VALIDATION:
        report.warnings.append(
            f"dataset_label={label!r} is not {APPROVED_REAL_LFS_VALIDATION!r} -- governance "
            f"checks are only enforced for that label; nothing was validated."
        )
        return report

    if not (manifest or {}).get("split_name"):
        report.errors.append("manifest.split_name is missing -- required for any labelled experiment.")

    if card is None:
        report.errors.append(
            f"dataset_label={APPROVED_REAL_LFS_VALIDATION!r} but no DatasetCard was supplied -- "
            f"a run cannot be labelled approved real LFS validation without governance documentation."
        )
        return report

    # ── Content-aware safeguard: reject synthetic-marked cards outright,
    # before even checking field presence (a fully-filled-out synthetic
    # fixture must never pass just because every field happens to be
    # non-blank) ──────────────────────────────────────────────────────
    synthetic_hit = _find_synthetic_marker(card)
    if synthetic_hit:
        report.errors.append(synthetic_hit)
        return report

    card_dict = card.model_dump()

    def _is_blank(value) -> bool:
        return value in (None, "", [], {})

    # ── Group: fields required regardless of label ─────────────────────
    for f in REQUIRED_CARD_FIELDS:
        if _is_blank(card_dict.get(f)):
            report.errors.append(f"DatasetCard.{f} is missing/blank.")

    # ── Group: identity / governance / labels / split / privacy ────────
    for f in REQUIRED_FOR_APPROVED_REAL_LFS:
        if _is_blank(card_dict.get(f)):
            report.errors.append(
                f"DatasetCard.{f} is missing/blank -- required specifically for "
                f"{APPROVED_REAL_LFS_VALIDATION} given the governance/provenance/annotation/"
                f"privacy/split completeness this label asserts."
            )

    # ── Booleans that must be explicitly True ───────────────────────────
    if card.consent_obtained is not True:
        report.errors.append(f"DatasetCard.consent_obtained must be True for {APPROVED_REAL_LFS_VALIDATION}.")

    for f in REQUIRED_TRUE_FOR_APPROVED_REAL_LFS:
        if card_dict.get(f) is not True:
            report.errors.append(
                f"DatasetCard.{f} must be explicitly True for {APPROVED_REAL_LFS_VALIDATION} "
                f"(missing, False, or unset all block)."
            )

    # ── Independent double-coding: hard requirement for approved real LFS
    # (upgraded from a warning -- see REVIEWER_RESPONSE_IMPLEMENTATION_MATRIX.md
    # / STEP3 audit notes for why this was tightened) ──────────────────
    if card.double_coded is not True:
        report.errors.append(
            f"DatasetCard.double_coded must be True for {APPROVED_REAL_LFS_VALIDATION} -- "
            f"single-coded labels are not accepted evidence for approved real LFS validation."
        )
    elif _is_blank(card_dict.get("adjudication_process")):
        report.errors.append(
            "DatasetCard.adjudication_process is missing/blank -- required when double_coded=True."
        )

    # ── sample_counts_by_language_and_task: dict-shaped, checked separately ──
    if _is_blank(card_dict.get("sample_counts_by_language_and_task")):
        report.errors.append(
            "DatasetCard.sample_counts_by_language_and_task is missing/blank -- "
            f"required for {APPROVED_REAL_LFS_VALIDATION}."
        )

    # ── inter_annotator_agreement: nullable by design (Task B.3) -- a
    # warning only, never blocks ────────────────────────────────────────
    if card_dict.get("inter_annotator_agreement") is None:
        report.warnings.append(
            "DatasetCard.inter_annotator_agreement is null -- acceptable (nullable until "
            "measured), but should be filled in before citing this dataset's label quality "
            "in the manuscript."
        )

    # ── Raw data paths under Git-tracked directories are rejected ──────
    for f in PATH_LIKE_FIELDS:
        value = card_dict.get(f)
        if isinstance(value, str) and _path_points_inside_repo(value):
            report.errors.append(
                f"DatasetCard.{f} appears to point inside this Git repository -- "
                f"real data (or its manifest) must be stored outside version control."
            )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path,
                         help="Path to a manifest JSON or JSONL file (eval.manifest.ExperimentRunManifest); the first line/object is used.")
    parser.add_argument("--dataset-card", type=Path, default=None,
                         help="Path to a DatasetCard JSON file.")
    args = parser.parse_args()

    if not args.manifest.exists():
        parser.error(f"Manifest not found: {args.manifest}")

    first_line = args.manifest.read_text(encoding="utf-8").splitlines()[0]
    manifest_data = json.loads(first_line)

    label = manifest_data.get("dataset_label", "")
    if label and label not in DATASET_LABELS:
        parser.error(
            f"manifest.dataset_label={label!r} is not one of {sorted(DATASET_LABELS)} -- "
            f"refusing to validate an unrecognised label."
        )

    card = None
    if args.dataset_card:
        if not args.dataset_card.exists():
            parser.error(f"Dataset card not found: {args.dataset_card}")
        card = DatasetCard.model_validate_json(args.dataset_card.read_text(encoding="utf-8"))

    report = validate(manifest_data, card)

    if report.warnings:
        print(f"{len(report.warnings)} warning(s):")
        for w in report.warnings:
            print(f"  WARNING: {w}")
    if report.errors:
        print(f"\n{len(report.errors)} error(s):")
        for e in report.errors:
            print(f"  ERROR: {e}")
        print(f"\nFAIL: this run cannot be labelled {APPROVED_REAL_LFS_VALIDATION!r} until the errors above are fixed.")
        sys.exit(1)

    print("\nPASS: governance checks satisfied." + (" (see warnings above)" if report.warnings else ""))


if __name__ == "__main__":
    main()
