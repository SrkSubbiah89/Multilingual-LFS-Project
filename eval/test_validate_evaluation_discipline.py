"""
Tests for eval/validate_evaluation_discipline.py and eval/split_manifest_schema.py
-- Conference I Reviewer #2 response, Step 4 (evaluation-readiness pass),
task D.4: "Add tests that reject: a missing split manifest; identical
development and test split IDs; absent dataset hash; completed status with
only null metrics; approved-real-LFS label without passing governance
validation; a synthetic run represented as manuscript-ready real
validation."

The last two scenarios are already enforced elsewhere (eval/manifest.py's
build_manifest() / eval/ablation_runner.py's run_config() for the
governance gate; eval/validate_real_lfs_governance.py's
check_manuscript_wording() for the manuscript-wording guard) -- this file
re-exercises them here, together with the four genuinely new checks, so all
six scenarios from the task spec are provably covered in one place.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import manifest as mf  # noqa: E402
import validate_evaluation_discipline as ved  # noqa: E402
import validate_real_lfs_governance as vrg  # noqa: E402
from split_manifest_schema import SplitEntry, SplitManifest  # noqa: E402
from dataset_card_schema import APPROVED_REAL_LFS_VALIDATION, DatasetCard  # noqa: E402

BASE_KWARGS = dict(
    run_id="run-1", classifier_method="isco_hierarchical_rag",
    split_name="dev_v1", dataset_version_hash="abc123",
)


def _valid_split_manifest(dev_id="DEV-1", heldout_id="HELDOUT-1") -> SplitManifest:
    return SplitManifest(
        dataset_id="SYN-1",
        split_hash="deadbeef",
        splits={
            "dev": SplitEntry(split_id=dev_id, purpose="parameter selection only", n_cases=10, frozen=False),
            "heldout": SplitEntry(split_id=heldout_id, purpose="frozen confirmation set", n_cases=20, frozen=True),
        },
        leakage_check_performed=True,
        leakage_check_method="no case_id overlap between splits",
    )


# ---------------------------------------------------------------------------
# 1. Missing split manifest
# ---------------------------------------------------------------------------

def test_missing_split_manifest_is_rejected():
    report = ved.validate_split_manifest(None)
    assert not report.ok
    assert any("missing" in e for e in report.errors)


def test_split_manifest_missing_dev_key_is_rejected():
    sm = SplitManifest(
        splits={"heldout": SplitEntry(split_id="H-1", purpose="frozen confirmation set", n_cases=20, frozen=True)},
        leakage_check_performed=True,
    )
    report = ved.validate_split_manifest(sm)
    assert not report.ok
    assert any("'dev'" in e for e in report.errors)


def test_valid_split_manifest_passes():
    report = ved.validate_split_manifest(_valid_split_manifest())
    assert report.ok
    assert report.errors == []


# ---------------------------------------------------------------------------
# 2. Identical development and test split IDs
# ---------------------------------------------------------------------------

def test_identical_dev_and_heldout_split_ids_are_rejected():
    sm = _valid_split_manifest(dev_id="SAME-ID", heldout_id="SAME-ID")
    report = ved.validate_split_manifest(sm)
    assert not report.ok
    assert any("identical" in e for e in report.errors)


def test_blank_split_id_is_rejected():
    sm = _valid_split_manifest(dev_id="", heldout_id="HELDOUT-1")
    report = ved.validate_split_manifest(sm)
    assert not report.ok
    assert any("blank split_id" in e for e in report.errors)


# ---------------------------------------------------------------------------
# 3. Absent dataset hash
# ---------------------------------------------------------------------------

def test_absent_dataset_hash_is_rejected():
    manifest = mf.build_manifest([], **{**BASE_KWARGS, "dataset_version_hash": ""})
    report = ved.validate_manifest_discipline(manifest)
    assert not report.ok
    assert any("dataset_version_hash" in e for e in report.errors)


def test_present_dataset_hash_passes_that_check():
    manifest = mf.build_manifest(
        [{"end_to_end_latency_ms": "100.0", "estimated_cost_usd": "0.0"}], **BASE_KWARGS,
    )
    report = ved.validate_manifest_discipline(manifest)
    assert not any("dataset_version_hash" in e for e in report.errors)


# ---------------------------------------------------------------------------
# 4. "measured" status with only null metrics
# ---------------------------------------------------------------------------

def test_measured_status_with_all_null_metrics_is_rejected():
    manifest = mf.build_manifest([], **BASE_KWARGS)  # evaluation_status defaults to "measured"; no case rows -> everything null
    assert manifest.evaluation_status == "measured"
    report = ved.validate_manifest_discipline(manifest)
    assert not report.ok
    assert any("null" in e and "measured" in e for e in report.errors)


def test_measured_status_with_a_real_metric_passes_that_check():
    manifest = mf.build_manifest(
        [{"end_to_end_latency_ms": "100.0", "estimated_cost_usd": "0.0"}], **BASE_KWARGS,
    )
    report = ved.validate_manifest_discipline(manifest)
    assert not any("only" in e and "null" in e for e in report.errors)


def test_dry_run_status_with_all_null_metrics_is_not_rejected_by_this_check():
    """A dry run legitimately has every metric null -- only evaluation_status
    == 'measured' claiming a real result with nothing to show is a problem."""
    manifest = mf.build_manifest([], **BASE_KWARGS, evaluation_status="dry_run_not_measured")
    report = ved.validate_manifest_discipline(manifest)
    assert not any("only" in e and "null" in e for e in report.errors)


# ---------------------------------------------------------------------------
# 5. approved-real-LFS label without passing governance validation
# (already enforced by eval/manifest.py's build_manifest(); re-exercised here)
# ---------------------------------------------------------------------------

def test_approved_real_lfs_label_without_governance_is_rejected():
    with pytest.raises(vrg.GovernanceError):
        mf.build_manifest(
            [], **{**BASE_KWARGS, "dataset_label": APPROVED_REAL_LFS_VALIDATION},
            dataset_card=None, on_governance_failure="raise",
        )


# ---------------------------------------------------------------------------
# 6. A synthetic run represented as manuscript-ready real validation
# (already enforced by eval/validate_real_lfs_governance.check_manuscript_wording();
# re-exercised here)
# ---------------------------------------------------------------------------

def test_synthetic_run_flagged_as_manuscript_ready_real_validation_is_rejected():
    banned_phrases_found = vrg.check_manuscript_wording(
        text="Our system was validated on real LFS data with 91% accuracy.",
        label="synthetic_or_operationally_realistic",
    )
    assert banned_phrases_found  # non-empty: the banned phrase was caught


def test_approved_real_lfs_label_permits_the_same_wording():
    banned_phrases_found = vrg.check_manuscript_wording(
        text="Our system was validated on real LFS data with 91% accuracy.",
        label=APPROVED_REAL_LFS_VALIDATION,
    )
    assert banned_phrases_found == []
