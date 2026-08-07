"""
Tests for Step 5's new evaluation_status="measured_synthetic_fixture_only"
value and the manuscript_eligible field (Conference I Reviewer #2 response,
Step 5: "execute a controlled evaluation and ablation study using a
user-supplied, locally stored, independently labelled dataset" -- run
against the existing eval/fixtures/synthetic_lfs_intake_package/ fixture
only, per the user's explicit "synthetic fixture integration run only"
instruction).

manuscript_eligible is computed automatically inside build_manifest() from
the FINAL evaluation_status and dataset_label -- never caller-supplied --
so these tests assert the computed value, not a passed-in one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import manifest as mf  # noqa: E402
from dataset_card_schema import APPROVED_REAL_LFS_VALIDATION, DEFAULT_DATASET_LABEL  # noqa: E402

BASE_KWARGS = dict(
    run_id="run-1", classifier_method="isco_hierarchical_rag",
    split_name="heldout", dataset_version_hash="abc123",
)


def test_measured_synthetic_fixture_only_is_a_valid_evaluation_status():
    assert "measured_synthetic_fixture_only" in mf.EVALUATION_STATUSES


def test_measured_synthetic_fixture_only_accepted_by_build_manifest():
    manifest = mf.build_manifest([], **BASE_KWARGS, evaluation_status="measured_synthetic_fixture_only")
    assert manifest.evaluation_status == "measured_synthetic_fixture_only"


def test_measured_synthetic_fixture_only_is_never_manuscript_eligible():
    manifest = mf.build_manifest(
        [{"end_to_end_latency_ms": "50.0", "estimated_cost_usd": "0.0"}],
        **BASE_KWARGS, evaluation_status="measured_synthetic_fixture_only",
        dataset_label=DEFAULT_DATASET_LABEL,
    )
    assert manifest.manuscript_eligible is False


def test_synthetic_fixture_status_stays_ineligible_even_with_approved_label_string():
    """Defence in depth: even if a caller mistakenly passed the approved-real
    label alongside a synthetic-fixture-only run, manuscript_eligible must
    still be False (it requires evaluation_status == 'measured' exactly).
    This scenario would also fail governance validation (no real DatasetCard
    supplied) and raise before reaching this point in the default
    on_governance_failure='raise' mode -- tested via on_governance_failure=
    'downgrade' here specifically to inspect the resulting manifest instead
    of catching an exception."""
    manifest = mf.build_manifest(
        [], **BASE_KWARGS, evaluation_status="measured_synthetic_fixture_only",
        dataset_label=APPROVED_REAL_LFS_VALIDATION, dataset_card=None,
        on_governance_failure="downgrade",
    )
    assert manifest.manuscript_eligible is False


def test_plain_measured_status_with_synthetic_label_is_not_manuscript_eligible():
    """A normal 'measured' run against merely synthetic/operationally-
    realistic data (dataset_label default) must also never be
    manuscript_eligible -- only measured + approved_real_lfs_validation is."""
    manifest = mf.build_manifest(
        [{"end_to_end_latency_ms": "50.0", "estimated_cost_usd": "0.0"}],
        **BASE_KWARGS, evaluation_status="measured", dataset_label=DEFAULT_DATASET_LABEL,
    )
    assert manifest.manuscript_eligible is False


def test_dry_run_is_not_manuscript_eligible():
    manifest = mf.build_manifest([], **BASE_KWARGS, evaluation_status="dry_run_not_measured")
    assert manifest.manuscript_eligible is False


def test_invalid_evaluation_status_still_rejected():
    with pytest.raises(ValueError, match="evaluation_status"):
        mf.build_manifest([], **BASE_KWARGS, evaluation_status="bogus_status")


def test_ablation_runner_evaluation_status_override_reaches_manifest(tmp_path, monkeypatch):
    """eval/ablation_runner.py::run_config()'s evaluation_status_override
    param (Step 5) must reach the built manifest unchanged when supplied,
    independent of --dry-run."""
    import csv
    from types import SimpleNamespace
    import ablation_runner as ar

    monkeypatch.setattr(ar, "RAW_RUNS_DIR", tmp_path / "raw_runs")
    monkeypatch.setattr(ar, "DEV_SELECTION_DIR", tmp_path / "dev_selection")

    csv_path = tmp_path / "raw_runs" / "case_result.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case_id", "end_to_end_latency_ms", "escalation_triggered", "reranker_fired", "estimated_cost_usd", "peak_memory_mb"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow({"case_id": "c0", "end_to_end_latency_ms": "10.0", "escalation_triggered": "False", "reranker_fired": "True", "estimated_cost_usd": "0.0", "peak_memory_mb": ""})

    test_set = tmp_path / "test_set.csv"
    test_set.write_text("case_id,input_text\nc0,x\n", encoding="utf-8")

    def runner(argv, capture_output, text):
        return SimpleNamespace(returncode=0, stdout=f"Wrote 1 row(s) to {csv_path}", stderr="")

    _, manifest = ar.run_config(
        "flat_baseline", test_set, "heldout", reranker_model="m",
        subprocess_runner=runner, evaluation_status_override="measured_synthetic_fixture_only",
    )
    assert manifest.evaluation_status == "measured_synthetic_fixture_only"
    assert manifest.manuscript_eligible is False
