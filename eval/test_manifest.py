"""
Tests for eval/manifest.py -- Section D of the Conference I Reviewer #2
response (evaluation/reproducibility instrumentation, reviewer comment 4).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import manifest as mf  # noqa: E402


def make_row(**overrides):
    row = {
        "end_to_end_latency_ms": "100.0",
        "peak_memory_mb": "",
        "reranker_fired": "True",
        "escalation_triggered": "False",
        "estimated_cost_usd": "0.0",
    }
    row.update(overrides)
    return row


BASE_KWARGS = dict(
    run_id="run-1", classifier_method="isco_hierarchical_rag",
    split_name="dev_v1", dataset_version_hash="abc123",
)


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------

def test_percentile_p50_of_sorted_list():
    assert mf._percentile([10, 20, 30, 40, 50], 50) == 30


def test_percentile_p95_of_sorted_list():
    assert mf._percentile(list(range(1, 101)), 95) == 95


def test_percentile_empty_raises():
    with pytest.raises(ValueError):
        mf._percentile([], 50)


# ---------------------------------------------------------------------------
# build_manifest: latency aggregation
# ---------------------------------------------------------------------------

def test_latency_mean_and_percentiles_computed():
    rows = [make_row(end_to_end_latency_ms=str(v)) for v in [100, 200, 300, 400, 500]]
    m = mf.build_manifest(rows, **BASE_KWARGS)
    assert m.latency_mean_ms == 300.0
    assert m.latency_p50_ms == 300.0
    assert m.latency_unavailable_reason is None


def test_latency_unavailable_when_no_rows_have_it():
    rows = [make_row(end_to_end_latency_ms="")]
    m = mf.build_manifest(rows, **BASE_KWARGS)
    assert m.latency_mean_ms is None
    assert m.latency_unavailable_reason is not None
    assert m.throughput_unavailable_reason is not None


def test_throughput_computed_from_latencies():
    rows = [make_row(end_to_end_latency_ms="1000.0") for _ in range(5)]  # 5 cases, 1s each -> 5s total
    m = mf.build_manifest(rows, **BASE_KWARGS)
    assert m.throughput_cases_per_sec == pytest.approx(1.0)  # 5 cases / 5s


# ---------------------------------------------------------------------------
# build_manifest: peak process memory -- honest gap
# ---------------------------------------------------------------------------

def test_peak_process_memory_unavailable_when_never_populated():
    rows = [make_row(peak_memory_mb="") for _ in range(3)]
    m = mf.build_manifest(rows, **BASE_KWARGS)
    assert m.peak_process_memory_mb is None
    assert "peak_memory_mb" in m.peak_process_memory_unavailable_reason


def test_peak_process_memory_computed_when_present():
    rows = [make_row(peak_memory_mb=v) for v in ["100.0", "250.5", "80.0"]]
    m = mf.build_manifest(rows, **BASE_KWARGS)
    assert m.peak_process_memory_mb == 250.5
    assert m.peak_process_memory_unavailable_reason is None


# ---------------------------------------------------------------------------
# build_manifest: retrieval_count is always "not measured" (documented gap)
# ---------------------------------------------------------------------------

def test_retrieval_count_always_unavailable():
    rows = [make_row()]
    m = mf.build_manifest(rows, **BASE_KWARGS)
    assert m.retrieval_count is None
    assert m.retrieval_count_unavailable_reason is not None


# ---------------------------------------------------------------------------
# build_manifest: reranker invocation count
# ---------------------------------------------------------------------------

def test_reranker_invocation_count_counts_true_values():
    rows = [make_row(reranker_fired=v) for v in ["True", "False", "True", "True"]]
    m = mf.build_manifest(rows, **BASE_KWARGS)
    assert m.reranker_invocation_count == 3


def test_reranker_invocation_count_none_when_column_absent():
    rows = [{"end_to_end_latency_ms": "100"}]
    m = mf.build_manifest(rows, **BASE_KWARGS)
    assert m.reranker_invocation_count is None


# ---------------------------------------------------------------------------
# build_manifest: hitl_escalation_rate (harness heuristic, not production)
# ---------------------------------------------------------------------------

def test_hitl_escalation_rate_computed():
    rows = [make_row(escalation_triggered=v) for v in ["True", "False", "False", "False"]]
    m = mf.build_manifest(rows, **BASE_KWARGS)
    assert m.hitl_escalation_rate == pytest.approx(0.25)


def test_hitl_escalation_rate_unavailable_when_all_empty():
    rows = [make_row(escalation_triggered="")]
    m = mf.build_manifest(rows, **BASE_KWARGS)
    assert m.hitl_escalation_rate is None
    assert m.hitl_escalation_rate_unavailable_reason is not None


# ---------------------------------------------------------------------------
# build_manifest: cost
# ---------------------------------------------------------------------------

def test_estimated_cost_summed_when_present():
    rows = [make_row(estimated_cost_usd=v) for v in ["0.01", "0.02", "0.005"]]
    m = mf.build_manifest(rows, **BASE_KWARGS)
    assert m.estimated_cost_usd == pytest.approx(0.035)
    assert m.estimated_cost_method is not None


def test_estimated_cost_zero_is_a_real_zero_not_missing():
    rows = [make_row(estimated_cost_usd="0.0") for _ in range(3)]
    m = mf.build_manifest(rows, **BASE_KWARGS)
    assert m.estimated_cost_usd == 0.0
    assert m.estimated_cost_method is not None


# ---------------------------------------------------------------------------
# build_manifest: empty case_rows
# ---------------------------------------------------------------------------

def test_empty_case_rows_all_unavailable():
    m = mf.build_manifest([], **BASE_KWARGS)
    assert m.n_cases == 0
    assert m.latency_unavailable_reason is not None
    assert m.throughput_unavailable_reason is not None
    assert m.peak_process_memory_unavailable_reason is not None
    assert m.retrieval_count_unavailable_reason is not None
    assert m.hitl_escalation_rate_unavailable_reason is not None


# ---------------------------------------------------------------------------
# build_manifest: metadata fields
# ---------------------------------------------------------------------------

def test_manifest_carries_through_metadata():
    rows = [make_row()]
    m = mf.build_manifest(
        rows, run_id="r1", classifier_method="isco_hierarchical_rag",
        split_name="heldout", dataset_version_hash="deadbeef",
        dataset_label="synthetic_or_operationally_realistic", language_filter="ar",
        model_versions={"reranker": "claude-3-5-sonnet"},
        retrieval_params={"beam": 2}, llm_params={"temperature": 0.0},
        git_commit="abc1234", utc_timestamp="2026-01-01T00:00:00+00:00",
    )
    assert m.run_id == "r1"
    assert m.split_name == "heldout"
    assert m.dataset_label == "synthetic_or_operationally_realistic"
    assert m.language_filter == "ar"
    assert m.model_versions == {"reranker": "claude-3-5-sonnet"}
    assert m.git_commit == "abc1234"
    assert m.utc_timestamp == "2026-01-01T00:00:00+00:00"


def test_manifest_default_dataset_label_is_synthetic_or_operationally_realistic():
    m = mf.build_manifest([make_row()], **BASE_KWARGS)
    assert m.dataset_label == "synthetic_or_operationally_realistic"
    assert m.dataset_label == mf.DEFAULT_DATASET_LABEL


def test_manifest_rejects_unrecognised_dataset_label():
    with pytest.raises(ValueError, match="synthetic_or_operationally_realistic"):
        mf.build_manifest([make_row()], **BASE_KWARGS, dataset_label="real_world")


def test_manifest_governance_validation_errors_defaults_to_empty_list():
    m = mf.build_manifest([make_row()], **BASE_KWARGS)
    assert m.governance_validation_errors == []


def test_manifest_hardware_and_dependency_versions_populated():
    m = mf.build_manifest([make_row()], **BASE_KWARGS)
    assert isinstance(m.hardware, mf.HardwareInfo)
    assert m.hardware.cpu_count is not None
    assert isinstance(m.dependency_versions, dict)
    assert len(m.dependency_versions) > 0


# ---------------------------------------------------------------------------
# Hardware probe never raises
# ---------------------------------------------------------------------------

def test_probe_hardware_does_not_raise():
    hw = mf.probe_hardware()
    assert isinstance(hw, mf.HardwareInfo)
    # On a GPU-less CI/dev box, gpu_mem_gb is None with a reason -- never fabricated
    if hw.gpu_mem_gb is None:
        assert hw.gpu_unavailable_reason is not None


def test_git_commit_hash_does_not_raise():
    result = mf.git_commit_hash()
    assert isinstance(result, str)


def test_sha256_of_missing_file_is_empty_string(tmp_path):
    assert mf.sha256_of_file(tmp_path / "does_not_exist.csv") == ""


def test_sha256_of_file_is_64_hex_chars(tmp_path):
    p = tmp_path / "f.csv"
    p.write_text("hello", encoding="utf-8")
    assert len(mf.sha256_of_file(p)) == 64


# ---------------------------------------------------------------------------
# write_manifest: CSV + JSONL output
# ---------------------------------------------------------------------------

def test_write_manifest_produces_csv_and_jsonl(tmp_path):
    m = mf.build_manifest([make_row()], **BASE_KWARGS)
    csv_path, jsonl_path = mf.write_manifest(m, tmp_path)
    assert csv_path.exists()
    assert jsonl_path.exists()


def test_write_manifest_jsonl_round_trips_full_structure(tmp_path):
    m = mf.build_manifest([make_row()], **BASE_KWARGS)
    _, jsonl_path = mf.write_manifest(m, tmp_path)
    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["run_id"] == "run-1"
    assert payload["hardware"]["cpu_count"] == m.hardware.cpu_count


def test_write_manifest_csv_flattens_hardware_and_json_fields(tmp_path):
    import csv as csv_mod
    m = mf.build_manifest([make_row()], **BASE_KWARGS)
    csv_path, _ = mf.write_manifest(m, tmp_path)
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv_mod.DictReader(f))
    assert len(rows) == 1
    assert "hardware_cpu_count" in rows[0]
    assert "hardware" not in rows[0]


# ---------------------------------------------------------------------------
# build_manifest: approved_real_lfs_validation governance fail-fast gate
# (Section F, expanded in the Step 3 real-LFS-intake hardening pass)
# ---------------------------------------------------------------------------

_COMPLETE_CARD_PATH = Path(__file__).resolve().parent / "fixtures" / "complete_approved_real_lfs_dataset_card_example.json"
_MINIMAL_SYNTHETIC_CARD_PATH = Path(__file__).resolve().parent / "fixtures" / "synthetic_lfs_dataset_card_example.json"


def _load_complete_card():
    from dataset_card_schema import DatasetCard
    return DatasetCard.model_validate_json(_COMPLETE_CARD_PATH.read_text(encoding="utf-8"))


def _load_minimal_synthetic_card():
    from dataset_card_schema import DatasetCard
    return DatasetCard.model_validate_json(_MINIMAL_SYNTHETIC_CARD_PATH.read_text(encoding="utf-8"))


def test_build_manifest_blocks_approved_real_lfs_label_without_card_default_raise():
    with pytest.raises(ValueError, match="approved_real_lfs_validation"):
        mf.build_manifest([make_row()], **BASE_KWARGS, dataset_label="approved_real_lfs_validation")


def test_build_manifest_succeeds_with_complete_card():
    card = _load_complete_card()
    m = mf.build_manifest(
        [make_row()], **BASE_KWARGS, dataset_label="approved_real_lfs_validation", dataset_card=card,
    )
    assert m.dataset_label == "approved_real_lfs_validation"
    assert m.governance_validation_errors == []


def test_build_manifest_rejects_synthetic_card_for_approved_label_default_raise():
    """Backward-compatible default ('raise'): a synthetic-marked card must
    never silently produce an approved_real_lfs_validation manifest."""
    card = _load_minimal_synthetic_card()
    with pytest.raises(ValueError):
        mf.build_manifest([make_row()], **BASE_KWARGS, dataset_label="approved_real_lfs_validation", dataset_card=card)


def test_build_manifest_non_approved_label_ignores_missing_card():
    m = mf.build_manifest([make_row()], **BASE_KWARGS, dataset_label="synthetic_or_operationally_realistic", dataset_card=None)
    assert m.dataset_label == "synthetic_or_operationally_realistic"


# ---------------------------------------------------------------------------
# build_manifest: on_governance_failure="downgrade" -- the third, visible
# outcome state (invalid_incomplete_governance), Step 3 addition
# ---------------------------------------------------------------------------

def test_downgrade_mode_does_not_raise_and_labels_invalid_incomplete_governance():
    card = _load_minimal_synthetic_card()
    m = mf.build_manifest(
        [make_row()], **BASE_KWARGS, dataset_label="approved_real_lfs_validation",
        dataset_card=card, on_governance_failure="downgrade",
    )
    assert m.dataset_label == "invalid_incomplete_governance"
    assert m.governance_validation_errors  # non-empty -- records exactly why


def test_downgrade_mode_error_list_matches_validator_output():
    card = _load_minimal_synthetic_card()
    m = mf.build_manifest(
        [make_row()], **BASE_KWARGS, dataset_label="approved_real_lfs_validation",
        dataset_card=card, on_governance_failure="downgrade",
    )
    assert any("synthetic" in e.lower() for e in m.governance_validation_errors)


def test_downgrade_mode_never_labels_approved_when_validation_passes():
    """Downgrade mode does not change behaviour on a SUCCESSFUL validation
    -- the requested label is used as-is."""
    card = _load_complete_card()
    m = mf.build_manifest(
        [make_row()], **BASE_KWARGS, dataset_label="approved_real_lfs_validation",
        dataset_card=card, on_governance_failure="downgrade",
    )
    assert m.dataset_label == "approved_real_lfs_validation"


def test_invalid_on_governance_failure_value_rejected():
    with pytest.raises(ValueError, match="on_governance_failure"):
        mf.build_manifest([make_row()], **BASE_KWARGS, on_governance_failure="ignore")
