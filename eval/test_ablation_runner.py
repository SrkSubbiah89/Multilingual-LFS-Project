"""
Tests for eval/ablation_runner.py -- Section E of the Conference I
Reviewer #2 response (ablation infrastructure). The case-runner
(eval/run_eval.py) is always mocked/stubbed here -- no live Qdrant/Ollama
call is ever made by this test file.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ablation_runner as ar  # noqa: E402


# ---------------------------------------------------------------------------
# CONFIGS / build_argv -- pure config resolution, no execution
# ---------------------------------------------------------------------------

def test_five_named_configs_exist():
    assert set(ar.CONFIGS.keys()) == {
        "flat_baseline", "hierarchical_no_rerank", "hierarchical_with_rerank",
        "no_sre", "with_sre",
    }


def test_flat_baseline_argv():
    argv = ar.build_argv(ar.CONFIGS["flat_baseline"], Path("t.csv"), Path("out"), reranker_model="ollama/llama3.2:1b")
    assert "--system" in argv and argv[argv.index("--system") + 1] == "flat"
    assert "--reranker-model" in argv


def test_hierarchical_no_rerank_argv():
    argv = ar.build_argv(ar.CONFIGS["hierarchical_no_rerank"], Path("t.csv"), Path("out"), reranker_model="m")
    assert argv[argv.index("--use-llm-reranker") + 1] == "off"
    assert argv[argv.index("--system") + 1] == "hierarchical"


def test_hierarchical_with_rerank_argv():
    argv = ar.build_argv(ar.CONFIGS["hierarchical_with_rerank"], Path("t.csv"), Path("out"), reranker_model="m")
    assert argv[argv.index("--use-llm-reranker") + 1] == "on"


def test_no_sre_argv():
    argv = ar.build_argv(ar.CONFIGS["no_sre"], Path("t.csv"), Path("out"), reranker_model="m")
    assert argv[argv.index("--sre") + 1] == "off"


def test_with_sre_argv():
    argv = ar.build_argv(ar.CONFIGS["with_sre"], Path("t.csv"), Path("out"), reranker_model="m")
    assert argv[argv.index("--sre") + 1] == "on"


def test_hierarchical_config_without_reranker_model_raises():
    with pytest.raises(ValueError, match="reranker-model"):
        ar.build_argv(ar.CONFIGS["hierarchical_with_rerank"], Path("t.csv"), Path("out"))


def test_argv_includes_run_id_and_limit_when_given():
    argv = ar.build_argv(ar.CONFIGS["with_sre"], Path("t.csv"), Path("out"), reranker_model="m", limit=5, run_id="r1")
    assert argv[argv.index("--limit") + 1] == "5"
    assert argv[argv.index("--run-id") + 1] == "r1"


# ---------------------------------------------------------------------------
# Split routing
# ---------------------------------------------------------------------------

def test_dev_split_routes_to_dev_selection_dir():
    assert ar._split_output_dir("dev") == ar.DEV_SELECTION_DIR


def test_heldout_split_routes_to_raw_runs_dir():
    assert ar._split_output_dir("heldout") == ar.RAW_RUNS_DIR


def test_invalid_split_raises():
    with pytest.raises(ValueError):
        ar._split_output_dir("bogus")


# ---------------------------------------------------------------------------
# run_config: stubbed subprocess, no live infra
# ---------------------------------------------------------------------------

def _make_case_csv(path: Path, n=3):
    fieldnames = ["case_id", "end_to_end_latency_ms", "escalation_triggered", "reranker_fired", "estimated_cost_usd", "peak_memory_mb"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i in range(n):
            w.writerow({
                "case_id": f"c{i}", "end_to_end_latency_ms": "100.0",
                "escalation_triggered": "False", "reranker_fired": "True",
                "estimated_cost_usd": "0.0", "peak_memory_mb": "",
            })


def make_stub_runner(csv_path: Path, returncode=0, stderr=""):
    def _runner(argv, capture_output, text):
        return SimpleNamespace(returncode=returncode, stdout=f"Wrote 3 row(s) to {csv_path}", stderr=stderr)
    return _runner


def test_run_config_unknown_name_raises_keyerror(tmp_path):
    with pytest.raises(KeyError):
        ar.run_config("bogus_config", tmp_path / "t.csv", "heldout", reranker_model="m")


def test_run_config_writes_manifest_from_stubbed_subprocess(tmp_path, monkeypatch):
    monkeypatch.setattr(ar, "RAW_RUNS_DIR", tmp_path / "raw_runs")
    monkeypatch.setattr(ar, "DEV_SELECTION_DIR", tmp_path / "dev_selection")

    csv_path = tmp_path / "raw_runs" / "case_result.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    _make_case_csv(csv_path)

    test_set = tmp_path / "test_set.csv"
    test_set.write_text("case_id,input_text\nc0,x\n", encoding="utf-8")

    runner = make_stub_runner(csv_path)
    result_csv, manifest = ar.run_config(
        "hierarchical_with_rerank", test_set, "heldout",
        reranker_model="ollama/llama3.2:1b", subprocess_runner=runner,
    )
    assert result_csv == csv_path
    assert manifest.n_cases == 3
    assert manifest.classifier_method == "hierarchical_with_rerank"
    assert manifest.split_name == "heldout"
    manifest_path = tmp_path / "raw_runs" / "manifests" / "manifest_hierarchical_with_rerank.jsonl"
    assert manifest_path.exists()


def test_run_config_manifest_records_sre_status_in_retrieval_params(tmp_path, monkeypatch):
    """Task D.3 ('every run records ... SRE status'): retrieval_params must
    carry the config's sre on/off setting, not just system/reranker -- this
    was a real gap (SRE was silently absent from every prior manifest even
    though no_sre/with_sre are two of the five named ablation configs)."""
    monkeypatch.setattr(ar, "RAW_RUNS_DIR", tmp_path / "raw_runs")
    monkeypatch.setattr(ar, "DEV_SELECTION_DIR", tmp_path / "dev_selection")

    csv_path = tmp_path / "raw_runs" / "case_result.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    _make_case_csv(csv_path)

    test_set = tmp_path / "test_set.csv"
    test_set.write_text("case_id,input_text\nc0,x\n", encoding="utf-8")

    runner = make_stub_runner(csv_path)
    _, manifest_no_sre = ar.run_config(
        "no_sre", test_set, "heldout", reranker_model="m", subprocess_runner=runner,
    )
    assert manifest_no_sre.retrieval_params["sre"] == "off"

    _, manifest_with_sre = ar.run_config(
        "with_sre", test_set, "heldout", reranker_model="m", subprocess_runner=runner,
    )
    assert manifest_with_sre.retrieval_params["sre"] == "on"


def test_run_config_nonzero_exit_raises_runtimeerror(tmp_path, monkeypatch):
    monkeypatch.setattr(ar, "RAW_RUNS_DIR", tmp_path / "raw_runs")

    def failing_runner(argv, capture_output, text):
        return SimpleNamespace(returncode=1, stdout="", stderr="boom")

    with pytest.raises(RuntimeError, match="boom"):
        ar.run_config(
            "flat_baseline", tmp_path / "t.csv", "heldout",
            reranker_model="m", subprocess_runner=failing_runner,
        )


def test_run_config_dev_split_writes_under_dev_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(ar, "RAW_RUNS_DIR", tmp_path / "raw_runs")
    monkeypatch.setattr(ar, "DEV_SELECTION_DIR", tmp_path / "dev_selection")

    csv_path = tmp_path / "dev_selection" / "case_result.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    _make_case_csv(csv_path)
    test_set = tmp_path / "dev.csv"
    test_set.write_text("case_id,input_text\nc0,x\n", encoding="utf-8")

    runner = make_stub_runner(csv_path)
    _, manifest = ar.run_config(
        "no_sre", test_set, "dev", reranker_model="m", subprocess_runner=runner,
    )
    assert manifest.split_name == "dev"
    manifest_path = tmp_path / "dev_selection" / "manifests" / "manifest_no_sre.jsonl"
    assert manifest_path.exists()
    # never written under raw_runs/ -- a dev-split run must not appear as a confirmed result
    assert not (tmp_path / "raw_runs" / "manifests" / "manifest_no_sre.jsonl").exists()


# ---------------------------------------------------------------------------
# Governance gate: blocks BEFORE any subprocess call
# (Step 3 real-LFS-intake hardening pass -- new label vocabulary,
# on_governance_failure="downgrade" default, in-repo-path/synthetic-marker
# safeguards)
# ---------------------------------------------------------------------------

_COMPLETE_CARD_PATH = Path(__file__).resolve().parent / "fixtures" / "complete_approved_real_lfs_dataset_card_example.json"
_MINIMAL_SYNTHETIC_CARD_PATH = Path(__file__).resolve().parent / "fixtures" / "synthetic_lfs_dataset_card_example.json"


def _load_complete_card():
    from dataset_card_schema import DatasetCard
    return DatasetCard.model_validate_json(_COMPLETE_CARD_PATH.read_text(encoding="utf-8"))


def _load_minimal_synthetic_card():
    from dataset_card_schema import DatasetCard
    return DatasetCard.model_validate_json(_MINIMAL_SYNTHETIC_CARD_PATH.read_text(encoding="utf-8"))


def test_run_config_blocks_approved_real_lfs_without_card(tmp_path, monkeypatch):
    monkeypatch.setattr(ar, "RAW_RUNS_DIR", tmp_path / "raw_runs")
    calls = []

    def spy_runner(argv, capture_output, text):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with pytest.raises(ar.GovernanceError):
        ar.run_config(
            "flat_baseline", tmp_path / "t.csv", "heldout",
            reranker_model="m", dataset_label="approved_real_lfs_validation",
            subprocess_runner=spy_runner,
        )
    assert calls == []  # subprocess never invoked -- governance check ran first


def test_run_config_rejects_unrecognised_dataset_label(tmp_path):
    with pytest.raises(ValueError, match="approved_real_lfs_validation"):
        ar.run_config(
            "flat_baseline", tmp_path / "t.csv", "heldout",
            reranker_model="m", dataset_label="real_world",
        )


def test_run_config_default_downgrade_mode_writes_invalid_manifest_record(tmp_path, monkeypatch):
    """Default behaviour (on_governance_failure='downgrade'): the rejected
    attempt is still recorded on disk as invalid_incomplete_governance --
    visible evidence rather than a silently discarded exception -- even
    though run_config() still raises (the classification never runs)."""
    monkeypatch.setattr(ar, "RAW_RUNS_DIR", tmp_path / "raw_runs")
    card = _load_minimal_synthetic_card()

    with pytest.raises(ar.GovernanceError):
        ar.run_config(
            "flat_baseline", tmp_path / "t.csv", "heldout",
            reranker_model="m", dataset_label="approved_real_lfs_validation",
            dataset_card=card, run_id="rejected-attempt",
        )

    manifest_path = tmp_path / "raw_runs" / "manifests" / "manifest_rejected-attempt.jsonl"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["dataset_label"] == "invalid_incomplete_governance"
    assert payload["governance_validation_errors"]


def test_run_config_raise_mode_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(ar, "RAW_RUNS_DIR", tmp_path / "raw_runs")
    card = _load_minimal_synthetic_card()

    with pytest.raises(ar.GovernanceError):
        ar.run_config(
            "flat_baseline", tmp_path / "t.csv", "heldout",
            reranker_model="m", dataset_label="approved_real_lfs_validation",
            dataset_card=card, on_governance_failure="raise",
        )
    manifests_dir = tmp_path / "raw_runs" / "manifests"
    assert not manifests_dir.exists() or not list(manifests_dir.glob("*.jsonl"))


def test_run_config_allows_approved_real_lfs_with_complete_card(tmp_path, monkeypatch):
    monkeypatch.setattr(ar, "RAW_RUNS_DIR", tmp_path / "raw_runs")
    card = _load_complete_card()

    csv_path = tmp_path / "raw_runs" / "case_result.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    _make_case_csv(csv_path)
    test_set = tmp_path / "t.csv"
    test_set.write_text("case_id,input_text\nc0,x\n", encoding="utf-8")

    runner = make_stub_runner(csv_path)
    _, manifest = ar.run_config(
        "flat_baseline", test_set, "heldout", reranker_model="m",
        dataset_label="approved_real_lfs_validation", dataset_card=card, subprocess_runner=runner,
    )
    assert manifest.dataset_label == "approved_real_lfs_validation"


def test_run_config_synthetic_intake_package_still_rejected(tmp_path, monkeypatch):
    """The fully-filled-out synthetic intake package fixture (Task D) must
    never pass as approved real LFS, even via run_config()'s full path."""
    from dataset_card_schema import DatasetCard
    monkeypatch.setattr(ar, "RAW_RUNS_DIR", tmp_path / "raw_runs")
    intake_card_path = Path(__file__).resolve().parent / "fixtures" / "synthetic_lfs_intake_package" / "dataset_card.json"
    card = DatasetCard.model_validate_json(intake_card_path.read_text(encoding="utf-8"))

    with pytest.raises(ar.GovernanceError):
        ar.run_config(
            "flat_baseline", tmp_path / "t.csv", "heldout",
            reranker_model="m", dataset_label="approved_real_lfs_validation",
            dataset_card=card,
        )


# ---------------------------------------------------------------------------
# Evaluation table: "not yet run" unless a real manifest exists
# ---------------------------------------------------------------------------

def test_table_rows_all_not_yet_run_when_no_manifests(tmp_path, monkeypatch):
    monkeypatch.setattr(ar, "RAW_RUNS_DIR", tmp_path / "raw_runs")
    rows = ar.build_evaluation_table_rows()
    assert len(rows) == 5
    assert all(r["n_cases"] == "not yet run" for r in rows)


def test_table_rows_reflect_real_manifest_when_present(tmp_path, monkeypatch):
    manifests_dir = tmp_path / "raw_runs" / "manifests"
    manifests_dir.mkdir(parents=True)
    monkeypatch.setattr(ar, "RAW_RUNS_DIR", tmp_path / "raw_runs")

    payload = {
        "n_cases": 130, "latency_mean_ms": 250.5, "hitl_escalation_rate": 0.12,
        "split_name": "heldout", "dataset_label": "approved_real_lfs_validation",
    }
    (manifests_dir / "manifest_with_sre.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")

    rows = {r["config"]: r for r in ar.build_evaluation_table_rows()}
    assert rows["with_sre"]["n_cases"] == 130
    assert rows["with_sre"]["latency_mean_ms"] == 250.5
    assert rows["with_sre"]["dataset_label"] == "approved_real_lfs_validation"
    assert rows["no_sre"]["n_cases"] == "not yet run"
    assert rows["no_sre"]["dataset_label"] == "not yet run"


def test_write_evaluation_table_markdown(tmp_path, monkeypatch):
    monkeypatch.setattr(ar, "RAW_RUNS_DIR", tmp_path / "raw_runs")
    out_path = tmp_path / "evaluation_table_template.md"
    ar.write_evaluation_table_markdown(out_path)
    text = out_path.read_text(encoding="utf-8")
    assert "not yet run" in text
    assert "flat_baseline" in text
    assert "Data label" in text


def test_evaluation_table_markdown_shows_real_dataset_label(tmp_path, monkeypatch):
    manifests_dir = tmp_path / "raw_runs" / "manifests"
    manifests_dir.mkdir(parents=True)
    monkeypatch.setattr(ar, "RAW_RUNS_DIR", tmp_path / "raw_runs")
    payload = {
        "n_cases": 5, "latency_mean_ms": 1.0, "hitl_escalation_rate": 0.0,
        "split_name": "heldout", "dataset_label": "invalid_incomplete_governance",
    }
    (manifests_dir / "manifest_no_sre.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")

    out_path = tmp_path / "evaluation_table_template.md"
    ar.write_evaluation_table_markdown(out_path)
    text = out_path.read_text(encoding="utf-8")
    assert "invalid_incomplete_governance" in text


# ---------------------------------------------------------------------------
# CLI: --dataset-label choices= rejects an unrecognised label
# ---------------------------------------------------------------------------

def test_cli_rejects_invalid_dataset_label_choice(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "ablation_runner.py", "run", "--config", "flat_baseline",
        "--test-set", "t.csv", "--split", "heldout", "--dataset-label", "real_world",
    ])
    with pytest.raises(SystemExit) as exc_info:
        ar.main()
    assert exc_info.value.code == 2  # argparse invalid-choice exit code
