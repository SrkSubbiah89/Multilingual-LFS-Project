"""
Tests for eval/figure_exports/*.py -- Section I of the Conference I
Reviewer #2 response (figure-support data exports).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "figure_exports"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import export_agent_role_diagram as agent_export  # noqa: E402
import export_classifier_hierarchy as hier_export  # noqa: E402
import export_coverage_charts as cov_export  # noqa: E402
import export_evaluation_results as eval_export  # noqa: E402
import export_latency_scalability as lat_export  # noqa: E402


# ---------------------------------------------------------------------------
# export_classifier_hierarchy
# ---------------------------------------------------------------------------

def test_isco_stages_are_four_implemented_stages():
    assert len(hier_export.ISCO_STAGES) == 4
    assert all(s.status == "implemented" for s in hier_export.ISCO_STAGES)
    assert [s.stage_name for s in hier_export.ISCO_STAGES] == ["major", "submajor", "minor", "unit"]


def test_isic_iscedf_stages_are_implemented_but_unevaluated():
    stages = hier_export.IMPLEMENTED_UNEVALUATED_STAGES
    assert stages
    assert all(s.status == "implemented_unevaluated" for s in stages)
    # Real collection names (Task 05), never a placeholder string.
    assert all(s.collection.startswith(("isic_rev4_", "iscedf2013_")) for s in stages)
    assert all(s.weight > 0.0 for s in stages)


def test_hierarchy_export_writes_json_and_csv(tmp_path):
    rows = hier_export.all_rows()
    hier_export.write_json(rows, tmp_path / "h.json")
    hier_export.write_csv(rows, tmp_path / "h.csv")
    payload = json.loads((tmp_path / "h.json").read_text(encoding="utf-8"))
    assert len(payload["stages"]) == len(rows)
    assert (tmp_path / "h.csv").exists()


def test_hierarchy_export_cli(tmp_path, monkeypatch):
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["export_classifier_hierarchy.py", "--out", str(out_dir)])
    hier_export.main()
    assert (out_dir / "classifier_hierarchy.json").exists()
    assert (out_dir / "classifier_hierarchy.csv").exists()


# ---------------------------------------------------------------------------
# export_agent_role_diagram
# ---------------------------------------------------------------------------

def test_nodes_cover_all_registry_components():
    nodes = agent_export.build_nodes()
    component_names = {n.component for n in nodes}
    assert "ISCOClassifier" in component_names
    assert "ISICClassifier" in component_names


def test_edges_count_matches_registry_length():
    from backend.agents.method_registry import REGISTRY
    edges = agent_export.build_edges()
    assert len(edges) == len(REGISTRY)


def test_edges_mark_all_registry_methods_as_implemented():
    """As of Task 05, every REGISTRY row (including ISIC/ISCED-F
    hierarchical retrieval) describes real code -- is_implemented is now
    always True. Evaluated-vs-not is a separate distinction, tracked by
    ClassifierMethodEntry.evaluated, not this diagram field."""
    edges = agent_export.build_edges()
    hier_edges = [e for e in edges if e.method_id in ("isic_hierarchical_retrieval", "iscedf_hierarchical_retrieval")]
    assert hier_edges
    assert all(e.is_implemented is True for e in edges)


def test_agent_role_diagram_cli(tmp_path, monkeypatch):
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["export_agent_role_diagram.py", "--out", str(out_dir)])
    agent_export.main()
    assert (out_dir / "agent_role_diagram.json").exists()
    assert (out_dir / "agent_role_diagram_nodes.csv").exists()
    assert (out_dir / "agent_role_diagram_edges.csv").exists()


# ---------------------------------------------------------------------------
# export_evaluation_results -- "no data yet" path
# ---------------------------------------------------------------------------

def test_no_manifests_found_produces_empty_but_valid_export(tmp_path):
    rows = eval_export.load_manifest_summaries(tmp_path / "does_not_exist")
    assert rows == []
    eval_export.write_json(rows, tmp_path / "r.json")
    payload = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert payload["no_manifests_found"] is True
    assert payload["results"] == []


def test_manifest_found_produces_real_row(tmp_path):
    results_dir = tmp_path / "results" / "raw_runs" / "manifests"
    results_dir.mkdir(parents=True)
    payload = {"run_id": "r1", "classifier_method": "hierarchical_with_rerank",
               "split_name": "heldout", "dataset_label": "synthetic_or_operationally_realistic", "n_cases": 10,
               "latency_mean_ms": 120.0, "latency_p95_ms": 200.0,
               "hitl_escalation_rate": 0.1, "estimated_cost_usd": 0.0,
               "git_commit": "abc", "utc_timestamp": "2026-01-01T00:00:00+00:00"}
    (results_dir / "manifest_r1.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")

    rows = eval_export.load_manifest_summaries(tmp_path / "results")
    assert len(rows) == 1
    assert rows[0]["run_id"] == "r1"


def test_evaluation_results_cli_no_data(tmp_path, monkeypatch, capsys):
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "export_evaluation_results.py", "--out", str(out_dir),
        "--results-dir", str(tmp_path / "nonexistent"),
    ])
    eval_export.main()
    payload = json.loads((out_dir / "evaluation_results.json").read_text(encoding="utf-8"))
    assert payload["no_manifests_found"] is True


# ---------------------------------------------------------------------------
# export_latency_scalability
# ---------------------------------------------------------------------------

def test_latency_rows_flatten_hardware_fields(tmp_path):
    results_dir = tmp_path / "results" / "raw_runs" / "manifests"
    results_dir.mkdir(parents=True)
    payload = {
        "run_id": "r1", "classifier_method": "hierarchical_with_rerank", "n_cases": 5,
        "latency_mean_ms": 100.0, "latency_p50_ms": 90.0, "latency_p95_ms": 150.0,
        "throughput_cases_per_sec": 8.0,
        "hardware": {"cpu_count": 8, "ram_gb": 16.0, "gpu_model": None},
        "peak_process_memory_mb": None, "peak_gpu_memory_mb": None,
    }
    (results_dir / "manifest_r1.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")

    rows = lat_export.load_latency_rows(tmp_path / "results")
    assert len(rows) == 1
    assert rows[0]["hardware_cpu_count"] == 8
    assert rows[0]["hardware_ram_gb"] == 16.0


def test_latency_no_manifests_found(tmp_path):
    rows = lat_export.load_latency_rows(tmp_path / "nonexistent")
    assert rows == []


def test_latency_scalability_cli(tmp_path, monkeypatch):
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "export_latency_scalability.py", "--out", str(out_dir),
        "--results-dir", str(tmp_path / "nonexistent"),
    ])
    lat_export.main()
    payload = json.loads((out_dir / "latency_scalability.json").read_text(encoding="utf-8"))
    assert payload["no_manifests_found"] is True


# ---------------------------------------------------------------------------
# export_coverage_charts
# ---------------------------------------------------------------------------

def test_coverage_no_reports_found(tmp_path):
    rows = cov_export.load_coverage_rows(tmp_path / "nonexistent")
    assert rows == []


def test_coverage_finds_latest_report_per_standard_key(tmp_path):
    source_dir = tmp_path / "generated"
    source_dir.mkdir()
    older = {"reports": [{"standard_name": "ISCO-08", "level_name": "major",
                           "official_count_verified": None, "official_count_unverified": 10,
                           "implemented_count": 1, "coverage_percentage": None,
                           "coverage_percentage_status": "old", "duplicate_codes": [], "malformed_codes": []}]}
    newer = {"reports": [{"standard_name": "ISCO-08", "level_name": "major",
                           "official_count_verified": None, "official_count_unverified": 10,
                           "implemented_count": 10, "coverage_percentage": None,
                           "coverage_percentage_status": "new", "duplicate_codes": ["x"], "malformed_codes": []}]}
    (source_dir / "coverage_audit_isco08_20260101T000000Z.json").write_text(json.dumps(older), encoding="utf-8")
    (source_dir / "coverage_audit_isco08_20260102T000000Z.json").write_text(json.dumps(newer), encoding="utf-8")

    rows = cov_export.load_coverage_rows(source_dir)
    assert len(rows) == 1
    assert rows[0]["coverage_percentage_status"] == "new"
    assert rows[0]["duplicate_count"] == 1


def test_coverage_handles_multi_underscore_standard_key(tmp_path):
    source_dir = tmp_path / "generated"
    source_dir.mkdir()
    payload = {"reports": [{"standard_name": "ISCED 2011", "level_name": "level",
                             "official_count_verified": None, "official_count_unverified": 9,
                             "implemented_count": 9, "coverage_percentage": None,
                             "coverage_percentage_status": "s", "duplicate_codes": [], "malformed_codes": []}]}
    (source_dir / "coverage_audit_isced2011_and_iscedf2013_20260101T000000Z.json").write_text(
        json.dumps(payload), encoding="utf-8",
    )
    rows = cov_export.load_coverage_rows(source_dir)
    assert len(rows) == 1
    assert rows[0]["standard_name"] == "ISCED 2011"


def test_coverage_charts_cli_no_data(tmp_path, monkeypatch):
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "export_coverage_charts.py", "--out", str(out_dir),
        "--source-dir", str(tmp_path / "nonexistent"),
    ])
    cov_export.main()
    payload = json.loads((out_dir / "coverage_charts.json").read_text(encoding="utf-8"))
    assert payload["no_coverage_reports_found"] is True
