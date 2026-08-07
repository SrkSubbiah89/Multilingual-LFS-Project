"""
eval/generate_dry_run_readiness_report.py

Conference I Reviewer #2 response, Step 4 (evaluation-readiness pass), task
E: "Generate only synthetic dry-run artifacts." Orchestrates the pieces
already built/verified in tasks A-D into ONE synthetic dry run and writes
every resulting artifact under a Git-ignored local directory (see
eval/local_runs/ in .gitignore) -- never under a tracked eval/results/ path.

This script performs NO live Qdrant/Ollama/LLM/API/network call: it drives
eval/ablation_runner.py's --dry-run path (which itself never constructs a
classifier -- see eval/run_eval.py's --dry-run branch) for all 5 named
ablation configs, then separately exercises eval/coverage_audit.py (reads
only in-repo code tables), eval/sre_eval_format.py (pure function, no
labels supplied here), and eval/figure_exports/*.py (reads only the
manifests this same run just wrote) -- pointed at the local run directory,
never at the tracked eval/results/ tree.

Every artifact this script writes states dataset_label=
synthetic_or_operationally_realistic and evaluation_status=
dry_run_not_measured (or, for the coverage audit -- which is not a
classification evaluation and has no evaluation_status/dataset_label of its
own -- a REPORT_KIND: coverage_audit marker plus an explicit note that it
carries no synthetic/real evaluation claim at all).

Usage
-----
    python eval/generate_dry_run_readiness_report.py \\
        --test-set eval/fixtures/synthetic_lfs_intake_package/synthetic_test_set.csv \\
        --dataset-card eval/fixtures/synthetic_lfs_intake_package/dataset_card.json \\
        --split-manifest eval/fixtures/synthetic_lfs_intake_package/split_manifest.json

Writes to eval/local_runs/dry_run_<UTC timestamp>/ by default (override with
--output-root). See Documentation/Conference_I_Reviewer_2/
EVALUATION_READINESS_AND_DRY_RUN.md for the full readiness report this
script's output backs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import ablation_runner as ar  # noqa: E402
from dataset_card_schema import DatasetCard  # noqa: E402
from validate_evaluation_discipline import (  # noqa: E402
    validate_manifest_discipline,
    validate_split_manifest,
)
from split_manifest_schema import SplitManifest  # noqa: E402
from sre_eval_format import evaluate_sre_labels  # noqa: E402

_MEASUREMENT_FIELDS_WITH_REASONS = [
    ("latency_mean_ms", "latency_unavailable_reason"),
    ("throughput_cases_per_sec", "throughput_unavailable_reason"),
    ("peak_process_memory_mb", "peak_process_memory_unavailable_reason"),
    ("peak_gpu_memory_mb", "peak_gpu_memory_unavailable_reason"),
    ("retrieval_count", "retrieval_count_unavailable_reason"),
    ("mean_retrieval_candidate_pool_size", "mean_retrieval_candidate_pool_size_unavailable_reason"),
    ("hitl_escalation_rate", "hitl_escalation_rate_unavailable_reason"),
    ("estimated_cost_usd", "estimated_cost_method"),
]


def run_all_dry_run_configs(
    test_set: Path, output_root: Path,
    dataset_card_path: Path = None, split_manifest_path: Path = None,
) -> dict:
    """Runs all 5 named ablation configs with dry_run=True, --split heldout,
    under output_root. Returns {config_name: ExperimentRunManifest}."""
    card = None
    if dataset_card_path is not None:
        card = DatasetCard.model_validate_json(dataset_card_path.read_text(encoding="utf-8"))

    manifests = {}
    for name in ar.CONFIGS:
        _, manifest = ar.run_config(
            name, test_set, "heldout",
            reranker_model=None,
            dataset_card=card,
            run_id=name,
            dry_run=True,
            split_manifest_path=split_manifest_path,
            output_root=output_root,
        )
        manifests[name] = manifest
    return manifests


def write_ablation_status_table(manifests: dict, out_path: Path) -> None:
    lines = [
        "# Dry-Run Ablation Status Table (SYNTHETIC -- not a measured result)",
        "",
        "Every row below is a `--dry-run` invocation: no classifier was constructed, "
        "no Qdrant/Ollama/LLM/API/network call was made. `evaluation_status` is "
        "`dry_run_not_measured` for every row -- this table proves the pipeline runs "
        "end-to-end for all 5 required ablation configs, nothing more.",
        "",
        "| Config | dataset_label | evaluation_status | split | n_cases | git_commit | utc_timestamp |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in manifests.items():
        lines.append(
            f"| {name} | {m.dataset_label} | {m.evaluation_status} | {m.split_name} | "
            f"{m.n_cases} | {m.git_commit} | {m.utc_timestamp} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_null_metric_report(manifests: dict, out_path: Path) -> None:
    report = {}
    for name, m in manifests.items():
        rows = []
        for value_field, reason_field in _MEASUREMENT_FIELDS_WITH_REASONS:
            value = getattr(m, value_field)
            reason = getattr(m, reason_field, None)
            rows.append({"field": value_field, "value": value, "reason": reason})
        report[name] = {
            "dataset_label": m.dataset_label,
            "evaluation_status": m.evaluation_status,
            "null_metrics": rows,
        }
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def write_discipline_check(manifests: dict, split_manifest: SplitManifest, out_path: Path) -> None:
    """Task D's validators, run against this dry run's own artifacts --
    proves the rejection logic also accepts a genuinely well-formed run."""
    split_report = validate_split_manifest(split_manifest)
    per_manifest = {
        name: {
            "ok": validate_manifest_discipline(m).ok,
            "errors": validate_manifest_discipline(m).errors,
            "warnings": validate_manifest_discipline(m).warnings,
        }
        for name, m in manifests.items()
    }
    out = {
        "split_manifest_check": {"ok": split_report.ok, "errors": split_report.errors, "warnings": split_report.warnings},
        "per_config_manifest_check": per_manifest,
    }
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


def run_coverage_audit(out_dir: Path) -> Path:
    """eval/coverage_audit.py reads only in-repo code tables -- no network,
    no synthetic/real data claim of its own. Run as linked evidence
    alongside the dry run (Task E: 'coverage-audit linkage')."""
    result = subprocess.run(
        [sys.executable, str(_HERE / "coverage_audit.py"), "--out", str(out_dir)],
        capture_output=True, text=True,
    )
    (out_dir / "coverage_audit_stdout.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
    return out_dir


def run_sre_eval_schema_check(out_path: Path) -> None:
    """No labelled incoherence fixture exists in this repo (see
    eval/sre_eval_format.py's module docstring) -- this call demonstrates
    the schema's honest no-labels path, not a measurement."""
    metrics = evaluate_sre_labels([])
    out = {
        "note": "SYNTHETIC DRY RUN -- no labelled SRE incoherence fixture supplied; this is the honest no-labels path, not a measurement.",
        "status": metrics.status,
        "precision": metrics.precision, "recall": metrics.recall,
        "fpr": metrics.fpr, "fnr": metrics.fnr,
        "reviewer_workload_delta": metrics.reviewer_workload_delta,
        "n_labeled": metrics.n_labeled,
    }
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


def run_figure_export_validation(manifests_results_dir: Path, coverage_dir: Path, out_dir: Path) -> None:
    """Points eval/figure_exports/*.py at THIS run's local manifests/coverage
    output (never the tracked eval/results/ or Documentation/.../generated/
    trees) to exercise input validation end-to-end without touching tracked
    output."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("export_evaluation_results.py", ["--results-dir", str(manifests_results_dir)], "evaluation_results.json"),
        ("export_latency_scalability.py", ["--results-dir", str(manifests_results_dir)], "latency_scalability.json"),
        ("export_coverage_charts.py", ["--source-dir", str(coverage_dir)], "coverage_charts.json"),
    ]
    log_lines = []
    for script, extra_args, out_name in jobs:
        out_path = out_dir / out_name
        result = subprocess.run(
            [sys.executable, str(_HERE / "figure_exports" / script), "--out", str(out_path), *extra_args],
            capture_output=True, text=True,
        )
        log_lines.append(f"=== {script} (exit {result.returncode}) ===\n{result.stdout}\n{result.stderr}")
    (out_dir / "figure_export_stdout.txt").write_text("\n".join(log_lines), encoding="utf-8")


def write_readiness_summary(output_root: Path, manifests: dict) -> None:
    lines = [
        "# Dry-Run Readiness Summary (SYNTHETIC ONLY -- no measurement)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "**Every artifact under this directory is a synthetic dry run.** "
        "`dataset_label=synthetic_or_operationally_realistic` and "
        "`evaluation_status=dry_run_not_measured` on every manifest. No "
        "classifier was constructed; no Qdrant/Ollama/LLM/API/network call "
        "was made anywhere in this run. Nothing here may be cited as a "
        "measured result in the manuscript.",
        "",
        "## Contents",
        "- `heldout/manifests/manifest_<config>.jsonl` -- one manifest per named ablation config (5 total)",
        "- `heldout/manifests/manifest_<config>.csv` -- same, flattened CSV",
        "- `heldout/*.csv` -- per-case CaseResult rows (all `pred_*` columns blank, `evaluation_status=dry_run_not_measured`)",
        "- `ablation_status_table.md` -- one row per config, dry-run status only",
        "- `null_metric_report.json` -- every unmeasured field + its explicit reason, per config",
        "- `discipline_check.json` -- Task D's evaluation-discipline validators run against this run's own manifests/split manifest",
        "- `coverage_audit/` -- eval/coverage_audit.py output (real, in-repo code-table counts; not a classification-evaluation result and carries no dataset_label/evaluation_status of its own)",
        "- `sre_eval_schema_check.json` -- eval/sre_eval_format.py's honest no-labels-supplied path",
        "- `figure_export_validation/` -- eval/figure_exports/*.py run against this run's own manifests/coverage output, proving the export scripts parse dry-run manifests without fabricating figures",
        "",
        "## Configs run",
        "",
    ]
    for name, m in manifests.items():
        lines.append(f"- `{name}`: n_cases={m.n_cases}, dataset_label={m.dataset_label}, evaluation_status={m.evaluation_status}")
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-set", required=True, type=Path)
    parser.add_argument("--dataset-card", type=Path, default=None)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    output_root = args.output_root or (_HERE / "local_runs" / f"dry_run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    output_root.mkdir(parents=True, exist_ok=True)

    manifests = run_all_dry_run_configs(
        args.test_set, output_root,
        dataset_card_path=args.dataset_card, split_manifest_path=args.split_manifest,
    )

    write_ablation_status_table(manifests, output_root / "ablation_status_table.md")
    write_null_metric_report(manifests, output_root / "null_metric_report.json")

    split_manifest_obj = None
    if args.split_manifest is not None:
        split_manifest_obj = SplitManifest.model_validate_json(args.split_manifest.read_text(encoding="utf-8"))
    write_discipline_check(manifests, split_manifest_obj, output_root / "discipline_check.json")

    coverage_dir = output_root / "coverage_audit"
    coverage_dir.mkdir(parents=True, exist_ok=True)
    run_coverage_audit(coverage_dir)

    run_sre_eval_schema_check(output_root / "sre_eval_schema_check.json")

    run_figure_export_validation(
        output_root / "heldout" / "manifests", coverage_dir,
        output_root / "figure_export_validation",
    )

    write_readiness_summary(output_root, manifests)

    print(f"Wrote all Task E dry-run readiness artifacts to {output_root}")


if __name__ == "__main__":
    main()
