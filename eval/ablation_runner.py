"""
eval/ablation_runner.py

Ablation infrastructure (Conference I Reviewer #2 response, Section E).
Drives eval/run_eval.py through 5 uniform, named configurations by
composing its EXISTING CLI flags -- run_eval.py stays the single source of
truth for how a case is scored; this module only resolves configs,
orchestrates the run (as a subprocess, so this file has no import-time
dependency on run_eval.py's heavy classifier imports and so a config's
argv can be inspected/tested without ever touching Qdrant/Ollama), and
wraps the resulting CaseResult CSV into an eval.manifest.
ExperimentRunManifest.

Named configs
-------------
flat_baseline              --system flat  (single-stage dense retrieval, same reranker rule as hierarchical)
hierarchical_no_rerank     --system hierarchical --use-llm-reranker off
hierarchical_with_rerank   --system hierarchical --use-llm-reranker on  (default hierarchical behaviour)
no_sre                     --system hierarchical --sre off
with_sre                   --system hierarchical --sre on   (default hierarchical behaviour)

These 5 configs cover 3 independent research questions: does hierarchical
retrieval beat flat retrieval (flat_baseline vs hierarchical_with_rerank),
does LLM reranking help (hierarchical_no_rerank vs hierarchical_with_rerank),
and does enabling the SRE change the run (no_sre vs with_sre).

Dev/held-out discipline
------------------------
--split dev writes to eval/results/dev_selection/ (parameter selection
only -- never a citable confirmation result). --split heldout writes to
eval/results/raw_runs/ (run_eval.py's own default RESULTS_DIR -- the only
split whose manifest may be cited as a confirmed result). This mirrors the
dev_set_v1.csv / test_set_full130.csv separation eval/validate_dev_set.py
already enforces for ISCO K-selection.

Real-LFS-validation governance gate
-------------------------------------
--dataset-label real_lfs_validation runs
eval/validate_real_lfs_governance.py BEFORE the experiment, aborting with
no run and no manifest if governance documentation is missing/incomplete
(same fail-fast gate eval.manifest.build_manifest() already enforces --
checked here too, before spending any compute on a run that could never
produce a citable manifest).

No results without a run
--------------------------
--emit-markdown-table only fills a cell in the generated evaluation table
when a real manifest file exists on disk (under eval/results/raw_runs/
manifests/) for that config -- otherwise the cell is the literal string
"not yet run". Never inferred, never estimated.

Usage
-----
    python eval/ablation_runner.py run --config hierarchical_with_rerank \
        --test-set eval/test_set_full130.csv --reranker-model anthropic/claude-3-5-sonnet-20241022 \
        --split heldout
    python eval/ablation_runner.py emit-table
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from dataset_card_schema import (  # noqa: E402
    APPROVED_REAL_LFS_VALIDATION,
    DATASET_LABELS,
    DEFAULT_DATASET_LABEL,
    INVALID_INCOMPLETE_GOVERNANCE,
)
import manifest as mf  # noqa: E402
from manifest import build_manifest, sha256_of_file, write_manifest  # noqa: E402
import validate_real_lfs_governance as vrg  # noqa: E402
from validate_real_lfs_governance import GovernanceError  # noqa: E402

RUN_EVAL_PATH = _HERE / "run_eval.py"
RAW_RUNS_DIR = _HERE / "results" / "raw_runs"
DEV_SELECTION_DIR = _HERE / "results" / "dev_selection"
DOCS_GENERATED_DIR = _HERE.parent / "Documentation" / "Conference_I_Reviewer_2" / "generated"


@dataclass
class ExperimentConfig:
    name: str
    system: str  # "flat" | "hierarchical" | "bm25"
    use_llm_reranker: str  # "on" | "off"
    sre: str  # "on" | "off"
    description: str


CONFIGS: dict[str, ExperimentConfig] = {
    "flat_baseline": ExperimentConfig(
        "flat_baseline", "flat", "on", "on",
        "Single-stage flat dense retrieval, same reranking rule as hierarchical.",
    ),
    "hierarchical_no_rerank": ExperimentConfig(
        "hierarchical_no_rerank", "hierarchical", "off", "on",
        "4-stage hierarchical retrieval, LLM reranking disabled (top pooled candidate used directly).",
    ),
    "hierarchical_with_rerank": ExperimentConfig(
        "hierarchical_with_rerank", "hierarchical", "on", "on",
        "4-stage hierarchical retrieval with LLM reranking (default hierarchical production behaviour).",
    ),
    "no_sre": ExperimentConfig(
        "no_sre", "hierarchical", "on", "off",
        "Hierarchical + reranking, SemanticRelationEngine coherence check disabled.",
    ),
    "with_sre": ExperimentConfig(
        "with_sre", "hierarchical", "on", "on",
        "Hierarchical + reranking, SemanticRelationEngine coherence check enabled (identical run.py invocation to hierarchical_with_rerank -- the SRE axis is the only difference from no_sre).",
    ),
}

VALID_SPLITS = ("dev", "heldout")

# GovernanceError is defined in validate_real_lfs_governance.py (the single
# source of truth for the governance gate) and re-exported here so existing
# callers importing it from this module (`ablation_runner.GovernanceError`)
# keep working unchanged.


def _split_output_dir(split: str, output_root: Optional[Path] = None) -> Path:
    """output_root, if given, overrides the default eval/results/{dev_selection,
    raw_runs}/ locations with output_root/{dev,heldout}/ -- used by the
    Step 4 dry-run readiness pipeline to write demonstration artifacts
    under the gitignored eval/local_runs/ instead of the tracked
    eval/results/ tree. Omitted (None, the default): byte-for-byte the
    original behaviour, unchanged for every existing caller."""
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")
    if output_root is not None:
        return output_root / split
    return DEV_SELECTION_DIR if split == "dev" else RAW_RUNS_DIR


def build_argv(
    config: ExperimentConfig,
    test_set: Path,
    output_dir: Path,
    reranker_model: Optional[str] = None,
    limit: Optional[int] = None,
    run_id: Optional[str] = None,
    dry_run: bool = False,
) -> list[str]:
    """Pure config -> argv resolution, no execution -- directly testable.

    dry_run : bool, default False
        Appends --dry-run to the resolved argv and skips the
        --reranker-model requirement for hierarchical/flat configs (a dry
        run never constructs a classifier, so nothing pings the reranker
        model -- see eval/run_eval.py's --dry-run docstring). A
        reranker_model, if supplied anyway, is still recorded (harmless).
    """
    argv = [
        "--test-set", str(test_set),
        "--system", config.system,
        "--config", config.name,
        "--output-dir", str(output_dir),
        "--sre", config.sre,
        "--use-llm-reranker", config.use_llm_reranker,
    ]
    if config.system in ("hierarchical", "flat"):
        if reranker_model:
            argv += ["--reranker-model", reranker_model]
        elif not dry_run:
            raise ValueError(f"config {config.name!r} (system={config.system!r}) requires --reranker-model")
    if limit is not None:
        argv += ["--limit", str(limit)]
    if run_id:
        argv += ["--run-id", run_id]
    if dry_run:
        argv += ["--dry-run"]
    return argv


def _find_written_csv(stdout: str, output_dir: Path, config_name: str) -> Optional[Path]:
    """run_eval.py prints 'Wrote N row(s) to <path>' -- parse that line first
    (authoritative); fall back to the newest matching file in output_dir."""
    for line in stdout.splitlines():
        if line.startswith("Wrote ") and " row(s) to " in line and line.rstrip().endswith(".csv"):
            candidate = Path(line.split(" row(s) to ", 1)[1].strip())
            if candidate.exists():
                return candidate
    matches = sorted(output_dir.glob(f"*_{config_name}.csv"))
    return matches[-1] if matches else None


def run_config(
    config_name: str,
    test_set: Path,
    split: str,
    reranker_model: Optional[str] = None,
    dataset_label: str = DEFAULT_DATASET_LABEL,
    dataset_card=None,
    limit: Optional[int] = None,
    run_id: Optional[str] = None,
    subprocess_runner=subprocess.run,
    on_governance_failure: str = "downgrade",
    dry_run: bool = False,
    split_manifest_path: Optional[Path] = None,
    output_root: Optional[Path] = None,
    evaluation_status_override: Optional[str] = None,
) -> tuple[Path, "object"]:
    """
    Run one named ablation config end-to-end: governance gate (if
    applicable) -> run_eval.py subprocess -> build+write a manifest.

    subprocess_runner is injectable for tests (defaults to subprocess.run;
    tests pass a stub that never touches Qdrant/Ollama).

    dry_run : bool, default False
        Conference I Reviewer #2, Step 4 (evaluation-readiness pass).
        Threads --dry-run into the eval/run_eval.py subprocess invocation
        (no classifier constructed, no Qdrant/LLM/network call there
        either) and sets the resulting manifest's evaluation_status to
        "dry_run_not_measured". The governance gate above still runs in
        full (dry_run does NOT bypass it) -- a dry run against a rejected
        DatasetCard is still rejected, and a dry run against a complete,
        valid one still legitimately carries
        dataset_label=approved_real_lfs_validation (with
        evaluation_status=dry_run_not_measured alongside it -- see
        eval.manifest.ExperimentRunManifest.evaluation_status).
    split_manifest_path : Path, optional
        Forwarded to build_manifest() -- see that function's docstring.
    output_root : Path, optional
        Overrides the default eval/results/{dev_selection,raw_runs}/
        output location with output_root/{dev,heldout}/ -- see
        _split_output_dir(). Omitted (None, the default): unchanged
        original behaviour.
    evaluation_status_override : str, optional
        Conference I Reviewer #2, Step 5 (controlled synthetic-fixture
        integration run). When None (the default), evaluation_status is
        "dry_run_not_measured" if dry_run else "measured" (unchanged prior
        behaviour). When supplied, MUST be one of manifest.EVALUATION_
        STATUSES and OVERRIDES that default -- used to mark a REAL
        (non-dry-run) classification run as
        "measured_synthetic_fixture_only" when it was only run against a
        small synthetic integration fixture, never a substantive
        evaluation. Has no effect on dry_run itself (still threaded into
        the run_eval.py subprocess normally); only changes which
        evaluation_status string the resulting manifest records.

    dataset_label : str
        MUST be one of dataset_card_schema.DATASET_LABELS -- any other
        string raises ValueError before anything else happens (no
        subprocess, no manifest). See dataset_card_schema.py.
    on_governance_failure : {"downgrade", "raise"}, default "downgrade"
        When dataset_label == APPROVED_REAL_LFS_VALIDATION but governance
        validation fails: "downgrade" (default here -- this IS "the
        evaluation runner" the governance objective means to make invalid
        attempts visible in) writes a manifest labelled
        INVALID_INCOMPLETE_GOVERNANCE recording exactly which checks
        failed, THEN still raises GovernanceError -- the classification
        subprocess is NEVER run either way (no compute spent on a rejected
        attempt), but with "downgrade" the attempt leaves a paper trail on
        disk instead of vanishing into an exception with no artifact.
        "raise" reproduces the original, stricter behaviour (exception
        only, nothing written).

    Returns (case_csv_path, ExperimentRunManifest).
    """
    if config_name not in CONFIGS:
        raise KeyError(f"Unknown ablation config {config_name!r}; expected one of {sorted(CONFIGS)}")
    config = CONFIGS[config_name]

    if dataset_label not in DATASET_LABELS:
        raise ValueError(
            f"dataset_label must be exactly one of {sorted(DATASET_LABELS)}, got {dataset_label!r}."
        )
    if on_governance_failure not in ("raise", "downgrade"):
        raise ValueError(f"on_governance_failure must be 'raise' or 'downgrade', got {on_governance_failure!r}")

    run_id = run_id or config_name

    if dataset_label == APPROVED_REAL_LFS_VALIDATION:
        report = vrg.validate({"dataset_label": dataset_label, "split_name": split}, dataset_card)
        if not report.ok:
            message = (
                f"Cannot run config {config_name!r} labelled {APPROVED_REAL_LFS_VALIDATION!r}: "
                + "; ".join(report.errors)
            )
            if on_governance_failure == "downgrade":
                invalid_output_dir = _split_output_dir(split, output_root)
                invalid_output_dir.mkdir(parents=True, exist_ok=True)
                invalid_manifest = build_manifest(
                    [], run_id=run_id, classifier_method=config_name, split_name=split,
                    dataset_version_hash=sha256_of_file(test_set),
                    dataset_label=dataset_label, dataset_card=dataset_card,
                    on_governance_failure="downgrade",
                )
                manifests_dir = invalid_output_dir / "manifests"
                write_manifest(invalid_manifest, manifests_dir)
                message += f" -- a record of this rejected attempt was written as {INVALID_INCOMPLETE_GOVERNANCE!r} to {manifests_dir}."
            raise GovernanceError(message)

    output_dir = _split_output_dir(split, output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    argv = build_argv(config, test_set, output_dir, reranker_model=reranker_model, limit=limit, run_id=run_id, dry_run=dry_run)

    result = subprocess_runner(
        [sys.executable, str(RUN_EVAL_PATH), *argv], capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"eval/run_eval.py failed for config {config_name!r} (exit {result.returncode}):\n{result.stderr}"
        )

    csv_path = _find_written_csv(result.stdout, output_dir, config_name)
    if csv_path is None:
        raise RuntimeError(f"Could not locate the CaseResult CSV run_eval.py wrote for config {config_name!r}")

    with open(csv_path, newline="", encoding="utf-8") as f:
        case_rows = list(csv.DictReader(f))

    manifest = build_manifest(
        case_rows,
        run_id=run_id,
        classifier_method=config_name,
        split_name=split,
        dataset_version_hash=sha256_of_file(test_set),
        dataset_label=dataset_label,
        dataset_card=dataset_card,
        retrieval_params={"system": config.system, "use_llm_reranker": config.use_llm_reranker, "sre": config.sre},
        llm_params={"reranker_model": reranker_model or ""},
        evaluation_status=evaluation_status_override or ("dry_run_not_measured" if dry_run else "measured"),
        split_manifest_path=split_manifest_path,
    )
    write_manifest(manifest, output_dir / "manifests")

    return csv_path, manifest


# ---------------------------------------------------------------------------
# Validate all 5 named configs resolve to a valid argv (no execution) --
# Conference I Reviewer #2, Step 4 (evaluation-readiness pass, task A/B: "the
# pipeline covers the five required ablation configurations")
# ---------------------------------------------------------------------------

def validate_all_configs(reranker_model: str = "dry-run-placeholder") -> dict[str, list[str]]:
    """Calls build_argv() for every one of the 5 named CONFIGS with a
    placeholder test set path (never read -- build_argv() is pure argv
    resolution, no filesystem access) and dry_run=True (so no
    --reranker-model is actually required). Returns {config_name: argv}
    for every config; raises if any config fails to resolve. Used by
    `python eval/ablation_runner.py validate-configs` and by the Step 4
    dry-run readiness pipeline."""
    resolved = {}
    placeholder_test_set = Path("placeholder_test_set.csv")
    placeholder_output_dir = Path("placeholder_output_dir")
    for name, config in CONFIGS.items():
        resolved[name] = build_argv(
            config, placeholder_test_set, placeholder_output_dir,
            reranker_model=reranker_model, dry_run=True,
        )
    return resolved


# ---------------------------------------------------------------------------
# Markdown evaluation table -- "not yet run" unless a real manifest exists
# ---------------------------------------------------------------------------

def _load_manifest_for_config(config_name: str) -> Optional[dict]:
    """Only looks under RAW_RUNS_DIR (heldout) -- dev_selection manifests
    are parameter-selection artifacts, never citable confirmed results."""
    manifest_path = RAW_RUNS_DIR / "manifests" / f"manifest_{config_name}.jsonl"
    if not manifest_path.exists():
        return None
    lines = manifest_path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return None
    return json.loads(lines[0])


def build_evaluation_table_rows() -> list[dict]:
    rows = []
    for name, config in CONFIGS.items():
        manifest = _load_manifest_for_config(name)
        if manifest is None:
            rows.append({
                "config": name, "description": config.description,
                "n_cases": "not yet run", "latency_mean_ms": "not yet run",
                "hitl_escalation_rate": "not yet run", "split": "not yet run",
                "dataset_label": "not yet run",
            })
        else:
            rows.append({
                "config": name, "description": config.description,
                "n_cases": manifest.get("n_cases", "not yet run"),
                "latency_mean_ms": manifest.get("latency_mean_ms") if manifest.get("latency_mean_ms") is not None else "not yet run",
                "hitl_escalation_rate": manifest.get("hitl_escalation_rate") if manifest.get("hitl_escalation_rate") is not None else "not yet run",
                "split": manifest.get("split_name", "not yet run"),
                # Always the actual dataset_label the manifest carries --
                # never inferred/assumed. A manuscript may only write
                # "validated on real LFS data" for a row whose dataset_label
                # is exactly APPROVED_REAL_LFS_VALIDATION -- see
                # validate_real_lfs_governance.check_manuscript_wording()
                # and REAL_LFS_DATA_INTAKE_CHECKLIST.md.
                "dataset_label": manifest.get("dataset_label", "unknown"),
            })
    return rows


def write_evaluation_table_markdown(path: Path) -> None:
    rows = build_evaluation_table_rows()
    lines = [
        "# Evaluation Table (generated)",
        "",
        "Generated by `python eval/ablation_runner.py emit-table`. A cell reads "
        "\"not yet run\" until a real manifest exists under "
        "`eval/results/raw_runs/manifests/manifest_<config>.jsonl` -- never inferred. "
        "The **Data label** column carries the manifest's actual `dataset_label` -- "
        "manuscript wording such as \"validated on real LFS data\" is only safe for a "
        f"row labelled `{APPROVED_REAL_LFS_VALIDATION}`.",
        "",
        "| Config | Description | Split | Data label | n cases | Latency mean (ms) | HITL escalation rate |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['config']} | {r['description']} | {r['split']} | {r['dataset_label']} | {r['n_cases']} | "
            f"{r['latency_mean_ms']} | {r['hitl_escalation_rate']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run one named ablation config")
    run_p.add_argument("--config", required=True, choices=sorted(CONFIGS))
    run_p.add_argument("--test-set", required=True, type=Path)
    run_p.add_argument("--split", required=True, choices=VALID_SPLITS)
    run_p.add_argument("--reranker-model", type=str, default=None)
    # choices= closes the CLI-level gap where an arbitrary/mistyped string
    # (e.g. wrong case, or a value that isn't in DATASET_LABELS at all)
    # used to silently bypass the governance check by just never matching
    # the one exact string the validator looked for. build_manifest()/
    # run_config() re-check this too (defense at the function level, not
    # just argparse), but rejecting it here means a bad --dataset-label
    # never even reaches Python code.
    run_p.add_argument("--dataset-label", type=str, default=DEFAULT_DATASET_LABEL, choices=sorted(DATASET_LABELS))
    run_p.add_argument("--dataset-card", type=Path, default=None)
    run_p.add_argument("--split-manifest", type=Path, default=None,
                        help="Optional split-manifest file (e.g. a dev/test split assignment doc) -- its sha256 is recorded as the manifest's split_manifest_hash. Content is never read beyond hashing.")
    run_p.add_argument("--limit", type=int, default=None)
    run_p.add_argument("--run-id", type=str, default=None)
    run_p.add_argument("--output-root", type=Path, default=None,
                        help="Override eval/results/{dev_selection,raw_runs}/ with <output-root>/{dev,heldout}/ -- e.g. a gitignored eval/local_runs/... directory for dry-run/demonstration output.")
    run_p.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Conference I Reviewer #2, Step 4 (evaluation-readiness pass). "
            "Threads --dry-run into the eval/run_eval.py subprocess (no "
            "classifier constructed, no Qdrant/LLM/network call). The "
            "governance gate still runs in full. Resulting manifest carries "
            "evaluation_status=dry_run_not_measured."
        ),
    )
    run_p.add_argument(
        "--evaluation-status", type=str, default=None, choices=list(mf.EVALUATION_STATUSES),
        help=(
            "Conference I Reviewer #2, Step 5 (controlled synthetic-fixture "
            "integration run). Overrides the manifest's evaluation_status "
            "(default: dry_run_not_measured if --dry-run else measured). "
            "Use measured_synthetic_fixture_only for a REAL (non-dry-run) "
            "classification run against only a small synthetic integration "
            "fixture -- never a substantive evaluation. manuscript_eligible "
            "is computed automatically and can never be True for this value."
        ),
    )

    sub.add_parser("emit-table", help="Regenerate the Markdown evaluation table from existing manifests")

    sub.add_parser(
        "validate-configs",
        help="Confirm all 5 named ablation configs resolve to a valid argv, with no execution and no network call.",
    )

    args = parser.parse_args()

    if args.command == "run":
        card = None
        if args.dataset_card:
            from dataset_card_schema import DatasetCard
            card = DatasetCard.model_validate_json(args.dataset_card.read_text(encoding="utf-8"))
        csv_path, manifest = run_config(
            args.config, args.test_set, args.split,
            reranker_model=args.reranker_model, dataset_label=args.dataset_label,
            dataset_card=card, limit=args.limit, run_id=args.run_id,
            dry_run=args.dry_run, split_manifest_path=args.split_manifest,
            output_root=args.output_root,
            evaluation_status_override=args.evaluation_status,
        )
        print(f"Wrote case CSV: {csv_path}")
        print(f"Manifest run_id={manifest.run_id} n_cases={manifest.n_cases} "
              f"evaluation_status={manifest.evaluation_status} "
              f"manuscript_eligible={manifest.manuscript_eligible} "
              f"latency_mean_ms={manifest.latency_mean_ms}")
    elif args.command == "emit-table":
        out_path = DOCS_GENERATED_DIR / "evaluation_table_template.md"
        write_evaluation_table_markdown(out_path)
        print(f"Wrote {out_path}")
    elif args.command == "validate-configs":
        resolved = validate_all_configs()
        for name, argv in resolved.items():
            print(f"{name}: python eval/run_eval.py {' '.join(argv)}")
        print(f"\nAll {len(resolved)} named ablation configs resolved successfully (no execution, no network call).")


if __name__ == "__main__":
    main()
