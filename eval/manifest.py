"""
eval/manifest.py

Experiment-run manifest builder for Conference I Reviewer #2 response,
Section D (evaluation/reproducibility instrumentation, reviewer comment 4:
computational analysis is missing).

Aggregates ONE manifest per evaluation run from the per-case rows
eval/run_eval.py already produces (its ``CaseResult`` dataclass -- see that
module for the full per-case schema). This module does not add any new
per-case instrumentation to run_eval.py; every field below is either read
straight off existing CaseResult columns, computed by simple aggregation
over them, or probed independently at manifest-build time (hardware, OS,
installed package versions).

Honesty rule, applied throughout: a metric that cannot actually be measured
from the given case rows / this environment is ``None`` with a sibling
``<field>_unavailable_reason`` string explaining why -- never a placeholder
number, and never silently omitted. In particular:

  - ``peak_process_memory_mb`` is None (with a reason) for every run today,
    because ``CaseResult.peak_memory_mb`` is a declared field that
    run_eval.py never actually populates (grep-verified: no assignment
    anywhere in that module) -- this is an honest, pre-existing gap, not
    something this module papers over.
  - ``retrieval_count`` (the number of underlying Qdrant query_points calls)
    is None (with a reason) because CaseResult does not log a per-case
    query count -- only candidate lists and per-stage latency, neither of
    which is a reliable proxy for call count without guessing at beam
    branching. Reporting an estimate here as if it were measured would be
    exactly the kind of fabrication this tooling exists to avoid.
  - ``hitl_escalation_rate`` reflects ``CaseResult.escalation_triggered``,
    which is the EVALUATION HARNESS's own research-only escalation
    heuristic (confidence + SRE severity + stratified sampling) -- NOT the
    production ``HITLQualityManager`` decision (see
    backend/agents/hitl_quality_manager.py / method_registry.py). This
    manifest field must not be read as "production HITL rate."

Usage
-----
    from eval.manifest import build_manifest, write_manifest
    manifest = build_manifest(case_rows, run_id=..., classifier_method=...,
                               split_name="dev_v1", dataset_version_hash=...)
    write_manifest(manifest, Path("eval/results/manifests/"))
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent

# eval/ is not a package (no __init__.py) -- add it to sys.path so sibling
# modules (validate_real_lfs_governance, dataset_card_schema) can be
# imported by bare name, same convention as every eval/test_*.py file.
sys.path.insert(0, str(_HERE))

from dataset_card_schema import (  # noqa: E402
    APPROVED_REAL_LFS_VALIDATION,
    DATASET_LABELS,
    DEFAULT_DATASET_LABEL,
    INVALID_INCOMPLETE_GOVERNANCE,
)

# "measured": a real run against operationally-realistic or approved-real
# data. "dry_run_not_measured": eval/run_eval.py --dry-run /
# eval/ablation_runner.py --dry-run output. "measured_synthetic_fixture_only"
# (Step 5): a real classification run, but against the tiny synthetic
# integration fixture only -- see ExperimentRunManifest.evaluation_status's
# docstring for the full explanation of each value and why they're kept
# distinct rather than collapsed into "measured" vs "not measured."
MEASURED = "measured"
DRY_RUN_NOT_MEASURED = "dry_run_not_measured"
MEASURED_SYNTHETIC_FIXTURE_ONLY = "measured_synthetic_fixture_only"
EVALUATION_STATUSES = (MEASURED, DRY_RUN_NOT_MEASURED, MEASURED_SYNTHETIC_FIXTURE_ONLY)


# ---------------------------------------------------------------------------
# Hardware probing (best-effort, never fabricated)
# ---------------------------------------------------------------------------

@dataclass
class HardwareInfo:
    cpu_model: Optional[str] = None
    cpu_count: Optional[int] = None
    ram_gb: Optional[float] = None
    gpu_model: Optional[str] = None
    gpu_mem_gb: Optional[float] = None
    gpu_unavailable_reason: Optional[str] = None


def probe_hardware() -> HardwareInfo:
    cpu_count = os.cpu_count()
    cpu_model = platform.processor() or None

    ram_gb = None
    try:
        import psutil  # already a project dependency
        ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except Exception:
        ram_gb = None

    gpu_model, gpu_mem_gb, gpu_reason = None, None, None
    try:
        import torch  # optional; not a required project dependency
        if torch.cuda.is_available():
            gpu_model = torch.cuda.get_device_name(0)
            gpu_mem_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
        else:
            gpu_reason = "torch installed but torch.cuda.is_available() is False (no CUDA device detected)"
    except ImportError:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                name, mem_mib = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
                gpu_model = name
                gpu_mem_gb = round(float(mem_mib) / 1024, 2)
            else:
                gpu_reason = "torch not installed and nvidia-smi returned no GPU (probably no NVIDIA GPU present)"
        except Exception:
            gpu_reason = "torch not installed and nvidia-smi is not available on PATH"

    return HardwareInfo(
        cpu_model=cpu_model, cpu_count=cpu_count, ram_gb=ram_gb,
        gpu_model=gpu_model, gpu_mem_gb=gpu_mem_gb, gpu_unavailable_reason=gpu_reason,
    )


def probe_dependency_versions() -> dict[str, str]:
    """Full installed-package snapshot via importlib.metadata (avoids a
    hand-maintained, driftable list)."""
    versions = {}
    for dist in importlib.metadata.distributions():
        try:
            name = dist.metadata["Name"]
            if name:
                versions[name] = dist.version
        except Exception:
            continue
    return versions


def git_commit_hash() -> str:
    """Same subprocess pattern as eval/run_eval.py's main() -- best-effort,
    never fatal."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_HERE, capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def sha256_of_file(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

@dataclass
class ExperimentRunManifest:
    run_id: str
    utc_timestamp: str
    git_commit: str
    dataset_version_hash: str
    split_name: str
    dataset_label: str  # one of dataset_card_schema.DATASET_LABELS -- see that module's docstring
    classifier_method: str
    language_filter: Optional[str]
    n_cases: int

    hardware: HardwareInfo
    os_name: str
    python_version: str

    # Conference I Reviewer #2, Step 4 (evaluation-readiness dry-run pass) +
    # Step 5 (controlled synthetic-fixture integration run).
    # "measured" for a real run against operationally-realistic or approved
    # real data (default -- every existing caller that doesn't pass this
    # gets identical behaviour to before this field existed).
    # "dry_run_not_measured" for eval/run_eval.py --dry-run /
    # eval/ablation_runner.py --dry-run output -- orthogonal to
    # dataset_label (which describes the DATA's nature: synthetic/approved-
    # real/invalid-governance); this field describes whether metrics were
    # actually MEASURED. A dry run against a fully governance-compliant
    # real dataset card can legitimately carry
    # dataset_label=approved_real_lfs_validation AND
    # evaluation_status=dry_run_not_measured at the same time -- that
    # combination means "yes this is real, approved data, but nothing has
    # actually been measured against it yet."
    # "measured_synthetic_fixture_only" (Step 5): a REAL classification run
    # (an actual classifier was invoked, metrics are real numbers, not
    # null-by-construction like a dry run) but against the tiny, obviously-
    # synthetic eval/fixtures/synthetic_lfs_intake_package/ fixture used
    # only to integration-test this pipeline end-to-end -- never a
    # substantive evaluation. See manuscript_eligible below: this status
    # can NEVER combine with manuscript_eligible=True, unlike "measured".
    evaluation_status: str = "measured"

    # Step 5: computed automatically by build_manifest() (never caller-
    # supplied) -- True only when evaluation_status == "measured" AND
    # dataset_label == approved_real_lfs_validation. Every other
    # combination (dry run, synthetic-fixture-only, or a "measured" run
    # against merely synthetic/operationally-realistic data) is False. A
    # structural guarantee, not a documentation convention: a fixture or
    # synthetic run's manifest cannot be mistaken for manuscript-eligible
    # evidence just by reading one field.
    manuscript_eligible: bool = False

    # sha256 of the DatasetCard's canonical JSON, computed automatically
    # whenever a dataset_card is supplied to build_manifest() -- lets a
    # reader verify which exact governance submission a manifest was built
    # against without needing the card file itself.
    dataset_card_hash: Optional[str] = None

    # sha256 of a supplied split-manifest file (e.g. the dev/test split
    # assignment document) -- None + no reason needed, since omitting
    # --split-manifest is a normal, expected state for many runs (not
    # every run has a separate split-manifest file; the dev_v1/
    # test_set_full130 split discipline predates this field and is
    # enforced by eval/validate_dev_set.py instead). See build_manifest()'s
    # split_manifest_path parameter.
    split_manifest_hash: Optional[str] = None

    dependency_versions: dict[str, str] = field(default_factory=dict)
    model_versions: dict[str, str] = field(default_factory=dict)
    retrieval_params: dict = field(default_factory=dict)
    llm_params: dict = field(default_factory=dict)

    latency_mean_ms: Optional[float] = None
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    latency_unavailable_reason: Optional[str] = None

    throughput_cases_per_sec: Optional[float] = None
    throughput_unavailable_reason: Optional[str] = None

    peak_process_memory_mb: Optional[float] = None
    peak_process_memory_unavailable_reason: Optional[str] = None

    peak_gpu_memory_mb: Optional[float] = None
    peak_gpu_memory_unavailable_reason: Optional[str] = None

    retrieval_count: Optional[int] = None
    retrieval_count_unavailable_reason: Optional[str] = None

    # Mean of CaseResult.reranker_candidate_pool_size across case rows that
    # carry a value (the unique-codes-pooled-across-branches figure
    # hierarchical_store.py/hierarchy_engine.py already compute per case --
    # see backend/rag/candidate_pool.py). None + reason if no row carries
    # one (true for --system bm25/flat, which have no candidate pool, and
    # for every dry run).
    mean_retrieval_candidate_pool_size: Optional[float] = None
    mean_retrieval_candidate_pool_size_unavailable_reason: Optional[str] = None

    reranker_invocation_count: Optional[int] = None
    reranker_invocation_rate: Optional[float] = None  # reranker_invocation_count / n_cases

    hitl_escalation_rate: Optional[float] = None
    hitl_escalation_rate_unavailable_reason: Optional[str] = None

    estimated_cost_usd: Optional[float] = None
    estimated_cost_method: Optional[str] = None

    # Populated only when dataset_label was downgraded from
    # approved_real_lfs_validation to invalid_incomplete_governance (see
    # build_manifest()'s on_governance_failure="downgrade" path) -- the
    # exact governance errors that caused the downgrade, field names only,
    # never respondent data (DatasetCard never holds any).
    governance_validation_errors: list = field(default_factory=list)


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile, no numpy dependency (consistent with this
    project's hand-rolled-stats convention -- see eval/analyze.py). Uses
    ceil(pct/100 * n) rather than round() -- round() applies banker's
    rounding to exact .5 boundaries (e.g. n=5, pct=50 -> 2.5 -> round()
    gives 2, not the intuitive middle index 3), which would silently
    mis-rank the median for small, common sample sizes."""
    if not sorted_values:
        raise ValueError("cannot compute a percentile of an empty list")
    n = len(sorted_values)
    rank = max(1, min(n, math.ceil(pct / 100.0 * n)))
    return sorted_values[rank - 1]


def build_manifest(
    case_rows: list[dict],
    *,
    run_id: str,
    classifier_method: str,
    split_name: str,
    dataset_version_hash: str,
    dataset_label: str = DEFAULT_DATASET_LABEL,
    language_filter: Optional[str] = None,
    model_versions: Optional[dict[str, str]] = None,
    retrieval_params: Optional[dict] = None,
    llm_params: Optional[dict] = None,
    hardware: Optional[HardwareInfo] = None,
    dependency_versions: Optional[dict[str, str]] = None,
    git_commit: Optional[str] = None,
    utc_timestamp: Optional[str] = None,
    dataset_card=None,
    on_governance_failure: str = "raise",
    evaluation_status: str = "measured",
    split_manifest_path: Optional[Path] = None,
) -> ExperimentRunManifest:
    """
    Build one manifest from a list of case-row dicts (e.g.
    ``[r.__dict__ for r in results]`` from eval/run_eval.py's CaseResult
    list, or rows read back from a previously-written CaseResult CSV).
    Accepts plain dicts rather than the CaseResult class itself so this
    module has no import-time dependency on run_eval.py's heavy classifier
    imports (crewai/qdrant-client/sentence-transformers).

    evaluation_status : one of manifest.EVALUATION_STATUSES, default "measured"
        "measured" (default, unchanged prior behaviour) for a real run.
        "dry_run_not_measured" for eval/run_eval.py --dry-run /
        eval/ablation_runner.py --dry-run output. "measured_synthetic_
        fixture_only" (Step 5) for a real classification run against only
        the small synthetic integration fixture -- real metrics, but never
        manuscript evidence. Orthogonal to dataset_label -- see
        ExperimentRunManifest.evaluation_status's docstring for why. Any
        other value raises ValueError. manifest.manuscript_eligible is
        computed automatically from the FINAL evaluation_status and
        dataset_label (never caller-supplied) -- True only for
        evaluation_status="measured" + dataset_label=
        approved_real_lfs_validation.
    split_manifest_path : Path, optional
        If supplied, its sha256 is recorded as split_manifest_hash. Never
        read for content beyond hashing -- this module never inspects a
        split manifest's actual contents (e.g. case IDs), only proves
        which exact file version a run used.
    dataset_label : str
        MUST be one of dataset_card_schema.DATASET_LABELS -- any other
        string raises ValueError immediately (closes the gap where an
        arbitrary/mistyped label used to bypass governance checking
        entirely by never matching the one exact string this module
        checked for). See dataset_card_schema.py's module docstring for
        what each of the three labels means.
    dataset_card : eval.dataset_card_schema.DatasetCard, optional
        Fail-fast governance gate: when ``dataset_label ==
        APPROVED_REAL_LFS_VALIDATION``, this manifest cannot be built
        without a DatasetCard that passes
        eval.validate_real_lfs_governance.validate(). Ignored for every
        other dataset_label. See eval/validate_real_lfs_governance.py for
        the standalone, fail-late CLI equivalent of this same check.
    on_governance_failure : {"raise", "downgrade"}, default "raise"
        What happens when dataset_label == APPROVED_REAL_LFS_VALIDATION
        but governance validation fails. "raise" (default, preserves this
        function's original behaviour byte-for-byte): raises
        GovernanceError, no manifest is returned at all. "downgrade": does
        NOT raise -- instead returns a manifest with dataset_label forced
        to INVALID_INCOMPLETE_GOVERNANCE and governance_validation_errors
        populated with exactly why, so a rejected attempt is still visible
        evidence in generated reports rather than a silently-discarded
        exception. The requested label (APPROVED_REAL_LFS_VALIDATION)
        itself is NEVER returned on a manifest unless validation actually
        passed -- both modes agree on that; they differ only in whether
        the caller gets an exception or a downgraded, inspectable manifest.
    """
    if dataset_label not in DATASET_LABELS:
        raise ValueError(
            f"dataset_label must be exactly one of {sorted(DATASET_LABELS)}, got {dataset_label!r} "
            f"-- no free-form labels are accepted (see dataset_card_schema.DATASET_LABELS)."
        )
    if on_governance_failure not in ("raise", "downgrade"):
        raise ValueError(f"on_governance_failure must be 'raise' or 'downgrade', got {on_governance_failure!r}")
    if evaluation_status not in EVALUATION_STATUSES:
        raise ValueError(
            f"evaluation_status must be one of {EVALUATION_STATUSES}, got {evaluation_status!r}"
        )

    dataset_card_hash = None
    if dataset_card is not None:
        dataset_card_hash = hashlib.sha256(
            dataset_card.model_dump_json(exclude_none=False).encode("utf-8")
        ).hexdigest()

    split_manifest_hash = sha256_of_file(split_manifest_path) or None if split_manifest_path else None

    manifest = ExperimentRunManifest(
        run_id=run_id,
        utc_timestamp=utc_timestamp or datetime.now(timezone.utc).isoformat(),
        git_commit=git_commit if git_commit is not None else git_commit_hash(),
        dataset_version_hash=dataset_version_hash,
        split_name=split_name,
        dataset_label=dataset_label,
        classifier_method=classifier_method,
        language_filter=language_filter,
        n_cases=len(case_rows),
        evaluation_status=evaluation_status,
        dataset_card_hash=dataset_card_hash,
        split_manifest_hash=split_manifest_hash,
        hardware=hardware or probe_hardware(),
        os_name=f"{platform.system()} {platform.release()}",
        python_version=sys.version.split()[0],
        dependency_versions=dependency_versions if dependency_versions is not None else probe_dependency_versions(),
        model_versions=model_versions or {},
        retrieval_params=retrieval_params or {},
        llm_params=llm_params or {},
    )

    if dataset_label == APPROVED_REAL_LFS_VALIDATION:
        import validate_real_lfs_governance as vrg  # local import: only needed on this path

        report = vrg.validate(asdict(manifest), dataset_card)
        if not report.ok:
            if on_governance_failure == "raise":
                raise vrg.GovernanceError(
                    f"Cannot build a manifest labelled {APPROVED_REAL_LFS_VALIDATION!r}: "
                    + "; ".join(report.errors)
                )
            # on_governance_failure == "downgrade": never return a manifest
            # claiming approved status when validation failed -- force the
            # label to the third, explicitly-visible state instead.
            manifest.dataset_label = INVALID_INCOMPLETE_GOVERNANCE
            manifest.governance_validation_errors = list(report.errors)

    # manuscript_eligible: computed here (after any governance downgrade
    # above has already finalised dataset_label) so it always reflects the
    # manifest's REAL final state, never the caller's original request.
    manifest.manuscript_eligible = (
        manifest.evaluation_status == MEASURED
        and manifest.dataset_label == APPROVED_REAL_LFS_VALIDATION
    )

    if not case_rows:
        manifest.latency_unavailable_reason = "no case rows supplied"
        manifest.throughput_unavailable_reason = "no case rows supplied"
        manifest.peak_process_memory_unavailable_reason = "no case rows supplied"
        manifest.retrieval_count_unavailable_reason = "no case rows supplied"
        manifest.mean_retrieval_candidate_pool_size_unavailable_reason = "no case rows supplied"
        manifest.hitl_escalation_rate_unavailable_reason = "no case rows supplied"
        if evaluation_status == "dry_run_not_measured":
            manifest.estimated_cost_method = (
                "not measured -- dry run: no classifier was invoked, so no LLM/API cost was ever incurred"
            )
        return manifest

    # ── Latency (end_to_end_latency_ms is populated per-case by run_eval.py) ──
    latencies = [
        float(r["end_to_end_latency_ms"]) for r in case_rows
        if r.get("end_to_end_latency_ms") not in (None, "", "None")
    ]
    if latencies:
        latencies_sorted = sorted(latencies)
        manifest.latency_mean_ms = round(sum(latencies) / len(latencies), 2)
        manifest.latency_p50_ms = round(_percentile(latencies_sorted, 50), 2)
        manifest.latency_p95_ms = round(_percentile(latencies_sorted, 95), 2)
        total_s = sum(latencies) / 1000.0
        manifest.throughput_cases_per_sec = round(len(case_rows) / total_s, 4) if total_s > 0 else None
        if manifest.throughput_cases_per_sec is None:
            manifest.throughput_unavailable_reason = "sum of per-case latencies was 0 -- cannot divide"
    else:
        manifest.latency_unavailable_reason = "no case row carried a non-empty end_to_end_latency_ms"
        manifest.throughput_unavailable_reason = "latency data unavailable (see latency_unavailable_reason)"

    # ── Peak process memory: honest gap -- see module docstring ──────────
    mem_values = [
        float(r["peak_memory_mb"]) for r in case_rows
        if r.get("peak_memory_mb") not in (None, "", "None")
    ]
    if mem_values:
        manifest.peak_process_memory_mb = round(max(mem_values), 2)
    else:
        manifest.peak_process_memory_unavailable_reason = (
            "CaseResult.peak_memory_mb is not populated by any current eval/run_eval.py code path "
            "(declared field, never assigned) -- not measured in this run, not measured in any run to date"
        )

    # ── GPU memory: from the hardware probe only (peak DURING the run is
    # not separately tracked; this is a device-capacity figure, not a
    # per-run peak-usage figure -- documented distinction) ──────────────
    if manifest.hardware.gpu_mem_gb is None:
        manifest.peak_gpu_memory_unavailable_reason = (
            manifest.hardware.gpu_unavailable_reason or "no GPU detected"
        )

    # ── Retrieval count: genuinely not instrumented -- see module docstring ──
    manifest.retrieval_count_unavailable_reason = (
        "CaseResult does not log a per-case count of underlying Qdrant "
        "query_points() calls -- only candidate lists and per-stage "
        "latency, neither a reliable proxy without guessing at beam "
        "branching. Not measured."
    )

    # ── Retrieval candidate pool size: a genuine per-case figure
    # (reranker_candidate_pool_size) for --system hierarchical -- None +
    # reason for bm25/flat (no pool concept there) or a dry run (no
    # retrieval ever ran) ────────────────────────────────────────────
    pool_size_values = [
        int(r["reranker_candidate_pool_size"]) for r in case_rows
        if r.get("reranker_candidate_pool_size") not in (None, "", "None")
    ]
    if pool_size_values:
        manifest.mean_retrieval_candidate_pool_size = round(sum(pool_size_values) / len(pool_size_values), 2)
    else:
        manifest.mean_retrieval_candidate_pool_size_unavailable_reason = (
            "no case row carried a non-empty reranker_candidate_pool_size value "
            "(expected for --system bm25/flat, which have no candidate pool, or a dry run)"
        )

    # ── Reranker invocation count/rate: a genuine count (reranker_fired is
    # a real per-case bool) ─────────────────────────────────────────────
    fired_values = [r.get("reranker_fired") for r in case_rows if "reranker_fired" in r]
    fired_bools = [v for v in fired_values if v not in (None, "", "None")]
    if fired_bools:
        manifest.reranker_invocation_count = sum(
            1 for v in fired_bools if v is True or str(v).strip().lower() == "true"
        )
        manifest.reranker_invocation_rate = round(manifest.reranker_invocation_count / len(case_rows), 4)

    # ── HITL escalation rate: the eval harness's OWN research-only
    # heuristic (escalation_triggered), NOT production HITLQualityManager
    # -- see module docstring ──────────────────────────────────────────
    esc_values = [r.get("escalation_triggered") for r in case_rows if "escalation_triggered" in r]
    esc_bools = [v for v in esc_values if v not in (None, "", "None")]
    if esc_bools:
        n_escalated = sum(1 for v in esc_bools if v is True or str(v).strip().lower() == "true")
        manifest.hitl_escalation_rate = round(n_escalated / len(esc_bools), 4)
    else:
        manifest.hitl_escalation_rate_unavailable_reason = (
            "no case row carried a non-empty escalation_triggered value"
        )

    # ── Cost: sum of per-case estimated_cost_usd (already computed by
    # run_eval.py, e.g. via LiteLLM cost estimation for API-backed
    # rerankers; 0.0 for every row on a local Ollama run, which is a real
    # zero, not a missing value) ──────────────────────────────────────
    cost_values = [
        float(r["estimated_cost_usd"]) for r in case_rows
        if r.get("estimated_cost_usd") not in (None, "", "None")
    ]
    if cost_values:
        manifest.estimated_cost_usd = round(sum(cost_values), 6)
        manifest.estimated_cost_method = "sum of per-case CaseResult.estimated_cost_usd (LiteLLM cost estimation)"

    # ── Dry-run: never let a trivially-echoed default (e.g.
    # estimated_cost_usd's dataclass default of 0.0, which every dry-run
    # row satisfies the "non-empty" check with) read as if it were a real
    # measurement. This is the one metric that can slip through the
    # generic null+reason logic above for a dry run, because 0.0 is a
    # valid, non-blank CSV value -- so it needs an explicit override here
    # rather than relying on absence. See "Ensure no dry-run output can
    # accidentally be consumed as a completed result" in the Step 4 task.
    if evaluation_status == "dry_run_not_measured":
        manifest.estimated_cost_usd = None
        manifest.estimated_cost_method = (
            "not measured -- dry run: no classifier was invoked, so no LLM/API cost was ever incurred"
        )

    return manifest


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _flatten(manifest: ExperimentRunManifest) -> dict:
    d = asdict(manifest)
    hw = d.pop("hardware")
    for k, v in hw.items():
        d[f"hardware_{k}"] = v
    for json_field in ("dependency_versions", "model_versions", "retrieval_params", "llm_params"):
        d[json_field] = json.dumps(d[json_field], sort_keys=True, ensure_ascii=False)
    return d


def write_manifest(manifest: ExperimentRunManifest, out_dir: Path) -> tuple[Path, Path]:
    """Writes one CSV row (flattened) + one JSONL line (full nested
    structure) -- matching eval/run_eval.py's existing CSV+JSONL convention.
    Returns (csv_path, jsonl_path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"manifest_{manifest.run_id}"
    csv_path = out_dir / f"{base}.csv"
    jsonl_path = out_dir / f"{base}.jsonl"

    flat = _flatten(manifest)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(asdict(manifest), ensure_ascii=False) + "\n")

    return csv_path, jsonl_path
