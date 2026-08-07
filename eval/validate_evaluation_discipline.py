"""
eval/validate_evaluation_discipline.py

Conference I Reviewer #2 response, Step 4 (evaluation-readiness pass), task D
("enforce evaluation discipline"). Two independent checks, each returning a
DisciplineReport (never raising on its own -- callers decide whether a failed
report blocks a run):

  validate_split_manifest(split_manifest)
      -- rejects a missing/absent split manifest (by construction: caller
         must have one to validate) and rejects identical dev/heldout
         split_id values (a same-partition dev/test split is a data-leakage
         risk: any parameter chosen on "dev" would have been implicitly
         chosen on the confirmation set too).

  validate_manifest_discipline(manifest)
      -- rejects an absent/blank dataset_version_hash.
      -- rejects a manifest claiming evaluation_status == "measured" (i.e.
         NOT a dry run) while every real measurement field is null -- that
         combination means the run claims to be a completed measurement but
         recorded nothing, which is indistinguishable from a mislabelled dry
         run and must never be treated as a manuscript-ready result.

This module does not duplicate eval/validate_real_lfs_governance.py (which
already rejects an approved_real_lfs_validation label without a passing
governance check, and eval/dataset_card_schema.py's manuscript-wording guard,
which already rejects a synthetic run being described as manuscript-ready
real validation) -- see eval/test_validate_evaluation_discipline.py for tests
that exercise all six Task-D rejection scenarios together, re-using those
existing guards for the two already covered elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from split_manifest_schema import SplitManifest

# Manifest fields that, taken together, indicate SOME real measurement was
# recorded. If evaluation_status == "measured" and every one of these is
# still None, the run has nothing to show for itself.
_CORE_MEASUREMENT_FIELDS = (
    "latency_mean_ms",
    "throughput_cases_per_sec",
    "peak_process_memory_mb",
    "retrieval_count",
    "reranker_invocation_count",
    "hitl_escalation_rate",
    "estimated_cost_usd",
)


@dataclass
class DisciplineReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_split_manifest(split_manifest: Optional[SplitManifest]) -> DisciplineReport:
    """Reject a missing split manifest or identical dev/heldout split IDs."""
    errors: list[str] = []
    warnings: list[str] = []

    if split_manifest is None:
        return DisciplineReport(ok=False, errors=["split manifest is missing -- a run cannot be trusted without a recorded dev/heldout split assignment"])

    splits = split_manifest.splits
    if "dev" not in splits:
        errors.append("split manifest is missing a 'dev' split entry")
    if "heldout" not in splits:
        errors.append("split manifest is missing a 'heldout' split entry")

    if "dev" in splits and "heldout" in splits:
        dev_id = splits["dev"].split_id
        heldout_id = splits["heldout"].split_id
        if not dev_id or not heldout_id:
            errors.append("split manifest has a blank split_id for 'dev' and/or 'heldout'")
        elif dev_id == heldout_id:
            errors.append(
                f"dev split_id and heldout split_id are identical ({dev_id!r}) -- "
                "this means dev and heldout are the same partition, so any "
                "parameter selected on dev would be implicitly selected on "
                "the confirmation set too (data leakage)"
            )

    if not split_manifest.leakage_check_performed:
        warnings.append("split manifest does not record that a case-ID/text-overlap leakage check was performed between splits")

    return DisciplineReport(ok=not errors, errors=errors, warnings=warnings)


def validate_manifest_discipline(manifest) -> DisciplineReport:
    """Reject an absent dataset hash, or a 'measured' run with only null metrics.

    Accepts an eval.manifest.ExperimentRunManifest (or any object exposing
    the same attribute names -- kept duck-typed to avoid eval/manifest.py and
    this module importing each other).
    """
    errors: list[str] = []
    warnings: list[str] = []

    dataset_version_hash = getattr(manifest, "dataset_version_hash", None)
    if not dataset_version_hash:
        errors.append("dataset_version_hash is missing/blank -- every run must record which dataset version it evaluated against")

    evaluation_status = getattr(manifest, "evaluation_status", "measured")
    if evaluation_status == "measured":
        all_null = all(getattr(manifest, f, None) is None for f in _CORE_MEASUREMENT_FIELDS)
        if all_null:
            errors.append(
                "evaluation_status='measured' but every core measurement field "
                f"({', '.join(_CORE_MEASUREMENT_FIELDS)}) is null -- a completed "
                "run must record at least one real measurement, or it must be "
                "relabelled evaluation_status='dry_run_not_measured'"
            )

    split_manifest_hash = getattr(manifest, "split_manifest_hash", None)
    if not split_manifest_hash:
        warnings.append("split_manifest_hash is missing -- this run cannot be tied to a recorded dev/heldout split assignment")

    return DisciplineReport(ok=not errors, errors=errors, warnings=warnings)
