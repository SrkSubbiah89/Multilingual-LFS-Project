"""
eval/validate_controlled_benchmark.py

Conference I Reviewer #2 response, Step 6, Phase F: eligibility validator
for a CONTROLLED BENCHMARK package (list of
controlled_benchmark_schema.BenchmarkRecord). Fail-closed, mirroring
eval/validate_real_lfs_governance.py's pattern: a pure validation function
returning a report with .ok/errors/warnings, no side effects, and error
messages that never echo back input_text/gold_code content (only field
names, counts, and record IDs) so a validation report itself can never leak
a benchmark's actual content.

Usage
-----
    from controlled_benchmark_schema import BenchmarkRecord
    from validate_controlled_benchmark import validate_benchmark_package
    report = validate_benchmark_package(records)
    if not report.ok:
        ...
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from controlled_benchmark_schema import (  # noqa: E402
    ADJUDICATION_STATUSES,
    BENCHMARK_TASKS,
    BenchmarkRecord,
    DATA_SOURCE_TYPES,
    DOUBLE_CODING_STATUSES,
    INDEPENDENT_LABEL_STATUS_NOT_INDEPENDENT,
    INDEPENDENT_LABEL_STATUSES,
    LABEL_SOURCE_SYSTEM_SELF_GENERATED,
    LABEL_SOURCE_TYPES,
    LABEL_SOURCE_UNKNOWN,
    SPLIT_DEV,
    SPLIT_EXCLUDED,
    SPLIT_HELDOUT,
    SPLITS,
)
from dataset_card_schema import APPROVED_REAL_LFS_VALIDATION, DATASET_LABELS  # noqa: E402


@dataclass
class ValidationReport:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _record_ref(r: BenchmarkRecord) -> str:
    """A record reference safe to put in an error message -- the ID only,
    never input_text/gold_code/labeler_identifier content."""
    return f"benchmark_id={r.benchmark_id!r}"


def validate_benchmark_package(
    records: list[BenchmarkRecord],
    dataset_card=None,
    dataset_hash: Optional[str] = None,
) -> ValidationReport:
    """
    Rejects a package represented as benchmark-ready when (Step 6, Phase F.2):
      - gold labels are absent (for any non-excluded, non-ambiguous record)
      - provenance is absent (unknown/blank source or labeler fields)
      - the system generated its own gold labels (self-generated label source,
        or independent_label_status == not_independent)
      - development and test splits overlap (same benchmark_id in both)
      - hashes are absent (per-record record_hash, or the package-level
        dataset_hash argument)
      - the dataset claims real LFS status without governance approval
        (dataset_label == approved_real_lfs_validation without a DatasetCard
        that passes eval.validate_real_lfs_governance.validate())

    Never echoes input_text/gold_code/labeler_identifier_or_role content in
    any error/warning message -- only record IDs, field names, and counts.
    """
    report = ValidationReport()

    if not records:
        report.errors.append("benchmark package is empty -- no records supplied")
        return report

    if dataset_hash is None or not dataset_hash.strip():
        report.errors.append("package-level dataset_hash is missing -- every benchmark package must record a stable hash")

    dev_ids: set[str] = set()
    heldout_ids: set[str] = set()
    seen_ids: set[str] = set()

    for r in records:
        ref = _record_ref(r)

        if r.benchmark_id in seen_ids:
            report.errors.append(f"duplicate benchmark_id ({ref})")
        seen_ids.add(r.benchmark_id)

        if r.task not in BENCHMARK_TASKS:
            report.errors.append(f"{ref}: task {r.task!r} is not one of {sorted(BENCHMARK_TASKS)}")
        if r.data_source_type not in DATA_SOURCE_TYPES:
            report.errors.append(f"{ref}: data_source_type {r.data_source_type!r} is not one of {sorted(DATA_SOURCE_TYPES)}")
        if r.label_source_type not in LABEL_SOURCE_TYPES:
            report.errors.append(f"{ref}: label_source_type {r.label_source_type!r} is not one of {sorted(LABEL_SOURCE_TYPES)}")
        if r.independent_label_status not in INDEPENDENT_LABEL_STATUSES:
            report.errors.append(f"{ref}: independent_label_status {r.independent_label_status!r} is not one of {sorted(INDEPENDENT_LABEL_STATUSES)}")
        if r.double_coding_status not in DOUBLE_CODING_STATUSES:
            report.errors.append(f"{ref}: double_coding_status {r.double_coding_status!r} is not one of {sorted(DOUBLE_CODING_STATUSES)}")
        if r.adjudication_status not in ADJUDICATION_STATUSES:
            report.errors.append(f"{ref}: adjudication_status {r.adjudication_status!r} is not one of {sorted(ADJUDICATION_STATUSES)}")
        if r.split not in SPLITS:
            report.errors.append(f"{ref}: split {r.split!r} is not one of {sorted(SPLITS)}")
        if r.dataset_label not in DATASET_LABELS:
            report.errors.append(f"{ref}: dataset_label {r.dataset_label!r} is not one of {sorted(DATASET_LABELS)}")

        # ── Provenance absent ────────────────────────────────────────────
        if r.label_source_type == LABEL_SOURCE_UNKNOWN:
            report.errors.append(f"{ref}: label_source_type is 'unknown' -- provenance is absent")
        if not r.labeler_identifier_or_role.strip():
            report.errors.append(f"{ref}: labeler_identifier_or_role is blank -- provenance is absent")

        # ── System self-generated gold labels ────────────────────────────
        if r.label_source_type == LABEL_SOURCE_SYSTEM_SELF_GENERATED:
            report.errors.append(f"{ref}: label_source_type is 'system_self_generated' -- the system under evaluation cannot be its own gold-label source")
        if r.independent_label_status == INDEPENDENT_LABEL_STATUS_NOT_INDEPENDENT:
            report.errors.append(f"{ref}: independent_label_status is 'not_independent' -- gold label was not produced independently of the system under evaluation")

        # ── Gold labels absent ────────────────────────────────────────────
        if r.split != SPLIT_EXCLUDED and not r.ambiguity_flag:
            if not r.gold_code or not r.gold_code.strip():
                report.errors.append(f"{ref}: gold_code is blank on a non-excluded, non-ambiguous record")
        if r.ambiguity_flag and not (r.exclusion_reason and r.exclusion_reason.strip()):
            report.errors.append(f"{ref}: ambiguity_flag is true but exclusion_reason is blank -- an ambiguous case must explain why, never silently forced to a gold code")
        if r.split == SPLIT_EXCLUDED and not (r.exclusion_reason and r.exclusion_reason.strip()):
            report.errors.append(f"{ref}: split is 'excluded' but exclusion_reason is blank")

        # ── Hashes absent (per-record) ───────────────────────────────────
        if not r.record_hash or not r.record_hash.strip():
            report.errors.append(f"{ref}: record_hash is missing")

        # ── Split bookkeeping ─────────────────────────────────────────────
        if r.split == SPLIT_DEV:
            dev_ids.add(r.benchmark_id)
        elif r.split == SPLIT_HELDOUT:
            heldout_ids.add(r.benchmark_id)

    overlap = dev_ids & heldout_ids
    if overlap:
        report.errors.append(
            f"development and heldout splits overlap on {len(overlap)} record(s) "
            "-- a record cannot be in both splits"
        )
    if not dev_ids:
        report.warnings.append("no records assigned to the 'dev' split")
    if not heldout_ids:
        report.warnings.append("no records assigned to the 'heldout' split")

    # ── Real-LFS status claimed without governance approval ─────────────
    claims_approved_real = any(r.dataset_label == APPROVED_REAL_LFS_VALIDATION for r in records)
    if claims_approved_real:
        import validate_real_lfs_governance as vrg  # local import: only needed on this path

        gov_report = vrg.validate({"dataset_label": APPROVED_REAL_LFS_VALIDATION, "split_name": "heldout"}, dataset_card)
        if not gov_report.ok:
            report.errors.append(
                "one or more records claim dataset_label=approved_real_lfs_validation but "
                "governance validation did not pass: " + "; ".join(gov_report.errors)
            )
        report.errors.append(
            "a controlled benchmark package must never claim dataset_label=approved_real_lfs_validation "
            "-- a controlled benchmark is reference/example data, not Labour Force Survey respondent data, "
            "regardless of governance status"
        )

    return report
