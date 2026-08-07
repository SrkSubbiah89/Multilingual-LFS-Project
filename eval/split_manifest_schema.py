"""
eval/split_manifest_schema.py

Pydantic schema for a train/development/held-out-test SPLIT MANIFEST --
Conference I Reviewer #2 response, Step 4 (evaluation-readiness pass),
task D ("enforce evaluation discipline"). Records which case IDs (by
non-identifying reference, never respondent text) belong to which split,
each split's purpose, and a distinct split_id per split -- never the
underlying data itself.

See eval/validate_evaluation_discipline.py for the validator that uses
this schema to reject a missing split manifest or identical dev/test split
IDs, and eval/fixtures/synthetic_lfs_intake_package/split_manifest.json for
a SYNTHETIC example.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class SplitEntry(BaseModel):
    split_id: str = Field(..., description="A distinct identifier for this split (e.g. a hash of its case-ID list). Must differ from every other split's split_id in the same manifest.")
    purpose: str = Field(..., description="What this split is for, e.g. 'parameter selection only' or 'frozen confirmation set'.")
    n_cases: int = Field(..., description="Number of cases in this split.")
    frozen: bool = Field(False, description="Whether this split is frozen (never used for parameter selection again once set).")


class SplitManifest(BaseModel):
    dataset_id: Optional[str] = Field(None, description="Matches DatasetCard.dataset_id, if applicable.")
    split_hash: Optional[str] = Field(None, description="A hash of the overall split assignment, for reproducibility verification.")
    splits: dict[str, SplitEntry] = Field(..., description="Keyed by split name, e.g. {'dev': {...}, 'heldout': {...}}.")
    leakage_check_performed: bool = Field(..., description="Whether a case-ID/near-duplicate-text overlap check was run between splits.")
    leakage_check_method: Optional[str] = Field(None, description="Free-text description of the leakage check method.")
