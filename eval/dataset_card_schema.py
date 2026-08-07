"""
eval/dataset_card_schema.py

Pydantic schema for a dataset card documenting the governance of a
real-world Labour Force Survey dataset (Conference I Reviewer #2 response,
Section F; expanded in the Step 3 real-LFS-intake hardening pass).
Mirrors
Documentation/Conference_I_Reviewer_2/REAL_LFS_VALIDATION_DATASET_CARD_TEMPLATE.md
field-for-field.

NO real respondent data lives in this module or anywhere in this repo --
this schema records METADATA ABOUT a dataset (its provenance, ethics
status, labelling process, split discipline, privacy posture), never the
dataset's actual records. See eval/validate_real_lfs_governance.py for the
validator that uses this schema to block mislabelled experiments, and
eval/fixtures/ for SYNTHETIC examples used only by tests.

Dataset labels (governance status)
------------------------------------
Every experiment-run manifest / generated report carries EXACTLY one of
three labels -- see DATASET_LABELS below. There is no free-form label: any
other string is rejected by eval/manifest.py and eval/ablation_runner.py at
the point they'd otherwise attach a label to output, closing the gap where
an arbitrary string (e.g. a typo'd or differently-cased variant of the old
"real_lfs_validation" value) used to bypass governance checking entirely.

    synthetic_or_operationally_realistic
        No real respondent data. Default for everything -- smoke tests,
        dev-set parameter selection, synthetic fixtures, operationally
        realistic (but not respondent-derived) traffic. No governance
        checks apply.
    approved_real_lfs_validation
        Real, permissioned Labour Force Survey respondent data, with a
        DatasetCard that has passed eval.validate_real_lfs_governance's
        full governance/provenance/annotation/privacy/split check. This is
        the ONLY label that may ever appear in manuscript wording like
        "validated on real LFS data" -- see REAL_LFS_DATA_INTAKE_CHECKLIST.md.
    invalid_incomplete_governance
        Someone attempted approved_real_lfs_validation but the DatasetCard
        failed governance validation. The attempt is still recorded (run
        details, which checks failed) so a rejected attempt is visible
        evidence, not a silently discarded exception -- but this label can
        NEVER be requested directly; it is only ever assigned by
        eval/manifest.py when an approved_real_lfs_validation request is
        downgraded after failing validation.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Dataset labels -- the single source of truth for the closed 3-value enum.
# Every other module (manifest.py, ablation_runner.py,
# validate_real_lfs_governance.py) imports these rather than redefining
# label strings, so there is exactly one place that can ever add a 4th value.
# ---------------------------------------------------------------------------

SYNTHETIC_OR_OPERATIONALLY_REALISTIC = "synthetic_or_operationally_realistic"
APPROVED_REAL_LFS_VALIDATION = "approved_real_lfs_validation"
INVALID_INCOMPLETE_GOVERNANCE = "invalid_incomplete_governance"

DatasetLabel = Literal[
    "synthetic_or_operationally_realistic",
    "approved_real_lfs_validation",
    "invalid_incomplete_governance",
]

DATASET_LABELS: frozenset[str] = frozenset({
    SYNTHETIC_OR_OPERATIONALLY_REALISTIC,
    APPROVED_REAL_LFS_VALIDATION,
    INVALID_INCOMPLETE_GOVERNANCE,
})

DEFAULT_DATASET_LABEL = SYNTHETIC_OR_OPERATIONALLY_REALISTIC


class DatasetCard(BaseModel):
    """
    Governance/provenance metadata for one dataset. Fields kept as they
    were (required, no default) predate this pass and stay required for
    backward compatibility with every existing caller/fixture. Every field
    added in this pass is Optional[...] = None at the Pydantic level --
    NONE of them are required to construct a DatasetCard at all (a
    synthetic/dev-selection card still only needs the original fields) --
    but eval/validate_real_lfs_governance.py's REQUIRED_FOR_APPROVED_*
    lists make almost all of them mandatory specifically for the
    approved_real_lfs_validation label. This mirrors the existing
    ethics_approval_ref/deidentification_method pattern: optional on the
    model, mandatory in the validator, for exactly the standards/level
    that need it.
    """

    # ── 1. Dataset identity ────────────────────────────────────────────
    source: str = Field(..., description="Who/what collected this dataset, e.g. 'UAE MOHRE LFS 2023 wave', 'Synthetic — not real respondent data'.")
    dataset_id: Optional[str] = Field(None, description="A stable identifier for this dataset (custodian-assigned or internally minted).")
    data_owner_or_custodian: Optional[str] = Field(None, description="The organisation/individual with legal custody of the data (may differ from 'source').")
    region: str = Field(..., description="Country/region the data was collected in.")
    survey_programme_name: Optional[str] = Field(None, description="The name of the survey programme this data comes from, e.g. 'UAE Labour Force Survey'.")
    wave: str = Field(..., description="Survey wave/round identifier, e.g. 'Q3 2023'.")
    collection_start_date: str = Field(..., description="ISO date (YYYY-MM-DD) data collection began.")
    collection_end_date: str = Field(..., description="ISO date (YYYY-MM-DD) data collection ended.")
    languages: list[str] = Field(default_factory=list, description="Languages present in the raw respondent text.")
    dataset_version: Optional[str] = Field(None, description="Version/release identifier for this specific dataset extract.")
    dataset_hash: Optional[str] = Field(None, description="A hash (e.g. sha256) the custodian computed over their own local dataset file, recorded here for immutability/provenance tracking -- NEVER the file itself.")

    # ── 2. Governance ───────────────────────────────────────────────────
    data_sharing_agreement_ref: Optional[str] = Field(None, description="Reference/identifier for the data-sharing agreement or approval permitting this use.")
    ethics_approval_ref: Optional[str] = Field(None, description="Ethics/IRB approval, exemption, or governance-review reference number. Required for approved_real_lfs_validation.")
    consent_obtained: bool = Field(..., description="Whether informed consent was obtained from respondents (or another lawful basis applies -- see lawful_basis_description).")
    lawful_basis_description: Optional[str] = Field(None, description="Free-text description of the consent process or other lawful basis for using this data.")
    permitted_research_purpose: Optional[str] = Field(None, description="The specific research purpose(s) this data is permitted to be used for.")
    retention_period: Optional[str] = Field(None, description="How long this data may be retained, per the governing agreement.")
    access_control_description: Optional[str] = Field(None, description="Who may access this data and how access is restricted/audited.")
    deidentification_method: Optional[str] = Field(None, description="How PII was removed/pseudonymised before this dataset was used. Required for approved_real_lfs_validation.")
    deidentification_status: Optional[str] = Field(None, description="Current de-identification status, e.g. 'fully de-identified', 'pseudonymised with re-identification key held by custodian'.")
    prohibited_disclosure_rules: Optional[str] = Field(None, description="Any outputs/disclosures that are prohibited under the governing agreement (e.g. cell sizes below a threshold, verbatim free text).")

    # ── 3. Classification labels ────────────────────────────────────────
    classification_standard_version: str = Field(..., description="Legacy summary field: exact standard + version the gold labels use, e.g. 'ISCO-08'. Superseded for approved_real_lfs_validation by the four standard-specific *_version fields below.")
    isco08_version: Optional[str] = Field(None, description="Exact ISCO-08 version/edition the gold occupation labels use.")
    isic_rev4_version: Optional[str] = Field(None, description="Exact ISIC Rev.4 version/edition the gold industry labels use.")
    isced2011_version: Optional[str] = Field(None, description="Exact ISCED 2011 version/edition the gold education-level labels use.")
    iscedf2013_version: Optional[str] = Field(None, description="Exact ISCED-F 2013 version/edition the gold education-field labels use.")
    label_source: Optional[str] = Field(None, description="How gold labels were produced, e.g. 'expert human coder', 'administrative record match'.")
    labelling_process: str = Field(..., description="How gold labels were produced, e.g. 'single coder', 'double-coded with adjudication'.")
    labeler_type: Optional[str] = Field(None, description="Type of labeler, e.g. 'professional statistical coder', 'trained research assistant'.")
    coder_qualifications: str = Field(..., description="Qualifications/training of the human coder(s) who produced gold labels.")
    double_coded: bool = Field(..., description="Whether every record was independently coded twice and adjudicated.")
    adjudication_process: Optional[str] = Field(None, description="How disagreements between coders were resolved, if double_coded=True.")
    inter_annotator_agreement: Optional[float] = Field(None, description="Measured inter-annotator agreement (e.g. Cohen's kappa), 0-1. Nullable until actually measured -- absence is not itself a failure, but see the validator for the accompanying warning.")

    # ── 4. Evaluation discipline ────────────────────────────────────────
    split_manifest_ref: Optional[str] = Field(None, description="Reference/filename for the train/development/test split manifest (never the data itself).")
    split_hash: Optional[str] = Field(None, description="Hash of the split manifest/assignment, for reproducibility verification.")
    frozen_heldout_test_status: Optional[str] = Field(None, description="Confirmation that the held-out test split is frozen and has never been used for parameter selection, e.g. 'frozen as of 2026-08-07'.")
    parameter_selection_dataset_ref: Optional[str] = Field(None, description="Reference to the SEPARATE dataset/split used for parameter selection (must differ from the frozen test split).")
    test_set_access_restriction: Optional[str] = Field(None, description="Who may access the frozen test set and under what conditions, to prevent leakage.")
    sample_counts_by_language_and_task: Optional[dict] = Field(None, description="Sample counts broken down by language and classification task, e.g. {'en': {'isco': 50, 'isic': 50}}.")
    missing_data_policy: Optional[str] = Field(None, description="How missing/incomplete responses are handled in evaluation.")
    exclusion_criteria: Optional[str] = Field(None, description="Criteria used to exclude records from this dataset/evaluation.")

    # ── 5. Privacy ───────────────────────────────────────────────────────
    direct_identifiers_removed_confirmed: Optional[bool] = Field(None, description="Explicit confirmation that direct identifiers (name, national ID, phone, exact address, etc.) have been removed. Required True for approved_real_lfs_validation.")
    quasi_identifier_risk_note: Optional[str] = Field(None, description="Assessment of re-identification risk from quasi-identifiers (e.g. rare occupation + small region + age).")
    free_text_redaction_method: Optional[str] = Field(None, description="How free-text respondent answers were screened/redacted for incidental PII before use.")
    secure_storage_location_description: Optional[str] = Field(None, description="DESCRIPTION (not a literal path to real data) of where the real dataset is stored -- must be outside this Git repository, e.g. 'encrypted institutional drive, access restricted to named PI'.")
    raw_records_excluded_from_git_confirmed: Optional[bool] = Field(None, description="Explicit confirmation that raw respondent records are excluded from source control. Required True for approved_real_lfs_validation.")
