"""
eval/controlled_benchmark_schema.py

Conference I Reviewer #2 response, Step 6 ("audit and prepare a valid
controlled benchmark dataset"). Pydantic schema for a CONTROLLED-BENCHMARK
record -- distinct from eval/dataset_card_schema.py's DatasetCard, which
governs REAL LFS RESPONDENT data intake. A controlled benchmark is
reference/example data (e.g. an occupation-title dictionary, or authored
test cases) used to measure classifier accuracy against known-correct
codes -- it is never Labour Force Survey respondent data and must never be
labelled `approved_real_lfs_validation` (see DATASET_LABELS below; a
controlled benchmark is, definitionally, always
`synthetic_or_operationally_realistic`).

See Documentation/Conference_I_Reviewer_2/CONTROLLED_BENCHMARK_AUDIT.md for
the audit this schema supports and
Documentation/Conference_I_Reviewer_2/CONTROLLED_BENCHMARK_DATASET_CARD_TEMPLATE.md
for the dataset-level (not per-record) card template.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from dataset_card_schema import DATASET_LABELS, DEFAULT_DATASET_LABEL  # noqa: E402


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------

ISCO08 = "isco08"
ISIC_REV4 = "isic_rev4"
ISCED2011 = "isced2011"
ISCEDF2013 = "iscedf2013"
BENCHMARK_TASKS = frozenset({ISCO08, ISIC_REV4, ISCED2011, ISCEDF2013})

DATA_SOURCE_EXTERNAL_AUTHORITATIVE = "external_authoritative_publisher"
DATA_SOURCE_AUTHORED_BY_PROJECT = "authored_by_project_team"
DATA_SOURCE_CROWD_OR_SURVEY_PANEL = "crowd_or_survey_panel"
DATA_SOURCE_UNKNOWN = "unknown"
DATA_SOURCE_TYPES = frozenset({
    DATA_SOURCE_EXTERNAL_AUTHORITATIVE, DATA_SOURCE_AUTHORED_BY_PROJECT,
    DATA_SOURCE_CROWD_OR_SURVEY_PANEL, DATA_SOURCE_UNKNOWN,
})

LABEL_SOURCE_EXTERNAL_PUBLISHER = "external_publisher_gold"
LABEL_SOURCE_HUMAN_SINGLE_CODER = "human_single_coder"
LABEL_SOURCE_HUMAN_DOUBLE_CODED = "human_double_coded_adjudicated"
LABEL_SOURCE_SYSTEM_SELF_GENERATED = "system_self_generated"  # NEVER valid as benchmark-ready gold -- see validator
LABEL_SOURCE_UNKNOWN = "unknown"
LABEL_SOURCE_TYPES = frozenset({
    LABEL_SOURCE_EXTERNAL_PUBLISHER, LABEL_SOURCE_HUMAN_SINGLE_CODER,
    LABEL_SOURCE_HUMAN_DOUBLE_CODED, LABEL_SOURCE_SYSTEM_SELF_GENERATED,
    LABEL_SOURCE_UNKNOWN,
})

INDEPENDENT_LABEL_STATUS_INDEPENDENT = "independent_of_system_under_test"
INDEPENDENT_LABEL_STATUS_NOT_INDEPENDENT = "not_independent"
INDEPENDENT_LABEL_STATUS_UNKNOWN = "unknown"
INDEPENDENT_LABEL_STATUSES = frozenset({
    INDEPENDENT_LABEL_STATUS_INDEPENDENT, INDEPENDENT_LABEL_STATUS_NOT_INDEPENDENT,
    INDEPENDENT_LABEL_STATUS_UNKNOWN,
})

DOUBLE_CODING_YES = "double_coded"
DOUBLE_CODING_SINGLE = "single_coded"
DOUBLE_CODING_NOT_APPLICABLE = "not_applicable_external_source"
DOUBLE_CODING_UNKNOWN = "unknown"
DOUBLE_CODING_STATUSES = frozenset({
    DOUBLE_CODING_YES, DOUBLE_CODING_SINGLE, DOUBLE_CODING_NOT_APPLICABLE, DOUBLE_CODING_UNKNOWN,
})

ADJUDICATION_RESOLVED = "resolved_by_adjudicator"
ADJUDICATION_NOT_NEEDED = "not_needed_no_disagreement"
ADJUDICATION_NOT_APPLICABLE = "not_applicable"
ADJUDICATION_UNKNOWN = "unknown"
ADJUDICATION_STATUSES = frozenset({
    ADJUDICATION_RESOLVED, ADJUDICATION_NOT_NEEDED, ADJUDICATION_NOT_APPLICABLE, ADJUDICATION_UNKNOWN,
})

SPLIT_DEV = "dev"
SPLIT_HELDOUT = "heldout"
SPLIT_EXCLUDED = "excluded"
SPLITS = frozenset({SPLIT_DEV, SPLIT_HELDOUT, SPLIT_EXCLUDED})


class BenchmarkRecord(BaseModel):
    """One controlled-benchmark example. Every field the Step 6 task
    specification requires, verbatim, is present below."""

    benchmark_id: str = Field(..., description="Stable, unique identifier for this record within its benchmark package. Never a real respondent identifier.")
    task: str = Field(..., description="One of controlled_benchmark_schema.BENCHMARK_TASKS.")
    data_source_type: str = Field(..., description="One of DATA_SOURCE_TYPES -- where this record's input text and gold code originated.")
    dataset_label: str = Field(DEFAULT_DATASET_LABEL, description="One of dataset_card_schema.DATASET_LABELS. A controlled benchmark built from reference/dictionary data is synthetic_or_operationally_realistic by definition -- never approved_real_lfs_validation.")

    language: str = Field(..., description="BCP-47-ish language code, e.g. 'en', 'ar', 'ur', 'hi', 'tl'.")
    input_text: str = Field(..., description="The text to be classified (occupation title, industry description, education description, etc.). Never real respondent free text unless governed separately as approved_real_lfs_validation.")
    context_fields: Optional[dict] = Field(None, description="Optional extra context (e.g. industry_text/education_text for cross-standard SRE evaluation), only when applicable to the task.")

    gold_code: str = Field(..., description="The reference/gold classification code. May be blank ONLY if ambiguity_flag is true and exclusion_reason explains why (see validator).")
    gold_code_title: str = Field("", description="Human-readable title of gold_code, for reviewer legibility. May be blank if not available from the source.")
    classification_standard: str = Field(..., description="e.g. 'ISCO-08', 'ISIC Rev.4', 'ISCED 2011', 'ISCED-F 2013'.")
    classification_version: str = Field(..., description="The exact declared version/edition of the standard.")
    hierarchy_level: str = Field(..., description="e.g. 'unit_group_4digit', 'section', 'level', 'detailed_field'.")

    label_source_type: str = Field(..., description="One of LABEL_SOURCE_TYPES.")
    labeler_identifier_or_role: str = Field(..., description="Non-personal identifier or role only -- e.g. 'external_publisher:wisco', 'coder_role:senior_annotator'. NEVER a real name or other direct identifier.")
    independent_label_status: str = Field(..., description="One of INDEPENDENT_LABEL_STATUSES -- whether the gold label is independent of the system under evaluation.")
    double_coding_status: str = Field(..., description="One of DOUBLE_CODING_STATUSES.")
    adjudication_status: str = Field(..., description="One of ADJUDICATION_STATUSES.")

    ambiguity_flag: bool = Field(False, description="True if this case is ambiguous/unclassifiable -- see exclusion_reason. Never force a false gold_code to avoid setting this.")
    exclusion_reason: Optional[str] = Field(None, description="Required (non-null) when ambiguity_flag is true, or when this record is split='excluded'. Explains why, never fabricated.")

    split: str = Field(..., description="One of SPLITS ('dev', 'heldout', 'excluded').")
    record_hash: Optional[str] = Field(None, description="sha256 of this record's canonical JSON, computed by the builder -- lets a reader verify a record was not altered after gold-label assignment.")
