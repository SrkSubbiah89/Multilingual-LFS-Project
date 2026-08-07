# Annotation and Adjudication Guide

Governs how gold labels for ISCO-08 (occupation), ISIC Rev.4 (industry),
ISCED 2011 (education level), and ISCED-F 2013 (education field) must be
produced for any dataset intended to be labelled `approved_real_lfs_validation`
(Conference I Reviewer #2 response, Step 3 real-LFS-intake hardening pass).
This guide is process documentation for human coders and adjudicators — it
does not itself contain, and must never be used to record, any real
respondent data. Use
`eval/fixtures/synthetic_lfs_intake_package/annotation_adjudication_record_template.json`
as the record-keeping template.

**Step 6 update**: Sections 1-7 below apply equally to labelling a
**controlled benchmark** (`eval/controlled_benchmark_schema.py`'s
`BenchmarkRecord` — reference/example data, never Labour Force Survey
respondent data; see
`Documentation/Conference_I_Reviewer_2/CONTROLLED_BENCHMARK_AUDIT.md` and
`CONTROLLED_BENCHMARK_DATASET_CARD_TEMPLATE.md`). The only differences: a
controlled benchmark uses `BenchmarkRecord`'s fields
(`label_source_type`/`independent_label_status`/`double_coding_status`/
`adjudication_status`/`ambiguity_flag`/`exclusion_reason`) instead of the
`DatasetCard`/annotation-record fields named throughout, and is validated
by `eval/validate_controlled_benchmark.py` instead of
`eval/validate_real_lfs_governance.py`. Section 10 (new, below) covers the
benchmark-specific workflow items in full.

## 1. Annotation instructions per task

### Occupation (ISCO-08)

- Read the respondent's free-text job-title (and, where available, job-
  duties) response in full before assigning a code.
- Assign the most specific 4-digit ISCO-08 unit-group code the response
  supports. If the response only supports a broader level (e.g. only a
  3-digit minor group is clearly indicated), record the coder's best
  4-digit judgement AND flag `ambiguous_or_unknown_flag=true` with a note
  explaining the uncertainty — never silently round up to a parent code
  without flagging it.
- Do not infer occupation from industry or education alone; the job-title/
  duties text is the primary evidence. Industry/education context may be
  used to disambiguate between close candidates, but must be noted in
  `coder_1_confidence_or_note`.

### Industry (ISIC Rev.4)

- Code from the respondent's description of their employer's main economic
  activity, not from the occupation title alone (e.g. an accountant working
  for a hospital is industry Q "Human health and social work activities",
  not industry M "Professional, scientific and technical activities").
- Assign the most specific 4-digit ISIC Rev.4 class the response supports,
  following the same specificity/flagging discipline as occupation coding.

### Education level (ISCED 2011)

- Code the respondent's HIGHEST COMPLETED level, not current enrolment
  (e.g. someone currently enrolled in but not having completed a Master's
  is coded at their highest COMPLETED level, typically Bachelor's/ISCED 6),
  unless the survey instrument specifically asks about current enrolment —
  note which convention applies in the dataset card's
  `missing_data_policy`/`exclusion_criteria` fields if it differs from this
  default.
- Use the 0–8 ISCED 2011 level scale exactly as defined in
  `backend/agents/isced_classifier.py`'s `_ISCED_LEVELS` table and the
  official UNESCO ISCED 2011 Operational Manual (see
  `STANDARDS_SOURCE_PROVENANCE.md`).

### Education field (ISCED-F 2013)

- Code the field of the respondent's HIGHEST COMPLETED qualification,
  using the same specificity/flagging discipline as the other tasks:
  broad (2-digit) → narrow (3-digit) → detailed (4-digit).
- If a respondent's field doesn't map cleanly to a single ISCED-F category
  (e.g. a genuinely interdisciplinary qualification), code the closest
  detailed field AND flag `ambiguous_or_unknown_flag=true`.

## 2. Coder qualifications and training requirements

A coder is qualified to produce gold labels for `approved_real_lfs_validation`
only if they meet ALL of:

1. Familiarity with the exact standard/version being coded (ISCO-08, ISIC
   Rev.4, ISCED 2011, ISCED-F 2013) — completion of a structured training
   session covering the standard's structure and worked examples.
2. Fluency (native or professional working proficiency) in at least one of
   the dataset's declared languages, with access to translation support for
   the others.
3. Completion of a calibration exercise (a small set of test cases with
   known/agreed answers) before coding real records, with results reviewed
   by a senior coder or the adjudicator.

Record each coder's qualifications in the dataset card's
`coder_qualifications` and `labeler_type` fields — free text, but specific
enough that a reader can judge whether the qualification bar above was met
(e.g. "postgraduate statistics degree; completed 2-day ISCO-08/ISIC Rev.4
coding training; calibration exercise: 18/20 agreement with reference set"
— not just "trained coder").

## 3. Independent double-coding procedure

**Every record intended for `approved_real_lfs_validation` must be coded
independently by two different coders before adjudication.** This is a hard
requirement (`DatasetCard.double_coded` must be `true`) — single-coded
datasets are not accepted evidence for approved real LFS validation.

1. Coder 1 and Coder 2 code the same record independently, without seeing
   each other's labels, and without seeing the system's own prediction
   (see Section 6).
2. Record both coders' labels using the fields `coder_1_id`/`coder_1_label`
   and `coder_2_id`/`coder_2_label` in the annotation record.
3. Compute agreement (`agreement = coder_1_label == coder_2_label`) per
   record, then in aggregate (Cohen's kappa or equivalent) across the
   dataset — record the aggregate figure in the dataset card's
   `inter_annotator_agreement` field once computed (leave `null` until it
   actually is).

## 4. Disagreement and adjudication workflow

When `agreement=false`:

1. A third coder (the adjudicator — must not be either of the original two
   coders for that record) reviews both labels and the original response
   text.
2. The adjudicator either confirms one of the two original labels or
   assigns a different one, and records `adjudicated_final_label` plus a
   free-text `adjudication_rationale` explaining the decision.
3. The adjudicator's label becomes the gold label for that record. The two
   original (disagreeing) labels are retained in the record for
   transparency — never discarded.
4. Record the overall adjudication process (who adjudicates, how often,
   escalation path for persistent disagreement) in the dataset card's
   `adjudication_process` field.

## 5. Handling unknown, ambiguous, multilingual, or incomplete responses

- **Unknown/refused**: if a respondent's answer is "don't know", refused,
  or otherwise non-substantive, do not force a code — record it as
  excluded per the dataset card's `exclusion_criteria`/`missing_data_policy`
  fields, never as a guessed code.
- **Ambiguous**: set `ambiguous_or_unknown_flag=true` and record the
  resolution in `ambiguous_or_unknown_resolution` (e.g. "coded to the
  closest unit group per ISCO-08 index rules, section 3.2").
- **Multilingual/code-switched responses**: code from the full response
  regardless of which language(s) it's in; if translation was needed,
  note the translation method in `ambiguous_or_unknown_resolution`. Do not
  discard code-switched responses by default — only exclude per the
  documented exclusion criteria.
- **Incomplete responses**: apply the dataset card's `missing_data_policy`
  consistently across all coders — document the policy before coding
  begins, not case-by-case.

## 6. Separation between system prediction and reference label

**Coders must never see this system's own classifier prediction while
producing a gold label.** The `system_prediction_shown_to_coder` field in
the annotation record must be `false` for every record used as a reference/
gold label for evaluation. Showing coders the system's prediction before
they label independently would contaminate the evaluation (the coder could
anchor on the system's answer rather than judging independently), making
any resulting accuracy figure unpublishable as an independent validation
result. If a workflow ever needs coders to review system predictions (e.g.
for error analysis, not gold-label production), keep those records clearly
separate and never mix them into the evaluation split.

## 7. Privacy and access-control requirements

- Coders access only the fields needed for their task (free-text job title/
  industry/education responses, non-identifying case IDs) — never direct
  identifiers, which must already be removed per the dataset card's
  `deidentification_method`/`direct_identifiers_removed_confirmed` fields
  before any coder sees the data.
- Coding must happen within the access-control boundary described in the
  dataset card's `access_control_description` field (e.g. a secured
  institutional environment) — never by exporting records to an
  uncontrolled location (personal laptops, cloud drives outside the
  agreement's scope, and never into this Git repository).
- Any incidental PII noticed by a coder during labelling (e.g. a name typed
  into a free-text response) must be reported through the custodian's
  redaction process (`free_text_redaction_method`), not silently ignored
  or, worse, copied into notes/tickets/annotation records.

## 8. Validating a completed submission

```bash
python eval/validate_real_lfs_governance.py \
  --manifest path/to/manifest.jsonl \
  --dataset-card path/to/your_dataset_card.json
```

See `REAL_LFS_VALIDATION_DATASET_CARD_TEMPLATE.md` for the full field list
and `REAL_LFS_DATA_INTAKE_CHECKLIST.md` for the end-to-end intake process
this annotation work feeds into.

## 9. Real data must remain outside Git

**No annotation record, coder note, adjudication log, or any file
containing real respondent text or real gold labels may ever be committed
to this repository.** Store them in the location described by the dataset
card's `secure_storage_location_description` field — a description of a
location OUTSIDE this Git repository. `eval/validate_real_lfs_governance.py`
rejects any `secure_storage_location_description` (or split-reference
field) that resolves to a path inside this repository, but that automated
check is a backstop, not a substitute for never attempting it in the first
place. `eval/local_catalogues/` and any other locally-created data
directories are gitignored for the same reason — see `.gitignore` and
`COVERAGE_AUDIT_GUIDE.md`'s "Do not commit the catalogue" section for the
parallel policy on classification-standard catalogue files.

## 10. Controlled-benchmark-specific workflow (Step 6)

Applies when producing or upgrading a `BenchmarkRecord` package (e.g.
converting an external source like WISCO, or manually authoring new
benchmark cases) rather than real-LFS respondent data. Sections 1-7 above
already cover the substance (per-task coding instructions, coder
qualifications, double-coding, adjudication, ambiguous/multilingual
handling, prediction separation, privacy); this section covers what's
specific to a benchmark package.

### 10.1 Measuring inter-annotator agreement (IAA)

When two or more independent labels exist for the same record (i.e.
`double_coding_status="double_coded"`), compute and record agreement at two
levels:

- **Raw percent agreement**: `n_agreeing_records / n_double_coded_records`.
- **Chance-corrected agreement (Cohen's kappa)**, when the label space is a
  closed, comparably-sized set across both coders — see
  `eval/analyze.py::wilson_score_interval()`/`mcnemar_test()` for this
  project's existing hand-rolled-statistics convention (no `scipy`
  dependency); a kappa implementation would follow the same style if/when a
  double-coded controlled benchmark is actually built. Record the computed
  figure in the dataset card's IAA field — never a placeholder, and never
  computed and then silently dropped if it turns out low. A low kappa is
  itself an important, reportable finding (it means the task is genuinely
  ambiguous or the coding guide needs revision), not something to hide.
- If only one coder's label exists (`double_coding_status="single_coded"`
  or `"unknown"`, as is honestly the case for the current WISCO-derived
  package — see `CONTROLLED_BENCHMARK_AUDIT.md` Phase B, item 6), IAA
  cannot be computed and the dataset card must say so explicitly
  (`inter_annotator_agreement: null` with a reason), never omit the field
  silently.

### 10.2 Excluding unclear or insufficiently detailed responses

Identical discipline to Section 5 above: set `BenchmarkRecord.
ambiguity_flag=true` and a non-blank `exclusion_reason`, and use
`split="excluded"` for cases that cannot support any gold code — never
force a code to avoid having an excluded record.
`eval/validate_controlled_benchmark.py` rejects any record where
`ambiguity_flag=true` but `exclusion_reason` is blank, and any non-excluded,
non-ambiguous record where `gold_code` is blank — so an incomplete
exclusion cannot silently pass validation.

### 10.3 Maintaining separation between benchmark labels and model predictions

Same rule as Section 6, restated for the benchmark schema:
`BenchmarkRecord.independent_label_status` must be
`"independent_of_system_under_test"` for any record used as evaluation
ground truth. `label_source_type="system_self_generated"` and
`independent_label_status="not_independent"` are both hard rejections in
`eval/validate_controlled_benchmark.py` — a classifier's own output (or an
LLM asked to "generate labels" for its own evaluation) can never become
that evaluation's gold standard. This is also why an LLM-generated label is
never accepted as "independent ground truth" (Step 6 restriction 2) even
when the LLM is a different model from the one under test — the schema has
no vocabulary entry that would let such a label pass as independent;
`unknown` is the honest value until real independent provenance exists.

### 10.4 Upgrading an existing non-independently-labelled test set

If a project wants to upgrade a dataset like `eval/test_set_full130.csv`
(audited in `CONTROLLED_BENCHMARK_AUDIT.md` as
`not_eligible_unknown_provenance`) to controlled-benchmark status: apply
Sections 1-4 above to have two independent coders label the existing
`input_text` values **blind to the file's current `gold_isco_4digit`
values** (never show a coder the existing label before they produce their
own — that would defeat the independence requirement), adjudicate any
disagreement, and only then populate a `BenchmarkRecord` package with
`label_source_type="human_double_coded_adjudicated"`. This is not done as
part of Step 6 (audit-and-prepare only) — see `CONTROLLED_BENCHMARK_AUDIT.md`
Phase C for the disposition decision.
