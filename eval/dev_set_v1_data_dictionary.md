# `eval/dev_set_v1.csv` — data dictionary

`eval/dev_set_v1.csv` is the **B2 development set**, used exclusively to
select the candidate-pool size K ∈ {5, 8, 10, 15, 20} for the B2
candidate-capacity experiment. See `eval/PRE_RUN_B2_CHECKLIST.md` and
`eval/dev_sweep.py`'s module docstring for how K selection uses this file.

**This is the data dictionary. The canonical schema definition lives in
`eval/dev_set_schema.md`** (including the full migration note for prior,
now-superseded column names) — this document restates it for readers who
land here first, and must stay in sync with it. If the two ever disagree,
`eval/dev_set_schema.md` is authoritative.

As of this writing `eval/dev_set_v1.csv` is an empty, header-only intake
template — see `eval/dev_set_v1_provenance.md` and
`eval/dev_set_v1_readiness_report.md` for current population status.

## Purpose and scope limitation

This dataset selects a hyperparameter (K). It is explicitly **not** a
held-out evaluation set and **must never be reported as, or substituted
for, final confirmation evidence**. The only file that plays that role is
`eval/test_set_full130.csv`, run through `eval/run_eval.py` **exactly
once**, after K has been selected from this dev set and frozen. Any
accuracy/recall number computed from `dev_set_v1.csv` describes "how K
performs on the dev set," not "how the system performs" — those are
different claims and must not be conflated in a paper draft or report.

## Columns (7, canonical — see `eval/dev_set_schema.md` for the authoritative version)

| Column | Type | Required | Description |
|---|---|---|---|
| `case_id` | string | yes | Unique identifier for this dev-set case. Must be distinct from every `case_id` in `eval/test_set_smoke20.csv` and `eval/configs/full130_leakage_manifest.json`'s `case_ids` — prefix with a scheme that makes collisions structurally unlikely (e.g. `devv1_0001`). |
| `language` | `en` \| `ar` \| `mixed` | yes | Respondent's input language. Coverage target: at least 15 Arabic cases, remainder English. `mixed` (code-switched) only for genuinely code-switched text. |
| `respondent_text` | string | yes | **The exact text sent to the classifier** (`ISCOClassifier.classify(job_title=...)`, via `eval/dev_sweep.py`'s translation of this column into `run_eval.py`'s `input_text`) — stored directly, never assembled by concatenating other fields, and never normalised/translated/rewritten before use. The free-text occupation description a respondent gave, phrased as an actual survey answer — not a title copied from an ISCO-08 reference table. |
| `gold_isco_code` | string, 4 digits, in the classifier-supported ISCO catalogue | yes | The correct ISCO-08 **unit-group** code for this case, as determined by the process in `gold_label_source`. Must match `^[0-9]{4}$` **and** exist in `backend/rag/load_full_isco.py`'s `_UNIT` catalogue (441 codes this project's classifier can actually predict — **not** independently verified against the official ILO ISCO-08 standard, see below) — syntactically valid but nonexistent codes like `0000`/`9999` are rejected. See `eval/dev_set_schema.md`'s "Semantic ISCO-08 code validation" section, including the required follow-up audit note on the 441-vs-436 discrepancy. |
| `gold_label_source` | string | yes | How the gold code was determined. Recommended values: `human_coder_single` (one qualified coder), `human_coder_double_agreement` (two independent coders agreed), `adjudicated_panel` (coders disagreed, resolved by a named adjudication process), or `authoritative_source:<name>` (an existing labelled corpus with documented ISCO-08 coding methodology, `<name>` traceable to a real, citable source). Must not be blank. |
| `annotator_or_adjudication_reference` | string | yes | A reference to who/what produced the gold label — an annotator ID, coder initials, adjudication ticket/session identifier, or citation key. Identifies the **labeller**, never the respondent — see § Confidentiality. |
| `dataset_split` | string, constant | yes | Always the literal value `dev_v1` for every row in this file. |

**No `job_title`/`job_description` split, and no `major_group`/
`difficulty_level`/`notes` columns** — see `eval/dev_set_schema.md`'s
migration note for why these were consolidated/dropped during schema
reconciliation. `major_group` is now *derived* from `gold_isco_code[:1]` by
`eval/validate_dev_set.py` at validation time rather than stored redundantly.

## Validation rules (summary — enforced by `eval/validate_dev_set.py`)

- The CSV itself must be well-formed: no malformed/unterminated quoting, no ragged rows (checked before any field-level validation — `validate_csv_structure()`). Properly quoted multiline `respondent_text` cells are explicitly supported, not rejected.
- All 7 columns present with non-blank (or whitespace-only) values — leading/trailing whitespace is stripped before this check, so `"   "` counts as blank.
- `case_id` unique within the file (case-sensitive — `"Dev001"` ≠ `"dev001"`), and disjoint from `eval/test_set_smoke20.csv` and the full130 leakage manifest's `case_ids` (same case-sensitive comparison).
- `gold_isco_code` exactly 4 digits **and** a code in the classifier-supported ISCO catalogue (see above) — not just format-valid. `eval/validate_dev_set.py`'s CLI treats a missing/unparsable catalogue as FATAL (exit 1) rather than silently skipping this check.
- The CSV header itself (column names, order, no duplicates/extras) is validated independently of row count — see `eval/dev_set_schema.md`'s "Header validation" section.
- `language` ∈ {`en`, `ar`, `mixed`}.
- `dataset_split` == `dev_v1` for every row (hard requirement).
- `respondent_text` must not exact-duplicate (after `normalize_text()`) any `eval/test_set_smoke20.csv` `input_text`, nor hash-match any entry in the full130 leakage manifest — and the manifest's hashes are only trusted if their recorded `normalization_fingerprint` matches the live `normalize_text()` (see `eval/dev_set_schema.md`'s "Normalization-version / fingerprint enforcement").
- `respondent_text` must not duplicate another row's `respondent_text` within this file itself.

**A PASS on all of the above means "schema-valid," not "ready."** See
`eval/dev_set_schema.md`'s "Schema validity vs. readiness with real
labels" section — human/authoritative label provenance is a separate
question this validator cannot check.

## Labelling / adjudication process (required, not yet executed against real data)

Every gold code in this file must originate from a real person (or a cited
authoritative, already-published labelled corpus) assigning the code, with
the assignment traceable via `annotator_or_adjudication_reference`. No code
in this file may be:
- Generated, guessed, translated, or paraphrased by an LLM or automated tool.
- Copied from `eval/test_set_full130.csv` or `eval/test_set_smoke20.csv` for a similar-sounding title.
- Back-filled from the system under test's own predictions.

## Source and provenance standard

Every row must have an auditable source record (who supplied it, when, and
under what process) maintained separately from this CSV if it would expose
confidential respondent information — see `eval/dev_set_v1_provenance.md`
for the current provenance record and § Confidentiality below for what may
vs. may not appear in this public CSV.

## Confidentiality

`respondent_text` should be a free-text occupation description, not a full
survey transcript — avoid names, contact details, exact employer
identifiers, or other personally identifying information.
`annotator_or_adjudication_reference` identifies the labeller, never the
respondent. If a supplied record cannot be sufficiently de-identified,
exclude it and log the reason (see `eval/dev_set_v1_provenance.md`'s
exclusions section) rather than redacting it into an unusable fragment.

## Independence from `eval/test_set_smoke20.csv` and `eval/test_set_full130.csv`

- `eval/test_set_smoke20.csv` has already been used for earlier model/beam/routing experiments in this project, so it cannot be reused as a K-selection dev set.
- `eval/test_set_full130.csv` is the frozen confirmation set. Independence from it is checked via `eval/configs/full130_leakage_manifest.json` only — `eval/validate_dev_set.py`, `eval/dev_sweep.py`, and `eval/pre_run_check.py` never open `eval/test_set_full130.csv` directly. See `eval/dev_set_v1_readiness_report.md` for the actual check result.

## This is for K selection only

Repeating the scope limitation from the top of this document because it is
the single most important constraint on how these numbers may be used: a
result computed against `eval/dev_set_v1.csv` selects a hyperparameter. It
is not, and must never be reported as, evidence of final system accuracy —
that role belongs exclusively to the one, pre-specified confirmation run on
`eval/test_set_full130.csv`.
