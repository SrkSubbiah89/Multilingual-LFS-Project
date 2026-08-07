# `eval/dev_set_v1.csv` — provenance record

Schema reference: `eval/dev_set_schema.md` (canonical, v1 — 7 columns:
`case_id`, `language`, `respondent_text`, `gold_isco_code`,
`gold_label_source`, `annotator_or_adjudication_reference`,
`dataset_split`). This document assumes that schema throughout; see
`eval/dev_set_schema.md`'s migration note if you're looking for an older
column name (`coder_or_adjudicator`, `major_group`, `difficulty_level`,
`notes`, or a `job_title`/`job_description` split).

## Status: NOT YET POPULATED

As of this writing, `eval/dev_set_v1.csv` contains **zero data rows**
(header only). No job-description records or gold ISCO-08 codes have been
supplied by a human expert or an approved external labelled source. This
document records that fact honestly rather than describing a collection
process that has not happened — per this task's explicit instruction not
to claim independent labelling unless evidence is actually available.

Everything below is split into **(a) what is true right now** and **(b)
what the required process will look like once real records are supplied**
— these are clearly separated so this document cannot be misread as a
retroactive description of work that occurred.

## (a) Current state

| Field | Value |
|---|---|
| Collection date | Not applicable — no collection has occurred |
| Source type | Not applicable — no source has been engaged |
| Number of cases | 0 |
| English-language cases | 0 |
| Arabic-language cases | 0 |
| ISCO-08 major-group distribution | Not applicable — no data |
| Annotator/adjudication procedure executed | None — no cases have been labelled |
| Exclusions logged | None — no candidate records have been received or reviewed |

## (b) Required process once records are supplied

This section documents the REQUIRED procedure for populating this file —
it is a specification for future intake, not a report of intake already
performed.

### Source requirements

Records must come from one of:
1. **Human expert coding** — a qualified coder (or two, for
   `human_coder_double_agreement`) assigns each `gold_isco_code` directly
   from a respondent's job title/description, following standard ISCO-08
   coding index practice.
2. **Adjudicated disagreement** — where two coders disagree, a named
   adjudicator resolves the code, and the adjudication session/ticket is
   recorded in `annotator_or_adjudication_reference`.
3. **An approved external authoritative source** — an already-published,
   independently labelled occupational-coding corpus with documented
   ISCO-08 coding methodology, cited by name in `gold_label_source` as
   `authoritative_source:<name>` and traceable to a real, checkable
   reference.

No record may be added to this file by generating, translating,
paraphrasing, or guessing either the job text or the gold code — including
by an LLM — and no record may be back-filled from this project's own
classifier's predictions.

### What must be logged for every ingested case

- Collection date (the date the record was received/finalised for this
  dataset, not necessarily the date the underlying respondent interaction
  occurred).
- Source type (which of the three categories above).
- The `annotator_or_adjudication_reference` value and what it resolves to
  (kept in an internal, non-public record if it would expose confidential
  respondent or annotator information — this public provenance document
  should describe the *process*, not necessarily list every reference
  value verbatim if doing so would be identifying).

### Exclusions

Any candidate record that is questionable — ambiguous gold code, unclear
provenance, possible overlap with `eval/test_set_smoke20.csv` or
`eval/test_set_full130.csv`, insufficiently de-identified respondent
content — must be **excluded from `eval/dev_set_v1.csv` and logged here**
with a reason, rather than forced into the dataset to hit the case-count
target. This section will be updated with the actual exclusion log once
candidate records exist and are reviewed. Currently: no candidates have
been received, so no exclusions have occurred.

### Target coverage (not yet met — see `eval/dev_set_v1_readiness_report.md`)

- 50+ total cases (30 absolute minimum if genuinely data-constrained).
- 15+ Arabic-language cases, remainder English.
- Representation across multiple ISCO-08 major groups (not concentrated in
  one occupation type).

## Next step

This file cannot progress past "not yet populated" without human-labelled
or authoritative-source records being supplied to this project. See
`eval/dev_set_v1_readiness_report.md` for the specific blocking status and
`eval/dev_set_v1_data_dictionary.md` for the exact schema those records
must conform to.
