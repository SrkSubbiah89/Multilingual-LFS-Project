# Controlled Benchmark Dataset Card — Template

Conference I Reviewer #2 response, Step 6. Governs a **controlled
benchmark**: reference/example data used to measure classifier accuracy
against known-correct codes. A controlled benchmark is **never** Labour
Force Survey respondent data — see
`Documentation/Conference_I_Reviewer_2/CONTROLLED_BENCHMARK_AUDIT.md` for
the audit that motivated this template and
`REAL_LFS_VALIDATION_DATASET_CARD_TEMPLATE.md` for the separate template
that governs actual respondent-data intake (`approved_real_lfs_validation`
— a controlled benchmark must never use that label; see
`eval/validate_controlled_benchmark.py`, which rejects it unconditionally).

Copy this file, fill in every field marked **REQUIRED** (or write "N/A —
[reason]" if it genuinely does not apply), and keep the per-record schema
in `eval/controlled_benchmark_schema.py`'s `BenchmarkRecord` in sync with
whatever you declare here. See
`eval/fixtures/controlled_benchmark_synthetic_example/synthetic_benchmark_records.json`
for a small, schema-complete SYNTHETIC example (test fixture only), and
`eval/local_benchmarks/wisco_isco08_v1/` (Git-ignored; regenerate with
`python eval/build_wisco_isco_benchmark.py --out-root eval/local_benchmarks/wisco_isco08_v1`)
for this repository's actual first controlled-benchmark package.

---

## 1. Benchmark identity

- **Benchmark ID** (REQUIRED): a stable identifier for this package, e.g.
  `WISCO-ISCO08-BENCHMARK-v1`.
- **Task(s) covered** (REQUIRED): one or more of `isco08` / `isic_rev4` /
  `isced2011` / `iscedf2013` (`eval/controlled_benchmark_schema.BENCHMARK_TASKS`).
- **Data source type** (REQUIRED): one of `external_authoritative_publisher`
  / `authored_by_project_team` / `crowd_or_survey_panel` / `unknown`
  (`DATA_SOURCE_TYPES`). `unknown` always fails
  `eval/validate_controlled_benchmark.py`.
- **Dataset label** (REQUIRED, fixed): `synthetic_or_operationally_realistic`
  — a controlled benchmark can never be `approved_real_lfs_validation`,
  regardless of label quality.
- **Source citation** (REQUIRED if `data_source_type` is
  `external_authoritative_publisher`): full citation, licence, and version
  identifier (DOI, release tag, etc.) — see
  `Documentation/Phase_2/Week_1/PROVENANCE.md` for the WISCO example of
  what this should look like (Zenodo version DOI, publisher, licence,
  MD5-verified download integrity).
- **Classification standard(s) and version(s)** (REQUIRED): e.g. "ISCO-08
  (2008)"; must match `BenchmarkRecord.classification_version` exactly for
  every record claiming that standard.

## 2. Language coverage

- **Languages present** (REQUIRED): list every `BenchmarkRecord.language`
  value actually used, with a per-language record count.
- **Language/code-switching annotation note** (REQUIRED if applicable): how
  mixed-language or code-switched input was handled — see
  `ANNOTATION_AND_ADJUDICATION_GUIDE.md` §5 (Language and code-switching
  annotation).

## 3. Label provenance (per `eval/validate_controlled_benchmark.py`)

- **Label source type** (REQUIRED): one of `external_publisher_gold` /
  `human_single_coder` / `human_double_coded_adjudicated` /
  `system_self_generated` / `unknown` (`LABEL_SOURCE_TYPES`).
  `system_self_generated` is **always rejected** — the system under
  evaluation can never be its own gold-label source.
- **Labeler identifier or role** (REQUIRED, non-personal): e.g.
  `external_publisher:wisco_surveycodings_wageindicator_foundation` or
  `coder_role:senior_annotator` — never a real name.
- **Independent-label status** (REQUIRED): one of
  `independent_of_system_under_test` / `not_independent` / `unknown`
  (`INDEPENDENT_LABEL_STATUSES`). `not_independent` is always rejected.
- **Double-coding status** (REQUIRED): one of `double_coded` /
  `single_coded` / `not_applicable_external_source` / `unknown`. Report
  `unknown` honestly rather than assuming double-coding happened just
  because the source is authoritative — see the WISCO audit finding in
  `CONTROLLED_BENCHMARK_AUDIT.md` Phase B, item 6.
- **Adjudication status** (REQUIRED): one of `resolved_by_adjudicator` /
  `not_needed_no_disagreement` / `not_applicable` / `unknown`.
- **Ambiguous/unclassifiable case policy** (REQUIRED): confirm the package
  uses `ambiguity_flag=true` + a populated `exclusion_reason` rather than
  forcing a false gold code for any case that cannot be confidently
  classified — see `ANNOTATION_AND_ADJUDICATION_GUIDE.md` §5.

## 4. Split discipline

- **Dev split size / purpose** (REQUIRED): count and confirmation it is
  used for parameter selection only, never cited as a confirmed result.
- **Heldout split size / frozen status** (REQUIRED): count and confirmation
  it is frozen (never used to select parameters).
- **Split assignment method** (REQUIRED): the exact deterministic rule used
  (e.g. "sha256 hash of the source occupation key, mod 10") — must be
  reproducible from the method description alone, and must avoid splitting
  a single underlying source item (e.g. one WISCO occupation) across both
  dev and heldout under different language variants.
- **Split manifest hash** (REQUIRED): the `split_hash` from the package's
  `split_manifest.json` (see `eval/split_manifest_schema.py`).

## 5. Hashes and reproducibility

- **Dataset hash** (REQUIRED): sha256 over the full records payload (see
  `build_summary.json`'s `dataset_hash` field for the WISCO package).
- **Source file hash** (REQUIRED if derived from an external file): sha256
  of the raw/parsed source file the benchmark was built from.
- **Build command** (REQUIRED): the exact command used to regenerate this
  package from source, e.g.:
  ```bash
  python eval/build_wisco_isco_benchmark.py --out-root eval/local_benchmarks/wisco_isco08_v1
  ```

## 6. Storage and retention

- **Storage location** (REQUIRED): confirm whether the package output is
  Git-ignored (`eval/local_benchmarks/` — the default for anything derived
  from a source larger than a small fixture) or, for a small, fully
  synthetic example only, committed under `eval/fixtures/`.
- **Retention plan** (REQUIRED): how to regenerate the package if the local
  copy is lost (must be possible from tracked source + script alone, per
  requirement 1 of Step 6 Phase D — raw/derived data outside Git unless
  explicitly synthetic).

## 7. Validation

Run before any package is used for Step 7:

```bash
python -c "
import json, sys
sys.path.insert(0, 'eval')
from controlled_benchmark_schema import BenchmarkRecord
from validate_controlled_benchmark import validate_benchmark_package
data = json.loads(open('eval/local_benchmarks/<package>/records.json', encoding='utf-8').read())
records = [BenchmarkRecord(**r) for r in data['records']]
dataset_hash = open('eval/local_benchmarks/<package>/dataset_hash.txt').read().strip()
report = validate_benchmark_package(records, dataset_hash=dataset_hash)
print('ok=', report.ok, 'errors=', report.errors, 'warnings=', report.warnings)
"
```

Record the exact output (ok/errors/warnings, never the record content
itself) here or in the package's own `build_summary.json`.
