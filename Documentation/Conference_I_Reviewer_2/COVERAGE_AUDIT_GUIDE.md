# Coverage Audit Guide

Covers Section C of the Reviewer #2 response and its official
classification-source/coverage-denominator follow-up, directly answering
reviewer comment 5: the manuscript abstract must not imply complete ISIC
(or ISCO/ISCED) coverage without disclosing the real, current limitation
— and must not claim a coverage *percentage* until the official
denominator is verified, not just sourced.

**Read `STANDARDS_SOURCE_PROVENANCE.md` alongside this guide** — that
document records exactly where every `official_count_unverified` number
below came from and how confident that provenance is.

## Two tiers of official count — the central concept

| | `official_count_unverified` | `official_count_verified` |
|---|---|---|
| Lives in | `eval/standards_reference.yaml` (human-maintained) | `eval/verified_catalogue_counts.yaml` (machine-generated) |
| Produced by | Transcribing a number from a cited `source_url` | `eval/catalogue_importer.py` validating a real catalogue file with **zero** issues |
| Used for `coverage_percentage`? | **Never** | **Only this** |

`eval/coverage_audit.py` reports both tiers side by side so a reader can
see the sourced-but-unverified figure for context, but it will **never**
compute a `coverage_percentage` from `official_count_unverified` — see
`_compute_coverage_percentage()` in that module, which doesn't even accept
an unverified count as a parameter (enforced by
`eval/test_coverage_audit.py::test_coverage_percentage_never_uses_unverified_count`).

## What `eval/coverage_audit.py` computes, per standard/level

- **standard name + version**
- **`official_count_unverified`** + **`official_count_unverified_note`** —
  from `eval/standards_reference.yaml`. `null` + a note if not sourced yet.
- **`official_count_verified`** + **`official_count_verified_note`** —
  from `eval/verified_catalogue_counts.yaml`. `null` + a note if no
  catalogue has been imported for that level.
- **`implemented_count`** — computed live from this codebase's actual
  embedded data tables (`backend/rag/load_full_isco.py`,
  `backend/agents/isic_classifier.py`'s `_ISIC_DATA`,
  `backend/agents/isced_classifier.py`'s `_ISCED_LEVELS`/`_ISCED_FIELDS`).
- **`coverage_percentage`** + **`coverage_percentage_status`** —
  `implemented_count / official_count_verified * 100`, rounded to 2
  decimals, **only** when `official_count_verified` is present and
  positive. `null` + a status string otherwise.
- **duplicate codes**, **malformed codes** (regex-checked against the
  expected code shape for that level)
- **missing codes** — always `[]` today. Computing this correctly
  requires the FULL official code list, not just a count. A catalogue
  import does see the full list while validating, but
  `eval/catalogue_importer.py` deliberately persists only per-level
  *counts* to `verified_catalogue_counts.yaml`, never the code list
  itself — see "Why counts only, not the code list" below.
- **source file + sha256** of the *implemented-code* source file audited
  (e.g. `isic_classifier.py`), for traceability.

## Running the audit

```bash
python eval/coverage_audit.py --out Documentation/Conference_I_Reviewer_2/generated/
```

Writes 3 JSON+CSV+Markdown report sets (one per standard group: `isco08`,
`isic_rev4`, `isced2011_and_iscedf2013`) to `generated/`. With no catalogue
ever imported (the default state of this repo), every `coverage_percentage`
in every report is `null` — this is expected and correct, not a bug.

## Getting a verified coverage percentage: supplying an official catalogue

**1. Obtain the official catalogue yourself.** This repository does not
ship one, and `eval/catalogue_importer.py` does not fetch one — see "Do
not commit the catalogue" below for why. Extract a code list from the
official source cited in `STANDARDS_SOURCE_PROVENANCE.md` (e.g. ISIC
Rev.4's UN Statistical Papers Series M No.4/Rev.4 publication) into a CSV.

**2. Format it as a 4-column CSV**: `level,code,parent_code,label`.

- `level` must be one of the standard's declared level names (see
  `eval/standards_reference.yaml`'s `hierarchy_levels` for that standard,
  e.g. `section`/`division`/`group`/`class` for `isic_rev4`).
- `code` must match that level's `code_pattern` regex.
- `parent_code` is the code's parent at the parent level — **blank** for
  top-level rows (e.g. ISIC sections), **required** for every other level.
- Rows must be ordered **top-down**: every row's parent must already have
  appeared as a row at the parent level earlier in the file.
- `label` is free text, not validated.

Example (a tiny, valid excerpt for `isic_rev4`):

```csv
level,code,parent_code,label
section,A,,Agriculture, Forestry and Fishing
division,01,A,Crop and animal production...
group,011,01,Growing of non-perennial crops
class,0111,011,Growing of cereals...
```

**3. Save it under `eval/local_catalogues/`** (gitignored — see "Do not
commit the catalogue" below) or anywhere outside the repo.

**4. Run the importer:**

```bash
python eval/catalogue_importer.py --standard isic_rev4 --catalogue eval/local_catalogues/isic_rev4_full.csv
```

- On a **clean** validation (zero issues), this writes verified counts +
  the catalogue file's path/sha256/timestamp to
  `eval/verified_catalogue_counts.yaml`. This file is safe to commit — it
  contains only counts and a hash, never any classification-standard
  content itself.
- On **any** issue (malformed code, duplicate, orphan parent, unknown
  level), nothing is written — fail-closed. The CLI prints every issue
  (row number, level, code, exact problem) and exits non-zero.
- Use `--dry-run` to validate without writing, e.g. while iterating on
  your extraction.

**5. Regenerate the audit** (step "Running the audit" above) — every
level with a matching verified count now shows a real
`coverage_percentage`.

## Validation the importer performs

1. **Code format** — code fully matches the level's `code_pattern`.
2. **Hierarchy level** — `level` is one of the standard's declared names.
3. **Uniqueness** — no duplicate code within a level.
4. **Parent-child consistency** — every non-top-level row's `parent_code`
   matches an already-seen code at the parent level; top-level rows have
   an empty `parent_code`.

See `eval/test_catalogue_importer.py` for a worked example of every
failure mode, using the small synthetic fixtures
`eval/fixtures/synthetic_catalogue_clean.csv` and
`synthetic_catalogue_with_issues.csv` (never real standard data).

## Do not commit the catalogue

Per the governing instructions for this work: **do not automatically
commit official source PDFs, restricted data, or copyrighted material into
Git.** `eval/local_catalogues/` is gitignored specifically for this reason
(see `.gitignore`). Only `eval/verified_catalogue_counts.yaml` — counts,
a file path, and a sha256 hash, never the underlying classification
content — is meant to be committed.

## Why counts only, not the code list

`eval/catalogue_importer.py` sees the full code list while validating a
catalogue, but intentionally persists only per-level *counts* (plus a hash
of the source file) to `verified_catalogue_counts.yaml`, not the codes
themselves. This keeps the one machine-generated file this workflow
commits free of any actual classification-standard content — which may be
copyrighted — while still producing a citable, verified denominator. A
side effect: `eval/coverage_audit.py`'s `missing_codes` field stays `[]`
even after a verified import, since computing it would need the actual
code list, not just a count.

## Filling in `official_count_unverified`

Edit `eval/standards_reference.yaml`. Only fill in a level's
`official_count_unverified` when you have an actual, checkable
`source_url` (record it at the standard level, plus a `retrieval_date` and
a specific `unverified_source_note` quoting or describing what the source
said). Leave it `null` with a note otherwise — **do not** fill in a number
from memory or general knowledge, even a widely-known one, without citing
where it came from. See `STANDARDS_SOURCE_PROVENANCE.md` for the exact
research trail behind every number currently in that file, including two
that were deliberately left `null` (ISIC Rev.4's counts are corroborated
via secondary sources only, not primary-document-confirmed; ISCED-F 2013's
detailed-field count is only ever cited as "about 80" in available
sources, too imprecise to record as exact).

## What this means for the abstract, right now

**No coverage percentage is supportable for any of the four standards
today** — no catalogue has ever been imported in this repository, so
`eval/verified_catalogue_counts.yaml` does not exist and every
`coverage_percentage` in every generated report is `null`. See the
`generated/coverage_audit_isic_rev4_*.md` report for ISIC's real
`implemented_count` per level (as of the last run in this session: 21
sections, 68 divisions, 118 groups, 134 classes implemented in
`ISICClassifier._ISIC_DATA` — real, computed numbers) alongside its
sourced-but-**unverified** official counts (21/88/238/419, see
`STANDARDS_SOURCE_PROVENANCE.md` for why these are corroborated, not
primary-confirmed). Recommended framing once a real catalogue has been
imported and a verified percentage exists:

> "The system implements ISIC Rev.4 classification for N of M
> (X%) four-digit classes, covering the sections most relevant to Labour
> Force Survey occupations; full ISIC Rev.4 coverage is not yet
> implemented."

Fill in N/M/X% **only** from a freshly regenerated
`coverage_audit_isic_rev4_*.md` report whose `coverage_percentage` column
is non-null (i.e. after running `eval/catalogue_importer.py` successfully)
— never from `official_count_unverified`, and never estimated.

## Tests

- `eval/test_coverage_audit.py` — duplicate/malformed detection on small
  fixture catalogues, `_level_info()`'s honest-default behaviour,
  `_compute_coverage_percentage()`'s verified-only contract, and
  integration tests against the real embedded tables + real
  `standards_reference.yaml` confirming `coverage_percentage` is `null`
  by default (never asserting a specific implemented count as ground
  truth, since discovering those is exactly what this tool is for).
- `eval/test_catalogue_importer.py` — every validation rule exercised in
  isolation plus via the two synthetic fixture CSVs, fail-closed behaviour
  (a catalogue with issues writes nothing), and the CLI's exit codes.
