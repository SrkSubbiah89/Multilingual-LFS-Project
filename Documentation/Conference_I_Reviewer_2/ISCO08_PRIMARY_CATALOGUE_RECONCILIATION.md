# ISCO-08 Primary Catalogue Reconciliation

Task 20: resolves the catalogue-identity blockers Task 19 left open by
importing the official ILO ISCO-08 structure directly from a primary,
machine-readable ILO source and running a full, reproducible,
code-by-code diff against `backend/rag/load_full_isco.py`'s hand-authored
`_MAJOR`/`_SUBMAJOR`/`_MINOR`/`_UNIT` lists. **No production hierarchy
data was changed, no Qdrant collection was touched, no evaluation was
run, and no WISCO file was read at any point in this task.**

```text
OFFICIAL_ISCO08_CATALOGUE_RECONCILED: yes
```

"Reconciled" here means: every one of the 33 previously-unidentified
code-level discrepancies Task 19 flagged is now individually identified
against a primary source, and a new `eval/verified_catalogue_counts.yaml`
exists for the first time in this project. It does **not** mean the
project's own catalogue now matches the standard — it still doesn't (see
§4) — and this task deliberately did not touch it (see §6 for the
un-taken next step, which needs separate approval).

## 1. Official sources used

| Source | URL | Role |
|---|---|---|
| Primary English structure workbook | `https://webapps.ilo.org/ilostat-files/ISCO/newdocs-08-2021/ISCO-08/ISCO-08%20EN%20Structure%20and%20definitions.xlsx` | Sole authority for all code/parent/title/count decisions in this task |
| Optional English-Arabic workbook | `https://webapps.ilo.org/ilostat-files/Documents/ISCO_08_AR_structure_V1.0.xlsx` | Downloaded and its schema documented (§2) for future reference only; **not parsed, imported, or used for any English code/hierarchy/count decision** |
| ILO ISCO-08 landing page | `https://isco.ilo.org/en/isco-08/` | Confirmed, by direct fetch in this task, that it states: *"ISCO-08 is a four-level hierarchically structured classification that allows all jobs in the world to be classified into 436 unit groups"*, aggregated into *"130 minor groups, 43 sub-major groups and 10 major groups"* — an exact match to the expected counts this task required |

**No WISCO file, path, package, code, title, split, or output was read at
any stage of this task** — confirmed by design (the normalizer and
reconciliation script take only the ILO workbook / `load_full_isco.py`
as input; neither imports anything from `backend/evaluation/wisco/` or
`eval/local_benchmarks/`).

## 2. Retrieval provenance

| | English workbook | Arabic workbook |
|---|---|---|
| Retrieved (UTC) | 2026-08-08T18:14:27.67Z | 2026-08-08T18:14:39.21Z |
| File size | 303,771 bytes | 66,388 bytes |
| SHA-256 | `df10744491b4216ed3580e8263b6324502e774c16667a1d7ecdbee50e85505e7` | `3dd8cf35c11a81ec55554dec174f9b50cfafaa5b3f5ebb43c9462a06e5f03718` |
| Sheets | `ISCO-08 EN Struct and defin` (619 data rows), `Sheet1` (empty) | `Information`, `ISCO-08 Structure EN AR V.1.0` (634 data rows), `c` |

Both files were saved only under the git-ignored
`eval/local_catalogues/ilo_isco08_2021/` directory (confirmed covered by
the existing `eval/local_catalogues/` `.gitignore` entry) and are never
committed.

**English workbook schema** (the sheet this task actually parses):
single flat table, one row per hierarchy node at any level, columns
`Level` (`"1"`-`"4"`), `ISCO 08 Code`, `Title EN`, plus four free-text
columns this task ignores (`Definition`, `Tasks include`, `Included
occupations`, `Excluded occupations`, `Notes`). Rows are in depth-first
top-down order (each parent row precedes all of its children) — verified
programmatically by the normalizer, not assumed.

**Arabic workbook schema** (documented, not parsed): a wide-format
table with one column per hierarchy level (`Major Group`, `Sub Major
Group`, `Minor Group`, `Unit Group`), an English `Description` column,
and an Arabic translation column, produced by *"The Regional Working
Group on Labour Indicators in the Arab Region"* in collaboration with
*"the ILO Department of Statistics and the ILO Regional Office of the
Arab States (ROAS)"*, year 2021. This is recorded for future reference
only (e.g. a future task wanting official Arabic unit-group labels
instead of this project's own hand-translated ones) — it played no role
in any decision in this task.

## 3. Parser, importer, and validation

### 3.1 Normalizer

No existing tool in this repository accepts the ILO workbook's schema
directly — `eval/catalogue_importer.py` requires a flat
`level,code,parent_code,label` CSV, top-down ordered. A new, tested
normalizer was added:

```text
eval/normalize_ilo_isco08_catalogue.py
eval/test_normalize_ilo_isco08_catalogue.py   (16 tests, all synthetic fixtures)
```

It parses only the documented English structure sheet, derives
`parent_code` as each code's own prefix (e.g. a unit code's parent is
its first 3 characters), and fails closed (raises, writes nothing) on a
missing/ambiguous sheet, a missing column, a blank/non-numeric/wrong-
length code, a duplicate code, an out-of-order (orphan-parent) row, a
blank title, or an unexpected per-level count. No catalogue row is
hard-coded anywhere in the parser or its tests.

```bash
python eval/normalize_ilo_isco08_catalogue.py \
  --xlsx eval/local_catalogues/ilo_isco08_2021/ISCO-08_EN_Structure_and_definitions.xlsx \
  --out-csv eval/local_catalogues/ilo_isco08_2021/normalized/isco08_official_normalized.csv \
  --report-out eval/local_catalogues/ilo_isco08_2021/normalized/normalization_report.json
```

Result: **619 rows parsed cleanly. Counts by level: `{major: 10,
submajor: 43, minor: 130, unit: 436}` — an exact match to the expected
counts. Zero malformed codes, zero duplicates, zero orphan parents, zero
blank titles, zero count violations.** Normalized CSV SHA-256:
`29b7539e25752b9d5b869baaa67d93f395781a107bbe64d371c00f4adaadeea3`.

`eval/catalogue_importer.py` was **not** modified — the normalizer alone
was sufficient.

### 3.2 Existing importer validation

```bash
python eval/catalogue_importer.py --standard isco08 \
  --catalogue eval/local_catalogues/ilo_isco08_2021/normalized/isco08_official_normalized.csv \
  --dry-run
```

Result: `rows=619 ok=True`, counts by level `{major: 10, submajor: 43,
minor: 130, unit: 436}`, **zero issues** — an independent confirmation
using the project's own existing, previously-tested fail-closed
validator (`--dry-run` used so this run itself made no write; the
tracked `eval/verified_catalogue_counts.yaml` was composed separately,
see §3.3, with the richer metadata this task requires).

### 3.3 `eval/verified_catalogue_counts.yaml`

Because both validations passed cleanly, this file was created for the
first time in this project (it did not exist before this task, and
`eval/coverage_audit.py`'s `official_count_verified` field was `null`
for every ISCO-08 level until now). It contains metadata and the four
verified counts only — no raw catalogue rows, codes, or titles.
Highlights: both source URLs, both retrieval timestamps, both raw-
workbook SHA-256 values, the normalized-catalogue SHA-256, the
normalizer/importer script SHA-256 values, the sheet/column mapping,
the verified counts (10/43/130/436), the exact validation commands and
their clean results, a licence/use note (no explicit reuse licence text
was found on the landing page or in the workbook's document properties
— consistent with the existing characterization in
`STANDARDS_SOURCE_PROVENANCE.md`), and an explicit WISCO-independence
statement.

## 4. Code-by-code reconciliation results

Produced by a local, reproducible, non-tracked script
(`eval/local_catalogues/ilo_isco08_2021/reconcile_against_project.py`,
ignored — reproducible by anyone who re-downloads the same workbook)
that compares the normalized official catalogue against
`backend/rag/load_full_isco.py`'s `_MAJOR`/`_SUBMAJOR`/`_MINOR`/`_UNIT`
lists, imported read-only (never modified). Full diff (every affected
code, every title pair) is at
`eval/local_catalogues/ilo_isco08_2021/normalized/reconciliation_diff.json`
(git-ignored, not committed) — SHA-256 of the complete diff artifact:
`6bc1689f53eb08fb999a9a8e86acb541d15010bdb7bca7280584a94255ac6eb8`.

| Level | Official | Project | Shared | Project-only | Official-only | Parent mismatches | Title mismatches* |
|---|---|---|---|---|---|---|---|
| major | 10 | 10 | 10 | 0 | 0 | 0 | 0 |
| submajor | 43 | 43 | 43 | 0 | 0 | 0 | 4 |
| minor | 130 | **131** | 130 | **1** | 0 | 0 | 10 |
| unit | 436 | **441** | 422 | **19** | **14** | 0 | **84** |

*After whitespace/case normalization; exact original strings preserved
in the full local diff. **Zero parent-code mismatches at every level**
— every shared code's derived parent (first N-1 characters) agrees
between the official structure and the project's own `code[:-1]`
derivation.

**Code ordering**: `major` and `submajor` have identical code sets but
different row order (the two SHA-256 hashes over each level's sorted
code set are identical, confirming this is a pure ordering difference,
not an identity difference); `minor` and `unit` differ in both identity
and, trivially, order (since the sets themselves differ).

### 4.1 The 131-vs-130 minor-group gap — now fully explained

**Exactly one project-only minor-group code**: `913` ("Building and
Related Caretakers"). This code does not exist anywhere in the official
ISCO-08 minor-group list. Zero official-only minor codes (nothing is
missing at this level). This closes the previously **undocumented**
discrepancy Task 19 found — its exact cause (this single fabricated
code) is now known.

### 4.2 The 441-vs-436 unit-group gap — now fully explained, code-for-code

**19 project-only unit codes** (exist in `_UNIT`, not in the official
standard) and **14 official-only unit codes** (exist in the official
standard, missing from `_UNIT`) — an **exact match, code-for-code, to
the list `Documentation/Phase_2/Week_1/module_a_week1_report.md` §5.1/
§5.2 already published** from an independent WISCO-based comparison.
This is an important cross-validation: the primary ILO source, consulted
directly for the first time in this project, confirms every single one
of the codes that earlier WISCO-based comparison identified, with no
additions and no removals.

19 project-only unit codes: `1347, 6124, 6141, 6142, 6150, 6161, 6162,
6163, 6164, 7116, 9131, 9132, 9141, 9151, 9152, 9153, 9161, 9162, 9420`

14 official-only unit codes: `6210, 6221, 6222, 6223, 6224, 6310, 6320,
6330, 6340, 7119, 9622, 9623, 9624, 9629`

**The one previously-verified root cause is now confirmed by primary-
source titles, not just code presence**: `6161`/`6162`/`6163`/`6164`'s
project titles ("Subsistence Crop Farmers" / "Subsistence Livestock
Farmers" / "Subsistence Mixed Crop and Livestock Farmers" / "Subsistence
Fishers, Hunters, Trappers and Gatherers") are **byte-identical** to the
official titles at `6310`/`6320`/`6330`/`6340` respectively — this
project's catalogue has this cluster under the wrong sub-major group
(`61`, not the official `63`), exactly as previously suspected, now
proven directly against the ILO source rather than inferred via WISCO.

### 4.3 A new finding: 84 unit-group title mismatches among the 422 shared codes

Not previously discovered — no prior task compared *titles* code-by-code
against a primary source (the Week 1 WISCO comparison only checked code
*presence*). Roughly 20% of the 422 codes present in both catalogues
have a title that differs after whitespace/case normalization. Most are
clearly cosmetic (British/American spelling: "Specialised" vs
"Specialized"; abbreviation expansion: "exc" vs "excluding", "NEC" vs
"Not Elsewhere Classified"; punctuation/hyphenation). A representative
sample:

| Code | Official title | Project title | Character |
|---|---|---|---|
| 13 | Production and Specialized Services Managers | Production and Specialised Services Managers | cosmetic (spelling) |
| 71 | Building and Related Trades Workers (excluding Electricians) | Building and Related Trades Workers (exc Electricians) | cosmetic (abbreviation) |
| 1219 | Business Services and Administration Managers Not Elsewhere Classified | Business Services and Administration Managers, NEC | cosmetic (abbreviation) |
| 2522 | Systems Aministrators *(sic, official typo)* | Systems Administrators | cosmetic (official source typo) |

A smaller subset appears to reflect **genuine content misalignment**,
not spelling — flagged here for priority human review, not resolved by
this task:

| Code | Official title | Project title |
|---|---|---|
| 5221 | Shopkeepers | Shop Salespersons |
| 5223 | Shop Sales Assistants | Cashiers and Ticket Clerks |
| 5230 | Cashiers and Ticket Clerks | Fuel Station Attendants |
| 9510 | Street and Related Service Workers | Refuse Sorters |
| 9520 | Street Vendors (excluding Food) | Odd Job Workers |
| 9611 | Garbage and Recycling Collectors | Water and Firewood Collectors |
| 9612 | Refuse Sorters | Odd Job Workers |
| 9613 | Sweepers and Related Labourers | Scrap Collectors and Recyclers |
| 9621 | Messengers, Package Deliverers and Luggage Porters | Subsistence Agricultural, Forestry, Fishing and Hunting Labourers |

These two code ranges (522x-523x, 951x-962x) show titles that appear
systematically offset from their neighbors, as if shifted by one or more
positions within the range, rather than randomly wrong — worth a
dedicated, focused review in the future correction task (§6), not
diagnosed further here. Verified directly against `backend/rag/
load_full_isco.py` source lines (e.g. lines 549-552) to confirm this is
a real property of the source file, not a reconciliation-script defect.

The complete list of all 84 (plus the 4 submajor and 10 minor
mismatches) is in the local, git-ignored `reconciliation_diff.json` —
not reproduced in full here.

## 5. What this does and does not establish

**Established**: a primary, source-hashed, reproducibly-verified
official ISCO-08 catalogue now exists (`eval/verified_catalogue_counts.yaml`);
every one of Task 19's 33 unidentified code-level discrepancies is now
individually identified; a new, larger title-mismatch issue was
discovered and documented; zero WISCO input was used anywhere in this
process.

**Not established, and not attempted**: the project's `_MAJOR`/
`_SUBMAJOR`/`_MINOR`/`_UNIT` lists were not changed. No Qdrant collection
was built, queried, or mutated. No evaluation was run. **The existing
Task 17 hierarchical WISCO result remains what it always was — a raw,
integrity-checked, non-accuracy artifact — and is not a citable ISCO-08
accuracy result now or after this task.** Any future accuracy claim
still requires the staged plan in §6, each step separately approved.

## 6. Staged future plan (not started)

1. Human review of the full mismatch list (§4, plus the complete local
   diff) and an explicit, approved decision on final accepted catalogue
   scope — including the two flagged content-misalignment ranges (§4.3).
2. A dedicated source-data correction task for `backend/rag/
   load_full_isco.py`'s `_MINOR`/`_UNIT` tables, with hermetic tests,
   scoped only to applying the reviewed corrections from step 1.
3. A separately approved local Qdrant rebuild (`isco08_minor_groups`,
   `isco08_unit_groups`) from the corrected list.
4. A new strict full WISCO run for both a future unit-group-only flat
   comparator (per `FLAT_BASELINE_COVERAGE_AUDIT.md` §4's now-partially-
   unblocked specification) and the hierarchical system.
5. A new fail-closed analysis task, structured like Task 18's
   `eval/analyze_wisco_tier1.py`.

None of these five steps were taken in this task.

## 7. Local artifact paths (git-ignored, not committed)

```text
eval/local_catalogues/ilo_isco08_2021/ISCO-08_EN_Structure_and_definitions.xlsx
eval/local_catalogues/ilo_isco08_2021/ISCO_08_AR_structure_V1.0.xlsx
eval/local_catalogues/ilo_isco08_2021/normalized/isco08_official_normalized.csv
eval/local_catalogues/ilo_isco08_2021/normalized/normalization_report.json
eval/local_catalogues/ilo_isco08_2021/normalized/catalogue_importer_output.txt
eval/local_catalogues/ilo_isco08_2021/normalized/reconciliation_diff.json
eval/local_catalogues/ilo_isco08_2021/reconcile_against_project.py
```
