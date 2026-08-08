# Task 20 Final Report — Official ILO ISCO-08 Catalogue Reconciliation

Produced in response to
`Documentation/AI_HANDOFF/CLAUDE_TASK_20_OFFICIAL_ILO_ISCO08_CATALOGUE_RECONCILIATION.md`.
Official-catalogue acquisition, validation, and discrepancy documentation
only — no production hierarchy correction, no Qdrant operation, no
evaluation, no WISCO read.

```text
OFFICIAL_ISCO08_CATALOGUE_RECONCILED: yes
```

## 1. Source SHA, branch, final commit, push, clean-tree status

| | |
|---|---|
| Base branch | `reviewer2-flat-baseline-coverage-audit-20260808` |
| Required SHA | `c158897f691e9b94855f670503a70d02877d84d1` |
| Verified `origin` SHA | `c158897f691e9b94855f670503a70d02877d84d1` — match |
| New branch | `reviewer2-isco08-official-catalogue-reconciliation-20260808` |
| Final commit SHA | recorded after this report's commit (see push confirmation below) |

Working tree was clean before branching and remained clean throughout
except for the files listed in §7, all of which are either tracked
deliverables this task is permitted to add/modify or git-ignored local
artifacts.

## 2. Official source URLs, retrieval, raw-workbook hashes, schema, Arabic-workbook use

| | English structure workbook | Arabic structure workbook |
|---|---|---|
| URL | `https://webapps.ilo.org/ilostat-files/ISCO/newdocs-08-2021/ISCO-08/ISCO-08%20EN%20Structure%20and%20definitions.xlsx` | `https://webapps.ilo.org/ilostat-files/Documents/ISCO_08_AR_structure_V1.0.xlsx` |
| Retrieved (UTC) | 2026-08-08T18:14:27.67Z | 2026-08-08T18:14:39.21Z |
| File size | 303,771 bytes | 66,388 bytes |
| SHA-256 | `df10744491b4216ed3580e8263b6324502e774c16667a1d7ecdbee50e85505e7` | `3dd8cf35c11a81ec55554dec174f9b50cfafaa5b3f5ebb43c9462a06e5f03718` |
| Sheets | `ISCO-08 EN Struct and defin` (619 data rows, parsed), `Sheet1` (empty) | `Information`, `ISCO-08 Structure EN AR V.1.0` (634 data rows, documented only), `c` |

Also fetched: `https://isco.ilo.org/en/isco-08/` (landing page), directly
confirming *"436 unit groups ... aggregated into 130 minor groups, 43
sub-major groups and 10 major groups"* — an exact match to the required
figures, and confirming no explicit reuse/redistribution licence
statement exists beyond a bare copyright notice.

**Sheet/column mapping** (English workbook, the only one parsed): `Level`
(values `"1"`-`"4"`) → `major`/`submajor`/`minor`/`unit`; `ISCO 08 Code`
→ code (source-provided, already correctly zero-padded); `Title EN` →
label; four other columns (`Definition`, `Tasks include`, `Included
occupations`, `Excluded occupations`, `Notes`) ignored.

**Arabic workbook**: downloaded and its wide-format schema documented
(one column per level, English description, Arabic translation,
attributed to *"The Regional Working Group on Labour Indicators in the
Arab Region"* with the ILO Department of Statistics and ROAS, 2021) —
**not parsed, imported, or used for any English code/hierarchy/count
decision**, per the task's own restriction. Both raw workbooks were
saved only under the git-ignored `eval/local_catalogues/ilo_isco08_2021/`
directory and were never committed (confirmed: this path is line 43 of
`.gitignore`, already present before this task).

## 3. Parser/importer commands and validation result

No existing tool accepted the ILO workbook's schema directly
(`eval/catalogue_importer.py` requires a flat `level,code,parent_code,
label` CSV). A new normalizer was added and used:

```bash
python eval/normalize_ilo_isco08_catalogue.py \
  --xlsx eval/local_catalogues/ilo_isco08_2021/ISCO-08_EN_Structure_and_definitions.xlsx \
  --out-csv eval/local_catalogues/ilo_isco08_2021/normalized/isco08_official_normalized.csv \
  --report-out eval/local_catalogues/ilo_isco08_2021/normalized/normalization_report.json
```

Result: **619 rows parsed cleanly; counts by level `{major: 10,
submajor: 43, minor: 130, unit: 436}` — exact match to the expected
figures; zero malformed codes, zero duplicates, zero orphan parents,
zero blank titles, zero count violations.**

```bash
python eval/catalogue_importer.py --standard isco08 \
  --catalogue eval/local_catalogues/ilo_isco08_2021/normalized/isco08_official_normalized.csv \
  --dry-run
```

Result: `rows=619 ok=True`, counts `{major: 10, submajor: 43, minor: 130,
unit: 436}`, **zero issues** — confirmed by the project's own existing,
previously-tested fail-closed validator, independent of the new
normalizer's own checks. `eval/catalogue_importer.py` itself was **not**
modified — no bug was found that required it.

## 4. Official hierarchy counts and verified-catalogue metadata status

**Official counts, verified**: 10 major / 43 sub-major / 130 minor / 436
unit groups — matching both the ILO landing page's own statement and
the normalized workbook parse.

`eval/verified_catalogue_counts.yaml` was created for the first time in
this project (it did not exist on any prior branch). It contains
metadata and the four verified counts only — no raw catalogue rows.
Full contents: both source URLs, both retrieval timestamps, both raw-
workbook SHA-256 values and sizes, the normalized-catalogue SHA-256, the
normalizer/importer script SHA-256 values, the sheet/column mapping, the
verified counts, the exact validation commands and their clean results,
a licence/use note, and an explicit WISCO-independence statement.
`eval/coverage_audit.py`'s `official_count_verified` for ISCO-08 is no
longer `null` as a result (not re-run in this task — that is a separate,
out-of-scope operation).

## 5. Reconciliation results by level

Produced by a local, reproducible, non-tracked script
(`eval/local_catalogues/ilo_isco08_2021/reconcile_against_project.py`,
git-ignored) comparing the normalized official catalogue against
`backend/rag/load_full_isco.py`'s `_MAJOR`/`_SUBMAJOR`/`_MINOR`/`_UNIT`
lists, imported read-only.

| Level | Official | Project | Shared | Project-only | Official-only | Parent mismatches | Title mismatches |
|---|---|---|---|---|---|---|---|
| major | 10 | 10 | 10 | 0 | 0 | 0 | 0 |
| submajor | 43 | 43 | 43 | 0 | 0 | 0 | 4 |
| minor | 130 | 131 | 130 | **1** (`913`) | 0 | 0 | 10 |
| unit | 436 | 441 | 422 | **19** | **14** | 0 | **84** |

Zero parent-code mismatches at every level. The 19 project-only and 14
official-only unit codes are an **exact, code-for-code match** to the
list `Documentation/Phase_2/Week_1/module_a_week1_report.md` §5.1/§5.2
already published from an earlier, independent WISCO-based comparison —
confirming that earlier finding directly against a primary source for
the first time. The verified root cause (`6161`-`6164` should be
`6310`-`6340`) is now confirmed by byte-identical primary-source titles,
not merely inferred. The previously-**undocumented** 131-vs-130 minor-
group gap is now fully explained: exactly one fabricated code, `913`
("Building and Related Caretakers"), not present in the official
standard.

**New finding**: 84 of the 422 shared unit codes have a title that
differs after whitespace/case normalization — not previously checked by
any prior task (the WISCO-based comparison only checked code presence,
never titles). Most are cosmetic (spelling, abbreviation expansion); a
smaller subset in two code ranges (`5221`-`5230`, `9510`-`9629`) shows
titles that look systematically offset/misaligned rather than randomly
wrong — flagged for priority human review, not resolved here. Full
detail, all 84+10+4 pairs, and every affected code's official/project
label and parent: `Documentation/Conference_I_Reviewer_2/
ISCO08_PRIMARY_CATALOGUE_RECONCILIATION.md` §4 (summary in the tracked
doc) and the local `reconciliation_diff.json` (full raw diff, git-
ignored; SHA-256 `6bc1689f53eb08fb999a9a8e86acb541d15010bdb7bca7280584a94255ac6eb8`).

## 6. Ignored local artifact paths

```text
eval/local_catalogues/ilo_isco08_2021/ISCO-08_EN_Structure_and_definitions.xlsx
eval/local_catalogues/ilo_isco08_2021/ISCO_08_AR_structure_V1.0.xlsx
eval/local_catalogues/ilo_isco08_2021/normalized/isco08_official_normalized.csv
eval/local_catalogues/ilo_isco08_2021/normalized/normalization_report.json
eval/local_catalogues/ilo_isco08_2021/normalized/catalogue_importer_output.txt
eval/local_catalogues/ilo_isco08_2021/normalized/reconciliation_diff.json
eval/local_catalogues/ilo_isco08_2021/reconcile_against_project.py
```

All confirmed covered by the pre-existing `eval/local_catalogues/`
`.gitignore` entry; none were committed.

## 7. Exact changed tracked files

```text
Documentation/Conference_I_Reviewer_2/ISCO08_PRIMARY_CATALOGUE_RECONCILIATION.md   (new)
Documentation/Conference_I_Reviewer_2/FLAT_BASELINE_COVERAGE_AUDIT.md              (updated)
Documentation/Conference_I_Reviewer_2/STANDARDS_SOURCE_PROVENANCE.md               (updated)
Documentation/Conference_I_Reviewer_2/REVIEWER_RESPONSE_IMPLEMENTATION_MATRIX.md   (updated)
Documentation/Conference_I_Reviewer_2/README.md                                    (updated)
eval/normalize_ilo_isco08_catalogue.py                                             (new)
eval/test_normalize_ilo_isco08_catalogue.py                                        (new)
eval/verified_catalogue_counts.yaml                                                (new)
Documentation/AI_HANDOFF/CLAUDE_TASK_20_FINAL_REPORT.md                            (new, this file)
```

No production hierarchy data (`_MAJOR`/`_SUBMAJOR`/`_MINOR`/`_UNIT`),
`vector_store.py`, `hierarchical_store.py`, classifier code, Qdrant data,
or evaluation configuration was changed. `eval/catalogue_importer.py`
was read but not modified.

## 8. Focused and full test results

```
python -m pytest eval/test_normalize_ilo_isco08_catalogue.py eval/test_catalogue_importer.py eval/test_coverage_audit.py eval/test_docs_consistency.py -q
→ 72 passed in 38.54s

python -m pytest backend/tests eval/ -q
→ 1995 passed, 1 deselected, 1 warning in 298.21s (0:04:58)
```

**Zero failures in either run.** `1995 = 1979` (Task 19's baseline) `+
16` (all-new `eval/test_normalize_ilo_isco08_catalogue.py` tests) —
confirming zero regressions. `eval/test_docs_consistency.py` (included
in the focused run) passed against the updated documentation.

## 9. Confirmation: no WISCO input, Qdrant/model/LLM/evaluation, production correction, or benchmark occurred

- **WISCO**: zero WISCO file, path, package, code, title, split, or
  output was read at any point — the normalizer's only input is the ILO
  workbook path passed on the command line; the reconciliation script's
  only inputs are the normalized official CSV and `backend/rag/
  load_full_isco.py`, imported read-only. Neither touches
  `backend/evaluation/wisco/` or `eval/local_benchmarks/`.
- **Qdrant**: zero connection, query, count, build, populate, mutation,
  rebuild, or deletion. No Qdrant client was ever instantiated in this
  task.
- **Model/LLM**: zero SentenceTransformer load/download, zero Ollama/
  CrewAI/LLM/paid-API call. The only network operations performed were
  the two explicitly authorized workbook downloads and the landing-page
  fetch for provenance confirmation (§2).
- **Evaluation**: zero invocation of `eval/run_eval.py`,
  `eval/analyze.py`, or `eval/analyze_wisco_tier1.py` on real data. The
  only code execution beyond the normalizer/importer/reconciliation
  scripts themselves was the two hermetic pytest runs in §8.
- **Production correction**: `_MAJOR`/`_SUBMAJOR`/`_MINOR`/`_UNIT`
  remain byte-identical to the Task 19 base (not diffed here again since
  §7's changed-file list already proves no `backend/rag/*.py` file was
  touched).
- **Benchmark**: no B1 re-freeze, B2 sweep, ISIC/ISCED/SRE work, or flat-
  comparator implementation occurred.

## 10. Protected-branch and clean-tree confirmation

| Branch | Status |
|---|---|
| `master` | not touched |
| `conference1-b2-evaluation` | not touched |
| every prior `reviewer2-*` task/integration branch (through `reviewer2-flat-baseline-coverage-audit-20260808`) | not touched |

No `git merge`, `git rebase`, `git reset`, `git clean`, `git stash`,
`git pull`, or force-push occurred. No PR was created. Working tree was
clean before branching and clean immediately before this report's own
commit (only the files in §7 changed).

## 11. Precise next staged action (needs separate approval)

Per `ISCO08_PRIMARY_CATALOGUE_RECONCILIATION.md` §6, the next step is
**human review of the full mismatch list** (the 20 non-standard codes,
14 missing codes, and 84 title mismatches — with priority attention to
the two apparently-misaligned unit-code ranges, `5221`-`5230` and
`9510`-`9629`) and an explicit, approved decision on final accepted
catalogue scope. Only after that approval should a dedicated correction
task touch `backend/rag/load_full_isco.py`, followed separately by an
approved Qdrant rebuild, a new strict full WISCO run, and a new
fail-closed analysis task. **None of these were started in this task.**
The existing Task 17 hierarchical WISCO result remains a raw, non-
accuracy artifact and is not citable as an ISCO-08 accuracy result until
that entire staged process completes.
