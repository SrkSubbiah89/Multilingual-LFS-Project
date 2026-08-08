# Task 20 — Official ILO ISCO-08 Catalogue Reconciliation

## Purpose

Resolve the catalogue-identity blockers established by Task 19 using a primary, machine-readable ILO source. This task must produce a reproducible, source-hashed official ISCO-08 structure import and a code-by-code reconciliation against the project’s hand-authored `_MAJOR`, `_SUBMAJOR`, `_MINOR`, and `_UNIT` hierarchy lists.

This task does **not** correct production hierarchy data, build a Qdrant collection, run an evaluation, rerun WISCO, or report any accuracy result. It is official-catalogue acquisition, validation, and discrepancy documentation only.

The official ILO source, not WISCO, is the sole authority for all code/parent/count decisions in this task. WISCO must remain an independent benchmark and must not be read, imported, parsed, or used as a crosswalk source.

## Required source state

Fetch `origin` and verify before any change:

| Role | Branch | Required SHA |
|---|---|---|
| Task 19 audit base | `reviewer2-flat-baseline-coverage-audit-20260808` | `c158897f691e9b94855f670503a70d02877d84d1` |

Require a clean working tree, then create and work only on:

```text
reviewer2-isco08-official-catalogue-reconciliation-20260808
```

Push only this new branch. Do not create a pull request. Do not merge, rebase, reset, clean, stash, pull, force-push, or modify any protected/prior branch.

## Permitted primary ILO source files

Use the following official ILO download URLs only:

1. Primary English structure and definitions workbook, authoritative for code, title, hierarchy, and counts:

```text
https://webapps.ilo.org/ilostat-files/ISCO/newdocs-08-2021/ISCO-08/ISCO-08%20EN%20Structure%20and%20definitions.xlsx
```

2. Optional official ILO English-Arabic structure workbook, usable only to document available official Arabic labels. It must not alter the English code/hierarchy interpretation:

```text
https://webapps.ilo.org/ilostat-files/Documents/ISCO_08_AR_structure_V1.0.xlsx
```

Also retain the ILO ISCO-08 landing page as the primary human-readable source for the official hierarchy totals:

```text
https://isco.ilo.org/en/isco-08/
```

The expected official hierarchy counts are:

```text
10 major groups
43 sub-major groups
130 minor groups
436 unit groups
```

If a downloaded workbook or its schema contradicts those figures, do not choose a convenient interpretation. Record the exact conflict, preserve the downloaded bytes locally, and stop without creating a verified catalogue-count record.

## Local official-source handling

1. Download source workbook(s) only to an ignored local directory:

```text
eval/local_catalogues/ilo_isco08_2021/
```

2. Never commit a raw official workbook, raw normalized official CSV, or a copied catalogue table.
3. Record in an ignored local provenance JSON:
   - exact URL;
   - retrieval timestamp in UTC;
   - SHA-256 of each downloaded byte stream;
   - file size;
   - workbook/sheet names;
   - source column names;
   - parser version/script hash.
4. Preserve the raw workbook unmodified.
5. No WISCO path, package, code, title, split, or output may be read at any stage of this task.

## Existing importer assessment and normalization

First inspect `eval/catalogue_importer.py` and its tests.

### If it already accepts the official XLSX schema directly

Use it with the official English workbook and record the exact command.

### If it does not accept the official XLSX schema directly

Add a small deterministic parser/normalizer:

```text
eval/normalize_ilo_isco08_catalogue.py
eval/test_normalize_ilo_isco08_catalogue.py
```

Requirements for the normalizer:

1. Use `openpyxl` or an already installed project dependency. Do not call a network, Qdrant, model, or subprocess.
2. Accept explicit input/output paths.
3. Parse only the official English workbook’s documented structure sheet.
4. Fail closed on missing/ambiguous sheet or column names, blank/invalid codes, duplicate codes, malformed hierarchy levels, invalid parent relationships, or unexpected record counts.
5. Output a local ignored normalized CSV or JSON in the input shape accepted by `catalogue_importer.py`.
6. Do not hard-code a catalogue row list in the parser or tests.
7. Tests must use small synthetic temporary XLSX fixtures, never a copied official workbook and never WISCO.
8. Test successful parsing, sheet/column rejection, malformed code rejection, duplicate rejection, parent-link validation, and all expected-level-count validations.

Do not modify `catalogue_importer.py` unless a narrowly justified parser bug prevents importing a valid normalized official catalogue. If a change is necessary, add focused regression coverage and explain why normalization alone was insufficient.

## Official catalogue validation

From the primary English ILO workbook, derive an authoritative local normalized structure and validate, at minimum:

1. exactly 10 one-digit major-group codes;
2. exactly 43 two-digit sub-major-group codes;
3. exactly 130 three-digit minor-group codes;
4. exactly 436 four-digit unit-group codes;
5. every code is unique at its level;
6. every 2-digit code has a parent equal to its first digit;
7. every 3-digit code has a parent equal to its first two digits;
8. every 4-digit code has a parent equal to its first three digits;
9. every unit-group title is nonblank;
10. no code is fabricated, padded, reclassified, or sourced from WISCO.

Run the existing fail-closed `catalogue_importer.py` validation against the normalized official catalogue. Retain only ignored local normalized data and importer output.

If all primary-source validations pass, add:

```text
eval/verified_catalogue_counts.yaml
```

This tracked file must contain **metadata and verified counts only**, not raw catalogue rows. It must include:

- standard key `isco08`;
- issuing organization `International Labour Organization`;
- primary source landing-page URL;
- exact official workbook URL;
- retrieval date/time;
- raw workbook SHA-256;
- normalized-catalogue SHA-256;
- parser/importer version or script hashes;
- sheet/column mapping;
- verified counts at all four hierarchy levels;
- validation status and command;
- a licence/usage note copied or accurately characterized from the official source where available, otherwise `not stated in retrieved source`;
- a statement that this file is based on a primary official ILO import and has no WISCO input.

Only create this file after all validations pass. Do not create a partially verified version.

## Code-by-code reconciliation against project hierarchy

Using the primary-source normalized local catalogue, compare it programmatically and reproducibly against the project lists in:

```text
backend/rag/load_full_isco.py
```

Compare each level separately: `_MAJOR`, `_SUBMAJOR`, `_MINOR`, `_UNIT`.

For each level report:

1. official count;
2. project static-list count;
3. shared code count;
4. project-only codes;
5. official-only codes;
6. parent mismatch codes;
7. English title mismatches after whitespace/case normalization, while preserving exact original title strings in the machine-readable diff;
8. code ordering differences, clearly separated from code-identity problems;
9. a deterministic SHA-256 for each code list and the complete diff artifact.

Write the full machine-readable diff only under the ignored local output directory. It must include every affected code and its official/project labels/parents where available.

Do **not** change `_MAJOR`, `_SUBMAJOR`, `_MINOR`, `_UNIT`, `vector_store.py`, `hierarchical_store.py`, classifier code, Qdrant data, or any evaluation configuration in this task.

## Documentation

Add:

```text
Documentation/Conference_I_Reviewer_2/ISCO08_PRIMARY_CATALOGUE_RECONCILIATION.md
```

Update:

```text
Documentation/Conference_I_Reviewer_2/FLAT_BASELINE_COVERAGE_AUDIT.md
Documentation/Conference_I_Reviewer_2/STANDARDS_SOURCE_PROVENANCE.md
Documentation/Conference_I_Reviewer_2/REVIEWER_RESPONSE_IMPLEMENTATION_MATRIX.md
Documentation/Conference_I_Reviewer_2/README.md
```

The documentation must:

1. cite the ILO landing page and exact workbook URL;
2. distinguish official verified counts from the project’s current static-list counts;
3. report the exact code-by-code reconciliation figures;
4. state that no WISCO content was used to correct or validate the catalogue;
5. state that the current Task 17 hierarchical result is **not** a citable ISCO-08 accuracy result until a later, separately approved correction/rebuild/re-evaluation process uses the reconciled official catalogue;
6. describe the 441/436 and 131/130 discrepancies as resolved only if the full primary-source code-by-code diff proves they are resolved;
7. otherwise retain the `UNRESOLVED` state and list exact residual codes;
8. update the reviewer matrix status honestly: official-catalogue provenance strengthened, but no new classifier benchmark evidence yet.

## Correction plan, but no correction

If the official import and code-by-code diff succeed, document a staged future plan only:

1. human review of the full mismatch list and acceptance of exact official scope;
2. a dedicated source-data correction task with hermetic tests;
3. a separately approved local Qdrant rebuild from the corrected official list;
4. a new strict full WISCO run for both a unit-group-only flat comparator and hierarchical system;
5. a new fail-closed analysis task.

Do not take any of these steps now.

## Prohibited operations

Do not run:

- `eval/run_eval.py`, `eval/analyze.py`, `eval/analyze_wisco_tier1.py` on real data, or any benchmark/evaluation;
- Qdrant connection, query, count, build, populate, mutation, rebuild, or deletion;
- SentenceTransformer model load/download;
- Ollama, CrewAI, LLM, paid API, or external inference;
- WISCO data/source/split/output read, write, or comparison;
- production hierarchy correction;
- flat comparator implementation;
- B1 re-freeze, B2 sweep, ISIC/ISCED/SRE work.

The only permitted network operation is retrieval of the two explicitly listed official ILO workbook URLs and the ILO landing page for provenance confirmation.

## Tests

Run, as applicable:

```bash
pytest eval/test_normalize_ilo_isco08_catalogue.py eval/test_catalogue_importer.py eval/test_coverage_audit.py eval/test_docs_consistency.py -q
pytest backend/tests eval/ -q
```

If no normalizer was needed, omit only its nonexistent test file and report that fact. Do not weaken, skip, xfail, or alter unrelated tests/code.

## Final report

Create:

```text
Documentation/AI_HANDOFF/CLAUDE_TASK_20_FINAL_REPORT.md
```

It must include:

1. verified source SHA, branch, final commit SHA, push confirmation, and clean-tree status;
2. official source URLs, retrieval timestamps, raw-workbook SHA-256 values, file sizes, sheet/column mapping, and use/non-use of optional Arabic workbook;
3. exact parser/importer command(s) and validation result;
4. all official hierarchy counts and verified-catalogue metadata status;
5. exact reconciliation results by level, including all code-identity and parent/title mismatch totals;
6. ignored local paths to raw workbooks, normalized official catalogue, provenance, importer output, and full diff;
7. exact changed tracked files;
8. focused and full test results;
9. confirmation no WISCO input, Qdrant/model/LLM/evaluation operation, production correction, or benchmark occurred;
10. clear status:

```text
OFFICIAL_ISCO08_CATALOGUE_RECONCILED: yes
```

or:

```text
OFFICIAL_ISCO08_CATALOGUE_RECONCILED: no
```

11. a precise next staged action and statement that it needs a separate approval.

After committing and pushing the final report, stop. Do not correct production code or begin a collection rebuild/evaluation.
