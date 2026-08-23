# `eval/dev_set_v1.csv` — readiness report

Generated: 2026-08-05 (validator-strengthening pass). Reflects the actual,
current state of `eval/dev_set_v1.csv` — **0 data rows**, canonical
7-column schema (see `eval/dev_set_schema.md`). This report is a real run
against the real (empty) file, not a projection of a populated one.

> **Update, 2026-08-22**: this report's "441-code catalogue" mentions below
> are a historical record from this pass and are left unedited. The
> 441-vs-436 discrepancy they reference was resolved 2026-08-12 (primary-
> source ILO cross-check); the catalogue is now 436 codes, live-reconfirmed
> 2026-08-22. See `eval/dev_set_schema.md`'s "Resolved: former 441-versus-436
> discrepancy" section for the current, maintained state.

## What changed in this pass

Schema is unchanged (still 7 columns; see `eval/dev_set_schema.md`'s
migration note). What's new is the *strength* of validation applied to
whatever data eventually populates this file:
- **Semantic ISCO-08 code validation** — `gold_isco_code` must now be a
  real unit-group code (441-code catalogue from `backend/rag/
  load_full_isco.py`), not just 4 digits. `0000`/`9999`-style codes are
  now rejected.
- **Normalization-integrity enforcement** — `eval/configs/
  full130_leakage_manifest.json` now carries a `normalization_fingerprint`;
  both `eval/validate_dev_set.py` and `eval/pre_run_check.py` fail closed
  if the live `normalize_text()` no longer matches it.
- **CSV structural validation** — malformed quoting and ragged rows are
  now rejected before field-level checks run; multiline quoted
  `respondent_text` cells are confirmed supported.
- **`eval/build_full130_leakage_manifest.py`** formalises the one
  authorised path that reads `eval/test_set_full130.csv`, with a
  regression-guard test suite proving `validate_dev_set.py`/
  `pre_run_check.py` never do.
- **Exact classifier-input regression test** added (`eval/
  test_run_eval_b2.py`) — proves `respondent_text` reaches
  `ISCOClassifier.classify()` verbatim and that the dev-set mapping is
  identical to B0/B1's. No production mapping code changed.

## Checklist

| # | Check | Result | Detail |
|---|---|---|---|
| 1 | Minimum case-count target (50+; 30 absolute floor) | **FAIL** | 0 cases. Blocked on human-labelled/authoritative-source record intake — see `eval/dev_set_v1_provenance.md`. |
| 2 | Arabic-case target (15+) | **FAIL** | 0 Arabic cases (0 cases of any language). |
| 3 | Schema validation (`eval/validate_dev_set.py`) | **FAIL** | Ran `python eval/validate_dev_set.py --dev-set eval/dev_set_v1.csv`; exited 1 with `ERROR: Dev set is empty (no data rows).` This is the *only* reason for failure. Before this, the run printed `Manifest normalization integrity OK` — the new fail-closed gate passed and did not block the run. |
| 4 | Valid 4-digit **and existent** ISCO codes | **N/A** | No rows to check. Semantic-catalogue loading itself was verified separately (441 codes, real source) — see `eval/test_validate_dev_set.py::test_load_isco_unit_group_catalogue_against_real_source`. |
| 5 | Language counts | **FAIL** | 0 en / 0 ar / 0 mixed. |
| 6 | ISCO major-group coverage | **N/A** | No rows to check. Major group is derived from `gold_isco_code[:1]`, not stored. |
| 7 | Exact-ID leakage check vs. full130 | **PASS (vacuous)** | `check_full130_manifest_overlap()` against `eval/configs/full130_leakage_manifest.json` → `No case_id or normalized-text collisions with full130's 130 case(s) (checked via manifest only -- full130 itself was never opened)`. Vacuous (0 dev rows); re-run once real data exists. |
| 8 | Normalized-text-hash leakage check vs. full130 | **PASS (vacuous)** | Same invocation as #7. Manifest's `normalization_fingerprint` independently confirmed matching the live `normalize_text()` before this check ran (see #9). |
| 9 | **Normalization-integrity gate (new)** | **PASS** | `check_manifest_normalization_integrity()` → `normalization_version='v1', fingerprint confirmed live-matching.` Proven to fail closed on a mismatched/missing fingerprint by `eval/test_validate_dev_set.py::test_changed_normalize_text_implementation_causes_validation_failure` and 3 related tests. |
| 10 | **CSV structural validation (new)** | **PASS** | `validate_csv_structure()` on the header-only template → no errors (0 data rows, header parses cleanly). |
| 11 | Human/authoritative label provenance | **FAIL / BLOCKED** | No records supplied yet, so no `gold_label_source`/`annotator_or_adjudication_reference` values exist to verify. This is the binding blocker — see below. |
| 12 | Unresolved data-quality issues | **0 open items** | The schema-name mismatch flagged in an earlier version of this report is resolved (prior pass). No new open items from this pass. |

## Achieved vs. recommended — explicit distinction

**Achieved (verifiable right now):**
- The file exists with the canonical column headers, consistent across
  every script that reads it.
- `eval/validate_dev_set.py --dev-set eval/dev_set_v1.csv` fails *only*
  because the file is empty.
- The manifest's normalization fingerprint is confirmed live-matching —
  its hashes are currently trustworthy for comparison.
- The leakage-check tooling never opens `eval/test_set_full130.csv`,
  confirmed by both an AST source scan and a runtime `open()`-guard test.
- No fabricated rows were added to force any check to pass.

**Not yet achieved (recommended targets, blocked on data intake):**
- 50+ cases (30 minimum).
- 15+ Arabic cases.
- Multi-major-group coverage.
- Any actual leakage/semantic-code verification with real content (current
  PASSes are vacuous — zero rows means zero chance of a violation, not
  zero risk).
- Human/authoritative-source label provenance.

**Schema validity ≠ readiness.** See `eval/dev_set_schema.md`'s "Schema
validity vs. readiness with real labels" section: everything this report
can check is mechanical (format, structure, catalogue membership,
non-overlap). Whether the eventual gold labels are *correct* and whether
`respondent_text` is a *genuine* respondent utterance are provenance
questions this tooling cannot answer — only human review can.

## Commands run for this report (all non-inference, no full130 raw access)

```
python eval/validate_dev_set.py --dev-set eval/dev_set_v1.csv
```
```python
# direct library call, not the full pre-run gate
import pre_run_check as prc
prc.check_full130_manifest_overlap(dev_rows, manifest)  # manifest = full130_leakage_manifest.json
```
```
python -m pytest eval/test_validate_dev_set.py eval/test_dev_sweep.py \
    eval/test_pre_run_check.py eval/test_run_eval_b2.py
```
See the validator-strengthening task's final response for the exact pass count.

## Binding blocker

**This dataset cannot progress past FAIL on checks 1, 2, 5, and 11 without
human-labelled or authoritative-source records being supplied.** No amount
of further tooling work resolves this — it requires either qualified human
coders to label real respondent job-title/description text against
ISCO-08 (using only codes from the real 441-code catalogue), or an
approved, citable external labelled corpus. See `eval/dev_set_v1_provenance.md`
for the exact intake process those records must follow, and
`eval/dev_set_schema.md` for the exact 7-column schema and every
validation rule they must satisfy.
