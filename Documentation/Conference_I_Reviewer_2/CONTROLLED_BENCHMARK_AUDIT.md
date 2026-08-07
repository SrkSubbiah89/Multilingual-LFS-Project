# Controlled Benchmark Audit

Conference I Reviewer #2 response, Step 6: "Determine whether the
repository already contains a defensible controlled benchmark... If it is
defensible, package and document it correctly. If it is not defensible,
create a complete benchmark-preparation and annotation workflow without
inventing labels."

No fabricated labels, provenance, or real-LFS status appear anywhere in
this document. Every "missing" field below is reported missing, never
inferred or assumed present.

## Phase A — Candidate dataset inventory

| File | Type | Sample count | Task(s) covered | Language coverage | Apparent label source | Split status | Data sensitivity | Safe to retain in Git? |
|---|---|---|---|---|---|---|---|---|
| `eval/test_set_full130.csv` | CSV | 130 | ISCO-08 only (`gold_isic`/`gold_isced` columns exist but are **0/130 populated**) | en 100, ar 30 | **Undocumented** — no `gold_label_source`/coder field exists in this file or anywhere referencing it; git history shows one bulk commit (`a00d973`, "Add ISCO/ISIC/ISCED classifiers, evaluation harness...") with no separate annotation trail | Frozen (used as the confirmation set for B0/B1/B2), but **not independently split from its own label-creation process** | None — occupation-title sentences, no PII | Yes (already committed; contains no real respondent data) |
| `eval/test_set_sys_compare50.csv` | CSV | 50 | ISCO-08 only (same empty ISIC/ISCED columns) | Not yet re-verified per-row (same schema as full130) | Undocumented, same as full130 | Not documented as dev/heldout; purpose is system-comparison smoke testing | None | Yes |
| `eval/test_set_smoke20.csv` | CSV | 25 (despite the "20" in the filename) | ISCO-08 + ISIC (section) + ISCED (level) — the only existing test set with `industry_text`/`education_text` populated for all rows | en 19, ar 6 | Undocumented — same gap as full130 | Explicitly **not** independent (already reused across "model, beam, and routing experiments" per `eval/dev_set_schema.md`) | None | Yes |
| `eval/dev_set_v1_template.csv` | CSV | 0 (template only, never populated) | N/A | N/A | N/A — file is a header-only schema template | N/A | None | Yes |
| `eval/dev_set_schema.md` | Markdown | N/A (schema doc) | N/A | N/A | N/A | Documents required leakage-safety discipline for a *future* dev set | None | Yes |
| `eval/fixtures/synthetic_lfs_intake_package/synthetic_test_set.csv` | CSV | 5 | ISCO + ISIC + ISCED | en 3, ar 2 | Explicitly synthetic (`SYNTHETIC EXAMPLE -` prefixed), fixture labels invented for test coverage (Step 4/5 disclosure) | dev/heldout split manifest exists (`split_manifest.json`, distinct `split_id`s) | None | Yes (already the project's canonical "safe to commit" synthetic fixture) |
| `eval/results/raw_runs/*.csv` (25 files) | CSV | Varies | ISCO (B0-B2 experiment outputs) | N/A | **Not a label source** — these are *derived result* files (predictions from prior runs against full130/smoke20), not independent gold-label datasets | N/A | None | Yes (already committed; no gold labels of their own) |
| `backend/evaluation/wisco/data/raw/occupations_ISCO08_5dgt_55languages_4000titles_surveycodings_20230818.xlsx` | XLSX (raw, external) | ~4,745 rows in CODESET sheet (4,232 clean occupation records after parsing) | ISCO-08 (4-digit, direct); industry crosswalk via NACE Rev.1.1/Rev.2 (**not** direct ISIC Rev.4); ISCO skill-level 1-4 (**not** ISCED — a documented proxy only) | 61 base languages incl. all 5 project target languages (en 4,230 / ar 4,167 / ur 3,989 / hi 4,202 / tl 4,172) | **External, authoritative**: WISCO, published by SurveyCodings/WageIndicator Foundation (creator Kea Tijdens), Zenodo version DOI `10.5281/zenodo.8262593`, CC-BY-4.0, MD5-verified download integrity — see `Documentation/Phase_2/Week_1/PROVENANCE.md` | **No split exists yet** — raw external reference data, never split into dev/heldout | None — occupation-title/code reference data, no PII, no respondent data of any kind | Already committed (19 MB total across 2 xlsx files) — see note below on repo size |
| `backend/evaluation/wisco/data/processed/wisco_raw_parsed.json` | JSON (parsed) | 4,232 | ISCO-08 (4-digit unit group + full hierarchy path + skill level) | Same 5-language subset extracted from the raw file | Same as raw file (derived, not re-labelled) | No split yet | None | Yes (already committed) |
| `backend/evaluation/wisco/data/processed/wisco_industry_crosswalk.json` | JSON (parsed) | 4,291 | NACE Rev.1.1/Rev.2 industry codes keyed to WISCO's ISCO-08 codes — **not** ISIC Rev.4 directly | N/A (code-to-code crosswalk) | Same WISCO provenance | No split yet | None | Yes (already committed) |
| `Documentation/Phase_2/Week_1/PROVENANCE.md`, `module_a_week1_report.md`, `wisco_structure_inspection.md`, `language_mapping_note.md` | Markdown | N/A | N/A | N/A | These ARE the provenance documentation for WISCO | N/A | None | Yes |

**Data-sensitivity note on the WISCO raw files**: both `.xlsx` files (7.0 MB + 8.6 MB, 19 MB total) are already tracked in Git. They contain zero personal/respondent data — pure occupation-title/code reference tables — so retention is safe from a privacy standpoint. This audit does not flag file size as an in-scope concern (`.gitignore`'s large-binary policy targets installers/executables, not data files), but notes it for completeness since a future contributor might otherwise wonder why an evaluation-adjacent directory contains multi-megabyte binaries.

**Not found**: no other CSV/JSON/YAML file anywhere in the repository (searched `eval/`, `backend/`, `Documentation/`, excluding `node_modules`/`__pycache__`/`eval/local_runs/`) matches the search terms B0/B1/B2/flat/hierarchical/ISCO/ISIC/ISCED/dev/test/heldout/gold/label/benchmark beyond the files listed above and the already-audited Step 3-5 synthetic fixtures.

## Phase B — Label-provenance audit

Checklist items, evaluated per dataset. `✓` = evidence present and verified in-repo. `✗` = evidence absent (never inferred). `partial` = some evidence exists but is incomplete or requires further work to be usable.

| # | Checklist item | `test_set_full130.csv` | `test_set_smoke20.csv` | `test_set_sys_compare50.csv` | WISCO (ISCO-08 only) |
|---|---|---|---|---|---|
| 1 | Clearly documented source | ✗ | ✗ | ✗ | ✓ (Zenodo DOI 10.5281/zenodo.8262593, `PROVENANCE.md`) |
| 2 | Non-personal / de-identified input text | ✓ (authored occupation-title sentences) | ✓ | ✓ | ✓ (occupation titles, not respondent text) |
| 3 | Declared classification version | ✓ (ISCO-08, by convention of the codebase; not stated in-file) | ✓ | ✓ | ✓ (ISCO-08 explicit; NACE Rev.1.1/2.0 for industry) |
| 4 | Gold labels not generated by the system under evaluation | **unconfirmed** — no record of how labels were produced; cannot rule out that they were written with the classifier's own taxonomy open as reference | unconfirmed, same reason | unconfirmed, same reason | ✓ (external publisher, predates and is independent of this codebase) |
| 5 | Human or authoritative-source label provenance | ✗ (no coder identity, no source citation) | ✗ | ✗ | ✓ (SurveyCodings/WageIndicator Foundation, a named institutional data publisher) |
| 6 | Independent double-coding / adjudication evidence | ✗ | ✗ | ✗ | ✗ — not documented in this repo (WISCO's own internal coding methodology is not reproduced here; absence recorded, not assumed) |
| 7 | Frozen held-out test split | partial (used *as if* frozen for B0-B2 confirmation, but never formally split from a paired dev set at creation time) | ✗ (already reused across prior experiments — explicitly disqualified from "clean" status by `eval/dev_set_schema.md`) | ✗ (no split role documented) | ✗ (no split exists yet — raw reference data) |
| 8 | Distinct development split | ✗ (`dev_set_v1.csv` was never populated — 0 rows) | ✗ | ✗ | ✗ (not yet constructed) |
| 9 | Stable dataset and split hashes | ✗ (no hash ever computed/recorded prior to this audit) | ✗ | ✗ | ✗ (raw file MD5 is recorded in `PROVENANCE.md`; the *parsed* JSON and any *split* have no hash yet) |
| 10 | Valid dataset card | ✗ | ✗ | ✗ | ✗ |

### Status labels (exact vocabulary, per task instructions)

| Dataset | Status |
|---|---|
| `eval/test_set_full130.csv` | `not_eligible_unknown_provenance` |
| `eval/test_set_smoke20.csv` | `not_eligible_unknown_provenance` (compounded by `not_eligible_missing_split_discipline` — already reused across prior experiments, per `eval/dev_set_schema.md`'s own leakage-safety rationale) |
| `eval/test_set_sys_compare50.csv` | `not_eligible_unknown_provenance` |
| `eval/dev_set_v1_template.csv` | Not applicable — template only, 0 records, nothing to audit |
| WISCO (ISCO-08 dimension) | `conditionally_eligible_needs_documentation` — real external authoritative-source labels (items 1, 2, 3, 4, 5 satisfied), but items 6, 7, 8, 9, 10 all require new work before it is benchmark-ready (see Phase C) |
| WISCO (ISIC dimension, via NACE crosswalk) | `not_eligible_missing_independent_labels` — no direct ISIC Rev.4 code exists in the source; the NACE→ISIC hop is undocumented and only ~17% row coverage on the cleaner (NACE2.0) path per `parse_wisco.py`'s own analysis |
| WISCO (ISCED dimension, via skill-level proxy) | `not_eligible_missing_independent_labels` — `isco08_skill_level` (1-4) is an ISCO-08 property, not an ISCED code; no documented skill-level → ISCED-level crosswalk exists in this repo |
| Step 4/5 synthetic fixture (`eval/fixtures/synthetic_lfs_intake_package/`) | Already correctly labelled `synthetic_or_operationally_realistic` with `manuscript_eligible=false` per Steps 4-5.1 — not re-audited here, no change |

No dataset in this repository qualifies for `eligible_controlled_benchmark` (which would require zero missing items) or `not_eligible_contains_sensitive_data` (nothing found contains sensitive data).

## Phase C — Disposition

### The 130-case ISCO set (`eval/test_set_full130.csv`)

**Lacks independent labels and documented provenance** (Phase B: 6 of 10
checklist items fail or are unconfirmed). Per the task's Phase C.2 branch:

- **Not used as final evaluation evidence.** No claim of "controlled
  benchmark" status is made for this file by this audit.
- **Preserved as a development/engineering fixture only** — it continues to
  serve its existing role (B0/B1/B2 system-comparison smoke testing,
  `eval/ablation_runner.py`'s dry-run and Step-4/5-style integration
  checks) with an explicit disclosure of what it is not.
- **Missing evidence, documented** (see Phase B table above): no
  `gold_label_source`, no coder identity, no double-coding, no adjudication,
  no dataset card, no independent split, no hash.
- **Upgrade path**: the same annotation/adjudication workflow this Step 6
  pass documents for a new controlled benchmark (`ANNOTATION_AND_
  ADJUDICATION_GUIDE.md`) could, in principle, be applied retroactively to
  `full130` — i.e., have two independent coders re-label its 130 cases
  blind to the existing `gold_isco_4digit` values, adjudicate disagreements,
  and only then promote it. This is **not done in this step** (Step 6 is
  audit-and-prepare only, per restriction 5) and would itself need a
  legal/ethical review of whether the original 130 occupation-title
  sentences may be redistributed/relabelled (they appear to be
  author-written, not sourced from a third party, so this is likely
  low-risk, but that judgement belongs to the authors, not this audit).

### WISCO — the actually-defensible candidate

The audit's most consequential finding is that **`eval/test_set_full130.csv`
is not the repository's best available controlled-benchmark candidate.**
The dormant WISCO dataset under `backend/evaluation/wisco/` — downloaded
and parsed during an earlier project phase (`Documentation/Phase_2/Week_1/`)
for a different purpose (comparing this system's own ISCO-08 knowledge-base
coverage against the real standard) — has real, externally-sourced,
CC-BY-4.0-licensed, DOI-anchored ISCO-08 gold labels across all 5 of this
project's target languages, and has **never been used as an evaluation
benchmark**.

For **ISCO-08**, WISCO is `conditionally_eligible_needs_documentation`: the
label-provenance items that matter most (source, non-personal text,
declared version, external/independent origin, authoritative publisher) are
already satisfied. What's missing is entirely mechanical, not evidentiary:
a dataset card, a dev/heldout split with hashes, and confirmation of
double-coding/adjudication evidence (which may exist in WISCO's own
methodology documentation, not reproduced in this repo — the honest
position is "not confirmed here," not "absent").

For **ISIC and ISCED**, WISCO does **not** currently provide usable direct
gold labels (see Phase B) — building a defensible NACE→ISIC crosswalk and
an ISCO-skill-level→ISCED crosswalk is out of scope for this step and would
need its own documented methodology before any claim of ISIC/ISCED coverage
from WISCO could be made.

**Disposition**: per Phase D below, a WISCO-derived controlled-benchmark
package is prepared (dataset card, split manifest, record schema) for
**ISCO-08 only**, labelled `synthetic_or_operationally_realistic` (never
`approved_real_lfs_validation` — WISCO is reference/dictionary data, not
Labour Force Survey respondent data, so it can never carry that label
regardless of its label quality). It is made ready for Step 7 but **not
run** in this step (restriction 5).

## Step 7A update — leakage audit and group-aware split correction

A strict leakage audit (`eval/audit_wisco_benchmark_leakage.py`) found that
the v1 package above, while free of raw source-key leakage, had **4
cross-split exact-duplicate-text groups**: different WISCO occupation keys
sharing byte-identical titles (in one language each) and identical gold
codes, split across dev/heldout by coincidence of the per-key hash. A
corrected, group-aware v2 (`eval/local_benchmarks/wisco_isco08_v2_group_split/`,
fixed seed 42, union-find text-duplicate grouping) is now the valid split
for any future evaluation — v1 is retained unchanged as the audit record of
this finding, never deleted. Full detail:
`Documentation/Conference_I_Reviewer_2/WISCO_LEAKAGE_AUDIT_AND_RUN_PLAN.md`.
