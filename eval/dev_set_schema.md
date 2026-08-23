# `eval/dev_set_v1.csv` schema (canonical, v1)

This is the **single, authoritative schema** for the B2 development set,
used only to select the candidate-capacity parameter `K`
(`--reranker-candidates`) before the single, pre-specified confirmation run
on `eval/test_set_full130.csv`. It is consistent across every script that
reads this file: `eval/validate_dev_set.py`, `eval/dev_sweep.py`,
`eval/pre_run_check.py`, their test suites, and the companion
`eval/dev_set_v1_data_dictionary.md` / `eval/dev_set_v1_provenance.md` /
`eval/dev_set_v1_readiness_report.md` documents. There is exactly one
column name for each concept — see the migration note at the bottom if
you've seen a different name in an older report or draft.

## Why this file exists (leakage-safety rationale)

`eval/test_set_smoke20.csv` has already been used for model, beam, and
routing experiments earlier in this project, so it is not a clean
development set for a new selection decision — any K chosen against it could
be overfit to prior exploration, not genuinely validated. `eval/
test_set_full130.csv` is the frozen held-out set for B0 vs. B1 vs. B2
confirmation and must never be used to *choose* a hyperparameter, only to
*confirm* one already chosen elsewhere. `dev_set_v1.csv` is a third,
independent set whose only purpose is K selection.

## Canonical columns (7, in order)

| # | Column | Type | Description |
|---|---|---|---|
| 1 | `case_id` | string | Unique identifier. **Must not collide with any `case_id` in `test_set_smoke20.csv` or the full130 leakage manifest** (see below) — prefix your IDs (e.g. `dev001`) to make collisions structurally impossible rather than relying on numbering luck. |
| 2 | `language` | `en` \| `ar` \| `mixed` | Respondent's input language. `mixed` is reserved for genuinely code-switched text, not a way around the en/ar coverage target. |
| 3 | `respondent_text` | string | **The single field sent to the classifier as job-title/description input** (see "Classifier-input field" below). The free-text a respondent would give, phrased as a survey answer — not an ISCO dictionary title copy-pasted from a reference table. |
| 4 | `gold_isco_code` | string, 4 digits, **and in the classifier-supported ISCO catalogue** | The correct ISCO-08 **unit-group** code for `respondent_text`, as determined by the process in `gold_label_source`. Must match `^[0-9]{4}$` **and** exist in the project's classifier-supported ISCO catalogue — see "Semantic ISCO-08 code validation" below. `0000` and `9999` are syntactically valid but rejected, since they are not codes the classifier can ever predict. |
| 5 | `gold_label_source` | string | How the gold code was determined. Must not be blank. Recommended values (not hard-enforced by `validate_dev_set.py`, but expected by `eval/dev_set_v1_data_dictionary.md`'s provenance standard): `human_coder_single`, `human_coder_double_agreement`, `adjudicated_panel`, or `authoritative_source:<name>`. |
| 6 | `annotator_or_adjudication_reference` | string | Identifier/reference (coder ID, adjudication ticket, citation key) for who/what produced the gold label. Identifies the **labeller**, never the respondent — see confidentiality note below. |
| 7 | `dataset_split` | string, constant | Always the literal value `dev_v1`. Lets rows retain their split identity if concatenated with other case sets downstream. |

## Classifier-input field

**`respondent_text` is the one, unambiguous classifier-input-text field.**
It is:
- **Stored directly**, not assembled by concatenating other columns — this
  schema does not have separate title/description columns to concatenate.
- **Sent to the classifier verbatim / unnormalised.** `eval/dev_sweep.py`'s
  `_load_dev_rows_as_test_set()` maps it straight through to
  `run_eval.run_one_case()`'s `input_text`, which `run_eval.py` passes as
  `job_title=input_text` to `ISCOClassifier.classify()` — exactly the same
  path B0/B1 already use for `test_set_smoke20.csv`/`test_set_full130.csv`'s
  `input_text` column. No normalisation, translation, or rewriting happens
  between the CSV and the classifier — see "Text normalisation" below for
  why that's a deliberate, separate concern from leakage-hash comparison.
- **Never AI-generated, translated, paraphrased, or otherwise rewritten.**
  It must be the respondent's own words (or a faithful verbatim transcript
  of them), exactly as `eval/dev_set_v1_provenance.md`'s labelling-process
  section requires.

There is no separate `job_title` / `job_description` split in this schema
(see migration note) — a single field keeps "what does the classifier see"
unambiguous, matching the single `input_text` column
`test_set_smoke20.csv`/`test_set_full130.csv` already use.

## Semantic ISCO-08 code validation (classifier-supported catalogue)

Format validation (`^[0-9]{4}$`) alone accepts nonsense codes like `0000`
or `9999` that no version of this project's classifier could ever predict.
`eval/validate_dev_set.py`'s `load_isco_unit_group_catalogue()` closes this
gap by checking `gold_isco_code` against the **classifier-supported ISCO
catalogue** — the codes this project's classifier can actually return. As
of the 2026-08-12 primary-source ILO cross-check documented in `CLAUDE.md`
("Knowledge base construction"), this catalogue has been independently
verified against the official ILO ISCO-08 unit-group list (see the
resolved-discrepancy note below):

- **Source**: `backend/rag/load_full_isco.py`'s `_UNIT: list[tuple[str,
  str]] = [...]` module-level literal — the same data that populates the
  `isco08_unit_groups` Qdrant collection every classification query
  actually runs against. There is no separately-maintained JSON/CSV
  catalogue in this repo; this list *is* the catalogue this validator uses.
- **How it's loaded**: read as **plain text and regex-parsed**
  (`\(\s*"(\d{4})"\s*,`), never imported as a Python module — importing
  `load_full_isco.py` would pull in `qdrant_client`/`sentence_transformers`
  at import time, which this validator must never risk triggering.
- **Count**: 436 unique 4-digit codes, live-verified 2026-08-22 (both
  directly against `_UNIT` and against the live `isco08_unit_groups`
  Qdrant collection). This matches the official ISCO-08 unit-group count
  exactly.
- **Versioned/documented**: the source path
  (`_DEFAULT_ISCO_CATALOGUE_SOURCE`) and parsing logic are both in
  `eval/validate_dev_set.py`, overridable via `--isco-catalogue-source` for
  testing. If `backend/rag/load_full_isco.py`'s `_UNIT` list is ever
  restructured, this parser (and this documentation) need updating together.
- **Fails CLOSED, not open**: `eval/validate_dev_set.py`'s CLI (`main()`)
  treats an empty/unparsable catalogue as **FATAL** — it prints an error and
  exits 1 before any dataset can pass, rather than silently downgrading to
  format-only validation (which would let `0000`/`9999`-style nonexistent
  codes slip through unnoticed). `eval/pre_run_check.py`'s
  `isco_catalogue_loaded` checklist item is likewise a **hard failure** — a
  missing/unparsable catalogue blocks `OVERALL: PASS`. The underlying pure
  function, `validate_dev_set()`, remains independently testable with an
  injected (possibly empty) catalogue for unit tests — only the CLI/gate
  entry points are fail-closed, not the library function itself.

### Resolved: former 441-versus-436 discrepancy

This section previously flagged an open discrepancy: `_UNIT` contained 441
entries while the module docstring claimed 436. **This is now resolved.**
A 2026-08-12 primary-source ILO cross-check (isco.ilo.org's official CSV
export) found 19 non-standard codes, 14 missing official codes, and 6
further codes carrying the wrong occupation's label — all fixed. `_UNIT`
now contains exactly 436 unique codes, independently verified against the
official ILO ISCO-08 standard. Live-reconfirmed 2026-08-22: `_UNIT` has
436 entries in source, and the live `isco08_unit_groups` Qdrant collection
also serves exactly 436 points (the fix was actually deployed via
`--recreate`, not just committed to source). Full detail and the one
disclosed remaining exception (Major Group 0 code-length convention):
`CLAUDE.md`'s "Knowledge base construction" section and
`backend/tests/test_load_full_isco_catalogue_consistency.py`.

## Text normalisation (leakage checks only — never classifier input)

`eval/validate_dev_set.py.normalize_text()` is:

```python
def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())
```

i.e. strip, lowercase, collapse internal whitespace. This is the **one**
normalisation function used everywhere text needs to be compared for
leakage/duplication:
1. Hashing `respondent_text` for the full130 manifest comparison
   (`eval/configs/full130_leakage_manifest.json`'s `normalized_text_sha256`
   entries are `sha256(normalize_text(input_text))` for each full130 row —
   built by the same normalisation, so a hash computed on a dev-set row is
   directly comparable).
2. Comparing `respondent_text` against `test_set_smoke20.csv`'s `input_text`
   (raw normalised-string comparison — smoke20 isn't manifest-restricted,
   so `eval/validate_dev_set.py` can and does read it directly).
3. Detecting duplicate `respondent_text` values within the dev set itself.

It is **not** applied to what actually reaches the classifier (see
"Classifier-input field" above) — normalising classifier input would be a
real behavioural change to B0/B1/B2's retrieval/reranking path, which is
explicitly out of scope for this dev-set schema work.

It is intentionally mechanical, not semantic: this catches exact/
whitespace/case duplicates only, via an exact hash or string match — **not**
via an LLM, translation, semantic-similarity, or embedding-based check. Near-
duplicate detection beyond this mechanical check (e.g. a genuine paraphrase
that survives normalisation) is **not automated** and must be handled by
human review during data intake — see `eval/dev_set_v1_provenance.md`'s
exclusions process. Do not read an automated PASS on this check as proof
that no paraphrase-level leakage exists.

### Normalization-version / fingerprint enforcement

`eval/configs/full130_leakage_manifest.json`'s hashes are only meaningful
if they were built by the *exact same* `normalize_text()` a validator is
about to compare them against. To make that a checked property rather than
an assumption, the manifest records:

- `normalization_version` — a short human string (`"v1"`, bumped by hand
  whenever `normalize_text()`'s behaviour intentionally changes).
- `normalization_fingerprint` — `sha256(inspect.getsource(normalize_text))`,
  computed by `eval/validate_dev_set.py`'s `compute_normalize_text_
  fingerprint()`. Unlike the version string, this can't be forgotten to
  bump: it changes automatically the instant `normalize_text()`'s source is
  edited at all.

Both `eval/validate_dev_set.py` (`main()`) and `eval/pre_run_check.py`
(`full130_manifest_normalization_integrity` checklist item) call
`check_manifest_normalization_integrity(manifest)` **before** trusting the
manifest for any hash comparison, and **fail closed** (hard error, no case
gets evaluated) if:
- the manifest predates fingerprinting (missing either field — treated as
  untrusted, not silently accepted as "no fingerprint to check"), or
- the live `normalize_text()` fingerprint no longer matches the manifest's
  recorded one (someone edited `normalize_text()` without rebuilding the
  manifest via `eval/build_full130_leakage_manifest.py --write`).

`eval/test_validate_dev_set.py::test_changed_normalize_text_implementation_
causes_validation_failure` proves this end-to-end by monkeypatching
`normalize_text()` to a different implementation and confirming the
integrity check fails against a manifest built for the original.

## Coverage targets (see `validate_dev_set.py --dev-set eval/dev_set_v1.csv`)

- **Preferred**: 50+ cases total, at least 15 Arabic (`language == "ar"`).
- **Absolute minimum**: 30 cases, if data collection is genuinely constrained.
  `validate_dev_set.py` fails (non-zero exit) below 30 and warns (exit 0)
  between 30 and the preferred target.
- **Stratification**: cases should span multiple ISCO major groups, not
  cluster in one or two. Major group is *derived* from `gold_isco_code[:1]`
  at validation time (not a stored column — see migration note); when a
  caller supplies known full130 major groups, `validate_dev_set.py` flags
  any group with zero dev-set coverage as a warning, never a hard error.
- **Both languages present**: at least one `en` and one `ar` case is a hard
  requirement — a dev set that can't measure both languages can't be used
  to select a single K for a multilingual system.

## Independence requirements (hard leakage-safety gate)

`validate_dev_set.py` fails the file if:

1. Any `case_id` also appears in `test_set_smoke20.csv` or the full130
   leakage manifest's `case_ids`.
2. Any `respondent_text`, after `normalize_text()`, exact-string-matches a
   `test_set_smoke20.csv` `input_text` value, OR its sha256 hash matches an
   entry in the full130 leakage manifest's `normalized_text_sha256` list.
3. Gold codes are not independently derived — cases must be labelled or
   adjudicated by a person (or a cited authoritative source), never copied
   from `test_set_full130.csv` for a similar-sounding title.

**`eval/test_set_full130.csv` itself is never opened by `validate_dev_set.py`,
`eval/dev_sweep.py`, or `eval/pre_run_check.py`.** All three read only
`eval/configs/full130_leakage_manifest.json` (case_ids and
`normalize_text()`-then-sha256 hashes of `input_text` — no gold labels, no
raw text) for the full130 side of any check.

**The one authorised exception** is `eval/build_full130_leakage_manifest.py`
-- a standalone, manually-run (never automated/CI) script that reads
`eval/test_set_full130.csv` to (re)build the manifest, using
`eval/validate_dev_set.py`'s own `normalize_text()` so the manifest's
hashes are guaranteed built by the same function every consumer will
compare against. It defaults to a dry run (prints a summary, touches
nothing) — pass `--write` to actually update the checked-in manifest.

The runtime guard itself lives in **one shared module**,
`eval/full130_access_guard.py` (`guard_against_full130_access()`), used by
both `eval/validate_dev_set.py` (wraps `main()`'s entire execution path)
and `eval/pre_run_check.py` (wraps its whole checklist run) — not two
independently-maintained copies. It patches **five** independent
file-reading entry points, not just `builtins.open()`: `builtins.open`,
`io.open` (a *separate* name binding from `builtins.open`, so a
builtins-only patch would miss code that calls `io.open(...)` directly),
`pathlib.Path.open`, `pathlib.Path.read_text`, and `pathlib.Path.
read_bytes`. Any call whose path argument contains `"test_set_full130"`
raises `Full130AccessBlocked` immediately; unrelated paths (including the
leakage manifest) pass through unaffected. `eval/
test_full130_access_guard.py` tests all five vectors individually, proves
unrelated paths stay readable, and proves the real `validate_dev_set.py`/
`pre_run_check.py` execution paths never trigger it. `eval/
test_validate_dev_set.py` and `eval/test_pre_run_check.py` additionally
contain an AST-scan regression guard confirming neither module contains a
direct `open()`-family call referencing `eval/test_set_full130.csv` in its
own source, with a locked-in check that the manifest builder *does*.

## Confidentiality

`respondent_text` should be a free-text occupation description, not a full
survey transcript — avoid names, contact details, exact employer
identifiers, or other personally identifying information.
`annotator_or_adjudication_reference` identifies the labeller (an ID or
citation), never the respondent.

## Header validation (runs before, and independently of, row count)

`eval/validate_dev_set.py`'s `read_csv_header()` + `validate_csv_header()`
check the header row itself, separately from and *before* any row-count or
per-row check: the header must equal `CANONICAL_HEADER` (== `REQUIRED_
COLUMNS`, the 7 columns above, in exactly that order) with no missing,
duplicate, reordered, or extra columns. This matters specifically because
`validate_dev_set()`'s per-row logic only ever inspects column names once
`dev_rows` is non-empty (it short-circuits on "Dev set is empty" before
reaching any column check) — without a standalone header check, a
malformed header on a header-only (0 data row) file would be masked by the
generic "empty" error rather than reported as what it actually is. `eval/
validate_dev_set.py --dev-set ...` runs this check first and exits 1 with
an explicit header/schema error (never conflated with "Dev set is empty")
if it fails; `eval/pre_run_check.py`'s `dev_set_header_schema` checklist
item does the same, and short-circuits the remaining checklist (further
parsing is unreliable against an unknown header).

## CSV structure and edge cases

`eval/validate_dev_set.py`'s `validate_csv_structure()` checks the CSV
*grammar* itself, separately from field-value validation:
- **Malformed/unterminated quoting** and **ragged rows** (a data row with
  more or fewer fields than the header) are rejected as structural errors,
  checked before any field-level validation runs (a ragged row can't be
  reliably mapped to columns at all).
- **Multiline `respondent_text` cells are supported**, not rejected — a
  properly double-quoted CSV field may contain embedded newlines; Python's
  `csv` module (with `newline=""` on open, which every reader in this
  codebase already uses) parses this correctly as a single field, and
  `validate_csv_structure()` does not flag it.
- **UTF-8 Arabic text** round-trips correctly through every loader in this
  codebase (`load_csv_rows()`, `_load_dev_rows_as_test_set()`) — all file
  I/O is explicit `encoding="utf-8"`.
- **Leading/trailing whitespace** on `case_id`/`respondent_text` is
  stripped before comparison (duplicate detection, blank checks) but not
  silently accepted as distinct from its stripped form — a whitespace-only
  value is rejected as blank, and a value that differs from another row
  only by incidental leading/trailing whitespace is still caught as a
  duplicate.
- **`case_id` matching is case-sensitive** (documented policy, not an
  accident of implementation): `"Dev001"` and `"dev001"` are different
  case_ids for both internal-duplicate detection and external
  (smoke20/full130) collision checks. If you want IDs to be
  unambiguous, pick one casing convention and use it consistently — this
  validator will not catch a same-ID-different-case collision for you.

## Schema validity vs. readiness with real labels

**These are two different questions, and this schema only answers the
first one.** `eval/validate_dev_set.py --dev-set eval/dev_set_v1.csv`
passing means: the CSV is well-formed, every required column is present
and non-blank, every `gold_isco_code` is a real ISCO-08 unit-group code,
`dataset_split` is correct, and there is no detected overlap with
smoke20/full130. **It does not, and cannot, verify that the gold labels
are actually correct**, that `gold_label_source`/
`annotator_or_adjudication_reference` describe a real, auditable labelling
process, or that `respondent_text` is a genuine (not fabricated)
respondent utterance — those are provenance questions, answered by human
review and recorded in `eval/dev_set_v1_provenance.md`, not by this
validator. See `eval/dev_set_v1_readiness_report.md` for the current,
explicit split between "schema-valid" and "ready to select K from."

## Non-requirements (explicitly out of scope for this file)

- `industry_text` / `education_text` (used for SRE coherence scoring
  elsewhere in `run_eval.py`) are not required here — B2 dev-set selection
  only needs Candidate Recall@K and Top-1, neither of which touches SRE.
- No `gold_isic` / `gold_isced` columns — same reason.

## Migration note (schema reconciliation)

This file went through three revisions before settling on the schema
above:

1. **Original (dev_set_schema.md v0)**: 9 columns — `case_id`, `language`,
   `respondent_text`, `gold_isco_code`, `gold_label_source`,
   `coder_or_adjudicator`, `major_group`, `difficulty_level`, `notes`.
   `respondent_text` already matched what `eval/dev_sweep.py` and
   `eval/pre_run_check.py` expect.
2. **A later revision** introduced `job_title` + `job_description` (split
   from `respondent_text`) and `annotator_or_adjudication_reference` +
   `dataset_split` (new), while dropping `major_group`/`difficulty_level`/
   `notes`. This broke consistency: `eval/validate_dev_set.py` and
   `eval/dev_sweep.py` still expected the v0 column names, so a populated
   file in this intermediate schema would have failed schema validation or
   been silently mis-mapped.
3. **This reconciliation (canonical v1, current)**: reverted the
   title/description split back to a single `respondent_text` field
   (matching v0 and the existing pipeline's `input_text` convention
   throughout the rest of this project), kept the clearer
   `annotator_or_adjudication_reference` name and the useful `dataset_split`
   addition from revision 2, and dropped `major_group` (now derived from
   `gold_isco_code[:1]` instead of stored redundantly) and
   `difficulty_level`/`notes` (informational-only, not required by the B2
   pipeline).

**Final field mapping (old → canonical):**

| Old name(s) | Canonical name | Change |
|---|---|---|
| `respondent_text` (v0) / `job_title` + `job_description` (rev. 2) | `respondent_text` | Reverted to a single field; classifier input is unambiguous. |
| `coder_or_adjudicator` (v0) | `annotator_or_adjudication_reference` | Renamed for clarity/traceability (kept from rev. 2). |
| `major_group` (v0) | *(removed — derived from `gold_isco_code[:1]`)* | Eliminated a redundant column that could disagree with the gold code. |
| `difficulty_level` (v0) | *(removed)* | Informational-only, not required by K-selection. |
| `notes` (v0) | *(removed)* | Informational-only, not required by K-selection. |
| *(none)* | `dataset_split` | New (from rev. 2) — literal `dev_v1`, retained for provenance tracking. |

As of this reconciliation, `eval/validate_dev_set.py`'s `main()` also
stopped reading `eval/test_set_full130.csv` directly (a pre-existing
inconsistency versus `eval/pre_run_check.py`, which had always used the
manifest) — it now reads only `eval/configs/full130_leakage_manifest.json`,
via `--full130-manifest` (replacing the old `--full130` flag).

### Validator-strengthening pass (after the schema reconciliation above)

A later pass closed remaining validation gaps before real data collection
began, without changing the 7-column schema itself:
- Semantic ISCO-08 code validation (rejecting `0000`/`9999`-style
  syntactically-valid-but-nonexistent codes) — see "Semantic ISCO-08 code
  validation" above.
- `normalization_version`/`normalization_fingerprint` recorded in the
  leakage manifest and enforced fail-closed — see "Normalization-version /
  fingerprint enforcement" above.
- `eval/build_full130_leakage_manifest.py` formalised as the one,
  documented, manually-run path allowed to read `eval/test_set_full130.csv`
  — see "Independence requirements" above.
- CSV-grammar validation (`validate_csv_structure()`) — malformed quoting
  and ragged rows now rejected explicitly, multiline quoted fields
  confirmed supported, case-sensitive `case_id` policy documented.
- `eval/test_run_eval_b2.py` gained a direct regression test proving
  `respondent_text`/`input_text` reaches `ISCOClassifier.classify()`
  completely unchanged, and that the dev-set mapping is identical to the
  one B0/B1 already use for smoke20/full130 — see "Classifier-input field"
  above. No production mapping code was changed; the test confirmed no
  inconsistency existed.
