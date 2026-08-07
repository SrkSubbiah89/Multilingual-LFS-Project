# B2 pre-run checklist

Read this before running `eval/dev_sweep.py` for real. B2's whole point is a
clean, leakage-safe, reproducible comparison against the frozen B1 result
(54/130 on `eval/test_set_full130.csv`) -- every item below exists because
skipping it would silently break that comparability.

## Prerequisites

1. **A fresh, validated dev set** at `eval/dev_set_v1.csv`, following
   [`eval/dev_set_schema.md`](dev_set_schema.md). Not `eval/
   test_set_smoke20.csv` (reused for earlier experiments) and never
   `eval/test_set_full130.csv` (the confirmation set -- see "full130 usage
   rule" below).
2. **`eval/configs/b1_frozen.json`** present and internally consistent with
   the current codebase and the currently-installed Ollama model. This repo
   ships one already, built from the real 54/130 confirmation run -- you
   normally don't need to touch it, but re-run `eval/pre_run_check.py` if
   you've edited `backend/agents/isco_classifier.py`,
   `backend/rag/hierarchical_store.py`, or re-pulled the pinned Ollama
   model, since any of those can legitimately fail the codebase checks.
3. **Ollama running locally** with the pinned model (`ollama/llama3.2:latest`
   as of this file's `_verified_on` date -- check
   `eval/configs/b1_frozen.json`'s `reranker_model` field, not this
   document, for the current source of truth) actually pulled.
4. **A clean git working tree**, or an explicit decision to override (see
   below).

## Required manual confirmation if beam remains inferred

`eval/configs/b1_frozen.json`'s `beam_evidence.status` is currently
`"inferred"` -- beam=3 is corroborated by the source CSV's filename and by
`full130_3b_analysis.txt`'s paired-comparison note, but neither is an
independently recorded value (a saved CLI invocation, config file, or
reproducibility manifest), so it is **not** treated as confirmed. Both
`eval/pre_run_check.py` and `eval/dev_sweep.py` refuse to run by default
while this is true.

**If you have independent access to the original run history** (shell
history, a notebook, a chat log, anything that directly states `--beam 3`
was passed for the run that produced
`eval/results/raw_runs/20260805T055741Z_full130_leafvote_beam3_llama3b_pooled.csv`)
and can confirm beam=3:
- Pass `--confirm-inferred-beam 3` to both `eval/pre_run_check.py` and
  `eval/dev_sweep.py`. The exact value is required -- any other number is a
  hard failure, not a silent correction.
- Consider updating `eval/configs/b1_frozen.json`'s `beam_evidence.status`
  to `"confirmed"` with the source you found, so future runs don't need the
  override at all.

**If you cannot confirm it**, do not pass the override. Treat any B2 result
that required `--confirm-inferred-beam` as conditional on that assumption,
and say so if these numbers reach a paper draft.

## Step 1: validate the dev set

```
python eval/validate_dev_set.py --dev-set eval/dev_set_v1.csv
```

Must exit 0 (PASS; warnings are fine, errors are not) before continuing.
This checks:
- **Header validity FIRST, independently of row count** -- the header must
  exactly match the canonical 7-column order (no missing/duplicate/
  reordered/extra columns). A malformed header-only file is reported as an
  explicit header/schema error, never conflated with the generic "Dev set
  is empty" message a merely-unpopulated (but correctly-headed) file gets.
- CSV structure (malformed quoting, ragged rows -- rejected before any
  field-level check runs).
- Schema/field validity, including that `gold_isco_code` is not just
  4 digits but exists in the **classifier-supported** ISCO catalogue
  (441 codes from `backend/rag/load_full_isco.py`, read as text --
  `0000`/`9999`-style codes are rejected even though they're syntactically
  valid). This is "codes the classifier can predict," not an independently
  verified statement of official ILO ISCO-08 coverage -- see
  `eval/dev_set_schema.md`'s follow-up audit note on the 441-vs-436
  discrepancy. If this catalogue can't be loaded at all, the run is FATAL
  (exit 1) rather than silently downgrading to format-only validation.
- Leakage-safety against `eval/test_set_smoke20.csv` (read directly -- it
  is not held-out) and `eval/test_set_full130.csv` (checked via
  `eval/configs/full130_leakage_manifest.json` **only** -- this script does
  NOT read `eval/test_set_full130.csv` directly, see "full130 usage rule"
  below; that was true even in an earlier revision of this document that
  said otherwise).
- The leakage manifest's `normalization_fingerprint` matches this script's
  live `normalize_text()` -- fails closed (before any leakage comparison
  runs) if not, since a mismatched fingerprint means the manifest's hashes
  aren't trustworthy to compare against.

## Step 2: run the pre-run gate

```
python eval/pre_run_check.py \
    --baseline-config eval/configs/b1_frozen.json \
    --dev-set eval/dev_set_v1.csv
```

Add `--confirm-inferred-beam 3` if you've resolved the beam-provenance gap
above, and/or `--allow-dirty-tree` if you have an intentional reason to run
against uncommitted changes (both are recorded prominently in the checklist
output and, when you proceed to `dev_sweep.py`, in its report and metadata
JSONL too).

Must print `OVERALL: PASS` and exit 0. Its checklist includes, among the
items covered in Step 1 above (surfaced here as individual PASS/FAIL lines
-- `dev_set_header_schema`, `dev_set_csv_structure`, `isco_catalogue_loaded`
(**hard failure** if the classifier-supported ISCO catalogue can't be
loaded -- see `eval/dev_set_schema.md`'s "Semantic ISCO-08 code validation"
section for why "classifier-supported" and "official ILO ISCO-08" are not
the same claim), `full130_manifest_normalization_integrity`,
`dev_set_validation_vs_smoke20`, `dev_set_no_overlap_with_full130` -- plus
the baseline/beam/git-tree gates from `eval/configs/b1_frozen.json`). This
script never opens `eval/test_set_full130.csv` -- dev-set overlap against
it is checked via `eval/configs/full130_leakage_manifest.json` (case_ids
and text hashes only, no labels), and the whole check sequence runs inside
`eval/full130_access_guard.py`'s shared runtime guard, which patches FIVE
independent file-reading entry points (`builtins.open`, `io.open`,
`pathlib.Path.open`/`.read_text()`/`.read_bytes()` -- not just
`builtins.open`, since `io.open` is a separate name binding that a
builtins-only patch would miss) and raises immediately if anything ever
tries to open a path matching `test_set_full130`. `eval/validate_dev_set.py`
wraps its own `main()` in the exact same shared guard. Three independent
regression-test layers prove this: `eval/test_full130_access_guard.py`
(all five vectors individually, plus real-execution-path non-triggering),
and an AST source scan in `eval/test_validate_dev_set.py`/`eval/
test_pre_run_check.py` finding no direct `open()`-family call referencing
that filename in either module's own source.

## Step 3: run the dev-set K-sweep (only after Step 2 PASSes)

```
python eval/dev_sweep.py \
    --dev-set eval/dev_set_v1.csv \
    --baseline-config eval/configs/b1_frozen.json \
    --k-values 5,8,10,15,20 \
    [--confirm-inferred-beam 3] [--allow-dirty-tree]
```

`dev_sweep.py` re-runs the same baseline/beam/git-tree gates itself (it does
not trust that Step 2 was actually run) -- Step 2 exists so you see the full
picture before spending wall-clock time, not as the only enforcement point.
Produces a Markdown report (`eval/results/dev_sweep/*_devsweep_report.md`)
recording the chosen K, the operational-eligibility rationale, the seed and
execution-order determinism record, and full provenance (git commit,
Ollama/Qdrant versions, environment).

## full130 usage rule

**`eval/test_set_full130.csv` must not be used until K is selected and
frozen from Step 3's report.** Neither `eval/validate_dev_set.py` (Step 1)
nor `eval/pre_run_check.py` (Step 2) ever opens it directly -- both check
dev-set independence via `eval/configs/full130_leakage_manifest.json`
only (case_ids + `normalize_text()`-then-sha256 hashes, no labels, no raw
text). The **one** script in this repo authorised to read
`eval/test_set_full130.csv` directly is `eval/
build_full130_leakage_manifest.py`, a manual, non-automated script that
(re)builds that manifest -- see its own module docstring and
`eval/dev_set_schema.md`'s "Independence requirements" section. It must
never be passed to `eval/run_eval.py` until you have a single,
pre-specified K chosen from the dev-sweep report. Once K is frozen,
`eval/run_eval.py` is run against `eval/test_set_full130.csv` **exactly
once**, for confirmation only -- not for further tuning.
