# WISCO Gold ISCO-08 Label Audit (2026-08-10)

**Trigger:** direct operator request to double-check whether the WISCO
benchmark's `gold_isco_4digit` labels are actually correct, for both the
flat and hierarchical arms of the official Tier-1 evaluation
(`OFFICIAL_WISCO_TIER1_CONTROLLED_RESULTS.md`).

**Bottom line: the gold labels are sound.** One narrow, genuine data-quality
issue was found (2 of 20,760 records: an English-language text collision
with a disagreeing gold code) and has been fixed at the source. It has
**zero effect** on the already-published Task 36/37/37.1 headline numbers
— the exact CSV those tasks measured is proven byte-identical before and
after the fix (see §5).

## 1. What was checked

| # | Check | Method | Result |
|---|---|---|---|
| 1 | Are gold codes valid, official ISCO-08 unit-group codes? | Set-diff WISCO v2's 436 distinct `gold_code` values against `eval/local_catalogues/ilo_isco08_2021/normalized/isco08_official_normalized.csv`'s 436 unit codes | **Exact match.** 0 codes only in WISCO, 0 official codes missing from WISCO. |
| 2 | Is the ISCO0801-04 hierarchy internally consistent in the parser? | Read `backend/evaluation/wisco/parse_wisco.py` — every row's 4-digit code is validated to start with its own declared 3/2/1-digit parents before being accepted; inconsistent rows are quarantined (8 found, independently explained — genuine Armed-Forces mis-tagging in WISCO's own source, not a parser bug) | Parser already fails closed on this class of error; no defect found. |
| 3 | Does `heldout_run_eval_format.csv` (what Task 36 actually fed to the evaluator) match `records.json`'s gold codes/text for all 18,747 rows? | Direct row-by-row comparison, `case_id`/`gold_code`/`input_text`/`split` | **0 mismatches, 0 missing rows, 0 split leakage.** |
| 4 | Do any two records share byte-identical (language, normalized text) but disagree on gold code? | Grouped all 20,760 records by `(language, normalize_text(input_text))`, checked for >1 distinct `gold_code` per group | **1 conflict found** (2 records) — see §2. |
| 5 | Are `MAPPINGS` sheet duplicate keys ("last wins" in the parser) safe? | Re-read the raw workbook directly; compared `ISCO0804` across all 61 duplicate-key groups | All 61 duplicate groups agree on `ISCO0804` — "last wins" changed nothing. |
| 6 | Manual semantic spot-check | 25 random English records: WISCO's specific occupation title vs. the official unit-group's broader title | All 25 plausible on inspection (e.g. "Farmer - hop" → 6112 Tree and Shrub Crop Growers; "Plastic surgeon" → 2212 Specialist Medical Practitioners; "Chef cook" → 3434 Chefs). |

## 2. The one genuine finding

Two **different** WISCO source occupation keys carry the identical English
string `"Veterinary assistant"` but disagree on gold code:

| `benchmark_id` | WISCO key | gold_code | Official title | WISCO's own label |
|---|---|---|---|---|
| `WISCO-3240000400018-en` | `3240000400018` | `3240` | Veterinary Technicians and Assistants | "Veterinary assistant" |
| `WISCO-5164140000000-en` | `5164140000000` | `5164` | Pet Groomers and Animal Care Workers | "Veterinary helper" |

Both keys' **other** language variants (Arabic, Urdu, Hindi, and — for the
second key — Tagalog) are genuinely distinct strings between the two keys
and are **not** ambiguous; only the English `en_US` locale column happened
to collide. Both records land in the `heldout` split (the v2 group-split
grouping already correctly keeps them in the same split — this was one of
the "13 within-split exact-duplicate-text groups" the existing leakage
audit already reported as a text duplicate, `ok: true`; what the leakage
audit never checked is whether a duplicate-text group agrees on gold code).
Checked exhaustively: this is the **only** such conflict among all 20,760
records / 20,747 distinct (language, text) pairs across all 5 languages.

Because the classifier pipeline is deterministic, a given input string
always produces the same prediction — confirmed directly in Task 36's raw
output:

| Method | `WISCO-3240000400018-en` (gold 3240) | `WISCO-5164140000000-en` (gold 5164) |
|---|---|---|
| Flat | pred `3240` → **correct** | pred `3240` → **incorrect** |
| Hierarchical | pred `3240` → **correct** | pred `3240` → **incorrect** |

Both arms predicted the same code both times (as expected for identical
input text), so at most one of these two rows could ever be scored
correct, regardless of classifier quality. This is symmetric across flat
and hierarchical — it does not favor either method in the paired
comparison — and affects exactly 1 case out of 18,747 (≈0.0053 percentage
points), far below the 2-decimal resolution of every published number in
`OFFICIAL_WISCO_TIER1_CONTROLLED_RESULTS.md`.

## 3. Fix applied

`eval/build_wisco_isco_benchmark_v2_group_split.py` gained a new
post-processing pass, `detect_gold_code_text_conflicts()`: after records
are built (same text-normalization logic already used by
`audit_wisco_benchmark_leakage.py`), any record whose exact
(language, text) pair is shared with another record carrying a
**different** `gold_code` is marked `ambiguity_flag=True` with a populated
`exclusion_reason` (required together by
`eval/validate_controlled_benchmark.py`'s existing validator). `gold_code`
itself is **left exactly as WISCO published it** — never blanked or
guessed — consistent with this project's "never fabricate, never silently
force a gold code" discipline already encoded in the schema.

Only the two specific colliding English records are flagged; the other
language variants of both keys, and every other duplicate-text group in
the dataset (12 others, all internally consistent on gold code), are
untouched.

`eval/local_benchmarks/wisco_isco08_v1/` (the frozen leakage-audit record)
was **not** touched, per that directory's own standing rule (see the v2
script's module docstring) — this fix lives only in the v2 group-split
builder.

## 4. Regeneration

Re-ran `python eval/build_wisco_isco_benchmark_v2_group_split.py --out-root eval/local_benchmarks/wisco_isco08_v2_group_split`:

```text
Wrote 20760 benchmark records ({'dev': 2013, 'heldout': 18747})
dataset_hash=2bbd5fa1e2815ce01757ae2fbfb927dc4e759fabb4a28ae49b9ee0b719a662e2
group_meta={'fixed_seed': 42, 'n_source_keys': 4232, 'n_groups': 4220,
            'n_merges_from_duplicate_text': 12,
            'n_groups_with_more_than_one_member': 12,
            'n_records_flagged_gold_code_text_conflict': 2}
validation: ok=True errors=0
leakage audit: leakage_found=False overall_ok=True
```

Record/dev/heldout counts, grouping, and leakage-audit outcome are
unchanged from before the fix — only 2 records' `ambiguity_flag`/
`exclusion_reason` fields changed, which changes `records.json`'s overall
content hash (dataset_hash `a3b3c1a31abd24369643d265c17d13dea8a5bcc9dbf1d6582b13153011dd287c`
→ `2bbd5fa1e2815ce01757ae2fbfb927dc4e759fabb4a28ae49b9ee0b719a662e2`).

## 5. Proof the already-published Task 36 numbers are unaffected

`eval/export_benchmark_to_run_eval_csv.py` (the script that produced the
actual evaluator input) only ever reads `benchmark_id`/`input_text`/
`language`/`gold_code`/`split`/`task` — none of which changed for any
record. Re-exporting the heldout split from the regenerated `records.json`
and diffing against the file already on disk (the one Task 36's evaluator
actually consumed) confirms this directly:

```text
$ python eval/export_benchmark_to_run_eval_csv.py --records .../records.json --split heldout --out /tmp/heldout_reexport_check.csv
$ diff /tmp/heldout_reexport_check.csv eval/local_benchmarks/wisco_isco08_v2_group_split/heldout_run_eval_format.csv
(no output -- byte-identical)
$ sha256sum both files
41c20fcc9eeec42358bdd90f211f6a344a47394b4b76fdf02c4ed5305cd1931c  (both)
```

This SHA-256 is the exact **"Heldout export"** value already recorded in
`OFFICIAL_WISCO_TIER1_CONTROLLED_RESULTS.md`'s "Raw-file identities"
table. **The flat/hierarchical accuracy numbers, Wilson intervals, and
McNemar result already published for Task 36/37/37.1 remain exactly
correct and exactly reproducible from the identical frozen input** — this
fix only adds data-quality metadata to `records.json` (used for auditing
and any *future* run), it does not and cannot retroactively change what
was measured.

## 6. Operator decisions (resolved 2026-08-10)

- **Canonical doc:** `OFFICIAL_WISCO_TIER1_CONTROLLED_RESULTS.md`'s
  "WISCO v2 split/provenance" section has been updated to cite the new
  dataset hash (`2bbd5fa1...`), with the old hash (`a3b3c1a3...`) and a
  pointer to this document kept inline for traceability. No headline
  number, raw-file hash, or evidence-chain row in that document was
  touched.
- **Git handling:** committed on a new branch off the current one, pushed,
  no PR — see the final report / commit for the exact branch and SHA.

## 7. Files touched

- `eval/build_wisco_isco_benchmark_v2_group_split.py` — added
  `detect_gold_code_text_conflicts()` and wired it into `build_records()`.
- `eval/test_build_wisco_isco_benchmark_v2_group_split.py` (new) — 9 tests.
- `eval/local_benchmarks/wisco_isco08_v2_group_split/*` — regenerated
  (git-ignored, not part of the repository).
- This document.

`eval/local_benchmarks/wisco_isco08_v1/` and every Task 36-41 evidence
file were not touched. Full suite: `2294 passed, 1 deselected, 1 warning`
(2285 baseline + 9 new), zero regressions.
