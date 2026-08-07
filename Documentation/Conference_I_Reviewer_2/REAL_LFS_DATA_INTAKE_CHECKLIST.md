# Real LFS Data Intake Checklist

Step-by-step checklist for a data custodian (or a researcher working with
one) to move from "we have access to a real, permissioned Labour Force
Survey dataset" to "this codebase can validate against it and the
manuscript can safely cite the result." Conference I Reviewer #2 response,
Step 3 real-LFS-intake hardening pass.

**No real respondent data is ever added to this Git repository at any step
of this checklist.** Everything committed is metadata ABOUT the data
(governance documentation, counts, hashes), never the data itself.

## Before you start

- [ ] Read `REAL_LFS_VALIDATION_DATASET_CARD_TEMPLATE.md` in full.
- [ ] Read `ANNOTATION_AND_ADJUDICATION_GUIDE.md` if you are also producing
      new gold labels for this dataset (skip if labels already exist and
      meet the same standard).
- [ ] Confirm you actually have: a real dataset, a data-sharing agreement/
      approval, an ethics/governance review reference, and independently
      double-coded gold labels (or the ability to produce them per the
      annotation guide). If any of these is missing, this dataset cannot
      be labelled `approved_real_lfs_validation` yet — that's fine, use
      `synthetic_or_operationally_realistic` in the meantime.

## Step 1 — Store the real data outside Git

- [ ] Store the actual dataset file(s) in a location OUTSIDE this
      repository (e.g. an encrypted institutional drive) — never inside
      any directory tracked by this repo's `.gitignore`-exempted paths.
- [ ] If you want a local scratch copy for your own convenience, put it
      under `eval/local_catalogues/` (already gitignored — see
      `.gitignore`) or another gitignored location, and confirm with
      `git status` that it does NOT appear as an untracked/trackable file.
- [ ] Compute a hash (sha256 recommended) of your dataset file and record
      it as `dataset_hash` in the dataset card — this lets you later prove
      which exact file version a result came from, without ever putting
      the file itself under version control.

## Step 2 — Fill in the dataset card

- [ ] Copy `REAL_LFS_VALIDATION_DATASET_CARD_TEMPLATE.md`'s field list (or
      `eval/fixtures/complete_approved_real_lfs_dataset_card_example.json`
      as a structural starting point — note its `_notice`: it is a TEST
      FIXTURE, not real data, replace every value) into a new JSON file
      matching `eval/dataset_card_schema.py::DatasetCard`.
- [ ] Fill in every field marked REQUIRED in the template. Do not write
      "SYNTHETIC", "fake", "dummy", "not real", or similar wording into
      any field — the validator rejects a card containing such markers
      by design, regardless of completeness.
- [ ] Save the completed card OUTSIDE this repository, or under a
      gitignored path (see Step 1) if you want a local copy for repeated
      validation runs.

## Step 3 — Produce/verify gold labels

- [ ] If gold labels don't already exist: follow
      `ANNOTATION_AND_ADJUDICATION_GUIDE.md` in full (independent double-
      coding, adjudication for disagreements, coders never shown the
      system's own prediction).
- [ ] If gold labels already exist from a prior process: confirm they meet
      the same standard (double-coded, adjudicated, coders qualified) —
      document this in the dataset card's `label_source`/`labeler_type`/
      `double_coded`/`adjudication_process` fields. A dataset with
      single-coded or unverified labels cannot be labelled
      `approved_real_lfs_validation`.
- [ ] Compute inter-annotator agreement (e.g. Cohen's kappa) across the
      double-coded records and record it in `inter_annotator_agreement`.

## Step 4 — Define and freeze the evaluation split

- [ ] Split the dataset into a development set (for parameter selection)
      and a held-out test set (frozen, confirmation-only) — following the
      same discipline `eval/validate_dev_set.py` / `EVALUATION_PROTOCOL.md`
      already establish for the dev_v1/test_set_full130 split.
- [ ] Confirm zero case-ID or near-duplicate-text overlap between the two
      splits.
- [ ] Record the split manifest reference, split hash, frozen-status
      confirmation, and parameter-selection-dataset reference in the
      dataset card. Once frozen, the held-out split must never be touched
      again for parameter selection — any future access should go through
      the `test_set_access_restriction` control you document.

## Step 5 — Validate the dataset card

Run:

```bash
python eval/validate_real_lfs_governance.py \
  --manifest path/to/manifest.jsonl \
  --dataset-card path/to/your_dataset_card.json
```

- [ ] Fix every reported error (each names a missing/invalid FIELD, never
      any respondent data) and re-run until you see `PASS`.
- [ ] If you don't have a manifest yet, you can still validate the card in
      isolation via a minimal manifest dict — see
      `eval/test_validate_real_lfs_governance.py` for examples, or build
      one first with `eval/manifest.py`'s `build_manifest()` /
      `eval/ablation_runner.py run --dataset-label approved_real_lfs_validation`.

## Step 6 — Run the evaluation

```bash
python eval/ablation_runner.py run --config hierarchical_with_rerank \
  --test-set <your held-out test set, outside this repo> \
  --reranker-model anthropic/claude-3-5-sonnet-20241022 \
  --split heldout \
  --dataset-label approved_real_lfs_validation \
  --dataset-card path/to/your_dataset_card.json
```

- [ ] If governance validation fails at this step, the run does NOT
      execute (no compute spent), and a manifest labelled
      `invalid_incomplete_governance` is written recording exactly which
      checks failed — go back to Step 2/3/4 and fix them.
- [ ] On success, the resulting manifest carries
      `dataset_label="approved_real_lfs_validation"` — this is now
      real, citable evidence.

## Step 7 — Check generated reports before citing

- [ ] Confirm the manifest/report you intend to cite actually carries
      `dataset_label="approved_real_lfs_validation"` — check
      `eval/ablation_runner.py emit-table`'s **Data label** column, or the
      manifest JSON's `dataset_label` field directly. Never cite a row
      labelled `synthetic_or_operationally_realistic` or
      `invalid_incomplete_governance` as real-data validation.

## Manuscript wording: what's safe

| Wording | Safe when... |
|---|---|
| "validated on real LFS data" | ONLY when the underlying manifest's `dataset_label == approved_real_lfs_validation`. |
| "real-world validation" | Same condition. |
| "evaluated on a synthetic/operationally realistic test set" | Any `synthetic_or_operationally_realistic` result — always safe, and required framing until Step 1–7 above are complete. |

`eval/validate_real_lfs_governance.py`'s `check_manuscript_wording()`
function programmatically flags the banned phrases above when the
associated label isn't `approved_real_lfs_validation` — use it (or this
table) as the final check before writing evaluation numbers into the paper.
