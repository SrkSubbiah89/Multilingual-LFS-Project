# Synthetic LFS Intake Package — Example Only

**SYNTHETIC EXAMPLE ONLY; NOT REAL LFS DATA.**

This directory demonstrates the *shape* of a complete real-LFS-validation
intake package — every file here is fully synthetic, contains no personal
information, and copies no respondent text from any real survey. It exists
so a data custodian (or a developer) can see exactly what a complete
submission looks like before assembling a real one.

Every file in this directory is deliberately marked with the word
"SYNTHETIC" in multiple fields, specifically so
`eval/validate_real_lfs_governance.py`'s content-aware safeguard
(`_find_synthetic_marker()`) will always reject `dataset_card.json` here if
someone tries to label a run `approved_real_lfs_validation` using it — see
`eval/test_validate_real_lfs_governance.py::
test_synthetic_intake_package_cannot_be_relabelled_as_approved_real_lfs`.

## Files

| File | What it demonstrates |
|---|---|
| `dataset_card.json` | The full `DatasetCard` schema (`eval/dataset_card_schema.py`) — identity, governance, classification-label provenance, split discipline, privacy. |
| `governance_manifest.json` | A governance sign-off record (approval status, reviewer, review date) — a human-facing summary distinct from the field-by-field dataset card. |
| `data_dictionary.json` | Column-by-column description of what a real intake file would contain — field names, types, descriptions, and **synthetic example values only**, never real respondent text. |
| `label_schema.json` | The four classification standards (ISCO-08, ISIC Rev.4, ISCED 2011, ISCED-F 2013) this system labels against, with their versions and a few example codes. |
| `language_metadata.json` | Supported languages/dialects and script notes. |
| `deidentification_declaration.json` | A focused de-identification status declaration. |
| `split_manifest.json` | Train/development/test split summary — counts and hashes only, never records. |
| `annotation_adjudication_record_template.json` | The template structure for recording one annotation + adjudication decision — fields only, no filled example (filling it in with synthetic content would misleadingly resemble a real coding decision). |

## Validating this package

```bash
python eval/validate_real_lfs_governance.py \
  --manifest <a manifest.jsonl with dataset_label=approved_real_lfs_validation> \
  --dataset-card eval/fixtures/synthetic_lfs_intake_package/dataset_card.json
```

This will **FAIL** — intentionally. See `REAL_LFS_DATA_INTAKE_CHECKLIST.md`
for what a real custodian must supply to pass.
