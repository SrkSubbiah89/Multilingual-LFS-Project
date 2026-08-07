# Real LFS Validation Dataset Card — Template

**Status of this document: TEMPLATE.** No real Labour Force Survey respondent
data has been added to this repository as part of this work, and none should
ever be committed to it. This template exists so that if/when a real,
permissioned dataset becomes available, its governance can be documented
completely and consistently — and so that `eval/validate_real_lfs_governance.py`
can mechanically block any evaluation run from being labelled
`approved_real_lfs_validation` until every field below is filled in.

Copy this file, fill in every field marked **REQUIRED** (or explicitly write
"N/A" with a reason if a field genuinely does not apply — an empty/blank
field always blocks), and save it as a `DatasetCard` JSON matching
`eval/dataset_card_schema.py`'s schema. See
`eval/fixtures/complete_approved_real_lfs_dataset_card_example.json` for a
fully populated, schema-complete TEST FIXTURE that shows every field filled
in correctly (it is not real governance documentation — see that file's own
`_notice` field), and
`eval/fixtures/synthetic_lfs_intake_package/dataset_card.json` for a
SYNTHETIC example that is deliberately rejected by the validator, to show
what "not real data" looks like from the tooling's point of view.

**IMPORTANT**: do not write "SYNTHETIC", "fake", "dummy", "not real", or
similar wording into any field of a REAL submission — the validator's
content safeguard (`_find_synthetic_marker()`) will reject a card containing
any such marker, by design, regardless of how complete the rest of the card
is.

---

## 1. Dataset identity

- **Source** (REQUIRED): [Who/what collected this dataset? e.g. "UAE MOHRE
  Labour Force Survey, Q3 2023 wave" — this field is the primary citation
  trail for any manuscript claim built on this data.]
- **Dataset ID** (REQUIRED): [A stable identifier for this dataset.]
- **Data owner / custodian** (REQUIRED): [The organisation/individual with
  legal custody of the data — may differ from Source.]
- **Region** (REQUIRED): [Country/region the data was collected in.]
- **Survey programme name** (REQUIRED): [The name of the survey programme,
  e.g. "UAE Labour Force Survey".]
- **Wave** (REQUIRED): [Survey wave/round or collection-period identifier.]
- **Collection start date** (REQUIRED): [ISO date YYYY-MM-DD]
- **Collection end date** (REQUIRED): [ISO date YYYY-MM-DD]
- **Languages** (REQUIRED): [Languages present in the raw respondent text,
  e.g. en, ar, ar-gulf, ur, hi, tl.]
- **Dataset version** (REQUIRED): [Version/release identifier for this
  specific extract.]
- **Dataset hash** (REQUIRED): [A hash (e.g. sha256) YOU compute over your
  own local dataset file, recorded here for immutability/provenance
  tracking. Never upload the file itself.]

## 2. Governance

- **Data-sharing agreement reference** (REQUIRED): [Reference/identifier
  for the agreement permitting this use.]
- **Ethics/IRB approval, exemption, or governance-review reference**
  (REQUIRED): [Approval number/reference.]
- **Consent obtained** (REQUIRED, must be `true`): [Was informed consent
  obtained from respondents for this specific use?]
- **Lawful basis description** (REQUIRED): [Free-text description of the
  consent process or other lawful basis.]
- **Permitted research purpose** (REQUIRED): [The specific purpose(s) this
  data is permitted to be used for.]
- **Retention period** (REQUIRED): [How long this data may be retained.]
- **Access-control description** (REQUIRED): [Who may access this data and
  how access is restricted/audited.]
- **De-identification method** (REQUIRED): [How PII was removed/
  pseudonymised before this dataset was used.]
- **De-identification status** (REQUIRED): [Current status, e.g. "fully
  de-identified", "pseudonymised with re-identification key held by
  custodian".]
- **Prohibited disclosure rules** (REQUIRED): [Any outputs/disclosures
  prohibited under the governing agreement, e.g. minimum cell sizes.]

## 3. Classification labels

- **Legacy classification-standard summary** (REQUIRED): [e.g. "ISCO-08" —
  kept for backward compatibility; the four fields below are what's
  actually checked for approved real-LFS validation.]
- **ISCO-08 version** (REQUIRED)
- **ISIC Rev.4 version** (REQUIRED)
- **ISCED 2011 version** (REQUIRED)
- **ISCED-F 2013 version** (REQUIRED)
- **Label source** (REQUIRED): [How gold labels were produced, e.g. "expert
  human coder", "administrative record match".]
- **Labelling process** (REQUIRED): [e.g. "double-coded with adjudication".]
- **Labeler type** (REQUIRED): [e.g. "professional statistical coder".]
- **Coder qualifications** (REQUIRED): [Training/qualifications of coders —
  see `ANNOTATION_AND_ADJUDICATION_GUIDE.md` for the required standard.]
- **Double-coded** (REQUIRED, must be `true`): every record must be
  independently coded twice. **This is a hard requirement, not a
  warning** — a single-coded dataset cannot be labelled
  `approved_real_lfs_validation`.
- **Adjudication process** (REQUIRED when double-coded, which is always):
  [How coder disagreements were resolved — see
  `ANNOTATION_AND_ADJUDICATION_GUIDE.md`.]
- **Inter-annotator agreement** (nullable — not required, but strongly
  recommended before citing label quality in the manuscript; leave `null`
  if not yet measured, never estimate it).

## 4. Evaluation discipline

- **Train/development/test split manifest reference** (REQUIRED)
- **Split hash** (REQUIRED)
- **Frozen held-out test status** (REQUIRED): [Confirmation the held-out
  test split is frozen and was never used for parameter selection.]
- **Parameter-selection dataset reference** (REQUIRED): [The SEPARATE
  dataset/split used for parameter selection — must differ from the
  frozen test split.]
- **Test-set access restriction** (REQUIRED): [Who may access the frozen
  test set, to prevent leakage.]
- **Sample counts by language and classification task** (REQUIRED): [e.g.
  `{"en": {"isco": 50, "isic": 50, "isced": 50}, "ar": {...}}`.]
- **Missing-data policy** (REQUIRED): [How missing/incomplete responses
  are handled in evaluation.]
- **Exclusion criteria** (REQUIRED): [Criteria used to exclude records.]

## 5. Privacy

- **Direct identifiers removed — confirmed** (REQUIRED, must be `true`)
- **Quasi-identifier risk note** (REQUIRED): [Assessment of re-
  identification risk from quasi-identifiers.]
- **Free-text redaction method** (REQUIRED): [How free-text answers were
  screened for incidental PII.]
- **Secure storage location description** (REQUIRED): [A DESCRIPTION —
  never a literal path into this repository — of where the real dataset is
  stored, e.g. "encrypted institutional drive, access restricted to named
  PI". A value that resolves to a path inside this Git repository is
  rejected automatically.]
- **Raw records excluded from Git — confirmed** (REQUIRED, must be `true`)

---

## How this card is enforced

`eval/validate_real_lfs_governance.py` reads a filled-in copy of this card
(as JSON, matching `eval/dataset_card_schema.py::DatasetCard`) and BLOCKS
`eval.manifest.build_manifest()` / `eval.ablation_runner.run_config()` from
producing a manifest labelled `dataset_label="approved_real_lfs_validation"`
unless:

1. Every field above marked REQUIRED is non-blank.
2. `consent_obtained`, `double_coded`,
   `direct_identifiers_removed_confirmed`, and
   `raw_records_excluded_from_git_confirmed` are all explicitly `true`.
3. No field contains a synthetic/placeholder marker word (SYNTHETIC, FAKE,
   DUMMY, "not real", ...) — a card that reads as non-real data is rejected
   regardless of field completeness.
4. No path-shaped field (`secure_storage_location_description`,
   `split_manifest_ref`, `parameter_selection_dataset_ref`) resolves to a
   location inside this Git repository.
5. The associated run manifest has a non-empty `split_name`.

A dataset lacking real governance information can still be used and
evaluated — just not labelled `approved_real_lfs_validation`. Use
`dataset_label="synthetic_or_operationally_realistic"` instead, which
carries no governance requirement. An attempt that fails these checks is
recorded with the label `invalid_incomplete_governance` rather than
silently discarded — see `REAL_LFS_DATA_INTAKE_CHECKLIST.md`.

## Exact command to validate a filled-in card

```bash
python eval/validate_real_lfs_governance.py \
  --manifest path/to/manifest.jsonl \
  --dataset-card path/to/your_dataset_card.json
```

Exits 0 and prints `PASS` if every check above is satisfied; exits 1 and
lists every failing field (by name only, never by value) otherwise.
