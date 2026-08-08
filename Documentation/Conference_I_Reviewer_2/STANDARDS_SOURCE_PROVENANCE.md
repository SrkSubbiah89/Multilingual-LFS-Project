# Standards Source Provenance

Records the authoritative sources identified for the four classification
standards this system implements, and the exact research trail behind
every `official_count_unverified` value in `eval/standards_reference.yaml`.
This document exists so that any number cited from that file can be traced
back to where it came from and how confident that provenance is —
supporting Reviewer #2 comment 8 (references and technical claims need
traceable evidence) as well as comment 5 (ISIC coverage disclosure).

**Read this alongside `COVERAGE_AUDIT_GUIDE.md`.** This document is about
*where the official counts come from*; that one is about *how to turn a
real catalogue file into a verified, citable coverage percentage*. Nothing
in this document is, by itself, sufficient to support a coverage
percentage in the manuscript — see "Two tiers of official count" below.

## Two tiers of official count

| Tier | Where it lives | How it's produced | Used for `coverage_percentage`? |
|---|---|---|---|
| **Unverified** | `eval/standards_reference.yaml`, per level, as `official_count_unverified` | A real, cited number transcribed from the `source_url` below, by a human or an AI assistant reading that source in a single research session | **Never** |
| **Verified** | `eval/verified_catalogue_counts.yaml` (machine-generated) | `eval/catalogue_importer.py` validates a real, user-supplied catalogue CSV (code format, hierarchy level, uniqueness, parent-child consistency) with **zero** issues | **Only this tier** |

This split exists because a number read off a webpage — even an official
one — is not the same epistemic category as a number derived from
validating an actual, structured catalogue of codes. The unverified tier
is still useful context (it's real and sourced, not invented), but
`eval/coverage_audit.py` is hard-coded to never compute a
`coverage_percentage` from it.

**Update (Task 20): ISCO-08 now has a real Verified-tier entry.**
`eval/verified_catalogue_counts.yaml` was created for the first time in
this project by downloading the official ILO ISCO-08 structure workbook
directly (`https://webapps.ilo.org/ilostat-files/ISCO/newdocs-08-2021/
ISCO-08/ISCO-08%20EN%20Structure%20and%20definitions.xlsx`), normalizing
it (`eval/normalize_ilo_isco08_catalogue.py`), and validating it with
`eval/catalogue_importer.py` — zero issues, verified counts `{major: 10,
submajor: 43, minor: 130, unit: 436}`, an exact match to this document's
own unverified figure below. See
[ISCO08_PRIMARY_CATALOGUE_RECONCILIATION.md](ISCO08_PRIMARY_CATALOGUE_RECONCILIATION.md)
for the full retrieval/validation/reconciliation trail. This does
**not** mean this project's own classifier catalogue is standards-
conformant — it still has 20 non-standard codes and is missing 14 real
ones (see that document) — only that the *official count itself* is now
independently verified rather than merely cited. ISIC Rev.4, ISCED 2011,
and ISCED-F 2013 remain Unverified-tier only; this update covers ISCO-08
alone.

## Source-provenance table

| Standard | Issuing organization | Source URL | Retrieved | Licence/use note |
|---|---|---|---|---|
| ISCO-08 | International Labour Organization (ILO) | [isco.ilo.org/en/isco-08](https://isco.ilo.org/en/isco-08) | 2026-08-07 | Freely accessible for reference; no bulk-redistribution licence text located — verify before redistributing an extracted catalogue. |
| ISIC Rev.4 | United Nations Statistics Division (UNSD) | [unstats.un.org/unsd/classifications/Econ/isic](https://unstats.un.org/unsd/classifications/Econ/isic) (structure corroborated via [classification.codes/classifications/industry/isic](https://classification.codes/classifications/industry/isic)) | 2026-08-07 | UN Statistical Papers Series M No.4/Rev.4, freely accessible for reference; no bulk-redistribution licence text located. |
| ISCED 2011 | UNESCO Institute for Statistics (UIS) | [uis.unesco.org/en/methods-and-tools/isced](https://www.uis.unesco.org/en/methods-and-tools/isced) | 2026-08-07 | Freely accessible for reference; no bulk-redistribution licence text located. |
| ISCED-F 2013 | UNESCO Institute for Statistics (UIS) | [UIS detailed field descriptions PDF](https://www.uis.unesco.org/sites/default/files/medias/fichiers/2025/04/international-standard-classification-of-education-fields-of-education-and-training-2013-detailed-field-descriptions-2015-en.pdf) | 2026-08-07 | Freely accessible for reference; no bulk-redistribution licence text located. |

## Per-level unverified counts and exactly how each was obtained

### ISCO-08 (ILO)

**10 major / 43 sub-major / 130 minor / 436 unit groups.**

Confirmed by directly fetching `isco.ilo.org/en/isco-08` in this session.
The page states: *"a four-level hierarchically structured classification
... 436 unit groups ... 130 minor groups ... 43 sub-major groups and 10
major groups."* This is a direct, primary-source confirmation — the
highest confidence tier available without importing a full catalogue.

### ISIC Rev.4 (UNSD)

**21 sections / 88 divisions / 238 groups / 419 classes.**

The primary UNSD publication (`unstats.un.org/unsd/publication/seriesm/
seriesm_4rev4e.pdf`, UN Statistical Papers Series M No.4/Rev.4) could
**not** be programmatically text-extracted in this session (the fetch tool
returned an undecodable PDF stream). These counts are instead corroborated
identically across two independent sources consulted in this session: a
web-search results summary, and `classification.codes/classifications/
industry/isic`, which itself cites UNSD documentation. Both agree exactly
on 21/88/238/419. This is **secondary corroboration, not primary-document
confirmation** — a meaningfully lower confidence tier than ISCO-08's. Do
not treat these four numbers as verified; import an official ISIC Rev.4
catalogue via `eval/catalogue_importer.py` before citing a coverage
percentage.

### ISCED 2011 (UNESCO UIS)

**9 levels (0–8).**

Confirmed by directly fetching `uis.unesco.org/en/methods-and-tools/isced`
in this session, which lists all 9 levels by name (Early childhood
education through Doctoral or equivalent level). Also structurally
consistent with this codebase's own `_ISCED_LEVELS` table
(`backend/agents/isced_classifier.py`), which independently encodes the
same 9 levels — though that consistency is circumstantial corroboration,
not itself a source (this codebase's own table cannot verify itself).

### ISCED-F 2013 (UNESCO UIS)

**11 broad fields / 29 narrow fields / detailed fields: left `null`.**

11 and 29 are corroborated identically across multiple independent
secondary sources consulted in this session (web-search summaries citing
UNESCO UIS's ISCED-F 2013 documentation). The detailed-field count is
**deliberately left unverified (`null`)** — every source found in this
session hedges it as *"about 80 detailed fields,"* never a precise figure.
An approximate figure is not precise enough to record as if it were exact;
recording "80" as `official_count_unverified` would misrepresent the
actual confidence behind that number. Import an official ISCED-F 2013
catalogue for an exact, verified count.

## What this means for the manuscript today

**No coverage percentage is supportable yet for any of the four
standards**, including ISIC. `eval/coverage_audit.py`'s `coverage_percentage`
field is `null` for every level, for every standard, as of this session —
confirmed by `eval/test_coverage_audit.py::
test_real_standards_reference_yaml_has_unverified_counts_but_no_verified_ones_by_default`
and by inspecting a freshly-generated `coverage_audit_*.json` report (see
`generated/`). Closing this gap requires supplying a real catalogue file
and running `eval/catalogue_importer.py` — see `COVERAGE_AUDIT_GUIDE.md`
for exactly how.

## Honesty notes for future maintainers

- If you re-run the web research behind this document, record a fresh
  `retrieval_date` and update the notes above — do not silently reuse
  these dates for a different research session.
- If ISIC Rev.4's primary PDF becomes extractable (e.g. via a different
  tool or an OCR pass), re-verify 21/88/238/419 directly and upgrade the
  note in `eval/standards_reference.yaml` accordingly — this document's
  "secondary corroboration" caveat should be removed only once that
  happens, not before.
- Never copy an official standard's full code list into this repository
  (see `.gitignore`'s `eval/local_catalogues/` entry and
  `COVERAGE_AUDIT_GUIDE.md`'s "Do not commit the catalogue" section) —
  only aggregate counts and hashes belong in version control.
