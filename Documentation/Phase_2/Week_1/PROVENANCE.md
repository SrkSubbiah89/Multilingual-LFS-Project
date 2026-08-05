# Data Provenance Record — WISCO

> Location: `Documentation/Phase_2/Week_1/PROVENANCE.md`
> Raw workbooks live at: `backend/evaluation/wisco/data/raw/`
> This record is cited in the thesis reproducibility appendix and in the IEEE paper.
> **Corrected 2026-08-02** after an external review caught that the wrong file version had
> been used. Both the original error and the correction are documented below in full —
> this is a provenance record, and silently overwriting the mistake would defeat its purpose.

---

## Correction note (read this first)

The file originally downloaded and used for all of Module A's Week 1 analysis
(`Occupations_ISCO08_5dgt_55languages_4000titles_with_mapping_20230202.xlsx`) was the
**oldest of 4 versions** published under this Zenodo concept DOI. The original Module A
plan's own `PROVENANCE.md` template expected a file named
`..._20230818.xlsx` — a **newer** file that exists under the *same* concept DOI but was
not the one fetched. The original download also contained a wrong claim: that a related
DOI (**7871194**) "does not resolve to a WISCO record." It does — 7871194 is another
version in the same version chain (published 2023-04-25). That claim has been removed.

**All Module A artefacts (inspection, language mapping, parsing, integrity checks) were
redone against the correct file** (see `module_a_week1_report.md` and
`wisco_structure_inspection.md` for the current, correct numbers). The old file is kept on
disk (`Occupations_ISCO08_5dgt_55languages_4000titles_with_mapping_20230202.xlsx`) purely
as an audit trail of the error — no script uses it.

## Version chain (Zenodo concept DOI 10.5281/zenodo.7598567)

| Zenodo record id | Version DOI | Published | File(s) | Used? |
|---|---|---|---|---|
| 7598568 | 10.5281/zenodo.7598568 | 2023-02-02 | `Occupations_ISCO08_5dgt_55languages_4000titles_with_mapping_20230202.xlsx` | **No — superseded, kept for audit trail only** |
| 7775374 | 10.5281/zenodo.7775374 | 2023-02-02 | `occupations_ISCO08_5dgt_55languages_4000titles_with_mapping_20230327.xlsx` | No |
| 7871194 | 10.5281/zenodo.7871194 | 2023-04-25 | `WISCO occupations_..._20230425.xlsx` | No |
| **8262593** | **10.5281/zenodo.8262593** | **2023-04-25 (record); latest file within it dated 2023-08-18)** | `WISCO occupations_..._20230425.xlsx` **and** `occupations_ISCO08_5dgt_55languages_4000titles_surveycodings_20230818.xlsx` | **Yes — canonical, this is what the original plan expected** |

The concept DOI (10.5281/zenodo.7598567, note: no trailing "8") always resolves to whichever
version is current — **do not cite the concept DOI**, since it doesn't pin what was
actually used. This record cites the specific version DOI below.

## Canonical file (used for all Module A results)

| Field | Value |
|---|---|
| Dataset name | WISCO — World database of ISCO Occupations |
| Publisher | SurveyCodings / WageIndicator Foundation (Zenodo record creator: Kea Tijdens) |
| Version DOI | **10.5281/zenodo.8262593** |
| Zenodo record title | `WISCO occupations_ISCO08_5dgt_55languages_4000titles_with_mapping_surveycodings_20230425` |
| Filename as received | `occupations_ISCO08_5dgt_55languages_4000titles_surveycodings_20230818.xlsx` |
| Internal citation string (from the workbook's own CITATION sheet) | `Tijdens, K.G. (2023). Occupations_ISCO08_5dgt_55languages_4000titles_with_mapping_20230613. Netherlands, WageIndicator Foundation` — note this internal date (20230613) is yet a *third* date label distinct from both the Zenodo record's publish date and the filename's date suffix; the source itself is inconsistently dated internally. Cite the Zenodo version DOI as the authoritative anchor. |
| Retrieval URL | `https://zenodo.org/api/records/8262593/files/occupations_ISCO08_5dgt_55languages_4000titles_surveycodings_20230818.xlsx/content` |
| Downloaded by | Sivar (git identity on this machine — **confirm full name for the thesis record**) |
| Download timestamp (UTC) | 2026-08-02 |
| File size (bytes) | 8,601,590 |
| MD5 checksum (Zenodo-published) | `9cdd0270720d6eb16677f58379c33376` — **matches** the value published in the Zenodo record's file metadata; download integrity confirmed |
| Licence / terms of use | CC-BY-4.0 (as recorded in the Zenodo record) |

### Attribution obligations (both required — this is new: the second one was missed originally)

1. **General use** — cite the Zenodo version DOI (10.5281/zenodo.8262593) and creator (Kea Tijdens), per CC-BY-4.0.
2. **If `OCC>>INDUSTRY` / NACE2004 predictions are used anywhere (Module D)** — the workbook's own CITATION sheet requires a *second*, separate citation:
   > Belloni, M, Tijdens, K.G. (2017). Occupation > industry predictions for measuring industry in surveys, Deliverable 8.11 of the SERISS project funded under the European Union's Horizon 2020 research and innovation programme GA No: 654221, DOI 10.13140/RG.2.2.31328.02566

Both citation strings need to appear in the repo README and the thesis, and any files
derived from this data (the parsed JSON outputs) carry the same obligation forward.

## Stated scale (per publisher, description text unchanged across all 4 versions)

- Occupational titles: ~4,000
- Languages: 55 (the canonical file actually has **61** distinct base languages — the
  description text was not updated when languages were added between versions)
- Gold mapping precision: **there is no official 5-digit ISCO-08 level.** The finest
  precision this workbook provides is 4-digit (`ISCO0804`), matching ISCO-08's own
  unit-group level. The "5dgt" in the filename refers to SurveyCodings' internal
  occupation-ID scheme, not ISCO-08 code precision — corrected from an earlier planning
  assumption that this workbook offered 5-digit gold codes.

## Observed scale (canonical file, measured 2026-08-02)

| Measure | Observed |
|---|---|
| Total rows in titles sheet (CODESET) | 4,745 (of which 246 are section/category heading rows, not occupations — see `wisco_structure_inspection.md`) |
| Distinct base language codes present | 61 |
| Distinct ISCO-08 unit groups (4-digit) present | 436 |
| Occupation records successfully parsed (title + valid code) | 4,232 |
| Rows retained for the 5 target languages | en 4,230 / ar 4,167 / ur 3,989 / hi 4,202 / tl 4,172 — see `module_a_week1_report.md` §4 |

Full detail: `wisco_structure_inspection.md` and `module_a_week1_report.md` (both in this
same `Documentation/Phase_2/Week_1/` folder).

## Notes

- The archive is a **single Excel workbook** with 9 sheets (CODESET, STRUCTURE, MAPPINGS,
  OCC>>INDUSTRY, LABELSET, CODING RULES, CITATION, UPDATES, SUMMARY CHECKS) — see
  `wisco_structure_inspection.md` for the full sheet inventory.
- File integrity: size (8,601,590 bytes) and MD5 checksum matched Zenodo's published file
  metadata exactly.
- The `UPDATES` sheet (1,423 rows) documents a real translation-improvement changelog
  between versions, none of which touches our 5 target languages directly (updates are
  concentrated in Russian-family, Czech, Georgian, Slovenian, Dutch, and Italian locales,
  plus 128 "Improved English" master-label edits) — but the underlying occupation/title
  *count* did grow materially (4,232 → 4,745 CODESET rows) between the version originally
  downloaded and the canonical one.
