# WISCO Workbook Structure — Inspection Report

> Location: `Documentation/Phase_2/Week_1/wisco_structure_inspection.md`
> Produced: 2026-08-02, from `backend/evaluation/wisco/inspect_wisco.py` and `analyze_wisco.py`
> **Corrected 2026-08-02**: rewritten against the canonical file
> (`occupations_ISCO08_5dgt_55languages_4000titles_surveycodings_20230818.xlsx`) after
> discovering the originally-downloaded file was an older version. See `PROVENANCE.md` for
> the full correction note. Numbers below differ from an earlier draft of this document —
> this is the current, correct version.
> Raw machine-readable output: `backend/evaluation/wisco/data/interim/wisco_structure_inspection.json`, `wisco_language_analysis.json`.

## 1. Sheets present

| Sheet name | Rows (excl. header) | Columns | Role |
|---|---|---|---|
| `CODESET` | 4,745 | 233 | Occupational titles, wide format — but see §6 item 1: 246 of these rows are section/category headings, not occupations |
| `STRUCTURE` | 2,909 | 3 | Internal SurveyCodings grouping hierarchy (negative-numbered Level 1/2 categories → occupation keys at Level 3). Not used by Module A; the negative-number scheme it uses is the same one that leaks into CODESET's key column (§6 item 1). |
| `MAPPINGS` | 4,469 | 9 | Gold ISCO-08 codes (4 precision levels) + skill level + ISEI + NACE2.0 |
| `OCC>>INDUSTRY` | 4,232 | 35 | Occupation × industry (NACE **2004**, i.e. Rev.1.1) cross-tab — see §6 item 3 for why this is not simply "the" industry source |
| `LABELSET ISCO-1-4dgt NACE` | 952 | 4 | Human-readable label lookups for ISCO/NACE numeric codes |
| `CODING RULES` | 6 | 2 | **New in this version.** Documents the join key's own internal digit structure — see §6 item 2 |
| `CITATION` | 4 | 1 | Citation string, plus a *second*, separate citation required if `OCC>>INDUSTRY` is used (see `PROVENANCE.md`) |
| `UPDATES` | 1,423 | 9 | **New in this version.** Changelog of translation fixes between file versions |
| `SUMMARY CHECKS` | 7 | 4 | **New in this version.** QA sign-off log for a subset of languages (does not cover our 5 target languages) |

## 2. Sheet → expected role mapping

| Expected role | Documentation says | Actually found | Match? |
|---|---|---|---|
| Occupational titles | CODESET | `CODESET` | Yes, with the caveat in §6 item 1 |
| Gold ISCO-08 codes | MAPPINGS | `MAPPINGS` | Yes |
| Occupation × industry cross-tab | OCC>>INDUSTRY | `OCC>>INDUSTRY` | Yes, but see §6 item 3 — this is NACE Rev.1.1, not the ISIC-Rev.4-comparable source |

**Cross-tab status: present and usable, with a coverage/precision trade-off, not a clean "yes."**
`OCC>>INDUSTRY` gives 100% row coverage (every occupation has ≥1 real NACE2004 code, avg 9.4
per occupation) but is NACE Rev.1.1 — one classification generation behind ISIC Rev.4, requiring
an extra hop. `MAPPINGS.NACE2.0` (NACE Rev.2, which maps cleanly to ISIC Rev.4 at class level) only
covers 16.7% of occupations (745/4,469), and where both exist, they agree on only ~26% of rows —
they are measuring different things (a single "canonical" industry vs. a broader "occupation could
appear in these industries" list), not two versions of the same fact. See
`module_a_week1_report.md` §5 for the Module D recommendation.

## 3. Column inventory

### CODESET sheet (titles)

| Column | Dtype as read | Example values | Interpretation |
|---|---|---|---|
| `occupai3_API_13dgt` | mixed: int / negative int / imprecise float | `110000100018` | Join key — **not uniformly typed**, see §6 item 1 |
| `MASTER LABEL 4000` | str | `"Air force captain"` | Canonical/reference English label. **Column name changed from `MASTER LABEL` in the superseded file.** |
| 231 locale columns | str | `ar_AE`: `"قائد سلاح الجو"` | One column per language-country variant, 61 distinct base languages (up from 55 in the superseded file) — full disposition in `language_mapping_note.md` |

Note: `it_IT` appears **twice** as a column header (data-quality issue in the source; only the
first occurrence is reachable via name-based lookup — flagged, not used by any target-language logic).

### MAPPINGS sheet (gold codes)

| Column | Dtype as read | Interpretation |
|---|---|---|
| `occupai3_API_13dgt` | int (clean, no negative/float issues here) | Join key |
| `occupai3_API_17dgt` | int | Finer-grained internal ID, not an ISCO code |
| `ISCO0804` / `0803` / `0802` / `0801` | int | ISCO-08 code at 4/3/2/1-digit precision. `ISCO0804` loses leading zeros on read for major-group-0 codes — see §6 item 4 |
| `ISCO08lv` | int, 1–4 | ISCO-08 skill level. Distribution: level 1=261, level 2=2,149, level 3=1,073, level 4=986. Extracted into the parsed output for the Week 7 ISCO↔ISCED cross-check (skill level is defined to align with ISCED groups). |
| `ISCOISEI` | — | International Socio-Economic Index score, not used by Module A |
| `NACE2.0` | int | NACE **Rev.2** primary industry code. Also loses leading zeros on read (154/745 non-null values were 3-digit after `int()` conversion) — zero-padded the same way as `ISCO0804`. |

### CODING RULES sheet (new)

Documents the join key's digit positions: 1–4 = ISCO-08 4-digit code, 5–6 = follow-up code
within that group, 7–8 = country-specific follow-up (not used after 2020/1), 9–11 = country
code (not used after 2020/1), 12–13 = year the occupation was added since 2017.

**Checked and rejected**: whether the key's own leading 4 digits could be used as a fallback/
cross-check for the `ISCO0804` column value. On a 2,000-row sample this agreed with `ISCO0804`
98.35% of the time, but disagreed for exactly the cases that matter most — including
known-correct rows like "Air force captain" (key prefix says major group 1, but the verified
correct code, confirmed via `LABELSET`, is major group 0). The key encoding is evidently a
legacy/placeholder scheme that doesn't reliably track ISCO-08 reclassifications. **Not used as
a correction source** — see §6 item 5 for the 8 rows this matters for.

## 4. Join key

- Column linking titles (CODESET) to codes (MAPPINGS): `occupai3_API_13dgt`
- Cardinality: **mostly one-to-one, with real exceptions** — CODESET has 7 duplicate keys and
  183 rows with no key at all (before excluding category rows); MAPPINGS has 107 duplicate
  keys. The parser resolves duplicates "last row wins" and logs the count; see
  `module_a_week1_report.md` §3 (C6).
- Orphan rows: 79 CODESET titles have no matching MAPPINGS code, and separately 125 MAPPINGS
  codes have no matching CODESET title. Both counts are fully diagnosed, not just measured —
  see §6 items 1, 5, and 6.

## 5. Language code column

Same wide-format structure as previously documented, now with more coverage: 61 distinct base
languages across 231 locale columns (up from 55/184 in the superseded 2023-02-02 file). Our 5
target languages: `en` now has 65 country variants (was 40), `ar` has 22 (was 16, new: DJ, ER,
KM, LY, SY, YE), `ur`/`hi`/`tl` remain single-locale (`ur_PK`, `hi_IN`, `tl_PH`).

## 6. Surprises

| # | Surprise | Parser adjustment or note |
|---|---|---|
| 1 | 246 CODESET rows are section/category headings leaking into the occupation data range — their key is a small negative integer (e.g. `-20509`) matching the STRUCTURE sheet's own negative-numbered Level 1/2 scheme, and their "master label" is a category name ("Waterworks", "Oil, gas", "Management, direction"), not a job title | `parse_wisco.py` excludes any row with a negative key before treating it as an occupation |
| 2 | A separate 71 CODESET rows *are* real occupations ("Pre-school teacher", "Chef de Partie", "Yoga trainer", ...) but openpyxl reads their key as an imprecise float (e.g. `1.34400030001688e+16`) because the value exceeds float64's exact-integer range | Rounding to the nearest int is attempted but **does not recover a match** against MAPPINGS for any of these 71 — the precision loss originates in the source file itself. These are genuine, currently-unrecoverable orphans; there is no alternative join path (MAPPINGS has no title text to fuzzy-match against). |
| 3 | `OCC>>INDUSTRY`'s NACE2004 codes are NACE **Rev.1.1** (one generation behind ISIC Rev.4), while `MAPPINGS.NACE2.0` is NACE Rev.2 (maps cleanly to ISIC Rev.4 at class level) — but NACE2.0 only covers 16.7% of rows, and the two sources agree on only ~26% of rows where both exist | Both are extracted into a separate `wisco_industry_crosswalk.json` output with clear provenance tags (`nace2_0_rev2`, `nace2004_rev1_1`) — the parser does not pick one for Module D; see `module_a_week1_report.md` §5 |
| 4 | `ISCO0804` and `NACE2.0` are both read by openpyxl as Python `int`, silently stripping leading zeros (33 major-group-0 Armed Forces codes for `ISCO0804`; 154/745 values for `NACE2.0`) | Both are `str(int(v)).zfill(n)`'d before being treated as code strings |
| 5 | 8 rows are doubly broken in the source data itself, independent of the parser: their MAPPINGS `ISCO0801`–`0804` hierarchy is internally consistent but semantically wrong (master labels are unambiguously Armed Forces occupations — "Commissioned officer armed forces", "Military weapons specialist" — but the codes claim major group 1/2/3, not 0), **and** their CODESET key doesn't match their MAPPINGS key for the same occupation at all (CODESET has `1100100000008420`, MAPPINGS has `110010000000` for "Commissioned officer armed forces") | Quarantined by the parser's prefix-consistency check; not guessed at or auto-corrected |
| 6 | `it_IT` appears twice as a CODESET column header | Flagged as a data-quality note; doesn't affect any of our 5 target languages |
| 7 | The `occupai3_API_13dgt` / `_17dgt` naming, and the filename's "5dgt", do not refer to ISCO-08 5-digit precision — there is no official 5-digit ISCO-08 level. The finest gold-code precision actually present is `ISCO0804` (4-digit), matching our own `ISCOClassifier`'s target precision exactly. | Any earlier reference to "5-digit gold codes" should be corrected — see the Week 14 audit list in `module_a_week1_report.md` §8 |

## 7. Constants used by `parse_wisco.py`

```python
TITLES_SHEET = "CODESET"
CODES_SHEET  = "MAPPINGS"
CROSSTAB_SHEET = "OCC>>INDUSTRY"
KEY_COL      = "occupai3_API_13dgt"
MASTER_LABEL_COL = "MASTER LABEL 4000"   # was "MASTER LABEL" in the superseded file
ISCO4_COL    = "ISCO0804"   # int in source; MUST str().zfill(4) before use
ISCO3_COL    = "ISCO0803"
ISCO2_COL    = "ISCO0802"
ISCO1_COL    = "ISCO0801"
ISCO08LV_COL = "ISCO08lv"
NACE2_0_COL  = "NACE2.0"    # also needs str().zfill(4)

TARGET_LOCALE_COL = {
    "en": "en_US",
    "ar": "ar_AE",
    "ur": "ur_PK",
    "hi": "hi_IN",
    "tl": "tl_PH",
}

OCC_INDUSTRY_SENTINEL_NONE = 99999
```
