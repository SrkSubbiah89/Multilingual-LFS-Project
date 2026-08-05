# WISCO Language Code Mapping

> Location: `Documentation/Phase_2/Week_1/language_mapping_note.md`
> Produced: 2026-08-02, from `backend/evaluation/wisco/analyze_wisco.py` output (`wisco_language_analysis.json`)
> **Corrected 2026-08-02, twice over:** (1) rewritten against the canonical file after the
> version error was found (see `PROVENANCE.md`); (2) the Arabic-locale rationale in §4.1 was
> **wrong** in the original draft and has been replaced after external review prompted a direct
> test. Read §4.1 even if you've seen this document before — the conclusion changed.
> **This is a documented research decision, not a lookup.** WISCO's language codes and the
> system's 5 `LanguageProcessor` categories were built by different people for different
> purposes. This note records how they were reconciled, because the Week 3 per-language
> accuracy figures and the Week 9 dialect-normalisation plan both depend on it.

## 1. Target categories

| System category | LanguageProcessor label | Notes |
|---|---|---|
| English | `en` | |
| Arabic | `ar` (MSA + Gulf dialect handling) | `_GULF_NORMALISE` 30-rule dialect-to-MSA normaliser applies |
| Urdu | `ur` | |
| Hindi | `hi` | |
| Tagalog | `tl` | |

## 2. Structural finding that changes how this mapping works

WISCO is **not** a long table with one language-code column and repeated title rows per
language. It is a **wide table**: 4,745 occupation rows (246 of which are section/category
headings, not occupations — see `wisco_structure_inspection.md`) × 231 columns, where each
column is a `{base language}_{country}` locale (e.g. `ar_AE`, `en_US`, `hi_IN`). The canonical
file has **61 distinct base languages** (up from 55 in the superseded 2023-02-02 file that an
earlier draft of this note was based on).

WISCO already groups every locale under its correct base language — there is no cross-base
folding to do. The judgement call is **which locale to pick as the primary column** for
languages with more than one (`en`: 65 options, `ar`: 22 options; `ur`, `hi`, `tl` each have
exactly one).

## 3. Disposition of every WISCO base language code

All 61 base codes in the canonical file, each with a disposition and a reason.

| WISCO base code | Locale variants (count) | Label | Disposition | Mapped to |
|---|---|---|---|---|
| `am` | 1 | Amharic | excluded (out of scope) | - |
| `ar` | 22 | Arabic | matched | ar |
| `az` | 1 | Azerbaijani | excluded (out of scope) | - |
| `ba` | 1 | Bashkir (ISO 639-1) / locale suggests Indonesia -- ambiguous, unresolved | excluded (out of scope) | - |
| `bg` | 1 | Bulgarian | excluded (out of scope) | - |
| `bn` | 1 | Bengali | excluded (out of scope) | - |
| `bs` | 1 | Bosnian | excluded (out of scope) | - |
| `cs` | 1 | Czech | excluded (out of scope) | - |
| `da` | 1 | Danish | excluded (out of scope) | - |
| `de` | 5 | German | excluded (out of scope) | - |
| `dz` | 1 | Dzongkha | excluded (out of scope) | - |
| `el` | 2 | Greek | excluded (out of scope) | - |
| `en` | 65 | English | matched | en |
| `es` | 24 | Spanish | excluded (out of scope) | - |
| `et` | 1 | Estonian | excluded (out of scope) | - |
| `fa` | 2 | Persian (Farsi) | excluded (out of scope) | - |
| `fi` | 1 | Finnish | excluded (out of scope) | - |
| `fr` | 28 | French | excluded (out of scope) | - |
| `he` | 1 | Hebrew | excluded (out of scope) | - |
| `hi` | 1 | Hindi | matched | hi |
| `hr` | 2 | Croatian | excluded (out of scope) | - |
| `hu` | 1 | Hungarian | excluded (out of scope) | - |
| `hy` | 1 | Armenian | excluded (out of scope) | - |
| `is` | 1 | Icelandic | excluded (out of scope) | - |
| `it` | 2 (one column name duplicated — see `wisco_structure_inspection.md` §6) | Italian | excluded (out of scope) | - |
| `ja` | 1 | Japanese | excluded (out of scope) | - |
| `ka` | 1 | Georgian | excluded (out of scope) | - |
| `kk` | 1 | Kazakh | excluded (out of scope) | - |
| `km` | 1 | Khmer | excluded (out of scope) | - |
| `ko` | 2 | Korean | excluded (out of scope) | - |
| `lo` | 1 | Lao | excluded (out of scope) | - |
| `lt` | 1 | Lithuanian | excluded (out of scope) | - |
| `lv` | 1 | Latvian | excluded (out of scope) | - |
| `mk` | 1 | Macedonian | excluded (out of scope) | - |
| `mn` | 1 | Mongolian | excluded (out of scope) | - |
| `ms` | 2 | Malay | excluded (out of scope) | - |
| `my` | 1 | Burmese | excluded (out of scope) | - |
| `ne` | 1 | Nepali | excluded (out of scope) | - |
| `nl` | 4 | Dutch | excluded (out of scope) | - |
| `no` | 1 | Norwegian | excluded (out of scope) | - |
| `pl` | 1 | Polish | excluded (out of scope) | - |
| `pt` | 8 | Portuguese | excluded (out of scope) | - |
| `ro` | 2 | Romanian | excluded (out of scope) | - |
| `ru` | 7 | Russian | excluded (out of scope) | - |
| `si` | 1 | Sinhala | excluded (out of scope) | - |
| `sk` | 1 | Slovak | excluded (out of scope) | - |
| `sl` | 1 | Slovenian | excluded (out of scope) | - |
| `so` | 1 | Somali | excluded (out of scope) | - |
| `sq` | 3 | Albanian | excluded (out of scope) | - |
| `sr` | 2 | Serbian | excluded (out of scope) | - |
| `sv` | 1 | Swedish | excluded (out of scope) | - |
| `sw` | 2 | Swahili | excluded (out of scope) | - |
| `th` | 1 | Thai | excluded (out of scope) | - |
| `tk` | 1 | Turkmen | excluded (out of scope) | - |
| `tl` | 1 | Tagalog | matched | tl |
| `tr` | 1 | Turkish | excluded (out of scope) | - |
| `uk` | 1 | Ukrainian | excluded (out of scope) | - |
| `ur` | 1 | Urdu | matched | ur |
| `uz` | 2 | Uzbek | excluded (out of scope) | - |
| `vi` | 1 | Vietnamese | excluded (out of scope) | - |
| `zh` | 4 | Chinese | excluded (out of scope) | - |

## 4. Judgement calls

### 4.1 Arabic — corrected finding, read this even if you've seen an earlier draft

**What WISCO provides:** one base code `ar` split by country into 22 locale variants (was 16
in the superseded file; canonical file adds DJ, ER, KM, LY, SY, YE).

**Original (wrong) reasoning:** an earlier draft of this note picked `ar_AE` and claimed this
gives "a natively Gulf-dialect-flavoured Arabic test set for free," reasoning that the LFS
system targets UAE respondents and already implements Gulf-dialect normalisation.

**Why that was wrong, and how it was checked:** external review pointed out that WISCO's
Arabic columns are translations of a single English source, not respondent utterances — so
there's no a priori reason to expect dialectal variation between the 22 "country" columns at
all, only a reason to *hope* for it. This was tested directly rather than left as a caveat:
every non-`ar_AE` Arabic column was diff'd against `ar_AE` across all rows where both are
populated (4,168 rows).

**Result: 0 of 21 comparison columns differ from `ar_AE` in a single row, across the entire
dataset.** The 22 "country" Arabic columns are the same Modern Standard Arabic translation,
mechanically duplicated 22 times. (On the superseded file, one shared mistranslation existed —
"ice hockey coach" mislabelled as "basketball coach" in 5 columns; the canonical file has fixed
even that, so the agreement is now literally 100%.)

**Consequence:**
- The locale choice among the 22 Arabic columns is **cosmetic** — any one of them gives
  identical text. `ar_AE` is still used, simply because a locale has to be picked and this one
  is at least nominally aligned with the deployment context, but it carries no evidentiary
  weight about dialect.
- **The Week 9 dialect-normalisation A/B test cannot be run against WISCO as originally
  planned.** WISCO's Arabic titles are already clean MSA with no dialectal content to
  normalise — running `_GULF_NORMALISE` against them would trivially show no effect, and that
  null result would say nothing about whether the normaliser works; it would only confirm the
  input had nothing to normalise. Week 9 needs either (a) a different data source with genuine
  Gulf-dialect text (e.g. synthetic Gulf-dialect job-title variants, or real respondent
  free-text captured during the pilot study once Module E is underway), or (b) a redefinition
  of what the Week 9 experiment measures if WISCO remains the only source in scope.
- **Chapter 6 needs an explicit caveat, independent of the Week 9 point:** "WISCO titles are
  translations of a single reference English source, not naturally-occurring respondent text."
  This affects how any WISCO-based accuracy figure should be read for *all 5* languages, not
  just Arabic — translated, professionally-curated text is a best-case input, systematically
  cleaner than what real respondents will type.

### 4.2 Tagalog / Filipino
- What WISCO provides: exactly one code, `tl_PH`.
- Decision: use directly. No ambiguity.

### 4.3 Urdu and Hindi
- Checked whether these are merged in the source (a merged set would inflate apparent
  cross-language performance, since they share vocabulary but not script).
- **Finding: not merged.** Distinct base codes, each with exactly one locale (`ur_PK`,
  `hi_IN`), correct distinct scripts confirmed by inspection.
- Decision: use `ur_PK` and `hi_IN` directly.

### 4.4 English
- WISCO provides **65** English country-locale columns (was 40 in the superseded file).
- Decision: use **`en_US`** — highest fill rate (4,230/4,232 in the parsed output), and
  matches `MASTER LABEL 4000` in every sampled row.
- No material consequence for later weeks.

### 4.5 Other notes
- `ba_ID`: ISO 639-1 `ba` is Bashkir, but the `_ID` suffix suggests Indonesia — inconsistent
  pairing, unresolved, doesn't affect our 5 target languages.
- `it_IT` appears as a duplicate column header in CODESET (data-quality issue in the source,
  doesn't affect our target languages) — see `wisco_structure_inspection.md` §6.

## 5. Retained title counts and statistical power

Non-empty counts for each target language's primary column, out of 4,232 successfully-parsed
occupation records (see `module_a_week1_report.md` for how 4,745 raw CODESET rows reduce to 4,232):

| Language | Column used | Titles retained | Coverage | Sufficient for a per-language accuracy estimate? |
|---|---|---|---|---|
| English | `en_US` | 4,230 | 99.95% | Yes |
| Arabic | `ar_AE` | 4,167 | 98.5% | Yes (but see §4.1 — coverage is not the concern here, dialectal content is) |
| Urdu | `ur_PK` | 3,989 | 94.3% | Yes, though noticeably lower than the others — worth a one-line note in Chapter 6 |
| Hindi | `hi_IN` | 4,202 | 99.3% | Yes |
| Tagalog | `tl_PH` | 4,172 | 98.6% | Yes |

**Flag threshold:** any language under ~200 titles should be raised as a reporting decision.
All five are comfortably above that. Urdu's 94.3% (vs. 98.5–99.95% for the others) is not a
statistical-power problem but is the largest gap among the five and worth a one-line mention
in Chapter 6 for completeness.

**Languages flagged for thin coverage: none.**

## 6. Wording for Chapter 6

> WISCO's 231 language-locale columns were grouped into 61 base language codes by ISO 639-1
> prefix. Five — English, Arabic, Urdu, Hindi, and Tagalog — correspond to the system's target
> languages. Urdu and Hindi were verified to be represented as distinct, unmerged columns in
> the correct scripts. For English (65 country variants) and Arabic (22 country variants),
> `en_US` and `ar_AE` were selected as primary columns. Critically, all 22 Arabic country
> columns were confirmed — by direct row-by-row comparison, not assumption — to contain
> identical Modern Standard Arabic text with zero dialectal variation; the locale choice is
> therefore cosmetic, and WISCO cannot support the planned dialect-normalisation evaluation.
> More generally, WISCO's titles are translations of a single reference source rather than
> naturally-occurring respondent utterances, which should be read as a ceiling-case rather than
> a representative input when interpreting any WISCO-derived accuracy figure in this thesis.
