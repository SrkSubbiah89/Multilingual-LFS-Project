# Module A — Week 1 Report

> Location: `Documentation/Phase_2/Week_1/module_a_week1_report.md`
> Produced: 2026-08-02. **Rewritten 2026-08-02 (same day)** after external review caught a
> wrong file version and several under-read findings. This version supersedes the original
> in every section — see §7 for the itemised list of what changed and why.

## 1. Summary

Module A is complete against the **canonical** WISCO file (version DOI
10.5281/zenodo.8262593, filename `..._20230818.xlsx`) — the original download had used the
oldest of 4 published versions, caught during external review and corrected same-day; see
`PROVENANCE.md`. The canonical file is materially larger (4,745 vs 4,232 CODESET rows, 61 vs
55 languages) and includes new QA/changelog sheets absent from the older file.

4,232 clean occupation records were parsed with full, individually-explained accounting for
every excluded or orphaned row — nothing is an unexplained residual. All 9 integrity checks
pass, with two genuine defects caught and fixed (C4's leading-zero strip, now confirmed on two
separate columns; C6's join-key inconsistencies, fully diagnosed rather than just counted).

The most consequential finding is **not** a WISCO data-quality issue: cross-referencing
WISCO's 436 ISCO-08 unit groups (which matches the true ISCO-08 standard count exactly)
against our own system's 441-entry knowledge base shows our system carries **19 non-standard
codes that a real classifier can return to a respondent**, is **missing 14 real ISCO-08 unit
groups**, and at least 4 of those 19/14 are the same occupational content filed under the
wrong code (see §5). This is a defect in the LFS system itself, not a WISCO coverage gap, and
belongs in Chapter 3 and Chapter 6, not a follow-up ticket.

Two further findings materially change Week 2/3/9 planning: WISCO's Arabic data has **zero**
dialectal content (§6), so the planned Week 9 dialect-normalisation test cannot run against it
as originally scoped; and the Module D industry-crosswalk source needs a two-source design,
not a single pick (§5.4).

## 2. Artefacts produced

| Artefact | Path | Status |
|---|---|---|
| Canonical raw workbook | `backend/evaluation/wisco/data/raw/occupations_ISCO08_5dgt_55languages_4000titles_surveycodings_20230818.xlsx` | Downloaded, checksum-verified |
| Superseded raw workbook (audit trail only) | `backend/evaluation/wisco/data/raw/Occupations_ISCO08_5dgt_55languages_4000titles_with_mapping_20230202.xlsx` | Kept, not used by any script |
| Provenance record | `Documentation/Phase_2/Week_1/PROVENANCE.md` | Rewritten with full version-correction note |
| Structure inspection | `Documentation/Phase_2/Week_1/wisco_structure_inspection.md` | Rewritten against canonical file |
| Language mapping note | `Documentation/Phase_2/Week_1/language_mapping_note.md` | Rewritten; Arabic finding corrected |
| Parsed dataset | `backend/evaluation/wisco/data/processed/wisco_raw_parsed.json` | 4,232 records, includes ISCO08 skill level |
| Industry crosswalk (new) | `backend/evaluation/wisco/data/processed/wisco_industry_crosswalk.json` | Both NACE2.0 and NACE2004 sources, separately tagged |
| Parse summary | `backend/evaluation/wisco/data/processed/wisco_parse_summary.json` | Full diagnostic counts |
| Inspection script | `backend/evaluation/wisco/inspect_wisco.py` | Repointed to canonical file |
| Analysis script | `backend/evaluation/wisco/analyze_wisco.py` | Repointed; adds Arabic-distinctness and NACE-comparison checks |
| Parser | `backend/evaluation/wisco/parse_wisco.py` | Repointed; adds key normalisation, NACE2.0/ISCO08lv extraction, industry crosswalk output |
| Environment lock | `backend/evaluation/wisco/requirements.lock.txt` | New — see §8 |
| Tagged commit | Not yet tagged | Still pending — see §8 |

## 3. Integrity check results

| # | Check | Result | Notes |
|---|---|---|---|
| C1 | Sheet and column contract | **PASS** | 9 sheets in the canonical file (vs. 5 in the superseded one): CODESET, STRUCTURE, MAPPINGS, OCC>>INDUSTRY, LABELSET, CODING RULES, CITATION, UPDATES, SUMMARY CHECKS |
| C2 | Per-language title counts | **PASS** | en 4,230 / ar 4,167 / ur 3,989 / hi 4,202 / tl 4,172, all well above the 200-title flag threshold |
| C3 | Code validity (well-formed) | **PASS, with 8 rows correctly quarantined** | See §6 for the full diagnosis — these are genuine WISCO data defects, not zero-padding artefacts |
| C4 | Leading-zero preservation | **PASS (after fix, on two columns)** | Both `ISCO0804` (33 Armed Forces codes) and `NACE2.0` (154/745 values) are read as Python `int` by openpyxl, silently stripping leading zeros. Both are `zfill()`'d before use. The `NACE2.0` instance of this bug was not caught in the original (superseded) pass — found only when NACE2.0 was investigated in response to review feedback. |
| C5 | Digit-derivation consistency | **PASS** | Adapted from "5-to-4-digit derivation": no 5-digit ISCO column exists in this workbook (confirmed — see §7 item 3), so this instead verifies 4-digit codes' leading digits match their declared 1/2/3-digit parents. The 8 C3 quarantines are exactly the rows that fail this. |
| C6 | Duplicates and orphans | **PASS, fully diagnosed, not just counted** | CODESET: 7 duplicate keys, 183 null keys, **246 rows are section/category headings, not occupations** (excluded), 71 rows have real titles but an openpyxl-imprecise float key. MAPPINGS: 107 duplicate keys (resolved last-row-wins). 79 orphans remain after exclusions, and **every one is individually explained**: 8 are the Armed Forces rows from C3 (whose CODESET key doesn't even match their MAPPINGS key), 71 are the float-precision rows (rounding does not recover a match — the precision loss originates in the source file itself, not in the read). Nothing here is an unexplained residual. |
| C7 | Encoding and directionality | **PASS** | 20-row spot check per RTL/non-Latin language: 100% of alphabetic characters in the correct Unicode block for Arabic, Urdu, and Hindi |
| C8 | Whitespace and case normalisation | **PASS (deferred to query time, by design)** | Parser applies `.strip()` on ingestion, matching production's `.strip()` call. Production additionally `.lower()`s at query time — deliberately **not** applied here, so the parsed dataset preserves original casing for later case-sensitivity testing. **This means Week 2's evaluation harness must apply `.lower()` itself before querying the classifier** — added explicitly to the Week 2 brief's input checklist (it had been implicit and unlisted before this review). |
| C9 | Cross-tab presence | **PASS, with a real precision/coverage trade-off, not a simple "present and usable"** | See §5.4 |

## 4. Dataset as parsed

| Language | Titles retained | Distinct 4-digit unit groups covered | % of WISCO's own 436 |
|---|---|---|---|
| English | 4,230 | 436 | 100.0% |
| Arabic | 4,167 | 436 | 100.0% |
| Urdu | 3,989 | 436 | 100.0% |
| Hindi | 4,202 | 436 | 100.0% |
| Tagalog | 4,172 | 436 | 100.0% |
| **Total (any language)** | **4,232** | **436** | **100.0%** |

Every unit group covered by any language is covered by all five (this is a translated dataset —
translators worked from the same source list — so coverage doesn't vary by language, only
per-title fill rate does; this was measured directly, not assumed, including checking whether
Arabic's larger gap versus the other languages drops any unit group to zero members — it
doesn't).

## 5. Unit-group coverage vs. our system — reframed

**The original version of this report undersold this finding by framing it as "14 unit groups
possibly missing from our system." Run the other way: official ISCO-08 has 436 unit groups.
WISCO — independently, as a translated/coded academic dataset — also has 436. Our system's
knowledge base has 441, of which 19 are not in WISCO's coverage.** Since WISCO's count matches
the official standard exactly, the 19 "extra" codes are the more serious half of this finding:
they are very likely codes that don't exist in the real ISCO-08 standard, sitting in a live
retrieval index that a real classifier can return to a real survey respondent. This is a defect
in `backend/rag/load_full_isco.py`, and belongs in Chapter 3 (which currently states 441 unit
groups — the thesis's own proposal says ~436, and `load_full_isco.py`'s own module docstring
says 436 while its `_UNIT` list loads 441; three numbers in three places, only one of which is
right) and in Chapter 6, not a follow-up ticket outside Phase II's scope.

### 5.1 The 19 codes in our system that WISCO's (== official) coverage doesn't have

| Code | Our system's label |
|---|---|
| 1347 | Professional Services Managers, NEC |
| 6124 | Apiarists and Sericulturists |
| 6141 | Forestry Workers |
| 6142 | Charcoal Burners and Related Workers |
| 6150 | Aquaculture Workers |
| 6161 | Subsistence Crop Farmers |
| 6162 | Subsistence Livestock Farmers |
| 6163 | Subsistence Mixed Crop and Livestock Farmers |
| 6164 | Subsistence Fishers, Hunters, Trappers and Gatherers |
| 7116 | Other Building Frame and Related Trades Workers |
| 9131 | Domestic Housekeepers |
| 9132 | Restaurant Services Workers |
| 9141 | Building Caretakers |
| 9151 | Messengers, Package Deliverers and Luggage Porters |
| 9152 | Doorkeepers and Related Workers |
| 9153 | Vending Machine Operators and Related Workers |
| 9161 | Refuse Workers |
| 9162 | Sweepers and Related Labourers |
| 9420 | Street and Related Service Workers |

### 5.2 The 14 codes WISCO has that our system doesn't

| Code | WISCO's label |
|---|---|
| 6210 | Charcoal burner |
| 6221 | Farm worker - oyster |
| 6222 | Captain coastal waters fishing boat |
| 6223 | Captain deep-sea fishing boat |
| 6224 | Seal hunter |
| 6310 | Vegetable farmer (subsistence farming) |
| 6320 | Livestock farmer (subsistence farming) |
| 6330 | Farm worker - subsistence farming |
| 6340 | Subsistence fisher |
| 7119 | Asbestos remover |
| 9622 | Odd-job worker |
| 9623 | Coin meter collector |
| 9624 | Water carrier |
| 9629 | Amusement park attendant |

### 5.3 One specific, verified root cause: the subsistence-farming cluster is under the wrong sub-major group

Comparing §5.1 and §5.2 by content, not just by code, surfaces a concrete bug rather than a
diffuse gap: our system's `6161`–`6164` ("Subsistence Crop/Livestock/Mixed/Fishers Farmers") are
the same occupational content as WISCO's `6310`–`6340` ("subsistence farming" titles). ISCO-08's
real structure places "Subsistence Farmers, Fishers, Hunters and Gatherers" under **sub-major
group 63**, not under sub-major group 61 ("Market-oriented Skilled Agricultural Workers", where
our system currently nests them as a fabricated minor group "616" that doesn't exist in the
official standard). **This is a specific, actionable bug report for `load_full_isco.py`: the
entire subsistence-farming/fishing cluster needs to move from 616x to 63xx.** The remaining 15
of 19 and 10 of 14 codes don't show as clean a 1:1 correspondence by label and would need a full
cross-check against the ILO's official ISCO-08 structure document to resolve individually —
that full cross-check is out of Module A's scope but is now a well-scoped, concrete task rather
than an open-ended one.

### 5.4 Module D industry-crosswalk source — two sources, not one

The original report recommended treating `OCC>>INDUSTRY` as "present and usable" for Module D
without qualification. On investigation, prompted by review feedback that `OCC>>INDUSTRY` is
NACE **2004** (Rev.1.1, one generation behind ISIC Rev.4) while `MAPPINGS.NACE2.0` (Rev.2, maps
cleanly to ISIC Rev.4 at class level) sits unused:

| Source | Sheet | Coverage | Precision | |
|---|---|---|---|---|
| `NACE2.0` | MAPPINGS | 16.7% of rows (745/4,469) | NACE Rev.2 — single hop to ISIC Rev.4 | Sparse but clean |
| `NACE2004_01..33` | OCC>>INDUSTRY | 100% of rows | NACE Rev.1.1 — needs an extra hop | Comprehensive but one generation behind |

**These are not redundant measurements of the same fact.** Where both exist for the same
occupation, they agree on only **26.1%** of rows (checked directly, not assumed) — `NACE2.0`
appears to encode a single "canonical" industry per occupation, while the `OCC>>INDUSTRY`
cross-tab encodes a broader "this occupation could appear in any of these industries" list.
Switching wholesale to `NACE2.0` (as a literal reading of "use the cleaner-hop source" would
suggest) would silently drop coverage from 100% to 16.7% — worse than the status quo. **Both
are extracted, separately tagged, into `wisco_industry_crosswalk.json`
(`nace2_0_rev2` / `nace2004_rev1_1`)**, so Module D can choose per its own precision/coverage
need rather than this report choosing for it. Using `OCC>>INDUSTRY` triggers a second citation
obligation — see `PROVENANCE.md`.

## 6. The 8 quarantined/orphaned Armed Forces rows — full diagnosis

Keys `110010000000`, `110020000000`, `210010000000`, `310020000000`–`310060000000` all have
master labels unambiguously in ISCO-08 major group 0 ("Commissioned officer armed forces",
"Military weapons specialist", "Special forces crew member", ...), but their MAPPINGS
`ISCO0801`–`0804` columns are internally consistent with each other while claiming major group
1/2/3 — not a zero-padding artefact, a genuine tagging error in WISCO's own MAPPINGS sheet.
**Additionally** — found only while chasing this down — their CODESET key
(e.g. `1100100000008420`) doesn't match their MAPPINGS key for the same occupation
(`110010000000`) at all; the two sheets disagree on this occupation's identity, not just its
code. A theory that the join key's own leading 4 digits could recover the correct ISCO code
(documented in the new `CODING RULES` sheet as the key's intended structure) was tested on a
2,000-row sample: 98.35% agreement with `ISCO0804`, but it disagreed for exactly the cases that
matter, including known-correct rows like "Air force captain." **Not used as a correction
source.** These 8 rows are quarantined/orphaned, not guessed at.

## 7. What changed from the original version of this report

| # | Original claim | Correction |
|---|---|---|
| 1 | Used file `..._20230202.xlsx` | That was the oldest of 4 Zenodo versions. Canonical file is `..._20230818.xlsx` (version DOI 10.5281/zenodo.8262593). All numbers in this report are from the canonical file. |
| 2 | "14 unit groups possibly missing from our system" | Reframed: 19 non-standard codes in our own retrieval index is the more serious half of this finding; a specific root cause (subsistence-farming cluster under the wrong sub-major group) was identified and verified. |
| 3 | `OCC>>INDUSTRY` "present and usable" for Module D without qualification | Two non-interchangeable sources exist (`NACE2.0` sparse/clean, `NACE2004` comprehensive/one-generation-behind); both are now extracted separately rather than one being silently preferred. |
| 4 | `ar_AE` chosen for its "Gulf-dialect-flavoured" text | Tested directly: 0/21 other Arabic columns differ from `ar_AE` in any row. WISCO's Arabic data has zero dialectal content. Week 9's dialect-normalisation test cannot run against WISCO as planned. |
| 5 | `wisco_structure_inspection.md` §5 said `fr` has 26 variants (table said 27) | Both now correctly say 28 (canonical file) — the arithmetic (sum of all locale counts = 231) was used to catch and fix this class of error going forward. |
| 6 | "5-digit gold-mapped precision" (this report, the language mapping note, and other Phase II planning documents) | ISCO-08 has no 5-digit level; 4-digit (`ISCO0804`) is the finest precision this workbook provides. Corrected everywhere in this document set; added to the Week 14 audit list (§9) for any remaining references elsewhere. |
| 7 | PROVENANCE.md claimed DOI 7871194 "does not resolve to a WISCO record" | It does — it's another version in the same chain (published 2023-04-25). Claim removed; the correct citation obligation (including a second, previously-missed citation for `OCC>>INDUSTRY` use) is documented in `PROVENANCE.md`. |
| 8 | C6 reported duplicate/orphan counts without diagnosing them | Every orphan and duplicate is now individually explained (§6, and `wisco_structure_inspection.md` §6) |

## 8. Not previously captured — now addressed or logged

- **Pinned environment.** `backend/evaluation/wisco/requirements.lock.txt` added (`pip freeze` output for this module's dependencies: `openpyxl`, plus the shared repo dependencies it imports from `backend.rag.load_full_isco`). This work still exists only on one machine and is uncommitted — see the outstanding repository-state note carried over from the Phase 1 audit.
- **ISCO08lv (skill level).** Now extracted into `wisco_raw_parsed.json` (`isco08_skill_level` field, values 1–4). Distribution: level 1=261, level 2=2,149, level 3=1,073, level 4=986. Reserved for the Week 7 ISCO↔ISCED cross-check.
- **CC-BY-4.0 / SERISS attribution.** Both citation strings are now logged in `PROVENANCE.md` and need to be added to the repo README and thesis — tracked as an open item, not yet done in the README (see the revised Week 2 brief).
- **C8's lowercase-at-query-time requirement.** Explicitly added to the Week 2 evaluation-harness input checklist (see the revised Week 2 brief) — it was correct in principle before but not written down anywhere a Week 2 implementer would see it.
- **Week 3 compute estimate.** 4,232 titles × 5 languages = 21,160 classification calls through a 4-stage RAG pipeline with conditional LLM re-ranking, inside one week. See the revised Week 2 brief for the cost/wall-clock estimate and stratified-subsample recommendation.

## 9. Week 14 audit list additions

- Remove any remaining "ISCO-08 5-digit" / "5dgt gold-mapped precision" claims across Phase II
  planning documents (this report and the language mapping note are already fixed; other
  planning documents referenced during review — "Phase II Comprehensive §4.1", "Summary",
  "Week 3 plan" — were not available to check in this pass and should be swept during the Week
  14 audit alongside the already-known ISIC-419 and ISCED-80 corrections).
- Confirm the second Zenodo DOI (7871194) is not cited anywhere as "does not resolve" — it does
  resolve, and the correction is in `PROVENANCE.md`.
- Verify `load_full_isco.py`'s subsistence-farming code fix (§5.3) has landed before citing
  "441 unit groups" (or whatever the corrected count becomes) in Chapter 3.

## 10. Inputs ready for Week 2

- [x] `wisco_raw_parsed.json` passing all 9 checks, against the canonical file
- [x] `wisco_industry_crosswalk.json` — both NACE sources, separately tagged
- [x] `language_mapping_note.md` — all 61 base codes dispositioned, Arabic finding corrected
- [x] Per-language retained counts (§4)
- [x] Provenance record complete, including the version correction and both citation obligations
- [x] Pinned environment file

Week 2 can proceed, with two plan changes carried into the revised Week 2 brief: the Module D
source decision (§5.4) and the Week 9 dialect-test rescoping (§6 of `language_mapping_note.md`).
