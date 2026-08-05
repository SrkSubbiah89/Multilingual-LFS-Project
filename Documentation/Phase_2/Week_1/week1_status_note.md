> **Partially filled, and revised same-day after external review.** The Module A (WISCO)
> paragraph below reflects the corrected results — the original draft of this note (and of
> Module A itself) used the wrong WISCO file version; both have been fixed, see
> `module_a_week1_report.md` §7 for the full change list. The Module E paragraph, and the
> `From`/`Date` header fields, are left as `<FILL>` because no ethics submission has happened —
> there is no real committee reference, submission date, or decision timeline to report, and
> none will be invented for a note that goes to your supervisor. Fill those in once Module E is
> actually underway.

---

# Week 1 Status Note — to Supervisor

> Send Day 5. Keep it to one screen. The supervisor needs three things: whether the schedule
> still holds, what the dataset looks like, and anything requiring a decision.

---

**To:** Dr. Goutam Mali
**From:** `<FILL>`
**Date:** `<FILL>`
**Subject:** Phase II Week 1 — Module A complete and corrected; Module E not started

Dear Dr. Mali,

**Ethics (Module E) — the item that actually matters this week.** No application has been
submitted. `<FILL — once real: submission date, body, reference number, next committee sitting
date, expected decision date, whether this keeps recruitment on track for Week 5 or puts it
behind schedule, how many languages the consent materials were submitted in.>` The plan assumes
recruitment starts Week 5, which assumes committee approval by roughly Week 4. I don't yet know
the committee's sitting date — that is the single fact the 14-week schedule's realism depends
on, and establishing it is a one-day task I'm prioritising ahead of further Module A polish.

**WISCO dataset (Module A) — complete, and corrected same-day.** The dataset originally
downloaded (DOI 10.5281/zenodo.7598568) turned out to be the oldest of 4 published versions;
the correct one is version DOI 10.5281/zenodo.8262593 (file `..._20230818.xlsx`), confirmed by
matching the exact filename the original plan expected. All of Module A's analysis was redone
against the correct file the same day this was caught. Corrected parsing retains 4,232 clean
occupation records:

| Language | Titles retained |
|---|---|
| English | 4,230 |
| Arabic | 4,167 |
| Urdu | 3,989 |
| Hindi | 4,202 |
| Tagalog | 4,172 |

All nine integrity checks pass. Two things were worth catching, not just passing: the source
stores both `ISCO0804` and `NACE2.0` as plain integers, silently dropping leading zeros (fixed,
verified on real data — e.g. all 33 Armed Forces codes now correctly zero-padded); and 8
occupation rows turned out to be doubly broken in WISCO itself (wrong major-group tagging *and*
a CODESET/MAPPINGS key mismatch) — quarantined, not guessed at.

**Findings that changed the plan, not just the numbers.**
- **The 441-vs-436 ISCO-08 unit-group gap is our own defect, not a WISCO shortfall.** WISCO's
  436 matches the true ISCO-08 standard exactly; our system's knowledge base has 441, of which
  19 don't exist in the standard — meaning our classifier can return invalid codes to a real
  respondent. I traced one specific, verified cause: our subsistence-farming occupations
  (codes 6161–6164) are filed under the wrong sub-major group and should be 6310–6340. This
  needs to go into Chapter 3 and Chapter 6, not a follow-up ticket — Chapter 3 currently claims
  441, the proposal says ~436, and `load_full_isco.py`'s own docstring says 436 while loading
  441. An examiner who knows ISCO-08 will catch this.
- **WISCO's Arabic data has zero dialectal content — checked directly, not assumed.** All 22
  Arabic country-locale columns are the same MSA translation, byte-for-byte identical across
  4,168 rows in every case but the two I diffed and found nothing. This means the planned Week
  9 dialect-normalisation A/B test cannot run against WISCO as scoped — there is nothing for a
  dialect normaliser to normalise in translated, professionally-curated text. Week 9 needs a
  different data source or a redefined experiment; flagging now rather than reporting a
  meaningless null result in Week 9.
- **Module D's industry-crosswalk source isn't a single pick.** `OCC>>INDUSTRY` (NACE 2004,
  Rev.1.1) has 100% coverage but is one classification generation behind ISIC Rev.4; `NACE2.0`
  (Rev.2, in MAPPINGS) maps cleanly to ISIC Rev.4 but only covers 17% of rows, and the two
  sources agree on just 26% of rows where both exist. Both are now extracted separately for
  Module D to choose from, rather than the parser silently picking one.
- Using `OCC>>INDUSTRY` triggers a second citation obligation (a 2017 SERISS deliverable,
  separate from the main WISCO citation) — logged in `PROVENANCE.md`, not yet added to the
  README.

**Decisions or guidance needed.**
- None from Module A. Everything from Module E, starting with the committee sitting date.

Week 2 builds the per-language test sets and the evaluation harness, now incorporating the
Module D two-source design and the Week 9 rescoping above.

Best regards,
`<FILL>`
