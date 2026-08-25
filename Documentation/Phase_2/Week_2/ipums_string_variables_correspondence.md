# IPUMS correspondence — original occupation/industry/education string variables

> Location: `Documentation/Phase_2/Week_2/ipums_string_variables_correspondence.md`
> Recorded: 2026-08-25, from a direct email reply received from IPUMS staff.
> **Why this was asked at all:** `language_mapping_note.md` §4.1 (Week 1) found that WISCO's
> Arabic titles are translations of a single English reference source, not naturally-occurring
> respondent text, and that this makes the Week 9 dialect-normalisation A/B test (Module G)
> impossible to run against WISCO as scoped — it needs "a different data source with genuine
> Gulf-dialect text" (real respondent free-text, not curated/translated titles). IPUMS
> (International and USA) was approached as one candidate source of real, original-language
> occupation/industry/education free-text responses. This note records what came back, so the
> next person evaluating data sources for Module G doesn't re-ask the same question.

## What was asked

Whether IPUMS could provide the **original string variables** (the raw free-text responses, in
the original language, before coding) underlying occupation, industry, and education variables
— for IPUMS International and/or IPUMS USA extracts.

## What IPUMS said (verbatim reply, summarised below)

**IPUMS International:**
- IPUMS International generally does **not have access to the original string variables** for
  occupation, industry, or education from the original national data providers. It receives
  already-**coded** variables.
- IPUMS International does sometimes perform *additional* classification on top of what it
  receives (example given: `OCCISCO` is derived from `OCC`), but the **primary coding of raw
  strings into codes is done by the original data providers** (national statistical agencies),
  before IPUMS ever receives the data.
- IPUMS International's **data-sharing agreements with providers generally preclude
  redistributing data outside the IPUMS International extract system** — so even in a
  hypothetical case where IPUMS did hold original strings, dissemination outside the extract
  system would likely not be possible.
- IPUMS staff had, at the time of the reply, escalated an inquiry internally to the IPUMS
  International team to double-check this — **response pending**, not yet received as of this
  recording.
- Direct suggestion from IPUMS: **contact the original national data providers directly** to
  request access to unaltered string variables.

**IPUMS USA:**
- IPUMS USA does provide **historical full-count U.S. census datasets**, including restricted
  versions, that **do include many original string variables** — but only because historical
  U.S. census data were made available via **digitization and transcription of the original
  enumeration forms**.
- **More modern U.S. censuses/surveys are processed differently**, and IPUMS staff explicitly
  said they don't know whether the Census Bureau itself retains any record of the original
  string responses (e.g. for occupation/industry/education) for modern data collection.

## Why this is a dead end for Module G specifically — not just "no data"

Two independent reasons, not one:

1. **Coverage never existed in the first place.** IPUMS International — the arm that would
   plausibly carry non-English, multi-country occupation data resembling this project's 5
   target languages — was told directly it generally never had the original strings to begin
   with; national statistical agencies code them before IPUMS receives anything. This isn't a
   licensing restriction sitting on top of real data; there is likely no such data to license.
2. **Even a hypothetical positive answer would still fail the redistribution requirement.**
   IPUMS International's own agreements with providers preclude redistributing data outside its
   extract system — so even if the pending internal follow-up comes back positive, the practical
   path would be "query it inside IPUMS's own extract system," not "obtain a redistributable
   dataset to check into this repository or otherwise integrate directly."
3. **The one real archive of original strings IPUMS does hold (IPUMS USA historical full-count
   census) doesn't fit this project's actual need.** It's English-only, U.S.-only, and
   historical (transcribed paper forms) — it has no bearing on genuine Gulf-Arabic, Urdu, Hindi,
   or Tagalog dialectal occupation text, which is what Module G's dialect-normalisation A/B test
   specifically needs.

## Disposition

**IPUMS (International or USA) is not a viable source of genuine, redistributable,
multilingual respondent free-text for occupation/industry/education**, closing off one
candidate alternative data source raised in response to the Week 1 finding above.

This does **not** change Module G's status in `CLAUDE.md` — it was already "unchanged: needs a
different data source or a redefined experiment," and remains exactly that. It does narrow the
realistic remaining options to what `language_mapping_note.md` §4.1 already named:
- synthetic Gulf-dialect job-title variants, or
- real respondent free-text captured during the pilot study, once Module E (currently not yet
  started — see `CLAUDE.md`'s Module E status) is actually underway.

**Follow-up action, not yet closed:** IPUMS's internal escalation to their International team
was still pending at the time of this reply. If a further reply arrives, it should be appended
here rather than starting a new file, and `CLAUDE.md`'s Module G line updated only if that reply
actually changes the disposition above (i.e., if it turns out original strings *do* exist
somewhere in the IPUMS International pipeline after all — recorded provenance here says this is
unlikely, but not yet fully ruled out).

**Suggested next step if this is pursued further:** contact original national statistical
agencies directly (IPUMS's own suggestion) for the specific countries feeding the project's 5
target languages, rather than continuing to pursue this through IPUMS itself.
