> **STATUS: CLOSED — definitive negative answer, verified 2026-08-27.** IPUMS International
> does not have access to, and cannot distribute, original verbatim occupation/industry/
> education write-in text for any of its samples, including the Arabic (Egypt/Jordan), Hindi
> (India), and Urdu (Pakistan) samples this project asked about specifically. This closes off
> IPUMS as a source of real, externally-sourced multilingual text for ISIC/ISCED-F validation.
> It does **not** invalidate the synthetic-data work in `eval/generate_synthetic_isic_iscedf_
> benchmark.py` — see CLAUDE.md's "Knowledge base construction" log for how that work is scoped
> and disclosed. The one remaining path to real (non-synthetic) ISIC/ISCED-F validation data is
> Module E (the pilot) or a national LFS microdata release with public-use write-in text, neither
> of which this correspondence provides.

---

# IPUMS International correspondence — verbatim occupation/industry/education text

> Location: `Documentation/Phase_2/Week_1/ipums_correspondence_log.md`
> Correspondent: Isabel Pastoor, IPUMS User Support (ipums@umn.edu)
> Sender: Sivarama Krishnan Subbiah (sivarama.24wp1107@iiitkottayam.ac.in)
> Referenced from: CLAUDE.md ("Knowledge base construction" — the ISIC/ISCED-F synthetic
> benchmark entry) and `Documentation/Conference_I_Reviewer_2/
> ISIC_ISCEDF_HIERARCHICAL_RETRIEVAL_IMPLEMENTATION.md`.

## Why this correspondence happened

WISCO (the dataset behind every ISCO-08 accuracy number in this project) is occupation-only —
it carries no industry (ISIC) or field-of-study (ISCED-F) gold labels. IPUMS International was
investigated as a possible source of real, externally-sourced multilingual industry/education
text that could serve the same role for ISIC/ISCED-F that WISCO serves for ISCO-08, specifically
for three of this project's five languages: Arabic (Egypt/Jordan samples), Hindi (India), and
Urdu (Pakistan).

## 1. Outgoing query — 2026-08-24, 2:45 PM

Sent to `ipums@umn.edu`. Asked, for the Egypt/Jordan (Arabic), India (Hindi), and Pakistan (Urdu)
samples specifically: does the distributed microdata (or an available extract option) include
the original verbatim/write-in text response an enumerator recorded for the occupation and/or
industry questions, in the original survey language — or are only final coded values (`OCC`,
`IND`, `OCCISCO`, and similar harmonized variables) ever distributed, with source text retained
only internally by the national statistical office and never released? The query noted that
IPUMS USA's `OCCSTR` verbatim-text variable appears to exist only for historical 1850–1930
samples, suggesting modern samples may not retain source text at all.

## 2. First reply — 2026-08-25, 8:31 PM (Isabel Pastoor, IPUMS User Support)

Key points, transcribed from the received email:

- "IPUMS International generally does not have access to the original string variables for
  occupation, industry, or education from the original data providers. Instead, we receive
  coded variables. We sometimes are able to perform additional classifications (e.g., we do
  create OCCISCO based on OCC) but the primary coding of strings is performed by original data
  providers, usually national statistical agencies, before we receive the data."
- "Our agreements with IPUMS International data providers generally preclude us from
  disseminating data outside the IPUMS International data extract system. Unfortunately I do
  not believe it would be possible to provide you with any original string variables even if we
  did have them ourselves, but I have sent a message to my colleagues on the IPUMS International
  team to inquire."
- Suggested reaching out to the original national data providers directly to request access to
  unaltered string variables.
- On IPUMS USA specifically (not this project's use case, but noted for completeness): historical
  full-count U.S. census datasets, including restricted versions, do include many original string
  variables via digitization/transcription of historical enumeration forms. Modern U.S. census/
  survey data is processed differently, and it was not known whether the Census Bureau retains
  original string responses for those.

## 3. Follow-up reply — 2026-08-25, 10:59 PM, same day (Isabel Pastoor)

After consulting the IPUMS International team directly:

- "I spoke with a colleague on the IPUMS International team and unfortunately we do not have
  access to any original string variables for these samples. While we very occasionally have
  string variables available in some samples in our internal versions of the data, we are not
  able to share these with IPUMS users because of our agreements with original data providers."
- "I'm sure this is not the answer you were hoping for."

## 4. Conclusion — verified 2026-08-27

This is a definitive, sourced negative finding, not an unanswered inquiry — the same class of
result as Module D's ISCO-08↔ISIC crosswalk search (a real, complete document was checked and
found not to contain what was needed, rather than never checked). IPUMS International:

1. Does not hold original-language verbatim occupation/industry/education text for the Egypt/
   Jordan, India, or Pakistan samples (or, per Isabel Pastoor's colleague, essentially any sample).
2. Could not distribute it even where it exists internally, due to data-provider agreements.
3. Directed further inquiry to the original national statistical agencies — a real but separate,
   much higher-effort path (direct requests to Egypt/Jordan CAPMAS-equivalent, India's NSSO/MoSPI,
   and Pakistan's PBS) not pursued further in this pass.

**This closes the IPUMS avenue for this project.** It does not change the status of the synthetic
ISIC/ISCED-F benchmark (`eval/generate_synthetic_isic_iscedf_benchmark.py`), which was always
disclosed as a distinct, non-substitute methodology — see CLAUDE.md. The two remaining paths to
real (non-synthetic) ISIC/ISCED-F evaluation data are: (a) Module E, the n=30 pilot, if its
consent/data-collection scope is extended to capture industry/education alongside occupation, or
(b) a direct approach to one of the three national statistical agencies named above, which was not
attempted in this pass and would need its own time budget and, likely, its own ethics
consideration given it is a distinct new data-access request.
