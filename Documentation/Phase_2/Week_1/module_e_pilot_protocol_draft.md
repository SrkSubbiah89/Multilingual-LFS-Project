> **STATUS: DRAFT, NOT SUBMITTED, NOT REVIEWED.** This is a starting document
> for you to take to Dr. Mali and, through them, to the IIIT Kottayam IEC —
> not a finished protocol and not evidence of ethics progress in its own
> right. Every fact below is either (a) a real, already-decided project
> parameter, sourced from `Documentation/Phase_2/Week_1/
> ethics_submission_log.md`'s own Section 5, or (b) standard human-subjects
> protocol structure. Every institution-specific fact this project does not
> yet have (IEC-specific forms, committee dates, your supervisor's own
> requirements) is marked `<FILL — confirm with Dr. Mali>` rather than
> guessed. Do not submit this as-is; it exists so the first submission draft
> isn't a blank page.

---

# Module E Pilot — Draft Protocol

> Location: `Documentation/Phase_2/Week_1/module_e_pilot_protocol_draft.md`
> Companion to: `ethics_submission_log.md` (tracking) and
> `Documentation/Phase_2/FINAL_RESULTS_PACKAGE.md` (which this pilot's
> outcomes are meant to feed, per that document's own Table 6.5).

## 1. Title

Multilingual Conversational AI vs. Traditional Interview for Labour Force
Survey Data Collection: A Parallel-Arm Pilot Study

`<FILL — confirm exact title with Dr. Mali; this is a working title>`

## 2. Background and rationale

Labour Force Surveys collect occupation, industry, and education data used
to derive official employment statistics. In multilingual settings, survey
quality depends on accurate real-time classification of free-text responses
into standard schemes (ISCO-08 for occupation, ISIC Rev.4 for industry,
ISCED 2011 / ISCED-F 2013 for education). This project has built and
internally tested a conversational AI system (FastAPI + CrewAI backend,
Next.js frontend) that conducts LFS interviews in five languages (English,
Arabic MSA, Gulf Arabic dialect, Urdu, Hindi, Tagalog) and classifies
responses using retrieval-augmented generation against official
classification catalogues, cross-validated by a Semantic Relation Engine.
Internal evaluation against WISCO (a large occupation-coding benchmark) has
produced real, published accuracy figures — see CLAUDE.md's WISCO Tier-1
result — but **no evaluation against real human respondents has yet been
conducted.** This pilot is the first planned real-respondent evaluation.

## 3. Objectives

**Primary**: compare the AI-conducted interview against a traditional
(human-interviewer) LFS interview on:
- Completion time per interview
- Item nonresponse rate
- Internal contradiction rate (responses the Semantic Relation Engine or
  Validation Agent would flag)
- Respondent satisfaction (CSAT)
- Per-interview cost

**Secondary** `<FILL — add if Dr. Mali wants secondary outcomes, e.g.
classification accuracy against a human-coder gold standard for this
specific sample, or per-language subgroup comparisons>`.

## 4. Design

Parallel-arm randomised pilot. n=30 total (15 AI-conducted arm, 15
traditional-interviewer arm) — already decided per
`ethics_submission_log.md` Section 5.

Randomisation method: `<FILL — e.g. simple randomisation via a random
number generator, block randomisation by language>`.
Randomisation performed by: `<FILL — must not be the person running
sessions, per ethics_submission_log.md's own existing note>`.

## 5. Participants

**Inclusion criteria**: `<FILL — e.g. adults 18+, currently or recently
employed or seeking work, fluent in one of the five supported languages>`.

**Exclusion criteria**: `<FILL — e.g. inability to give informed consent,
no access to a device/network for the AI arm>`.

**Recruitment**: `<FILL — where/how participants will be found; consider
whether this needs its own IEC sign-off separate from the interview
protocol itself>`.

**Incentive**: `<FILL — or "none">`.

## 6. Procedures

1. Consent obtained (see Section 8) in the participant's preferred
   language among the five supported.
2. Participant randomised to AI arm or traditional-interviewer arm.
3. **AI arm**: participant completes the LFS interview via the system's
   chat interface (web, `/chat`). An AI disclosure statement is shown
   before the interview begins (see Section 9). A human-interviewer
   alternative is offered at any point if the participant wants to switch
   — `<FILL — confirm this offer is acceptable as the switch/withdrawal
   mechanism, or specify a different one>`.
4. **Traditional arm**: participant completes the same core LFS questions
   via a human interviewer, following the same question set the AI arm's
   `ConversationManager` FSM enforces (see
   `Documentation/Conference_I_Reviewer_2/` for the FSM's real, tested
   question ordering) so the two arms are comparable.
5. Post-interview: CSAT collected from both arms via the same short
   instrument. `<FILL — attach/reference the actual CSAT instrument once
   drafted>`.
6. Completion time, item nonresponse, and contradiction flags are logged
   automatically for the AI arm (the system already logs these — see
   `agent_decision_logs`, `hitl_queue`, and `quality_reviews` tables in
   `backend/database/models.py`) and recorded manually by the traditional
   arm's interviewer using a matching log template
   `<FILL — build this template if not already built>`.

## 7. Risks and benefits

**Risks**: `<FILL — e.g. minimal risk beyond a standard interview; note
any sensitivity around occupation/income-adjacent questions;
data-confidentiality risk if de-identification fails>`.

**Benefits**: no direct benefit to participants beyond any incentive
offered; broader benefit is methodological evidence for multilingual LFS
data collection.

## 8. Consent process

Informed consent obtained before randomisation, in the participant's
chosen language. Consent form and Participant Information Sheet (PIS)
status: `<FILL — see ethics_submission_log.md Section 4 for per-language
translation tracking; this protocol assumes those exist before
recruitment starts>`.

Withdrawal procedure: `<FILL — how and to whom a participant withdraws,
and what happens to already-collected data>`.

## 9. AI disclosure

Participants in the AI arm must be told, before the interview starts,
that they are interacting with an AI system, not a human interviewer, and
must be offered a human-interviewer alternative — this is a real,
already-decided project commitment (see `ethics_submission_log.md`
Section 3's "AI disclosure statement" row). Exact wording:
`<FILL — draft the actual disclosure text per language>`.

## 10. Data management

**What is collected**: interview responses, classification outputs
(ISCO/ISIC/ISCED codes and confidence scores), completion time, CSAT
score. `<FILL — confirm whether raw audio/video is collected at all; this
project's system as built is text-chat only, so likely no>`.

**Storage location**: `<FILL>`.
**Retention period**: `<FILL>`.
**Destruction schedule**: `<FILL>`.
**De-identification method**: `<FILL — how participant identity is
separated from response data; note the system's real `person_register`
table already exists and its data-handling constraints are documented in
CLAUDE.md's "Knowledge base construction" / Module F notes — confirm this
protocol's data handling is consistent with that existing constraint
("real register data must not leave SCAD")>`.

## 11. Analysis plan

Primary outcomes compared between arms using `<FILL — e.g. two-sample
t-test or Mann-Whitney U for completion time; chi-square or Fisher's exact
for nonresponse/contradiction rates, given the small n=30>`. Given n=30
(15/15), this is explicitly a **pilot** — underpowered for definitive
inference, intended to produce feasibility evidence and effect-size
estimates for a future, larger study, not a conclusive comparison. State
this limitation explicitly in any write-up of results.

## 12. What this protocol does NOT cover

- ISCO/ISIC/ISCED-F classification accuracy against a human-coder gold
  standard for this specific pilot sample — out of scope unless added as
  a secondary outcome (Section 3).
- Any claim that this pilot substitutes for the synthetic ISIC/ISCED-F
  benchmark work already documented in CLAUDE.md, or vice versa — they
  measure genuinely different things (real respondents vs. LLM-generated
  text) and neither substitutes for the other.

## 13. Open items before this can be submitted

- [ ] Confirm title, objectives, and secondary outcomes with Dr. Mali.
- [ ] Identify IIIT Kottayam IEC's actual submission requirements, forms,
      and meeting calendar (`ethics_submission_log.md` Section 2 — the
      single highest-priority open item in this whole project per that
      file's own header).
- [ ] Draft PIS/consent forms in all 5 languages, verified by a native
      speaker per language.
- [ ] Fill every `<FILL>` above.
- [ ] Have Dr. Mali review this full draft before it goes anywhere near
      the IEC.
