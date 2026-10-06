# Thesis LaTeX — Verification Notes

Tracks the rewrite of the thesis draft found in `Multilingual_Conversational_AI_LFS.zip`
(dated 2026-05-02, predates almost all of this project's own verification work).
Original files kept as `*_ORIGINAL_UNVERIFIED.tex` for reference — **do not compile
or submit those**. Corrected files have no suffix. Full citation-by-citation audit
is in `CITATION_AUDIT.md`.

## Why this rewrite exists

A full scan of the original draft found its entire quantitative core fabricated
or stale, most seriously:
- **85.3% ISCO-08 top-1 accuracy** — this is *literally* the fabricated Phase I
  baseline CLAUDE.md's own provenance section already documents finding and
  correcting once (`Documentation/Phase_2/PHASE_II_PLAN_CORRECTIONS.md`). Real,
  best-tested number: **40.95%** (flat retrieval + enriched catalogue + e5-large).
- **"Hierarchical RAG outperformed flat RAG by 6.4pp"** — the exact opposite of
  the real, published, statistically decisive finding: hierarchical
  **underperforms** flat by 10.84pp (McNemar p≈1.86×10⁻³⁰¹).
- **A full set of fabricated pilot-study results** (Chapter 5 originally) — 390
  ISCO-08 classification events, Cohen's κ=0.87, double-coded by "ILO-trained
  coders" — for a pilot that has never been submitted for ethics approval, let
  alone run. Removed entirely, not edited down.
- **"ILO Geneva 2012 ISCO-ISIC correspondence table"** cited as the SRE's source
  — a real search of the complete 433-page ISCO-08 Vol. I PDF found no such
  table exists anywhere. The crosswalk is hand-built by necessity.
- **441 unit groups** with a fabricated "five supplementary categories from ILO
  Geneva 2012 guidance" footnote with no basis in any source checked. Real,
  verified count: 436.
- **"E1 wage gate" / "F3 not-in-labour-force re-routing"** attributed to
  `ValidationAgent`'s ten rules — checked directly against the live
  `validation_agent.py`: false. The real ten rules (R01–R10) are basic
  hours/employment-status contradiction checks. (E1 and F3 *do* exist for real,
  just as `ConversationManager` FSM skip-gates, not `ValidationAgent` rules —
  see Chapter 3's correction for the precise distinction.)
- **Claude 3.5 Sonnet** presented as actively routing critical tasks throughout
  — real, designed behaviour (`TaskType.CRITICAL`, confirmed in `llm_client.py`),
  but production use was blocked by zero Anthropic credit the whole time.
- A fabricated SRE "demo" appendix whose Arabic explanation field was literal
  untranslated placeholder text (`[AR_TEXT] [AR_TEXT]...`).
- ~16 of ~40 bibliography entries had a real, checkable error: 3 could not be
  matched to any real paper at all; 13 were real papers cited with a wrong
  author, title, year, or a false "Anonymous" attribution. Full detail in
  `CITATION_AUDIT.md`.
- Multiple other unverified accuracy figures (92.1%, 81.4%, 87.3%, 76.8%, 98.2%,
  92.4%) with no matching number anywhere in the verified project record.

## Source of truth for every correction

Two sources, in order of use: (1) `CLAUDE.md` (repo root) — verified directly
against the running code and live services, with its own dated correction
history; (2) the **live source code itself**, read directly where CLAUDE.md
didn't already cover the specific claim (`validation_agent.py`, `llm_client.py`,
`conversation_manager.py`'s own module docstring, `isced_classifier.py`,
`context_memory.py`, `rag_expert.py`, `semantic_relation.py`,
`requirements.txt`, `frontend/`). Every numeric/factual claim in the corrected
chapters traces to one of these two. Where neither source resolves a claim, the
thesis says so explicitly via a `%% VERIFY AGAINST CODE` marker rather than
guessing.

## Citation audit — DONE (2026-08-29)

All ~40 `bib.bib` entries checked via direct web search, one at a time. Full
writeup in `CITATION_AUDIT.md`. Summary: 3 entries could not be matched to any
real paper (`bach2025rag`, `hamed2025isco`, `liu2025agenteval` — commented out
in `bib.bib`, no `\cite{}` to any of them remains anywhere); 13 were real
papers with a wrong author/title/year/false-anonymity (corrected in `bib.bib`
with an inline audit-trail comment on each); ~23 were confirmed accurate as
originally cited. One further, different issue found: `zhang2025synthetic` is
a real paper but was cited for claims (ILO/OECD data scale, item-nonresponse
rates, per-interview cost) its real content — a synthetic-survey-data-generation
methods comparison — doesn't actually support; de-cited from those specific
claims rather than left misattached.

**Verified mechanically**: every `\cite{}` key used across all 8 `.tex` files
resolves to a real, uncommented `bib.bib` entry (checked via script — zero
broken references).

## Status by file — all corrected

| File | Status |
|---|---|
| `report.tex` | **Corrected** — abstract fully rewritten around real Chapter 6 numbers; title/declaration/certificate pages unchanged (already accurate) |
| `chapter1.tex` | **Corrected** — architecture facts, objectives, scope/limitations; UAE-specific framing confirmed and reinstated (was over-cautiously generalised in an earlier pass); citation misattachments fixed |
| `chapter2.tex` | **Corrected** — false "this thesis" claims (441, Agent-N numbering, 85%+, "addresses all gaps") fixed; all citations audited, 3 fabricated ones removed with in-text disclosure |
| `chapter3.tex` | **Corrected** — heaviest original fabrication. Real 13-module agent list (no SurveyOrchestrator, no PersonRegister-as-agent), real R01–R10 *and* the real, separate E1/F3 FSM skip-gates (confirmed both exist, just not where the original draft put them), real `TaskType.GENERAL`/`CRITICAL` behaviour, real SRE formula, real field list (16 all-paths + path-specific fields, sourced from the module's own docstring), no custom HNSW config (verified absent), real frontend component names, hierarchical-vs-flat corrected |
| `chapter4.tex` | **Corrected** — real dependency versions; `camel-tools` confirmed absent, `sendgrid`/`python-jose`/`langdetect` confirmed *present* (with real versions) — both directions checked, not a blanket rewrite; real R01–R10; real test count (2,501) |
| `chapter5.tex` | **Fully rewritten** — removed fabricated pilot-study results entirely; real methodology description (WISCO, synthetic ISIC/ISCED-F benchmark + IPUMS closure + refusal-bug story); the "ILO-certified coders" claim for the legacy 100-case set checked directly against the script and found false (real gold labels are a hardcoded developer-authored list) |
| `chapter6.tex` | **Fully rewritten** — rebuilt entirely from verified numbers: WISCO canonical (21.19%/10.35%), additive experiments (40.95% headline), 6 reranking-null checks, synthetic ISIC/ISCED-F (82.85%/13.14%, contamination and blocked re-run disclosed), Gulf dialect null result, real SRE validation, real computational-efficiency numbers, real objective-status table |
| `chapter7.tex` | **Corrected** — limitations/future-work rewritten around real pilot status and real coverage numbers |
| `appendices.tex` | **Corrected** — real field table (names/codes/labels/values sourced from live `_CORRECTION_FIELD_SCHEMA`, confirms UAE dirham currency); fabricated SRE demo replaced with real validation evidence and the real bilingual explanation string (which itself surfaced a genuine wording bug in the live production code, disclosed as a byproduct finding); YAML "agent configs" marked illustrative (real agents are constructed in Python, not YAML); Appendix D (consent/ethics) was already honest in the original — kept intact |
| `arch_diagram.tex` | **Corrected** — rebuilt with the real 13 modules, real LLM tiers, no hierarchical-process claim, no fictitious LLM providers |
| `bib.bib` | **Corrected** — see Citation audit above and `CITATION_AUDIT.md` |

## A genuine byproduct finding, disclosed for action outside this LaTeX task

While sourcing the real bilingual SRE explanation text for Appendix C, found
that `semantic_relation.py`'s own live `_build_explanations()` method generates
text saying "according to ILO ISCO-ISIC-ISCED crosswalk tables" — which is not
accurate, since Chapter 3 (and the real code search behind it) confirms no such
official table exists; the crosswalk is hand-built. This is a real wording bug
in the production system itself, not just a documentation issue. Flagged in
Appendix C; **not fixed in the actual codebase** as part of this task (out of
scope), but should be corrected in `_build_explanations()` at some point.

## Structural checks performed (no LaTeX distribution available on this machine)

No `pdflatex`/`xelatex`/`lualatex` is installed, so this could not be
compile-tested end-to-end. Three mechanical checks were run instead, across
every corrected file, after the full edit pass:
1. Brace balance — all files pass (`report.tex` through `bib.bib`).
2. `\begin{}`/`\end{}` environment-name matching — all files pass.
3. Every `\cite{}` key resolves to a real, uncommented `bib.bib` entry — zero
   broken references.

This does **not** guarantee a clean compile — it cannot catch undefined
commands, missing packages, or malformed table column specs. **A real
`pdflatex` run (locally after installing MiKTeX/TeX Live, or via Overleaf) is
still the first thing to do the next time this is worked on.**

## What remains genuinely open

1. A real `pdflatex`/Overleaf compile — not done, no LaTeX distribution
   available in this environment.
2. A handful of narrower implementation details, each marked
   `%% VERIFY AGAINST CODE` in the `.tex` source rather than asserted:
   the *exact* full skip-gate count (chapter 3 confirms at least 6 explicit
   gates from the module docstring, but did not exhaustively re-count every
   conditional in the function bodies), exact quick-reply/field UI counts,
   the "8,720+ chunks" / "50+ synonyms" exact figures (the *mechanism* — a
   real keyword pre-filter and real synonym dictionary — is confirmed; the
   exact counts are not independently re-tallied), and the exact spoken
   question wording per field (the field's *label and accepted values* are
   confirmed real; the literal turn-by-turn phrasing was not separately
   sourced).
3. Two low-confidence bibliography entries flagged rather than resolved:
   `reimann2025conveval` (wrong arXiv ID found; a real underlying source may
   exist but wasn't independently located) and `eyolfson2026mas`'s specific
   author (real title now correct; author not independently confirmed).
4. Module E (the pilot) actually starting — nothing in this LaTeX rewrite
   changes that; it remains the single largest gap the thesis discloses, and
   no amount of further LaTeX correction closes it.

## Standing rules for any future edits to this thesis

1. Never state a specific accuracy/count/version number without a direct line
   to CLAUDE.md's verified text or the live source code.
2. Never describe hierarchical retrieval as outperforming flat retrieval.
3. Never describe the Module E pilot as having produced data or results.
4. Never cite the ILO Geneva 2012 correspondence table as existing.
5. Never re-add a `bib.bib` citation that this audit commented out without
   first re-verifying it independently.
6. Where CLAUDE.md itself says something is unresolved/blocked (e.g. the
   694→1091-row synthetic accuracy re-run, blocked on memory), say so plainly
   rather than omitting it or implying it's done.
