# Phase II Comprehensive Plan — Corrections Against Verified Project State

> Produced 2026-08-11. Reviews the "Phase II Comprehensive Plan" document supplied for this
> project against two independent sources of ground truth already in this repository:
> (1) the exhaustively-verified official WISCO Tier-1 evaluation evidence chain (Tasks 36-43,
> `Documentation/Conference_I_Reviewer_2/OFFICIAL_WISCO_TIER1_CONTROLLED_RESULTS.md` and
> related reports), and (2) the **already-completed, real** Phase 2 Module A Week 1/2 work
> (`Documentation/Phase_2/Week_1/`, `Week_2/`), which the supplied plan does not appear to have
> had access to. Every correction below cites its source. Nothing here is inferred or guessed.

## How to use this document

The supplied Phase II plan's structure (Modules A-H, timeline, risk register) is largely sound
and does not need to be rewritten wholesale. What needs fixing are specific factual claims that
contradict this project's own verified evidence. Fix these before the plan — or anything derived
from it — goes into a thesis chapter.

---

## 1. Executive Summary table — the Phase I baseline is wrong

| Metric | Plan claims | Actually verified in this repo | Source |
|---|---|---|---|
| ISCO-08 top-1 accuracy | 85.3% | **21.19%** (3,973/18,747, official heldout, flat retrieval, no reranking) | `OFFICIAL_WISCO_TIER1_CONTROLLED_RESULTS.md` |
| Hierarchical vs. flat RAG | **+6.4pp improvement** | **-10.84pp** — hierarchical is *worse* than flat, not better (10.35% vs 21.19%, McNemar p ≈ 1.86×10⁻³⁰¹, 18,747 paired cases) | same |
| ISCO-08 top-3 / kappa / ISIC / ISCED / language-ID / code-switching figures | 92.1% / 0.83 / 81.4% / 87.3% / 76.8% / 98.2% / 92.4% | **Not independently verified against this project's real evaluation evidence in this review.** Unlike the ISCO-08 top-1 and hierarchical-vs-flat figures, no WISCO-based or equivalent external measurement of these specific numbers was found in the repo's evidence chain. Treat as unverified internal figures, not confirmed — do not present them as validated until a source is found or they are re-measured. |

**This is not a rounding difference.** The plan's headline finding (hierarchical retrieval
improves accuracy) is the *opposite* of the measured result (hierarchical retrieval is
dramatically worse than flat, an already-published, heavily cross-checked, statistically
overwhelming finding — see `MANUSCRIPT_SAFE_WISCO_WORDING.md`'s explicit instruction to *never*
write "hierarchical retrieval improved accuracy"). If this table's numbers are cited anywhere in
the thesis, correct them to the verified figures or remove the comparison entirely.

**Also unverified and not measured in this review:** a real Claude 3.5 Sonnet reranking pass
against the full WISCO dev/heldout split (blocked by Anthropic account credit as of this
writing — see `CLAUDE_TASK_43_LIVE_RERANKER_WISCO_RUN_FINAL_REPORT.md`). The only *real*
reranked-accuracy measurement obtained so far used a free local model (Ollama `llama3.2:latest`,
not Claude) on the 2,013-case dev split: **17.19%** (346/2,013) — also below the flat-only,
no-reranking baseline. Do not conflate this local-model figure with a Claude 3.5 Sonnet result in
the thesis; it is a different, weaker model, run only because paid credit was unavailable.

## 2. Section 2 "corrections" — the WISCO DOI correction is itself wrong

The plan states: *"The WISCO dataset citation is corrected to DOI 10.5281/zenodo.7598568
[...] A different DOI (...7871194) [...] could not be verified [...] that DOI does not resolve
to a WISCO record."*

**Both halves of this claim are wrong**, per this project's own dated, MD5-checksum-verified
provenance record (`Documentation/Phase_2/Week_1/PROVENANCE.md`, produced 2026-08-02):

| Version DOI | Published | Status |
|---|---|---|
| 10.5281/zenodo.7598568 | 2023-02-02 | **Superseded — the oldest of 4 versions.** This is the file an earlier draft of *this project's own* Module A work mistakenly used before catching and correcting the error same-day. It is not the canonical citation. |
| 10.5281/zenodo.7871194 | 2023-04-25 | **Does resolve** — a real, later version in the same version chain. The claim that it doesn't resolve is incorrect. |
| **10.5281/zenodo.8262593** | **record 2023-04-25, canonical file dated 2023-08-18** | **Canonical — cite this one.** File `occupations_ISCO08_5dgt_55languages_4000titles_surveycodings_20230818.xlsx`, MD5-verified against Zenodo's published metadata, licence CC-BY-4.0. |

**Additional correction the plan is missing entirely:** using this dataset's `OCC>>INDUSTRY`
sheet (needed for Module D's industry crosswalk) triggers a **second, separate citation
obligation** (Belloni & Tijdens 2017, SERISS Deliverable 8.11, DOI 10.13140/RG.2.2.31328.02566) —
documented in `PROVENANCE.md`, not yet added to the repo README or thesis.

**Also correct, but incomplete:** the plan's claim that the source has no 5-digit ISCO precision
is right, but the reason given should be sharpened — per `PROVENANCE.md`, the "5dgt" in the
filename refers to SurveyCodings' own internal occupation-ID scheme, not ISCO-08 code precision;
the finest gold-code precision this workbook provides is 4-digit (`ISCO0804`), matching ISCO-08's
own unit-group level.

## 3. Module A — this work is already done, with different, real numbers

The plan's Section 4 describes Module A as pending, 2-week, weeks-1-2 work
("download WISCO, run classifier, compute accuracy"). **This has already happened**, in this
repository, on 2026-08-02:

| Plan assumption | What actually happened | Source |
|---|---|---|
| ~4,000 titles, 55 languages | **4,745 CODESET rows** (246 are section headings, not occupations; **4,232 clean parsed occupation records**), **61 distinct base languages** (not 55 — the description text wasn't updated between versions) | `module_a_week1_report.md` §1, §4 |
| Per-language extraction still to do | Done: en 4,230 / ar 4,167 / ur 3,989 / hi 4,202 / tl 4,172 titles retained, 100% unit-group coverage (436/436) in every language | `module_a_week1_report.md` §4 |
| Run classifier, compute accuracy | **Not yet run** — Week 2 (test-harness build) is the actual next step, not Week 1's "download & filter." A 21,160-call compute/cost estimate exists (§4 below) but the actual classification pass has not executed. | `week2_brief.md` §4 |
| (not mentioned) | **A real, root-caused system defect was found and has now been fully resolved**: this project's own legacy ISCO-08 knowledge base (`load_full_isco.py`) had 441 unit groups against the official 436 — 19 non-standard codes returnable to a real respondent, 14 real codes missing. The subsistence-farming cluster (6161-6164) was fixed first (commit `fdc101e`); the remaining 15/10-code discrepancy — plus 6 further mislabeled-but-code-valid entries the original WISCO-only comparison couldn't catch — was resolved 2026-08-12 via a direct primary-source cross-check against isco.ilo.org's official ISCO-08 structure export. The KB now has exactly 436 unit groups, with one disclosed, deliberately-unfixed exception (Armed Forces code format). | `module_a_week1_report.md` §5, §5.3a; `backend/tests/test_load_full_isco_catalogue_consistency.py` |
| (not mentioned) | WISCO's Arabic data has **zero dialectal content** (all 22 Arabic locale columns are byte-identical MSA, checked directly on real data) — the plan's Module G dialect-normalisation test (§10) **cannot run against WISCO as scoped**. Needs a different data source or a redefined experiment before Week 9. | `module_a_week1_report.md` §6, `week2_brief.md` §2.2 |
| Single industry-crosswalk source | Two non-interchangeable sources exist (`NACE2.0`: 16.7% coverage, clean single hop to ISIC Rev.4; `NACE2004`/`OCC>>INDUSTRY`: 100% coverage, one generation behind, needs an extra hop). They agree on only 26.1% of rows where both exist. Both are now extracted separately; Module D must report accuracy for each, not collapse them into one number. | `module_a_week1_report.md` §5.4 |

**Action:** replace Section 4 of the plan with a pointer to the real Week 1/2 documents, and
retarget Module A's remaining scope to what Week 2/3 actually still need: build the evaluation
harness (with the `.lower().strip()` normalization and dual-industry-source requirements already
identified in `week2_brief.md`), then run the real classification pass — ideally the
already-recommended 300-500-case stratified pilot first, not the full 21,160-call run blind (cost
estimate: $38-89 and 3-10 hours depending on the real, currently-unmeasured LLM reranking trigger
rate — see `week2_brief.md` §4).

## 4. "13-agent CrewAI system" — does not match the actual architecture

Checked directly against `backend/agents/*.py`: **12 modules actually instantiate a
`crewai.Agent`** (`audit_logger.py`, `context_memory.py`, `conversation_manager.py`,
`emotional_intelligence.py`, `hitl_quality_manager.py`, `isco_classifier.py`,
`isic_classifier.py`, `language_processor.py`, `rag_expert.py`, `report_generator.py`,
`semantic_relation.py`, `validation_agent.py`). Several further modules
(`isced_classifier.py`, `nationality_classifier.py`, `person_register.py`,
`survey_orchestrator.py`, `isco_reranker_strict.py`) provide real functionality without
constructing a CrewAI `Agent` object directly. Neither "13" nor the original archived
`CLAUDE.md`'s 10-agent plan (`agent_01_auth.py` ... `agent_10_audit.py`) matches the actual,
current file layout — that document is explicitly marked superseded in its own header. Use
"12-agent CrewAI architecture" (or list the actual module names) in Chapter 2/6, not "13-agent."

## 5. Two citations correctly removed — confirmed, no further action

The plan's own Section 2 already correctly identifies and removes two unverifiable citations (a
claimed 2026 IEEE Access CrewAI/LangGraph paper, a claimed "Digital Dubai synthetic LFS pilot").
This review did not find either cited anywhere in this repository's own documentation, consistent
with the plan's removal. No correction needed here — flagged only for completeness.

## 6. What does NOT need correction

Modules B, C, D (crosswalk rebuild plan), E (pilot), F (synthetic data, correctly scoped as
supplementary-only), G, H, the 14-week timeline structure, and the risk register are not
contradicted by anything in this project's verified evidence and do not need rewriting — only
the specific factual claims itemized above do. The literature review (Section 3) and
classification-revision context (Section 12) were not independently re-verified in this review
(would require live web access this review did not perform); they are unchanged from the
supplied plan.

## 7. Recommended next action

1. Correct the Executive Summary table (Section 1 above) before this plan is shown to anyone else.
2. Correct the WISCO DOI citation everywhere it appears (Section 2 above) — the plan's current
   "corrected" DOI is the wrong one.
3. Replace Module A's Section 4 with a pointer to the real Week 1/2 work and retarget its
   remaining scope to Week 2/3 (Section 3 above).
4. Fix the "13-agent" claim (Section 4 above).
5. Decide how to handle the hierarchical-underperforms-flat finding in Chapter 2's novelty
   framing — this is a real, negative, already-disclosed result (see
   `MANUSCRIPT_SAFE_WISCO_WORDING.md` for pre-drafted safe language), not something to omit or
   soften.

---

## 8. Module-by-module implementation review (2026-08-11)

A deeper pass through Modules A-H against the actual code, not just the plan's claims. Full test
suite green throughout (2,347 passed at time of writing); every number below was read directly
from the running code, not assumed from documentation.

### Module A — one more correction, plus confirmation the real evaluation was unaffected

`Week_2/week2_brief.md` §2.3 claimed production's `ISCOClassifier.classify()` lowercases query
text before embedding, citing a specific line number. **Checked directly and found wrong**: the
real `classify()` entry point only strips whitespace; the only `.lower()` call anywhere in the
retrieval path (`isco_classifier.py`, `hierarchical_store.py`, `vector_store.py` — checked
exhaustively) is inside a separate keyword-hint helper that never touches the embedded text.
**No harm done to the real evaluation**: `eval/run_eval.py` and the WISCO benchmark builders
never applied `.lower()` either, so the actual Task 36-43 evaluation already matches true
production behaviour by coincidence. Corrected `week2_brief.md` in place with a dated note — do
not "fix" `run_eval.py` to add lowercasing based on the old version of that section.

### Modules B & C — infrastructure is real and well-tested; the gap is data + deployment, not code

The plan treats ISIC/ISCED-F hierarchical retrieval as unbuilt, unscoped future work. **It
already exists**: `backend/rag/hierarchy_nodes.py` + `standard_hierarchical_store.py` +
`build_standard_hierarchical_collections.py` reuse ISCO-08's own generic beam-search engine for
both standards, fully tested (92 tests passing across `test_isic_classifier.py`,
`test_isced_classifier.py`, `test_hierarchy_nodes.py`, `test_standard_hierarchical_store.py`,
`test_build_standard_hierarchical_collections.py`), with an explicit, tested,
never-silently-fabricated fallback contract to the legacy keyword/LLM pipeline. See
`ISIC_ISCEDF_HIERARCHICAL_RETRIEVAL_IMPLEMENTATION.md` for the full design — that document
itself is careful not to overclaim ("These counts are NOT an official-catalogue coverage claim").

**What's actually missing, and it's not a code bug:**
1. Data coverage: 134/419 ISIC classes, 63/~80 ISCED-F detailed fields — the same gap the plan
   already correctly identifies, just via a different (keyword-table) mechanism than the plan
   assumed.
2. **Deployment**: the live Qdrant collections (`isic_rev4_*`, `iscedf2013_*_fields`) have never
   been populated — the `--execute` step is explicitly "NOT run as part of this repository's own
   work." Until it is, `method="isic_hierarchical_retrieval"` will always fall back to the legacy
   keyword/LLM path in practice, regardless of how much data-coverage work happens.

Neither of these was fixed in this pass — (1) needs real official ISIC Rev.4 / ISCED-F 2013
source text authored by someone with access to and time to work through the primary documents,
not a guess; (2) is a real infrastructure decision (a live Qdrant write) that needs the same
explicit go-ahead every other live-infrastructure action in this project has required.

### Module D — the LOW-severity gap the plan describes is already fixed

The plan's Module D section (and an earlier project planning document) describes
`SemanticViolation.severity` as only ever emitting `HIGH`/`MODERATE` despite `LOW` being a
documented valid value. **Checked directly: this is already fixed** — a genuine third severity
band exists (`semantic_relation.py` line 528), with `rule_id`s (`SR-ISCO-ISCED-01/02/03`) and
15 passing tests. What the plan correctly identifies as still needed: the `_ISCO_MAJOR_TO_ISIC` /
`_ISCO_SUBMAJOR_TO_ISIC` / `_ISCO_MAJOR_TO_ISCED` / `_ISCO_SUBMAJOR_TO_ISCED_MIN` tables are
confirmed **still hand-built**, not yet rebuilt from the official ILO ISCO-08 Volume I
correspondence tables or UNESCO ISCED 2011 Operational Manual Table 7. Not attempted in this
pass — rebuilding a crosswalk by guessing at official correspondences would be exactly the kind
of fabrication this project has consistently avoided; it needs the real source documents.

### Module G — solid, 102 passing tests, no new findings beyond the plan's own (WISCO Arabic
has no dialectal content, already documented in `Week_1/module_a_week1_report.md` §6).

### Module H — the plan's own framing doesn't match this system's actual architecture

The plan's "Delegation Correctness" metric assumes a CrewAI hierarchical process (a manager
agent delegating to worker agents). **Checked exhaustively: this system never uses one.** Zero
occurrences of `Process.hierarchical` or `manager_agent` anywhere in `backend/`. All 12 agents
that construct a `crewai.Agent` (`audit_logger`, `context_memory`, `conversation_manager`,
`emotional_intelligence`, `hitl_quality_manager`, `isco_classifier`, `isic_classifier`,
`language_processor`, `rag_expert`, `report_generator`, `semantic_relation`, `validation_agent`)
consistently set `allow_delegation=False` — confirming this by design, not omission. The
original archived `CLAUDE.md` plan (superseded, per its own header) *did* describe a hierarchical
manager-delegation pattern; the actual implemented system abandoned it in favour of direct,
independent invocation from application code. **Module H's evaluation target needs rescoping**:
"orchestration correctness" (does the calling code invoke the right agent at the right time) is
the real, evaluable equivalent in this architecture — "delegation correctness" as literally
described in the plan has nothing to measure here.

### Modules E & F — not code-review targets

Module E (pilot) requires real human participants and ethics approval — not something any code
review or implementation pass can substitute for; status unchanged from the last check (no
application submitted). Module F (synthetic data) is explicitly scoped as supplementary-only and
lower priority by the plan itself; no synthetic-data generation code exists yet in this repo, and
building it wasn't attempted in this pass pending an explicit decision on priority and exact
Person Register schema fields to target.
