# Project: Multilingual Conversational AI for Labour Force Surveys

M.Tech thesis project (IIIT Kottayam, supervisor Dr. Goutam Mali). A
FastAPI + CrewAI backend and Next.js 14 frontend conducting Labour Force
Survey interviews in 5 languages (English, Arabic MSA + Gulf dialect,
Urdu, Hindi, Tagalog), with hierarchical RAG ISCO-08 coding, ISIC Rev.4 /
ISCED 2011 + ISCED-F 2013 classification, and a Semantic Relation Engine
(SRE) cross-validating all three standards. Currently in **Phase II**:
closing coverage/validation gaps identified in an internal Module A Week 1
review and an externally-supplied revision plan whose own factual claims
required correction (see "Provenance of this document" below).

**Every fact in this document was verified directly against the running
code and live repository on 2026-08-12** — not inferred from the thesis,
not copied from a prior draft. Where something genuinely could not be
verified in this pass, it says so explicitly rather than guessing.

## Provenance of this document

This replaces an externally-supplied `CLAUDE.md` draft that contained
extensive, specific, unhedged claims — exact dependency versions, file
names, agent counts — that turned out not to match this repository at
all (not stale; wrong in ways suggesting it was written without real
repo access; see the corrections table below). That draft's own
"Repository structure" and "Coding conventions" sections admitted as
much ("Not yet verified against the real repo... treat as a starting
hypothesis, not a fact"), but its other sections stated equally
unverified claims as fact without that hedge. Do not trust a document
like that one for anything specific without independently checking it
against the repo first — this is the same failure mode as the WISCO DOI
error and the fabricated 85.3%-accuracy Phase I baseline found and
corrected earlier in this project's history
(`Documentation/Phase_2/PHASE_II_PLAN_CORRECTIONS.md`).

An even older, fully archived kickoff prompt lives at
`Documentation/CLAUDE.md` — it describes a `hierarchical_rag/`,
10-agent, `Process.hierarchical` architecture that was never actually
built; the real system is the flatter `backend/agents/` / `backend/rag/`
/ `backend/api/` structure described below. That file is marked
ARCHIVED — SUPERSEDED in its own header and should not be used to
navigate or modify current code.

## Corrections to the externally-supplied draft

| Claim | Draft said | Actually verified |
|---|---|---|
| `crewai` version | `0.60` | **`1.9.3`** |
| `qdrant-client` version | `1.9` | **`1.17.0`** |
| `fastapi` version | `0.110` | **`0.131.0`** |
| `alembic` version | `1.13` | **`1.18.4`** |
| `redis` (Python client) version | `5.0` | **`7.2.0`** |
| `sentence-transformers` version | `3.0` | **`5.2.3`** |
| Arabic dialect normalization | `camel-tools==1.5` | **Not a dependency at all** — hand-built 79-rule dict (`_GULF_NORMALISE` in `language_processor.py`), no external library |
| SRE file name | `semantic_relation_engine.py` | **`semantic_relation.py`** |
| Agent count | "13-agent CrewAI system" | **11-12 modules** construct a `crewai.Agent` (count depends on exact grep pattern; never 13). **Zero** use of CrewAI hierarchical delegation anywhere — no `Process.hierarchical`, no `manager_agent`. "SurveyOrchestrator — CrewAI hierarchical process" does not match the real architecture. |
| WISCO DOI | `10.5281/zenodo.7598568` | **`10.5281/zenodo.8262593`** — `7598568` is the old, superseded version (this exact error was already found and corrected once before, in a different draft) |
| ISCO-08 "441 = 436 official + 5 supplementary (ILO Geneva 2012 guidance)" | Stated as an already-correct, deliberate design choice | **Not what was found.** Real, root-caused audit: 19 non-standard codes, 14 missing official codes — no "5 supplementary, ILO-sanctioned" explanation exists in any source checked. **Fully resolved 2026-08-12** via a direct primary-source ILO cross-check — see "Knowledge base construction" below. |
| `backend/tests/` file count | 18 files | **44 files** (88 combined with `eval/`'s 44) |
| API surface | "All survey interaction goes through one endpoint" | **17 distinct routes** (counted from the live OpenAPI spec) — session CRUD, responses, HITL queue/review, reports, auth, health/debug, etc. |
| ISIC coverage | "100+ of 419" | **134 of 419** — real, checked directly |
| ISCED-F coverage | "30+ of ~80 detailed fields" | **63 of ~80** — real, checked directly (the draft undercounted; this was already at 63 before this document was written) |

Everything else below is either confirmed accurate from the draft or
independently verified fresh.

## Environment & deployment

```bash
# Full native startup (Docker infra + native backend/frontend):
start.bat
# or, Docker-only:
docker compose -f docker/docker-compose.yml up --build
```

| Service | URL | Verified state (2026-08-12) |
|---|---|---|
| Frontend (Next.js 14) | http://localhost:3000 | 4 pages: `/`, `/chat`, `/report`, `/supervisor_review` — all confirmed loading |
| Backend (FastAPI) | http://localhost:8000 | `/docs` (Swagger), `/ready`, `/health`, `/debug/isco/{job_title}` — all confirmed responding |
| PostgreSQL 15 | internal :5432 | 11 tables (see below), Alembic migrations, confirmed at head as of 2026-08-12 |
| Qdrant | http://localhost:6333 | 10 live collections (legacy + official-profile ISCO-08, both hierarchical and flat) |
| Redis 7 | internal :6379 | confirmed healthy |
| Ollama | http://localhost:11434 | confirmed healthy; models present: `llama3.2:1b`, `llama3.2:latest`, `gemma3:4b` |

First-boot note: the embedding model actually in use **by default** is
`intfloat/multilingual-e5-small` (not `-large` as some older documents
say) — confirmed in `backend/rag/hierarchical_store.py` and
`backend/rag/vector_store.py`. **2026-08-24: a real, larger alternative
now exists and is measurably better, but is not yet the default.** A new,
additive `official_ilo2021_v1_e5large` catalogue profile
(`intfloat/multilingual-e5-large`, 1024-dim) was built, live-verified,
and evaluated end-to-end against an independent 500-case WISCO dev
sample (flat retrieval, no LLM reranking — same config as the published
21.19% baseline): **29.20% (146/500) vs. e5-small's 20.60% (103/500)**,
McNemar exact p ≈ 1.77×10⁻⁶, Wilson 95% CIs non-overlapping
([25.39%,33.33%] vs. [17.29%,24.36%]). Gain is largest on Arabic
(12.93%→26.72%) and consistent across every language. Cost: ~5.2x
slower per query (145ms vs. 28ms — both still fast in absolute terms).
Real artifacts: `eval/local_runs/e5large_build_manifest_20260823.json`,
`eval/local_runs/e5large_vs_e5small_dev500_20260824/results.csv`. **Not
yet switched to production default** — `official_ilo2021_v1` (e5-small)
remains the default `isco_catalogue_profile`; switching requires
rebuilding the flat/hierarchical/ISIC/ISCED-F collections too and is a
call for you and your supervisor, same as the flat-vs-hierarchical
default decision already flagged elsewhere in this document. New code:
`backend/rag/build_official_isco08_collections_e5large.py`,
`embedding_config_for_profile()` in `official_isco08_catalogue.py`. 10
new tests, full suite re-run: 2,342 passed, 0 failed. **Immediate
follow-up, same day**: does LLM reranking help more on top of the
stronger retrieval? Re-ran the identical 500-case sample with Groq
reranking enabled (`reranker_model="groq/openai/gpt-oss-120b"` —
Gemini's free-tier daily quota was already exhausted). Result:
**29.20% (146/500) — byte-identical to the no-rerank e5-large result**;
499/500 predicted codes matched exactly, the reranker fired on 100% of
cases and changed nothing. This is the third independent confirmation
that retrieval quality, not reranking/reasoning quality, is the real
accuracy ceiling (see "Semantic Relation Engine" section's earlier
Gemini-vs-local-model finding) — now shown on a stronger retrieval base
and a different provider, which rules out a one-off coincidence.

Real, currently-pinned dependency versions (from `requirements-dev.txt`,
the file that actually pins exact versions — `requirements.txt` itself
pins nothing, all bare package names, verified across this project's
entire git history):

```
crewai==1.9.3
fastapi==0.131.0
sqlalchemy==2.0.46
alembic==1.18.4
qdrant-client==1.17.0
sentence-transformers==5.2.3
redis==7.2.0
pytest==9.0.2
pytest-asyncio==1.3.0
```

## Database (PostgreSQL, 11 tables — confirmed exact names)

`users`, `otp_codes`, `survey_sessions`, `survey_responses`, `audit_logs`,
`data_access_logs`, `agent_decision_logs`, `quality_reviews`,
`hitl_queue`, `person_register`, `survey_report_records`.

Migrations (`backend/database/migrations/versions/`, confirmed current
chain, applied through head as of 2026-08-21):
`001_initial_schema` → `976b9b9c96d4_add_hitlqueue_evaluation_tables` →
`f514fcb81c72_add_deleted_at_soft_delete_columns` →
`a3f9c1d2e4b6_add_survey_response_supersedes_id` →
`c7e2a48f9d31_add_report_isic_isced_coherence_columns` — fixes a real bug
found 2026-08-21: `ReportGenerator.generate()` (`backend/agents/
report_generator.py`) correctly computed ISIC/ISCED/semantic-coherence
results on first generation, but `survey_report_records` had no columns
to persist them, so every subsequent cache-hit read of an already-
generated report (`regenerate=False`, the default — i.e. any normal
report re-view) silently returned `null` for all three, even when the
underlying data was present and classifiable. Reproduced against a real
account before being fixed; regression tests in
`backend/tests/test_report_generator.py::TestEnrichmentPersistence`.

**A second, independent bug in the same code path, found by that same
regression test**: `report_generator.py`'s `semantic_coherence` dict was
built with `dataclasses.asdict(v) for v in val` over an already-recursed
`dataclasses.asdict(sc)` result — `asdict()` already converts nested
dataclasses (e.g. `sc.violations`) to plain dicts, so the second call
raised `TypeError` on every single invocation, silently swallowed by a
bare `except Exception: pass`. This meant `semantic_coherence` was `null`
in every report this method ever generated, independent of the caching
bug above. Fixed to a single `dataclasses.asdict(sc)` call — the same
fix already applied earlier to the identical pattern in
`backend/api/survey_routes.py`'s `semantic_coherence_out` (that fix did
not get propagated to this second, separate occurrence at the time).

## API (17 routes, from the live OpenAPI spec)

```
POST   /auth/request-otp
POST   /auth/verify-otp
POST   /auth/request-sms-otp
POST   /auth/verify-sms-otp
POST   /auth/logout
POST   /survey/sessions
GET    /survey/sessions
GET    /survey/sessions/{session_id}
DELETE /survey/sessions/{session_id}
PATCH  /survey/sessions/{session_id}/complete
POST   /survey/sessions/{session_id}/message
GET    /survey/sessions/{session_id}/report
POST   /survey/sessions/{session_id}/responses
GET    /survey/sessions/{session_id}/responses
PATCH  /survey/sessions/{session_id}/responses/{response_id}
GET    /survey/hitl/queue
POST   /survey/hitl/review
GET    /health
GET    /debug/isco/{job_title}
GET    /ready
```

Login flow tested end-to-end 2026-08-12: `POST /auth/request-otp
{"email": ...}` → dev-mode response includes `dev_otp` directly → `POST
/auth/verify-otp {"email", "code"}` → JWT → `POST /survey/sessions
{"language": "en"}` → real session created with pre-filled Person
Register fields.

## Real, current agent modules (`backend/agents/`)

Not all of these construct a `crewai.Agent` — confirmed which do:

| File | Constructs `crewai.Agent`? |
|---|---|
| `audit_logger.py` | Yes |
| `context_memory.py` | Yes |
| `conversation_manager.py` | Yes |
| `emotional_intelligence.py` | Yes |
| `hitl_quality_manager.py` | Yes |
| `isco_classifier.py` | Yes |
| `isic_classifier.py` | Yes |
| `language_processor.py` | Yes |
| `rag_expert.py` | Yes |
| `report_generator.py` | Yes |
| `semantic_relation.py` | Yes |
| `validation_agent.py` | Yes |
| `isced_classifier.py` | No — keyword/rule-based, no LLM |
| `nationality_classifier.py` | No |
| `person_register.py` | No — deterministic |
| `isco_reranker_strict.py` | No |
| `classifier_methods.py` | Not an agent — shared constants |
| `method_registry.py` | Not an agent — registry/introspection utility |

All 12 that do construct an `Agent` set `allow_delegation=False`
consistently (checked directly). There is no CrewAI hierarchical
manager-delegation process anywhere in this codebase.

## Semantic Relation Engine — verified formula

Confirmed present in `backend/agents/semantic_relation.py` (lines
358-377):

```python
raw_score = 0.55 * score_isic + 0.45 * score_isced
is_coherent = raw_score >= 0.70
# score >= 0.90 -> NONE / highest coherence band
# additional bands down to HIGH severity below 0.50 (LOW severity now
# genuinely emitted -- see below, this was previously a documented gap)
```

The sub-major strict rules (Health Professionals → ISIC Q + ISCED≥7, ICT
Professionals → ISIC J, Chief Executives → ISCED≥6) are **confirmed
present** but **confirmed still hand-built**, not yet rebuilt from the
official ILO ISCO-08 Vol. I / UNESCO ISCED 2011 Operational Manual Table
7 correspondence tables — this part of the backlog (Module D) is real
and still open.

**A gap noted in earlier planning documents — "LOW severity is never
actually emitted" — is already fixed**, confirmed directly: a genuine
third severity band exists with `rule_id`s (`SR-ISCO-ISCED-01/02/03`) and
dedicated passing tests.

## Knowledge base construction

- **ISCO-08 (legacy profile)**: `backend/rag/load_full_isco.py`, now
  **436 unit groups**, matching the official ISCO-08 count exactly —
  fixed 2026-08-12 via a direct cross-check against the primary ILO
  ISCO-08 structure source (isco.ilo.org's official CSV export, not
  WISCO, not a guess). Originally 441 entries: 19 non-standard + 14
  missing per Module A Week 1's WISCO-only comparison; the 2026-08-12
  primary-source pass resolved all of those plus 6 further ones that
  comparison couldn't catch (several official, structurally valid codes
  — 9510, 9520, 9611, 9612, 9613, 9621, and the 6121/6122/6123 trio —
  carried an entirely different occupation's label, invisible to a
  code-existence check alone). **One disclosed exception remains,
  deliberately not fixed**: major group 0 (Armed Forces) is represented
  as 4-digit `"0110"/"0210"/"0310"` instead of ISCO-08's own bare
  3-digit convention (`"110"/"210"/"310"`) — changing code string length
  would ripple into every other 4-digit-code assumption in this codebase
  (CSV joins, WISCO gold-code comparison, `_digits()`), so this needs a
  coordinated decision, not a unilateral rename. See
  `backend/tests/test_load_full_isco_catalogue_consistency.py` for the
  full verification and `Documentation/Phase_2/Week_1/
  module_a_week1_report.md` for the original WISCO-only finding this
  fix supersedes. **This required an explicit live Qdrant reload**
  (`python -m backend.rag.load_full_isco --recreate`) to actually take
  effect; the source fix alone does nothing until that runs, because
  `start.bat`'s own logic skips reloading collections that already
  exist.
- **ISCO-08 (official ILO 2021 profile)**: a separate, later-built,
  independently verified catalogue (10/43/130/436 major/submajor/minor/
  unit groups), used for the actual published WISCO Tier-1 evaluation
  results (see below). Distinct collections, distinct code path
  (`profile="official_ilo2021_v1"`).
- **ISIC Rev.4 / ISCED-F 2013 hierarchical retrieval**: real,
  well-tested infrastructure already exists
  (`backend/rag/hierarchy_nodes.py`, `standard_hierarchical_store.py`,
  `build_standard_hierarchical_collections.py`), reusing the same
  generic beam-search engine ISCO-08 uses. **The live Qdrant collections
  were built and populated 2026-08-23** (`--execute` run for both
  standards: `isic_rev4_{sections,divisions,groups,classes}` — 21/68/118/134
  nodes — and `iscedf2013_{broad,narrow,detailed}_fields` — 11/25/63 nodes),
  confirmed live directly: `ISICClassifier().classify(text,
  method=ISIC_HIERARCHICAL_RETRIEVAL)` and the ISCED-F equivalent both
  return `fallback_used=False` on a real query — genuine hierarchical
  retrieval, not the legacy fallback. **Accuracy against a labelled test
  set is still not yet evaluated** — that remains open. Data coverage
  (unchanged by this): 134/419 ISIC classes, 63/~80 ISCED-F detailed
  fields.
- **ISIC/ISCED-F reranker parity + a real dead-code bug, fixed 2026-08-23**:
  `ISICClassifier`/`ISCEDClassifier` gained a `reranker_model` constructor
  param mirroring `ISCOClassifier`'s exactly (fail-closed via
  `get_llm_strict()`, `None` default preserves prior behaviour byte-for-
  byte). Wiring it up surfaced a genuine, previously-undiscovered bug: both
  classifiers' keyword-confidence score normalised the top candidate by
  *its own hit count* (`count / max_hits`), which is algebraically always
  1.0 — since the LLM-rerank gate was `score >= 0.85`, the LLM branch
  (present since these classifiers were first written) was **permanently
  unreachable in production**, confirmed by computing real scores against
  real queries. Fixed by renormalising to a genuine match-coverage
  fraction plus a gap-based trigger reusing the top1/top2-candidate-gap
  method already proven for ISCO (`_MIN_CANDIDATE_GAP = 0.15`, a disclosed
  provisional judgement call, not independently measured). 17 new tests;
  full suite (2,332 tests) passes with zero regressions. Live-verified
  against the real Gemini API: genuinely-ambiguous inputs (e.g. "medical
  and dental studies") now correctly fire `method="llm"`; unambiguous ones
  correctly stay `"keyword"`. A smaller, separate quirk (a keyword
  repeated within one catalogue entry's own keyword string is indexed once
  per repetition, inflating that entry's score) was found but not fixed —
  flagged for a future pass. See
  `Documentation/Conference_I_Reviewer_2/` and the RAG Implementation
  Dossier artifact for the full writeup.

## The actual published WISCO evaluation result — read this before citing any accuracy number

**This is the single most important fact this document can convey.** The
official WISCO Tier-1 evaluation has been run, analyzed, and published
(not a future task):

| Metric | Flat retrieval | Strict hierarchical retrieval |
|---|---:|---:|
| Exact 4-digit accuracy (18,747-case official heldout) | **21.19%** | **10.35%** |
| 95% Wilson CI | [20.61%, 21.78%] | [9.93%, 10.80%] |

Hierarchical is **10.84 percentage points worse** than flat (McNemar
p ≈ 1.86×10⁻³⁰¹) — a genuine, disclosed negative finding. **Never**
describe hierarchical retrieval as outperforming flat retrieval in any
manuscript-facing text; pre-approved safe wording is in
`Documentation/Conference_I_Reviewer_2/MANUSCRIPT_SAFE_WISCO_WORDING.md`.
Canonical source of these numbers, character-for-character:
`Documentation/Conference_I_Reviewer_2/OFFICIAL_WISCO_TIER1_CONTROLLED_RESULTS.md`.

**A real Claude 3.5 Sonnet reranked accuracy number does not exist yet**
— blocked purely on Anthropic account credit as of this writing. A free
local-model (Ollama `llama3.2:latest`) fallback result exists (17.19% on
the 2,013-case dev split) but must never be cited as a Claude result.

## Testing

```bash
pytest backend/tests eval/ -q
```
→ **2,282 passed, 0 failed** (2026-08-21, count only — not a full
document re-verification; 1,450 in `backend/tests` + 832 in `eval/`, 89
files total). The count dropped from a prior 2,363 because
`backend/agents/survey_orchestrator.py` — confirmed dead code, never
imported by the live API, see the API-surface note below — was removed
from the codebase along with its dedicated 65-test file and the
orchestrator-dependent tests in `test_hitl_and_e2e_extended.py`. No
standing known failures.

## Citation policy — unchanged, still correct

Do not add a citation (paper, dataset, standard) unless independently
verified to exist. Two specific citations already failed verification and
must never be reintroduced: a claimed 2026 IEEE Access CrewAI/LangGraph
paper with specific performance figures, and a claimed "Digital Dubai
synthetic LFS pilot." Neither returned any matching source after direct
search.

## Active task backlog — real status, not the external draft's assumed status

**Correction, 2026-08-24**: everything below this line was stale. This
document's own header claims verification "on 2026-08-12," but a real,
committed, 9-prompt work sequence (commits `cabcc75` through `df813af`,
dated 2026-08-15 through 2026-08-21 — all on this branch, confirmed via
`git log`) substantially completed Modules A/D/F/I/J *after* that date
and was never folded back into this document. Corrected below by reading
the actual committed evidence directly, not by trusting the prior text.

- **Module A (WISCO external validation)**: substantially done, **and
  the per-language breakdown this document previously said was missing
  already exists**: `Documentation/Conference_I_Reviewer_2/generated/
  wisco_tier1_topk_kappa.json` (commit `cabcc75`) — real per-language
  top-1/top-3/Cohen's κ/95% CI for all 5 languages, both flat and
  hierarchical, reproduced in `Documentation/Phase_2/
  FINAL_RESULTS_PACKAGE.md` Table 6.1b.
- **Module B (ISIC full coverage)**: infrastructure ready and tested;
  hierarchical-retrieval Qdrant collections populated and live-verified
  2026-08-23 (see above); `reranker_model` cloud-LLM parity added
  2026-08-23, plus a genuine dead-code confidence-scoring bug found and
  fixed the same day (see below). Data coverage (134/419 classes) is the
  real remaining gap — unchanged, and no official ISIC Rev.4 catalogue
  has ever been imported/verified (unlike ISCO-08's Task 20/21 primary-
  source pass), so `official_count_verified` stays `null`.
- **Module C (ISCED-F full coverage)**: same status as Module B —
  collections live, reranker parity added, same bug fixed. Data coverage
  (63/~80 detailed fields) is the real remaining gap.
- **Module D (SRE official crosswalk)**: more resolved than "needs work."
  LOW-severity gap fixed. Expanded to a real 61-case validation set with
  `n_mismatches=0` (commit `f44cd79`, see `Documentation/Phase_2/
  FINAL_RESULTS_PACKAGE.md` Table 6.3). **A HIGH-severity SRE result now
  genuinely escalates to the live HITL queue** — wired and verified 3x
  against the real `/survey/sessions/{id}/message` endpoint, 100%/15 HIGH
  escalated, 0%/46 non-HIGH escalated, byte-identical across all 3 runs
  (commit `0eba4d4`). The crosswalk-table sourcing question this
  document previously framed as "find the ILO/UNESCO document" was
  **investigated directly and found to be a dead end, not a to-do**: the
  real, complete 433-page ISCO-08 Vol. I PDF was searched page-by-page —
  no ISCO-08↔ISIC correspondence table exists in it, and no other ILO
  document mapping ISCO-08 to ISIC was found either (commit `f44cd79`).
  The crosswalk tables remain hand-built by necessity, not by omission —
  any future improvement needs a different sourcing strategy, not "the
  document we haven't found yet."
- **Module E (pilot, n=30)**: confirmed still not started — re-checked
  directly against `Documentation/Phase_2/Week_1/
  ethics_submission_log.md`, every field is still an unfilled `<FILL>`
  placeholder. No ethics application submitted. Still the single most
  time-critical open item in the whole project, independent of all code
  work.
- **Module F (synthetic Person Register data)**: **done, not "not
  started"** — `eval/synthetic_person_register_stress_test.py` (commit
  `506e794`, 2026-08-15) exists, is committed, and stress-tests
  `PersonRegisterService`'s real pre-fill logic against statistically-
  sampled (not GAN-generated — disclosed reasoning in the file) synthetic
  records. Correctly and explicitly scoped as supplementary/stress-test
  only, never pilot evidence.
- **Module G (multilingual validation)**: unchanged — WISCO's Arabic
  data confirmed to have zero dialectal content, the planned dialect-
  normalization A/B test cannot run against it as originally scoped;
  needs a different data source or a redefined experiment.
- **Module H (CrewAI architecture evaluation)**: confirmed still not
  started — no commit or document anywhere in this repo's history
  mentions it. Its "delegation correctness" framing needs rescoping
  first — this system never uses CrewAI delegation (see agent table
  above); the real evaluable target is "orchestration correctness" (does
  the calling code invoke the right agent at the right time).
- **Module I (computational efficiency)**: **done, not "unverified"** —
  real measurements for every metric measurable on available hardware
  (commit `b73dbca`; full writeup `Documentation/Phase_2/Week_2/
  module_i_computational_efficiency_report.md`): hierarchical/flat RAG
  latency (133.9ms / 31.1ms mean, from the real 18,747-case Task 36 run),
  embedding compute (19.1ms mean), Qdrant on-disk size (~4.3MB) and RSS
  (94.9MB), and a real load-test breaking point (100% success through 42
  concurrent users, fails at 50). Genuinely still unmeasured, disclosed
  as such: LLM re-rank trigger rate and any real Claude 3.5 Sonnet
  cost/latency figure (a prior attempt was found to be a 100%-silent-
  fallback artifact from zero Anthropic credit, not used) — needs the
  stratified 300-500-case pilot Week 2 already recommends.
- **Module J (LLM tier-routing ablation)**: **complete, not "unverified"**
  — all 3 GENERAL-tier agents (LanguageProcessor/NER, ConversationManager,
  EmotionalIntelligence) have real, warmed, 3-run comparisons of
  llama3.2 vs. qwen2.5:3b (commits `b01d768`, `ad47312`; full writeup
  `Documentation/Phase_2/Week_2/
  module_j_llm_task_routing_ablation_status.md`). Finding: qwen2.5:3b
  matches or beats llama3.2 on every comparison, most clearly on Arabic-
  script NER (F1 0.509 vs. 0.421 mean). **This is a measurement, not a
  decision** — `llm_client.py`'s actual default (`OLLAMA_MODEL=llama3.2`)
  was deliberately left unchanged, pending a human call.
- A separate, much larger implementation plan (method registry, coverage
  audit, evaluation manifest/reproducibility instrumentation, ablation
  runner, figure-data exports — a formal A/C/D/E/F/G/H/I/J build spanning
  `eval/` and `backend/rag/hierarchy_engine.py`) was drafted for branch
  `master` earlier in this project and approved via plan mode.
  **Re-verified 2026-08-24**: its deliverables (method registry, coverage
  audit, manifest/analyze, ablation runner, SRE evidence fields, figure
  exports) are confirmed present, tested, and *this branch has them too*
  — see `Documentation/Conference_I_Reviewer_2/`. What's still genuinely
  missing: a real, full-scale ablation run against WISCO with reranking
  (the "reranking tier" / Phase D.2 / Tier 2) has **not** been executed —
  only the non-reranked Tier-1 run (table above) and a tiny 5-case
  synthetic-fixture integration run exist. The authoritative, per-
  reviewer-comment status lives in `Documentation/Conference_I_Reviewer_2/
  REVIEWER_RESPONSE_IMPLEMENTATION_MATRIX.md` — dated 2026-08-10, so
  itself older than the Module A/D/F/I/J work above and due for a
  refresh; only comments #1 (Springer formatting) and #6 (LLM roles
  documented) are marked "Ready for paper update" there, everything else
  is "Partially evidenced" / "Awaiting data" / "Awaiting measurement."

## Do not

- Do not resubmit on internal-only evidence — the real, external WISCO
  evaluation now exists (table above); use it.
- Do not drop the pilot study (Module E) or substitute synthetic data for
  it.
- Do not remove or shrink the SRE.
- Do not describe hierarchical retrieval as outperforming flat retrieval
  — the measured result is the opposite.
- Do not cite the free Ollama reranking result as if it were a Claude 3.5
  Sonnet result.
- Do not trust exact dependency versions, file names, or specific counts
  from any planning document (including earlier versions of this one)
  without checking them against the running repo first — this document's
  own corrections table above is the demonstration of why.
