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

First-boot note: the embedding model actually in use is
`intfloat/multilingual-e5-small` (not `-large` as some older documents
say) — confirmed in `backend/rag/hierarchical_store.py` and
`backend/rag/vector_store.py`.

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
chain, applied through head as of 2026-08-12):
`001_initial_schema` → `976b9b9c96d4_add_hitlqueue_evaluation_tables` →
`f514fcb81c72_add_deleted_at_soft_delete_columns` →
`a3f9c1d2e4b6_add_survey_response_supersedes_id`.

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
  for these have never been populated** (`--execute` never run) — so in
  practice, `method="isic_hierarchical_retrieval"` always falls back to
  the legacy keyword/LLM pipeline today. Data coverage: 134/419 ISIC
  classes, 63/~80 ISCED-F detailed fields.

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

- **Module A (WISCO external validation)**: substantially done. Real
  Week 1 data-prep work exists (`Documentation/Phase_2/Week_1/`); the
  actual official-profile classification run happened (table above).
  Per-language accuracy breakdown not yet compiled as a standalone table.
- **Module B (ISIC full coverage)**: infrastructure ready and tested;
  data coverage 134/419 is the real remaining gap.
- **Module C (ISCED-F full coverage)**: infrastructure ready and tested;
  data coverage 63/~80 is the real remaining gap.
- **Module D (SRE official crosswalk)**: LOW-severity gap already fixed.
  Crosswalk tables still hand-built — real work needed, requires the
  actual official ILO/UNESCO source documents, not a guess.
- **Module E (pilot, n=30)**: not started. No ethics application
  submitted as of the last check — the single most time-critical open
  item in the whole project, independent of all code work.
- **Module F (synthetic Person Register data)**: not started, no code
  exists yet, correctly scoped as supplementary-only.
- **Module G (multilingual validation)**: WISCO's Arabic data confirmed
  to have zero dialectal content — the planned dialect-normalization A/B
  test cannot run against it as originally scoped; needs a different
  data source or a redefined experiment.
- **Module H (CrewAI architecture evaluation)**: not started. Its
  "delegation correctness" framing needs rescoping first — this system
  never uses CrewAI delegation (see agent table above); the real
  evaluable target is "orchestration correctness" (does the calling code
  invoke the right agent at the right time).
- **Modules I/J (computational efficiency, LLM tier-routing ablation)**:
  not independently verified against the repo in this pass — treat as
  genuinely not-yet-started unless checked directly.
- A separate, much larger implementation plan (method registry, coverage
  audit, evaluation manifest/reproducibility instrumentation, ablation
  runner, figure-data exports — a formal A/C/D/E/F/G/H/I/J build spanning
  `eval/` and `backend/rag/hierarchy_engine.py`) was drafted for branch
  `master` earlier in this project and approved via plan mode, but its
  execution status against the current branch has not been re-verified
  as part of this document. Do not assume it is either done or pending
  without checking current code and asking the user which branch it was
  meant to target.

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
