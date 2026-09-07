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
`Documentation/Archive/ARCHIVED_ORIGINAL_KICKOFF_PROMPT.md` (moved and
renamed 2026-08-24 — it used to be named `Documentation/CLAUDE.md`,
which was confusing enough in practice, two files both named
`CLAUDE.md`, that it was relocated specifically to stop that). It
describes a `hierarchical_rag/`, 10-agent, `Process.hierarchical`
architecture that was never actually built; the real system is the
flatter `backend/agents/` / `backend/rag/` / `backend/api/` structure
described below. That file is marked ARCHIVED — SUPERSEDED in its own
header and should not be used to navigate or modify current code.

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
| Agent count | "13-agent CrewAI system" | **11-12 modules** construct a `crewai.Agent` (count depends on exact grep pattern; never 13) *as of 2026-08-12, when this comparison was made* — **now stale: a real 13th agent-constructing module was added 2026-08-24, see "Real, current agent modules" below.* **Zero** use of CrewAI hierarchical delegation anywhere — no `Process.hierarchical`, no `manager_agent`. "SurveyOrchestrator — CrewAI hierarchical process" does not match the real architecture. |
| WISCO DOI | `10.5281/zenodo.7598568` | **`10.5281/zenodo.8262593`** — `7598568` is the old, superseded version (this exact error was already found and corrected once before, in a different draft) |
| ISCO-08 "441 = 436 official + 5 supplementary (ILO Geneva 2012 guidance)" | Stated as an already-correct, deliberate design choice | **Not what was found.** Real, root-caused audit: 19 non-standard codes, 14 missing official codes — no "5 supplementary, ILO-sanctioned" explanation exists in any source checked. **Fully resolved 2026-08-12** via a direct primary-source ILO cross-check — see "Knowledge base construction" below. |
| `backend/tests/` file count | 18 files | **44 files** (88 combined with `eval/`'s 44) |
| API surface | "All survey interaction goes through one endpoint" | **20 distinct routes** (counted from the live OpenAPI spec — corrected from a previously-stated 17, which undercounted against this document's own already-listed 20-line route table below; found during the 2026-08-24 documentation-completeness audit) — session CRUD, responses, HITL queue/review, reports, auth, health/debug, etc. |
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
| Qdrant | http://localhost:6333 | **43 live collections** (corrected 2026-08-25, re-counted directly via a live `GET /collections` call — up from this same day's earlier "41" count, since the real-official-source enrichment work documented below adds 2 more `_flat_enriched_e5large` collections): 5 legacy ISCO-08 (`isco_occupations` + major/submajor/minor/unit), 5 `official_ilo2021_v1`, 5 `official_ilo2021_v1_e5large`, 5 `official_ilo2021_v1_enriched`, 5 `official_ilo2021_v1_enriched_e5large` (each of the 4 ISCO-08 profiles = major/submajor/minor/unit + flat-unit), 4 `isic_rev4_*` (sections/divisions/groups/classes), 4 `isic_rev4_*_e5large` (same 4, e5-large), 1 `isic_rev4_classes_flat_e5large`, 1 `isic_rev4_classes_flat_enriched_e5large` (121 points — 134 minus 13 disclosed non-standard codes, see below), 3 `iscedf2013_*` (broad/narrow/detailed fields), 3 `iscedf2013_*_e5large` (same 3, e5-large), 1 `iscedf2013_detailed_fields_flat_e5large`, 1 `iscedf2013_detailed_fields_flat_enriched_e5large` (61 points — 63 minus 2 disclosed non-standard codes) |
| Redis 7 | internal :6379 | confirmed healthy |
| Ollama | http://localhost:11434 | confirmed healthy; **7 models present** (corrected 2026-08-24, re-checked live via `GET /api/tags` — the prior 3-model list was missing `qwen2.5:3b`, which Module J's own findings elsewhere in this document actively depend on, plus 3 more): `llama3.2:1b`, `llama3.2:latest`, `gemma3:4b`, `qwen2.5:3b`, `aya:latest`, `hf.co/CohereLabs/tiny-aya-global-GGUF:Q4_K_M`, `hf.co/CohereLabs/tiny-aya-fire-GGUF:Q4_K_M` |

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
**Full-scale confirmation, same day**: the 500-case e5-large-vs-e5-small
result was a preliminary sample — re-ran flat retrieval, no reranking,
on the complete 18,747-case heldout split (the identical cases and
config as the canonical 21.19% result). **Result: 29.70% (5,567/18,747)
vs. e5-small's 21.19% (3,973/18,747) — +8.50pp, McNemar exact
p ≈ 7.34×10⁻¹⁶⁷, Wilson 95% CIs [29.05%,30.35%] vs. [20.61%,21.78%],
non-overlapping.** This is now the headline, full-scale, statistically
decisive result — not an estimate. Real per-language breakdown (paired,
computed directly from both runs' per-case CSVs, not carried over from
the 500-case sample): Arabic 14.62%→28.84% (+14.22pp), Hindi
23.07%→30.95% (+7.88pp), Tagalog 14.47%→24.19% (+9.72pp), Urdu
15.33%→26.66% (+11.34pp), **English 37.98%→37.59% (−0.39pp, flat, not
meaningful)**. The gain is concentrated almost entirely in the four
non-English languages — English alone does not benefit. Real cost:
202ms/case full-scale mean (vs. e5-small's ~28ms/case under the same
measurement method) — ~7x slower, still fast in absolute terms. Real
artifact: `eval/local_runs/e5large_full_heldout_20260824/results.csv`.
Still not switched to production default — same open decision as above,
now backed by full-scale rather than sampled evidence.

**Fourth reranking check, later the same day** — the first of the four
to show any non-zero effect at all. Run against the real, previously-
unused **validation split** (642 cases, built 2026-08-23 — see the
3-way dev/validation/heldout split note below) on the *production-
default* profile (e5-small) rather than e5-large, covering a combination
the three checks above hadn't: **18.22%→18.69% (+0.47pp), 638/642
predictions byte-identical, McNemar exact p=0.25 on the 4 that
differed — not significant.** Investigated the 4 differences rather than
stopping at the aggregate number: **all 4 are the same occupation, "Air
force captain," gold code `0110`** — the already-disclosed Armed Forces
4-digit-vs-3-digit catalogue quirk (see "Knowledge base construction"
below). 3 of 4 flipped wrong→right under reranking. Read honestly: a
narrow, single-occupation effect plausibly tied to one known catalogue
oddity, not evidence that reranking generally helps — the three prior
confirmations' conclusion stands. Real artifacts:
`eval/results/dev_selection/validation_reranking_check/*.csv` (both
runs; correctly routed to `dev_selection/`, not `raw_runs/`, since a
validation-split result is never a citable confirmed result per this
project's own dev/validation-selects, heldout-confirms discipline).

**The biggest single accuracy finding in this project, same day, prompted
directly by "why isn't ISCO-08 accuracy better — go find the real
issue."** Root-caused rather than assumed: every one of the 436 official
catalogue entries' `embedding_text` was just `"{code} {2-4 word title}"`
(`backend/rag/official_isco08_catalogue.py:282`, e.g. `"6122 Poultry
Producers"`) — while the official ILO source workbook this project
already has on disk (`ISCO-08_EN_Structure_and_definitions.xlsx`) carries
full `Definition`, `Tasks include`, and `Included occupations` (real
example job titles) columns for every entry, silently discarded by
`eval/normalize_ilo_isco08_catalogue.py` at the very first processing
step. Measured a real "magnet" effect this caused: several thin-text
codes (e.g. `5165` "Driving Instructors") were predicted 15–55× more
often than their true frequency in the WISCO heldout set. **Validated
the fix against the real embedding model before building anything**: 19
of 20 real wrongly-matched queries showed reduced similarity to the
wrong "magnet" code once enriched; 4 of 5 real cases flipped from wrong
to correct when both the wrong and true codes were enriched (e.g. "Piano
tutor (private tuition)" correctly favored `2354` Other Music Teachers —
whose official ILO example list literally contains "Piano teacher
(private tuition)" — over `5165`, once both carried real text).

Built additively, same discipline as `E5LARGE_PROFILE`: `eval/normalize_
ilo_isco08_catalogue.py` gained an opt-in `normalize_enriched()` (the
original 4-column output's SHA-256 is unchanged, verified byte-for-byte
identical — the existing hash-locked pipeline is completely untouched);
`backend/rag/official_isco08_catalogue.py` gained `ENRICHED_PROFILE`
(`"official_ilo2021_v1_enriched"`) and a separate `load_enriched_
catalogue()` with its own hash check; `backend/rag/
build_official_isco08_collections_enriched.py` (new, mirrors the
e5-large build script) built and verified all 5 collections (619 + 436
points). Same `intfloat/multilingual-e5-small` model as the default
profile — only the input text changed, not the model.

**Full 18,747-case heldout result (identical cases/config as the
canonical 21.19% baseline)**: **32.55% (6,102/18,747) vs. 21.19%
(3,973/18,747) — +11.36pp, McNemar p ≈ 2.19×10⁻²⁷⁴.** Larger than the
e5-large gain (+8.50pp) and markedly cheaper — same model, same
inference speed, zero extra compute. Per-language, and unlike e5-large
this genuinely helps every language including English (which e5-large
left flat): Arabic 14.62%→25.15% (+10.53pp), English 37.98%→**54.85%**
(**+16.87pp**, the largest gain of any language), Hindi 23.07%→33.80%
(+10.73pp), Tagalog 14.47%→23.13% (+8.66pp), Urdu 15.33%→25.19%
(+9.87pp). Magnet effect confirmed resolved on the same codes: `6122`
468→24 predictions, `5165` 276→31, `6114` 208→6, `8153` 203→7 — all now
close to their true frequency instead of vacuuming up unrelated queries.
Real artifacts: `eval/results/dev_selection/enriched_catalogue_
validation_check/` (642-case first check, +14.33pp, McNemar
p=2.65×10⁻¹⁷), `eval/results/raw_runs/enriched_catalogue_
heldout_20260824/` (the full, citable 18,747-case confirmation).

**A real caveat, caught by direct user correction, then actually
tested**: the 32.55% figure above was run with `--use-llm-reranker off`
— retrieval-only, not real production behavior. `enable_llm=True` is
production's actual default, and the LLM step only skips via a fast
path at confidence ≥ 0.92; checked directly against the full heldout
run's own data — **99.7% of cases (18,692/18,747) are below that
threshold**, so the LLM step fires for nearly every real query. Genuine
gap in what was first reported, not a technicality. **Re-tested with the
reranker actually enabled** (Groq, 642-case validation split,
`reranker_fired=True` confirmed on 642/642): **32.55%→32.87%, +0.31pp,
McNemar p=0.5 — not significant**, only 3/642 predictions changed (same
Armed Forces `0110` edge case as before). **5th independent confirmation**
that LLM reranking doesn't move ISCO-08 accuracy, now shown on the
enriched catalogue too. Correct framing going forward: the LLM step
fires on ~99.7% of real queries, but essentially never changes the
outcome when it does — state both facts, not just the aggregate number.

**Combination test, answered at full scale, same day.** Built a third
profile, `official_ilo2021_v1_enriched_e5large` (new: `backend/rag/
build_official_isco08_collections_enriched_e5large.py`) — same enriched
embedding_text as `ENRICHED_PROFILE`, embedded with
`intfloat/multilingual-e5-large` instead of `-small`. Same additive
discipline: new collection names, zero changes to anything already
published. Validation-split preview first (642 cases, +7.32pp over
enrichment alone, McNemar p=7.3×10⁻⁵), then the full 18,747-case
heldout: **32.55%→40.95% (+8.40pp over enrichment alone, +19.75pp over
the original 21.19% baseline). 95% Wilson CI [40.24%, 41.65%]. McNemar
p≈9.8×10⁻¹⁴³ vs. enrichment alone, p≈0 vs. the original baseline.** The
two interventions are genuinely additive, not redundant — confirming
they work via independent mechanisms (embedding-model capacity vs.
input-text richness). Per-language, every language gained substantially
over the original baseline: Arabic 14.62%→37.77% (+23.15pp), English
37.98%→56.91% (+18.94pp), Hindi 23.07%→40.89% (+17.82pp), Tagalog
14.47%→33.17% (+18.69pp), Urdu 15.33%→35.53% (+20.21pp). **This is now
the headline ISCO-08 accuracy result for the thesis — nearly double the
original canonical baseline.** Real artifacts: `eval/results/
dev_selection/enriched_e5large_validation_check/` (642-case preview),
`eval/results/raw_runs/enriched_e5large_heldout_20260824/` (the full,
citable confirmation). **Not yet switched to production default** — same
category of decision as before, now resting on the strongest evidence in
the project.

**Corrective RAG retry, re-tested on the new best config, same day.**
The one prior data point (n=63, "within run-to-run noise") predates
every finding above. `eval/run_eval.py` gained a `--use-corrective-retry
{on,off}` flag (previously untested by the main harness at all). First
attempt via Groq — **invalid, same standard as the earlier
Gemini-quota case**: 425 rate-limit errors, 28 failed reformulation
calls across 60 cases, Groq's 8000 TPM limit couldn't keep pace with
corrective retry's doubled LLM-call pattern. Re-ran with a local model
(`ollama/qwen2.5:3b`, no rate limit) — clean, zero errors: **31/60 =
51.67% both with and without corrective retry, exactly identical,
McNemar exact p=1** (2 cases improved, 2 regressed, perfectly
cancelling). Extends the "generation doesn't move accuracy" finding to
a genuinely different mechanism — corrective retry reformulates the
query itself, not just candidate selection like reranking — so this
wasn't a guaranteed replication, and it held anyway. **6th independent
confirmation.** n=60 is real and clean but still modest — a solid
signal, not full-scale certainty. Real artifacts: `eval/results/
dev_selection/corrective_retry_check_ollama/` and
`corrective_retry_check_ollama_off/`.

Real, currently-pinned dependency versions. **Correction, 2026-08-24**:
this section previously claimed `requirements.txt` "pins nothing, all
bare package names, verified across this project's entire git history"
— checked directly against the live file and that claim is false, and
has been since commit `c5aad9c` ("Pin requirements.txt to real,
pip-freeze-verified versions (28 packages)"), well before this document's
prior verification passes. `requirements.txt` is the real, full,
pip-freeze-verified application dependency file (28 packages, `==`
pins throughout) — it is the primary source, not `requirements-dev.txt`.
`requirements-dev.txt` is a deliberately narrower, separately-pinned
subset (10 packages) covering only what the B2 non-inference test suite
imports (see its own header comment for the exact test list); where the
two files list the same package, their versions agree (`crewai`,
`qdrant-client`, `sentence-transformers`, `pytest`, `pytest-asyncio`,
`python-dotenv`, `rank-bm25` all match). A third, separately-scoped file,
`backend/evaluation/wisco/requirements.lock.txt`, pins only what Module
A's WISCO scripts add on top (`openpyxl`, `et_xmlfile`) — self-documented
in its own header, not a duplicate of either file above. Versions below
are from `requirements.txt` directly:

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

## API (20 routes, from the live OpenAPI spec)

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
| `isced_classifier.py` | **Yes — changed 2026-08-24** (see below) |
| `language_processor.py` | Yes |
| `rag_expert.py` | Yes |
| `report_generator.py` | Yes |
| `semantic_relation.py` | Yes |
| `validation_agent.py` | Yes |
| `nationality_classifier.py` | No |
| `person_register.py` | No — deterministic |
| `isco_reranker_strict.py` | No |
| `classifier_methods.py` | Not an agent — shared constants |
| `method_registry.py` | Not an agent — registry/introspection utility |

**Correction, 2026-08-24** (found during a documentation-completeness
audit — this table had gone stale and nobody had re-checked it against
the code since): `isced_classifier.py` used to be correctly "No" here,
but the same-day corrective-RAG-retry port (see "Knowledge base
construction" below) added a real `Agent(...)` construction at two call
sites (`backend/agents/isced_classifier.py:749` and `:846`, the LLM
rerank and corrective-retry-reformulation paths) — this table was never
updated when that code shipped. **13 modules now construct a
`crewai.Agent`, not 12** — this also makes the "11-12 modules... never
13" claim in the corrections table near the top of this document
(under "Agent count") stale; that row is left as written since it
describes a comparison against the externally-supplied draft made on
2026-08-12, before this change existed, but should not be read as a
current fact. All 13 that do construct an `Agent` set
`allow_delegation=False` consistently (checked directly, including both
of `isced_classifier.py`'s new call sites). There is still no CrewAI
hierarchical manager-delegation process anywhere in this codebase.

## Semantic Relation Engine — verified formula

**Corrected 2026-08-24** — re-checked directly against the live file
during the documentation-completeness audit; the previous version of
this section had drifted on three counts: stale line numbers, renamed
variables (`score_isic`/`score_isced` → `score_parts[0]`/`score_parts[1]`
at some point after this was first written), and — more substantively —
it omitted a real partial-credit step entirely, and conflated two
genuinely different scoring mechanisms (the confidence-adjustment bands
vs. the violation-severity bands) as if they were one. Confirmed present
in `backend/agents/semantic_relation.py`, inside `analyse()`
(method starts line 340):

```python
# Lines 392-406 — weighted coherence score, with graceful degradation
# when only one cross-standard dimension is available:
if isic_section and isced_level is not None:
    raw_score = 0.55 * score_parts[0] + 0.45 * score_parts[1]
elif isic_section:
    raw_score = float(score_parts[0])
elif isced_level is not None:
    raw_score = float(score_parts[1])
else:
    raw_score = 0.80   # no cross-standard data -> assume plausible

# Partial credit: MODERATE-severity violations soften the penalty --
# this step was previously undocumented here entirely.
moderate_violations = sum(1 for v in violations if v.severity == "MODERATE")
raw_score = max(0.0, raw_score + 0.15 * moderate_violations * (1 - raw_score))

score = round(min(raw_score, 1.0), 4)
is_coherent = score >= 0.70

# Lines 408-416 — confidence adjustment bands (affect the CALLER's
# confidence, not violation severity): >=0.90 -> +0.10, >=0.70 -> +0.05,
# >=0.50 -> -0.05, else -0.20.
```

**Violation severity (LOW/MODERATE/HIGH) is a separate, gap-based
mechanism** (lines 551-564 of `_check_isco_isced`, not the score bands
above) — how many ISCED levels the respondent's level falls outside the
expected `[min_l, max_l]` range: `gap<=1` → LOW, `gap==2` → MODERATE,
`gap>2` → HIGH. LOW is a real, genuinely-emitted third tier (`rule_id`s
`SR-ISCO-ISCED-01/02/03`), confirmed by reading the actual severity
assignment, not inferred from the comment claiming it.

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
- **A second, more serious real bug in the same functions, found and
  fixed 2026-08-24 while porting corrective retry to ISIC/ISCED**:
  `ISCEDClassifier._score_level()`/`_score_field_candidates()` and
  `ISICClassifier._keyword_score()` tokenised query text via
  `set(re.findall(...))`. A Python `set`'s iteration order depends on
  `PYTHONHASHSEED`, which is **randomised by default every time a Python
  process starts** — so whenever two candidates tied on keyword-hit
  count (a common case with this integer-count scoring — e.g. "bachelor"
  (ISCED level 6) and "education" (level 0, via the level-0 keyword
  string's own tokenised "no education") both hitting once for a real
  input text), `max(hit_counts, key=...)`'s tie-break silently depended
  on which token the hash-randomised set happened to iterate first.
  **Confirmed directly**: the identical input text classified to a
  different ISCED level across different `PYTHONHASHSEED` values (tested
  0 through 6) — meaning **the same respondent answer could classify
  differently depending on which process/server restart handled it**, a
  real production non-determinism bug, not just a test flake (it was
  first caught as an intermittently-failing test). Root cause traced by
  comparing against `ISCOClassifier._keyword_major_hint()`, which was
  never affected because it already iterates `re.findall()`'s list
  directly rather than wrapping it in `set()`. Fixed in all three
  functions by switching to `dict.fromkeys(re.findall(...))`, which
  dedupes while preserving each token's real first-appearance order in
  the source text — deterministic, and a more defensible tie-break rule
  than an arbitrary hash. 5 new regression tests, including one that
  greps the fixed function's own source for `dict.fromkeys(` and asserts
  `set(re.findall` is absent, as a guard against this exact bug being
  reintroduced by a future refactor.
- **Corrective RAG retry ported to ISIC/ISCED, 2026-08-24**: previously
  ISCO-08 only. `ISICClassifier`/`ISCEDClassifier` gained
  `enable_corrective_retry` mirroring `ISCOClassifier`'s exactly (same
  gap-based accept rule: retry replaces the original only if it produces
  a strictly wider top1/top2 candidate-score gap, never on raw
  confidence alone). ISCED's retry is scoped to the FIELD dimension only
  — level stays deterministic. Default `False`, zero behavioural change
  for existing callers. 10 new tests.
- **ISIC/ISCED-F e5-large embedding profile, 2026-08-25** — prompted
  directly by "close the ISIC/ISCED gap with ISCO-08." Checked before
  building anything: does the ISCO-08 "magnet effect" catalogue-text bug
  (see the 32.55% enrichment finding above) apply here too? **No** — real
  investigation, not assumed. `_ISIC_DATA`/`_ISCED_FIELDS` already carry
  real multilingual keyword strings per entry, and
  `backend/rag/hierarchy_nodes.py` already aggregates descendant keywords
  bottom-up into every internal node's index text — the richness ISCO-08's
  catalogue was missing before enrichment. The real, verified gap was
  different: ISIC/ISCED-F's hierarchical store had no e5-large option at
  all (`MODEL_NAME` was hardcoded to e5-small in
  `backend/rag/standard_hierarchical_store.py`, no profile concept
  existed), unlike ISCO-08's own store. Closed additively, same pattern as
  every ISCO-08 profile: `PROFILE_MODEL_CONFIG`,
  `ISIC_COLLECTIONS_BY_PROFILE`/`ISCEDF_COLLECTIONS_BY_PROFILE`, and a
  `profile=` parameter on `isic_stages()`/`iscedf_stages()` and both
  `get_*_hierarchical_store()` factories; `ISIC_COLLECTIONS`/
  `ISCEDF_COLLECTIONS` (no suffix) remain exact aliases of the `e5_small`
  profile, so every existing caller — including both classifiers'
  `_classify_hierarchical()`, which still calls the factories with no
  `profile=` argument — is byte-for-byte unaffected. Built and verified
  live: `isic_rev4_{sections,divisions,groups,classes}_e5large` (341
  nodes total) and `iscedf2013_{broad,narrow,detailed}_fields_e5large` (99
  nodes). Direct store-level verification (not through the classifiers,
  to avoid touching production code without evaluation evidence behind
  it): real queries against both new stores returned `ready=True`,
  `unavailable_reason==""`, and semantically correct codes — e.g. "I build
  mobile apps at a software company" → ISIC `6201` (Computer programming
  activities, the exact example in `isic_classifier.py`'s own docstring),
  "construction labourer on a residential building site" → ISIC `4100`
  (Construction of buildings), "Bachelor's degree in computer science" →
  ISCED-F `0613`. Proof the collections are live and serving, **not an
  accuracy claim**. 26 new tests
  (`backend/tests/test_standard_hierarchical_store_e5large_profile.py`
  plus 2 in `test_build_standard_hierarchical_collections.py`); full
  suite re-run with zero regressions. **The classifiers were not changed
  and do not use this profile in production** — infrastructure parity
  with ISCO-08, not a production switch.

  **What this does NOT close, and no amount of further engineering can**:
  unlike ISCO-08, there is still no labelled evaluation dataset for
  ISIC/ISCED-F. WISCO (every ISCO-08 accuracy number in this document) is
  occupation-only — no industry or field-of-study gold labels. The only
  labelled data touching ISIC/ISCED-F anywhere in this repo is
  `eval/fixtures/synthetic_lfs_intake_package/synthetic_test_set.csv` — 5
  rows, every field explicitly prefixed `"SYNTHETIC EXAMPLE"`, built for
  Module F's pre-fill stress test, and missing ISCED-F **field** gold
  labels entirely (only the independent ISCED **level** dimension is
  present). Deliberately not used as an accuracy source — 5 synthetic
  rows dressed up as a benchmark would be exactly the kind of
  overclaiming this document exists to prevent. So the honest state stays:
  ISCO-08 has a real, full-scale, best-tested number (40.95%); ISIC/ISCED-F
  have real, tested, now-at-parity infrastructure and a real "method used"
  label, but **no accuracy number exists or can be honestly produced for
  them without new labelled data** — see
  `Documentation/Conference_I_Reviewer_2/
  ISIC_ISCEDF_HIERARCHICAL_RETRIEVAL_IMPLEMENTATION.md`'s new "Why this
  remains the one gap infrastructure cannot close" section for the full
  writeup.
- **ISIC/ISCED-F flat retrieval, 2026-08-25, same day** — a direct,
  explicit user correction to the entry above: "we need to use the same
  implementation what ISCO08 is implemented, because its novel
  contribution." Checked what that actually requires before building:
  ISCO-08's own **best-tested, headline configuration is FLAT retrieval**
  (40.95%), not hierarchical — ISCO-08's hierarchical retrieval measurably
  **underperformed** its own flat retrieval (10.35% vs 21.19%, see the
  WISCO Tier-1 table above). So the hierarchical-only path the e5-large
  entry above added, on its own, did not actually mirror ISCO-08's real
  best implementation — it mirrored ISCO-08's *worse* one. Closed for
  real: `backend/rag/standard_hierarchical_store.py` gained
  `StandardFlatStore` (single-collection direct query, no parent-chain
  traversal — deliberately not built on `HierarchyBeamSearchEngine`,
  which requires ≥2 stages by its own docstring) plus
  `ISIC_FLAT_COLLECTIONS_BY_PROFILE`/`ISCEDF_FLAT_COLLECTIONS_BY_PROFILE`
  and `get_isic_flat_store()`/`get_iscedf_flat_store()`; the build script
  gained `--flat`, reusing the SAME leaf-level derived nodes as the
  hierarchical build's own final stage (no new node-derivation logic —
  matches ISCO-08's own precedent of building "flat" and hierarchical-leaf
  collections from identical source records). Both classifiers gained
  `classify(text, method=ISIC_FLAT_RETRIEVAL / ISCEDF_FLAT_RETRIEVAL)`,
  **hardcoded internally to `profile="e5_large"`** — not a caller choice
  — because flat+e5-large specifically is the recipe being mirrored, not
  flat-with-whatever-model. Built and verified live:
  `isic_rev4_classes_flat_e5large` (134 points) and
  `iscedf2013_detailed_fields_flat_e5large` (63 points). Full
  section/division/group (ISIC) and broad/narrow (ISCED-F) ancestry is
  reconstructed via new `_ENTRY_BY_CLASS`/`_ENTRY_BY_DETAILED` lookups,
  since the flat result only ever carries the single leaf code. 34 new
  tests (`test_standard_flat_store.py` plus additions to
  `test_isic_classifier.py`/`test_isced_classifier.py`/
  `test_build_standard_hierarchical_collections.py`); full suite re-run,
  zero regressions. `classify(text)` (no `method=`) is byte-for-byte
  unchanged for both classifiers — same discipline as every prior
  addition, infrastructure/architecture parity, not a production-default
  switch. **What this does not, and cannot, resolve**: whether flat (or
  hierarchical) retrieval is actually more accurate than the existing
  keyword/LLM pipeline for ISIC or ISCED-F is still genuinely untested —
  the no-labelled-data blocker above is unchanged by this. This closes
  the *implementation-parity* gap the user pointed at (the architecture
  diagram showing ISCO-08 with a best-tested RAG config and ISIC/ISCED-F
  with only "keyword match"), not an accuracy gap — those remain two
  different, correctly-distinguished things.
- **ISIC/ISCED-F real official-source enrichment + a genuine magnet-effect
  regression found and fixed, 2026-08-25, same day** — a direct user
  instruction to double-check the "already rich, no fix needed" claim two
  entries above, before moving on. Re-verified rather than re-asserted:
  measured ISIC/ISCED-F's keyword text objectively (mean 11.4 / 9.7 words)
  against the REAL ISCO-08 official workbook's actual enriched content
  (50-150+ word prose definitions + real example lists) — not against
  ISCO-08's original bug state as the earlier comparison had done. The
  richness gap was real. Checked whether an equivalent official source
  even exists for ISIC/ISCED-F (it hadn't been checked before dismissing
  the idea) — it does: the UN Statistics Division's ISIC Rev.4 structure
  publication and UNESCO UIS's ISCED-F 2013 detailed field descriptions,
  both real, both public. Downloaded both (`eval/local_catalogues/
  isic_rev4_2008/`, `eval/local_catalogues/iscedf_2013/`), built a
  reproducible parser (`eval/parse_official_isic_iscedf_definitions.py`,
  using `pdfplumber`, already a dependency) extracting real per-code
  definitions/examples — 419/419 ISIC classes and 92 ISCED-F fields parsed
  correctly, verified via multiple full read-throughs against the raw PDF
  text, not just spot-checked.

  **A second, larger, unplanned finding surfaced by this same
  verification**: cross-checking `_ISIC_DATA`/`_ISCED_FIELDS`'s already-
  embedded codes against the real official document found 13 ISIC codes
  and 2 ISCED-F codes that **do not exist in the official standard at
  all** — e.g. `_ISIC_DATA`'s `"7311 Advertising agencies"` vs the real
  official `"7310 Advertising"`; `"9001 Performing arts"`/`"9003 Artistic
  creation"` vs the real single class `"9000 Creative, arts and
  entertainment activities"` (no 9001/9003 split exists in ISIC Rev.4 at
  all). The same class of bug Task 20/21's primary-source audit found and
  fixed in the ISCO-08 catalogue (19 non-standard codes there) — genuinely
  new here, not previously known, and **not fixed in this pass** (deciding
  the correct replacement code for each requires its own careful audit,
  out of scope for a text-enrichment task). Disclosed precisely via
  `backend/rag/official_source_enrichment.py`'s `NON_STANDARD_ISIC_CODES`
  / `NON_STANDARD_ISCEDF_CODES`.

  Built `backend/rag/official_source_enrichment.py`
  (`build_enriched_text()`, real definition+examples where a match exists,
  falls back to the existing title+keywords text otherwise — never
  fabricates), wired into a new `"enriched_e5large"` profile on the flat
  collections only (mirrors the specific flat+enriched+e5-large recipe
  that IS ISCO-08's actual best-tested config). **`ISIC_FLAT_RETRIEVAL`/
  `ISCEDF_FLAT_RETRIEVAL` now use this profile** (changed from the plain
  `e5_large` profile the prior entry built) — this is genuinely the same
  implementation ISCO-08 uses, not just the same architecture family.

  **Live-tested before declaring this done, per this project's own
  standing discipline — and a real regression was caught doing so**: the
  first live build (all 134/63 codes included, non-standard ones kept
  with their thin fallback text) showed a genuine magnet effect —
  "I build mobile apps at a software company" and "construction labourer
  on a residential building site" both wrongly matched non-standard code
  `8899` instead of their real codes (`6201`, `4100`); a 15-query broad
  smoke test showed 3 of the 13 non-standard codes actively capturing
  unrelated queries. Exact same mechanism as ISCO-08's own pre-enrichment
  magnet bug (see the 32.55% enrichment finding above) — once every OTHER
  code's text got much richer, the already-thin non-standard entries stood
  out as disproportionately generic-looking matches. **Fixed by excluding
  `NON_STANDARD_ISIC_CODES`/`NON_STANDARD_ISCEDF_CODES` from this specific
  collection entirely** (not a coverage loss in any meaningful sense,
  since these codes were already confirmed not to correspond to any real
  official code — a query that would have hit one now correctly falls
  through to its real neighbouring code, e.g. excluding `7311` means
  "advertising agency" queries now correctly land on `7310`, the actual
  official code for the same concept). Rebuilt both collections (121/61
  points); re-ran the same 15-query smoke test — zero repeated codes
  across all 15 queries (every prior magnet resolved), remaining
  differences are ordinary adjacent-category ambiguity (e.g. "nurse" →
  `8690` Other human health activities vs the expected `8610` Hospital
  activities), not a systemic bug. Live-verified end-to-end through the
  real classifiers, not just the store layer.

  60 new tests (`test_official_source_enrichment.py` plus additions to
  `test_build_standard_hierarchical_collections.py`); full suite re-run,
  zero regressions. **Still cannot be turned into an accuracy claim** — no
  labelled ISIC/ISCED-F evaluation data exists, unchanged by any of this
  work. What changed: ISIC/ISCED-F's `ISIC_FLAT_RETRIEVAL`/
  `ISCEDF_FLAT_RETRIEVAL` methods are now genuinely, not just
  architecturally, the same implementation ISCO-08's own best-tested
  config uses — real official enriched text, e5-large embeddings, flat
  retrieval — with a real, live-caught, live-fixed data-quality issue
  along the way. `classify(text)` (no `method=`) remains byte-for-byte
  unchanged for both classifiers throughout.
- **A real, if synthetic and incomplete, ISIC/ISCED-F accuracy number now
  exists, 2026-08-26/27** — directly supersedes the "still cannot be
  turned into an accuracy claim" line in the entry directly above. Prompted
  by an explicit instruction to generate a real evaluation dataset via a
  standard, defensible synthetic-data methodology, covering all 5 project
  languages, since WISCO doesn't cover these standards and — at the time
  this work started — the IPUMS correspondence (see below) had not yet
  received a reply.

  **Method**: taxonomy-grounded LLM paraphrase data augmentation — a real,
  standard technique for bootstrapping labelled evaluation data from a
  classification schema's own official definitions when real annotated
  examples don't exist (the same family used across NLP to build
  taxonomy-classification benchmarks from a schema document). Every
  generated example is grounded in the real official definition/examples
  from `backend/rag/official_source_enrichment.py` (the same data behind
  the `enriched_e5large` profile above) and explicitly instructed to
  paraphrase into casual respondent language, not echo official
  terminology — so a keyword classifier can't trivially "win" by matching
  the source text back to itself. New script:
  `eval/generate_synthetic_isic_iscedf_benchmark.py`. Full disclosure in
  that script's own docstring: **not real respondent data, not a
  substitute for real correspondence-sourced data or the Module E pilot**
  — genuinely useful for regression-testing and directional comparison
  between methods, never to be cited as WISCO-equivalent or pilot-grade
  accuracy.

  **IPUMS International correspondence — closed, 2026-08-27, definitive
  negative answer, not a non-reply.** The line above ("had not yet
  received a reply") was true only when the synthetic-data work started;
  a real reply arrived and was verified before this document was updated.
  Full transcript: `Documentation/Phase_2/Week_1/
  ipums_correspondence_log.md`. Sivarama emailed IPUMS User Support
  2026-08-24 asking specifically whether the Egypt/Jordan (Arabic), India
  (Hindi), and Pakistan (Urdu) samples distribute original verbatim
  occupation/industry write-in text, or only final coded values.
  Isabel Pastoor (IPUMS User Support) replied 2026-08-25 (two messages,
  the second after consulting the IPUMS International team directly):
  **IPUMS International does not have access to original string variables
  for occupation, industry, or education for these samples at all** — only
  coded variables are ever received from national statistical agencies,
  and even where a string variable exists internally for some samples, data-
  provider agreements preclude IPUMS from ever distributing it to users.
  This is a real, sourced dead end — the same class of finding as Module
  D's ISCO-08 Vol. I crosswalk search (a real check that came back
  negative, not an unanswered question) — and it **closes the IPUMS
  avenue for this project**, not just for this correspondence. It does
  not change the synthetic benchmark's own disclosed status above. The
  two remaining real (non-synthetic) paths, neither attempted in this
  pass: extending Module E's pilot scope to capture industry/education
  alongside occupation, or a direct approach to the relevant national
  statistical agencies (Isabel Pastoor's own suggestion) — a materially
  higher-effort path with its own likely ethics-consideration
  requirements, not something this correspondence itself provides.

  **Model selection, and a real quality-driven correction along the
  way**: a first attempt used local Ollama (`qwen2.5:3b`) exclusively.
  Directly inspecting the output caught real, disqualifying problems —
  Arabic generations mixed in literal Chinese characters mid-sentence
  (e.g. "أنا程序员"), Urdu output was grammatically broken, and a larger
  multilingual-specialist model (`aya:latest`) timed out (>120s/call) on
  this hardware. Switched to Groq (`groq/openai/gpt-oss-120b` — already
  used successfully for ISCO-08 reranking) after directly comparing
  output: clean, natural text in every language tested, ~1-2s/call.

  **Coverage actually achieved, and why it's incomplete — disclosed, not
  hidden**: target was 1,092 rows (182 matched codes × 6 languages × 1
  example). Groq's real, confirmed constraints — first an 8000
  tokens-PER-MINUTE ceiling (fixed via proactive request pacing plus
  retry/backoff, both real code, not just intent), then, after roughly 90
  minutes of combined generation across two passes, a hard **200,000
  tokens-PER-DAY** ceiling (confirmed directly from the API's own error:
  "Used 199,703, Limit 200,000") — capped the real total at **372 rows**
  (34% of target). A same-session attempt to route the remainder through
  OpenRouter's free-tier models failed outright (every tested free model
  slug on this account returned 404/unavailable). This is the same class
  of constraint as this document's own already-disclosed Gemini-quota
  precedent — marked invalid/incomplete honestly rather than presented as
  full coverage. Real per-language counts in the final 372: en 66, hi 69,
  ur 68, tl 63, ar 58, ar-gulf 48 — reasonably even, not concentrated in
  one language. A resume/fill-in capability
  (`--skip-existing`, deterministic case-ID matching) was built into the
  generator specifically so a future session can top up the remaining 720
  rows once the daily quota resets, without re-spending budget on the 372
  that already succeeded.

  **The result** (`eval/run_synthetic_isic_iscedf_eval.py`, real computed
  numbers over the full 372-row set, both standards, all 6 languages,
  Wilson 95% CI, McNemar exact test):

  | | n | legacy keyword/LLM | flat_retrieval (enriched_e5large) | McNemar p |
  |---|---:|---:|---:|---:|
  | Overall | 372 | 13.98% [10.82%, 17.87%] | **83.06%** [78.92%, 86.53%] | 9.37×10⁻⁶⁸ |
  | ISIC | 254 | 12.99% | 80.71% | 2.89×10⁻⁴⁴ |
  | ISCED-F | 118 | 16.10% | 88.14% | 1.14×10⁻²⁴ |
  | en | 66 | 31.82% | 86.36% | 5.63×10⁻⁹ |
  | ar | 58 | 17.24% | 84.48% | 3.82×10⁻¹¹ |
  | ar-gulf | 48 | 22.92% | 79.17% | 4.63×10⁻⁷ |
  | hi | 69 | 1.45% | 84.06% | 2.08×10⁻¹⁶ |
  | ur | 68 | 4.41% | 83.82% | 1.58×10⁻¹⁵ |
  | tl | 63 | 9.52% | 79.37% | 1.14×10⁻¹³ |

  A large, statistically overwhelming gap in every breakdown — the legacy
  keyword pipeline is essentially non-functional on Hindi/Urdu/Tagalog
  (1-10%, since `_ISIC_DATA`/`_ISCED_FIELDS`'s hand-built "keywords" field
  has virtually no coverage in those languages), while `flat_retrieval`
  (multilingual e5-large embeddings) stays in a consistent 79-88% band
  across every one of the 6 languages. **Read honestly, per this
  document's own discipline**: this measures how well each method
  recovers the class its own LLM-generated prompt was built from — a real
  signal, the best available in the current absence of real respondent
  data, but not equivalent to a WISCO-style external validation. The
  directional conclusion (RAG-based multilingual retrieval dramatically
  outperforms English-only keyword matching for non-English LFS
  respondents) is exactly what the project's architecture already
  predicted; this is the first real, computed number behind that
  prediction for these two standards.

  **A genuine bonus finding — closes a previously-blocked gap (Module G,
  "multilingual validation")**: Module G's planned Gulf Arabic dialect-
  normalization A/B test had been blocked because WISCO's Arabic data was
  confirmed to have zero dialectal content — no real Gulf-dialect text
  existed to test `LanguageProcessor._normalise_gulf_arabic()`'s 79-term
  dictionary against. The `ar-gulf` rows here are genuine dialectal text
  (LLM-instructed to use colloquial Khaleeji Arabic; confirmed using real
  Gulf markers like "إحنا" vs. MSA "نحن"). New script:
  `eval/gulf_arabic_dialect_normalization_ab_test.py`. Real result on the
  full 48 ar-gulf rows: the marker dictionary's detection rate is
  **12.5% (6/48)** — most LLM-generated Gulf dialect text doesn't trip any
  of the 79 hand-curated markers, a real, disclosed, low-coverage finding.
  Classification accuracy was **byte-identical with and without
  normalization** (38/48 = 79.17% both ways, McNemar b=0 c=0 p=1) — the
  multilingual e5-large embedding model already handles Gulf dialect
  vocabulary robustly without help from the hand-built normalizer, for
  this specific downstream task. This extends the project's now-repeated
  "extra processing doesn't move the needle" pattern (LLM reranking, 6
  independent nulls; corrective retry, 1 null) to dialect normalization —
  a 7th instance, on real generated dialectal text, in an area that
  previously had NO real data to test against at all.

  19 new tests across `test_generate_synthetic_isic_iscedf_benchmark.py`
  and `test_run_synthetic_isic_iscedf_eval.py` (hermetic — prompt
  construction, definition truncation, resume/skip-existing logic, and
  the accuracy-reporting arithmetic against hand-computed fixture values;
  no live LLM call in the test suite itself); full suite re-run, zero
  regressions.

  **2026-08-27, same effort continued** — the Groq daily quota reset
  (confirmed live: a trivial test call succeeded again), so the
  `--skip-existing` resume path was used to top up the missing (code,
  language) pairs a user check had correctly flagged (only 4 of 131
  populated codes had all 6 languages at the 372-row stage). A second
  fill-in pass added 77 more rows before hitting the same daily TPD
  ceiling again (confirmed via the identical error message, now at
  199,632/200,000) — **final real total: 449/1,092 rows (41%
  coverage)**, 10 of 143 populated codes now have all 6 languages.
  Per-language counts: hi 88, ur 78, en 78, tl 73, ar 71, ar-gulf 61.

  **A real infrastructure bug caught and corrected, not glossed over**:
  the first evaluation re-run on the enlarged 449-row set, and the first
  Gulf A/B re-run, were both launched concurrently and both silently
  degraded — many `StandardFlatStore` calls failed with a genuine OS-level
  error ("The paging file is too small for this operation to complete",
  Windows error 1455) and Ollama's local server also returned HTTP 500s.
  Root cause, confirmed directly (`docker stats`, `wmic OS get
  FreePhysicalMemory`): this machine has only ~8GB total RAM; Qdrant alone
  now holds ~1.3GB resident (41 collections, several with 1024-dim
  e5-large vectors, built across this session's work), leaving too little
  headroom for two concurrently-loaded `multilingual-e5-large`
  `SentenceTransformer` instances (~1-1.5GB each) plus CrewAI/local-LLM
  overhead. The degraded run's Gulf-Arabic accuracy (45.90%, badly out of
  line with the clean 79-80% band) was the tell. **Fixed by re-running
  both evaluations strictly sequentially** (never concurrently) —
  confirmed zero paging errors on the clean re-runs, and the resulting
  numbers landed close to the earlier smaller-sample results, as expected
  for a real, stable effect: **overall 82.85% (372/449) vs. 13.14%
  (59/449), McNemar p≈5.5×10⁻⁸³** (vs. 83.06%/13.98% at n=372 — consistent
  within sampling noise); Gulf A/B **80.33% (49/61) identical with/without
  normalization** (vs. 79.17% at n=48; detection rate 13.11%, vs. 12.5%).
  **A standing, disclosed operational constraint for this session's local
  dev machine, not fixed at the OS level**: further concurrent heavy
  eval/generation work on this machine should be run strictly
  sequentially, one memory-heavy process at a time, given the real,
  confirmed low-RAM ceiling.

  **Code review, 2026-08-27, same effort continued** — a requested
  line-by-line review of the whole ISIC/ISCED-F flat-retrieval diff
  (multi-agent, 4 independent angles). Every finding was independently
  re-verified before being acted on, not accepted at face value:

  - **Real, reproduced bug, fixed**: `python -m backend.rag.
    build_standard_hierarchical_collections --standard isic --dry-run
    --profile enriched_e5large` (a documented, argparse-valid flag
    combination, just missing `--flat`) crashed with a bare, unexplained
    `KeyError: 'enriched_e5large'` — reproduced directly before touching
    any code. Root cause: `enriched_e5large` conflates two independent
    axes (embedding model vs. catalogue-text source) into one profile
    key, and only has entries in the *flat* collection dicts, not the
    hierarchical ones. Fixed with a shared `_hierarchical_collections_for()`
    guard (`standard_hierarchical_store.py`) used by `isic_stages()`/
    `iscedf_stages()` *and* the CLI's `dry_run()`/`execute_run()` — now a
    clear `ValueError` explaining exactly what to do instead, from every
    call site, not just one. 4 new regression tests reproduce the exact
    prior crash and assert the new message.
  - **Real test gap, fixed**: neither `test_isic_classifier.py` nor
    `test_isced_classifier.py`'s flat-store test mocks ever asserted
    *which* profile string `_classify_flat` actually requests — the
    mock's `lambda profile="e5_large": store` silently accepted and
    ignored any value, including the real `"enriched_e5large"` argument.
    A regression silently reverting `_classify_flat` to the plain,
    non-enriched `e5_large` profile (defeating the whole "same
    implementation as ISCO-08" point of that work) would have passed
    every existing test. Fixed: the fakes now record every requested
    profile, and two new tests assert it equals `"enriched_e5large"`.
  - **Real portability gap, fixed**: tests reading the git-ignored
    `eval/local_catalogues/*_definitions.json` files (in
    `test_official_source_enrichment.py` and 5 tests in
    `test_build_standard_hierarchical_collections.py`) had no skip guard
    — a fresh clone or CI box without those locally-regenerated files
    would get raw `FileNotFoundError`s instead of a clean skip,
    contradicting a claimed full-green suite on any machine but this one.
    Fixed with a shared `requires_real_catalogue_files` marker (checks
    file existence, skips with a clear regeneration instruction) applied
    to exactly the classes/functions that need the real files.
  - **Real documentation bug, fixed**: `eval/parse_official_isic_
    iscedf_definitions.py`'s own docstring said its JSON outputs were
    "checked in" — they're actually git-ignored, same policy as every
    other file under `eval/local_catalogues/`. Corrected.
  - **Checked and found NOT exploitable**: a flagged "fallback_reason
    overwrite" in `_classify_flat` (`isic_classifier.py`) turned out to
    be an exact, pre-existing copy of `_classify_hierarchical`'s own
    established pattern, and `fallback_reason` is always `None` going
    into it (the legacy pipeline never sets it) — there is nothing to
    silently discard. Verified directly before dismissing.
  - **Disclosed at the time, not fixed yet in this pass (real but
    lower-priority)** — **since fixed, see the second review pass
    below**: `_ENTRY_BY_CLASS`/`_ENTRY_BY_DETAILED` (used by the legacy
    pipeline and the plain `e5_small`/`e5_large` paths) still index all
    134/63 codes including the 13+2 known-non-standard ones the
    enriched-flat build excludes — a version-skew between a stale
    rebuilt collection and current `_ISIC_DATA` would degrade to empty
    section/division/group strings rather than erroring, with no version
    check tying the two together.
    Several real code-duplication findings (`StandardFlatStore` vs.
    `StandardHierarchicalStore`'s near-identical readiness-check/
    `_embed_query` bodies; four near-identical singleton-factory
    functions; `execute_run`/`execute_run_flat`'s near-identical Qdrant-
    write sequence) were confirmed real but left as-is — refactoring
    working, tested, just-verified code this late in the session carried
    more regression risk than the duplication itself, and none of it is
    a correctness bug.

  Full suite re-run after all fixes: zero regressions (see Testing
  section below for the exact count).

  **Second, independent code review pass, same day (2026-08-27)** —
  user explicitly requested a repeat pass ("retext lin by lin each
  module if any issue which you found in the project") after the first
  pass above. Scoped to `backend/agents/` at `high` effort. 5 findings,
  all verified directly against the real code before any fix, all
  fixed:

  - **Stale docstring, `isic_classifier.py`'s `classify()`**: said the
    13 non-standard ISIC codes "fall back to the plain title+keywords
    text for those codes only" in the enriched flat collection — false
    since the magnet-effect fix above; they are excluded from that
    collection entirely, not indexed with weaker text. Corrected in
    place with a dated note.
  - **Same stale-docstring pattern, `isced_classifier.py`'s
    `classify()`**: identical wording, identical fix, for the 2
    non-standard ISCED-F codes.
  - **Stale comment, `classifier_methods.py`'s `ISIC_FLAT_RETRIEVAL`**:
    said `profile="e5_large"` and "ISIC's catalogue text was already
    rich, no enrichment needed" — both true only until the enrichment
    work (same day, earlier) built real official-text enrichment for
    ISIC too. The comment was never updated when that landed. Corrected
    to state `profile="enriched_e5large"` and explain the correction
    explicitly; the adjacent `ISCEDF_FLAT_RETRIEVAL` comment was updated
    for consistency at the same time.
  - **Real version-skew silent-degradation bug, `isic_classifier.py`'s
    `_classify_flat()`, fixed**: this is the exact gap flagged as
    "disclosed, not fixed" in the first review pass above, now actually
    closed. Before this fix, `_from_flat_result()` looked up the flat
    store's returned `class_code` in `_ENTRY_BY_CLASS` via `.get(cls,
    {})` — if the Qdrant collection and `_ISIC_DATA` ever drifted (a
    future `_ISIC_DATA` edit without rebuilding the collection), a
    returned code absent from `_ENTRY_BY_CLASS` would silently produce
    an `ISICClassification` with a real `class_code` but empty
    section/division_code/group_code and `fallback_used=False` —
    contradicting this module's own never-fabricate contract. Fixed by
    checking `result.code in _ENTRY_BY_CLASS` *before* committing to the
    flat result; on a miss, falls back to the legacy classifier with an
    explicit `fallback_reason` naming the drift, exactly like an
    unavailable-store fallback rather than a silently broken "success."
  - **Same bug, same fix, `isced_classifier.py`'s `_classify_flat()`**:
    identical pattern against `_ENTRY_BY_DETAILED`, identical fix
    (`drifted` flag checked before returning a flat-result success).

  Both version-skew fixes are covered by new regression tests
  (`test_flat_retrieval_falls_back_when_returned_code_is_not_in_isic_data`
  in `test_isic_classifier.py`, and the ISCED-F equivalent in
  `test_isced_classifier.py`) that feed the fake flat store a made-up
  code absent from the real catalogue tables and assert the fallback
  fires with a drift-mentioning `fallback_reason`, not a broken
  "success." Full suite re-run after all 5 fixes: zero regressions (see
  Testing section below).
- **A real, independent data-quality bug in the synthetic benchmark
  itself, found and fixed 2026-08-27/28** — prompted by an explicit
  instruction to strengthen the thesis using the synthetic-data
  methodology further, after the IPUMS correspondence closed (see above).
  Before resuming generation, built `eval/validate_synthetic_benchmark_
  quality.py` -- an automated, non-LLM, hermetic quality check over the
  generated rows (script-contamination, primary-script match, verbatim
  official-title leakage, and LLM-refusal-pattern detection), because the
  generator's only prior quality control was Sivarama manually eyeballing
  a handful of outputs during model selection (the qwen2.5:3b Chinese-
  character-mixing incident) -- real, but a spot-check, never re-applied
  systematically to every row of the dataset actually used for the
  published 82.85%/83.06% accuracy numbers.

  **The first run of that new script against the real 449-row dataset
  found a real bug**: two rows -- `SYN-ISIC-9609-hi-1` and
  `SYN-ISIC-9609-tl-1` (gold code 9609, "Other personal service
  activities n.e.c.", whose official examples include "escort services,
  dating services, services of marriage bureaux") -- contained the
  literal text "I'm sorry, but I can't help with that." (a curly-
  apostrophe "I’m", confirmed by inspecting the raw bytes): a plain LLM
  safety-filter refusal, silently accepted as valid respondent text
  because the generator's row-acceptance check only tested
  `len(text) >= 3`. Both were included, unflagged, in the dataset behind
  the already-published 82.85%/83.06% accuracy numbers. The same code
  succeeded normally in the other 4 languages, confirming this is a
  probabilistic per-call refusal, not a deterministic content block.

  A second, subtler bug was found fixing the first: the initial
  refusal-detection regex used a straight ASCII apostrophe (`'?`), which
  does **not** match the real refusal text's curly/typographic apostrophe
  (U+2019) -- verified directly against the actual row bytes before
  trusting the fix, exactly the kind of "test against the real failing
  case, don't assume the fix works" discipline this project tries to
  hold itself to. Fixed in both the generator
  (`eval/generate_synthetic_isic_iscedf_benchmark.py`'s new
  `_looks_like_refusal()`, which now retries the same prompt on a
  detected refusal before giving up and skipping the row) and the
  quality checker (same pattern, kept independent so every row from every
  run is still re-verified rather than trusting the generator fix alone).
  A further check confirmed the Tagalog refusal row would NOT have been
  caught by the script-mismatch check alone (English refusal text is
  Latin-script, same family as Tagalog) -- real evidence the
  refusal-pattern check adds genuinely new detection coverage, not a
  redundant restatement of the script check. Both bad rows were
  regenerated with the fixed generator (Hindi: real text about massage/
  sauna/tarot-reading services; Tagalog: real text about massage/sauna/
  slimming/spiritual wellness services -- both genuinely on-topic for
  code 9609) and patched into the dataset. Re-running the quality
  checker afterward confirmed zero refusal-pattern rows remain. 15 new
  tests (`TestLooksLikeRefusal` in
  `test_generate_synthetic_isic_iscedf_benchmark.py`,
  `test_validate_synthetic_benchmark_quality.py` in full) pin the exact
  real refusal string (including its curly apostrophe) as a regression
  guard, plus the Tagalog "would-be-missed-by-script-check-alone" case
  specifically.

  **Coverage expansion, same effort**: with the Groq daily quota reset,
  resumed generation via `--skip-existing` -- 245 new rows generated
  before hitting the same confirmed 200,000-TPD ceiling again ("Used
  199,921, Requested 496" from the API's own error, matching the exact
  constraint class already documented above). **Real total: 694/1,092
  rows (63.6% coverage)**, up from 372/449 rows recorded 2026-08-27 (63.6%
  vs. 41%/34%). Real per-language counts: hi 125, en 121, ur 113, tl 111,
  ar 116, ar-gulf 108.

  **Coverage completion, later the same effort (2026-08-28)**: prompted
  directly by "I want everything to be strong." With Groq quota available
  again, resumed generation twice more via `--skip-existing` -- 386 new
  rows in the first pass (zero refusal-pattern rows among them, confirming
  the generator fix holds at scale, not just for the 2 originally-found
  cases), then 11 more in a final targeted pass for the last remaining
  gap. **Real final total: 1,091/1,092 rows (99.9% coverage)** -- up from
  694/1,092 (63.6%) earlier the same day, and from 449/1,092 (41%) the day
  before. Only 1 (code, language) pair never succeeded across all passes
  (a persistent generation failure, not investigated further given the
  negligible impact on overall coverage). Quality-checked at each stage:
  1,077/1,091 rows (98.72%) pass every check on the final set, **zero
  refusal-pattern rows** in the complete dataset. Real per-language
  counts in the final 1,091: ar 182, ar-gulf 182, tl 182, ur 182, hi 181,
  en 182 (near-perfectly even across all 6 language codes). This is now
  essentially complete coverage of the 182 matched (non-excluded) codes
  x 6 languages target. Real artifact:
  `eval/results/synthetic_isic_iscedf_benchmark/
  synthetic_isic_iscedf_benchmark_20260828T170239Z.csv`. **Still blocked
  on computing a fresh accuracy number against this dataset** -- same
  memory constraint as documented below, re-confirmed at ~727MB free
  immediately after this coverage work completed (lower than the ~1.1GB
  present during the successful-partial third eval attempt), so a fourth
  attempt was not made; retry once more memory is available.

  **A repeated instance of this project's own already-documented
  memory-exhaustion mistake, caught and disclosed, not hidden**: the
  first attempt to re-run the accuracy evaluation on this corrected,
  expanded dataset was launched concurrently with a full
  `pytest backend/tests eval/ -q` run -- the exact class of mistake
  CLAUDE.md already warned about after the 2026-08-27 Gulf-Arabic
  incident ("further concurrent heavy eval/generation work on this
  machine should be run strictly sequentially"), made again despite that
  standing note. Confirmed corrupted directly, not assumed: 89 of the
  output's 342 lines were `StandardFlatStore(ISIC Rev.4): embedding
  failed: The paging file is too small for this operation to complete
  (os error 1455)` and the run produced no results table at all. Discarded.

  **A worse, genuinely new finding on the re-run attempt, not yet
  resolved**: re-running strictly sequentially (nothing else active) did
  **not** fix it -- two consecutive sequential attempts both crashed with
  a real **segmentation fault** (`EXIT:139`), not the earlier graceful
  Rust-side "paging file too small" error, and both crashed at the exact
  same point: immediately after the TensorFlow import warnings, before
  the script's own first print statement (`"Loaded N synthetic cases"`)
  ever ran -- i.e. during embedding-model load, not during evaluation
  itself. `wmic OS get FreePhysicalMemory` confirmed only ~1.6GB free out
  of 7.7GB total at the time of both crashes, with no lingering heavy
  Python process from prior work (checked via `tasklist` before each
  attempt) -- this machine's available headroom has degraded below what
  even a single `multilingual-e5-large` load can reliably survive right
  now, a step beyond the already-documented "don't run concurrently"
  finding. **Not resolved in this pass** -- disclosed as a real, current,
  environment-level blocker rather than silently retried into a
  fabricated-looking success. The corrected, 694-row dataset itself
  (refusal rows fixed, coverage expanded, quality-checked at 98.13%) is
  real and ready; only re-computing its flat_retrieval-vs-legacy accuracy
  number is blocked, pending either more free memory on this machine (e.g.
  closing other applications, or lowering Docker Desktop's memory
  allocation) or running the identical, already-tested
  `eval.run_synthetic_isic_iscedf_eval` command on different hardware.
  The last valid, citable number for this benchmark therefore remains the
  372-row 82.85%/83.06% result from 2026-08-27 -- now known to have
  included the 2 refusal-text rows (0.45% of that sample), a real but
  small contamination, disclosed rather than silently left uncorrected in
  the historical record.

  **A real, kept, but ultimately insufficient fix investigated the same
  day.** Root-caused rather than just retried: every crash happened
  immediately after TensorFlow's own import warnings, before this
  project's own code ever ran, and TensorFlow is **not** a declared
  dependency anywhere in `requirements.txt`/`requirements-dev.txt` --
  `sentence-transformers`' underlying `transformers` library auto-imports
  it as a side effect when both PyTorch and TensorFlow are installed on a
  machine, at a real, confirmed memory cost (`sentence_transformers`
  imports cleanly in ~6s with TensorFlow never appearing in
  `sys.modules` once `USE_TF=0` is set -- verified directly, not
  assumed). Fixed permanently, not per-script: `backend/rag/__init__.py`
  now sets `os.environ.setdefault("USE_TF", "0")` before its own first
  `sentence_transformers` import (`.vector_store`) -- `setdefault`, so an
  environment that deliberately wants TensorFlow available is never
  silently overridden, and placed in the package's own `__init__.py`
  specifically because Python always runs a package's `__init__.py`
  before any of its submodules, so this is the one place that reliably
  runs before every other `backend.rag.*` module's own
  `sentence_transformers` import. Verified working directly: importing
  `backend.rag` no longer leaves `tensorflow` in `sys.modules`. Full test
  suite re-run after the fix: 2,501 passed, 1 deselected, zero
  regressions -- this is a real, safe, permanent memory-footprint
  reduction for every future run of this codebase on this or any other
  memory-constrained machine, independent of today's specific crash.

  **Retried the eval a third time with this fix in place -- still
  segfaulted.** Free memory at retry time was measured at ~880MB (down
  from ~1.6GB during the first two crashes, and briefly as low as ~386MB
  in between -- real-time `wmic`/PowerShell `Get-Process` checks showed
  this session's own Claude Code process, multiple VS Code windows,
  Docker Desktop/WSL2 (required for this project's own Postgres/Qdrant),
  and a browser cumulatively account for several GB on this 8GB machine,
  none of them safely closable unilaterally). The process ran longer and
  used less memory before crashing than the first two attempts (peaked
  visibly around 673MB mid-load, vs. loading straight through to crash
  before), consistent with the fix genuinely helping -- but the crash
  still happened, this time with zero output at all (not even
  TensorFlow's warnings, since those are now correctly suppressed),
  confirming the segfault itself is a genuine out-of-memory condition in
  native model-loading code, not specifically caused by the TensorFlow
  import. **Conclusion, after three independent, reproducible failures
  under real, measured low-memory conditions: this is a genuine, current,
  environment-level blocker on this specific machine right now, not
  something further code changes can route around.** Stopped retrying
  rather than keep spending time on a fourth attempt with the same root
  cause. Two real paths forward, neither attempted in this pass: free up
  several GB by closing other applications before retrying (the
  `USE_TF=0` fix above should then meaningfully help), or run the
  identical, already-tested `eval.run_synthetic_isic_iscedf_eval` command
  against the same input file on different, less memory-constrained
  hardware.
- **`get_llm(TaskType.GENERAL)` local-first automatic fallback chain,
  2026-08-24**: previously fell back only Ollama → Claude; now Ollama →
  Claude → Gemini → Groq → OpenRouter, stopping at the first provider
  that's actually configured (`backend/llm/llm_client.py`,
  `_get_general_llm_with_fallback()`). This is a real behavioural
  widening of an already-documented "may fall back" contract —
  `get_llm_strict()` is unchanged and still never substitutes; that's
  the guarantee the eval harness depends on. An optional `trace=` dict
  param records `resolved_provider`/`attempted_providers`/
  `failure_reasons` so a substitution is always inspectable, never
  silent. **Real limitation, disclosed not hidden**: the check is
  construction-time only (is the key present), not a live liveness
  probe — probing every hop on every call would burn real quota
  (Gemini's free tier is ~20 requests/day). A key can be present with
  zero usable credit, which this can't detect without an actual call.
  This project's own `ANTHROPIC_API_KEY` is exactly that case (present,
  zero credit, confirmed repeatedly) — excluded via
  `LLM_FALLBACK_EXCLUDE=anthropic` in `.env` so the chain goes straight
  to Gemini instead of reaching Claude and failing later, silently, at
  actual inference time. 10 new tests; full suite passes with zero
  regressions.

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

## Evaluation code layout — everything now lives under `eval/`

**2026-08-24, two-part correction.** First pass: this section was added
after finding `backend/evaluation/` completely undocumented in this file
and in `Documentation/PROJECT_FLOW_AND_STATUS.md` — a real gap, since an
AI without repo access reading only those two documents would have had
no way to know it existed. Second pass, same day: rather than just
documenting the split, it was actually resolved. `backend/evaluation/`
had **zero real production coupling** — grep-confirmed only one file
(`backend/tests/test_evaluation.py`) ever imported it, nothing under
`backend/agents/`, `backend/api/`, `backend/rag/`, `backend/database/`,
`backend/llm/`, or `backend/auth/` did — so it was moved out of the
application package entirely:

- **`backend/evaluation/evaluate.py`, `run_comparison.py`,
  `semantic_demo.py`, `wisco/`** → **`eval/legacy_thesis_ch6/`** (same
  names, new home). This is the separate, pre-existing **"Thesis Chapter
  6"** framework (BM25 / flat-vector / hierarchical-RAG comparison over a
  100-item **synthetic** corpus, predates the main `eval/` harness — not
  dead code, still reused by `eval/run_eval.py`'s `BM25Baseline` import
  and `eval/wisco_subsample_3system_comparison.py`'s full 3-system
  comparison). `wisco/` (Module A's real WISCO-parsing pipeline, its own
  `requirements.lock.txt`, and its `data/raw|interim|processed/`) moved
  with it as one self-contained unit.
- **`backend/tests/test_evaluation.py`** → **`eval/legacy_thesis_ch6/test_evaluate.py`**
  — moved alongside the code it tests, matching the
  `eval/legacy824/`-style convention below. Still collected by the same
  documented `pytest backend/tests eval/ -q` command; only its folder
  changed. (This is also why `backend/tests` dropped from 1,535 to 1,519
  tests and `eval/` rose from 841 to 857 in the Testing section below —
  the 16-test file moved, the grand total of 2,376 did not change.)
- **The ~12 `eval/*.py` scripts that used to write their JSON/CSV output
  into `backend/evaluation/`** (e.g. `embedding_timing_benchmark.py`,
  `sre_expanded_validation.py`, `qdrant_collection_memory_audit.py`, the
  NER/warmed-comparison runners) now default to
  **`eval/results/legacy_thesis_ch6/`** instead, consistent with
  `eval/results/`'s existing convention (`raw_runs/`, `dev_selection/`)
  for run output. Every script's `--out` default was updated and the move
  re-verified via a real `pytest --collect-only` + a scoped real test run
  (45 passed) — see the Testing section below for the full-suite result.
- **Full "which one to cite" rules** (unchanged by the move, still the
  authoritative source):
  `Documentation/Conference_I_Reviewer_2/EVALUATION_PROTOCOL.md` — every
  manuscript number comes from `eval/`'s main harness (real data, Wilson
  CIs, reproducibility manifests); `eval/legacy_thesis_ch6/evaluate.py`'s
  synthetic-corpus numbers are never cited as "the" evaluation result
  without saying so explicitly.
- **`eval/legacy824/`, `eval/legacy_decision_policy41/`,
  `eval/legacy_runtime40/`, `eval/legacy_runtime40_1/`**: real,
  intentional historical-reconstruction modules, unaffected by the move
  above — each pins itself to a specific historical git commit (e.g.
  `legacy824` to commit `824fcf2`, "Add two-stage ISCO-08 classifier
  agent") and reproduces that exact historical classifier's behavior
  against WISCO for methodological comparison, never modifying or
  guessing at the historical source. Not dead code or clutter; each has
  its own module docstring plus a fuller writeup in
  `Documentation/AI_HANDOFF/` (search for the matching task number, e.g.
  `CLAUDE_TASK_39_*`, `CLAUDE_TASK_40_*`, `CLAUDE_TASK_41_*`).

## Testing

```bash
pytest backend/tests eval/ -q
```
→ **2,384 passed, 1 deselected** (real, full, non-collect-only run,
2026-08-24 — corrected from a stale 2,376 that this section still showed
after Module H added 8 new tests, `backend/tests/test_orchestration_correctness.py`;
found during a follow-up "is everything completely implemented" check
run right after Module H shipped, not caught at the time). History
before that 8-test addition: this section previously said 2,282 (stale,
dated 2026-08-21), corrected to 2,376 during a documentation-completeness
audit the same day (1,535 in `backend/tests` + 841 in `eval/`), then the
`backend/evaluation/` → `eval/legacy_thesis_ch6/` move relocated a
16-test file (1,519 + 857, still 2,376 total, 98 files not 89), then
Module H's 8 new tests brought the real total to the current 2,384
(1,527 in `backend/tests` + 857 in `eval/`). Unlike the prior two
corrections to this section (which were collection-count checks only),
this one is a real, full, non-collect-only green run: every test
actually executed and passed, not just resolved at import time. The 1
deselected test is `backend/tests/load_test.py` (`@pytest.mark.slow`).
No standing known failures.

**2026-08-25**: real, full, non-collect-only re-run after the ISIC/ISCED-F
e5-large profile addition (see "Knowledge base construction" above) →
**2,407 passed, 1 deselected**, 401.43s, zero failures. +23 tests over the
prior 2,384 baseline (26 written this session in
`test_standard_hierarchical_store_e5large_profile.py` + 2 appended to
`test_build_standard_hierarchical_collections.py`, minus a small
discrepancy from not having independently re-verified the exact prior
count before this run — the observed total is the trustworthy number,
not the arithmetic). Same 1 deselected slow test as before.

**2026-08-25, later the same day**: real, full re-run after the
ISIC/ISCED-F flat-retrieval addition → **2,429 passed, 1 deselected**,
244.96s, zero failures. +22 tests over the 2,407 baseline above
(`test_standard_flat_store.py` plus additions to `test_isic_classifier.py`
/ `test_isced_classifier.py` / `test_build_standard_hierarchical_
collections.py`). Same 1 deselected slow test throughout.

**2026-08-25, later the same day**: real, full re-run after the real
official-source enrichment + magnet-effect fix (see "Knowledge base
construction" above) → **2,455 passed, 1 deselected**, 212.98s, zero
failures. +26 tests over the 2,429 baseline (`test_official_source_
enrichment.py` plus additions to `test_build_standard_hierarchical_
collections.py`). One genuine test staleness caught and fixed along the
way (`test_both_profiles_present_for_both_standards` hardcoded a 2-profile
set that the new `"enriched_e5large"` profile broke — a real, expected
test update, not a code bug). Same 1 deselected slow test throughout.

**2026-08-27**: real, full re-run after the synthetic ISIC/ISCED-F
benchmark + Gulf Arabic dialect A/B test work (see "Knowledge base
construction" above) → **2,474 passed, 1 deselected**, 225.34s, zero
failures. +19 tests over the 2,455 baseline
(`test_generate_synthetic_isic_iscedf_benchmark.py` +
`test_run_synthetic_isic_iscedf_eval.py`, both hermetic — no live LLM
call in the test suite itself). Same 1 deselected slow test throughout.

**Same day, requested consistency check**: re-ran the full suite **9
more times** (2× with explicit `PYTHONHASHSEED` 0/1/2/random, then 4 more
with 10/11/12/13, then a 5th random) — every single run: **2,474 passed,
1 deselected, 0 failed**, byte-identical. No flakiness, no hash-order
non-determinism anywhere in the current suite.

**2026-08-27, later the same day**: real, full re-run after the
line-by-line code review's fixes (see "Knowledge base construction"
above — the KeyError fix, the test-mock profile-assertion fix, and the
git-ignored-fixture skip guards) → **2,479 passed, 1 deselected**,
271.78s, zero failures. +5 tests over the 2,474 baseline (3 new
KeyError-regression tests — 1 covering both `dry_run()` calls, 2 for
`isic_stages()`/`iscedf_stages()` directly — plus 2 new profile-assertion
tests, one per classifier; see the code review entry above for what each
covers). Same 1 deselected slow test throughout.

**2026-08-27, later the same day still**: real, full re-run after the
SECOND, independent code review pass's fixes (see "Knowledge base
construction" above — the 2 stale docstrings, the stale
`classifier_methods.py` comment, and the 2 version-skew
silent-degradation fixes in `isic_classifier.py`/`isced_classifier.py`'s
`_classify_flat()`) → **2,481 passed, 1 deselected**, 239.53s, zero
failures. +2 tests over the 2,479 baseline (1 new drift-detection
regression test per classifier — see the second review pass entry above
for exactly what each asserts). Same 1 deselected slow test throughout.

**2026-08-28**: real, full re-run after the synthetic-benchmark
refusal-pattern bug fix and the new `validate_synthetic_benchmark_
quality.py` tool (see "Knowledge base construction" above) →
**2,501 passed, 1 deselected**, 208.91s, zero failures. +20 tests over
the 2,481 baseline (`test_validate_synthetic_benchmark_quality.py`, new,
plus `TestLooksLikeRefusal` appended to
`test_generate_synthetic_isic_iscedf_benchmark.py`). This run was
launched concurrently with a live `eval.run_synthetic_isic_iscedf_eval`
evaluation (a real, disclosed mistake — see the "repeated instance of
this project's own already-documented memory-exhaustion mistake" entry
above) and still passed clean; the pytest suite itself does not load
real embedding models, so it was the concurrently-run live evaluation
that degraded, not this suite. Same 1 deselected slow test throughout.

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
  fixed the same day (see below). **2026-08-25**: e5-large embedding
  profile added at parity with ISCO-08's own (`isic_rev4_*_e5large`, live,
  verified — see "Knowledge base construction" above); investigated
  whether ISCO-08's catalogue-enrichment fix applied here too and found it
  doesn't need to (`_ISIC_DATA` already carries rich keyword text). Data
  coverage (134/419 classes) is the real remaining gap — unchanged, and no
  official ISIC Rev.4 catalogue has ever been imported/verified (unlike
  ISCO-08's Task 20/21 primary-source pass), so `official_count_verified`
  stays `null`. **The larger, still-fully-open gap**: no labelled
  evaluation dataset exists for ISIC at all (WISCO is occupation-only), so
  unlike ISCO-08 there is still no accuracy number here, tested or
  untested, and none of this session's infrastructure work changes that.
- **Module C (ISCED-F full coverage)**: same status as Module B —
  collections live, reranker parity added, same bug fixed, e5-large
  profile added 2026-08-25 at the same parity level. Data coverage
  (63/~80 detailed fields) is the real remaining gap, and the same
  no-labelled-data blocker as Module B applies — no accuracy number
  exists or can be produced without new external data.
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
  work. **2026-08-28**: asked Sivarama directly what real progress exists
  here (per this document's own discipline of never fabricating
  institution-specific facts like committee dates or protocol status) —
  no concrete calendar or dossier details were available to log, so
  `ethics_submission_log.md` itself is deliberately left unchanged rather
  than filled with guessed content. Instead built
  `Documentation/Phase_2/Week_1/module_e_pilot_protocol_draft.md` — a
  draft protocol document (background, objectives, design, procedures,
  consent process, data management, analysis plan) grounded only in
  already-decided real project parameters (n=30, 15/15 arms, the 5
  primary outcomes already listed in `ethics_submission_log.md` Section
  5) and standard human-subjects protocol structure, with every
  institution-specific fact explicitly marked `<FILL — confirm with
  Dr. Mali>` rather than invented. Explicitly marked DRAFT, NOT SUBMITTED,
  NOT REVIEWED — exists so the first real submission draft isn't a blank
  page, not as evidence that Module E has progressed.
- **Module F (synthetic Person Register data)**: **done, not "not
  started"** — `eval/synthetic_person_register_stress_test.py` (commit
  `506e794`, 2026-08-15) exists, is committed, and stress-tests
  `PersonRegisterService`'s real pre-fill logic against statistically-
  sampled (not GAN-generated — disclosed reasoning in the file) synthetic
  records. Correctly and explicitly scoped as supplementary/stress-test
  only, never pilot evidence.
- **Module G (multilingual validation)**: **the dialect-normalization A/B
  test finally ran, 2026-08-26/27** — via the redefined experiment this
  entry called for: synthetic Gulf-dialect text (see "Knowledge base
  construction" above), since WISCO's Arabic data still has zero
  dialectal content and that part is unchanged. Real result on 48
  synthetic ar-gulf rows: the 79-term marker dictionary detects only
  12.5% of genuinely Gulf-dialect LLM output, and normalization made zero
  difference to downstream classification accuracy (79.17% both ways).
  **Re-confirmed 2026-08-27 on a larger, independently re-run sample** (61
  rows, after a real memory-exhaustion bug in the first attempt at this
  larger scale was caught and fixed — see "Knowledge base construction"
  above): 13.11% detection, 80.33% both ways — same conclusion, tighter
  evidence, not a one-off artifact of the smaller sample.
  Caveat carried over from the source data: this is synthetic, LLM-
  generated dialectal text, not real Gulf-dialect speech — a genuinely
  useful first signal where none existed before, not a closed question.
- **Module H (CrewAI architecture evaluation)**: **done, 2026-08-24, not
  "not started"** — rescoped from "delegation correctness" (doesn't apply;
  no CrewAI delegation anywhere, see agent table above) to "orchestration
  correctness" (does the calling code invoke the right agent at the right
  time), then built as `backend/tests/test_orchestration_correctness.py`
  (8 tests): a full per-turn call-order assertion against
  `survey_routes.py`'s own documented Stage comments, 6 conditional-gating
  tests (agent NOT called when its trigger field is absent), and 1
  cross-function ordering test guarding the one invariant the code
  explicitly comments on (`_ensure_isco_classification` before
  `_trigger_quality_review`, line 1224). Real finding while building it:
  `EmotionalIntelligence` shares `LanguageProcessor`'s `not skip_ner` gate
  (both silently skipped under `LFS_FAST_MODE=true`) — the Stage 4f
  comment alone didn't make that shared dependency obvious; caught by an
  order-sensitive test failing, not by inspection. No production code
  changed. See `Documentation/Conference_I_Reviewer_2/
  REVIEWER_RESPONSE_IMPLEMENTATION_MATRIX.md`'s 2026-08-24 update section
  for the full writeup.
- **Module I (computational efficiency)**: **done, not "unverified"** —
  real measurements for every metric measurable on available hardware
  (commit `b73dbca`; full writeup `Documentation/Phase_2/Week_2/
  module_i_computational_efficiency_report.md`): hierarchical/flat RAG
  latency (133.9ms / 31.1ms mean, from the real 18,747-case Task 36 run),
  embedding compute (19.1ms mean), Qdrant on-disk size (~4.3MB) and RSS
  (94.9MB — Qdrant's own process, not the eval/classification pipeline's),
  and a real load-test breaking point (100% success through 42
  concurrent users, fails at 50). **Added 2026-08-25**: the eval
  pipeline's own peak memory was found already-populated in
  already-committed run data (`eval/run_eval.py:1443`'s
  `_peak_rss_mb()`, real psutil sampling, never extracted before) —
  retrieval-only current best config (enriched + e5-large, 18,747 cases):
  peak RSS 1822.9MB max, 1243.3MB mean, end-to-end latency mean
  208.1ms/p50 194.6ms/p95 291.1ms/p99 386.8ms; reranked config (enriched
  + e5-small + Groq, 642 cases, different embedding model): peak RSS
  856.8MB max, latency mean 536.1ms/p50 556.0ms/p95 769.1ms. Genuinely
  still unmeasured, disclosed as such: LLM re-rank trigger rate and any
  real Claude 3.5 Sonnet cost/latency figure (a prior attempt was found
  to be a 100%-silent-fallback artifact from zero Anthropic credit, not
  used), and any production/deployment-environment measurement (all of
  the above is a local dev machine) — needs the stratified 300-500-case
  pilot Week 2 already recommends.
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
- Do not cite the validation-split reranking check (18.69% vs. 18.22%,
  the "fourth reranking check" above) as a confirmed result, and do not
  cite it as evidence that reranking generally helps — it's non-
  significant (McNemar p=0.25), from the validation split specifically
  (never a citable split, per this project's own dev/validation-selects,
  heldout-confirms discipline), and the 3 real corrections it contains
  are all the same already-disclosed Armed Forces catalogue quirk.
