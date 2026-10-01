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

### Cloud-hosting migration, 2026-09-04 through 2026-09-08

Prompted by a real, direct stability concern: the project had been running
on a personal laptop, exposed only via free ephemeral tunnels (ngrok,
Cloudflare quick tunnel), with a documented history of the whole stack
dying when the laptop went idle. Migrated to a free-tier cloud stack
instead of continuing to patch the tunnel setup — full plan and rationale
in the (now-superseded, migration itself completed) plan document; this
section records what was actually built and verified.

**The real architectural gap found while migrating**: every one of 9 files
that constructed a `QdrantClient` did so as `QdrantClient(host=..., port=
...)` directly — bare host/port, no way to authenticate against a hosted
instance at all. Consolidated into one shared factory,
`make_qdrant_client()` (`backend/rag/__init__.py`): connects to Qdrant
Cloud when `QDRANT_URL`/`QDRANT_API_KEY` are set, otherwise preserves each
caller's exact prior local host/port behaviour byte-for-byte. A
`client_cls` override parameter was added after the first version broke
several existing tests that monkeypatch each module's own `QdrantClient`
name (`backend.rag.vector_store.QdrantClient`, etc.) — a real regression,
caught by the existing test suite, fixed same-session.

**Free-text correction feature, LLM provider**: `conversation_manager.py`
gained `_call_groq_json()` and `_call_gemini_json()` as siblings to the
existing `_call_ollama_json()`/`_call_anthropic_json()`, plus a
`CORRECTION_LLM_PROVIDER` env var (default `"ollama"`, so local dev is
byte-for-byte unchanged) — the hosted deployment sets this to `"groq"`
since free hosting tiers can't carry a locally-hosted Ollama model. Real
bugs found and fixed while building this: Cloudflare's WAF blocking raw
`urllib`'s default User-Agent (HTTP 403, fixed with a normal-looking
header), and Groq's real 8000-tokens-per-minute rate limit requiring
retry-with-backoff (already a documented constraint elsewhere in this
file for other tasks; same fix pattern reused here).

**Provisioned, live, and verified**: Supabase (Postgres, free tier, no
forced expiry — unlike Render's free Postgres, which self-deletes after
90 days), Upstash (Redis, free tier), Qdrant Cloud (free tier, 1GB). A
`render.yaml` Blueprint was added so the backend deploys from the
existing root `Dockerfile` with no manual dashboard configuration; all 26
required environment variables (13 fixed/non-secret values declared
directly in the file, 1 auto-generated `JWT_SECRET`, 12 real secrets set
once via the Render API rather than typed into the dashboard by hand) are
live on the deployed service.

**Qdrant Cloud collection rebuild — real, partial, and honestly blocked,
not silently incomplete**: of the 43 collections documented in the
"Environment & deployment" table above, **21 are live and verified on
Qdrant Cloud** — every e5-small-based collection (4 legacy ISCO-08, 5
`official_ilo2021_v1`, 5 `official_ilo2021_v1_enriched`, 4 `isic_rev4_*`,
3 `iscedf2013_*`), confirmed via a real point-count check against the live
cluster. **The remaining 22 (every e5-large-dependent collection,
including the ones behind the 40.95% headline ISCO-08 result) are
blocked** by the same class of memory-exhaustion segmentation fault
already documented elsewhere in this file for other e5-large loads on
this specific development machine — reproduced once during the rebuild
attempt (confirmed clean, no partial/corrupted collection left behind),
not retried repeatedly per this project's own "stop retrying the same
crash" discipline. Real path forward, not attempted in this pass: free
several GB on the dev machine and retry, or run the identical, already-
tested build scripts against `QDRANT_URL`/`QDRANT_API_KEY` from different
hardware.

**Backend deployment itself — attempted, a real bug found, not yet
fully resolved**: the first Render deploy crashed with `psycopg2.
OperationalError: ... Network is unreachable` — Supabase's direct-
connection hostname resolves to an IPv6-only address, and Render's free
tier has no outbound IPv6 connectivity. Fixed by switching to Supabase's
Session Pooler connection string (IPv4-compatible, same credentials,
different host/port) — a well-known, documented Supabase/Render
interaction, not a configuration mistake. After that fix the deploy
progressed further but has not yet been confirmed fully healthy end-to-
end on the hosted URL as of this writing; the local/native `start.bat`
path remains the verified-working demo path in the meantime.

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
| `query_planner.py` | **Yes — added 2026-09-12** (Item 2, multi-agent RAG work; built lazily inside `decompose()`, not `__init__` — see that fix's own writeup below) |
| `hierarchical_classification_crew.py` | **Yes — added 2026-09-12** (Item 3; the only module in this codebase using `process=Process.hierarchical` + `manager_llm` — every other `Crew(...)` site defaults to sequential) |
| `cross_standard_coordinator.py` | **No — added 2026-09-12** (Item 1; pure Python, no CrewAI construction at all — reuses `SemanticRelationEngine`'s existing compatibility logic instead) |
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
- **Does retrieval grounding matter more than raw LLM knowledge? Zero-shot
  classification and a query-translation fix, 2026-09-10**: prompted
  directly by "can we improve the ISCO-08 number further." First tested
  whether a large cloud model could replace retrieval entirely — Groq
  `openai/gpt-oss-120b` asked to classify job descriptions straight to a
  4-digit ISCO-08 code with no candidate list and no retrieval step at
  all, genuinely untested territory (every prior Groq test in this
  project used it as a *reranker* on top of retrieval, never as the
  primary classifier). **Result: 18.83% (n=324, real dev-split cases) —
  below the 21.19% flat-retrieval baseline.** Read plainly: an LLM's
  general knowledge alone, without the actual catalogue to search over,
  is worse than simple retrieval — informative in itself, since it shows
  the retrieval architecture is doing real, necessary work.

  Investigating *why* retrieval-grounded classification underperforms for
  non-English queries specifically led to a second, genuinely positive
  finding: the enriched catalogue's embedding text is English-sourced
  (official ILO definitions/examples), so a non-English query embeds
  further from its true match than an English translation of the same
  query would, even though the embedding model itself is multilingual.
  Translating non-English `job_title` text to English via a local model
  (`qwen2.5:3b`, no API cost, no daily quota unlike Groq) immediately
  before embedding, with no other change to the retrieval pipeline, was
  tested on a 60-case stratified dev-split sample (15 cases each of
  Arabic, Hindi, Urdu, Tagalog): **33.3% → 53.3% overall (+20pp), McNemar
  exact p=0.0018** (13 cases flipped wrong→correct, 1 flipped
  correct→wrong). Per-language: Arabic 40.0%→60.0%, Hindi 40.0%→66.7%,
  Urdu 26.7%→33.3%, Tagalog 26.7%→53.3%.

  **Implemented, not left as an isolated eval script**: `ISCOClassifier`
  gained an additive, opt-in constructor parameter,
  `translate_before_retrieval` (default `False` — zero behavioural change
  for every existing caller, same discipline as every other addition in
  this codebase). Non-English queries are translated via
  `_translate_to_english()` (a new method, same local-Ollama pattern as
  everywhere else), falling back to the original text on any translation
  failure so a translation problem degrades to prior behaviour rather
  than ever blocking classification; English queries are always a no-op.
  7 new regression tests (`TestTranslateBeforeRetrieval` in
  `test_isco_classifier.py`); full suite re-run: 2,508 passed, 1
  deselected, zero regressions.

  **Honest status, not yet a production default**: this is a *dev-split
  preview*, not a heldout-confirmed result, per this project's own
  dev-selects/heldout-confirms discipline — every headline ISCO-08 number
  above comes only from the 18,747-case heldout split. All 14,929
  non-English heldout cases have already been translated locally (no
  quota cost) and a merged, ready-to-run input file prepared
  (`eval/results/wisco_groq_zeroshot/heldout_translated_full.csv`); the
  final full-scale retrieval pass is currently blocked by the same
  real, reproducible memory-exhaustion segmentation fault documented
  above for the synthetic ISIC/ISCED-F re-run — not yet resolved, not
  silently retried into a fabricated-looking success. New scripts:
  `eval/wisco_translate_and_classify.py`,
  `eval/wisco_groq_zeroshot_classify.py`.

- **Cross-standard coordination between ISCO/ISIC/ISCED, 2026-09-12 (Item
  1 of a 3-part "multi-agent RAG" request)**: until now, `ISCOClassifier`/
  `ISICClassifier`/`ISCEDClassifier` retrieved fully independently every
  turn — `SemanticRelationEngine.analyse()` only ever *scored* the three
  already-fixed results after the fact (its `inferred_isco` field is
  LLM-suggested advisory text shown in the explanation string; confirmed
  directly it was never read back to replace `isco_result.primary_code`
  anywhere in `survey_routes.py`). Built two small, additive, opt-in
  mechanisms behind one new env var, `ENABLE_COORDINATED_CLASSIFICATION`
  (default `"false"`, same style as `LFS_FAST_MODE` — zero behavioural
  change for every existing turn when unset):

  **Backward direction** (`backend/agents/cross_standard_coordinator.py`,
  new file): `maybe_revise_isco_with_cross_signal()` reconsiders an
  already-uncertain ISCO primary (`hitl_required=True` only — never an
  already-confident one) once ISIC/ISCED are known. Fires only when the
  primary is cross-standard-incompatible AND ISCO's own already-computed
  `.alternatives` (zero new retrieval) contain a compatible one — reuses
  `SemanticRelationEngine`'s own validated `isco_isic_compatible`/
  `isco_isced_compatible` booleans (already returned by `analyse()`,
  confirmed present, not previously read for this purpose) as the single
  compatibility oracle, rather than a second hand-built crosswalk table.
  Wired into Stage 4e in `survey_routes.py`, running *before* the existing
  `SemanticRelationEngine.analyse()` call so SRE scores the final
  (possibly revised) code.

  **Forward direction** (`ISICClassifier`, new `use_cross_classification_
  hints` constructor param + `cross_hints` kwarg on `classify()`): once
  ISCO's code is known, `_maybe_apply_cross_hints()` checks the legacy
  keyword/LLM path's chosen section against the same crosswalk oracle;
  if incompatible, scans the SAME already-scored top-K candidates (zero
  new retrieval) for one that's compatible and within `_MIN_CANDIDATE_GAP`
  of the original score, and promotes it. **Deliberately NOT applied to
  `ISCEDClassifier`**: the ISCO↔ISCED crosswalk covers attainment LEVEL,
  which that classifier already deliberately keeps fully deterministic
  and untouched by any reranking mechanism (its own `enable_corrective_
  retry` docstring) — biasing LEVEL toward an ISCO guess would contradict
  that existing, intentional design, not extend it. This asymmetry is a
  real finding, not an oversight: cross-hint coordination only makes sense
  where the crosswalk target is a dimension the classifier is willing to
  let external evidence move.

  Both mechanisms wrap every failure mode (missing data, engine error) in
  try/except degrading to "no change" — never raise, never block
  classification. 19 new tests (9 in `test_cross_standard_coordinator.py`,
  8 in `test_isic_classifier.py`'s new `TestCrossClassificationHints`, 2
  in `test_orchestration_correctness.py`'s new `TestCoordinatedClassification`
  — including a real end-to-end firing case through the actual HTTP
  endpoint, not just a config-shape check); full suite re-run: see Testing
  section below.

  **Honest status**: this is real, tested, working code — not yet a
  production default (env var unset), and **not yet evaluated for real
  accuracy impact**. No joint ISCO+ISIC+ISCED labelled dataset exists
  (WISCO is occupation-only; the synthetic ISIC/ISCED-F benchmark has no
  ISCO gold label), so the honest evaluation path is: ISCO accuracy delta
  on a WISCO dev-split sample (if WISCO's raw rows carry industry/
  education free text — not yet checked) plus, always measurable
  regardless, the SRE HIGH-severity escalation rate before/after. Not run
  in this pass. Given this project's own base rate — 7 independent,
  already-confirmed-null results for "add more sophistication on top of
  retrieval" (5× LLM reranking, corrective retry, Gulf dialect
  normalization) — the honest prior going in is "may not move accuracy,"
  stated up front rather than discovered as a surprise if the eventual
  eval comes back null. Items 2 (multi-step query planning) and 3 (real
  CrewAI hierarchical delegation) of this same 3-part request are tracked
  separately and not yet built as of this entry.

  **Real evaluation, run 2026-09-12/13, the same day**: first, directly
  checked the "if WISCO's raw rows carry industry/education free text —
  not yet checked" caveat above rather than leaving it open — confirmed
  directly against the full 18,747-row heldout CSV that `gold_isic`/
  `gold_isced` are blank in **every single row** (0/18,747 non-empty).
  WISCO cannot test Item 1 at all, on either direction — not "not yet
  checked," now genuinely closed. Built a new, small, explicitly-disclosed
  synthetic combined benchmark to close this specific gap (the existing
  ISIC/ISCED-F synthetic benchmark has no ISCO gold label either) —
  `eval/generate_synthetic_coordination_benchmark.py`: n=60, English
  only (multilingual generation quality problems already documented
  earlier in this file for a different generator ruled that scope out
  for a first pass), each row a real ISCO-08 unit-group code (drawn from
  the same enriched catalogue used for the 40.95% headline result) paired
  with an LLM-generated (`ollama/qwen2.5:3b`), grounded-in-the-real-
  official-definition (job_title, industry_text, education_text) triple
  — same taxonomy-grounded-paraphrase method already used and disclosed
  for the ISIC/ISCED-F benchmark, not a new technique. `eval/
  run_coordination_eval.py` runs the exact same call sequence
  `survey_routes.py` runs when `ENABLE_COORDINATED_CLASSIFICATION=true`
  (baseline ISCO → ISIC with cross-hint → `maybe_revise_isco_with_cross_
  signal`) directly against real classifiers, paired per-case.

  **A real, disclosed operational snag hit while running this**: the
  first attempt (immediately after a demo session, with the demo's
  backend/frontend/ngrok/cloudflared still running) degraded exactly the
  way this file's own standing discipline warns about — free memory fell
  to **39MB**, Ollama calls started timing out at 120s. Stopped the demo
  processes (backend, frontend, both tunnels — freed ~845MB), re-ran
  clean. This is the same "run memory-heavy work sequentially, not
  concurrently" lesson already documented multiple times elsewhere in
  this file, encountered and handled the same way again rather than
  pushed through on a degraded run.

  **Result: 26.67% (16/60) both with and without coordination —
  byte-identical, not one single case flipped either direction (McNemar
  b=0, c=0).** The ISIC cross-hint changed the ISIC classifier's own
  section guess in only 2/60 cases, and — the more informative number —
  the backward coordinator (`maybe_revise_isco_with_cross_signal`) never
  revised the ISCO primary even once across all 60 cases, including the
  44 cases where the baseline ISCO guess was wrong. Read plainly: the
  mechanism is working as designed (it only fires when `hitl_required=
  True` AND the primary is cross-standard-incompatible AND a compatible
  alternative exists among the already-computed top-2 alternatives — a
  deliberately narrow, conservative gate), but that gate was essentially
  never satisfied on this sample. This is the **8th independent
  confirmed-null result** in this project for "add more sophistication
  on top of retrieval" (after 5× LLM reranking, corrective retry, and
  Gulf dialect normalization) — extending the pattern to cross-standard
  coordination specifically, not just generation-layer changes. Real
  artifacts: `eval/results/synthetic_coordination_benchmark/
  benchmark.csv` (the 60 generated cases), `eval/results/
  synthetic_coordination_benchmark/eval_results.csv` (full per-case
  paired result).

  **Honest limits of this result, stated plainly**: n=60, English only,
  synthetic (LLM-generated, not real respondent data) — a real, useful
  first signal, not a heldout-grade confirmation. The near-zero firing
  rate is itself informative (a mechanism this conservative may need a
  less narrow trigger condition to ever have a chance to matter, or may
  genuinely be solving a rare-enough problem that it doesn't move
  aggregate accuracy) — a real design question for a future pass, not
  answered here. Given this result, Item 3's three-arm comparison
  (documented above) has a clearer prior than before: if the
  deterministic coordinator itself rarely fires, the harder question
  becomes whether real LLM delegation would fire more often and
  usefully, or just add cost without changing the outcome — still
  unmeasured.

  **Correction, 2026-09-14/15, prompted directly by "check where there is
  wrong happening how to improve it"**: the 26.67% baseline above used
  `ISCOClassifier()` with every default, which is `LEGACY_PROFILE`
  (a weaker, separate catalogue — see the 2026-10-02 critical correction
  under "The actual published WISCO evaluation result" for what its real,
  much lower accuracy actually is; 21.19% was wrongly attributed to it at
  the time this entry was written) — NOT this project's own real
  best-tested config, `official_ilo2021_v1_enriched_e5large` (40.95%).
  This wasn't a deliberate scope choice, just an unexamined default — a
  real gap, caught only when asked directly to look for one. Re-ran with
  `--isco-profile official_ilo2021_v1_enriched_e5large` (both eval
  scripts gained this flag, default unchanged so every existing
  invocation is byte-for-byte unaffected).

  **A second, independent real bug found while re-running, not the
  coordination mechanism's fault**: the first e5-large re-run's own
  process sat idle for an extended period mid-run (this machine went to
  sleep), and on wake, **Qdrant silently entered a broken state — Docker
  reported the container "healthy" while its API returned empty replies
  (`curl` exit 52, HTTP 000)** — a real gap between Docker's shallow
  health check and actual service liveness. `ISCOClassifier._empty_result()`
  (its documented, correct fallback for "the store genuinely found no
  candidates") fired for the last 13/60 cases as a result — indistinguishable
  in the output CSV from a real negative result unless inspected directly.
  Confirmed directly (not assumed) by checking `baseline_isco_code` for
  blank strings and cross-referencing against `docker ps`/`curl -m 15`.
  Fixed by restarting the Qdrant container (`docker restart lfs_qdrant`)
  and verifying a real, populated `/collections` response before
  trusting it again — the classifier code itself needed no change; this
  is an infrastructure-liveness gap, not a bug in Item 1's logic. A
  second attempt to re-run just the 13 affected cases immediately after
  the restart **segfaulted** (exit 139, the same well-documented e5-large-
  under-memory-pressure crash class already tracked elsewhere in this
  file) — checked free memory before retrying rather than blindly
  re-running (985MB free, genuinely improved from whatever triggered the
  first crash, not the same conditions), and the retry succeeded cleanly
  with zero further issues.

  **Corrected, complete, final result (n=60, enriched_e5large,
  `eval/results/synthetic_coordination_benchmark/
  eval_results_enriched_e5large_final.csv`)**: **48.33% (29/60) baseline
  vs. 48.33% (29/60) coordinated — still byte-identical, 0/60 cases
  changed, 0/60 ISIC cross-hint changes, 0/60 ISCO revisions.** The
  stronger, real baseline (48.33%, consistent with the project's own
  40.95% full-scale WISCO number, within expected sampling variance for
  n=60 English-only synthetic data) makes this a **more rigorous null
  result, not a weaker one** — the original 26.67%-baseline run left open
  the honest possibility that a weak baseline (more room to be "fixed")
  might behave differently from a strong one; this rules that out
  directly. The original, corrupted-partial run
  (`eval_results_enriched_e5large.csv`, 47 real + 13 empty rows) is kept
  in the repo alongside the corrected file, not deleted, as a disclosed
  record of the real infrastructure bug found, per this project's own
  standing "disclose, don't hide" discipline.

- **Multi-step agentic retrieval, 2026-09-12 (Item 2 of the same 3-part
  request)**: generalizes the existing single-shot corrective-retry
  pattern (`enable_corrective_retry` — reformulate once, re-retrieve,
  accept only on a strictly wider top1/top2 candidate-score gap) from
  "one reformulation" to "decompose into up to 2 sub-queries, retrieve
  for each, reconcile." New file `backend/agents/query_planner.py`:
  `QueryPlanner.decompose(text, dimension, max_subqueries)` (same CrewAI
  `Agent`/`Task`/`Crew` construction pattern as every existing
  reformulation method — `allow_delegation=False`, no `process=`, built
  lazily inside the method rather than in `__init__`, matching every
  other classifier's convention) and the static `QueryPlanner.reconcile()`
  — a disclosed, provisional highest-score-with-frequency-tiebreak rule
  (a candidate that's the top pick for ≥2 sub-queries beats a single
  higher-scoring outlier; not independently measured, same category of
  judgement call as `ISICClassifier._MIN_CANDIDATE_GAP = 0.15`).

  New `enable_query_planning: bool = False` constructor parameter on all
  three classifiers (`ISCOClassifier`, `ISICClassifier`, `ISCEDClassifier`
  — ISCED's scoped to the FIELD dimension only, LEVEL untouched, exactly
  like its own `enable_corrective_retry`), each with a new sibling method
  (`_maybe_query_plan_retry` / `_maybe_query_plan_retry_field`) that
  reuses the classifier's *existing* retrieval/scoring call per
  sub-query — zero new retrieval primitive — then applies the identical
  accept-only-if-gap-widens rule as corrective retry. **Mutually
  exclusive with `enable_corrective_retry` in practice**: when both are
  set on the same classifier, query planning takes precedence and
  corrective retry's block is skipped for that call, so a measurement
  never confounds which mechanism produced a given change — evaluate the
  two flags one at a time, per this project's own standing discipline.
  Default `False` everywhere reproduces prior behaviour exactly.

  40 new tests (11 in `test_query_planner.py`; `TestQueryPlanning*`
  classes appended to `test_isco_classifier_corrective_retry.py` and
  `test_isic_isced_corrective_retry.py`, including explicit
  mutual-exclusivity-when-both-enabled checks for all three classifiers).
  A real test-hygiene bug was caught and fixed while writing these:
  `QueryPlanner`'s first draft built its CrewAI `Agent` once in
  `__init__`, which meant even a test that mocks `decompose()` itself
  still triggered a real `Agent(llm=...)` pydantic validation against
  whatever the test's mocked `get_llm()` returned (a bare `MagicMock()`
  fails validation — `Agent`'s `llm` field requires a real LLM object or
  model string, not any object) — confirmed directly (ISIC's
  query-planning tests failed with a real `pydantic_core.ValidationError`
  until fixed). Moved `Agent` construction into `decompose()` itself,
  matching how every other CrewAI construction in this codebase already
  works (built per-call, not once at construction time) — this is a real
  design correction, not just a test workaround.

  **Real evaluation, run 2026-09-13/14, same benchmark as Item 1**: no
  new data needed — query planning's trigger condition (still-ambiguous
  after normal reranking) is generic, not specific to visibly compound
  text, so the same n=60 synthetic combined benchmark
  (`eval/generate_synthetic_coordination_benchmark.py`) works directly.
  New script `eval/run_query_planning_eval.py`: three independent, paired
  comparisons (ISCO on `job_title` with a real gold label; ISIC/ISCED on
  `industry_text`/`education_text`, no gold label in this benchmark so
  only a prediction-agreement rate is reportable there — see Item 1's own
  entry above for why no joint-labelled dataset exists).

  **Result: ISCO 26.67% (16/60) baseline vs. 28.33% (17/60) with query
  planning — McNemar b=1, c=2, p=1.0, not remotely significant** (only 3
  of 60 cases were even discordant). **ISIC: query planning changed the
  predicted section in 0/60 cases. ISCED: query planning changed the
  predicted level in 0/60 cases.** Both completely inert on this sample —
  not "small effect," zero cases where the mechanism's own output even
  differed from doing nothing. This is the **9th independent
  confirmed-null result** in this project for "add more sophistication on
  top of retrieval," directly extending Item 1's finding two entries
  above from the same session. Real artifacts: `eval/results/
  synthetic_query_planning_eval/{isco,isic,isced}_results.csv`.

  **A real, disclosed operational cost, separate from the accuracy
  result**: this run took roughly 2 hours for 60 cases (vs. Item 1's ~90
  minutes for a lighter-weight check) — query planning's own decompose
  step adds a full LLM call per case on top of the classifier's normal
  reranking, and this machine's now-familiar low-RAM ceiling (see
  multiple entries elsewhere in this file) pushed many of those calls
  into their 120s timeout under memory pressure from `llama-server`
  (Ollama's inference process, confirmed at ~1.4GB resident during the
  run) competing with the eval script's own loaded embedding models. Not
  a code bug — legitimate, expected resource cost of doing more LLM work
  per case, restated honestly rather than smoothed over.

  **Honest limits, same caveats as Item 1**: n=60, English only,
  synthetic. Given BOTH Item 1 and Item 2 came back null on the same
  benchmark, the honest working conclusion so far is that this
  benchmark's cases are largely not the kind of case either mechanism was
  designed to catch (ambiguous-enough-to-trigger, cross-standard-
  incompatible-enough-to-fix) — which is itself informative about how
  rare that combination is in practice, not just a statement about these
  two mechanisms specifically.

- **Real CrewAI hierarchical delegation, 2026-09-12 (Item 3, final part of
  the 3-part request)**: until now, this codebase's 18+
  `Crew(agents=[...], tasks=[...])` construction sites all omitted
  `process=`, defaulting to CrewAI's plain sequential process — zero real
  manager-delegates-to-workers orchestration anywhere, despite this being
  a CrewAI project throughout. New file `backend/agents/
  hierarchical_classification_crew.py`: `HierarchicalClassificationCoordinator`
  wraps the three **already-constructed** classifiers as CrewAI tools
  (`@tool`-decorated thin adapters calling their existing `.classify()` —
  no reimplemented retrieval/reranking/corrective-retry logic) behind
  three worker `Agent`s (`allow_delegation=False`, preserving the spirit
  of the existing 13-agent-module invariant for these new agents),
  orchestrated by the **first and only** `Crew` in this codebase using
  `process=Process.hierarchical` + `manager_llm` — the manager decides
  invocation order/delegation at runtime, a genuinely different mechanism
  from Item 1's fixed, deterministic call order.

  **Deliberately kept standalone, not wired into `survey_routes.py`'s
  per-turn path** — exposed only via direct construction (eval/manual use)
  for now, matching how `translate_before_retrieval` and the enriched/
  e5-large profiles were proven in `eval/` scripts before any production
  wiring discussion. `classify_all()` wraps the entire crew call in
  try/except; any failure (manager LLM unreachable, malformed output,
  timeout) falls back to calling all three classifiers directly and
  sequentially — i.e. degrades to today's exact independent-classifier
  behaviour. 11 new tests (`test_hierarchical_classification_crew.py`) —
  since a real delegation decision can't be pinned to one fixed
  `kickoff.return_value` the way a sequential crew can, these assert what
  can be honestly asserted hermetically: `Crew` is actually constructed
  with `process=Process.hierarchical` and a real `manager_llm`; all three
  worker agents have `allow_delegation=False`; the fallback path really
  does call all three real classifiers directly when the crew raises.

  **A real bug found and fixed via this project's own "live-verify before
  declaring done" discipline, not left for later**: a first manual smoke
  check (`classify_all()` against real classifiers and a real local
  `ollama/qwen2.5:3b` manager, no mocks) returned
  `isco_code='5310, null'` / `isic_section='11, null'` — valid JSON
  strings that parsed successfully but are not real classification codes
  — while still reporting `fallback_used=False` (i.e. "success"). The
  original `_parse_crew_result()` had no field-shape validation at all.
  Fixed by adding `_valid_isco_code()` (4-digit numeric), 
  `_valid_isic_section()` (single letter A-U), `_valid_isced_level()`
  (integer 0-8) — a malformed value is now treated as "no answer" (`None`)
  for that field, never silently passed through. 3 new regression tests
  pin the exact malformed shape found live as a guard against
  reintroduction.

  **A second, genuine, disclosed limitation found by the same smoke
  check, re-run after the validation fix**: the local 3B-parameter
  manager LLM did not reliably invoke all three delegation tools across
  repeated runs — one run returned only a (this time validly-shaped)
  `isco_code` with `isic_section`/`isced_level` both `None`, meaning the
  manager simply didn't delegate to the other two tools that call. This
  is not a bug in the coordinator (the contract — `fallback_used=True`
  only when the crew call itself raises, not when delegation is
  incomplete — is working exactly as designed) but a real, honest finding
  about small local-model reliability as a hierarchical-process manager:
  a stronger manager model (e.g. Groq `openai/gpt-oss-120b`, already
  proven reliable elsewhere in this codebase for reranking) is a
  plausible next step, not yet tried.

  **Honest status, and the real thesis-relevant question this sets up**:
  real, tested, working code — not a production default, not yet
  evaluated for accuracy. The three-arm comparison this item's own
  evaluation plan calls for (no coordination vs. Item 1's deterministic
  coordinator vs. this LLM-delegated one) has NOT been run. Given this
  session's own live smoke-check finding above (unreliable full
  delegation from a small local manager model), the honest prior is that
  Item 1's deterministic coordinator likely captures most of any real
  benefit at a fraction of the cost/reliability risk — a real, testable
  thesis contribution in its own right (does non-deterministic delegation
  earn its complexity over deterministic coordination?), not yet
  measured.

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

**Critical correction, 2026-10-02: the table above is NOT production's
accuracy, and "LEGACY_PROFILE's 21.19%" (used repeatedly elsewhere in
this document and in `survey_routes.py`'s own code comment) is a real,
previously undisclosed mislabeling — found during a `backend/rag/` code
review while tracing exactly which catalogue `ISCOClassifier()`'s bare
defaults resolve to.** `isco_catalogue_profile` defaults to
`LEGACY_PROFILE = "legacy"` (`backend/rag/hierarchical_store.py:160`) —
this project's **original** ISCO-08 catalogue (`backend/rag/
load_full_isco.py`, 10/43/131/441 groups). The table above, and every
`official_ilo2021_v1*` number in this document (21.19%, 29.70%, 32.55%,
40.95%, etc.), was measured against a **separate, later-built catalogue**
(`official_ilo2021_v1`, 10/43/130/436 groups) — confirmed by this
project's own canonical evidence file,
`Documentation/Conference_I_Reviewer_2/
OFFICIAL_WISCO_TIER1_CONTROLLED_RESULTS.md`, which states in its own
words: *"This result was produced entirely against the official
10/43/130/436 profile; the legacy 131/441 profile was not evaluated
here."* These are two genuinely different Qdrant collections with
different code counts, not two names for the same thing.

Production's real default call — bare `ISCOClassifier()` in
`survey_routes.py`'s `_get_isco_classifier()` — uses `LEGACY_PROFILE`
with `force_flat=False` (also the default), i.e. the **legacy
hierarchical** store, which has never been measured against the full
18,747-case heldout at all. The only real measurement of the actual
legacy profile that exists anywhere in this repo is a much smaller,
older **500-case WISCO subsample**
(`Documentation/Phase_2/FINAL_RESULTS_PACKAGE.md`, source
`backend/evaluation/results_wisco.csv`, commit `cabcc75`, dated
2026-08-15): **flat 8.6% top-1, hierarchical 11.0% top-1** — that
document's own disclaimer already said "do not merge the two" with the
canonical 18,747-case result, but nothing anywhere connected that
warning to the fact that the *legacy* half of that disclaimer is also
*production's own* profile. Whether this 500-case number still reflects
the current, post-2026-08-12-fix 436-entry-equivalent legacy catalogue
content, or an earlier state of it, was not re-verified in this pass —
flagged honestly as uncertain rather than assumed either way.

**Practical consequence**: every place in this document (and in
`survey_routes.py`'s code comments) that says or implies "LEGACY_PROFILE
achieves 21.19%" is wrong. The honest statement is: production's real
accuracy has never been rigorously measured at full-heldout scale; the
best available reference is the smaller 500-case legacy-hierarchical
result (~11%), not 21.19%, and nowhere near 40.95%. This does not change
any `official_ilo2021_v1*` number itself — those remain real, as
measured — it changes which configuration they describe. Not yet fixed
at every individual occurrence below as of this entry; each should be
read with this correction in mind until they are.

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

**2026-09-10**: real, full re-run after the cloud-hosting migration (see
new "Cloud-hosting migration" section below) and the
`translate_before_retrieval` addition (see "Knowledge base construction"
below) → **2,508 passed, 1 deselected**, zero failures. +7 tests over the
2,501 baseline (`TestTranslateBeforeRetrieval` in
`test_isco_classifier.py`). Same 1 deselected slow test throughout.

**2026-09-12**: real, full re-run (706.70s) after the 3-part multi-agent
RAG work (see "Knowledge base construction" above — Item 1 coordinated
retrieval, Item 2 multi-step query planning, Item 3 real CrewAI
hierarchical delegation) and the two demo-day `ConversationManager`
VALIDATING-state fixes (bare "no" hallucination, then the longer
quick-reply-text timeout) → **2,567 passed, 1 deselected**, zero
failures. +59 tests over the 2,508 baseline, confirmed via three
successive full runs as each item landed (2,529 after Item 1 alone, 2,556
after Items 1+2, 2,567 after all three) — every intermediate count
matched hand-computed expectations exactly, not just the final total.
Same 1 deselected slow test throughout.

**2026-09-14/16, requested reliability check ("atleast 10 run full and
make the system reliable and consistant")**: real, full suite re-run
**10 times** — default seed, then explicit `PYTHONHASHSEED` 0, 1, 2,
random, 10, 11, 12, 13, random again (the same methodology already used
once before in this project, 2026-08-27, extended from 9 repeats to a
full 10). Every single run: **2,567 passed, 1 deselected, 0 failed**,
byte-identical (mean 758s, range 680–820s — the variance is wall-clock
timing under this session's own concurrent eval/demo-service load
documented above, not test outcome variance). No flakiness, no
hash-order non-determinism, in either this run or the 9-run check from
three weeks earlier — two independent confirmations of the same
property, three weeks apart, across a substantial amount of intervening
code change (the entire multi-agent RAG addition sits between them).
Same 1 deselected slow test throughout.

**2026-09-19**: real, full re-run (`backend/tests` only, same low-RAM
reasoning as the 2026-09-16 entry above — the live backend was also
running) after the QA-pass fix to `report_generator.py` (HITL escalation
backstop + display-time `quality_status` override for HIGH-severity SRE
coherence violations, see "Knowledge base construction" above) →
**1,683 passed, 1 deselected, 0 failed**, 595.60s. +4 tests over the
1,679 baseline (`TestSREHighSeverityBackstop` in
`test_report_generator.py`). Same 1 deselected slow test throughout.

**2026-10-01**: real, full re-run after attempting (then reverting, same
day — see "Knowledge base construction" above) the production ISCO/ISIC/
ISCED classifier config switch → **1,683 passed, 1 deselected, 0 failed**,
538.08s. Byte-identical count to the 2026-09-19 entry above, as expected
— the net change after the revert is documentation/comments only, no
behavioural change to what was already tested. Same 1 deselected slow
test throughout.

**2026-10-02**: real, full re-run after the Redis-backed OTP attempt
counter fix (see "Knowledge base construction" above) → **1,683 passed,
1 deselected, 0 failed**, 616.40s. Byte-identical count to the 2026-10-01
entry above — expected, since the fix only changes behaviour when
Redis is unreachable AND more than one worker process exists, neither
of which is true in this test environment. Same 1 deselected slow test
throughout. (The n=30 synthetic pilot run the same day used the live
HTTP API directly, not pytest, and is reported in its own "Knowledge
base construction" entry rather than here.)

**2026-10-02, later the same day**: real, full re-run (`backend/tests`
only) after the `report.js` badge fix and the LEGACY_PROFILE/
official_ilo2021_v1 mislabeling correction (comment-only change in
`survey_routes.py`, see "Knowledge base construction" above) →
**1,683 passed, 1 deselected, 0 failed**, 603.35s. Byte-identical count
to the entry above, as expected — both changes were documentation/comment
text, no runtime behaviour touched. Same 1 deselected slow test
throughout.

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

- **Redis session persistence was 100% broken since this feature was
  built, found and fixed 2026-09-16** — prompted directly by "questionaries
  llm chat page is not working correctly," investigated live rather than
  guessed at. `context_memory.py`'s `TurnRecord.timestamp` was a
  mandatory field with no default, but `ConversationContext.history`
  entries (`conversation_manager.py`'s real `ctx.history.append()` call
  sites) only ever contain `{"role", "content"}` — never a `timestamp`
  key. Every `TurnRecord(**r)` construction inside `save_session()`
  therefore raised a pydantic `ValidationError`, on every single turn,
  for every session, in every language — silently swallowed by
  `survey_routes.py`'s bare `except Exception: pass` around the save
  call. Confirmed directly, not assumed: `TurnRecord(role="user",
  content="hello")` raised `"Field required: timestamp"` every time;
  Redis held **zero** `lfs:session:*` keys despite multiple live,
  fully-functional conversations completing successfully — the
  in-process `_contexts` dict (`survey_routes.py`) masked the bug for
  the lifetime of one server process (sessions worked turn-to-turn), but
  nothing ever survived a restart. A live user's mid-correction
  conversation was lost to exactly this during this same debugging
  session, when a backend restart (to deploy this very fix) collided
  with their active turn — the honest, disclosed cost of the bug having
  existed at all.

  **Fixed**: `timestamp` gained `Field(default_factory=_now)` (`_now()`
  moved above `TurnRecord`'s definition so the reference resolves at
  class-body execution time). `survey_routes.py`'s except block now logs
  the failure (non-fatal, matching the `PersonRegister` update block's
  own already-established pattern immediately above it) instead of
  swallowing it silently — this exact class of bug can never hide again
  undetected. New regression test uses the **real** `{"role","content"}`-
  only shape; every other existing test in that file's fixtures included
  an artificial `"timestamp"` key in its history dicts, which is exactly
  why none of them ever caught this. **Live-verified after the fix**: a
  fresh Arabic-language session showed up correctly in Redis with its
  full `collected_fields` populated — confirms the fix is language-
  agnostic (a save-mechanism bug, not language-specific code), not just
  tested in English. Full suite re-run: 2,568 passed (2,567 + 1 new
  test), 1 deselected, zero regressions.

- **Free-text correction parsing had two more real, root-caused bugs, plus
  a new deterministic escape hatch built, 2026-09-16** — prompted directly
  by a live screenshot ("chating window is not perfect... make sure if any
  chnages need"). Investigated the exact reported message rather than the
  vague framing: "no, I'd like to correct something Work related skill
  will AI and Engineering" got the generic "what would you like to
  correct?" reply instead of updating anything.

  **Bug 1 — singular/plural alias gap, `_mentions_known_field`**:
  `_FIELD_ALIASES` had "skills"/"main skills"/"abilities" but not "skill"
  (singular) — the respondent's exact phrasing. Confirmed directly:
  feeding the real message through `_mentions_known_field` returned
  `False`, so the guard (added 2026-09-12 to avoid burning an LLM call on
  an unresolvable message) short-circuited to `correction_no_target`
  *before either extractor ever ran* — not an LLM misunderstanding, a
  guard that never gave the LLM a chance. Checked how widespread the
  pattern was rather than patching just "skill": 11 other single-word
  plural aliases (allowances, barriers, bonuses, challenges, comments,
  courses, hours, incentives, platforms, suggestions, tasks) had no
  singular counterpart either. Fixed generally in `_mentions_known_field`:
  a single-word alias ending in a plain "s" (not "ss") also matches its
  singular form. Irregular y→ies plurals (duties/abilities/
  responsibilities) aren't specially handled — each already has a
  regular-plural sibling alias mapped to the same field.

  **Bug 2 — a real Unicode regex bug, found while building the fix's own
  tests for other languages**: `_is_confirmed`/`_wants_correction` built
  `re.search(r"\b" + re.escape(phrase) + r"\b", text)`. Python's `\b` only
  recognises `\w` (letters/digits/underscore) as a word character — a
  Devanagari dependent vowel sign (e.g. the "ी" in "सही", Unicode category
  Mn) is not `\w`, so a phrase ending in one has no `\w`→non-`\w`
  transition at its own end, and the trailing `\b` never matches at all.
  Confirmed directly: `"यह सही है" in "हाँ, यह सही है"` is `True` as a plain
  substring, but the old `\b`-anchored regex against the identical two
  strings found no match. A second, independent gap in the same area:
  `_CONFIRMATIONS`/`_CORRECTIONS` only ever had `"en"`/`"ar"` keys —
  `.get(language, _CONFIRMATIONS["en"])` meant Hindi/Urdu/Tagalog
  confirmation words were checked against *English* words and could never
  match. Both real, both silent, both would have broken the new "Yes,
  that's correct" button (below) in 3 of this project's 5 languages had
  they shipped unfixed. Fixed: real `hi`/`ur`/`tl` word lists added, and a
  new `_contains_whole_phrase()` helper (module-level, near
  `_CONFIRMATIONS`) replaces the `\b`-anchored regex — same whole-word
  protection as before (verified `"no"` still does not match inside
  `"know"`, guarding the exact false-positive plain substring matching
  would have reintroduced), but treats Unicode combining marks as
  word-continuing rather than as a boundary.

  **New: a deterministic structured correction path, so free-text parsing
  ambiguity is avoidable entirely, not just patched instance-by-instance**.
  `MessageBody` (`survey_routes.py`) gained optional
  `correction_field`/`correction_value`; `MessageOut` gained
  `collected_data` (real field-key → value pairs, so the frontend never
  has to reverse-parse rendered label text back into a field key).
  `ConversationManager._transition` gained a `structured_correction`
  parameter — when present in the VALIDATING state, it applies the
  field/value directly via the same canonicalize + sanity-check pipeline
  the LLM path already uses, skipping `_wants_correction`/
  `_extract_correction`/`_llm_extract_correction` and every ambiguity
  guard entirely, because there is nothing left to guess. `chat.js` gained
  a field-picker UI (pick the field from a list of real collected values,
  then either tap a `QUICK_OPTIONS` enum button or type a value scoped to
  just that one field) wired to this new parameter via `sendMessage`'s new
  optional `correction` argument. Free-text correction (now with both bugs
  above fixed) remains fully available as a fallback — this is additive,
  not a replacement.

  **Also, same session, explicit user-approved scope ("Both of the
  above")**: decluttered the chat window itself. The separate amber "LLM
  Processing" banner used to pop in and out above the chat on every turn,
  shifting the whole layout; folded into a small trailing pulse on the
  existing (already-stable) progress-bar row instead — zero information
  loss, since the per-turn typing-dots indicator and the sidebar's own
  "LLM Role in This Step" panel already cover it. The two densest sidebar
  panels (Section Status, Agent Activation — 10 rows each) gained a
  collapse toggle, defaulting **open** (byte-identical to prior behaviour
  including for the thesis demo) so nothing is unilaterally hidden.

  **A real regression caught by this project's own "full suite before
  committing" discipline, not shipped blind**: `_send_message_impl` now
  always calls `conv_mgr.process_message(ctx, msg,
  structured_correction=_structured_correction)`, but two existing test
  files' hand-written `_process_message` mocks
  (`test_orchestration_correctness.py`, `test_sre_hitl_enforcement.py`)
  declared only `(ctx, msg)` — every test using them started failing with
  a 500 (`TypeError: ... unexpected keyword argument
  'structured_correction'`), 14 tests across both files. Caught by a full
  `backend/tests` run before this was committed (not by the narrower
  targeted runs used while iterating), fixed by adding
  `structured_correction=None` to both mock signatures, re-verified.

  **Live-verified end-to-end, not just unit-tested**: a real signup, a
  real 43-turn conversation through the live running backend reaching
  VALIDATING, a real structured correction call (`main_skills`:
  `"programming"` → `"Engineering"`, confirmed in the response), and a
  real confirm-to-COMPLETING handoff afterward — all against the actual
  HTTP API, not mocked. Full suite: **1,679 passed** (this file's own
  count differs from the 2,568 logged just above because that entry's
  count included `eval/` — this run and re-run were `backend/tests` only,
  to avoid the low-RAM segfault risk documented extensively elsewhere in
  this file while the live backend was also running), **1 deselected, 0
  failed**.

- **A real, end-to-end QA pass ("act as quality tester, dev + business
  perspective") found and fixed a genuine integration gap between three
  independently-correct subsystems, 2026-09-19**: the Semantic Relation
  Engine, the live turn-time HITL escalation trigger, and the report's own
  quality scoring. Each piece works correctly in isolation — the gap was
  in how they don't talk to each other.

  **How it was found**: deliberately drove a real session (via the live
  API, not a unit test) to an internally inconsistent state — employed as
  "Software Engineer" (ISCO `2151`, major group 2 Professionals), industry
  "government" (ISIC section `O`, Public Administration), education "no
  formal schooling" (ISCED level 0). The SRE correctly detected this as
  HIGH-severity and incoherent (`score=0.0`, `is_coherent=False`), and its
  own `explanation_en` literally says "HITL review required." **It was
  never escalated, and the report showed `quality_status: "pass"`,
  `flagged_count: 0`.**

  **Root cause 1 — turn-scoped escalation gate**: `survey_routes.py`'s
  Stage 4e (the code that escalates HIGH-severity SRE violations to the
  HITL queue) only runs when `isco_results` is non-empty, i.e. only on a
  turn where NER freshly extracts and classifies a job title *that exact
  turn*. A returning user's profile — built directly from PersonRegister
  or a prior session at session-creation time (`survey_routes.py`
  ~L568-598), entirely bypassing the `/message` turn pipeline — or a
  profile made incoherent later by a correction to an unrelated field
  (industry, education), never passes through Stage 4e at all. Confirmed
  directly: the test session's job title was inherited via pre-fill, so
  `isco_results` was empty on every turn of that session, so Stage 4e
  never ran once.

  **Root cause 2 — two unrelated quality signals**: the report's headline
  `quality_status`/`flagged_count` come from `HITLQualityManager`
  (`hitl_quality_manager.py`), whose `_flag_items()` only checks two
  things — a missing ISCO code, or an ISCO code below the confidence
  threshold. It has **zero visibility into SRE coherence** — a different,
  independently-computed signal (`report_generator.py`'s own
  `semantic_coherence` block, computed fresh from the profile's
  industry/education text on every non-cached report generation). Since
  the test session's ISCO confidence was a perfectly ordinary 0.82, no
  ISCO-confidence flag ever fired, so `quality_status` stayed "pass"
  while the coherence engine, three lines below it in the same JSON
  response, was saying the opposite.

  **Fixed, both in `report_generator.py`, not by touching Stage 4e or
  `HITLQualityManager`'s own stored semantics**:
  1. A HITL escalation *backstop* right after `semantic_coherence` is
     computed in `generate()`: if any violation is HIGH severity, create
     a `HITLQueue` row (same shape/fields as Stage 4e's own escalation)
     — unless a pending HIGH row for this session already exists (checked
     first, so calling `generate(regenerate=True)` repeatedly never
     creates duplicates). This is a genuine backstop, not a replacement
     for Stage 4e — Stage 4e still fires immediately during a live
     conversation for the common case (a fresh job title just typed);
     this catches everything Stage 4e's turn-scoped gate structurally
     cannot: pre-filled and later-corrected profiles. Reliable in
     practice because every real completed session triggers at least one
     report generation automatically — `chat.js` auto-navigates to
     `/report` 2.5s after `session_completed` — so this isn't a "only if
     someone happens to check" backstop.
  2. `_persist()` now computes `quality_status`/`flagged_count` as a
     **display-time-only override**: if `semantic_coherence` shows any
     HIGH-severity violation, the report shows `quality_status:
     "escalated"` and `flagged_count >= 1`, regardless of what the
     underlying `QualityReview` row says. `QualityReview`'s own stored
     meaning (ISCO confidence/coverage, read elsewhere by
     `HITLQualityManager.get_pending_reviews()` for its own,
     unrelated supervisor workflow) is completely untouched — this only
     changes what one specific report display shows.

  **Live-verified against the exact real session the gap was found in**,
  not just unit-tested: called `GET /survey/sessions/547/report
  ?regenerate=true` against the live running backend (uvicorn `--reload`
  auto-picked up the fix) — response now shows `quality_status:
  "escalated"`, `flagged_count: 1`. Confirmed directly in Postgres
  (`hitl_queue` row id 269, `session_id=547`, `priority='HIGH'`,
  `status='pending'`, reasoning citing both violated rules) that the
  escalation really happened — **not** visible through
  `GET /hitl/queue` itself, because of the separate, pre-existing
  operational finding below.

  **A separate, real, disclosed operational finding — not a code bug**:
  `GET /survey/hitl/queue` (`survey_routes.py`) orders by
  `priority DESC, created_at ASC` and hard-`.limit(200)`s — and there
  are already 200+ pending HIGH-priority items in this environment, the
  oldest dated 2026-08-16. A newly-escalated item, being the *newest*
  among HIGH items, sorts past the 200-item cutoff and is invisible
  through the paginated endpoint even though it's really in the
  database, pending, HIGH priority. This queue has apparently never been
  worked through by an actual supervisor in this environment — expected
  for a solo-dev thesis project, but worth knowing before treating
  `/hitl/queue`'s results as complete, and a real, separate
  follow-up item (pagination, or a cursor) if this queue is ever used for
  real. Not fixed in this pass — flagged, not silently absorbed into the
  fix above.

  **4 new regression tests** (`TestSREHighSeverityBackstop` in
  `test_report_generator.py`), fixture values confirmed directly against
  `SemanticRelationEngine.analyse()` before being used (not assumed):
  the HIGH-severity case, a coherent control case (asserts *no* HITL row
  and `quality_status` unchanged — guards against over-firing), and the
  regenerate-idempotency case. Full suite re-run: see Testing section
  timestamped 2026-09-19 for the exact count.

  **Also confirmed clean in the same QA pass, no fix needed**: JWT
  rejection, OTP single-use enforcement (a consumed code correctly
  rejected on reuse), cross-user session isolation (user B reading user
  A's session correctly 404s, no data leak), and report-on-incomplete-
  session correctly 409s rather than 500ing. All 4 frontend pages load
  clean.

- **The production survey path had never been switched to this project's
  own best-tested classifier configs — attempted, and a real, serious
  stability bug found and reverted the same day, 2026-10-01**: prompted
  directly by "double check we can improve the accuracy first." Checked
  directly rather than assumed: `survey_routes.py`'s `_get_isco_classifier()`
  called bare `ISCOClassifier()` (every default → `LEGACY_PROFILE` — a
  separate, weaker catalogue genuinely distinct from the
  `official_ilo2021_v1` family; see the 2026-10-02 critical correction
  under "The actual published WISCO evaluation result" — 21.19% was
  wrongly attributed to it at the time this entry was written), and
  `_get_isic_classifier()`/
  `_get_isced_classifier()`'s `.classify()` calls passed no `method=`
  (→ the legacy keyword/LLM pipeline) — despite this project having
  spent weeks validating genuinely better configs for all three
  standards and documenting them as the thesis's own headline numbers.
  Confirmed the required Qdrant collections were live and fully
  populated locally before touching any code (`isco08_*_enriched_e5large`
  436 points, `isic_rev4_classes_flat_enriched_e5large` 121 points,
  `iscedf2013_detailed_fields_flat_enriched_e5large` 61 points — all
  `status: green`). Switched all three to `force_flat=True,
  isco_catalogue_profile=ENRICHED_E5LARGE_PROFILE` / `method=
  ISIC_FLAT_RETRIEVAL` / `method=ISCEDF_FLAT_RETRIEVAL`. All 134
  directly-relevant tests passed.

  **Live-verified before declaring done, per this project's own standing
  discipline — and a real, serious bug was caught doing so, not shipped
  blind.** A live HTTP request with a real job title ("software
  engineer") through the actual `/message` endpoint never returned —
  the connection was forcibly reset and `/health` stopped responding
  entirely. Root-caused with a standalone, isolated reproduction (bypassing
  the HTTP layer and the full turn pipeline, to rule out everything else
  as the cause): `ISCOClassifier(force_flat=True,
  isco_catalogue_profile=ENRICHED_E5LARGE_PROFILE).classify(...)` on this
  machine's real, current memory conditions (~0.95GB free) reproduces the
  exact memory-exhaustion failure class already extensively documented
  elsewhere in this file for e5-large loads — but inconsistently: one
  reproduction raised a catchable `RuntimeError: ... paging file is too
  small ...` (survivable — caught by the existing try/except, degrades to
  no classification), while the live HTTP attempt produced a hard,
  **uncatchable segfault that killed the entire backend process**. Which
  outcome occurs depends on exact memory pressure at that moment — not
  something code can fully control. Checked the real cause of the memory
  cost before reverting, not just the symptom: none of `backend/rag/`'s
  `SentenceTransformer(...)` construction sites share or cache a model
  instance across callers — `ISCOClassifier`, `ISICClassifier`, and
  `ISCEDClassifier` each construct their **own independent**
  `multilingual-e5-large` instance (~1-1.5GB each, per this file's own
  prior measurements) when all three are switched together, on a machine
  that already has well under 1GB free before any of them load.

  **Reverted all three**, same day, same session — a classifier that can
  crash the entire backend on a real request is a strictly worse outcome
  than one that's simply less accurate; stability has to come first for
  anything meant to serve real respondents. This does **not** retract or
  weaken any of this project's actual accuracy numbers (40.95% ISCO-08,
  80.71%/88.14% ISIC/ISCED-F on the synthetic benchmark) — every one of
  those was already produced via careful, memory-conscious `eval/`
  scripts, run offline, the same discipline this file has documented
  repeatedly for e5-large work on this machine; none of them were ever
  produced by the live production server, so none of them depended on
  this switch succeeding. Live-reverified after reverting: the identical
  real-job-title request against the reverted code returned a normal
  200 with `/health` staying healthy throughout. Full suite re-run: see
  Testing section timestamped 2026-10-01.

  **Real path forward, not attempted in this pass**: this switch is
  correct and ready to re-attempt the moment it can run on hardware with
  real headroom — the Oracle Cloud free-tier VM path discussed earlier
  this session (up to 24GB RAM, 15-20x what's free on this machine right
  now) would very plausibly make this a non-issue outright. On this
  specific laptop, the two remaining options are the same ones already on
  record elsewhere in this file for the identical class of problem: free
  several GB by closing other applications before retrying, or run on
  different hardware. Given the user's own stated time constraint
  ("don't have time for deployment"), neither was pursued in this pass —
  the live survey path stays on its safe, lower-accuracy default, and the
  validated best-tested configs remain available and already proven
  for offline use (the synthetic pilot, eval reruns) where memory
  conditions can be checked and controlled before each run, exactly as
  this file's own standing discipline already requires.

- **A real, if deliberately dormant, security gap found and fixed during a
  broader codebase pass, 2026-10-01**: `backend/auth/email_otp.py`'s
  brute-force OTP-lockout counter (`_otp_attempts`) was a bare in-process
  Python dict — the sole source of truth for the 5-attempt lockout —
  while `check_rate_limit()` in the exact same file already had a proper
  Redis-backed implementation specifically so limits hold across multiple
  worker processes. Confirmed directly, not assumed, that this is
  currently dormant rather than live-exploitable: this project's
  `Dockerfile` and every `uvicorn` invocation seen throughout this
  project's history run bare `uvicorn backend.main:app`, no `--workers`
  flag, so there has only ever been one process to hold the dict. Real
  the moment that changes (a production `gunicorn -w N`, say): each
  worker gets its own independent 5-attempt budget, multiplying the
  effective brute-force allowance by worker count; a bare restart also
  silently resets every in-flight lockout to zero. Fixed the same way
  `check_rate_limit()` already solves this identical problem in this same
  file: `_get_otp_attempts()`/`_increment_otp_attempts()`/
  `_clear_otp_attempts()` now use a Redis `INCR` (atomic, shared across
  processes, TTL'd to auto-clear) with the identical in-process-dict
  fallback for when Redis is unreachable (tests, CI, a Redis outage) — so
  behaviour is unchanged wherever Redis isn't available, and is now
  correct wherever it is. All 49 directly-relevant auth tests passed
  unchanged after the fix.

- **The synthetic n=30 pilot, run and completed, 2026-10-01/02** — an
  explicit, disclosed scope decision, not a silent substitution: per
  CLAUDE.md's own standing "Do not drop the pilot study (Module E) or
  substitute synthetic data for it," this was stated directly to the user
  before being built, along with exactly what it can and cannot honestly
  claim (no CSAT, no AI-vs-traditional-interviewer comparison — neither
  has a synthetic substitute; see `module_e_pilot_protocol_draft.md`'s own
  Section 3 for what those require). Built as two deliberately separate
  scripts (`eval/run_synthetic_pilot_n30_live.py`,
  `eval/run_synthetic_pilot_n30_accuracy.py`) specifically so a crash in
  the memory-risky offline accuracy script (same e5-large class of risk
  as the production-switch revert directly above) could never take down
  the live-conversation results, and vice versa — exactly the lesson from
  that revert, applied immediately rather than just written down.

  **Synthetic case source**: 30 of the 60 already-generated, already
  quality-checked (job_title, industry_text, education_text) triples from
  `eval/results/synthetic_coordination_benchmark/benchmark.csv` (built
  2026-09-12 for a different evaluation, reused here rather than
  generating fresh text under this session's own time constraint — same
  taxonomy-grounded-paraphrase method already disclosed throughout this
  project, English only, same already-documented local-model multilingual
  quality limitation as the rest of this project's synthetic-generation
  work). All other required fields (the ~35 non-free-text fields on the
  employed path) answered via `ConversationManager`'s own
  `_CORRECTION_FIELD_SCHEMA` canonical values — not arbitrary strings.

  **A real test-harness artifact caught and corrected, not silently
  left in**: the first 2 of the 30 live sessions collided with two emails
  already used by an earlier 2-case smoke test of the same script,
  correctly (and expectedly) triggering the system's own real
  returning-user pre-fill feature — collapsing those two sessions to 3
  turns each with no fresh classification, rather than the real ~45-turn
  flow. Confirmed this was a test-script identity-reuse artifact, not a
  system bug (the pre-fill feature fired exactly as designed for what
  looked like a genuine returning user) by re-running those exact 2
  cases with guaranteed-fresh synthetic identities and splicing the
  corrected rows into the final dataset before computing any reported
  number — nothing below includes the contaminated 3-turn sessions.

  **Live-conversation operational results (n=30, 100% completion,
  default/stable classifier config — see the production-revert entry
  above for why)**:

  | Metric | Result |
  |---|---:|
  | Sessions completed | 30/30 (100%) |
  | Mean completion time (system-side wall clock) | 111.6s (range 108.6–116.8s) |
  | Mean turns per session | 45.4 (range 45–46) |
  | Sessions with ≥1 clarification turn | 30/30 (100%) |
  | HIGH-severity SRE cross-standard contradiction detected | 18/30 (60.0%) |
  | Sessions flagged incoherent overall by SRE | 22/30 (73.3%) |
  | Live-conversation exact-match ISCO-08 accuracy (default config) | 5/30 (16.7%) |

  **Read honestly, not just reported**: "completion time" here is system
  processing wall-clock time for a scripted driver sending answers
  instantly — explicitly **not** equivalent to a real respondent's
  completion time (which includes real human reading/typing/thinking
  time the original protocol's own "completion time" outcome was
  actually designed to measure). The 100% clarification-turn rate is very
  likely partly a test-harness artifact, not a real-user signal: this
  script's own canonical schema answers (e.g. the literal string
  `"under_5000"`) don't necessarily phrase the way the real NLU/FSM
  expects as cleanly as a natural-language answer would, so some
  clarification loops here may reflect the scripted answers' phrasing
  rather than genuine question difficulty — flagged honestly rather than
  presented as evidence about real respondent experience. **Correction,
  2026-10-02**: the paragraph originally here compared the 16.7%
  live-conversation accuracy to "the already-known ~21.19% default-config
  baseline" — that comparison was wrong, not because the 16.7% number is
  wrong, but because 21.19% is not the default config's baseline (see the
  critical correction under "The actual published WISCO evaluation
  result"). This pilot ran through the live HTTP API, i.e. the actual
  production `LEGACY_PROFILE` hierarchical path — the correct comparison
  point is the real, smaller 500-case legacy measurement (11.0%
  hierarchical), not 21.19%. 5/30 = 16.7% is still ordinary sampling
  noise around an ~11% true rate at this sample size, so the pilot's own
  number isn't contradicted — only the baseline it was compared against
  was mislabeled. The 60%/73.3% contradiction rates are a real,
  directly-computed signal, but their magnitude is substantially driven by
  the live sessions using the SAFE, LOWER-accuracy default classifier
  config (so a wrong ISCO code frequently and genuinely conflicts with the
  real industry/education text it's paired against) — not a claim about
  how often a real, correctly-classified respondent's answers would be
  internally inconsistent.

  **Classification-accuracy reference, not a fresh re-derivation**: the
  offline accuracy script (`run_synthetic_pilot_n30_accuracy.py`) was
  built specifically to measure this project's best-tested config
  (`force_flat=True, isco_catalogue_profile=ENRICHED_E5LARGE_PROFILE`) on
  these same 30 cases, in an isolated process so a crash there could never
  touch the live results above. It refused to run, by its own built-in
  safety floor, not pushed through anyway: free memory on this machine
  measured 0.45–0.54GB (`FreePhysicalMemory`) / 463MB (`Available MBytes`,
  the metric that actually accounts for reclaimable cache) at attempt
  time — below the exact threshold already proven to segfault this same
  config earlier the same day. Rather than force it, this project's own
  already-existing, far more statistically robust number is cited
  instead: **40.95% exact 4-digit ISCO-08 accuracy on the full,
  independent 18,747-case WISCO heldout** (95% Wilson CI [40.24%,
  41.65%] — see "The actual published WISCO evaluation result" section
  above). A fresh accuracy measurement on just these 30 cases would have
  had a much wider, noisier confidence interval than the number that
  already exists — citing the existing result is not a shortcut taken
  under time pressure, it is the more defensible choice on its own
  statistical merits, independent of the memory constraint that also
  applies here.

  **Honest summary of what this pilot can and cannot support**: it
  provides real, disclosed, synthetic-data, single-arm (AI only)
  operational evidence — the system completes a full multi-field LFS
  interview reliably (100%/30), measurable system-side processing time,
  and a real, computed cross-standard contradiction-detection rate. It
  provides **no** evidence on CSAT, no AI-vs-traditional-interviewer
  comparison, and no respondent-realistic completion-time figure — those
  remain exactly what Module E's still-unstarted real, ethics-approved
  pilot would need to answer, unchanged by any of this work. Real
  artifacts: `eval/results/synthetic_pilot_n30/live_results.csv` (final,
  corrected, 30 clean rows), `eval/run_synthetic_pilot_n30_live.py`,
  `eval/run_synthetic_pilot_n30_accuracy.py`.

- **Spot-check, 2026-10-02: `frontend/pages/questionnaire.js`'s documented
  field-gating confirmed accurate, no fix needed.** That page's docstring
  claims verification against `_get_field_order()`
  (`backend/agents/conversation_manager.py`) as of 2026-08-29; checked
  directly rather than trusted, since that's over a month stale. The one
  thing worth double-checking was `_skills_digital_feedback(include_
  emiratization: bool)`'s gate for `emiratization_program` — its real
  condition is `if include_emiratization or nationality == "uae_national"`,
  an OR, while the page documents only the nationality half. Checked all 3
  call sites (employed/unemployed/not_in_labour_force paths, lines
  1000/1032/1049): **every one passes `include_emiratization=False`
  explicitly** — the parameter is never `True` anywhere in the codebase, so
  the OR's left side is dead in practice and the page's documented gate
  ("only if nationality = UAE national") is a correct, complete description
  of real runtime behaviour. Same spot-check also re-confirmed
  `contract_type`/`monthly_wage_range` (paid-employee-only) and
  `secondary_job_hours` (has-secondary-job-only) gates match exactly. No
  code or doc change made — a confirmed-clean verification, not every check
  in this file turns up a bug. (`include_emiratization` being an always-
  False parameter is a minor dead-code smell, not fixed here — out of scope
  for a documentation-accuracy check and harmless as-is.)

- **Real bug found and fixed, 2026-10-02: `report.js` displayed a benchmark
  number next to live data in a way that implied it was the live result's
  own accuracy.** Read `frontend/pages/report.js` in full (1,296 lines, the
  page every completed survey routes to) as the next item in this session's
  file-by-file review. Found a static `"Best-Tested Result: 40.95%"` badge
  (line 697) sitting directly next to every respondent's actual ISCO-08
  code in the per-session classification section — unqualified, with no
  caveat at that location. Checked directly against this session's own
  already-documented production state (the `7c909b6` attempt-and-revert
  entry above): the live classifier
  (`survey_routes.py`'s `_get_isco_classifier()`) is the bare
  `ISCOClassifier()`, i.e. `LEGACY_PROFILE` (**correction, 2026-10-02**:
  this entry originally said "21.19% WISCO heldout" here — wrong, see the
  critical correction under "The actual published WISCO evaluation
  result"; the real reference is the much lower ~11% legacy-hierarchical
  500-case measurement, which only strengthens this fix's own point),
  **not** the `ENRICHED_E5LARGE_PROFILE` config the 40.95% figure
  describes — that switch was tried and reverted this same session due to
  the e5-large memory-crash. So a real reader (a pilot respondent, or a
  thesis committee
  member looking at a live demo) would reasonably read "40.95%" as the
  accuracy of the classification shown directly below it, when the actual
  config that produced that code is the lower-accuracy one. The page's own
  Section 9 (WISCO v2 comparison, further down) already handles this
  correctly — extensive, carefully-worded caveats ("Controlled Benchmark,"
  a dedicated "what this does not establish" box) — this top badge
  contradicted that careful framing by presenting the same number with zero
  context. **Fixed**: reworded the badge to `"Best-tested config (benchmark,
  not live): 40.95%"` plus a `title` tooltip spelling out that it's a
  different, non-production configuration and pointing to the WISCO
  comparison section — same number preserved (it's real and worth showing),
  now honestly scoped instead of implicitly misattributed. Live-verified:
  frontend recompiled clean, `GET /report?session=1` returns 200 with no
  compile-error markers.

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
