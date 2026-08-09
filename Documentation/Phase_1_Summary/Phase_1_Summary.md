# Phase 1 Summary — Multilingual LFS Conversational AI

**M.Tech Thesis | IIIT Kottayam 2026 | Supervisor: Dr. Goutam Mali**
**Prepared: 2026-08-02**

---

## 2026-08-10 Evidence Addendum

**Everything below this addendum is the original 2026-08-02 point-in-time
snapshot, preserved unchanged as historical record. It is not current.**
This addendum does not rewrite that snapshot — it adds what has changed
since, with pointers to the current authoritative sources.

- **Canonical WISCO result**: a real, controlled ISCO-08 accuracy
  measurement now exists — see
  `Documentation/Conference_I_Reviewer_2/OFFICIAL_WISCO_TIER1_CONTROLLED_RESULTS.md`
  (Tasks 36/37.1, 2026-08-10) for the full, exact, citable record. In
  brief: on the WISCO v2 controlled multilingual ISCO-08 benchmark
  (18,747-case heldout split, official ILO 2021 ISCO-08 catalogue
  profile, no LLM reranking), flat retrieval reached 21.1927% exact
  4-digit accuracy (3,973/18,747) versus strict hierarchical retrieval's
  10.3537% (1,941/18,747; McNemar exact two-sided p ≈ 1.8574e-301).
  **This is a controlled WISCO v2 benchmark result, not real Labour
  Force Survey validation**, and does not resolve the thesis's real-LFS
  validation gap.
- **Catalogue count distinction**: §4/§5 below describe this project's
  **historical legacy** ISCO-08 implementation snapshot as of
  2026-08-02 — "441 unit groups" and "131 minor groups" across "10
  major / 43 submajor / 131 minor / 441 unit" collections
  (`backend/rag/load_full_isco.py`). The WISCO result above uses a
  **separate, later-built official ILO 2021 ISCO-08 catalogue profile**
  with **verified** counts **10 major / 43 sub-major / 130 minor / 436
  unit groups** (`eval/verified_catalogue_counts.yaml`,
  `backend/rag/official_isco08_catalogue.py`). These are two different
  catalogues; the legacy 131/441 counts below do not equal, and must
  never be reported as equal to, the official 130/436 counts.
- **Test count**: §6 below reports **1,178 passed** as of 2026-08-02.
  That count is superseded. The current verified full-suite result (Task
  37.1, 2026-08-10, independently reproduced with zero live
  Qdrant/network/model dependency) is:
  ```
  2185 passed, 1 deselected, 1 warning
  ```
  See `Documentation/AI_HANDOFF/CLAUDE_TASK_37_1_FINAL_REPORT.md` for
  the exact command and full verification detail. The growth from 1,178
  to 2,185 reflects substantial work across Tasks 09-37.1 (WISCO
  evidence line, official-catalogue runtime, client-side deadline
  hardening, and this evidence-analysis tooling), not a correction of
  the 2026-08-02 count, which was accurate for its own date.
- **Manuscript-safe wording**: for any Reviewer #2 response or
  manuscript text citing the WISCO result, use
  `Documentation/Conference_I_Reviewer_2/MANUSCRIPT_SAFE_WISCO_WORDING.md`
  — it lists explicitly prohibited phrasing alongside ready-to-paste
  safe wording.

---

## 1. Purpose of This Document

This is the Phase 1 status summary for the thesis project *"Multilingual Conversational AI for Labour Force Surveys."* It consolidates:

- The current, **verified** system architecture and implementation status
- Live test-suite results (not carried over from an older run)
- A documentation accuracy audit performed on 2026-08-02, including every discrepancy found between the previously drafted README and the actual codebase, and the fixes applied
- The current repository/commit state, so Phase 2 planning starts from ground truth rather than assumption

Everything numeric in this document was checked against the live code or a live test run on 2026-08-02, not copied from a prior draft. See `Documentation/Test_Suite_Report.md` for the full per-test breakdown and `README.md` (repo root) for the living architecture reference — this document is a point-in-time snapshot for the Phase 1 review.

---

## 2. Project Overview

An AI-powered Labour Force Survey (LFS) system that conducts employment interviews in **English, Arabic (MSA + Gulf dialect), Urdu, Hindi, and Tagalog**, classifies:

- Occupation → **ISCO-08** (4-digit unit group)
- Industry → **ISIC Rev.4** (Section → Division → Group → 4-digit Class)
- Education field → **ISCED-F 2013** (Broad → Narrow → 4-digit Detailed)
- Education attainment → **ISCED 2011** (levels 0–8)

...and implements the complete **UAE Labour Force Survey questionnaire** (Sections A–K, 56 fields, ILO ICLS-19 standards) with dynamic skip logic across three employment paths (Employed / Unemployed / Outside Labour Force).

The thesis's novel contribution is the **Semantic Relation Engine** — a deterministic three-way crosswalk that cross-validates ISCO-08, ISIC Rev.4, and ISCED 2011 classifications against each other using ILO/UNESCO correspondence tables, producing a `SemanticCoherence` score (0–1) that adjusts ISCO confidence and can trigger HITL escalation on high-severity mismatches.

**Stack:** FastAPI (backend) · Next.js 14 + Tailwind (frontend) · PostgreSQL 15 · Redis 7 · Qdrant (vector DB) · CrewAI agents · Ollama/Llama 3.2 (general LLM tasks) · Claude 3.5 Sonnet (critical tasks: validation, ISCO re-ranking, reports).

---

## 3. System Architecture — 14 Agents

| # | Agent | File | Role |
|---|---|---|---|
| ① | LanguageProcessor | `language_processor.py` | Language detection (6 codes), Gulf Arabic normalisation, code-switch detection, NER |
| ② | ConversationManager | `conversation_manager.py` | 5-state FSM driving the 56-field UAE LFS questionnaire across 3 employment paths |
| ③ | ISCOClassifier | `isco_classifier.py` | 4-stage hierarchical RAG → ISCO-08 unit-group classification |
| ④ | ISICClassifier | `isic_classifier.py` | ISIC Rev.4 4-level industry classification (keyword + LLM) |
| ⑤ | ISCEDClassifier | `isced_classifier.py` | Dual ISCED classification: attainment level (2011) + field of specialisation (F-2013) |
| ⑥ | NationalityClassifier | `nationality_classifier.py` | UN M49 + ISO 3166-1 alpha-3 nationality resolution |
| ⑦ | ValidationAgent | `validation_agent.py` | 10 cross-answer consistency rules (R01–R10, ILO ICLS-19) |
| ⑧ | PersonRegister | `person_register.py` | Pre-fill from previous survey rounds (question-burden reduction) |
| ⑨ | HITLQualityManager | `hitl_quality_manager.py` | Automated quality scoring + escalation queue |
| ⑩ | AuditLogger | `audit_logger.py` | Immutable GDPR audit trail, 10-year retention |
| ⑪ | ReportGenerator | `report_generator.py` | Bilingual EN+AR employment report |
| ⑫ | EmotionalIntelligence | `emotional_intelligence.py` | Abandonment-risk detection, culturally adapted support messages |
| ⑬ | SemanticRelationEngine | `semantic_relation.py` | ISCO↔ISIC↔ISCED three-way crosswalk (thesis contribution) |
| ⑭ | SurveyOrchestrator | `survey_orchestrator.py` | Top-level per-turn agent pipeline coordinator |

Supporting infrastructure: `context_memory.py` (Redis session persistence), `rag_expert.py` + `hierarchical_store.py` (ISCO retrieval), `audit_logger.py`.

---

## 4. Implementation Status (verified 2026-08-02)

All items below are implemented and confirmed against the live codebase.

| Component | Status | Detail |
|---|---|---|
| Email OTP + JWT Auth | Done | Gmail SMTP (SendGrid fallback) + HS256 JWT, auto-fill in dev mode |
| Language Detection (6 codes) | Done | en / ar / ar-gulf / ur / hi / tl; Devanagari fast-path |
| Gulf Arabic Normalisation | Done | ~30 dialect→MSA token replacements before NER + embedding |
| Code-Switch Detection | Done | Arabic+Latin and Devanagari+Latin mixing detection |
| NER — 5 languages | Done | CrewAI agent, JOB_TITLE / INDUSTRY / LOCATION / EDUCATION |
| Conversation FSM (5 states) | Done | 56 fields, 3 employment paths, dynamic conditional skip gates |
| UAE LFS Questionnaire (A–K) | Done | All sections implemented; dynamic skip logic per ILO ICLS-19 |
| ILO ICLS-19 E1 wage gate | Done | monthly_wage_range skipped for employer/self-employed |
| ILO ICLS-19 F3 re-routing | Done | F1=no + F3=no → auto-reclassify to not_in_labour_force |
| ISCO-08 Knowledge Base | Done | 441 unit groups across 4 Qdrant collections (10 major / 43 submajor / 131 minor / 441 unit) |
| Hierarchical RAG (4-stage) | Done | Major→Sub-major→Minor→Unit with parent_code filtering |
| Per-stage Confidence Scoring | Done | Weighted: 0.10×s1 + 0.20×s2 + 0.20×s3 + 0.50×s4 |
| LLM Re-ranking | Done | Claude 3.5 Sonnet; skipped when top-1 similarity ≥ 0.92 |
| HITL Escalation (< 0.70) | Done | HITLQueue DB + priority ordering (HIGH first) |
| Supervisor Review Dashboard | Done | Approve / correct / reject with inline form |
| Evaluation Framework | Done | BM25 / Flat / Hierarchical 3-system comparison, 100 synthetic cases |
| Validation Agent (R01–R10) | Done | ILO ICLS-19 cross-answer consistency rules; wired into VALIDATING FSM state |
| ISCO-08 Keyword Pre-filter | Done | Major-group anchor prevents semantic drift |
| ISCO-08 Synonym Enrichment | Done | 50+ unit-group descriptions enriched with synonyms for better recall |
| ISIC Rev.4 Classification (4-digit) | Done | Full Section→Division→Group→Class hierarchy; keyword + LLM; 134 class entries; EN+AR |
| ISCED-F 2013 Field of Specialisation | Done | Broad→Narrow→Detailed; keyword two-pass; 11 broad fields, 63 detailed codes; EN+AR |
| ISCED 2011 Attainment Level | Done | Levels 0–8, combined with ISCED-F in single classifier; EN+AR |
| UN M49 Nationality Classification | Done | 50 countries, ISO 3166-1 alpha-3 + UN M49 codes, EN+AR+UR/HI/TL aliases |
| Person Register Pre-fill | Done | Question reduction from previous survey round |
| Emotional Intelligence Monitor | Done | Abandonment-risk detection + culturally adapted responses; wired into every turn |
| Audit / GDPR Compliance | Done | Immutable trail, Art. 15 subject-access, 10-year retention |
| Redis Context Persistence | Done | ContextMemory load/save/delete around every turn |
| Report Generator | Done | Bilingual EN+AR employment profile + recommendations |
| Quick-reply Pill Buttons | Done | 39 categorical fields × 5 languages in frontend |
| RTL Arabic Support | Done | Noto Sans Arabic, full RTL layout in chat + report |
| **Semantic Relation Engine** | **Done** | ISCO↔ISIC↔ISCED crosswalk; SemanticCoherence score; confidence adj. ±5–20%; HITL on HIGH violations |
| Cross-Standard Coherence API | Done | `semantic_coherence` field in every message response |
| Coherence Report Panel | Done | Score bar, compatibility flags, violation list (EN+AR) on report page |
| Rate Limiting | Done | Sliding-window (OTP) + `slowapi` 30 req/min per-IP on `/message` |
| OTP Brute-force Lockout | Done | Locks OTP after 5 consecutive wrong codes |
| Security Headers Middleware | Done | CSP, X-Frame-Options, X-Content-Type-Options, Referrer-Policy, Permissions-Policy |
| LFS_FAST_MODE | Done | Deterministic stub responses for fast local dev without live LLM calls |
| CORS Allowlist | Done | Explicit origin/method/header allowlist from `CORS_ORIGINS`; `allow_credentials=True` |
| X-Request-ID Middleware | Done | UUID correlation header on every response |
| Soft-Delete | Done | `deleted_at` on User, SurveySession, SurveyResponse; enforced in all query paths |
| Graceful Shutdown | Done | Lifespan handler closes Redis + disposes DB engine on SIGTERM |
| Load Test Suite | Done | `@pytest.mark.slow`; 5 users, 2 workers, ≥80% success-rate assertion |

Full detail and code-level references for each row: `README.md` (repo root).

---

## 5. Database Schema — 11 Tables

`users`, `otp_codes`, `survey_sessions`, `survey_responses`, `audit_logs`, `data_access_logs`, `agent_decision_logs`, `quality_reviews`, `hitl_queue`, `survey_report_records`, `person_register`.

Alembic migration chain (3 revisions): `001_initial_schema` → `976b9b9c96d4_add_hitlqueue_evaluation_tables` → `f514fcb81c72_add_deleted_at_soft_delete_columns`.

---

## 6. Test Suite Results (live run, 2026-08-02)

```
1178 passed, 1 deselected, 1 warning in 340.50s (0:05:40)
```

| Metric | Value |
|---|---|
| Total tests | 1,178 |
| Passed | 1,178 |
| Failed | 0 |
| Test files | 25 (+ `conftest.py` + slow-marked `load_test.py`) |
| Infrastructure required | **None** — SQLite in-memory DB, FakeRedis, mocked Qdrant/LLM/CrewAI |

### Tests by module

| Test File | Component | Tests |
|---|---|---|
| `test_context_memory.py` | ContextMemory | 83 |
| `test_validation_agent.py` | ValidationAgent | 82 |
| `test_hitl_quality_manager.py` | HITLQualityManager | 81 |
| `test_isco_classifier_extended.py` | ISCOClassifier (extended) | 80 |
| `test_audit_logger.py` | AuditLogger | 77 |
| `test_emotional_intelligence.py` | EmotionalIntelligence | 74 |
| `test_survey_orchestrator.py` | SurveyOrchestrator | 65 |
| `test_conversation_manager.py` | ConversationManager | 63 |
| `test_validation_agent_extended.py` | ValidationAgent (extended) | 59 |
| `test_language_processor.py` | LanguageProcessor | 52 |
| `test_language_processor_extended.py` | LanguageProcessor (extended) | 47 |
| `test_rag_expert.py` | RAGExpert | 46 |
| `test_report_generator.py` | ReportGenerator | 45 |
| `test_nationality_classifier.py` | NationalityClassifier | 45 |
| `test_conversation_manager_extended.py` | ConversationManager (extended) | 42 |
| `test_hitl_and_e2e_extended.py` | HITL & E2E (extended) | 37 |
| `test_isco_classifier.py` | ISCOClassifier | 31 |
| `test_vector_store.py` | VectorStore | 29 |
| `test_survey_routes.py` | Survey Routes (API) | 29 |
| `test_auth_and_api_extended.py` | Auth & API (extended) | 27 |
| `test_auth_routes.py` | Authentication Routes | 22 |
| `test_isic_classifier.py` | ISICClassifier | 19 |
| `test_isced_classifier.py` | ISCEDClassifier | 17 |
| `test_evaluation.py` | Evaluation Framework | 16 |
| `test_person_register.py` | PersonRegister | 10 |
| **TOTAL** | | **1,178** |

Full per-test breakdown (all 1,178 tests, class-by-class): `Documentation/Test_Suite_Report.md`.

---

## 7. Documentation Audit — Findings & Corrections (2026-08-02)

Before this Phase 1 summary was written, `README.md` (already substantially rewritten in a prior, uncommitted session) was audited claim-by-claim against the live codebase. The test-suite claim was independently re-verified by actually running pytest. The following discrepancies were found and corrected:

| # | Claim (as previously drafted) | Verified actual value | Fixed in |
|---|---|---|---|
| 1 | "55 countries" (nationality classifier) | **50** countries (`_COUNTRY_DATA` in `nationality_classifier.py`) | README.md (4 locations) |
| 2 | "130" ISCO-08 minor groups | **131** (`_MINOR` list in `load_full_isco.py`) | README.md architecture diagram |
| 3 | "10 broad fields" (ISCED-F 2013) | **11** broad codes, 00–10 (`_ISCED_FIELDS` in `isced_classifier.py`) | README.md |
| 4 | "1,130 tests across 28 test files" (2 locations) | **1,178 tests across 25 test files** — confirmed live | README.md; contradicted the doc's own accurate header |
| 5 | "Credentials are not exposed (`allow_credentials=False`)" | Code has `allow_credentials=True` in `backend/main.py` | README.md Security Features |
| 6 | Field-count-by-path table (Employed 31+13=44, Unemployed 22+9=31, Outside LF 16+7=23, shared base 16) | Recomputed via live `ConversationManager._get_field_order()` calls: Employed 38+7=45, Unemployed 23+9=32, Outside LF 20+7=27, shared base 18; union of all fields = 56 | README.md field-count table |
| 7 | 3rd Alembic migration (`f514fcb81c72_...`) missing from project structure tree | File exists and is correctly chained | README.md project structure |
| 8 | Undocumented: `SecurityHeadersMiddleware`, `slowapi` 30 req/min rate limiter on `/message`, `LFS_FAST_MODE` env var | All three exist and are wired in `backend/main.py` / `survey_routes.py` | README.md Security Features + Env Vars |
| 9 | `Documentation/Test_Suite_Report.md` — 765 tests, dated 2026-03-08 | Stale (14 files, 765 tests); regenerated from a live `pytest --collect-only` run | Fully regenerated |
| 10 | `Documentation/CLAUDE.md` — describes a `hierarchical_rag/` + numbered-agent-file (`agent_01_auth.py`…) layout | This layout was never built; actual code lives under `backend/agents/`, `backend/rag/`, `backend/api/` | Added "ARCHIVED — SUPERSEDED" banner pointing to README.md |

Everything else checked (ISCO/ISIC/ISCED classifier structural claims, 39×5 pill buttons, 56-field total, 11 DB tables, R01–R10 validation rules, SemanticCoherence engine mechanics, HITL frontend/backend wiring and request schema) matched the README precisely and required no changes.

---

## 8. Repository State (as of 2026-08-02)

**Important for Phase 2 planning:** the verified state above reflects the working tree, not what is committed to git. As of this summary:

- ~9,600 lines of uncommitted changes across 31 tracked files
- ~40 untracked new files, including 5 new agents (`isic_classifier.py`, `isced_classifier.py`, `nationality_classifier.py`, `semantic_relation.py`, `person_register.py`), the hierarchical RAG store, the evaluation framework, 2 Alembic migrations, and ~13 new/extended test files
- Latest commit on `master`: `fe0e5b7` ("Add report API, report page, and initial Alembic migration") — everything described in Sections 3–6 above post-dates that commit and is not yet in git history
- This Phase 1 summary, the regenerated `Test_Suite_Report.md`, and the `Documentation/CLAUDE.md` archival banner are themselves new/uncommitted files as of writing

**Recommendation:** commit this body of work (in logical groups — new classifiers, new tests, doc updates) before starting Phase 2, so the verified state has a permanent record and isn't at risk of being lost.

---

## 9. Suggested Phase 2 Candidates

Not yet implemented or explicitly out of scope for Phase 1 (not verified as gaps during this audit — listed for planning discussion, confirm against current code before committing to any of these):

- Committing the current working tree to git (see §8)
- Multi-worker / distributed rate limiting (current limiters are in-process; noted as a known limitation in README)
- Production CORS/secrets hardening for a non-localhost deployment target
- Extending the Semantic Relation Engine's per-violation severity levels beyond the two currently emitted in practice (HIGH/MODERATE)
