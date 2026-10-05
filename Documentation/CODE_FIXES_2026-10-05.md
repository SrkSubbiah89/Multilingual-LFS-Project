# Code fixes and verification — 5 October 2026

**Subsequent activity:** [local activation and browser verification](LOCAL_ACTIVATION_2026-10-05.md) records the later applied migration, requested supervisor grant, startup, and additional readiness/mobile fixes. Configuration/database statements below describe the completed code-fix pass before that activation.

The application fixes below address the 28 findings in [the 3 October review](CODE_REVIEW_2026-10-03.md), plus defects found while independently reviewing the fixes. They are working-tree changes based on `f9c4f5c`; no commit or production deployment was performed. The original review records the behavior before these changes.

## Changes by original finding

| Finding | Implemented change | Main regression evidence |
| --- | --- | --- |
| 1 — supervisor authorization | Both HITL endpoints require an active, explicitly configured reviewer. Empty reviewer configuration denies access. Review scope is global over active sessions and users. Deleted sessions are excluded. | `test_review_persistence_fixes.py`: ordinary-user denial, authorized corrections, deleted-session exclusion. |
| 2 — translated employment answers | Quick buttons send canonical field/value pairs alongside their localized labels. The backend validates the current question and enum, normalizes typed labels, and reasks unknown status values. | Conversation regressions across languages; frontend canonical-button/handler tests. |
| 3 — fabricated wages | A number becomes a wage only on the wage question or in an explicit wage/currency statement. Amounts are associated with the salary phrase. | Residence duration, experience, tax, hours, explicit wages, native digits, and refusals. |
| 4 — top-3 denominator | Eligible retrieval pools count even when gold is absent. Missing telemetry is distinguished from a retrieval miss. Corrected summaries preserve original raw CSVs. | Analyzer tests and hashed correction sidecars. Full130 retrieval top-3 is **61/130 = 46.92%**, smoke25 is **21/25 = 84%**. |
| 5 — fresh database startup | Backend image packages Alembic configuration and upgrades to the schema head before Uvicorn starts. | Fresh PostgreSQL 15 migration, repeated upgrade, complete table/column comparison. |
| 6 — duties refinement | The displayed refined occupation is saved as an active answer revision, with obsolete review entries retired. | Actual route regression for duties refinement and next-turn replay. |
| 7 — title corrections | Value changes trigger revisions and classification, including when NER is skipped. Completion and reports use active revisions. | Title correction, stored code, supersession, completion, and report-profile assertions. |
| 8 — deleted prefills | Returning-user queries exclude deleted sessions and inactive response revisions. | Deleted-session and superseded-answer prefill regression. |
| 9 — encoder provenance | New evaluation rows, hashes, and manifests derive encoder identity from resolved configuration/store state. Historical sidecars separate intended configuration from unproven executed encoder identity. | Evaluation/exporter/manifest tests; source-hashed provenance sidecars. Historical E5-large identity remains unresolved. |
| 10 — dry-run accuracy | Unmeasured rows do not become measured accuracy; legacy status assumptions are explicit. | Analyzer dry-run and legacy-row tests. |
| 11 — survey statistics | Removed the unsupported 155-question skipped/reduction calculation. Progress uses the implemented path and records actual prefills. | Frontend build and source review of progress rendering. |
| 12 — misleading report badge | Live pending-review state and completed human decisions take precedence over confidence; unassessed reports have a neutral state. | Backend report-state and frontend status regressions. |
| 13 — SMS rate limits | Client and normalized-recipient limits run before account creation, OTP generation, and delivery. | Both rejection paths assert no delivery/account creation. No SMS was sent. |
| 14 — reviewer confidence | Supervisor cells read `ai_confidence`; unauthorized reviewers receive a clear access message. | Frontend API-shaped rendering review and production build. |
| 15 — negated confirmation | Negation/correction is processed before confirmation, including common contractions and curly apostrophes. | Conversation confirmation/correction regressions. |
| 16 — unlinked SRE reviews | New semantic escalations link to the active occupation. Legacy unlinked entries resolve that occupation before a review is applied. | Linked/unlinked human correction and report backstop tests. |
| 17 — stale reports/quality | Human decisions save revisions, invalidate cached reports, and reassess occupation quality. Report coherence is recomputed when the invalidated report is regenerated. | Revision, report-cache, quality, and authoritative decision-state assertions. |
| 18 — never-worked escalations | Non-applicable occupation sentinels are exempt. Genuinely missing required occupations remain flagged. No model confidence is invented for an unclassified profile. | Never-worked pass and missing-employed-occupation escalation. |
| 19 — language changes as answers | A dedicated PATCH changes language and rerenders the current prompt without a user answer or duplicated assistant turns. | Repeated language changes preserve answers, question, and transcript length. |
| 20 — abandoned interviews on reload | The frontend persists the session ID and restores history/current state. Temporary API errors preserve the ID; only confirmed absence starts a replacement. | Frontend reload/error tests; backend resume/history tests. |
| 21 — reference-label integrity | Official and WISCO analyzers validate result gold labels against canonical reference labels before scoring. | Altered-gold fixtures fail validation. |
| 22 — retry telemetry | All classification, reformulation, and query-planning calls contribute usage and timings. Winning retry output, hierarchy, alternatives, decision, and model-specific cost agree with the final prediction. | Corrective-retry, query-planner, and exporter regressions. |
| 23 — gap-aware shortcut | Enabled ambiguity checks run before high-confidence early returns. Review-required explanations match the result. | Near-tie high-confidence regression. |
| 24 — wage band boundaries | Canonical options and inequalities are recognized before bare-number bucketing. | Lowest/highest bands and translated options. |
| 25 — absent frontend public directory | The builder creates the optional public directory before build/copy. | Dockerfile review and Next production build. A full Docker image build was not run. |
| 26 — unavailable curl healthcheck | Backend healthcheck uses Python's standard-library HTTP client. | Docker/Compose declaration review. Full Compose startup was not run. |
| 27 — incomplete FSM restoration | Memory round-trips clarification target/count, correction flags, returning-user state, and original prefill keys. | ContextMemory serialization and resumed clarification regressions. |
| 28 — stale worker contexts | Each turn refreshes Redis state under local and distributed session locks. Leases renew during long turns; ownership checks protect context, response, report, and quality writes. Storage failures propagate. | Worker-refresh, busy/unavailable lock, renewal, and lost-ownership regressions. |

## Additional fixes from independent review

- Revision helpers flush before querying, covering the production `autoflush=False` configuration. Coordinator promotion retires an unflushed obsolete escalation and saves the final displayed code.
- Raw response POST/PATCH uses the same revision, cache-invalidation, context, and locking machinery. PATCH cannot move an answer to another question.
- Deletion, completion, snapshot, language, review, and report operations share session locking. Ownership/deletion checks run inside the relevant lock. Report lock failures retain HTTP 503.
- If Redis state expires or is malformed, both resume and direct-message paths reconstruct active database answers. Old localized routing values are normalized; invalid gates are reasked. A direct message no longer retires saved answers by starting from an empty context. The old transcript cannot be reconstructed from answer rows after cache loss.
- Correcting education, secondary-job, or platform-work gates reopens newly applicable questions. Structured quick answers require both field and value and reject stale questions before mutating conversation history.
- Generated semantic explanations identify the project's heuristic rules rather than claiming an official ILO crosswalk or unsupported confidence adjustment.
- Alembic safely accepts percent-encoded database passwords.

## Final validation

Final backend/evaluation suite: **2,768 passed, one slow test deselected, two warnings, in 172.37 seconds**. [The 5 October JUnit file](CODE_FIXES_2026-10-05_TEST_RESULTS.xml) preserves per-test results. The warnings concern an already-imported test plugin and an intentionally short JWT secret in a wrong-secret rejection fixture. The intermediate [4 October run](CODE_FIXES_2026-10-04_TEST_RESULTS.xml) had 2,740 passes and nine failures; those failures were resolved and the affected modules passed before the final run. Focused and full-suite counts overlap and must not be added together.

Frontend regression tests: **13 passed**. The Next.js production build completed and generated seven routes.

`git diff --check` and Python compilation of `backend`, `eval`, and `scripts` passed. The independent artifact audit verified all **nine raw-CSV source hashes** and both referenced manifest hashes; no top-1 predictions/scores were changed.

[Fresh database verification](CODE_FIXES_2026-10-05_DEPLOYMENT_RESULTS.json): **11 model tables**, all model columns present, head **`d841b2079c65`**, second upgrade idempotent, and percent-encoded password verified. The verifier created and removed its own PostgreSQL container without accessing the user database.

[Real Redis verification](CODE_FIXES_2026-10-05_REDIS_RESULTS.json): **four integration groups passed** against a disposable Redis 7 container: complete state restoration across workers, visibility of a second worker's update, cross-thread lock exclusion/reacquisition, and rejection after ownership loss. The task's containers were removed after identity checks. This complements the unit tests; long-duration renewal and production load were not exercised against real Redis.

Reproduce from the repository root with Python 3.11 and installed project dependencies:

```powershell
py -3.11 scripts/run_review_tests.py backend/tests eval -q --tb=short --junitxml=Documentation/CODE_FIXES_2026-10-05_TEST_RESULTS.xml
npm --prefix frontend test
npm --prefix frontend run build
py -3.11 scripts/verify_fresh_database.py --output Documentation/CODE_FIXES_2026-10-05_DEPLOYMENT_RESULTS.json
py -3.11 scripts/verify_fresh_redis.py --output Documentation/CODE_FIXES_2026-10-05_REDIS_RESULTS.json
```

The test runner disables `.env` loading, live providers, real database/Redis access, and startup warmup. It uses a test-only database/JWT, mocked route memory with the real serializer, and local mock servers. The integration verifiers require Docker with the existing `postgres:15` and `redis:7` images, use ephemeral loopback ports, and construct no LLM agents.

## Applying the working-tree changes

Before restarting an existing deployment, apply the migration with its intended database configuration:

```powershell
py -3.11 -m alembic upgrade head
```

Docker startup performs this upgrade automatically. Configure `HITL_REVIEWER_USER_IDS` with comma-separated database IDs of authorized supervisors; an empty setting denies all supervisor access. This grants global review access over active respondents. Redis must be available for survey updates; the API returns an error if it cannot save a turn. Rebuild/restart the application processes to load the changes. The actual `.env` and existing survey database were not changed during this task.

## Evidence limits

These checks establish the listed regressions and build/schema behavior. They do not establish live LLM accuracy, human usability, complete browser workflows, a full Docker/Compose deployment, or production concurrency/load behavior. Redis leases with ownership checkpoints do not establish an atomic transaction spanning PostgreSQL and Redis. Historical raw results, original manifests, prior thesis changes, archives, and local logs were preserved. The thesis PDF describes the 3 October review snapshot; this document records the later application fixes.

Historical encoder identity and contamination of previously collected answers cannot be repaired by inventing provenance or rewriting respondent data. The new code prevents the reviewed defects prospectively, and the correction sidecars explicitly retain those evidence limits.
