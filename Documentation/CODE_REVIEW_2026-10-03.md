# Code review — 3 October 2026

**Follow-up, 5 October 2026:** application fixes and their validation are recorded in [CODE_FIXES_2026-10-05.md](CODE_FIXES_2026-10-05.md). The findings and original test evidence below are the historical pre-fix review.

Reviewed the working tree at commit `f9c4f5c` in `C:\Multilingual_LFS_Project`. This review covers the main authentication, survey, classification, reporting, frontend, evaluation, and Docker startup paths. Application source was left unchanged. The existing untracked local log files were left alone.

The most urgent findings concern access to other respondents' data, incorrect questionnaire routing, silently fabricated wage answers, an inflated evaluation metric, and first-boot deployment failure. P1 means address before deployment or using the affected behavior as thesis evidence; P2 means an actionable correctness, reliability, or presentation defect.

**1. P1 — Ordinary respondents can read and modify other respondents' supervisor reviews.**

Locations: [survey_routes.py:1832](C:/Multilingual_LFS_Project/backend/api/survey_routes.py:1832), [survey_routes.py:1863](C:/Multilingual_LFS_Project/backend/api/survey_routes.py:1863).

Both HITL endpoints require only `get_current_user`. The queue query is global, and the review endpoint accepts any escalation ID without checking a supervisor role or authority over the target session. The review then changes the associated respondent's stored ISCO code. In an isolated database, a second ordinary user successfully listed the first user's queue item and changed its response code. Add a supervisor authorization dependency to both endpoints and enforce the intended scope of review access on the server.

**2. P1 — Hindi and Urdu employment-status buttons can bypass almost the entire survey.**

Locations: [chat.js:146](C:/Multilingual_LFS_Project/frontend/pages/chat.js:146), [chat.js:728](C:/Multilingual_LFS_Project/frontend/pages/chat.js:728), [conversation_manager.py:3564](C:/Multilingual_LFS_Project/backend/agents/conversation_manager.py:3564).

The frontend sends the translated button label as the answer. Employment-status extraction recognizes English and Arabic phrases, but does not normalize the Hindi and Urdu labels. Executing the actual conversation methods in isolation showed that these clicks enter clarification. After three further clarification answers, the fallback stores the translated label as the status. Because that value is not `employed`, `unemployed`, or `not_in_labour_force`, field order collapses to employment status and education. Answering education then enters validation. Send canonical option values alongside display labels, or normalize all supported labels before routing; validate the stored status against its enum.

**3. P1 — Numbers in unrelated answers become wage answers.**

Location: [conversation_manager.py:3733](C:/Multilingual_LFS_Project/backend/agents/conversation_manager.py:3733).

Wage extraction scans every employed respondent's answer for a number whenever `monthly_wage_range` is missing. It does not require the wage question or a reference to wages. Reproduction: while the next field is `uae_residence_duration`, answering `5-9 years` sets both residence duration and `monthly_wage_range='under_5000'`. The later wage question is skipped because the field already exists. Restrict this extraction to the wage question or an explicit wage/currency statement, and review existing collected data for this contamination.

**4. P1 — Top-3 evaluation accuracy excludes cases where retrieval misses the gold code.**

Locations: [analyze.py:188](C:/Multilingual_LFS_Project/eval/analyze.py:188), [run_eval.py:716](C:/Multilingual_LFS_Project/eval/run_eval.py:716).

`gold_rank_in_pool` is blank when the gold occupation is absent from the retrieved pool. The analyzer discards those rows from the denominator. Direct execution against `eval/results/raw_runs/20260805T055741Z_full130_leafvote_beam3_llama3b_pooled.csv` found 130 nonempty pools, 108 containing the gold code, and 61 top-3 hits. The analyzer reports **56.48% (61/108)**; accuracy across those 130 cases is **46.92% (61/130)**. The confidence interval is affected too. Count eligible retrieval misses as failures and explicitly distinguish missing telemetry from an absent gold candidate. Recompute affected top-3 summaries. This finding concerns this metric; it does not establish that the separate heldout top-1 results are wrong.

**5. P1 — A fresh Docker deployment starts without creating database tables.**

Location: [Dockerfile:22](C:/Multilingual_LFS_Project/Dockerfile:22).

The initialization command imports `Base` and `engine` from `connection.py` and calls `Base.metadata.create_all()` before importing the model declarations. A fresh-process check confirmed that this metadata contains **zero tables**. The later application import registers the models but does not rerun schema creation. With a new PostgreSQL volume, authentication therefore encounters missing tables. Package and run Alembic migrations during initialization; at minimum, import the models before attempting metadata-based creation. An existing database can hide this defect.

**6. P2 — Classification refined with job duties is displayed but not saved.**

Location: [survey_routes.py:1034](C:/Multilingual_LFS_Project/backend/api/survey_routes.py:1034).

The duties-triggered reclassification replaces `isco_results` without updating or superseding the stored `SurveyResponse`, or updating its review entry. An isolated endpoint implementation check returned refined code `1213` while the database retained `2512`; the next turn returned cached code `2512` again. The final report consequently uses the earlier classification. Persist the refined classification and synchronize its review state before returning it.

**7. P2 — Corrected job titles can be lost from the completed report.**

Locations: [survey_routes.py:846](C:/Multilingual_LFS_Project/backend/api/survey_routes.py:846), [survey_routes.py:1776](C:/Multilingual_LFS_Project/backend/api/survey_routes.py:1776).

When NER is skipped or misses the corrected job title, correction changes only the value of an existing context key, so the key-set comparison does not treat `job_title` as newly collected and the fallback classification is not retriggered. Completion then skips job title whenever a response already exists. Reproduction using the real structured-correction transition with NER skipped: the context changed from `Engineer` to `Doctor`, the API classification still named `Engineer`, and the completed report profile still named `Engineer`. Use value changes or `corrected_fields` to create a new response revision, supersede the old row, reclassify, and consistently select active revisions. The model already has `supersedes_id`, but this path does not use it.

**8. P2 — Deleted surveys are reused as returning-user prefills.**

Location: [survey_routes.py:608](C:/Multilingual_LFS_Project/backend/api/survey_routes.py:608).

The previous-completed-session query does not exclude sessions with `deleted_at`, and its response aggregation does not exclude deleted or superseded responses. Starting another survey can therefore restore answers from a survey the respondent deleted, or use obsolete revisions. The deleted-session behavior was reproduced in an isolated database. Filter both sessions and responses to active records and select the latest active answer deterministically.

**9. P2 — E5-large evaluation artifacts identify their encoder as E5-small.**

Location: [run_eval.py:662](C:/Multilingual_LFS_Project/eval/run_eval.py:662).

Every non-BM25 row receives the constant `EMBEDDING_MODEL_NAME`, rather than the model resolved from the selected catalogue profile. The configuration hash also uses this constant. The committed `enriched_e5large_heldout_20260824/20260824T123941Z_flat.csv` names the enriched E5-large method but records `embedding_model_version=intfloat/multilingual-e5-small`. Derive the encoder identity from the actual profile/classifier configuration and correct affected provenance metadata and downstream manifests. Do not infer a changed accuracy score from this metadata defect alone.

**10. P2 — Dry-run output is analyzed as measured accuracy.**

Location: [analyze.py:161](C:/Multilingual_LFS_Project/eval/analyze.py:161).

Exact-match analysis ignores `evaluation_status` and emits `status='measured'` whenever gold labels exist. A row marked `dry_run` with gold `2512` and an empty prediction produced measured 0% accuracy with a confidence interval, although classification never occurred. Reject or separately report unmeasured rows, and handle legacy CSVs without a status column explicitly.

**11. P2 — The frontend's skipped-question statistics use an unsupported 155-question baseline.**

Locations: [chat.js:422](C:/Multilingual_LFS_Project/frontend/pages/chat.js:422), [chat.js:578](C:/Multilingual_LFS_Project/frontend/pages/chat.js:578).

The UI subtracts the current path length from `TOTAL_QUESTIONS=155`. Enumerating the backend's actual field-order branches yields 56 distinct fields and a maximum path of 45. A common employed path with total 38 immediately shows **117 skipped questions and 75% reduction**. These numbers do not measure questions skipped within the implemented questionnaire or benefits from returning-user prefill. Derive the baseline from the implemented questionnaire; if a separate manual questionnaire is the intended comparator, identify it and establish a defensible mapping before presenting these numbers.

**12. P2 — Reports can label escalated classifications “AI Verified.”**

Location: [report.js:727](C:/Multilingual_LFS_Project/frontend/pages/report.js:727).

The badge uses only ISCO confidence below 0.70 to decide whether human review is needed. The backend also escalates high-confidence classifications for HIGH semantic-coherence violations and reports `quality_status='escalated'`. Evaluating the actual badge condition with confidence 0.95 selects “AI Verified” even for that supported escalation scenario. Base the badge on authoritative pending-review/escalation state and use confidence as supplementary information.

**13. P2 — SMS OTP requests have no request rate limit.**

Location: [auth_routes.py:189](C:/Multilingual_LFS_Project/backend/api/auth_routes.py:189).

Unlike email OTP requests, SMS requests proceed to OTP generation and Twilio delivery without client or phone-number limits. When SMS delivery is configured, an unauthenticated caller can repeatedly request paid messages and invalidate a recipient's outstanding code. Apply client and normalized-recipient limits before code generation and delivery. This finding was established from code; no SMS was sent during review.

**14. P2 — Supervisor confidence cells read the wrong API property.**

Location: [supervisor_review.js:314](C:/Multilingual_LFS_Project/frontend/pages/supervisor_review.js:314).

The UI reads `item.confidence`, while `HITLQueueItem` exposes `ai_confidence`. Evaluating the actual rendering expression with an API-shaped item containing `ai_confidence: 0.40` produces `—`. Use the documented API field so supervisors can assess the classifier's confidence.

The continued review added these findings:

**15. P1 — Explicit disagreement can complete the survey instead of opening correction.**

Location: [conversation_manager.py:3472](C:/Multilingual_LFS_Project/backend/agents/conversation_manager.py:3472).

The validating-state transition checks confirmation before checking correction intent. Confirmation detection matches positive words inside negative phrases. Executing the original methods with `not right`, Arabic `غير صحيح`, Urdu `درست نہیں`, Hindi `सही नहीं`, and Tagalog `hindi tama` returned both confirmation and correction as true, then transitioned to `completing`. `Yes, but change my industry to healthcare` also completed without applying the requested change. Resolve explicit disagreement and correction intent before confirmation, and make confirmation detection account for negation. These checks used the actual logic without an LLM or live services.

**16. P2 — Semantic-coherence review corrections can succeed without changing survey data.**

Locations: [report_generator.py:516](C:/Multilingual_LFS_Project/backend/agents/report_generator.py:516), [survey_routes.py:1268](C:/Multilingual_LFS_Project/backend/api/survey_routes.py:1268), [survey_routes.py:1902](C:/Multilingual_LFS_Project/backend/api/survey_routes.py:1902).

The SRE escalation paths create queue entries with `response_id=None`, but the review endpoint propagates corrections only when this ID exists. An isolated check of the actual report-generation backstop with the real SRE created such an entry. Correcting its occupation from `2512` to `1213` returned a success message and marked the item reviewed, while the stored response and a regenerated report retained `2512`. Link the escalation to the active occupation response or implement explicit session-level correction persistence.

**17. P2 — Successful human corrections leave cached reports and quality assessments stale.**

Locations: [report_generator.py:380](C:/Multilingual_LFS_Project/backend/agents/report_generator.py:380), [survey_routes.py:1908](C:/Multilingual_LFS_Project/backend/api/survey_routes.py:1908).

Even when a queue entry is linked and the correction updates its response, the endpoint neither invalidates the cached report nor recomputes `QualityReview`. Verification changed the database code from `2512` to `1213`; an ordinary report request still returned the same report ID and code `2512`. Explicit regeneration updated the code but retained the preceding quality assessment (`fail`). Invalidate affected report caches and recompute applicable quality/coherence state after human decisions. This is separate from the unlinked-escalation failure in finding 16.

**18. P2 — Respondents who have never worked receive false occupation-quality escalations.**

Locations: [hitl_quality_manager.py:498](C:/Multilingual_LFS_Project/backend/agents/hitl_quality_manager.py:498), [hitl_quality_manager.py:540](C:/Multilingual_LFS_Project/backend/agents/hitl_quality_manager.py:540).

The quality manager treats every `job_title` or `last_job_title` row as requiring ISCO and assigns zero confidence/coverage when no occupation applies. A completed outside-labour-force profile with every required field and the actual `never_worked` sentinel produced quality score `0.20`, a missing-ISCO flag, and `quality_status='escalated'`. Exempt non-applicable occupation sentinels and assess sessions with no expected occupational classification separately from genuinely missing required codes.

**19. P2 — Changing the language can record an answer the respondent never supplied.**

Location: [chat.js:787](C:/Multilingual_LFS_Project/frontend/pages/chat.js:787).

Before the first real respondent answer, the language selector sends another literal `hello` to the existing session. Its initial greeting has already advanced to answer collection, so this request is interpreted as an employment-status answer. Source-extracted frontend and backend checks showed that one language change enters clarification and four changes store `employment_status='hello'`, reducing the normal survey path to education. Update the language and rerender the current prompt without advancing the conversation or supplying a synthetic answer.

**20. P2 — Reloading the chat abandons access to the current interview.**

Locations: [chat.js:649](C:/Multilingual_LFS_Project/frontend/pages/chat.js:649), [survey_routes.py:590](C:/Multilingual_LFS_Project/backend/api/survey_routes.py:590).

Every chat mount creates a new session. Its ID exists only in React state, and the frontend does not offer session resumption or transcript restoration. Executing the original initialization twice created distinct session IDs while storage kept only authentication and language. Because backend prefill consults completed sessions only, reloading midway presents a fresh questionnaire and makes current progress inaccessible through the UI. Persist the active session ID and restore its context/history on return. The old server records remain; this finding does not claim they are deleted.

**21. P2 — WISCO analysis does not verify result gold labels against the hashed reference.**

Locations: [analyze_wisco_tier1.py:580](C:/Multilingual_LFS_Project/eval/analyze_wisco_tier1.py:580), [analyze_wisco_tier1.py:266](C:/Multilingual_LFS_Project/eval/analyze_wisco_tier1.py:266), [analyze_wisco_tier1.py:403](C:/Multilingual_LFS_Project/eval/analyze_wisco_tier1.py:403).

The reference validation passes only case IDs downstream, result validation checks gold-code format, and accuracy uses the result CSV's labels. A temporary 24-case reference covering all required risk IDs had canonical gold `2512`; both result files used gold `2221` and prediction `2221`. The actual validation gates accepted these files and reported 100% accuracy, versus 0% against the canonical reference. Compare each result's gold label with its reference and score using canonical labels. This demonstrates an integrity-check gap; it does not establish that the committed WISCO results contain altered labels.

**22. P2 — Corrective retry exports incomplete usage and an earlier reranker decision.**

Locations: [isco_classifier.py:1208](C:/Multilingual_LFS_Project/backend/agents/isco_classifier.py:1208), [isco_classifier.py:1363](C:/Multilingual_LFS_Project/backend/agents/isco_classifier.py:1363), [run_eval.py:767](C:/Multilingual_LFS_Project/eval/run_eval.py:767).

Retry reranking does not receive the trace, and query reformulation does not collect usage, so evaluation exports initial-call telemetry. With original classifier/exporter methods and mocked Crew/store, a corrective run made three LLM calls and two retrievals but exported 100/20 prompt/completion tokens instead of the total 300/60. Under the same configured prices, the estimate was one-third of the complete run's cost. The final prediction was `2221`, while `reranker_output.code` remained `2512` and `retry_count` was zero. Capture each attempt, aggregate usage, and identify the winning attempt explicitly.

**23. P2 — The high-confidence shortcut bypasses enabled gap-aware escalation.**

Locations: [isco_classifier.py:990](C:/Multilingual_LFS_Project/backend/agents/isco_classifier.py:990), [isco_classifier.py:1062](C:/Multilingual_LFS_Project/backend/agents/isco_classifier.py:1062).

The classifier returns immediately at similarity at least 0.92, before the gap-aware ambiguity check. Original-method verification with `use_gap_aware_confidence=True` and candidate scores 0.950/0.949 returned `hitl_required=False` despite the 0.001 gap falling below the configured 0.01 ambiguity threshold. Evaluate gap-based escalation before the early return. This affects explicitly enabled gap-aware behavior; the flag is off by default.

The resumed review checked additional deployment, restart, and answer-normalization paths:

**24. P2 — The wage buttons for the lowest and highest bands select the wrong band.**

Location: [conversation_manager.py:3738](C:/Multilingual_LFS_Project/backend/agents/conversation_manager.py:3738).

Even when the wage question is the current question, the generic numeric parser runs before the inequality checks. Executing the original extraction method with the actual frontend option `Less than 5,000` stored `5000_10000`; `More than 50,000` stored `20001_50000`. These answers should map to `under_5000` and `over_50000`, respectively. Recognize the canonical categorical option and inequality before bucketing a bare number. This is distinct from finding 3, which concerns numbers extracted from unrelated questions.

**25. P1 — A clean frontend Docker build copies a directory that does not exist.**

Location: [frontend/Dockerfile:26](C:/Multilingual_LFS_Project/frontend/Dockerfile:26).

The runner stage unconditionally copies `/app/public` from the builder, but `frontend/public` is absent and has no tracked files. The builder copies the frontend source and runs `next build`; the installed Next.js implementation treats this directory as optional and does not create it. The runner-stage copy therefore fails on a clean build. Remove the copy when no public assets are needed, or explicitly create and package the directory. Verification checked the filesystem, tracked files, build script, and installed Next.js build code; a real Docker build was not run.

**26. P1 — The backend healthcheck requires curl, which the image does not install.**

Locations: [docker-compose.yml:103](C:/Multilingual_LFS_Project/docker/docker-compose.yml:103), [Dockerfile:7](C:/Multilingual_LFS_Project/Dockerfile:7), [docker-compose.yml:120](C:/Multilingual_LFS_Project/docker/docker-compose.yml:120).

The healthcheck runs `curl`, while the backend Dockerfile uses the Python slim image and installs only `gcc` and `libpq-dev`. The [official Python slim Dockerfile](https://raw.githubusercontent.com/docker-library/python/master/3.11/slim-trixie/Dockerfile) does not supply curl either. Consequently the healthcheck cannot succeed even when the API responds, and the frontend's `service_healthy` dependency prevents its startup. Install curl or implement the check using Python's standard-library HTTP client. This was verified from image/build declarations; no container was started. It is separate from the empty database-metadata initialization in finding 5.

**27. P2 — Restarting during clarification loses the question target and retry count.**

Locations: [context_memory.py:174](C:/Multilingual_LFS_Project/backend/agents/context_memory.py:174), [context_memory.py:331](C:/Multilingual_LFS_Project/backend/agents/context_memory.py:331), [survey_routes.py:1651](C:/Multilingual_LFS_Project/backend/api/survey_routes.py:1651).

The serializer and restoration preserve `state='clarifying'` but omit `clarification_target` and `clarification_count`. Original-method verification saved a conversation clarifying employment status with count 2. After restoration, the target was `None` and count was 0. Five further `not sure` answers left the state clarifying with count 5 and no collected data, because the configured fallback requires a target. Persist and restore the FSM fields needed to resume clarification correctly.

**28. P2 — Multiple backend workers overwrite answers from stale cached contexts.**

Locations: [survey_routes.py:1647](C:/Multilingual_LFS_Project/backend/api/survey_routes.py:1647), [survey_routes.py:1662](C:/Multilingual_LFS_Project/backend/api/survey_routes.py:1662), [survey_routes.py:1394](C:/Multilingual_LFS_Project/backend/api/survey_routes.py:1394).

Redis is consulted only on a worker's first request for a session. Later requests reuse that worker's cached snapshot and overwrite Redis with it. A sequential A → B → A reproduction saved employment status on A, then education `bachelor` on B, which requested field of study. A reused its earlier snapshot and interpreted `Mechanical Engineering` as education, overwriting the bachelor answer and removing the field-of-study path. Refresh/version context between turns and serialize session updates. This affects deployments using multiple processes; the default launch uses one worker. Verification used the original methods and an in-memory Redis substitute.

Validation completed:

- **Continued pass: 2,579 tests passed, one slow test deselected, two warnings, in 232.82 seconds.** This ran `backend/tests` and `eval` with the repository's default non-slow selection. [JUnit test results](C:/Multilingual_LFS_Project/Documentation/CODE_REVIEW_2026-10-03_TEST_RESULTS.xml) preserve the per-test results. The process disabled `.env` loading and live service connections, used test-only database/JWT/provider settings, disabled real Redis access, and replaced application startup warmup with an empty lifespan. Local mock-server transport tests still ran. Initial isolation settings conflicted with some tests' Qdrant/provider mocks; those settings were corrected before the successful complete run.
- **194 focused existing tests passed** across `test_conversation_manager.py`, `test_conversation_manager_extended.py`, `test_orchestration_correctness.py`, `test_analyze.py`, `test_analyze_official_tier1.py`, and `test_analyze_wisco_tier1.py`. Orchestration was rerun after correcting the isolated runner's JWT and route fast-mode settings; test-setup failures are not reported as application defects.
- Hindi/Urdu routing and wage extraction were reproduced by executing the original source methods through AST isolation, without replacing their logic.
- Classification refinement, title correction, and report aggregation were exercised with an in-memory SQLite database and mocked AI services.
- Supervisor authorization and deleted-session prefill were checked against actual route functions with an isolated database.
- Evaluation findings were checked with the real analyzer and committed CSV data; frontend field/condition mismatches were checked with Node expressions extracted from the source.
- The continued pass reproduced negated confirmation, semantic-review persistence, stale reports, no-work-history quality scoring, language-change requests, chat remounts, reference-label integrity, corrective-retry telemetry, and gap-aware confidence using the original methods with isolated dependencies.
- The resumed pass reproduced wage-band parsing and both persistence failures using original methods with isolated dependencies. Deployment findings were verified from local build inputs and source declarations. No application changes were made, so the previously saved 2,579-test run was retained rather than rerun.

The slow test, real application startup warmup, browser workflows, a fresh Docker deployment, and live LLM accuracy were not run. The findings above are a focused review, not a certification that every other project path is correct. Existing survey data and external messaging services were not used for the checks. Application source and the existing thesis edits under `Documentation/LaTeX/thesis` were left unchanged.
