# Project title alignment — 6 October 2026

Project title: **Multilingual Conversational AI for Labour Force Surveys: A Multi-Agent RAG System with CrewAI Framework**.

This audit began on 5 October against application commit `db734cb` and continued on 6 October. It checks executable paths rather than inferring implementation from class names or diagrams. Earlier thesis edits and historical evaluation artifacts remain separate.

## Implementation against the title

| Title component | Executable implementation | Scope |
| --- | --- | --- |
| Multilingual | `LanguageProcessor`, five-language `ConversationManager` prompts/templates, localized canonical survey options and UI | English, Arabic MSA/Gulf, Urdu, Hindi, Tagalog. NER uses original respondent wording; normalization supplies supplementary context. |
| Conversational AI | CrewAI language/conversation tasks around a persisted interview state machine; clarification, correction and resume | Explicit fast mode supplies deterministic responses. Full AI mode executes configured models; state and routing remain validated application logic. |
| Labour force surveys | Project-adapted sections A–K and employment/unemployment/outside-labour-force paths; answer revisions, quality review and reports | The application is a research survey implementation. A title does not establish official questionnaire approval or empirical usability. |
| Multi-agent | Live `SurveyClassificationCrew`: requested occupation/industry/education specialists plus an evidence auditor, with task context passed to the auditor | At least two agents for one requested dimension, up to four for all three. Tools hold exact inputs; original classifier objects determine saved answers and review requirements. |
| RAG | Live ISCO hierarchical Qdrant retrieval with multilingual embeddings and optional candidate-grounded CrewAI reranking | Four legacy ISCO collections contain 10/43/130/436 groups. Other profiles and standard-specific retrieval methods are available; live ISIC/ISCED defaults remain keyword/rule paths with subset coverage. |
| CrewAI framework | Actual installed CrewAI 1.9.3 `Agent`, `Task`, `Crew` and sequential process in the live classification route | Sequential collaboration is a supported CrewAI process. The experimental hierarchical manager coordinator is separate from the hierarchical retrieval algorithm. |

The configured crew requires specialist tool calls and an audit tool call before accepting completion. Model-generated replacement codes, confidence, or human-review decisions never become authoritative. Completed results survive later failures; only unfinished dimensions enter direct fallback. API metadata exposes whether cooperation actually occurred.

## Gaps fixed

- Connected a shared classification crew to the live survey. It runs only when coding inputs change and preserves full classifier confidence, hierarchy, alternatives and HITL decisions. Previous occupations for unemployed and outside-labour-force respondents use the same workflow; never-worked sentinels are excluded.
- Hardened the experimental manager coordinator against fabricated or partial output, malformed JSON, invalid ISCED levels, and occupation-only delegation. A manager result must match actual specialist-tool evidence.
- Corrected multilingual goals, task expected-output languages, validation language labels and original-text NER offsets. Hindi segment labels, natural localized qualification answers and native-language correction labels now follow the implemented survey gates.
- Moved detected-language selection before generating the current reply. An explicit language preference retains precedence.
- Kept ambiguous routing answers in clarification after repeated attempts, while retaining supported refusals and open-text fallback. Localized quick options now remain available during clarification.
- Fixed prior-occupation semantic checks and code promotion, prevented an old code being replayed after a failed title correction, and forwarded classified industry/attainment values to the PersonRegister on completion.
- Deduplicated and bounded query decomposition so repeated phrases do not manufacture repeated retrieval evidence. Retrieval explanations retain sources when generation is unavailable.
- Replaced guessed UI agent activity and a hard-coded Claude claim with returned component execution status. Cached, skipped and failed work are shown as reported.
- Made chat review badges reflect pending occupation reviews, including high-confidence semantic escalations. Completed human decisions remove the pending badge. Rejected reports retain their human-decision badge when the ISCO code is cleared.
- Added opt-in local-only inference so full local AI can run without falling back to configured cloud providers. Explicit evaluation model pins retain their original contract.
- Corrected README architecture and unsupported confidence-adjustment claims. The semantic coherence score remains a separate project heuristic.

## Evidence and verification

Read-only [catalogue inspection](TITLE_ALIGNMENT_2026-10-05_CATALOGUE_RESULTS.json) records all 43 existing Qdrant collections, counts and vector dimensions. Counts do not prove encoder identity, catalogue completeness beyond the checked source sets, or classification accuracy.

The [initial local-model run](TITLE_ALIGNMENT_2026-10-05_LIVE_CREW_INITIAL_RESULTS.json) omitted required tools. The [guarded follow-up](TITLE_ALIGNMENT_2026-10-05_LIVE_CREW_RESULTS.json) retained classifications through direct fallback but did not complete the evidence audit and took 1,282.733 seconds. Both failures are retained. They motivated explicit tool-completion checks and a bounded local tool-calling path; they are not successful collaboration evidence.

The [successful native local-model run](TITLE_ALIGNMENT_2026-10-06_LIVE_CREW_RESULTS.json) completed in **84.98 seconds** with actual CrewAI 1.9.3 and `ollama/qwen2.5:3b`. All three specialist tools and the evidence auditor executed; cooperation was verified, with no fallback, failed dimensions, blocked network requests or runtime logging observations. It used one synthetic English profile. Occupation evidence came from the existing backend's read-only semantic debug endpoint, which did not forward context/language or request LLM reranking. This verifies actual collaboration rather than full interview persistence or multilingual classification accuracy.

- **2,966 backend/evaluation tests passed**, one slow test deselected, in **224.62 seconds**. The final run includes the pending-review badge fix. [JUnit results](TITLE_ALIGNMENT_2026-10-06_TEST_RESULTS.xml). Its 38 warnings comprise one known pytest plugin warning, one deliberately short JWT test key warning, and 36 installed CrewAI deprecation warnings.
- **18 frontend tests passed**; the final production build generated seven routes. The report-card regressions render actual JSX and translated English/Arabic rejection labels with the existing Next compiler.
- **12 Chromium workflow checks passed**, with no JavaScript errors or unexpected requests. [Workflow results](TITLE_ALIGNMENT_2026-10-06_BROWSER_RESULTS.json). These cover seeded authentication, five-language controls, canonical Hindi answers, resume, phone controls, education correction, supervisor restrictions, human correction and reports.
- **Two additional Chromium rejection checks passed** in English and Arabic after the final UI rebuild. Actual isolated API rejection produced a null ISCO code and a rejected human status; both LTR/RTL reports retained the decision badge. [Rejection results](TITLE_ALIGNMENT_2026-10-06_BROWSER_REJECTION_RESULTS.json).

Browser checks use real application routes and the production frontend with task-owned disposable PostgreSQL/Redis containers, synthetic accounts/answers and controlled AI outputs. They establish application behavior independently of the native model probe. OTP delivery is intercepted; no real message is sent.

Independent code and documentation reviews found and resolved the routing, persistence, review-state and stale-architecture issues above. The final code review found no remaining blocking defect within the reviewed changes.

## Local activation

The local configuration selects `LFS_FAST_MODE=false`, `ENABLE_SURVEY_CLASSIFICATION_CREW=true`, `LFS_LOCAL_ONLY=true`, and `OLLAMA_MODEL=qwen2.5:3b`. CrewAI remote tracing and telemetry are disabled. The local-only application factory uses the configured installed Ollama model for unpinned general and critical inference; explicitly pinned evaluations retain their original provider contract.

The rebuilt production frontend is at **http://127.0.0.1:3000**, compiled against **http://localhost:8000**. Its existing tunnel environment file remains available for a separate tunnel build. The backend is restarted from the reviewed source at **http://127.0.0.1:8000**.

Final readiness, supervisor access, database counts and disposable-service cleanup are recorded in [runtime evidence](TITLE_ALIGNMENT_2026-10-06_RUNTIME_RESULTS.json). New records exist since the earlier activation; the evidence records those increases without assuming their source. No table count decreased, and counts remained unchanged during final read-only verification. The authorized supervisor grant remains existing active user ID 5. Runtime checks do not conduct a real respondent interview or alter those records.

## Runtime and evidence limits

Fast mode explicitly bypasses collaborative generation. A real framework execution, a passing routing regression and a classification accuracy benchmark establish different facts. The live default's legacy retrieval profile cannot inherit accuracy numbers from another evaluated profile. Historical encoder provenance remains unresolved where the earlier review recorded it. ISIC/ISCED-F coverage is a subset; report narratives and auxiliary support remain EN/AR although their UI labels and interviews support five languages.

Human usability, expert-reviewed classification accuracy and production load performance require separate empirical evaluation. This audit does not invent those results or alter historical respondent answers.
