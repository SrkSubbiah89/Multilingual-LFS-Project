# Occupation duties routing and retrieval follow-up — 2026-10-06

The survey now uses explicitly supplied **current-job duties** in occupation retrieval across the direct classifier, CrewAI tool, later duties corrections, and completion fallback. Previously, the active parent-document classifier ignored the auxiliary context containing duties, while the completion fallback separately appended them. A shared adapter now makes those paths consistent.

The existing title-only benchmark remains **38.83% (7,279/18,747)** for the active parent-document RAG, compared with **32.55% (6,102/18,747)** for same-profile dense retrieval. A catalogue-alignment experiment reached **39.16%**, but is retained as an experiment because Tagalog top-one performance decreased. No new field accuracy or perfect classification is claimed.

## Implemented behavior

[`occupation_inputs.py`](../backend/agents/occupation_inputs.py) composes the original title and explicit duties once before retrieval. The adapter preserves native-script input, bounds duties to 1,600 characters with a diagnostic truncation flag, and ignores known missing values, repeated duties, or duties with no letters in the retained text. It does not determine whether alphabetic text is a meaningful duties description. When duties are absent or filtered, the original title-only invocation and result object remain unchanged.

The title and duties stay separate in respondent records. Current-job duties and industry context do not enter previous-job classification. Reclassifying corrected duties retains the existing response revision and human-review lifecycle. Retrieval scores, alternatives, and human-review requirements are preserved; cosine scores are not calibrated correctness probabilities.

Combined parent retrieval uses the distinct method `isco_parent_document_duties_rag`, with `duties_accuracy_evaluated=false` in optional diagnostics. The frozen title-only parent selection, child catalogue, encoder, and ranking implementation are unchanged. This fixes how evidence reaches the classifier; it does not extend the title-only benchmark to duties descriptions. ISCO groups jobs by tasks and duties, which supports using this information when respondents provide it. [ILO ISCO-08](https://isco.ilo.org/en/isco-08/).

The read-only debug endpoint and local native CrewAI verification bridge support the same explicit duties input. Industry metadata cannot substitute for duties in this bridge. Diagnostic traces can contain supplied text and are intended for local verification.

## Measured retrieval comparison

The alternative fits a regularized transform using only the 436 paired official catalogue vectors: verified 384-dimensional small-encoder vectors and the existing stored 1,024-dimensional parent vectors. It projects small-encoder queries into the stored geometry and blends that evidence with the frozen parent retrieval. This does **not** run a large query encoder. The historical stored large vectors' encoder revision and weight identity remain unverified.

Seven configurations were declared before development evaluation. Development selected ridge `0.1` and projected-parent blend `0.5`; that selection, source hashes, and cache identities were frozen before validation and the historical held-out run. No query or benchmark label was used to fit the transform. Validation and historical held-out evaluation ran only the selected configuration; no subsequent retuning occurred.

| Split | Cases | Frozen parent RAG | Catalogue-aligned experiment | Difference |
|---|---:|---:|---:|---:|
| Development | 1,371 | 523 / 38.15% | 524 / 38.22% | +1 case / +0.07 pp |
| Validation | 642 | 236 / 36.76% | 239 / 37.23% | +3 cases / +0.47 pp |
| Historical held-out | 18,747 | 7,279 / 38.83% | 7,342 / 39.16% | +63 cases / +0.34 pp |

The held-out paired comparison has 402 gains and 339 losses; exact two-sided **case-level** McNemar p = 0.0226847880. Translations and repeated occupation-title families are dependent, so this case-level value is descriptive rather than independent-respondent evidence. The enriched catalogue was previously informed by held-out errors, so freezing this comparison does not establish unbiased generalization. The historically reused WISCO benchmark also does not constitute new Labour Force Survey field validation.

| Language | Historical cases | Change in correct top-one codes | Top-three change | Top-five change |
|---|---:|---:|---:|---:|
| Arabic | 3,762 | +30 | +23 | +30 |
| English | 3,818 | +19 | +6 | +13 |
| Hindi | 3,793 | +24 | −18 | +18 |
| Tagalog | 3,766 | −12 | +2 | +15 |
| Urdu | 3,608 | +2 | −7 | +24 |

Before held-out evaluation, the promotion gate required global improvement, case-level p < 0.05, and no language regression. Tagalog top-one regression fails that gate; Hindi and Urdu top-three coverage also declined. The application therefore keeps the existing parent retrieval. Every control top-five sequence matched the prior frozen baseline on development, validation, and held-out cases.

The [aggregate experiment evidence](RAG_CATALOGUE_ALIGNMENT_2026-10-06_RESULTS.json) preserves all split metrics, paired counts, the promotion decision, selection identity, and source/cache hashes. Full predictions and differences remain in ignored local experiment directories. Earlier **40.95%** historical output remains separate evidence with unresolved executed-encoder provenance; see the [accuracy audit](RAG_ACCURACY_AUDIT_2026-10-06.md).

## Verification and practical limits

The developer-authored duties fixture was frozen before observing predictions. The [live diagnostics](RAG_DUTIES_2026-10-06_DIAGNOSTIC_RESULTS.json) made 35 read-only requests across 25 cases in five scenario families, translated into English, Arabic, Urdu, Hindi, and Tagalog. All mandatory routing, native-text, official-candidate, human-review, and empty-duties parity checks passed. **18/25** semantic expectations matched and **7 misses remain recorded**. These expectations were not independently expert or native-speaker validated and must not be reported as field accuracy. Neither the fixture nor its wording was revised to hide misses.

The [native CrewAI run](RAG_DUTIES_2026-10-06_LIVE_CREW_RESULTS.json) completed in **78.097 seconds** using CrewAI 1.9.3 and local `qwen2.5:3b`. All three specialist tools and the evidence auditor executed with no fallback. Occupation retrieval returned `2512`, used explicit duties, and required human review. This was one synthetic English profile through the read-only bridge, without survey persistence, cloud inference, or external messages.

The [full isolated backend/evaluation suite](RAG_DUTIES_2026-10-06_TEST_RESULTS.xml) passed **3,439 tests** in 397.52 seconds. One slow test was deselected by the repository's existing `pytest.ini` configuration. After the final retained-prefix guard refinement, [47 final input checks](RAG_DUTIES_2026-10-06_FINAL_INPUT_TEST_RESULTS.xml) passed separately. The isolated runner uses in-memory SQLite, disables dotenv loading and startup warmup, and blocks production database/infrastructure and remote network access.

The [final local runtime evidence](RAG_DUTIES_2026-10-06_RUNTIME_RESULTS.json) records backend readiness, frontend HTTP 200, five title-only language probes, and three live guard-parity probes after restart. The actual local application also received separate authentication and session-creation requests during the observation window. Database row counts therefore changed; the evidence preserves the deltas without claiming equality or establishing the origin of those requests. Verification used read-only SQL and synthetic GETs, with no actual authentication, survey persistence, or external-message requests. PostgreSQL, Redis, and Qdrant remain the original three local containers; Qdrant retains 44 collections.

Further accuracy measurement needs a fresh expert-labeled multilingual title-and-duties sample. The recorded synthetic misses and language regressions are reasons to collect that evidence rather than claim that duties or catalogue alignment solve all classification errors.
