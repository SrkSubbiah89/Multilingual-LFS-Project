# Parent-document RAG improvement — 2026-10-06

The local application now uses **parent-document ISCO RAG** inside the existing CrewAI classification workflow. On the 18,747-case WISCO reference benchmark it achieved **38.83% (7,279 correct)**, compared with **32.55% (6,102 correct)** for dense retrieval using the same official enriched catalogue and E5-small encoder: **+6.28 percentage points**, or 1,177 additional correct codes.

This is a controlled comparison on historically reused occupation-title data. The enriched catalogue was previously designed in response to historical held-out errors. These results do not establish accuracy on fresh respondent interviews, duties descriptions, or the Labour Force Survey population. The historical 21.19% flat / 10.35% strict-hierarchy experiment remains unchanged in the report; it used a different catalogue configuration. The historical intended E5-large result is a separate experiment and is not surpassed by this result; its executed encoder provenance remains unresolved.

The [accuracy reconciliation and follow-up](RAG_ACCURACY_AUDIT_2026-10-06.md) confirms that the active 38.83% result improves the unchanged 32.55% dense baseline but remains 2.12 percentage points below the historical 40.95% output. It records the encoder metadata conflict, corrects the report labels, and preserves two development-only follow-up experiments that did not improve accuracy.

## Measured results

| Split | Cases | Same-profile dense | Parent-document RAG | Difference |
|---|---:|---:|---:|---:|
| Development, parameter selection | 1,371 | 438 / 31.95% | 523 / 38.15% | +6.20 pp |
| Validation, frozen parameters | 642 | 209 / 32.55% | 236 / 36.76% | +4.21 pp |
| Historical held-out, frozen parameters | 18,747 | 6,102 / 32.55% | 7,279 / 38.83% | +6.28 pp |

On the historical held-out cases, top-three candidate coverage increased from **48.93% to 54.56%** and top-five coverage from **55.94% to 60.90%**. Case-level Wilson intervals for top-one accuracy are [31.88%, 33.22%] for dense and [38.13%, 39.53%] for parent-document retrieval. Multilingual titles share source families, so these case-level intervals are descriptive, not independent-respondent uncertainty.

The paired comparison contains 1,897 cases corrected by parent-document retrieval and 720 cases where dense retrieval was correct but the new method was wrong. A deterministic 2,000-replicate bootstrap over 3,808 occupation/title clusters gives a descriptive difference interval of **[+5.59, +6.96] percentage points**. Clusters combine WISCO occupation keys and keys sharing an identical normalized retained title within one language. The benchmark reuse limits still apply.

| Input language | Historical cases | Dense accuracy | Parent-document accuracy |
|---|---:|---:|---:|
| English | 3,818 | 54.85% | 63.91% |
| Arabic | 3,762 | 25.15% | 27.94% |
| Urdu | 3,608 | 25.19% | 31.60% |
| Hindi | 3,793 | 33.80% | 40.18% |
| Tagalog | 3,766 | 23.13% | 29.85% |

Validation is smaller and not uniformly improved: Arabic validation decreased from **36/128 (28.13%) to 30/128 (23.44%)**. Both splits and their language-specific counts are retained in the frontend comparison and [aggregate evidence](RAG_IMPROVEMENT_2026-10-06_RESULTS.json).

## Retrieval and framework integration

The independent official ILO catalogue produces **2,833 child fragments**: occupation titles, 120-word definition passages, and included occupation examples. Each fragment retains its authoritative parent code. The index contains official text only; benchmark titles, translations, labels, and respondent answers were not added.

For each query, the method scores every one of the **436 official unit groups**. It combines the best child cosine and full parent-definition cosine with the development-selected weight **0.5**, using maximum child aggregation. No early major-group decision removes candidate occupations. This is sentence-embedding parent-document retrieval, rather than ColBERT token-level late interaction or an executed four-stage traversal. Taxonomy prefixes are displayed as an ISCO code hierarchy.

The live wrapper queries Qdrant with exact search for all parents and groups children by their unit code. It checks complete code coverage, official payloads, vector hashes, encoder weights, and the frozen selection snapshot. Missing or mismatched evidence fails explicitly. A lock prevents concurrent initialization from loading duplicate encoders on this machine.

Scores are uncalibrated similarity blends. Their original values are retained, optional LLM reranking does not execute for this method, and classifications require human review. Existing HITL correction/rejection, semantic plausibility checks, and the three-specialist CrewAI evidence-auditor workflow remain active. The measured method uses the occupation input text; the separate context/duties argument is not used. The official catalogue provides English labels, so the new result does not invent an Arabic catalogue label.

The session-completion fallback now also persists a required occupation review, including high-similarity results for current or previous jobs. Repeated completion checks retain a single pending entry. A focused 115-test review/routing run passed after this edge-case fix; the final full suite is recorded separately.

The local `.env` selects `ISCO_RETRIEVAL_STRATEGY=parent_document`, with full local AI and CrewAI enabled. The shipped example defaults to `legacy` so a clean installation does not claim an index that has not been built. The parent index is a new content-addressed Qdrant collection; existing collections and respondent records were not rewritten.

## Selection and provenance

Basic Unicode BM25 plus dense reciprocal-rank fusion, soft hierarchy, and offline hierarchy beams were tried first on development data. They did not outperform dense retrieval. The best RRF configuration chose zero lexical weight, making it identical to dense retrieval; that is not reported as a hybrid improvement. Those negative experiments remain in the aggregate evidence.

Parent-document selection searched eight development-only configurations: child weights 0.25, 0.5, 0.75, 1.0 with maximum or top-two-mean aggregation. It selected 0.5 / maximum before validation and historical held-out evaluation. A subsequent independent review strengthened selection integrity guards, including development-report verification and dependent source hashes. Re-evaluation preserved every prediction CSV byte-for-byte across all three splits; no retrieval algorithm, search grid, or selected parameter was changed after viewing held-out results.

The pinned encoder is `intfloat/multilingual-e5-small`, revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`, normalized 384-dimensional vectors, with E5 query/passage prefixes. All 436 stored enriched parent vectors were re-encoded and matched the recorded query encoder with minimum cosine **0.9999998808**. Historical catalogue metadata did not record its original weights revision; this reproduction check establishes compatibility for the vectors used in this comparison without inventing historical provenance.

The original parent-evaluator reports inherited BM25/RRF descriptive metadata from a shared baseline helper, although those operations did not execute in the parent method. The aggregate corrects those curated labels to a cosine child/parent blend and marks RRF fields inapplicable. It also retains the original frozen reports for source-hash verification; predictions and numerical results are unchanged.

The [runtime configuration](../backend/rag/parent_isco_config.json) binds to the [exact frozen development selection](../backend/rag/parent_isco_selection.json), SHA-256 `632598fd8a70cdabfef3841ef871573dbe6e3ae53dcacf3a2009e294973563ef`. Raw generated cases, query/index vector caches and per-case prediction CSVs remain in ignored local directories. Committed aggregate evidence retains their hashes, metrics, source provenance, and integrity comparisons.

## Verification

- Read-only Qdrant serving verification: **50/50** sampled cases across five languages returned the exact offline top-five code sequence, with all 436 codes scored.
- Production frontend build completed; **31 frontend tests passed**.
- **12 Chromium workflow checks passed** using actual routes, isolated disposable databases, controlled AI and intercepted OTP delivery.
- **Six additional Chromium checks passed** for the new comparison card across five languages, RTL/LTR layout, mobile width, and absence of JavaScript errors. [Browser evidence](RAG_IMPROVEMENT_2026-10-06_BROWSER_EVIDENCE_RESULTS.json).
- The actual restarted backend returned the new method and required-review flag for five-language read-only probes. Database counts were unchanged between the before/after activation checks; migration head remained `d841b2079c65`. [Runtime evidence](RAG_IMPROVEMENT_2026-10-06_RUNTIME_RESULTS.json).
- Actual local CrewAI specialists and evidence auditor completed with `ollama/qwen2.5:3b` in **67.305 seconds**, with the occupation tool invoking the newly activated retriever through its read-only endpoint. This uses one synthetic English profile and verifies cooperation, not a stored interview or multilingual population accuracy. [Native cooperation evidence](RAG_IMPROVEMENT_2026-10-06_LIVE_CREW_RESULTS.json).

The final backend/evaluation regression results are recorded in [JUnit evidence](RAG_IMPROVEMENT_2026-10-06_TEST_RESULTS.xml). No cloud inference, paid API call, actual OTP delivery, or respondent write was needed for this work.

The final full regression run passed **3,193 backend/evaluation tests**, with one slow test deselected, in **224.85 seconds**. Its 38 warnings comprise one pytest plugin warning, one intentionally short JWT test-key warning, and 36 installed CrewAI deprecation warnings. The reviewed source is running locally at ports 8000/3000, and the final [readiness/count check](RAG_IMPROVEMENT_2026-10-06_FINAL_RUNTIME_RESULTS.json) confirms that the respondent table counts remained unchanged throughout this activation.

## Reproduction and activation

Use the existing official catalogue acquisition/normalization tooling and WISCO v3 split builder to populate the ignored source directories. The evaluator requires the measured enriched profile and frozen query/index identities; it aborts on incompatible data or changed selected sources. Install the declared project dependencies and cache the pinned E5-small revision locally.

1. Run `scripts/cache_rag_query_embeddings.py --input <cases.csv> --output <queries.npz>` for development/validation and held-out cases. Only query-prefixed input text enters the encoder; IDs and language identify cached records. Gold labels are not model inputs.
2. Run `eval/compare_hybrid_isco.py snapshot --profile official_ilo2021_v1_enriched --output <catalogue.npz>` to inspect and cache the existing official Qdrant profile without writing it.
3. Run `eval/compare_parent_document_isco.py build --output <fragments.npz> --encoder-metadata <queries.meta.json> --catalogue <catalogue.npz>` to create official child vectors and verify parent-vector compatibility.
4. Run its `select` command on development cases; use the resulting selection with `evaluate --selection <selected_config.json> --split validation` and `evaluate --selection <selected_config.json> --split heldout`. Each run requires `--cases`, `--queries`, `--fragments`, `--catalogue`, and a new `--output` directory. Selection hashes freeze exact source bytes, including line endings.
5. Run `scripts/publish_parent_isco_index.py --cache <fragments.npz> --source <official-enriched.csv> --manifest <new-private-manifest.json>`. It writes a new local content-addressed collection and refuses to overwrite an existing collection. A partially failed publish is retained for inspection.
6. Bind the runtime configuration/selection to the verified index and encoder hashes, set `ISCO_RETRIEVAL_STRATEGY=parent_document`, and restart the backend. Verify `/ready`, the read-only debug method, and `scripts/check_parent_isco_runtime_equivalence.py` before interpreting any live result as this evaluated configuration.

Reproduction uses locally generated artifacts and exact provenance checks. A different encoder, catalogue, fragment construction, retrieval weight, or source version requires a new development selection and evaluation.

The next empirical activity is a fresh, expert-labeled multilingual title-and-duties evaluation, followed by calibration of review/clarification thresholds. The present accuracy gain improves the local reference result; it is not a claim of perfect classification.
