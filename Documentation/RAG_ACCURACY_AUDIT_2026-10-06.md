# Accuracy reconciliation and retrieval follow-up — 6 October 2026

The local application still serves **parent-document ISCO RAG, measured at 38.83%** on the historically reused WISCO reference split. **32.55% is its dense comparison baseline**, not the active method's measured accuracy. The earlier **40.95%** output is numerically higher, but its executed encoder cannot be established from the saved run metadata.

The report now distinguishes these three results in all five interface languages. The historical percentage was removed from the individual survey classification card. Benchmark percentages describe aggregate reference results; they are not the confidence or field accuracy of an individual survey classification.

## Reconciled results

All 18,747 case IDs, input texts, languages and gold codes match across the historical and current prediction files. Every current dense prediction matches the earlier enriched E5-small dense run: **zero prediction differences**. This is not a regression of that baseline.

| Result | Exact codes | Accuracy | Identity/status |
|---|---:|---:|---|
| Historical intended E5-large profile | 7,676 / 18,747 | 40.95% | Executed encoder unresolved; offline reference |
| Enriched dense E5-small baseline | 6,102 / 18,747 | 32.55% | Unchanged comparator |
| Parent-document E5-small RAG | 7,279 / 18,747 | 38.83% | Active local method |

Parent RAG improves the same-encoder baseline by **1,177 correct codes / 6.28 percentage points**. It remains **397 correct codes / 2.12 points below** the historical 40.95% output. That gap is real in the saved predictions; it does not establish an encoder-specific cause.

The historical CSV reports `intfloat/multilingual-e5-small` while its method/profile label intends E5-large. The October 4 [provenance correction](../eval/results/corrections_20261004/raw_runs/enriched_e5large_heldout_20260824/20260824T123941Z_flat.provenance_correction.json) explicitly sets the executed model to null and records `historical_execution_identity_unresolved`. Neither the profile name nor the old configuration hash proves which encoder ran. The score's arithmetic remains valid.

The current parent result binds to the frozen selection, serving configuration, encoder revision and weights, and official catalogue. Its existing live verification is supplemented by the new read-only runtime check. The [final history audit](RAG_ACCURACY_AUDIT_2026-10-06_HISTORY_FINAL_RESULTS.json) records hashes, current identity, per-language counts and paired comparisons without publishing raw cases.

| Language | Cases | Historical intended-large output | Current dense baseline | Active parent RAG |
|---|---:|---:|---:|---:|
| English | 3,818 | 56.91% | 54.85% | 63.91% |
| Arabic | 3,762 | 37.77% | 25.15% | 27.94% |
| Urdu | 3,608 | 35.53% | 25.19% | 31.60% |
| Hindi | 3,793 | 40.89% | 33.80% | 40.18% |
| Tagalog | 3,766 | 33.17% | 23.13% | 29.85% |

These are reused occupation-title cases. Catalogue enrichment was previously informed by historical held-out errors. The comparisons do not establish accuracy on fresh Labour Force Survey interviews or title-and-duties descriptions. The smaller Arabic validation decline documented in the original parent comparison is preserved in the interface.

## Bounded development experiments

Two alternatives used the existing pinned E5-small query and official catalogue vectors. Neither loaded a new encoder, indexed benchmark text, tuned language-specific parameters or changed production retrieval. Their fixed grids were evaluated only on the 1,371 development cases.

| Experiment | Configurations | Best new alternative | Original parent control | Decision |
|---|---:|---:|---:|---|
| Catalogue-only density correction | 20, including zero controls | 518 / 37.78% | 523 / 38.15% | Selected correction strength zero |
| Fragment-kind balancing | 10 weight sets, plus shared max control | 497 / 36.25% | 523 / 38.15% | Retained maximum pooling |
| Count-normalized log-mean-exp pooling | 4 temperatures, plus shared max control | 491 / 35.81% | 523 / 38.15% | Retained maximum pooling |

Density correction subtracts a parent popularity offset derived from independent official title vectors, excluding the same parent. These title references are passage-prefixed; this is an approximation, not standard cross-domain similarity local scaling with query-prefixed references. Kind balancing gives title, definition and included-example evidence separate global weights, renormalizing for missing kinds. Log-mean-exp uses a stable calculation with child-count normalization. Every occupation remains a candidate.

Neither selected a new configuration. No validation or held-out evaluation was run for these alternatives, and no new accuracy improvement is claimed. The density integrity review strengthened guards before any such evaluation; re-running development preserved its prediction CSV byte-for-byte. The pooling control matched the original parent development top-five sequence for every case.

The [experiment evidence](RAG_ACCURACY_AUDIT_2026-10-06_EXPERIMENT_RESULTS.json) preserves both grids, all outcomes, selected controls, source/cache hashes and frozen selection identities. Raw per-case outputs remain in ignored local experiment directories. The new experiment helpers are available for reproduction; the live classifier continues using the previously validated parent method.

## Review and verification

Independent review checked the arithmetic, identical-case alignment, catalogue-only reference construction, pooling algebra and selection integrity. Fixes reject mismatched development reports, incompatible parent aggregation, overlap with either method's development IDs, and invalid encoder compatibility evidence. The history audit now rejects blank or malformed gold codes, preserves leading-zero codes and abstentions, derives percentages from counts, and binds current predictions to the frozen serving identity.

Verification evidence:

- [Final backend/evaluation regressions](RAG_ACCURACY_AUDIT_2026-10-06_FINAL_REVIEW_TEST_RESULTS.xml): **229 passed**, including existing parent retrieval/serving and all 17 final history-audit tests. One existing pytest plugin warning was emitted.
- Frontend: **33 tests passed**; the production build passed and was restarted locally.
- [Chromium report checks](RAG_ACCURACY_AUDIT_2026-10-06_BROWSER_RESULTS.json): **6 passed**, covering all five languages, active/baseline roles, historical uncertainty, RTL/LTR, mobile width and absence of JavaScript errors.
- [Read-only runtime verification](RAG_ACCURACY_AUDIT_2026-10-06_RUNTIME_RESULTS.json): backend ready, frontend HTTP 200, all five occupation probes return `isco_parent_document_rag` with human review required; database counts unchanged. Disposable browser databases were removed.

No respondent changes, actual OTP delivery, cloud inference or public deployment were needed. The earlier 3,193-test full regression result remains separate evidence in [the parent improvement report](RAG_IMPROVEMENT_2026-10-06.md).

## Reproduction and next accuracy measurement

Run the history audit with `py -3.11 scripts/audit_isco_accuracy_history.py --predictions <current-predictions.csv> --comparison-report <current-heldout-report.json> --output <new-audit.json>`. It verifies the original historical sources and correction, pairs identical cases, and checks current counts and frozen serving identity.

The experiment CLIs are `eval/compare_isco_density.py` and `eval/compare_balanced_pooling_isco.py`. Their `select` command requires development `--cases`, cached `--queries`, official `--fragments`, `--catalogue`, `--parent-selection`, and a new `--output` directory. A changed encoder/catalogue/source needs a new controlled selection. `evaluate` additionally requires the selected snapshot and explicit split, checks integrity before evaluation inputs, and never searches evaluation labels for a better configuration.

A reliable improvement above the current result needs a verified encoder/candidate reranker experiment and a fresh expert-labeled multilingual title-and-duties evaluation. Historical attempts to run the larger encoder on this local machine failed under its memory constraints; this follow-up did not retry or assume success. A successful larger-encoder run must establish its actual weights, matching query/catalogue vectors and new measured accuracy. It cannot inherit the historical 40.95% claim.
