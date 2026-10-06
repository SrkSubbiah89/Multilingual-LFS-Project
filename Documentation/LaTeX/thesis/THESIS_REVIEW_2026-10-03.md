# Thesis review and gap report — 3 October 2026

> **Status note added 7 October 2026 — read before citing the gap table below.**
> This document records the state on 3 October and is kept unchanged as the
> historical record. Several gaps it lists as open have since been closed, so
> the table must not be read as the current state:
>
> - **All four P1 application defects are repaired and regression-tested** —
>   reviewer authorization, status/wage extraction, fresh-database startup, and
>   the top-3 evaluation denominator (corrected to 61/130 = 46.92%). Evidence
>   per finding: [CODE_FIXES_2026-10-05.md](../../CODE_FIXES_2026-10-05.md).
>   In particular, the "ordinary respondents can access and modify other
>   respondents' HITL records" entry **no longer describes the system**;
>   both HITL endpoints now require an explicitly configured active reviewer.
> - **Historical execution provenance** was investigated rather than closed:
>   the enriched-large run's executed encoder is formally unresolved, recorded
>   in committed provenance corrections and now stated in the thesis itself.
> - **The served retrieval configuration changed** on 6 October to
>   fragment-to-parent retrieval (38.83%), reported in thesis Section 6.1.4
>   with its own verification; see
>   [RAG_ACCURACY_AUDIT_2026-10-06.md](../../RAG_ACCURACY_AUDIT_2026-10-06.md).
>
> Gaps that remain genuinely open: independent final benchmark on untouched
> data, real multi-standard labels and coverage, SRE validity and calibration,
> conversational language quality, human outcomes, and production runtime
> evidence.

**Outcome:** all seven active chapters, the abstract, bibliography, architecture diagram, and appendices were revised for academic accuracy, traceability, and consistency with the actual implementation and saved experiments. A results graph and experiment-provenance appendix were added. The thesis remains a prototype evaluation with disclosed limitations; this review is not a certification of perfection, institutional approval, or international survey compliance.

Main deliverable: [reviewed thesis PDF](thesis_reviewed.pdf). Build instructions: [README](README.md). Primary-reference review: [citation audit](CITATION_AUDIT.md). Numerical evidence: [results verification](results_verification.json).

The separate [code review](../../CODE_REVIEW_2026-10-03.md) contains 14 actionable findings, including five P1 findings. Application source was not changed during this thesis revision. Existing local logs, original thesis backups, and raw evaluation predictions/manifests were preserved.

## Changes completed

| Area | Result |
|---|---|
| Research problem and gaps | Four scoped gaps and RQ1–RQ4 linked to O1–O8, measurable assessment criteria, available evidence, and remaining validation. |
| Literature | Rebuilt focused comparison from primary sources. Corrected SOCbot's published identity and use of retrieval/dynamic probing, the IEA report, and claims about CAPI. Retained 23 actually cited sources and added missing standards, dataset, model and statistics references. |
| Contributions | Distinguish comparative retrieval evidence, heuristic cross-standard screening, an integrated prototype, and diagnostic artifacts. Removed unsupported global-first novelty and proof of respondent benefit. |
| Architecture | Separate the live API's legacy/hierarchical/small-encoder default from official enriched/large flat evaluation. Distinguish direct orchestration, optional planning, and experimental manager-mediated CrewAI delegation. |
| Implementation | Corrected model routing, semantic validation, coverage, questionnaire fields, deployment, authorization, persistence, retention, and interface claims against source. |
| Methodology | Identify WISCO as externally sourced reference titles with project-generated group-separated splits; disclose repeated benchmark use, synthetic label limitations, two refusal rows, and an uncompleted human pilot. |
| Statistics | Recomputed counts, paired outcomes, intervals, tiny exact probabilities, per-language results, translation, reranking, coordination, runtime units, and pilot summaries. Distinguish changed codes from changed correctness. |
| Reproducibility | Appendix F records available historical settings, artifact paths, commit markers, environment records, and unresolved source/encoder provenance. Added an offline standard-library verifier with 31 input hashes. |
| Presentation | A4 layout, 12-point TeX Gyre Termes, consistent spacing, portable TeX fonts, corrected appendix numbering, contained architecture diagram, embedded vector results graph, and resolved citations/references. Fixed clipped title content and long filename overflow found through rendered-page inspection. |
| Historical records | Earlier audits are archived and explicitly superseded; the prior bibliography is retained separately. Originals are not submission sources. |

## Key recalculated results

All five principal occupation runs contain the same 18,747 identifiers and reference labels.

| Recorded configuration | Correct / 18,747 | Exact accuracy |
|---|---:|---:|
| Title-only, E5-small, hierarchical | 1,941 | 10.35% |
| Title-only, E5-small, flat | 3,973 | 21.19% |
| Title-only, E5-large, flat | 5,567 | 29.70% |
| Enriched, E5-small, flat | 6,102 | 32.55% |
| Enriched, E5-large, flat | 7,676 | 40.95% |

These are occupation-reference records, not actual LFS interviews. Later configurations reused the benchmark during development. The best result requires independent confirmation and has incomplete executed-source provenance.

Three incorrect paired probabilities were replaced by exact recalculations:

| Comparison | Baseline-only / alternative-only successes | Nominal exact p |
|---|---:|---:|
| Enriched-small versus title-only-small | 743 / 2,872 | 8.59 × 10^-293 |
| Enriched-large versus title-only-small | 668 / 4,371 | 7.69 × 10^-663 |
| Enriched-large versus enriched-small | 1,125 / 2,699 | 5.40 × 10^-147 |

Record-level intervals and tests assume conditions that do not account for related language variants, repeated selection, or multiple exploratory comparisons. They are not population uncertainty estimates.

The 324-record zero-shot run has 61 successes (18.83%); the matched retrieval baseline has 86 (26.54%), with nominal paired p=0.00967. Comparing the zero-shot subset with the entire benchmark was replaced by this matched analysis.

Corrective retrieval changes nine final codes but leaves aggregate accuracy at 31/60, with two gains and two losses. Query planning changes 21 occupation codes, with one loss and two gains. Unchanged totals are not unchanged predictions or proof of equivalence.

The synthetic industry/education comparison retains its original 449 rows, including two later-identified refusal texts: 59 legacy successes and 372 flat successes (13.14% and 82.85%). No completed clean, expanded-corpus result was substituted.

## Remaining gaps and what would close them

These requirements depend on the claim being made. A thesis can report a limited prototype and negative results honestly; missing evidence must not be replaced by optimistic wording.

| Priority / gap | Why it matters | Concrete next work / acceptance evidence |
|---|---|---|
| P1: reviewer authorization | Ordinary respondents can access and modify other respondents' HITL records. | Server-enforced reviewer roles and permitted scope; isolated tests showing ordinary users cannot list/review foreign items. |
| P1: status and wage extraction | Hindi/Urdu labels can truncate routing; unrelated numbers can silently populate wages. | Canonical option values and validated enums; wage extraction restricted to wage input; complete routing tests in supported languages. |
| P1: fresh deployment | Docker startup calls create_all before model registration and omits migration setup. | Packaged migration configuration and successful startup against a fresh isolated database volume. |
| P1: evaluation denominator | Top-3 coverage omits retrieval misses. | Include all eligible pools; example corrected from 61/108=56.48% to 61/130=46.92%; regenerate affected summaries without overwriting original runs. |
| P2: persistence and reports | Refined/corrected occupation data may not be saved; deleted sessions reappear in prefill; badges can misstate review status. | Active response revisions, synchronized saved classifications/review records, deleted-data exclusion, and authoritative review-state display. |
| Independent final benchmark | Reused WISCO results can reflect adaptive selection; language records are dependent. | Freeze selected settings and evaluate on an untouched grouped test set; group-aware uncertainty and a declared comparison plan. |
| Historical execution provenance | Enriched-large CSV reports the small encoder; its recorded HEAD lacks the named enriched-large profile. Later manifests do not establish a complete executed source snapshot. | Repeat with complete source snapshot/diff, resolved model revision, taxonomy hashes, dependencies and command saved alongside predictions. |
| Real multi-standard labels and coverage | Generation and retrieval share taxonomy descriptions; intended labels are not independently judged. Enriched coverage is 121 ISIC classes and 61 ISCED-F fields. | Permitted respondent descriptions, independent expert coding/adjudication and agreement measures; complete validated catalogue coverage or documented exclusions. |
| SRE validity and calibration | 61 constructed cases test agreement with authored rules. Repeating them does not estimate real contradiction detection. | Independent expert-labelled combinations, including legitimate atypical cases; measure false positives/negatives and review burden; calibrate on separate data if probability claims are intended. |
| Conversational language quality | Five-language title retrieval does not establish correct interviews, entity extraction, code-switching or dialect interpretation. Normalized Gulf per-case outputs are absent. | Native-speaker review and independent conversational annotations; complete language-specific routing tests; retain both arms of future dialect experiments. |
| Human outcomes | The completed n=30 run is synthetic English API traffic. Burden, satisfaction, accessibility and comparative cost remain unmeasured. | Institutional ethics approval, finalized translated consent/data-management plans, participant feasibility study and a justified subsequent effectiveness design. |
| Runtime and operational evidence | Local sequential timings, sampled RSS and 42 successful request sequences do not establish production throughput or completed interviews under load. | Matched hardware/configuration benchmarks, measured failure/recovery/concurrency, completed-workflow load criteria, and transparent cost accounting. |

The code review also covers SMS OTP limiting, confidence-field mismatch, dry-run metrics, and configuration metadata. No participant recruitment, live messaging, or new expensive model experiments were performed in this review.

## Validation and limits

- The thesis compiles with Tectonic 0.17.0 (XeTeX/BibTeX).
- Active document validation resolves 23 cited sources and 89 labels with no duplicate/undefined references, missing inputs/graphics, or unbalanced environments.
- The final compilation contains no overfull boxes or missing-character messages. Non-fatal underfull paragraph warnings remain; these concern justification and do not establish research validity.
- Numerical verification recalculates the saved results using Python's standard library and records SHA-256 hashes for 31 source artifacts. It initializes no application models and accesses no network.
- The graph generator verifies counts and identical case/reference pairs before plotting.
- Rendered title, abstract, architecture, results, gap tables, questionnaire, configuration appendix and bibliography were inspected. PDF page dimensions are A4; all-page content-boundary checks found no text outside the page.
- The earlier code review passed 194 focused existing tests and used isolated databases/mocked AI for selected reproductions. It did not run every application test, a browser acceptance suite, a fresh Docker deployment, or live model accuracy.
- Institutional formatting and signed declarations/certificates require the university/supervisor's final check. No specific institutional formatting specification was supplied.
- This review does not verify originality through a plagiarism database, establish ethics approval, or certify legal/regulatory or complete international statistical compliance.

A defensible next submission is the reviewed PDF plus this gap report for supervisor assessment, with claims limited to the evidence actually available.
