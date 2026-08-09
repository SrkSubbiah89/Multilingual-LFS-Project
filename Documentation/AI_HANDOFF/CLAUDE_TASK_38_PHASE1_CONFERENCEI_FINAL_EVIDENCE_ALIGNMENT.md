# Task 38: Phase 1 and Conference I Final Evidence Alignment

## Purpose

Bring the Phase 1 summary and Conference I Reviewer #2 documentation into an evidence-accurate final state using the validated Task 36 raw run and the clean offline Task 37.1 reproduction.

This is a documentation-only task. It must make the reviewer response more honest and specific, not more optimistic. In particular, it must state that the official ILO 2021 ISCO-08 profile was evaluated on the controlled WISCO v2 benchmark and that, in this exact four-digit, no-reranker comparison, the flat comparator outperformed the strict hierarchical method.

This does not resolve the reviewer request for real Labour Force Survey validation.

## Required source and branch

- Base branch: `reviewer2-wisco-official-tier1-analysis-clean-reproduction-20260810`
- Required base SHA: `6b797f77291046f7a89268a33ab116e868ef7503`
- New branch: `reviewer2-phase1-conference1-final-evidence-alignment-20260810`
- Fetch `origin` and verify both local base and `origin/reviewer2-wisco-official-tier1-analysis-clean-reproduction-20260810` equal the required SHA before branching.
- Do not merge, rebase, reset, clean, stash, pull, force-push, or open a PR.
- Do not touch protected or prior-task branches.

## Evidence hierarchy

Use only the following validated sources for Task 36/37.1 claims:

- Task 36 final report:
  `Documentation/AI_HANDOFF/CLAUDE_TASK_36_FINAL_REPORT.md`
- Task 37 final report:
  `Documentation/AI_HANDOFF/CLAUDE_TASK_37_FINAL_REPORT.md`
- Task 37.1 final report:
  `Documentation/AI_HANDOFF/CLAUDE_TASK_37_1_FINAL_REPORT.md`
- Task 36 raw input/output hashes, as independently re-verified in Task 37.1:
  - flat CSV: `d0692b4a87db11945dbd046ead79a32dcc36d715fbaa66fcc4e4853102be5f02`
  - hierarchical CSV: `b72193c8411b076df827abf2fa8bc6c2c72ac60f86799e435b94ff5eb237a8a4`
  - heldout export: `41c20fcc9eeec42358bdd90f211f6a344a47394b4b76fdf02c4ed5305cd1931c`
- Official catalogue identity/count evidence already validated in the chain:
  - 10 major groups;
  - 43 sub-major groups;
  - 130 minor groups;
  - 436 unit groups.

Task 37's one out-of-scope read-only Qdrant metadata call remains permanently disclosed. Do not delete, revise, hide, minimize, or reinterpret it. Task 37.1 independently reproduced the analysis from local raw files without a Qdrant connection. The final documentation must cite Task 37.1 as the clean reproduction record and may mention Task 37 only with this qualification.

## Fixed results that may be documented

All values below are exact targets to transcribe from the validated reports, not values to recompute or tune:

| Metric | Official flat | Strict hierarchical |
|---|---:|---:|
| Heldout cases | 18,747 | 18,747 |
| Exact 4-digit ISCO-08 correct | 3,973 | 1,941 |
| Accuracy | 21.1927% | 10.3537% |
| 95% Wilson interval | [20.6136%, 21.7836%] | [9.9256%, 10.7979%] |

Paired comparison:

- both correct: 1,341;
- flat-only correct: 2,632;
- hierarchical-only correct: 600;
- both incorrect: 14,174;
- hierarchical minus flat: -10.8391 percentage points;
- exact two-sided McNemar p-value: `1.8573559951149046e-301`.

Operational observations that may be documented only as local, single-run descriptions:

- flat query duration mean / median / p95 / p99 / max: 8.37 / 7.76 / 10.58 / 26.50 / 119.21 ms;
- hierarchical stage 1 mean / median / p95 / p99 / max: 6.71 / 5.55 / 9.97 / 17.87 / 208.01 ms;
- hierarchical stage 2: 15.32 / 13.90 / 27.73 / 38.86 / 86.73 ms;
- hierarchical stage 3: 31.26 / 28.62 / 49.09 / 70.53 / 2,581.40 ms;
- hierarchical stage 4: 54.30 / 49.68 / 83.26 / 116.30 / 202.29 ms;
- 228,232 hierarchical stage-level queries;
- zero retries, exceptions, fallbacks, unavailable outcomes, or stage-budget exhaustion in the Task 36 full run;
- no reranker, LLM, ISIC, ISCED, or SRE activity in either comparison arm.

## Claims that must remain prohibited

Do not state or imply:

- real Labour Force Survey validation;
- real respondent text;
- an ISIC, ISCED, or Semantic Relation Engine evaluation or improvement;
- LLM reranking comparison or conclusion;
- a hierarchy-accuracy improvement, novelty gain, or superiority;
- an official coverage percentage or generalization claim;
- cost, memory, scalability, throughput, production latency, or SLA claim;
- that WISCO is an official ILO dataset;
- that WISCO is the same as ISCO-08;
- that historical legacy 441/131 counts equal the official 436/130 catalogue profile;
- that the Task 36 benchmark resolves Reviewer #2's real-LFS-data request;
- that the Task 37 scope breach did not occur.

Use this terminology consistently:

- **WISCO v2 controlled multilingual ISCO-08 benchmark** for the dataset;
- **official ILO 2021 ISCO-08 catalogue profile** for the runtime/profile;
- **controlled exact-code evaluation**, not “field validation”;
- **local single-run operational observation**, not “performance benchmark” or “SLA”.

## Documentation changes required

Read every target file in full before editing. Preserve useful historical material and append or clearly supersede it rather than silently deleting audit history.

### Create

1. `Documentation/Conference_I_Reviewer_2/OFFICIAL_WISCO_TIER1_CONTROLLED_RESULTS.md`

This is the canonical durable, manuscript-ready evidence note. It must contain:

- a clear scope label and evidence chain;
- the exact Task 36 and Task 37.1 branch/commit identities;
- raw-file hashes;
- WISCO v2 split/provenance description;
- exact correctness definition;
- eligibility and integrity criteria;
- headline table;
- paired contingency table and exact McNemar method;
- descriptive language and major-group subgroup tables copied only from the Task 37.1 derived result;
- local-only operational table;
- a clear interpretation: flat was more accurate than strict hierarchical retrieval in this specific controlled configuration;
- a “what this does not establish” section containing every prohibited-claim category above;
- reviewer-response-ready safe wording and explicitly unsafe wording;
- a direct note that Task 37's scope breach remains disclosed and Task 37.1 is the clean offline reproduction;
- links/paths to Task 36, Task 37, Task 37.1 reports and the Git-ignored raw/derived artifact locations with SHA-256 identities.

2. `Documentation/Conference_I_Reviewer_2/MANUSCRIPT_SAFE_WISCO_WORDING.md`

This must provide concise, ready-to-paste, evidence-safe language for:

- abstract/summary;
- methods;
- results;
- limitations;
- response to Reviewer #2 comment 2 (novelty);
- response to Reviewer #2 comment 3 (real LFS data);
- response to Reviewer #2 comment 4 (computational analysis).

It must include an explicit “do not write” list. It must not use numeric evidence in the abstract wording unless it begins with the controlled-benchmark qualification. It must never say that hierarchical retrieval improved accuracy.

### Update

3. `Documentation/Conference_I_Reviewer_2/REVIEWER_RESPONSE_IMPLEMENTATION_MATRIX.md`

Update it as the authoritative response matrix:

- Preserve that Reviewer #2 comment 3 remains **Awaiting data**. State Task 36/37.1 does not use real LFS data and does not close that request.
- For comment 2, retain a cautious status such as **Partially evidenced**, add the new controlled comparison, and state the result does not support a hierarchy-accuracy superiority claim. State what evidence remains required for a novelty claim.
- For comment 4, move only as far as evidence supports, for example **Partially evidenced: controlled local retrieval measurements exist**, while explicitly listing still-missing memory, throughput/scalability, production/deployment, reranking, and real-LFS measurements.
- Preserve comment 5’s ISIC limitations and clearly distinguish the official ILO 2021 ISCO-08 profile from unresolved ISIC coverage.
- Add a dated Task 36/37.1 WISCO section with exact numbers, identifiers, limitations, and safe interpretation.
- Remove or supersede statements that say WISCO has no benchmark measurement, Step 7B is not completed, or no WISCO accuracy number exists. Do not erase historical Steps 4–7A or synthetic-fixture records.

4. `Documentation/Conference_I_Reviewer_2/README.md`

- Update the task chronology to cover Tasks 23 through 37.1 accurately.
- State that Task 36/37.1 produced one controlled WISCO v2 exact-code comparison with official ILO 2021 profile collections.
- Replace obsolete “no WISCO accuracy number” and “Step 7B not yet completed” wording with a precise, limited status.
- Link to the two new canonical documentation files.
- Keep all ISIC/ISCED-F hierarchical retrieval warnings and no-claim limits intact.
- Keep the statement that WISCO is not real LFS validation.

5. `Documentation/Conference_I_Reviewer_2/WISCO_LEAKAGE_AUDIT_AND_RUN_PLAN.md`

- Preserve the v1/v2 leakage-audit history.
- Mark the non-reranked official-profile controlled Tier-1 full run as completed and link to Task 36/37.1 evidence.
- Mark reranking and ISIC/ISCED/SRE axes as not evaluated on WISCO.
- Replace any stale command/pending language that contradicts the completed run.
- Add the correct outcome: flat outperformed strict hierarchy on this controlled exact-code comparison.

6. `Documentation/Conference_I_Reviewer_2/generated/MEASURED_EVALUATION_EVIDENCE_SUMMARY.md`

- Preserve the historical five-row synthetic-fixture evidence as a distinct, manuscript-ineligible record.
- Append a clearly separated “Task 36/37.1 controlled WISCO v2 official-profile evidence” section.
- Include the headline and paired results, raw hash identities, zero-reranker/zero-ISIC/zero-ISCED/zero-SRE context, and the strict non-real-LFS limitation.
- Do not merge the five-row synthetic numbers with the 18,747-case WISCO results.

7. `Documentation/Phase_1_Summary/Phase_1_Summary.md`

Do not rewrite the 2026-08-02 historical snapshot as though it were current. Add a clearly dated **2026-08-10 evidence addendum** that:

- directs readers to the canonical controlled-results note;
- distinguishes the historical legacy 10/43/131/441 implementation snapshot from the official ILO 2021 profile’s 10/43/130/436 catalogue counts;
- states the Task 36/37.1 controlled exact-code result without a real-LFS claim;
- corrects any impression that 1,178 tests is the current final test count by stating the current Task 37.1 verification result: 2,185 passed, 1 deselected, 1 warning;
- preserves historical information rather than silently changing it.

## Documentation integrity requirements

1. Every Task 36/37.1 number must agree character-for-character in all updated documents.
2. Add exact commit IDs and input hashes wherever a document presents headline results.
3. Do not introduce web citations, new external facts, or new references. Link only to repository documents/commits already in this evidence chain.
4. Do not commit raw local-run CSV/JSON/Markdown artifacts from Git-ignored folders.
5. Do not change any source code, test, configuration, generated executable script, dataset, collection, or benchmark output.
6. Do not change the Task 36, Task 37, or Task 37.1 final reports.
7. Add/update only documentation files named above plus the final Task 38 report. If another file appears necessary, stop and explain why rather than expanding scope.

## Required verification

Run:

```bash
python -m pytest backend/tests eval/ -q
```

Expected result: `2185 passed, 1 deselected, 1 warning`.

Run repository documentation consistency tests and a deterministic documentation audit that:

- scans every changed documentation file for all fixed numbers and checks agreement;
- scans for prohibited phrases/claims;
- verifies all internal links and referenced report paths resolve;
- proves Task 36/37/37.1 reports are byte-identical to the base;
- shows only allowed documentation files changed.

Do not weaken or skip an existing documentation test. If tests or the audit fail, fix only the documentation error. Do not modify source/test code to accommodate documentation.

## Final report

Create:

`Documentation/AI_HANDOFF/CLAUDE_TASK_38_FINAL_REPORT.md`

Include:

1. final branch, base SHA, final SHA, push confirmation, and clean working tree;
2. complete changed-file list and confirmation no source/test/data/config/output artifact changed;
3. per-document summary of changes;
4. exact values and hashes used, with a cross-document consistency result;
5. the complete prohibited-claim audit result;
6. Task 37 scope-breach disclosure preserved exactly, and Task 37.1 clean-reproduction role;
7. full test and documentation-audit commands/output;
8. protected-branch preservation;
9. a final evidence statement:
   - Phase 1 and Conference I Reviewer #2 documentation is now aligned with the controlled benchmark evidence;
   - the new result is a negative hierarchy-versus-flat finding in one controlled setup, not a claim of system superiority;
   - real LFS, ISIC, ISCED, SRE, reranking, cost, coverage, generalization, and production claims remain unresolved or unsupported;
   - B1 remains stale/quarantined.

Push only the new branch. Stop after the final report.
