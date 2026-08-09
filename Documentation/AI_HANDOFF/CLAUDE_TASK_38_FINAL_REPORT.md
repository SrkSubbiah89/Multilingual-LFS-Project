# Task 38 Final Report — Phase 1 and Conference I Final Evidence Alignment

Produced in response to
`Documentation/AI_HANDOFF/CLAUDE_TASK_38_PHASE1_CONFERENCEI_FINAL_EVIDENCE_ALIGNMENT.md`.
This is a documentation-only task: it brings the Phase 1 summary and
Conference I Reviewer #2 documentation into an evidence-accurate final
state using the validated Task 36 raw run and the clean offline Task
37.1 reproduction. No source, test, dataset, or configuration file was
touched.

## 1. Branch, base SHA, final SHA, push, working tree

| | |
|---|---|
| Base branch | `reviewer2-wisco-official-tier1-analysis-clean-reproduction-20260810` |
| Required/verified base SHA | `6b797f77291046f7a89268a33ab116e868ef7503` (confirmed against both the local branch and `origin` before branching, re-confirmed identical throughout) |
| New branch | `reviewer2-phase1-conference1-final-evidence-alignment-20260810` |
| Final commit SHA | reported in Claude's end-of-turn response, not inside this file |

Working tree was clean before branching; `git status --porcelain`
before this report's own commit showed exactly the 7 allowed
documentation files (below) plus this report itself.

## 2. Complete changed-file list; confirmation no source/test/data/config/output artifact changed

**Created (2):**
- `Documentation/Conference_I_Reviewer_2/OFFICIAL_WISCO_TIER1_CONTROLLED_RESULTS.md`
- `Documentation/Conference_I_Reviewer_2/MANUSCRIPT_SAFE_WISCO_WORDING.md`

**Updated (5):**
- `Documentation/Conference_I_Reviewer_2/REVIEWER_RESPONSE_IMPLEMENTATION_MATRIX.md`
- `Documentation/Conference_I_Reviewer_2/README.md`
- `Documentation/Conference_I_Reviewer_2/WISCO_LEAKAGE_AUDIT_AND_RUN_PLAN.md`
- `Documentation/Conference_I_Reviewer_2/generated/MEASURED_EVALUATION_EVIDENCE_SUMMARY.md`
- `Documentation/Phase_1_Summary/Phase_1_Summary.md`

**Report (1):** `Documentation/AI_HANDOFF/CLAUDE_TASK_38_FINAL_REPORT.md`

No other file appeared necessary — the 7 files above (plus this
report) were sufficient for every required change. `git status
--porcelain` confirmed exactly this set both before and after work
(Section 7). No `backend/`, `eval/` source/test file, dataset,
catalogue, Qdrant collection, benchmark output, or Git-ignored raw/
derived artifact was created, modified, or deleted.

## 3. Per-document summary of changes

**`OFFICIAL_WISCO_TIER1_CONTROLLED_RESULTS.md` (new)** — the canonical,
manuscript-ready evidence note: scope label, evidence chain (Task
36/37/37.1 branch/SHA identities), Task 37 scope-breach disclosure
section, raw-file hashes, WISCO v2 split/provenance, official catalogue
profile identity (10/43/130/436, distinct from the legacy 131/441),
exact correctness definition, eligibility criteria, headline table,
paired contingency + McNemar method, full language and major-group
subgroup tables (copied verbatim from the Task 37.1 derived result),
local-only operational table, interpretation ("flat was more accurate
than strict hierarchical retrieval in this specific controlled
configuration"), a "what this does not establish" section covering
every prohibited-claim category, and paths/hashes to the Git-ignored
raw/derived artifact locations.

**`MANUSCRIPT_SAFE_WISCO_WORDING.md` (new)** — ready-to-paste wording
for abstract/methods/results/limitations and responses to Reviewer #2
comments 2, 3, and 4; an explicit "do not write" list; numeric evidence
in the abstract pattern always opens with the controlled-benchmark
qualification; explicitly never states hierarchical retrieval improved
accuracy.

**`REVIEWER_RESPONSE_IMPLEMENTATION_MATRIX.md`** — rows 2, 3, 4, 5
updated with the new evidence (elaboration text placed in the
"Evidence output"/"Remaining action" columns, never inside the Status
column, to preserve the existing `test_docs_consistency.py` constraint
that Status cells contain only one of the 5 allowed labels verbatim).
Comment 2 (novelty): status unchanged (**Partially evidenced**); added
the real non-dry-run result and the explicit statement that it does not
support a hierarchy-accuracy superiority claim. Comment 3 (real LFS
data): status unchanged (**Awaiting data**); added that Task 36/37.1
does not use real LFS data and does not close this request. Comment 4
(computational analysis): status moved from **Awaiting measurement** to
**Partially evidenced**, citing the real controlled local
retrieval-latency measurements while explicitly listing still-missing
memory, throughput/scalability, reranked-configuration, production, and
real-LFS measurements. Comment 5 (ISIC): status unchanged (**Partially
evidenced**); clarified that Tasks 23-37.1 closed the "no fresh ISCO-08
evaluation run" gap for the official profile specifically, without any
ISIC coverage implication. Added a new dated
"2026-08-10 — Tasks 23-37.1" section with the full result and safe
interpretation; the Step 7A section's "No benchmark measurement was run"
sentence was narrowed to Step 7A specifically with a forward pointer,
never deleted.

**`README.md` (Conference_I_Reviewer_2)** — added table rows for the
two new canonical files; added a new "Tasks 23-37.1" subsection
documenting the completed run; replaced the "No WISCO accuracy number
has been produced by any of these tasks" sentence (now scoped
accurately to "as of Task 22") and the "Step 7B... has not happened
yet" sentence (now states the non-reranked tier completed 2026-08-10).
All ISIC/ISCED-F hierarchical retrieval warnings and no-claim limits,
and the "WISCO is not real LFS validation" statement, left intact.

**`WISCO_LEAKAGE_AUDIT_AND_RUN_PLAN.md`** — preserved the full v1/v2
leakage-audit history unchanged. Added inline forward-pointers at Phase
D's header and D.1's heading noting the non-reranked tier is now
complete (Tier 2/reranking remains not executed — that statement is
preserved as accurate). Appended a new "Post-execution update: Tasks
23-37.1" section at the end with the exact completed-run outcome (flat
outperformed strict hierarchy) and full limitations, following the same
pattern as the file's existing Task 12/13/15-21 post-execution update
sections.

**`generated/MEASURED_EVALUATION_EVIDENCE_SUMMARY.md`** — preserved the
5-row synthetic-fixture record (§1-8) completely unchanged. Appended a
new, clearly separated "§9. Task 36/37.1 — controlled WISCO v2
official-profile evidence" section with the headline/paired results,
raw hashes, zero-reranker/ISIC/ISCED/SRE context, and the strict
non-real-LFS limitation — explicitly stated never to be merged or
averaged with the `n=5` numbers above it.

**`Phase_1_Summary.md`** — the 2026-08-02 historical snapshot (§1-9)
was left completely unchanged. Added a new "2026-08-10 Evidence
Addendum" section immediately after the title block (before the
original §1), which: points to the canonical results note; explicitly
distinguishes the historical legacy 10/43/**131**/**441** snapshot from
the official ILO 2021 profile's 10/43/**130**/**436** counts; states the
Task 36/37.1 result without any real-LFS claim; and states the current
verified test count (`2185 passed, 1 deselected, 1 warning`) while
explaining the 1,178 → 2,185 growth is due to substantial intervening
work (Tasks 09-37.1), not a correction of the 2026-08-02 count.

## 4. Exact values and hashes used; cross-document consistency result

Every updated/created document was checked to contain the exact fixed
values from the task file, transcribed character-for-character:
18,747; 3,973; 21.1927%; 1,941; 10.3537%; [20.6136%, 21.7836%];
[9.9256%, 10.7979%]; 1,341; 2,632; 600; 14,174; -10.8391 percentage
points; `1.8573559951149046e-301`; catalogue counts 10/43/130/436; and
the three raw-file SHA-256 hashes
(`d0692b4a87db11945dbd046ead79a32dcc36d715fbaa66fcc4e4853102be5f02`,
`b72193c8411b076df827abf2fa8bc6c2c72ac60f86799e435b94ff5eb237a8a4`,
`41c20fcc9eeec42358bdd90f211f6a344a47394b4b76fdf02c4ed5305cd1931c`).

A deterministic Python audit script (session scratchpad only, not
committed) scanned every changed document for: (a) presence and exact
match of every fixed number in the canonical document; (b) any
transcription-error-shaped near-miss numeric token (e.g. a mistyped
"21.1%" where "21.1927%" was expected); (c) prohibited-phrase patterns;
(d) markdown-link and backtick-path resolution; (e) byte-identity of
the Task 36/37/37.1 final reports against a pre-work hash snapshot; (f)
that only allowed files appear in `git status --porcelain`.

**Result**: the canonical document contains every fixed number exactly.
The script additionally flagged 18 items requiring manual review, all
of which were verified by hand to be false positives from the script's
own crude heuristics, not real documentation defects:
- 15 "near-miss" flags were all the correctly-rounded 2-decimal-place
  citation (`21.19%`/`10.35%`, the exact rounding of `21.1927%`/
  `10.3537%`) used consistently in prose alongside or after the
  full-precision figure — confirmed by direct grep of every occurrence
  across all five affected documents; every instance matched the
  correct rounding with no transcription error.
- 3 "prohibited phrase" flags were all phrases quoted inside explicit
  "do not write" / "what this does not establish" negated contexts
  (e.g. `**"hierarchical retrieval improved accuracy"** or any phrasing
  implying...` inside the "Do not write" list) — confirmed by reading
  each flagged context in full.
- 2 "link" flags were the two raw CSV filenames referenced as
  Git-ignored artifact locations, explicitly labeled as such
  ("Raw and derived artifacts are Git-ignored and not committed to this
  repository") immediately before the list — intentional, per the
  task's own instruction to cite Git-ignored raw-artifact locations.

Zero genuine cross-document inconsistency was found.

## 5. Prohibited-claim audit result

Checked every updated/created document against every category in the
task's "Claims that must remain prohibited" list. None of the following
appears asserted anywhere: real LFS validation; real respondent text;
an ISIC/ISCED/SRE evaluation or improvement; an LLM reranking
comparison/conclusion; a hierarchy-accuracy improvement, novelty gain,
or superiority claim (the documented result is explicitly the
opposite); an official coverage percentage or generalization claim; a
cost/memory/scalability/throughput/production-latency/SLA claim; WISCO
described as an official ILO dataset; WISCO equated with ISCO-08; the
legacy 441/131 counts equated with the official 436/130 counts; a
claim that Task 36/37.1 resolves Reviewer #2's real-LFS request; or any
statement that Task 37's scope breach did not occur. Terminology
("WISCO v2 controlled multilingual ISCO-08 benchmark", "official ILO
2021 ISCO-08 catalogue profile", "controlled exact-code evaluation",
"local single-run operational observation") is used consistently in
every new/updated document.

## 6. Task 37 scope-breach disclosure; Task 37.1's role

Every document that cites this evidence chain includes Task 37's
permanent, unrevised disclosure: one read-only `get_collection()` Qdrant
point-count call made outside its own declared scope, by an
operator-side preservation check (not the analyzer itself), causing no
mutation. Every document explicitly designates **Task 37.1 — not Task
37 — as the clean, citable, strictly-offline reproduction record**, and
states that Task 37 may be mentioned only with this qualification. This
disclosure is identical in substance across all documents (no
rewording that would soften or omit it) — verified by direct read of
each occurrence while drafting.

## 7. Test and documentation-audit commands/output

```bash
python -m pytest backend/tests eval/ -q
```
Result: `2185 passed, 1 deselected, 1 warning in 319.90s` — matches the
required result exactly. (Unaffected by this task's changes, since no
source/test file was touched; run to confirm.)

```bash
python -m pytest eval/test_docs_consistency.py -q
```
Result: `7 passed` — all pre-existing documentation-consistency tests
(matrix Status-label constraint, 8-row count, `generated/` path
traceability, guide-doc existence, standards-reference citation
traceability) still pass unmodified. No existing documentation test was
weakened, skipped, or altered.

Deterministic documentation audit (scratchpad script, Section 4):
fixed-number agreement confirmed exact in the canonical document; 18
manually-reviewed-and-cleared false-positive flags (Section 4); Task
36/37/37.1 report byte-identity confirmed (Section 8); only the 7
allowed files changed (confirmed via `git status --porcelain`, Section
8).

## 8. Protected-branch preservation

No merge, rebase, reset, clean, stash, pull, or force-push was
performed. No protected or prior-task branch was touched. Only
`reviewer2-phase1-conference1-final-evidence-alignment-20260810` was
created, and only that branch is pushed.

**Task 36/37/37.1 final report preservation** — SHA-256 hashed before
branching and re-hashed at task completion:

| File | SHA-256 | Match |
|---|---|---|
| `CLAUDE_TASK_36_FINAL_REPORT.md` | `d14e9c68c7d08844a3077866e681dff1595801d457af9cd626d5a3e4ce12b0bc` | yes |
| `CLAUDE_TASK_37_FINAL_REPORT.md` | `17ebc0fc97091166582083ee839e48a2d69b099f5981c008411635f7da5d8970` | yes |
| `CLAUDE_TASK_37_1_FINAL_REPORT.md` | `c7f2ffad9a5ebc34c3932a8281162091ab239d14995ba1a6caa0e8f22b0f5b8d` | yes |

`git status --porcelain` immediately before staging showed exactly:
```
 M Documentation/Conference_I_Reviewer_2/README.md
 M Documentation/Conference_I_Reviewer_2/REVIEWER_RESPONSE_IMPLEMENTATION_MATRIX.md
 M Documentation/Conference_I_Reviewer_2/WISCO_LEAKAGE_AUDIT_AND_RUN_PLAN.md
 M Documentation/Conference_I_Reviewer_2/generated/MEASURED_EVALUATION_EVIDENCE_SUMMARY.md
 M Documentation/Phase_1_Summary/Phase_1_Summary.md
?? Documentation/Conference_I_Reviewer_2/MANUSCRIPT_SAFE_WISCO_WORDING.md
?? Documentation/Conference_I_Reviewer_2/OFFICIAL_WISCO_TIER1_CONTROLLED_RESULTS.md
```
— exactly the 7 allowed files, no others.

## 9. Final evidence statement

- Phase 1 and Conference I Reviewer #2 documentation is now aligned
  with the controlled benchmark evidence produced by Tasks 36/37.1: the
  canonical results note, safe-wording guide, implementation matrix,
  README, leakage-audit/run-plan, measured-evidence summary, and Phase
  1 summary addendum all cite the same exact numbers, hashes, and task
  identities.
- The new result is a **negative hierarchy-versus-flat finding in one
  controlled setup** — flat retrieval was substantially more accurate
  than strict hierarchical retrieval on the WISCO v2 official-profile,
  non-reranked, full-heldout comparison. It is **not** a claim of
  system superiority, novelty proof, or general statement about
  hierarchical retrieval beyond this exact configuration.
- Real Labour Force Survey validation, ISIC, ISCED, Semantic Relation
  Engine evaluation, LLM reranking comparison, cost, coverage,
  generalization, and production-performance claims all remain
  unresolved or unsupported by this evidence — every updated document
  says so explicitly.
- B1 remains stale/quarantined; nothing in this task touches or
  revalidates it.
