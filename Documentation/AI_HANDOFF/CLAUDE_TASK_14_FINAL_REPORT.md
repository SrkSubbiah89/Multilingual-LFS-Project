# Task 14 Final Report — Integrate Hierarchical Integrity Hardening into the Strict WISCO Benchmark Baseline

Produced in response to
`Documentation/AI_HANDOFF/CLAUDE_TASK_14_INTEGRATE_STRICT_WISCO_BASELINE.md`
(task ID `14-integrate-strict-wisco-baseline`). Git integration and
hermetic test verification only — no WISCO read/rebuild/export/evaluate,
no Qdrant connection, no embedding-model load, no LLM/API call, no
benchmark script invocation.

## Verified source SHAs

| Role | Branch | SHA |
|---|---|---|
| Failed-run/report base | `reviewer2-wisco-tier1-results-20260808` | `6710192ea255c2cfb00599db9e55f65f9a53ca53` |
| Integrity-hardening feature | `reviewer2-hierarchical-integrity-hardening-20260808` | `68cfcd452b880b7cd62db416c66e5db96681f306` |

`git fetch origin` confirmed both `origin/...` refs at exactly these SHAs.
The starting checkout was on the Task 13 feature branch itself, clean, at
`68cfcd4...`, before this task began.

## Merge SHA / final SHA

| | |
|---|---|
| New branch | `reviewer2-wisco-strict-benchmark-baseline-20260808` |
| Merge commit SHA | `4c1840bfb75ae1254de913b34ee40e7ba242699b` |
| Merge commit parents | `6710192ea255c2cfb00599db9e55f65f9a53ca53` (Task 12 results base), `68cfcd452b880b7cd62db416c66e5db96681f306` (Task 13 feature) |
| Final SHA | commit created and pushed as this task's own closing step, containing this report (see push confirmation below) |

Branch created with `git switch -c
reviewer2-wisco-strict-benchmark-baseline-20260808
reviewer2-wisco-tier1-results-20260808`, then merged with
`git merge --no-ff reviewer2-hierarchical-integrity-hardening-20260808`
(explicit two-parent merge commit, no fast-forward).

## Conflict status

**No merge conflicts.** `git merge --no-ff` reported "Merge made by the
'ort' strategy" with no conflict markers, and `git status --short`
immediately after was empty. Consistent with the task's own statement
that Task 13 is exactly one commit ahead of the Task 12 results branch,
touching only files the results branch itself never modified.

## Exact base-to-merge file scope

`git diff --stat 6710192..4c1840b` → **6 files changed, 1292 insertions(+), 8 deletions(-)**

- `backend/rag/hierarchical_store.py` (modified) — keyword-anchor retry
  recovery, bounded Qdrant timeout.
- `eval/run_eval.py` (modified) — `--require-genuine-hierarchical`,
  `--max-stage-latency-ms`, `check_strict_hierarchical()`.
- `backend/tests/test_hierarchical_store.py` (new, 18 tests).
- `eval/test_require_genuine_hierarchical.py` (new, 19 tests).
- `Documentation/Conference_I_Reviewer_2/WISCO_LEAKAGE_AUDIT_AND_RUN_PLAN.md`
  (modified) — Task 12/13 post-execution update section.
- `Documentation/AI_HANDOFF/CLAUDE_TASK_13_FINAL_REPORT.md` (new).

Exactly the six files Task 13 produced. No conflict-resolution edit was
made (none was needed).

## Preservation-check evidence

**1. Task 12 raw output CSVs remain local, Git-ignored, and byte-identical:**

```
md5sum -c task14_task12_artifact_checksums_before.txt
→ eval/local_runs/wisco_v2_tier1_full_20260807T234643Z/hierarchical/20260808T000252Z_wisco_v2_tier1_full_hierarchical_norerank.csv: OK
→ eval/local_runs/wisco_v2_tier1_full_20260807T234643Z/flat/20260807T234757Z_wisco_v2_tier1_full_flat_norerank.csv: OK
```

Checksums recorded immediately before this task began, matching the
checksums Task 13 itself recorded — confirmed unchanged by both tasks.
`git check-ignore -v` confirms the run root remains covered by the
existing `eval/local_runs/` `.gitignore` pattern. Neither file was opened
for writing at any point in this task.

**2. Task 12 final report unchanged, still states `TIER1_COMPLETED: no`:**
confirmed via `grep` — line 7 of
`Documentation/AI_HANDOFF/CLAUDE_TASK_12_FINAL_REPORT.md` reads
`## TIER1_COMPLETED: no`, unmodified.

**3. Task 05/05.1/05.2 ISIC/ISCED-F hierarchy files present:**
`backend/rag/hierarchy_nodes.py`, `backend/rag/standard_hierarchical_store.py`,
`backend/rag/build_standard_hierarchical_collections.py` all confirmed
present via `ls`.

**4. Task 07 repaired `get_llm` mock present, including the `temperature`
assertion:** confirmed in `backend/tests/test_isco_classifier_extended.py`
— `get_llm_mock` fixture (`side_effect=lambda *args, **kwargs: MagicMock()`)
and `assert "temperature" in call_kwargs` both present.

**5. B1 quarantine byte-identical to the Task 12 base:**

```
git diff 6710192ea255c2cfb00599db9e55f65f9a53ca53..HEAD -- eval/configs/b1_frozen.json eval/dev_sweep.py
→ (no output)
```

Confirmed zero diff — B1 remains quarantined exactly as Task 04 left it;
no re-freeze, no B2 sweep bypass.

**6. Task 13 behavior confirmed present after the merge** (via direct
grep against the merged tree, not inference):

- Keyword-anchor retry logic (`retried = True`,
  `trace.update(attempt_trace)`) present in
  `backend/rag/hierarchical_store.py`.
- `QDRANT_DEFAULT_TIMEOUT_SECONDS = 30` and
  `_resolve_qdrant_timeout_seconds()` present (finite default, safe
  environment override).
- `check_strict_hierarchical()`, `--require-genuine-hierarchical`, and
  `--max-stage-latency-ms` all present in `eval/run_eval.py`.

**7. Task 09 model-free behavior confirmed present:**
`ISCOClassifier.__init__`'s `enable_llm: bool = True` parameter and
`eval/run_eval.py`'s `any_row_has_paired_industry_education` gating logic
both confirmed present via grep.

## Test commands and exact outputs

```
pytest backend/tests/test_hierarchical_store.py eval/test_require_genuine_hierarchical.py backend/tests/test_hierarchy_engine.py backend/tests/test_isco_classifier.py backend/tests/test_isco_classifier_extended.py eval/test_model_free_isco_evaluation.py eval/test_run_eval_b2.py eval/test_sre_isic_isced_coupling_fix.py -q
→ 238 passed in 2.57s

pytest backend/tests eval/ -q
→ 1948 passed, 1 deselected, 1 warning in 297.26s (0:04:57)
```

**Zero failures.** Full-suite pass count is an exact match to Task 13's
own final-state numbers (1948 passed, 1 deselected) — confirming this
integration introduced zero regressions and zero new failures. This claim
is based on the actual command output above, not assumed.

## Confirmation: zero live WISCO/Qdrant/model/LLM/benchmark activity

- The merge itself is a pure Git tree operation — no code was executed.
- All verification commands ran the existing hermetic test suites
  (`FakeQdrantClient`/`FakeEmbedder`/mocked `ISCOClassifier`/`get_llm`/
  `Agent`/`Task`/`Crew` throughout), none of which construct a real
  `QdrantClient` connection, load a real `SentenceTransformer` model, or
  call Ollama/an LLM API.
- No WISCO data was read, rebuilt, exported, or evaluated — the Task 12
  raw CSVs were only checksummed (read-only) for the preservation check
  above, never opened for writing, and no `eval/build_wisco_isco_benchmark*.py`
  or `eval/export_benchmark_to_run_eval_csv.py` was invoked.
- No `eval/run_eval.py`, `eval/analyze.py`, or `eval/ablation_runner.py`
  was invoked outside the hermetic test suite (where every classifier is
  mocked).
- No Qdrant collection was listed, connected to, or mutated. No B1
  re-freeze, no B2 sweep.

## Protected branch status

| Branch | Status |
|---|---|
| `master` | not touched |
| `conference1-b2-evaluation` | not touched |
| `reviewer2-wip-snapshot-20260807` | not touched |
| `reviewer2-b2-integration-20260807` | not touched |
| `reviewer2-isic-iscedf-hierarchical-rag-20260808` | not touched |
| `reviewer2-isic-iscedf-hierarchical-resilience-20260808` | not touched |
| `reviewer2-isic-iscedf-hierarchical-model-init-resilience-20260808` | not touched |
| `reviewer2-isic-iscedf-integration-20260808` | not touched |
| `reviewer2-isic-iscedf-test-green-20260808` | not touched |
| `reviewer2-pre-evaluation-baseline-20260808` | not touched |
| `reviewer2-model-free-isco-evaluation-20260808` | not touched |
| `reviewer2-wisco-measurement-baseline-20260808` | not touched |
| `reviewer2-wisco-tier1-preflight-20260808` | not touched |
| `reviewer2-wisco-tier1-results-20260808` | not touched (used only as the merge base; never checked out for writing) |
| `reviewer2-hierarchical-integrity-hardening-20260808` | not touched (merged FROM, not merged INTO; the branch itself was never modified) |

No `git merge` into any protected branch, no `git rebase`, `git reset`,
`git clean`, `git stash`, `git pull`, or force-push was run at any point
in this task. No PR was created. No production code, test, configuration,
requirements, dataset, benchmark split, manuscript wording, or Task 12
artifact was changed beyond the clean merge and this report.

## Working-tree status

Clean before this task started (verified on the Task 13 feature branch),
clean immediately after the merge (`git status --short` empty), and clean
again immediately before this report's commit.

## Remaining manuscript limits

Unchanged from Tasks 12 and 13 — this task is integration-only and adds
no new evidence:

- No Tier-1 accuracy, latency, or comparison figure exists anywhere as of
  this task's completion.
- No claim that the Task 12 severe-latency stalls' root cause is
  understood or resolved — Task 13's Qdrant timeout only bounds a class
  of future blocking behaviour, it is not a confirmed fix for what
  actually happened in Task 12.
- No real-LFS, ISIC, ISCED, SRE, or reranking conclusion is supported.
- `reviewer2-wisco-strict-benchmark-baseline-20260808` is now the single
  integrated branch combining the stopped Task 12 lineage with Task 13's
  hardening, ready for the next task to independently inspect and
  authorize a larger strict preflight before any new full benchmark run.
  No WISCO preflight or rerun was performed here, per the stop condition.
