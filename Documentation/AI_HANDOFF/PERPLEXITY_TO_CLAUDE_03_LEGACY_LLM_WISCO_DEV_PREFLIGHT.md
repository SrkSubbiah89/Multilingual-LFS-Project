# Task 39: Literal Legacy ISCO Conditional-LLM WISCO Development Preflight

## Decision and evidence boundary

This task performs the **first historical-logic experiment exactly as requested**. It uses the earliest verified ISCO classifier implementation, without substituting a current reranker policy, local Ollama, official ILO profile, hierarchy engine, strict hierarchy guard, new prompt, new parser, new candidate count, or new threshold.

The historical source of truth is:

```text
824fcf235ae2f8787706cf479a07620519c914de
Add two-stage ISCO-08 classifier agent
```

The old classifier is a **flat, two-stage retrieval-augmented classifier**, not a hierarchical classifier:

```text
Stage 1: query the legacy flat isco_occupations collection for five candidates.
Stage 2: if top confidence is below 0.92, ask Claude to select one supplied candidate.
```

This task must not claim to test hierarchical retrieval. A later, separately authorised task may recover the later hierarchy-enabled historical implementation, but only after this literal original pipeline has been documented.

The WISCO v2 development split is a new input to the historical classifier. It is not the old paper's original test set, so this is an **observed development-only historical-pipeline result**, not a reproduction of the paper's original accuracy. The 18,747-row WISCO heldout split is forbidden.

No current evidence is replaced, deleted, reinterpreted, or promoted by this task. The approved original paper and all of its references remain untouched.

## Branch and source commitments

### Repository

Use only:

```text
https://github.com/SrkSubbiah89/Multilingual-LFS-Project.git
```

Keep the remote name `origin`. Do not create another repository, modify remote settings, alter protection, make a tag, create a pull request, merge, rebase, reset, clean, stash, pull, or force-push.

### New branch

Create exactly one branch:

```text
reviewer2-literal-legacy-llm-wisco-dev-preflight-20260810
```

It must start from:

```text
reviewer2-phase1-conference1-final-evidence-alignment-20260810
@ 1c62efaec4495f183fcfbe9b011ce5a73e7e846e
```

The historical source commit is read-only evidence. Do not merge it, cherry-pick it, or replace current source files with it.

### Required Git procedure

```bash
REPO_URL="https://github.com/SrkSubbiah89/Multilingual-LFS-Project.git"
BASE_BRANCH="reviewer2-phase1-conference1-final-evidence-alignment-20260810"
BASE_SHA="1c62efaec4495f183fcfbe9b011ce5a73e7e846e"
LEGACY_SHA="824fcf235ae2f8787706cf479a07620519c914de"
TASK_BRANCH="reviewer2-literal-legacy-llm-wisco-dev-preflight-20260810"

git remote get-url origin
git status --porcelain
git fetch origin "$BASE_BRANCH"
git fetch origin "$LEGACY_SHA"
git rev-parse "origin/$BASE_BRANCH"
git cat-file -e "$LEGACY_SHA^{commit}"
git show -s --format='%H%n%P%n%s' "$LEGACY_SHA"
git ls-remote --heads origin "$TASK_BRANCH"
git switch --create "$TASK_BRANCH" "origin/$BASE_BRANCH"
git status --porcelain
```

Required outcomes:

1. `origin` equals `REPO_URL`.
2. The working tree is clean before branch creation.
3. `origin/$BASE_BRANCH` equals `BASE_SHA`.
4. The recovered historical commit equals `LEGACY_SHA` and has the expected subject.
5. The task branch does not already exist. If it exists, stop and report its SHA; never reuse or overwrite it.
6. The task branch is created directly from the verified base.

At the end:

```bash
git status --porcelain
git diff --name-only
git add <only-task-script-tests-and-final-report-files>
git diff --cached --name-only
git diff --cached --check
git diff --cached --stat
git commit -m "Add literal legacy LLM WISCO development preflight"
git status --porcelain
git push --set-upstream origin "$TASK_BRANCH"
git status --porcelain
```

If a non-task file is staged, unstage only that path using `git restore --staged <path>` and investigate. Never use `git reset`.

## Exact historical decision policy

Inspect these files directly from `LEGACY_SHA` and quote the relevant lines in the final report:

```text
backend/agents/isco_classifier.py
backend/rag/vector_store.py
backend/llm/llm_client.py
```

The task is correct only if inspection confirms every one of the following, before any implementation or live activity:

| Historical property | Required literal value |
|---|---|
| Classifier flow | direct `VectorStore.search(job_title, top_k=top_k)` followed by conditional LLM selection |
| Candidate count | 5 |
| High-confidence condition | top candidate confidence `>= 0.92` |
| High-confidence result | return semantic top candidate without an LLM call |
| Low-confidence result | call `get_llm(TaskType.CRITICAL)` once and select only from supplied candidates |
| Legacy model route | `anthropic/claude-3-5-sonnet-20241022` |
| Temperature | `0.0` |
| Prompt evidence | job title, language/context, candidate code, English/Arabic title, level, score, description |
| Accepted LLM response | JSON containing `selected_code` and `reasoning` |
| Validation | selected code must be one of the five supplied candidates |
| Historical malformed/invalid response path | semantic top candidate with the legacy fallback reasoning |
| Method labels | `semantic` and `llm_ranked` |
| Legacy index | `isco_occupations`, using the historical VectorStore implementation and its historical embedding configuration |

If any item differs from direct historical inspection, stop. Do not silently choose a more convenient later version.

## Isolation requirement

The original VectorStore can create or populate `isco_occupations`; it must never point at the current local Qdrant instance or any existing evidence collection.

Create an isolated, disposable local environment solely for this task:

1. Create a detached Git worktree at `LEGACY_SHA`; do not edit it.
2. Run any required Qdrant service on a distinct local port and a new task-specific data directory. It must be visibly different from the production/current local Qdrant endpoint.
3. Point only the detached historical worktree and its WISCO adapter at that isolated endpoint.
4. Permit the historical code to construct its own legacy `isco_occupations` collection there, exactly as it historically would.
5. Never connect the literal historical process to `localhost:6333`, current `ilo2021_v1` collections, or the current project database.
6. Record isolated endpoint, data-directory identifier, collection name, collection point count, embedding model identifier, and collection creation/population log evidence.
7. Destroy neither the isolated output nor current collections during the task. Keep all generated live artefacts Git-ignored and task-specific.

This is environment isolation, not a decision-logic modification. The historical classifier source and decision flow must remain byte-identical in the detached worktree.

### Cache and dependency gate

The historical VectorStore requires its own historical embedding model. No model may be downloaded in this task.

Before the collection is created, verify the required historical embedding model is already locally cached. If it is not cached, set the task status to `LITERAL_LEGACY_LLM_WISCO_DEV_PREFLIGHT_READY: no`, report the missing dependency, and stop. Do not download a model or substitute another embedder.

## WISCO adapter scope

The historical classifier was not written for WISCO CSV input. Add the smallest new adapter outside the detached historical worktree, for example under a new task-specific `eval/legacy824/` directory.

The adapter may:

- read only the 2,013-row WISCO v2 development split;
- map `case_id`, `language`, `occupation_text`, and gold ISCO code into a one-row-at-a-time call to the unmodified historical classifier;
- write a new Git-ignored output CSV with the historical output and an adjacent input identity;
- calculate descriptive exact four-digit accuracy after every row has completed successfully; and
- retain raw classifier outputs, semantic candidate list, top semantic confidence, whether the LLM fired, the five candidate codes, selected code, reasoning, method label, latency, and exceptions.

The adapter must not:

- alter candidate ordering, confidence values, threshold, top-k value, prompt text, model route, temperature, JSON parser, candidate validation, fallback policy, or method label;
- call a hierarchy engine;
- call the current `ISCOClassifier`, current `run_eval.py`, official catalogue loader, current official flat comparator, ISIC, ISCED, SRE, reranker retry code, or current Qdrant deadline pool;
- read, write, list, hash, or evaluate any heldout row;
- deduplicate, translate, rewrite, enrich, or otherwise alter WISCO occupation text before it reaches the historical classifier; or
- create a model-selection sweep.

Store input-to-output identifiers only in the adapter. The historical classifier receives exactly its historical call inputs: job title, optional language/context, and `top_k=5`.

## LLM approval and live-call boundary

The literal route is Claude 3.5 Sonnet, not Ollama. Do not substitute a local model, a different provider, a mock, or a model fallback for the live historical run.

The task has two phases:

### Phase A: no-cost preflight

Complete all repository, source-identity, dependency-cache, dataset, isolation, adapter, and hermetic-test checks. The LLM is mocked only in tests.

### Phase B: one live development run

Phase B is allowed only after all Phase A gates pass and a valid `ANTHROPIC_API_KEY` is already configured locally. Do not print, log, commit, or disclose the key.

Immediately before the first network LLM request, stop and show the operator:

```text
Literal historical run ready:
- model: anthropic/claude-3-5-sonnet-20241022
- temperature: 0.0
- scope: WISCO development split only (2,013 records)
- LLM is called only when semantic confidence < 0.92
- maximum possible calls: 2,013
- no heldout rows will be read
- isolated legacy Qdrant endpoint only
```

Require the operator's explicit confirmation in the Claude Code session before launching the run. If confirmation is absent, stop after Phase A and report `LITERAL_LEGACY_LLM_WISCO_DEV_PREFLIGHT_READY: no` with reason `awaiting_live_llm_confirmation`.

After confirmation, run the adapter exactly once. There is no evaluator retry, no per-row LLM retry, and no provider fallback. If a row raises an LLM/API/parse exception, preserve its raw error, stop the run, mark the task `no`, and do not rerun.

## Preservation gates

Before Phase A and again after every live operation, prove byte identity for:

1. Task 36 raw official flat and strict-hierarchical WISCO CSVs;
2. Task 37 and Task 37.1 analysis reports and result JSON;
3. Task 38 documentation-only evidence-alignment files;
4. official ILO catalogue and official collection-builder source;
5. B1 frozen configuration and B2 sweep safety files; and
6. all existing historical Task 24–38 evidence artefacts enumerated by the latest Task 38 report.

Use checksums for Git-ignored artefacts and `git diff`/hashes for tracked files. The current Qdrant instance and every official or legacy collection in it must be unchanged, because this task must not connect to it.

## Dataset gate

Verify, without opening any heldout-row content:

```text
WISCO v2 package hash matches the established package hash.
Total records: 20,760.
Development records: 2,013.
Heldout records: 18,747.
Development/heldout leakage: zero.
Development codes: syntactically valid ISCO-08 gold labels.
```

Only the development input file may be read by the adapter. Log a file-access manifest proving this.

## Required hermetic tests

Add tests for the new adapter/isolation glue. Mock all LLM, Qdrant, embedding, filesystem-heavy, and subprocess dependencies. Tests must never create a real collection, load an embedding model, call Anthropic, or read WISCO data.

At minimum:

1. Historical source identity check rejects a changed threshold, top-k, model route, temperature, prompt field, parser condition, method label, or fallback behavior.
2. Adapter calls the historical classifier once per supplied development record with unchanged occupation text and `top_k=5`.
3. Confidence exactly 0.92 does not call the mocked LLM.
4. Confidence above 0.92 does not call the mocked LLM.
5. Confidence below 0.92 calls the mocked LLM exactly once.
6. The LLM receives exactly five ordered candidates and can choose only among them.
7. A valid `selected_code` preserves historical `llm_ranked` behavior.
8. Invalid JSON, an out-of-candidate code, and an LLM exception preserve the historical fallback behavior in the classifier test, while the development-run adapter converts any such live condition into a stopped preflight result.
9. The adapter rejects a row with a malformed or non-four-digit gold code rather than modifying it.
10. The adapter refuses any heldout path or row count.
11. Isolation configuration refuses port 6333 and any endpoint matching current Qdrant configuration.
12. The adapter does not import current `run_eval.py`, current classifiers, hierarchy engine, official catalogue, ISIC, ISCED, or SRE.
13. The historical worktree source files are byte-identical to `git show LEGACY_SHA:<path>` before and after the preflight.
14. The final output schema retains sufficient per-row raw evidence to audit the conditional decision.

Run focused tests first. Then run:

```text
python -m pytest backend/tests eval/ -q
```

Record literal output. Any test failure blocks Phase B.

## Phase A gates

All gates must pass, in this order:

1. **Git gate**: verified base, recovered `LEGACY_SHA`, clean new branch.
2. **Historical-source gate**: all values in the exact-policy table verified by direct source inspection.
3. **Preservation gate**: all protected evidence byte-identical.
4. **Dataset gate**: correct WISCO package, exact development count, no heldout read.
5. **Cache gate**: historical embedding model already cached.
6. **Isolation gate**: detached unmodified worktree and distinct Qdrant service/port/data path; no connection to current Qdrant.
7. **Legacy-index gate**: historical `isco_occupations` collection built only in isolated Qdrant; collection provenance and point count logged.
8. **Adapter gate**: static and hermetic tests prove no decision-policy substitution and no imports from prohibited current evaluation components.
9. **Full-test gate**: full suite is green.
10. **API-readiness gate**: exact Claude route is configured, but no API call has yet occurred.
11. **Live-confirmation gate**: explicit operator confirmation is recorded.

Failure of any gate means stop. Do not run WISCO, alter the historical logic, switch models, create a fallback mode, or attempt a repair.

## Development-run gates

After explicit confirmation, allow one live adapter run against the 2,013 development rows only.

The run passes only if:

1. exactly 2,013 rows finish;
2. each row has a preserved `case_id`, language, raw job title identity, gold code, predicted code, method, top semantic score, ordered candidates, LLM-fired boolean, and elapsed time;
3. every direct semantic decision has top confidence at least 0.92 and method `semantic`;
4. every reranked decision has top confidence below 0.92, exactly one LLM invocation, method `llm_ranked`, and a selected code among its five candidates;
5. no LLM error, timeout, invalid JSON, out-of-candidate selection, malformed output, or unrecorded fallback occurs;
6. the isolated Qdrant collection remains internally stable after initial construction;
7. current Qdrant and all prior evidence remain unchanged;
8. no heldout file was opened; and
9. no hierarchy, official-profile, ISIC, ISCED, SRE, reranking-off, or current-model evaluator component was invoked.

If a run gate fails, stop. Do not retry the failed record or rerun the dataset. Preserve the partial output with the failure reason.

## Allowed descriptive result

Only if every development-run gate passes, calculate:

- exact four-digit accuracy: correct / 2,013;
- semantic-only count and accuracy;
- conditional-LLM count and accuracy;
- number and percentage of LLM decisions that differ from the semantic top candidate;
- raw method-label counts;
- median and p95 per-row latency;
- no-error confirmation; and
- a five-row, non-sensitive audit sample containing candidate codes, method, selected code, and correctness.

These are development observations only. Do not compare them statistically to the 18,747-row heldout result, update any paper, claim an improvement, call the result “reproduced,” or select a new configuration using these figures.

## Stop status

Use one exact status:

```text
LITERAL_LEGACY_LLM_WISCO_DEV_PREFLIGHT_READY: yes
LITERAL_LEGACY_LLM_WISCO_DEV_PREFLIGHT_READY: no
```

Set `yes` only after Phase A, explicit live-call confirmation, and all development-run gates pass. Otherwise set `no` and state the exact gate that stopped the task.

## Final report

Create:

```text
Documentation/AI_HANDOFF/CLAUDE_TASK_39_LITERAL_LEGACY_LLM_WISCO_DEV_PREFLIGHT_FINAL_REPORT.md
```

Include:

1. task branch, final SHA, verified base branch/SHA, and historical source SHA;
2. the direct code evidence for every historical policy value;
3. changed files and why none change the historical classifier;
4. isolated-worktree, isolated-Qdrant, and cache/dependency evidence;
5. WISCO development-only access manifest and proof heldout was not read;
6. preflight-gate table and exact stop status;
7. exact test commands and outputs;
8. the operator's explicit live-call confirmation, if Phase B occurred;
9. exact model route, temperature, threshold, top-k, prompt/parser provenance, and call count;
10. literal command used for the single live adapter run;
11. development-run gate table and any allowed descriptive result;
12. hashes proving historical evidence and current collections remained unchanged;
13. confirmation of no collection contact outside the isolated local service;
14. confirmation of no paper/manuscript edit, heldout evaluation, hierarchy test, model substitution, model download, PR, merge, rebase, reset, clean, stash, pull, or force-push;
15. protected branches left untouched and clean working-tree evidence before and after commit; and
16. the explicit boundary: stop after this original flat legacy observation. Any later hierarchy-era legacy policy requires a new independent task.

Commit only new task-specific adapter/test files and this report. Push only the new task branch. Do not open a pull request.

## Claude Code execution prompt

```text
Execute Task 39 exactly as specified in:
Documentation/AI_HANDOFF/PERPLEXITY_TO_CLAUDE_03_LEGACY_LLM_WISCO_DEV_PREFLIGHT.md

The previous modernised Ollama/official-profile brief is superseded. Follow the literal earliest historical ISCO implementation at commit 824fcf235ae2f8787706cf479a07620519c914de, not a later hierarchy implementation and not current reranking logic.

This task has one historical arm only: original flat VectorStore retrieval over the legacy isco_occupations index, top_k=5, semantic return at confidence >=0.92, otherwise one TaskType.CRITICAL Claude 3.5 Sonnet reranking call at temperature 0.0 using the original prompt, parser, candidate validation, method labels, and fallback behavior. Do not call it hierarchical.

Use WISCO development rows only (2,013). Do not read or evaluate the 18,747-row heldout split. Do not modify the historical classifier: use it byte-identically in a detached worktree. Keep it isolated from all current Qdrant collections by using a distinct disposable local Qdrant service, port, and data directory. Do not download any model; if the historical embedder is not cached, stop. Build and test only minimal adapter/isolation glue, then run the full suite before any live operation.

Before the first Anthropic request, stop and ask the operator for explicit live-call confirmation with the exact model, maximum possible call count, development-only scope, threshold, and isolated-Qdrant boundary. If confirmation is not given, complete Phase A only and report status no with awaiting_live_llm_confirmation. After confirmation, make exactly one development-run attempt. No LLM retry, evaluator retry, provider fallback, threshold change, candidate-count change, prompt change, model substitution, hierarchy run, official-profile run, paper update, heldout run, or result-promotion is authorised.

Follow the required branch, Git verification, preservation checks, tests, report, commit, and push process exactly. Push only reviewer2-literal-legacy-llm-wisco-dev-preflight-20260810 and do not open a PR.
```
