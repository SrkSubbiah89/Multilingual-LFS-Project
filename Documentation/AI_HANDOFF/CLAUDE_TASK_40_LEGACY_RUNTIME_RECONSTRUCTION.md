# Task 40: Historical Qdrant Runtime Reconstruction for Literal Legacy ISCO Reproduction

## Purpose and strict boundary

Task 39 recovered and verified the exact earliest legacy ISCO classifier logic at:

```text
824fcf235ae2f8787706cf479a07620519c914de
Add two-stage ISCO-08 classifier agent
```

Task 39 then stopped correctly before any WISCO development row or Anthropic call because the current shared dependency `qdrant-client 1.17.0` does not expose `QdrantClient.search()`, while the byte-identical historical `VectorStore.search()` calls that method.

This task reconstructs the historical **runtime dependency** in a dedicated isolated environment. It does not change, patch, wrap, shim, monkeypatch, translate, fork, or copy-edit the legacy classifier or legacy VectorStore code.

This task does not run WISCO, does not read WISCO data, does not call an LLM or Anthropic, does not download or load an embedding model, does not create a new evaluator, does not compute an accuracy result, and does not update a paper.

The only permitted live query is a bounded read-only smoke query against Task 39's already-isolated legacy Qdrant collection, using the exact historical `VectorStore.search()` implementation from an unmodified detached worktree.

## Required branch and source identities

Repository:

```text
https://github.com/SrkSubbiah89/Multilingual-LFS-Project.git
```

Create exactly one new branch:

```text
reviewer2-legacy-runtime-reconstruction-20260810
```

It must start from the verified Task 39 branch:

```text
reviewer2-literal-legacy-llm-wisco-dev-preflight-20260810
@ 4b6bfd7e90f79c38588fb0f06aedf5eb9a16a618
```

Required immutable historical source:

```text
824fcf235ae2f8787706cf479a07620519c914de
```

Do not merge, rebase, cherry-pick, reset, clean, stash, pull, force-push, alter remotes, modify protected branches, or create a pull request.

Run:

```bash
REPO_URL="https://github.com/SrkSubbiah89/Multilingual-LFS-Project.git"
BASE_BRANCH="reviewer2-literal-legacy-llm-wisco-dev-preflight-20260810"
BASE_SHA="4b6bfd7e90f79c38588fb0f06aedf5eb9a16a618"
LEGACY_SHA="824fcf235ae2f8787706cf479a07620519c914de"
TASK_BRANCH="reviewer2-legacy-runtime-reconstruction-20260810"

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

Stop if:

1. `origin` differs from `REPO_URL`;
2. the working tree is not clean;
3. the base SHA differs;
4. the historical source commit or subject differs;
5. the new branch already exists; or
6. any protected branch would be modified.

## Historical dependency provenance gate

The acceptable runtime version must come from historical evidence, not a guess.

Inspect, at `LEGACY_SHA` and its reachable contemporaneous project history, all likely provenance sources:

```text
requirements.txt
requirements*.txt
pyproject.toml
poetry.lock
Pipfile
Pipfile.lock
setup.py
setup.cfg
Dockerfile*
docker-compose*.yml
.github/workflows/*
README*
Documentation/*
```

Also inspect the legacy commit's parent and the first later commit that changed Qdrant dependencies, if present.

Create a provenance table:

| Evidence source | Commit/path | Exact qdrant-client value or constraint | Conclusion |
|---|---|---|---|

### Allowed outcomes

**Outcome A: exact pin recovered**

An exact historical `qdrant-client==X.Y.Z` value is evidenced by a lockfile, requirements file, container definition, CI environment, archived dependency export, or a directly versioned project record. Only that exact version may be installed in the isolated runtime.

**Outcome B: bounded compatible range recovered**

Historical evidence establishes a bounded range but not an exact pin. The report must state that the runtime is historically compatible, not bit-for-bit reconstructed. Install only the oldest exact candidate in that evidenced range that exposes `QdrantClient.search()`, and disclose the selection rule before installation.

**Outcome C: no defensible version provenance**

No exact pin or bounded version range can be established. Stop with:

```text
HISTORICAL_LEGACY_RUNTIME_READY: no
reason: historical_qdrant_client_version_not_proven
```

Do not search random versions, install a guessed older library, create an API compatibility shim, or continue to a smoke query.

## Isolated environment rules

Create a new task-specific virtual environment or container environment. Do not alter the project-wide Python environment, global pip packages, current Qdrant client, existing isolated Task 39 worktree, or current repository dependency files.

The reconstructed environment must:

1. use a task-specific path/name and be Git-ignored;
2. install only the historically evidenced dependency set needed for the smoke test;
3. record Python version, platform, package versions, and `pip freeze`;
4. prove `qdrant_client.QdrantClient.search` exists before the legacy source is invoked;
5. keep `qdrant-client 1.17.0` untouched outside the dedicated runtime;
6. never write a compatibility method onto a client object;
7. never modify a file in the detached legacy worktree; and
8. never install, upgrade, downgrade, or uninstall anything in the shared environment.

Before any dependency installation, record:

```text
current shared qdrant-client version
new isolated environment path identifier
provenance evidence supporting the candidate version
planned exact package installation command
```

If the required historically evidenced package cannot be installed or cannot import in isolation, stop. Do not substitute a different version or edit source code.

## Detached worktree and collection preservation

Reuse Task 39's detached historical worktree and isolated Qdrant infrastructure only after proving:

1. the worktree source files are byte-identical to `git show LEGACY_SHA:<path>`;
2. the isolated service remains on the documented non-production port(s), not port 6333;
3. the isolated legacy collection is exactly `isco_occupations`;
4. its point count is 124 before the smoke query;
5. no current local Qdrant endpoint, official `ilo2021_v1` collection, production collection, or benchmark output is contacted;
6. current official collection point counts are identical before and after the task; and
7. Task 39's preserved artifacts remain byte-identical.

If Task 39's isolated infrastructure is unavailable, recreate it exactly as Task 39 documented, in a new isolated port/data directory, without changing legacy source. This task may initialise the historical legacy collection only in that isolated service. It may not contact or mutate the current Qdrant instance.

## Smoke-query scope

The smoke test must use the exact `VectorStore.search()` method from:

```text
backend/rag/vector_store.py @ LEGACY_SHA
```

Do not call a current store, hierarchy engine, collection builder, official catalogue loader, current `ISCOClassifier`, current `eval/run_eval.py`, ISIC, ISCED, SRE, reranker retry logic, deadline pool, or WISCO adapter.

The only permitted operation is:

1. construct the legacy VectorStore in the detached worktree using the reconstructed isolated runtime;
2. issue one fixed, non-WISCO, non-sensitive English occupation query chosen before execution, for example `software developer`;
3. call `VectorStore.search(query, top_k=5)` once;
4. record method availability, returned candidate count, candidate codes, titles, scores, elapsed time, and any exception; and
5. verify the isolated collection point count is still 124 afterward.

The test passes only if:

```text
QdrantClient.search exists;
the historical VectorStore.search call returns exactly five ordered candidates;
no exception occurs;
no source file changed;
the isolated legacy collection remains 124 points; and
no non-isolated Qdrant endpoint was contacted.
```

Do not instantiate or invoke the legacy LLM route. Do not call `ISCOClassifier.classify()`. Do not issue a second smoke query or retry the query for any reason.

## Required hermetic tests

Add only the smallest task-specific test and smoke tooling necessary to prove the environment boundary. Tests must use fakes or mocks. No test may load an embedding model, connect Qdrant, create a Docker container, install packages, call an LLM, or read WISCO.

At minimum test:

1. provenance parser rejects unpinned or unsupported dependency evidence;
2. selected version is exactly the historical pin or follows the disclosed bounded-range rule;
3. isolated-runtime command construction never targets the shared environment;
4. endpoint validation rejects port 6333 and the current Qdrant endpoint;
5. source identity check rejects a modified historical VectorStore;
6. smoke tool refuses a non-legacy worktree or current VectorStore;
7. smoke tool invokes only `VectorStore.search(query, top_k=5)`;
8. smoke tool refuses a second query or a retry;
9. WISCO paths and imports are forbidden;
10. no current evaluator, classifier, hierarchy, official-profile, ISIC, ISCED, SRE, or LLM import is permitted; and
11. result artifact schema contains runtime provenance, source hashes, endpoint identity, collection count before/after, and query output.

Run focused tests, then:

```text
python -m pytest backend/tests eval/ -q
```

Record exact output. Any failure blocks all live smoke activity.

## Stop conditions

Stop immediately and set `HISTORICAL_LEGACY_RUNTIME_READY: no` if:

- historical version provenance is insufficient;
- version installation fails;
- `QdrantClient.search` does not exist in the reconstructed environment;
- the legacy source is not byte-identical;
- isolated Qdrant cannot be proven distinct from current Qdrant;
- collection count/provenance differs;
- any test fails;
- any WISCO, LLM, Anthropic, current evaluator, or non-isolated Qdrant contact occurs; or
- the one smoke query raises an exception or does not return exactly five candidates.

Do not fix an error, retry, change code, or continue after a stop condition.

## Success status

Use exactly one:

```text
HISTORICAL_LEGACY_RUNTIME_READY: yes
HISTORICAL_LEGACY_RUNTIME_READY: no
```

`yes` means only that the historical flat retrieval runtime can perform one isolated read-only search with the exact unmodified legacy method. It does not authorise WISCO, LLM calls, accuracy analysis, paper updates, or a claim of reproducing prior results.

## Final report

Create:

```text
Documentation/AI_HANDOFF/CLAUDE_TASK_40_HISTORICAL_LEGACY_RUNTIME_RECONSTRUCTION_FINAL_REPORT.md
```

Include:

1. branch, final SHA, verified base SHA, and historical source SHA;
2. historical dependency provenance table and outcome A/B/C;
3. exact selected version and why it was permitted;
4. isolated environment identity, Python version, package inventory, and confirmation that the shared environment was untouched;
5. source hashes before/after and detached-worktree proof;
6. isolated Qdrant endpoint, collection provenance, and point counts before/after;
7. smoke command, fixed query, method-existence proof, output summary, and elapsed time;
8. all required test commands and literal output;
9. checksums for Task 39 evidence, current official evidence, B1/B2 safety files, and collection-count preservation;
10. confirmation that zero WISCO files/rows, zero LLM/Anthropic calls, zero model downloads, and zero current Qdrant queries occurred;
11. stop status and exact reason if `no`;
12. explicit statement that no paper result, WISCO result, hierarchy result, or accuracy claim exists; and
13. confirmation of clean working tree, untouched protected branches, no PR, and no forbidden Git operation.

Commit only the new task-specific smoke/test tooling and final report. Push only:

```text
reviewer2-legacy-runtime-reconstruction-20260810
```

Do not open a pull request.

## Claude Code execution prompt

```text
Execute Task 40 exactly as specified in:
Documentation/AI_HANDOFF/CLAUDE_TASK_40_LEGACY_RUNTIME_RECONSTRUCTION.md

Start from reviewer2-literal-legacy-llm-wisco-dev-preflight-20260810 @ 4b6bfd7e90f79c38588fb0f06aedf5eb9a16a618 and create only reviewer2-legacy-runtime-reconstruction-20260810.

The objective is limited to restoring the historically evidenced qdrant-client runtime needed by the exact legacy source commit 824fcf235ae2f8787706cf479a07620519c914de. Do not modify, shim, wrap, monkeypatch, or replace the historical VectorStore.search() method or any legacy classifier logic.

Recover dependency provenance first. If an exact historical qdrant-client pin or a defensible bounded range cannot be proven from the historical repository record, stop and report no. Do not guess old package versions or brute-force version searching.

Use a dedicated Git-ignored environment only. Do not modify the shared Python environment or current qdrant-client 1.17.0. Reuse Task 39's isolated Qdrant only after proving it is distinct from production, remains non-6333, and its legacy isco_occupations collection has 124 points. Never contact current Qdrant, official collections, WISCO, an LLM, Anthropic, an embedding-model download, or a current evaluator.

After full tests pass, issue exactly one read-only non-WISCO smoke query through the byte-identical legacy VectorStore.search(query, top_k=5) in the detached historical worktree. The query is fixed before execution, no retries are allowed, and success requires QdrantClient.search to exist and exactly five candidates to return. Stop on any exception or discrepancy.

Follow all preservation checks, reporting, commit, and push instructions in the task document. Push only the new task branch and do not open a PR.
```
