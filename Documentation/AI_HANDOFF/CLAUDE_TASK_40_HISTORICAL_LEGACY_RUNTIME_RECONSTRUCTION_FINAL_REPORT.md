HISTORICAL_LEGACY_RUNTIME_READY: no
reason: historical_qdrant_client_version_not_proven

# Task 40 Final Report — Historical Qdrant Runtime Reconstruction for Literal Legacy ISCO Reproduction

Produced in response to
`Documentation/AI_HANDOFF/CLAUDE_TASK_40_LEGACY_RUNTIME_RECONSTRUCTION.md`.
This task stops at the historical dependency provenance gate, before
any isolated environment was created, before any package was
installed, and before any smoke query was attempted. No exact
historical `qdrant-client` pin or defensible bounded range exists
anywhere in this repository's reachable history — Outcome C, exactly
as the task's own decision table anticipates.

## 1. Branch, final SHA, verified base SHA, historical source SHA

| | |
|---|---|
| Base branch | `reviewer2-literal-legacy-llm-wisco-dev-preflight-20260810` |
| Required/verified base SHA | `4b6bfd7e90f79c38588fb0f06aedf5eb9a16a618` (confirmed via `git rev-parse origin/$BASE_BRANCH` before branching) |
| New branch | `reviewer2-legacy-runtime-reconstruction-20260810` |
| Final commit SHA | reported in Claude's end-of-turn response, not inside this file |
| Historical source commit (LEGACY_SHA) | `824fcf235ae2f8787706cf479a07620519c914de` — verified present (`git cat-file -e`) and subject confirmed exactly `"Add two-stage ISCO-08 classifier agent"` |

Required Git procedure executed exactly as specified: `git remote get-url origin` matched the required repo URL; `git status --porcelain` was clean; `git fetch origin "$BASE_BRANCH"` and `git fetch origin "$LEGACY_SHA"` both succeeded; `git rev-parse origin/$BASE_BRANCH` matched `BASE_SHA` exactly; `git ls-remote --heads origin "$TASK_BRANCH"` was empty (branch did not already exist); `git switch --create "$TASK_BRANCH" "origin/$BASE_BRANCH"` succeeded, working tree clean immediately after.

## 2. Historical dependency provenance table and Outcome A/B/C

Every provenance source the task lists was inspected directly (`git show <sha>:<path>` / `git log --all` — never assumed, never inferred from a later unrelated file):

| Evidence source | Commit/path | Exact qdrant-client value or constraint | Conclusion |
|---|---|---|---|
| `requirements.txt` | `824fcf2:requirements.txt` (LEGACY_SHA itself) | `qdrant-client` | No version specifier |
| `requirements.txt` | `8bba6e2:requirements.txt` (initial commit, 2026-02-23) | `qdrant-client` | No version specifier (earliest occurrence in the whole repository) |
| `requirements.txt` | `1d03e75:requirements.txt` (LEGACY_SHA's direct parent) | `qdrant-client` | No version specifier |
| `requirements.txt` | `64e0415` / `a00d973` / `825a6e5` / `675121b` / `e20af39` (every later commit that ever touched the file) | `qdrant-client` | No version specifier, unchanged across the file's entire history |
| `docker/docker-compose.yml` | `824fcf2:docker/docker-compose.yml` | `image: qdrant/qdrant:latest` | Not a client-library pin; a floating tag gives no version constraint even for the server |
| `Dockerfile` (root) | `4705afde:Dockerfile` (added 2026-02-24, one day after LEGACY_SHA) | `RUN pip install --no-cache-dir -r requirements.txt tf-keras` | No override; installs the unpinned `requirements.txt` verbatim |
| `README.md` | `824fcf2:README.md` | (no mention) | No evidence |
| `pyproject.toml`, `poetry.lock`, `Pipfile`, `Pipfile.lock`, `setup.py`, `setup.cfg` | entire repository history | (never existed) | No evidence — confirmed via `git log --all --diff-filter=A` returning zero results for all six filenames |
| `.github/workflows/*` | entire repository history | (never existed) | No evidence |
| `requirements-dev.txt` | `66bac86:requirements-dev.txt` (added 2026-08-06) | `qdrant-client==1.17.0` | **Exact pin found, but not contemporaneous**: dated over five months after LEGACY_SHA's 2026-02-23 commit date, added during the current (Conference I Reviewer #2 response) era for entirely unrelated reasons, and identical to the currently-installed version already proven incompatible in Task 39 — using it as "historical" provenance would be exactly the "guess an old version from an unrelated later file" behavior this task explicitly forbids |

**Outcome: C — no defensible version provenance.** Every genuinely contemporaneous source (everything at or immediately around LEGACY_SHA's own commit date) leaves `qdrant-client` completely unpinned; the only exact-version evidence found anywhere in the repository's full history is dated five-plus months later and reflects a different, current-era environment, not the historical one this task needs to reconstruct.

```text
HISTORICAL_LEGACY_RUNTIME_READY: no
reason: historical_qdrant_client_version_not_proven
```

Per the task's explicit instruction under Outcome C: no version was guessed, no brute-force search across old versions was performed, no API compatibility shim was created, and no smoke query was attempted.

## 3. Exact selected version and why it was permitted

Not applicable — no version was selected. Outcome C explicitly forbids selecting or installing any version.

## 4. Isolated environment identity, Python version, package inventory, shared-environment confirmation

**No isolated environment was created.** The task's own structure gates environment creation behind a successful Outcome A or B; Outcome C stops before that step. The shared project Python environment and its `qdrant-client==1.17.0` installation were never touched by this task — confirmed by the fact that no `pip install`/`pip uninstall`/venv-creation command was ever run in this task's execution, and independently by the full-suite test result (Section 8) passing identically to the pre-task baseline.

## 5. Source hashes before/after; detached-worktree proof

Task 39's detached worktree (`C:/task39_legacy824_worktree`, `824fcf2` detached HEAD) remains registered and was re-verified unchanged in this task via `git diff --quiet 824fcf235ae2f8787706cf479a07620519c914de -- <path>` (run from inside the worktree, exit code 0 for all three files) for:
- `backend/agents/isco_classifier.py`
- `backend/rag/vector_store.py`
- `backend/llm/llm_client.py`

No file in the worktree was read for any purpose beyond this diff check in this task (no import, no execution — the provenance investigation needed only `git show`/`git log` output, not the worktree's own files).

## 6. Isolated Qdrant endpoint, collection provenance, point counts before/after

Task 39's isolated Qdrant container (`task39_isolated_qdrant_20260810`, ports `17333`/`17334`, volume `task39_isolated_qdrant_data_20260810`) was re-verified, not recreated (Task 40 never reached the point where recreation would have been needed):

| Check | Result |
|---|---|
| Container status | `Up` (running throughout, from Task 39) |
| Endpoint | `localhost:17333` — confirmed distinct from `6333` |
| Collections present | exactly `["isco_occupations"]` |
| `isco_occupations` point count | 124 (unchanged from Task 39) |

This task issued **zero** Qdrant queries of any kind, isolated or otherwise — the provenance gate stopped before the "Detached worktree and collection preservation" reuse step was ever reached. The re-verification above used only `get_collections()`/`get_collection()` metadata reads against the isolated endpoint (never the production endpoint), performed as part of this report's own preservation evidence, not as part of any classification or smoke activity.

## 7. Smoke command, fixed query, method-existence proof, output, elapsed time

**Not applicable — no smoke query was attempted.** Per Outcome C's explicit instruction, the task stops before "continue to a smoke query." No `VectorStore.search()` call, no `QdrantClient.search` existence check against the current or any reconstructed environment, and no legacy classifier instantiation occurred anywhere in this task.

## 8. Required test commands and literal output

Focused (new Task 40 tests only):
```bash
python -m pytest eval/legacy_runtime40/ -v
```
Result: `11 passed in 0.11s`.

Full suite:
```bash
python -m pytest backend/tests eval/ -q
```
Result: `2239 passed, 1 deselected, 1 warning in 407.14s` (2,228 Task 39
baseline + 11 new). Zero regressions, zero skipped, zero xfailed.

## 9. Checksums for Task 39 evidence, current official evidence, B1/B2 safety files, collection-count preservation

**160-file preservation snapshot** (Task 24-38 raw-output roots, official catalogue, B1 frozen config, `full130` leakage guard/manifest, WISCO dataset/records/split-manifest, all Task 37/37.1/38 documentation and analyzer source, every prior task's final report through Task 39, and Task 39's own `eval/legacy824/` adapter/test package) — hashed immediately after branching and re-hashed after the provenance investigation and hermetic-test work completed: **zero mismatches, zero missing files, all 160/160 byte-identical.**

**Qdrant point counts** — checked before and after all work, for both the five official `ilo2021_v1` collections and every legacy collection on the **production** instance:

| Collection | Before | After |
|---|---:|---:|
| `isco08_major_groups_ilo2021_v1` | 10 | 10 |
| `isco08_submajor_groups_ilo2021_v1` | 43 | 43 |
| `isco08_minor_groups_ilo2021_v1` | 130 | 130 |
| `isco08_unit_groups_ilo2021_v1` | 436 | 436 |
| `isco08_unit_groups_flat_ilo2021_v1` | 436 | 436 |
| `isco08_major_groups` | 10 | 10 |
| `isco08_submajor_groups` | 43 | 43 |
| `isco08_minor_groups` | 131 | 131 |
| `isco08_unit_groups` | 441 | 441 |
| `isco_occupations` (production) | 124 | 124 |

Exact match, before and after, on every collection. The isolated instance's own `isco_occupations` collection (Section 6) is also unchanged at 124 points.

## 10. Confirmation of zero WISCO, LLM, model-download, or current-Qdrant activity

- **Zero WISCO files or rows read.** No `eval/local_benchmarks/` path was opened; no WISCO adapter (Task 39's `eval/legacy824/adapter.py`) was invoked.
- **Zero LLM/Anthropic calls.** `TaskType.CRITICAL`/`get_llm`/`Crew`/`ISCOClassifier.classify()` were never instantiated or called anywhere in this task.
- **Zero model downloads.** No embedding model was loaded, cached, or downloaded — the provenance investigation is pure git-history reading; no `SentenceTransformer` import occurred anywhere in this task's own code.
- **Zero current-Qdrant queries.** The only live Qdrant reads in this entire task were the isolated-instance (`localhost:17333`) and production-instance (`localhost:6333`) **read-only metadata checks** in Sections 6 and 9, performed solely to prove preservation — no collection was created, populated, or mutated on either instance.

## 11. Stop status and exact reason

```text
HISTORICAL_LEGACY_RUNTIME_READY: no
reason: historical_qdrant_client_version_not_proven
```

## 12. Explicit statement on result claims

**No paper result, WISCO result, hierarchy result, or accuracy claim exists from this task.** This task did not run WISCO, did not call an LLM, did not compute an accuracy figure, and did not update any manuscript or Conference I Reviewer #2 documentation file. Its only output is a provenance determination (Outcome C) and the reusable, hermetically-tested classifier module that produced it.

## 13. Clean working tree, protected branches, no PR, no forbidden Git operation

`git status --porcelain` immediately before this task's commit showed exactly:
```
?? eval/legacy_runtime40/
```
— the new provenance module/test package and nothing else. Working tree was clean immediately before branch creation and remains clean of any other change. No merge, rebase, cherry-pick, reset, clean, stash, pull, force-push, or remote alteration was performed. No protected or prior-task branch was touched. Only `reviewer2-legacy-runtime-reconstruction-20260810` was created, and only that branch is pushed. No pull request was opened.

## Explicit boundary

This task establishes only that no defensible historical `qdrant-client` version can be reconstructed for `824fcf235ae2f8787706cf479a07620519c914de` from this repository's own history. It does not authorize a compatibility shim, a guessed version, WISCO evaluation, an LLM call, or any accuracy claim. Any future attempt at a literal legacy flat-retrieval smoke test would require either external provenance this repository does not itself contain (e.g. an author's personal environment record, if one exists outside this repository), or a separately and explicitly authorized decision to use a non-historical compatibility approach (such as the shim option this task's predecessor, Task 39, presented and the operator declined) — neither of which this task attempts or recommends.
