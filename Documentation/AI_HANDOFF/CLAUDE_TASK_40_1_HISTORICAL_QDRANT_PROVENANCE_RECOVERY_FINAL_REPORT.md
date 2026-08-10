# Task 40.1 — Historical `qdrant-client` Runtime Provenance Recovery: Final Report

**Task file:** `Documentation/AI_HANDOFF/CLAUDE_TASK_40_1_HISTORICAL_RUNTIME_PROVENANCE_RECOVERY.md`
**Branch:** `reviewer2-legacy-runtime-provenance-recovery-20260810`
**Base:** `reviewer2-legacy-runtime-reconstruction-20260810` @ `bf725b0b373941e770720fa55e266cab9fee15e9` (Task 40's final commit; verified via `git merge-base`)
**LEGACY_SHA (unchanged, reused from Task 39/40):** `824fcf235ae2f8787706cf479a07620519c914de` — commit date `2026-02-23T23:35:37+04:00`

## HISTORICAL_QDRANT_PROVENANCE_RECOVERED: yes

**Recovered version: `qdrant-client==1.17.0`, Grade B evidence.**

**This finding does NOT unblock Task 39.** `qdrant-client 1.17.0` is the exact same version already proven, in Task 39, to lack `QdrantClient.search()` (`hasattr(QdrantClient, "search")` → `False`; replaced by `query_points`/`query`). Recovering this version does not make literal reproduction of LEGACY_SHA's `VectorStore.search()` possible — it independently confirms, via a completely different evidence path than Task 40's, that the incompatibility was present essentially from day one of this project's Docker build history, not introduced later. See "What this outcome means" below before taking any further action.

---

## 1. What was searched, and where (bounded scope as instructed)

| Category | Scope searched | Result |
|---|---|---|
| Narrow local archival search | `C:\` top level; the user's home directory top level (no recursive full-home scan, no shell history, no browser profiles, no personal files) | No sibling project checkout, `.zip`/`.7z`/`.tar` backup, or old-project-name directory found. Two manuscript files were found under `Downloads` (a `.tex` source and PDF exports of the submitted paper) — see §4, Privacy exclusions. |
| Project-attributable Python environment search | Search for `pyvenv.cfg` / `Scripts\activate` / project-referencing `Lib\site-packages` trees attributable to this repository | None found anywhere. |
| Docker metadata search | `docker images -a`, `docker image inspect`, `docker history --no-trunc`, `docker create` + `docker export` (read-only filesystem metadata only — see §6, forbidden-activity confirmation) | One project-attributable backend image found: `docker-backend:latest`. Full findings in §2. |
| GitHub archival search | `gh release list`, `git ls-remote --tags origin`, `gh run list`, `gh issue list --state all`, `gh pr list --state all` | All completely empty. Zero GitHub archival evidence exists anywhere for this repository. |
| Repository-history (git) record | Reused verbatim from Task 40 (`eval/legacy_runtime40/provenance.py::real_historical_evidence()`), re-run through this task's own combined classifier | Reconfirms Task 40's Outcome C exactly — see §3. |

## 2. Docker metadata search — the recovered evidence

- **Image:** `docker-backend:latest`, image ID `sha256:0ebaafa2ff9e3b658655a8e3eb7011809dff0a65d619f3795ca016be0284ac07`. This is the **only** backend-related image anywhere in the local image store — confirmed via `docker images -a` (which also lists dangling/untagged images): no other image, at any date, competes or conflicts.
- **Project attribution:** `docker image inspect` labels: `com.docker.compose.project=docker`, `com.docker.compose.service=backend`, `com.docker.compose.version=5.0.2` — these are this repository's own `docker/docker-compose.yml` labels, not a generic/unrelated image.
- **Build lineage verified, not assumed:** `docker history --no-trunc docker-backend:latest` shows build layers `WORKDIR /app` → `apt-get install gcc libpq-dev` → `COPY requirements.txt .` → `RUN pip install --no-cache-dir -r requirements.txt tf-keras`. This is byte-identical to `git show 4705afdea3932c0327d08d85a5c9220f6ba84509:Dockerfile` — the commit that added the Dockerfile, **`2026-02-24T22:19:10+04:00`, one day after LEGACY_SHA**. `requirements.txt` at that same commit still shows the unpinned `qdrant-client` line (confirmed via `git show 4705afd:requirements.txt`), consistent with Task 40's finding that the file was never pinned anywhere in this repository's history.
- **Timestamp / contemporaneity:** Image `Created`: `2026-02-24T19:48:27.823487637Z` (`2026-02-24T23:48:27+04:00`). Relative to LEGACY_SHA's commit date (`2026-02-23T23:35:37+04:00`), this is **1.0089 days** after — the tightest contemporaneous bound available anywhere in this repository's history, since the backend Dockerfile itself did not exist before that commit.
- **Extraction method (no forbidden Docker command used):** `docker create docker-backend:latest` → container `cd76e766a588b0432f3f957c65e3f869141ea9478cf3549d66174d8fc6635826` (created, **never started, never `exec`'d**). `docker export <container> -o docker_backend_export.tar` (scratchpad-local, ~10.6 GB) — `export`/`create` are not on the task's forbidden list (`pull`, `build`, `run`, `start`, `exec`, `cp`, `commit`, `rm`, `rmi`) and read filesystem contents without executing any code from the image.
- **Package metadata found:** exactly one `qdrant_client-*.dist-info/` directory anywhere in the exported filesystem: `usr/local/lib/python3.11/site-packages/qdrant_client-1.17.0.dist-info/`. Its `METADATA` file was extracted (`tar -xf ... -O <path>`, printed to a local file, never executed) and reads:
  ```
  Metadata-Version: 2.4
  Name: qdrant-client
  Version: 1.17.0
  ```
- **Evidence file hash:** SHA-256 of the extracted `METADATA` file: `1156249a11208b116a672ccac9fa49c736682283ff1fb68963da57d126f3dfce`.

This satisfies the task's Grade B definition in full: *"A preserved project-attributable venv/Docker image/container whose package metadata exposes an exact installed version and whose provenance/time linkage is documented."* No other Grade A/B/C evidence was found anywhere (local archival, Python environment, or GitHub searches all returned nothing; the git-history search — §3 — found only unpinned/non-contemporaneous evidence). There is therefore a single, unambiguous, non-conflicting exact version: `1.17.0`.

## 3. Repository-history (git) record — Task 40's finding, reconfirmed

Re-run verbatim through `eval.legacy_runtime40.provenance.real_historical_evidence()` (unchanged from Task 40) and converted into this task's own evidence table via `eval.legacy_runtime40_1.evidence_grades.git_history_evidence_as_candidates()`: every contemporaneous `requirements.txt` occurrence (LEGACY_SHA itself, the initial commit, LEGACY_SHA's direct parent, and all five later commits that ever touched the file) shows `qdrant-client` with **no version specifier**. The only exact pin found anywhere in git history (`requirements-dev.txt`, `qdrant-client==1.17.0`, added `2026-08-06` — 5+ months after LEGACY_SHA) remains correctly excluded as non-contemporaneous, exactly as Task 40 determined. In isolation, the git-history record alone still yields Outcome C — this is unchanged from Task 40. It is only the Docker-image evidence (§2), a category Task 40 did not search, that changes the final combined outcome.

## 4. Privacy exclusions

- Two files matching the project name were found under the user's `Downloads` folder: a `.tex` manuscript source and PDF exports of the submitted paper. These are manuscript/paper documents — outside every permitted evidence category for this task (not a source-code archive, backup, or dependency manifest) — and were **not opened or read**, consistent with this project's standing discipline of never touching the paper itself. They are not represented as a candidate anywhere in the evidence table (`eval/legacy_runtime40_1/test_evidence_grades.py::test_excluded_manuscript_note_documents_exclusion_not_a_candidate` asserts this).
- No shell history, browser profile, credential store, or unrelated personal directory was inspected at any point.
- No broad/recursive home-directory scan was performed — only the top level of `C:\` and the top level of the user's home directory, per the task's explicit bound.

## 5. Candidate-evidence table (full)

| Category | Source | Grade | Contemporaneous | Exact version | Accepted | Reason |
|---|---|---|---|---|---|---|
| git_history | `requirements.txt` @ 5 distinct contemporaneous commits (LEGACY_SHA, initial commit, parent, +3 later) | D | yes | — | no | no version specifier |
| git_history | `requirements-dev.txt` @ `66bac86` (2026-08-06) | D | no | 1.17.0 | no | non-contemporaneous exact pin; excluded |
| git_history | `docker/docker-compose.yml`, `Dockerfile`, `README.md`, `pyproject.toml`/lockfile family, `.github/workflows/*` | D | yes | — | no | no evidence (server-image tag only / never existed) |
| local_archive | narrow local archival search | D | — | — | no | no sibling checkout, archive, or backup found |
| python_env | project-attributable Python environment search | D | — | — | no | no venv/conda environment found |
| **docker_image** | **`docker-backend:latest` (image `0ebaafa2ff9e...`)** | **B** | **yes (1.01 days)** | **1.17.0** | **yes** | **project-attributable, contemporaneous, exact version from real extracted metadata** |
| github | releases / tags / Actions runs / issues / PRs | D | — | — | no | all empty |

Full table (including exact source labels, attribution basis, and evidence file hashes) is produced programmatically by `eval.legacy_runtime40_1.real_findings.real_all_candidates()` and is exercised by the test suite (§7).

## 6. Confirmation of zero forbidden activity

- No environment was created; no package was installed; no `pip install` of any kind was run.
- No Qdrant server was started, queried, or connected to (the isolated `task39_isolated_qdrant_20260810` container and `lfs_qdrant` were left exactly as-is, never touched).
- No WISCO data, model, classifier, LLM, or Anthropic call was made anywhere in this task.
- No secrets, shell history, browser profile, credential, or unrelated personal file was inspected.
- Docker commands used were limited to: `docker images -a`, `docker ps -a`, `docker image inspect`, `docker history`, `docker create`, `docker export`. None of `pull`, `build`, `run`, `start`, `exec`, `cp`, `commit`, `rm`, `rmi` were used at any point.
- No version was guessed, inferred, or brute-forced. The recovered version (`1.17.0`) came from a real, extracted `dist-info/METADATA` file's literal `Version:` field — the same standard the task requires for Grade A/B/C evidence.

## 7. Tests and full-suite output

New: `eval/legacy_runtime40_1/` (`__init__.py`, `evidence_grades.py`, `real_findings.py`, `test_evidence_grades.py`) — 15 hermetic tests (no git/Docker/filesystem/network access in the test file itself; `real_findings.py` returns only already-gathered plain data). Covers: Grade B acceptance/rejection on each of the 4 independent criteria (attribution, build-lineage verification, contemporaneity, real-version-with-hash), git-history-row conversion (unpinned → not accepted, non-contemporaneous pin → not accepted), combined-outcome logic (single Grade B recovers; zero establishing evidence does not recover; conflicting versions do not recover; Grade D rows never contribute even if they happen to carry a version string), and the real Task 40.1 findings run through the identical code path used by every synthetic test.

```
eval/legacy_runtime40_1/test_evidence_grades.py: 15 passed
```

Full suite (`pytest backend/tests eval/ -q`):

```
2254 passed, 1 deselected, 1 warning in 477.18s
```

(Task 40's baseline was `2239 passed, 1 deselected, 1 warning`; `2239 + 15 = 2254` — zero regressions.) The single atexit `colorama`/`crewai` teardown exception printed after the summary line is pre-existing, harmless interpreter-shutdown noise (stream-close ordering in a third-party dependency), not a test failure — the reported result line `2254 passed, 1 deselected, 1 warning` is authoritative.

## 8. Preservation evidence (before/after)

`git status --porcelain` before and after this task's work shows only `eval/legacy_runtime40_1/` as new/untracked — no existing file was modified. Spot-checked via SHA-256 (unchanged from before this task started):

| File | SHA-256 |
|---|---|
| `eval/legacy824/historical_loader.py` | `fa34538b8eaa80182934437f11ab9b536947ef6b2a9a485f268ed2fdacc89bce` |
| `eval/legacy824/adapter.py` | `779a18084132d82006a97acd01a8adb77757f682d86afee6e501588c03430564` |
| `eval/legacy824/dataset_gate.py` | `88c22153207d8750bb538764ebf104aa6a0b62518dee01a607129c53eccc4c9b` |
| `eval/legacy_runtime40/provenance.py` | `6b656249e1905ed07374a60efa452c3102328e32bddd0a3417099e24cdadc5e4` |
| `eval/legacy_runtime40/__init__.py` | `5a342dc8f538d785e0ae75f0181f6c43aa744736b3f6667e0940280e1c5409ad` |
| `backend/rag/official_isco08_catalogue.py` | `02c1d0dae8d58fe7fb5bda6d1e0824b912a8f79779923e94c927f9f9741197b5` |
| `backend/rag/build_official_isco08_collections.py` | `63c061b4a663841cd629684cbc7e3fdc9dbc2542eb58d942b15e447e95e4480d` |
| `eval/configs/b1_frozen.json` | `f479ffef7e2e3b8342ffdbc11712df3cc2e78a405c20f09a940a2ec840a4967b` |
| `eval/PRE_RUN_B2_CHECKLIST.md` | `d51d7bac13f1cfa48983609e9de54bf304cbe0aee18c4439b8d0c570c32cb5d8` |
| `eval/test_run_eval_b2.py` | `291152b2b34779dbd4254676f68dea49d86ad8b8e06f05d0de4f5fea00c75711` |
| `Documentation/AI_HANDOFF/CLAUDE_B1_BASELINE_STATUS_REPORT.md` | `be3b0e5133a54972c2ed55e182415a950b336927a0efb9528d987c00c4105499` |
| `Documentation/AI_HANDOFF/CLAUDE_B2_INTEGRATION_REPORT.md` | `34cc0307cbc806a6cd25cb03719797386355e54d79f89e678c8556d7f75a6540` |

Task 39's detached worktree (`C:/task39_legacy824_worktree`, HEAD detached at `824fcf2`), Task 39's isolated Qdrant container (`task39_isolated_qdrant_20260810`) and its `isco_occupations` collection, and Task 40's report/evidence were all confirmed still present and untouched. This task's own new Docker artifacts — the created-but-never-started container `cd76e766a588...` and the local export tar (`docker_backend_export.tar`, scratchpad-only, not part of the repository) — are left in place for the same reason Task 39 left its isolated Qdrant container in place: `docker rm` is on the forbidden list, and deleting the container would destroy the traceable link between this report's evidence table and its literal source.

## 9. Clean-tree / protected-branch evidence

- `git status --porcelain` at task start (immediately after `git switch --create reviewer2-legacy-runtime-provenance-recovery-20260810`) and again at the time of writing this report: clean except for the new `eval/legacy_runtime40_1/` directory.
- Branch created via `git switch --create`, base verified via `git merge-base HEAD bf725b0b373941e770720fa55e266cab9fee15e9` = `bf725b0b373941e770720fa55e266cab9fee15e9` exactly (the exact SHA specified).
- No merge, rebase, reset, clean, stash, pull, or force-push was ever run. `master`, `reviewer2-legacy-runtime-reconstruction-20260810`, and `reviewer2-literal-legacy-llm-wisco-dev-preflight-20260810` all remain exactly as they were (confirmed via `git branch -a`).
- No PR was opened.

## 10. What this outcome means — the essential caveat

`HISTORICAL_QDRANT_PROVENANCE_RECOVERED: yes` answers exactly the question this task asked: *is there Grade A/B/C, contemporaneous, project-attributable evidence establishing an exact historical `qdrant-client` version?* Yes — `1.17.0`, from the Docker image built one day after LEGACY_SHA.

It does **not** mean literal reproduction of LEGACY_SHA's classifier (Task 39's goal) is now possible. `1.17.0` is the identical version already proven, in Task 39, to lack `QdrantClient.search()`. The practical implication is the opposite of unblocking: this project's own unpinned dependency-installation process resolved to the incompatible version within roughly 24 hours of LEGACY_SHA's commit — there is no recovered evidence, anywhere searched, of a `qdrant-client` version installed via this project's own documented process that both (a) is contemporaneous with LEGACY_SHA and (b) exposes `.search()`.

## 11. Next permitted action

Per this task's own explicit boundary, no installation, environment creation, smoke query, or "Task 40.2" follow-on is authorized as a result of this "yes" — a separately approved task is required for any further action. Given §10, any such follow-on task cannot be framed as "install the recovered version and run the literal reproduction" (that would reproduce the known-broken state). Consistent with Task 39's own closing framing, any future experiment involving LEGACY_SHA's classifier logic would need to be scoped as a **non-literal historical-code compatibility study** (e.g., an explicitly-labelled, disclosed adaptation layer) rather than a byte-identical reproduction — and that scoping decision belongs to the operator, not to this task.

---

*Report ends. This task's own boundary now applies: stop here.*
