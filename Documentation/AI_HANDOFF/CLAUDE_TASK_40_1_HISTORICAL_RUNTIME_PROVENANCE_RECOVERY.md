# Task 40.1: Historical Qdrant Runtime Provenance Recovery

## Purpose

Task 40 correctly stopped with:

```text
HISTORICAL_LEGACY_RUNTIME_READY: no
reason: historical_qdrant_client_version_not_proven
```

The historical source commit:

```text
824fcf235ae2f8787706cf479a07620519c914de
Add two-stage ISCO-08 classifier agent
```

uses `QdrantClient.search()`, but the repository contains no contemporaneous exact `qdrant-client` pin or bounded compatible version range. The current later `qdrant-client==1.17.0` record is known incompatible and may not be used as historical provenance.

This task performs one final, bounded archival provenance search. It does not guess a version, install a package, create a virtual environment, modify the historical source, start a service, run Qdrant, query Qdrant, read WISCO, load/download an embedding model, call an LLM, call Anthropic, or run any classifier.

The purpose is only to answer:

> Is there independently attributable contemporaneous evidence of the exact historical `qdrant-client` runtime version?

## Repository and branch

Repository:

```text
https://github.com/SrkSubbiah89/Multilingual-LFS-Project.git
```

Create exactly one branch:

```text
reviewer2-legacy-runtime-provenance-recovery-20260810
```

Verified base:

```text
reviewer2-legacy-runtime-reconstruction-20260810
@ bf725b0b373941e770720fa55e266cab9fee15e9
```

Historical source remains immutable:

```text
824fcf235ae2f8787706cf479a07620519c914de
```

Use:

```bash
REPO_URL="https://github.com/SrkSubbiah89/Multilingual-LFS-Project.git"
BASE_BRANCH="reviewer2-legacy-runtime-reconstruction-20260810"
BASE_SHA="bf725b0b373941e770720fa55e266cab9fee15e9"
LEGACY_SHA="824fcf235ae2f8787706cf479a07620519c914de"
TASK_BRANCH="reviewer2-legacy-runtime-provenance-recovery-20260810"

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

Stop immediately if the remote, base SHA, historical source, branch-existence condition, or clean-tree condition differs.

Do not merge, rebase, reset, clean, stash, pull, force-push, alter remote settings, alter protected branches, or open a pull request.

## Permitted evidence sources

Search only the following bounded sources. The search is read-only.

### Repository-adjacent project artefacts

1. Existing task reports, handoff documents, benchmark manifests, execution logs, and archived outputs under the repository and its known project-specific sibling directories.
2. Earlier source archives, zip files, release bundles, backup copies, or duplicate checkouts whose name and path clearly identify this project.
3. GitHub repository releases, Actions workflow runs, job logs, artifacts, commit attachments, tags, and issue/comment attachments, if publicly or account-accessibly available.
4. Existing project-specific Docker images, stopped containers, or volumes, but only metadata inspection. Do not start, exec into, build, pull, remove, or modify them.
5. Existing project-specific virtual environments or Conda environments only when their directory naming, adjacent source checkout, timestamps, and package metadata jointly attribute them to the original project period.

### Forbidden evidence sources

Do not inspect, print, parse, copy, or hash:

- `.env`, credential files, SSH keys, browser profiles, cloud configuration, API keys, tokens, password stores, or shell history;
- unrelated user directories or unrelated projects;
- personal documents, chats, email, downloads not visibly project-related, or system-wide telemetry;
- pip cache entries unless a project-specific contemporaneous record independently ties that exact wheel/version to the original project environment.

Do not perform a broad unrestricted home-directory search. Use narrow, documented paths and filenames. If a candidate path is ambiguous, do not open it; record it as excluded for privacy/scope reasons.

## Evidence hierarchy

Classify every candidate as one of the following.

| Grade | Evidence requirement | Can establish the historical version? |
|---|---|---|
| A | Contemporaneous `pip freeze`, lockfile, requirements export, environment manifest, CI log, or archived runtime record that identifies both the project and an exact `qdrant-client==X.Y.Z` | Yes |
| B | A preserved project-attributable virtual environment, Docker image, or container whose package metadata exposes an exact installed version and whose provenance/time linkage is documented | Yes, with the stated attribution evidence |
| C | An archive or project backup containing the exact dependency file and a verifiable contemporaneous relationship to the original code | Yes |
| D | Pip cache filename, package release date, unrelated environment, later repository record, memory, inference, or an unlinked wheel | No |

Only Grade A, B, or C evidence can establish a version.

If two Grade A/B/C sources disagree, stop with:

```text
HISTORICAL_QDRANT_PROVENANCE_RECOVERED: no
reason: conflicting_contemporaneous_evidence
```

If no Grade A/B/C source exists, stop with:

```text
HISTORICAL_QDRANT_PROVENANCE_RECOVERED: no
reason: no_attributable_contemporaneous_version_evidence
```

Do not select, install, test, or recommend any candidate version in either failure outcome.

## Required search procedure

### Repository record confirmation

Reproduce Task 40's repository-history conclusion first:

1. Inspect `requirements.txt` across its complete reachable history.
2. Inspect `requirements-dev.txt`, `pyproject.toml`, `poetry.lock`, `Pipfile`, `Pipfile.lock`, `setup.py`, `setup.cfg`, Docker files, Compose files, CI files, README files, and documentation at `LEGACY_SHA` and relevant surrounding commits.
3. Confirm whether each path existed at the historical date.
4. Confirm that the only known exact later `qdrant-client==1.17.0` record is post-historical and incompatible.

This is a preservation check, not a second attempt to infer a version from repository history.

### Narrow local archival search

Search only explicit candidate roots that visibly relate to the project, for example:

```text
<current-repository-parent>/Multilingual-LFS-Project*
<current-repository-parent>/Multilingual_Conversational_AI_Labour_Force_Surveys*
<known project workspace>/archives/
<known project workspace>/backups/
<known project workspace>/artifacts/
```

For each candidate archive/check-out:

1. record path identifier, file name, size, and modification timestamp;
2. establish project attribution before opening internal files;
3. inspect only dependency manifests, package metadata, or execution logs;
4. record a SHA-256 hash of the specific evidence file only;
5. do not extract or copy unrelated files; and
6. never alter the source artefact.

### Project-attributable Python environment search

Inspect an existing virtual/Conda environment only if it is adjacent to an attributable project copy or its own metadata clearly ties it to the original project timeframe.

Permitted metadata reads:

```text
qdrant_client-*.dist-info/METADATA
qdrant_client-*.dist-info/RECORD
pip freeze output already stored inside that environment
conda-meta/qdrant-client-*.json
```

Record only:

```text
environment path identifier
project-attribution basis
relevant timestamps
exact qdrant-client version
SHA-256 of the metadata file
```

Never activate, modify, install into, or execute code from the candidate environment.

### Docker metadata search

List existing images/containers only. For a candidate visibly attributable to the original project, inspect labels, creation timestamps, configured command, mounted project paths, image digest, and image filesystem package metadata only if this can be read without starting or executing the container.

Do not:

```text
docker pull
docker build
docker run
docker start
docker exec
docker cp
docker commit
docker rm
docker rmi
```

An unlabeled generic Python/Qdrant image is Grade D and cannot establish the original project version.

### GitHub archival search

Read-only search is allowed for:

- repository releases and attached assets;
- accessible Actions job logs and artifacts;
- tags;
- issues, comments, and attached project artefacts; and
- commit-associated files.

Do not create an issue, comment, release, workflow, artifact, tag, branch, or PR. Do not edit GitHub content.

## Preservation checks

Before and after the task, verify byte identity for:

1. Task 39 report, detached-worktree metadata, isolated-Qdrant metadata, and generated artefacts;
2. Task 40 report and provenance-classification logic;
3. Task 36 raw official WISCO evidence, Task 37/37.1 analysis evidence, and Task 38 documentation evidence;
4. official ILO catalogue and collection-builder source;
5. B1 frozen configuration and B2 sweep safety files; and
6. every protected artefact enumerated by Task 40's 160-file snapshot.

This task must not contact a Qdrant server. Collection counts may be verified only from already-recorded Task 39/40 metadata or read-only filesystem manifests; do not issue a Qdrant client request.

## Required hermetic tests

Add only narrowly scoped, hermetic provenance-classification tests. No test may inspect a real archive, current filesystem, Docker daemon, network service, Qdrant, WISCO, an embedding model, an LLM, Anthropic, or a secret.

At minimum cover:

1. Grade A exact `pip freeze` evidence passes.
2. Grade B attributable virtual-environment metadata passes only when attribution fields are complete.
3. Grade C project backup with exact dependency file passes.
4. An unpinned requirement fails.
5. A later `qdrant-client==1.17.0` record fails as post-historical.
6. A pip-cache wheel without independent project attribution fails.
7. A generic Docker image fails.
8. Conflicting Grade A/B/C records fail closed.
9. Missing timestamp or project linkage fails.
10. Forbidden evidence path types are rejected without file reads.
11. The final result cannot emit an installation command or selected runtime version after a failed evidence outcome.
12. The archive search manifest excludes forbidden paths and never includes raw file contents.

Run focused tests, then:

```text
python -m pytest backend/tests eval/ -q
```

Any failure blocks the archival conclusion and must be reported.

## Outcome and stop rules

Use exactly one status:

```text
HISTORICAL_QDRANT_PROVENANCE_RECOVERED: yes
HISTORICAL_QDRANT_PROVENANCE_RECOVERED: no
```

Set `yes` only when one exact version is established by Grade A/B/C evidence, with unambiguous project and contemporaneous attribution.

On `yes`, report the exact version and evidence, then stop. Do not install it, create an environment, run a smoke query, or begin Task 40.2.

Set `no` if no qualifying evidence exists, if qualifying records conflict, if a candidate cannot be attributed, or if scope/privacy constraints prevent inspection. State the exact reason, then stop. Do not guess, test, or recommend a version.

## Final report

Create:

```text
Documentation/AI_HANDOFF/CLAUDE_TASK_40_1_HISTORICAL_QDRANT_PROVENANCE_RECOVERY_FINAL_REPORT.md
```

The report must include:

1. branch, final SHA, verified base, and historical source SHA;
2. every searched source category, bounded path/root identifier, and whether it existed;
3. a candidate-evidence table with grade, attribution basis, timestamp, exact version if present, file hash, and accept/reject outcome;
4. explicit privacy exclusions and confirmation that no forbidden source was opened;
5. repository-history confirmation, including why later `1.17.0` is not evidence;
6. tests and exact full-suite output;
7. full preservation-check evidence;
8. exact status and outcome reason;
9. confirmation of zero Qdrant connection, zero environment creation, zero package installation, zero embedding-model activity, zero WISCO read, zero LLM/Anthropic call, zero source modification, and zero GitHub mutation beyond this task branch/report;
10. confirmation that no version was guessed or recommended on a `no` outcome;
11. clean working tree and protected-branch evidence; and
12. the next permitted action:
    - if `yes`: a separately approved Task 40.2 that uses only the recovered version;
    - if `no`: literal runtime reconstruction ends, and any future experiment must be labelled a non-literal historical-code compatibility study.

Commit only task-specific provenance helper/tests and the final report. Push only:

```text
reviewer2-legacy-runtime-provenance-recovery-20260810
```

Do not open a pull request.

## Claude Code execution prompt

```text
Execute Task 40.1 exactly as specified in:
Documentation/AI_HANDOFF/CLAUDE_TASK_40_1_HISTORICAL_RUNTIME_PROVENANCE_RECOVERY.md

Create only reviewer2-legacy-runtime-provenance-recovery-20260810 from reviewer2-legacy-runtime-reconstruction-20260810 @ bf725b0b373941e770720fa55e266cab9fee15e9.

This is a bounded, read-only archival provenance search to establish the exact historical qdrant-client version for legacy source commit 824fcf235ae2f8787706cf479a07620519c914de. Follow the evidence hierarchy exactly. Only Grade A, B, or C contemporaneous, project-attributable evidence can establish a version.

Do not guess or install a version. Do not create an environment, start/query Qdrant, touch WISCO, load/download a model, call a classifier, LLM, or Anthropic. Do not inspect secrets, shell history, browser profiles, personal files, unrelated directories, or broad home-directory contents. Follow the narrow source/path rules in the task.

If exact provenance is recovered, report it and stop without installation. If it is not recovered or evidence conflicts, report no and stop without proposing a version. Run the required hermetic tests and full suite, preserve all prior evidence, commit only task-specific files/report, push only the new branch, and do not open a PR.
```
