"""
eval/legacy_runtime40_1/real_findings.py

Task 40.1's actual archival-provenance-search findings, reproduced literally
here (values copied verbatim from real, locally-run `git show`/`git log`/
`docker image inspect`/`docker history`/extracted dist-info METADATA output
-- see the Task 40.1 final report for the exact commands). This module
performs no filesystem/Docker/network/git access itself; it only returns the
already-gathered facts as plain data, run through the exact same classifier
code path (`evidence_grades.py`) as any synthetic test.

Search categories covered (per the task's own bounded scope):
  - Narrow local archival search: no sibling project checkout, `.zip`/`.7z`/
    `.tar` backup, or old-project-name directory found anywhere under `C:\\`
    or the user's top-level home directory. (Two manuscript files were found
    under Downloads matching the project name -- a `.tex` source and PDF
    exports of the paper -- and are explicitly EXCLUDED here: they are
    manuscript/paper documents, not a source-code archive, backup, or
    dependency manifest, and this project's standing discipline never opens
    the paper itself. Recorded in the final report's evidence table as
    found-but-excluded, not as a candidate.)
  - Project-attributable Python environment search: no venv/conda
    environment directory (`pyvenv.cfg`, `Scripts/activate`,
    `Lib/site-packages`) attributable to this project found anywhere.
  - Docker metadata search: `docker-backend:latest` (image ID
    `sha256:0ebaafa2ff9e3b658655a8e3eb7011809dff0a65d619f3795ca016be0284ac07`)
    is the only backend-related image found in the entire local image store
    (`docker images -a`, no dangling/untagged competitors at any other
    date) and the only container built from it
    (`cd76e766a588b0432f3f957c65e3f869141ea9478cf3549d66174d8fc6635826`,
    created via `docker create`, never started/run/exec'd). Its
    `com.docker.compose.project`/`com.docker.compose.service` labels and
    `docker history --no-trunc` build layers (verified against this
    repository's own `Dockerfile`/`requirements.txt` at commit `4705afd`,
    the commit that added the Dockerfile, one day after LEGACY_SHA)
    unambiguously attribute it to this project's own backend service. Its
    package metadata was read via `docker export` (no `pull`/`build`/`run`/
    `start`/`exec`/`cp`/`commit`/`rm`/`rmi` -- all explicitly forbidden for
    this task) to a local tar file, from which exactly one
    `qdrant_client-*.dist-info/` directory was found and its `METADATA`
    file extracted: `Version: 1.17.0`.
  - GitHub archival search: `gh release list`, `git ls-remote --tags
    origin`, `gh run list`, `gh issue list --state all`, `gh pr list
    --state all` all returned completely empty -- zero GitHub archival
    evidence exists anywhere for this repository.
"""

from __future__ import annotations

from eval.legacy_runtime40.provenance import real_historical_evidence
from eval.legacy_runtime40_1.evidence_grades import (
    ArchivalCandidate,
    classify_docker_image_candidate,
    git_history_evidence_as_candidates,
)

LEGACY_SHA_COMMIT_DATE = "2026-02-23T23:35:37+04:00"

DOCKER_IMAGE_REPO_TAG = "docker-backend:latest"
DOCKER_IMAGE_ID = "sha256:0ebaafa2ff9e3b658655a8e3eb7011809dff0a65d619f3795ca016be0284ac07"
DOCKER_IMAGE_CREATED_ISO = "2026-02-24T23:48:27+04:00"
DOCKER_IMAGE_CREATED_UTC = "2026-02-24T19:48:27.823487637Z"
DOCKER_COMPOSE_PROJECT_LABEL = "docker"
DOCKER_COMPOSE_SERVICE_LABEL = "backend"
DOCKER_DAYS_AFTER_LEGACY_SHA = 1.0089120370370371  # (image Created) - (LEGACY_SHA commit date), in days
DOCKER_MAX_CONTEMPORANEOUS_DAYS = 7.0  # generous bound; actual finding is ~1.01 days, far inside it

DOCKER_CONTAINER_ID = "cd76e766a588b0432f3f957c65e3f869141ea9478cf3549d66174d8fc6635826"
DOCKER_DIST_INFO_PATH = (
    "usr/local/lib/python3.11/site-packages/qdrant_client-1.17.0.dist-info/METADATA"
)
DOCKER_QDRANT_CLIENT_VERSION = "1.17.0"
DOCKER_METADATA_FILE_SHA256 = "1156249a11208b116a672ccac9fa49c736682283ff1fb68963da57d126f3dfce"

# Excluded from evidence consideration entirely -- recorded for transparency
# only, never treated as a candidate (out of the task's permitted evidence
# categories, and this project's standing discipline never opens the paper).
EXCLUDED_DOWNLOADS_MANUSCRIPT_FILES_NOTE = (
    "Files matching the project name were found under the user's Downloads "
    "folder (a .tex source and PDF exports of the submitted manuscript). "
    "These are manuscript/paper documents, not a source-code archive, "
    "backup, or dependency manifest -- outside every permitted evidence "
    "category for this task -- and were not opened or read, consistent "
    "with this project's standing discipline of never touching the paper "
    "itself. Excluded from the evidence table below."
)


def real_docker_candidate() -> ArchivalCandidate:
    """The one Docker-image-derived candidate found in this task's bounded
    Docker metadata search, classified through the real Grade B/D rule."""
    return classify_docker_image_candidate(
        source_label=f"{DOCKER_IMAGE_REPO_TAG} (image {DOCKER_IMAGE_ID})",
        compose_project_label=DOCKER_COMPOSE_PROJECT_LABEL,
        compose_service_label=DOCKER_COMPOSE_SERVICE_LABEL,
        build_matches_repo_dockerfile=True,  # verified: docker history layers == git show 4705afd:Dockerfile
        image_created_iso=DOCKER_IMAGE_CREATED_ISO,
        days_after_legacy_sha=DOCKER_DAYS_AFTER_LEGACY_SHA,
        max_contemporaneous_days=DOCKER_MAX_CONTEMPORANEOUS_DAYS,
        exact_version=DOCKER_QDRANT_CLIENT_VERSION,
        evidence_file_path=DOCKER_DIST_INFO_PATH,
        evidence_file_sha256=DOCKER_METADATA_FILE_SHA256,
    )


def real_local_archive_candidates() -> list[ArchivalCandidate]:
    """No sibling checkout, archive, or backup was found anywhere in the
    narrow local archival search -- a single Grade D 'no evidence' row
    documents the negative result."""
    return [
        ArchivalCandidate(
            category="local_archive",
            source_label="narrow local archival search (C:\\ top level, user home top level)",
            attribution_basis="n/a -- no candidate file found",
            timestamp=None,
            contemporaneous=False,
            exact_version=None,
            evidence_file_path=None,
            evidence_file_sha256=None,
            grade="D",
            accepted=False,
            reason="no sibling project checkout, archive, or backup found anywhere in scope",
        )
    ]


def real_python_env_candidates() -> list[ArchivalCandidate]:
    """No project-attributable venv/conda environment was found anywhere."""
    return [
        ArchivalCandidate(
            category="python_env",
            source_label="project-attributable Python environment search",
            attribution_basis="n/a -- no candidate environment found",
            timestamp=None,
            contemporaneous=False,
            exact_version=None,
            evidence_file_path=None,
            evidence_file_sha256=None,
            grade="D",
            accepted=False,
            reason="no venv/conda environment directory attributable to this project found anywhere",
        )
    ]


def real_github_candidates() -> list[ArchivalCandidate]:
    """GitHub releases, tags, Actions runs, issues, and PRs were all empty."""
    return [
        ArchivalCandidate(
            category="github",
            source_label="GitHub archival search (releases, tags, Actions runs, issues, PRs)",
            attribution_basis="n/a -- no candidate evidence found",
            timestamp=None,
            contemporaneous=False,
            exact_version=None,
            evidence_file_path=None,
            evidence_file_sha256=None,
            grade="D",
            accepted=False,
            reason="gh release list / git ls-remote --tags / gh run list / gh issue list / gh pr list all empty",
        )
    ]


def real_all_candidates() -> list[ArchivalCandidate]:
    """The complete Task 40.1 evidence table: Task 40's git-history findings
    (reused, not re-derived) plus this task's own archival categories."""
    candidates: list[ArchivalCandidate] = []
    candidates.extend(git_history_evidence_as_candidates(real_historical_evidence()))
    candidates.extend(real_local_archive_candidates())
    candidates.extend(real_python_env_candidates())
    candidates.append(real_docker_candidate())
    candidates.extend(real_github_candidates())
    return candidates
