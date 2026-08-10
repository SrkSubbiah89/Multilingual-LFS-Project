"""
eval/legacy_runtime40_1/evidence_grades.py

Task 40.1's Grade A/B/C/D archival-evidence hierarchy and classifiers.

Grade definitions (as specified for this task):
  A -- a contemporaneous pip freeze / lockfile / requirements export /
       environment manifest / CI log identifying both this project and an
       exact qdrant-client version.
  B -- a preserved, project-attributable virtual environment, Docker image,
       or container whose package metadata exposes an exact installed
       version and whose provenance/time linkage is documented.
  C -- an archive/backup containing an exact dependency file with a
       verifiable contemporaneous relationship to this project.
  D -- anything that cannot establish a version on its own: a pip cache
       filename, a package's own release date, an unrelated environment, a
       later/non-contemporaneous repository record, memory, inference, or an
       unlinked wheel file.

Only A/B/C evidence may set a recovered version. This module never touches
git, the filesystem, Docker, or the network itself -- callers (typically
built from real `docker image inspect`/`docker history`/extracted
dist-info METADATA output, or Task 40's own git-history findings) supply
plain data, so every classification here is trivially hermetic to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

GRADE_A = "A"
GRADE_B = "B"
GRADE_C = "C"
GRADE_D = "D"

ESTABLISHING_GRADES = frozenset({GRADE_A, GRADE_B, GRADE_C})


@dataclass
class ArchivalCandidate:
    category: str                          # "local_archive" | "python_env" | "docker_image" | "github" | "git_history"
    source_label: str                      # human-readable identity of the evidence item
    attribution_basis: str                 # why (or why not) this is project-attributable
    timestamp: Optional[str]               # ISO-8601 timestamp of the evidence itself, if known
    contemporaneous: bool                  # within a defensible window of LEGACY_SHA
    exact_version: Optional[str]
    evidence_file_path: Optional[str]
    evidence_file_sha256: Optional[str]
    grade: str                             # "A" | "B" | "C" | "D"
    accepted: bool
    reason: str


@dataclass
class ProvenanceRecoveryReport:
    candidates: list[ArchivalCandidate] = field(default_factory=list)
    recovered: bool = False
    recovered_version: Optional[str] = None
    recovered_grade: Optional[str] = None
    reason: str = ""


def classify_docker_image_candidate(
    source_label: str,
    compose_project_label: Optional[str],
    compose_service_label: Optional[str],
    build_matches_repo_dockerfile: bool,
    image_created_iso: str,
    days_after_legacy_sha: float,
    max_contemporaneous_days: float,
    exact_version: Optional[str],
    evidence_file_path: Optional[str],
    evidence_file_sha256: Optional[str],
) -> ArchivalCandidate:
    """
    Classifies one Docker-image-derived candidate as Grade B (accepted) or
    Grade D (rejected). Grade B requires ALL of:
      - project attribution: both compose labels present AND the image's
        build lineage matches this repository's own Dockerfile/
        requirements.txt (never assumed -- caller must have verified this,
        e.g. via `docker history --no-trunc`);
      - a timestamp within `max_contemporaneous_days` of LEGACY_SHA;
      - an exact version recovered from real, extracted package metadata
        (never a guess/inference), with its own file hash recorded.
    Missing any one of these degrades the candidate to Grade D
    (unrelated/insufficiently-attributed environment, non-contemporaneous,
    or no real extracted version) and it is not accepted.
    """
    attribution_ok = bool(compose_project_label) and bool(compose_service_label) and build_matches_repo_dockerfile
    contemporaneous = days_after_legacy_sha <= max_contemporaneous_days
    has_real_version = bool(exact_version) and bool(evidence_file_sha256)

    if attribution_ok and contemporaneous and has_real_version:
        return ArchivalCandidate(
            category="docker_image",
            source_label=source_label,
            attribution_basis=(
                f"docker-compose labels project={compose_project_label!r} "
                f"service={compose_service_label!r}; build layers verified to match "
                f"this repository's own Dockerfile/requirements.txt lineage"
            ),
            timestamp=image_created_iso,
            contemporaneous=True,
            exact_version=exact_version,
            evidence_file_path=evidence_file_path,
            evidence_file_sha256=evidence_file_sha256,
            grade=GRADE_B,
            accepted=True,
            reason=(
                f"project-attributable Docker image built {days_after_legacy_sha:.2f} day(s) "
                f"after LEGACY_SHA; exact version exposed by real extracted dist-info METADATA"
            ),
        )

    reasons = []
    if not attribution_ok:
        reasons.append("insufficient project attribution (missing compose labels or unverified build lineage)")
    if not contemporaneous:
        reasons.append(
            f"not contemporaneous ({days_after_legacy_sha:.2f} days after LEGACY_SHA "
            f"exceeds the {max_contemporaneous_days}-day bound)"
        )
    if not has_real_version:
        reasons.append("no exact version recoverable from real extracted package metadata")

    return ArchivalCandidate(
        category="docker_image",
        source_label=source_label,
        attribution_basis="insufficient" if not attribution_ok else "ok",
        timestamp=image_created_iso,
        contemporaneous=contemporaneous,
        exact_version=exact_version,
        evidence_file_path=evidence_file_path,
        evidence_file_sha256=evidence_file_sha256,
        grade=GRADE_D,
        accepted=False,
        reason="; ".join(reasons) or "rejected",
    )


def git_history_evidence_as_candidates(evidence_rows) -> list[ArchivalCandidate]:
    """
    Converts Task 40's `EvidenceRow` findings (from
    `eval.legacy_runtime40.provenance`) into `ArchivalCandidate` rows for a
    single combined evidence table. All of Task 40's contemporaneous
    requirements.txt rows concluded "no version specifier" -- Grade D, not
    accepted. The one non-contemporaneous exact pin found anywhere
    (requirements-dev.txt, 1.17.0, 5+ months after LEGACY_SHA) is preserved
    for transparency but stays Grade D / not accepted, exactly as Task 40
    itself determined.
    """
    candidates = []
    for row in evidence_rows:
        exact_version = None
        if row.conclusion.startswith("exact pin"):
            exact_version = row.conclusion.split()[-1]
        accepted_here = bool(exact_version) and row.contemporaneous
        grade = GRADE_D
        reason = row.conclusion
        if not row.contemporaneous and exact_version:
            reason = f"non-contemporaneous exact pin ({row.conclusion}); excluded from Grade A/B/C consideration"
        candidates.append(
            ArchivalCandidate(
                category="git_history",
                source_label=f"{row.source} @ {row.commit_or_path}",
                attribution_basis="tracked in this repository's own git history",
                timestamp=None,
                contemporaneous=row.contemporaneous,
                exact_version=exact_version if accepted_here else None,
                evidence_file_path=row.commit_or_path,
                evidence_file_sha256=None,
                grade=grade,
                accepted=False,
                reason=reason,
            )
        )
    return candidates


def determine_final_outcome(candidates: list[ArchivalCandidate]) -> ProvenanceRecoveryReport:
    """
    Combines every candidate (git-history + archival) into one final
    HISTORICAL_QDRANT_PROVENANCE_RECOVERED verdict. Only accepted Grade
    A/B/C candidates with a real exact_version count. A single consistent
    exact version across all such candidates -> recovered=True. Zero such
    candidates, or two or more disagreeing on the exact version -> False.
    """
    report = ProvenanceRecoveryReport(candidates=list(candidates))
    establishing = [
        c for c in candidates
        if c.accepted and c.grade in ESTABLISHING_GRADES and c.exact_version
    ]
    versions = {c.exact_version for c in establishing}

    if len(versions) == 1:
        version = next(iter(versions))
        sources = ", ".join(sorted({c.source_label for c in establishing}))
        best_grade = sorted({c.grade for c in establishing})[0]
        report.recovered = True
        report.recovered_version = version
        report.recovered_grade = best_grade
        report.reason = (
            f"single consistent exact version {version!r} established by Grade "
            f"{'/'.join(sorted({c.grade for c in establishing}))} evidence ({sources}); "
            f"no conflicting Grade A/B/C evidence found"
        )
        return report

    if len(versions) > 1:
        report.recovered = False
        report.reason = f"conflicting exact versions across Grade A/B/C evidence: {sorted(versions)}"
        return report

    report.recovered = False
    report.reason = "no Grade A/B/C evidence found establishing an exact version"
    return report
