"""
Tests for eval/legacy_runtime40_1/evidence_grades.py and real_findings.py
(Task 40.1).

Fully hermetic: no git, Docker, filesystem, or network call anywhere in
this file -- every candidate is hand-constructed, or (for the
`real_*`-prefixed tests) built from the literal, already-gathered facts in
`real_findings.py`, which itself performs no live access.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.legacy_runtime40_1.evidence_grades import (  # noqa: E402
    ArchivalCandidate,
    classify_docker_image_candidate,
    determine_final_outcome,
    git_history_evidence_as_candidates,
)
from eval.legacy_runtime40.provenance import EvidenceRow  # noqa: E402
from eval.legacy_runtime40_1 import real_findings  # noqa: E402


# ---------------------------------------------------------------------------
# classify_docker_image_candidate
# ---------------------------------------------------------------------------

def _docker_kwargs(**overrides):
    kwargs = dict(
        source_label="example-backend:latest (image abc123)",
        compose_project_label="docker",
        compose_service_label="backend",
        build_matches_repo_dockerfile=True,
        image_created_iso="2026-02-24T23:48:27+04:00",
        days_after_legacy_sha=1.0,
        max_contemporaneous_days=7.0,
        exact_version="1.7.3",
        evidence_file_path="usr/local/lib/python3.11/site-packages/qdrant_client-1.7.3.dist-info/METADATA",
        evidence_file_sha256="deadbeef" * 8,
    )
    kwargs.update(overrides)
    return kwargs


def test_docker_candidate_accepted_grade_b_when_all_criteria_met():
    row = classify_docker_image_candidate(**_docker_kwargs())
    assert row.grade == "B"
    assert row.accepted is True
    assert row.exact_version == "1.7.3"


def test_docker_candidate_rejected_when_missing_compose_labels():
    row = classify_docker_image_candidate(**_docker_kwargs(compose_project_label=None))
    assert row.grade == "D"
    assert row.accepted is False
    assert "attribution" in row.reason


def test_docker_candidate_rejected_when_build_lineage_not_verified():
    row = classify_docker_image_candidate(**_docker_kwargs(build_matches_repo_dockerfile=False))
    assert row.grade == "D"
    assert row.accepted is False


def test_docker_candidate_rejected_when_not_contemporaneous():
    row = classify_docker_image_candidate(**_docker_kwargs(days_after_legacy_sha=400.0))
    assert row.grade == "D"
    assert row.accepted is False
    assert "not contemporaneous" in row.reason


def test_docker_candidate_rejected_when_no_exact_version():
    row = classify_docker_image_candidate(**_docker_kwargs(exact_version=None, evidence_file_sha256=None))
    assert row.grade == "D"
    assert row.accepted is False
    assert "no exact version" in row.reason


# ---------------------------------------------------------------------------
# git_history_evidence_as_candidates
# ---------------------------------------------------------------------------

def test_git_history_evidence_conversion_marks_unpinned_rows_grade_d_not_accepted():
    rows = [EvidenceRow("requirements.txt", "abc:requirements.txt", "qdrant-client",
                         "no version specifier", contemporaneous=True)]
    candidates = git_history_evidence_as_candidates(rows)
    assert len(candidates) == 1
    assert candidates[0].grade == "D"
    assert candidates[0].accepted is False
    assert candidates[0].exact_version is None


def test_git_history_evidence_conversion_marks_noncontemporaneous_pin_not_accepted():
    rows = [EvidenceRow("requirements-dev.txt", "later:requirements-dev.txt", "qdrant-client==1.17.0",
                         "exact pin 1.17.0", contemporaneous=False)]
    candidates = git_history_evidence_as_candidates(rows)
    assert candidates[0].accepted is False
    assert "non-contemporaneous" in candidates[0].reason


# ---------------------------------------------------------------------------
# determine_final_outcome
# ---------------------------------------------------------------------------

def test_determine_final_outcome_single_grade_b_candidate_recovers_version():
    candidates = [classify_docker_image_candidate(**_docker_kwargs())]
    report = determine_final_outcome(candidates)
    assert report.recovered is True
    assert report.recovered_version == "1.7.3"
    assert report.recovered_grade == "B"


def test_determine_final_outcome_no_establishing_evidence_returns_not_recovered():
    candidates = git_history_evidence_as_candidates(
        [EvidenceRow("requirements.txt", "abc:requirements.txt", "qdrant-client",
                      "no version specifier", contemporaneous=True)]
    )
    report = determine_final_outcome(candidates)
    assert report.recovered is False
    assert report.recovered_version is None
    assert "no Grade A/B/C evidence" in report.reason


def test_determine_final_outcome_conflicting_versions_returns_not_recovered():
    candidates = [
        classify_docker_image_candidate(**_docker_kwargs(exact_version="1.7.3", source_label="image-a")),
        classify_docker_image_candidate(**_docker_kwargs(exact_version="1.9.0", source_label="image-b")),
    ]
    report = determine_final_outcome(candidates)
    assert report.recovered is False
    assert "conflicting" in report.reason


def test_determine_final_outcome_grade_d_candidates_never_contribute():
    grade_d_with_version = ArchivalCandidate(
        category="python_env", source_label="unrelated venv", attribution_basis="none",
        timestamp=None, contemporaneous=True, exact_version="9.9.9",
        evidence_file_path=None, evidence_file_sha256=None, grade="D", accepted=False,
        reason="not project-attributable",
    )
    report = determine_final_outcome([grade_d_with_version])
    assert report.recovered is False


# ---------------------------------------------------------------------------
# real_findings -- the actual Task 40.1 investigation, run through the exact
# same classifier code path as any synthetic test above.
# ---------------------------------------------------------------------------

def test_real_docker_candidate_is_accepted_grade_b_with_version_1_17_0():
    row = real_findings.real_docker_candidate()
    assert row.grade == "B"
    assert row.accepted is True
    assert row.exact_version == "1.17.0"
    assert row.evidence_file_sha256 == real_findings.DOCKER_METADATA_FILE_SHA256


def test_real_all_candidates_has_exactly_one_establishing_candidate():
    candidates = real_findings.real_all_candidates()
    establishing = [c for c in candidates if c.accepted and c.grade in ("A", "B", "C")]
    assert len(establishing) == 1
    assert establishing[0].category == "docker_image"
    non_establishing_categories = {c.category for c in candidates if not c.accepted}
    assert non_establishing_categories == {"git_history", "local_archive", "python_env", "github"}


def test_real_findings_final_outcome_recovers_1_17_0_with_grade_b():
    report = determine_final_outcome(real_findings.real_all_candidates())
    assert report.recovered is True
    assert report.recovered_version == "1.17.0"
    assert report.recovered_grade == "B"


def test_excluded_manuscript_note_documents_exclusion_not_a_candidate():
    note = real_findings.EXCLUDED_DOWNLOADS_MANUSCRIPT_FILES_NOTE
    assert "manuscript" in note.lower()
    assert "not" in note.lower()
    all_source_labels = " ".join(c.source_label for c in real_findings.real_all_candidates())
    assert "manuscript" not in all_source_labels.lower()
