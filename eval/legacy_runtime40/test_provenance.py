"""
Tests for eval/legacy_runtime40/provenance.py (Task 40).

Fully hermetic: no git, filesystem, network, Qdrant, or model call
anywhere in this file -- every evidence row is hand-constructed or
built from a small in-memory requirements-file string.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.legacy_runtime40.provenance import (  # noqa: E402
    EvidenceRow,
    classify_requirements_text,
    determine_outcome,
    real_historical_evidence,
)


# ---------------------------------------------------------------------------
# classify_requirements_text
# ---------------------------------------------------------------------------

def test_classifies_exact_pin():
    row = classify_requirements_text("requirements.txt", "abc123:requirements.txt", "qdrant-client==1.7.3\nfastapi\n")
    assert row.conclusion == "exact pin 1.7.3"


def test_classifies_bounded_constraint():
    row = classify_requirements_text("requirements.txt", "abc123:requirements.txt", "qdrant-client>=1.6,<1.8\n")
    assert row.conclusion.startswith("bounded constraint >=1.6")


def test_classifies_unpinned():
    row = classify_requirements_text("requirements.txt", "abc123:requirements.txt", "qdrant-client\nfastapi\n")
    assert row.conclusion == "no version specifier"


def test_classifies_absent():
    row = classify_requirements_text("requirements.txt", "abc123:requirements.txt", "fastapi\nuvicorn\n")
    assert row.conclusion == "no evidence"


# ---------------------------------------------------------------------------
# determine_outcome -- scenario 1 (rejects unpinned/unsupported evidence) and
# scenario 2 (selected version matches disclosed rule)
# ---------------------------------------------------------------------------

def test_outcome_a_single_consistent_exact_pin():
    evidence = [
        EvidenceRow("requirements.txt", "sha1:requirements.txt", "qdrant-client==1.7.3", "exact pin 1.7.3", True),
        EvidenceRow("Dockerfile", "sha1:Dockerfile", "pip install -r requirements.txt", "no evidence", True),
    ]
    report = determine_outcome(evidence)
    assert report.outcome == "A"
    assert report.selected_version == "1.7.3"
    assert report.selection_rule is not None


def test_outcome_c_conflicting_exact_pins():
    evidence = [
        EvidenceRow("requirements.txt", "sha1:requirements.txt", "qdrant-client==1.7.3", "exact pin 1.7.3", True),
        EvidenceRow("Dockerfile", "sha2:Dockerfile", "qdrant-client==1.6.0", "exact pin 1.6.0", True),
    ]
    report = determine_outcome(evidence)
    assert report.outcome == "C"
    assert "conflicting" in report.reason


def test_outcome_b_bounded_range_only():
    evidence = [
        EvidenceRow("requirements.txt", "sha1:requirements.txt", "qdrant-client>=1.6,<1.8",
                    "bounded constraint >=1.6", True),
    ]
    report = determine_outcome(evidence)
    assert report.outcome == "B"
    assert report.selection_rule is not None
    assert "oldest exact candidate" in report.selection_rule


def test_outcome_c_no_version_specifier_anywhere():
    evidence = [
        EvidenceRow("requirements.txt", "sha1:requirements.txt", "qdrant-client", "no version specifier", True),
        EvidenceRow("Dockerfile", "sha1:Dockerfile", "pip install -r requirements.txt", "no evidence", True),
    ]
    report = determine_outcome(evidence)
    assert report.outcome == "C"
    assert report.reason == "historical_qdrant_client_version_not_proven"


def test_outcome_c_no_evidence_at_all():
    report = determine_outcome([])
    assert report.outcome == "C"
    assert report.reason == "historical_qdrant_client_version_not_proven"


def test_non_contemporaneous_exact_pin_is_never_used_for_outcome_a():
    """A later, unrelated file pinning an exact version must never be
    treated as historical provenance -- this is exactly the 'guess an
    old version from an unrelated later file' behavior Task 40 forbids."""
    evidence = [
        EvidenceRow("requirements.txt", "sha1:requirements.txt", "qdrant-client", "no version specifier", True),
        EvidenceRow("requirements-dev.txt", "much_later_sha:requirements-dev.txt", "qdrant-client==1.17.0",
                    "exact pin 1.17.0", contemporaneous=False),
    ]
    report = determine_outcome(evidence)
    assert report.outcome == "C"
    assert report.selected_version is None


# ---------------------------------------------------------------------------
# The real Task 40 investigation, run through the exact same classifier
# ---------------------------------------------------------------------------

def test_real_historical_evidence_yields_outcome_c():
    report = determine_outcome(real_historical_evidence())
    assert report.outcome == "C"
    assert report.reason == "historical_qdrant_client_version_not_proven"
    assert report.selected_version is None
    # Every contemporaneous requirements.txt occurrence (five distinct
    # commits spanning the entire project history) must show "no
    # version specifier", never a pin -- otherwise Outcome C above
    # would be wrong.
    contemporaneous_requirements_rows = [
        r for r in report.rows if r.source == "requirements.txt" and r.contemporaneous
    ]
    assert len(contemporaneous_requirements_rows) >= 4
    assert all(r.conclusion == "no version specifier" for r in contemporaneous_requirements_rows)
    # The only exact pin found anywhere is present in the evidence table
    # (for transparency) but is explicitly marked non-contemporaneous.
    non_contemporaneous = [r for r in report.rows if not r.contemporaneous]
    assert len(non_contemporaneous) == 1
    assert "1.17.0" in non_contemporaneous[0].raw_value
