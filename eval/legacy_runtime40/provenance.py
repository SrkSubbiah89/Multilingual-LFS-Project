"""
eval/legacy_runtime40/provenance.py

Task 40: classifies dependency-provenance evidence for the historical
`qdrant-client` package version needed to run LEGACY_SHA's byte-identical
`VectorStore.search()`, and determines Outcome A (exact pin), Outcome B
(defensible bounded range), or Outcome C (no defensible provenance) --
never by guessing a version, brute-forcing, or inferring from an
unrelated/non-contemporaneous file.

This module only classifies evidence rows a caller supplies (typically
built from real `git show`/`git log` output at and around LEGACY_SHA);
it never runs git itself, so it is trivially hermetic to test with
synthetic evidence, and the real historical findings (Task 40's actual
investigation, reproduced in `real_historical_evidence()` below) are
exercised through the exact same code path as any test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

_PIN_RE = re.compile(r"^qdrant-client\s*==\s*([0-9][0-9A-Za-z.\-]*)\s*$")
_RANGE_RE = re.compile(r"^qdrant-client\s*(>=|<=|>|<|~=)\s*([0-9][0-9A-Za-z.\-]*)")


@dataclass
class EvidenceRow:
    source: str            # e.g. "requirements.txt"
    commit_or_path: str    # e.g. "824fcf2:requirements.txt"
    raw_value: str         # the literal qdrant-client line, or a note
    conclusion: str        # "exact pin X" | "bounded constraint OP X" | "no version specifier" | "no evidence"
    contemporaneous: bool = True  # False for evidence dated well after LEGACY_SHA


@dataclass
class ProvenanceReport:
    rows: list[EvidenceRow] = field(default_factory=list)
    outcome: str = "C"  # "A" | "B" | "C"
    selected_version: Optional[str] = None
    selection_rule: Optional[str] = None
    reason: Optional[str] = None


def classify_requirements_text(source_label: str, commit_or_path: str, requirements_text: str,
                                contemporaneous: bool = True) -> EvidenceRow:
    """Looks for a qdrant-client line in *requirements_text* (the raw
    content of one requirements-style file) and classifies it."""
    for line in requirements_text.splitlines():
        stripped = line.strip()
        if not stripped.lower().startswith("qdrant-client"):
            continue
        pin_match = _PIN_RE.match(stripped)
        if pin_match:
            return EvidenceRow(source_label, commit_or_path, stripped,
                                f"exact pin {pin_match.group(1)}", contemporaneous)
        range_match = _RANGE_RE.match(stripped)
        if range_match:
            return EvidenceRow(source_label, commit_or_path, stripped,
                                f"bounded constraint {range_match.group(1)}{range_match.group(2)}", contemporaneous)
        return EvidenceRow(source_label, commit_or_path, stripped, "no version specifier", contemporaneous)
    return EvidenceRow(source_label, commit_or_path, "(qdrant-client not mentioned)", "no evidence", contemporaneous)


def determine_outcome(evidence: list[EvidenceRow]) -> ProvenanceReport:
    """
    Determines Outcome A/B/C from *evidence*. Only CONTEMPORANEOUS rows
    (contemporaneous=True -- i.e. genuinely from LEGACY_SHA or its
    immediately reachable, dated-close history, never a much-later
    unrelated snapshot) count towards an exact pin or bounded range;
    non-contemporaneous rows are recorded in the report for
    transparency but never used to justify Outcome A or B, since doing
    so would be exactly the "guess an old version from an unrelated
    later file" behavior Task 40 explicitly forbids.
    """
    report = ProvenanceReport(rows=list(evidence))
    contemporaneous = [r for r in evidence if r.contemporaneous]

    exact_pins = {r.conclusion.split()[-1] for r in contemporaneous if r.conclusion.startswith("exact pin")}
    if len(exact_pins) == 1:
        version = next(iter(exact_pins))
        report.outcome = "A"
        report.selected_version = version
        report.selection_rule = (
            f"single consistent exact pin {version!r} found across all contemporaneous evidence"
        )
        return report
    if len(exact_pins) > 1:
        report.outcome = "C"
        report.reason = f"conflicting contemporaneous exact pins found: {sorted(exact_pins)}"
        return report

    bounded = [r for r in contemporaneous if r.conclusion.startswith("bounded constraint")]
    if bounded:
        report.outcome = "B"
        report.selection_rule = (
            "historical evidence establishes a bounded range but not an exact pin; "
            "the oldest exact candidate in that range exposing QdrantClient.search() "
            "must be selected and disclosed before installation"
        )
        return report

    report.outcome = "C"
    report.reason = "historical_qdrant_client_version_not_proven"
    return report


def real_historical_evidence() -> list[EvidenceRow]:
    """
    Task 40's actual investigation findings, reproduced literally here
    (values copied verbatim from `git show`/`git log` output run
    directly against this repository -- see the Task 40 final report
    for the exact commands). Exhaustively covers every provenance
    source the task requires: requirements.txt (at LEGACY_SHA, at the
    initial commit, and at every later commit that ever touched the
    file), docker/docker-compose.yml, the root Dockerfile (added one
    day after LEGACY_SHA), README.md, and the only qdrant-client
    version pin found anywhere in the repository's full history
    (requirements-dev.txt, added 2026-08-06 -- over five months after
    LEGACY_SHA's 2026-02-23 date, for the current/modern environment,
    marked non-contemporaneous). No pyproject.toml, poetry.lock,
    Pipfile, Pipfile.lock, setup.py, setup.cfg, or .github/workflows/*
    has ever existed anywhere in this repository's history.
    """
    return [
        EvidenceRow("requirements.txt", "824fcf2:requirements.txt (LEGACY_SHA itself)",
                    "qdrant-client", "no version specifier", contemporaneous=True),
        EvidenceRow("requirements.txt", "8bba6e2:requirements.txt (initial commit, 2026-02-23)",
                    "qdrant-client", "no version specifier", contemporaneous=True),
        EvidenceRow("requirements.txt", "1d03e75:requirements.txt (LEGACY_SHA's direct parent)",
                    "qdrant-client", "no version specifier", contemporaneous=True),
        EvidenceRow("requirements.txt", "64e0415/a00d973/825a6e5/675121b/e20af39:requirements.txt "
                    "(every later commit that ever touched the file)",
                    "qdrant-client", "no version specifier", contemporaneous=True),
        EvidenceRow("docker/docker-compose.yml", "824fcf2:docker/docker-compose.yml",
                    "image: qdrant/qdrant:latest",
                    "no evidence (server image floating tag, not a client-library pin)", contemporaneous=True),
        EvidenceRow("Dockerfile", "4705afde:Dockerfile (added 2026-02-24, one day after LEGACY_SHA)",
                    "RUN pip install --no-cache-dir -r requirements.txt tf-keras",
                    "no evidence (installs unpinned requirements.txt verbatim)", contemporaneous=True),
        EvidenceRow("README.md", "824fcf2:README.md",
                    "(no qdrant-client mention)", "no evidence", contemporaneous=True),
        EvidenceRow("pyproject.toml/poetry.lock/Pipfile/Pipfile.lock/setup.py/setup.cfg", "(entire repository history)",
                    "(file never existed)", "no evidence", contemporaneous=True),
        EvidenceRow(".github/workflows/*", "(entire repository history)",
                    "(never existed)", "no evidence", contemporaneous=True),
        EvidenceRow("requirements-dev.txt", "66bac86:requirements-dev.txt (added 2026-08-06)",
                    "qdrant-client==1.17.0",
                    "non-contemporaneous exact pin 1.17.0 (excluded: 5+ months after LEGACY_SHA's "
                    "2026-02-23 date; reflects the current/modern environment, identical to the "
                    "already-proven-incompatible version currently installed)",
                    contemporaneous=False),
    ]
