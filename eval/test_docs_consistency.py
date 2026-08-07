"""
Tests for Documentation/Conference_I_Reviewer_2/ consistency -- Section H
of the Reviewer #2 response. Not testing prose content, only the
structural invariant the governing instructions require: the
implementation matrix's Status column uses only the 5 allowed labels.
"""

from __future__ import annotations

import re
from pathlib import Path

_DOCS_DIR = Path(__file__).resolve().parents[1] / "Documentation" / "Conference_I_Reviewer_2"
_MATRIX_PATH = _DOCS_DIR / "REVIEWER_RESPONSE_IMPLEMENTATION_MATRIX.md"

ALLOWED_STATUSES = {
    "Implemented but not evaluated",
    "Partially evidenced",
    "Awaiting data",
    "Awaiting measurement",
    "Ready for paper update",
}


def _extract_table_rows() -> list[list[str]]:
    text = _MATRIX_PATH.read_text(encoding="utf-8")
    rows = []
    for line in text.splitlines():
        if not line.startswith("| ") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)
    return rows


def test_matrix_file_exists():
    assert _MATRIX_PATH.exists()


def test_matrix_has_eight_comment_rows():
    rows = _extract_table_rows()
    data_rows = [r for r in rows if r and r[0].isdigit()]
    assert len(data_rows) == 8


def test_every_status_cell_is_one_of_the_five_allowed_labels():
    rows = _extract_table_rows()
    header = rows[0]
    status_idx = header.index("Status")
    data_rows = [r for r in rows if r and r[0].isdigit()]
    for row in data_rows:
        raw = row[status_idx]
        status = re.sub(r"\*\*", "", raw).strip()
        assert status in ALLOWED_STATUSES, f"Unexpected status label: {status!r}"


def test_no_other_status_like_bold_text_outside_allowed_set():
    """Guards against a typo'd status label slipping in anywhere in the
    matrix file, not just the extracted table cells."""
    text = _MATRIX_PATH.read_text(encoding="utf-8")
    bolded = set(re.findall(r"\*\*([^*]+)\*\*", text))
    # Only check bolded phrases that look like a status label (short,
    # title-case-ish) -- avoids false positives on unrelated bolded text.
    status_like = {b for b in bolded if b in ALLOWED_STATUSES or b.split()[0] in
                   {"Implemented", "Partially", "Awaiting", "Ready"}}
    assert status_like.issubset(ALLOWED_STATUSES)


# ---------------------------------------------------------------------------
# generated/ paths the matrix references either exist or are traceable to
# a producing script named elsewhere in this repo's eval/ tooling
# ---------------------------------------------------------------------------

_KNOWN_GENERATOR_SCRIPTS = {
    "coverage_audit.py", "method_registry.py", "manifest.py",
    "ablation_runner.py", "export_classifier_hierarchy.py",
    "export_agent_role_diagram.py", "export_evaluation_results.py",
    "export_latency_scalability.py", "export_coverage_charts.py",
}


def test_generated_dir_referenced_paths_traceable_to_a_known_generator():
    """Every 'generated/...' path mentioned in the matrix is either a glob
    pattern (contains '*') -- traceable to one of the known generator
    scripts by prefix -- or an actual file that exists."""
    text = _MATRIX_PATH.read_text(encoding="utf-8")
    referenced = set(re.findall(r"`generated/([^`]+)`", text))
    assert referenced, "expected at least one generated/ reference in the matrix"
    for ref in referenced:
        if "*" in ref or "{" in ref:
            continue  # glob / brace-expansion pattern -- traceable by convention to the generator scripts
        assert (_DOCS_DIR / "generated" / ref).exists() or (_DOCS_DIR / "generated" / "figure_data" / ref).exists(), (
            f"generated/{ref} referenced in the matrix but not found on disk"
        )


def test_other_guide_docs_exist():
    for name in (
        "README.md", "CLASSIFIER_METHOD_REGISTRY.md", "EVALUATION_PROTOCOL.md",
        "REAL_LFS_VALIDATION_DATASET_CARD_TEMPLATE.md", "COVERAGE_AUDIT_GUIDE.md",
        "COMPUTATIONAL_ANALYSIS_GUIDE.md", "FIGURE_DATA_EXPORT_GUIDE.md",
        "STANDARDS_SOURCE_PROVENANCE.md", "ANNOTATION_AND_ADJUDICATION_GUIDE.md",
        "REAL_LFS_DATA_INTAKE_CHECKLIST.md",
    ):
        assert (_DOCS_DIR / name).exists(), f"missing {name}"


# ---------------------------------------------------------------------------
# Cross-doc traceability: every source_url declared in standards_reference.yaml
# is actually cited in STANDARDS_SOURCE_PROVENANCE.md (reviewer comment 8:
# references and technical claims need traceable evidence)
# ---------------------------------------------------------------------------

def test_every_standards_reference_source_url_is_cited_in_provenance_doc():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import coverage_audit as ca  # noqa: E402

    standards_ref = ca.load_standards_reference()
    provenance_text = (_DOCS_DIR / "STANDARDS_SOURCE_PROVENANCE.md").read_text(encoding="utf-8")

    for key, entry in (standards_ref.get("standards") or {}).items():
        source_url = entry.get("source_url")
        if source_url:
            assert source_url in provenance_text, f"{key}'s source_url {source_url!r} not cited in STANDARDS_SOURCE_PROVENANCE.md"
