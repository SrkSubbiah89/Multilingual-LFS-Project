"""
eval/coverage_audit.py

Coverage-audit tooling for ISCO-08 / ISIC Rev.4 / ISCED 2011 / ISCED-F 2013
(Conference I Reviewer #2 response, Section C + the official
classification-source/coverage-denominator follow-up). Directly answers
reviewer comment 5 (the manuscript abstract implies complete ISIC coverage
without disclosing the real limitation) with real, computed numbers instead
of assumed or claimed ones.

For each standard/level this reports:
  - standard name + version
  - official_count_unverified: a real, sourced count transcribed from
    eval/standards_reference.yaml's provenance entry (source_url +
    retrieval_date recorded there) -- but NEVER used to compute a
    coverage_percentage. See Documentation/Conference_I_Reviewer_2/
    STANDARDS_SOURCE_PROVENANCE.md for the full research trail behind each
    number, and why "sourced" is not the same as "verified" here.
  - official_count_verified: read ONLY from eval/verified_catalogue_counts.yaml,
    which is machine-generated ONLY by eval/catalogue_importer.py after it
    validates a real, user-supplied catalogue file with ZERO issues
    (fail-closed). This is the ONLY count this module will use to compute
    coverage_percentage.
  - implemented_count: the IMPLEMENTED unique valid code count, computed
    live from this codebase's actual embedded data tables
    (backend/rag/load_full_isco.py's _MAJOR/_SUBMAJOR/_MINOR/_UNIT,
    backend/agents/isic_classifier.py's _ISIC_DATA,
    backend/agents/isced_classifier.py's _ISCED_LEVELS / _ISCED_FIELDS).
  - coverage_percentage: implemented_count / official_count_verified * 100,
    rounded to 2 decimal places -- ONLY computed when official_count_verified
    is present and > 0. `null` otherwise, with coverage_percentage_status
    explaining why. NEVER computed from official_count_unverified -- see
    "Keep coverage percentage null unless verified official catalogue input
    is present" in the governing instructions for this work.
  - duplicate codes and malformed codes (regex-checked against the expected
    code shape for that level) -- both computed directly from the data.
  - missing_codes is always [] today: computing it correctly requires the
    FULL official list of valid codes for a level, not just a count. A
    catalogue import (eval/catalogue_importer.py) DOES have the full list
    while validating, but currently only persists per-level COUNTS to
    eval/verified_catalogue_counts.yaml, not the code list itself -- see
    COVERAGE_AUDIT_GUIDE.md for the rationale (keeping verified_catalogue_
    counts.yaml free of any actual classification-standard content, which
    may be copyrighted, while still recording a verified count).
  - source file path + sha256 of the file actually audited, for traceability.

Usage
-----
    python eval/coverage_audit.py --out Documentation/Conference_I_Reviewer_2/generated/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

_HERE = Path(__file__).resolve().parent
_DEFAULT_STANDARDS_REF = _HERE / "standards_reference.yaml"
_DEFAULT_VERIFIED_COUNTS = _HERE / "verified_catalogue_counts.yaml"

# Allows `python eval/coverage_audit.py` (not just `python -m eval.coverage_audit`)
# to find the `backend` package -- same pattern as eval/run_eval.py.
sys.path.insert(0, str(_HERE.parent))


@dataclass
class CoverageReport:
    standard_name: str
    version: str
    level_name: str

    official_count_verified: Optional[int]
    official_count_verified_note: str

    official_count_unverified: Optional[int]
    official_count_unverified_note: str

    implemented_count: int

    coverage_percentage: Optional[float]
    coverage_percentage_status: str

    duplicate_codes: list[str] = field(default_factory=list)
    malformed_codes: list[str] = field(default_factory=list)
    missing_codes: list[str] = field(default_factory=list)
    source_file: str = ""
    source_file_sha256: str = ""


# ---------------------------------------------------------------------------
# standards_reference.yaml / verified_catalogue_counts.yaml access
# ---------------------------------------------------------------------------

def load_standards_reference(path: Path = _DEFAULT_STANDARDS_REF) -> dict:
    if not path.exists():
        return {"standards": {}}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {"standards": {}}


def load_verified_counts(path: Path = _DEFAULT_VERIFIED_COUNTS) -> dict:
    """eval/catalogue_importer.py-written file; empty dict if no catalogue
    has ever been imported (the default, honest state for this repo)."""
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _level_info(
    standards_ref: dict, verified_counts: dict, standard_key: str, level_key: str,
) -> tuple[str, str, Optional[int], str, Optional[int], str]:
    """(standard_name, version, official_count_unverified, unverified_note,
    official_count_verified, verified_note) for one level, with honest
    'not configured' / 'not imported' defaults."""
    entry = (standards_ref.get("standards") or {}).get(standard_key, {})
    standard_name = entry.get("standard_name", standard_key)
    version = entry.get("version", "unknown")

    level_cfg = (entry.get("levels") or {}).get(level_key, {})
    official_count_unverified = level_cfg.get("official_count_unverified")
    unverified_note = level_cfg.get(
        "unverified_source_note",
        "not configured in eval/standards_reference.yaml",
    )

    verified_entry = (verified_counts.get(standard_key) or {}).get(level_key)
    if verified_entry:
        official_count_verified = verified_entry.get("count")
        verified_note = (
            f"verified via eval/catalogue_importer.py from "
            f"{verified_entry.get('source_file', '(unknown file)')} "
            f"(sha256={verified_entry.get('source_file_sha256', '')[:12]}..., "
            f"imported_at_utc={verified_entry.get('imported_at_utc', '')})"
        )
    else:
        official_count_verified = None
        verified_note = (
            "no verified official catalogue has been imported for this level -- "
            "run eval/catalogue_importer.py with a real catalogue file (see "
            "COVERAGE_AUDIT_GUIDE.md); official_count_unverified is NEVER used "
            "as a substitute"
        )

    return standard_name, version, official_count_unverified, unverified_note, official_count_verified, verified_note


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _sha256_of_file(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_duplicates(codes: list[str]) -> list[str]:
    """Codes appearing more than once, each listed once, in first-seen order."""
    seen: set[str] = set()
    dupes: list[str] = []
    for c in codes:
        if c in seen and c not in dupes:
            dupes.append(c)
        seen.add(c)
    return dupes


def find_malformed(codes: list[str], pattern: str) -> list[str]:
    """Codes that do NOT fully match `pattern` (e.g. r'\\d{4}' for a 4-digit
    numeric code, or r'[A-Z]' for a 1-letter ISIC section)."""
    rx = re.compile(pattern)
    return [c for c in codes if not rx.fullmatch(c)]


def _compute_coverage_percentage(
    implemented_count: int, official_count_verified: Optional[int],
) -> tuple[Optional[float], str]:
    """ONLY ever computed from official_count_verified -- see module
    docstring. Returns (percentage_or_None, status_string)."""
    if official_count_verified is None:
        return None, (
            "unavailable -- no verified official catalogue import for this level "
            "(official_count_unverified, if present, is a sourced-but-unverified "
            "figure and is never used for this computation)"
        )
    if official_count_verified <= 0:
        return None, f"unavailable -- verified official count is {official_count_verified} (not positive)"
    pct = round(100.0 * implemented_count / official_count_verified, 2)
    return pct, "verified -- computed from implemented_count / official_count_verified"


def _build_level_report(
    codes: list[str], pattern: str,
    standards_ref: dict, verified_counts: dict, standard_key: str, level_key: str,
    source_file: Path,
) -> CoverageReport:
    (standard_name, version, official_count_unverified, unverified_note,
     official_count_verified, verified_note) = _level_info(
        standards_ref, verified_counts, standard_key, level_key,
    )

    implemented_count = len(set(codes))
    coverage_percentage, coverage_percentage_status = _compute_coverage_percentage(
        implemented_count, official_count_verified,
    )

    return CoverageReport(
        standard_name=standard_name,
        version=version,
        level_name=level_key,
        official_count_verified=official_count_verified,
        official_count_verified_note=verified_note,
        official_count_unverified=official_count_unverified,
        official_count_unverified_note=unverified_note,
        implemented_count=implemented_count,
        coverage_percentage=coverage_percentage,
        coverage_percentage_status=coverage_percentage_status,
        duplicate_codes=find_duplicates(codes),
        malformed_codes=find_malformed(codes, pattern),
        missing_codes=[],  # see module docstring: requires the full code list, not just a count
        source_file=str(source_file),
        source_file_sha256=_sha256_of_file(source_file),
    )


# ---------------------------------------------------------------------------
# Per-standard audits
# ---------------------------------------------------------------------------

def audit_isco(standards_ref: Optional[dict] = None, verified_counts: Optional[dict] = None) -> list[CoverageReport]:
    from backend.rag import load_full_isco as m

    standards_ref = standards_ref or load_standards_reference()
    verified_counts = verified_counts if verified_counts is not None else load_verified_counts()
    source_file = Path(m.__file__)
    levels = [
        ("major", [c for c, *_ in m._MAJOR], r"\d"),
        ("submajor", [c for c, *_ in m._SUBMAJOR], r"\d{2}"),
        ("minor", [c for c, *_ in m._MINOR], r"\d{3}"),
        ("unit", [c for c, *_ in m._UNIT], r"\d{4}"),
    ]
    return [
        _build_level_report(codes, pattern, standards_ref, verified_counts, "isco08", level_key, source_file)
        for level_key, codes, pattern in levels
    ]


def audit_isic(standards_ref: Optional[dict] = None, verified_counts: Optional[dict] = None) -> list[CoverageReport]:
    from backend.agents import isic_classifier as m

    standards_ref = standards_ref or load_standards_reference()
    verified_counts = verified_counts if verified_counts is not None else load_verified_counts()
    source_file = Path(m.__file__)

    # _ISIC_DATA is a FLAT list of full leaf paths (one row per class code),
    # not separate per-level tables -- dedupe by each level's own code to
    # derive unique per-level entries before counting.
    sections = [e["section"] for e in m._ISIC_DATA]
    divisions = [e["division_code"] for e in m._ISIC_DATA]
    groups = [e["group_code"] for e in m._ISIC_DATA]
    classes = [e["class_code"] for e in m._ISIC_DATA]

    levels = [
        ("section", sections, r"[A-Z]"),
        ("division", divisions, r"\d{2}"),
        ("group", groups, r"\d{3}"),
        ("class", classes, r"\d{4}"),
    ]
    return [
        _build_level_report(codes, pattern, standards_ref, verified_counts, "isic_rev4", level_key, source_file)
        for level_key, codes, pattern in levels
    ]


def audit_isced(standards_ref: Optional[dict] = None, verified_counts: Optional[dict] = None) -> list[CoverageReport]:
    from backend.agents import isced_classifier as m

    standards_ref = standards_ref or load_standards_reference()
    verified_counts = verified_counts if verified_counts is not None else load_verified_counts()
    source_file = Path(m.__file__)

    levels_codes = [str(e["level"]) for e in m._ISCED_LEVELS]
    broad_codes = [e["broad_code"] for e in m._ISCED_FIELDS]
    narrow_codes = [e["narrow_code"] for e in m._ISCED_FIELDS]
    detailed_codes = [e["detailed_code"] for e in m._ISCED_FIELDS]

    levels = [
        ("level", levels_codes, r"[0-8]"),
        ("broad", broad_codes, r"\d{2}"),
        ("narrow", narrow_codes, r"\d{3}"),
        ("detailed", detailed_codes, r"\d{4}"),
    ]
    reports = []
    for level_key, codes, pattern in levels:
        standard_key = "isced2011" if level_key == "level" else "iscedf2013"
        reports.append(_build_level_report(codes, pattern, standards_ref, verified_counts, standard_key, level_key, source_file))
    return reports


def audit_all(standards_ref: Optional[dict] = None, verified_counts: Optional[dict] = None) -> dict[str, list[CoverageReport]]:
    standards_ref = standards_ref or load_standards_reference()
    verified_counts = verified_counts if verified_counts is not None else load_verified_counts()
    return {
        "isco08": audit_isco(standards_ref, verified_counts),
        "isic_rev4": audit_isic(standards_ref, verified_counts),
        "isced2011_and_iscedf2013": audit_isced(standards_ref, verified_counts),
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_json(reports: list[CoverageReport], path: Path) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reports": [asdict(r) for r in reports],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(reports: list[CoverageReport], path: Path) -> None:
    fieldnames = [
        "standard_name", "version", "level_name",
        "official_count_verified", "official_count_verified_note",
        "official_count_unverified", "official_count_unverified_note",
        "implemented_count", "coverage_percentage", "coverage_percentage_status",
        "duplicate_count", "malformed_count", "missing_count",
        "source_file", "source_file_sha256",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in reports:
            writer.writerow({
                "standard_name": r.standard_name, "version": r.version,
                "level_name": r.level_name,
                "official_count_verified": r.official_count_verified,
                "official_count_verified_note": r.official_count_verified_note,
                "official_count_unverified": r.official_count_unverified,
                "official_count_unverified_note": r.official_count_unverified_note,
                "implemented_count": r.implemented_count,
                "coverage_percentage": r.coverage_percentage,
                "coverage_percentage_status": r.coverage_percentage_status,
                "duplicate_count": len(r.duplicate_codes),
                "malformed_count": len(r.malformed_codes),
                "missing_count": len(r.missing_codes),
                "source_file": r.source_file,
                "source_file_sha256": r.source_file_sha256,
            })


def write_markdown(reports: list[CoverageReport], path: Path) -> None:
    lines = [
        "# Coverage Audit",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Generated by `python eval/coverage_audit.py` -- do not hand-edit. "
        "`official_count_verified` comes ONLY from `eval/verified_catalogue_counts.yaml` "
        "(machine-written by `eval/catalogue_importer.py` after a clean catalogue import); "
        "`official_count_unverified` is a sourced-but-not-imported figure from "
        "`eval/standards_reference.yaml` and is NEVER used to compute `coverage_percentage`. "
        "`null` means not available, never zero.",
        "",
        "| Standard | Level | Verified count | Unverified count | Implemented | Coverage % | Coverage status | Duplicates | Malformed |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        verified = r.official_count_verified if r.official_count_verified is not None else "null"
        unverified = r.official_count_unverified if r.official_count_unverified is not None else "null"
        pct = f"{r.coverage_percentage:.2f}%" if r.coverage_percentage is not None else "null"
        lines.append(
            f"| {r.standard_name} | {r.level_name} | {verified} | {unverified} | "
            f"{r.implemented_count} | {pct} | {r.coverage_percentage_status} | "
            f"{len(r.duplicate_codes)} | {len(r.malformed_codes)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--standards-ref", type=Path, default=_DEFAULT_STANDARDS_REF)
    parser.add_argument("--verified-counts", type=Path, default=_DEFAULT_VERIFIED_COUNTS)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    standards_ref = load_standards_reference(args.standards_ref)
    verified_counts = load_verified_counts(args.verified_counts)
    by_standard = audit_all(standards_ref, verified_counts)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for standard_key, reports in by_standard.items():
        json_path = args.out / f"coverage_audit_{standard_key}_{ts}.json"
        csv_path = args.out / f"coverage_audit_{standard_key}_{ts}.csv"
        md_path = args.out / f"coverage_audit_{standard_key}_{ts}.md"
        write_json(reports, json_path)
        write_csv(reports, csv_path)
        write_markdown(reports, md_path)
        print(f"{standard_key}: wrote {json_path.name}, {csv_path.name}, {md_path.name}")


if __name__ == "__main__":
    main()
