"""
eval/catalogue_importer.py

Reproducible importer for OFFICIAL classification catalogue files (ISCO-08,
ISIC Rev.4, ISCED 2011, ISCED-F 2013) -- Conference I Reviewer #2 response,
Section C follow-up (official classification-source and coverage-denominator
work).

This module does not fetch, embed, or commit any official source data
itself. It only VALIDATES a catalogue file YOU supply (see
Documentation/Conference_I_Reviewer_2/COVERAGE_AUDIT_GUIDE.md for exactly
how) and, only on a fully clean validation, records a VERIFIED count per
level to eval/verified_catalogue_counts.yaml -- the one file
eval/coverage_audit.py trusts for a citable coverage_percentage. See that
module's docstring for why eval/standards_reference.yaml's
official_count_unverified is never used for that computation.

Input catalogue file format (CSV)
----------------------------------
Columns: level, code, parent_code, label

    level        one of the standard's declared hierarchy level names
                 (see eval/standards_reference.yaml's hierarchy_levels)
    code         the code at that level (must match that level's
                 code_pattern regex)
    parent_code  the code of this row's parent at the parent level
                 (blank for top-level rows, i.e. levels with
                 parent_level: null)
    label        free-text label, optional, not validated

Rows must be ordered top-down: every row's parent_code must already have
appeared as a code at its parent level EARLIER in the file (parent before
child) -- this lets validation run in a single streaming pass without
needing to hold the whole file in memory or make a second pass.

Validation performed (see validate_catalogue())
-------------------------------------------------
1. code format      -- code fully matches the level's code_pattern
2. hierarchy level   -- level is one of the standard's declared level names
3. uniqueness        -- no duplicate code within a level
4. parent-child      -- non-top-level rows have a non-empty parent_code that
   consistency          matches an already-seen code at the parent level;
                        top-level rows have an EMPTY parent_code

Fail-closed: if ANY row has ANY issue, the whole import is marked not-ok
and eval/verified_catalogue_counts.yaml is NOT written/updated for that
standard. A partially-clean catalogue produces zero verified counts, not
partial ones -- reporting "mostly verified" would be exactly the kind of
soft fabrication this tooling exists to prevent.

Usage
-----
    python eval/catalogue_importer.py --standard isco08 --catalogue path/to/your_isco08_catalogue.csv
    python eval/catalogue_importer.py --standard isic_rev4 --catalogue path/to/your_isic_catalogue.csv --dry-run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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


# ---------------------------------------------------------------------------
# Level spec (read from eval/standards_reference.yaml's hierarchy_levels)
# ---------------------------------------------------------------------------

@dataclass
class LevelSpec:
    name: str
    code_pattern: str
    parent_level: Optional[str]
    description: str = ""


def load_level_specs(standards_ref: dict, standard_key: str) -> list[LevelSpec]:
    entry = (standards_ref.get("standards") or {}).get(standard_key)
    if entry is None:
        raise KeyError(f"Unknown standard {standard_key!r}; not present in standards_reference.yaml")
    raw = entry.get("hierarchy_levels") or []
    if not raw:
        raise ValueError(f"standard {standard_key!r} has no hierarchy_levels declared in standards_reference.yaml")
    return [
        LevelSpec(name=lv["name"], code_pattern=lv["code_pattern"],
                  parent_level=lv.get("parent_level"), description=lv.get("description", ""))
        for lv in raw
    ]


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------

@dataclass
class ImportIssue:
    row_number: int
    level: str
    code: str
    parent_code: str
    issue_type: str  # see module docstring's 4 checks, plus "unknown_level"
    detail: str


@dataclass
class ImportResult:
    standard_key: str
    ok: bool
    counts_by_level: dict = field(default_factory=dict)
    issues: list = field(default_factory=list)
    row_count: int = 0
    source_file: str = ""
    source_file_sha256: str = ""
    imported_at_utc: str = ""


def _sha256_of_file(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_catalogue(standard_key: str, levels: list[LevelSpec], rows: list[dict]) -> ImportResult:
    """Pure validation logic, no filesystem I/O -- takes already-loaded
    rows (list of dicts with 'level'/'code'/'parent_code' keys) so this is
    directly unit-testable against small synthetic catalogues."""
    level_by_name = {lv.name: lv for lv in levels}
    seen_codes_by_level: dict[str, set] = {lv.name: set() for lv in levels}
    counts_by_level: dict[str, int] = {lv.name: 0 for lv in levels}
    issues: list[ImportIssue] = []

    for i, row in enumerate(rows, start=1):
        level = (row.get("level") or "").strip()
        code = (row.get("code") or "").strip()
        parent_code = (row.get("parent_code") or "").strip()

        if level not in level_by_name:
            issues.append(ImportIssue(
                row_number=i, level=level, code=code, parent_code=parent_code,
                issue_type="unknown_level",
                detail=f"level {level!r} is not one of {sorted(level_by_name)} for standard {standard_key!r}",
            ))
            continue

        spec = level_by_name[level]

        if not code or not re.fullmatch(spec.code_pattern, code):
            issues.append(ImportIssue(
                row_number=i, level=level, code=code, parent_code=parent_code,
                issue_type="malformed_code",
                detail=f"code {code!r} does not match pattern {spec.code_pattern!r} for level {level!r}",
            ))
            continue

        if code in seen_codes_by_level[level]:
            issues.append(ImportIssue(
                row_number=i, level=level, code=code, parent_code=parent_code,
                issue_type="duplicate_code",
                detail=f"code {code!r} appears more than once at level {level!r}",
            ))
            continue

        if spec.parent_level is None:
            if parent_code:
                issues.append(ImportIssue(
                    row_number=i, level=level, code=code, parent_code=parent_code,
                    issue_type="unexpected_parent_for_top_level",
                    detail=f"level {level!r} is a top-level (parent_level=null); parent_code must be empty, got {parent_code!r}",
                ))
                continue
        else:
            if not parent_code:
                issues.append(ImportIssue(
                    row_number=i, level=level, code=code, parent_code=parent_code,
                    issue_type="missing_parent_code",
                    detail=f"level {level!r} requires a non-empty parent_code (parent level {spec.parent_level!r})",
                ))
                continue
            if parent_code not in seen_codes_by_level[spec.parent_level]:
                issues.append(ImportIssue(
                    row_number=i, level=level, code=code, parent_code=parent_code,
                    issue_type="orphan_parent",
                    detail=(
                        f"parent_code {parent_code!r} not found among already-seen "
                        f"{spec.parent_level!r} codes -- rows must be ordered top-down "
                        f"(parent row before child row)"
                    ),
                ))
                continue

        seen_codes_by_level[level].add(code)
        counts_by_level[level] += 1

    return ImportResult(
        standard_key=standard_key,
        ok=(len(issues) == 0),
        counts_by_level=counts_by_level,
        issues=issues,
        row_count=len(rows),
        imported_at_utc=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_standards_reference(path: Path = _DEFAULT_STANDARDS_REF) -> dict:
    if not path.exists():
        return {"standards": {}}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {"standards": {}}


def load_catalogue_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_verified_counts(path: Path = _DEFAULT_VERIFIED_COUNTS) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_verified_counts(result: ImportResult, out_path: Path = _DEFAULT_VERIFIED_COUNTS) -> None:
    """Only called when result.ok is True (see import_catalogue_file()).
    Merges this standard's entry into the existing file, preserving any
    other standards' previously-verified entries -- machine-generated only,
    never hand-edited, so a plain yaml.safe_load/dump round-trip is safe
    (unlike eval/standards_reference.yaml, which has hand-authored comments
    a naive YAML dump would destroy)."""
    existing = load_verified_counts(out_path)
    existing[result.standard_key] = {
        level: {
            "count": count,
            "source_file": result.source_file,
            "source_file_sha256": result.source_file_sha256,
            "imported_at_utc": result.imported_at_utc,
            "row_count": result.row_count,
        }
        for level, count in result.counts_by_level.items()
    }
    out_path.write_text(
        "# eval/verified_catalogue_counts.yaml\n"
        "# MACHINE-GENERATED by eval/catalogue_importer.py -- do not hand-edit.\n"
        "# Only written when a supplied catalogue file validates with ZERO issues\n"
        "# (fail-closed). eval/coverage_audit.py reads ONLY this file for\n"
        "# official_count_verified -- never eval/standards_reference.yaml's\n"
        "# official_count_unverified.\n\n"
        + yaml.safe_dump(existing, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )


def import_catalogue_file(
    standard_key: str,
    catalogue_path: Path,
    standards_ref_path: Path = _DEFAULT_STANDARDS_REF,
    verified_counts_path: Path = _DEFAULT_VERIFIED_COUNTS,
    dry_run: bool = False,
) -> ImportResult:
    standards_ref = load_standards_reference(standards_ref_path)
    levels = load_level_specs(standards_ref, standard_key)
    rows = load_catalogue_csv(catalogue_path)

    result = validate_catalogue(standard_key, levels, rows)
    result.source_file = str(catalogue_path)
    result.source_file_sha256 = _sha256_of_file(catalogue_path)

    if result.ok and not dry_run:
        write_verified_counts(result, verified_counts_path)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--standard", required=True,
                         help="Standard key from eval/standards_reference.yaml, e.g. isco08 / isic_rev4 / isced2011 / iscedf2013")
    parser.add_argument("--catalogue", required=True, type=Path, help="Path to your local catalogue CSV (never committed -- see COVERAGE_AUDIT_GUIDE.md)")
    parser.add_argument("--standards-ref", type=Path, default=_DEFAULT_STANDARDS_REF)
    parser.add_argument("--verified-counts-out", type=Path, default=_DEFAULT_VERIFIED_COUNTS)
    parser.add_argument("--dry-run", action="store_true", help="Validate only -- never writes verified_catalogue_counts.yaml, even on success")
    args = parser.parse_args()

    if not args.catalogue.exists():
        parser.error(f"Catalogue file not found: {args.catalogue}")

    result = import_catalogue_file(
        args.standard, args.catalogue,
        standards_ref_path=args.standards_ref,
        verified_counts_path=args.verified_counts_out,
        dry_run=args.dry_run,
    )

    print(f"standard={result.standard_key}  rows={result.row_count}  ok={result.ok}")
    if result.issues:
        print(f"\n{len(result.issues)} issue(s) -- NO verified counts were written:")
        for issue in result.issues[:50]:
            print(f"  row {issue.row_number}: [{issue.issue_type}] level={issue.level!r} code={issue.code!r}: {issue.detail}")
        if len(result.issues) > 50:
            print(f"  ... and {len(result.issues) - 50} more")
        sys.exit(1)

    print("\nCounts by level (clean import):")
    for level, count in result.counts_by_level.items():
        print(f"  {level}: {count}")

    if args.dry_run:
        print("\n--dry-run: verified_catalogue_counts.yaml was NOT written.")
    else:
        print(f"\nWrote verified counts to {args.verified_counts_out}")


if __name__ == "__main__":
    main()
