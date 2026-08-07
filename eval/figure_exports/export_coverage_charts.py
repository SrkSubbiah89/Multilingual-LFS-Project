"""
eval/figure_exports/export_coverage_charts.py

Exports a clean CSV/JSON summary from eval/coverage_audit.py's generated
reports (Conference I Reviewer #2 response, Section I; directly supports
reviewer comment 5 -- the abstract's ISIC coverage disclosure) for a
coverage chart -- NOT a screenshot.

Reads Documentation/Conference_I_Reviewer_2/generated/coverage_audit_*.json
(the most recent JSON per standard group, by filename timestamp). If no
coverage-audit reports exist yet (eval/coverage_audit.py has not been run),
writes a structured "no data yet" JSON/CSV rather than fabricating rows.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

_DEFAULT_SOURCE_DIR = (
    Path(__file__).resolve().parents[2] / "Documentation" / "Conference_I_Reviewer_2" / "generated"
)

_FIELDS = [
    "standard_name", "level_name",
    "official_count_verified", "official_count_unverified",
    "implemented_count", "coverage_percentage", "coverage_percentage_status",
    "duplicate_count", "malformed_count",
]


def find_latest_reports_per_standard(source_dir: Path = _DEFAULT_SOURCE_DIR) -> list[Path]:
    """coverage_audit.py names files coverage_audit_<standard_key>_<timestamp>.json
    -- one JSON per standard GROUP (isco08 / isic_rev4 / isced2011_and_iscedf2013),
    each containing multiple per-level reports. Returns the most recent file
    per standard_key, by the timestamp embedded in the filename (lexically
    sortable, matches coverage_audit.py's %Y%m%dT%H%M%SZ format)."""
    if not source_dir.exists():
        return []
    by_key: dict[str, Path] = {}
    for path in source_dir.glob("coverage_audit_*.json"):
        # coverage_audit_<key>_<timestamp>.json -- key may itself contain
        # underscores (e.g. "isced2011_and_iscedf2013"), so split off the
        # timestamp (last underscore-separated segment) instead of the key.
        stem = path.stem  # coverage_audit_<key>_<timestamp>
        parts = stem.split("_")
        if len(parts) < 3:
            continue
        key = "_".join(parts[2:-1])
        existing = by_key.get(key)
        if existing is None or path.name > existing.name:
            by_key[key] = path
    return sorted(by_key.values())


def load_coverage_rows(source_dir: Path = _DEFAULT_SOURCE_DIR) -> list[dict]:
    rows = []
    for path in find_latest_reports_per_standard(source_dir):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for report in payload.get("reports", []):
            rows.append({k: report.get(k) for k in (
                            "standard_name", "level_name",
                            "official_count_verified", "official_count_unverified",
                            "implemented_count", "coverage_percentage", "coverage_percentage_status",
                        )}
                        | {"duplicate_count": len(report.get("duplicate_codes", [])), "malformed_count": len(report.get("malformed_codes", []))})
    return rows


def write_json(rows: list[dict], path: Path) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "no_coverage_reports_found": len(rows) == 0,
        "coverage": rows,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source-dir", type=Path, default=_DEFAULT_SOURCE_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = load_coverage_rows(args.source_dir)
    write_json(rows, args.out / "coverage_charts.json")
    write_csv(rows, args.out / "coverage_charts.csv")
    if rows:
        print(f"Wrote {len(rows)} row(s) to {args.out}")
    else:
        print(f"No coverage-audit reports found under {args.source_dir} -- wrote a 'no data yet' export to {args.out}")


if __name__ == "__main__":
    main()
