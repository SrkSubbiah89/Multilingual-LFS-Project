"""
eval/normalize_ilo_isco08_catalogue.py

Conference I Reviewer #2 response, Task 20: deterministic, offline
normalizer that converts the official ILO ISCO-08 "EN Structure and
definitions" workbook into the flat CSV shape
`eval/catalogue_importer.py` accepts (`level,code,parent_code,label`,
rows ordered top-down so every child row's parent already appeared).

This module does not embed, hard-code, or ship any ISCO-08 catalogue
row. It parses whatever workbook path it is given, at runtime, using
`openpyxl` (an existing project dependency) only -- no network, Qdrant,
model, or subprocess call is made anywhere in this file.

Fail-closed design
-------------------
`normalize()` raises `NormalizationError` (never returns a partial
result) on: a missing/unreadable workbook, a missing or ambiguous
structure sheet, a missing required column, a blank/malformed code, a
code whose length does not match its declared `Level` value, a
duplicate code within a level, a row whose derived parent code was not
already seen (rows must appear in top-down/depth-first order in the
source, which the official workbook does -- this is verified, not
assumed), a blank title, or an observed per-level record count that
does not match the caller-supplied expected counts (default: the
ILO-published 10/43/130/436). No normalized CSV is ever written for a
run that raises.

Source sheet contract (as published by the ILO in the 2021 revision of
"ISCO-08 EN Structure and definitions.xlsx")
--------------------------------------------------------------------
Sheet columns (row 1 header, exact names): `Level`, `ISCO 08 Code`,
`Title EN`, plus several free-text columns this normalizer ignores
(`Definition`, `Tasks include`, `Included occupations`, `Excluded
occupations`, `Notes`). `Level` values are `1`/`2`/`3`/`4` (major /
sub-major / minor / unit); `ISCO 08 Code` is the code string, already
correctly zero-padded by the source (e.g. `"03"`, `"031"`, `"0310"` for
Armed Forces codes) -- this normalizer never re-pads or otherwise
mutates a code string, it only validates it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import openpyxl

DEFAULT_SHEET_NAME = "ISCO-08 EN Struct and defin"
_REQUIRED_COLUMNS = ("Level", "ISCO 08 Code", "Title EN")

_LEVEL_NAME_BY_DIGIT = {"1": "major", "2": "submajor", "3": "minor", "4": "unit"}
_LEVEL_CODE_LENGTH = {"major": 1, "submajor": 2, "minor": 3, "unit": 4}
_LEVEL_PARENT = {"major": None, "submajor": "major", "minor": "submajor", "unit": "minor"}
_CODE_RE = re.compile(r"^\d+$")

DEFAULT_EXPECTED_COUNTS = {"major": 10, "submajor": 43, "minor": 130, "unit": 436}


class NormalizationError(Exception):
    """Raised on any fail-closed violation. normalize() never returns a
    partial result when this is raised -- callers must not catch this
    and proceed with whatever rows were parsed so far."""


@dataclass
class NormalizedRow:
    level: str
    code: str
    parent_code: str
    label: str


@dataclass
class NormalizationReport:
    source_path: str
    source_sha256: str
    sheet_name: str
    header_row: list
    n_rows_by_level: dict = field(default_factory=dict)
    total_rows: int = 0
    normalized_at_utc: str = ""
    parser_script_sha256: str = ""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_workbook(
    xlsx_path: Path,
    sheet_name: str = DEFAULT_SHEET_NAME,
) -> tuple[list[NormalizedRow], NormalizationReport]:
    """Parse *xlsx_path* and return (rows, report). Raises
    NormalizationError on any violation; never returns a partial rows
    list in that case."""
    if not xlsx_path.exists():
        raise NormalizationError(f"workbook not found: {xlsx_path}")

    try:
        wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    except Exception as exc:  # noqa: BLE001 -- fail closed with context, never silently proceed
        raise NormalizationError(f"could not open workbook {xlsx_path}: {exc}") from exc

    if sheet_name not in wb.sheetnames:
        raise NormalizationError(
            f"sheet {sheet_name!r} not found in {xlsx_path}; available sheets: {wb.sheetnames}"
        )
    matching = [s for s in wb.sheetnames if s == sheet_name]
    if len(matching) != 1:
        raise NormalizationError(f"sheet name {sheet_name!r} is ambiguous in {xlsx_path}: {matching}")

    ws = wb[sheet_name]
    all_rows = list(ws.iter_rows(values_only=True))
    if not all_rows:
        raise NormalizationError(f"sheet {sheet_name!r} in {xlsx_path} has no rows")

    header = list(all_rows[0])
    header_index = {str(h).strip(): i for i, h in enumerate(header) if h is not None}
    missing_cols = [c for c in _REQUIRED_COLUMNS if c not in header_index]
    if missing_cols:
        raise NormalizationError(
            f"sheet {sheet_name!r} is missing required column(s) {missing_cols}; "
            f"found columns: {list(header_index)}"
        )

    level_col = header_index["Level"]
    code_col = header_index["ISCO 08 Code"]
    title_col = header_index["Title EN"]

    rows: list[NormalizedRow] = []
    seen_codes_by_level: dict[str, set] = {lv: set() for lv in _LEVEL_NAME_BY_DIGIT.values()}
    n_by_level: dict[str, int] = {lv: 0 for lv in _LEVEL_NAME_BY_DIGIT.values()}

    for row_number, raw_row in enumerate(all_rows[1:], start=2):
        if raw_row is None or all(v is None for v in raw_row):
            continue  # a fully blank row is not a data row; not a violation by itself

        raw_level = raw_row[level_col] if level_col < len(raw_row) else None
        raw_code = raw_row[code_col] if code_col < len(raw_row) else None
        raw_title = raw_row[title_col] if title_col < len(raw_row) else None

        level_digit = str(raw_level).strip() if raw_level is not None else ""
        if level_digit not in _LEVEL_NAME_BY_DIGIT:
            raise NormalizationError(
                f"row {row_number}: malformed hierarchy level {raw_level!r} "
                f"(expected one of {sorted(_LEVEL_NAME_BY_DIGIT)})"
            )
        level = _LEVEL_NAME_BY_DIGIT[level_digit]

        code = str(raw_code).strip() if raw_code is not None else ""
        if not code or not _CODE_RE.match(code):
            raise NormalizationError(f"row {row_number}: blank or non-numeric code {raw_code!r} at level {level!r}")
        expected_len = _LEVEL_CODE_LENGTH[level]
        if len(code) != expected_len:
            raise NormalizationError(
                f"row {row_number}: code {code!r} has length {len(code)}, expected "
                f"{expected_len} for level {level!r}"
            )

        if code in seen_codes_by_level[level]:
            raise NormalizationError(f"row {row_number}: duplicate code {code!r} at level {level!r}")

        parent_level = _LEVEL_PARENT[level]
        if parent_level is None:
            parent_code = ""
        else:
            parent_code = code[:_LEVEL_CODE_LENGTH[parent_level]]
            if parent_code not in seen_codes_by_level[parent_level]:
                raise NormalizationError(
                    f"row {row_number}: code {code!r}'s derived parent {parent_code!r} "
                    f"(level {parent_level!r}) has not appeared yet -- source rows must be "
                    f"in top-down/depth-first order"
                )

        title = str(raw_title).strip() if raw_title is not None else ""
        if not title:
            raise NormalizationError(f"row {row_number}: blank title for code {code!r} at level {level!r}")

        seen_codes_by_level[level].add(code)
        n_by_level[level] += 1
        rows.append(NormalizedRow(level=level, code=code, parent_code=parent_code, label=title))

    report = NormalizationReport(
        source_path=str(xlsx_path),
        source_sha256=_sha256_file(xlsx_path),
        sheet_name=sheet_name,
        header_row=[str(h) if h is not None else None for h in header],
        n_rows_by_level=n_by_level,
        total_rows=len(rows),
        normalized_at_utc=datetime.now(timezone.utc).isoformat(),
        parser_script_sha256=_sha256_file(Path(__file__)),
    )
    return rows, report


def validate_expected_counts(
    n_rows_by_level: dict[str, int],
    expected_counts: dict[str, int] = DEFAULT_EXPECTED_COUNTS,
) -> None:
    """Raises NormalizationError if any level's observed count differs
    from *expected_counts*. Called separately from parse_workbook() so
    callers can choose their own expected counts (e.g. in tests) without
    duplicating the parsing logic."""
    mismatches = {
        lv: (n_rows_by_level.get(lv, 0), expected_counts[lv])
        for lv in expected_counts
        if n_rows_by_level.get(lv, 0) != expected_counts[lv]
    }
    if mismatches:
        detail = ", ".join(f"{lv}: observed={obs} expected={exp}" for lv, (obs, exp) in mismatches.items())
        raise NormalizationError(f"unexpected record count(s): {detail}")


def write_normalized_csv(rows: list[NormalizedRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["level", "code", "parent_code", "label"])
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))


def normalize(
    xlsx_path: Path,
    out_csv_path: Path,
    sheet_name: str = DEFAULT_SHEET_NAME,
    expected_counts: Optional[dict[str, int]] = DEFAULT_EXPECTED_COUNTS,
) -> NormalizationReport:
    """End-to-end: parse, validate counts (unless expected_counts is
    None), write the normalized CSV. Raises NormalizationError and
    writes nothing on any violation."""
    rows, report = parse_workbook(xlsx_path, sheet_name=sheet_name)
    if expected_counts is not None:
        validate_expected_counts(report.n_rows_by_level, expected_counts)
    write_normalized_csv(rows, out_csv_path)
    report_dict = asdict(report)
    report_dict["normalized_csv_path"] = str(out_csv_path)
    report_dict["normalized_csv_sha256"] = _sha256_file(out_csv_path)
    return report, report_dict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xlsx", required=True, type=Path, help="Path to the official ILO ISCO-08 EN structure workbook")
    parser.add_argument("--out-csv", required=True, type=Path, help="Path to write the normalized level,code,parent_code,label CSV")
    parser.add_argument("--sheet-name", default=DEFAULT_SHEET_NAME)
    parser.add_argument("--report-out", type=Path, default=None, help="Optional path to write a JSON normalization report")
    parser.add_argument("--skip-count-check", action="store_true", help="Skip the 10/43/130/436 expected-count validation")
    args = parser.parse_args()

    expected = None if args.skip_count_check else DEFAULT_EXPECTED_COUNTS
    try:
        report, report_dict = normalize(args.xlsx, args.out_csv, sheet_name=args.sheet_name, expected_counts=expected)
    except NormalizationError as exc:
        print(f"NORMALIZATION FAILURE: {exc}")
        raise SystemExit(1) from exc

    print(f"Parsed {report.total_rows} rows from {args.xlsx} (sheet {args.sheet_name!r})")
    print(f"Counts by level: {report.n_rows_by_level}")
    print(f"Wrote normalized CSV to {args.out_csv}")
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report_dict, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote normalization report to {args.report_out}")


if __name__ == "__main__":
    main()
