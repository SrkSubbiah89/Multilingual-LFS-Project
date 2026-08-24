"""
Inspects the raw WISCO workbook and prints/writes a structural report BEFORE
any parsing logic is written, so parse_wisco.py can assert a verified
contract (sheet names, column names, dtypes) instead of an assumed one.

CANONICAL FILE (corrected 2026-08-02): the 2023-02-02 file originally
downloaded here was the *oldest* of 4 Zenodo versions under this concept
DOI. The 2023-08-18 file below is the correct/latest one -- it is the exact
filename the original Module A plan expected, and it is materially larger
and more current (4,745 vs 4,232 titles; 1,423-row changelog of
translation fixes). The old file is kept on disk for the provenance
record's audit trail but is no longer used by any script. See
Documentation/Phase_2/Week_1/PROVENANCE.md for the full correction note.

Usage:
    python eval/legacy_thesis_ch6/wisco/inspect_wisco.py
"""
import hashlib
import json
from pathlib import Path

import openpyxl

RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"
WORKBOOK = RAW_DIR / "occupations_ISCO08_5dgt_55languages_4000titles_surveycodings_20230818.xlsx"
SUPERSEDED_WORKBOOK = RAW_DIR / "Occupations_ISCO08_5dgt_55languages_4000titles_with_mapping_20230202.xlsx"
OUT_JSON = Path(__file__).resolve().parent / "data" / "interim" / "wisco_structure_inspection.json"


def main() -> None:
    assert WORKBOOK.exists(), f"Workbook not found at {WORKBOOK}"

    wb = openpyxl.load_workbook(WORKBOOK, read_only=True, data_only=True)

    report = {"sheets": []}

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        n_rows = ws.max_row
        n_cols = ws.max_column

        header_row = next(ws.iter_rows(min_row=1, max_row=1, values_only=True), ())
        header_row = [h for h in header_row if h is not None]

        # Sample a few data rows (row 2..6) for each column to see example values.
        sample_rows = list(ws.iter_rows(min_row=2, max_row=6, values_only=True))

        columns = []
        for col_idx, col_name in enumerate(header_row):
            examples = []
            for r in sample_rows:
                if col_idx < len(r) and r[col_idx] is not None:
                    examples.append(r[col_idx])
            columns.append({
                "name": col_name,
                "index": col_idx,
                "examples": examples[:5],
            })

        report["sheets"].append({
            "name": sheet_name,
            "rows": n_rows,
            "cols": n_cols,
            "header": header_row,
            "columns": columns,
        })

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print(f"Wrote structural report to {OUT_JSON}")
    for s in report["sheets"]:
        print(f"\n=== Sheet: {s['name']} ({s['rows']} rows x {s['cols']} cols) ===")
        print("Header:", s["header"])


if __name__ == "__main__":
    main()
