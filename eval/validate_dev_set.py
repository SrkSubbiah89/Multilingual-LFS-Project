"""
eval/validate_dev_set.py

Leakage-safety and coverage validator for eval/dev_set_v1.csv -- the B2
development set used only to select --reranker-candidates (K) before the
single, pre-specified confirmation run on eval/test_set_full130.csv. See
eval/dev_set_schema.md for the canonical field-by-field schema and the
rationale for why this file must be independent of both
eval/test_set_smoke20.csv (reused for prior model/beam/routing experiments)
and eval/test_set_full130.csv (the frozen confirmation set).

This script does not run any classification -- it only checks the CSV
itself. Run it after you've filled in dev_set_v1.csv and before it is ever
passed to dev_sweep.py or pre_run_check.py.

full130 access policy: this script NEVER opens eval/test_set_full130.csv.
The full130 side of the leakage check reads ONLY
eval/configs/full130_leakage_manifest.json (case_ids and sha256 hashes of
normalized text, no gold labels or raw text) -- see --full130-manifest and
check_full130_manifest_overlap-equivalent hash comparison below. This
matches eval/pre_run_check.py's full130 access policy; both scripts use
the same eval.dev_set_schema-canonical normalize_text() to build/consume
that hash, so a hash computed here and a hash computed there are always
comparable.

Usage
-----
    python eval/validate_dev_set.py --dev-set eval/dev_set_v1.csv

Exit codes
----------
0   No hard errors. May still print warnings (e.g. below the *preferred*
    50-case / 15-Arabic target but at/above the 30-case absolute minimum).
1   At least one hard error (leakage against smoke20/full130, missing
    required column, below the 30-case absolute minimum, no cases in one
    of the two languages, malformed gold code, invalid dataset_split,
    etc.) -- do not use this file for K selection until every hard error
    is fixed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from full130_access_guard import guard_against_full130_access  # noqa: E402

# Canonical schema (see eval/dev_set_schema.md for full documentation and
# the migration note explaining prior, now-superseded column names):
#   case_id, language, respondent_text, gold_isco_code, gold_label_source,
#   annotator_or_adjudication_reference, dataset_split
# respondent_text is the SAME field name eval/dev_sweep.py and
# eval/pre_run_check.py already read -- there is exactly one authoritative
# column name for "the text sent to the classifier" across every script
# that touches this file.
REQUIRED_COLUMNS = [
    "case_id", "language", "respondent_text", "gold_isco_code",
    "gold_label_source", "annotator_or_adjudication_reference", "dataset_split",
]

# REQUIRED_COLUMNS doubles as the canonical header: not just "these columns
# must be present" (validate_dev_set()'s per-row check, which only ever
# runs once dev_rows is non-empty) but "the header row must equal this
# list EXACTLY, in this exact order" -- see read_csv_header()/
# validate_csv_header() below, which run independently of row count so a
# malformed header is caught even on a header-only (0 data row) file.
CANONICAL_HEADER = REQUIRED_COLUMNS

VALID_LANGUAGES = {"en", "ar", "mixed"}
VALID_DATASET_SPLIT = "dev_v1"

PREFERRED_MIN_TOTAL = 50
PREFERRED_MIN_ARABIC = 15
ABSOLUTE_MIN_TOTAL = 30

_DEFAULT_DIR = Path(__file__).resolve().parent
_DEFAULT_SMOKE20 = _DEFAULT_DIR / "test_set_smoke20.csv"
_DEFAULT_FULL130_MANIFEST = _DEFAULT_DIR / "configs" / "full130_leakage_manifest.json"

# Semantic ISCO-08 code validation source: the SAME data this project's
# Qdrant-backed classifier is actually built from, not a separately
# maintained copy of the ISCO-08 standard that could drift from it. There
# is no standalone JSON/CSV catalogue in this repo -- backend/rag/
# load_full_isco.py's `_UNIT: list[tuple[str, str]] = [...]` module-level
# literal (436 (code, label_en) tuples) IS the catalogue; it is what
# populates the isco08_unit_groups Qdrant collection every classification
# actually queries against. Read as plain text and regex-parsed here
# rather than imported, specifically so loading the catalogue never pulls
# in qdrant_client/sentence_transformers or risks executing any Qdrant/
# model-loading code path -- see load_isco_unit_group_catalogue().
#
# 2026-08-12: the discrepancy this comment used to document (docstring
# claimed 436, the literal list actually had 441) is resolved -- a direct
# primary-source cross-check against isco.ilo.org's official ISCO-08
# structure export fixed the literal list to genuinely contain 436 unique
# 4-digit codes, matching the official standard exactly (see
# backend/tests/test_load_full_isco_catalogue_consistency.py). This
# catalogue still reflects "codes the classifier can actually return",
# which remains the right standard for THIS check (rejecting a gold code
# the system could never predict) -- it now also happens to be a
# byte-for-byte-equivalent set to the official ISCO-08 unit-group codes,
# with one disclosed exception (Armed Forces codes keep this codebase's
# existing 4-digit format rather than ISCO-08's own bare 3-digit
# convention).
_DEFAULT_ISCO_CATALOGUE_SOURCE = (
    Path(__file__).resolve().parents[1] / "backend" / "rag" / "load_full_isco.py"
)
_ISCO_CATALOGUE_LIST_MARKER = "_UNIT:"  # NOT "_UNIT" alone -- that substring
# also matches "COL_UNIT" earlier in the same file, which would anchor the
# search at the wrong position entirely.


def load_isco_unit_group_catalogue(path: Path = None) -> set:
    """Returns the set of valid 4-digit ISCO-08 unit-group code strings
    from backend/rag/load_full_isco.py's `_UNIT` list, parsed as plain
    text (no import of that module -- see the module-level comment above
    for why). Returns an empty set (not an error) if the source file is
    missing or the expected `_UNIT = [...]` block can't be located --
    callers must decide what an empty catalogue means (main() treats it as
    "semantic ISCO validation skipped for this run" and prints a loud
    warning, never a silent pass)."""
    path = path or _DEFAULT_ISCO_CATALOGUE_SOURCE
    if not path.exists():
        return set()
    src = path.read_text(encoding="utf-8")
    marker_pos = src.find(_ISCO_CATALOGUE_LIST_MARKER)
    if marker_pos == -1:
        return set()
    list_start = src.find("[", marker_pos)
    if list_start == -1:
        return set()
    list_end = src.find("\n]", list_start)
    if list_end == -1:
        return set()
    block = src[list_start:list_end]
    return set(re.findall(r'\(\s*"(\d{4})"\s*,', block))


NORMALIZATION_VERSION = "v1"


def normalize_text(s: str) -> str:
    """Lowercase + collapse internal whitespace. Intentionally mechanical,
    not semantic -- see dev_set_schema.md's "Independence requirements"
    note: this catches exact/whitespace/case duplicates only; a human
    reviewer should still check for paraphrases that survive this check.

    NORMALIZATION_VERSION above and compute_normalize_text_fingerprint()
    below exist so eval/configs/full130_leakage_manifest.json's hashes
    (built by hashing THIS function's output, see
    eval/build_full130_leakage_manifest.py) can never silently drift out
    of sync with what's actually running -- if this function is ever
    edited, its fingerprint changes and check_manifest_normalization_
    integrity() fails closed until the manifest is rebuilt."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def compute_normalize_text_fingerprint() -> str:
    """sha256 of normalize_text()'s live source, full 64-hex-char digest.
    Same pattern as eval/dev_sweep.py's compute_composite_fingerprint() --
    an objective, code-derived drift detector rather than a hand-maintained
    version string that's easy to forget to bump."""
    return hashlib.sha256(inspect.getsource(normalize_text).encode()).hexdigest()


def check_manifest_normalization_integrity(manifest: dict) -> tuple:
    """Returns (ok: bool, message: str). Fails closed (ok=False) if the
    manifest predates fingerprinting (missing fields -- treated as
    untrusted, NOT silently accepted) or if the live normalize_text()
    no longer matches what built the manifest's hashes -- in either case,
    the manifest's normalized_text_sha256 values are not trustworthy for
    a leakage comparison against the CURRENT normalize_text()."""
    manifest_version = manifest.get("normalization_version")
    manifest_fingerprint = manifest.get("normalization_fingerprint")
    if not manifest_version or not manifest_fingerprint:
        return False, (
            "manifest is missing normalization_version/normalization_fingerprint -- "
            "predates normalization-integrity enforcement and cannot be trusted for "
            "a hash comparison until rebuilt with eval/build_full130_leakage_manifest.py."
        )
    live_fingerprint = compute_normalize_text_fingerprint()
    if manifest_fingerprint != live_fingerprint:
        return False, (
            f"manifest normalization_fingerprint={manifest_fingerprint!r} does not match "
            f"the live normalize_text() fingerprint={live_fingerprint!r} -- normalize_text() "
            f"has changed since the manifest was built. Its normalized_text_sha256 hashes are "
            f"no longer comparable to a hash computed now. Rebuild the manifest with "
            f"eval/build_full130_leakage_manifest.py before trusting any leakage check."
        )
    if manifest_version != NORMALIZATION_VERSION:
        return False, (
            f"manifest normalization_version={manifest_version!r} does not match the live "
            f"NORMALIZATION_VERSION={NORMALIZATION_VERSION!r} (even though the fingerprint "
            f"matched -- this should not happen; treat as a bug and investigate before trusting "
            f"the manifest)."
        )
    return True, f"normalization_version={manifest_version!r}, fingerprint confirmed live-matching."


def load_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate_csv_structure(path: Path) -> list:
    """CSV-grammar checks distinct from schema/field-value validation --
    malformed/unterminated quoting and ragged rows (a row with more or
    fewer fields than the header, which csv.DictReader would otherwise
    silently pad with None or drop into a None-keyed overflow list rather
    than flag). Returns a list of error strings; empty means the file
    parses as well-formed CSV. Does not check for MISSING required
    COLUMNS (REQUIRED_COLUMNS, checked separately in validate_dev_set())
    -- only that the CSV grammar itself is sound. A properly quoted field
    containing embedded newlines (multiline respondent_text) is valid CSV
    and is NOT flagged here -- Python's csv module (opened with
    newline="", as load_csv_rows() already does) handles that natively."""
    errors = []
    if not path.exists():
        return errors  # missing-file handling belongs to the caller
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f, strict=True)
            try:
                header = next(reader)
            except StopIteration:
                return errors  # empty file -- "0 data rows" is handled elsewhere
            expected_n = len(header)
            for i, row in enumerate(reader, start=2):  # 1-based, +1 for the header row
                if len(row) != expected_n:
                    errors.append(
                        f"Row {i}: has {len(row)} field(s), expected {expected_n} "
                        f"(matching the {expected_n}-column header) -- malformed row or "
                        f"unterminated/mismatched quoting."
                    )
    except csv.Error as exc:
        errors.append(f"CSV parse error: {exc}")
    except UnicodeDecodeError as exc:
        errors.append(f"File is not valid UTF-8: {exc}")
    return errors


def read_csv_header(path: Path) -> list:
    """Returns the raw header row as a list of column names, in file
    order -- or None if the file doesn't exist or has no header row at all
    (0 bytes / immediately EOF). Deliberately separate from load_csv_rows()
    (which returns dict rows keyed by header, losing the distinction
    between "no header" and "header present but zero data rows") so header
    validation can run independently of, and before, any row-count check."""
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return None


def validate_csv_header(header: list) -> list:
    """Returns a list of error strings; empty means header == CANONICAL_
    HEADER exactly (same 7 columns, same order, no duplicates, no extras).
    Runs independently of row count -- must reject a malformed header even
    when the file has zero data rows, rather than only reporting "empty"
    once row-count validation runs (see main()'s call order). Reports
    EVERY problem found (missing/duplicate/extra/reordered), not just the
    first, so a single run tells a user everything wrong with the header
    at once."""
    if header is None:
        return ["CSV has no header row (file is empty or unreadable)."]

    errors = []
    dupes = sorted({c for c in header if header.count(c) > 1})
    if dupes:
        errors.append(f"Duplicate column(s) in header: {dupes}")

    header_set = set(header)
    missing = [c for c in CANONICAL_HEADER if c not in header_set]
    if missing:
        errors.append(f"Missing required column(s): {missing}")

    extra = [c for c in header if c not in CANONICAL_HEADER]
    if extra:
        errors.append(f"Unexpected extra column(s): {extra}")

    if not errors and header != CANONICAL_HEADER:
        errors.append(
            f"Header columns are reordered: got {header}, expected exactly "
            f"{CANONICAL_HEADER} (same columns, wrong order)."
        )
    return errors


def load_case_ids(rows: list[dict], id_column: str = "case_id") -> set[str]:
    return {r[id_column] for r in rows if r.get(id_column)}


def load_normalized_texts(rows: list[dict], text_column: str) -> set[str]:
    return {normalize_text(r[text_column]) for r in rows if r.get(text_column)}


@dataclass
class ValidationReport:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_dev_set(
    dev_rows: list[dict],
    other_case_ids: set,
    other_normalized_texts: set = None,
    other_text_hashes: set = None,
    other_major_groups: set = None,
    valid_isco_codes: set = None,
) -> ValidationReport:
    """Pure validation logic, no filesystem/CLI -- takes already-loaded rows.

    other_case_ids: case_ids from smoke20 and/or the full130 leakage
        manifest -- collision with any of them is an error regardless of
        source.
    other_normalized_texts: RAW normalized text from sources this script is
        allowed to open directly (smoke20). Compared via exact string match
        after normalize_text().
    other_text_hashes: sha256(normalize_text(...)) hashes from sources this
        script must NOT open directly (full130 -- see
        eval/configs/full130_leakage_manifest.json). Compared by hashing
        each dev-set row's own normalized text and checking membership --
        eval/test_set_full130.csv's raw text is never seen by this
        function or its caller.
    other_major_groups (optional): major groups known to be present in
        full130, used only to flag zero-coverage groups as a warning, never
        a hard error. Derived by the caller from whatever full130-adjacent
        source they have available (the leakage manifest does not carry
        gold labels, so this is often empty/unavailable -- see main()).
    valid_isco_codes (optional): the set of real ISCO-08 unit-group codes
        (see load_isco_unit_group_catalogue()). When None or empty, the
        semantic-existence check is skipped for this call (format-only
        4-digit validation still applies) -- main() always attempts to
        load and pass this, and warns loudly rather than silently skipping
        when the catalogue can't be loaded.
    """
    report = ValidationReport()
    other_normalized_texts = other_normalized_texts or set()
    other_text_hashes = other_text_hashes or set()
    other_major_groups = other_major_groups or set()
    valid_isco_codes = valid_isco_codes or set()

    if not dev_rows:
        report.errors.append("Dev set is empty (no data rows).")
        return report

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in dev_rows[0]]
    if missing_cols:
        report.errors.append(f"Missing required column(s): {', '.join(missing_cols)}")
        return report  # remaining checks assume all columns exist

    seen_ids: set = set()
    seen_texts: dict = {}  # normalized text -> case_id that first used it
    lang_counts = {"en": 0, "ar": 0, "mixed": 0}
    major_group_counts: dict = {}
    unknown_lang_rows = []

    for row in dev_rows:
        cid = (row.get("case_id") or "").strip()
        text = row.get("respondent_text") or ""
        norm_text = normalize_text(text)
        norm_text_hash = hashlib.sha256(norm_text.encode()).hexdigest() if norm_text else ""
        lang = (row.get("language") or "").strip().lower()
        gold = (row.get("gold_isco_code") or "").strip()
        label_source = (row.get("gold_label_source") or "").strip()
        annotator_ref = (row.get("annotator_or_adjudication_reference") or "").strip()
        dataset_split = (row.get("dataset_split") or "").strip()

        if not cid:
            report.errors.append("Row with blank case_id.")
            continue
        if cid in seen_ids:
            report.errors.append(f"Duplicate case_id within dev set: {cid!r}")
        seen_ids.add(cid)

        if cid in other_case_ids:
            report.errors.append(
                f"case_id {cid!r} collides with an existing smoke20/full130 case_id."
            )

        if not text.strip():
            report.errors.append(f"case_id={cid!r}: respondent_text is blank.")
        elif norm_text in other_normalized_texts:
            report.errors.append(
                f"case_id={cid!r}: respondent_text duplicates (or near-duplicates, "
                f"after whitespace/case normalisation) an existing smoke20 case."
            )
        elif norm_text_hash and norm_text_hash in other_text_hashes:
            report.errors.append(
                f"case_id={cid!r}: respondent_text's normalized-text hash matches an "
                f"existing full130 case (checked via the leakage manifest, not the raw file)."
            )
        elif norm_text in seen_texts:
            report.errors.append(
                f"case_id={cid!r}: respondent_text duplicates case_id="
                f"{seen_texts[norm_text]!r} within the dev set itself."
            )
        seen_texts.setdefault(norm_text, cid)

        if lang not in VALID_LANGUAGES:
            unknown_lang_rows.append(cid)
        else:
            lang_counts[lang] += 1

        if not re.fullmatch(r"\d{4}", gold):
            report.errors.append(
                f"case_id={cid!r}: gold_isco_code {gold!r} is not a 4-digit code."
            )
        else:
            # major_group is not a stored column (see eval/dev_set_schema.md's
            # migration note) -- derived here so stratification can still be
            # reported without a redundant column that could disagree with
            # gold_isco_code[0].
            major_group_counts[gold[:1]] = major_group_counts.get(gold[:1], 0) + 1
            # Semantic check: syntactically valid (4 digits) but not a real
            # ISCO-08 unit-group code -- e.g. 0000 or 9999 -- would silently
            # pass format validation alone. See load_isco_unit_group_catalogue().
            if valid_isco_codes and gold not in valid_isco_codes:
                report.errors.append(
                    f"case_id={cid!r}: gold_isco_code {gold!r} is a syntactically valid "
                    f"4-digit code but does not exist in the ISCO-08 unit-group catalogue "
                    f"(backend/rag/load_full_isco.py's _UNIT list, {len(valid_isco_codes)} "
                    f"code(s)) -- the classifier could never predict this code."
                )

        if not label_source:
            report.errors.append(f"case_id={cid!r}: gold_label_source is blank.")

        if not annotator_ref:
            report.errors.append(f"case_id={cid!r}: annotator_or_adjudication_reference is blank.")

        if dataset_split != VALID_DATASET_SPLIT:
            report.errors.append(
                f"case_id={cid!r}: dataset_split={dataset_split!r} must be "
                f"{VALID_DATASET_SPLIT!r}."
            )

    if unknown_lang_rows:
        report.errors.append(
            f"{len(unknown_lang_rows)} row(s) have a language outside "
            f"{sorted(VALID_LANGUAGES)}: case_ids={unknown_lang_rows}"
        )

    total = len(dev_rows)
    report.stats.update({
        "total_cases": total,
        "language_counts": lang_counts,
        "major_group_counts": major_group_counts,
    })

    if total < ABSOLUTE_MIN_TOTAL:
        report.errors.append(
            f"Only {total} case(s); absolute minimum is {ABSOLUTE_MIN_TOTAL}."
        )
    elif total < PREFERRED_MIN_TOTAL:
        report.warnings.append(
            f"Only {total} case(s); preferred target is {PREFERRED_MIN_TOTAL}+ "
            f"(above the {ABSOLUTE_MIN_TOTAL}-case absolute minimum, so this is "
            f"usable, but a larger dev set would give a more reliable K choice)."
        )

    if lang_counts["ar"] < PREFERRED_MIN_ARABIC:
        report.warnings.append(
            f"Only {lang_counts['ar']} Arabic case(s); preferred target is "
            f"{PREFERRED_MIN_ARABIC}+."
        )
    if lang_counts["en"] == 0:
        report.errors.append("No English cases at all.")
    if lang_counts["ar"] == 0:
        report.errors.append("No Arabic cases at all.")

    missing_major_groups = other_major_groups - set(major_group_counts.keys())
    if missing_major_groups:
        report.warnings.append(
            f"Major group(s) with zero dev-set coverage but present in "
            f"full130: {sorted(missing_major_groups)}"
        )

    return report


def load_full130_manifest_raw(path: Path) -> dict:
    """Returns the raw manifest dict (including normalization_version/
    normalization_fingerprint, for check_manifest_normalization_integrity()),
    or {} if missing. NEVER opens eval/test_set_full130.csv itself."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_full130_manifest(path: Path) -> tuple:
    """Returns (case_ids: set, text_hashes: set) from the pre-built leakage
    manifest -- NEVER opens eval/test_set_full130.csv itself. Returns two
    empty sets (not an error) if the manifest is missing, matching
    load_csv_rows()'s existing "missing file -> empty, caller decides what
    that means" convention; main() prints a warning in that case."""
    manifest = load_full130_manifest_raw(path)
    return set(manifest.get("case_ids", [])), set(manifest.get("normalized_text_sha256", []))


def _run(args) -> None:
    """The actual CLI execution path, called from main() wrapped in
    guard_against_full130_access() -- see that function for why this
    script must never open eval/test_set_full130.csv directly, and
    eval/full130_access_guard.py for what "wrapped in the guard" actually
    checks at runtime."""
    # main() already validated existence via parser.error() before calling
    # _run() -- this defensive re-check only matters if _run() is ever
    # called directly (e.g. from a test) without going through main() first.
    if not args.dev_set.exists():
        print(f"FATAL: Dev set not found: {args.dev_set}", file=sys.stderr)
        sys.exit(1)

    # Header validation runs FIRST and independently of row count -- a
    # malformed header must be reported as a header/schema error, not
    # masked by (or conflated with) the "0 data rows" empty-set error that
    # validate_dev_set() would otherwise report first for a header-only file.
    header = read_csv_header(args.dev_set)
    header_errors = validate_csv_header(header)
    if header_errors:
        print(f"FATAL: {args.dev_set} header does not match the canonical schema:", file=sys.stderr)
        for e in header_errors:
            print(f"  ERROR: {e}", file=sys.stderr)
        print(f"  Expected exactly: {CANONICAL_HEADER}", file=sys.stderr)
        sys.exit(1)

    structure_errors = validate_csv_structure(args.dev_set)
    if structure_errors:
        print(f"FATAL: {args.dev_set} is not well-formed CSV:", file=sys.stderr)
        for e in structure_errors:
            print(f"  ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    manifest = load_full130_manifest_raw(args.full130_manifest)
    if manifest:
        norm_ok, norm_msg = check_manifest_normalization_integrity(manifest)
        if not norm_ok:
            print(f"FATAL: {args.full130_manifest} normalization integrity check failed: "
                  f"{norm_msg}", file=sys.stderr)
            sys.exit(1)
        print(f"Manifest normalization integrity OK: {norm_msg}")

    # Semantic ISCO catalogue: FATAL (not a warning) if it can't be loaded.
    # An empty/unparsable catalogue here would silently downgrade every
    # gold_isco_code check to format-only (4-digit) validation, which would
    # let 0000/9999-style nonexistent codes pass -- treated as a hard
    # failure so that can never happen unnoticed. validate_dev_set() itself
    # (the pure function) still accepts an empty/None valid_isco_codes for
    # unit testing -- this fail-closed behaviour is specific to the CLI
    # entry point, not the underlying library function.
    valid_isco_codes = load_isco_unit_group_catalogue(args.isco_catalogue_source)
    if not valid_isco_codes:
        print(
            f"FATAL: could not load the classifier-supported ISCO-08 catalogue from "
            f"{args.isco_catalogue_source} -- refusing to validate gold_isco_code with "
            f"format-only (4-digit) checking, which would silently accept nonexistent "
            f"codes like 0000/9999. Fix the catalogue source before re-running.",
            file=sys.stderr,
        )
        sys.exit(1)

    dev_rows = load_csv_rows(args.dev_set)
    smoke20_rows = load_csv_rows(args.smoke20)
    full130_ids = set(manifest.get("case_ids", []))
    full130_text_hashes = set(manifest.get("normalized_text_sha256", []))

    if not smoke20_rows:
        print(f"WARNING: could not load {args.smoke20} (0 rows) -- case_id/text "
              f"collision checks against it will be skipped.", file=sys.stderr)
    if not full130_ids:
        print(f"WARNING: could not load {args.full130_manifest} (0 case_ids) -- full130 "
              f"collision checks will be skipped. This does NOT fall back to reading "
              f"eval/test_set_full130.csv directly.", file=sys.stderr)

    other_ids = load_case_ids(smoke20_rows) | full130_ids
    other_texts = load_normalized_texts(smoke20_rows, "input_text")
    # other_major_groups is intentionally NOT populated from full130 here --
    # the leakage manifest carries no gold labels (by design, see
    # eval/configs/full130_leakage_manifest.json's own docstring), and this
    # script must not open eval/test_set_full130.csv to derive them another
    # way. The major-group-coverage warning therefore only ever fires from
    # data a caller explicitly supplies to validate_dev_set() directly.

    report = validate_dev_set(
        dev_rows, other_ids, other_texts, full130_text_hashes,
        valid_isco_codes=valid_isco_codes,
    )

    print(f"Loaded {len(dev_rows)} row(s) from {args.dev_set}")
    if report.stats:
        print(f"stats: {report.stats}")
    if report.warnings:
        print(f"\n{len(report.warnings)} warning(s):")
        for w in report.warnings:
            print(f"  WARNING: {w}")
    if report.errors:
        print(f"\n{len(report.errors)} error(s):")
        for e in report.errors:
            print(f"  ERROR: {e}")
        print("\nFAIL: fix the error(s) above before using this file for K selection.")
        sys.exit(1)

    print("\nPASS: no hard errors." + (" (see warnings above)" if report.warnings else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-set", required=True, type=Path)
    parser.add_argument("--smoke20", type=Path, default=_DEFAULT_SMOKE20)
    parser.add_argument(
        "--full130-manifest", type=Path, default=_DEFAULT_FULL130_MANIFEST,
        help=(
            "Leakage-detection manifest (case_ids + normalized-text sha256 hashes only, "
            "no labels) -- NOT eval/test_set_full130.csv itself, which this script never opens."
        ),
    )
    parser.add_argument(
        "--isco-catalogue-source", type=Path, default=_DEFAULT_ISCO_CATALOGUE_SOURCE,
        help=(
            "Source of valid ISCO-08 unit-group codes for semantic validation -- "
            "backend/rag/load_full_isco.py, read as plain text (never imported). An "
            "empty/unparsable catalogue is FATAL (exit 1), not a soft warning."
        ),
    )
    args = parser.parse_args()

    if not args.dev_set.exists():
        parser.error(f"Dev set not found: {args.dev_set}")

    with guard_against_full130_access():
        _run(args)


if __name__ == "__main__":
    main()
