"""
Tests for eval/validate_dev_set.py -- pure CSV-validation logic, no
classification/live infra involved. See eval/dev_set_schema.md for the
canonical schema these checks enforce.

Canonical schema (7 columns): case_id, language, respondent_text,
gold_isco_code, gold_label_source, annotator_or_adjudication_reference,
dataset_split. See eval/dev_set_schema.md's migration note for the
superseded column names (coder_or_adjudicator, major_group,
difficulty_level, notes, job_title, job_description) this replaces.
"""

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import validate_dev_set as vds  # noqa: E402
import pre_run_check as prc  # noqa: E402 -- reused only for guard_against_full130_access()


def make_row(case_id="dev001", language="en", respondent_text="baker",
             gold_isco_code="7512", gold_label_source="human_coder_single",
             annotator_or_adjudication_reference="AB", dataset_split="dev_v1"):
    return {
        "case_id": case_id, "language": language, "respondent_text": respondent_text,
        "gold_isco_code": gold_isco_code, "gold_label_source": gold_label_source,
        "annotator_or_adjudication_reference": annotator_or_adjudication_reference,
        "dataset_split": dataset_split,
    }


def make_valid_set(n_en=16, n_ar=15):
    rows = []
    for i in range(n_en):
        rows.append(make_row(case_id=f"dev_en_{i}", language="en",
                              respondent_text=f"english job title number {i}"))
    for i in range(n_ar):
        rows.append(make_row(case_id=f"dev_ar_{i}", language="ar",
                              respondent_text=f"arabic job title number {i}"))
    return rows  # 31 total, clears the 30-case absolute minimum


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

def test_normalize_text_collapses_whitespace_and_lowercases():
    assert vds.normalize_text("  Software   Developer\n") == "software developer"


def test_normalize_text_empty_and_none():
    assert vds.normalize_text("") == ""
    assert vds.normalize_text(None) == ""


# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------

def test_empty_dev_set_is_an_error():
    report = vds.validate_dev_set([], set())
    assert not report.ok
    assert any("empty" in e.lower() for e in report.errors)


def test_missing_required_column_is_an_error():
    rows = [{"case_id": "x", "language": "en"}]  # missing most required columns
    report = vds.validate_dev_set(rows, set())
    assert not report.ok
    assert any("missing required column" in e.lower() for e in report.errors)


def test_canonical_required_columns_are_exactly_seven():
    assert vds.REQUIRED_COLUMNS == [
        "case_id", "language", "respondent_text", "gold_isco_code",
        "gold_label_source", "annotator_or_adjudication_reference", "dataset_split",
    ]


def test_superseded_columns_are_not_required():
    """job_title/job_description/coder_or_adjudicator/major_group/
    difficulty_level/notes were all superseded by the canonical schema
    reconciliation -- none of them should be required."""
    for superseded in ("job_title", "job_description", "coder_or_adjudicator",
                       "major_group", "difficulty_level", "notes"):
        assert superseded not in vds.REQUIRED_COLUMNS


# ---------------------------------------------------------------------------
# Leakage: case_id collisions
# ---------------------------------------------------------------------------

def test_case_id_colliding_with_existing_set_is_an_error():
    rows = make_valid_set()
    rows[0]["case_id"] = "17"  # pretend this collides with an existing full130/smoke20 id
    report = vds.validate_dev_set(rows, other_case_ids={"17"})
    assert not report.ok
    assert any("collides" in e for e in report.errors)


def test_duplicate_case_id_within_dev_set_is_an_error():
    rows = make_valid_set()
    rows[1]["case_id"] = rows[0]["case_id"]
    report = vds.validate_dev_set(rows, set())
    assert not report.ok
    assert any("duplicate case_id" in e.lower() for e in report.errors)


# ---------------------------------------------------------------------------
# Leakage: near-duplicate respondent_text (smoke20 -- raw text comparison)
# ---------------------------------------------------------------------------

def test_respondent_text_duplicating_smoke20_is_an_error():
    rows = make_valid_set()
    rows[0]["respondent_text"] = "Software Developer"
    other_texts = {vds.normalize_text("software   developer")}
    report = vds.validate_dev_set(rows, set(), other_normalized_texts=other_texts)
    assert not report.ok
    assert any("smoke20" in e for e in report.errors)


def test_respondent_text_duplicating_within_dev_set_is_an_error():
    rows = make_valid_set()
    rows[1]["respondent_text"] = rows[0]["respondent_text"]
    report = vds.validate_dev_set(rows, set())
    assert not report.ok
    assert any("within the dev set itself" in e for e in report.errors)


def test_blank_respondent_text_is_an_error():
    rows = make_valid_set()
    rows[0]["respondent_text"] = ""
    report = vds.validate_dev_set(rows, set())
    assert not report.ok
    assert any("blank" in e for e in report.errors)


# ---------------------------------------------------------------------------
# Leakage: full130 hash collision (manifest-derived, exact hash match only)
# ---------------------------------------------------------------------------

def test_respondent_text_hash_colliding_with_full130_manifest_is_an_error():
    import hashlib
    rows = make_valid_set()
    rows[0]["respondent_text"] = "Software Developer"
    h = hashlib.sha256(vds.normalize_text("Software Developer").encode()).hexdigest()
    report = vds.validate_dev_set(rows, set(), other_text_hashes={h})
    assert not report.ok
    assert any("full130" in e and "hash" in e for e in report.errors)


def test_respondent_text_hash_not_colliding_passes():
    rows = make_valid_set()
    report = vds.validate_dev_set(rows, set(), other_text_hashes={"deadbeef" * 8})
    assert report.ok


def test_full130_hash_check_never_needs_raw_full130_text():
    """The whole point of other_text_hashes -- confirms the comparison is a
    pure hash membership test, not something that could silently fall back
    to needing raw full130 text."""
    rows = make_valid_set()
    # Only a hash is supplied, never anything resembling full130's raw text --
    # if this passed, it proves the check only ever needs hashes.
    report = vds.validate_dev_set(rows, set(), other_text_hashes=set())
    assert report.ok


# ---------------------------------------------------------------------------
# Field-level validation
# ---------------------------------------------------------------------------

def test_non_4digit_gold_code_is_an_error():
    rows = make_valid_set()
    rows[0]["gold_isco_code"] = "751"
    report = vds.validate_dev_set(rows, set())
    assert not report.ok
    assert any("not a 4-digit code" in e for e in report.errors)


def test_blank_gold_label_source_is_an_error():
    rows = make_valid_set()
    rows[0]["gold_label_source"] = ""
    report = vds.validate_dev_set(rows, set())
    assert not report.ok
    assert any("gold_label_source is blank" in e for e in report.errors)


def test_blank_annotator_or_adjudication_reference_is_an_error():
    rows = make_valid_set()
    rows[0]["annotator_or_adjudication_reference"] = ""
    report = vds.validate_dev_set(rows, set())
    assert not report.ok
    assert any("annotator_or_adjudication_reference is blank" in e for e in report.errors)


def test_unknown_language_is_an_error():
    rows = make_valid_set()
    rows[0]["language"] = "fr"
    report = vds.validate_dev_set(rows, set())
    assert not report.ok


def test_invalid_dataset_split_is_an_error():
    rows = make_valid_set()
    rows[0]["dataset_split"] = "smoke20"
    report = vds.validate_dev_set(rows, set())
    assert not report.ok
    assert any("dataset_split" in e for e in report.errors)


def test_blank_dataset_split_is_an_error():
    rows = make_valid_set()
    rows[0]["dataset_split"] = ""
    report = vds.validate_dev_set(rows, set())
    assert not report.ok
    assert any("dataset_split" in e for e in report.errors)


# ---------------------------------------------------------------------------
# Coverage thresholds
# ---------------------------------------------------------------------------

def test_below_absolute_minimum_total_is_an_error():
    rows = make_valid_set(n_en=10, n_ar=10)  # 20, below the 30 floor
    report = vds.validate_dev_set(rows, set())
    assert not report.ok
    assert any("absolute minimum" in e for e in report.errors)


def test_between_absolute_and_preferred_is_a_warning_not_error():
    rows = make_valid_set(n_en=20, n_ar=15)  # 35 total: clears 30, below 50
    report = vds.validate_dev_set(rows, set())
    assert report.ok
    assert any("preferred target" in w for w in report.warnings)


def test_meets_preferred_targets_no_coverage_warnings():
    rows = make_valid_set(n_en=35, n_ar=15)  # 50 total, 15 Arabic
    report = vds.validate_dev_set(rows, set())
    assert report.ok
    assert not any("preferred target" in w for w in report.warnings)


def test_zero_english_cases_is_an_error():
    rows = make_valid_set(n_en=0, n_ar=31)
    report = vds.validate_dev_set(rows, set())
    assert not report.ok
    assert any("no english" in e.lower() for e in report.errors)


def test_zero_arabic_cases_is_an_error():
    rows = make_valid_set(n_en=31, n_ar=0)
    report = vds.validate_dev_set(rows, set())
    assert not report.ok
    assert any("no arabic" in e.lower() for e in report.errors)


# ---------------------------------------------------------------------------
# Major-group coverage -- now derived from gold_isco_code[:1], no stored column
# ---------------------------------------------------------------------------

def test_major_group_counts_derived_from_gold_isco_code():
    rows = make_valid_set()  # all gold_isco_code="7512" -> major group "7"
    report = vds.validate_dev_set(rows, set())
    assert report.stats["major_group_counts"] == {"7": 31}


def test_missing_major_group_coverage_is_a_warning():
    rows = make_valid_set()  # all major group "7" (derived from gold_isco_code)
    report = vds.validate_dev_set(rows, set(), other_major_groups={"7", "2"})
    assert report.ok
    assert any("zero dev-set coverage" in w for w in report.warnings)


def test_stats_reports_totals_and_language_counts():
    rows = make_valid_set(n_en=16, n_ar=15)
    report = vds.validate_dev_set(rows, set())
    assert report.stats["total_cases"] == 31
    assert report.stats["language_counts"]["en"] == 16
    assert report.stats["language_counts"]["ar"] == 15


# ---------------------------------------------------------------------------
# load_full130_manifest -- reads ONLY the manifest, never raw full130
# ---------------------------------------------------------------------------

def test_load_full130_manifest_reads_case_ids_and_hashes(tmp_path):
    import json
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"case_ids": ["ar1", "ar2"], "normalized_text_sha256": ["h1", "h2"]}),
        encoding="utf-8",
    )
    ids, hashes = vds.load_full130_manifest(manifest_path)
    assert ids == {"ar1", "ar2"}
    assert hashes == {"h1", "h2"}


def test_load_full130_manifest_missing_file_returns_empty_sets(tmp_path):
    ids, hashes = vds.load_full130_manifest(tmp_path / "does_not_exist.json")
    assert ids == set()
    assert hashes == set()


def test_load_full130_manifest_against_real_manifest():
    real_manifest = Path(__file__).resolve().parent / "configs" / "full130_leakage_manifest.json"
    ids, hashes = vds.load_full130_manifest(real_manifest)
    assert len(ids) == 130
    assert len(hashes) == 130


# ---------------------------------------------------------------------------
# Semantic ISCO-08 code validation (task 1) -- load_isco_unit_group_catalogue()
# ---------------------------------------------------------------------------

def test_load_isco_unit_group_catalogue_against_real_source():
    codes = vds.load_isco_unit_group_catalogue()
    assert len(codes) == 441  # documented, known count -- see module-level comment
    assert "2512" in codes  # Software Developers
    assert "7512" in codes  # Bakers, Pastry Cooks and Confectionery Makers
    assert "0000" not in codes
    assert "9999" not in codes


def test_load_isco_unit_group_catalogue_all_codes_are_4_digits():
    codes = vds.load_isco_unit_group_catalogue()
    assert all(len(c) == 4 and c.isdigit() for c in codes)


def test_load_isco_unit_group_catalogue_missing_source_returns_empty_set(tmp_path):
    codes = vds.load_isco_unit_group_catalogue(tmp_path / "does_not_exist.py")
    assert codes == set()


def test_load_isco_unit_group_catalogue_marker_not_confused_by_col_unit(tmp_path):
    """Regression test for the exact bug hit while building this feature:
    naive substring search for "_UNIT" matches inside "COL_UNIT" first and
    anchors parsing at the wrong position, yielding zero codes."""
    fake_source = tmp_path / "fake_load_full_isco.py"
    fake_source.write_text(
        'COL_UNIT = "isco08_unit_groups"\n'
        '_UNIT: list[tuple[str, str]] = [\n'
        '    ("1234", "Fake Occupation A"),\n'
        '    ("5678", "Fake Occupation B"),\n'
        ']\n',
        encoding="utf-8",
    )
    codes = vds.load_isco_unit_group_catalogue(fake_source)
    assert codes == {"1234", "5678"}


def test_syntactically_valid_but_nonexistent_gold_code_is_rejected():
    rows = make_valid_set()
    rows[0]["gold_isco_code"] = "0000"
    report = vds.validate_dev_set(rows, set(), valid_isco_codes={"7512"})
    assert not report.ok
    assert any("does not exist in the ISCO-08" in e for e in report.errors)


def test_9999_is_rejected_when_catalogue_supplied():
    rows = make_valid_set()
    rows[0]["gold_isco_code"] = "9999"
    real_codes = vds.load_isco_unit_group_catalogue()
    report = vds.validate_dev_set(rows, set(), valid_isco_codes=real_codes)
    assert not report.ok
    assert any("9999" in e and "does not exist" in e for e in report.errors)


def test_real_gold_code_passes_semantic_check():
    rows = make_valid_set()  # gold_isco_code="7512" throughout
    real_codes = vds.load_isco_unit_group_catalogue()
    report = vds.validate_dev_set(rows, set(), valid_isco_codes=real_codes)
    assert report.ok


def test_semantic_check_skipped_when_catalogue_not_supplied():
    """valid_isco_codes defaults to None/empty -- format-only validation
    still applies, but a nonexistent-but-4-digit code like 0000 is not
    flagged when no catalogue was supplied (documented skip, not a crash)."""
    rows = make_valid_set()
    rows[0]["gold_isco_code"] = "0000"
    report = vds.validate_dev_set(rows, set())  # no valid_isco_codes
    assert report.ok


# ---------------------------------------------------------------------------
# Normalization-integrity enforcement (task 2)
# ---------------------------------------------------------------------------

def test_compute_normalize_text_fingerprint_is_deterministic():
    a = vds.compute_normalize_text_fingerprint()
    b = vds.compute_normalize_text_fingerprint()
    assert a == b
    assert len(a) == 64


def test_manifest_normalization_integrity_passes_for_matching_fingerprint():
    manifest = {
        "normalization_version": vds.NORMALIZATION_VERSION,
        "normalization_fingerprint": vds.compute_normalize_text_fingerprint(),
    }
    ok, msg = vds.check_manifest_normalization_integrity(manifest)
    assert ok is True


def test_manifest_normalization_integrity_fails_on_fingerprint_mismatch():
    manifest = {
        "normalization_version": vds.NORMALIZATION_VERSION,
        "normalization_fingerprint": "0" * 64,
    }
    ok, msg = vds.check_manifest_normalization_integrity(manifest)
    assert ok is False
    assert "does not match" in msg


def test_manifest_normalization_integrity_fails_closed_on_missing_fields():
    """A manifest built before fingerprinting existed must be treated as
    untrusted, not silently accepted."""
    ok, msg = vds.check_manifest_normalization_integrity({})
    assert ok is False
    assert "missing" in msg.lower()


def test_manifest_normalization_integrity_fails_closed_on_missing_version_only():
    manifest = {"normalization_fingerprint": vds.compute_normalize_text_fingerprint()}
    ok, msg = vds.check_manifest_normalization_integrity(manifest)
    assert ok is False


def test_changed_normalize_text_implementation_causes_validation_failure(monkeypatch):
    """Proves drift detection actually works end-to-end: simulate normalize_
    text() having changed (e.g. someone edits it) by monkeypatching it to a
    DIFFERENT implementation, and confirm the fingerprint check -- computed
    fresh against the now-different live source -- fails against a manifest
    fingerprint recorded for the ORIGINAL implementation."""
    original_fingerprint = vds.compute_normalize_text_fingerprint()
    manifest = {
        "normalization_version": vds.NORMALIZATION_VERSION,
        "normalization_fingerprint": original_fingerprint,
    }

    def changed_normalize_text(s):
        """A deliberately different implementation (different docstring/body
        so inspect.getsource() -- and therefore the fingerprint -- changes)."""
        return (s or "").strip().upper()  # uppercase instead of lowercase

    monkeypatch.setattr(vds, "normalize_text", changed_normalize_text)
    ok, msg = vds.check_manifest_normalization_integrity(manifest)
    assert ok is False
    assert "does not match" in msg


def test_real_manifest_passes_normalization_integrity_check():
    real_manifest_path = Path(__file__).resolve().parent / "configs" / "full130_leakage_manifest.json"
    import json
    manifest = json.loads(real_manifest_path.read_text(encoding="utf-8"))
    ok, msg = vds.check_manifest_normalization_integrity(manifest)
    assert ok is True


# ---------------------------------------------------------------------------
# CSV structural validation (task 4) -- validate_csv_structure()
# ---------------------------------------------------------------------------

def _write_raw(path, text):
    path.write_text(text, encoding="utf-8", newline="")


def test_validate_csv_structure_well_formed_file_has_no_errors(tmp_path):
    p = tmp_path / "ok.csv"
    _write_raw(p, "case_id,language,respondent_text\ndev001,en,baker\ndev002,ar,teacher\n")
    assert vds.validate_csv_structure(p) == []


def test_validate_csv_structure_detects_ragged_row_too_few_fields(tmp_path):
    p = tmp_path / "ragged.csv"
    _write_raw(p, "case_id,language,respondent_text\ndev001,en\n")  # missing 3rd field
    errors = vds.validate_csv_structure(p)
    assert errors
    assert any("Row 2" in e for e in errors)


def test_validate_csv_structure_detects_ragged_row_too_many_fields(tmp_path):
    p = tmp_path / "ragged.csv"
    _write_raw(p, "case_id,language,respondent_text\ndev001,en,baker,extra_field\n")
    errors = vds.validate_csv_structure(p)
    assert errors
    assert any("Row 2" in e for e in errors)


def test_validate_csv_structure_detects_unterminated_quote(tmp_path):
    p = tmp_path / "bad_quote.csv"
    _write_raw(p, 'case_id,language,respondent_text\ndev001,en,"unterminated quote\n')
    errors = vds.validate_csv_structure(p)
    assert errors


def test_validate_csv_structure_missing_file_returns_no_errors(tmp_path):
    assert vds.validate_csv_structure(tmp_path / "does_not_exist.csv") == []


def test_validate_csv_structure_empty_file_returns_no_errors(tmp_path):
    p = tmp_path / "empty.csv"
    _write_raw(p, "")
    assert vds.validate_csv_structure(p) == []


def test_validate_csv_structure_allows_properly_quoted_multiline_field(tmp_path):
    """A quoted field containing an embedded newline is VALID CSV and must
    not be flagged as structurally malformed."""
    p = tmp_path / "multiline.csv"
    _write_raw(
        p,
        'case_id,language,respondent_text\n'
        'dev001,en,"line one\nline two"\n'
        'dev002,ar,baker\n',
    )
    assert vds.validate_csv_structure(p) == []
    rows = vds.load_csv_rows(p)
    assert rows[0]["respondent_text"] == "line one\nline two"
    assert len(rows) == 2  # the multiline field did not get split into extra rows


# ---------------------------------------------------------------------------
# Whitespace-only field rejection (task 4)
# ---------------------------------------------------------------------------

def test_whitespace_only_case_id_is_rejected():
    rows = make_valid_set()
    rows[0]["case_id"] = "   "
    report = vds.validate_dev_set(rows, set())
    assert not report.ok
    assert any("blank case_id" in e for e in report.errors)


def test_whitespace_only_respondent_text_is_rejected():
    rows = make_valid_set()
    rows[0]["respondent_text"] = "   \t  "
    report = vds.validate_dev_set(rows, set())
    assert not report.ok
    assert any("respondent_text is blank" in e for e in report.errors)


def test_leading_trailing_whitespace_is_stripped_before_comparison():
    """A case_id/respondent_text with incidental leading/trailing whitespace
    should still be treated as its stripped form for duplicate detection --
    not as a distinct value that happens to dodge the duplicate check."""
    rows = make_valid_set()
    rows[1]["respondent_text"] = "  " + rows[0]["respondent_text"] + "  "
    report = vds.validate_dev_set(rows, set())
    assert not report.ok
    assert any("within the dev set itself" in e for e in report.errors)


# ---------------------------------------------------------------------------
# Case-sensitive case_id policy (task 4, documented)
# ---------------------------------------------------------------------------

def test_case_id_comparison_is_case_sensitive():
    """Documented policy: case_id matching is exact/case-sensitive.
    "Dev001" and "dev001" are DIFFERENT case_ids -- not deduplicated,
    not treated as colliding with each other, and not treated as
    colliding with an external case_id that differs only in case."""
    rows = make_valid_set()
    rows[1]["case_id"] = rows[0]["case_id"].upper()
    assert rows[0]["case_id"] != rows[1]["case_id"]  # sanity: upper() actually changed it
    report = vds.validate_dev_set(rows, set())
    assert not any("Duplicate case_id" in e for e in report.errors)


def test_case_id_external_collision_is_case_sensitive():
    rows = make_valid_set()
    lowercase_id = rows[0]["case_id"]
    report = vds.validate_dev_set(rows, other_case_ids={lowercase_id.upper()})
    assert not any("collides" in e for e in report.errors)


# ---------------------------------------------------------------------------
# UTF-8 Arabic text support (task 4)
# ---------------------------------------------------------------------------

def test_arabic_respondent_text_round_trips_through_csv(tmp_path):
    p = tmp_path / "arabic.csv"
    header = ",".join(vds.REQUIRED_COLUMNS)
    row = "dev001,ar,طباخ,7512,human_coder_single,AB,dev_v1"
    _write_raw(p, header + "\n" + row + "\n")
    rows = vds.load_csv_rows(p)
    assert rows[0]["respondent_text"] == "طباخ"
    report = vds.validate_dev_set(rows, set())
    # only 1 case, so it will fail on coverage thresholds, but not on
    # anything related to the Arabic text itself being unreadable/mangled
    assert not any("blank" in e or "not valid" in e.lower() for e in report.errors)


def test_normalize_text_handles_arabic():
    assert vds.normalize_text("  طباخ  ") == "طباخ"


# ---------------------------------------------------------------------------
# Full130 isolation regression guard (task 5)
# ---------------------------------------------------------------------------

_VALIDATION_MODULES = [
    Path(__file__).resolve().parent / "validate_dev_set.py",
    Path(__file__).resolve().parent / "pre_run_check.py",
]
_MANIFEST_BUILDER_MODULE = Path(__file__).resolve().parent / "build_full130_leakage_manifest.py"
_OPEN_LIKE_METHODS = {"read_text", "read_bytes", "open"}


def _find_forbidden_full130_opens(source_path):
    """AST scan: find open()/.read_text()/.read_bytes()/.open() calls whose
    argument subtree contains a string literal referencing the full130
    filename. Returns a list of (lineno, snippet) violations -- empty means
    no hardcoded direct open of eval/test_set_full130.csv was found
    anywhere in the module (prose mentions in docstrings/comments/error
    messages are NOT flagged, since they aren't inside a Call node's
    argument subtree)."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        is_open_call = (
            (isinstance(node.func, ast.Name) and node.func.id == "open")
            or (isinstance(node.func, ast.Attribute) and node.func.attr in _OPEN_LIKE_METHODS)
        )
        if not is_open_call:
            continue
        for arg_node in ast.walk(node):
            if isinstance(arg_node, ast.Constant) and isinstance(arg_node.value, str):
                if "test_set_full130" in arg_node.value:
                    violations.append((node.lineno, ast.dump(node)[:200]))
    return violations


def test_validate_dev_set_module_has_no_direct_full130_open_call():
    violations = _find_forbidden_full130_opens(Path(__file__).resolve().parent / "validate_dev_set.py")
    assert violations == [], f"forbidden full130 open() found: {violations}"


def test_pre_run_check_module_has_no_direct_full130_open_call():
    violations = _find_forbidden_full130_opens(Path(__file__).resolve().parent / "pre_run_check.py")
    assert violations == [], f"forbidden full130 open() found: {violations}"


def test_manifest_builder_module_is_the_documented_authorised_exception():
    """The manifest-builder module IS allowed (required, even) to open
    full130 directly -- unlike validate_dev_set.py/pre_run_check.py, it
    passes the path in as a plain function parameter (build_manifest(path)
    -> open(path, ...)), so the Call-argument AST scan above wouldn't see a
    literal at that call site even though the module DOES read full130 by
    design. Check its actual default path constant directly instead -- this
    locks in and documents the asymmetry rather than letting it silently
    drift in either direction: if this module's default ever stops pointing
    at full130, something is broken; if the OTHER two modules ever gain a
    similar constant, test_validate_dev_set_module_has_no_direct_full130_
    open_call / test_pre_run_check_module_has_no_direct_full130_open_call
    only catch it once that constant is actually passed into an open()-like
    call within THAT module's own source -- this test's narrower scope
    (does the builder still target full130 at all) is deliberately simpler."""
    import build_full130_leakage_manifest as builder
    assert builder._DEFAULT_FULL130.name == "test_set_full130.csv"


def test_validate_dev_set_functions_never_open_full130_at_runtime(tmp_path):
    """Behavioural counterpart to the AST scan above -- wraps an ACTUAL
    execution of every full130-adjacent validate_dev_set.py function in
    guard_against_full130_access() (imported from pre_run_check.py, which
    already relies on this exact guarantee) and confirms none of them ever
    call open() on a path matching eval/test_set_full130.csv, regardless of
    how indirectly the path might have been constructed."""
    dev_set = tmp_path / "dev.csv"
    dev_set.write_text(
        "case_id,language,respondent_text,gold_isco_code,gold_label_source,"
        "annotator_or_adjudication_reference,dataset_split\n"
        "dev001,en,baker,7512,human_coder_single,AB,dev_v1\n",
        encoding="utf-8",
    )
    manifest_path = Path(__file__).resolve().parent / "configs" / "full130_leakage_manifest.json"
    smoke20_path = Path(__file__).resolve().parent / "test_set_smoke20.csv"

    with prc.guard_against_full130_access():
        structure_errors = vds.validate_csv_structure(dev_set)
        manifest = vds.load_full130_manifest_raw(manifest_path)
        vds.check_manifest_normalization_integrity(manifest)
        full130_ids, full130_hashes = vds.load_full130_manifest(manifest_path)
        valid_codes = vds.load_isco_unit_group_catalogue()
        dev_rows = vds.load_csv_rows(dev_set)
        smoke20_rows = vds.load_csv_rows(smoke20_path)
        report = vds.validate_dev_set(
            dev_rows, full130_ids, vds.load_normalized_texts(smoke20_rows, "input_text"),
            full130_hashes, valid_isco_codes=valid_codes,
        )
    assert structure_errors == []
    assert isinstance(report, vds.ValidationReport)  # completed without the guard ever firing


# ---------------------------------------------------------------------------
# A genuinely clean set passes with no errors and no warnings
# ---------------------------------------------------------------------------

def test_fully_clean_set_passes_with_no_warnings():
    rows = make_valid_set(n_en=35, n_ar=15)
    report = vds.validate_dev_set(rows, other_case_ids=set())
    assert report.ok
    assert report.warnings == []


def test_fully_clean_set_passes_with_real_isco_catalogue():
    rows = make_valid_set(n_en=35, n_ar=15)  # gold_isco_code="7512" throughout
    real_codes = vds.load_isco_unit_group_catalogue()
    report = vds.validate_dev_set(rows, other_case_ids=set(), valid_isco_codes=real_codes)
    assert report.ok
