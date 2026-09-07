"""
Tests for backend/rag/official_source_enrichment.py -- real official
definitions/examples for ISIC Rev.4 / ISCED-F 2013, parsed from the
primary-source UN/UNESCO publications by
eval/parse_official_isic_iscedf_definitions.py. Mirrors ISCO-08's own
ENRICHED_PROFILE (backend/rag/official_isco08_catalogue.py) for the two
standards that didn't have an equivalent official-source enrichment before
this session.
"""

import pytest

from backend.agents.isced_classifier import _ISCED_FIELDS
from backend.agents.isic_classifier import _ISIC_DATA
from backend.rag.official_source_enrichment import (
    _ISCEDF_DEFINITIONS_PATH,
    _ISIC_DEFINITIONS_PATH,
    NON_STANDARD_ISCEDF_CODES,
    NON_STANDARD_ISIC_CODES,
    build_enriched_text,
    load_iscedf_definitions,
    load_isic_definitions,
)

# eval/local_catalogues/ is git-ignored (same policy as ISCO-08's own
# official workbook -- see .gitignore and this module's own docstring), so
# these two JSON files only exist on a machine that has run
# `python -m eval.parse_official_isic_iscedf_definitions` locally. Real gap
# found by code review (2026-08-27): tests reading them had no skip guard,
# so a fresh clone/CI box got confusing FileNotFoundErrors instead of a
# clean skip. Only classes that read the REAL files need this guard --
# TestMissingFileErrorMessage and TestBuildEnrichedText use tmp_path
# fixtures / hand-built dicts and never touch the real files.
_HAS_REAL_CATALOGUE_FILES = _ISIC_DEFINITIONS_PATH.exists() and _ISCEDF_DEFINITIONS_PATH.exists()
_SKIP_REASON = (
    "eval/local_catalogues/*_definitions.json are git-ignored and not present on "
    "this machine -- run `python -m eval.parse_official_isic_iscedf_definitions` "
    "after downloading the source PDFs (see that script's own docstring) to "
    "regenerate them locally before running these tests."
)
requires_real_catalogue_files = pytest.mark.skipif(not _HAS_REAL_CATALOGUE_FILES, reason=_SKIP_REASON)


class TestMissingFileErrorMessage:
    def test_missing_definitions_file_raises_actionable_error(self, tmp_path, monkeypatch):
        """These JSON files are git-ignored (same policy as ISCO-08's own
        official workbook) -- a fresh clone without them must get a clear,
        actionable error, not a bare FileNotFoundError."""
        import backend.rag.official_source_enrichment as mod
        missing = tmp_path / "does_not_exist.json"
        monkeypatch.setattr(mod, "_ISIC_DEFINITIONS_PATH", missing)
        mod.load_isic_definitions.cache_clear()
        try:
            with pytest.raises(FileNotFoundError, match="parse_official_isic_iscedf_definitions"):
                mod.load_isic_definitions()
        finally:
            mod.load_isic_definitions.cache_clear()


@requires_real_catalogue_files
class TestLoaders:
    def test_isic_definitions_load_and_cover_all_419_official_classes(self):
        defs = load_isic_definitions()
        assert len(defs) == 419

    def test_iscedf_definitions_load_and_have_real_entries(self):
        defs = load_iscedf_definitions()
        assert len(defs) > 60

    def test_loaders_are_cached_singletons(self):
        assert load_isic_definitions() is load_isic_definitions()
        assert load_iscedf_definitions() is load_iscedf_definitions()


@requires_real_catalogue_files
class TestNonStandardCodesDisclosed:
    """These are real, verified findings (codes in the existing catalogues
    that do not match any code in the official structure document) -- not
    fixed by this module, but never silently unaccounted for either."""

    def test_isic_non_standard_codes_are_real_isic_data_entries(self):
        isic_codes = {e["class_code"] for e in _ISIC_DATA}
        assert NON_STANDARD_ISIC_CODES.issubset(isic_codes)
        assert len(NON_STANDARD_ISIC_CODES) == 13

    def test_iscedf_non_standard_codes_are_real_isced_fields_entries(self):
        iscedf_codes = {e["detailed_code"] for e in _ISCED_FIELDS}
        assert NON_STANDARD_ISCEDF_CODES.issubset(iscedf_codes)
        assert len(NON_STANDARD_ISCEDF_CODES) == 2

    def test_non_standard_isic_codes_genuinely_absent_from_official_definitions(self):
        defs = load_isic_definitions()
        for code in NON_STANDARD_ISIC_CODES:
            assert code not in defs

    def test_non_standard_iscedf_codes_genuinely_absent_from_official_definitions(self):
        defs = load_iscedf_definitions()
        for code in NON_STANDARD_ISCEDF_CODES:
            assert code not in defs

    def test_every_matched_isic_code_has_real_non_empty_content(self):
        """Every _ISIC_DATA code NOT in the disclosed non-standard set must
        have a genuine, non-empty official definition or example list --
        guards against a future silent regression re-introducing an empty
        match (the exact bug class this module's docstring documents being
        found and fixed during parsing)."""
        defs = load_isic_definitions()
        isic_codes = {e["class_code"] for e in _ISIC_DATA}
        for code in isic_codes - NON_STANDARD_ISIC_CODES:
            entry = defs.get(code)
            assert entry is not None, f"{code} should match the official document"
            assert entry["definition"] or entry["examples"], f"{code} matched but has no real content"

    def test_every_matched_iscedf_code_has_real_non_empty_content(self):
        defs = load_iscedf_definitions()
        iscedf_codes = {e["detailed_code"] for e in _ISCED_FIELDS}
        for code in iscedf_codes - NON_STANDARD_ISCEDF_CODES:
            entry = defs.get(code)
            assert entry is not None, f"{code} should match the official document"
            assert entry["definition"] or entry["examples"], f"{code} matched but has no real content"


class TestBuildEnrichedText:
    def test_matched_code_uses_real_definition_and_examples(self):
        defs = {"6201": {"title": "Computer programming activities",
                          "definition": "This class includes the writing of software.",
                          "examples": ["designing systems software", "customizing applications"]}}
        text = build_enriched_text("6201", "Computer programming activities", "software code app", defs)
        assert text.startswith("6201 Computer programming activities.")
        assert "This class includes the writing of software." in text
        assert "Examples: designing systems software; customizing applications." in text
        # Enriched text must not just be the old keyword text with extra
        # words appended -- it should be structurally different (real
        # prose), not a superset check that would pass trivially.
        assert text != f"6201 Computer programming activities software code app"

    def test_unmatched_code_falls_back_to_title_and_keywords_unchanged(self):
        text = build_enriched_text("7311", "Advertising agencies", "advertising marketing agency", {})
        assert text == "7311 Advertising agencies advertising marketing agency"

    def test_matched_but_empty_entry_falls_back_same_as_unmatched(self):
        defs = {"9999": {"title": "X", "definition": "", "examples": []}}
        text = build_enriched_text("9999", "X", "keyword", defs)
        assert text == "9999 X keyword"

    def test_definition_only_no_examples(self):
        defs = {"8610": {"title": "Hospital activities", "definition": "Real definition text.", "examples": []}}
        text = build_enriched_text("8610", "Hospital activities", "hospital medical", defs)
        assert "Real definition text." in text
        assert "Examples:" not in text

    def test_examples_only_no_definition(self):
        defs = {"8610": {"title": "Hospital activities", "definition": "", "examples": ["treating patients"]}}
        text = build_enriched_text("8610", "Hospital activities", "hospital medical", defs)
        assert "Examples: treating patients." in text


@requires_real_catalogue_files
class TestRealIsicSampleFromOfficialDocument:
    """Sanity checks against real, hand-verified parsed content (not
    fixtures) -- confirms the checked-in JSON genuinely reflects the
    official ISIC Rev.4 publication for a few well-known classes."""

    def test_computer_programming_6201(self):
        defs = load_isic_definitions()
        entry = defs["6201"]
        assert "software" in entry["definition"].lower()

    def test_hospital_activities_8610(self):
        defs = load_isic_definitions()
        entry = defs["8610"]
        assert entry["examples"]

    def test_construction_of_buildings_4100(self):
        defs = load_isic_definitions()
        entry = defs["4100"]
        assert "residential" in entry["definition"].lower()


@requires_real_catalogue_files
class TestRealIscedfSampleFromOfficialDocument:
    def test_software_development_0613(self):
        defs = load_iscedf_definitions()
        entry = defs["0613"]
        assert "computer" in entry["definition"].lower()
        assert "Computer programming" in entry["examples"]

    def test_basic_programmes_0011(self):
        defs = load_iscedf_definitions()
        entry = defs["0011"]
        assert entry["definition"]
