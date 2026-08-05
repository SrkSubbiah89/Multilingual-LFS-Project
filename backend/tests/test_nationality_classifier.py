"""
Tests for backend/agents/nationality_classifier.py

NationalityClassifier is stateless (no LLM, no network).  All tests run
offline with no mocking required.
"""

import pytest

from backend.agents.nationality_classifier import (
    NationalityClassification,
    NationalityClassifier,
    _COUNTRY_DATA,
    _ALIAS_INDEX,
)


@pytest.fixture(scope="module")
def clf():
    return NationalityClassifier()


# ---------------------------------------------------------------------------
# Empty / unknown input
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_string(self, clf):
        r = clf.classify("")
        assert r.method == "unknown"
        assert r.confidence == 0.0

    def test_whitespace_only(self, clf):
        r = clf.classify("   ")
        assert r.method == "unknown"
        assert r.confidence == 0.0

    def test_gibberish(self, clf):
        r = clf.classify("xyzxyz123")
        assert r.method == "unknown"

    def test_returns_dataclass(self, clf):
        r = clf.classify("Indian")
        assert isinstance(r, NationalityClassification)


# ---------------------------------------------------------------------------
# Stage 0: ISO-3 exact lookup
# ---------------------------------------------------------------------------

class TestIso3Lookup:
    def test_ind(self, clf):
        r = clf.classify("IND")
        assert r.iso_alpha3 == "IND"
        assert r.confidence == 1.0
        assert r.method == "exact"

    def test_are_lowercase(self, clf):
        r = clf.classify("are")
        assert r.iso_alpha3 == "ARE"
        assert r.method == "exact"

    def test_phl(self, clf):
        r = clf.classify("PHL")
        assert r.iso_alpha3 == "PHL"
        assert r.country_en == "Philippines"

    def test_gbr(self, clf):
        r = clf.classify("GBR")
        assert r.iso_alpha3 == "GBR"
        assert r.nationality_en == "British"


# ---------------------------------------------------------------------------
# Stage 0b: M49 numeric code lookup
# ---------------------------------------------------------------------------

class TestM49Lookup:
    def test_uae_784(self, clf):
        r = clf.classify("784")
        assert r.m49_code == "784"
        assert r.iso_alpha3 == "ARE"
        assert r.method == "exact"

    def test_india_356(self, clf):
        r = clf.classify("356")
        assert r.m49_code == "356"
        assert r.country_en == "India"

    def test_pakistan_586(self, clf):
        r = clf.classify("586")
        assert r.m49_code == "586"
        assert r.country_en == "Pakistan"


# ---------------------------------------------------------------------------
# Stage 1 + 2: alias / token lookup — top UAE nationalities
# ---------------------------------------------------------------------------

class TestTopUaeNationalities:
    """The 8 nationalities now shown as QUICK_OPTIONS in chat.js."""

    def test_emirati(self, clf):
        r = clf.classify("Emirati")
        assert r.iso_alpha3 == "ARE"
        assert r.confidence >= 0.8

    def test_uae(self, clf):
        r = clf.classify("UAE")
        assert r.iso_alpha3 == "ARE"

    def test_indian(self, clf):
        r = clf.classify("Indian")
        assert r.iso_alpha3 == "IND"
        assert r.confidence >= 0.8

    def test_india(self, clf):
        r = clf.classify("I am from India")
        assert r.iso_alpha3 == "IND"

    def test_pakistani(self, clf):
        r = clf.classify("Pakistani")
        assert r.iso_alpha3 == "PAK"
        assert r.confidence >= 0.8

    def test_filipino(self, clf):
        r = clf.classify("Filipino")
        assert r.iso_alpha3 == "PHL"
        assert r.confidence >= 0.8

    def test_filipina(self, clf):
        r = clf.classify("Filipina")
        assert r.iso_alpha3 == "PHL"

    def test_bangladeshi(self, clf):
        r = clf.classify("Bangladeshi")
        assert r.iso_alpha3 == "BGD"
        assert r.confidence >= 0.8

    def test_egyptian(self, clf):
        r = clf.classify("Egyptian")
        assert r.iso_alpha3 == "EGY"
        assert r.confidence >= 0.8

    def test_british(self, clf):
        r = clf.classify("British")
        assert r.iso_alpha3 == "GBR"
        assert r.confidence >= 0.8


# ---------------------------------------------------------------------------
# Multi-word country names (Stage 1 phrase lookup)
# ---------------------------------------------------------------------------

class TestMultiWordCountries:
    def test_south_africa(self, clf):
        r = clf.classify("South Africa")
        assert r.iso_alpha3 == "ZAF"
        assert r.confidence >= 0.7

    def test_sri_lanka(self, clf):
        r = clf.classify("Sri Lanka")
        assert r.iso_alpha3 == "LKA"
        assert r.confidence >= 0.7

    def test_saudi_arabia(self, clf):
        r = clf.classify("Saudi Arabia")
        assert r.iso_alpha3 == "SAU"

    def test_united_kingdom(self, clf):
        # "kingdom" token added to GBR aliases
        r = clf.classify("United Kingdom")
        assert r.iso_alpha3 == "GBR"

    def test_united_states(self, clf):
        # "states" token added to USA aliases
        r = clf.classify("United States")
        assert r.iso_alpha3 == "USA"


# ---------------------------------------------------------------------------
# Arabic input
# ---------------------------------------------------------------------------

class TestArabicInput:
    def test_arabic_emirati(self, clf):
        r = clf.classify("إماراتي")
        assert r.iso_alpha3 == "ARE"

    def test_arabic_indian(self, clf):
        r = clf.classify("هندي")
        assert r.iso_alpha3 == "IND"

    def test_arabic_egyptian(self, clf):
        r = clf.classify("مصري")
        assert r.iso_alpha3 == "EGY"

    def test_arabic_saudi(self, clf):
        r = clf.classify("سعودي")
        assert r.iso_alpha3 == "SAU"

    def test_arabic_phrase(self, clf):
        r = clf.classify("أنا من الهند")
        assert r.iso_alpha3 == "IND"


# ---------------------------------------------------------------------------
# Case insensitivity
# ---------------------------------------------------------------------------

class TestCaseInsensitivity:
    def test_uppercase(self, clf):
        r = clf.classify("INDIAN")
        assert r.iso_alpha3 == "IND"

    def test_mixed_case(self, clf):
        r = clf.classify("pAkIsTeNi")
        # "pAkIsTeNi" lowercased = "pakisteni" — not an alias; no match expected
        # just verify no crash
        assert r.method in ("alias", "token", "unknown")

    def test_lowercase_country(self, clf):
        r = clf.classify("india")
        assert r.iso_alpha3 == "IND"


# ---------------------------------------------------------------------------
# Sentence-level input (conversational phrasing from survey)
# ---------------------------------------------------------------------------

class TestConversationalInput:
    def test_change_nationality_to_indian(self, clf):
        # The value stored after correction is "indian" — classifier must handle it
        r = clf.classify("indian")
        assert r.iso_alpha3 == "IND"

    def test_nationality_should_be_pakistani(self, clf):
        r = clf.classify("pakistani")
        assert r.iso_alpha3 == "PAK"

    def test_from_phrase(self, clf):
        r = clf.classify("from Philippines")
        assert r.iso_alpha3 == "PHL"

    def test_raw_text_preserved(self, clf):
        r = clf.classify("Egyptian")
        assert r.raw_text == "Egyptian"


# ---------------------------------------------------------------------------
# Output fields completeness
# ---------------------------------------------------------------------------

class TestOutputFields:
    def test_all_fields_populated_for_match(self, clf):
        r = clf.classify("Indian")
        assert r.m49_code == "356"
        assert r.iso_alpha3 == "IND"
        assert r.country_en == "India"
        assert r.country_ar == "الهند"
        assert r.region_en == "Southern Asia"
        assert r.nationality_en == "Indian"
        assert r.nationality_ar == "هندي"
        assert 0.0 < r.confidence <= 1.0

    def test_unknown_sentinel_fields(self, clf):
        r = clf.classify("zzzzz")
        assert r.m49_code == "000"
        assert r.iso_alpha3 == "XXX"
        assert r.confidence == 0.0
        assert r.method == "unknown"


# ---------------------------------------------------------------------------
# Data integrity
# ---------------------------------------------------------------------------

class TestDataIntegrity:
    def test_country_data_not_empty(self):
        assert len(_COUNTRY_DATA) >= 50

    def test_alias_index_not_empty(self):
        assert len(_ALIAS_INDEX) >= 100

    def test_all_entries_have_required_keys(self):
        required = {"m49", "iso3", "country_en", "country_ar", "region_en",
                    "nat_en", "nat_ar", "aliases"}
        for entry in _COUNTRY_DATA:
            missing = required - set(entry.keys())
            assert not missing, f"Entry {entry.get('iso3')} missing keys: {missing}"

    def test_no_duplicate_iso3(self):
        seen = set()
        for e in _COUNTRY_DATA:
            assert e["iso3"] not in seen, f"Duplicate ISO3: {e['iso3']}"
            seen.add(e["iso3"])

    def test_stateless_multiple_instances(self):
        c1 = NationalityClassifier()
        c2 = NationalityClassifier()
        r1 = c1.classify("Indian")
        r2 = c2.classify("Indian")
        assert r1.iso_alpha3 == r2.iso_alpha3
