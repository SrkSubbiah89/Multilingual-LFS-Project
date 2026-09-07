"""
Tests for eval/generate_synthetic_isic_iscedf_benchmark.py -- fully
offline, no live LLM/network calls. Only the deterministic, non-LLM parts
are exercised: prompt construction, definition truncation, entry loading
(which reads the same real, checked-locally definitions files
backend/rag/official_source_enrichment.py uses), and CLI validation.
"""

import pytest

from eval.generate_synthetic_isic_iscedf_benchmark import (
    _LANGUAGES,
    _MAX_DEFINITION_CHARS,
    _build_prompt,
    _ISCEDF_PROMPT_TEMPLATE,
    _ISIC_PROMPT_TEMPLATE,
    _load_existing_case_ids,
    _load_matched_isic_entries,
    _load_matched_iscedf_entries,
    _looks_like_refusal,
)


class TestLanguageCodes:
    def test_matches_language_processor_supported_languages(self):
        """Must stay in sync with backend/agents/language_processor.py's
        SUPPORTED_LANGUAGES (minus "other", which isn't a real generation
        target) -- the whole point of this benchmark is real multilingual
        coverage matching the project's actual supported languages."""
        from backend.agents.language_processor import SUPPORTED_LANGUAGES
        assert set(_LANGUAGES) == SUPPORTED_LANGUAGES - {"other"}

    def test_six_language_codes(self):
        assert len(_LANGUAGES) == 6
        assert set(_LANGUAGES) == {"en", "ar", "ar-gulf", "ur", "hi", "tl"}


class TestBuildPrompt:
    def test_uses_definition_when_present(self):
        entry = {"title": "Computer programming activities",
                  "definition": "This class includes writing software.",
                  "examples": ["designing systems"]}
        prompt = _build_prompt(_ISIC_PROMPT_TEMPLATE, entry, "English")
        assert "Computer programming activities" in prompt
        assert "This class includes writing software." in prompt
        assert "English" in prompt

    def test_falls_back_to_examples_when_no_definition(self):
        entry = {"title": "X", "definition": "", "examples": ["example one", "example two"]}
        prompt = _build_prompt(_ISIC_PROMPT_TEMPLATE, entry, "Hindi")
        assert "example one" in prompt

    def test_truncates_long_definition_to_bound_token_usage(self):
        entry = {"title": "X", "definition": "word " * 200, "examples": []}
        prompt = _build_prompt(_ISIC_PROMPT_TEMPLATE, entry, "English")
        # The definition portion embedded in the prompt should be bounded,
        # not the full 1000-char input.
        assert len(entry["definition"]) > _MAX_DEFINITION_CHARS
        assert "..." in prompt

    def test_never_cuts_mid_word(self):
        entry = {"title": "X", "definition": "alpha beta gamma delta " * 20, "examples": []}
        prompt = _build_prompt(_ISIC_PROMPT_TEMPLATE, entry, "English")
        # The truncated definition (before "...") should end on a word
        # boundary -- rsplit(" ", 1) guarantees this.
        before_ellipsis = prompt.split("...")[0]
        assert not before_ellipsis.endswith(("alph", "bet", "gamm", "delt"))

    def test_iscedf_template_asks_about_field_of_study(self):
        entry = {"title": "Software development", "definition": "Study of software.", "examples": []}
        prompt = _build_prompt(_ISCEDF_PROMPT_TEMPLATE, entry, "Tagalog")
        assert "study" in prompt.lower()


class TestMatchedEntryLoading:
    """These read the same real, locally-parsed definitions files
    backend/rag/official_source_enrichment.py uses -- if those files are
    present (as they are in this dev environment), these must exclude the
    disclosed non-standard codes exactly."""

    def test_isic_entries_exclude_non_standard_codes(self):
        from backend.rag.official_source_enrichment import NON_STANDARD_ISIC_CODES
        entries = _load_matched_isic_entries()
        codes = {e["code"] for e in entries}
        assert codes.isdisjoint(NON_STANDARD_ISIC_CODES)
        assert len(entries) == 121

    def test_iscedf_entries_exclude_non_standard_codes(self):
        from backend.rag.official_source_enrichment import NON_STANDARD_ISCEDF_CODES
        entries = _load_matched_iscedf_entries()
        codes = {e["code"] for e in entries}
        assert codes.isdisjoint(NON_STANDARD_ISCEDF_CODES)
        assert len(entries) == 61

    def test_every_entry_has_real_grounding_content(self):
        for e in _load_matched_isic_entries():
            assert e["definition"] or e["examples"], f"{e['code']} has no grounding content"

    def test_no_duplicate_codes(self):
        entries = _load_matched_isic_entries()
        codes = [e["code"] for e in entries]
        assert len(codes) == len(set(codes))


class TestLoadExistingCaseIds:
    def test_none_path_returns_empty_set(self):
        assert _load_existing_case_ids(None) == set()

    def test_reads_case_ids_from_a_real_csv(self, tmp_path):
        import csv
        p = tmp_path / "prior.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["case_id", "input_text"])
            w.writeheader()
            w.writerow({"case_id": "SYN-ISIC-6201-en-1", "input_text": "x"})
            w.writerow({"case_id": "SYN-ISIC-6201-ar-1", "input_text": "y"})
        result = _load_existing_case_ids(str(p))
        assert result == {"SYN-ISIC-6201-en-1", "SYN-ISIC-6201-ar-1"}


class TestLooksLikeRefusal:
    """Real, live-caught defect (2026-08-27): SYN-ISIC-9609-hi-1 and
    SYN-ISIC-9609-tl-1 (gold code 9609, whose official examples include
    "escort services, dating services") both contained a literal LLM
    refusal, silently accepted as valid respondent text by the prior
    `len(text) >= 3` check. Found by a first run of
    eval/validate_synthetic_benchmark_quality.py against the real 449-row
    dataset -- see CLAUDE.md's "Knowledge base construction" log."""

    def test_detects_the_actual_real_refusal_text_found_in_the_dataset(self):
        """The real text from SYN-ISIC-9609-hi-1/tl-1, verbatim -- including
        its curly/typographic apostrophe (U+2019), not a straight ASCII
        one. A straight-quote-only version of this regex was tested and
        confirmed NOT to match this exact string before being corrected --
        this test pins that fix."""
        assert _looks_like_refusal("I’m sorry, but I can’t help with that.")

    def test_detects_straight_apostrophe_variant_too(self):
        assert _looks_like_refusal("I'm sorry, but I can't help with that.")

    def test_detects_common_refusal_phrasings(self):
        for text in [
            "As an AI, I cannot generate that content.",
            "I am unable to help with this request.",
            "I cannot assist with that topic.",
        ]:
            assert _looks_like_refusal(text)

    def test_does_not_flag_ordinary_respondent_text(self):
        for text in [
            "I run a spa offering massages, saunas, and wellness treatments.",
            "I studied chemistry, learning about stuff and how they mix.",
            "Main gaadiyon aur jahazon se samaan bhejta hoon.",
        ]:
            assert not _looks_like_refusal(text)

    def test_does_not_flag_text_that_merely_contains_the_word_sorry(self):
        """A real respondent could plausibly use "sorry" mid-sentence
        without it being a refusal -- this regex is deliberately anchored
        to whole refusal PHRASES ("I'm sorry, ..."), not the bare word."""
        assert not _looks_like_refusal("Sorry I'm late, I work as a night-shift security guard.")
