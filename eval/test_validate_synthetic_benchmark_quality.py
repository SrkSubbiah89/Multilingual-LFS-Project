"""
Tests for eval/validate_synthetic_benchmark_quality.py -- fully offline,
no live LLM/network calls, no dependency on the real generated dataset
being present on disk.
"""

from eval.validate_synthetic_benchmark_quality import (
    check_row,
    check_csv,
    summarise_by_language,
)


class TestScriptContamination:
    def test_flags_cjk_mixed_into_arabic_text(self):
        """The exact defect class the discarded qwen2.5:3b attempt
        produced (see the generator's own docstring): CJK characters
        mixed mid-sentence into otherwise-Arabic text."""
        result = check_row(
            case_id="x", input_text="اسمي هو 程序员", input_language="ar",
            standard="isic", gold_code="6201", gold_title="Computer programming activities",
        )
        assert result.cjk_contamination is True
        assert "cjk_contamination" in result.failure_reasons
        assert result.passed is False

    def test_clean_arabic_text_has_no_cjk_contamination(self):
        result = check_row(
            case_id="x", input_text="أنا أعمل مبرمجاً في شركة برمجيات",
            input_language="ar", standard="isic", gold_code="6201", gold_title="Computer programming activities",
        )
        assert result.cjk_contamination is False


class TestScriptMatch:
    def test_hindi_written_in_devanagari_matches(self):
        result = check_row(
            case_id="x", input_text="मैं एक सॉफ्टवेयर कंपनी में प्रोग्रामर हूं",
            input_language="hi", standard="isic", gold_code="6201", gold_title="Computer programming activities",
        )
        assert result.script_match is True

    def test_hindi_written_in_latin_script_flagged_as_mismatch(self):
        """Real case caught in the actual dataset (SYN-ISIC-5229-hi-1):
        genuine Hindi content, but transliterated into Latin script
        ("Main gaadiyon aur..."). Not necessarily a fabrication defect --
        real Hindi speakers do write this way -- but it does not match
        what the "hi" language code is expected to look like for this
        benchmark's embedding-based downstream evaluation, so it's
        correctly flagged rather than silently passed."""
        result = check_row(
            case_id="x", input_text="Main gaadiyon aur jahazon se samaan bhejta hoon.",
            input_language="hi", standard="isic", gold_code="4922", gold_title="Freight transport",
        )
        assert result.script_match is False
        assert "script_mismatch" in result.failure_reasons

    def test_english_and_tagalog_use_latin_script_family(self):
        for lang in ("en", "tl"):
            result = check_row(
                case_id="x", input_text="I work as a software developer at a tech company",
                input_language=lang, standard="isic", gold_code="6201", gold_title="Computer programming activities",
            )
            assert result.script_match is True

    def test_urdu_uses_arabic_script_family(self):
        result = check_row(
            case_id="x", input_text="میں ایک سافٹ ویئر کمپنی میں پروگرامر ہوں",
            input_language="ur", standard="isic", gold_code="6201", gold_title="Computer programming activities",
        )
        assert result.script_match is True


class TestLengthBounds:
    def test_very_short_text_fails(self):
        result = check_row(
            case_id="x", input_text="ok", input_language="en",
            standard="isic", gold_code="6201", gold_title="Computer programming activities",
        )
        assert result.length_ok is False
        assert "length_out_of_bounds" in result.failure_reasons

    def test_reasonable_length_text_passes(self):
        result = check_row(
            case_id="x", input_text="I write and maintain software applications for a small tech company.",
            input_language="en", standard="isic", gold_code="6201", gold_title="Computer programming activities",
        )
        assert result.length_ok is True


class TestTitleLeak:
    def test_verbatim_english_title_flagged(self):
        """Real case from the actual dataset: 'I studied chemistry,
        learning about stuff and how they mix.' against gold title
        'Chemistry' -- the paraphrase instruction was not followed for
        this row."""
        result = check_row(
            case_id="x", input_text="I studied chemistry, learning about stuff and how they mix.",
            input_language="en", standard="iscedf", gold_code="0531", gold_title="Chemistry",
        )
        assert result.title_leak is True

    def test_paraphrased_text_not_flagged(self):
        result = check_row(
            case_id="x", input_text="I learned about atoms, molecules, and how substances react together.",
            input_language="en", standard="iscedf", gold_code="0531", gold_title="Chemistry",
        )
        assert result.title_leak is False

    def test_short_generic_title_does_not_trigger_false_positive(self):
        """A title like 'Other' would trivially substring-match many
        unrelated casual sentences -- the minimum-length guard exists
        specifically to avoid that class of false positive."""
        result = check_row(
            case_id="x", input_text="I help other people move their furniture around.",
            input_language="en", standard="isic", gold_code="9999", gold_title="Other",
        )
        assert result.title_leak is False


class TestRefusalPattern:
    """Real, live-caught defect (2026-08-27): see
    test_generate_synthetic_isic_iscedf_benchmark.py's TestLooksLikeRefusal
    for the full incident writeup. This is the same check, applied as a
    post-hoc, re-runnable verification independent of the generator's own
    (now also fixed) in-line retry logic."""

    def test_flags_the_actual_real_refusal_text_verbatim(self):
        result = check_row(
            case_id="SYN-ISIC-9609-hi-1", input_text="I’m sorry, but I can’t help with that.",
            input_language="hi", standard="isic", gold_code="9609",
            gold_title="Other personal service activities n.e.c.",
        )
        assert result.refusal_pattern is True
        assert result.passed is False

    def test_this_check_catches_what_script_match_alone_would_miss(self):
        """The real reason this check was added as a fourth, separate
        check rather than folded into script_mismatch: the Tagalog
        refusal row (SYN-ISIC-9609-tl-1) is English text, but English and
        Tagalog share the Latin script family, so script_mismatch alone
        does NOT catch it -- only the refusal-pattern check does."""
        result = check_row(
            case_id="SYN-ISIC-9609-tl-1", input_text="I'm sorry, but I can't help with that.",
            input_language="tl", standard="isic", gold_code="9609",
            gold_title="Other personal service activities n.e.c.",
        )
        assert result.script_match is True  # Latin-on-Latin -- would silently pass this check alone
        assert result.refusal_pattern is True
        assert result.passed is False

    def test_ordinary_text_not_flagged(self):
        result = check_row(
            case_id="x", input_text="I run a spa offering massages, saunas, and wellness treatments.",
            input_language="en", standard="isic", gold_code="9609",
            gold_title="Other personal service activities n.e.c.",
        )
        assert result.refusal_pattern is False


class TestCheckCsvAndSummary:
    def test_check_csv_and_summarise_end_to_end(self):
        rows = [
            {"case_id": "a", "input_text": "I write software at a tech company for a living.",
             "input_language": "en", "standard": "isic", "gold_code": "6201",
             "gold_title": "Computer programming activities"},
            {"case_id": "b", "input_text": "I’m sorry, but I can’t help with that.",
             "input_language": "hi", "standard": "isic", "gold_code": "9609",
             "gold_title": "Other personal service activities n.e.c."},
        ]
        results = check_csv(rows)
        assert len(results) == 2
        assert results[0].passed is True
        assert results[1].passed is False

        summaries = summarise_by_language(results)
        by_lang = {s.language: s for s in summaries}
        assert by_lang["en"].n == 1 and by_lang["en"].n_passed == 1
        assert by_lang["hi"].n == 1 and by_lang["hi"].n_passed == 0
        assert by_lang["hi"].n_refusal_pattern == 1
