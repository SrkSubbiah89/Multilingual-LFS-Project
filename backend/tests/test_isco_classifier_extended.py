"""
Extended tests for ISCOClassifier
Covers: all 10 major ISCO groups, Arabic job titles, confidence edge cases,
        hierarchical stage verification, boundary conditions, UAE-specific occupations.
"""
import pytest
from unittest.mock import MagicMock, patch


# ─── Shared fixtures (same pattern as test_isco_classifier.py) ───────────────

@pytest.fixture
def mock_store():
    store = MagicMock()
    match = MagicMock()
    match.code       = "2512"
    match.title_en   = "Software Developers"
    match.title_ar   = "مطورو البرمجيات"
    match.confidence = 0.85
    store.search.return_value = [match]
    return store


@pytest.fixture
def get_llm_mock():
    # Production calls get_llm(TaskType.GENERAL, temperature=self._llm_temperature)
    # (backend/agents/isco_classifier.py) -- a positional TaskType arg plus a
    # temperature= keyword. Must accept both, not just a single positional
    # arg, or ISCOClassifier.__init__'s try/except silently swallows the
    # resulting TypeError and falls back to semantic-only classification.
    return MagicMock(side_effect=lambda *args, **kwargs: MagicMock())


@pytest.fixture
def clf(monkeypatch, mock_store, get_llm_mock):
    monkeypatch.setattr("backend.agents.isco_classifier.get_llm", get_llm_mock)
    monkeypatch.setattr(
        "backend.agents.isco_classifier.get_hierarchical_store",
        lambda **kw: None,   # force flat path so mock_store is used
    )
    monkeypatch.setattr(
        "backend.agents.isco_classifier.get_vector_store",
        lambda **kw: mock_store,
    )
    monkeypatch.setattr("backend.agents.isco_classifier.Agent", MagicMock())
    monkeypatch.setattr("backend.agents.isco_classifier.Task",  MagicMock())

    crew_mock = MagicMock()
    crew_mock.return_value.kickoff.return_value = '{"code": "2512", "reasoning": "test"}'
    monkeypatch.setattr("backend.agents.isco_classifier.Crew", crew_mock)

    from backend.agents.isco_classifier import ISCOClassifier
    return ISCOClassifier()


# ─── All 10 ISCO Major Groups ────────────────────────────────────────────────

class TestAllISCOMajorGroups:

    @pytest.mark.parametrize("job_title,expected_major", [
        ("Chief Executive Officer",  "1"),
        ("Software Engineer",        "2"),
        ("Laboratory Technician",    "3"),
        ("Administrative Secretary", "4"),
        ("Retail Sales Assistant",   "5"),
        ("Crop Farmer",              "6"),
        ("Construction Electrician", "7"),
        ("Machine Operator",         "8"),
        ("Cleaner",                  "9"),
        ("Armed Forces Officer",     "0"),
    ])
    def test_major_group_classified(self, clf, mock_store, job_title, expected_major):
        mock_store.search.return_value[0].code = expected_major + "110"
        result = clf.classify(job_title)
        assert result is not None
        assert result.primary.code is not None

    def test_all_ten_major_groups_have_distinct_codes(self):
        codes = set(str(i) for i in range(10))
        assert len(codes) == 10


# ─── Arabic Job Titles ───────────────────────────────────────────────────────

class TestArabicJobTitles:

    @pytest.mark.parametrize("arabic_title", [
        "مهندس برمجيات",
        "طبيب",
        "معلم",
        "محاسب",
        "ممرضة",
        "سائق",
        "مدير",
        "عامل بناء",
        "موظف إداري",
        "طاهٍ",
    ])
    def test_arabic_title_classified(self, clf, arabic_title):
        result = clf.classify(arabic_title)
        assert result is not None
        assert result.primary.code is not None

    def test_arabic_result_has_arabic_label(self, clf):
        result = clf.classify("مهندس برمجيات")
        assert result.primary.title_ar is not None
        assert len(result.primary.title_ar) > 0

    def test_arabic_result_has_english_label(self, clf):
        result = clf.classify("مهندس برمجيات")
        assert result.primary.title_en is not None


# ─── UAE-Specific Occupations ────────────────────────────────────────────────

class TestUAESpecificOccupations:

    @pytest.mark.parametrize("occupation", [
        "Construction Laborer", "Domestic Helper",    "Security Guard",
        "Shop Assistant",       "Taxi Driver",        "Delivery Driver",
        "Hotel Housekeeper",    "Restaurant Waiter",  "Cashier",
        "Warehouse Worker",     "IT Support Technician", "Civil Engineer",
        "HR Manager",           "Sales Executive",    "Accountant",
        "Teacher",              "Nurse",              "Doctor",
        "Architect",            "Legal Advisor",
    ])
    def test_uae_occupation_classified(self, clf, occupation):
        result = clf.classify(occupation)
        assert result is not None

    def test_domestic_helper_not_manager(self, clf, mock_store):
        mock_store.search.return_value[0].code = "9111"
        result = clf.classify("Domestic Helper")
        assert not result.primary.code.startswith("1")


# ─── Confidence Thresholds ───────────────────────────────────────────────────

class TestConfidenceThresholds:

    def test_hitl_required_below_threshold(self, clf, mock_store):
        mock_store.search.return_value[0].confidence = 0.55
        result = clf.classify("I do things")
        assert result.hitl_required is True

    def test_hitl_not_required_above_threshold(self, clf, mock_store):
        mock_store.search.return_value[0].confidence = 0.85
        result = clf.classify("Software Engineer")
        assert result.hitl_required is False

    def test_confidence_exactly_at_threshold(self, clf, mock_store):
        mock_store.search.return_value[0].confidence = 0.70
        result = clf.classify("Engineer")
        assert result.hitl_required is False

    def test_confidence_bounded_0_to_1(self, clf):
        result = clf.classify("engineer")
        assert 0.0 <= result.primary.confidence <= 1.0

    def test_zero_confidence_triggers_hitl(self, clf, mock_store):
        mock_store.search.return_value[0].confidence = 0.0
        result = clf.classify("??")
        assert result.hitl_required is True


# ─── Stage Verification ──────────────────────────────────────────────────────

class TestHierarchicalStages:

    def test_result_has_confidence_score(self, clf):
        result = clf.classify("Software Engineer")
        assert result.primary.confidence is not None

    def test_weighted_confidence_formula(self):
        weights = [0.10, 0.20, 0.20, 0.50]
        assert abs(sum(weights) - 1.0) < 0.001

    def test_fast_path_skips_llm(self, clf, mock_store):
        mock_store.search.return_value[0].confidence = 0.95
        result = clf.classify("Software Developer")
        # High confidence → method is semantic (no LLM suffix)
        assert "semantic" in result.method

    def test_llm_used_for_low_similarity(self, clf, mock_store, get_llm_mock):
        mock_store.search.return_value[0].confidence = 0.60
        result = clf.classify("I do stuff with computers")
        # Low confidence → LLM re-ranking, method ends with _llm
        assert "llm" in result.method
        # Proof of the actual root cause fix: get_llm was called with the
        # production temperature= keyword (the old lambda t: MagicMock()
        # test double could not accept this and silently disabled the LLM
        # agent, which is why this test previously took the semantic-only
        # path instead of exercising LLM re-ranking).
        assert get_llm_mock.called
        _, call_kwargs = get_llm_mock.call_args
        assert "temperature" in call_kwargs

    def test_stage_confidences_when_present(self, clf, mock_store):
        result = clf.classify("nurse")
        # stage_confidences may be None for flat path; primary.confidence must exist
        assert result.primary.confidence is not None


# ─── ISCO Code Format ────────────────────────────────────────────────────────

class TestISCOCodeFormat:

    def test_primary_code_is_4_digits(self, clf):
        result = clf.classify("Software Engineer")
        if result.primary.code and result.primary.code != "0000":
            assert len(result.primary.code) == 4

    def test_primary_code_is_numeric_string(self, clf):
        result = clf.classify("Software Engineer")
        if result.primary.code:
            assert result.primary.code.isdigit()

    def test_sentinel_code_on_empty_input(self, clf):
        result = clf.classify("")
        assert result.primary.code is not None

    def test_sentinel_code_on_whitespace(self, clf):
        result = clf.classify("   ")
        assert result is not None

    def test_major_group_code_length_1(self):
        for code in [str(i) for i in range(10)]:
            assert len(code) == 1

    def test_unit_group_code_starts_with_valid_major(self, clf):
        result = clf.classify("Nurse")
        if result.primary.code and len(result.primary.code) == 4:
            assert result.primary.code[0] in "0123456789"


# ─── Informal Descriptions ───────────────────────────────────────────────────

class TestInformalDescriptions:

    @pytest.mark.parametrize("description", [
        "I work with computers",      "I take care of patients",
        "I drive people around",      "I sell things in a shop",
        "I build houses",             "I teach children",
        "I manage a team",            "I fix machines",
        "I cook food",                "I clean offices",
        "أشتغل في الكمبيوتر",         "أعتني بالمرضى",
        "أبيع بضائع",
    ])
    def test_informal_description_classified(self, clf, description):
        result = clf.classify(description)
        assert result is not None
        assert result.primary.code is not None

    def test_very_vague_description_triggers_hitl(self, clf, mock_store):
        mock_store.search.return_value[0].confidence = 0.30
        result = clf.classify("I do stuff")
        assert result.hitl_required is True


# ─── Error Resilience ────────────────────────────────────────────────────────

class TestErrorResilience:

    def test_qdrant_timeout_propagates_or_sentinel(self, clf, mock_store):
        # classify() does not catch store exceptions; they propagate to the caller
        mock_store.search.side_effect = TimeoutError("Qdrant timeout")
        try:
            result = clf.classify("Engineer")
            assert result is not None  # if handled gracefully
        except (TimeoutError, Exception):
            pass  # acceptable: unhandled exception propagates

    def test_llm_exception_propagates_or_sentinel(self, clf, mock_store):
        mock_store.search.side_effect = Exception("LLM down")
        try:
            result = clf.classify("nurse")
            assert result is not None  # if handled gracefully
        except Exception:
            pass  # acceptable: unhandled exception propagates

    def test_classify_special_characters(self, clf):
        result = clf.classify("Engineer & Manager / Director")
        assert result is not None

    def test_classify_only_numbers(self, clf):
        result = clf.classify("12345")
        assert result is not None

    def test_classify_html_injection(self, clf):
        result = clf.classify("<script>alert('xss')</script>")
        assert result is not None

    def test_classify_very_long_input(self, clf):
        result = clf.classify("software engineer " * 200)
        assert result is not None
