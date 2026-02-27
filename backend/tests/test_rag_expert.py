"""
Tests for backend/agents/rag_expert.py

Pure-logic functions (_detect_script, _build_hierarchy_info,
_expand_hierarchy, _parse_explanations) are tested without mocking.

retrieve() tests patch VectorStore and Agent/Crew to avoid network calls
and bypass CrewAI's LLM validation entirely.
"""

from unittest.mock import MagicMock

import pytest

from backend.agents.rag_expert import (
    HierarchyInfo,
    OccupationCandidate,
    RAGExpert,
    RAGExpertResult,
    _detect_script,
)
from backend.rag.vector_store import OccupationMatch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_match(
    code: str = "2512",
    title_en: str = "Software Developers",
    title_ar: str = "مطورو البرمجيات",
    level: int = 4,
    confidence: float = 0.75,
) -> OccupationMatch:
    return OccupationMatch(
        code=code,
        title_en=title_en,
        title_ar=title_ar,
        level=level,
        description="Design, develop and maintain software applications.",
        confidence=confidence,
    )


_LLM_RESPONSE = (
    '[{"code": "2512", "explanation_en": "Matches software developer.", '
    '"explanation_ar": "يطابق مطور البرمجيات."}]'
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_store():
    store = MagicMock()
    store.search.return_value = [make_match()]
    return store


@pytest.fixture
def mock_crew(monkeypatch):
    crew_instance = MagicMock()
    crew_instance.kickoff.return_value = _LLM_RESPONSE
    crew_class = MagicMock(return_value=crew_instance)

    monkeypatch.setattr("backend.agents.rag_expert.Agent", MagicMock())
    monkeypatch.setattr("backend.agents.rag_expert.Crew", crew_class)
    monkeypatch.setattr("backend.agents.rag_expert.Task", MagicMock())
    return crew_instance


@pytest.fixture
def expert(monkeypatch, mock_store, mock_crew):
    """RAGExpert with LLM and VectorStore patched out."""
    monkeypatch.setattr(
        "backend.agents.rag_expert.get_llm",
        lambda *a, **kw: MagicMock(),
    )
    monkeypatch.setattr(
        "backend.agents.rag_expert.get_vector_store",
        lambda **kw: mock_store,
    )
    return RAGExpert()


# ---------------------------------------------------------------------------
# _detect_script (module-level helper)
# ---------------------------------------------------------------------------

class TestDetectScript:
    def test_arabic_text_returns_ar(self):
        assert _detect_script("أنا مهندس برمجيات") == "ar"

    def test_english_text_returns_en(self):
        assert _detect_script("software engineer") == "en"

    def test_code_switched_returns_mixed(self):
        assert _detect_script("أنا software engineer") == "mixed"

    def test_empty_string_returns_other(self):
        assert _detect_script("") == "other"

    def test_digits_only_returns_other(self):
        assert _detect_script("123 456") == "other"

    def test_predominantly_arabic_returns_ar(self):
        # Single Latin char should not push ratio above 10%
        result = _detect_script("أنا أعمل في وزارة الصحة a")
        assert result in ("ar", "mixed")

    def test_pure_latin_no_arabic_returns_en(self):
        assert _detect_script("doctor nurse teacher") == "en"


# ---------------------------------------------------------------------------
# _build_hierarchy_info
# ---------------------------------------------------------------------------

class TestBuildHierarchyInfo:
    @pytest.fixture
    def exp(self, monkeypatch, mock_store, mock_crew):
        monkeypatch.setattr(
            "backend.agents.rag_expert.get_llm", lambda *a, **kw: MagicMock()
        )
        monkeypatch.setattr(
            "backend.agents.rag_expert.get_vector_store", lambda **kw: mock_store
        )
        return RAGExpert()

    def test_level1_only_major_populated(self, exp):
        h = exp._build_hierarchy_info("2", 1)
        assert h.major_code == "2"
        assert "Professional" in h.major_title_en
        assert h.sub_major_code is None
        assert h.sub_major_title_en is None

    def test_level2_has_major_and_sub_major(self, exp):
        h = exp._build_hierarchy_info("25", 2)
        assert h.major_code == "2"
        assert h.sub_major_code == "25"
        assert "ICT" in h.sub_major_title_en or "Information" in h.sub_major_title_en

    def test_level4_has_both_parent_groups(self, exp):
        h = exp._build_hierarchy_info("2512", 4)
        assert h.major_code == "2"
        assert h.sub_major_code == "25"
        assert h.major_title_en != "Unknown"
        assert h.sub_major_title_en != "Unknown"

    def test_unknown_sub_major_returns_unknown_label(self, exp):
        # "9" is a known major group; "98" is not in the dataset
        h = exp._build_hierarchy_info("9812", 4)
        assert h.major_title_en != "Unknown"          # "9" exists → resolved
        assert h.sub_major_title_en == "Unknown"      # "98" not in dataset
        assert h.sub_major_title_ar == "غير معروف"

    def test_hierarchy_info_is_pydantic_model(self, exp):
        h = exp._build_hierarchy_info("2512", 4)
        assert isinstance(h, HierarchyInfo)


# ---------------------------------------------------------------------------
# _expand_hierarchy
# ---------------------------------------------------------------------------

class TestExpandHierarchy:
    @pytest.fixture
    def exp(self, monkeypatch, mock_store, mock_crew):
        monkeypatch.setattr(
            "backend.agents.rag_expert.get_llm", lambda *a, **kw: MagicMock()
        )
        monkeypatch.setattr(
            "backend.agents.rag_expert.get_vector_store", lambda **kw: mock_store
        )
        return RAGExpert()

    def test_returns_capped_at_top_k(self, exp):
        raw = [make_match(code=c) for c in ["2512", "2511", "2513", "2519", "2521",
                                             "2522", "2141", "2142", "2143", "2144"]]
        result = exp._expand_hierarchy(raw, top_k=5)
        assert len(result) <= 5

    def test_all_raw_results_tagged_as_semantic(self, exp):
        raw = [make_match(code="2512"), make_match(code="2511")]
        result = exp._expand_hierarchy(raw, top_k=5)
        semantic = [t for t in result if t[2] == "semantic"]
        assert len(semantic) >= 2

    def test_parent_groups_added_as_hierarchical_expansion(self, exp):
        # Single unit-group result → its parents should be added
        raw = [make_match(code="2512", confidence=0.80)]
        result = exp._expand_hierarchy(raw, top_k=5)
        stages = [t[2] for t in result]
        assert "hierarchical_expansion" in stages

    def test_no_duplicate_codes(self, exp):
        raw = [make_match(code="2512"), make_match(code="25", level=2),
               make_match(code="2", level=1)]
        result = exp._expand_hierarchy(raw, top_k=10)
        codes = [t[0].code for t in result]
        assert len(codes) == len(set(codes))

    def test_parent_confidence_lower_than_child(self, exp):
        raw = [make_match(code="2512", confidence=0.80)]
        result = exp._expand_hierarchy(raw, top_k=5)
        code_conf = {t[0].code: t[0].confidence for t in result}
        # Sub-major 25 and major 2 should have lower confidence than unit 2512
        if "25" in code_conf:
            assert code_conf["25"] < code_conf["2512"]
        if "2" in code_conf:
            assert code_conf["2"] < code_conf["2512"]

    def test_sorted_by_confidence_descending(self, exp):
        raw = [make_match(code="2512", confidence=0.80),
               make_match(code="2511", confidence=0.60)]
        result = exp._expand_hierarchy(raw, top_k=5)
        confs = [t[0].confidence for t in result]
        assert confs == sorted(confs, reverse=True)

    def test_returns_list_of_triples(self, exp):
        raw = [make_match()]
        result = exp._expand_hierarchy(raw, top_k=5)
        for item in result:
            assert len(item) == 3
            match, hier, stage = item
            assert isinstance(match, OccupationMatch)
            assert isinstance(hier, HierarchyInfo)
            assert isinstance(stage, str)


# ---------------------------------------------------------------------------
# _parse_explanations
# ---------------------------------------------------------------------------

class TestParseExplanations:
    @pytest.fixture
    def exp(self, monkeypatch, mock_store, mock_crew):
        monkeypatch.setattr(
            "backend.agents.rag_expert.get_llm", lambda *a, **kw: MagicMock()
        )
        monkeypatch.setattr(
            "backend.agents.rag_expert.get_vector_store", lambda **kw: mock_store
        )
        return RAGExpert()

    def _make_enriched(self, code="2512"):
        match = make_match(code=code)
        hier  = HierarchyInfo(
            major_code="2",
            major_title_en="Professionals",
            major_title_ar="المهنيون",
            sub_major_code="25",
            sub_major_title_en="ICT Professionals",
            sub_major_title_ar="مهنيو تكنولوجيا المعلومات",
        )
        return [(match, hier, "semantic")]

    def test_valid_json_produces_occupation_candidates(self, exp):
        enriched = self._make_enriched("2512")
        raw = ('[{"code": "2512", "explanation_en": "Software dev match.", '
               '"explanation_ar": "تطابق مطور البرمجيات."}]')
        result = exp._parse_explanations(raw, enriched)
        assert len(result) == 1
        assert isinstance(result[0], OccupationCandidate)

    def test_explanation_en_populated_from_llm(self, exp):
        enriched = self._make_enriched("2512")
        raw = ('[{"code": "2512", "explanation_en": "Custom EN explanation.", '
               '"explanation_ar": "شرح عربي."}]')
        result = exp._parse_explanations(raw, enriched)
        assert result[0].explanation_en == "Custom EN explanation."

    def test_explanation_ar_populated_from_llm(self, exp):
        enriched = self._make_enriched("2512")
        raw = ('[{"code": "2512", "explanation_en": "EN.", '
               '"explanation_ar": "شرح عربي مخصص."}]')
        result = exp._parse_explanations(raw, enriched)
        assert result[0].explanation_ar == "شرح عربي مخصص."

    def test_missing_code_falls_back_to_auto_explanation(self, exp):
        enriched = self._make_enriched("2512")
        result = exp._parse_explanations("[]", enriched)
        assert "2512" in result[0].explanation_en or "Software" in result[0].explanation_en

    def test_malformed_json_uses_fallback(self, exp):
        enriched = self._make_enriched("2512")
        result = exp._parse_explanations("not valid json", enriched)
        assert len(result) == 1
        assert result[0].code == "2512"

    def test_rank_starts_at_1(self, exp):
        enriched = self._make_enriched("2512")
        raw = ('[{"code": "2512", "explanation_en": "EN.", "explanation_ar": "AR."}]')
        result = exp._parse_explanations(raw, enriched)
        assert result[0].rank == 1

    def test_strips_markdown_fences(self, exp):
        enriched = self._make_enriched("2512")
        raw = ('```json\n[{"code": "2512", "explanation_en": "EN fenced.", '
               '"explanation_ar": "AR fenced."}]\n```')
        result = exp._parse_explanations(raw, enriched)
        assert result[0].explanation_en == "EN fenced."

    def test_hierarchy_info_attached_to_candidate(self, exp):
        enriched = self._make_enriched("2512")
        raw = '[{"code": "2512", "explanation_en": "EN.", "explanation_ar": "AR."}]'
        result = exp._parse_explanations(raw, enriched)
        assert result[0].hierarchy.major_code == "2"
        assert result[0].hierarchy.sub_major_code == "25"

    def test_retrieval_stage_preserved(self, exp):
        match = make_match(code="2512")
        hier  = HierarchyInfo(
            major_code="2", major_title_en="P", major_title_ar="م"
        )
        enriched = [(match, hier, "hierarchical_expansion")]
        raw = '[{"code": "2512", "explanation_en": "EN.", "explanation_ar": "AR."}]'
        result = exp._parse_explanations(raw, enriched)
        assert result[0].retrieval_stage == "hierarchical_expansion"


# ---------------------------------------------------------------------------
# retrieve — edge cases
# ---------------------------------------------------------------------------

class TestRetrieveEdgeCases:
    def test_empty_query_returns_empty_candidates(self, expert):
        result = expert.retrieve("")
        assert isinstance(result, RAGExpertResult)
        assert result.candidates == []
        assert result.total_retrieved == 0

    def test_whitespace_only_returns_empty_candidates(self, expert):
        result = expert.retrieve("   ")
        assert result.candidates == []

    def test_no_store_results_returns_empty_candidates(self, expert, mock_store):
        mock_store.search.return_value = []
        result = expert.retrieve("something obscure")
        assert result.candidates == []
        assert result.total_retrieved == 0


# ---------------------------------------------------------------------------
# retrieve — language detection
# ---------------------------------------------------------------------------

class TestRetrieveLanguage:
    def test_arabic_query_detected_as_ar(self, expert):
        result = expert.retrieve("مهندس برمجيات")
        assert result.language == "ar"

    def test_english_query_detected_as_en(self, expert):
        result = expert.retrieve("software engineer")
        assert result.language == "en"

    def test_mixed_query_detected_as_mixed(self, expert):
        result = expert.retrieve("أنا software engineer")
        assert result.language == "mixed"


# ---------------------------------------------------------------------------
# retrieve — result shape
# ---------------------------------------------------------------------------

class TestRetrieveResultShape:
    def test_result_is_rag_expert_result(self, expert):
        result = expert.retrieve("nurse")
        assert isinstance(result, RAGExpertResult)

    def test_candidates_are_occupation_candidates(self, expert):
        result = expert.retrieve("nurse")
        for c in result.candidates:
            assert isinstance(c, OccupationCandidate)

    def test_candidates_ranked_from_1(self, expert):
        result = expert.retrieve("nurse")
        if result.candidates:
            assert result.candidates[0].rank == 1

    def test_candidate_ranks_are_sequential(self, expert, mock_store):
        mock_store.search.return_value = [
            make_match(code="2221", confidence=0.80),
            make_match(code="3221", confidence=0.70),
        ]
        result = expert.retrieve("nurse")
        ranks = [c.rank for c in result.candidates]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_total_retrieved_matches_store_output(self, expert, mock_store):
        matches = [make_match(code=str(i) * 4) for i in range(1, 6)]
        mock_store.search.return_value = matches
        result = expert.retrieve("something")
        assert result.total_retrieved == len(matches)

    def test_confidence_scores_in_valid_range(self, expert, mock_store):
        mock_store.search.return_value = [make_match(confidence=0.80)]
        result = expert.retrieve("software developer")
        for c in result.candidates:
            assert 0.0 <= c.confidence <= 1.0

    def test_query_field_echoed_in_result(self, expert):
        result = expert.retrieve("software engineer")
        assert result.query == "software engineer"

    def test_method_is_string(self, expert):
        result = expert.retrieve("nurse")
        assert result.method in ("hierarchical", "semantic_only")


# ---------------------------------------------------------------------------
# retrieve — hierarchical method flag
# ---------------------------------------------------------------------------

class TestRetrieveMethodFlag:
    def test_method_is_hierarchical_when_parents_added(self, expert, mock_store):
        # Return a unit-group result; parent expansion should occur
        mock_store.search.return_value = [make_match(code="2512", level=4, confidence=0.80)]
        result = expert.retrieve("software developer")
        # Parents 25 and 2 should be in the candidate list, triggering "hierarchical"
        assert result.method == "hierarchical"

    def test_method_is_semantic_only_when_no_parents_needed(self, expert, mock_store):
        # Return only major-group results (no unit groups → no expansion needed)
        mock_store.search.return_value = [
            make_match(code="2", level=1, confidence=0.70),
            make_match(code="3", level=1, confidence=0.60),
        ]
        result = expert.retrieve("professional")
        assert result.method == "semantic_only"


# ---------------------------------------------------------------------------
# retrieve — top_k clamping
# ---------------------------------------------------------------------------

class TestTopK:
    def test_top_k_clamped_to_10(self, expert, mock_store):
        mock_store.search.return_value = [make_match()]
        expert.retrieve("nurse", top_k=50)
        # search should be called with top_k=10*2=20 (clamped 10 × 2)
        call_kwargs = mock_store.search.call_args
        called_top_k = call_kwargs[1].get("top_k") or call_kwargs[0][1]
        assert called_top_k <= 20

    def test_top_k_minimum_is_1(self, expert, mock_store):
        mock_store.search.return_value = [make_match()]
        result = expert.retrieve("nurse", top_k=0)
        assert result.candidates is not None
