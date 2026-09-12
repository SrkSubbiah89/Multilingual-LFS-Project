"""
Tests for backend/agents/cross_standard_coordinator.py (Item 1b of the
2026-09-12 multi-agent RAG work: deterministic backward-direction
cross-standard coordination).

Pure Python -- no CrewAI/LLM mocking needed, since
maybe_revise_isco_with_cross_signal() never calls an LLM. It reuses the
REAL SemanticRelationEngine(use_llm=False).analyse() as the compatibility
oracle, so these tests use real ISCO/ISIC/ISCED crosswalk facts (see
semantic_relation.py's _ISCO_MAJOR_TO_ISIC / _ISCO_SUBMAJOR_TO_ISIC /
_ISCO_MAJOR_TO_ISCED tables) rather than mocks:

- ISCO submajor "25" (ICT Professionals) -> ISIC "J" only.
- ISCO submajor "61" (Market Gardeners)   -> ISIC "A" only.
- ISCO submajor "26" (Legal/Social/Cultural) -> ISIC ["M", "R", "J"].
- ISCO major "2" (Professionals) -> ISCED 6-8.
"""

from types import SimpleNamespace

from backend.agents.cross_standard_coordinator import maybe_revise_isco_with_cross_signal


def make_isco_result(primary_code, hitl_required=True, alternatives=None):
    return SimpleNamespace(
        primary_code=primary_code,
        hitl_required=hitl_required,
        alternatives=alternatives or [],
    )


def make_alt(code, confidence=0.5):
    return SimpleNamespace(code=code, title_en="x", title_ar="x", confidence=confidence)


class TestMaybeReviseIscoWithCrossSignal:
    def test_already_compatible_primary_no_revision(self):
        # 2512 (submajor 25 -> ISIC "J" only) with isic_section="J": compatible.
        isco_result = make_isco_result("2512", hitl_required=True, alternatives=[make_alt("6111")])
        promoted, reason = maybe_revise_isco_with_cross_signal(isco_result, "J", 7)
        assert promoted is None
        assert reason == ""

    def test_incompatible_primary_with_compatible_alternative_promotes(self):
        # 2512 (submajor 25 -> ISIC "J" only) with isic_section="A": incompatible.
        # Alternative 6111 (submajor 61 -> ISIC "A" only): compatible.
        isco_result = make_isco_result(
            "2512", hitl_required=True, alternatives=[make_alt("6111"), make_alt("2621", 0.3)]
        )
        promoted, reason = maybe_revise_isco_with_cross_signal(isco_result, "A", None)
        assert promoted is not None
        assert promoted.code == "6111"
        assert "2512" in reason and "6111" in reason

    def test_no_compatible_alternative_no_revision(self):
        # 2512 incompatible with "A"; alternative 2621 (submajor 26 -> M/R/J)
        # is ALSO incompatible with "A" -- nothing to promote.
        isco_result = make_isco_result("2512", hitl_required=True, alternatives=[make_alt("2621")])
        promoted, reason = maybe_revise_isco_with_cross_signal(isco_result, "A", None)
        assert promoted is None
        assert reason == ""

    def test_no_alternatives_at_all_no_revision(self):
        isco_result = make_isco_result("2512", hitl_required=True, alternatives=[])
        promoted, reason = maybe_revise_isco_with_cross_signal(isco_result, "A", None)
        assert promoted is None
        assert reason == ""

    def test_not_hitl_required_never_fires(self):
        # Even though "A" is incompatible and 6111 would qualify, a
        # confident (non-hitl_required) primary must never be reconsidered.
        isco_result = make_isco_result("2512", hitl_required=False, alternatives=[make_alt("6111")])
        promoted, reason = maybe_revise_isco_with_cross_signal(isco_result, "A", None)
        assert promoted is None
        assert reason == ""

    def test_no_isic_or_isced_known_never_fires(self):
        isco_result = make_isco_result("2512", hitl_required=True, alternatives=[make_alt("6111")])
        promoted, reason = maybe_revise_isco_with_cross_signal(isco_result, None, None)
        assert promoted is None
        assert reason == ""

    def test_malformed_isco_result_degrades_to_no_revision(self):
        # Missing primary_code entirely -- must never raise.
        isco_result = SimpleNamespace(hitl_required=True, alternatives=[make_alt("6111")])
        promoted, reason = maybe_revise_isco_with_cross_signal(isco_result, "A", None)
        assert promoted is None
        assert reason == ""

    def test_alternative_as_plain_dict_also_works(self):
        # maybe_revise_isco_with_cross_signal supports dict-shaped
        # alternatives too, not just attribute-access objects.
        isco_result = make_isco_result(
            "2512", hitl_required=True, alternatives=[{"code": "6111", "confidence": 0.4}]
        )
        promoted, reason = maybe_revise_isco_with_cross_signal(isco_result, "A", None)
        assert promoted == {"code": "6111", "confidence": 0.4}
        assert "6111" in reason

    def test_engine_failure_degrades_to_no_revision(self, monkeypatch):
        import backend.agents.semantic_relation as sr

        def _boom(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(sr, "get_semantic_relation_engine", _boom)
        isco_result = make_isco_result("2512", hitl_required=True, alternatives=[make_alt("6111")])
        promoted, reason = maybe_revise_isco_with_cross_signal(isco_result, "A", None)
        assert promoted is None
        assert reason == ""
