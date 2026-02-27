"""
Tests for backend/agents/survey_orchestrator.py

Approach
--------
All eight sub-agents are replaced with lightweight fakes injected through
SurveyOrchestrator's constructor — no live LLM, Redis, Qdrant, or PostgreSQL
is touched.

Fake hierarchy
--------------
  FakeLanguageProcessor   – returns a configurable _FakeLPResult
  FakeEmotionalIntelligence – returns a configurable EmotionalAnalysis
  FakeContextMemory        – records append_turn / save_session calls
  FakeConversationManager  – returns a fixed reply; optionally sets COMPLETING
  FakeValidationAgent      – returns a configurable ValidationResult
  FakeISCOClassifier       – returns a fixed ISCOClassification (or raises)
  FakeAuditLogger          – records log_interaction / log_agent_decision calls
  FakeHITLManager          – records review_session calls; returns a QualityReport
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import pytest

import backend.agents.survey_orchestrator as _mod
from backend.agents.survey_orchestrator import (
    ISCOMatch,
    SurveyOrchestrator,
    TurnResult,
    _neutral_fallback,
    get_survey_orchestrator,
)
from backend.agents.conversation_manager import ConversationContext, ConversationState
from backend.agents.emotional_intelligence import (
    EmotionalAnalysis,
    EmotionalState,
    SurveyAction,
)
from backend.agents.hitl_quality_manager import (
    QualityMetrics,
    QualityReport,
    ReviewStatus,
)
from backend.agents.validation_agent import RuleViolation, ValidationResult


# ---------------------------------------------------------------------------
# Minimal data carriers
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class _FakeEntity:
    text:     str
    label:    str
    language: str = "en"


@dataclasses.dataclass
class _FakeLPResult:
    """Mimics LanguageProcessorResult with only the fields the orchestrator reads."""
    raw_text:          str   = "test"
    detected_language: str   = "en"
    confidence:        float = 0.95
    is_code_switched:  bool  = False
    arabic_ratio:      float = 0.0
    latin_ratio:       float = 1.0
    segments:          list  = dataclasses.field(default_factory=list)
    entities:          list  = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class _FakeOccMatch:
    """Mimics OccupationMatch with only the fields the orchestrator reads."""
    code:        str   = "2512"
    title_en:    str   = "Software Developers"
    title_ar:    str   = "مطورو البرمجيات"
    confidence:  float = 0.90
    level:       int   = 4
    description: str   = ""


@dataclasses.dataclass
class _FakeISCOClassification:
    """Mimics ISCOClassification with only the fields the orchestrator reads."""
    query:      str             = "software engineer"
    language:   str             = "en"
    primary:    _FakeOccMatch   = dataclasses.field(default_factory=_FakeOccMatch)
    candidates: list            = dataclasses.field(default_factory=list)
    reasoning:  str             = "Good semantic match."
    method:     str             = "semantic"


# ---------------------------------------------------------------------------
# Fake agents
# ---------------------------------------------------------------------------

class FakeLanguageProcessor:
    def __init__(self, result: Optional[_FakeLPResult] = None) -> None:
        self._result = result or _FakeLPResult()

    def process(self, text: str) -> _FakeLPResult:
        self._result.raw_text = text
        return self._result


class FakeEmotionalIntelligence:
    def __init__(self, result: Optional[EmotionalAnalysis] = None) -> None:
        self._result = result

    def analyze(self, text: str, language: str = "en") -> EmotionalAnalysis:
        return self._result or _neutral_fallback(text, language)


class FakeContextMemory:
    def __init__(self, missing: Optional[list] = None, fail: bool = False) -> None:
        self._missing = missing if missing is not None else []
        self._fail    = fail
        self.appended: list[tuple] = []
        self.saved:    list[tuple] = []

    def append_turn(
        self, session_id: int, role: str, content: str,
        detected_language: Optional[str] = None,
    ) -> None:
        if self._fail:
            raise RuntimeError("Redis down")
        self.appended.append((session_id, role, content))

    def save_session(
        self, session_id: int, state: str, language: str,
        collected_fields: dict, history: list,
    ) -> None:
        if self._fail:
            raise RuntimeError("Redis down")
        self.saved.append((session_id, state))

    def get_missing_fields(self, session_id: int) -> list[str]:
        if self._fail:
            raise RuntimeError("Redis down")
        return self._missing


class FakeConversationManager:
    def __init__(self, reply: str = "Next question?", completing: bool = False) -> None:
        self._reply     = reply
        self._completing = completing

    def new_context(self, session_id: int, language: str) -> ConversationContext:
        ctx = ConversationContext(session_id=session_id, language=language)
        if self._completing:
            ctx.state = ConversationState.COMPLETING
        return ctx

    def process_message(self, ctx: ConversationContext, message: str) -> str:
        if self._completing:
            ctx.state = ConversationState.COMPLETING
        return self._reply


class FakeValidationAgent:
    def __init__(
        self,
        is_valid: bool = True,
        violations: Optional[list[RuleViolation]] = None,
        fail: bool = False,
    ) -> None:
        self._is_valid   = is_valid
        self._violations = violations or []
        self._fail       = fail

    def validate(self, responses: dict, language: str = "en") -> ValidationResult:
        if self._fail:
            raise RuntimeError("Validation error")
        return ValidationResult(
            is_valid=self._is_valid,
            confidence=1.0,
            rule_violations=self._violations,
            semantic_issues=[],
            explanation_en="OK",
            explanation_ar="موافق",
            validated_data=responses,
        )


class FakeISCOClassifier:
    def __init__(self, result: Optional[_FakeISCOClassification] = None, fail: bool = False) -> None:
        self._result = result or _FakeISCOClassification()
        self._fail   = fail

    def classify(self, job_title: str, context: str = "", top_k: int = 5):
        if self._fail:
            raise RuntimeError("ISCO error")
        return self._result


class FakeAuditLogger:
    def __init__(self, fail: bool = False) -> None:
        self._fail       = fail
        self.interactions: list[str] = []
        self.decisions:    list[str] = []

    def log_interaction(
        self, event_type: str, description: str,
        session_id: Optional[int] = None, user_id: Optional[int] = None,
        **kw,
    ) -> None:
        if self._fail:
            raise RuntimeError("Audit error")
        self.interactions.append(event_type)

    def log_agent_decision(
        self, agent_name: str, decision_type: str, input_summary: str,
        output_summary: str, confidence: Optional[float] = None,
        session_id: Optional[int] = None, **kw,
    ) -> None:
        if self._fail:
            raise RuntimeError("Audit error")
        self.decisions.append(agent_name)


def _make_quality_report(session_id: int = 1) -> QualityReport:
    return QualityReport(
        review_id=1,
        session_id=session_id,
        quality_score=0.90,
        status=ReviewStatus.PASS,
        metrics=QualityMetrics(
            session_id=session_id,
            total_responses=2,
            responses_with_isco=2,
            responses_with_score=2,
            avg_confidence=0.90,
            isco_coverage=1.0,
            low_confidence_count=0,
            missing_isco_count=0,
            flagged_count=0,
        ),
        flagged_items=[],
        escalated=False,
        escalation_reason=None,
        report_en="Quality passed.",
        report_ar="اجتاز الجودة.",
        generated_at="2026-02-28T00:00:00Z",
    )


class FakeHITLManager:
    def __init__(self, fail: bool = False) -> None:
        self._fail    = fail
        self.reviewed: list[int] = []

    def review_session(self, session_id: int) -> QualityReport:
        if self._fail:
            raise RuntimeError("HITL error")
        self.reviewed.append(session_id)
        return _make_quality_report(session_id)


# ---------------------------------------------------------------------------
# Shared fixture builder
# ---------------------------------------------------------------------------

def _make_orch(
    lp:   Optional[FakeLanguageProcessor]     = None,
    ei:   Optional[FakeEmotionalIntelligence] = None,
    mem:  Optional[FakeContextMemory]         = None,
    conv: Optional[FakeConversationManager]   = None,
    va:   Optional[FakeValidationAgent]       = None,
    isco: Optional[FakeISCOClassifier]        = None,
    audit:Optional[FakeAuditLogger]           = None,
    hitl: Optional[FakeHITLManager]           = None,
) -> SurveyOrchestrator:
    return SurveyOrchestrator(
        language_processor=lp    or FakeLanguageProcessor(),
        emotional_intelligence=ei or FakeEmotionalIntelligence(),
        context_memory=mem         or FakeContextMemory(),
        conversation_manager=conv  or FakeConversationManager(),
        validation_agent=va        or FakeValidationAgent(),
        isco_classifier=isco       or FakeISCOClassifier(),
        audit_logger=audit         or FakeAuditLogger(),
        hitl_manager=hitl          or FakeHITLManager(),
    )


@pytest.fixture()
def orch() -> SurveyOrchestrator:
    return _make_orch()


# ===========================================================================
# Tests
# ===========================================================================


class TestISCOMatchDataclass:
    def test_fields_exist(self):
        m = ISCOMatch(
            job_title="nurse", code="2221",
            title_en="Nursing Professionals", title_ar="ممرضون",
            confidence=0.88, method="semantic",
        )
        assert m.job_title == "nurse"
        assert m.code == "2221"
        assert m.title_en == "Nursing Professionals"
        assert m.title_ar == "ممرضون"
        assert m.confidence == 0.88
        assert m.method == "semantic"

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(ISCOMatch)


class TestTurnResultDataclass:
    def test_required_fields(self, orch):
        r = orch.process_turn(1, "Hello")
        assert isinstance(r, TurnResult)
        assert r.session_id == 1
        assert isinstance(r.reply, str)
        assert isinstance(r.state, str)
        assert isinstance(r.detected_language, str)
        assert isinstance(r.is_code_switched, bool)
        assert isinstance(r.entities, list)
        assert isinstance(r.isco_matches, list)
        assert isinstance(r.emotional_state, str)
        assert isinstance(r.survey_action, str)
        assert isinstance(r.adapted_prompt, str)
        assert isinstance(r.validation_passed, bool)
        assert isinstance(r.validation_violations, list)
        assert isinstance(r.missing_fields, list)
        assert isinstance(r.session_completed, bool)

    def test_quality_report_none_by_default(self, orch):
        r = orch.process_turn(1, "Hello")
        assert r.quality_report is None

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(TurnResult)


class TestProcessTurnBasic:
    def test_reply_from_conversation_manager(self):
        orch = _make_orch(conv=FakeConversationManager(reply="What is your job?"))
        r = orch.process_turn(1, "Hi")
        assert r.reply == "What is your job?"

    def test_session_id_propagated(self, orch):
        r = orch.process_turn(42, "Hi")
        assert r.session_id == 42

    def test_state_is_string(self, orch):
        r = orch.process_turn(1, "Hi")
        assert isinstance(r.state, str)
        # default state is greeting
        assert r.state == ConversationState.GREETING.value

    def test_session_not_completed_by_default(self, orch):
        r = orch.process_turn(1, "Hi")
        assert r.session_completed is False

    def test_validation_passed_when_no_collected_data(self, orch):
        r = orch.process_turn(1, "Hi")
        assert r.validation_passed is True
        assert r.validation_violations == []

    def test_no_isco_matches_when_no_job_entities(self, orch):
        r = orch.process_turn(1, "I live in Riyadh.")
        assert r.isco_matches == []

    def test_empty_entities_list(self, orch):
        r = orch.process_turn(1, "Hello")
        assert r.entities == []


class TestProcessTurnLanguageDetection:
    def test_english_detection_preserved(self):
        lp = FakeLanguageProcessor(_FakeLPResult(detected_language="en"))
        orch = _make_orch(lp=lp)
        r = orch.process_turn(1, "Hello", language="en")
        assert r.detected_language == "en"

    def test_arabic_detection_preserved(self):
        lp = FakeLanguageProcessor(_FakeLPResult(detected_language="ar"))
        orch = _make_orch(lp=lp)
        r = orch.process_turn(1, "مرحبا", language="ar")
        assert r.detected_language == "ar"

    def test_unknown_lang_falls_back_to_session_language(self):
        lp = FakeLanguageProcessor(_FakeLPResult(detected_language="fr"))
        orch = _make_orch(lp=lp)
        r = orch.process_turn(1, "Bonjour", language="en")
        assert r.detected_language == "en"

    def test_code_switched_flag_propagated(self):
        lp = FakeLanguageProcessor(_FakeLPResult(is_code_switched=True))
        orch = _make_orch(lp=lp)
        r = orch.process_turn(1, "أنا software engineer")
        assert r.is_code_switched is True

    def test_arabic_lang_uses_arabic_adapted_prompt(self):
        ei_result = _neutral_fallback("test", "ar")
        ei_result.adapted_prompt_ar = "الرجاء الاستمرار."
        ei = FakeEmotionalIntelligence(result=ei_result)
        lp = FakeLanguageProcessor(_FakeLPResult(detected_language="ar"))
        orch = _make_orch(lp=lp, ei=ei)
        r = orch.process_turn(1, "مرحبا")
        assert r.adapted_prompt == "الرجاء الاستمرار."

    def test_english_lang_uses_english_adapted_prompt(self):
        ei_result = _neutral_fallback("test", "en")
        ei_result.adapted_prompt_en = "Please continue."
        ei = FakeEmotionalIntelligence(result=ei_result)
        lp = FakeLanguageProcessor(_FakeLPResult(detected_language="en"))
        orch = _make_orch(lp=lp, ei=ei)
        r = orch.process_turn(1, "Hello")
        assert r.adapted_prompt == "Please continue."

    def test_context_language_updated_on_detection_change(self):
        # Start with English context, detect Arabic
        lp = FakeLanguageProcessor(_FakeLPResult(detected_language="ar"))
        orch = _make_orch(lp=lp)
        orch.process_turn(1, "مرحبا", language="en")
        ctx = orch.get_conv_context(1)
        assert ctx.language == "ar"


class TestProcessTurnEntities:
    def test_job_title_entity_produces_isco_match(self):
        lp = FakeLanguageProcessor(_FakeLPResult(
            entities=[_FakeEntity(text="nurse", label="JOB_TITLE")],
        ))
        orch = _make_orch(lp=lp)
        r = orch.process_turn(1, "I am a nurse.")
        assert len(r.isco_matches) == 1
        m = r.isco_matches[0]
        assert m.job_title == "nurse"
        assert m.code == "2512"       # from _FakeISCOClassification default
        assert m.method == "semantic"

    def test_non_job_entity_skipped(self):
        lp = FakeLanguageProcessor(_FakeLPResult(
            entities=[_FakeEntity(text="Riyadh", label="LOCATION")],
        ))
        orch = _make_orch(lp=lp)
        r = orch.process_turn(1, "I live in Riyadh.")
        assert r.isco_matches == []

    def test_multiple_job_titles_produce_multiple_matches(self):
        lp = FakeLanguageProcessor(_FakeLPResult(
            entities=[
                _FakeEntity(text="nurse", label="JOB_TITLE"),
                _FakeEntity(text="teacher", label="JOB_TITLE"),
            ],
        ))
        orch = _make_orch(lp=lp)
        r = orch.process_turn(1, "I work as nurse and used to be a teacher.")
        assert len(r.isco_matches) == 2

    def test_entities_serialised_as_dicts(self):
        lp = FakeLanguageProcessor(_FakeLPResult(
            entities=[_FakeEntity(text="nurse", label="JOB_TITLE", language="en")],
        ))
        orch = _make_orch(lp=lp)
        r = orch.process_turn(1, "I am a nurse.")
        assert r.entities == [{"text": "nurse", "label": "JOB_TITLE", "language": "en"}]

    def test_isco_match_fields_mapped_correctly(self):
        custom = _FakeISCOClassification(
            primary=_FakeOccMatch(
                code="3212", title_en="Medical Lab", title_ar="معمل",
                confidence=0.75,
            ),
            method="llm_ranked",
        )
        lp = FakeLanguageProcessor(_FakeLPResult(
            entities=[_FakeEntity(text="lab tech", label="JOB_TITLE")],
        ))
        orch = _make_orch(lp=lp, isco=FakeISCOClassifier(result=custom))
        r = orch.process_turn(1, "I am a lab tech.")
        m = r.isco_matches[0]
        assert m.code == "3212"
        assert m.title_en == "Medical Lab"
        assert m.confidence == 0.75
        assert m.method == "llm_ranked"


class TestProcessTurnEmotional:
    def test_neutral_emotional_state(self, orch):
        r = orch.process_turn(1, "I work full time.")
        assert r.emotional_state == EmotionalState.NEUTRAL.value

    def test_stressed_emotional_state_propagated(self):
        ei_result = EmotionalAnalysis(
            raw_text="I'm so stressed",
            detected_language="en",
            emotional_state=EmotionalState.STRESSED,
            confidence=0.8,
            intensity=0.7,
            signals=[],
            adapted_prompt_en="Take your time.",
            adapted_prompt_ar="خذ وقتك.",
            support_message_en="It's okay to pause.",
            support_message_ar="لا بأس بالتوقف.",
            survey_action=SurveyAction.SLOW_DOWN,
            action_reason="High stress detected.",
        )
        orch = _make_orch(ei=FakeEmotionalIntelligence(result=ei_result))
        r = orch.process_turn(1, "I'm so stressed")
        assert r.emotional_state == "stressed"
        assert r.survey_action == "slow_down"

    def test_frustrated_end_action(self):
        ei_result = EmotionalAnalysis(
            raw_text="I HATE THIS",
            detected_language="en",
            emotional_state=EmotionalState.FRUSTRATED,
            confidence=0.9,
            intensity=0.9,
            signals=[],
            adapted_prompt_en="We can stop here.",
            adapted_prompt_ar="يمكننا التوقف.",
            support_message_en="Thank you.",
            support_message_ar="شكرًا.",
            survey_action=SurveyAction.END,
            action_reason="High frustration.",
        )
        orch = _make_orch(ei=FakeEmotionalIntelligence(result=ei_result))
        r = orch.process_turn(1, "I HATE THIS")
        assert r.survey_action == "end"

    def test_survey_action_continue_by_default(self, orch):
        r = orch.process_turn(1, "Hello")
        assert r.survey_action == SurveyAction.CONTINUE.value

    def test_ei_failure_returns_neutral_fallback(self):
        class _FailEI:
            def analyze(self, text, language="en"):
                raise RuntimeError("EI down")

        orch = _make_orch(ei=_FailEI())
        r = orch.process_turn(1, "Hello")
        assert r.emotional_state == "neutral"
        assert r.survey_action == "continue"


class TestProcessTurnValidation:
    def test_no_validation_when_empty_collected_data(self, orch):
        r = orch.process_turn(1, "Hello")
        assert r.validation_passed is True
        assert r.validation_violations == []

    def test_validation_pass_propagated(self):
        conv = FakeConversationManager()
        # Inject some collected data
        orch = _make_orch(conv=conv, va=FakeValidationAgent(is_valid=True))
        ctx = orch._get_or_create_context(1, "en")
        ctx.collected_data["employment_status"] = "employed"
        r = orch.process_turn(1, "I work full time.")
        assert r.validation_passed is True

    def test_validation_failure_propagated(self):
        violation = RuleViolation(
            rule_id="R06",
            field="hours_per_week",
            severity="error",
            message_en="Unemployed but reported hours.",
            message_ar="عاطل لكن أبلغ عن ساعات.",
        )
        conv = FakeConversationManager()
        va = FakeValidationAgent(is_valid=False, violations=[violation])
        orch = _make_orch(conv=conv, va=va)
        ctx = orch._get_or_create_context(1, "en")
        ctx.collected_data["employment_status"] = "unemployed"
        ctx.collected_data["hours_per_week"] = "40"
        r = orch.process_turn(1, "I don't work.")
        assert r.validation_passed is False
        assert any("R06" in v for v in r.validation_violations)

    def test_validation_error_falls_back_to_pass(self):
        conv = FakeConversationManager()
        orch = _make_orch(conv=conv, va=FakeValidationAgent(fail=True))
        ctx = orch._get_or_create_context(1, "en")
        ctx.collected_data["employment_status"] = "employed"
        r = orch.process_turn(1, "I work full time.")
        assert r.validation_passed is True   # safe fallback


class TestProcessTurnCompletion:
    def test_session_completed_flag(self):
        orch = _make_orch(conv=FakeConversationManager(completing=True))
        r = orch.process_turn(1, "Thank you.")
        assert r.session_completed is True

    def test_quality_report_set_on_completion(self):
        hitl = FakeHITLManager()
        orch = _make_orch(conv=FakeConversationManager(completing=True), hitl=hitl)
        r = orch.process_turn(1, "Done.")
        assert r.quality_report is not None
        assert isinstance(r.quality_report, QualityReport)

    def test_hitl_review_called_with_session_id(self):
        hitl = FakeHITLManager()
        orch = _make_orch(conv=FakeConversationManager(completing=True), hitl=hitl)
        orch.process_turn(42, "Done.")
        assert 42 in hitl.reviewed

    def test_conv_context_cleared_on_completion(self):
        orch = _make_orch(conv=FakeConversationManager(completing=True))
        orch.process_turn(1, "Done.")
        assert orch.get_conv_context(1) is None

    def test_completion_logs_session_completed_event(self):
        audit = FakeAuditLogger()
        orch = _make_orch(conv=FakeConversationManager(completing=True), audit=audit)
        orch.process_turn(1, "Done.")
        from backend.agents.audit_logger import EventType
        assert EventType.SESSION_COMPLETED in audit.interactions

    def test_quality_report_none_when_hitl_fails(self):
        hitl = FakeHITLManager(fail=True)
        orch = _make_orch(conv=FakeConversationManager(completing=True), hitl=hitl)
        r = orch.process_turn(1, "Done.")
        assert r.quality_report is None   # safe fallback

    def test_quality_report_none_when_not_completed(self, orch):
        r = orch.process_turn(1, "Hello")
        assert r.quality_report is None


class TestProcessTurnContextMemory:
    def test_turn_appended_to_memory(self):
        mem = FakeContextMemory()
        orch = _make_orch(mem=mem)
        orch.process_turn(1, "Hello")
        # Two appended calls: user turn + assistant turn
        assert len(mem.appended) == 2
        roles = [t[1] for t in mem.appended]
        assert "user" in roles
        assert "assistant" in roles

    def test_session_saved_to_memory(self):
        mem = FakeContextMemory()
        orch = _make_orch(mem=mem)
        orch.process_turn(5, "Hi")
        assert len(mem.saved) == 1
        assert mem.saved[0][0] == 5

    def test_memory_failure_does_not_abort_turn(self):
        orch = _make_orch(mem=FakeContextMemory(fail=True))
        # Should complete without raising
        r = orch.process_turn(1, "Hello")
        assert r.reply == "Next question?"

    def test_missing_fields_returned(self):
        missing = ["industry", "employment_type"]
        orch = _make_orch(mem=FakeContextMemory(missing=missing))
        r = orch.process_turn(1, "Hello")
        assert r.missing_fields == missing

    def test_missing_fields_empty_on_memory_failure(self):
        orch = _make_orch(mem=FakeContextMemory(fail=True))
        r = orch.process_turn(1, "Hello")
        assert r.missing_fields == []


class TestProcessTurnAuditLogging:
    def test_interaction_logged(self):
        audit = FakeAuditLogger()
        orch = _make_orch(audit=audit)
        orch.process_turn(1, "Hello", user_id=7)
        from backend.agents.audit_logger import EventType
        assert EventType.MESSAGE_SENT in audit.interactions

    def test_agent_decision_logged(self):
        audit = FakeAuditLogger()
        orch = _make_orch(audit=audit)
        orch.process_turn(1, "Hello")
        assert "LanguageProcessor" in audit.decisions

    def test_audit_failure_does_not_abort_turn(self):
        orch = _make_orch(audit=FakeAuditLogger(fail=True))
        r = orch.process_turn(1, "Hello")
        assert r.reply == "Next question?"


class TestProcessTurnISCOFailure:
    def test_isco_failure_skipped_gracefully(self):
        lp = FakeLanguageProcessor(_FakeLPResult(
            entities=[_FakeEntity(text="nurse", label="JOB_TITLE")],
        ))
        orch = _make_orch(lp=lp, isco=FakeISCOClassifier(fail=True))
        r = orch.process_turn(1, "I am a nurse.")
        # ISCO failed but turn still succeeds with empty matches
        assert r.isco_matches == []
        assert r.reply == "Next question?"


class TestContextManagement:
    def test_get_conv_context_none_before_first_turn(self):
        orch = _make_orch()
        assert orch.get_conv_context(99) is None

    def test_get_conv_context_after_turn(self, orch):
        orch.process_turn(1, "Hello")
        ctx = orch.get_conv_context(1)
        assert ctx is not None
        assert isinstance(ctx, ConversationContext)

    def test_context_reused_across_turns(self, orch):
        orch.process_turn(1, "Hello")
        ctx1 = orch.get_conv_context(1)
        orch.process_turn(1, "I work as an engineer.")
        ctx2 = orch.get_conv_context(1)
        assert ctx1 is ctx2   # same object

    def test_separate_sessions_have_separate_contexts(self, orch):
        orch.process_turn(1, "Hello")
        orch.process_turn(2, "Hello")
        assert orch.get_conv_context(1) is not orch.get_conv_context(2)

    def test_drop_context_removes_cache_entry(self, orch):
        orch.process_turn(1, "Hello")
        orch.drop_context(1)
        assert orch.get_conv_context(1) is None

    def test_drop_context_noop_for_unknown_session(self, orch):
        # Should not raise
        orch.drop_context(9999)

    def test_context_not_present_after_completion(self):
        orch = _make_orch(conv=FakeConversationManager(completing=True))
        orch.process_turn(1, "Done.")
        assert orch.get_conv_context(1) is None

    def test_new_context_created_after_drop(self, orch):
        orch.process_turn(1, "Hello")
        orch.drop_context(1)
        orch.process_turn(1, "Hello again")
        ctx = orch.get_conv_context(1)
        assert ctx is not None
        assert ctx.state == ConversationState.GREETING   # fresh context


class TestNeutralFallback:
    def test_returns_emotional_analysis(self):
        r = _neutral_fallback("test", "en")
        assert isinstance(r, EmotionalAnalysis)

    def test_state_is_neutral(self):
        r = _neutral_fallback("test", "en")
        assert r.emotional_state == EmotionalState.NEUTRAL

    def test_action_is_continue(self):
        r = _neutral_fallback("test", "en")
        assert r.survey_action == SurveyAction.CONTINUE

    def test_arabic_prompt_set(self):
        r = _neutral_fallback("test", "ar")
        assert len(r.adapted_prompt_ar) > 0

    def test_raw_text_preserved(self):
        r = _neutral_fallback("hello world", "en")
        assert r.raw_text == "hello world"


class TestGetSurveyOrchestratorSingleton:
    def setup_method(self):
        _mod._instance = None

    def teardown_method(self):
        _mod._instance = None

    def test_returns_survey_orchestrator(self, monkeypatch):
        # Prevent real agent construction
        monkeypatch.setattr(
            _mod, "SurveyOrchestrator",
            lambda **kw: object.__new__(SurveyOrchestrator),
        )
        _mod._instance = _make_orch()  # pre-set a real fake instance
        result = get_survey_orchestrator()
        assert result is _mod._instance

    def test_singleton_same_instance(self):
        _mod._instance = _make_orch()
        assert get_survey_orchestrator() is get_survey_orchestrator()

    def test_instance_is_none_initially(self):
        assert _mod._instance is None
