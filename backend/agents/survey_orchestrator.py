"""
backend/agents/survey_orchestrator.py

Central coordination layer that wires all LFS CrewAI agents into a single
conversational turn pipeline.

Pipeline (per user message)
----------------------------
1. LanguageProcessor    — detect language, extract NER entities
2. EmotionalIntelligence — gauge respondent emotional state
3. ConversationManager  — advance FSM, generate interviewer reply
4. ContextMemory        — append turns, sync collected fields to Redis
5. ValidationAgent      — check collected data for rule / semantic issues
6. ISCOClassifier       — classify each JOB_TITLE entity (non-blocking)
7. AuditLogger          — record interaction + agent decisions
8. HITLQualityManager   — run quality review when session completes

Only LanguageProcessor and ConversationManager are critical (exceptions
propagate to the caller). All enrichment steps are wrapped in try/except
so that infrastructure failures never abort a live survey turn.

Usage
-----
from backend.agents.survey_orchestrator import SurveyOrchestrator, TurnResult

orch = SurveyOrchestrator()
result: TurnResult = orch.process_turn(
    session_id=42,
    user_message="I work as a software engineer.",
    user_id=7,
    language="en",
)

print(result.reply)             # "Great, can you tell me which industry …"
print(result.emotional_state)   # "neutral"
print(result.isco_matches)      # [ISCOMatch(job_title="software engineer", …)]
print(result.missing_fields)    # ["industry", "employment_type", …]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from backend.agents.audit_logger import (
    AgentDecisionType,
    AuditLogger,
    EventType,
    get_audit_logger,
)
from backend.agents.context_memory import ContextMemory, get_context_memory
from backend.agents.conversation_manager import (
    ConversationContext,
    ConversationManager,
    ConversationState,
)
from backend.agents.emotional_intelligence import (
    EmotionalAnalysis,
    EmotionalIntelligence,
    EmotionalState,
    SurveyAction,
    get_emotional_intelligence,
)
from backend.agents.hitl_quality_manager import (
    HITLQualityManager,
    QualityReport,
    get_hitl_quality_manager,
)
from backend.agents.isco_classifier import ISCOClassifier, ISCOClassification
from backend.agents.language_processor import (
    LanguageProcessor,
    LanguageProcessorResult,
)
from backend.agents.validation_agent import ValidationAgent, ValidationResult


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

@dataclass
class ISCOMatch:
    """Compact ISCO-08 result for a single job-title mention in one turn."""

    job_title:  str
    code:       str
    title_en:   str
    title_ar:   str
    confidence: float
    method:     str


@dataclass
class TurnResult:
    """Complete enriched output for one conversational turn."""

    session_id:            int
    reply:                 str              # interviewer message to show the user
    state:                 str              # ConversationState.value
    detected_language:     str              # "en" | "ar"
    is_code_switched:      bool
    entities:              list[dict]       # [{text, label, language}] from NER
    isco_matches:          list[ISCOMatch]  # one per JOB_TITLE entity
    emotional_state:       str              # EmotionalState.value
    survey_action:         str              # SurveyAction.value
    adapted_prompt:        str              # language-appropriate interviewer hint
    validation_passed:     bool
    validation_violations: list[str]        # human-readable violation messages
    missing_fields:        list[str]        # LFS fields not yet collected
    session_completed:     bool
    quality_report:        Optional[QualityReport] = None


# ---------------------------------------------------------------------------
# Fallback EmotionalAnalysis returned when the EI agent is unavailable
# ---------------------------------------------------------------------------

def _neutral_fallback(text: str, language: str) -> EmotionalAnalysis:
    return EmotionalAnalysis(
        raw_text=text,
        detected_language=language,
        emotional_state=EmotionalState.NEUTRAL,
        confidence=1.0,
        intensity=0.0,
        signals=[],
        adapted_prompt_en="Please continue with the survey.",
        adapted_prompt_ar="يرجى الاستمرار في الاستبيان.",
        support_message_en="Thank you for your patience.",
        support_message_ar="شكرًا لصبركم.",
        survey_action=SurveyAction.CONTINUE,
        action_reason="Emotional analysis unavailable; defaulting to neutral.",
    )


# ---------------------------------------------------------------------------
# SurveyOrchestrator
# ---------------------------------------------------------------------------

class SurveyOrchestrator:
    """
    Coordinates all LFS agents into a single conversational turn pipeline.

    Parameters
    ----------
    language_processor, emotional_intelligence, context_memory,
    conversation_manager, validation_agent, isco_classifier,
    audit_logger, hitl_manager :
        Inject pre-built agent instances.  If None, the real
        singleton / default instances are constructed automatically.
        Pass fakes here for unit testing without live infrastructure.
    """

    def __init__(
        self,
        language_processor:     Optional[LanguageProcessor]     = None,
        emotional_intelligence: Optional[EmotionalIntelligence] = None,
        context_memory:         Optional[ContextMemory]         = None,
        conversation_manager:   Optional[ConversationManager]   = None,
        validation_agent:       Optional[ValidationAgent]       = None,
        isco_classifier:        Optional[ISCOClassifier]        = None,
        audit_logger:           Optional[AuditLogger]           = None,
        hitl_manager:           Optional[HITLQualityManager]    = None,
    ) -> None:
        self._lp    = language_processor     or LanguageProcessor()
        self._ei    = emotional_intelligence or get_emotional_intelligence()
        self._mem   = context_memory         or get_context_memory()
        self._conv  = conversation_manager   or ConversationManager()
        self._va    = validation_agent       or ValidationAgent()
        self._isco  = isco_classifier        or ISCOClassifier()
        self._audit = audit_logger           or get_audit_logger()
        self._hitl  = hitl_manager           or get_hitl_quality_manager()

        # Per-session ConversationContext cache (in-process, not Redis).
        # Mirrors the _contexts dict previously held in survey_routes.py.
        self._conv_contexts: dict[int, ConversationContext] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_turn(
        self,
        session_id:   int,
        user_message: str,
        user_id:      Optional[int] = None,
        language:     str = "en",
    ) -> TurnResult:
        """
        Process one conversational turn end-to-end.

        Parameters
        ----------
        session_id : int
            DB primary key of the active SurveySession.
        user_message : str
            Raw text sent by the respondent.
        user_id : int | None
            DB primary key of the authenticated User (used for audit
            logging only; may be None in test / internal contexts).
        language : str
            Current session language ("en" | "ar").  Overridden by the
            language detector when it is confident.

        Returns
        -------
        TurnResult
            Complete enriched result including the interviewer reply,
            emotional state, ISCO matches, validation outcome, missing
            LFS fields, and an optional quality report on completion.
        """
        # ── 1. Language detection + NER (critical — never caught) ─────
        lp_result: LanguageProcessorResult = self._lp.process(user_message)
        detected_lang = (
            lp_result.detected_language
            if lp_result.detected_language in ("en", "ar")
            else language
        )

        # ── 2. Emotional analysis (enrichment — safe fallback) ────────
        ei_result = self._safe_emotional_analysis(user_message, detected_lang)

        # ── 3. Conversation FSM turn (critical — never caught) ────────
        ctx = self._get_or_create_context(session_id, detected_lang)
        if detected_lang != ctx.language:
            ctx.language = detected_lang
        reply: str = self._conv.process_message(ctx, user_message)

        # ── 4. Sync Redis context memory (enrichment — safe) ──────────
        self._sync_memory(session_id, user_message, reply, ctx, lp_result)

        # ── 5. Determine missing LFS fields (enrichment — safe) ───────
        missing = self._safe_missing_fields(session_id)

        # ── 6. Validate collected data (enrichment — safe) ────────────
        validation_passed, validation_violations = self._run_validation(
            ctx.collected_data, detected_lang
        )

        # ── 7. ISCO classification per JOB_TITLE entity (enrichment) ──
        isco_matches = self._classify_entities(lp_result.entities, detected_lang)

        # ── 8. Audit logging (enrichment — safe) ──────────────────────
        self._log_turn(
            session_id, user_id, user_message, reply, lp_result, ei_result, ctx
        )

        # ── 9. Session completion + HITL quality review ───────────────
        session_completed = ctx.state == ConversationState.COMPLETING
        quality_report: Optional[QualityReport] = None
        if session_completed:
            quality_report = self._run_quality_review(session_id)
            self._safe_log_completion(session_id, user_id)
            # Remove in-process context — session is done
            self._conv_contexts.pop(session_id, None)

        # ── Choose the language-appropriate adapted prompt ─────────────
        adapted = (
            ei_result.adapted_prompt_ar
            if detected_lang == "ar"
            else ei_result.adapted_prompt_en
        )

        return TurnResult(
            session_id=session_id,
            reply=reply,
            state=ctx.state.value,
            detected_language=detected_lang,
            is_code_switched=lp_result.is_code_switched,
            entities=[
                {"text": e.text, "label": e.label, "language": e.language}
                for e in lp_result.entities
            ],
            isco_matches=isco_matches,
            emotional_state=ei_result.emotional_state.value,
            survey_action=ei_result.survey_action.value,
            adapted_prompt=adapted,
            validation_passed=validation_passed,
            validation_violations=validation_violations,
            missing_fields=missing,
            session_completed=session_completed,
            quality_report=quality_report,
        )

    def get_conv_context(self, session_id: int) -> Optional[ConversationContext]:
        """
        Return the cached ConversationContext for this session, or None.

        The route layer can use this to read ``ctx.collected_data`` for DB
        persistence without re-running the pipeline.
        """
        return self._conv_contexts.get(session_id)

    def drop_context(self, session_id: int) -> None:
        """
        Remove the cached ConversationContext for this session.

        Call this when a session is deleted so the in-process cache does
        not leak references.
        """
        self._conv_contexts.pop(session_id, None)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_or_create_context(
        self, session_id: int, language: str
    ) -> ConversationContext:
        if session_id not in self._conv_contexts:
            self._conv_contexts[session_id] = self._conv.new_context(
                session_id, language
            )
        return self._conv_contexts[session_id]

    def _safe_emotional_analysis(
        self, text: str, language: str
    ) -> EmotionalAnalysis:
        try:
            return self._ei.analyze(text, language=language)
        except Exception:
            return _neutral_fallback(text, language)

    def _sync_memory(
        self,
        session_id:   int,
        user_message: str,
        reply:        str,
        ctx:          ConversationContext,
        lp_result:    LanguageProcessorResult,
    ) -> None:
        try:
            self._mem.append_turn(
                session_id,
                role="user",
                content=user_message,
                detected_language=lp_result.detected_language,
            )
            self._mem.append_turn(
                session_id,
                role="assistant",
                content=reply,
                detected_language=ctx.language,
            )
            self._mem.save_session(
                session_id,
                state=ctx.state.value,
                language=ctx.language,
                collected_fields={
                    k: str(v) for k, v in ctx.collected_data.items() if v
                },
                history=[],
            )
        except Exception:
            pass  # Redis unavailability must not abort a survey turn

    def _safe_missing_fields(self, session_id: int) -> list[str]:
        try:
            return self._mem.get_missing_fields(session_id)
        except Exception:
            return []

    def _run_validation(
        self, collected_data: dict, language: str
    ) -> tuple[bool, list[str]]:
        if not collected_data:
            return True, []
        try:
            result: ValidationResult = self._va.validate(
                collected_data, language=language
            )
            violations = [
                f"[{v.rule_id}] {v.message_en}"
                for v in result.rule_violations
            ]
            return result.is_valid, violations
        except Exception:
            return True, []

    def _classify_entities(
        self, entities: list, language: str
    ) -> list[ISCOMatch]:
        matches: list[ISCOMatch] = []
        for entity in entities:
            if entity.label != "JOB_TITLE":
                continue
            try:
                clf: ISCOClassification = self._isco.classify(
                    entity.text,
                    context=f"language={language}",
                )
                matches.append(ISCOMatch(
                    job_title=entity.text,
                    code=clf.primary.code,
                    title_en=clf.primary.title_en,
                    title_ar=clf.primary.title_ar,
                    confidence=clf.primary.confidence,
                    method=clf.method,
                ))
            except Exception:
                pass  # ISCO enrichment must not abort the turn
        return matches

    def _log_turn(
        self,
        session_id:   int,
        user_id:      Optional[int],
        user_message: str,
        reply:        str,
        lp_result:    LanguageProcessorResult,
        ei_result:    EmotionalAnalysis,
        ctx:          ConversationContext,
    ) -> None:
        try:
            self._audit.log_interaction(
                event_type=EventType.MESSAGE_SENT,
                description=(
                    f"Turn in state '{ctx.state.value}'; "
                    f"lang={lp_result.detected_language}; "
                    f"emotion={ei_result.emotional_state.value}"
                ),
                session_id=session_id,
                user_id=user_id,
            )
            self._audit.log_agent_decision(
                agent_name="LanguageProcessor",
                decision_type=AgentDecisionType.LANGUAGE_DETECTION,
                input_summary=user_message[:200],
                output_summary=(
                    f"lang={lp_result.detected_language}, "
                    f"code_switched={lp_result.is_code_switched}, "
                    f"entities={len(lp_result.entities)}"
                ),
                confidence=lp_result.confidence,
                session_id=session_id,
            )
        except Exception:
            pass

    def _safe_log_completion(
        self, session_id: int, user_id: Optional[int]
    ) -> None:
        try:
            self._audit.log_interaction(
                event_type=EventType.SESSION_COMPLETED,
                description="Survey session completed by ConversationManager FSM.",
                session_id=session_id,
                user_id=user_id,
            )
        except Exception:
            pass

    def _run_quality_review(self, session_id: int) -> Optional[QualityReport]:
        try:
            return self._hitl.review_session(session_id)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: Optional[SurveyOrchestrator] = None


def get_survey_orchestrator() -> SurveyOrchestrator:
    """Return the process-wide SurveyOrchestrator singleton."""
    global _instance
    if _instance is None:
        _instance = SurveyOrchestrator()
    return _instance
