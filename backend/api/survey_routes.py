import logging
import os
import time
from datetime import datetime
from typing import Optional

from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=False)

_FAST_MODE = os.getenv("LFS_FAST_MODE", "false").lower() in ("1", "true", "yes")

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session, joinedload
from pydantic import BaseModel, Field

_logger = logging.getLogger("lfs.survey")

_perf_log = logging.getLogger("lfs.perf")

from backend.database.connection import get_db
from backend.database.models import HITLQueue, SurveyResponse, SurveySession, User
from backend.auth.jwt_handler import verify_access_token
from backend.agents.conversation_manager import ConversationContext, ConversationManager, ConversationState
from backend.agents.language_processor import LanguageProcessor, LanguageProcessorResult
from backend.agents.isco_classifier import ISCOClassifier
from backend.agents.hitl_quality_manager import HITLQualityManager
from backend.agents.report_generator import SurveyReport, get_report_generator
from backend.agents.isic_classifier import ISICClassifier
from backend.agents.isced_classifier import ISCEDClassifier
from backend.agents.nationality_classifier import NationalityClassifier, NationalityClassification
from backend.agents.person_register import PersonRegisterService
from backend.rag.vector_store import _ISCO_DATA as _ISCO_KB

# O(1) code → {title_en, title_ar} lookup used by ISCO cache-replay.
_ISCO_TITLE_MAP: dict[str, dict] = {e["code"]: e for e in _ISCO_KB}

router = APIRouter(prefix="/survey", tags=["survey"])
bearer_scheme = HTTPBearer()


# Rate limiting — track requests per IP with Redis
async def _check_rate_limit(request: Request) -> None:
    """Check if client has exceeded 30 requests/minute. Raises HTTPException(429) if so."""
    if not hasattr(request.app, "state") or not hasattr(request.app.state, "limiter"):
        return  # Limiter not configured; skip check
    try:
        limiter = request.app.state.limiter
        limiter.hit("send_message", request)
    except Exception as e:
        if "too many requests" in str(e).lower():
            raise HTTPException(status_code=429, detail="Too many requests. Max 30 per minute.")
        _logger.warning("Rate limit check failed (non-fatal): %s", e)


# ---------------------------------------------------------------------------
# Agent singletons (lazy-initialised on first /message request)
# ---------------------------------------------------------------------------
# Agents are expensive to construct (model loading, API client setup).
# We keep one instance per server process rather than rebuilding per request.
#
# NOTE: ConversationContext lives in _contexts (in-memory), so conversation
# state is lost on server restart.  Replace with Redis-backed storage for
# multi-process or persistent deployments.

_conversation_manager: Optional[ConversationManager] = None
_language_processor:   Optional[LanguageProcessor]   = None
_isco_classifier:      Optional[ISCOClassifier]       = None
_hitl_quality_manager: Optional[HITLQualityManager]  = None
_isic_classifier:         Optional[ISICClassifier]         = None
_isced_classifier:        Optional[ISCEDClassifier]        = None
_nationality_classifier:  Optional[NationalityClassifier]  = None
_person_register_svc:     Optional[PersonRegisterService]  = None

# Secondary agents — wired into live flow for audit, memory, emotional support, validation
_context_memory:          Optional[object] = None   # ContextMemory (lazy)
_audit_logger_agent:      Optional[object] = None   # AuditLogger (lazy)
_emotional_intelligence:  Optional[object] = None   # EmotionalIntelligence (lazy)
_validation_agent:        Optional[object] = None   # ValidationAgent (lazy)

# { session_id: ConversationContext }
_contexts: dict[int, ConversationContext] = {}


def _get_agents() -> tuple[ConversationManager, LanguageProcessor]:
    """Return the conversation and language agents (lightweight — initialises fast)."""
    global _conversation_manager, _language_processor
    if _conversation_manager is None:
        _conversation_manager = ConversationManager()
        _language_processor   = LanguageProcessor()
    return _conversation_manager, _language_processor


def _get_isco_classifier() -> ISCOClassifier:
    """Return the ISCO classifier, initialising it lazily on first call.

    Separated from _get_agents() because ISCOClassifier loads a 1.3 GB
    SentenceTransformer model and populates Qdrant on first use, which
    can take 30–120 s.  We only pay that cost when the first JOB_TITLE
    entity is detected, not on every first /message call.
    """
    global _isco_classifier
    # Retry if store was unavailable at first init (e.g. Qdrant slow to start)
    if _isco_classifier is None or (
        _isco_classifier._hierarchical_store is None and _isco_classifier._flat_store is None
    ):
        _isco_classifier = ISCOClassifier()
    return _isco_classifier


def _get_hitl_quality_manager() -> HITLQualityManager:
    """Return the HITL quality manager singleton, initialising it lazily."""
    global _hitl_quality_manager
    if _hitl_quality_manager is None:
        _hitl_quality_manager = HITLQualityManager()
    return _hitl_quality_manager


def _get_isic_classifier() -> ISICClassifier:
    global _isic_classifier
    if _isic_classifier is None:
        _isic_classifier = ISICClassifier()
    return _isic_classifier


def _get_isced_classifier() -> ISCEDClassifier:
    global _isced_classifier
    if _isced_classifier is None:
        _isced_classifier = ISCEDClassifier()
    return _isced_classifier


def _get_nationality_classifier() -> NationalityClassifier:
    global _nationality_classifier
    if _nationality_classifier is None:
        _nationality_classifier = NationalityClassifier()
    return _nationality_classifier


def _get_person_register_svc() -> PersonRegisterService:
    global _person_register_svc
    if _person_register_svc is None:
        _person_register_svc = PersonRegisterService()
    return _person_register_svc


def _get_context_memory():
    global _context_memory
    if _context_memory is None:
        from backend.agents.context_memory import ContextMemory
        _context_memory = ContextMemory()
    return _context_memory


def _get_audit_logger():
    global _audit_logger_agent
    if _audit_logger_agent is None:
        from backend.agents.audit_logger import AuditLogger
        _audit_logger_agent = AuditLogger()
    return _audit_logger_agent


def _get_emotional_intelligence():
    global _emotional_intelligence
    if _emotional_intelligence is None:
        from backend.agents.emotional_intelligence import EmotionalIntelligence
        _emotional_intelligence = EmotionalIntelligence()
    return _emotional_intelligence


def _get_validation_agent():
    global _validation_agent
    if _validation_agent is None:
        from backend.agents.validation_agent import ValidationAgent
        _validation_agent = ValidationAgent()
    return _validation_agent


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Resolve a Bearer JWT token to the authenticated User."""
    user_id = verify_access_token(credentials.credentials)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = db.query(User).filter(User.id == int(user_id), User.deleted_at.is_(None)).first()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SessionCreateBody(BaseModel):
    language: str


class SessionResponse(BaseModel):
    id: int
    user_id: int
    status: str
    language: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    prefilled_fields: list[str] = []

    class Config:
        from_attributes = True


class ResponseSubmitBody(BaseModel):
    question_id: str
    answer: str
    isco_code: Optional[str] = None
    confidence_score: Optional[float] = None


class SurveyResponseOut(BaseModel):
    id: int
    session_id: int
    question_id: str
    answer: str
    isco_code: Optional[str] = None
    confidence_score: Optional[float] = None

    class Config:
        from_attributes = True


class MessageBody(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    preferred_language: Optional[str] = None


class EntityOut(BaseModel):
    text: str
    label: str
    language: str


class ISCOAlternative(BaseModel):
    code: str
    title_en: str
    title_ar: str
    confidence: float


class ISCOResult(BaseModel):
    job_title: str
    primary_code: str
    primary_title_en: str
    primary_title_ar: str
    confidence: float
    method: str
    stage_confidences: Optional[dict] = None      # {"stage1":0.91,"stage2":0.88,...}
    hierarchy_path: Optional[list] = None          # ["2","25","251","2512"]
    hitl_required: bool = False
    alternatives: list[ISCOAlternative] = []


class SurveyProgress(BaseModel):
    answered: int
    total: int
    pct: int
    current_field: Optional[str] = None


# Short messages that carry no extractable entities — skip NER LLM call.
_NER_SKIP_TOKENS = frozenset({
    # Acknowledgements
    "yes", "no", "ok", "okay", "sure", "yep", "nope", "correct",
    "wrong", "right", "hello", "hi", "hey", "thanks", "thank you",
    "نعم", "لا", "صحيح", "خطأ", "حسنًا", "موافق", "شكرًا",
    # Language toggles
    "english", "arabic", "عربي", "انجليزي",
    # Employment status (C1)
    "employed", "unemployed", "not in labour force", "not in the labour force",
    "موظف", "عاطل", "خارج سوق العمل",
    # Employment nature (C3)
    "paid employee", "self-employed", "employer", "unpaid family worker",
    "contributing family worker",
    # Employment sector (C4)
    "government", "private", "semi-government", "ngo", "international org",
    # Employment type (D6)
    "full-time", "part-time", "seasonal", "casual",
    "full time", "part time",
    # Monthly wage (E1)
    "< 5,000", "5,000–10,000", "10,001–20,000", "20,001–50,000", "> 50,000",
    "prefer not to say", "under 5000", "less than 5000",
    "5000-10000", "10001-20000", "20001-50000", "over 50000",
    # Job search (F1)
    "job_search_active yes", "job_search_active no",
    # Available for work (F3)
    "available_for_work yes", "available_for_work no",
    # Reason left job (G3)
    "laid off", "contract ended", "resigned", "business closed",
    "family reasons", "other", "made redundant",
    # Outside LF reason (F6)
    "housework", "student", "retired", "disabled", "discouraged",
    "homemaker", "illness or disability",
    # Education level (B5)
    "no formal", "primary", "lower secondary", "upper secondary",
    "diploma", "bachelor", "master", "phd", "vocational",
    "bachelor's", "master's", "bachelor degree", "master degree",
    "bachelor's degree", "master's degree", "phd or higher",
    "no formal education",
    # AI preference (K3)
    "ai", "human", "no preference", "prefer ai", "prefer human",
    # Data confidence (K4)
    "very confident", "confident", "somewhat", "not confident",
    "somewhat confident",
    # Gender (B1)
    "male", "female", "prefer not to say",
    "ذكر", "أنثى",
    # Nationality — common demonyms (B3)
    "emirati", "indian", "pakistani", "filipino", "bangladeshi",
    "egyptian", "saudi", "jordanian", "lebanese", "syrian",
    "nepali", "sri lankan", "indonesian", "british", "american",
    "yemeni", "iraqi", "omani", "kuwaiti", "bahraini", "qatari",
    "moroccan", "sudanese", "ethiopian", "kenyan", "nigerian",
    "iranian", "afghan", "chinese", "korean", "japanese",
    "canadian", "australian", "french", "german", "russian",
    "uae national", "uae", "asian", "arab", "western", "african",
    "هندي", "باكستاني", "فلبيني", "إماراتي", "مصري", "بريطاني",
    # Marital status (B4)
    "single", "married", "divorced", "widowed",
    "أعزب", "متزوج", "مطلق", "أرمل",
    # Emirate (B8)
    "abu dhabi", "dubai", "sharjah", "ajman",
    "umm al quwain", "ras al khaimah", "fujairah",
    "أبوظبي", "دبي", "الشارقة", "عجمان", "الفجيرة",
    # UAE residence duration (B9)
    "born in uae", "less than 1 year", "1-4 years", "5-9 years",
    "10-19 years", "20+ years", "1–4 years", "5–9 years", "10–19 years",
    "مولود في الإمارات",
    # Vocational training (B7)
    # "yes"/"no" already covered
    # Contract type (D7)
    "permanent", "fixed-term", "probation period", "no written contract",
    "fixed-term (less than 1 year)", "fixed-term (1-3 years)",
    # Remote work (D8)
    "always", "mostly", "partially", "never",
    # Underemployment (D5)
    "already overemployed", "yes — i want more hours", "yes - i want more hours",
    # Salary allowances (E2)
    "housing", "transport", "food", "schooling", "none",
    # Bonuses (E3)
    "yes — annual bonus", "yes — performance bonus", "yes — other",
    "yes - annual bonus", "yes - performance bonus",
    # Health insurance (E4)
    "full coverage", "partial coverage", "i pay for my own",
    # Pension (E5)
    "not sure",
    # Qualification match (H2)
    "overqualified", "well matched", "underqualified",
    # Skills (H1)
    "digital/it", "management", "finance", "engineering", "healthcare",
    "education", "trades/technical", "hospitality/service", "sales/marketing",
    # Training (H3)
    "yes — employer-funded", "yes — self-funded", "yes — government program",
    "yes - employer-funded", "yes - self-funded",
    # Barriers (H5)
    "language barrier", "lack of experience", "qualification mismatch",
    "salary expectations", "discrimination", "location", "no barriers",
    # Platform work (I1)
    "yes — as my primary income", "yes — as supplementary income",
    "yes - as my primary income", "yes - as supplementary income",
    # Job satisfaction (J1) — numeric scores
    "1", "2", "3", "4", "5",
    "1 — very dissatisfied", "2 — dissatisfied", "3 — neutral",
    "4 — satisfied", "5 — very satisfied",
    "1 - very dissatisfied", "2 - dissatisfied", "3 - neutral",
    "4 - satisfied", "5 - very satisfied",
    # Work safety (J2)
    # "always"/"mostly"/"never" already covered
    "sometimes", "rarely",
    # Workplace issues (J3)
    "harassment", "discrimination", "wage theft", "contract violation",
    # Work-life balance (J4)
    "somewhat", "somewhat yes",
    # Question clarity (K1) — numeric
    # "1"-"5" already covered
    # Difficulty answering (K2)
    "no — all questions were clear", "no - all questions were clear",
    "no, all questions were clear",
    # Secondary job (D3)
    # "yes"/"no" already covered
    # Secondary job status pill text
    "statistics center",
})


class ISICResult(BaseModel):
    industry_text: str
    section: str
    section_title: str
    division_code: str
    division_title: str
    group_code: str = ""
    group_title: str = ""
    class_code: str = ""
    class_title: str = ""
    confidence: float
    method: str


class ISCEDResult(BaseModel):
    education_text: str
    level: int
    level_title: str
    broad_code: str = ""
    broad_title: str = ""
    narrow_code: str = ""
    narrow_title: str = ""
    detailed_code: str = ""
    detailed_title: str = ""
    confidence: float
    method: str


class NationalityResult(BaseModel):
    raw_text: str
    m49_code: str
    iso_alpha3: str
    country_en: str
    country_ar: str
    region_en: str
    nationality_en: str
    nationality_ar: str
    confidence: float
    method: str


class MessageOut(BaseModel):
    reply: str
    state: str
    next_field: Optional[str] = None  # next unanswered field when state is collecting_info
    detected_language: str
    is_code_switched: bool
    entities: list[EntityOut]
    isco_classifications: list[ISCOResult]
    isic_classification: Optional[ISICResult] = None
    isced_classification: Optional[ISCEDResult] = None
    nationality_classification: Optional[NationalityResult] = None
    survey_progress: Optional[SurveyProgress] = None
    session_completed: bool
    semantic_coherence: Optional[dict] = None
    latency_ms: int = 0
    emotional_state: Optional[str] = None            # e.g. "neutral", "frustrated", "confused"
    emotional_support_message: Optional[str] = None  # shown to interviewer when distress detected
    validation_issues: list[str] = []                # rule violations when in VALIDATING state
    is_data_valid: Optional[bool] = None             # None outside VALIDATING state


class MessageResponse(BaseModel):
    message: str


# ---------------------------------------------------------------------------
# Session endpoints
# ---------------------------------------------------------------------------


def _empty_lp_result(text: str, language: str = "en") -> LanguageProcessorResult:
    """Return a no-op LanguageProcessorResult when NER is skipped."""
    from backend.agents.language_processor import CodeSegment
    return LanguageProcessorResult(
        raw_text=text,
        detected_language=language,
        confidence=1.0,
        is_code_switched=False,
        arabic_ratio=0.0,
        latin_ratio=1.0,
        segments=[],
        entities=[],
    )

@router.post(
    "/sessions",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start a new survey session",
    description="Creates a new in-progress survey session for the authenticated user.",
)
def create_session(
    body: SessionCreateBody,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = SurveySession(
        user_id=current_user.id,
        status="in_progress",
        language=body.language,
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    # ── Returning-user pre-fill ──────────────────────────────────────────────
    # Query the most recent COMPLETED session's SurveyResponse rows.
    # This is more reliable than PersonRegister because it's populated on every
    # completion, not just when PersonRegisterService.update_from_session() is called.
    prefilled_fields: list[str] = []
    try:
        prev_session = (
            db.query(SurveySession)
            .options(joinedload(SurveySession.responses))
            .filter(
                SurveySession.user_id == current_user.id,
                SurveySession.status == "completed",
            )
            .order_by(SurveySession.completed_at.desc())
            .first()
        )
        if prev_session and prev_session.responses:
            prev_responses = prev_session.responses
            if prev_responses:
                prefill_data = {r.question_id: r.answer for r in prev_responses}
                prefilled_fields = list(prefill_data.keys())
                ctx = ConversationContext(
                    session_id=session.id,
                    language=body.language,
                    collected_data=prefill_data,
                    is_returning=True,
                )
                _contexts[session.id] = ctx
                _logger.info(
                    "Returning user %d: pre-filled %d fields from session %d",
                    current_user.id, len(prefilled_fields), prev_session.id,
                )
    except Exception as _pf_err:
        _logger.warning("Pre-fill failed (non-fatal): %s", _pf_err)

    # Audit: log session creation
    try:
        from backend.agents.audit_logger import EventType, AccessType
        al = _get_audit_logger()
        al.log_interaction(
            event_type=EventType.SESSION_STARTED,
            description=f"Survey session {session.id} created for user {current_user.id}.",
            session_id=session.id,
            user_id=current_user.id,
        )
        al.log_data_access(
            user_id=current_user.id,
            resource_type="survey_session",
            resource_id=session.id,
            access_type=AccessType.WRITE,
            purpose="Labour Force Survey data collection under GDPR Art. 6(1)(e).",
        )
    except Exception:
        pass

    # Build a response dict manually so we can include prefilled_fields
    # (SQLAlchemy model has no such column — it's computed at request time).
    return {
        "id": session.id,
        "user_id": session.user_id,
        "status": session.status,
        "language": session.language,
        "started_at": session.started_at,
        "completed_at": session.completed_at,
        "prefilled_fields": prefilled_fields,
    }


@router.get(
    "/sessions",
    response_model=list[SessionResponse],
    summary="List all sessions for the current user",
)
def list_sessions(
    skip: int = Query(0, ge=0, description="Number of sessions to skip"),
    limit: int = Query(100, ge=1, le=500, description="Maximum sessions to return"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return (
        db.query(SurveySession)
        .filter(
            SurveySession.user_id == current_user.id,
            SurveySession.deleted_at.is_(None),
        )
        .order_by(SurveySession.started_at.desc(), SurveySession.id.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


@router.get(
    "/sessions/{session_id}",
    response_model=SessionResponse,
    summary="Get a specific session",
)
def get_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = _get_owned_session(db, session_id, current_user.id)
    return session


@router.patch(
    "/sessions/{session_id}/complete",
    response_model=SessionResponse,
    summary="Mark a session as completed",
    description="Sets the session status to 'completed' and records the completion timestamp.",
)
def complete_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = _get_owned_session(db, session_id, current_user.id)

    if session.status == "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Session is already completed.",
        )

    session.status = "completed"
    session.completed_at = datetime.utcnow()
    db.commit()
    db.refresh(session)
    return session


@router.delete(
    "/sessions/{session_id}",
    response_model=MessageResponse,
    summary="Delete a session and all its responses",
)
def delete_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = _get_owned_session(db, session_id, current_user.id)
    # Soft-delete: stamp deleted_at instead of hard-deleting the row.
    # The row is retained for GDPR audit trail purposes.
    session.deleted_at = datetime.utcnow()
    _contexts.pop(session_id, None)
    db.commit()
    return {"message": f"Session {session_id} deleted."}


# ---------------------------------------------------------------------------
# Conversational message endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/sessions/{session_id}/message",
    response_model=MessageOut,
    summary="Send a message and advance the survey conversation",
    description=(
        "Processes one conversational turn:\n"
        "1. **LanguageProcessor** — detects language, identifies code-switching, "
        "extracts LFS-relevant entities (job titles, organisations, locations, …).\n"
        "2. **ConversationManager** — advances the FSM and generates the next "
        "interviewer reply.\n"
        "3. **ISCOClassifier** — for every JOB_TITLE entity found, runs a "
        "two-stage semantic + LLM classification and stores the result.\n\n"
        "When the conversation reaches the *completing* state the session is "
        "automatically marked as completed and all collected survey fields are "
        "persisted."
    ),
)
def send_message(
    session_id: int,
    body: MessageBody,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    _: None = Depends(_check_rate_limit),
):
    try:
        return _send_message_impl(session_id, body, db, current_user)
    except HTTPException:
        raise
    except Exception as _exc:
        import traceback as _tb
        _logger.error("send_message 500 sid=%d: %s\n%s", session_id, _exc, _tb.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {type(_exc).__name__}: {_exc}",
        )


def _send_message_impl(
    session_id: int,
    body: MessageBody,
    db: Session,
    current_user: User,
):
    t_start = time.perf_counter()
    session = _get_owned_session(db, session_id, current_user.id)

    if session.status == "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Session is already completed.",
        )

    conv_mgr, lang_proc = _get_agents()
    ctx = _get_or_create_context(conv_mgr, session_id, session.language)
    msg = body.message.strip()

    # If the user explicitly selected a display language, honour it before NER
    if body.preferred_language and body.preferred_language in (
        "en", "ar", "ar-gulf", "ur", "hi", "tl"
    ):
        ctx.language = body.preferred_language
        session.language = body.preferred_language

    # ── Short-circuit NER for simple acknowledgement tokens, or in FAST_MODE ──
    # In FAST_MODE the conversation reply uses the deterministic stub which
    # does not need entity spans.  The _fields_new ISCO fallback below handles
    # classification from collected_data without requiring NER.
    skip_ner = _FAST_MODE or msg.lower() in _NER_SKIP_TOKENS or len(msg) <= 3

    # Snapshot fields before processing so we can detect which field was just set
    _fields_before = set(ctx.collected_data.keys())

    # Tracks every HITLQueue row created during this single turn (regardless
    # of reason) so a later escalation reason (e.g. SRE incoherence) can
    # append to an existing pending entry for this turn instead of creating a
    # second, redundant queue row -- see Stage 4e below.
    _hitl_entries_this_turn: list["HITLQueue"] = []

    # ── Stages 1 + 3: NER then conversation (sequential) ────────────────────
    # Both agents target the same local Ollama instance which handles one
    # request at a time.  Running them in parallel saturates Ollama and causes
    # both to queue/time-out.  Sequential keeps each call within the timeout.
    t_parallel = time.perf_counter()
    lp_result = _empty_lp_result(msg, ctx.language) if skip_ner else lang_proc.process(msg)
    reply = conv_mgr.process_message(ctx, msg)
    t_parallel_ms = int((time.perf_counter() - t_parallel) * 1000)

    # Fields set for the first time in this turn
    _fields_new = set(ctx.collected_data.keys()) - _fields_before

    # ── Stage 2: update context language (only when no explicit preference set)
    if not body.preferred_language and lp_result.detected_language in (
        "en", "ar", "ar-gulf", "ur", "hi", "tl"
    ):
        ctx.language = lp_result.detected_language
        session.language = lp_result.detected_language

    # ── Stage 4: ISCO classification for every JOB_TITLE entity ─────────────
    job_title_entities = [e for e in lp_result.entities if e.label == "JOB_TITLE"]
    isco_results: list[ISCOResult] = []
    t_isco = time.perf_counter()

    for entity in job_title_entities:
        try:
            # Build context-enriched string from all collected occupational data.
            # This enriches the embedding query (not just the LLM re-ranking prompt)
            # so the hierarchical vector search benefits from duties/industry/sector.
            _ctx_parts = [f"language={lp_result.detected_language}"]
            for _fld in ("job_duties", "industry", "employment_sector", "employment_nature"):
                _val = ctx.collected_data.get(_fld, "")
                if _val:
                    _ctx_parts.append(_val)
            _isco_context = " | ".join(_ctx_parts)

            clf = _get_isco_classifier().classify(
                entity.text,
                context=_isco_context,
            )
            isco_resp = SurveyResponse(
                session_id=session_id,
                question_id="job_title",
                answer=entity.text,
                isco_code=clf.primary.code or None,
                confidence_score=clf.primary.confidence,
            )
            db.add(isco_resp)
            if clf.hitl_required:
                db.flush()  # get isco_resp.id before creating HITL entry
                _hq = HITLQueue(
                    session_id=session_id,
                    response_id=isco_resp.id,
                    raw_text=entity.text,
                    ai_code=clf.primary.code,
                    ai_confidence=clf.primary.confidence,
                    ai_reasoning=getattr(clf, "reasoning", None),
                    hierarchy_path=str(clf.hierarchy_path) if clf.hierarchy_path else None,
                    priority="HIGH" if clf.primary.confidence < 0.50 else "MEDIUM",
                    status="pending",
                    created_at=datetime.utcnow(),
                )
                db.add(_hq)
                _hitl_entries_this_turn.append(_hq)
            try:
                from backend.agents.audit_logger import AgentDecisionType
                _get_audit_logger().log_agent_decision(
                    agent_name="ISCOClassifier",
                    decision_type=AgentDecisionType.ISCO_CLASSIFICATION,
                    input_summary=f"Job title: '{entity.text}'",
                    output_summary=f"ISCO-08 {clf.primary.code} '{clf.primary.title_en}' "
                                   f"(conf {clf.primary.confidence:.2f}, method={clf.method})",
                    confidence=clf.primary.confidence,
                    session_id=session_id,
                )
            except Exception:
                pass
            isco_results.append(ISCOResult(
                job_title=entity.text,
                primary_code=clf.primary.code,
                primary_title_en=clf.primary.title_en,
                primary_title_ar=clf.primary.title_ar,
                confidence=clf.primary.confidence,
                method=clf.method,
                stage_confidences=clf.stage_confidences,
                hierarchy_path=clf.hierarchy_path,
                hitl_required=clf.hitl_required,
                alternatives=[
                    ISCOAlternative(
                        code=a.code,
                        title_en=a.title_en,
                        title_ar=a.title_ar,
                        confidence=a.confidence,
                    ) for a in clf.alternatives
                ],
            ))
        except Exception:
            pass

    # ── Stage 4 fallback: ISCO from stored job_title when NER missed it ────────
    # Handles cases where:
    #   (a) NER was skipped (short/categorical answer)
    #   (b) NER ran but failed to extract a JOB_TITLE entity
    # We run ISCO only on the turn job_title was first stored so it doesn't
    # re-run every subsequent turn.
    if not isco_results and "job_title" in _fields_new:
        _stored_title = ctx.collected_data.get("job_title", "").strip()
        if _stored_title and len(_stored_title) >= 3:
            try:
                _ctx_parts = [f"language={lp_result.detected_language}"]
                for _fld in ("job_duties", "industry", "employment_sector", "employment_nature"):
                    _val = ctx.collected_data.get(_fld, "")
                    if _val:
                        _ctx_parts.append(_val)
                _isco_context = " | ".join(_ctx_parts)
                clf = _get_isco_classifier().classify(_stored_title, context=_isco_context, use_llm=not _FAST_MODE)
                _fb_resp = SurveyResponse(
                    session_id=session_id,
                    question_id="job_title",
                    answer=_stored_title,
                    isco_code=clf.primary.code or None,
                    confidence_score=clf.primary.confidence,
                )
                db.add(_fb_resp)
                if clf.hitl_required:
                    db.flush()
                    _hq = HITLQueue(
                        session_id=session_id,
                        response_id=_fb_resp.id,
                        raw_text=_stored_title,
                        ai_code=clf.primary.code,
                        ai_confidence=clf.primary.confidence,
                        ai_reasoning=getattr(clf, "reasoning", None),
                        hierarchy_path=str(clf.hierarchy_path) if clf.hierarchy_path else None,
                        priority="HIGH" if clf.primary.confidence < 0.50 else "MEDIUM",
                        status="pending",
                        created_at=datetime.utcnow(),
                    )
                    db.add(_hq)
                    _hitl_entries_this_turn.append(_hq)
                isco_results.append(ISCOResult(
                    job_title=_stored_title,
                    primary_code=clf.primary.code,
                    primary_title_en=clf.primary.title_en,
                    primary_title_ar=clf.primary.title_ar,
                    confidence=clf.primary.confidence,
                    method=clf.method,
                    stage_confidences=clf.stage_confidences,
                    hierarchy_path=clf.hierarchy_path,
                    hitl_required=clf.hitl_required,
                    alternatives=[
                        ISCOAlternative(
                            code=a.code,
                            title_en=a.title_en,
                            title_ar=a.title_ar,
                            confidence=a.confidence,
                        ) for a in clf.alternatives
                    ],
                ))
            except Exception as _isco_err:
                _logger.warning("ISCO fallback classify failed for %r: %s", _stored_title, _isco_err)

    # ── Stage 4 cache-replay: surface previous ISCO result on later turns ───────
    # ISCO only computes on the turn job_title is first stored (expensive model).
    # On all subsequent turns (including VALIDATING) isco_results is empty even
    # though the DB has a classification.  Re-surface it so the UI always shows
    # a result once a job title has been classified.
    if not isco_results and ctx.collected_data.get("job_title"):
        _prev_row = (
            db.query(SurveyResponse)
            .filter(
                SurveyResponse.session_id == session_id,
                SurveyResponse.question_id == "job_title",
                SurveyResponse.isco_code.isnot(None),
            )
            .order_by(SurveyResponse.id.desc())
            .first()
        )
        if _prev_row:
            _kb_entry = _ISCO_TITLE_MAP.get(_prev_row.isco_code, {})
            isco_results = [ISCOResult(
                job_title=_prev_row.answer or ctx.collected_data["job_title"],
                primary_code=_prev_row.isco_code,
                primary_title_en=_kb_entry.get("title_en", ""),
                primary_title_ar=_kb_entry.get("title_ar", ""),
                confidence=_prev_row.confidence_score or 0.0,
                method="cached",
                stage_confidences={},
                hierarchy_path=[],
                hitl_required=(_prev_row.confidence_score or 1.0) < 0.70,
                alternatives=[],
            )]

    t_isco_ms = int((time.perf_counter() - t_isco) * 1000)

    # ── Stage 4 re-classify: re-run ISCO with job_duties context once available ─
    # job_duties was just stored this turn and we already have job_title → re-run
    # so the classification benefits from the full duties description.
    if "job_duties" in _fields_new and ctx.collected_data.get("job_title"):
        _stored_title = ctx.collected_data["job_title"].strip()
        try:
            _ctx_parts = [f"language={lp_result.detected_language}"]
            for _fld in ("job_duties", "industry", "employment_sector", "employment_nature"):
                _val = ctx.collected_data.get(_fld, "")
                if _val:
                    _ctx_parts.append(_val)
            _isco_context = " | ".join(_ctx_parts)
            clf = _get_isco_classifier().classify(_stored_title, context=_isco_context, use_llm=not _FAST_MODE)
            # Replace previous classification with the enriched one
            isco_results = [ISCOResult(
                job_title=_stored_title,
                primary_code=clf.primary.code,
                primary_title_en=clf.primary.title_en,
                primary_title_ar=clf.primary.title_ar,
                confidence=clf.primary.confidence,
                method=clf.method,
                stage_confidences=clf.stage_confidences,
                hierarchy_path=clf.hierarchy_path,
                hitl_required=clf.hitl_required,
                alternatives=[
                    ISCOAlternative(
                        code=a.code,
                        title_en=a.title_en,
                        title_ar=a.title_ar,
                        confidence=a.confidence,
                    ) for a in clf.alternatives
                ],
            )]
        except Exception:
            pass

    # ── Stage 4b: ISIC industry classification ───────────────────────────────
    isic_result: Optional[ISICResult] = None
    industry_text = ctx.collected_data.get("industry")
    if industry_text and not isic_result:
        try:
            isic_clf = _get_isic_classifier().classify(industry_text)
            isic_result = ISICResult(
                industry_text=industry_text,
                section=isic_clf.section,
                section_title=isic_clf.section_title,
                division_code=isic_clf.division_code,
                division_title=isic_clf.division_title,
                group_code=isic_clf.group_code,
                group_title=isic_clf.group_title,
                class_code=isic_clf.class_code,
                class_title=isic_clf.class_title,
                confidence=isic_clf.confidence,
                method=isic_clf.method,
            )
        except Exception:
            pass

    # ── Stage 4c: ISCED education classification ─────────────────────────────
    isced_result: Optional[ISCEDResult] = None
    # Combine education_level (ISCED 2011 level) + field_of_study (ISCED-F field)
    # so the classifier can produce both the attainment level AND the specialisation.
    # "bachelor Computer Science" → level 6 + detailed_code "0613"
    # "bachelor" alone → level 6 + field defaults to "0011" generic
    _edu_level = ctx.collected_data.get("education_level")
    _edu_field = ctx.collected_data.get("field_of_study")
    edu_text = (
        " ".join(filter(None, [_edu_level, _edu_field]))
        or next(
            (e.text for e in lp_result.entities if e.label == "EDUCATION"), None
        )
    )
    if edu_text:
        try:
            isced_clf = _get_isced_classifier().classify(edu_text)
            isced_result = ISCEDResult(
                education_text=edu_text,
                level=isced_clf.level,
                level_title=isced_clf.level_title,
                broad_code=isced_clf.broad_code,
                broad_title=isced_clf.broad_title,
                narrow_code=isced_clf.narrow_code,
                narrow_title=isced_clf.narrow_title,
                detailed_code=isced_clf.detailed_code,
                detailed_title=isced_clf.detailed_title,
                confidence=isced_clf.confidence,
                method=isced_clf.method,
            )
        except Exception:
            pass

    # ── Stage 4d: Nationality / UN M49 classification ────────────────────────
    nationality_result: Optional[NationalityResult] = None
    # Use collected nationality field first; fall back to LOCATION NER entities
    nat_text = (
        ctx.collected_data.get("nationality")
        or next(
            (e.text for e in lp_result.entities if e.label == "LOCATION"), None
        )
    )
    if nat_text:
        try:
            nat_clf = _get_nationality_classifier().classify(nat_text)
            if nat_clf.method != "unknown":
                nationality_result = NationalityResult(
                    raw_text=nat_text,
                    m49_code=nat_clf.m49_code,
                    iso_alpha3=nat_clf.iso_alpha3,
                    country_en=nat_clf.country_en,
                    country_ar=nat_clf.country_ar,
                    region_en=nat_clf.region_en,
                    nationality_en=nat_clf.nationality_en,
                    nationality_ar=nat_clf.nationality_ar,
                    confidence=nat_clf.confidence,
                    method=nat_clf.method,
                )
        except Exception:
            pass

    # ── Stage 4e: Semantic cross-classification coherence ───────────────────
    semantic_coherence_out: Optional[dict] = None
    if isco_results and ctx.collected_data:
        try:
            from backend.agents.semantic_relation import get_semantic_relation_engine
            _sr = get_semantic_relation_engine(use_llm=False)
            isced_raw = isced_result.level if isced_result else None
            sc = _sr.analyse(
                isco_code    = isco_results[0].primary_code,
                isic_section = isic_result.section if isic_result else None,
                isced_level  = isced_raw,
                job_title    = str(ctx.collected_data.get("job_title", "")),
                language     = lp_result.detected_language,
            )
            import dataclasses
            # dataclasses.asdict() already recursively converts nested
            # dataclasses (e.g. each SemanticViolation in sc.violations)
            # into plain dicts -- a prior version of this code re-applied
            # asdict() to those already-converted dicts, which raises
            # TypeError("asdict() should be called on dataclass instances")
            # whenever sc.violations was non-empty. Since that raise was
            # caught by the surrounding try/except, semantic_coherence was
            # silently None in the API response for every case that had
            # any violation at all -- exactly the cases where it mattered.
            semantic_coherence_out = dataclasses.asdict(sc)

            # ── Mandatory HITL escalation for HIGH-severity SRE violations ──
            # Added 2026-08-16 (Conference I Reviewer #2 response, Module D
            # Step 5.5) to close the gap Step 5 found: semantic_coherence was
            # computed but never enforced anywhere. This IS the real
            # enforcement point: it's the only live code path that computes
            # semantic_coherence at all -- backend/agents/survey_orchestrator.py
            # also computes it, but that module is never imported by the live
            # API (confirmed by grep across all of backend/), so it has no
            # effect on production traffic regardless of what it does with it.
            _high_violations = [v for v in sc.violations if v.severity == "HIGH"]
            if _high_violations:
                try:
                    _sre_summary = "; ".join(f"[{v.rule_id}] {v.message_en}" for v in _high_violations)
                    _sre_context = (
                        f"SRE HIGH-severity incoherence (score={sc.score}): {_sre_summary} "
                        f"[ISCO={sc.isco_code} ISIC={sc.isic_section} ISCED={sc.isced_level}]"
                    )
                    if _hitl_entries_this_turn:
                        # Fold into the existing pending entry for this turn
                        # rather than creating a second, redundant row for the
                        # same underlying response -- avoids duplicate/
                        # conflicting HITLQueue entries.
                        _existing = _hitl_entries_this_turn[-1]
                        _existing.ai_reasoning = f"{_existing.ai_reasoning or ''}\n{_sre_context}".strip()
                        _existing.priority = "HIGH"
                    else:
                        _hq = HITLQueue(
                            session_id=session_id,
                            response_id=None,
                            raw_text=str(ctx.collected_data.get("job_title", "")),
                            ai_code=sc.isco_code,
                            ai_confidence=sc.score,
                            ai_reasoning=_sre_context,
                            hierarchy_path=None,
                            priority="HIGH",
                            status="pending",
                            created_at=datetime.utcnow(),
                        )
                        db.add(_hq)
                        _hitl_entries_this_turn.append(_hq)
                except Exception as _sre_hitl_exc:
                    _logger.error(
                        "SRE HIGH-severity escalation failed to queue for session=%d: %s",
                        session_id, _sre_hitl_exc,
                    )
        except Exception:
            pass

    # ── Stage 4f: EmotionalIntelligence — detect respondent emotional state ─────
    # Runs only on real messages (not NER-skipped short tokens) to avoid
    # wasting LLM cycles on one-word acknowledgements.
    emotional_state: Optional[str] = None
    emotional_support_message: Optional[str] = None
    if not skip_ner and len(msg) > 10:
        try:
            _ei_result = _get_emotional_intelligence().analyze(msg, ctx.language)
            emotional_state = _ei_result.state
            # Surface a support message to the interviewer when distress is detected
            if _ei_result.state in ("stressed", "frustrated", "confused", "distressed"):
                emotional_support_message = (
                    _ei_result.support_message_en
                    if ctx.language in ("en", "ur", "hi", "tl")
                    else _ei_result.support_message_ar
                )
        except Exception:
            pass

    # ── Stage 4g: ValidationAgent — run rule checks when in VALIDATING state ──
    # The ValidationAgent's rule-based stage 1 (no LLM) runs synchronously.
    # LLM stage 2 is skipped here to keep latency low; it's used in the report.
    validation_issues: list[str] = []
    is_data_valid: Optional[bool] = None
    if ctx.state == ConversationState.VALIDATING and ctx.collected_data:
        try:
            from backend.agents.validation_agent import ValidationAgent as _VA
            _val_result = _get_validation_agent().validate(
                ctx.collected_data,
                language=ctx.language,
            )
            is_data_valid = _val_result.is_valid
            validation_issues = [
                (v.message_en if ctx.language in ("en", "ur", "hi", "tl") else v.message_ar)
                for v in _val_result.rule_violations
            ]
            if not _val_result.is_valid:
                try:
                    from backend.agents.audit_logger import AgentDecisionType
                    _get_audit_logger().log_agent_decision(
                        agent_name="ValidationAgent",
                        decision_type=AgentDecisionType.VALIDATION,
                        input_summary=f"session={session_id}, fields={len(ctx.collected_data)}",
                        output_summary=f"is_valid={_val_result.is_valid}, "
                                       f"violations={len(_val_result.rule_violations)}",
                        confidence=_val_result.confidence,
                        session_id=session_id,
                    )
                except Exception:
                    pass
        except Exception:
            pass

    # ── Stage 5: session completion ──────────────────────────────────────────
    session_completed = ctx.state == ConversationState.COMPLETING
    if session_completed:
        _persist_collected_data(db, session_id, ctx.collected_data)
        session.status = "completed"
        session.completed_at = datetime.utcnow()
        _contexts.pop(session_id, None)

    db.commit()

    # ── Save to PersonRegister for future pre-fill ───────────────────────────
    if session_completed:
        try:
            from calendar import month_abbr
            _now = datetime.utcnow()
            _period = f"{_now.year}-Q{(_now.month - 1) // 3 + 1}"
            _isco = None
            _isic = None
            _isced = None
            if isco_results:
                _isco = isco_results[0].primary_code if hasattr(isco_results[0], "primary_code") else None
            _get_person_register_svc().update_from_session(
                user_id=session.user_id,
                reference_period=_period,
                collected_data=ctx.collected_data,
                isco_code=_isco,
                isic_code=_isic,
                isced_level=_isced,
                db=db,
            )
        except Exception as _pr_err:
            _logger.warning("PersonRegister update failed (non-fatal): %s", _pr_err)

    # ── Stage 6: post-completion ISCO fallback + quality review ─────────────
    # Runs only once per session, after all response rows are committed.
    # _ensure_isco_classification must run before _trigger_quality_review
    # so the quality metrics include the freshly assigned ISCO code.
    if session_completed:
        _ensure_isco_classification(db, session_id, ctx.collected_data)
        _trigger_quality_review(session_id)

    # ── ContextMemory: persist full session state to Redis after every turn ─────
    # Survives server restarts and allows context sharing across multiple workers.
    try:
        _get_context_memory().save_session(
            session_id=session_id,
            state=ctx.state.value,
            language=ctx.language,
            collected_fields=ctx.collected_data,
            history=ctx.history,
        )
        if session_completed:
            _get_context_memory().delete_session(session_id)
    except Exception:
        pass

    # ── AuditLogger: log message event ──────────────────────────────────────
    try:
        from backend.agents.audit_logger import EventType
        _get_audit_logger().log_interaction(
            event_type=EventType.MESSAGE_SENT,
            description=f"Turn in session {session_id}: state={ctx.state.value}",
            session_id=session_id,
            user_id=current_user.id,
        )
    except Exception:
        pass

    total_ms = int((time.perf_counter() - t_start) * 1000)

    _perf_log.info(
        "session=%d ner_skipped=%s parallel_ms=%d isco_ms=%d total_ms=%d "
        "state=%s lang=%s",
        session_id, skip_ner, t_parallel_ms, t_isco_ms, total_ms,
        ctx.state.value, lp_result.detected_language,
    )

    # Determine the next unanswered field so the frontend can show option buttons.
    # Use the same dynamic field order as the ConversationManager so the path
    # (employed / unemployed / outside_lf) is correctly reflected.
    next_field = None
    survey_progress: Optional[SurveyProgress] = None
    _field_order = ConversationManager._get_field_order(ctx.collected_data)
    if ctx.state == ConversationState.COLLECTING_INFO:
        next_field = next((f for f in _field_order if f not in ctx.collected_data), None)
    _answered = [f for f in _field_order if f in ctx.collected_data]
    _total    = max(len(_field_order), 1)
    survey_progress = SurveyProgress(
        answered=len(_answered),
        total=_total,
        pct=round(len(_answered) / _total * 100),
        current_field=next_field,
    )

    return MessageOut(
        reply=reply,
        state=ctx.state.value,
        next_field=next_field,
        detected_language=lp_result.detected_language,
        is_code_switched=lp_result.is_code_switched,
        entities=[
            EntityOut(text=e.text, label=e.label, language=e.language)
            for e in lp_result.entities
        ],
        isco_classifications=isco_results,
        isic_classification=isic_result,
        isced_classification=isced_result,
        nationality_classification=nationality_result,
        survey_progress=survey_progress,
        session_completed=session_completed,
        semantic_coherence=semantic_coherence_out,
        latency_ms=total_ms,
        emotional_state=emotional_state,
        emotional_support_message=emotional_support_message,
        validation_issues=validation_issues,
        is_data_valid=is_data_valid,
    )


# ---------------------------------------------------------------------------
# Report endpoint
# ---------------------------------------------------------------------------

@router.get(
    "/sessions/{session_id}/report",
    response_model=SurveyReport,
    summary="Get the bilingual employment report for a completed session",
    description=(
        "Returns the cached report if one already exists, otherwise generates "
        "a new bilingual (EN + AR) report using the ReportGenerator agent.\n\n"
        "Add `?regenerate=true` to force a fresh LLM run even if a report "
        "already exists."
    ),
)
def get_report(
    session_id: int,
    regenerate: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = _get_owned_session(db, session_id, current_user.id)

    if session.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Report is only available for completed sessions.",
        )

    try:
        report = get_report_generator().generate(
            session_id=session_id,
            regenerate=regenerate,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report generation failed: {exc}",
        )

    return report


# ---------------------------------------------------------------------------
# Raw response endpoints  (kept for manual overrides / testing)
# ---------------------------------------------------------------------------

@router.post(
    "/sessions/{session_id}/responses",
    response_model=SurveyResponseOut,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a response to a survey question",
)
def submit_response(
    session_id: int,
    body: ResponseSubmitBody,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = _get_owned_session(db, session_id, current_user.id)

    if session.status == "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot add responses to a completed session.",
        )

    response = SurveyResponse(
        session_id=session.id,
        question_id=body.question_id,
        answer=body.answer,
        isco_code=body.isco_code,
        confidence_score=body.confidence_score,
    )
    db.add(response)
    db.commit()
    db.refresh(response)
    return response


@router.get(
    "/sessions/{session_id}/responses",
    response_model=list[SurveyResponseOut],
    summary="List all responses for a session",
)
def list_responses(
    session_id: int,
    skip: int = Query(0, ge=0, description="Number of responses to skip"),
    limit: int = Query(200, ge=1, le=1000, description="Maximum responses to return"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_owned_session(db, session_id, current_user.id)
    return (
        db.query(SurveyResponse)
        .filter(
            SurveyResponse.session_id == session_id,
            SurveyResponse.deleted_at.is_(None),
        )
        .offset(skip)
        .limit(limit)
        .all()
    )


@router.patch(
    "/sessions/{session_id}/responses/{response_id}",
    response_model=SurveyResponseOut,
    summary="Update ISCO code or confidence score on a response",
)
def update_response(
    session_id: int,
    response_id: int,
    body: ResponseSubmitBody,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_owned_session(db, session_id, current_user.id)
    response = db.query(SurveyResponse).filter(
        SurveyResponse.id == response_id,
        SurveyResponse.session_id == session_id,
        SurveyResponse.deleted_at.is_(None),
    ).first()

    if not response:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Response not found.",
        )

    response.question_id = body.question_id
    response.answer = body.answer
    response.isco_code = body.isco_code
    response.confidence_score = body.confidence_score
    db.commit()
    db.refresh(response)
    return response


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_owned_session(db: Session, session_id: int, user_id: int) -> SurveySession:
    """Fetch a non-deleted session and enforce ownership. Raises 404 if absent."""
    session = db.query(SurveySession).filter(
        SurveySession.id == session_id,
        SurveySession.user_id == user_id,
        SurveySession.deleted_at.is_(None),
    ).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found.",
        )
    return session


def _get_or_create_context(
    mgr: ConversationManager,
    session_id: int,
    language: str,
) -> ConversationContext:
    """Return the cached ConversationContext for this session.

    Priority: in-process dict → Redis (ContextMemory) → fresh context.
    Redis restore allows conversation state to survive server restarts
    and works across multiple worker processes.
    """
    if session_id not in _contexts:
        try:
            mem = _get_context_memory().load_session(session_id)
            if mem:
                _contexts[session_id] = ConversationContext(
                    session_id=session_id,
                    language=mem.language,
                    state=ConversationState(mem.state),
                    collected_data=dict(mem.collected_fields),
                    history=[{"role": t.role, "content": t.content} for t in mem.history],
                )
            else:
                _contexts[session_id] = mgr.new_context(session_id, language)
        except Exception:
            _contexts[session_id] = mgr.new_context(session_id, language)
    return _contexts[session_id]


def _ensure_isco_classification(
    db: Session,
    session_id: int,
    collected_data: dict,
) -> None:
    """
    Fallback ISCO classification for the FSM-extracted occupation.

    When NER fails to detect a JOB_TITLE entity (e.g. Ollama returned empty
    entities), the job_title SurveyResponse row is written without an isco_code.
    This function detects that case and runs the ISCOClassifier directly on the
    FSM-extracted text so the quality review and final report always show a
    classification.

    Also handles the unemployed path where the occupation field is last_job_title
    (UAE LFS G1) rather than job_title (UAE LFS C5).

    Called only after db.commit() at session completion.
    """
    status = collected_data.get("employment_status", "")

    # For unemployed respondents the relevant occupation is their last job
    if status == "unemployed":
        occ_field = "last_job_title"
        occ_value = collected_data.get("last_job_title", "")
        # If they never worked there's nothing to classify
        if not occ_value or occ_value in ("N/A", "never_worked"):
            return
        isco_query = occ_value
        industry = ""   # unemployed path doesn't collect current industry
    else:
        occ_field = "job_title"
        occ_value = collected_data.get("job_title", "")
        if not occ_value or occ_value == "N/A":
            return
        # Build enriched query: title + duties (UAE LFS C5 + C5a)
        # The duties description significantly improves ISCO semantic matching.
        job_duties = collected_data.get("job_duties", "")
        isco_query = f"{occ_value} {job_duties}".strip() if job_duties and job_duties != "N/A" else occ_value
        industry = collected_data.get("industry", "")

    occ_row = (
        db.query(SurveyResponse)
        .filter(
            SurveyResponse.session_id == session_id,
            SurveyResponse.question_id == occ_field,
        )
        .first()
    )
    if occ_row is None or occ_row.isco_code:
        return  # already classified — nothing to do

    try:
        clf = _get_isco_classifier().classify(
            isco_query,
            context=f"industry={industry}",
        )
        occ_row.isco_code        = clf.primary.code or None
        occ_row.confidence_score = clf.primary.confidence
        db.commit()
    except Exception:
        pass  # classification failure must never break session completion


def _trigger_quality_review(session_id: int) -> None:
    """
    Run the HITL quality review for a newly completed session.

    Persists a QualityReview DB row which ReportGenerator reads when
    producing the final bilingual report (quality_score, quality_status,
    flagged_count fields).  Called after _ensure_isco_classification()
    so the quality metrics see the most up-to-date ISCO codes.

    Silently swallows all errors — a failed quality review must never
    prevent the survey completion response from reaching the frontend.
    """
    try:
        _get_hitl_quality_manager().review_session(session_id)
    except Exception:
        pass


def _persist_collected_data(
    db: Session,
    session_id: int,
    collected_data: dict,
) -> None:
    """
    Save all ConversationManager-collected fields to SurveyResponse rows.

    Fields: employment_status, job_title, industry, hours_per_week,
            employment_type (and any others the FSM extracted).

    job_title is skipped if the ISCO classification loop already wrote a row
    for it (those rows carry the ISCO code and are more complete).  If no
    ISCO row exists yet — because NER missed the entity — we fall back to
    writing the raw FSM-extracted value.
    """
    existing_job_title = (
        db.query(SurveyResponse)
        .filter(
            SurveyResponse.session_id == session_id,
            SurveyResponse.question_id == "job_title",
        )
        .first()
    ) is not None

    for field, value in collected_data.items():
        if not value:
            continue
        # Avoid duplicate: ISCO loop already wrote job_title with code + confidence
        if field == "job_title" and existing_job_title:
            continue
        db.add(SurveyResponse(
            session_id=session_id,
            question_id=field,
            answer=str(value),
        ))


# ---------------------------------------------------------------------------
# HITL occupation review endpoints  (Gap 3 — RQ4)
# ---------------------------------------------------------------------------

class HITLQueueItem(BaseModel):
    id:               int
    session_id:       Optional[int]   = None
    raw_text:         str
    ai_code:          str
    ai_confidence:    float
    ai_reasoning:     Optional[str]   = None
    hierarchy_path:   Optional[str]   = None
    priority:         str             # HIGH | MEDIUM
    status:           str             # pending | reviewed | rejected
    reviewer_code:    Optional[str]   = None
    created_at:       datetime

    class Config:
        from_attributes = True


class HITLReviewBody(BaseModel):
    escalation_id:  int
    action:         str              # "approve" | "correct" | "reject"
    code:           Optional[str]   = None   # required when action == "correct"
    notes:          Optional[str]   = None


class HITLReviewResponse(BaseModel):
    escalation_id: int
    action:        str
    final_code:    Optional[str]
    message:       str


@router.get(
    "/hitl/queue",
    response_model=list[HITLQueueItem],
    summary="List all pending HITL occupation reviews",
    description=(
        "Returns all HITLQueue records where status='pending', ordered by priority "
        "(HIGH first) then created_at. Requires authentication."
    ),
)
def get_hitl_queue(
    status_filter: str = "pending",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    q = db.query(HITLQueue)
    if status_filter != "all":
        q = q.filter(HITLQueue.status == status_filter)
    return (
        q.order_by(
            HITLQueue.priority.desc(),   # HIGH before MEDIUM
            HITLQueue.created_at.asc(),
        )
        .limit(200)
        .all()
    )


@router.post(
    "/hitl/review",
    response_model=HITLReviewResponse,
    summary="Submit a human review decision for a HITL escalation",
    description=(
        "Actions:\n"
        "- **approve**: Accept the AI's suggested ISCO code as correct.\n"
        "- **correct**: Override with a different code (provide `code` field).\n"
        "- **reject**: Mark as unresolvable (no code assigned).\n\n"
        "Updates `HITLQueue.status` and optionally updates the corresponding "
        "`SurveyResponse.isco_code` in the database."
    ),
)
def submit_hitl_review(
    body: HITLReviewBody,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    item = db.query(HITLQueue).filter(HITLQueue.id == body.escalation_id).first()
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"HITL queue item {body.escalation_id} not found.",
        )
    if item.status != "pending":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Item {body.escalation_id} is already {item.status}.",
        )
    if body.action not in ("approve", "correct", "reject"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="action must be 'approve', 'correct', or 'reject'.",
        )
    if body.action == "correct" and not body.code:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="code is required when action is 'correct'.",
        )

    final_code: Optional[str] = None
    if body.action == "approve":
        final_code = item.ai_code
        item.reviewer_code = item.ai_code
    elif body.action == "correct":
        final_code = body.code
        item.reviewer_code = body.code
    # reject: no code

    item.status = "reviewed" if body.action != "reject" else "rejected"
    item.reviewer_notes = body.notes
    item.reviewed_by = current_user.id
    item.reviewed_at = datetime.utcnow()

    # Propagate the final code to the corresponding SurveyResponse row
    if final_code and item.response_id:
        resp = db.query(SurveyResponse).filter(
            SurveyResponse.id == item.response_id,
            SurveyResponse.deleted_at.is_(None),
        ).first()
        if resp:
            resp.isco_code = final_code
            resp.confidence_score = 1.0  # human-reviewed = 100% confidence

    db.commit()

    return HITLReviewResponse(
        escalation_id=body.escalation_id,
        action=body.action,
        final_code=final_code,
        message=(
            f"ISCO code {final_code} confirmed."
            if body.action == "approve"
            else f"ISCO code updated to {final_code}."
            if body.action == "correct"
            else "Escalation rejected — no code assigned."
        ),
    )
