import logging
import os
import time
import copy
import json
import math
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from typing import Annotated, Optional, TYPE_CHECKING
from types import SimpleNamespace

from pathlib import Path
from dotenv import load_dotenv
from backend.agents.occupation_inputs import classify_occupation_input
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=False)

_FAST_MODE = os.getenv("LFS_FAST_MODE", "false").lower() in ("1", "true", "yes")
_CREW_CLASSIFICATION_ENABLED = os.getenv(
    "ENABLE_SURVEY_CLASSIFICATION_CREW", "true"
).lower() in ("1", "true", "yes")

# Real, opt-in cross-standard coordination (2026-09-12) -- default False
# reproduces every existing turn's behaviour exactly. See
# backend/agents/cross_standard_coordinator.py (backward: ISIC/ISCED ->
# revise an uncertain ISCO primary) and ISICClassifier's
# use_cross_classification_hints (forward: ISCO -> bias ISIC's own
# ambiguous-match tie-break) for what this actually does when enabled.
_COORDINATED_CLASSIFICATION = os.getenv(
    "ENABLE_COORDINATED_CLASSIFICATION", "false"
).lower() in ("1", "true", "yes")

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import case
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, joinedload
from pydantic import BaseModel, Field, model_validator
from redis.exceptions import RedisError

_logger = logging.getLogger("lfs.survey")

_perf_log = logging.getLogger("lfs.perf")

from backend.database.connection import get_db
from backend.database.models import AuditLog, HITLQueue, SurveyResponse, SurveySession, User, QualityReview
from backend.database.response_revisions import active_responses, save_response_revision, invalidate_reports, retire_response
from backend.auth.jwt_handler import verify_access_token
from backend.auth.email_otp import check_rate_limit
from backend.agents.conversation_manager import ConversationContext, ConversationManager, ConversationState
from backend.agents.language_processor import LanguageProcessor, LanguageProcessorResult
from backend.agents.isco_classifier import ISCOClassifier
if TYPE_CHECKING:
    from backend.agents.parent_document_isco_classifier import ParentDocumentISCOClassifier
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


def _client_ip(request: Request) -> str:
    """Best-effort client IP extraction (proxy-aware) -- same logic as auth_routes.py's
    own helper of the same name; duplicated rather than imported since it's a tiny,
    dependency-free function and auth_routes.py's copy is module-private."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# Rate limiting — track requests per IP with Redis.
#
# FIXED 2026-09-02: this previously called request.app.state.limiter.hit(...),
# a real, reproduced bug -- slowapi.Limiter (the actually-installed 0.1.9) has
# no `.hit()` method at all (confirmed directly: dir(Limiter) lists exempt/
# get_app_config/limit/limiter/reset/shared_limit/slowapi_startup, no hit).
# Every single call raised AttributeError, which the bare `except Exception`
# caught, logged as "non-fatal", and silently swallowed -- meaning the
# documented "30 requests/minute" cap on this endpoint has never actually
# been enforced, in any deployment, since this code was written. Confirmed
# live: every message in every session log from this entire work session
# logged that exact warning.
#
# Fixed by reusing backend/auth/email_otp.py's check_rate_limit() -- a real,
# independent, already-working Redis-backed rate limiter (with an in-process
# fallback) already proven in production by auth_routes.py's OTP endpoints.
# Simpler and more consistent with the rest of the codebase than getting
# slowapi's lower-level `limits` API (Limiter.limiter.hit(item, *ids)) right,
# and avoids introducing a second rate-limiting mechanism.
async def _check_rate_limit(request: Request) -> None:
    """Check if client has exceeded 30 requests/minute. Raises HTTPException(429) if so."""
    if not check_rate_limit(f"send_message:{_client_ip(request)}", max_requests=30, window_seconds=60):
        raise HTTPException(status_code=429, detail="Too many requests. Max 30 per minute.")


# ---------------------------------------------------------------------------
# Agent singletons (lazy-initialised on first /message request)
# ---------------------------------------------------------------------------
# Agents are expensive to construct (model loading, API client setup).
# We keep one instance per server process rather than rebuilding per request.
#
# Redis holds conversation state; _contexts is a per-process working cache.
# Active database answers provide recovery when the Redis entry expires.

_conversation_manager: Optional[ConversationManager] = None
_language_processor:   Optional[LanguageProcessor]   = None
_isco_classifier: Optional["ISCOClassifier | ParentDocumentISCOClassifier"] = None
_isco_classifier_init_lock = threading.Lock()
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
_session_locks: dict[int, threading.RLock] = {}
_session_locks_guard = threading.Lock()
_session_guard = ContextVar("lfs_session_guard", default=None)


def _check_session_ownership():
    guard = _session_guard.get()
    if guard is not None:
        guard()


def _commit_session(db):
    _check_session_ownership()
    db.commit()


def _committed_outcome_after_lease_failure(session_id, outcome, exc):
    """Keep a result only after this turn has already passed _commit_session.

    session_lock raises LockError while leaving the lock, after the turn body
    has returned. _session_turn reports every RedisError as "not saved".
    That sentence is for a turn that did not commit. The caller must assign
    outcome only from a return that is reached after _commit_session.
    """
    if outcome is None or not isinstance(exc, HTTPException) or exc.status_code != 503:
        return None
    _logger.warning(
        "Committed outcome retained after lease teardown failure for session %s",
        session_id,
    )
    return outcome


@contextmanager
def _session_turn(session_id):
    """Serialize a complete load/process/save operation across workers."""
    with _session_locks_guard:
        local_lock = _session_locks.setdefault(session_id, threading.RLock())
    with local_lock:
        try:
            with _get_context_memory().session_lock(session_id) as guard:
                token = _session_guard.set(guard)
                try:
                    yield
                finally:
                    _session_guard.reset(token)
        except HTTPException:
            raise
        except RedisError as exc:
            _contexts.pop(session_id, None)
            _logger.warning("Session update unavailable for %s: %s", session_id, exc)
            # "Please retry shortly" understated the real wait. The turn is
            # refused rather than half-saved (fail-closed), and the session
            # write-lock is deliberately NOT force-released, because releasing a
            # lock whose owner may still be alive would break the single-writer
            # guarantee that stops one worker overwriting another's turn. If the
            # previous owner died mid-turn the lock can only lapse by its TTL,
            # so the honest worst case is that long.
            from backend.agents.context_memory import SESSION_LOCK_TIMEOUT_SECONDS
            minutes = max(1, SESSION_LOCK_TIMEOUT_SECONDS // 60)
            raise HTTPException(
                status_code=503,
                detail=(
                    "Your last answer was not saved, so nothing has been recorded twice. "
                    "Please try again. If the session stays unavailable it can take up to "
                    f"{minutes} minutes to clear, after which your answers so far are still there."
                ),
            ) from exc


def _get_agents() -> tuple[ConversationManager, LanguageProcessor]:
    """Return the conversation and language agents (lightweight — initialises fast)."""
    global _conversation_manager, _language_processor
    if _conversation_manager is None:
        _conversation_manager = ConversationManager()
        _language_processor   = LanguageProcessor()
    return _conversation_manager, _language_processor


def _get_isco_classifier() -> "ISCOClassifier | ParentDocumentISCOClassifier":
    """Lazily load the configured occupation retriever and its local encoder."""
    with _isco_classifier_init_lock:
        return _initialise_isco_classifier()


def _initialise_isco_classifier():
    """Called under the initialization lock to avoid duplicate model loads."""
    global _isco_classifier
    # Retry if store was unavailable at first init (e.g. Qdrant slow to start)
    if _isco_classifier is None or (not getattr(_isco_classifier, "ready", False) and
        getattr(_isco_classifier, "_hierarchical_store", None) is None
        and getattr(_isco_classifier, "_flat_store", None) is None
    ):
        # Legacy preserves earlier runtime behavior. Parent-document retrieval
        # uses the measured official enriched E5-small index; E5-large cannot
        # be loaded reliably on this local machine. See the dated RAG report.
        strategy = os.getenv("ISCO_RETRIEVAL_STRATEGY", "legacy").strip().lower()
        if strategy == "parent_document":
            from backend.agents.parent_document_isco_classifier import ParentDocumentISCOClassifier
            _isco_classifier = ParentDocumentISCOClassifier()
        elif strategy == "legacy":
            _isco_classifier = ISCOClassifier()
        else:
            raise ValueError("Unknown ISCO_RETRIEVAL_STRATEGY: " + strategy)
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
        _isic_classifier = ISICClassifier(
            use_cross_classification_hints=_COORDINATED_CLASSIFICATION
        )
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


def _audit_best_effort(operation: str, session_id, user_id, record) -> None:
    """Persist an audit action without changing the HTTP result.

    The survey session is committed first. ``AuditLogger._write`` then uses
    its own session, so the session row and these audit rows are not one
    transaction. A session can therefore exist without its audit records.
    An audit failure stays best-effort: it is recorded as the exception type
    only, and the request result is unchanged.
    """
    try:
        record()
    except Exception as exc:
        _logger.warning(
            "audit_persistence_failed operation=%s session_id=%s user_id=%s error_type=%s",
            operation, session_id, user_id, type(exc).__name__,
        )


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


def _require_reviewer(user: User) -> User:
    """Configured reviewers have global scope over active survey sessions."""
    reviewer_ids = {value.strip() for value in os.getenv("HITL_REVIEWER_USER_IDS", "").split(",") if value.strip()}
    if not user.is_active or user.deleted_at is not None or str(user.id) not in reviewer_ids:
        raise HTTPException(status_code=403, detail="Supervisor review access is required.")
    return user


def get_current_reviewer(current_user: User = Depends(get_current_user)) -> User:
    return _require_reviewer(current_user)


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


class ResponseUpdateBody(ResponseSubmitBody):
    class Config:
        extra = "forbid"


class SurveyResponseOut(BaseModel):
    id: int
    session_id: int
    question_id: str
    answer: str
    isco_code: Optional[str] = None
    confidence_score: Optional[float] = None
    retired_fields: list[str] = Field(default_factory=list)
    routing_notice: Optional[str] = None

    class Config:
        from_attributes = True


class MessageBody(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    preferred_language: Optional[str] = None
    # Structured correction picker (2026-09-16) -- see
    # ConversationManager._transition's own docstring for why this exists:
    # free-text correction parsing is inherently ambiguous (a real,
    # live-reported bug -- a respondent naming a field in its natural
    # singular form, "skill", went unrecognised against an alias list that
    # only had "skills" -- is exactly the class of failure this sidesteps).
    # When the frontend's field picker is used instead of typing a
    # correction, it sends the exact field key (from MessageOut's own
    # collected_data keys) and the new value here; `message` is still sent
    # too, as the human-readable text shown in the chat transcript. Both
    # optional and independent of `message` -- omitted, behaviour is
    # byte-for-byte the existing free-text path.
    correction_field: Optional[str] = None
    correction_value: Optional[str] = None
    answer_field: Optional[str] = None
    answer_value: Optional[str] = None

    @model_validator(mode="after")
    def require_answer_pair(self):
        if (self.answer_field is None) != (self.answer_value is None):
            raise ValueError("answer_field and answer_value must be supplied together.")
        return self


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
    agent_execution: dict[str, str] = Field(default_factory=dict)
    classification_execution: Optional[dict] = None
    # Real field keys -> current values (2026-09-16), so the frontend's
    # structured correction picker can list actual fields/values without
    # re-parsing the rendered summary text back into field keys (fragile --
    # label text isn't guaranteed to reverse-map 1:1 onto a field key).
    collected_data: dict[str, str] = {}


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
    db.flush()
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
                SurveySession.deleted_at.is_(None),
            )
            .order_by(SurveySession.completed_at.desc())
            .first()
        )
        if prev_session and prev_session.responses:
            prev_responses = active_responses(db, prev_session.id)
            if prev_responses:
                prefill_data = {r.question_id: r.answer for r in prev_responses}
                prefilled_fields = list(prefill_data.keys())
                ctx = ConversationContext(
                    session_id=session.id,
                    language=body.language,
                    collected_data=prefill_data,
                    is_returning=True,
                    prefilled_fields=prefilled_fields,
                )
                _contexts[session.id] = ctx
                _logger.info(
                    "Returning user %d: pre-filled %d fields from session %d",
                    current_user.id, len(prefilled_fields), prev_session.id,
                )
    except Exception as _pf_err:
        _logger.warning("Pre-fill failed (non-fatal): %s", _pf_err)

    # Publish the initial snapshot before another worker can resume this ID.
    manager, _ = _get_agents()
    context = _contexts.setdefault(session.id, manager.new_context(session.id, body.language))
    ConversationManager.sanitize_context(context)
    prefilled_fields = list(context.prefilled_fields)
    try:
        _save_context(context)
        _commit_session(db)
    except RedisError as exc:
        db.rollback()
        _contexts.pop(session.id, None)
        raise HTTPException(status_code=503, detail="Interview storage is unavailable. Please retry shortly.") from exc

    # Best-effort and separate from the committed session. See _audit_best_effort.
    def _record_session_creation_audit():
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
    _audit_best_effort(
        "session_started", session.id, current_user.id, _record_session_creation_audit,
    )

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
    description="Validates applicable questionnaire requirements before recording completion.",
)
def complete_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    committed = None
    try:
        with _session_turn(session_id):
            committed = _complete_session_impl(session_id, db, current_user)
            return committed
    except SQLAlchemyError as exc:
        db.rollback()
        _contexts.pop(session_id, None)
        _logger.exception("Completion persistence failed for session %s", session_id)
        raise HTTPException(status_code=503, detail="Completion could not be saved. Please retry.") from exc
    except HTTPException as exc:
        retained = _committed_outcome_after_lease_failure(session_id, committed, exc)
        if retained is not None:
            return retained
        db.rollback()
        raise
    except Exception:
        db.rollback()
        raise


def _require_completion(collected_data):
    errors = ConversationManager.completion_errors(collected_data)
    if errors:
        invalid = {error.field for error in errors if error.rule_id not in {"R01", "Q_REQUIRED"}}
        missing = {error.field for error in errors if error.rule_id in {"R01", "Q_REQUIRED"}} - invalid
        raise HTTPException(status_code=422, detail={
            "message": "Survey completion requires all applicable mandatory answers and no blocking validation errors.",
            "missing_fields": sorted(missing),
            "invalid_fields": sorted(invalid),
            "rule_violations": [
                {"rule_id": error.rule_id, "field": error.field, "severity": error.severity,
                 "message_en": error.message_en, "message_ar": error.message_ar}
                for error in errors
            ],
        })


def _reconcile_completion_context(ctx, session):
    """SQL completion is authoritative when recovering a terminal cache entry."""
    if session.status == "completed" or ctx.state != ConversationState.COMPLETING:
        return False
    ctx.state = ConversationState.VALIDATING
    ctx.clarification_target = None
    ctx.clarification_count = 0
    ctx.correction_applied = False
    ctx.corrected_fields.clear()
    ctx.correction_rejected_field = None
    ctx.correction_no_target = False
    ctx.validation_block_messages = []
    return True


def _complete_session_impl(session_id, db, current_user):
    session = _get_owned_session(db, session_id, current_user.id)

    if session.status == "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Session is already completed.",
        )

    # Persisted active revisions are authoritative; cached prefill is not an answer.
    answers = {row.question_id: row.answer for row in active_responses(db, session_id)}
    _require_completion(answers)
    memory = _get_context_memory()
    cached = memory.load_session(session_id)
    history = [{"role": turn.role, "content": turn.content} for turn in cached.history] if cached else []
    _check_session_ownership()
    # Evict stale state before SQL completion. If Redis is unavailable, reject
    # before changing the database. A missing cache can recover from SQL.
    memory.delete_session(session_id)
    _contexts.pop(session_id, None)
    invalidate_reports(db, session_id)
    session.status = "completed"
    session.completed_at = datetime.utcnow()
    _commit_session(db)
    db.refresh(session)
    terminal = ConversationContext(
        session_id=session_id, language=session.language,
        state=ConversationState.COMPLETING, collected_data=answers, history=history,
    )
    try:
        _stamp_response_revisions(terminal, db)
        _save_context(terminal)
    except RedisError:
        # SQL completion is durable and the stale cache was already removed.
        _logger.warning("Completed session %s; terminal cache will recover from SQL", session_id, exc_info=True)
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
    committed = None
    try:
        with _session_turn(session_id):
            committed = _delete_session_impl(session_id, db, current_user)
            return committed
    except HTTPException as exc:
        retained = _committed_outcome_after_lease_failure(session_id, committed, exc)
        if retained is not None:
            return retained
        db.rollback()
        raise


def _delete_session_impl(session_id, db, current_user):
    session = _get_owned_session(db, session_id, current_user.id)
    # Soft-delete: stamp deleted_at instead of hard-deleting the row.
    # The row is retained for GDPR audit trail purposes.
    session.deleted_at = datetime.utcnow()
    _contexts.pop(session_id, None)
    _get_context_memory().delete_session(session_id)
    invalidate_reports(db, session_id)
    _commit_session(db)
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
    committed = None
    try:
        with _session_turn(session_id):
            committed = _send_message_impl(session_id, body, db, current_user)
            return committed
    except HTTPException as exc:
        retained = _committed_outcome_after_lease_failure(session_id, committed, exc)
        if retained is not None:
            return retained
        db.rollback()
        _contexts.pop(session_id, None)
        raise
    except Exception as _exc:
        db.rollback()
        _contexts.pop(session_id, None)
        raise _public_server_error(_exc)


def _public_server_error(exc: BaseException) -> HTTPException:
    """Return a fixed client message. Exception text stays off the response and the log."""
    reference = uuid.uuid4().hex[:12]
    _logger.error("send_message failed reference=%s type=%s", reference, type(exc).__name__)
    return HTTPException(
        status_code=500,
        detail=f"The interview could not be completed. Reference: {reference}",
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
    ctx = _get_or_create_context(conv_mgr, session_id, session.language, db=db)
    if _reconcile_completion_context(ctx, session):
        ConversationManager.sanitize_context(ctx)
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
    _values_before = dict(ctx.collected_data)
    _fields_before = set(_values_before)
    # Keep the unsanitized SQL base separately: existing canonicalization and
    # routing cleanup remain changes, while an unchanged cache is not a write.
    _persistence_base = {row.question_id: row.answer for row in active_responses(db, session_id)}

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
    agent_execution = {"LanguageProcessor": "skipped" if skip_ner else "completed"}
    # The response generated on this turn must use the detected language.
    if not body.preferred_language and lp_result.detected_language in (
        "en", "ar", "ar-gulf", "ur", "hi", "tl"
    ):
        ctx.language = lp_result.detected_language
        session.language = lp_result.detected_language
    _structured_correction = (
        (body.correction_field, body.correction_value)
        if body.correction_field and body.correction_value
        else None
    )
    _answer_options = {"structured_answer": (body.answer_field, body.answer_value)} if body.answer_field is not None else {}
    try:
        reply = conv_mgr.process_message(ctx, msg, structured_correction=_structured_correction, **_answer_options)
        agent_execution["ConversationManager"] = "completed"
    except ValueError as exc:
        from backend.agents.conversation_manager import StructuredAnswerError
        if isinstance(exc, StructuredAnswerError):
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        raise
    if ctx.state == ConversationState.COMPLETING:
        _require_completion(ctx.collected_data)
    t_parallel_ms = int((time.perf_counter() - t_parallel) * 1000)

    # Fields set for the first time in this turn
    _fields_new = {key for key, value in ctx.collected_data.items() if key not in _values_before or _values_before[key] != value}

    # Changed classification inputs share one CrewAI workflow. Original tool
    # results retain their codes, confidence, hierarchy and review requirements.
    classification_execution = {
        "mode": "fast" if _FAST_MODE else "independent",
        "crew_attempted": False, "crew_completed": False,
        "requested_dimensions": [],
    }
    classification_batch = None
    title_input = ""
    industry_input = ""
    education_input = ""
    occupation_field = "last_job_title" if ctx.collected_data.get("employment_status") in (
        "unemployed", "not_in_labour_force"
    ) else "job_title"
    occupation_fields = ({occupation_field, "job_duties", "industry", "employment_sector", "employment_nature"}
                         if occupation_field == "job_title" else {occupation_field})
    # Current-job tasks cannot be evidence about a previous occupation.
    occupation_duties = ctx.collected_data.get("job_duties", "") if occupation_field == "job_title" else ""
    if _CREW_CLASSIFICATION_ENABLED and not _FAST_MODE:
        if _fields_new.intersection(occupation_fields):
            title_input = str(ctx.collected_data.get(occupation_field, "")).strip()
            if len(title_input) < 3 or title_input in ("N/A", "never_worked"):
                title_input = ""
        if "industry" in _fields_new:
            industry_input = str(ctx.collected_data.get("industry", "")).strip()
        if _fields_new.intersection({"education_level", "field_of_study"}):
            education_input = " ".join(filter(None, (
                ctx.collected_data.get("education_level"), ctx.collected_data.get("field_of_study")
            )))
        if title_input or industry_input or education_input:
            from backend.agents.survey_classification_crew import SurveyClassificationCrew
            def lazy_classifier(getter):
                return SimpleNamespace(classify=lambda text, **kwargs: getter().classify(text, **kwargs))
            def lazy_occupation_classifier():
                return SimpleNamespace(classify=lambda text, **kwargs: classify_occupation_input(
                    _get_isco_classifier(), text, duties=occupation_duties, **kwargs))
            context_parts = [f"language={lp_result.detected_language}"] + [
                str(ctx.collected_data[field]) for field in
                ("industry", "employment_sector", "employment_nature")
                if occupation_field == "job_title" and ctx.collected_data.get(field)
            ]
            classification_batch = SurveyClassificationCrew(
                isco_classifier=lazy_occupation_classifier() if title_input else None,
                isic_classifier=lazy_classifier(_get_isic_classifier) if industry_input else None,
                isced_classifier=lazy_classifier(_get_isced_classifier) if education_input else None,
            ).classify(
                job_title=title_input, industry_text=industry_input, education_text=education_input,
                language=ctx.language, isco_context=" | ".join(context_parts), use_llm=True,
                isic_cross_hints="from_isco" if _COORDINATED_CLASSIFICATION and title_input else None,
            )
            classification_execution = classification_batch.execution
            for dimension, name in (("isco", "ISCOClassifier"), ("isic", "ISICClassifier"), ("isced", "ISCEDClassifier")):
                if dimension in classification_execution.get("requested_dimensions", []):
                    agent_execution[name] = "completed" if getattr(classification_batch, dimension) is not None else "failed"
            if classification_execution.get("audit_tool_executed"):
                agent_execution["ClassificationEvidenceAuditor"] = "completed"

    def classify_occupation(text, **kwargs):
        if classification_batch is not None and text == title_input and title_input:
            if classification_batch.isco is None:
                raise RuntimeError("Occupation classification failed in the survey crew")
            return classification_batch.isco
        result = classify_occupation_input(_get_isco_classifier(), text, duties=occupation_duties, **kwargs)
        agent_execution["ISCOClassifier"] = "completed"
        return result

    def classify_industry(text, **kwargs):
        if classification_batch is not None and text == industry_input and industry_input:
            if classification_batch.isic is None:
                raise RuntimeError("Industry classification failed in the survey crew")
            return classification_batch.isic
        result = _get_isic_classifier().classify(text, **kwargs)
        agent_execution["ISICClassifier"] = "completed"
        return result

    def classify_education(text):
        if classification_batch is not None and text == education_input and education_input:
            if classification_batch.isced is None:
                raise RuntimeError("Education classification failed in the survey crew")
            return classification_batch.isced
        result = _get_isced_classifier().classify(text)
        agent_execution["ISCEDClassifier"] = "completed"
        return result

    # ── Stage 4: ISCO classification for every JOB_TITLE entity ─────────────
    # Occupation persistence follows the questionnaire answer, not unrelated
    # job-title mentions extracted from later conversational turns.
    job_title_entities = [e for e in lp_result.entities if e.label == "JOB_TITLE" and "job_title" in _fields_new and e.text == ctx.collected_data.get("job_title")]
    isco_results: list[ISCOResult] = []
    t_isco = time.perf_counter()

    for entity in job_title_entities:
        try:
            # Duties enter the shared query adapter once. Auxiliary context
            # retains the legacy classifier's industry/language contract.
            _ctx_parts = [f"language={lp_result.detected_language}"]
            for _fld in ("industry", "employment_sector", "employment_nature"):
                _val = ctx.collected_data.get(_fld, "")
                if _val:
                    _ctx_parts.append(_val)
            _isco_context = " | ".join(_ctx_parts)

            clf = classify_occupation(
                entity.text,
                context=_isco_context,
            )
            isco_resp = save_response_revision(db, session_id, "job_title", entity.text,
                isco_code=clf.primary.code or None, confidence_score=clf.primary.confidence)
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
    if not isco_results and occupation_field == "job_title" and "job_title" in _fields_new:
        _stored_title = ctx.collected_data.get("job_title", "").strip()
        if _stored_title and len(_stored_title) >= 3 and _stored_title not in ("N/A", "never_worked"):
            try:
                _ctx_parts = [f"language={lp_result.detected_language}"]
                for _fld in ("industry", "employment_sector", "employment_nature"):
                    _val = ctx.collected_data.get(_fld, "")
                    if _val:
                        _ctx_parts.append(_val)
                _isco_context = " | ".join(_ctx_parts)
                clf = classify_occupation(_stored_title, context=_isco_context, use_llm=not _FAST_MODE)
                _fb_resp = save_response_revision(db, session_id, "job_title", _stored_title,
                    isco_code=clf.primary.code or None, confidence_score=clf.primary.confidence)
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

    # The unemployed path codes the last occupation with the same evidence
    # workflow and revision/review rules as a current occupation.
    if occupation_field == "last_job_title" and "last_job_title" in _fields_new:
        previous_title = str(ctx.collected_data.get("last_job_title", "")).strip()
        if previous_title and previous_title not in ("N/A", "never_worked"):
            try:
                clf = classify_occupation(previous_title, context=f"language={ctx.language}", use_llm=not _FAST_MODE)
                response = save_response_revision(db, session_id, "last_job_title", previous_title,
                    isco_code=clf.primary.code or None, confidence_score=clf.primary.confidence)
                if clf.hitl_required:
                    review = HITLQueue(session_id=session_id, response_id=response.id,
                        raw_text=previous_title, ai_code=clf.primary.code, ai_confidence=clf.primary.confidence,
                        ai_reasoning=getattr(clf, "reasoning", None),
                        priority="HIGH" if clf.primary.confidence < 0.50 else "MEDIUM",
                        status="pending", created_at=datetime.utcnow())
                    db.add(review)
                    _hitl_entries_this_turn.append(review)
                isco_results = [ISCOResult(job_title=previous_title,
                    primary_code=clf.primary.code, primary_title_en=clf.primary.title_en,
                    primary_title_ar=clf.primary.title_ar, confidence=clf.primary.confidence,
                    method=clf.method, stage_confidences=clf.stage_confidences,
                    hierarchy_path=clf.hierarchy_path, hitl_required=clf.hitl_required,
                    alternatives=[ISCOAlternative(code=alt.code, title_en=alt.title_en,
                        title_ar=alt.title_ar, confidence=alt.confidence) for alt in clf.alternatives])]
            except Exception as exc:
                _logger.warning("Last occupation classification failed for session=%s: %s", session_id, type(exc).__name__)

    # ── Stage 4 cache-replay: surface previous ISCO result on later turns ───────
    # ISCO only computes on the turn job_title is first stored (expensive model).
    # On all subsequent turns (including VALIDATING) isco_results is empty even
    # though the DB has a classification.  Re-surface it so the UI always shows
    # a result once a job title has been classified.
    if not isco_results and ctx.collected_data.get(occupation_field):
        _prev_row = (
            db.query(SurveyResponse)
            .filter(
                SurveyResponse.session_id == session_id,
                SurveyResponse.question_id == occupation_field,
                SurveyResponse.answer == str(ctx.collected_data[occupation_field]),
                SurveyResponse.deleted_at.is_(None),
                SurveyResponse.isco_code.isnot(None),
            )
            .order_by(SurveyResponse.id.desc())
            .first()
        )
        if _prev_row:
            agent_execution.setdefault("ISCOClassifier", "cached")
            _kb_entry = _ISCO_TITLE_MAP.get(_prev_row.isco_code, {})
            isco_results = [ISCOResult(
                job_title=_prev_row.answer or ctx.collected_data[occupation_field],
                primary_code=_prev_row.isco_code,
                primary_title_en=_kb_entry.get("title_en", ""),
                primary_title_ar=_kb_entry.get("title_ar", ""),
                confidence=_prev_row.confidence_score or 0.0,
                method="cached",
                stage_confidences={},
                hierarchy_path=[],
                hitl_required=(_prev_row.confidence_score if _prev_row.confidence_score is not None else 0.0) < 0.70,
                alternatives=[],
            )]

    t_isco_ms = int((time.perf_counter() - t_isco) * 1000)

    # ── Stage 4 re-classify: re-run ISCO with job_duties context once available ─
    # job_duties was just stored this turn and we already have job_title → re-run
    # so the classification benefits from the full duties description.
    if occupation_field == "job_title" and "job_title" not in _fields_new and _fields_new.intersection({"job_duties", "industry", "employment_sector", "employment_nature"}) and ctx.collected_data.get("job_title") not in (None, "", "N/A", "never_worked"):
        _stored_title = ctx.collected_data["job_title"].strip()
        try:
            _ctx_parts = [f"language={lp_result.detected_language}"]
            for _fld in ("industry", "employment_sector", "employment_nature"):
                _val = ctx.collected_data.get(_fld, "")
                if _val:
                    _ctx_parts.append(_val)
            _isco_context = " | ".join(_ctx_parts)
            clf = classify_occupation(_stored_title, context=_isco_context, use_llm=not _FAST_MODE)
            new_code = clf.primary.code or None
            # Decide before any revision write. save_response_revision replaces a
            # human-confirmed row when only the confidence differs, and this block
            # used to insert a pending review even when that write returned the
            # same row. Duties and industry remain on their own answer fields.
            # This schema has no separate review-evidence history, so an unchanged
            # code does not rewrite the existing pending task.
            active_occupation = (
                db.query(SurveyResponse)
                .filter(
                    SurveyResponse.session_id == session_id,
                    SurveyResponse.question_id == "job_title",
                    SurveyResponse.deleted_at.is_(None),
                )
                .order_by(SurveyResponse.id.desc())
                .first()
            )
            human_confirmed = (
                active_occupation is not None
                and active_occupation.answer == _stored_title
                and active_occupation.isco_code == new_code
                and active_occupation.confidence_score == 1.0
                and bool(new_code)
                and db.query(HITLQueue.id).filter(
                    HITLQueue.response_id == active_occupation.id,
                    HITLQueue.status == "reviewed",
                    HITLQueue.reviewer_code == new_code,
                ).first() is not None
            )
            if not human_confirmed:
                refined_response = save_response_revision(db, session_id, "job_title", _stored_title,
                    isco_code=new_code, confidence_score=clf.primary.confidence)
                if clf.hitl_required and new_code:
                    duplicate_pending = db.query(HITLQueue.id).filter(
                        HITLQueue.session_id == session_id,
                        HITLQueue.response_id == refined_response.id,
                        HITLQueue.ai_code == new_code,
                        HITLQueue.status == "pending",
                    ).first()
                    if duplicate_pending is None:
                        entry = HITLQueue(session_id=session_id, response_id=refined_response.id,
                            raw_text=_stored_title, ai_code=new_code, ai_confidence=clf.primary.confidence,
                            ai_reasoning=getattr(clf, "reasoning", None),
                            hierarchy_path=str(clf.hierarchy_path) if clf.hierarchy_path else None,
                            priority="HIGH" if clf.primary.confidence < 0.50 else "MEDIUM",
                            status="pending", created_at=datetime.utcnow())
                        db.add(entry)
                        _hitl_entries_this_turn.append(entry)
                # Replace the displayed classification only when this turn's
                # result is allowed to become the stored one.
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
            _isic_cross_hints = (
                {"isco_code": isco_results[0].primary_code}
                if _COORDINATED_CLASSIFICATION and isco_results
                else None
            )
            # Switching this to method=ISIC_FLAT_RETRIEVAL was attempted and
            # reverted 2026-10-01, same real memory-exhaustion finding as
            # the ISCO revert above (both load multilingual-e5-large) --
            # see that comment for the full story. The legacy keyword/LLM
            # path stays the live production default; the flat_retrieval+
            # enriched_e5large result (80.71% vs. 12.99% on the synthetic
            # ISIC/ISCED-F benchmark, CLAUDE.md) remains real and evidenced,
            # produced via eval/ scripts offline, just not wired into the
            # live, memory-constrained survey path on this machine.
            isic_clf = classify_industry(industry_text, cross_hints=_isic_cross_hints)
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
            # Same revert and reasoning as ISIC's Stage 4b above --
            # method=ISCEDF_FLAT_RETRIEVAL also loads multilingual-e5-large.
            isced_clf = classify_education(edu_text)
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

            # ── Item 1b: deterministic backward-direction coordination ──
            # (2026-09-12) -- reconsider an already-uncertain ISCO primary
            # using ISIC/ISCED evidence that wasn't available when ISCO
            # committed. Runs BEFORE the SRE analyse() call below so SRE
            # scores the final (possibly revised) code, not the
            # pre-revision one. Off/unset -> byte-identical to before; see
            # backend/agents/cross_standard_coordinator.py's own docstring
            # for the exact, narrow firing conditions.
            if _COORDINATED_CLASSIFICATION:
                try:
                    from backend.agents.cross_standard_coordinator import (
                        maybe_revise_isco_with_cross_signal,
                    )
                    _promoted, _coord_reason = maybe_revise_isco_with_cross_signal(
                        isco_results[0],
                        isic_result.section if isic_result else None,
                        isced_raw,
                    )
                    if _promoted is not None:
                        _old_primary = isco_results[0]
                        isco_results[0] = ISCOResult(
                            job_title=_old_primary.job_title,
                            primary_code=_promoted.code,
                            primary_title_en=_promoted.title_en,
                            primary_title_ar=_promoted.title_ar,
                            confidence=_promoted.confidence,
                            method=f"{_old_primary.method}+cross_revised",
                            stage_confidences=_old_primary.stage_confidences,
                            hierarchy_path=_old_primary.hierarchy_path,
                            hitl_required=True,
                            alternatives=[
                                a for a in _old_primary.alternatives if a.code != _promoted.code
                            ] + [ISCOAlternative(
                                code=_old_primary.primary_code,
                                title_en=_old_primary.primary_title_en,
                                title_ar=_old_primary.primary_title_ar,
                                confidence=_old_primary.confidence,
                            )],
                        )
                        promoted_response = save_response_revision(db, session_id, occupation_field, _old_primary.job_title,
                            isco_code=_promoted.code, confidence_score=_promoted.confidence)
                        _hitl_entries_this_turn = [entry for entry in _hitl_entries_this_turn if entry.status == "pending"]
                        promoted_review = HITLQueue(session_id=session_id, response_id=promoted_response.id,
                            raw_text=_old_primary.job_title, ai_code=_promoted.code, ai_confidence=_promoted.confidence,
                            ai_reasoning=_coord_reason, priority="HIGH" if _promoted.confidence < 0.50 else "MEDIUM",
                            status="pending", created_at=datetime.utcnow())
                        db.add(promoted_review)
                        _hitl_entries_this_turn.append(promoted_review)
                        _logger.info(
                            "cross_standard_coordinator revised ISCO for session=%s: %s",
                            session_id, _coord_reason,
                        )
                except Exception:
                    pass

            sc = _sr.analyse(
                isco_code    = isco_results[0].primary_code,
                isic_section = isic_result.section if isic_result else None,
                isced_level  = isced_raw,
                job_title    = str(ctx.collected_data.get(occupation_field, "")),
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
            agent_execution["SemanticRelationEngine"] = "completed"

            # ── Mandatory HITL escalation for HIGH-severity SRE violations ──
            # Added 2026-08-16 (Conference I Reviewer #2 response, Module D
            # Step 5.5) to close the gap Step 5 found: semantic_coherence was
            # computed but never enforced anywhere. This IS the real
            # enforcement point: it's the only live code path that computes
            # semantic_coherence at all -- an earlier module,
            # backend/agents/survey_orchestrator.py, also computed it, but was
            # confirmed never imported by the live API and was removed from
            # this codebase after that finding was documented.
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
                        occupation = _active_occupation_response(db, session_id)
                        previous = db.query(HITLQueue).filter(
                            HITLQueue.session_id == session_id,
                            HITLQueue.response_id == occupation.id if occupation else HITLQueue.response_id.is_(None),
                            HITLQueue.ai_code == sc.isco_code,
                            HITLQueue.status.in_(("pending", "reviewed")),
                        ).order_by(HITLQueue.id.desc()).first()
                        if previous and previous.status == "pending":
                            previous.priority = "HIGH"
                            if _sre_context not in (previous.ai_reasoning or ""):
                                previous.ai_reasoning = f"{previous.ai_reasoning or ''}\n{_sre_context}".strip()
                        elif previous and previous.status == "reviewed" and f"[ISCO={sc.isco_code} ISIC={sc.isic_section} ISCED={sc.isced_level}]" in (previous.ai_reasoning or ""):
                            pass
                        else:
                            _hq = HITLQueue(
                            session_id=session_id,
                            response_id=occupation.id if occupation else None,
                            raw_text=str(ctx.collected_data.get(occupation_field, "")),
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
            agent_execution["EmotionalIntelligence"] = "completed"
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
            agent_execution["ValidationAgent"] = "completed"
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
        _require_completion(ctx.collected_data)
    retired = _persist_collected_data(db, session_id, ctx.collected_data,
                                     previous_data=_persistence_base, routing_change=ctx.routing_change)
    _record_routing_change(db, session, ctx, retired)
    if session_completed:
        session.status = "completed"
        session.completed_at = datetime.utcnow()
        _contexts.pop(session_id, None)

    _commit_session(db)

    # Later steps cannot undo this commit. A failure here must still return the
    # committed turn. Rollback clears only the unfinished follow-up transaction.
    try:
        if session_completed:
            try:
                from calendar import month_abbr
                _now = datetime.utcnow()
                _period = f"{_now.year}-Q{(_now.month - 1) // 3 + 1}"
                _isco = None
                _isic = isic_result.division_code if isic_result else None
                _isced = isced_result.level if isced_result else None
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

        # Stage 6 runs only after the response rows are committed.
        # Classification before quality review keeps the new code in that review.
        if session_completed:
            _ensure_isco_classification(db, session_id, ctx.collected_data)
            _trigger_quality_review(session_id)

        for occupation_result in isco_results:
            pending_review = db.query(HITLQueue.id).outerjoin(
                SurveyResponse, HITLQueue.response_id == SurveyResponse.id,
            ).filter(
                HITLQueue.session_id == session_id,
                HITLQueue.status == "pending",
                (
                    (SurveyResponse.session_id == session_id)
                    & (SurveyResponse.question_id == occupation_field)
                    & SurveyResponse.deleted_at.is_(None)
                    & (SurveyResponse.answer == occupation_result.job_title)
                ) | (
                    HITLQueue.response_id.is_(None)
                    & (HITLQueue.raw_text == occupation_result.job_title)
                    & (HITLQueue.ai_code == occupation_result.primary_code)
                ),
            ).first()
            occupation_result.hitl_required = pending_review is not None

        _stamp_response_revisions(ctx, db)
        _publish_context_after_commit(session_id, ctx)

        try:
            from backend.agents.audit_logger import EventType
            _get_audit_logger().log_interaction(
                event_type=EventType.MESSAGE_SENT,
                description=f"Turn in session {session_id}: state={ctx.state.value}",
                session_id=session_id,
                user_id=current_user.id,
            )
            agent_execution["AuditLogger"] = "completed"
        except Exception:
            pass

        total_ms = int((time.perf_counter() - t_start) * 1000)
        _perf_log.info(
            "session=%d ner_skipped=%s parallel_ms=%d isco_ms=%d total_ms=%d "
            "state=%s lang=%s",
            session_id, skip_ner, t_parallel_ms, t_isco_ms, total_ms,
            ctx.state.value, lp_result.detected_language,
        )
    except Exception as exc:
        total_ms = int((time.perf_counter() - t_start) * 1000)
        _logger.warning(
            "Session %s answer committed; post-commit step failed: %s",
            session_id, type(exc).__name__,
        )
        db.rollback()
        _contexts.pop(session_id, None)

    # Determine the next unanswered field so the frontend can show option buttons.
    # Use the same dynamic field order as the ConversationManager so the path
    # (employed / unemployed / outside_lf) is correctly reflected.
    next_field = None
    survey_progress: Optional[SurveyProgress] = None
    _field_order = ConversationManager._get_field_order(ctx.collected_data)
    if ctx.state == ConversationState.CLARIFYING:
        next_field = ctx.clarification_target
    elif ctx.state == ConversationState.COLLECTING_INFO:
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
        collected_data={k: str(v) for k, v in ctx.collected_data.items() if v is not None},
        agent_execution=agent_execution,
        classification_execution=classification_execution,
    )


# ---------------------------------------------------------------------------
# Report endpoint
# ---------------------------------------------------------------------------

class LanguageUpdateBody(BaseModel):
    language: str = Field(..., pattern=r"^(en|ar|ar-gulf|ur|hi|tl)$")


def _conversation_snapshot(db, session, *, language=None):
    db.refresh(session)
    manager, _ = _get_agents()
    ctx = _get_or_create_context(manager, session.id, session.language, db=db, sanitize=False)
    if language:
        ctx.language = language
        session.language = language
    if session.status == "completed":
        ctx.state = ConversationState.COMPLETING
    terminal_repaired = _reconcile_completion_context(ctx, session)
    repaired = ConversationManager.sanitize_context(ctx) or terminal_repaired or ctx.answers_reconciled
    if terminal_repaired and ctx.state == ConversationState.VALIDATING:
        use_ar = ctx.language in ("ar", "ar-gulf")
        ctx.validation_block_messages = [
            error.message_ar if use_ar else error.message_en
            for error in ConversationManager.completion_errors(ctx.collected_data)
        ]
    initial = ctx.state == ConversationState.GREETING
    if initial:
        ctx.state = ConversationState.COLLECTING_INFO if any(
            field not in ctx.collected_data for field in manager._get_field_order(ctx.collected_data)
        ) else ConversationState.VALIDATING
    # The renderer may reset one-turn flags/purge stray fields; render a copy.
    rendered = copy.deepcopy(ctx)
    reply = manager._greeting_with_first_question(rendered) if initial and not ctx.collected_data else manager._dev_stub_response(rendered)
    if session.status == "completed" and not language:
        # Direct completion preserves the transcript, including its previous
        # confirmation prompt. Present the terminal reply and record it once.
        if not ctx.history or ctx.history[-1] != {"role": "assistant", "content": reply}:
            ctx.history.append({"role": "assistant", "content": reply})
    elif (language or repaired) and ctx.history and ctx.history[-1]["role"] == "assistant":
        ctx.history[-1] = {"role": "assistant", "content": reply}
    elif not ctx.history or language:
        ctx.history.append({"role": "assistant", "content": reply})
    else:
        reply = next((turn["content"] for turn in reversed(ctx.history) if turn["role"] == "assistant"), reply)
    fields = manager._get_field_order(ctx.collected_data)
    next_field = ctx.clarification_target if ctx.state == ConversationState.CLARIFYING else (
        next((field for field in fields if field not in ctx.collected_data), None)
        if ctx.state == ConversationState.COLLECTING_INFO else None
    )
    answered = sum(field in ctx.collected_data for field in fields)
    payload = MessageOut(
        reply=reply, state=ctx.state.value, next_field=next_field,
        detected_language=ctx.language, is_code_switched=False, entities=[],
        isco_classifications=[], session_completed=session.status == "completed",
        collected_data={key: str(value) for key, value in ctx.collected_data.items()},
        survey_progress=SurveyProgress(answered=answered, total=max(len(fields), 1),
            pct=round(answered / max(len(fields), 1) * 100), current_field=next_field),
    ).model_dump()
    payload.update(history=list(ctx.history), language=ctx.language,
        prefilled_fields=list(ctx.prefilled_fields))
    _save_context(ctx)
    _commit_session(db)
    return payload


@router.get("/sessions/{session_id}/conversation", summary="Restore the current interview without submitting an answer")
def get_conversation(session_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    committed = None
    try:
        with _session_turn(session_id):
            session = _get_owned_session(db, session_id, current_user.id)
            committed = _conversation_snapshot(db, session)
            return committed
    except HTTPException as exc:
        retained = _committed_outcome_after_lease_failure(session_id, committed, exc)
        if retained is not None:
            return retained
        db.rollback()
        raise


@router.patch("/sessions/{session_id}/language", summary="Change interview language without advancing the question")
def update_conversation_language(session_id: int, body: LanguageUpdateBody, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    committed = None
    try:
        with _session_turn(session_id):
            session = _get_owned_session(db, session_id, current_user.id)
            committed = _conversation_snapshot(db, session, language=body.language)
            return committed
    except HTTPException as exc:
        retained = _committed_outcome_after_lease_failure(session_id, committed, exc)
        if retained is not None:
            return retained
        db.rollback()
        raise


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
    report = None
    try:
        with _session_turn(session_id):
            session = _get_owned_session(db, session_id, current_user.id)
            if session.status != "completed":
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Report is only available for completed sessions.",
                )
            # Active answers are the completion gate. A legacy completed row can
            # still hold a blocking contradiction, and a cached report must not
            # be returned ahead of that check. Historical deleted revisions are
            # not part of this answer set. The response is the existing 422
            # completion body. Answers, status, and the stored report row stay
            # unchanged.
            answers = {row.question_id: row.answer for row in active_responses(db, session_id)}
            _require_completion(answers)
            report = get_report_generator().generate(
                session_id=session_id, regenerate=regenerate, write_guard=_check_session_ownership,
            )
    except HTTPException as exc:
        retained = _committed_outcome_after_lease_failure(session_id, report, exc)
        if retained is not None:
            return retained
        raise
    except ValueError as exc:
        # Preserve the existing 404 contract, but never return an internal
        # exception's potentially sensitive text to a respondent.
        _logger.warning("report_lookup_failed type=%s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found.",
        ) from None
    except Exception as exc:
        reference = uuid.uuid4().hex[:12]
        _logger.error("report_generation_failed reference=%s type=%s", reference, type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report generation failed. Reference: {reference}",
        ) from None

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
    committed = None
    try:
        with _session_turn(session_id):
            committed = _submit_response_impl(session_id, body, db, current_user)
            return committed
    except HTTPException as exc:
        retained = _committed_outcome_after_lease_failure(session_id, committed, exc)
        if retained is not None:
            return retained
        raise


def _submit_response_impl(session_id, body, db, current_user):
    session = _get_owned_session(db, session_id, current_user.id)

    if session.status == "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot add responses to a completed session.",
        )

    previous_data = _raw_routing_base(db, session, body.question_id)
    response = save_response_revision(db, session.id, body.question_id, body.answer,
        isco_code=body.isco_code, confidence_score=body.confidence_score)
    ctx = _sync_raw_response(db, session, response, previous_data=previous_data)
    _commit_session(db)
    db.refresh(response)
    if ctx is not None:
        _stamp_response_revisions(ctx, db)
        response.retired_fields = ctx.routing_change.get("retired_fields", [])
        response.routing_notice = ConversationManager.routing_notice(ctx) or None
        _publish_context_after_commit(session_id, ctx)
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
    description=(
        "Completed answers are immutable. Identical-answer requests without metadata "
        "changes are no-ops. Classification changes on completed sessions require "
        "an authorized reviewer and preserve the respondent's answer. Omitted "
        "metadata fields on completed sessions retain their stored values."
    ),
)
def update_response(
    session_id: int,
    response_id: int,
    body: ResponseUpdateBody,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    completed_outcome = None
    try:
        with _session_turn(session_id):
            result = _update_response_impl(session_id, response_id, body, db, current_user)
            if isinstance(result, SurveyResponseOut):
                completed_outcome = result
            return result
    except HTTPException as exc:
        if exc.status_code == 503 and completed_outcome is not None:
            # Only completed no-ops/metadata updates return this captured outcome.
            # A lease-release failure cannot undo their confirmed SQL result;
            # no answer cache publication is needed for unchanged answer text.
            _logger.warning("Completed response outcome retained after lease teardown failure for %s", session_id)
            return completed_outcome
        raise


def _update_response_impl(session_id, response_id, body, db, current_user):
    session = _get_owned_session(db, session_id, current_user.id)
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

    if response.question_id != body.question_id:
        raise HTTPException(status_code=422, detail="A response update must keep its original question_id.")
    if session.status == "completed":
        if body.answer != response.answer:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Completed survey answers cannot be changed through this endpoint.",
            )
        return _update_completed_response_metadata(session, response, body, db, current_user)
    original_id = response.id
    previous_data = _raw_routing_base(db, session, body.question_id)
    response = save_response_revision(db, session_id, body.question_id, body.answer,
        isco_code=body.isco_code, confidence_score=body.confidence_score)
    if response.id != original_id and response.supersedes_id is None:
        response.supersedes_id = original_id
    ctx = _sync_raw_response(db, session, response, previous_data=previous_data)
    _reassess_quality(db, session_id)
    _commit_session(db)
    db.refresh(response)
    if ctx is not None:
        _stamp_response_revisions(ctx, db)
        response.retired_fields = ctx.routing_change.get("retired_fields", [])
        response.routing_notice = ConversationManager.routing_notice(ctx) or None
        _publish_context_after_commit(session_id, ctx)
    return response


def _update_completed_response_metadata(session, response, body, db, current_user):
    """Apply only explicitly supplied classification changes, with an audit trail."""
    metadata = {
        field: getattr(body, field) if field in body.model_fields_set else getattr(response, field)
        for field in ("isco_code", "confidence_score")
    }
    if metadata["confidence_score"] is not None and not math.isfinite(metadata["confidence_score"]):
        raise HTTPException(status_code=422, detail="confidence_score must be finite.")
    changes = {
        field: {"before": getattr(response, field), "after": value}
        for field, value in metadata.items() if getattr(response, field) != value
    }
    if not changes:
        return SurveyResponseOut.model_validate(response)

    _require_reviewer(current_user)
    current = {row.question_id: row for row in active_responses(db, session.id)}
    if current[response.question_id].id != response.id:
        raise HTTPException(status_code=409, detail="Only the current answer revision can be classified.")
    answers = {field: row.answer for field, row in current.items()}
    _require_completion(answers)
    original_id = response.id
    try:
        # Copy respondent fields from SQL, never from a metadata-update payload.
        revised = save_response_revision(db, session.id, response.question_id, response.answer, **metadata)
        if {row.question_id: row.answer for row in active_responses(db, session.id)} != answers:
            raise HTTPException(status_code=409, detail="Classification updates must preserve all completed answers.")
        db.add(AuditLog(
            session_id=session.id, user_id=current_user.id,
            event_type="classification_metadata_updated", actor="user",
            description="Reviewer updated classification metadata on an unchanged completed answer.",
            metadata_json=json.dumps({"previous_response_id": original_id,
                "response_id": revised.id, "changes": changes}),
            timestamp=datetime.utcnow(),
        ))
        _reassess_quality(db, session.id, reviewer=current_user.id,
            notes="Classification metadata updated on an unchanged completed answer.")
        outcome = SurveyResponseOut.model_validate(revised)
        _commit_session(db)
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail="Classification metadata update could not be saved.") from exc
    except (RedisError, HTTPException):
        db.rollback()
        raise
    return outcome


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_owned_session(db: Session, session_id: int, user_id: int) -> SurveySession:
    """Fetch a non-deleted session and enforce ownership. Raises 404 if absent."""
    session = db.query(SurveySession).filter(
        SurveySession.id == session_id,
        SurveySession.user_id == user_id,
        SurveySession.deleted_at.is_(None),
    ).populate_existing().first()
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
    *,
    db: Optional[Session] = None,
    sanitize: bool = True,
) -> ConversationContext:
    """Refresh shared state on every turn, while holding the session lock."""
    restored = False
    try:
        mem = _get_context_memory().load_session(session_id)
        if mem:
            _contexts[session_id] = ConversationContext(
                session_id=session_id, language=mem.language,
                state=ConversationState(mem.state), collected_data=dict(mem.collected_fields),
                history=[{"role": t.role, "content": t.content} for t in mem.history],
                clarification_target=mem.clarification_target,
                clarification_count=mem.clarification_count,
                is_returning=mem.is_returning,
                prefilled_fields=mem.prefilled_fields,
                correction_applied=mem.correction_applied,
                corrected_fields=set(mem.corrected_fields),
                correction_rejected_field=mem.correction_rejected_field,
                correction_no_target=mem.correction_no_target,
                response_revisions=mem.response_revisions,
            )
            restored = True
    except RedisError:
        raise
    except (TypeError, ValueError, AttributeError):
        # Recover malformed or expired cache entries from persisted answers.
        pass
    if not restored and db is not None:
        context = mgr.new_context(session_id, language)
        context.collected_data = {row.question_id: row.answer for row in active_responses(db, session_id)}
        if context.collected_data:
            context.state = ConversationState.COLLECTING_INFO if any(
                field not in context.collected_data for field in mgr._get_field_order(context.collected_data)
            ) else ConversationState.VALIDATING
        _contexts[session_id] = context
    elif session_id not in _contexts:
        _contexts[session_id] = mgr.new_context(session_id, language)
    if db is not None:
        _reconcile_answer_revisions(db, mgr, _contexts[session_id])
    if sanitize:
        ConversationManager.sanitize_context(_contexts[session_id])
    return _contexts[session_id]


def _reconcile_answer_revisions(db, mgr, ctx):
    """Restore committed answers, retaining only explicitly pending prefill.

    Redis owns dialogue metadata, not persisted response revisions. An accepted
    current turn is applied to this base under the session lease. Failed SQL
    writes therefore cannot become answers on the next turn, and failed cache
    publication cannot roll back a committed correction. Historical rows also
    distinguish pending prefill from a deliberately retired answer.
    """
    rows = active_responses(db, ctx.session_id)
    revisions = {row.question_id: row.id for row in rows}
    answers = {row.question_id: row.answer for row in rows}
    if ctx.is_returning and ctx.prefilled_fields:
        recorded = {field for (field,) in db.query(SurveyResponse.question_id).filter(
            SurveyResponse.session_id == ctx.session_id,
        ).distinct().all()}
        # A routing decision can also retire provisional prefill before its
        # first response row exists. Its committed audit prevents an older
        # cache from restoring that prefill after publication failure.
        for (metadata,) in db.query(AuditLog.metadata_json).filter(
            AuditLog.session_id == ctx.session_id,
            AuditLog.event_type == "questionnaire_routing_changed",
        ).all():
            try:
                recorded.update(json.loads(metadata)["retired_prefill_fields"])
            except (TypeError, ValueError, KeyError):
                # Unknown provenance cannot authorize provisional answers.
                recorded.update(ctx.prefilled_fields)
        for field in ctx.prefilled_fields:
            if field not in recorded and field in ctx.collected_data:
                answers[field] = ctx.collected_data[field]
    changed = ctx.collected_data != answers
    stale = ctx.response_revisions is not None and ctx.response_revisions != revisions
    ctx.collected_data = answers
    ctx.response_revisions = revisions
    ctx.answers_reconciled = changed or stale
    if changed:
        ctx.correction_applied = False
        ctx.corrected_fields.clear()
        ctx.correction_rejected_field = None
        ctx.correction_no_target = False
        ctx.validation_block_messages = []
        ctx.clarification_target = None
        ctx.clarification_count = 0
        if ctx.state != ConversationState.GREETING:
            ctx.state = (ConversationState.VALIDATING
                         if mgr._get_required_fields(answers).issubset(answers)
                         else ConversationState.COLLECTING_INFO)


def _stamp_response_revisions(ctx, db):
    """Call after commit; never attest uncommitted response IDs in Redis."""
    ctx.response_revisions = {row.question_id: row.id for row in active_responses(db, ctx.session_id)}


def _publish_context_after_commit(session_id, ctx):
    """Publish context after SQL commit. A cache miss recovers from active rows."""
    try:
        _save_context(ctx)
    except RedisError:
        _logger.warning(
            "Session %s answer committed; context cache will recover from SQL",
            session_id,
        )
        _contexts.pop(session_id, None)


def _save_context(ctx):
    _check_session_ownership()
    return _get_context_memory().save_session(
        session_id=ctx.session_id, state=ctx.state.value, language=ctx.language,
        collected_fields=ctx.collected_data, history=ctx.history,
        clarification_target=ctx.clarification_target, clarification_count=ctx.clarification_count,
        is_returning=ctx.is_returning, correction_applied=ctx.correction_applied,
        corrected_fields=ctx.corrected_fields, correction_rejected_field=ctx.correction_rejected_field,
        correction_no_target=ctx.correction_no_target,
        prefilled_fields=ctx.prefilled_fields,
        response_revisions=ctx.response_revisions,
    )


def _active_occupation_response(db, session_id):
    return db.query(SurveyResponse).filter(
        SurveyResponse.session_id == session_id,
        SurveyResponse.question_id.in_(("job_title", "last_job_title")),
        SurveyResponse.deleted_at.is_(None),
    ).order_by(SurveyResponse.id.desc()).first()


def _raw_routing_base(db, session, question_id):
    if question_id in ConversationManager._EXACT_QUESTIONS_EN:
        manager, _ = _get_agents()
        ctx = _get_or_create_context(manager, session.id, session.language, db=db, sanitize=False)
        return dict(ctx.collected_data)
    return {row.question_id: row.answer for row in active_responses(db, session.id)}


def _sync_raw_response(db, session, response, *, previous_data=None):
    ctx = None
    if response.question_id in ConversationManager._EXACT_QUESTIONS_EN:
        manager, _ = _get_agents()
        ctx = _get_or_create_context(manager, session.id, session.language, db=db)
        ctx.collected_data[response.question_id] = response.answer
        ConversationManager.sanitize_context(ctx)
        if previous_data is not None:
            manager.apply_routing_change(ctx, previous_data, explicit_fields={response.question_id})
            retired = {}
            for row in active_responses(db, session.id):
                if row.question_id in ctx.routing_change.get("retired_fields", []):
                    retire_response(db, row, reason=_routing_retirement_reason(ctx.routing_change))
                    retired[row.question_id] = row.id
            if retired:
                invalidate_reports(db, session.id)
            if ctx.routing_change:
                _record_routing_change(db, session, ctx, retired)
                db.flush()
                ctx.history.append({"role": "assistant", "content": manager.routing_notice(ctx)})
    if response.question_id in ("job_title", "last_job_title") and response.isco_code and (response.confidence_score or 0) < 0.70:
        pending = db.query(HITLQueue).filter(HITLQueue.response_id == response.id, HITLQueue.status == "pending").first()
        if pending is None:
            db.add(HITLQueue(session_id=session.id, response_id=response.id, raw_text=response.answer,
                ai_code=response.isco_code, ai_confidence=response.confidence_score or 0,
                ai_reasoning="Low-confidence saved occupation classification.",
                priority="HIGH" if (response.confidence_score or 0) < 0.5 else "MEDIUM",
                status="pending", created_at=datetime.utcnow()))
    return ctx


def _reassess_quality(db, session_id, reviewer=None, notes=None):
    flagged, metrics, score, review_status = HITLQualityManager.assess_responses(session_id, active_responses(db, session_id))
    db.add(QualityReview(session_id=session_id, quality_score=score,
        passed=review_status.value == "pass", flagged_count=len(flagged),
        flagged_items_json=json.dumps([flag.model_dump() for flag in flagged]),
        escalated=review_status.value == "escalated",
        escalation_reason=HITLQualityManager._escalation_reason(score, flagged) if review_status.value == "escalated" else None,
        reviewed_by=reviewer, reviewer_notes=notes, created_at=datetime.utcnow(),
        reviewed_at=datetime.utcnow() if reviewer is not None else None))


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
    if status in ("unemployed", "not_in_labour_force"):
        occ_field = "last_job_title"
        occ_value = collected_data.get("last_job_title", "")
        # If they never worked there's nothing to classify
        if not occ_value or occ_value in ("N/A", "never_worked"):
            return
        job_duties = ""
        industry = ""   # unemployed path doesn't collect current industry
    else:
        occ_field = "job_title"
        occ_value = collected_data.get("job_title", "")
        if not occ_value or occ_value == "N/A":
            return
        job_duties = collected_data.get("job_duties", "")
        industry = collected_data.get("industry", "")

    occ_row = (
        db.query(SurveyResponse)
        .filter(
            SurveyResponse.session_id == session_id,
            SurveyResponse.question_id == occ_field,
            SurveyResponse.deleted_at.is_(None),
        )
        .order_by(SurveyResponse.id.desc()).first()
    )
    if occ_row is None or occ_row.isco_code:
        return  # already classified — nothing to do

    try:
        clf = classify_occupation_input(
            _get_isco_classifier(), occ_value, duties=job_duties,
            context=f"industry={industry}",
            use_llm=not _FAST_MODE,
        )
        occ_row.isco_code        = clf.primary.code or None
        occ_row.confidence_score = clf.primary.confidence
        if clf.hitl_required and clf.primary.code:
            pending = db.query(HITLQueue).filter(
                HITLQueue.session_id == session_id,
                HITLQueue.response_id == occ_row.id,
                HITLQueue.status == "pending",
            ).first()
            if pending is None:
                db.add(HITLQueue(session_id=session_id, response_id=occ_row.id,
                    raw_text=occ_value, ai_code=clf.primary.code, ai_confidence=clf.primary.confidence,
                    ai_reasoning=getattr(clf, "reasoning", None),
                    hierarchy_path=str(clf.hierarchy_path) if clf.hierarchy_path else None,
                    priority="HIGH" if clf.primary.confidence < 0.50 else "MEDIUM",
                    status="pending", created_at=datetime.utcnow()))
        _commit_session(db)
    except RedisError:
        raise
    except Exception:
        pass  # classification failure must never break session completion


def _trigger_quality_review(session_id: int) -> None:
    """
    Run the HITL quality review for a newly completed session.

    Persists a QualityReview DB row which ReportGenerator reads when
    producing the final bilingual report (quality_score, quality_status,
    flagged_count fields).  Called after _ensure_isco_classification()
    so the quality metrics see the most up-to-date ISCO codes.

    Model failures leave completion intact; storage/ownership failures propagate.
    """
    try:
        _get_hitl_quality_manager().review_session(session_id, write_guard=_check_session_ownership)
    except RedisError:
        raise
    except Exception:
        pass


def _persist_collected_data(
    db: Session,
    session_id: int,
    collected_data: dict,
    *,
    previous_data: Optional[dict] = None,
    routing_change: Optional[dict] = None,
) -> dict[str, int]:
    """
    Save all ConversationManager-collected fields to SurveyResponse rows.

    Fields: employment_status, job_title, industry, hours_per_week,
            employment_type (and any others the FSM extracted).

    Unchanged active answers retain their classification. Changed answers
    create revisions; fields removed by questionnaire routing are retired.
    """
    existing = {row.question_id: row for row in active_responses(db, session_id)}
    retired = {}
    for field, value in collected_data.items():
        if previous_data is not None and field in existing and previous_data.get(field) == value:
            # Restoring a snapshot is not a respondent correction. In particular,
            # preserve a newer revision written elsewhere during this request.
            continue
        if not value:
            continue
        row = existing.get(field)
        if row and row.answer == str(value):
            continue
        save_response_revision(db, session_id, field, value)
    # A changed routing answer can remove fields from the active path.
    for field, row in existing.items():
        if field not in collected_data and (previous_data is None or field in previous_data):
            reason = (_routing_retirement_reason(routing_change)
                      if routing_change and field in routing_change["retired_fields"] else None)
            retire_response(db, row, reason=reason)
            retired[field] = row.id
            invalidate_reports(db, session_id)
    return retired


def _routing_retirement_reason(change):
    return "Answer no longer applicable after questionnaire routing change: " + ", ".join(sorted(change["changes"]))


def _record_routing_change(db, session, ctx, retired):
    """Commit the reason and historical row IDs atomically with retirements."""
    retired = {field: row_id for field, row_id in retired.items()
               if field in ctx.routing_change.get("retired_fields", [])}
    if ctx.routing_change:
        db.add(AuditLog(
            session_id=session.id, user_id=session.user_id,
            event_type="questionnaire_routing_changed", actor="agent:ConversationManager",
            description=ConversationManager.routing_notice(ctx),
            metadata_json=json.dumps({**ctx.routing_change, "retired_response_ids": retired,
                                      "retired_prefill_fields": sorted(set(ctx.routing_change["retired_fields"]) - set(retired))},
                                     ensure_ascii=False),
            timestamp=datetime.utcnow(),
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


# Deterministic review-queue ordering, shared by the endpoint and its tests.
#
# `priority` is a plain String column, so the previous `priority.desc()` sorted
# it ALPHABETICALLY: verified directly against this project's own Postgres,
# `ORDER BY priority DESC` returns MEDIUM, LOW, HIGH -- the exact opposite of
# what that line's own comment ("HIGH before MEDIUM") and this endpoint's
# docstring claimed. It was latent only because every row in this environment
# happens to be HIGH; one MEDIUM row would have sorted ahead of every HIGH one.
# An explicit rank fixes the intent, and `id` last makes the total order a
# strict one so `limit`/`offset` paging can never duplicate or skip a row when
# two records share a priority and a timestamp.
_HITL_PRIORITY_RANK = case(
    {"HIGH": 0, "MEDIUM": 1, "LOW": 2},
    value=HITLQueue.priority,
    else_=3,
)

_HITL_QUEUE_ORDER = (
    _HITL_PRIORITY_RANK.asc(),
    HITLQueue.created_at.asc(),
    HITLQueue.id.asc(),
)


@router.get(
    "/hitl/queue",
    response_model=list[HITLQueueItem],
    summary="List pending HITL occupation reviews (paginated)",
    description=(
        "Returns HITLQueue records ordered HIGH -> MEDIUM -> LOW, then oldest first, "
        "then by id. `status_filter` defaults to 'pending'; pass 'all' for every status. "
        "Paginate with `limit` (default 200, max 500) and `offset`. The total number of "
        "matching records is returned in the `X-Total-Count` header, so a reviewer can "
        "page through every eligible record. Requires reviewer authentication."
    ),
)
def get_hitl_queue(
    # Defaulted to None deliberately. FastAPI injects the live Response by type
    # annotation regardless of the default, so the pagination headers below are
    # still set over HTTP -- but several existing tests call this function
    # DIRECTLY as plain Python (test_review_persistence_fixes.py), and a
    # parameter with no default turns those into a TypeError. Keeping the
    # default preserves that calling convention.
    response: Response = None,
    status_filter: str = "pending",
    # Annotated, NOT `limit: int = Query(200, ...)`. With the Query object as
    # the default, a direct Python call receives the Query object itself rather
    # than an int, and `.limit(Query(200))` raises deep inside SQLAlchemy
    # ("int() argument must be ... not 'Query'"). Several existing tests call
    # this function directly, so the default has to be a real int; Annotated
    # keeps the validation and the OpenAPI constraints without occupying it.
    limit: Annotated[int, Query(ge=1, le=500, description="Rows per page (default 200, max 500).")] = 200,
    offset: Annotated[int, Query(ge=0, description="Rows to skip; use with limit to page.")] = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_reviewer),
):
    _require_reviewer(current_user)
    q = db.query(HITLQueue).join(SurveySession, HITLQueue.session_id == SurveySession.id).join(User, SurveySession.user_id == User.id).filter(
        SurveySession.deleted_at.is_(None), User.deleted_at.is_(None), User.is_active.is_(True),
    )
    if status_filter != "all":
        q = q.filter(HITLQueue.status == status_filter)

    # Report the full matching count before the page is cut, so the caller can
    # tell "this is the last page" from "there is more behind the limit" -- the
    # distinction the previous hard .limit(200) made impossible. Skipped on a
    # direct Python call, which has no response object to carry headers.
    if response is not None:
        response.headers["X-Total-Count"] = str(q.order_by(None).count())
        response.headers["X-Limit"] = str(limit)
        response.headers["X-Offset"] = str(offset)

    return (
        q.order_by(*_HITL_QUEUE_ORDER)
        .limit(limit)
        .offset(offset)
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
    current_user: User = Depends(get_current_reviewer),
):
    _require_reviewer(current_user)
    item = db.query(HITLQueue).filter(HITLQueue.id == body.escalation_id).first()
    if not item or not item.session_id:
        raise HTTPException(status_code=404, detail="Review item not found.")
    review_session_id = item.session_id
    committed = None
    try:
        with _session_turn(review_session_id):
            db.expire_all()
            committed = _submit_hitl_review_impl(body, db, current_user, expected_session_id=review_session_id)
            return committed
    except HTTPException as exc:
        retained = _committed_outcome_after_lease_failure(review_session_id, committed, exc)
        if retained is not None:
            return retained
        raise


def _hitl_record_values(row):
    return {column.name: getattr(row, column.name) for column in row.__table__.columns}


def _hitl_locked_rows(db, model, session_id):
    return db.query(model).filter(model.session_id == session_id).order_by(
        model.id.asc(),
    ).populate_existing().with_for_update().all()


def _hitl_review_target(item, responses):
    """Resolve one exact current occupation decision; never choose duplicates."""
    active = [row for row in responses if row.deleted_at is None]
    if item.response_id is not None:
        candidates = [row for row in active if row.id == item.response_id]
    else:
        # Legacy unlinked items remain supported only with unambiguous evidence.
        candidates = [row for row in active
            if row.question_id in ("job_title", "last_job_title")
            and row.answer == item.raw_text and (row.isco_code or "") == item.ai_code]
    if len(candidates) != 1:
        raise HTTPException(status_code=409, detail="Review target is stale or ambiguous; refresh the review queue.")
    response = candidates[0]
    if (response.question_id not in ("job_title", "last_job_title")
            or sum(row.question_id == response.question_id for row in active) != 1
            or response.answer != item.raw_text or (response.isco_code or "") != item.ai_code):
        raise HTTPException(status_code=409, detail="Review target conflicts with the current answer or classification.")
    return response


def _submit_hitl_review_impl(body, db, current_user, *, expected_session_id=None):
    _require_reviewer(current_user)
    try:
        return _apply_hitl_review(body, db, current_user, expected_session_id)
    except HTTPException as exc:
        db.rollback()
        if exc.status_code == 409:
            _logger.warning("HITL review conflict for item %s: %s", body.escalation_id, exc.detail)
        raise
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail="HITL review could not be saved.") from exc
    except RedisError:
        db.rollback()
        raise


def _apply_hitl_review(body, db, current_user, expected_session_id):
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
    if body.action == "correct" and (len(body.code) != 4 or not body.code.isascii() or not body.code.isdigit()):
        raise HTTPException(status_code=422, detail="code must contain four ISCO-08 digits.")

    session_id = expected_session_id
    if session_id is None:
        session_id = db.query(HITLQueue.session_id).filter(HITLQueue.id == body.escalation_id).scalar()
    session = db.query(SurveySession).join(User, SurveySession.user_id == User.id).filter(
        SurveySession.id == session_id, SurveySession.deleted_at.is_(None),
        User.deleted_at.is_(None), User.is_active.is_(True),
    ).populate_existing().with_for_update(of=(SurveySession, User)).first()
    if session is None:
        raise HTTPException(status_code=404, detail=f"HITL queue item {body.escalation_id} not found.")

    # Hold SQL locks through commit where supported, in addition to the public
    # session lease. SQLite ignores FOR UPDATE; its tests exercise the lease.
    queues = _hitl_locked_rows(db, HITLQueue, session.id)
    item = next((row for row in queues if row.id == body.escalation_id), None)
    if item is None:
        raise HTTPException(status_code=409, detail="Review session identity changed; refresh the review queue.")
    if item.status != "pending":
        raise HTTPException(status_code=409, detail=f"Item {body.escalation_id} is already {item.status}.")
    responses = _hitl_locked_rows(db, SurveyResponse, session.id)
    resp = _hitl_review_target(item, responses)
    before_responses = {row.id: _hitl_record_values(row) for row in responses}
    before_queues = {row.id: _hitl_record_values(row) for row in queues}
    before_session = _hitl_record_values(session)
    original_id, original_code = resp.id, resp.isco_code

    final_code: Optional[str] = None
    if body.action == "approve":
        final_code = item.ai_code or None
    elif body.action == "correct":
        final_code = body.code
    # reject: no code

    item.status = "reviewed" if body.action != "reject" else "rejected"
    item.reviewer_code = final_code
    item.reviewer_notes = body.notes
    item.reviewed_by = current_user.id
    decision_at = datetime.utcnow()
    item.reviewed_at = decision_at
    db.flush()

    revised = save_response_revision(db, session.id, resp.question_id, resp.answer,
        isco_code=final_code, confidence_score=1.0 if final_code else None)
    item.response_id = revised.id

    # Derive the only permitted response/history delta from pre-write values.
    # Neither a same-text extra active row nor changed classification evidence
    # may be hidden by comparing only a last-row-wins answer dictionary.
    expected_responses = {key: dict(value) for key, value in before_responses.items()}
    expected_queues = {key: dict(value) for key, value in before_queues.items()}
    if revised.id != original_id:
        retired_at = resp.deleted_at
        if retired_at is None or revised.id in before_responses:
            raise HTTPException(status_code=409, detail="Review revision conflict; the prior answer must remain historical.")
        expected_responses[original_id]["deleted_at"] = retired_at
        expected_responses[revised.id] = {
            **before_responses[original_id], "id": revised.id,
            "isco_code": final_code, "confidence_score": 1.0 if final_code else None,
            "supersedes_id": original_id, "deleted_at": None,
        }
        for key, previous in before_queues.items():
            if key != item.id and previous["response_id"] == original_id and previous["status"] == "pending":
                expected_queues[key].update(status="rejected", reviewed_at=retired_at,
                    reviewer_notes="Classification superseded by a newer answer revision.")
    expected_queues[item.id].update(response_id=revised.id,
        status="reviewed" if body.action != "reject" else "rejected",
        reviewer_code=final_code, reviewer_notes=body.notes,
        reviewed_by=current_user.id, reviewed_at=decision_at)

    db.add(AuditLog(session_id=session.id, user_id=current_user.id,
        event_type="hitl_review_decision", actor="user",
        description="Reviewer applied a classification decision without changing respondent answers.",
        metadata_json=json.dumps({"review_id": item.id, "previous_response_id": original_id,
            "response_id": revised.id, "question_id": resp.question_id, "action": body.action,
            "previous_code": original_code, "final_code": final_code}), timestamp=datetime.utcnow()))
    invalidate_reports(db, item.session_id)
    # Recompute deterministic quality from current active revisions in this
    # transaction; no model call is needed for a human review decision.
    _reassess_quality(db, item.session_id, reviewer=current_user.id, notes=body.notes)

    # Re-read flushed persistence, not the ORM pre-check, under the same locks
    # and transaction. Reject/roll back any unplanned answer or decision change.
    db.flush()
    actual_responses = {row.id: _hitl_record_values(row)
        for row in _hitl_locked_rows(db, SurveyResponse, session.id)}
    actual_queues = {row.id: _hitl_record_values(row)
        for row in _hitl_locked_rows(db, HITLQueue, session.id)}
    db.refresh(session, with_for_update=True)
    if (actual_responses != expected_responses or actual_queues != expected_queues
            or _hitl_record_values(session) != before_session):
        raise HTTPException(status_code=409, detail="Review persistence conflict; respondent answers and review evidence must remain unchanged.")
    _commit_session(db)

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
