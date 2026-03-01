from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session
from pydantic import BaseModel

from backend.database.connection import get_db
from backend.database.models import SurveyResponse, SurveySession, User
from backend.auth.jwt_handler import verify_access_token
from backend.agents.conversation_manager import ConversationContext, ConversationManager, ConversationState
from backend.agents.language_processor import LanguageProcessor
from backend.agents.isco_classifier import ISCOClassifier
from backend.agents.report_generator import SurveyReport, get_report_generator

router = APIRouter(prefix="/survey", tags=["survey"])
bearer_scheme = HTTPBearer()


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
    if _isco_classifier is None:
        _isco_classifier = ISCOClassifier()
    return _isco_classifier


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
    user = db.query(User).filter(User.id == int(user_id)).first()
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
    message: str


class EntityOut(BaseModel):
    text: str
    label: str
    language: str


class ISCOResult(BaseModel):
    job_title: str
    primary_code: str
    primary_title_en: str
    primary_title_ar: str
    confidence: float
    method: str


class MessageOut(BaseModel):
    reply: str
    state: str
    detected_language: str
    is_code_switched: bool
    entities: list[EntityOut]
    isco_classifications: list[ISCOResult]
    session_completed: bool


class MessageResponse(BaseModel):
    message: str


# ---------------------------------------------------------------------------
# Session endpoints
# ---------------------------------------------------------------------------

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
    return session


@router.get(
    "/sessions",
    response_model=list[SessionResponse],
    summary="List all sessions for the current user",
)
def list_sessions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return (
        db.query(SurveySession)
        .filter(SurveySession.user_id == current_user.id)
        .order_by(SurveySession.started_at.desc(), SurveySession.id.desc())
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
    db.delete(session)
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
):
    session = _get_owned_session(db, session_id, current_user.id)

    if session.status == "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Session is already completed.",
        )

    conv_mgr, lang_proc = _get_agents()

    # ── Stage 1: language detection + NER ───────────────────────────────────
    lp_result = lang_proc.process(body.message)

    # ── Stage 2: update context language if detector is confident ───────────
    ctx = _get_or_create_context(conv_mgr, session_id, session.language)
    if lp_result.detected_language in ("en", "ar"):
        ctx.language = lp_result.detected_language
        session.language = lp_result.detected_language

    # ── Stage 3: conversation turn ───────────────────────────────────────────
    reply = conv_mgr.process_message(ctx, body.message)

    # ── Stage 4: ISCO classification for every JOB_TITLE entity ─────────────
    job_title_entities = [e for e in lp_result.entities if e.label == "JOB_TITLE"]
    isco_results: list[ISCOResult] = []

    for entity in job_title_entities:
        try:
            clf = _get_isco_classifier().classify(
                entity.text,
                context=f"language={lp_result.detected_language}",
            )
            # Persist to DB
            db.add(SurveyResponse(
                session_id=session_id,
                question_id="job_title",
                answer=entity.text,
                isco_code=clf.primary.code or None,
                confidence_score=clf.primary.confidence,
            ))
            isco_results.append(ISCOResult(
                job_title=entity.text,
                primary_code=clf.primary.code,
                primary_title_en=clf.primary.title_en,
                primary_title_ar=clf.primary.title_ar,
                confidence=clf.primary.confidence,
                method=clf.method,
            ))
        except Exception:
            # ISCO classification is enrichment — don't fail the turn on error
            pass

    # ── Stage 5: session completion ──────────────────────────────────────────
    session_completed = ctx.state == ConversationState.COMPLETING
    if session_completed:
        _persist_collected_data(db, session_id, ctx.collected_data)
        session.status = "completed"
        session.completed_at = datetime.utcnow()
        _contexts.pop(session_id, None)

    db.commit()

    return MessageOut(
        reply=reply,
        state=ctx.state.value,
        detected_language=lp_result.detected_language,
        is_code_switched=lp_result.is_code_switched,
        entities=[
            EntityOut(text=e.text, label=e.label, language=e.language)
            for e in lp_result.entities
        ],
        isco_classifications=isco_results,
        session_completed=session_completed,
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
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_owned_session(db, session_id, current_user.id)
    return (
        db.query(SurveyResponse)
        .filter(SurveyResponse.session_id == session_id)
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
    """Fetch a session and enforce ownership. Raises 404 if not found or not owned."""
    session = db.query(SurveySession).filter(
        SurveySession.id == session_id,
        SurveySession.user_id == user_id,
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
    """Return the cached ConversationContext for this session, creating one if absent."""
    if session_id not in _contexts:
        _contexts[session_id] = mgr.new_context(session_id, language)
    return _contexts[session_id]


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
