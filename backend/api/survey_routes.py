from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session
from pydantic import BaseModel

from backend.database.connection import get_db
from backend.database.models import SurveyResponse, SurveySession, User
from backend.auth.jwt_handler import verify_access_token

router = APIRouter(prefix="/survey", tags=["survey"])
bearer_scheme = HTTPBearer()


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
        .order_by(SurveySession.started_at.desc())
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
# Response endpoints
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
