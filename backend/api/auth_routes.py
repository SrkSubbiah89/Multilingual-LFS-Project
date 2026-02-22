from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr

from backend.database.connection import get_db
from backend.database.models import User
from backend.auth.email_otp import generate_and_send_otp, verify_otp
from backend.auth.jwt_handler import create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class OTPRequestBody(BaseModel):
    email: EmailStr


class OTPVerifyBody(BaseModel):
    email: EmailStr
    code: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class MessageResponse(BaseModel):
    message: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/request-otp",
    response_model=MessageResponse,
    status_code=status.HTTP_200_OK,
    summary="Request an email OTP",
    description=(
        "Register the email if it does not exist, then generate and send "
        "a 6-digit OTP valid for 10 minutes."
    ),
)
def request_otp(body: OTPRequestBody, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()

    if not user:
        user = User(email=body.email)
        db.add(user)
        db.commit()
        db.refresh(user)

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This account has been deactivated.",
        )

    sent = generate_and_send_otp(db, user)
    if not sent:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to send OTP email. Please try again later.",
        )

    return {"message": "OTP sent. Please check your email."}


@router.post(
    "/verify-otp",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Verify OTP and receive JWT",
    description=(
        "Submit the 6-digit OTP received by email. "
        "Returns a signed JWT access token on success."
    ),
)
def verify_otp_and_login(body: OTPVerifyBody, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No account found for this email.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This account has been deactivated.",
        )

    valid = verify_otp(db, user.id, body.code)
    if not valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired OTP.",
        )

    token = create_access_token(subject=user.id)
    return {"access_token": token, "token_type": "bearer"}


@router.post(
    "/logout",
    response_model=MessageResponse,
    status_code=status.HTTP_200_OK,
    summary="Logout",
    description="Client-side logout. Instructs the client to discard its JWT.",
)
def logout():
    return {"message": "Logged out successfully. Please discard your token."}
