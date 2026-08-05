import os
import re
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr

from backend.database.connection import get_db
from backend.database.models import User
from backend.auth.email_otp import (
    _DEV_MODE,
    check_rate_limit,
    generate_and_send_otp,
    generate_and_send_sms_otp,
    verify_otp,
)
from backend.auth.jwt_handler import create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])


def _client_ip(request: Request) -> str:
    """Best-effort client IP extraction (proxy-aware)."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class OTPRequestBody(BaseModel):
    email: EmailStr


class OTPVerifyBody(BaseModel):
    email: EmailStr
    code: str


class SMSOTPRequestBody(BaseModel):
    phone: str


class SMSOTPVerifyBody(BaseModel):
    phone: str
    code: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class MessageResponse(BaseModel):
    message: str
    dev_otp: Optional[str] = None  # only populated when APP_ENV=development


# ---------------------------------------------------------------------------
# Phone normalisation helper
# ---------------------------------------------------------------------------

def _normalise_phone(raw: str) -> str:
    """Normalise a phone number to E.164 format.

    Handles common UAE input patterns:
      05XXXXXXXX  → +97105XXXXXXXX  (local, 10-digit)
      5XXXXXXXX   → +9715XXXXXXXX   (local, 9-digit)
      971XXXXXXXXX → +971XXXXXXXXX  (no leading +)
      +XXXXXXXXXXX → unchanged
    """
    phone = re.sub(r"[\s\-()]", "", raw.strip())
    if phone.startswith("05") and len(phone) == 10:
        return "+971" + phone[1:]
    if phone.startswith("5") and len(phone) == 9:
        return "+971" + phone
    if phone.startswith("971") and not phone.startswith("+"):
        return "+" + phone
    if not phone.startswith("+"):
        phone = "+" + phone
    return phone


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
def request_otp(body: OTPRequestBody, request: Request, db: Session = Depends(get_db)):
    # 5 OTP requests per IP per 10 minutes — skipped in dev mode
    if not _DEV_MODE and not check_rate_limit(f"otp_req:{_client_ip(request)}", max_requests=5, window_seconds=600):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many OTP requests. Please wait before trying again.",
        )
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

    sent, code = generate_and_send_otp(db, user)
    if not sent:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to send OTP email. Please try again later.",
        )

    is_dev = os.getenv("APP_ENV", "development").lower() == "development"
    return {
        "message": "OTP sent. Please check your email.",
        "dev_otp": code if is_dev else None,
    }


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
def verify_otp_and_login(body: OTPVerifyBody, request: Request, db: Session = Depends(get_db)):
    # 10 verify attempts per email per 10 minutes — skipped in dev mode
    if not _DEV_MODE and not check_rate_limit(f"otp_verify:{body.email}", max_requests=10, window_seconds=600):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many verification attempts. Please request a new OTP.",
        )
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
    "/request-sms-otp",
    response_model=MessageResponse,
    status_code=status.HTTP_200_OK,
    summary="Request an SMS OTP",
    description=(
        "Register the phone number if it does not exist, then generate and "
        "send a 6-digit OTP via Twilio SMS valid for 10 minutes."
    ),
)
def request_sms_otp(body: SMSOTPRequestBody, db: Session = Depends(get_db)):
    phone = _normalise_phone(body.phone)

    user = db.query(User).filter(User.phone == phone).first()
    if not user:
        # Use a synthetic email for phone-only accounts so the NOT NULL
        # constraint on users.email is satisfied without a migration.
        fake_email = f"sms_{phone.lstrip('+').replace(' ', '')}@lfs-sms.local"
        user = User(phone=phone, email=fake_email)
        db.add(user)
        db.commit()
        db.refresh(user)

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This account has been deactivated.",
        )

    sent, code = generate_and_send_sms_otp(db, user)
    if not sent:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to send SMS OTP. Please check the number and try again.",
        )

    is_dev = os.getenv("APP_ENV", "development").lower() == "development"
    return {"message": "OTP sent via SMS.", "dev_otp": code if is_dev else None}


@router.post(
    "/verify-sms-otp",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Verify SMS OTP and receive JWT",
    description=(
        "Submit the 6-digit OTP received by SMS. "
        "Returns a signed JWT access token on success."
    ),
)
def verify_sms_otp(body: SMSOTPVerifyBody, db: Session = Depends(get_db)):
    phone = _normalise_phone(body.phone)

    user = db.query(User).filter(User.phone == phone).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No account found for this phone number.",
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
