import os
import pyotp
from datetime import datetime, timedelta
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from backend.database.models import OTPCode, User

load_dotenv()

SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
SENDGRID_FROM_EMAIL = os.getenv("SENDGRID_FROM_EMAIL")
OTP_EXPIRY_MINUTES = 10


def generate_otp() -> str:
    """Generate a cryptographically secure 6-digit OTP."""
    totp = pyotp.TOTP(pyotp.random_base32(), digits=6, interval=OTP_EXPIRY_MINUTES * 60)
    return totp.now()


def store_otp(db: Session, user_id: int, code: str) -> OTPCode:
    """Persist OTP to the database with a 10-minute expiry.
    Invalidates any previously unused OTPs for the same user.
    """
    db.query(OTPCode).filter(
        OTPCode.user_id == user_id,
        OTPCode.is_used == False,  # noqa: E712
    ).update({"is_used": True})

    otp_entry = OTPCode(
        user_id=user_id,
        code=code,
        expires_at=datetime.utcnow() + timedelta(minutes=OTP_EXPIRY_MINUTES),
        is_used=False,
    )
    db.add(otp_entry)
    db.commit()
    db.refresh(otp_entry)
    return otp_entry


def send_otp_email(recipient_email: str, code: str) -> bool:
    """Send OTP to the user via SendGrid. Returns True on success."""
    message = Mail(
        from_email=SENDGRID_FROM_EMAIL,
        to_emails=recipient_email,
        subject="Your LFS Survey Verification Code",
        html_content=(
            f"<p>Your verification code is:</p>"
            f"<h2 style='letter-spacing: 4px;'>{code}</h2>"
            f"<p>This code expires in {OTP_EXPIRY_MINUTES} minutes.</p>"
            f"<p>If you did not request this code, please ignore this email.</p>"
        ),
    )
    try:
        client = SendGridAPIClient(SENDGRID_API_KEY)
        response = client.send(message)
        return response.status_code in (200, 202)
    except Exception as exc:
        print(f"[SendGrid] Failed to send OTP email: {exc}")
        return False


def verify_otp(db: Session, user_id: int, code: str) -> bool:
    """Verify a submitted OTP code.

    Returns True and marks the code as used if valid.
    Returns False if the code is wrong, expired, or already used.
    """
    otp_entry = (
        db.query(OTPCode)
        .filter(
            OTPCode.user_id == user_id,
            OTPCode.code == code,
            OTPCode.is_used == False,  # noqa: E712
            OTPCode.expires_at > datetime.utcnow(),
        )
        .first()
    )

    if not otp_entry:
        return False

    otp_entry.is_used = True
    db.commit()
    return True


def generate_and_send_otp(db: Session, user: User) -> bool:
    """Convenience function: generate, store, and email an OTP in one call."""
    code = generate_otp()
    store_otp(db, user.id, code)
    return send_otp_email(user.email, code)
