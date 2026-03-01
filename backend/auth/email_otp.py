import os
import logging
import smtplib
import pyotp
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from backend.database.models import OTPCode, User

load_dotenv(override=True)

logger = logging.getLogger(__name__)

OTP_EXPIRY_MINUTES = 10

# Delivery channel flags
_USE_SMS_OTP = os.getenv("USE_SMS_OTP", "false").lower() == "true"
_DEV_MODE = os.getenv("APP_ENV", "development").lower() == "development"

# Gmail SMTP (preferred — set GMAIL_APP_PASSWORD in .env)
_GMAIL_USER     = os.getenv("GMAIL_USER")       # your Gmail address
_GMAIL_APP_PASS = (os.getenv("GMAIL_APP_PASSWORD") or "").replace(" ", "") or None  # 16-char App Password (spaces stripped)

# SendGrid fallback (legacy)
_SENDGRID_API_KEY    = os.getenv("SENDGRID_API_KEY")
_SENDGRID_FROM_EMAIL = os.getenv("SENDGRID_FROM_EMAIL")


# ---------------------------------------------------------------------------
# OTP helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Email senders
# ---------------------------------------------------------------------------

def _html_body(code: str) -> str:
    return (
        f"<p>Your LFS Survey verification code is:</p>"
        f"<h2 style='letter-spacing:4px;'>{code}</h2>"
        f"<p>This code expires in {OTP_EXPIRY_MINUTES} minutes.</p>"
        f"<p>If you did not request this, please ignore this email.</p>"
    )


def _send_via_gmail(recipient_email: str, code: str) -> bool:
    """Send OTP using Gmail SMTP + App Password."""
    # Read credentials fresh on every call so .env changes take effect without restart
    load_dotenv(override=True)
    gmail_user = os.getenv("GMAIL_USER")
    gmail_pass = (os.getenv("GMAIL_APP_PASSWORD") or "").replace(" ", "") or None

    if not gmail_user or not gmail_pass:
        logger.error("[Gmail] GMAIL_USER or GMAIL_APP_PASSWORD not set in .env")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Your LFS Survey Verification Code"
    msg["From"]    = gmail_user
    msg["To"]      = recipient_email
    msg.attach(MIMEText(_html_body(code), "html"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(gmail_user, gmail_pass)
            smtp.sendmail(gmail_user, recipient_email, msg.as_string())
        logger.info("[Gmail] OTP sent to %s", recipient_email)
        return True
    except Exception as exc:
        logger.error("[Gmail] Failed to send OTP to %s: %s", recipient_email, exc)
        return False


def _send_via_sendgrid(recipient_email: str, code: str) -> bool:
    """Send OTP using SendGrid (fallback)."""
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail

        message = Mail(
            from_email=_SENDGRID_FROM_EMAIL,
            to_emails=recipient_email,
            subject="Your LFS Survey Verification Code",
            html_content=_html_body(code),
        )
        client   = SendGridAPIClient(_SENDGRID_API_KEY)
        response = client.send(message)
        if response.status_code in (200, 202):
            logger.info("[SendGrid] OTP sent to %s", recipient_email)
            return True
        logger.error(
            "[SendGrid] Status %s for %s. Body: %s",
            response.status_code, recipient_email, response.body,
        )
        return False
    except Exception as exc:
        logger.error("[SendGrid] Failed to send OTP to %s: %s", recipient_email, exc)
        return False


def send_otp_email(recipient_email: str, code: str) -> bool:
    """Send OTP email. Priority: Gmail SMTP → SendGrid → dev-console fallback.

    In development mode (APP_ENV=development) the OTP is always printed to
    the server console so the auth flow works even without email credentials.
    """
    # Always print to console in dev — usable even if email delivery fails
    if _DEV_MODE:
        print(
            f"\n{'='*52}\n"
            f"  [DEV MODE] OTP for {recipient_email}: {code}\n"
            f"  (expires in {OTP_EXPIRY_MINUTES} min)\n"
            f"{'='*52}\n"
        )

    # Re-read credentials fresh so the gate reflects the current .env state
    # (avoids stale None values if the module was imported before .env was loaded)
    load_dotenv(override=True)
    gmail_user = os.getenv("GMAIL_USER")
    gmail_pass = (os.getenv("GMAIL_APP_PASSWORD") or "").replace(" ", "") or None

    # 1. Gmail SMTP — preferred (set GMAIL_USER + GMAIL_APP_PASSWORD in .env)
    if gmail_user and gmail_pass:
        sent = _send_via_gmail(recipient_email, code)
    # 2. SendGrid fallback
    elif _SENDGRID_API_KEY:
        sent = _send_via_sendgrid(recipient_email, code)
    else:
        sent = False
        logger.error(
            "No email credentials configured. "
            "Set GMAIL_APP_PASSWORD or SENDGRID_API_KEY in .env"
        )

    # In dev mode succeed even if email delivery failed (OTP is in the console)
    return sent or _DEV_MODE


# ---------------------------------------------------------------------------
# OTP verification
# ---------------------------------------------------------------------------

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
    """Generate, store, and deliver an OTP.

    Delivery channel is controlled by the USE_SMS_OTP environment variable:
      false (default) — email (Gmail SMTP or SendGrid)
      true            — SMS via Twilio (not implemented; raises RuntimeError)
    """
    if _USE_SMS_OTP:
        raise RuntimeError(
            "USE_SMS_OTP=true but Twilio SMS is not implemented. "
            "Set USE_SMS_OTP=false in .env to use email OTP."
        )
    code = generate_otp()
    store_otp(db, user.id, code)
    return send_otp_email(user.email, code)
