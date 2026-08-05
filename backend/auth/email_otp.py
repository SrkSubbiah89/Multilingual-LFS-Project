import os
import logging
import smtplib
import pyotp
import threading
import time
from collections import defaultdict
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from backend.database.models import OTPCode, User

load_dotenv(override=True)

logger = logging.getLogger(__name__)

OTP_EXPIRY_MINUTES = 10
_MAX_OTP_ATTEMPTS = 5   # lock OTP after this many wrong guesses

# ---------------------------------------------------------------------------
# Rate limiter — Redis-backed sliding window (in-process fallback)
# ---------------------------------------------------------------------------
# Primary: Redis sorted-set per key; O(log N) ZADD + ZREMRANGEBYSCORE + ZCARD.
# Works correctly under multi-worker Gunicorn / uvicorn deployments.
# Fallback: thread-safe in-process dict used when Redis is unreachable.

_rl_lock = threading.Lock()
_rl_windows: dict[str, list[float]] = defaultdict(list)  # in-process fallback

_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
_rl_redis = None  # lazily initialised Redis client


def _get_rl_redis():
    """Return a shared Redis client, or None if Redis is unavailable."""
    global _rl_redis
    if _rl_redis is not None:
        return _rl_redis
    try:
        import redis as _redis_lib
        client = _redis_lib.from_url(_REDIS_URL, socket_connect_timeout=1, socket_timeout=1)
        client.ping()
        _rl_redis = client
        return _rl_redis
    except Exception:
        return None


def _check_rate_limit_inprocess(key: str, max_requests: int, window_seconds: int) -> bool:
    """Thread-safe in-process sliding-window fallback."""
    now = time.time()
    with _rl_lock:
        _rl_windows[key] = [t for t in _rl_windows[key] if now - t < window_seconds]
        if len(_rl_windows[key]) >= max_requests:
            return False
        _rl_windows[key].append(now)
        return True


def check_rate_limit(key: str, max_requests: int, window_seconds: int) -> bool:
    """
    Distributed sliding-window rate limiter.

    Uses Redis sorted sets as the primary store so the limit is shared across
    all worker processes.  Falls back to the in-process implementation if
    Redis is unavailable (single-worker deployments, tests, CI).

    Returns True when the caller is within the allowed rate,
    False when the limit has been exceeded.
    """
    r = _get_rl_redis()
    if r is None:
        return _check_rate_limit_inprocess(key, max_requests, window_seconds)

    now = time.time()
    window_start = now - window_seconds
    rl_key = f"lfs:rl:{key}"
    try:
        pipe = r.pipeline()
        pipe.zremrangebyscore(rl_key, 0, window_start)  # evict expired entries
        pipe.zadd(rl_key, {str(now): now})              # record this request
        pipe.zcard(rl_key)                              # count requests in window
        pipe.expire(rl_key, window_seconds + 10)        # auto-cleanup TTL
        results = pipe.execute()
        count = results[2]
        if count > max_requests:
            # Undo the zadd — we exceeded the limit, don't record this attempt
            r.zrem(rl_key, str(now))
            return False
        return True
    except Exception:
        # Redis error mid-pipeline — fall back to in-process
        return _check_rate_limit_inprocess(key, max_requests, window_seconds)


# ---------------------------------------------------------------------------
# Per-OTP attempt counter (brute-force lockout)
# ---------------------------------------------------------------------------

_otp_attempts: dict[int, int] = {}  # otp_entry.id → failed attempt count

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
    """Verify a submitted OTP code with brute-force lockout.

    Strategy:
    1. Find the latest valid (unused, unexpired) OTP for the user.
    2. Check if it has been locked out due to too many wrong guesses.
    3. Compare the submitted code — increment failure counter on mismatch.
    4. On success, mark as used and clear the attempt counter.

    Returns True and marks the code as used if valid.
    Returns False if wrong, expired, already used, or locked out.
    """
    # Fetch latest active OTP (not yet locked by is_used)
    otp_entry = (
        db.query(OTPCode)
        .filter(
            OTPCode.user_id == user_id,
            OTPCode.is_used == False,  # noqa: E712
            OTPCode.expires_at > datetime.utcnow(),
        )
        .order_by(OTPCode.expires_at.desc())
        .first()
    )

    if not otp_entry:
        return False

    # Brute-force guard: lock after too many wrong attempts
    attempts = _otp_attempts.get(otp_entry.id, 0)
    if attempts >= _MAX_OTP_ATTEMPTS:
        otp_entry.is_used = True   # permanently lock this OTP entry
        db.commit()
        logger.warning("OTP %d locked out after %d failed attempts.", otp_entry.id, attempts)
        return False

    if otp_entry.code != code:
        _otp_attempts[otp_entry.id] = attempts + 1
        logger.debug(
            "OTP mismatch for user %d (attempt %d/%d).",
            user_id, attempts + 1, _MAX_OTP_ATTEMPTS,
        )
        return False

    # Valid — consume and clear attempt counter
    otp_entry.is_used = True
    _otp_attempts.pop(otp_entry.id, None)
    db.commit()
    return True


def generate_and_send_otp(db: Session, user: User) -> tuple[bool, str]:
    """Generate, store, and deliver an OTP via email."""
    code = generate_otp()
    store_otp(db, user.id, code)
    return send_otp_email(user.email, code), code


# ---------------------------------------------------------------------------
# SMS sender (Twilio)
# ---------------------------------------------------------------------------

def send_sms_otp(phone: str, code: str) -> bool:
    """Send OTP via Twilio SMS.

    Falls back to dev-console in development mode if Twilio is not configured.
    """
    load_dotenv(override=True)
    sid    = os.getenv("TWILIO_ACCOUNT_SID")
    token  = os.getenv("TWILIO_AUTH_TOKEN")
    from_  = os.getenv("TWILIO_PHONE_NUMBER")

    if _DEV_MODE:
        print(
            f"\n{'='*52}\n"
            f"  [DEV MODE] SMS OTP for {phone}: {code}\n"
            f"  (expires in {OTP_EXPIRY_MINUTES} min)\n"
            f"{'='*52}\n"
        )

    placeholder_number = "+1234567890"
    if not sid or not token or not from_ or from_ == placeholder_number:
        logger.warning(
            "[Twilio] SMS credentials not fully configured "
            "(TWILIO_PHONE_NUMBER is placeholder or missing) — dev console only"
        )
        return _DEV_MODE  # succeed in dev, fail in production

    try:
        from twilio.rest import Client
        client = Client(sid, token)
        client.messages.create(
            body=(
                f"Your LFS Survey verification code is: {code}. "
                f"Valid for {OTP_EXPIRY_MINUTES} minutes."
            ),
            from_=from_,
            to=phone,
        )
        logger.info("[Twilio] SMS OTP sent to %s", phone)
        return True
    except Exception as exc:
        logger.error("[Twilio] Failed to send SMS to %s: %s", phone, exc)
        return _DEV_MODE  # in dev mode, still succeed so the flow works


def generate_and_send_sms_otp(db: Session, user: User) -> tuple[bool, str]:
    """Generate, store, and deliver an OTP via SMS."""
    code = generate_otp()
    store_otp(db, user.id, code)
    return send_sms_otp(user.phone, code), code
