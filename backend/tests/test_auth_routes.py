"""
Tests for /auth/* endpoints.

send_otp_email is patched throughout so no real SendGrid calls are made,
but OTP generation and DB storage still run normally.
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest
from jose import jwt

from backend.auth.jwt_handler import ALGORITHM, SECRET_KEY
from backend.database.models import OTPCode, User


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_otp(db, user, code="123456", expired=False, used=False):
    """Insert an OTP record directly into the DB."""
    expires_at = datetime.utcnow() + (
        timedelta(minutes=-1) if expired else timedelta(minutes=10)
    )
    otp = OTPCode(user_id=user.id, code=code, expires_at=expires_at, is_used=used)
    db.add(otp)
    db.commit()
    db.refresh(otp)
    return otp


# ---------------------------------------------------------------------------
# POST /auth/request-otp
# ---------------------------------------------------------------------------

class TestRequestOtp:
    def test_creates_new_user_on_first_request(self, client, db):
        with patch("backend.auth.email_otp.send_otp_email", return_value=True):
            resp = client.post("/auth/request-otp", json={"email": "new@test.com"})
        assert resp.status_code == 200
        assert db.query(User).filter(User.email == "new@test.com").count() == 1

    def test_does_not_duplicate_existing_user(self, client, db, user):
        with patch("backend.auth.email_otp.send_otp_email", return_value=True):
            client.post("/auth/request-otp", json={"email": user.email})
        assert db.query(User).filter(User.email == user.email).count() == 1

    def test_stores_otp_in_db(self, client, db):
        with patch("backend.auth.email_otp.send_otp_email", return_value=True):
            client.post("/auth/request-otp", json={"email": "new@test.com"})
        u = db.query(User).filter(User.email == "new@test.com").first()
        otp = db.query(OTPCode).filter(OTPCode.user_id == u.id).first()
        assert otp is not None
        assert otp.is_used is False

    def test_invalidates_previous_otp_on_new_request(self, client, db, user):
        with patch("backend.auth.email_otp.send_otp_email", return_value=True):
            client.post("/auth/request-otp", json={"email": user.email})
            client.post("/auth/request-otp", json={"email": user.email})
        otps = db.query(OTPCode).filter(OTPCode.user_id == user.id).all()
        used = [o for o in otps if o.is_used]
        unused = [o for o in otps if not o.is_used]
        assert len(used) == 1
        assert len(unused) == 1

    def test_returns_success_message(self, client):
        with patch("backend.auth.email_otp.send_otp_email", return_value=True):
            resp = client.post("/auth/request-otp", json={"email": "new@test.com"})
        assert resp.json() == {"message": "OTP sent. Please check your email."}

    def test_deactivated_user_returns_403(self, client, db):
        u = User(email="inactive@test.com", is_active=False)
        db.add(u)
        db.commit()
        resp = client.post("/auth/request-otp", json={"email": "inactive@test.com"})
        assert resp.status_code == 403
        assert "deactivated" in resp.json()["detail"]

    def test_email_send_failure_returns_502(self, client):
        with patch("backend.auth.email_otp.send_otp_email", return_value=False):
            resp = client.post("/auth/request-otp", json={"email": "new@test.com"})
        assert resp.status_code == 502

    def test_invalid_email_returns_422(self, client):
        resp = client.post("/auth/request-otp", json={"email": "notanemail"})
        assert resp.status_code == 422

    def test_missing_email_returns_422(self, client):
        resp = client.post("/auth/request-otp", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /auth/verify-otp
# ---------------------------------------------------------------------------

class TestVerifyOtp:
    def test_valid_otp_returns_200_with_jwt(self, client, db, user):
        make_otp(db, user, code="111111")
        resp = client.post("/auth/verify-otp", json={"email": user.email, "code": "111111"})
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_jwt_subject_matches_user_id(self, client, db, user):
        make_otp(db, user, code="222222")
        resp = client.post("/auth/verify-otp", json={"email": user.email, "code": "222222"})
        payload = jwt.decode(resp.json()["access_token"], SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["sub"] == str(user.id)

    def test_valid_otp_is_marked_used(self, client, db, user):
        otp = make_otp(db, user, code="333333")
        client.post("/auth/verify-otp", json={"email": user.email, "code": "333333"})
        db.refresh(otp)
        assert otp.is_used is True

    def test_wrong_code_returns_401(self, client, db, user):
        make_otp(db, user, code="444444")
        resp = client.post("/auth/verify-otp", json={"email": user.email, "code": "000000"})
        assert resp.status_code == 401
        assert "Invalid or expired" in resp.json()["detail"]

    def test_already_used_code_returns_401(self, client, db, user):
        make_otp(db, user, code="555555", used=True)
        resp = client.post("/auth/verify-otp", json={"email": user.email, "code": "555555"})
        assert resp.status_code == 401

    def test_expired_code_returns_401(self, client, db, user):
        make_otp(db, user, code="666666", expired=True)
        resp = client.post("/auth/verify-otp", json={"email": user.email, "code": "666666"})
        assert resp.status_code == 401

    def test_unknown_email_returns_404(self, client):
        resp = client.post("/auth/verify-otp", json={"email": "nobody@test.com", "code": "123456"})
        assert resp.status_code == 404
        assert "No account found" in resp.json()["detail"]

    def test_deactivated_user_returns_403(self, client, db):
        u = User(email="inactive@test.com", is_active=False)
        db.add(u)
        db.commit()
        make_otp(db, u, code="777777")
        resp = client.post("/auth/verify-otp", json={"email": "inactive@test.com", "code": "777777"})
        assert resp.status_code == 403
        assert "deactivated" in resp.json()["detail"]

    def test_invalid_email_format_returns_422(self, client):
        resp = client.post("/auth/verify-otp", json={"email": "notanemail", "code": "123456"})
        assert resp.status_code == 422

    def test_missing_code_returns_422(self, client, user):
        resp = client.post("/auth/verify-otp", json={"email": user.email})
        assert resp.status_code == 422

    def test_missing_email_returns_422(self, client):
        resp = client.post("/auth/verify-otp", json={"code": "123456"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /auth/logout
# ---------------------------------------------------------------------------

class TestLogout:
    def test_returns_200(self, client):
        resp = client.post("/auth/logout")
        assert resp.status_code == 200

    def test_returns_correct_message(self, client):
        resp = client.post("/auth/logout")
        assert resp.json() == {"message": "Logged out successfully. Please discard your token."}
