"""Stage 5C: unexpected message failures stay off the HTTP response and the log."""
import logging
from contextlib import contextmanager

from backend.api import survey_routes as routes


@contextmanager
def _no_lock(_session_id):
    yield


def test_message_http_500_hides_exception_text(auth_client, monkeypatch, caplog):
    secret = "postgresql://app:fixture-password-not-real@127.0.0.1:5432/lfs_stage5c_ci"
    leaked = (
        f"RuntimeError path C:/Users/private {secret} bearer aaa.bbb.ccc "
        "otp 123456 respondent@example.com"
    )

    def explode(*_args, **_kwargs):
        raise RuntimeError(leaked)

    monkeypatch.setattr(routes, "_session_turn", _no_lock)
    monkeypatch.setattr(routes, "_send_message_impl", explode)
    caplog.set_level(logging.ERROR, logger="lfs.survey")
    response = auth_client.post(
        "/survey/sessions/1/message",
        json={"message": "Employed"},
    )
    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail.startswith("The interview could not be completed. Reference: ")
    reference = detail.rsplit(" ", 1)[1]
    assert len(reference) == 12
    logged = " ".join(record.message for record in caplog.records)
    for fragment in (
        "fixture-password-not-real", "postgresql://", "bearer", "123456", "respondent@example.com",
        "C:/Users", "Internal error", leaked,
    ):
        assert fragment not in response.text
        assert fragment not in logged
    assert "RuntimeError" not in response.text
    assert "type=RuntimeError" in logged
    assert reference in logged
