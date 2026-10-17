"""Report HTTP errors must not return secrets from backend/model exceptions."""
import logging
from contextlib import contextmanager
from types import SimpleNamespace

from backend.api import survey_routes as routes


@contextmanager
def _no_lock(_session_id):
    yield


def _arrange_completed_report(monkeypatch, exc):
    monkeypatch.setattr(routes, "_session_turn", _no_lock)
    monkeypatch.setattr(routes, "_get_owned_session", lambda *_args: SimpleNamespace(status="completed"))
    monkeypatch.setattr(routes, "active_responses", lambda *_args: [])
    monkeypatch.setattr(routes, "_require_completion", lambda *_args: None)

    def fail_generation(**_kwargs):
        raise exc

    monkeypatch.setattr(
        routes, "get_report_generator", lambda: SimpleNamespace(generate=fail_generation)
    )


def test_report_http_500_hides_exception_text(auth_client, monkeypatch, caplog):
    leaked = ("postgresql://app:synthetic-secret@127.0.0.1:5432/ci "
              "bearer abc.def.ghi otp 123456 synthetic-user@example.test")
    _arrange_completed_report(monkeypatch, RuntimeError(leaked))
    caplog.set_level(logging.ERROR, logger="lfs.survey")
    response = auth_client.get("/survey/sessions/1/report")
    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail.startswith("Report generation failed. Reference: ")
    ref = detail.rsplit(" ", 1)[1]
    assert len(ref) == 12
    logged = " ".join(record.message for record in caplog.records)
    for fragment in (leaked, "synthetic-secret", "bearer", "123456", "synthetic-user@example.test"):
        assert fragment not in response.text
        assert fragment not in logged
    assert ref in logged
    assert "type=RuntimeError" in logged


def test_report_value_error_404_hides_exception_text(auth_client, monkeypatch):
    _arrange_completed_report(monkeypatch, ValueError("private-db-password synthetic-secret"))
    response = auth_client.get("/survey/sessions/1/report")
    assert response.status_code == 404
    assert response.json() == {"detail": "Report not found."}
    assert "synthetic-secret" not in response.text
