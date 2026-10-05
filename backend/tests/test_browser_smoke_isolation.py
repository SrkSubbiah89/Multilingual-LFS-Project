"""Pure checks preventing smoke fixtures from connecting to other services."""

from copy import deepcopy
from unittest.mock import MagicMock

import pytest

from scripts import run_browser_smoke_backend as smoke


@pytest.fixture
def verified_state(monkeypatch):
    state = {
        "status": "ready", "owner": "test-owner", "ownership_label": smoke.OWNER_LABEL,
        "services": {
            "postgres": {"id": "a" * 64},
            "redis": {"id": "b" * 64},
        },
        "DATABASE_URL": "postgresql://browser_smoke_user:test-only-password@127.0.0.1:54321/browser_smoke_db",
        "REDIS_URL": "redis://127.0.0.1:54322/0",
    }

    def inspected(service, owner, role):
        port = "54321" if role == "postgres" else "54322"
        return {
            "State": {"Running": True},
            "NetworkSettings": {"Ports": {smoke.CONTAINER_PORTS[role]: [{"HostPort": port}]}},
        }

    inspector = MagicMock(side_effect=inspected)
    monkeypatch.setattr(smoke, "inspect_owned", inspector)
    return state, inspector


def test_connection_urls_match_both_verified_container_ports(verified_state):
    state, inspector = verified_state

    smoke.validate_service_urls(state)

    assert inspector.call_count == 2
    assert {call.args[2] for call in inspector.call_args_list} == {"postgres", "redis"}


@pytest.mark.parametrize("key,value", [
    ("status", "stopped"),
    ("ownership_label", "unrelated-owner-label"),
    ("owner", ""),
    ("services", {"postgres": {"id": "a" * 64}}),
])
def test_incomplete_or_unowned_state_is_rejected_before_inspection(verified_state, key, value):
    state, inspector = verified_state
    state[key] = value

    with pytest.raises(RuntimeError, match="Complete ready"):
        smoke.validate_service_urls(state)

    inspector.assert_not_called()


@pytest.mark.parametrize("url", [
    "postgresql://browser_smoke_user:dummy@127.0.0.1:5432/browser_smoke_db",
    "postgresql://browser_smoke_user:dummy@remote.example.invalid:54321/browser_smoke_db",
    "postgresql://browser_smoke_user:dummy@127.0.0.1:54321/lfs_db",
    "postgresql://lfs_user:dummy@127.0.0.1:54321/browser_smoke_db",
    "postgresql://browser_smoke_user:dummy@127.0.0.1:54321/browser_smoke_db?host=127.0.0.1&port=5432",
    "postgresql://browser_smoke_user:dummy@127.0.0.1:54321/browser_smoke_db?service=respondent-db",
    "sqlite:///browser_smoke_db",
])
def test_other_database_targets_and_query_overrides_are_rejected(verified_state, url):
    state, _ = verified_state
    state["DATABASE_URL"] = url

    with pytest.raises(RuntimeError, match="Database URL does not target"):
        smoke.validate_service_urls(state)


@pytest.mark.parametrize("url", [
    "redis://127.0.0.1:6379/0",
    "redis://remote.example.invalid:54322/0",
    "redis://127.0.0.1:54322/1",
    "redis://127.0.0.1:54322/0?host=127.0.0.1&port=6379",
    "redis://127.0.0.1:54322/0#unexpected",
    "rediss://127.0.0.1:54322/0",
    "redis://unrelated-user:dummy@127.0.0.1:54322/0",
])
def test_other_redis_targets_and_query_overrides_are_rejected(verified_state, url):
    state, _ = verified_state
    state["REDIS_URL"] = url

    with pytest.raises(RuntimeError, match="Redis URL does not target"):
        smoke.validate_service_urls(state)


@pytest.mark.parametrize("record", [None, {"State": {"Running": False}}])
def test_missing_or_stopped_containers_cannot_receive_fixture_writes(verified_state, record):
    state, inspector = verified_state
    inspector.side_effect = None
    inspector.return_value = record

    with pytest.raises(RuntimeError, match="not running"):
        smoke.validate_service_urls(state)


@pytest.mark.parametrize("key,url", [
    ("DATABASE_URL", "malformed-test-only-private-password"),
    ("REDIS_URL", "redis://127.0.0.1:malformed-test-only-private-password/0"),
])
def test_malformed_urls_do_not_echo_private_input(verified_state, key, url):
    state, _ = verified_state
    state[key] = url

    with pytest.raises(RuntimeError) as failure:
        smoke.validate_service_urls(state)

    assert "test-only-private-password" not in str(failure.value)
    assert failure.value.__suppress_context__


def test_failed_validation_stops_main_before_environment_changes_or_migrations(verified_state, monkeypatch):
    import json
    import os
    from pathlib import Path

    state, _ = verified_state
    state["REDIS_URL"] = "redis://127.0.0.1:6379/0"
    monkeypatch.setattr(Path, "read_text", MagicMock(return_value=json.dumps(state)))
    original = deepcopy(dict(os.environ))

    with pytest.raises(RuntimeError, match="Redis URL does not target"):
        smoke.main()

    assert dict(os.environ) == original
