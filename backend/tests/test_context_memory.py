"""
backend/tests/test_context_memory.py

All Redis and CrewAI components are mocked — no live Redis or LLM needed.
"""
from __future__ import annotations

import json
from typing import Optional
from unittest.mock import MagicMock

import pytest

from backend.agents.context_memory import (
    ContextMemory,
    ContextSummary,
    SessionMemory,
    TurnRecord,
    _ALL_FIELDS,
    _KEY_PREFIX,
    _TTL_SECONDS,
    _now,
    get_context_memory,
)


# ---------------------------------------------------------------------------
# Fake Redis
# ---------------------------------------------------------------------------


class FakeRedis:
    """In-memory Redis substitute.  Accepts any constructor kwargs."""

    def __init__(self, *args, **kwargs):
        self._store: dict[str, str] = {}

    def get(self, key: str) -> Optional[str]:
        return self._store.get(key)

    def set(self, key: str, value: str, ex: Optional[int] = None) -> None:
        self._store[key] = value

    def delete(self, key: str) -> int:
        if key in self._store:
            del self._store[key]
            return 1
        return 0

    @classmethod
    def from_url(cls, url: str, **kwargs) -> "FakeRedis":
        return cls()


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_redis() -> FakeRedis:
    return FakeRedis()


@pytest.fixture()
def cm(fake_redis: FakeRedis, monkeypatch) -> ContextMemory:
    """ContextMemory backed by FakeRedis with CrewAI/LLM components mocked."""
    monkeypatch.setattr("backend.agents.context_memory.redis_lib.Redis", FakeRedis)
    monkeypatch.setattr(
        "backend.agents.context_memory.get_llm",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr("backend.agents.context_memory.Agent", MagicMock())
    instance = ContextMemory(redis_url="redis://localhost:6379")
    # Replace internal redis with the shared fake so tests can inspect it
    instance._redis = fake_redis
    return instance


def _make_mem(
    session_id: int = 1,
    state: str = "collecting_info",
    language: str = "en",
    collected_fields: Optional[dict] = None,
    history: Optional[list] = None,
    turn_count: int = 0,
) -> SessionMemory:
    return SessionMemory(
        session_id=session_id,
        language=language,
        state=state,
        collected_fields=collected_fields or {},
        history=history or [],
        turn_count=turn_count,
        created_at=_now(),
        last_updated=_now(),
    )


def _cm_with_all_mocks(monkeypatch, crew_output: str = "{}") -> ContextMemory:
    """Helper: fully-mocked ContextMemory with Crew returning *crew_output*."""
    monkeypatch.setattr("backend.agents.context_memory.redis_lib.Redis", FakeRedis)
    monkeypatch.setattr(
        "backend.agents.context_memory.get_llm",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr("backend.agents.context_memory.Agent", MagicMock())
    monkeypatch.setattr("backend.agents.context_memory.Task", MagicMock())

    crew_instance = MagicMock()
    crew_instance.kickoff.return_value = crew_output
    monkeypatch.setattr(
        "backend.agents.context_memory.Crew", MagicMock(return_value=crew_instance)
    )
    monkeypatch.delenv("REDIS_URL", raising=False)

    c = ContextMemory(host="localhost")
    c._redis = FakeRedis()
    return c


# ===========================================================================
# _now()
# ===========================================================================


class TestNow:
    def test_returns_string(self):
        assert isinstance(_now(), str)

    def test_contains_utc_indicator(self):
        result = _now()
        assert "+00:00" in result or result.endswith("Z")

    def test_monotonic(self):
        t1 = _now()
        t2 = _now()
        assert t2 >= t1


# ===========================================================================
# ContextMemory.__init__ — connection modes
# ===========================================================================


class TestContextMemoryInit:
    def test_url_mode_calls_from_url(self, monkeypatch):
        fake_cls = MagicMock()
        monkeypatch.setattr("backend.agents.context_memory.redis_lib.Redis", fake_cls)
        monkeypatch.setattr(
            "backend.agents.context_memory.get_llm", MagicMock(return_value=MagicMock())
        )
        monkeypatch.setattr("backend.agents.context_memory.Agent", MagicMock())
        ContextMemory(redis_url="redis://testhost:6379")
        fake_cls.from_url.assert_called_once_with(
            "redis://testhost:6379", decode_responses=True
        )

    def test_host_port_mode(self, monkeypatch):
        fake_cls = MagicMock()
        monkeypatch.setattr("backend.agents.context_memory.redis_lib.Redis", fake_cls)
        monkeypatch.setattr(
            "backend.agents.context_memory.get_llm", MagicMock(return_value=MagicMock())
        )
        monkeypatch.setattr("backend.agents.context_memory.Agent", MagicMock())
        monkeypatch.delenv("REDIS_URL", raising=False)
        ContextMemory(host="myhost", port=6380)
        fake_cls.assert_called_once_with(
            host="myhost", port=6380, db=0, decode_responses=True
        )

    def test_env_redis_url_used(self, monkeypatch):
        fake_cls = MagicMock()
        monkeypatch.setattr("backend.agents.context_memory.redis_lib.Redis", fake_cls)
        monkeypatch.setattr(
            "backend.agents.context_memory.get_llm", MagicMock(return_value=MagicMock())
        )
        monkeypatch.setattr("backend.agents.context_memory.Agent", MagicMock())
        monkeypatch.setenv("REDIS_URL", "redis://envhost:6379")
        ContextMemory()
        fake_cls.from_url.assert_called_once_with(
            "redis://envhost:6379", decode_responses=True
        )

    def test_default_ttl(self, monkeypatch):
        monkeypatch.setattr("backend.agents.context_memory.redis_lib.Redis", FakeRedis)
        monkeypatch.setattr(
            "backend.agents.context_memory.get_llm", MagicMock(return_value=MagicMock())
        )
        monkeypatch.setattr("backend.agents.context_memory.Agent", MagicMock())
        monkeypatch.delenv("REDIS_URL", raising=False)
        c = ContextMemory(host="localhost")
        assert c._ttl == _TTL_SECONDS

    def test_custom_ttl(self, monkeypatch):
        monkeypatch.setattr("backend.agents.context_memory.redis_lib.Redis", FakeRedis)
        monkeypatch.setattr(
            "backend.agents.context_memory.get_llm", MagicMock(return_value=MagicMock())
        )
        monkeypatch.setattr("backend.agents.context_memory.Agent", MagicMock())
        monkeypatch.delenv("REDIS_URL", raising=False)
        c = ContextMemory(host="localhost", ttl=3600)
        assert c._ttl == 3600


# ===========================================================================
# _key()
# ===========================================================================


class TestKey:
    def test_key_format(self, cm):
        assert cm._key(42) == f"{_KEY_PREFIX}42"

    def test_different_ids_produce_different_keys(self, cm):
        assert cm._key(1) != cm._key(2)

    def test_key_starts_with_prefix(self, cm):
        assert cm._key(0).startswith(_KEY_PREFIX)


# ===========================================================================
# save_session()
# ===========================================================================


class TestSaveSession:
    def test_returns_session_memory(self, cm):
        result = cm.save_session(1, "collecting_info", "en", {}, [])
        assert isinstance(result, SessionMemory)

    def test_persists_to_redis(self, cm, fake_redis):
        cm.save_session(1, "collecting_info", "en", {"employment_status": "employed"}, [])
        raw = fake_redis.get(f"{_KEY_PREFIX}1")
        assert raw is not None
        data = json.loads(raw)
        assert data["collected_fields"]["employment_status"] == "employed"

    def test_preserves_created_at_on_update(self, cm):
        mem1 = cm.save_session(1, "greeting", "en", {}, [])
        created = mem1.created_at
        mem2 = cm.save_session(1, "collecting_info", "en", {"job_title": "nurse"}, [])
        assert mem2.created_at == created

    def test_coerces_dict_turns_to_turn_record(self, cm):
        history = [{"role": "user", "content": "hello", "timestamp": _now()}]
        result = cm.save_session(1, "greeting", "en", {}, history)
        assert isinstance(result.history[0], TurnRecord)

    def test_counts_only_user_turns(self, cm):
        history = [
            {"role": "user", "content": "hi", "timestamp": _now()},
            {"role": "assistant", "content": "hello", "timestamp": _now()},
            {"role": "user", "content": "nurse", "timestamp": _now()},
        ]
        result = cm.save_session(1, "greeting", "en", {}, history)
        assert result.turn_count == 2

    def test_turn_count_zero_for_empty_history(self, cm):
        result = cm.save_session(1, "greeting", "en", {}, [])
        assert result.turn_count == 0

    def test_stores_language(self, cm):
        result = cm.save_session(1, "greeting", "ar", {}, [])
        assert result.language == "ar"

    def test_casts_field_values_to_str(self, cm):
        result = cm.save_session(1, "collecting_info", "en", {"hours_per_week": 40}, [])
        assert result.collected_fields["hours_per_week"] == "40"


# ===========================================================================
# load_session()
# ===========================================================================


class TestLoadSession:
    def test_returns_none_when_absent(self, cm):
        assert cm.load_session(99) is None

    def test_returns_session_memory_when_present(self, cm):
        cm.save_session(1, "collecting_info", "en", {}, [])
        result = cm.load_session(1)
        assert isinstance(result, SessionMemory)
        assert result.session_id == 1

    def test_returns_none_on_bad_json(self, cm, fake_redis):
        fake_redis.set(f"{_KEY_PREFIX}5", "not-valid-json")
        assert cm.load_session(5) is None

    def test_returns_none_on_empty_value(self, cm, fake_redis):
        fake_redis.set(f"{_KEY_PREFIX}5", "")
        assert cm.load_session(5) is None

    def test_state_round_trips(self, cm):
        cm.save_session(1, "validating", "en", {}, [])
        assert cm.load_session(1).state == "validating"

    def test_history_round_trips(self, cm):
        history = [TurnRecord(role="user", content="hi", timestamp=_now())]
        cm.save_session(1, "greeting", "en", {}, history)
        loaded = cm.load_session(1)
        assert len(loaded.history) == 1
        assert loaded.history[0].content == "hi"


# ===========================================================================
# delete_session()
# ===========================================================================


class TestDeleteSession:
    def test_returns_true_when_key_exists(self, cm):
        cm.save_session(1, "greeting", "en", {}, [])
        assert cm.delete_session(1) is True

    def test_returns_false_when_absent(self, cm):
        assert cm.delete_session(999) is False

    def test_key_removed_after_delete(self, cm, fake_redis):
        cm.save_session(1, "greeting", "en", {}, [])
        cm.delete_session(1)
        assert fake_redis.get(f"{_KEY_PREFIX}1") is None


# ===========================================================================
# append_turn()
# ===========================================================================


class TestAppendTurn:
    def test_returns_turn_record(self, cm):
        result = cm.append_turn(1, "user", "hello")
        assert isinstance(result, TurnRecord)

    def test_user_turn_increments_count(self, cm):
        cm.save_session(1, "greeting", "en", {}, [])
        cm.append_turn(1, "user", "message 1")
        cm.append_turn(1, "user", "message 2")
        assert cm.load_session(1).turn_count == 2

    def test_assistant_turn_does_not_increment(self, cm):
        cm.save_session(1, "greeting", "en", {}, [])
        cm.append_turn(1, "assistant", "response")
        assert cm.load_session(1).turn_count == 0

    def test_history_grows_on_each_append(self, cm):
        cm.save_session(1, "greeting", "en", {}, [])
        cm.append_turn(1, "user", "hi")
        cm.append_turn(1, "assistant", "hello")
        assert len(cm.load_session(1).history) == 2

    def test_creates_stub_when_session_absent(self, cm):
        cm.append_turn(99, "user", "hi")
        mem = cm.load_session(99)
        assert mem is not None
        assert len(mem.history) == 1

    def test_detected_language_stored(self, cm):
        turn = cm.append_turn(1, "user", "مرحبا", detected_language="ar")
        assert turn.detected_language == "ar"

    def test_content_stored(self, cm):
        turn = cm.append_turn(1, "user", "I am a nurse")
        assert turn.content == "I am a nurse"

    def test_role_stored(self, cm):
        turn = cm.append_turn(1, "assistant", "Thank you")
        assert turn.role == "assistant"


# ===========================================================================
# update_fields()
# ===========================================================================


class TestUpdateFields:
    def test_merges_new_fields(self, cm):
        cm.save_session(1, "collecting_info", "en", {"employment_status": "employed"}, [])
        result = cm.update_fields(1, {"job_title": "nurse"})
        assert result["employment_status"] == "employed"
        assert result["job_title"] == "nurse"

    def test_overwrites_existing_field(self, cm):
        cm.save_session(1, "collecting_info", "en", {"employment_status": "employed"}, [])
        result = cm.update_fields(1, {"employment_status": "self-employed"})
        assert result["employment_status"] == "self-employed"

    def test_ignores_empty_string_values(self, cm):
        cm.save_session(1, "collecting_info", "en", {"employment_status": "employed"}, [])
        result = cm.update_fields(1, {"employment_status": "", "job_title": "nurse"})
        assert result["employment_status"] == "employed"
        assert result["job_title"] == "nurse"

    def test_creates_stub_when_absent(self, cm):
        result = cm.update_fields(99, {"employment_status": "employed"})
        assert result["employment_status"] == "employed"

    def test_returns_full_collected_fields(self, cm):
        cm.save_session(1, "collecting_info", "en", {"employment_status": "employed"}, [])
        result = cm.update_fields(1, {"industry": "healthcare"})
        assert set(result.keys()) == {"employment_status", "industry"}


# ===========================================================================
# get_collected_fields()
# ===========================================================================


class TestGetCollectedFields:
    def test_returns_empty_when_absent(self, cm):
        assert cm.get_collected_fields(99) == {}

    def test_returns_fields_when_present(self, cm):
        cm.save_session(1, "collecting_info", "en", {"employment_status": "employed"}, [])
        assert cm.get_collected_fields(1) == {"employment_status": "employed"}

    def test_returns_copy_not_reference(self, cm):
        cm.save_session(1, "collecting_info", "en", {"employment_status": "employed"}, [])
        result = cm.get_collected_fields(1)
        result["foo"] = "bar"
        assert "foo" not in cm.get_collected_fields(1)


# ===========================================================================
# get_missing_fields()
# ===========================================================================


class TestGetMissingFields:
    def test_all_missing_when_no_session(self, cm):
        missing = cm.get_missing_fields(99)
        assert set(missing) == _ALL_FIELDS

    def test_subset_missing(self, cm):
        cm.save_session(
            1, "collecting_info", "en",
            {"employment_status": "employed", "job_title": "nurse"},
            [],
        )
        missing = cm.get_missing_fields(1)
        assert "employment_status" not in missing
        assert "job_title" not in missing
        assert len(missing) == 3

    def test_none_missing_when_all_collected(self, cm):
        all_fields = {f: "value" for f in _ALL_FIELDS}
        cm.save_session(1, "completing", "en", all_fields, [])
        assert cm.get_missing_fields(1) == []

    def test_returns_sorted_list(self, cm):
        missing = cm.get_missing_fields(99)
        assert missing == sorted(missing)

    def test_returns_list_type(self, cm):
        assert isinstance(cm.get_missing_fields(99), list)


# ===========================================================================
# get_history()
# ===========================================================================


class TestGetHistory:
    def test_empty_when_no_session(self, cm):
        assert cm.get_history(99) == []

    def test_returns_full_history(self, cm):
        history = [
            TurnRecord(role="user", content="hi", timestamp=_now()),
            TurnRecord(role="assistant", content="hello", timestamp=_now()),
        ]
        cm.save_session(1, "greeting", "en", {}, history)
        assert len(cm.get_history(1)) == 2

    def test_last_n_returns_tail(self, cm):
        history = [
            TurnRecord(role="user", content=f"msg{i}", timestamp=_now())
            for i in range(5)
        ]
        cm.save_session(1, "greeting", "en", {}, history)
        result = cm.get_history(1, last_n=3)
        assert len(result) == 3
        assert result[0].content == "msg2"
        assert result[-1].content == "msg4"

    def test_last_n_larger_than_history_returns_all(self, cm):
        history = [TurnRecord(role="user", content="hi", timestamp=_now())]
        cm.save_session(1, "greeting", "en", {}, history)
        assert len(cm.get_history(1, last_n=10)) == 1

    def test_returns_list_of_turn_records(self, cm):
        cm.save_session(1, "greeting", "en", {}, [
            TurnRecord(role="user", content="hi", timestamp=_now())
        ])
        result = cm.get_history(1)
        assert isinstance(result[0], TurnRecord)


# ===========================================================================
# _build_fallback_summary() — static method
# ===========================================================================


class TestBuildFallbackSummary:
    def test_no_fields_en_says_no_data(self):
        mem = _make_mem(collected_fields={})
        en, _ = ContextMemory._build_fallback_summary(mem, list(_ALL_FIELDS))
        assert "No employment data collected yet" in en

    def test_no_fields_ar_says_no_data(self):
        mem = _make_mem(collected_fields={})
        _, ar = ContextMemory._build_fallback_summary(mem, list(_ALL_FIELDS))
        assert "لم يتم جمع" in ar

    def test_some_fields_listed_en(self):
        mem = _make_mem(collected_fields={"employment_status": "employed"})
        en, _ = ContextMemory._build_fallback_summary(mem, ["job_title"])
        assert "Employment status" in en
        assert "employed" in en

    def test_all_fields_collected_en(self):
        all_fields = {f: "x" for f in _ALL_FIELDS}
        mem = _make_mem(collected_fields=all_fields)
        en, _ = ContextMemory._build_fallback_summary(mem, [])
        assert "All required fields collected" in en

    def test_all_fields_collected_ar(self):
        all_fields = {f: "x" for f in _ALL_FIELDS}
        mem = _make_mem(collected_fields=all_fields)
        _, ar = ContextMemory._build_fallback_summary(mem, [])
        assert "تم جمع جميع الحقول" in ar

    def test_missing_fields_listed_en(self):
        mem = _make_mem(collected_fields={"employment_status": "employed"})
        en, _ = ContextMemory._build_fallback_summary(mem, ["job_title"])
        assert "Job title" in en

    def test_known_state_label_used_en(self):
        mem = _make_mem(state="collecting_info")
        en, _ = ContextMemory._build_fallback_summary(mem, ["job_title"])
        assert "collecting data" in en

    def test_unknown_state_uses_raw_value(self):
        mem = _make_mem(state="custom_state")
        en, _ = ContextMemory._build_fallback_summary(mem, [])
        assert "custom_state" in en

    def test_turn_count_appears_in_summary(self):
        mem = _make_mem(turn_count=7)
        en, _ = ContextMemory._build_fallback_summary(mem, [])
        assert "7" in en

    def test_returns_tuple_of_two_strings(self):
        mem = _make_mem()
        result = ContextMemory._build_fallback_summary(mem, [])
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(s, str) for s in result)


# ===========================================================================
# _parse_summary_response()
# ===========================================================================


class TestParseSummaryResponse:
    """Tests use fully-mocked ContextMemory (no Redis calls needed)."""

    @pytest.fixture()
    def c(self, monkeypatch) -> ContextMemory:
        monkeypatch.setattr("backend.agents.context_memory.redis_lib.Redis", FakeRedis)
        monkeypatch.setattr(
            "backend.agents.context_memory.get_llm", MagicMock(return_value=MagicMock())
        )
        monkeypatch.setattr("backend.agents.context_memory.Agent", MagicMock())
        monkeypatch.delenv("REDIS_URL", raising=False)
        return ContextMemory(host="localhost")

    def test_valid_json(self, c):
        mem = _make_mem()
        raw = '{"summary_en": "En summary.", "summary_ar": "ملخص."}'
        en, ar = c._parse_summary_response(raw, mem, [])
        assert en == "En summary."
        assert ar == "ملخص."

    def test_json_with_markdown_fences(self, c):
        mem = _make_mem()
        raw = '```json\n{"summary_en": "En text.", "summary_ar": "نص."}\n```'
        en, ar = c._parse_summary_response(raw, mem, [])
        assert en == "En text."
        assert ar == "نص."

    def test_regex_fallback_on_surrounding_text(self, c):
        mem = _make_mem()
        raw = 'Here is the output: {"summary_en": "En.", "summary_ar": "عر."} Done.'
        en, ar = c._parse_summary_response(raw, mem, [])
        assert en == "En."
        assert ar == "عر."

    def test_missing_summary_en_triggers_fallback(self, c):
        mem = _make_mem()
        raw = '{"summary_ar": "ملخص."}'
        en, ar = c._parse_summary_response(raw, mem, [])
        assert len(en) > 0

    def test_missing_summary_ar_triggers_fallback(self, c):
        mem = _make_mem()
        raw = '{"summary_en": "En summary."}'
        en, ar = c._parse_summary_response(raw, mem, [])
        assert len(ar) > 0

    def test_invalid_json_triggers_fallback(self, c):
        mem = _make_mem()
        en, ar = c._parse_summary_response("completely invalid", mem, [])
        assert isinstance(en, str) and len(en) > 0
        assert isinstance(ar, str) and len(ar) > 0

    def test_empty_response_triggers_fallback(self, c):
        mem = _make_mem()
        en, ar = c._parse_summary_response("", mem, [])
        assert len(en) > 0
        assert len(ar) > 0


# ===========================================================================
# get_summary()
# ===========================================================================


class TestGetSummary:
    def test_returns_context_summary(self, monkeypatch):
        c = _cm_with_all_mocks(monkeypatch, '{"summary_en":"E","summary_ar":"ع"}')
        c.save_session(1, "greeting", "en", {}, [])
        assert isinstance(c.get_summary(1), ContextSummary)

    def test_session_id_in_result(self, monkeypatch):
        c = _cm_with_all_mocks(monkeypatch, '{"summary_en":"E","summary_ar":"ع"}')
        c.save_session(7, "greeting", "en", {}, [])
        assert c.get_summary(7).session_id == 7

    def test_missing_fields_correct(self, monkeypatch):
        c = _cm_with_all_mocks(monkeypatch, '{"summary_en":"E","summary_ar":"ع"}')
        c.save_session(1, "collecting_info", "en", {"employment_status": "employed"}, [])
        result = c.get_summary(1)
        assert "employment_status" not in result.missing_fields
        assert set(result.missing_fields) == _ALL_FIELDS - {"employment_status"}

    def test_non_existent_session_uses_stub(self, monkeypatch):
        c = _cm_with_all_mocks(monkeypatch, '{"summary_en":"E","summary_ar":"ع"}')
        result = c.get_summary(999)
        assert result.session_id == 999
        assert result.state == "greeting"
        assert result.collected_fields == {}
        assert set(result.missing_fields) == _ALL_FIELDS

    def test_llm_summaries_propagated(self, monkeypatch):
        raw = '{"summary_en": "English summary.", "summary_ar": "ملخص عربي."}'
        c = _cm_with_all_mocks(monkeypatch, raw)
        c.save_session(1, "greeting", "en", {}, [])
        result = c.get_summary(1)
        assert result.summary_en == "English summary."
        assert result.summary_ar == "ملخص عربي."

    def test_fallback_on_bad_llm_response(self, monkeypatch):
        c = _cm_with_all_mocks(monkeypatch, "oops not json")
        c.save_session(1, "greeting", "en", {}, [])
        result = c.get_summary(1)
        assert len(result.summary_en) > 0
        assert len(result.summary_ar) > 0

    def test_turn_count_in_result(self, monkeypatch):
        c = _cm_with_all_mocks(monkeypatch, '{"summary_en":"E","summary_ar":"ع"}')
        history = [TurnRecord(role="user", content="hi", timestamp=_now())] * 3
        c.save_session(1, "greeting", "en", {}, history)
        assert c.get_summary(1).turn_count == 3

    def test_language_from_session(self, monkeypatch):
        c = _cm_with_all_mocks(monkeypatch, '{"summary_en":"E","summary_ar":"ع"}')
        c.save_session(1, "greeting", "ar", {}, [])
        assert c.get_summary(1).language == "ar"

    def test_collected_fields_in_result(self, monkeypatch):
        c = _cm_with_all_mocks(monkeypatch, '{"summary_en":"E","summary_ar":"ع"}')
        c.save_session(1, "collecting_info", "en", {"employment_status": "employed"}, [])
        assert c.get_summary(1).collected_fields == {"employment_status": "employed"}


# ===========================================================================
# get_context_memory() singleton
# ===========================================================================


class TestGetContextMemory:
    def test_returns_context_memory_instance(self, monkeypatch):
        import backend.agents.context_memory as mod
        monkeypatch.setattr("backend.agents.context_memory.redis_lib.Redis", FakeRedis)
        monkeypatch.setattr(
            "backend.agents.context_memory.get_llm", MagicMock(return_value=MagicMock())
        )
        monkeypatch.setattr("backend.agents.context_memory.Agent", MagicMock())
        monkeypatch.delenv("REDIS_URL", raising=False)
        mod._instance = None
        result = get_context_memory()
        assert isinstance(result, ContextMemory)
        mod._instance = None  # cleanup

    def test_returns_same_instance_on_second_call(self, monkeypatch):
        import backend.agents.context_memory as mod
        monkeypatch.setattr("backend.agents.context_memory.redis_lib.Redis", FakeRedis)
        monkeypatch.setattr(
            "backend.agents.context_memory.get_llm", MagicMock(return_value=MagicMock())
        )
        monkeypatch.setattr("backend.agents.context_memory.Agent", MagicMock())
        monkeypatch.delenv("REDIS_URL", raising=False)
        mod._instance = None
        i1 = get_context_memory()
        i2 = get_context_memory()
        assert i1 is i2
        mod._instance = None  # cleanup

    def test_reuses_existing_instance(self, monkeypatch):
        import backend.agents.context_memory as mod
        existing = MagicMock(spec=ContextMemory)
        mod._instance = existing
        assert get_context_memory() is existing
        mod._instance = None  # cleanup
