"""
Tests for eval/legacy_decision_policy41/live_reranker.py (Task 43).

Fully hermetic: every "Anthropic client" here is a fake object -- no
`anthropic.Anthropic(...)` is ever constructed with a real API key, no
network call is ever made, no real cost is ever incurred by this file.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import anthropic
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.legacy_decision_policy41 import live_reranker as lr  # noqa: E402
from eval.legacy_decision_policy41.flat_retrieval_adapter import DevRow  # noqa: E402
from eval.legacy_decision_policy41.test_flat_retrieval_adapter import FakeStore, _five_hits  # noqa: E402


@dataclass
class _TextBlock:
    text: str
    type: str = "text"


@dataclass
class _FakeResponse:
    content: list


class FakeAnthropicClient:
    """Records every call; `responses` is a list of either a raw string
    (success) or an Exception instance/class (raised) to return/raise in
    sequence, one per call to `messages.create`."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.messages = self  # so `client.messages.create(...)` works

    def create(self, model, max_tokens, temperature, messages):
        self.calls.append({"model": model, "max_tokens": max_tokens, "temperature": temperature, "messages": messages})
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return _FakeResponse(content=[_TextBlock(text=item)])


# ---------------------------------------------------------------------------
# make_anthropic_reranker
# ---------------------------------------------------------------------------

def test_reranker_calls_correct_model_and_temperature():
    client = FakeAnthropicClient(['{"selected_code": "1000", "reasoning": "ok"}'])
    reranker = lr.make_anthropic_reranker(client)
    raw = reranker("some prompt text")
    assert raw == '{"selected_code": "1000", "reasoning": "ok"}'
    call = client.calls[0]
    assert call["model"] == "claude-3-5-sonnet-20241022"
    assert call["temperature"] == 0.0
    assert call["messages"] == [{"role": "user", "content": "some prompt text"}]


def _fake_httpx_response():
    resp = MagicMock()
    resp.request = MagicMock()
    return resp


def test_reranker_retries_on_transient_error_then_succeeds():
    client = FakeAnthropicClient([
        anthropic.RateLimitError("rate limited", response=_fake_httpx_response(), body=None),
        '{"selected_code": "2000", "reasoning": "recovered"}',
    ])
    reranker = lr.make_anthropic_reranker(client)
    raw = reranker("x")
    assert raw == '{"selected_code": "2000", "reasoning": "recovered"}'
    assert len(client.calls) == 2


def test_reranker_gives_up_after_max_attempts_on_persistent_transient_error():
    client = FakeAnthropicClient([
        anthropic.APITimeoutError(request=None),
        anthropic.APITimeoutError(request=None),
        anthropic.APITimeoutError(request=None),
    ])
    reranker = lr.make_anthropic_reranker(client)
    with pytest.raises(RuntimeError, match="failed after"):
        reranker("x")
    assert len(client.calls) == lr._MAX_ATTEMPTS


def test_reranker_does_not_retry_on_authentication_error():
    auth_err = anthropic.AuthenticationError("bad key", response=_fake_httpx_response(), body=None)
    client = FakeAnthropicClient([auth_err])
    reranker = lr.make_anthropic_reranker(client)
    with pytest.raises(anthropic.AuthenticationError):
        reranker("x")
    assert len(client.calls) == 1  # no retry attempted


# ---------------------------------------------------------------------------
# run_dev_with_live_reranker
# ---------------------------------------------------------------------------

def _dev_row(case_id="C1", text="plumber", lang="en", gold="1000"):
    return DevRow(case_id=case_id, input_text=text, input_language=lang, gold_isco_4digit=gold)


def test_fast_path_row_never_calls_reranker_and_is_scored():
    hits = _five_hits(base_score=0.99)  # above threshold
    store = FakeStore({"plumber": hits})
    reranker_calls = []
    reranker = lambda prompt: reranker_calls.append(prompt) or "should never be used"
    report = lr.run_dev_with_live_reranker(store, [_dev_row(gold="1000")], reranker)
    assert reranker_calls == []
    assert report.n_semantic == 1
    assert report.n_llm_ranked == 0
    assert report.n_correct == 1
    assert report.rows[0].method == "semantic"


def test_below_threshold_row_calls_reranker_and_is_scored():
    hits = _five_hits(base_score=0.5)
    store = FakeStore({"plumber": hits})
    reranker = lambda prompt: '{"selected_code": "1002", "reasoning": "chosen"}'
    report = lr.run_dev_with_live_reranker(store, [_dev_row(gold="1002")], reranker)
    assert report.n_llm_ranked == 1
    assert report.n_semantic == 0
    assert report.n_correct == 1
    assert report.n_llm_ranked_correct == 1
    assert report.rows[0].method == "llm_ranked"
    assert report.rows[0].predicted_code == "1002"


def test_aggregate_counts_and_breakdown_correct():
    fast_hits = _five_hits(base_score=0.99)
    slow_hits = _five_hits(base_score=0.5)
    store = FakeStore({"fast": fast_hits, "slow": slow_hits})
    rows = [
        _dev_row(case_id="C1", text="fast", gold="1000"),   # semantic, correct
        _dev_row(case_id="C2", text="fast", gold="9999"),   # semantic, incorrect
        _dev_row(case_id="C3", text="slow", gold="1002"),   # llm_ranked, correct
    ]
    reranker = lambda prompt: '{"selected_code": "1002", "reasoning": "x"}'
    report = lr.run_dev_with_live_reranker(store, rows, reranker)
    assert report.n_total == 3
    assert report.n_semantic == 2
    assert report.n_semantic_correct == 1
    assert report.n_llm_ranked == 1
    assert report.n_llm_ranked_correct == 1
    assert report.n_correct == 2


# ---------------------------------------------------------------------------
# Progress JSONL + fail-closed
# ---------------------------------------------------------------------------

def test_progress_jsonl_written_incrementally(tmp_path):
    hits = _five_hits(base_score=0.99)
    store = FakeStore({"fast": hits})
    reranker = lambda prompt: "unused"
    progress_path = tmp_path / "progress.jsonl"
    rows = [_dev_row(case_id=f"C{i}", text="fast", gold="1000") for i in range(3)]
    lr.run_dev_with_live_reranker(store, rows, reranker, progress_jsonl_path=progress_path)
    lines = progress_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    parsed = [json.loads(line) for line in lines]
    assert [p["case_id"] for p in parsed] == ["C0", "C1", "C2"]


def test_unexpected_exception_mid_run_preserves_partial_progress_file(tmp_path):
    class FlakyStore(FakeStore):
        def _query(self, collection, query_vec, limit):
            if len(self.query_calls) == 2:
                raise RuntimeError("simulated failure on row 3")
            return super()._query(collection, query_vec, limit)

    hits = _five_hits(base_score=0.99)
    store = FlakyStore({f"row{i}": hits for i in range(1, 6)})
    reranker = lambda prompt: "unused"
    progress_path = tmp_path / "progress.jsonl"
    rows = [_dev_row(case_id=f"C{i}", text=f"row{i}", gold="1000") for i in range(1, 6)]

    with pytest.raises(RuntimeError, match="simulated failure"):
        lr.run_dev_with_live_reranker(store, rows, reranker, progress_jsonl_path=progress_path)

    lines = progress_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2  # first 2 rows' results survived the mid-run crash


# ---------------------------------------------------------------------------
# Fatal-error tracker -- regression coverage for the 2026-08-10/11
# insufficient-credit incident (thousands of silently-fallen-back rows
# mislabeled "llm_ranked").
# ---------------------------------------------------------------------------

def _fake_insufficient_credit_error():
    return anthropic.BadRequestError(
        "Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing.",
        response=_fake_httpx_response(), body=None,
    )


def test_is_fatal_detects_insufficient_credit_bad_request_error():
    assert lr._is_fatal(_fake_insufficient_credit_error()) is True


def test_is_fatal_false_for_unrelated_bad_request_error():
    err = anthropic.BadRequestError("some other 400 problem", response=_fake_httpx_response(), body=None)
    assert lr._is_fatal(err) is False


def test_is_fatal_detects_auth_and_permission_errors():
    assert lr._is_fatal(anthropic.AuthenticationError("bad key", response=_fake_httpx_response(), body=None)) is True
    assert lr._is_fatal(anthropic.PermissionDeniedError("denied", response=_fake_httpx_response(), body=None)) is True


def test_is_fatal_false_for_retryable_errors():
    assert lr._is_fatal(anthropic.APITimeoutError(request=None)) is False


def test_reranker_records_fatal_error_into_tracker_and_reraises():
    client = FakeAnthropicClient([_fake_insufficient_credit_error()])
    tracker = {}
    reranker = lr.make_anthropic_reranker(client, fatal_tracker=tracker)
    with pytest.raises(anthropic.BadRequestError):
        reranker("x")
    assert tracker.get("error") is not None
    assert "credit balance" in str(tracker["error"]).lower()


def test_run_aborts_immediately_on_fatal_error_without_recording_misleading_row(tmp_path):
    """The exact regression this fix targets: classify_with_policy
    silently swallows the reranker's exception and returns a normal-
    looking fallback PolicyResult -- the fatal_tracker is the only way
    the driver can tell row 2 was never actually reranked, and it must
    abort BEFORE recording that row."""
    hits = _five_hits(base_score=0.5)  # below threshold -> reranker path taken every row
    store = FakeStore({"row1": hits, "row2": hits})
    tracker = {}

    call_count = {"n": 0}

    def reranker(prompt_text):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return '{"selected_code": "1000", "reasoning": "genuine reranking"}'
        tracker["error"] = _fake_insufficient_credit_error()
        raise tracker["error"]

    progress_path = tmp_path / "progress.jsonl"
    rows = [_dev_row(case_id="C1", text="row1", gold="1000"), _dev_row(case_id="C2", text="row2", gold="1000")]

    with pytest.raises(lr.FatalRerankerError, match="C2"):
        lr.run_dev_with_live_reranker(
            store, rows, reranker, progress_jsonl_path=progress_path, fatal_tracker=tracker,
        )

    lines = progress_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1  # only C1's genuine result was recorded -- C2's misleading fallback was not
    assert json.loads(lines[0])["case_id"] == "C1"


def test_run_without_fatal_tracker_keeps_prior_silent_fallback_behavior(tmp_path):
    """Backward compatibility: a caller that doesn't pass fatal_tracker
    (e.g. existing tests/callers written before this fix) sees the same
    behavior as before -- classify_with_policy's own fallback applies and
    the run completes normally. This documents that fatal_tracker is
    opt-in, not a silent behavior change for existing callers."""
    hits = _five_hits(base_score=0.5)
    store = FakeStore({"row1": hits})

    def reranker(prompt_text):
        raise _fake_insufficient_credit_error()

    report = lr.run_dev_with_live_reranker(store, [_dev_row(case_id="C1", text="row1", gold="1000")], reranker)
    assert report.n_total == 1
    assert report.rows[0].method == "llm_ranked"  # still mislabeled without the tracker -- opt-in fix
