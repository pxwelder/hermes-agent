"""429 handling end to end: provider error shape -> reset window -> rotation on a real pool.

The chat loop is simulated by ``Chat``: it holds one credential, hits a 429, asks
``recover_with_credential_pool`` what to do and follows the answer. Nothing sleeps: a test that
would wait out a Retry-After fails by the cascade it asserts on, not by wall-clock time.
"""
from __future__ import annotations

import logging
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.agent.new_chat_pool_support import MODEL, OTHER_MODEL, build_pool, dead, ids, manual_row


class RateLimitError(Exception):
    def __init__(self, *, body=None, headers=None, message="rate limited"):
        super().__init__(message)
        self.status_code = 429
        self.body = body
        self.response = SimpleNamespace(headers=headers or {}, status_code=429)


def iso(seconds_ahead: float) -> str:
    return (datetime.now(UTC) + timedelta(seconds=seconds_ahead)).isoformat()


def dispatched(entry):
    """What the chat would send as its key: the pool's runtime value for the entry."""
    return entry.runtime_api_key


class Chat:
    """One open chat bound to one pool entry, as ``AIAgent`` binds it."""

    def __init__(self, pool, model=MODEL):
        self.pool = pool
        first = pool.select(model=model)
        self.agent = SimpleNamespace(
            _credential_pool=pool, provider="anthropic", base_url=None, model=model,
            api_key=dispatched(first), _credential_pool_entry_id=first.id,
        )
        self.agent._swap_credential = self._swap
        self.swaps: list[str] = []

    def _swap(self, entry):
        self.agent.api_key = dispatched(entry)
        self.agent._credential_pool_entry_id = entry.id
        self.swaps.append(entry.id)

    @property
    def bound(self):
        return self.agent._credential_pool_entry_id

    def hit_429(self, error: RateLimitError, *, retried: bool = False):
        from agent.agent_runtime_helpers import extract_api_error_context, recover_with_credential_pool

        stub = SimpleNamespace(logger=logging.getLogger("test.rate_limit_rotation"))
        with patch("agent.agent_runtime_helpers._ra", return_value=stub), \
                patch("agent.agent_runtime_helpers.credential_pool_entry_serves_endpoint", return_value=True), \
                patch("time.sleep", side_effect=AssertionError("a 429 must rotate or retry, never sleep here")):
            return recover_with_credential_pool(
                self.agent, status_code=429, has_retried_429=retried,
                error_context=extract_api_error_context(error),
            )


@pytest.fixture
def three(tmp_path, monkeypatch):
    return build_pool(tmp_path, monkeypatch,
                      [manual_row("a", 0), manual_row("b", 1), manual_row("c", 2)], strategy="fill_first")


# ---- the reset window, as providers actually send it ----------------------------------------

LONG = [
    ("retry-after header, seconds", lambda: RateLimitError(headers={"retry-after": "600"})),
    ("retry-after header, just over the threshold", lambda: RateLimitError(headers={"retry-after": "31"})),
    ("body resets_at, epoch seconds", lambda: RateLimitError(body={"error": {"resets_at": time.time() + 600}})),
    ("body resets_at, epoch milliseconds", lambda: RateLimitError(body={"error": {"resets_at": (time.time() + 600) * 1000}})),
    ("body resets_at, ISO-8601", lambda: RateLimitError(body={"error": {"resets_at": iso(600)}})),
    ("body reset_at, numeric string", lambda: RateLimitError(body={"error": {"reset_at": str(time.time() + 600)}})),
    ("body resets_in_seconds", lambda: RateLimitError(body={"error": {"resets_in_seconds": 600}})),
    ("x-ratelimit-reset header, ISO-8601", lambda: RateLimitError(headers={"x-ratelimit-reset": iso(600)})),
    ("vendor reset header, duration", lambda: RateLimitError(headers={"x-ratelimit-reset-requests": "10m0s"})),
    ("vendor reset header, ISO-8601", lambda: RateLimitError(headers={"anthropic-ratelimit-tokens-reset": iso(900)})),
]
SHORT_OR_UNKNOWN = [
    ("retry-after header, a few seconds", lambda: RateLimitError(headers={"retry-after": "5"})),
    ("retry-after header, just under the threshold", lambda: RateLimitError(headers={"retry-after": "20"})),
    ("body resets_at already past", lambda: RateLimitError(body={"error": {"resets_at": iso(-60)}})),
    ("body resets_at, garbage", lambda: RateLimitError(body={"error": {"resets_at": "tomorrow-ish"}})),
    ("body without any window", lambda: RateLimitError(body={"error": {"type": "rate_limit_error"}})),
    ("no body and no headers", lambda: RateLimitError()),
]


@pytest.mark.parametrize("name,make", LONG, ids=[n for n, _ in LONG])
def test_a_long_window_rotates_on_the_first_429(name, make, three):
    chat = Chat(three)
    assert chat.hit_429(make()) == (True, False), name
    assert chat.bound != "a"


@pytest.mark.parametrize("name,make", SHORT_OR_UNKNOWN, ids=[n for n, _ in SHORT_OR_UNKNOWN])
def test_a_short_or_unknown_window_retries_the_same_key_first(name, make, three):
    chat = Chat(three)
    assert chat.hit_429(make()) == (False, True), name
    assert chat.bound == "a" and chat.swaps == []


def test_the_second_429_on_the_same_key_rotates_even_for_a_short_window(three):
    chat = Chat(three)
    assert chat.hit_429(RateLimitError(headers={"retry-after": "5"}), retried=True) == (True, False)
    assert chat.bound != "a"


# ---- cascades ---------------------------------------------------------------------------------

def test_three_accounts_in_a_row_walk_the_pool_without_waiting(three):
    chat = Chat(three)

    def wall():
        return RateLimitError(headers={"retry-after": "3600"})

    assert chat.hit_429(wall()) == (True, False)
    assert chat.hit_429(wall()) == (True, False)
    assert chat.swaps == ["b", "c"] and chat.bound == "c"
    assert chat.hit_429(wall()) == (False, True)  # nobody left: the caller falls back, still no sleep


def test_rotation_never_lands_on_a_dead_or_benched_account(tmp_path, monkeypatch):
    pool = build_pool(tmp_path, monkeypatch,
                      [manual_row("a", 0), manual_row("b", 1, **dead()), manual_row("c", 2), manual_row("d", 3)],
                      strategy="fill_first")
    entry = pool._find(lambda e: e.id == "c")
    pool._replace_entry(entry, replace(entry, model_cooldowns={MODEL: time.time() + 3600}))
    chat = Chat(pool)
    assert chat.hit_429(RateLimitError(headers={"retry-after": "600"})) == (True, False)
    assert chat.bound == "d"


def test_a_stamped_choice_opens_the_next_chat_but_a_429_moves_the_following_ones_off_it(three):
    three.prefer_for_new_chats("c")
    first = Chat(three)
    assert first.bound == "c"
    assert first.hit_429(RateLimitError(headers={"retry-after": "600"})) == (True, False)
    assert first.bound in {"a", "b"}
    assert Chat(three).bound in {"a", "b"}  # the account that just ran out is not offered again


def test_a_429_on_one_model_leaves_the_account_open_for_the_others(three):
    chat = Chat(three, model=MODEL)
    assert chat.hit_429(RateLimitError(headers={"retry-after": "600"})) == (True, False)
    assert three.next_for_new_session(model=OTHER_MODEL).id == "a"  # a is benched for MODEL only
    assert three.next_for_new_session(model=MODEL).id != "a"


def test_the_failed_account_is_the_one_that_gets_benched_not_the_pool_head(three):
    """The chat is bound to 'b' while 'a' heads the pool: the 429 belongs to 'b'."""
    three.prefer_for_new_chats("b")
    chat = Chat(three)
    assert chat.bound == "b"
    chat.hit_429(RateLimitError(headers={"retry-after": "600"}))
    assert three.next_for_new_session(model=MODEL).id == "a"
    assert {e.id: bool(e.model_cooldowns) for e in three.entries()} == {"a": False, "b": True, "c": False}


def test_two_chats_hitting_the_same_wall_both_leave_it_and_the_pool_stays_whole(three):
    one, two = Chat(three), Chat(three)
    for chat in (one, two):
        assert chat.hit_429(RateLimitError(headers={"retry-after": "600"})) == (True, False)
    assert one.bound != "a" and two.bound != "a"
    assert ids(three.entries()) == ["a", "b", "c"]


def test_recovery_logs_a_rotation_not_a_wait(three, caplog):
    chat = Chat(three)
    with caplog.at_level(logging.INFO):
        chat.hit_429(RateLimitError(headers={"retry-after": "600"}))
    assert any("rotated to pool entry" in r.getMessage() for r in caplog.records)
