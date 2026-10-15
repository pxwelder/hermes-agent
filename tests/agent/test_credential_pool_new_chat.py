"""Which account a NEW chat opens on, and when a 429 should not wait out its window."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _row(cid, priority, source):
    return {
        "id": cid, "label": cid, "auth_type": "oauth", "priority": priority, "source": source,
        "access_token": f"fixture-token-{cid}", "refresh_token": f"fixture-refresh-{cid}",
    }


@pytest.fixture
def pool(tmp_path, monkeypatch):
    """Same shape as the real store: two manual logins ahead of one seeded claude_code login.

    The seeded row is created by the real seeding pass from a stubbed Claude Code login, so
    the ordering contract under test is the one ``load_pool`` enforces, not one the test wrote.
    """
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "auth.json").write_text(json.dumps({
        "version": 1, "providers": {},
        "credential_pool": {"anthropic": [
            _row("manual-a", 0, "manual:hermes_pkce"),
            _row("manual-b", 1, "manual:hermes_pkce"),
        ]},
    }))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HOME", str(tmp_path))  # no real ~/.claude to seed from
    monkeypatch.setattr(
        "agent.anthropic_credentials.read_claude_code_credentials",
        lambda: {"accessToken": "fixture-token-seeded", "refreshToken": "fixture-refresh-seeded",
                 "expiresAt": int((time.time() + 86400) * 1000)},
    )
    from agent.credential_pool import load_pool
    loaded = load_pool("anthropic")
    monkeypatch.setattr(type(loaded), "_entry_needs_refresh", lambda self, e: False)
    seeded = [e for e in loaded.entries() if e.source == "claude_code"]
    assert len(seeded) == 1, "the stubbed Claude Code login must be seeded into the pool"
    loaded.seeded_id = seeded[0].id
    return loaded


def ids(entries):
    return [e.id for e in entries]


def bench(pool, credential_id, model, until):
    entry = pool._find(lambda e: e.id == credential_id)
    pool._replace_entry(entry, replace(entry, model_cooldowns={model: until}))


def test_fixture_keeps_all_three_logins(pool):
    assert ids(sorted(pool.entries(), key=lambda e: e.priority)) == ["manual-a", "manual-b", pool.seeded_id]


def test_priority_cannot_promote_a_seeded_login(pool):
    """The contract the stamp works around: position 0 for a seeded entry is reverted."""
    pool.move_entry(pool.seeded_id, 0)
    assert ids(sorted(pool.entries(), key=lambda e: e.priority)) == ["manual-a", "manual-b", pool.seeded_id]


def test_stamp_makes_a_seeded_login_the_next_chat(pool):
    assert pool.prefer_for_new_chats(pool.seeded_id)
    assert pool.next_for_new_session().id == pool.seeded_id
    assert pool.select().id == pool.seeded_id
    assert sorted(e.priority for e in pool.entries()) == [0, 1, 2]


def test_stamp_survives_reload(pool):
    from agent.credential_pool import load_pool
    pool.prefer_for_new_chats(pool.seeded_id)
    assert load_pool("anthropic").next_for_new_session().id == pool.seeded_id


def test_latest_stamp_wins(pool):
    pool.prefer_for_new_chats(pool.seeded_id)
    pool.prefer_for_new_chats("manual-b")
    assert pool.next_for_new_session().id == "manual-b"


def test_stamped_login_in_model_cooldown_is_skipped(pool):
    """The choice never hands a new chat an account that is benched for its model."""
    pool.prefer_for_new_chats(pool.seeded_id)
    bench(pool, pool.seeded_id, "claude-sonnet-5-5", time.time() + 3600)
    assert pool.next_for_new_session(model="claude-sonnet-5-5").id == "manual-a"
    assert pool.next_for_new_session(model="claude-haiku-5-5").id == pool.seeded_id


def test_head_in_model_cooldown_falls_to_next_available(pool):
    bench(pool, "manual-a", "claude-sonnet-5-5", time.time() + 3600)
    assert pool.next_for_new_session(model="claude-sonnet-5-5").id == "manual-b"


def test_clear_only_removes_the_callers_own_choice(pool):
    mine = pool.prefer_for_new_chats(pool.seeded_id)
    theirs = pool.prefer_for_new_chats("manual-b")
    assert pool.clear_new_chat_preference(mine) is False
    assert pool.next_for_new_session().id == "manual-b"
    assert pool.clear_new_chat_preference(theirs) is True
    assert pool.next_for_new_session().id == "manual-a"


def test_next_for_new_session_never_counts_a_request(pool):
    before = {e.id: e.request_count for e in pool.entries()}
    pool.next_for_new_session()
    assert {e.id: e.request_count for e in pool.entries()} == before


def test_unknown_id_is_refused(pool):
    assert pool.prefer_for_new_chats("missing") is None


def test_other_strategies_ignore_the_stamp(pool):
    from agent.credential_pool import STRATEGY_LEAST_USED
    pool.prefer_for_new_chats(pool.seeded_id)
    pool._strategy = STRATEGY_LEAST_USED
    assert pool.next_for_new_session().id == "manual-a"


class TestRateLimitWithLongReset:
    """Drives recover_with_credential_pool directly: importing run_agent boots against the real home."""

    def _agent(self):
        entries = [SimpleNamespace(id=f"cred-{i}", priority=i) for i in range(2)]
        pool = MagicMock()
        pool.provider = ""  # unscoped: skips the provider-mismatch guard
        pool.has_credentials.return_value = True
        pool.entries.return_value = entries
        pool.current.return_value = entries[0]
        pool.mark_exhausted_and_rotate = MagicMock(return_value=entries[1])
        agent = SimpleNamespace(
            _credential_pool=pool, _swap_credential=MagicMock(), provider="anthropic",
            api_key="fixture-key-a", base_url=None, model="claude-sonnet-5-5",
            _credential_pool_entry_id=entries[0].id,
        )
        return agent, pool, entries

    @staticmethod
    def _recover(agent, reset_at):
        from agent.agent_runtime_helpers import recover_with_credential_pool

        # _ra() imports run_agent only to reach its logger; hand it a plain one.
        stub = SimpleNamespace(logger=logging.getLogger("test.credential_pool_new_chat"))
        with patch("agent.agent_runtime_helpers._ra", return_value=stub), \
                patch("agent.agent_runtime_helpers.credential_pool_entry_serves_endpoint", return_value=True):
            return recover_with_credential_pool(
                agent, status_code=429, has_retried_429=False, error_context={"reset_at": reset_at},
            )

    def test_long_retry_after_rotates_on_the_first_429(self):
        agent, _pool, entries = self._agent()
        assert self._recover(agent, time.time() + 600) == (True, False)
        agent._swap_credential.assert_called_once_with(entries[1])

    def test_short_retry_after_keeps_the_same_key_for_the_first_429(self):
        agent, pool, _ = self._agent()
        assert self._recover(agent, time.time() + 5) == (False, True)
        pool.mark_exhausted_and_rotate.assert_not_called()

    def test_long_retry_after_with_no_other_account_still_retries(self):
        agent, pool, _ = self._agent()
        pool.mark_exhausted_and_rotate = MagicMock(return_value=None)
        assert self._recover(agent, time.time() + 600) == (False, True)

    def test_reset_in_milliseconds_is_understood(self):
        agent, _pool, _entries = self._agent()
        assert self._recover(agent, (time.time() + 600) * 1000) == (True, False)

    @pytest.mark.parametrize("value", [None, "soon", float("nan")])
    def test_unusable_reset_is_treated_as_unknown(self, value):
        agent, pool, _ = self._agent()
        assert self._recover(agent, value) == (False, True)
        pool.mark_exhausted_and_rotate.assert_not_called()
