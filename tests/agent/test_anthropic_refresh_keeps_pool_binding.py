# Copyright 2025 Nous Research (Licensed under the Apache License, Version 2.0)
"""The pre-request Anthropic token refresh must not move a pool-bound session to another account.

``_try_refresh_anthropic_client_credentials`` runs before EVERY Anthropic request. It resolved the
token through the global resolver (``resolve_anthropic_token``: the first manual pool OAuth, else
Claude Code), so a session bound to any other pool entry — after a 429 rotation or an explicit
account activation — had its key overwritten on the very next request. The pool id said one
account while the wire billed another, and the account-switch UI showed the wrong one "in use".
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.client_lifecycle import ClientLifecycleMixin
from agent.credential_pool import CredentialPool, PooledCredential

_BASE = "https://api.anthropic.com"


def _entry(entry_id, label, *, priority, token, source="manual"):
    return PooledCredential.from_dict("anthropic", {
        "id": entry_id, "label": label, "auth_type": "oauth", "priority": priority,
        "access_token": token, "base_url": _BASE, "source": source,
    })


def _pool():
    return CredentialPool(provider="anthropic", entries=[
        _entry("pessoal1", "pessoal", priority=0, token="tok-pessoal", source="manual:hermes_pkce"),
        _entry("team0001", "team", priority=1, token="tok-team", source="claude_code"),
    ])


class _Agent(ClientLifecycleMixin):
    """Just enough of AIAgent to drive the refresh path without building a real client."""

    def __init__(self, pool, entry_id, key):
        self.api_mode = "anthropic_messages"
        self.provider = "anthropic"
        self.model = "claude-opus-5"
        self.base_url = _BASE
        self._anthropic_base_url = _BASE
        self._anthropic_api_key = key
        self._anthropic_client = MagicMock()
        self._is_anthropic_oauth = True
        self._credential_pool = pool
        self._credential_pool_entry_id = entry_id

    def _anthropic_oauth_flag(self, token):
        return True

    def _build_direct_anthropic_client(self, token, base_url):
        return SimpleNamespace(token=token)


def _patched(pool, *, global_token="tok-pessoal"):
    return (
        patch("agent.credential_pool.load_pool", lambda _key: pool),
        patch("agent.anthropic_credentials.resolve_anthropic_token", return_value=global_token),
    )


def test_refresh_keeps_the_activated_account_on_the_wire():
    """The reported bug: activated Team, refresh flipped the key back to the personal account."""
    pool = _pool()
    agent = _Agent(pool, "team0001", "tok-team")
    load, resolve = _patched(pool)
    with load, resolve:
        assert agent._try_refresh_anthropic_client_credentials() is False
    assert agent._anthropic_api_key == "tok-team"


def test_refresh_adopts_a_renewed_token_of_the_same_entry():
    """Token renewal still works: the bound entry's NEW token is picked up and the client rebuilt."""
    pool = _pool()
    agent = _Agent(pool, "team0001", "tok-team-old")
    load, resolve = _patched(pool)
    with load, resolve:
        assert agent._try_refresh_anthropic_client_credentials() is True
    assert agent._anthropic_api_key == "tok-team"
    assert agent._anthropic_client.token == "tok-team"


def test_a_vanished_entry_keeps_the_key_in_hand_instead_of_switching():
    pool = _pool()
    agent = _Agent(pool, "gone0001", "tok-team")
    load, resolve = _patched(pool)
    with load, resolve:
        assert agent._try_refresh_anthropic_client_credentials() is False
    assert agent._anthropic_api_key == "tok-team"


def test_an_unreadable_pool_keeps_the_key_in_hand():
    pool = _pool()
    agent = _Agent(pool, "team0001", "tok-team")

    def _boom(_key):
        raise OSError("auth.json locked")

    with patch("agent.credential_pool.load_pool", _boom), \
            patch("agent.anthropic_credentials.resolve_anthropic_token", return_value="tok-pessoal"):
        assert agent._try_refresh_anthropic_client_credentials() is False
    assert agent._anthropic_api_key == "tok-team"


def test_a_session_without_a_pool_still_uses_the_global_resolver():
    """Env / Claude Code sessions with no pool keep the old behaviour (token rotation on disk)."""
    agent = _Agent(None, None, "tok-old")
    with patch("agent.anthropic_credentials.resolve_anthropic_token", return_value="tok-new"):
        assert agent._try_refresh_anthropic_client_credentials() is True
    assert agent._anthropic_api_key == "tok-new"


def test_a_foreign_pool_binding_never_falls_through_to_the_global_resolver():
    pool = _pool()
    agent = _Agent(pool, "team0001", "tok-team")
    with patch("agent.credential_pool.credential_pool_matches_provider", return_value=False), \
            patch("agent.credential_pool.load_pool", lambda _key: pool), \
            patch("agent.anthropic_credentials.resolve_anthropic_token", return_value="tok-pessoal"):
        assert agent._try_refresh_anthropic_client_credentials() is False
    assert agent._anthropic_api_key == "tok-team"


def test_a_renewed_pool_token_also_updates_agent_api_key():
    """Recovery and diagnostics read ``agent.api_key``: it must match the key on the wire."""
    pool = _pool()
    agent = _Agent(pool, "team0001", "tok-team-old")
    agent.api_key = "tok-team-old"
    load, resolve = _patched(pool)
    with load, resolve:
        assert agent._try_refresh_anthropic_client_credentials() is True
    assert agent.api_key == agent._anthropic_api_key == "tok-team"


def test_a_pool_with_its_binding_cleared_keeps_the_key_in_hand():
    """Rebind with nothing selectable / failed id sync leave the pool attached but no entry id."""
    pool = _pool()
    agent = _Agent(pool, None, "tok-team")
    load, resolve = _patched(pool)
    with load, resolve:
        assert agent._try_refresh_anthropic_client_credentials() is False
    assert agent._anthropic_api_key == "tok-team"


def test_a_swap_whose_client_build_raises_publishes_nothing(monkeypatch):
    """Build first, publish after: the old client, key and entry id survive a failed build."""
    import hermes_cli.anon_auth as anon

    monkeypatch.setattr(anon, "route_can_serve_model", lambda *a, **k: True)
    pool = _pool()
    agent = _Agent(pool, "pessoal1", "tok-pessoal")
    agent.api_key = "tok-pessoal"
    old_client = agent._anthropic_client

    def _boom(token, base_url):
        raise RuntimeError("TLS config exploded")

    agent._build_direct_anthropic_client = _boom
    team = next(e for e in pool.entries() if e.id == "team0001")
    try:
        agent._swap_credential(team)
    except RuntimeError:
        pass
    assert agent._credential_pool_entry_id == "pessoal1"
    assert agent._anthropic_api_key == agent.api_key == "tok-pessoal"
    assert agent._anthropic_client is old_client
    old_client.close.assert_not_called()


def test_a_refresh_whose_client_build_raises_keeps_the_old_client_open():
    """The pool entry renewed its token but the client build failed: the old client stays usable."""
    pool = _pool()
    agent = _Agent(pool, "team0001", "tok-team-velho")
    old_client = agent._anthropic_client

    def _boom(token, base_url):
        raise RuntimeError("TLS config exploded")

    agent._build_direct_anthropic_client = _boom
    load, resolve = _patched(pool)
    with load, resolve:
        assert agent._try_refresh_anthropic_client_credentials() is False
    assert agent._anthropic_client is old_client
    old_client.close.assert_not_called()
    assert agent._anthropic_api_key == "tok-team-velho"


def test_a_renewed_token_publishes_the_pool_it_came_from():
    """After renewal, 401/429 recovery must find the dispatched key in the agent's pool."""
    stale = _pool()
    renewed = CredentialPool(provider="anthropic", entries=[
        _entry("pessoal1", "pessoal", priority=0, token="tok-pessoal", source="manual:hermes_pkce"),
        _entry("team0001", "team", priority=1, token="tok-team-novo", source="claude_code"),
    ])
    agent = _Agent(stale, "team0001", "tok-team")
    load, resolve = _patched(renewed)
    with load, resolve:
        assert agent._try_refresh_anthropic_client_credentials() is True
    assert agent._anthropic_api_key == agent.api_key == "tok-team-novo"
    assert agent._credential_pool is renewed
    from agent.agent_runtime_helpers import _failed_credential_identity

    hint, cid = _failed_credential_identity(agent, agent._credential_pool)
    matched = agent._credential_pool._identify_failed_entry(cid, hint)
    assert matched is not None and matched.id == "team0001"
