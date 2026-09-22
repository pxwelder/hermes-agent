# Copyright 2025 Nous Research (Licensed under the Apache License, Version 2.0)
"""A credential the user deliberately activates is adopted by an ALREADY-OPEN session.

Reordering the pool (``hermes auth priority``, the account-switch chip) only steers sessions that
resolve their credential afterwards. A live chat kept billing its init-time account until a 429/402
rotated it off, so "switch account" silently meant "switch account, eventually". ``move_entry`` and
the reset paths stamp ``activated_at``; ``adopt_activated_credential`` reads it at the turn boundary
(before the turn's first API call, so nothing in flight is disturbed).

The negative cases matter as much: without an activation nothing rebinds, because rotating accounts
mid-conversation throws away the provider-side prompt cache.
"""

import time

from agent.agent_runtime_helpers import adopt_activated_credential
from agent.credential_pool import CredentialPool, PooledCredential

_BASE = "https://api.anthropic.com"


def _entry(entry_id, label, *, priority, token, base_url=_BASE):
    return PooledCredential.from_dict("anthropic", {
        "id": entry_id, "label": label, "auth_type": "api_key", "priority": priority,
        "access_token": token, "base_url": base_url, "source": "manual",
    })


def _pool(*entries):
    return CredentialPool(provider="anthropic", entries=list(entries))


class _LiveAgent:
    """Open chat stand-in: real pool + real adoption hook, no client build."""

    provider = "anthropic"
    model = "claude-opus-5"
    base_url = _BASE
    swap_refuses = False
    _credential_pool_revert_id: "str | None" = None

    def __init__(self, pool):
        self._credential_pool = pool
        first = pool.select()
        self._credential_pool_entry_id = first.id
        self.api_key = first.runtime_api_key
        self.statuses = []

    def _swap_credential(self, entry):
        if self.swap_refuses:
            return False
        self.api_key = entry.runtime_api_key
        self._credential_pool_entry_id = entry.id
        return True

    def _emit_diagnostic_status(self, text):
        self.statuses.append(str(text))


def _seed_pool(monkeypatch, pool):
    """Route ``load_pool`` (and the runtime key resolution it hangs off) at *pool*."""
    import agent.agent_runtime_helpers as arh
    import agent.credential_pool as cp

    monkeypatch.setattr(cp, "load_pool", lambda _key: pool)
    monkeypatch.setattr(arh, "resolve_runtime_pool_key", lambda _p, base_url=None: "anthropic")


def test_promote_moves_an_open_chat_at_the_next_turn(monkeypatch):
    pool = _pool(
        _entry("acct0001", "conta-1", priority=0, token="«redacted:sk-…»-one"),
        _entry("acct0002", "conta-2", priority=1, token="«redacted:sk-…»-two"),
    )
    _seed_pool(monkeypatch, pool)
    agent = _LiveAgent(pool)
    assert agent._credential_pool_entry_id == "acct0001"

    # First look only establishes the baseline: pre-existing stamps are history, not a choice
    # made during this conversation.
    assert adopt_activated_credential(agent) is False
    assert agent._credential_pool_entry_id == "acct0001"

    pool.move_entry("acct0002", 0)  # the user promotes the second account in the chip
    assert adopt_activated_credential(agent) is True
    assert agent._credential_pool_entry_id == "acct0002"
    assert agent.api_key == "«redacted:sk-…»-two"
    assert agent.statuses and "conta-2" in agent.statuses[0]

    # Idempotent: the same activation must not re-swap (and re-pay the prompt cache) every turn.
    agent.statuses.clear()
    assert adopt_activated_credential(agent) is False
    assert agent._credential_pool_entry_id == "acct0002" and not agent.statuses


def test_no_activation_keeps_the_session_and_its_prompt_cache(monkeypatch):
    """The hook is activation-driven, never a per-turn re-select: silence means stay put."""
    pool = _pool(
        _entry("acct0001", "conta-1", priority=0, token="«redacted:sk-…»-one"),
        _entry("acct0002", "conta-2", priority=1, token="«redacted:sk-…»-two"),
    )
    _seed_pool(monkeypatch, pool)
    agent = _LiveAgent(pool)
    for _ in range(3):
        assert adopt_activated_credential(agent) is False
    assert agent._credential_pool_entry_id == "acct0001"
    assert agent.api_key == "«redacted:sk-…»-one"


def test_bulk_reset_does_not_yank_the_session_onto_a_benched_account(monkeypatch):
    """``reset_statuses`` lifts cooldowns; it is not an account choice, so nothing rebinds."""
    pool = _pool(
        _entry("acct0001", "conta-1", priority=0, token="«redacted:sk-…»-one"),
        _entry("acct0002", "conta-2", priority=1, token="«redacted:sk-…»-two"),
    )
    _seed_pool(monkeypatch, pool)
    agent = _LiveAgent(pool)
    assert adopt_activated_credential(agent) is False  # baseline
    pool.mark_exhausted_and_rotate(
        status_code=429, credential_id="acct0002", failure_reason="rate_limit",
        error_context={"message": "Error"},
    )
    assert pool.reset_statuses() >= 1
    assert adopt_activated_credential(agent) is False
    assert agent._credential_pool_entry_id == "acct0001"


def test_targeted_reset_is_an_account_choice_and_rebinds(monkeypatch):
    """``reset_status(id)`` names one credential: that IS "use this one", so the session moves."""
    pool = _pool(
        _entry("acct0001", "conta-1", priority=0, token="«redacted:sk-…»-one"),
        _entry("acct0002", "conta-2", priority=1, token="«redacted:sk-…»-two"),
    )
    _seed_pool(monkeypatch, pool)
    agent = _LiveAgent(pool)
    assert adopt_activated_credential(agent) is False  # baseline
    pool.mark_exhausted_and_rotate(
        status_code=429, credential_id="acct0002", failure_reason="rate_limit",
        error_context={"message": "Error"},
    )
    assert pool.reset_status("acct0002") is not None
    assert adopt_activated_credential(agent) is True
    assert agent._credential_pool_entry_id == "acct0002"


def test_activation_on_a_foreign_endpoint_is_not_adopted(monkeypatch):
    """A same-provider entry pointing elsewhere (proxy/gateway) must not move this session's route."""
    pool = _pool(
        _entry("acct0001", "conta-1", priority=0, token="«redacted:sk-…»-one"),
        _entry("proxy001", "gateway", priority=1, token="«redacted:sk-…»-proxy",
               base_url="https://llm-proxy.internal"),
    )
    _seed_pool(monkeypatch, pool)
    agent = _LiveAgent(pool)
    assert adopt_activated_credential(agent) is False  # baseline
    pool.move_entry("proxy001", 0)
    assert adopt_activated_credential(agent) is False
    assert agent._credential_pool_entry_id == "acct0001"
    assert agent.api_key == "«redacted:sk-…»-one"


def test_activation_of_a_benched_account_waits_for_its_window(monkeypatch):
    """Promoting a rate-limited account must not hand the next turn a guaranteed 429.

    The baseline is NOT advanced in that case: once the cooldown lifts, a later turn adopts the
    account the user already chose, without asking them to click promote again.
    """
    import agent.credential_pool as cp

    pool = _pool(
        _entry("acct0001", "conta-1", priority=0, token="«redacted:sk-…»-one"),
        _entry("acct0002", "conta-2", priority=1, token="«redacted:sk-…»-two"),
    )
    _seed_pool(monkeypatch, pool)
    agent = _LiveAgent(pool)
    assert adopt_activated_credential(agent) is False  # baseline

    pool.mark_exhausted_and_rotate(
        status_code=429, credential_id="acct0002", failure_reason="rate_limit",
        error_context={"message": "Error"},
    )
    pool.move_entry("acct0002", 0)  # promoted while still cooling down
    assert adopt_activated_credential(agent) is False
    assert agent._credential_pool_entry_id == "acct0001"

    real_time = time.time
    monkeypatch.setattr(
        cp.time, "time", lambda: real_time() + cp.EXHAUSTED_TTL_429_SECONDS + 120,
    )
    assert adopt_activated_credential(agent) is True
    assert agent._credential_pool_entry_id == "acct0002"


def test_a_refused_swap_leaves_the_session_exactly_as_it_was(monkeypatch):
    """``_swap_credential`` refuses when the entry's route cannot serve the conversation's model."""
    pool = _pool(
        _entry("acct0001", "conta-1", priority=0, token="«redacted:sk-…»-one"),
        _entry("acct0002", "conta-2", priority=1, token="«redacted:sk-…»-two"),
    )
    _seed_pool(monkeypatch, pool)
    agent = _LiveAgent(pool)
    assert adopt_activated_credential(agent) is False  # baseline
    original_pool = agent._credential_pool
    agent.swap_refuses = True
    pool.move_entry("acct0002", 0)
    assert adopt_activated_credential(agent) is False
    assert agent._credential_pool_entry_id == "acct0001"
    assert agent.api_key == "«redacted:sk-…»-one"
    assert agent._credential_pool is original_pool


def test_activation_clears_a_pending_automatic_revert(monkeypatch):
    """An explicit account choice outranks the queued revert to a quota-benched credential."""
    pool = _pool(
        _entry("acct0001", "conta-1", priority=0, token="«redacted:sk-…»-one"),
        _entry("acct0002", "conta-2", priority=1, token="«redacted:sk-…»-two"),
    )
    _seed_pool(monkeypatch, pool)
    agent = _LiveAgent(pool)
    assert adopt_activated_credential(agent) is False  # baseline
    agent._credential_pool_revert_id = "acct0009"
    pool.move_entry("acct0002", 0)
    assert adopt_activated_credential(agent) is True
    assert agent._credential_pool_revert_id is None


def test_activation_stamp_survives_a_pool_round_trip():
    """``activated_at`` rides in ``extra``, so it must survive to_dict -> from_dict."""
    pool = _pool(
        _entry("acct0001", "conta-1", priority=0, token="«redacted:sk-…»-one"),
        _entry("acct0002", "conta-2", priority=1, token="«redacted:sk-…»-two"),
    )
    before = time.time()
    promoted = pool.move_entry("acct0002", 0)
    assert promoted is not None
    assert isinstance(promoted.activated_at, float) and promoted.activated_at >= before
    revived = PooledCredential.from_dict("anthropic", promoted.to_dict())
    assert revived.activated_at == promoted.activated_at


def _count_pool_reads(monkeypatch, pool):
    """Route ``load_pool`` at *pool* and count how often the turn boundary pays for it."""
    import agent.agent_runtime_helpers as arh
    import agent.credential_pool as cp

    reads = []
    monkeypatch.setattr(cp, "load_pool", lambda _key: (reads.append(1), pool)[1])
    monkeypatch.setattr(arh, "resolve_runtime_pool_key", lambda _p, base_url=None: "anthropic")
    return reads


def _stub_store(monkeypatch, fingerprint):
    """Pin the credential store's (mtime, size) so the test drives the short-circuit."""
    import agent.agent_runtime_helpers as arh

    box = {"value": fingerprint}
    monkeypatch.setattr(arh, "_auth_store_fingerprint", lambda: box["value"])
    return box


def test_an_untouched_store_is_not_reloaded_every_turn(monkeypatch):
    """An activation always writes auth.json, so an unchanged store means nothing to adopt.

    ``load_pool`` is expensive (on macOS it shells out to ``security`` for the Claude Code
    keychain entry), and this hook runs at every turn of every live session.
    """
    pool = _pool(
        _entry("acct0001", "conta-1", priority=0, token="«redacted:sk-…»-one"),
        _entry("acct0002", "conta-2", priority=1, token="«redacted:sk-…»-two"),
    )
    reads = _count_pool_reads(monkeypatch, pool)
    _stub_store(monkeypatch, (111, 222))
    agent = _LiveAgent(pool)

    assert adopt_activated_credential(agent) is False  # baseline: reads once, arms the fingerprint
    assert len(reads) == 1

    for _ in range(5):
        assert adopt_activated_credential(agent) is False
    assert len(reads) == 1, "an unchanged store must not be re-read"


def test_a_written_store_is_read_again_and_adopted(monkeypatch):
    """The short-circuit must not swallow a real activation: a new (mtime, size) reopens the path."""
    pool = _pool(
        _entry("acct0001", "conta-1", priority=0, token="«redacted:sk-…»-one"),
        _entry("acct0002", "conta-2", priority=1, token="«redacted:sk-…»-two"),
    )
    reads = _count_pool_reads(monkeypatch, pool)
    store = _stub_store(monkeypatch, (111, 222))
    agent = _LiveAgent(pool)
    assert adopt_activated_credential(agent) is False  # baseline

    pool.move_entry("acct0002", 0)
    store["value"] = (333, 444)  # the promote wrote auth.json

    assert adopt_activated_credential(agent) is True
    assert agent._credential_pool_entry_id == "acct0002"
    assert len(reads) == 2


def test_a_cooling_down_activation_keeps_being_retried(monkeypatch):
    """A cooldown expires with the clock, not with a write, so the fingerprint must NOT be armed.

    Arming it there would strand the activation until something else touched auth.json — the
    user would have to promote a second time to get the account they already chose.
    """
    import agent.credential_pool as cp

    pool = _pool(
        _entry("acct0001", "conta-1", priority=0, token="«redacted:sk-…»-one"),
        _entry("acct0002", "conta-2", priority=1, token="«redacted:sk-…»-two"),
    )
    reads = _count_pool_reads(monkeypatch, pool)
    store = _stub_store(monkeypatch, (111, 222))
    agent = _LiveAgent(pool)
    assert adopt_activated_credential(agent) is False  # baseline

    pool.mark_exhausted_and_rotate(
        status_code=429, credential_id="acct0002", failure_reason="rate_limit",
        error_context={"message": "Error"},
    )
    pool.move_entry("acct0002", 0)
    store["value"] = (333, 444)  # the promote wrote auth.json

    assert adopt_activated_credential(agent) is False  # benched: stays put
    assert adopt_activated_credential(agent) is False  # and keeps looking, store unchanged
    assert len(reads) == 3, "a pending activation must survive until its window reopens"

    real_time = time.time
    monkeypatch.setattr(
        cp.time, "time", lambda: real_time() + cp.EXHAUSTED_TTL_429_SECONDS + 120,
    )
    assert adopt_activated_credential(agent) is True
    assert agent._credential_pool_entry_id == "acct0002"


def test_a_store_that_cannot_be_stated_falls_back_to_reading(monkeypatch):
    """No fingerprint (missing or unreadable store) must degrade to the old always-read path."""
    pool = _pool(
        _entry("acct0001", "conta-1", priority=0, token="«redacted:sk-…»-one"),
        _entry("acct0002", "conta-2", priority=1, token="«redacted:sk-…»-two"),
    )
    reads = _count_pool_reads(monkeypatch, pool)
    _stub_store(monkeypatch, None)
    agent = _LiveAgent(pool)

    assert adopt_activated_credential(agent) is False  # baseline
    pool.move_entry("acct0002", 0)
    assert adopt_activated_credential(agent) is True
    assert len(reads) == 2


def test_equal_stamps_break_the_tie_by_priority(monkeypatch):
    """Two entries stamped in the same tick must resolve deterministically, not by dict order."""
    pool = _pool(
        _entry("acct0001", "conta-1", priority=0, token="«redacted:sk-…»-one"),
        _entry("acct0002", "conta-2", priority=1, token="«redacted:sk-…»-two"),
        _entry("acct0003", "conta-3", priority=2, token="«redacted:sk-…»-three"),
    )
    _seed_pool(monkeypatch, pool)
    agent = _LiveAgent(pool)
    assert adopt_activated_credential(agent) is False  # baseline

    stamp = time.time() + 1
    for entry_id in ("acct0003", "acct0002"):
        entry = next(e for e in pool.entries() if e.id == entry_id)
        pool._adopt(entry, persist=False, extra={**(entry.extra or {}), "activated_at": stamp})

    assert adopt_activated_credential(agent) is True
    assert agent._credential_pool_entry_id == "acct0002", "lowest priority wins an exact tie"


def test_a_choice_made_before_the_chat_was_built_is_honoured(monkeypatch):
    """The user picked an account, then the app restarted (or the chat was reopened).

    The rebuilt session bound the pool head through ``select()`` (priority), which a provider rule
    can pin to the OTHER account for good. A first-look baseline swallowed the stamp and the chat
    kept billing the account the user had switched away from until they clicked again.
    """
    pool = _pool(
        _entry("acct0001", "conta-1", priority=0, token="«redacted:sk-…»-one"),
        _entry("acct0002", "conta-2", priority=1, token="«redacted:sk-…»-two"),
    )
    chosen = next(e for e in pool.entries() if e.id == "acct0002")
    pool._adopt(chosen, persist=False, extra={**(chosen.extra or {}), "activated_at": time.time() - 60})
    _seed_pool(monkeypatch, pool)
    agent = _LiveAgent(pool)
    assert agent._credential_pool_entry_id == "acct0001"  # select() went by priority

    assert adopt_activated_credential(agent) is True
    assert agent._credential_pool_entry_id == "acct0002"
    assert adopt_activated_credential(agent) is False  # settled, no re-swap every turn


def test_an_activation_lands_between_the_api_calls_of_a_tool_loop(monkeypatch):
    """``prepare_iteration`` re-checks from the 2nd call on, so a click mid-turn is not parked
    until the user's next message (a tool loop can run for many minutes)."""
    import agent.turn_iteration_prep as tip

    calls = []
    monkeypatch.setattr(
        "agent.agent_runtime_helpers.adopt_activated_credential", lambda a: calls.append(a) or True,
    )

    class _Stop(Exception):
        pass

    class _A:
        _fallback_activated = False
        _nous_wire_pending = None
        step_callback = None
        _skill_nudge_interval = 0

        def _adopt_nous_key_before_expiry(self):
            pass

        def _drain_pending_steer(self):
            raise _Stop  # everything after the activation check is out of scope here

    agent = _A()
    # The loop passes the count PRE-increment: the 1st request sees 0 (covered by the turn
    # boundary), the 2nd sees 1 and must already pick up a click made during the 1st.
    for count, expected in ((0, 0), (1, 1), (2, 2)):
        try:
            tip.prepare_iteration(agent, messages=[], api_call_count=count)
        except _Stop:
            pass
        assert len(calls) == expected, f"api_call_count={count}"

    agent._fallback_activated = True  # on a fallback provider the primary's pool is not ours
    try:
        tip.prepare_iteration(agent, messages=[], api_call_count=3)
    except _Stop:
        pass
    assert len(calls) == 2


def test_a_swap_that_raises_rolls_back_and_retries_later(monkeypatch):
    """A client rebuild that raises must not leave a half-adopted session nor drop the choice."""
    pool = _pool(
        _entry("acct0001", "conta-1", priority=0, token="«redacted:sk-…»-one"),
        _entry("acct0002", "conta-2", priority=1, token="«redacted:sk-…»-two"),
    )
    _seed_pool(monkeypatch, pool)
    agent = _LiveAgent(pool)
    original_pool = agent._credential_pool
    pool.move_entry("acct0002", 0)

    def _boom(entry):
        agent._credential_pool_entry_id = entry.id  # partially applied before raising
        raise RuntimeError("TLS config exploded")

    agent._swap_credential = _boom
    assert adopt_activated_credential(agent) is False
    assert agent._credential_pool_entry_id == "acct0001"
    assert agent._credential_pool is original_pool

    del agent._swap_credential  # the next request retries the same choice and lands it
    assert adopt_activated_credential(agent) is True
    assert agent._credential_pool_entry_id == "acct0002"


def test_restoring_the_primary_applies_an_activation_made_during_fallback(monkeypatch):
    """The primary rebind selects by priority; the user's pending choice must win the first request."""
    import agent.agent_runtime_helpers as arh

    calls = []
    monkeypatch.setattr(arh, "adopt_activated_credential", lambda a, **kw: calls.append(a) or True)
    monkeypatch.setattr(arh, "_primary_reset_gate_blocks", lambda *a, **k: (False, None, None))
    for name in ("_apply_primary_runtime_fields", "_restore_runtime_capabilities", "_rebuild_primary_client",
                 "_rebind_primary_credential_pool"):
        monkeypatch.setattr(arh, name, lambda *a, **k: None)
    import agent.chat_completion_helpers as cch
    monkeypatch.setattr(cch, "_reset_stale_streak", lambda a: None)
    monkeypatch.setattr(cch, "rewrite_prompt_model_identity", lambda *a: None)

    class _Compressor:
        def update_model(self, **kw):
            pass

    class _A:
        _fallback_activated = True
        _rate_limited_until = 0
        model, provider, api_mode = "claude-opus-5", "anthropic", "anthropic_messages"
        _cache_disabled = False
        _compression_feasibility_checked = False
        _provider_fallback_active = False
        _primary_runtime = {
            "provider": "anthropic", "model": "claude-opus-5", "base_url": _BASE,
            "use_prompt_caching": True, "compressor_model": "m", "compressor_context_length": 1,
            "compressor_base_url": _BASE, "compressor_api_key": "k", "compressor_provider": "anthropic",
        }
        context_compressor = _Compressor()

        def _emit_diagnostic_status(self, text):
            pass

    monkeypatch.setattr("agent.fallback_cooldown._is_entitlement_rejected", lambda *a: False)
    agent = _A()
    assert arh.restore_primary_runtime(agent) is True
    assert calls == [agent]


def test_a_supplied_pool_never_arms_the_store_short_circuit(monkeypatch):
    """Primary restore hands in the pool it loaded; a click after that load must not be skipped."""
    pool = _pool(
        _entry("acct0001", "conta-1", priority=0, token="«redacted:sk-…»-one"),
        _entry("acct0002", "conta-2", priority=1, token="«redacted:sk-…»-two"),
    )
    reads = _count_pool_reads(monkeypatch, pool)
    store = _stub_store(monkeypatch, (111, 222))
    agent = _LiveAgent(pool)
    stale = _pool(
        _entry("acct0001", "conta-1", priority=0, token="«redacted:sk-…»-one"),
        _entry("acct0002", "conta-2", priority=1, token="«redacted:sk-…»-two"),
    )
    pool.move_entry("acct0002", 0)  # the click, after the caller's load
    store["value"] = (333, 444)

    assert adopt_activated_credential(agent, fresh=stale) is False  # stale snapshot: nothing new
    assert adopt_activated_credential(agent) is True  # next check reloads instead of short-circuiting
    assert agent._credential_pool_entry_id == "acct0002"
    assert len(reads) == 1


def test_a_swap_that_raises_after_publishing_rolls_every_field_back(monkeypatch):
    """Non-Anthropic swaps publish api_key/base_url before the client rebuild; a raise there must
    not leave the key of one entry bound to the id of another."""
    pool = _pool(
        _entry("acct0001", "conta-1", priority=0, token="«redacted:sk-…»-one"),
        _entry("acct0002", "conta-2", priority=1, token="«redacted:sk-…»-two"),
    )
    _seed_pool(monkeypatch, pool)
    agent = _LiveAgent(pool)
    agent.api_key = "«redacted:sk-…»-one"
    agent.base_url = _BASE
    agent._client_kwargs = {"api_key": "«redacted:sk-…»-one", "base_url": _BASE}

    def _half_swap(entry):
        agent.api_key = entry.runtime_api_key
        agent.base_url = "https://outra.example.test"
        agent._client_kwargs["api_key"] = entry.runtime_api_key
        raise RuntimeError("client rebuild exploded")

    agent._swap_credential = _half_swap
    pool.move_entry("acct0002", 0)
    attempts = []
    real_half_swap = _half_swap

    def _counted(entry):
        attempts.append(entry.id)
        return real_half_swap(entry)

    agent._swap_credential = _counted

    assert adopt_activated_credential(agent) is False
    assert attempts == ["acct0002"]  # the swap really ran and raised
    assert agent._credential_pool_entry_id == "acct0001"
    assert agent.api_key == "«redacted:sk-…»-one"
    assert agent.base_url == _BASE
    assert agent._client_kwargs["api_key"] == "«redacted:sk-…»-one"
