"""Invariants of "which account does a NEW chat open on": scenarios, strategies, persistence, races.

The pool is real (``load_pool`` over a temp store). The oracle below restates the contract
independently of the implementation: the newest explicit choice wins while its account can
serve the model, otherwise priority order, and an account that cannot serve is never returned.
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
import textwrap
import threading
import time
from dataclasses import replace

import pytest

from tests.agent.new_chat_pool_support import (
    MODEL, OTHER_MODEL, build_pool, claude_code_login, dead, exhausted, ids, manual_row, read_rows,
    reload_pool,
)


def can_serve(entry, model, now=None):
    now = now or time.time()
    if entry.last_status == "dead":
        return False
    cooldown = (entry.model_cooldowns or {}).get(model)
    if isinstance(cooldown, (int, float)) and cooldown > now:
        return False
    return not (entry.last_status == "exhausted" and (entry.last_error_reset_at or 0) > now)


def oracle_head(pool, model):
    serving = [e for e in sorted(pool.entries(), key=lambda e: e.priority) if can_serve(e, model)]
    stamped = [e for e in serving if (e.extra or {}).get("next_chat_at", 0) > 0]
    if stamped:
        return max(stamped, key=lambda e: e.extra["next_chat_at"])
    return serving[0] if serving else None


def mutate(pool, entry_id, **fields):
    entry = pool._find(lambda e: e.id == entry_id)
    pool._replace_entry(entry, replace(entry, **fields))


def random_pool(seed, tmp_path, monkeypatch):
    rng = random.Random(seed)
    rows = []
    for n in range(rng.randint(1, 3)):
        status = rng.choice(["ok", "ok", "ok", "exhausted", "dead"])
        rows.append(manual_row(f"m{n}", n, **({"exhausted": exhausted(), "dead": dead()}.get(status, {}))))
    seeded = (lambda: claude_code_login()) if rng.random() < 0.6 else None
    pool = build_pool(tmp_path, monkeypatch, rows, seeded=seeded, strategy="fill_first")
    until = time.time() + 3600
    for entry in list(pool.entries()):
        if rng.random() < 0.25:  # the seeded login only gets model cooldowns: exhaustion re-syncs it
            mutate(pool, entry.id, model_cooldowns={rng.choice([MODEL, OTHER_MODEL]): until})
    for entry in rng.sample(pool.entries(), k=rng.randint(0, len(pool.entries()))):
        pool.prefer_for_new_chats(entry.id)
    return pool


# ---- scenarios -------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(120))
def test_head_matches_the_independent_oracle(seed, tmp_path, monkeypatch):
    pool = random_pool(seed, tmp_path, monkeypatch)
    for model in (MODEL, OTHER_MODEL):
        expected, actual = oracle_head(pool, model), pool.next_for_new_session(model=model)
        assert getattr(actual, "id", None) == getattr(expected, "id", None), (seed, model)
        if actual is not None:
            assert can_serve(actual, model)


@pytest.mark.parametrize("seed", range(60))
def test_a_new_chat_binds_to_what_the_preview_promised(seed, tmp_path, monkeypatch):
    """The preview and select() are the same decision; only select() counts a request."""
    pool = random_pool(seed, tmp_path, monkeypatch)
    before = {e.id: e.request_count for e in pool.entries()}
    promised = pool.next_for_new_session(model=MODEL)
    assert {e.id: e.request_count for e in pool.entries()} == before
    chosen = pool.select(model=MODEL)
    assert getattr(chosen, "id", None) == getattr(promised, "id", None)
    if chosen is not None:
        after = {e.id: e.request_count for e in pool.entries()}
        assert after[chosen.id] == before[chosen.id] + 1
        assert all(after[i] == before[i] for i in before if i != chosen.id)


def test_stamping_never_reorders_priorities(tmp_path, monkeypatch):
    pool = build_pool(tmp_path, monkeypatch, [manual_row("a", 0), manual_row("b", 1)],
                      seeded=lambda: claude_code_login(), strategy="fill_first")
    order = ids(sorted(pool.entries(), key=lambda e: e.priority))
    for entry in pool.entries():
        pool.prefer_for_new_chats(entry.id)
    assert ids(sorted(pool.entries(), key=lambda e: e.priority)) == order
    assert [r["priority"] for r in read_rows(tmp_path / "hermes")] == sorted(r["priority"] for r in read_rows(tmp_path / "hermes"))


# ---- strategies ------------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", ["least_used", "round_robin"])
def test_preview_follows_the_strategy_select_will_apply(strategy, tmp_path, monkeypatch):
    """Under a deterministic non-default strategy the preview must still equal select()."""
    pool = build_pool(tmp_path, monkeypatch,
                      [manual_row("a", 0, request_count=9), manual_row("b", 1, request_count=2),
                       manual_row("c", 2, request_count=5)], strategy=strategy)
    pool.prefer_for_new_chats("a")  # a stamp is a fill_first concept: other strategies ignore it
    promised = pool.next_for_new_session(model=MODEL)
    assert pool.select(model=MODEL).id == promised.id


def test_least_used_prefers_the_account_with_fewest_requests(tmp_path, monkeypatch):
    pool = build_pool(tmp_path, monkeypatch,
                      [manual_row("a", 0, request_count=9), manual_row("b", 1, request_count=2)],
                      strategy="least_used")
    assert pool.next_for_new_session(model=MODEL).id == "b"


# ---- persistence -----------------------------------------------------------------------------

def test_stamp_is_stored_on_the_row_without_touching_secrets_or_order(tmp_path, monkeypatch):
    pool = build_pool(tmp_path, monkeypatch, [manual_row("a", 0), manual_row("b", 1)], strategy="fill_first")
    stamp = pool.prefer_for_new_chats("b")
    rows = {r["id"]: r for r in read_rows(tmp_path / "hermes")}
    assert rows["b"]["next_chat_at"] == stamp  # flat on the row, like every other extra key
    assert "next_chat_at" not in rows["a"]
    assert [rows[i]["priority"] for i in ("a", "b")] == [0, 1]
    assert rows["b"]["access_token"] == "fixture-token-b"


def test_stamp_survives_a_rotated_claude_code_login(tmp_path, monkeypatch):
    """Claude Code re-issues its tokens on its own schedule: the seeded row must keep the choice."""
    pool = build_pool(tmp_path, monkeypatch, [manual_row("a", 0)],
                      seeded=lambda: claude_code_login("v1"), strategy="fill_first")
    seeded_id = next(e.id for e in pool.entries() if e.source == "claude_code")
    pool.prefer_for_new_chats(seeded_id)
    fresh = reload_pool(monkeypatch, seeded=lambda: claude_code_login("v2"))
    assert fresh.next_for_new_session(model=MODEL).id == seeded_id


def test_stamp_survives_exhausting_and_rotating_another_account(tmp_path, monkeypatch):
    pool = build_pool(tmp_path, monkeypatch, [manual_row("a", 0), manual_row("b", 1), manual_row("c", 2)],
                      strategy="fill_first")
    pool.prefer_for_new_chats("b")
    pool.mark_exhausted_and_rotate(status_code=402, credential_id="c", model=MODEL)
    assert reload_pool(monkeypatch).next_for_new_session(model=MODEL).id == "b"


def test_stamped_account_falls_back_while_exhausted_and_returns_after_reset(tmp_path, monkeypatch):
    pool = build_pool(tmp_path, monkeypatch, [manual_row("a", 0), manual_row("b", 1)], strategy="fill_first")
    pool.prefer_for_new_chats("b")
    pool.mark_exhausted_and_rotate(status_code=402, credential_id="b", model=MODEL)
    assert pool.next_for_new_session(model=MODEL).id == "a"
    pool.reset_statuses()
    assert pool.next_for_new_session(model=MODEL).id == "b"  # reset is not a choice, and not an erase


def test_model_rate_limit_benches_only_that_model_for_the_stamped_account(tmp_path, monkeypatch):
    pool = build_pool(tmp_path, monkeypatch, [manual_row("a", 0), manual_row("b", 1)], strategy="fill_first")
    pool.prefer_for_new_chats("b")
    pool.mark_exhausted_and_rotate(status_code=429, credential_id="b", model=MODEL,
                                   error_context={"reset_at": time.time() + 600})
    assert pool.next_for_new_session(model=MODEL).id == "a"
    assert pool.next_for_new_session(model=OTHER_MODEL).id == "b"


def test_a_stale_pool_cannot_erase_a_newer_stamp(tmp_path, monkeypatch):
    """The HUD process and the agent hold different pool objects over one store."""
    stale = build_pool(tmp_path, monkeypatch, [manual_row("a", 0), manual_row("b", 1), manual_row("c", 2)],
                       strategy="fill_first")
    fresh = reload_pool(monkeypatch)
    fresh.prefer_for_new_chats("b")
    stale.mark_exhausted_and_rotate(status_code=402, credential_id="c", model=MODEL)  # persists its old view
    assert reload_pool(monkeypatch).next_for_new_session(model=MODEL).id == "b"


def test_removing_the_stamped_account_falls_back_to_priority(tmp_path, monkeypatch):
    pool = build_pool(tmp_path, monkeypatch, [manual_row("a", 0), manual_row("b", 1)], strategy="fill_first")
    pool.prefer_for_new_chats("b")
    index = next(i for i, e in enumerate(pool.entries(), start=1) if e.id == "b")
    pool.remove_index(index)
    assert pool.next_for_new_session(model=MODEL).id == "a"
    assert ids(reload_pool(monkeypatch).entries()) == ["a"]


@pytest.mark.parametrize("garbage", ["soon", None, -5, float("nan"), float("inf"), float("-inf"), [], {"a": 1}, True])
def test_unusable_stamp_on_disk_is_ignored(garbage, tmp_path, monkeypatch):
    rows = [manual_row("a", 0), manual_row("b", 1, next_chat_at=garbage)]
    pool = build_pool(tmp_path, monkeypatch, rows, strategy="fill_first")
    assert pool.next_for_new_session(model=MODEL).id == "a"
    assert pool.select(model=MODEL).id == "a"


def test_clearing_a_choice_cannot_be_undone_by_a_stale_pool(tmp_path, monkeypatch):
    stale = build_pool(tmp_path, monkeypatch, [manual_row("a", 0), manual_row("b", 1), manual_row("c", 2)],
                       strategy="fill_first")
    fresh = reload_pool(monkeypatch)
    stamp = fresh.prefer_for_new_chats("b")
    stale_after_choice = reload_pool(monkeypatch)  # a process that saw the choice
    assert fresh.clear_new_chat_preference(stamp) is True
    stale_after_choice.mark_exhausted_and_rotate(status_code=402, credential_id="c", model=MODEL)
    assert reload_pool(monkeypatch).next_for_new_session(model=MODEL).id == "a"
    assert stale.next_for_new_session(model=MODEL).id == "a"


def test_a_later_choice_beats_an_older_one_rewritten_by_a_stale_pool(tmp_path, monkeypatch):
    stale = build_pool(tmp_path, monkeypatch, [manual_row("a", 0), manual_row("b", 1), manual_row("c", 2)],
                       strategy="fill_first")
    fresh = reload_pool(monkeypatch)
    fresh.prefer_for_new_chats("b")
    saw_b = reload_pool(monkeypatch)
    fresh.prefer_for_new_chats("c")  # the user changed their mind
    saw_b.mark_exhausted_and_rotate(status_code=402, credential_id="a", model=MODEL)  # writes b's old stamp back
    assert reload_pool(monkeypatch).next_for_new_session(model=MODEL).id == "c"
    assert stale is not None


# ---- races -----------------------------------------------------------------------------------

def test_threads_stamping_one_pool_leave_one_consistent_winner(tmp_path, monkeypatch):
    pool = build_pool(tmp_path, monkeypatch, [manual_row(i, n) for n, i in enumerate("abcd")],
                      strategy="fill_first")
    errors: list[BaseException] = []

    def hammer(seed):
        rng = random.Random(seed)
        try:
            for _ in range(25):
                pool.prefer_for_new_chats(rng.choice("abcd"))
                pool.next_for_new_session(model=MODEL)
        except BaseException as exc:  # surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=hammer, args=(n,)) for n in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    stamps = [e.extra.get("next_chat_at", 0) for e in pool.entries()]
    assert len(set(s for s in stamps if s)) == len([s for s in stamps if s])  # no two choices tie
    winner = max(pool.entries(), key=lambda e: e.extra.get("next_chat_at", 0))
    assert pool.next_for_new_session(model=MODEL).id == winner.id
    assert json.loads((tmp_path / "hermes" / "auth.json").read_text())  # the store is still valid JSON


WORKER = textwrap.dedent("""
    import sys
    sys.path.insert(0, {repo!r})
    import agent.anthropic_credentials as ac
    ac.read_claude_code_credentials = lambda: None
    from agent.credential_pool import load_pool
    pool = load_pool("anthropic")
    pool._entry_needs_refresh = lambda e: False
    for _ in range(10):
        pool.prefer_for_new_chats(sys.argv[1])
""")


def test_processes_stamping_one_store_leave_a_valid_store_and_one_head(tmp_path, monkeypatch):
    from pathlib import Path

    repo = str(Path(__file__).resolve().parents[2])
    build_pool(tmp_path, monkeypatch, [manual_row(i, n) for n, i in enumerate("abcd")], strategy="fill_first")
    script = tmp_path / "worker.py"
    script.write_text(WORKER.format(repo=repo))
    env = {"PATH": "/usr/bin:/bin", "HERMES_HOME": str(tmp_path / "hermes"), "HOME": str(tmp_path)}
    procs = [subprocess.Popen([sys.executable, str(script), i], env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True) for i in "abcd"]
    outputs = [p.communicate(timeout=120) for p in procs]
    assert [p.returncode for p in procs] == [0, 0, 0, 0], outputs
    rows = read_rows(tmp_path / "hermes")
    stamps = {r["id"]: r.get("next_chat_at", 0) for r in rows}
    assert any(stamps.values())
    head = reload_pool(monkeypatch).next_for_new_session(model=MODEL)
    assert head.id == max(stamps, key=stamps.get)
    assert sorted(r["id"] for r in rows) == list("abcd")  # no row lost or duplicated
