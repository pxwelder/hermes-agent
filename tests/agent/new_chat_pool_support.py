"""Shared fixtures for the new-chat selection suites: a hermetic Anthropic pool built from rows.

No real home, no network, no keychain: ``HERMES_HOME`` and ``HOME`` point at a temp dir, the
Claude Code login is a stub, and token refresh is switched off. Tokens are synthetic.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

MODEL = "claude-sonnet-5-5"
OTHER_MODEL = "claude-opus-5-5"


def manual_row(cid: str, priority: int, **extra: Any) -> dict[str, Any]:
    return {
        "id": cid, "label": cid, "auth_type": "oauth", "priority": priority,
        "source": "manual:hermes_pkce", "access_token": f"fixture-token-{cid}",
        "refresh_token": f"fixture-refresh-{cid}", **extra,
    }


def exhausted(until_offset: float = 3600.0) -> dict[str, Any]:
    """Row fields for a credential-wide cooldown that ends ``until_offset`` seconds from now."""
    now = time.time()
    return {"last_status": "exhausted", "last_status_at": now, "last_error_code": 429,
            "last_error_reset_at": now + until_offset}


def dead() -> dict[str, Any]:
    return {"last_status": "dead", "last_status_at": time.time(), "last_error_code": 401}


def claude_code_login(tag: str = "seeded") -> dict[str, Any]:
    return {"accessToken": f"fixture-token-{tag}", "refreshToken": f"fixture-refresh-{tag}",
            "expiresAt": int((time.time() + 86400) * 1000)}


def write_store(home: Path, rows: list[dict[str, Any]]) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "auth.json").write_text(json.dumps({
        "version": 1, "providers": {}, "credential_pool": {"anthropic": rows},
    }))


def read_rows(home: Path) -> list[dict[str, Any]]:
    return json.loads((home / "auth.json").read_text())["credential_pool"]["anthropic"]


def build_pool(
    tmp_path: Path, monkeypatch, rows: list[dict[str, Any]], *,
    seeded: Optional[Callable[[], Optional[dict[str, Any]]]] = None, strategy: Optional[str] = None,
):
    """Load a real ``CredentialPool`` over *rows*; ``seeded`` is what the Claude Code login returns."""
    home = tmp_path / "hermes"
    write_store(home, rows)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HOME", str(tmp_path))  # no real ~/.claude, no keychain
    monkeypatch.setattr("agent.anthropic_credentials.read_claude_code_credentials",
                        seeded or (lambda: None))
    from agent.credential_pool import load_pool

    pool = load_pool("anthropic")
    monkeypatch.setattr(type(pool), "_entry_needs_refresh", lambda self, e: False)
    if strategy is not None:
        pool._strategy = strategy
    return pool


def reload_pool(monkeypatch, seeded: Optional[Callable[[], Optional[dict[str, Any]]]] = None):
    """A second, independent pool object over the same store (a different process, in effect)."""
    if seeded is not None:
        monkeypatch.setattr("agent.anthropic_credentials.read_claude_code_credentials", seeded)
    from agent.credential_pool import load_pool

    pool = load_pool("anthropic")
    monkeypatch.setattr(type(pool), "_entry_needs_refresh", lambda self, e: False)
    return pool


def ids(entries) -> list[str]:
    return [e.id for e in entries]
