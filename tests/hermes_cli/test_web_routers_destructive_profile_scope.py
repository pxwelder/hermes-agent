"""One backend serves every profile, so a REST handler that reads ``get_hermes_home()``
directly mutates the LAUNCH profile's data no matter which profile the request named.

These pin the two halves of the contract for the destructive routes:
* a named profile is the one that gets wiped, and the launch profile survives;
* an UNNAMED profile is refused (400) while several profiles are served, and still
  means the launch profile on a genuinely single-profile host.
"""
import json

import pytest
import yaml

import agent.secret_scope as _secret_scope


@pytest.fixture
def homes(tmp_path, monkeypatch, _isolate_hermes_home):
    """Isolated launch home + one named profile, both seeded with real files."""
    from hermes_constants import get_hermes_home
    from hermes_cli import profiles

    launch_home = get_hermes_home()
    profiles_root = launch_home / "profiles"
    beta = profiles_root / "worker_beta"
    for home in (launch_home, beta):
        (home / "memories").mkdir(parents=True, exist_ok=True)
        (home / "memories" / "MEMORY.md").write_text(f"memory of {home.name}\n", encoding="utf-8")
        (home / "memories" / "USER.md").write_text(f"user of {home.name}\n", encoding="utf-8")
        (home / "webhook_subscriptions.json").write_text(
            json.dumps({"alerts": {"secret": "s", "events": []}}), encoding="utf-8")
        (home / "config.yaml").write_text(
            yaml.safe_dump({"hooks": {"PreToolUse": [{"command": "/bin/true"}]}}), encoding="utf-8")
    (beta / ".env").write_text("", encoding="utf-8")

    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: launch_home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: profiles_root)
    return {"launch": launch_home, "worker_beta": beta}


@pytest.fixture
def client(monkeypatch, homes):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", homes["launch"] / "state.db")
    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def _hooks(home):
    return (yaml.safe_load((home / "config.yaml").read_text()) or {}).get("hooks") or {}


# (call, what proves the named profile was hit, what proves the other was not)
DESTRUCTIVE = {
    "memory-reset": (
        lambda c, q: c.post(f"/api/memory/reset{q}", json={"target": "all"}),
        lambda home: not (home / "memories" / "MEMORY.md").exists(),
    ),
    "webhook-delete": (
        lambda c, q: c.delete(f"/api/webhooks/alerts{q}"),
        lambda home: "alerts" not in json.loads((home / "webhook_subscriptions.json").read_text()),
    ),
    "hook-delete": (
        lambda c, q: c.request("DELETE", f"/api/ops/hooks{q}",
                               json={"event": "PreToolUse", "command": "/bin/true"}),
        lambda home: not _hooks(home),
    ),
}


@pytest.mark.parametrize("route", sorted(DESTRUCTIVE))
def test_destructive_route_hits_the_named_profile_only(client, homes, route):
    call, gone = DESTRUCTIVE[route]
    resp = call(client, "?profile=worker_beta")
    assert resp.status_code == 200, resp.text
    assert gone(homes["worker_beta"]), f"{route} did not act on worker_beta"
    assert not gone(homes["launch"]), f"{route} also hit the launch profile"


@pytest.mark.parametrize("route", sorted(DESTRUCTIVE))
def test_destructive_route_refuses_an_unnamed_profile_while_multiplexing(
    client, homes, monkeypatch, route
):
    call, gone = DESTRUCTIVE[route]
    monkeypatch.setattr(_secret_scope, "is_multiplex_active", lambda: True)
    resp = call(client, "")
    assert resp.status_code == 400, resp.text
    assert "explicit profile" in resp.json()["detail"]
    assert not gone(homes["launch"]), f"{route} mutated the launch profile despite the 400"
    assert not gone(homes["worker_beta"])


def test_unnamed_profile_still_means_the_launch_profile_on_a_single_profile_host(
    client, homes, monkeypatch
):
    """A plain ``hermes serve`` has nothing to confuse: `curl` with no profile is unchanged."""
    monkeypatch.setattr(_secret_scope, "is_multiplex_active", lambda: False)
    resp = client.post("/api/memory/reset", json={"target": "all"})
    assert resp.status_code == 200, resp.text
    assert not (homes["launch"] / "memories" / "MEMORY.md").exists()
    assert (homes["worker_beta"] / "memories" / "MEMORY.md").exists()
