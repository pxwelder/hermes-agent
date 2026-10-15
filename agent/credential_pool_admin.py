"""Locked credential-pool administration and target resolution."""
from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.credential_pool import PooledCredential


def _cleared_status_copy(entry: PooledCredential) -> PooledCredential:
    from agent.credential_pool import _CLEAR_STATUS

    # The reset marker lets a live pool in another process tell "reset after my cooldown" from
    # "never had a status" — both read as bare None on disk (#89415).
    return replace(entry, **_CLEAR_STATUS, model_cooldowns=None, status_cleared_at=time.time(),
                   extra={k: v for k, v in entry.extra.items() if k != "failure_reason"})


class CredentialNotSavedError(RuntimeError):
    """``add_entry`` wrote the pool but the store does not hold the new row."""


# Intent of "which account do new chats open on", stored as an epoch in ``entry.extra``.
# Priority cannot carry it: the Anthropic provider keeps manual logins ahead of seeded ones
# (documented ordering contract), so asking for position 0 on a seeded account is reverted by
# the load-time normalisation. A stamp survives that, and only a new chat reads it.
NEW_CHAT_KEY = "next_chat_at"


def new_chat_stamp(entry: Any) -> float:
    """Epoch of the last "use this account for new chats" choice, 0.0 when none."""
    try:
        return max(0.0, float((getattr(entry, "extra", None) or {}).get(NEW_CHAT_KEY) or 0.0))
    except (TypeError, ValueError):
        return 0.0


def order_for_new_chats(available: list[Any]) -> list[Any]:
    """``available`` with the most recently chosen entry first; unchanged when nothing is stamped.

    Applied to entries that already passed the availability filter (cooldown, model cooldown,
    dead), so a stamped account that cannot serve right now falls back to priority order.
    """
    stamped = [e for e in available if new_chat_stamp(e) > 0.0]
    if not stamped:
        return available
    top = max(stamped, key=lambda e: (new_chat_stamp(e), -int(getattr(e, "priority", 0) or 0)))
    return [top, *[e for e in available if e is not top]]


class CredentialPoolAdminMixin:
    def reset_status(self, credential_id: str) -> Optional[PooledCredential]:
        """Clear only the target's local error state, preserving sibling cooldowns."""
        with self._lock:
            entry = self._find(lambda e: e.id == credential_id)
            if entry is None:
                return None
            cleared = _cleared_status_copy(entry)
            self._replace_entry(entry, cleared)
            self._persist(status_cleared_ids=[cleared.id])
            return cleared
    def reset_statuses(self) -> int:
        """Clear exhaustion state on every entry. Returns how many were cleared.

        ``failure_reason`` lives in ``extra``, not a dataclass field, so it is
        stripped explicitly. The persist declares the cleared ids because the
        disk-recency merge reads a cleared ``last_status_at`` (None -> epoch 0)
        as a stale snapshot and would copy a still-binding cooldown back.
        """
        from agent.credential_pool import _CLEAR_STATUS

        with self._lock:
            stale = [
                e for e in self._entries
                if e.last_status or e.last_status_at or e.last_error_code or e.failure_reason or e.model_cooldowns
            ]
            if stale:
                stale_ids = {e.id for e in stale}
                self._entries = [
                    _cleared_status_copy(e) if e.id in stale_ids else e
                    for e in self._entries
                ]
                self._persist(status_cleared_ids=list(stale_ids))
            return len(stale)

    def remove_index(self, index: int) -> Optional[PooledCredential]:
        with self._lock:
            if index < 1 or index > len(self._entries):
                return None
            removed = self._entries.pop(index - 1)
            self._entries = [replace(e, priority=p) for p, e in enumerate(self._entries)]
            self._persist(removed_ids=[removed.id])
            if self._current_id == removed.id:
                self._current_id = None
            return removed

    def move_entry(self, credential_id: str, priority: int) -> Optional[PooledCredential]:
        """Place an entry at a clamped zero-based position and persist contiguous priorities."""
        from agent.credential_pool import _normalize_pool_priorities

        with self._lock:
            entry = self._find(lambda e: e.id == credential_id)
            if entry is None:
                return None
            others = [e for e in self._entries if e.id != credential_id]
            others.insert(max(0, min(int(priority), len(others))), entry)
            entries = [replace(e, priority=p) for p, e in enumerate(others)]
            # Apply load-time ordering now so the reported position survives reload.
            _normalize_pool_priorities(self.provider, entries)
            self._entries = sorted(entries, key=lambda e: e.priority)
            self._persist()
            return self._find(lambda e: e.id == credential_id)

    def prefer_for_new_chats(self, credential_id: str) -> Optional[float]:
        """Make *credential_id* the account new chats open on; returns the stamp, None if unknown.

        Never reorders priorities and never touches a chat that is already open: those keep
        the account they are bound to.
        """
        with self._lock:
            entry = self._find(lambda e: e.id == credential_id)
            if entry is None:
                return None
            newest = max((new_chat_stamp(e) for e in self._entries), default=0.0)
            stamp = max(time.time(), newest + 0.001)
            self._replace_entry(entry, replace(entry, extra={**entry.extra, NEW_CHAT_KEY: stamp}))
            self._persist()
            return stamp

    def clear_new_chat_preference(self, stamp: Optional[float] = None) -> bool:
        """Drop the new-chat choice. With *stamp*, only when it is still the latest one.

        The guard lets a caller undo its own choice without erasing a later one made by hand.
        Stored as 0.0 instead of deleting the key: the store merge treats a missing key as
        "keep the disk copy".
        """
        with self._lock:
            stamped = [e for e in self._entries if new_chat_stamp(e) > 0.0]
            if not stamped:
                return False
            if stamp is not None and max(new_chat_stamp(e) for e in stamped) != stamp:
                return False
            for e in stamped:
                self._replace_entry(e, replace(e, extra={**e.extra, NEW_CHAT_KEY: 0.0}))
            self._persist()
            return True

    def next_for_new_session(self, *, model: Optional[str] = None) -> Optional[PooledCredential]:
        """The entry a chat opened now would bind to. Read-only: no request count, no refresh."""
        from agent.credential_pool import STRATEGY_FILL_FIRST

        with self._lock:
            available, _pending = self._available_entries(model=model)
            if self._strategy == STRATEGY_FILL_FIRST:
                available = order_for_new_chats(available)
            return available[0] if available else None

    def resolve_target(self, target: Any) -> tuple[Optional[int], Optional[PooledCredential], Optional[str]]:
        raw = str(target or "").strip()
        if not raw:
            return None, None, "No credential target provided."

        with self._lock:
            for idx, entry in enumerate(self._entries, start=1):
                if entry.id == raw:
                    return idx, entry, None

            label_matches = [
                (idx, entry)
                for idx, entry in enumerate(self._entries, start=1)
                if entry.label.strip().lower() == raw.lower()
            ]
            if len(label_matches) == 1:
                return label_matches[0][0], label_matches[0][1], None
            if len(label_matches) > 1:
                return None, None, f'Ambiguous credential label "{raw}". Use the numeric index or entry id instead.'
            if raw.isdigit():
                index = int(raw)
                if 1 <= index <= len(self._entries):
                    return index, self._entries[index - 1], None
                return None, None, f"No credential #{index}."
            return None, None, f'No credential matching "{raw}".'

    def add_entry(self, entry: PooledCredential) -> PooledCredential:
        from agent.credential_pool import (
            _borrowed_single_use_pool_root, _next_priority, _profile_owns_pool_provider,
            write_credential_pool,
        )
        from hermes_cli import auth as auth_mod
        from hermes_cli.auth_oauth_grants import SINGLE_USE_REFRESH_POOL_PROVIDERS

        with self._lock:
            entry = replace(entry, priority=_next_priority(self._entries))
            self._entries.append(entry)
            borrowed_ids = self._borrowed_root_ids
            # Decided now, not from load_pool()'s snapshot: a profile with no
            # rows of its own borrows from root even when root has none, and
            # ``_persist`` would send the new row to the update-only root merge,
            # which drops it (#103694).
            profile_borrows = (
                self.provider in SINGLE_USE_REFRESH_POOL_PROVIDERS
                and not _profile_owns_pool_provider(self.provider)
                and _borrowed_single_use_pool_root() is not None
            )
            if borrowed_ids or profile_borrows:
                # ``hermes -p <profile> auth add <single-use provider>``: a fresh
                # login inside the profile is the profile's OWN credential, so
                # it goes to the profile store, never to root. Only that row is
                # written: the rest of this pool is root's (its rows, or its
                # ``providers.<id>`` login seeded through the fallback), and a
                # copy would fork that single-use refresh token (#100339). Rows
                # the profile seeds from its own sources come back on the next
                # load. Once the profile owns rows, the root fallback for this
                # provider is shadowed.
                self._entries = [entry]
                written = write_credential_pool(
                    self.provider, [entry.to_dict()],
                    token_bases=self._persisted_token_pairs,
                )
                self._persisted_token_pairs = auth_mod._token_pairs_by_id(written)
                self._borrowed_root_ids = set()
            else:
                self._persist()
            # Callers print "Added" on return; a row the store did not keep
            # must fail loudly instead.
            if not any(isinstance(row, dict) and row.get("id") == entry.id
                       for row in auth_mod.read_credential_pool(self.provider)):
                self._entries = [e for e in self._entries if e.id != entry.id]
                raise CredentialNotSavedError(
                    f"The {self.provider} credential was not saved: {auth_mod._auth_file_path()} "
                    f"does not contain it after the write, so nothing was added. Check that the file "
                    f"is writable, then run the command again."
                )
            return entry
