"""Caller-driven software watchdog (FR-07, PRD §11.1/§11.2). No threads, no wall clock.

Contracts:
- Time is INJECTED by the caller as monotonic nanoseconds; the watchdog never reads a clock,
  which makes expiry fully deterministic and testable.
- Fail-safe: a watchdog that has never been fed is expired.
- Expiry never auto-resets; only ``feed()`` re-arms.
- On timeout the decision is HOLD or STOP (config ``timeout_policy``) — never "keep executing
  the stale chunk" (PRD §11.1 Recovery).
"""

from __future__ import annotations

from enum import Enum

from wam.interfaces import SafetyIntervention
from wam.safety.config import SafetyConfig

_NS_PER_S = 1_000_000_000


class WatchdogAction(str, Enum):
    """Deterministic recovery decision on watchdog expiry."""

    HOLD = "hold"
    STOP = "stop"


class Watchdog:
    """Chunk-timeout watchdog. ``feed(now_ns)`` on every accepted chunk/heartbeat;
    ``expired(now_ns)``/``decide(now_ns)`` are checked by the control loop."""

    def __init__(self, timeout_s: float, action: WatchdogAction = WatchdogAction.HOLD) -> None:
        if not timeout_s > 0:
            raise ValueError(f"timeout_s must be > 0, got {timeout_s}")
        self._timeout_ns = round(timeout_s * _NS_PER_S)
        self._action = action
        self._last_feed_ns: int | None = None

    @classmethod
    def from_config(cls, config: SafetyConfig) -> Watchdog:
        return cls(
            timeout_s=config.chunk_timeout_s,
            action=WatchdogAction(config.timeout_policy),
        )

    @property
    def timeout_ns(self) -> int:
        return self._timeout_ns

    @property
    def action(self) -> WatchdogAction:
        return self._action

    @property
    def last_feed_ns(self) -> int | None:
        """Timestamp of the last feed; None if never fed (== expired)."""
        return self._last_feed_ns

    def feed(self, now_ns: int) -> None:
        """Re-arm the watchdog at ``now_ns`` (caller-provided monotonic time)."""
        self._last_feed_ns = int(now_ns)

    def expired(self, now_ns: int) -> bool:
        """True iff never fed, or more than ``timeout_ns`` elapsed since the last feed.
        Exactly at the deadline is NOT expired."""
        if self._last_feed_ns is None:
            return True
        return int(now_ns) - self._last_feed_ns > self._timeout_ns

    def decide(self, now_ns: int) -> WatchdogAction | None:
        """Configured HOLD/STOP action if expired, else None."""
        return self._action if self.expired(now_ns) else None

    def intervention(self, now_ns: int) -> SafetyIntervention | None:
        """Loggable intervention record if expired (FR-07: every event is logged), else None."""
        if not self.expired(now_ns):
            return None
        since = (
            "never fed"
            if self._last_feed_ns is None
            else (f"last feed {int(now_ns) - self._last_feed_ns} ns ago")
        )
        return SafetyIntervention(
            kind="watchdog_timeout",
            detail=f"timeout {self._timeout_ns} ns exceeded ({since}); decision={self._action.value}",
            timestamp_ns=int(now_ns),
        )
