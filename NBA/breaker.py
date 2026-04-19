"""File-backed circuit breaker that survives CLI restarts.

A solo CLI tool restarts on every invocation. In-memory pybreaker loses its
state every run, making reset_timeout meaningless. This module persists
breaker state in a JSON file so consecutive failures across runs actually
open the circuit.

State file: Claude/NBA/.breaker_state.json
Shape: {"<source_name>": {"consecutive_fails": int,
                           "last_fail_ts": float|null,
                           "opened_at": float|null}}

API
---
    from breaker import Breaker, BreakerOpen
    b = Breaker("nba_api")
    try:
        b.before_call()
        # ... do the network IO ...
        b.on_success()
    except BreakerOpen:
        # use fallback
    except Exception:
        b.on_fail()
        raise

Atomic writes (tmp file + os.replace) so a crashed run cannot corrupt state.
Single-writer assumption — two concurrent CLI runs against the same NBA dir
is out of scope.
"""

import json
import logging
import os
import time


_STATE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".breaker_state.json",
)

DEFAULT_FAIL_MAX = 3
DEFAULT_RESET_TIMEOUT = 300  # seconds


class BreakerOpen(Exception):
    """Raised by Breaker.before_call when the circuit is open."""


def _load_state():
    if not os.path.exists(_STATE_FILE):
        return {}
    try:
        with open(_STATE_FILE, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return data
    except (OSError, ValueError) as e:
        logging.warning("breaker: could not read %s (%s). Starting fresh.",
                        _STATE_FILE, e)
        return {}


def _save_state(state):
    tmp = _STATE_FILE + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, _STATE_FILE)
    except OSError as e:
        logging.warning("breaker: could not write %s (%s).", _STATE_FILE, e)
        try:
            os.remove(tmp)
        except OSError:
            pass


class Breaker:
    """Per-source circuit breaker with persistent state."""

    def __init__(self, source, fail_max=DEFAULT_FAIL_MAX,
                 reset_timeout=DEFAULT_RESET_TIMEOUT):
        self.source = source
        self.fail_max = int(fail_max)
        self.reset_timeout = int(reset_timeout)

    def _get(self):
        state = _load_state()
        return state.get(self.source, {
            "consecutive_fails": 0,
            "last_fail_ts": None,
            "opened_at": None,
        })

    def _put(self, entry):
        state = _load_state()
        state[self.source] = entry
        _save_state(state)

    def is_open(self):
        entry = self._get()
        opened_at = entry.get("opened_at")
        if opened_at is None:
            return False
        if (time.time() - float(opened_at)) >= self.reset_timeout:
            # Cooldown elapsed — half-open-like: let next call through.
            return False
        return True

    def before_call(self):
        if self.is_open():
            remaining = self.reset_timeout - (time.time() - float(self._get()["opened_at"]))
            raise BreakerOpen(
                "breaker [%s] open for another %ds" % (self.source, int(max(0, remaining)))
            )

    def on_success(self):
        entry = self._get()
        if entry.get("consecutive_fails", 0) != 0 or entry.get("opened_at") is not None:
            logging.info("breaker [%s]: recovered (resetting state)", self.source)
        self._put({
            "consecutive_fails": 0,
            "last_fail_ts": None,
            "opened_at": None,
        })

    def on_fail(self):
        entry = self._get()
        fails = int(entry.get("consecutive_fails", 0)) + 1
        now = time.time()
        opened_at = entry.get("opened_at")
        if fails >= self.fail_max and opened_at is None:
            opened_at = now
            logging.warning(
                "breaker [%s]: OPENED after %d consecutive fails (cooldown %ds)",
                self.source, fails, self.reset_timeout,
            )
        self._put({
            "consecutive_fails": fails,
            "last_fail_ts": now,
            "opened_at": opened_at,
        })

    def status(self):
        """Return a dict snapshot for display (health command)."""
        entry = self._get()
        return {
            "source": self.source,
            "consecutive_fails": int(entry.get("consecutive_fails", 0)),
            "last_fail_ts": entry.get("last_fail_ts"),
            "opened_at": entry.get("opened_at"),
            "is_open": self.is_open(),
        }


def reset(source):
    """Manually clear breaker state for a source. Useful for CLI debugging."""
    state = _load_state()
    if source in state:
        del state[source]
        _save_state(state)
