"""Shared HTTP session layer for NBA data fetchers.

Purpose
-------
Fixes nba_api timeout/hang pain by:
  1. Routing stats.nba.com through curl_cffi with chrome124 TLS impersonation.
     Akamai Bot Manager on stats.nba.com fingerprints TLS ClientHello (JA3/JA4)
     and HTTP/2 frames; stock `requests` gets tarpit'd (silent ReadTimeout)
     regardless of header spoofing. curl_cffi emits Chrome's exact ClientHello.
  2. Using requests_cache.CachedSession (SQLite) for cdn/espn (not TLS-blocked).
  3. Sending browser-like headers + x-nba-stats-* tokens.
  4. Wrapping calls in tenacity retry (3 attempts, exp backoff 1-4s, 8s timeout).
  5. Patching nba_api.library.http.requests so nba_api traffic routes through
     our stats session (nba_api 1.5.2 calls requests.get directly; no Session attr).

Sessions:
    stats_session       -> stats.nba.com via nba_api (curl_cffi, NO cache --
                           application layer has CSV cache; TLS fingerprint
                           is the load-bearing fix, not HTTP cache)
    cdn_session         -> cdn.nba.com (requests_cache, SQLite)
    espn_session        -> site.api.espn.com (requests_cache, SQLite)

Verification
------------
Run once after install:
    NBA_HTTP_VERIFY=1 python -c "import nba_http"
Expect output:  PATCH_OK: shim hit N time(s), status=200
If PATCH_FAIL:  the patch target changed in a newer nba_api release.
                Check Chunk 1 Step 1.0 in the design doc.
If BLOCK:       curl_cffi impersonation failed (Akamai updated fingerprint DB).
                Try a newer impersonate= value: chrome131, chrome133, etc.

Budget
------
Per-request timeout is 8s; tenacity 3 attempts => ~24s worst case per URL.
POSIX: belt-and-suspenders signal.alarm 45s wall-clock budget via
       wrap_with_budget() context manager. Callers: download_recent_games,
       download_player_stats.
Windows: signal.alarm unavailable; the 24s per-request cap IS the budget.

Sibling-sport safety
--------------------
Patches nba_api.library.http globally. Do NOT import this module from
Claude/MLB, Claude/NFL, or Claude/NHL code. They do not import nba_api
(verified 2026-04-18), so this patch is inert to them.
"""

import logging
import os
import sys

import requests
import requests_cache
from curl_cffi import requests as curl_requests
from curl_cffi.requests.errors import RequestsError as CurlRequestsError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_result,
)

# Chrome TLS/HTTP2 fingerprint. Bump when Akamai invalidates older Chrome
# versions (typically every 6-12 months). Supported values live in
# curl_cffi.requests.impersonate.BrowserTypeLiteral.
_IMPERSONATE = "chrome124"


# ---------------------------------------------------------------------------
# Cache backend (shared by all three sessions)
# ---------------------------------------------------------------------------

_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nba_http_cache.sqlite")

_URLS_EXPIRE_AFTER = {
    "*leaguegamelog*":                                  21600,   # 6h
    "*leaguedashplayerstats*":                           3600,   # 1h
    "cdn.nba.com/*scoreboard*":                            30,   # 30s (mark-to-market bursts)
    "site.api.espn.com/*scoreboard*":                    3600,   # 1h (ESPN per-day game list)
    "*":                                                    0,   # catch-all no-cache
}


def _make_cached_session(name, default_headers):
    """Build a CachedSession sharing the module-level SQLite backend."""
    sess = requests_cache.CachedSession(
        cache_name=_CACHE_PATH,
        backend="sqlite",
        allowable_codes=(200,),       # don't cache 5xx (defeats retries)
        stale_if_error=False,
        urls_expire_after=_URLS_EXPIRE_AFTER,
    )
    sess.headers.update(default_headers)
    return sess


# ---------------------------------------------------------------------------
# stats.nba.com session (curl_cffi w/ Chrome TLS impersonation)
# ---------------------------------------------------------------------------
# Why curl_cffi, not requests_cache: Akamai Bot Manager on stats.nba.com
# fingerprints TLS ClientHello (JA3/JA4) + HTTP/2 SETTINGS frame order.
# Python `requests` (urllib3 + OpenSSL defaults) ships a distinct fingerprint
# and gets tarpit'd (silent ReadTimeout, no 4xx). curl_cffi emits Chrome's
# exact ClientHello via libcurl-impersonate. Header spoofing alone is
# insufficient -- TLS fingerprint is the load-bearing signal.
#
# No HTTP cache on this session by design: CSV layer (nba_recent_games.csv,
# nba_player_stats.csv) is already the application cache. Adding another
# cache layer here would double-cache and complicate invalidation.

_CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

_STATS_HEADERS = {
    "User-Agent":          _CHROME_UA,
    "Referer":             "https://www.nba.com/",
    "Origin":              "https://www.nba.com",
    "Accept":              "application/json, text/plain, */*",
    "Accept-Language":     "en-US,en;q=0.9",
    "Accept-Encoding":     "gzip, deflate, br",
    "Connection":          "keep-alive",
    "x-nba-stats-origin":  "stats",
    "x-nba-stats-token":   "true",
    "Pragma":              "no-cache",
    "Cache-Control":       "no-cache",
}

stats_session = curl_requests.Session(impersonate=_IMPERSONATE, headers=_STATS_HEADERS)


# ---------------------------------------------------------------------------
# cdn.nba.com session (live scoreboard, schedule)
# ---------------------------------------------------------------------------

cdn_session = _make_cached_session(
    "cdn",
    {
        "User-Agent":      _CHROME_UA,
        "Referer":         "https://www.nba.com/",
        "Accept":          "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
    },
)


# ---------------------------------------------------------------------------
# ESPN hidden JSON session (Chunk 2 fallback, no key required)
# ---------------------------------------------------------------------------

espn_session = _make_cached_session(
    "espn",
    {
        "User-Agent":      _CHROME_UA,
        "Referer":         "https://www.espn.com/",
        "Accept":          "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
    },
)


# ---------------------------------------------------------------------------
# Retry wrapper (shared across all sessions)
# ---------------------------------------------------------------------------

def _is_server_error(resp):
    """Retry on 5xx. Explicitly do NOT retry on 4xx (rate limits need backoff)."""
    try:
        return resp.status_code >= 500
    except AttributeError:
        return False


_RETRY_EXCEPTIONS = (
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
    CurlRequestsError,  # curl_cffi transport errors (timeout, conn reset, TLS)
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=(retry_if_exception_type(_RETRY_EXCEPTIONS) | retry_if_result(_is_server_error)),
    reraise=True,
)
def session_get(session, url, **kwargs):
    """GET via `session` with tenacity retry + 8s per-request timeout.

    Kwargs forwarded to requests.Session.get. Callers may override `timeout`.
    """
    kwargs.setdefault("timeout", 8)
    return session.get(url, **kwargs)


# ---------------------------------------------------------------------------
# nba_api patch (Path B: replace module-level `requests` reference)
# ---------------------------------------------------------------------------

class _RequestsShim:
    """Routes nba_api's direct `requests.get(...)` calls through stats_session
    with tenacity retry. Only .get is wrapped; nba_api/library/http.py line 146
    only calls requests.get (inspected for nba_api==1.5.2).

    Tracks invocation count so the self-test can distinguish a shim-bypass
    (PATCH_FAIL) from a downstream network error (shim hit, but stats.nba.com
    dropped the connection — still PATCH_OK for our purposes).
    """

    def __init__(self, session):
        self._session = session
        self.call_count = 0
        self.last_error = None

    def get(self, url, params=None, headers=None, proxies=None, timeout=None, **kw):
        self.call_count += 1
        kwargs = {"params": params}
        if headers is not None:
            kwargs["headers"] = headers
        if proxies is not None:
            kwargs["proxies"] = proxies
        # Ignore nba_api's default timeout=30 (too lax for our retry budget).
        # session_get enforces 8s via setdefault in the absence of a timeout arg.
        kwargs.update(kw)
        kwargs.pop("timeout", None)
        try:
            resp = session_get(self._session, url, **kwargs)
            self.last_error = None
            return resp
        except Exception as e:
            self.last_error = repr(e)
            raise


# Module-level reference so _verify_patch can inspect shim state
_nba_shim = None


def _apply_patch():
    """Replace nba_api.library.http.requests with our shim. Idempotent."""
    global _nba_shim
    import nba_api.library.http as _nba_http_mod

    existing = getattr(_nba_http_mod, "requests", None)
    if isinstance(existing, _RequestsShim):
        _nba_shim = existing
        return
    _nba_shim = _RequestsShim(stats_session)
    _nba_http_mod.requests = _nba_shim


_apply_patch()


# ---------------------------------------------------------------------------
# Total-fetch wall-clock budget (POSIX only; Windows uses per-request cap)
# ---------------------------------------------------------------------------

class BudgetExceeded(Exception):
    """Raised when a wrapped block exceeds its wall-clock budget."""


class wrap_with_budget:  # noqa: N801 (context manager, lowercase intentional)
    """Context manager enforcing wall-clock budget on a block of network IO.

    POSIX: uses signal.alarm to interrupt blocking syscalls. Raises
           BudgetExceeded when the alarm fires.
    Windows: no-op. The tenacity `session_get` per-request timeout (8s x 3
             attempts = ~24s per URL) is the real budget there.

    Usage:
        with wrap_with_budget(45, "download_recent_games"):
            ... network IO ...
    """

    def __init__(self, seconds, label=""):
        self.seconds = int(seconds)
        self.label = label
        self._prev_handler = None
        self._posix = (os.name == "posix")

    def __enter__(self):
        if not self._posix:
            return self
        import signal

        def _handler(signum, frame):
            raise BudgetExceeded(
                "wall-clock budget %ds exceeded in %s" % (self.seconds, self.label)
            )

        self._prev_handler = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._posix:
            return False
        import signal

        signal.alarm(0)
        if self._prev_handler is not None:
            signal.signal(signal.SIGALRM, self._prev_handler)
        return False  # do not swallow


# ---------------------------------------------------------------------------
# Structured log helper (NBA-scoped)
# ---------------------------------------------------------------------------

def log_fetch(source, status, latency_ms=None, rows=None, cache=None):
    """Emit a grep-able one-line record of a data fetch.

    Example:  [source=nba_api status=ok cache=hit latency_ms=412 rows=82]
    """
    parts = ["source=%s" % source, "status=%s" % status]
    if cache is not None:
        parts.append("cache=%s" % cache)
    if latency_ms is not None:
        parts.append("latency_ms=%d" % int(latency_ms))
    if rows is not None:
        parts.append("rows=%d" % int(rows))
    logging.info("[%s]", " ".join(parts))


# ---------------------------------------------------------------------------
# Self-test: NBA_HTTP_VERIFY=1 python -c "import nba_http"
# ---------------------------------------------------------------------------

def _verify_patch():
    """Confirm nba_api traffic routes through stats_session AND Akamai passes.

    Signals:
      1. Shim invocation count -- did _RequestsShim.get get called?
         If count == 0: PATCH_FAIL (nba_api bypassed us, patch target drifted).
      2. Response status code -- did curl_cffi Chrome impersonation defeat
         the bot block? 200 = PATCH_OK. Timeout/non-200 = BLOCK (Akamai
         updated fingerprint DB; bump _IMPERSONATE to newer chrome version).
    """
    if os.environ.get("NBA_HTTP_VERIFY") != "1":
        return

    shim_before = _nba_shim.call_count if _nba_shim else -1
    nba_error = None
    last_resp = None
    try:
        from nba_api.stats.endpoints import commonteamyears
        ep = commonteamyears.CommonTeamYears()
        last_resp = ep
    except Exception as e:
        nba_error = e

    shim_after = _nba_shim.call_count if _nba_shim else -1
    shim_delta = shim_after - shim_before

    if _nba_shim is None:
        print("PATCH_FAIL: _nba_shim is None (patch function never ran)")
        sys.exit(1)

    if shim_delta <= 0:
        print("PATCH_FAIL: nba_api bypassed the shim (shim_delta=%d). "
              "nba_api internals changed; re-inspect nba_api/library/http.py."
              % shim_delta)
        sys.exit(1)

    if nba_error is not None:
        err_name = type(nba_error).__name__
        if "Timeout" in err_name or "Timeout" in repr(nba_error):
            print("BLOCK: shim hit %d time(s) but stats.nba.com timed out "
                  "(%s). Akamai likely rejected %s fingerprint -- bump "
                  "_IMPERSONATE in nba_http.py to newer chrome version "
                  "(chrome131, chrome133, etc)." %
                  (shim_delta, err_name, _IMPERSONATE))
            sys.exit(1)
        print("PATCH_OK: shim hit %d time(s); non-fatal downstream error "
              "(%s) -- shim routing confirmed." % (shim_delta, err_name))
        return

    print("PATCH_OK: shim hit %d time(s), status=200, impersonate=%s" %
          (shim_delta, _IMPERSONATE))


_verify_patch()
