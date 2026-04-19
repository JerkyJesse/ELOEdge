"""ESPN hidden JSON adapter for NBA game logs.

Why ESPN:
    - No API key, no account, no cursor pagination.
    - Same backend that feeds espn.com (always up).
    - Returns finalized games in clean JSON per-date query.
    - Stable schema; survives redesigns of espn.com frontend.

Endpoint:
    https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=YYYYMMDD

Response shape (relevant fields):
    events[].competitions[0].competitors[] — two entries (home/away)
        .homeAway: "home" | "away"
        .team.displayName: "Los Angeles Lakers"
        .score: "110"
    events[].competitions[0].status.type.completed: bool
    events[].competitions[0].status.type.name: "STATUS_FINAL" | ...
    events[].competitions[0].neutralSite: bool (sometimes)
    events[].competitions[0].venue.fullName

Strategy:
    Iterate day-by-day across the requested range. Each day = 1 HTTP call.
    Results cached for 1h (see nba_http._URLS_EXPIRE_AFTER). No rate limit
    documented; we self-throttle at 10 req/sec conservatively to avoid
    surprise 429s — a 730-day cold backfill completes in ~75 seconds.

Team name normalization:
    ESPN team displayNames match config.TEAM_ABBR keys directly for all 30
    franchises today. Two historical edge cases mapped defensively below.
"""

import logging
import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd

try:
    from .. import nba_http  # type: ignore
    from . import FallbackUnavailable, GAMES_CANONICAL_COLUMNS
except Exception:  # pragma: no cover — flat layout
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import nba_http
    from sources import FallbackUnavailable, GAMES_CANONICAL_COLUMNS


_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
_MIN_INTERVAL_S = 0.1   # 10 req/sec ceiling (conservative)

# ESPN displayName -> config.TEAM_ABBR full name.
# The 30 current franchises match directly; fix anything that drifts.
_TEAM_NAME_FIX = {
    "LA Clippers":        "Los Angeles Clippers",
    "LA Lakers":          "Los Angeles Lakers",
}


def _normalize_team_name(raw):
    if not raw:
        return ""
    return _TEAM_NAME_FIX.get(raw.strip(), raw.strip())


def _fetch_day(date_obj):
    """Fetch ESPN scoreboard for a single date. Returns list of game dicts."""
    params = {"dates": date_obj.strftime("%Y%m%d")}
    try:
        resp = nba_http.session_get(nba_http.espn_session, _BASE, params=params)
    except Exception as e:
        raise FallbackUnavailable("ESPN network error on %s: %s"
                                  % (date_obj.strftime("%Y-%m-%d"), e))
    if resp.status_code != 200:
        raise FallbackUnavailable(
            "ESPN HTTP %d on %s" % (resp.status_code, date_obj.strftime("%Y-%m-%d"))
        )
    try:
        data = resp.json()
    except ValueError as e:
        raise FallbackUnavailable("ESPN returned non-JSON: %s" % e)
    return data.get("events", []) or []


def _parse_event(event):
    """Map one ESPN event into a GAMES_CANONICAL_COLUMNS row.

    Returns None if the game is not finalized or data is incomplete.
    """
    comps = event.get("competitions") or []
    if not comps:
        return None
    comp = comps[0]

    status = (comp.get("status") or {}).get("type") or {}
    if not status.get("completed", False):
        return None

    competitors = comp.get("competitors") or []
    if len(competitors) != 2:
        return None

    home = away = None
    for c in competitors:
        side = c.get("homeAway", "").lower()
        if side == "home":
            home = c
        elif side == "away":
            away = c
    if home is None or away is None:
        return None

    home_name = _normalize_team_name((home.get("team") or {}).get("displayName", ""))
    away_name = _normalize_team_name((away.get("team") or {}).get("displayName", ""))
    if not home_name or not away_name:
        return None

    try:
        home_score = float(home.get("score", 0))
        away_score = float(away.get("score", 0))
    except (TypeError, ValueError):
        return None

    # ESPN timestamps are ISO-8601 UTC ("2026-04-17T23:00Z"). Convert to US/Eastern
    # (NBA canonical tz so late-night West-coast games don't roll to next day),
    # then strip tz and normalize — matches naive Timestamps produced by the
    # nba_api path so downstream pandas sort/merge does not crash on tz mismatch.
    raw_date = event.get("date") or comp.get("date") or ""
    try:
        ts = pd.to_datetime(raw_date)
        if getattr(ts, "tz", None) is not None:
            ts = ts.tz_convert("US/Eastern").tz_localize(None)
        game_date = ts.normalize()
    except (TypeError, ValueError):
        return None

    neutral = bool(comp.get("neutralSite", False))

    return {
        "date":         game_date,
        "home_team":    home_name,
        "away_team":    away_name,
        "home_score":   home_score,
        "away_score":   away_score,
        "neutral_site": neutral,
    }


def get_game_logs(start_date, end_date):
    """Fetch finalized games between start_date and end_date (YYYY-MM-DD, inclusive).

    Returns DataFrame with GAMES_CANONICAL_COLUMNS, same schema as
    data_games._fetch_and_process_season_type.

    Raises FallbackUnavailable on unrecoverable error.
    """
    try:
        d0 = datetime.strptime(start_date, "%Y-%m-%d").date()
        d1 = datetime.strptime(end_date,   "%Y-%m-%d").date()
    except ValueError as e:
        raise FallbackUnavailable("ESPN bad date range (%s -> %s): %s"
                                  % (start_date, end_date, e))
    if d1 < d0:
        raise FallbackUnavailable("ESPN end_date before start_date")

    total_days = (d1 - d0).days + 1
    if total_days > 30:
        logging.info("[source=espn] backfilling %d days (~%.1fs at 10 req/sec)",
                     total_days, total_days * _MIN_INTERVAL_S)

    rows = []
    failures = 0
    max_consecutive_fails = 5
    consecutive = 0
    current = d0
    last_request = 0.0

    while current <= d1:
        elapsed = time.time() - last_request
        if elapsed < _MIN_INTERVAL_S:
            time.sleep(_MIN_INTERVAL_S - elapsed)
        last_request = time.time()

        try:
            events = _fetch_day(current)
            consecutive = 0
        except FallbackUnavailable as e:
            failures += 1
            consecutive += 1
            logging.debug("ESPN day %s failed: %s", current, e)
            if consecutive >= max_consecutive_fails:
                raise FallbackUnavailable(
                    "ESPN: %d consecutive day fetches failed (last: %s)"
                    % (consecutive, e)
                )
            current += timedelta(days=1)
            continue

        for ev in events:
            row = _parse_event(ev)
            if row is not None:
                rows.append(row)
        current += timedelta(days=1)

    if not rows:
        raise FallbackUnavailable(
            "ESPN returned zero finalized games for %s..%s (%d day fetch failures)"
            % (start_date, end_date, failures)
        )

    df = pd.DataFrame(rows, columns=GAMES_CANONICAL_COLUMNS)
    df = df.drop_duplicates(subset=["date", "home_team", "away_team"]).reset_index(drop=True)
    if failures:
        logging.info("[source=espn] %d day fetches failed, %d games captured",
                     failures, len(df))
    return df
