"""Historical NBA game backfill via nba_api + existing `Claude/NBA/nba_http.py`.

Iterates `LeagueGameLog` per season (Regular Season + Playoffs) and merges the
results into the upstream CSV. Uses the production HTTP shim with curl_cffi
Chrome TLS impersonation, so Akamai Bot Manager on stats.nba.com is handled.

Delegates actual row parsing to `Claude/NBA/data_games._fetch_and_process_season_type`
(which understands the `LeagueGameLog` row-pair-per-game format), so there's
one source of truth for that logic.

Rate limit posture: per-request timeout 8s × 3 retries ≈ 24s worst case (set
by nba_http.session_get). 10 seasons × 2 season-types = 20 calls ≈ 8 min
worst case, usually much faster when Akamai is happy. No aggressive concurrent
fetching — one endpoint at a time.

Scope: historical games only. Does NOT fetch player stats, injuries, altitude,
etc. Those adjusters in the Tier 2 ablation remain inert until separately
backfilled. Documented in README.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Iterable, List, Tuple

import pandas as pd


_NBA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "NBA"))
if _NBA_DIR not in sys.path:
    sys.path.insert(0, _NBA_DIR)


SEASON_TYPES = ("Regular Season", "Playoffs")
DEFAULT_SEASONS: Tuple[int, ...] = tuple(range(2015, 2025))  # starting years 2015..2024


def _season_label(start_year: int) -> str:
    """NBA API season format: '2022-23' for starting year 2022."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def fetch_season(start_year: int, season_type: str) -> List[dict]:
    """Fetch one season's games via LeagueGameLog. Returns list of canonical-ish
    row dicts (date, home_team, away_team, home_score, away_score, neutral_site).
    Delegates to `data_games._fetch_and_process_season_type` for row shaping."""
    import nba_http  # noqa: F401  # applies the nba_api TLS-impersonation patch
    from data_games import _fetch_and_process_season_type

    # nba_api serializes None as the literal string "None" in query params,
    # which stats.nba.com rejects with 500. Pass empty string to mean "no filter".
    return _fetch_and_process_season_type(
        season_str=_season_label(start_year),
        season_type_str=season_type,
        date_from="",
    )


def backfill(
    seasons: Iterable[int] = DEFAULT_SEASONS,
    season_types: Iterable[str] = SEASON_TYPES,
    pause_seconds: float = 1.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Iterate seasons × season_types and return concatenated DataFrame.

    Pauses `pause_seconds` between fetches to stay polite (Akamai doesn't like
    rapid-fire). Partial progress returned even if a later fetch errors.
    """
    rows: List[dict] = []
    failures: List[Tuple[int, str, str]] = []

    for start_year in seasons:
        for st in season_types:
            label = f"{_season_label(start_year)} {st}"
            try:
                t0 = time.time()
                got = fetch_season(start_year, st)
                dt = time.time() - t0
                rows.extend(got)
                if verbose:
                    print(
                        f"nba_api_backfill: {label}: {len(got)} games in {dt:.1f}s",
                        file=sys.stderr,
                    )
            except Exception as e:
                failures.append((start_year, st, repr(e)))
                if verbose:
                    print(
                        f"nba_api_backfill: {label}: FAILED -- {e}",
                        file=sys.stderr,
                    )
            time.sleep(pause_seconds)

    df = pd.DataFrame(rows)
    if failures and verbose:
        print(
            f"nba_api_backfill: {len(failures)} fetch failures total — "
            f"partial data returned. Retry failing seasons individually.",
            file=sys.stderr,
        )
    if len(df) == 0:
        return df

    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(
        subset=["date", "home_team", "away_team", "home_score", "away_score"]
    ).sort_values("date").reset_index(drop=True)
    return df
