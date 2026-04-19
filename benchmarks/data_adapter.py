"""NBA data adapter: Claude/NBA/nba_recent_games.csv -> canonical schema (§4b).

Canonical columns: date, season, home, away, home_score, away_score.

Season convention (NBA standard, starting year): Oct-Dec -> year;
Jan-Jun -> year-1. Example: 2022-23 season == season 2022.
A game on 2022-10-20 -> season 2022. A game on 2023-04-10 -> season 2022.

Source file is populated by Claude/NBA/data_games.py from nba_api LeagueGameLog
through the Claude/NBA/nba_http.py session (curl_cffi Chrome TLS impersonation).
This adapter does NOT refetch; it consumes whatever rows are cached on disk.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import pandas as pd


NBA_GAMES_CSV = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "NBA", "nba_recent_games.csv")
)


def season_from_date(date: Any) -> int:
    """NBA starting-year season from a single date. Oct+ -> year, else year-1."""
    d = pd.to_datetime(date)
    return int(d.year if d.month >= 10 else d.year - 1)


def _season_series(dates: pd.Series) -> pd.Series:
    d = pd.to_datetime(dates)
    return (d.dt.year.where(d.dt.month >= 10, d.dt.year - 1)).astype(int)


def load_nba_games(
    csv_path: str = NBA_GAMES_CSV,
    drop_neutral: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load and canonicalize the NBA games CSV.

    Parameters
    ----------
    csv_path : path to `nba_recent_games.csv`.
    drop_neutral : drop `neutral_site == True` rows if column present.
    verbose : emit a row-count / date-range summary to stderr.

    Returns
    -------
    DataFrame, sorted ascending by date, columns:
      date (datetime64), season (int), home (str), away (str),
      home_score (int), away_score (int)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"NBA games CSV not found at {csv_path}. "
            "Run Claude/NBA/data_games.py::download_recent_games first."
        )
    raw = pd.read_csv(csv_path)
    required = {"date", "home_team", "away_team", "home_score", "away_score"}
    missing = required - set(raw.columns)
    if missing:
        raise KeyError(f"CSV missing columns: {missing}")

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["date"], errors="coerce"),
            "home": raw["home_team"].astype(str).str.strip(),
            "away": raw["away_team"].astype(str).str.strip(),
            "home_score": pd.to_numeric(raw["home_score"], errors="coerce"),
            "away_score": pd.to_numeric(raw["away_score"], errors="coerce"),
        }
    )
    if drop_neutral and "neutral_site" in raw.columns:
        neutral = raw["neutral_site"].astype(str).str.lower().isin(("true", "1", "yes"))
        df = df[~neutral.values].copy()

    before = len(df)
    df = df.dropna(subset=["date", "home", "away", "home_score", "away_score"]).copy()
    dropped = before - len(df)
    if dropped and verbose:
        print(
            f"data_adapter.load_nba_games: dropped {dropped} rows with null critical fields",
            file=sys.stderr,
        )

    df["season"] = _season_series(df["date"]).values
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    df = df[["date", "season", "home", "away", "home_score", "away_score"]]
    df = df.sort_values("date").reset_index(drop=True)

    if verbose:
        print(
            f"data_adapter.load_nba_games: {len(df)} rows, "
            f"seasons {sorted(df['season'].unique().tolist())}, "
            f"dates {df['date'].min().date()} -> {df['date'].max().date()}",
            file=sys.stderr,
        )
    return df
