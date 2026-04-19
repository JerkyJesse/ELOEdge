"""Import Basketball-Reference schedule CSVs into the canonical games frame.

Basketball-Reference exposes per-season schedule CSVs via the "Share & Export
→ Get table as CSV" action on:
    https://www.basketball-reference.com/leagues/NBA_{END_YEAR}_games.html
    (END_YEAR = season ending year; 2022-23 season -> NBA_2023_games.html)

The exported CSV has columns like:
    Date,Start (ET),Visitor/Neutral,PTS,Home/Neutral,PTS,,,Attend.,LOG,Arena,Notes
Date format: "Tue Oct 18, 2022". The second "PTS" column is the home score.

This importer:
  * parses date strings
  * maps home/visitor -> canonical home/away columns
  * adds `season` (starting year, per data_adapter convention)
  * merges across multiple per-season CSVs + dedupes
  * optionally writes the result to Claude/NBA/nba_recent_games.csv so all
    downstream harness machinery (data_adapter, splits, ablation) just works.

Historical team rename aliases are applied so BR's "Charlotte Bobcats",
"New Orleans Hornets", "Seattle SuperSonics", etc. get normalized to the
current franchise name (matching what Claude/NBA/data_games.py writes).
"""

from __future__ import annotations

import os
import sys
from typing import Iterable, List

import pandas as pd


HISTORICAL_TEAM_ALIASES = {
    "Charlotte Bobcats": "Charlotte Hornets",
    "New Orleans Hornets": "New Orleans Pelicans",
    "New Orleans/Oklahoma City Hornets": "New Orleans Pelicans",
    "Seattle SuperSonics": "Oklahoma City Thunder",
    "New Jersey Nets": "Brooklyn Nets",
    "Los Angeles Clippers": "LA Clippers",  # BR uses full LA; NBA API uses "LA Clippers"
}


def _normalize_team(name: str) -> str:
    name = str(name).strip()
    return HISTORICAL_TEAM_ALIASES.get(name, name)


def _season_from_start_date(dates: pd.Series) -> pd.Series:
    """Starting-year season: Oct-Dec -> year, Jan-Jun -> year-1 (matches data_adapter)."""
    d = pd.to_datetime(dates)
    return d.dt.year.where(d.dt.month >= 10, d.dt.year - 1).astype(int)


def parse_bref_csv(path: str) -> pd.DataFrame:
    """Parse one Basketball-Reference schedule CSV -> canonical games frame.

    Returns DataFrame columns: date, season, home, away, home_score, away_score.
    Rows with missing scores (future games on an in-progress season) are dropped.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    raw = pd.read_csv(path)

    def _find(*candidates: str) -> str:
        for c in candidates:
            if c in raw.columns:
                return c
        raise KeyError(
            f"no column matching any of {candidates} in {path}. "
            f"got columns: {list(raw.columns)}"
        )

    date_col = _find("Date")
    visitor_col = _find("Visitor/Neutral", "Visitor")
    home_col = _find("Home/Neutral", "Home")
    pts_cols = [c for c in raw.columns if c == "PTS" or c.startswith("PTS")]
    if len(pts_cols) < 2:
        raise KeyError(
            f"BR CSV should have two PTS columns (visitor then home). "
            f"got: {list(raw.columns)}"
        )
    visitor_pts_col, home_pts_col = pts_cols[0], pts_cols[1]

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_col], errors="coerce"),
            "away": raw[visitor_col].map(_normalize_team),
            "home": raw[home_col].map(_normalize_team),
            "away_score": pd.to_numeric(raw[visitor_pts_col], errors="coerce"),
            "home_score": pd.to_numeric(raw[home_pts_col], errors="coerce"),
        }
    )
    df = df.dropna(subset=["date", "home", "away", "home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["season"] = _season_from_start_date(df["date"]).values

    return df[["date", "season", "home", "away", "home_score", "away_score"]].sort_values(
        "date"
    ).reset_index(drop=True)


def parse_many(paths: Iterable[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for p in paths:
        f = parse_bref_csv(p)
        print(
            f"bref_import: {os.path.basename(p)} -> {len(f)} games, "
            f"seasons {sorted(f['season'].unique().tolist())}",
            file=sys.stderr,
        )
        frames.append(f)
    if not frames:
        return pd.DataFrame(
            columns=["date", "season", "home", "away", "home_score", "away_score"]
        )
    return pd.concat(frames, ignore_index=True)


def merge_with_existing(new_df: pd.DataFrame, existing_csv: str) -> pd.DataFrame:
    """Concatenate newly-parsed BR frame with existing nba_recent_games.csv.
    Dedupes on (date, home, away, home_score, away_score). Re-writes CSV in the
    original upstream format so Claude/NBA/data_games.py remains compatible."""
    if os.path.exists(existing_csv) and os.path.getsize(existing_csv) > 500:
        existing = pd.read_csv(existing_csv)
        existing["date"] = pd.to_datetime(existing["date"], errors="coerce")
        # Upstream uses "home_team" / "away_team"; normalize to common keys
        rename = {"home_team": "home", "away_team": "away"}
        existing = existing.rename(columns=rename)
    else:
        existing = pd.DataFrame(
            columns=["date", "home", "away", "home_score", "away_score", "neutral_site"]
        )

    # Harmonize columns for concat
    for col in ("neutral_site",):
        if col not in new_df.columns:
            new_df = new_df.assign(neutral_site=False)

    combined = pd.concat(
        [existing[["date", "home", "away", "home_score", "away_score", "neutral_site"]],
         new_df[["date", "home", "away", "home_score", "away_score", "neutral_site"]]],
        ignore_index=True,
    )
    combined = combined.drop_duplicates(
        subset=["date", "home", "away", "home_score", "away_score"]
    ).sort_values("date").reset_index(drop=True)
    return combined


def write_upstream_csv(combined: pd.DataFrame, csv_path: str) -> None:
    """Write out to the upstream (Claude/NBA/) column layout: date, home_team,
    away_team, home_score, away_score, neutral_site."""
    out = combined.rename(columns={"home": "home_team", "away": "away_team"})
    out = out[["date", "home_team", "away_team", "home_score", "away_score", "neutral_site"]]
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(csv_path, index=False)
