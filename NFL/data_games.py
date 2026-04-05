"""NFL game data download via ESPN public API."""

import os
import time
import logging
from datetime import datetime

import requests
import pandas as pd

from config import GAMES_FILE, get_current_season_year, is_cache_stale


def validate_games_df(df):
    required = {"date", "home_team", "away_team", "home_score", "away_score", "neutral_site"}
    if df is None or df.empty:
        logging.warning("Games dataframe is empty")
        return False
    if not required.issubset(df.columns):
        logging.warning("Games dataframe missing columns: %s", required - set(df.columns))
        return False
    if len(df) < 50:
        logging.warning("Games dataframe looks small: %d rows", len(df))
        return False
    return True


def _fetch_espn_daterange(date_start, date_end):
    """Fetch NFL games from ESPN scoreboard API using date range.

    ESPN's NFL scoreboard supports ?dates=YYYYMMDD-YYYYMMDD format.
    """
    url = ("https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
           "?dates=%s-%s&limit=200" % (date_start, date_end))
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logging.warning("ESPN fetch failed (%s to %s): %s", date_start, date_end, e)
        return []

    games = []
    for event in data.get("events", []):
        for comp in event.get("competitions", []):
            status = comp.get("status", {}).get("type", {})
            if not status.get("completed", False):
                continue
            neutral = comp.get("neutralSite", False)
            competitors = comp.get("competitors", [])
            if len(competitors) != 2:
                continue
            home_team = away_team = None
            home_score = away_score = 0
            for c in competitors:
                team_name = c.get("team", {}).get("displayName", "")
                score = int(c.get("score", 0) or 0)
                if c.get("homeAway") == "home":
                    home_team = team_name
                    home_score = score
                else:
                    away_team = team_name
                    away_score = score
            if home_team and away_team:
                # Skip Pro Bowl / All-Star games
                if home_team in ("AFC", "NFC") or away_team in ("AFC", "NFC"):
                    continue
                game_date = event.get("date", "")
                # Parse ISO date
                try:
                    dt = datetime.fromisoformat(game_date.replace("Z", "+00:00"))
                    date_str = dt.strftime("%Y-%m-%d")
                except Exception:
                    date_str = game_date[:10] if len(game_date) >= 10 else game_date
                games.append({
                    "date": date_str,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_score": float(home_score),
                    "away_score": float(away_score),
                    "neutral_site": neutral,
                })
    return games


def download_recent_games(csv_file=GAMES_FILE):
    """Download NFL games for current and previous season from ESPN."""
    if not is_cache_stale(csv_file):
        logging.info("Using cached %s (%d KB)", csv_file, os.path.getsize(csv_file) // 1024)
        return csv_file

    # --- Incremental download: load existing data and find last date ---
    existing_df = None
    last_date_dt = None
    if os.path.exists(csv_file) and os.path.getsize(csv_file) > 500:
        try:
            existing_df = pd.read_csv(csv_file)
            existing_df["date"] = pd.to_datetime(existing_df["date"], errors="coerce")
            last_date_dt = existing_df["date"].max()
            if pd.notna(last_date_dt):
                logging.info("Existing data has %d games through %s — fetching only newer",
                             len(existing_df), last_date_dt.strftime("%Y-%m-%d"))
        except Exception as e:
            logging.warning("Could not read existing file for incremental update: %s", e)
            existing_df = None

    logging.info("Downloading recent NFL games via ESPN scoreboard API...")
    current_season = get_current_season_year()
    seasons_to_fetch = [current_season - 1, current_season]

    all_games = []
    for season_year in seasons_to_fetch:
        months = [
            ("%d0901" % season_year,     "%d0930" % season_year),
            ("%d1001" % season_year,     "%d1031" % season_year),
            ("%d1101" % season_year,     "%d1130" % season_year),
            ("%d1201" % season_year,     "%d1231" % season_year),
            ("%d0101" % (season_year+1), "%d0131" % (season_year+1)),
            ("%d0201" % (season_year+1), "%d0228" % (season_year+1)),
        ]
        for date_start, date_end in months:
            # Skip months entirely before last known date
            if last_date_dt is not None:
                try:
                    end_dt = datetime.strptime(date_end, "%Y%m%d")
                    if end_dt < last_date_dt:
                        continue
                except ValueError:
                    pass
            games = _fetch_espn_daterange(date_start, date_end)
            all_games.extend(games)
            time.sleep(0.5)

    if not all_games and existing_df is None:
        logging.warning("No games retrieved from ESPN.")
        return csv_file if os.path.exists(csv_file) else None

    try:
        new_df = pd.DataFrame(all_games) if all_games else pd.DataFrame()
        if existing_df is not None and not new_df.empty:
            new_df["date"] = pd.to_datetime(new_df["date"], errors="coerce")
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            n_before = len(existing_df)
        elif existing_df is not None:
            combined = existing_df
            n_before = len(existing_df)
        else:
            combined = new_df
            n_before = 0
        games_df = (combined
                    .drop_duplicates(subset=["date", "home_team", "away_team", "home_score", "away_score"])
                    .sort_values("date").reset_index(drop=True))
        if not validate_games_df(games_df):
            return csv_file if os.path.exists(csv_file) else None
        n_new = len(games_df) - n_before
        games_df.to_csv(csv_file, index=False)
        logging.info("Total %d NFL games (%d new) -> %s", len(games_df), max(0, n_new), csv_file)
        return csv_file
    except Exception as e:
        logging.error("Game processing failed: %s", e, exc_info=True)
        return csv_file if os.path.exists(csv_file) else None
