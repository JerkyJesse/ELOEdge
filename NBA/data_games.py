"""Game data download and validation."""

import os
import time
import logging
from datetime import datetime, timedelta

import pandas as pd

from config import GAMES_FILE, get_season_label, is_cache_stale


def validate_games_df(df):
    required = {"date", "home_team", "away_team", "home_score", "away_score", "neutral_site"}
    if df is None or df.empty:
        logging.warning("Games dataframe is empty")
        return False
    if not required.issubset(df.columns):
        logging.warning("Games dataframe missing columns: %s", required - set(df.columns))
        return False
    if len(df) < 200:
        logging.warning("Games dataframe looks small: %d rows", len(df))
        return False
    return True


def _fetch_and_process_season_type(season_str, season_type_str, date_from):
    from nba_api.stats.endpoints import leaguegamelog
    response = leaguegamelog.LeagueGameLog(
        season=season_str, season_type_all_star=season_type_str,
        player_or_team_abbreviation="T", date_from_nullable=date_from,
    )
    df = response.get_data_frames()[0]
    if df.empty:
        return []
    processed = []
    for game_id, group in df.groupby("GAME_ID"):
        if len(group) != 2:
            continue
        row1, row2 = group.iloc[0], group.iloc[1]
        if " @ " in row1["MATCHUP"]:
            away_row, home_row = row1, row2
        elif " @ " in row2["MATCHUP"]:
            away_row, home_row = row2, row1
        else:
            continue
        home_team = home_row.get("TEAM_NAME", "").strip()
        away_team = away_row.get("TEAM_NAME", "").strip()
        if not home_team and " vs " in home_row["MATCHUP"]:
            parts = home_row["MATCHUP"].split(" vs ")
            home_team = parts[0].strip()
            if len(parts) > 1:
                away_team = parts[1].strip()
        if not away_team and " @ " in away_row["MATCHUP"]:
            parts = away_row["MATCHUP"].split(" @ ")
            away_team = parts[0].strip()
            if len(parts) > 1:
                home_team = parts[1].strip()
        if not home_team or not away_team:
            continue
        processed.append({
            "date": pd.to_datetime(row1["GAME_DATE"]),
            "home_team": home_team, "away_team": away_team,
            "home_score": float(home_row["PTS"]),
            "away_score": float(away_row["PTS"]),
            "neutral_site": False,
        })
    return processed


def download_recent_games(csv_file=GAMES_FILE):
    if not is_cache_stale(csv_file):
        logging.info("Using cached %s (%d KB)", csv_file, os.path.getsize(csv_file) // 1024)
        return csv_file

    # --- Incremental download: load existing data and only fetch new games ---
    existing_df = None
    last_date_str = None
    if os.path.exists(csv_file) and os.path.getsize(csv_file) > 500:
        try:
            existing_df = pd.read_csv(csv_file)
            existing_df["date"] = pd.to_datetime(existing_df["date"], errors="coerce")
            last_date = existing_df["date"].max()
            if pd.notna(last_date):
                last_date_str = last_date.strftime("%Y-%m-%d")
                logging.info("Existing data has %d games through %s — fetching only newer",
                             len(existing_df), last_date_str)
        except Exception as e:
            logging.warning("Could not read existing file for incremental update: %s", e)
            existing_df = None

    date_from = last_date_str if last_date_str else (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    season_str = get_season_label()
    logging.info("Downloading NBA games from %s via nba_api LeagueGameLog...", date_from)
    try:
        all_games = []
        all_games.extend(_fetch_and_process_season_type(season_str, "Regular Season", date_from))
        all_games.extend(_fetch_and_process_season_type(season_str, "Playoffs", date_from))
        if not all_games and existing_df is None:
            logging.warning("No games retrieved.")
            return csv_file if os.path.exists(csv_file) else None
        new_df = pd.DataFrame(all_games) if all_games else pd.DataFrame()
        # Merge with existing data
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
                    .drop_duplicates(subset=["date","home_team","away_team","home_score","away_score"])
                    .sort_values("date").reset_index(drop=True))
        if not validate_games_df(games_df):
            return csv_file if os.path.exists(csv_file) else None
        n_new = len(games_df) - n_before
        games_df.to_csv(csv_file, index=False)
        logging.info("Total %d games (%d new) -> %s", len(games_df), max(0, n_new), csv_file)
        return csv_file
    except Exception as e:
        logging.error("LeagueGameLog download failed: %s", e, exc_info=True)
        return csv_file if os.path.exists(csv_file) else None
