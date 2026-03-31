"""Game data download and validation via ESPN public API + NHL API for goalie data."""

import os
import time
import logging
from datetime import datetime, timedelta

import requests
import pandas as pd

from config import GAMES_FILE, TEAM_ABBR, get_season_label, is_cache_stale

ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
NHL_API_BASE = "https://api-web.nhle.com/v1"

# Map our full team names to NHL API abbreviations
_TEAM_TO_NHL_ABBR = {v: v for _, v in TEAM_ABBR.items()}
# Also build reverse: NHL abbr -> our full name
_NHL_ABBR_TO_FULL = {}
for full_name, abbr in TEAM_ABBR.items():
    _NHL_ABBR_TO_FULL[abbr] = full_name


def validate_games_df(df):
    required = {"date", "home_team", "away_team", "home_score", "away_score", "neutral_site"}
    if df is None or df.empty:
        logging.warning("Games dataframe is empty")
        return False
    if not required.issubset(df.columns):
        logging.warning("Games dataframe missing columns: %s", required - set(df.columns))
        return False
    if len(df) < 100:
        logging.warning("Games dataframe looks small: %d rows", len(df))
        return False
    return True


def _fetch_games_for_date(date_str):
    """Fetch completed NHL games for a single date from ESPN scoreboard API."""
    url = "%s?dates=%s" % (ESPN_SCOREBOARD_URL, date_str)
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logging.debug("ESPN fetch failed for %s: %s", date_str, e)
        return []

    games = []
    for event in data.get("events", []):
        for comp in event.get("competitions", []):
            status = comp.get("status", {}).get("type", {})
            if not status.get("completed", False):
                continue
            neutral = comp.get("neutralSite", False)
            home_team = away_team = ""
            home_score = away_score = 0
            for competitor in comp.get("competitors", []):
                team_name = competitor.get("team", {}).get("displayName", "")
                score = int(competitor.get("score", 0) or 0)
                if competitor.get("homeAway") == "home":
                    home_team = team_name
                    home_score = score
                else:
                    away_team = team_name
                    away_score = score
            if home_team and away_team:
                game_date = event.get("date", "")[:10]
                games.append({
                    "date": pd.to_datetime(game_date),
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_score": float(home_score),
                    "away_score": float(away_score),
                    "neutral_site": neutral,
                })
    return games


def _fetch_games_for_range(date_start, date_end):
    """Fetch completed NHL games for a date range (YYYYMMDD format)."""
    url = "%s?dates=%s-%s&limit=200" % (ESPN_SCOREBOARD_URL, date_start, date_end)
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
            home_team = away_team = ""
            home_score = away_score = 0
            for competitor in comp.get("competitors", []):
                team_name = competitor.get("team", {}).get("displayName", "")
                score = int(competitor.get("score", 0) or 0)
                if competitor.get("homeAway") == "home":
                    home_team = team_name
                    home_score = score
                else:
                    away_team = team_name
                    away_score = score
            if home_team and away_team:
                game_date = event.get("date", "")[:10]
                games.append({
                    "date": pd.to_datetime(game_date),
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_score": float(home_score),
                    "away_score": float(away_score),
                    "neutral_site": neutral,
                })
    return games


def _get_nhl_seasons_for_range(start_date, end_date):
    """Return NHL season codes (e.g., '20242025') covering the date range."""
    seasons = set()
    d = start_date
    while d <= end_date:
        if d.month >= 10:
            seasons.add("%d%d" % (d.year, d.year + 1))
        elif d.month <= 6:
            seasons.add("%d%d" % (d.year - 1, d.year))
        # July-September: offseason, skip
        d += timedelta(days=32)
        d = d.replace(day=1)
    return sorted(seasons)


def _fetch_goalie_starts(start_date, end_date):
    """Fetch starting goalie for each game from NHL API goalie game logs.
    Returns dict: (date_str, team_abbr) -> goalie_full_name"""
    seasons = _get_nhl_seasons_for_range(start_date, end_date)
    if not seasons:
        return {}
    logging.info("Fetching goalie data for seasons: %s", seasons)

    goalie_map = {}  # (date_str, team_abbr) -> goalie_name
    seen_goalies = set()  # avoid duplicate API calls
    team_abbrs = sorted(set(TEAM_ABBR.values()))

    for season in seasons:
        logging.info("  Season %s: fetching rosters...", season)
        for abbr in team_abbrs:
            try:
                url = "%s/roster/%s/%s" % (NHL_API_BASE, abbr, season)
                resp = requests.get(url, timeout=15)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                for g in data.get("goalies", []):
                    pid = g.get("id")
                    if pid and pid not in seen_goalies:
                        seen_goalies.add(pid)
                        fn = g.get("firstName", {}).get("default", "")
                        ln = g.get("lastName", {}).get("default", "")
                        goalie_name = "%s %s" % (fn, ln) if fn and ln else ""
                        if not goalie_name:
                            continue
                        # Fetch game log
                        try:
                            gl_url = "%s/player/%s/game-log/%s/2" % (NHL_API_BASE, pid, season)
                            gl_resp = requests.get(gl_url, timeout=15)
                            if gl_resp.status_code != 200:
                                continue
                            gl_data = gl_resp.json()
                            for entry in gl_data.get("gameLog", []):
                                gdate = str(entry.get("gameDate", ""))[:10]
                                team = entry.get("teamAbbrev", "")
                                if gdate and team:
                                    goalie_map[(gdate, team)] = goalie_name
                            time.sleep(0.2)
                        except Exception:
                            pass
                time.sleep(0.15)
            except Exception as e:
                logging.debug("Roster fetch failed for %s/%s: %s", abbr, season, e)

    logging.info("  Goalie map: %d (date, team) entries from %d goalies",
                 len(goalie_map), len(seen_goalies))
    return goalie_map


def _merge_goalie_data(games_df, goalie_map):
    """Add home_starter and away_starter columns to games dataframe."""
    home_starters = []
    away_starters = []
    matched = 0
    for _, row in games_df.iterrows():
        date_str = str(row["date"])[:10]
        home_full = row["home_team"]
        away_full = row["away_team"]
        home_abbr = TEAM_ABBR.get(home_full, home_full[:3].upper())
        away_abbr = TEAM_ABBR.get(away_full, away_full[:3].upper())
        h_goalie = goalie_map.get((date_str, home_abbr), "")
        a_goalie = goalie_map.get((date_str, away_abbr), "")
        home_starters.append(h_goalie)
        away_starters.append(a_goalie)
        if h_goalie or a_goalie:
            matched += 1
    games_df["home_starter"] = home_starters
    games_df["away_starter"] = away_starters
    logging.info("  Goalie data matched for %d/%d games (%.1f%%)",
                 matched, len(games_df), 100 * matched / max(len(games_df), 1))
    return games_df


def download_recent_games(csv_file=GAMES_FILE):
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

    logging.info("Downloading recent NHL games via ESPN scoreboard API...")
    try:
        all_games = []
        end_date = datetime.now()
        # If we have existing data, only fetch from last known date onward
        if last_date_dt is not None:
            start_date = last_date_dt.to_pydatetime() - timedelta(days=1)
        else:
            start_date = end_date - timedelta(days=730)

        current = start_date.replace(day=1)
        while current <= end_date:
            year, month = current.year, current.month
            if month == 12:
                next_month = current.replace(year=year + 1, month=1, day=1)
            else:
                next_month = current.replace(month=month + 1, day=1)
            last_day = (next_month - timedelta(days=1)).day
            date_start = "%d%02d01" % (year, month)
            date_end = "%d%02d%02d" % (year, month, last_day)
            games = _fetch_games_for_range(date_start, date_end)
            all_games.extend(games)
            logging.info("  %s-%s: %d games", date_start, date_end, len(games))
            current = next_month
            time.sleep(0.5)

        if not all_games and existing_df is None:
            logging.warning("No games retrieved.")
            return csv_file if os.path.exists(csv_file) else None

        new_df = pd.DataFrame(all_games) if all_games else pd.DataFrame()

        # Fetch goalie data only for newly fetched games
        if not new_df.empty:
            new_df["date"] = pd.to_datetime(new_df["date"], errors="coerce")
            goalie_start = start_date
            goalie_end = end_date
            logging.info("Fetching starting goalie data from NHL API...")
            try:
                goalie_map = _fetch_goalie_starts(goalie_start, goalie_end)
                if goalie_map:
                    new_df = _merge_goalie_data(new_df, goalie_map)
                else:
                    if "home_starter" not in new_df.columns:
                        new_df["home_starter"] = ""
                        new_df["away_starter"] = ""
            except Exception as e:
                logging.warning("Goalie data fetch failed: %s", e)
                if "home_starter" not in new_df.columns:
                    new_df["home_starter"] = ""
                    new_df["away_starter"] = ""

        # Merge with existing data
        if existing_df is not None and not new_df.empty:
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            n_before = len(existing_df)
        elif existing_df is not None:
            combined = existing_df
            n_before = len(existing_df)
        else:
            combined = new_df
            n_before = 0
            if "home_starter" not in combined.columns:
                combined["home_starter"] = ""
                combined["away_starter"] = ""

        games_df = (combined
                    .drop_duplicates(subset=["date", "home_team", "away_team", "home_score", "away_score"])
                    .sort_values("date").reset_index(drop=True))
        if not validate_games_df(games_df):
            return csv_file if os.path.exists(csv_file) else None
        n_new = len(games_df) - n_before
        games_df.to_csv(csv_file, index=False)
        logging.info("Total %d games (%d new) -> %s", len(games_df), max(0, n_new), csv_file)
        return csv_file
    except Exception as e:
        logging.error("ESPN game download failed: %s", e, exc_info=True)
        return csv_file if os.path.exists(csv_file) else None
