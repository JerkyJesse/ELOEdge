"""Constants, team abbreviations, settings I/O, and shared helpers."""

import os
import sys
import json
import logging
from datetime import datetime

try:
    from cache_utils import smart_cache_stale, get_cache_age_str
    HAS_SMART_CACHE = True
except ImportError:
    HAS_SMART_CACHE = False

SETTINGS_FILE          = "nfl_elo_settings.json"
PREDICTS_FILE          = "predicts_lots.csv"
PLAYER_STATS_FILE      = "nfl_player_stats.csv"
GAMES_FILE             = "nfl_recent_games.csv"
RATINGS_FILE           = "nfl_elo_ratings.json"
ADVANCED_STATS_FILE    = "nfl_advanced_stats.csv"
PLATT_SCALER_FILE      = "nfl_platt_scaler.json"
CACHE_MAX_AGE_HOURS    = 6

TEAM_ABBR = {
    "Arizona Cardinals": "ARI",    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",     "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",   "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",       "Denver Broncos": "DEN",
    "Detroit Lions": "DET",        "Green Bay Packers": "GB",
    "Houston Texans": "HOU",       "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",     "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR",     "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",    "New England Patriots": "NE",
    "New Orleans Saints": "NO",    "New York Giants": "NYG",
    "New York Jets": "NYJ",        "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",  "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",     "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",     "Washington Commanders": "WSH",
}

NFL_DIVISIONS = {
    "AFC East": ["Buffalo Bills", "Miami Dolphins", "New England Patriots", "New York Jets"],
    "AFC North": ["Baltimore Ravens", "Cincinnati Bengals", "Cleveland Browns", "Pittsburgh Steelers"],
    "AFC South": ["Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars", "Tennessee Titans"],
    "AFC West": ["Denver Broncos", "Kansas City Chiefs", "Las Vegas Raiders", "Los Angeles Chargers"],
    "NFC East": ["Dallas Cowboys", "New York Giants", "Philadelphia Eagles", "Washington Commanders"],
    "NFC North": ["Chicago Bears", "Detroit Lions", "Green Bay Packers", "Minnesota Vikings"],
    "NFC South": ["Atlanta Falcons", "Carolina Panthers", "New Orleans Saints", "Tampa Bay Buccaneers"],
    "NFC West": ["Arizona Cardinals", "Los Angeles Rams", "San Francisco 49ers", "Seattle Seahawks"],
}


def same_division(team_a, team_b):
    for div_teams in NFL_DIVISIONS.values():
        if team_a in div_teams and team_b in div_teams:
            return True
    return False


def get_team_abbr(full_name):
    return TEAM_ABBR.get(str(full_name).strip(), str(full_name)[:3].upper())


def get_current_season_year():
    """Return the year the current NFL season started.
    NFL seasons span two calendar years (Sep-Feb).
    If month >= 9, the season started this year.
    If month <= 8, the season started last year (we are in playoffs or off-season).
    """
    now = datetime.now()
    return now.year if now.month >= 9 else now.year - 1


def get_season_label(year=None):
    """Return the NFL season label, e.g. '2025' for the 2025-26 season."""
    if year is None:
        year = get_current_season_year()
    return str(year)


def current_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def is_cache_stale(filepath, max_age_hours=CACHE_MAX_AGE_HOURS, data_type="games"):
    """Check if cached data needs refreshing. Uses smart season-aware logic when available."""
    if HAS_SMART_CACHE:
        return smart_cache_stale(filepath, "nfl", data_type, max_age_hours=max_age_hours)
    # Fallback: simple age check
    if not os.path.exists(filepath) or os.path.getsize(filepath) < 500:
        return True
    age_hours = (datetime.now().timestamp() - os.path.getmtime(filepath)) / 3600
    return age_hours > max_age_hours


def load_elo_settings(filename=SETTINGS_FILE):
    defaults = {
        "base_rating": 1500.0, "k": 35.0, "home_adv": 28.0,
        "use_mov": True, "autoresolve_enabled": False,
        "player_boost": 30.0, "rest_factor": 6.0, "form_weight": 10.0,
        "travel_factor": 7.0, "sos_factor": 5.0,
        "playoff_hca_factor": 0.75, "pace_factor": 7.0,
        "division_factor": 30.0, "mean_reversion": 2.5,
        "b2b_penalty": 10.0, "road_trip_factor": 2.0,
        "homestand_factor": 17.0, "win_streak_factor": 18.0,
        "altitude_factor": 4.0, "season_phase_factor": 17.0,
        "scoring_consistency_factor": 2.0, "rest_advantage_cap": 4.0,
        "bye_week_factor": 10.0,
        "season_regress": 0.33,
        "starting_balance": 50.0, "kelly_fraction": 0.25,
        "auto_kalshi": False,
    }
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                saved = json.load(f)
            defaults.update(saved)
            logging.info("Loaded settings: K=%s, HomeAdv=%s", defaults["k"], defaults["home_adv"])
        except Exception as e:
            logging.warning("Settings load failed: %s", e)
    else:
        logging.info("Using default settings")
    return defaults


def save_elo_settings(settings, filename=SETTINGS_FILE):
    try:
        with open(filename, "w") as f:
            json.dump(settings, f, indent=2)
        logging.info("Saved settings -> %s", filename)
    except Exception as e:
        logging.warning("Save failed: %s", e)
