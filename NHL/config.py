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

SETTINGS_FILE          = "nhl_elo_settings.json"
PREDICTS_FILE = "predicts_lots.csv"
PLAYER_STATS_FILE      = "nhl_player_stats.csv"
GAMES_FILE             = "nhl_recent_games.csv"
RATINGS_FILE           = "nhl_elo_ratings.json"
ADVANCED_STATS_FILE    = "nhl_advanced_stats.csv"
PLATT_SCALER_FILE      = "nhl_platt_scaler.json"
CACHE_MAX_AGE_HOURS    = 6

TEAM_ABBR = {
    "Anaheim Ducks": "ANA",         "Boston Bruins": "BOS",
    "Buffalo Sabres": "BUF",        "Calgary Flames": "CGY",
    "Carolina Hurricanes": "CAR",   "Chicago Blackhawks": "CHI",
    "Colorado Avalanche": "COL",    "Columbus Blue Jackets": "CBJ",
    "Dallas Stars": "DAL",          "Detroit Red Wings": "DET",
    "Edmonton Oilers": "EDM",       "Florida Panthers": "FLA",
    "Los Angeles Kings": "LA",      "Minnesota Wild": "MIN",
    "Montreal Canadiens": "MTL",    "Nashville Predators": "NSH",
    "New Jersey Devils": "NJ",      "New York Islanders": "NYI",
    "New York Rangers": "NYR",      "Ottawa Senators": "OTT",
    "Philadelphia Flyers": "PHI",   "Pittsburgh Penguins": "PIT",
    "San Jose Sharks": "SJ",        "Seattle Kraken": "SEA",
    "St. Louis Blues": "STL",       "Tampa Bay Lightning": "TB",
    "Toronto Maple Leafs": "TOR",   "Utah Hockey Club": "UTA",
    "Vancouver Canucks": "VAN",     "Vegas Golden Knights": "VGK",
    "Washington Capitals": "WSH",   "Winnipeg Jets": "WPG",
}

NHL_DIVISIONS = {
    "Atlantic": ["Boston Bruins", "Buffalo Sabres", "Detroit Red Wings", "Florida Panthers", "Montreal Canadiens", "Ottawa Senators", "Tampa Bay Lightning", "Toronto Maple Leafs"],
    "Metropolitan": ["Carolina Hurricanes", "Columbus Blue Jackets", "New Jersey Devils", "New York Islanders", "New York Rangers", "Philadelphia Flyers", "Pittsburgh Penguins", "Washington Capitals"],
    "Central": ["Chicago Blackhawks", "Colorado Avalanche", "Dallas Stars", "Minnesota Wild", "Nashville Predators", "St. Louis Blues", "Utah Hockey Club", "Winnipeg Jets"],
    "Pacific": ["Anaheim Ducks", "Calgary Flames", "Edmonton Oilers", "Los Angeles Kings", "San Jose Sharks", "Seattle Kraken", "Vancouver Canucks", "Vegas Golden Knights"],
}


def same_division(team_a, team_b):
    for div_teams in NHL_DIVISIONS.values():
        if team_a in div_teams and team_b in div_teams:
            return True
    return False


# Map our abbreviations to NHL API abbreviations (where they differ)
NHL_API_ABBR = {
    "LA": "LAK", "NJ": "NJD", "SJ": "SJS", "TB": "TBL",
}


def get_team_abbr(full_name):
    return TEAM_ABBR.get(str(full_name).strip(), str(full_name)[:3].upper())


def get_current_season_year():
    now = datetime.now()
    return now.year + 1 if 10 <= now.month <= 12 else now.year


def get_season_label(year=None):
    year = year or get_current_season_year()
    return "%d-%s" % (year - 1, str(year)[-2:])


def current_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def is_cache_stale(filepath, max_age_hours=CACHE_MAX_AGE_HOURS, data_type="games"):
    """Check if cached data needs refreshing. Uses smart season-aware logic when available."""
    if HAS_SMART_CACHE:
        return smart_cache_stale(filepath, "nhl", data_type, max_age_hours=max_age_hours)
    # Fallback: simple age check
    # Files under 500 bytes are empty/corrupt stubs (valid CSV needs headers + rows)
    if not os.path.exists(filepath) or os.path.getsize(filepath) < 500:
        return True
    age_hours = (datetime.now().timestamp() - os.path.getmtime(filepath)) / 3600
    return age_hours > max_age_hours


def load_elo_settings(filename=SETTINGS_FILE):
    defaults = {
        "base_rating": 1500.0, "k": 5.0, "home_adv": 26.0,
        "use_mov": True, "autoresolve_enabled": False,
        "player_boost": 10.0, "starter_boost": 5.0,
        "rest_factor": 12.0, "form_weight": 4.0,
        "travel_factor": 8.0, "sos_factor": 10.0,
        "playoff_hca_factor": 1.0, "pace_factor": 10.0,
        "starting_balance": 50.0, "kelly_fraction": 0.25,
        "auto_kalshi": False,
        "division_factor": 5.0, "mean_reversion": 2.5,
        "b2b_penalty": 18.0, "road_trip_factor": 2.5,
        "homestand_factor": 3.0, "win_streak_factor": 2.0,
        "altitude_factor": 4.0, "season_phase_factor": 2.5,
        "scoring_consistency_factor": 1.5, "rest_advantage_cap": 1.5,
        "overtime_factor": 5.0,
        "season_regress": 0.33,
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
