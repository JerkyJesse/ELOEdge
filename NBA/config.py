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

SETTINGS_FILE          = "nba_elo_settings.json"
PREDICTS_FILE = "predicts_lots.csv"
PLAYER_STATS_FILE      = "nba_player_stats.csv"
GAMES_FILE             = "nba_recent_games.csv"
RATINGS_FILE           = "nba_elo_ratings.json"
ADVANCED_STATS_FILE    = "nba_advanced_stats.csv"
PLATT_SCALER_FILE      = "nba_platt_scaler.json"
CACHE_MAX_AGE_HOURS    = 6

TEAM_ABBR = {
    "Atlanta Hawks": "ATL",        "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",        "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",        "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",     "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",      "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",      "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "LA Clippers": "LAC",          "LA Lakers": "LAL",
    "Memphis Grizzlies": "MEM",    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",      "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC","Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",   "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR","Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",            "Washington Wizards": "WAS",
}

NBA_DIVISIONS = {
    "Atlantic": ["Boston Celtics", "Brooklyn Nets", "New York Knicks", "Philadelphia 76ers", "Toronto Raptors"],
    "Central": ["Chicago Bulls", "Cleveland Cavaliers", "Detroit Pistons", "Indiana Pacers", "Milwaukee Bucks"],
    "Southeast": ["Atlanta Hawks", "Charlotte Hornets", "Miami Heat", "Orlando Magic", "Washington Wizards"],
    "Northwest": ["Denver Nuggets", "Minnesota Timberwolves", "Oklahoma City Thunder", "Portland Trail Blazers", "Utah Jazz"],
    "Pacific": ["Golden State Warriors", "LA Clippers", "Los Angeles Lakers", "Phoenix Suns", "Sacramento Kings"],
    "Southwest": ["Dallas Mavericks", "Houston Rockets", "Memphis Grizzlies", "New Orleans Pelicans", "San Antonio Spurs"],
}


def same_division(team_a, team_b):
    for div_teams in NBA_DIVISIONS.values():
        if team_a in div_teams and team_b in div_teams:
            return True
    return False


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
        return smart_cache_stale(filepath, "nba", data_type, max_age_hours=max_age_hours)
    # Fallback: simple age check
    # Files under 500 bytes are empty/corrupt stubs (valid CSV needs headers + rows)
    if not os.path.exists(filepath) or os.path.getsize(filepath) < 500:
        return True
    age_hours = (datetime.now().timestamp() - os.path.getmtime(filepath)) / 3600
    return age_hours > max_age_hours


def load_elo_settings(filename=SETTINGS_FILE):
    defaults = {
        "base_rating": 1500.0, "k": 8.23, "home_adv": 34.0,
        "use_mov": True, "autoresolve_enabled": False,
        "player_boost": 35.0, "rest_factor": 25.0, "form_weight": 10.0,
        "travel_factor": 28.2, "sos_factor": 0.0,
        "playoff_hca_factor": 0.57, "pace_factor": 35.0,
        "starting_balance": 0.0, "kelly_fraction": 0.50,
        "auto_kalshi": False,
        "division_factor": 10.0, "mean_reversion": 0.0,
        "b2b_penalty": 0.0, "road_trip_factor": 3.47,
        "homestand_factor": 0.0, "win_streak_factor": 0.0,
        "altitude_factor": 0.0, "season_phase_factor": 0.0,
        "scoring_consistency_factor": 0.0, "rest_advantage_cap": 2.97,
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
