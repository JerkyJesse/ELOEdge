"""Build / train NFL model with season regression and Platt fit on startup."""

import os
import logging
from collections import defaultdict

import pandas as pd

from config import GAMES_FILE, load_elo_settings, get_season_label
from elo_model import NFLElo
from data_players import load_player_stats
from platt import regress_ratings_to_mean


def _calc_altitude_bonus(csv_file=GAMES_FILE):
    """Calculate extra home Elo bonus for high-altitude teams from game data.
    Only Denver Broncos (5280 ft) qualifies in the NFL.
    """
    import math
    ALTITUDE_TEAMS = {
        "Denver Broncos": 5280,
    }
    if not os.path.exists(csv_file):
        return {}
    try:
        games = pd.read_csv(csv_file, usecols=["home_team", "home_score", "away_score"])
        games = games.dropna(subset=["home_score", "away_score"])
        if len(games) < 50:
            return {}
        games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)
        league_home_wr = games["home_win"].mean()

        bonuses = {}
        for team, elevation in ALTITUDE_TEAMS.items():
            home_games = games[games["home_team"] == team]
            if len(home_games) < 5:
                continue
            team_home_wr = home_games["home_win"].mean()
            excess = team_home_wr - league_home_wr
            if excess <= 0:
                continue

            def wr_to_elo(wr):
                wr = max(0.01, min(0.99, wr))
                return 400.0 * math.log10(wr / (1.0 - wr))
            elo_bonus = wr_to_elo(team_home_wr) - wr_to_elo(league_home_wr)
            bonuses[team] = round(elo_bonus, 1)
        return bonuses
    except Exception as e:
        logging.warning("Could not calculate altitude bonus: %s", e)
        return {}


def _populate_game_history(model, csv_file=GAMES_FILE):
    """Scan game CSV to populate last game dates, locations, opponent Elos, and scores."""
    if not os.path.exists(csv_file):
        return
    try:
        games = pd.read_csv(csv_file)
        games["date"] = pd.to_datetime(games["date"], errors="coerce")
        games = games.dropna(subset=["date"])
        if games.empty:
            return
        games = games.sort_values("date")

        for _, row in games.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            dt   = row["date"]
            hs   = float(row.get("home_score", 0) or 0)
            as_  = float(row.get("away_score", 0) or 0)

            model._last_game_date[home] = dt
            model._last_game_date[away] = dt
            model._last_game_location[home] = home
            model._last_game_location[away] = home
            model._opponent_elos[home].append(model.ratings[away])
            model._opponent_elos[away].append(model.ratings[home])
            model._team_scores[home].append((hs, as_))
            model._team_scores[away].append((as_, hs))
            if hs > as_:
                model._recent_results[home].append(1.0)
                model._recent_results[away].append(0.0)
            elif as_ > hs:
                model._recent_results[home].append(0.0)
                model._recent_results[away].append(1.0)
            else:
                model._recent_results[home].append(0.5)
                model._recent_results[away].append(0.5)

            # Last margin for mean reversion
            model._last_margin[home] = hs - as_
            model._last_margin[away] = as_ - hs

            # Consecutive home/away tracking
            model._consecutive_home[home] = model._consecutive_home.get(home, 0) + 1
            model._consecutive_away[home] = 0
            model._consecutive_away[away] = model._consecutive_away.get(away, 0) + 1
            model._consecutive_home[away] = 0

            # Game number per team
            model._game_number[home] = model._game_number.get(home, 0) + 1
            model._game_number[away] = model._game_number.get(away, 0) + 1

        for team in list(model._opponent_elos.keys()):
            model._opponent_elos[team] = model._opponent_elos[team][-10:]
        for team in list(model._team_scores.keys()):
            model._team_scores[team] = model._team_scores[team][-10:]
        for team in list(model._recent_results.keys()):
            model._recent_results[team] = model._recent_results[team][-10:]

        logging.info("Game history loaded: %d teams (dates, locations, SOS, pace, form)",
                     len(model._last_game_date))
    except Exception as e:
        logging.warning("Could not load game history: %s", e)


def build_model(csv_file=GAMES_FILE):
    settings  = load_elo_settings()
    _elo_keys = {"base_rating","k","home_adv","use_mov","player_boost",
                  "rest_factor","form_weight","travel_factor","sos_factor",
                  "playoff_hca_factor","pace_factor",
                  "division_factor","mean_reversion",
                  "b2b_penalty","road_trip_factor","homestand_factor",
                  "win_streak_factor","altitude_factor","season_phase_factor",
                  "scoring_consistency_factor","rest_advantage_cap","bye_week_factor"}
    model     = NFLElo(**{k: v for k, v in settings.items() if k in _elo_keys})
    if model.load():
        _populate_game_history(model, csv_file)
        model._altitude_bonus = _calc_altitude_bonus(csv_file)
        player_df = load_player_stats()
        if not player_df.empty:
            model.set_player_stats(player_df)
        return model
    if not os.path.exists(csv_file):
        return model
    logging.info("Training full model...")
    games = pd.read_csv(csv_file)
    if "neutral_site" not in games.columns:
        games["neutral_site"] = False
    game_count = regular_count = playoff_count = 0
    prev_season = None
    for _, row in games.iterrows():
        try:
            row_season = None
            if "date" in row and pd.notna(row["date"]):
                try:
                    dt = pd.to_datetime(row["date"])
                    # NFL season boundary: month >= 9 = new season year
                    row_season = dt.year if dt.month >= 9 else dt.year - 1
                except Exception as e:
                    logging.debug("Season date parse error: %s", e)
            if row_season and prev_season and row_season != prev_season:
                logging.info("Season boundary %d->%d: regressing ratings to mean",
                             prev_season, row_season)
                model.ratings = defaultdict(
                    lambda: model.base_rating,
                    regress_ratings_to_mean(dict(model.ratings), factor=settings.get("season_regress", 0.33))
                )
            if row_season is not None:
                prev_season = row_season
            game_date = None
            if "date" in row and pd.notna(row["date"]):
                try:
                    game_date = pd.to_datetime(row["date"])
                except Exception as e:
                    logging.debug("Game date parse error: %s", e)
            model.update_game(
                row["home_team"], row["away_team"],
                row["home_score"], row["away_score"],
                neutral_site=bool(row["neutral_site"]),
                game_date=game_date,
            )
            game_count += 1
            if bool(row.get("is_playoff", False)):
                playoff_count += 1
            else:
                regular_count += 1
        except Exception as e:
            logging.warning("Training row error: %s", e)
    model.team_names = sorted(model.ratings.keys())
    model._rebuild_lookup()
    model._altitude_bonus = _calc_altitude_bonus(csv_file)
    player_df = load_player_stats()
    if not player_df.empty:
        model.set_player_stats(player_df)
    model.metadata.update({
        "season_label":  get_season_label(),
        "trained_games": int(game_count),
        "regular_games": int(regular_count),
        "playoff_games": int(playoff_count),
        "source_file":   csv_file,
        "settings":      model.settings_dict(),
    })
    model.save()
    model.show_all_teams()
    return model
