"""NFLElo model class."""

import os
import json
import math
import logging
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
from difflib import get_close_matches

from config import (
    TEAM_ABBR, RATINGS_FILE,
    get_season_label, current_timestamp,
)
from color_helpers import cok, cwarn, cdim, chi, div, hdr, cred, cgrn
from platt import load_platt_scaler, apply_platt

# Timezone offset (hours from UTC) for each team's home city
TEAM_TIMEZONE = {
    "Arizona Cardinals": -7,       "Atlanta Falcons": -5,
    "Baltimore Ravens": -5,        "Buffalo Bills": -5,
    "Carolina Panthers": -5,       "Chicago Bears": -6,
    "Cincinnati Bengals": -5,      "Cleveland Browns": -5,
    "Dallas Cowboys": -6,          "Denver Broncos": -7,
    "Detroit Lions": -5,           "Green Bay Packers": -6,
    "Houston Texans": -6,          "Indianapolis Colts": -5,
    "Jacksonville Jaguars": -5,    "Kansas City Chiefs": -6,
    "Las Vegas Raiders": -8,       "Los Angeles Chargers": -8,
    "Los Angeles Rams": -8,        "Miami Dolphins": -5,
    "Minnesota Vikings": -6,       "New England Patriots": -5,
    "New Orleans Saints": -6,      "New York Giants": -5,
    "New York Jets": -5,           "Philadelphia Eagles": -5,
    "Pittsburgh Steelers": -5,     "San Francisco 49ers": -8,
    "Seattle Seahawks": -8,        "Tampa Bay Buccaneers": -5,
    "Tennessee Titans": -6,        "Washington Commanders": -5,
}


class NFLElo:
    def __init__(self, base_rating=1500.0, k=28.36, home_adv=25.55,
                 use_mov=True, player_boost=24.61, rest_factor=0.68,
                 form_weight=20.0, travel_factor=0.0, sos_factor=0.0,
                 playoff_hca_factor=1.1, pace_factor=0.0,
                 division_factor=31.33, mean_reversion=0.0,
                 b2b_penalty=5.45, road_trip_factor=0.0,
                 homestand_factor=20.0, win_streak_factor=20.0,
                 altitude_factor=0.66, season_phase_factor=20.0,
                 scoring_consistency_factor=0.0, rest_advantage_cap=3.32,
                 bye_week_factor=0.0):
        self.base_rating   = base_rating
        self.k             = k
        self.home_adv      = home_adv
        self.use_mov       = use_mov
        self.player_boost  = player_boost
        self.rest_factor   = rest_factor
        self.form_weight   = form_weight
        self.travel_factor   = travel_factor
        self.sos_factor      = sos_factor
        self.playoff_hca_factor = playoff_hca_factor
        self.pace_factor     = pace_factor
        self.division_factor = division_factor
        self.mean_reversion  = mean_reversion
        self.b2b_penalty     = b2b_penalty      # extra penalty for short-week games (Thursday)
        self.road_trip_factor = road_trip_factor  # penalty for extended road trips
        self.homestand_factor = homestand_factor  # bonus for extended homestands
        self.win_streak_factor = win_streak_factor  # momentum from win/loss streaks
        self.altitude_factor = altitude_factor    # multiplier on altitude bonus (0=raw)
        self.season_phase_factor = season_phase_factor  # early-season dampener
        self.scoring_consistency_factor = scoring_consistency_factor  # penalty for volatile scoring
        self.rest_advantage_cap = rest_advantage_cap  # max rest days counted (0=uncapped)
        self.bye_week_factor = bye_week_factor    # bonus for teams coming off bye week
        self.ratings       = defaultdict(lambda: base_rating)
        self.team_names    = []
        self._team_lookup  = {}
        self._player_scores = {}
        self._last_margin  = {}
        self._last_game_date = {}
        self._recent_results = defaultdict(list)
        self._altitude_bonus = {}
        self._last_game_location = {}
        self._opponent_elos = defaultdict(list)
        self._team_scores = defaultdict(list)
        self._consecutive_away = defaultdict(int)  # team -> current away game streak
        self._consecutive_home = defaultdict(int)  # team -> current home game streak
        self._game_number = defaultdict(int)       # team -> games played this season
        self._platt_scaler = load_platt_scaler()
        self._xgb_model = None
        self._xgb_meta  = None
        self._mega_predictor = None  # MegaPredictor (31-model ensemble)
        self.metadata = {
            "season_label": get_season_label(), "trained_games": 0,
            "saved_at": None, "source_file": None, "settings": self.settings_dict(),
        }

    def settings_dict(self):
        return {
            "base_rating": self.base_rating, "k": self.k,
            "home_adv": self.home_adv, "use_mov": self.use_mov,
            "player_boost": self.player_boost,
            "rest_factor": self.rest_factor,
            "form_weight": self.form_weight,
            "travel_factor": self.travel_factor,
            "sos_factor": self.sos_factor,
            "playoff_hca_factor": self.playoff_hca_factor,
            "pace_factor": self.pace_factor,
            "division_factor": self.division_factor,
            "mean_reversion": self.mean_reversion,
            "b2b_penalty": self.b2b_penalty,
            "road_trip_factor": self.road_trip_factor,
            "homestand_factor": self.homestand_factor,
            "win_streak_factor": self.win_streak_factor,
            "altitude_factor": self.altitude_factor,
            "season_phase_factor": self.season_phase_factor,
            "scoring_consistency_factor": self.scoring_consistency_factor,
            "rest_advantage_cap": self.rest_advantage_cap,
            "bye_week_factor": self.bye_week_factor,
        }

    def set_player_stats(self, player_df):
        from data_players import build_league_player_scores
        self._player_scores = build_league_player_scores(player_df)
        if self.player_boost > 0 and len(self._player_scores) < 20:
            logging.warning(
                "player_boost=%.1f but only %d teams have scores -- "
                "consider set boost=0 until data is stable.",
                self.player_boost, len(self._player_scores)
            )

    def _rebuild_lookup(self):
        self._team_lookup = {t.lower(): t for t in self.team_names}

    def rest_days(self, team, game_date):
        """Days since team's last game. Returns None if unknown."""
        last = self._last_game_date.get(team)
        if last is None or game_date is None:
            return None
        delta = (game_date - last).days
        return max(0, delta)

    def rest_adjustment(self, team, game_date):
        """Elo adjustment for rest. NFL rest is huge -- Thursday games get big penalty.
        Normal rest (7 days) = 0 adjustment. Short rest (4 days/Thursday) = penalty.
        Long rest (10+ days, bye week) = bonus."""
        if self.rest_factor == 0:
            return 0.0
        rd = self.rest_days(team, game_date)
        if rd is None:
            return 0.0
        cap = int(self.rest_advantage_cap) if self.rest_advantage_cap > 0 else 14
        # Center at 7 days (normal NFL week). <7 = penalty, >7 = bonus (capped)
        adj = self.rest_factor * (min(rd, cap) - 7) / 7.0
        return adj

    def form_adjustment(self, team):
        """Elo adjustment based on recent form (last 5 games win%)."""
        if self.form_weight == 0:
            return 0.0
        results = self._recent_results.get(team, [])
        if len(results) < 3:
            return 0.0
        win_pct = sum(results[-5:]) / len(results[-5:])
        return self.form_weight * (win_pct - 0.5)

    def travel_adjustment(self, team, venue_team):
        """Elo penalty for timezone crossings since last game."""
        if self.travel_factor == 0:
            return 0.0
        last_loc = self._last_game_location.get(team)
        if last_loc is None:
            return 0.0
        prev_tz = TEAM_TIMEZONE.get(last_loc, -6)
        curr_tz = TEAM_TIMEZONE.get(venue_team, -6)
        tz_diff = abs(curr_tz - prev_tz)
        if tz_diff == 0:
            return 0.0
        return -self.travel_factor * tz_diff

    def sos_adjustment(self, team):
        """Elo adjustment for strength of schedule."""
        if self.sos_factor == 0:
            return 0.0
        opp_elos = self._opponent_elos.get(team, [])
        if len(opp_elos) < 3:
            return 0.0
        recent = opp_elos[-5:]
        avg_opp = sum(recent) / len(recent)
        return self.sos_factor * (avg_opp - self.base_rating) / 100.0

    def playoff_home_adv(self, game_date):
        """Return adjusted home advantage for playoffs.
        NFL playoffs run January through early February."""
        if game_date is None:
            return self.home_adv
        month = game_date.month
        # Playoff months: January and February (Super Bowl)
        if month in (1, 2):
            return self.home_adv * self.playoff_hca_factor
        return self.home_adv

    def is_playoff_game(self, game_date):
        """Detect if a game is likely a playoff game based on date."""
        if game_date is None:
            return False
        m = game_date.month
        return m in (1, 2)

    def pace_adjustment(self, team_a, team_b):
        """Elo adjustment based on pace mismatch.
        In NFL, pace relates to play tempo and total points scored."""
        if self.pace_factor == 0:
            return 0.0, 0.0
        scores_a = self._team_scores.get(team_a, [])
        scores_b = self._team_scores.get(team_b, [])
        if len(scores_a) < 3 or len(scores_b) < 3:
            return 0.0, 0.0
        recent_a = scores_a[-5:]
        recent_b = scores_b[-5:]
        pace_a = sum(pf + pa for pf, pa in recent_a) / (2.0 * len(recent_a))
        pace_b = sum(pf + pa for pf, pa in recent_b) / (2.0 * len(recent_b))
        pace_diff = pace_a - pace_b
        adj_a = -self.pace_factor * pace_diff / 10.0
        adj_b = self.pace_factor * pace_diff / 10.0
        return adj_a, adj_b

    def division_adjustment(self, team_a, team_b):
        """Reduce prediction confidence for divisional games"""
        if self.division_factor == 0:
            return 0.0, 0.0
        from config import same_division
        if same_division(team_a, team_b):
            ra, rb = self.ratings[team_a], self.ratings[team_b]
            diff = ra - rb
            adj = self.division_factor * diff / 100.0
            return -adj, adj
        return 0.0, 0.0

    def mean_reversion_adjustment(self, team):
        """After extreme results, expect regression to mean"""
        if self.mean_reversion == 0 or team not in self._last_margin:
            return 0
        margin = self._last_margin[team]
        if abs(margin) > 17:  # NFL blowout threshold: 3+ scores
            return -self.mean_reversion * margin / 100.0
        return 0

    def b2b_penalty_adjustment(self, team, game_date):
        """Extra penalty for short-week games (rest <= 5 days, e.g. Thursday games)."""
        if self.b2b_penalty == 0:
            return 0.0
        rd = self.rest_days(team, game_date)
        if rd is not None and rd <= 5:
            return -self.b2b_penalty
        return 0.0

    def road_trip_adjustment(self, team):
        """Penalty for extended road trips (2+ consecutive away games)."""
        if self.road_trip_factor == 0:
            return 0.0
        consec = self._consecutive_away.get(team, 0)
        if consec >= 2:
            return -self.road_trip_factor * min(consec - 1, 5)
        return 0.0

    def homestand_adjustment(self, team):
        """Bonus for extended homestands (2+ consecutive home games)."""
        if self.homestand_factor == 0:
            return 0.0
        consec = self._consecutive_home.get(team, 0)
        if consec >= 2:
            return self.homestand_factor * min(consec - 1, 5)
        return 0.0

    def win_streak_adjustment(self, team):
        """Momentum bonus for win streaks, penalty for loss streaks."""
        if self.win_streak_factor == 0:
            return 0.0
        results = self._recent_results.get(team, [])
        if len(results) < 2:
            return 0.0
        # Count streak from end
        streak = 0
        last = results[-1]
        for r in reversed(results):
            if r == last:
                streak += 1
            else:
                break
        streak = min(streak, 5)
        if last >= 0.5:
            return self.win_streak_factor * streak / 5.0
        else:
            return -self.win_streak_factor * streak / 5.0

    def season_phase_adjustment(self, team_a, team_b):
        """Reduce confidence in early-season predictions."""
        if self.season_phase_factor == 0:
            return 0.0
        gn_a = self._game_number.get(team_a, 0)
        gn_b = self._game_number.get(team_b, 0)
        avg_gn = (gn_a + gn_b) / 2.0
        game_frac = min(avg_gn / 18.0, 1.0)
        if game_frac < 0.20:
            # Early season: dampen rating difference
            return self.season_phase_factor * (0.20 - game_frac)
        return 0.0

    def scoring_consistency_adjustment(self, team):
        """Penalty for teams with volatile scoring (high std of recent scores)."""
        if self.scoring_consistency_factor == 0:
            return 0.0
        scores = self._team_scores.get(team, [])
        if len(scores) < 3:
            return 0.0
        recent_pf = [pf for pf, pa in scores[-5:]]
        std = float(np.std(recent_pf))
        # NFL avg scoring std ~7 points; penalize above-average volatility
        return -self.scoring_consistency_factor * (std - 7.0) / 10.0

    def bye_week_adjustment(self, team, game_date):
        """Bonus for teams coming off a bye week (rest >= 10 days)."""
        if self.bye_week_factor == 0:
            return 0.0
        rd = self.rest_days(team, game_date)
        if rd is not None and rd >= 10:
            return self.bye_week_factor
        return 0.0

    def expected_score(self, ra, rb):
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    def update_game(self, home_team, away_team, home_score, away_score,
                    neutral_site=False, game_date=None):
        self._opponent_elos[home_team].append(self.ratings[away_team])
        self._opponent_elos[away_team].append(self.ratings[home_team])
        for team in (home_team, away_team):
            if len(self._opponent_elos[team]) > 10:
                self._opponent_elos[team] = self._opponent_elos[team][-10:]
        self._team_scores[home_team].append((float(home_score), float(away_score)))
        self._team_scores[away_team].append((float(away_score), float(home_score)))
        for team in (home_team, away_team):
            if len(self._team_scores[team]) > 10:
                self._team_scores[team] = self._team_scores[team][-10:]

        ra = self.ratings[home_team] + (0.0 if neutral_site else self.home_adv)
        rb = self.ratings[away_team]
        ea = self.expected_score(ra, rb)
        eb = 1.0 - ea
        if home_score > away_score:
            sa, sb = 1.0, 0.0
        elif home_score < away_score:
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5
        # NFL MOV: log(max(1, abs(margin)) + 1)
        mov = math.log(max(1.0, abs(home_score - away_score)) + 1.0) if self.use_mov else 1.0
        self.ratings[home_team] += self.k * mov * (sa - ea)
        self.ratings[away_team] += self.k * mov * (sb - eb)
        if game_date is not None:
            self._last_game_date[home_team] = game_date
            self._last_game_date[away_team] = game_date
        self._last_game_location[home_team] = home_team
        self._last_game_location[away_team] = home_team
        self._last_margin[home_team] = home_score - away_score
        self._last_margin[away_team] = away_score - home_score
        self._recent_results[home_team].append(sa)
        self._recent_results[away_team].append(sb)
        for team in (home_team, away_team):
            if len(self._recent_results[team]) > 10:
                self._recent_results[team] = self._recent_results[team][-10:]
        # Track consecutive home/away games
        self._consecutive_home[home_team] = self._consecutive_home.get(home_team, 0) + 1
        self._consecutive_away[home_team] = 0
        self._consecutive_away[away_team] = self._consecutive_away.get(away_team, 0) + 1
        self._consecutive_home[away_team] = 0
        # Track game number per team
        self._game_number[home_team] = self._game_number.get(home_team, 0) + 1
        self._game_number[away_team] = self._game_number.get(away_team, 0) + 1

    def win_prob(self, team_a, team_b, team_a_home=True, neutral_site=False,
                 calibrated=True, game_date=None, use_injuries=True):
        ra = self.ratings[team_a]
        rb = self.ratings[team_b]
        if not neutral_site and team_a_home is not None:
            hca = self.playoff_home_adv(game_date)
            ra += hca if team_a_home else 0.0
            rb += 0.0 if team_a_home else hca
            if self._altitude_bonus:
                home_team = team_a if team_a_home else team_b
                alt_bonus = self._altitude_bonus.get(home_team, 0.0)
                if alt_bonus > 0:
                    if self.altitude_factor > 0:
                        alt_bonus = alt_bonus * self.altitude_factor
                    if team_a_home:
                        ra += alt_bonus
                    else:
                        rb += alt_bonus
        if self._player_scores and self.player_boost > 0.0:
            ra += self._player_scores.get(team_a, 0.0) * self.player_boost
            rb += self._player_scores.get(team_b, 0.0) * self.player_boost
        ra += self.rest_adjustment(team_a, game_date)
        rb += self.rest_adjustment(team_b, game_date)
        ra += self.form_adjustment(team_a)
        rb += self.form_adjustment(team_b)
        if self.travel_factor > 0 and not neutral_site and team_a_home is not None:
            venue_team = team_a if team_a_home else team_b
            ra += self.travel_adjustment(team_a, venue_team)
            rb += self.travel_adjustment(team_b, venue_team)
        ra += self.sos_adjustment(team_a)
        rb += self.sos_adjustment(team_b)
        pace_a, pace_b = self.pace_adjustment(team_a, team_b)
        ra += pace_a
        rb += pace_b
        div_a, div_b = self.division_adjustment(team_a, team_b)
        ra += div_a
        rb += div_b
        ra += self.mean_reversion_adjustment(team_a)
        rb += self.mean_reversion_adjustment(team_b)
        # Back-to-back (short week) penalty
        ra += self.b2b_penalty_adjustment(team_a, game_date)
        rb += self.b2b_penalty_adjustment(team_b, game_date)
        # Road trip / homestand
        ra += self.road_trip_adjustment(team_a)
        rb += self.road_trip_adjustment(team_b)
        ra += self.homestand_adjustment(team_a)
        rb += self.homestand_adjustment(team_b)
        # Win streak momentum
        ra += self.win_streak_adjustment(team_a)
        rb += self.win_streak_adjustment(team_b)
        # Scoring consistency
        ra += self.scoring_consistency_adjustment(team_a)
        rb += self.scoring_consistency_adjustment(team_b)
        # Bye week bonus
        ra += self.bye_week_adjustment(team_a, game_date)
        rb += self.bye_week_adjustment(team_b, game_date)
        # Season phase dampener (reduces diff in early season)
        phase_adj = self.season_phase_adjustment(team_a, team_b)
        if phase_adj > 0:
            diff = ra - rb
            ra -= phase_adj * diff / 100.0
            rb += phase_adj * diff / 100.0
        if use_injuries:
            from injuries import get_team_injuries, calc_injury_impact
            for team, rating_ref in ((team_a, "a"), (team_b, "b")):
                out = get_team_injuries(team)
                if out:
                    impact = calc_injury_impact(team, out)
                    if rating_ref == "a":
                        ra += impact
                    else:
                        rb += impact
        raw_p = self.expected_score(ra, rb)
        # XGBoost ensemble
        if calibrated and self._xgb_model is not None and self._xgb_meta is not None:
            xgb_prob = self._xgb_predict(team_a, team_b, raw_p, ra - rb, game_date)
            if xgb_prob is not None:
                elo_w = self._xgb_meta.get("elo_weight", 0.8)
                raw_p = elo_w * raw_p + (1.0 - elo_w) * xgb_prob
        # Mega-ensemble adjustment (if trained and available)
        if calibrated and self._mega_predictor is not None:
            mega_adj = self._mega_predictor.predict(
                team_a, team_b, raw_p, ra - rb, game_date
            )
            raw_p = max(0.02, min(0.98, raw_p + mega_adj))
        if calibrated and self._platt_scaler is not None:
            return apply_platt(raw_p, self._platt_scaler)
        return raw_p

    def _xgb_predict(self, team_a, team_b, elo_prob, elo_diff, game_date):
        try:
            import xgboost as xgb
            from enhanced_model import build_game_features
            scores_a = self._team_scores.get(team_a, [])
            scores_b = self._team_scores.get(team_b, [])
            if len(scores_a) < 3 or len(scores_b) < 3:
                return None

            def _rolling(scores, team):
                pf = [s[0] for s in scores[-5:]]
                pa = [s[1] for s in scores[-5:]]
                res = [1.0 if s[0] > s[1] else 0.0 for s in scores[-5:]]
                margins = [s[0] - s[1] for s in scores[-5:]]
                rd = self.rest_days(team, game_date)
                return {
                    "ppg": np.mean(pf), "papg": np.mean(pa),
                    "win_pct": np.mean(res), "avg_margin": np.mean(margins),
                    "off_rating": np.mean(pf), "def_rating": np.mean(pa),
                    "rest_days": rd if rd is not None else 7,
                    "games_played": len(pf),
                }

            home_feats = _rolling(scores_a, team_a)
            away_feats = _rolling(scores_b, team_b)
            rd_a = self.rest_days(team_a, game_date)
            rd_b = self.rest_days(team_b, game_date)
            home_feats["rest_days"] = rd_a if rd_a is not None else 7
            away_feats["rest_days"] = rd_b if rd_b is not None else 7

            player_diff = (self._player_scores.get(team_a, 0.0)
                           - self._player_scores.get(team_b, 0.0))
            fdict = build_game_features(home_feats, away_feats, elo_prob,
                                        elo_diff, player_diff)
            if fdict is None:
                return None
            feature_cols = self._xgb_meta.get("feature_cols")
            if not feature_cols:
                return None
            fvec = np.array([[fdict[c] for c in feature_cols]])
            dmat = xgb.DMatrix(fvec, feature_names=feature_cols)
            return float(self._xgb_model.predict(dmat)[0])
        except Exception as e:
            logging.debug("XGB predict failed: %s", e)
            return None

    def pick_winner(self, team_a, team_b, team_a_home=True, neutral_site=False,
                    game_date=None):
        if game_date is None:
            game_date = datetime.now()
        pa = self.win_prob(team_a, team_b, team_a_home, neutral_site,
                           game_date=game_date)
        return (team_a, pa) if pa >= 0.5 else (team_b, 1.0 - pa)

    def find_team(self, query):
        query = str(query).strip().lower()
        if not query:
            return None
        if query in self._team_lookup:
            return self._team_lookup[query]
        query_upper = query.upper()
        for full_name, abbr in TEAM_ABBR.items():
            if abbr == query_upper:
                found = self._team_lookup.get(full_name.lower())
                if found:
                    return found
                for key, name in self._team_lookup.items():
                    if any(part in key for part in full_name.lower().split() if len(part) > 3):
                        return name
        for key, name in self._team_lookup.items():
            if query in key:
                return name
        matches = get_close_matches(query, list(self._team_lookup.keys()), n=1, cutoff=0.6)
        if matches:
            return self._team_lookup[matches[0]]
        return None

    def save(self, filename=RATINGS_FILE):
        try:
            self.metadata["saved_at"] = current_timestamp()
            self.metadata["settings"] = self.settings_dict()
            payload = {"regular": dict(self.ratings), "metadata": self.metadata}
            with open(filename, "w") as f:
                json.dump(payload, f, indent=2)
            logging.info("Saved %d ratings -> %s", len(self.ratings), filename)
        except Exception as e:
            logging.warning("Save failed: %s", e)

    def load(self, filename=RATINGS_FILE):
        if not os.path.exists(filename):
            return False
        try:
            with open(filename, "r") as f:
                data = json.load(f)
            if isinstance(data, dict) and "regular" in data:
                self.ratings = defaultdict(lambda: self.base_rating, data.get("regular", {}))
                self.metadata.update(data.get("metadata", {}))
            else:
                self.ratings = defaultdict(lambda: self.base_rating, data)
            self.team_names = sorted(self.ratings.keys())
            self._rebuild_lookup()
            self._platt_scaler = load_platt_scaler()
            if self._platt_scaler:
                logging.info("Platt scaler active (n=%s samples)", self._platt_scaler.get("n_samples", "?"))
            logging.info("Loaded %d team ratings", len(self.team_names))
            return True
        except Exception as e:
            logging.warning("Load failed: %s", e)
            return False

    def show_settings(self):
        from config import load_elo_settings
        hdr("ELO SETTINGS")
        div()
        from enhanced_model import load_enhanced_model
        xgb_model, xgb_meta = load_enhanced_model()
        rows = [
            ("Base Rating",   self.base_rating),
            ("K Factor",      self.k),
            ("Home Adv",      self.home_adv),
            ("MOV Enabled",   self.use_mov),
            ("Player Boost",  self.player_boost),
            ("Rest Factor",   self.rest_factor),
            ("Travel Factor", self.travel_factor),
            ("SOS Factor",    self.sos_factor),
            ("Playoff HCA",  "%.0f%% of normal" % (self.playoff_hca_factor * 100)),
            ("Pace Factor",   self.pace_factor),
            ("B2B Penalty",   self.b2b_penalty),
            ("Road Trip",     self.road_trip_factor),
            ("Homestand",     self.homestand_factor),
            ("Win Streak",    self.win_streak_factor),
            ("Altitude Fac",  self.altitude_factor if self.altitude_factor > 0 else "raw"),
            ("Season Phase",  self.season_phase_factor),
            ("Score Consist", self.scoring_consistency_factor),
            ("Rest Cap",      int(self.rest_advantage_cap) if self.rest_advantage_cap > 0 else "off"),
            ("Bye Week",      self.bye_week_factor),
            ("Altitude Bonus", ", ".join("%s +%.1f" % (t, b) for t, b in self._altitude_bonus.items()) if self._altitude_bonus else "none (no data)"),
            ("Player Scores", "%d teams loaded" % len(self._player_scores)),
            ("XGBoost",       "ACTIVE (Elo=80% XGB=20%)" if xgb_model else "not trained -- run 'enhanced'"),
            ("Mega-Ensemble", self._mega_predictor.get_status() if self._mega_predictor else "not trained -- run 'mega' first"),
            ("Auto-Resolve",  "ON" if load_elo_settings().get("autoresolve_enabled") else "OFF"),
        ]
        for label, val in rows:
            print("  %-16s: %s" % (chi(label), cok(val)))
        if self._platt_scaler:
            ps = self._platt_scaler
            platt_s = cok("ACTIVE") + cdim(
                " (n=%s, coef=%.3f, intercept=%.3f, fitted %s)"
                % (ps.get("n_samples", "?"), ps.get("coef", 0),
                   ps.get("intercept", 0), ps.get("fitted_at", "?"))
            )
        else:
            platt_s = cwarn("not fitted") + cdim(" -- run 'backtest' to fit")
        print("  %-16s: %s" % (chi("Platt Scaler"), platt_s))
        div()

    def show_all_teams(self):
        sorted_teams = sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
        hdr("ALL %d NFL TEAMS - Elo Ratings" % len(sorted_teams))
        div()
        for i, (team, rating) in enumerate(sorted_teams, 1):
            if rating >= 1550:
                rs = cok("%.1f" % rating)
            elif rating >= 1450:
                rs = cwarn("%.1f" % rating)
            else:
                rs = cred("%.1f" % rating)
            bar = cdim("#" * int((rating - 1400) / 10))
            print("  %s %-28s  %s  %s" % (cdim("%3d." % i), team, rs, bar))
        div()
