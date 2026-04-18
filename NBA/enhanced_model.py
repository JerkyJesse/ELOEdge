"""Enhanced prediction model: rolling features + XGBoost ensemble with Elo."""

import os
import json
import logging
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import xgboost as xgb
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *a, **kw):
        return iterable if iterable is not None else range(0)
    tqdm.write = print

from config import GAMES_FILE, load_elo_settings
from elo_model import NBAElo
from data_players import load_player_stats
from metrics import log_loss_binary, brier_score_binary
from platt import (fit_platt_scaler, apply_platt,
                   fit_isotonic_scaler, apply_isotonic, save_isotonic_scaler)

ENHANCED_MODEL_FILE = "nba_enhanced_model.json"
WINDOW = 10  # rolling window for team stats
PYTH_EXP = 14.0  # NBA Morey Pythagorean exponent


class TeamTracker:
    """Tracks rolling stats per team from game-by-game data."""

    def __init__(self):
        self.points_scored = defaultdict(list)   # last N scores
        self.points_allowed = defaultdict(list)  # last N opponent scores
        self.results = defaultdict(list)         # last N win/loss (1/0)
        self.margins = defaultdict(list)         # last N score differentials
        self.last_date = {}                      # team -> last game datetime
        self.streak = defaultdict(int)
        # Unconventional tracking
        self.home_results = defaultdict(list)
        self.away_results = defaultdict(list)
        self.game_dates = defaultdict(list)
        self.close_results = defaultdict(list)
        self.blowout_results = defaultdict(list)
        # New state tracking
        self.consecutive_away = defaultdict(int)
        self.consecutive_home = defaultdict(int)
        self.game_number = defaultdict(int)
        self.opponent_elos = defaultdict(list)
        self.was_home = defaultdict(list)  # per-game is_home for B2B travel detection

    def get_features(self, team, game_date=None, is_home=True):
        """Return rolling features for a team. All based on past data only."""
        scored = self.points_scored.get(team, [])
        allowed = self.points_allowed.get(team, [])
        results = self.results.get(team, [])
        margins = self.margins.get(team, [])

        n = len(scored)
        if n < 3:
            return None  # not enough history

        ppg = np.mean(scored[-WINDOW:])
        papg = np.mean(allowed[-WINDOW:])
        win_pct = np.mean(results[-WINDOW:])
        avg_margin = np.mean(margins[-WINDOW:])

        # Offensive/defensive ratings (simplified: points per game proxies)
        off_rating = ppg
        def_rating = papg

        # Pythagorean win expectation (Morey formula for NBA)
        rs_exp = max(ppg, 0.1) ** PYTH_EXP
        ra_exp = max(papg, 0.1) ** PYTH_EXP
        pyth = rs_exp / (rs_exp + ra_exp)

        # Margin consistency (lower std = more predictable)
        consistency = float(np.std(margins[-WINDOW:])) if len(margins) >= 3 else 10.0

        # Trend: recent 5 vs full window win%
        recent_5 = np.mean(results[-5:]) if len(results) >= 5 else win_pct
        trend = recent_5 - win_pct

        # Rest days
        rest = None
        if game_date is not None and team in self.last_date:
            rest = max(0, (game_date - self.last_date[team]).days)

        # --- Unconventional features ---
        h_res = self.home_results.get(team, [])
        a_res = self.away_results.get(team, [])
        home_win_pct = np.mean(h_res[-WINDOW:]) if len(h_res) >= 3 else 0.5
        away_win_pct = np.mean(a_res[-WINDOW:]) if len(a_res) >= 3 else 0.5
        venue_win_pct = home_win_pct if is_home else away_win_pct

        fatigue = 0
        if game_date is not None:
            recent_dates = self.game_dates.get(team, [])
            cutoff = game_date - pd.Timedelta(days=7)
            fatigue = sum(1 for d in recent_dates if d >= cutoff)

        close_res = self.close_results.get(team, [])
        clutch = np.mean(close_res[-10:]) if len(close_res) >= 3 else 0.5

        score_vol = float(np.std(scored[-WINDOW:])) if len(scored) >= 5 else 10.0

        blow_res = self.blowout_results.get(team, [])
        dominance = np.mean(blow_res[-10:]) if len(blow_res) >= 3 else 0.5

        # New features: B2B, road trip, margin trend, ultra-recent form
        is_b2b = 1.0 if (rest is not None and rest <= 1) else 0.0
        road_trip_len = float(self.consecutive_away.get(team, 0))
        homestand_len = float(self.consecutive_home.get(team, 0))
        season_gn = float(self.game_number.get(team, 0))

        # Average opponent Elo (SOS feature)
        opp_elos = self.opponent_elos.get(team, [])
        avg_opp_elo = float(np.mean(opp_elos[-10:])) if len(opp_elos) >= 3 else 1500.0

        # Margin trend (slope over last N games) -- vectorized direct formula
        margin_trend = 0.0
        if len(margins) >= 5:
            recent_m = margins[-WINDOW:]
            n_mt = len(recent_m)
            x_mt = np.arange(n_mt, dtype=np.float64)
            x_mean = (n_mt - 1) / 2.0
            m_arr = np.asarray(recent_m, dtype=np.float64)
            m_mean = m_arr.mean()
            xd = x_mt - x_mean
            margin_trend = float(np.dot(xd, m_arr - m_mean) / max(np.dot(xd, xd), 1e-10))

        # Ultra-recent form (last 3 games)
        last3_wp = float(np.mean(results[-3:])) if len(results) >= 3 else 0.5

        # --- ADVANCED "OCCULT" FEATURES ---

        # Pythagorean residual: actual win% minus expected (luck factor)
        # Positive = team is "lucky" and due for regression
        pyth_residual = win_pct - pyth

        # Recency-weighted scoring (exponential decay, half-life ~5 games)
        decay_ppg = ppg
        decay_papg = papg
        if len(scored) >= 5:
            weights = np.array([0.5 ** ((len(scored[-WINDOW:]) - 1 - i) / 5.0)
                                for i in range(len(scored[-WINDOW:]))])
            weights /= weights.sum()
            decay_ppg = float(np.dot(scored[-WINDOW:], weights))
            decay_papg = float(np.dot(allowed[-WINDOW:], weights))

        # Momentum autocorrelation: correlation of consecutive margins
        # Positive = team performance is self-reinforcing (hot/cold streaks real)
        momentum_autocorr = 0.0
        if len(margins) >= 6:
            m = margins[-WINDOW:]
            if len(m) >= 6:
                a1 = np.asarray(m[:-1], dtype=np.float64)
                a2 = np.asarray(m[1:], dtype=np.float64)
                a1d = a1 - a1.mean()
                a2d = a2 - a2.mean()
                denom = np.sqrt(np.dot(a1d, a1d) * np.dot(a2d, a2d))
                momentum_autocorr = float(np.dot(a1d, a2d) / denom) if denom > 1e-10 else 0.0

        # Defensive trend: is defense improving or degrading? -- vectorized
        def_trend = 0.0
        if len(allowed) >= 5:
            recent_a = allowed[-WINDOW:]
            n_dt = len(recent_a)
            x_dt = np.arange(n_dt, dtype=np.float64)
            x_mean_dt = (n_dt - 1) / 2.0
            a_arr = np.asarray(recent_a, dtype=np.float64)
            a_mean = a_arr.mean()
            xd_dt = x_dt - x_mean_dt
            def_trend = float(np.dot(xd_dt, a_arr - a_mean) / max(np.dot(xd_dt, xd_dt), 1e-10))  # negative = defense improving

        # Scoring trend: is offense improving or degrading? -- vectorized
        off_trend = 0.0
        if len(scored) >= 5:
            recent_s = scored[-WINDOW:]
            n_ot = len(recent_s)
            x_ot = np.arange(n_ot, dtype=np.float64)
            x_mean_ot = (n_ot - 1) / 2.0
            s_arr = np.asarray(recent_s, dtype=np.float64)
            s_mean = s_arr.mean()
            xd_ot = x_ot - x_mean_ot
            off_trend = float(np.dot(xd_ot, s_arr - s_mean) / max(np.dot(xd_ot, xd_ot), 1e-10))  # positive = offense improving

        # Win% vs expected from margins (close game luck)
        # Teams winning close games at >60% are "clutch lucky" and will regress
        expected_wp_from_margins = 0.5
        if len(margins) >= 5:
            expected_wp_from_margins = float(np.mean([1.0 if m > 0 else 0.0 for m in margins[-WINDOW:]]))

        # SOS-adjusted win%: win% weighted by opponent strength
        sos_adj_wp = win_pct
        if len(opp_elos) >= 5 and len(results) >= 5:
            recent_opp = opp_elos[-WINDOW:]
            recent_res = results[-WINDOW:]
            min_n = min(len(recent_opp), len(recent_res))
            if min_n >= 3:
                opp_w = np.array(recent_opp[:min_n], dtype=float)
                res_w = np.array(recent_res[:min_n], dtype=float)
                opp_w = opp_w / 1500.0  # normalize around 1.0
                if opp_w.sum() > 0:
                    sos_adj_wp = float(np.average(res_w, weights=opp_w))

        # Rest quality (quadratic): optimal rest = 2 days for NBA
        # rest=1 (B2B) = bad, rest=2 (1 day off) = optimal, rest=4+ = rusty
        # Formula: 1.0 - (rest - 2)^2 / 4, clamped to [-1, 1]
        rest_quality = 0.0
        if rest is not None:
            rest_quality = max(-1.0, min(1.0, 1.0 - (rest - 2.0) ** 2 / 4.0))

        # --- NEW PREDICTION EDGE FEATURES ---

        # 1. Pythagorean Regression Risk: sample-size weighted regression signal
        games_played = n
        pyth_regression_risk = abs(pyth_residual) * min(1.0, games_played / 40.0)

        # 2. Back-to-Back with Travel: B2B is worse when team traveled
        was_home_list = self.was_home.get(team, [])
        if was_home_list and len(was_home_list) >= 1:
            last_was_home = was_home_list[-1]
            _is_b2b = (rest is not None and rest <= 1)
            b2b_travel = 1.0 if (_is_b2b and last_was_home != is_home) else 0.0
        else:
            b2b_travel = 0.0

        # 3. Win Streak Momentum: weighted recent form
        if len(results) >= 3:
            last3 = [1.0 if r >= 0.5 else -1.0 for r in results[-3:]]
            last5 = [1.0 if r >= 0.5 else -1.0 for r in results[-5:]] if len(results) >= 5 else last3
            streak_momentum = sum(last3) / 3.0 * 0.6 + sum(last5) / max(len(last5), 1) * 0.4
        else:
            streak_momentum = 0.0

        # 4. Home/Away Split: how much better at home vs away
        ha_home_results = self.home_results.get(team, [])
        ha_away_results = self.away_results.get(team, [])
        ha_home_wp = sum(ha_home_results) / max(len(ha_home_results), 1)
        ha_away_wp = sum(ha_away_results) / max(len(ha_away_results), 1)
        home_away_split = ha_home_wp - ha_away_wp

        # 5. Scoring Variance Ratio: ratio of offensive to defensive consistency
        off_scores_list = self.points_scored.get(team, [])
        def_scores_list = self.points_allowed.get(team, [])
        if len(off_scores_list) >= 5:
            off_recent = off_scores_list[-10:]
            def_recent = def_scores_list[-10:] if def_scores_list else []
            off_var = float(np.std(off_recent)) if off_recent else 1.0
            def_var = float(np.std(def_recent)) if def_recent else 1.0
            variance_ratio = off_var / max(def_var, 0.1)
        else:
            variance_ratio = 1.0

        return {
            "ppg": ppg,
            "papg": papg,
            "win_pct": win_pct,
            "avg_margin": avg_margin,
            "off_rating": off_rating,
            "def_rating": def_rating,
            "rest_days": rest,
            "games_played": n,
            "pyth": pyth,
            "streak": float(self.streak.get(team, 0)),
            "consistency": consistency,
            "trend": trend,
            "venue_win_pct": venue_win_pct,
            "fatigue": float(fatigue),
            "clutch": clutch,
            "score_vol": score_vol,
            "dominance": dominance,
            "is_b2b": is_b2b,
            "road_trip_len": road_trip_len,
            "homestand_len": homestand_len,
            "season_game_num": season_gn,
            "avg_opp_elo": avg_opp_elo,
            "margin_trend": margin_trend,
            "last3_win_pct": last3_wp,
            # Advanced occult features
            "pyth_residual": pyth_residual,
            "decay_ppg": decay_ppg,
            "decay_papg": decay_papg,
            "momentum_autocorr": momentum_autocorr,
            "def_trend": def_trend,
            "off_trend": off_trend,
            "sos_adj_wp": sos_adj_wp,
            "rest_quality": rest_quality,
            # New prediction edge features
            "pyth_regression_risk": pyth_regression_risk,
            "b2b_travel": b2b_travel,
            "streak_momentum": streak_momentum,
            "home_away_split": home_away_split,
            "variance_ratio": variance_ratio,
        }

    def update(self, team, pts_scored, pts_allowed, won, game_date=None,
               is_home=True, opp_elo=1500.0):
        self.points_scored[team].append(pts_scored)
        self.points_allowed[team].append(pts_allowed)
        self.results[team].append(1.0 if won else 0.0)
        self.margins[team].append(pts_scored - pts_allowed)
        # Update streak
        if won:
            self.streak[team] = max(0, self.streak.get(team, 0)) + 1
        else:
            self.streak[team] = min(0, self.streak.get(team, 0)) - 1
        # Track consecutive home/away and game number
        if is_home:
            self.consecutive_home[team] = self.consecutive_home.get(team, 0) + 1
            self.consecutive_away[team] = 0
        else:
            self.consecutive_away[team] = self.consecutive_away.get(team, 0) + 1
            self.consecutive_home[team] = 0
        self.game_number[team] = self.game_number.get(team, 0) + 1
        # Track opponent Elo for SOS
        self.opponent_elos[team].append(opp_elo)
        if len(self.opponent_elos[team]) > 25:
            self.opponent_elos[team] = self.opponent_elos[team][-25:]
        # Track per-game home/away for B2B travel detection
        self.was_home[team].append(is_home)
        if len(self.was_home[team]) > 25:
            self.was_home[team] = self.was_home[team][-25:]
        # Keep last 25 for flexibility (we slice to WINDOW when reading)
        for store in (self.points_scored, self.points_allowed,
                      self.results, self.margins):
            if len(store[team]) > 25:
                store[team] = store[team][-25:]
        # Track home/away splits
        if is_home:
            self.home_results[team].append(1.0 if won else 0.0)
            if len(self.home_results[team]) > 25:
                self.home_results[team] = self.home_results[team][-25:]
        else:
            self.away_results[team].append(1.0 if won else 0.0)
            if len(self.away_results[team]) > 25:
                self.away_results[team] = self.away_results[team][-25:]
        # Track game dates for fatigue calculation
        if game_date is not None:
            self.game_dates[team].append(game_date)
            if len(self.game_dates[team]) > 20:
                self.game_dates[team] = self.game_dates[team][-20:]
        # Track close games (margin <= 5 points) and blowouts (margin >= 15)
        margin = abs(pts_scored - pts_allowed)
        if margin <= 5:
            self.close_results[team].append(1.0 if won else 0.0)
            if len(self.close_results[team]) > 15:
                self.close_results[team] = self.close_results[team][-15:]
        if margin >= 15:
            self.blowout_results[team].append(1.0 if won else 0.0)
            if len(self.blowout_results[team]) > 15:
                self.blowout_results[team] = self.blowout_results[team][-15:]
        if game_date is not None:
            self.last_date[team] = game_date


def compute_team_stats(player_df, adv_df=None):
    """Aggregate player-level stats to team-level averages."""
    from config import TEAM_ABBR
    abbr_to_full = {v: k for k, v in TEAM_ABBR.items()}
    team_stats = {}
    if player_df is None or player_df.empty:
        return team_stats
    for abbr in player_df["Tm"].unique():
        team_name = abbr_to_full.get(abbr, abbr)
        tp = player_df[player_df["Tm"] == abbr]
        stats = {}
        for col, key in [("FG%", "fg_pct"), ("3P%", "three_pct"), ("FT%", "ft_pct"),
                         ("TRB", "reb"), ("AST", "ast"), ("TOV", "tov")]:
            if col in tp.columns:
                vals = pd.to_numeric(tp[col], errors="coerce").dropna()
                stats[key] = float(vals.mean()) if len(vals) > 0 else 0.0
            else:
                stats[key] = 0.0
        # Advanced stats
        if adv_df is not None and not adv_df.empty:
            ap = adv_df[adv_df["Tm"] == abbr]
            for col, key in [("NET_RATING", "net_rtg"), ("OFF_RATING", "off_rtg"),
                             ("DEF_RATING", "def_rtg")]:
                if col in ap.columns:
                    vals = pd.to_numeric(ap[col], errors="coerce").dropna()
                    stats[key] = float(vals.mean()) if len(vals) > 0 else 0.0
                else:
                    stats[key] = 0.0
        else:
            stats["net_rtg"] = 0.0
            stats["off_rtg"] = 0.0
            stats["def_rtg"] = 0.0
        team_stats[team_name] = stats
    return team_stats


def build_game_features(home_feats, away_feats, elo_prob, elo_diff,
                        player_diff=0.0, day_of_week=3, month=6,
                        elo_rating_h=1500.0, elo_rating_a=1500.0,
                        h_team_stats=None, a_team_stats=None):
    """Build feature vector for a single game."""
    if home_feats is None or away_feats is None:
        return None
    hs = h_team_stats or {}
    as_ = a_team_stats or {}
    return {
        "elo_prob": elo_prob,
        "elo_diff": elo_diff,
        "player_diff": player_diff,
        # Home team rolling stats
        "h_ppg": home_feats["ppg"],
        "h_papg": home_feats["papg"],
        "h_win_pct": home_feats["win_pct"],
        "h_margin": home_feats["avg_margin"],
        # Away team rolling stats
        "a_ppg": away_feats["ppg"],
        "a_papg": away_feats["papg"],
        "a_win_pct": away_feats["win_pct"],
        "a_margin": away_feats["avg_margin"],
        # Differentials (home advantage perspective)
        "ppg_diff": home_feats["ppg"] - away_feats["ppg"],
        "papg_diff": home_feats["papg"] - away_feats["papg"],
        "win_pct_diff": home_feats["win_pct"] - away_feats["win_pct"],
        "margin_diff": home_feats["avg_margin"] - away_feats["avg_margin"],
        "off_diff": home_feats["off_rating"] - away_feats["off_rating"],
        "def_diff": home_feats["def_rating"] - away_feats["def_rating"],
        # Rest
        "h_rest": home_feats["rest_days"] if home_feats["rest_days"] is not None else 1.0,
        "a_rest": away_feats["rest_days"] if away_feats["rest_days"] is not None else 1.0,
        "rest_diff": (home_feats["rest_days"] or 1) - (away_feats["rest_days"] or 1),
        # Pythagorean win expectation
        "h_pyth": home_feats["pyth"],
        "a_pyth": away_feats["pyth"],
        "pyth_diff": home_feats["pyth"] - away_feats["pyth"],
        # Streaks and momentum
        "h_streak": home_feats["streak"],
        "a_streak": away_feats["streak"],
        "streak_diff": home_feats["streak"] - away_feats["streak"],
        # Consistency
        "h_consistency": home_feats["consistency"],
        "a_consistency": away_feats["consistency"],
        # Trend
        "h_trend": home_feats["trend"],
        "a_trend": away_feats["trend"],
        "trend_diff": home_feats["trend"] - away_feats["trend"],
        # --- Unconventional features ---
        "h_venue_wp": home_feats.get("venue_win_pct", 0.5),
        "a_venue_wp": away_feats.get("venue_win_pct", 0.5),
        "venue_wp_diff": home_feats.get("venue_win_pct", 0.5) - away_feats.get("venue_win_pct", 0.5),
        "h_fatigue": home_feats.get("fatigue", 0.0),
        "a_fatigue": away_feats.get("fatigue", 0.0),
        "fatigue_diff": home_feats.get("fatigue", 0.0) - away_feats.get("fatigue", 0.0),
        "h_clutch": home_feats.get("clutch", 0.5),
        "a_clutch": away_feats.get("clutch", 0.5),
        "clutch_diff": home_feats.get("clutch", 0.5) - away_feats.get("clutch", 0.5),
        "h_score_vol": home_feats.get("score_vol", 10.0),
        "a_score_vol": away_feats.get("score_vol", 10.0),
        "dominance_diff": home_feats.get("dominance", 0.5) - away_feats.get("dominance", 0.5),
        "day_of_week": float(day_of_week),
        "month": float(month),
        # --- New game-history features ---
        "is_b2b_h": home_feats.get("is_b2b", 0.0),
        "is_b2b_a": away_feats.get("is_b2b", 0.0),
        "road_trip_len_h": home_feats.get("road_trip_len", 0.0),
        "road_trip_len_a": away_feats.get("road_trip_len", 0.0),
        "homestand_len_h": home_feats.get("homestand_len", 0.0),
        "homestand_len_a": away_feats.get("homestand_len", 0.0),
        "season_game_num": (home_feats.get("season_game_num", 0.0) + away_feats.get("season_game_num", 0.0)) / 2.0,
        "elo_rating_h": elo_rating_h,
        "elo_rating_a": elo_rating_a,
        "avg_opp_elo_h": home_feats.get("avg_opp_elo", 1500.0),
        "avg_opp_elo_a": away_feats.get("avg_opp_elo", 1500.0),
        "opp_elo_diff": home_feats.get("avg_opp_elo", 1500.0) - away_feats.get("avg_opp_elo", 1500.0),
        "margin_trend_h": home_feats.get("margin_trend", 0.0),
        "margin_trend_a": away_feats.get("margin_trend", 0.0),
        "last3_wp_h": home_feats.get("last3_win_pct", 0.5),
        "last3_wp_a": away_feats.get("last3_win_pct", 0.5),
        "last3_wp_diff": home_feats.get("last3_win_pct", 0.5) - away_feats.get("last3_win_pct", 0.5),
        "total_points_ou": (home_feats["ppg"] + away_feats["ppg"] + home_feats["papg"] + away_feats["papg"]) / 2.0,
        # --- Player stat features ---
        "h_fg_pct": hs.get("fg_pct", 0.0),
        "a_fg_pct": as_.get("fg_pct", 0.0),
        "fg_pct_diff": hs.get("fg_pct", 0.0) - as_.get("fg_pct", 0.0),
        "h_3p_pct": hs.get("three_pct", 0.0),
        "a_3p_pct": as_.get("three_pct", 0.0),
        "three_pct_diff": hs.get("three_pct", 0.0) - as_.get("three_pct", 0.0),
        "h_reb": hs.get("reb", 0.0),
        "a_reb": as_.get("reb", 0.0),
        "reb_diff": hs.get("reb", 0.0) - as_.get("reb", 0.0),
        "h_ast": hs.get("ast", 0.0),
        "a_ast": as_.get("ast", 0.0),
        "ast_diff": hs.get("ast", 0.0) - as_.get("ast", 0.0),
        "h_tov": hs.get("tov", 0.0),
        "a_tov": as_.get("tov", 0.0),
        "tov_diff": hs.get("tov", 0.0) - as_.get("tov", 0.0),
        "h_net_rtg": hs.get("net_rtg", 0.0),
        "a_net_rtg": as_.get("net_rtg", 0.0),
        "net_rtg_diff": hs.get("net_rtg", 0.0) - as_.get("net_rtg", 0.0),
        # Advanced occult features
        "h_pyth_resid": home_feats.get("pyth_residual", 0.0),
        "a_pyth_resid": away_feats.get("pyth_residual", 0.0),
        "pyth_resid_diff": home_feats.get("pyth_residual", 0.0) - away_feats.get("pyth_residual", 0.0),
        "h_decay_ppg": home_feats.get("decay_ppg", 0.0),
        "a_decay_ppg": away_feats.get("decay_ppg", 0.0),
        "decay_ppg_diff": home_feats.get("decay_ppg", 0.0) - away_feats.get("decay_ppg", 0.0),
        "h_decay_papg": home_feats.get("decay_papg", 0.0),
        "a_decay_papg": away_feats.get("decay_papg", 0.0),
        "decay_papg_diff": home_feats.get("decay_papg", 0.0) - away_feats.get("decay_papg", 0.0),
        "h_momentum": home_feats.get("momentum_autocorr", 0.0),
        "a_momentum": away_feats.get("momentum_autocorr", 0.0),
        "momentum_diff": home_feats.get("momentum_autocorr", 0.0) - away_feats.get("momentum_autocorr", 0.0),
        "h_def_trend": home_feats.get("def_trend", 0.0),
        "a_def_trend": away_feats.get("def_trend", 0.0),
        "def_trend_diff": home_feats.get("def_trend", 0.0) - away_feats.get("def_trend", 0.0),
        "h_off_trend": home_feats.get("off_trend", 0.0),
        "a_off_trend": away_feats.get("off_trend", 0.0),
        "off_trend_diff": home_feats.get("off_trend", 0.0) - away_feats.get("off_trend", 0.0),
        "h_sos_adj_wp": home_feats.get("sos_adj_wp", 0.5),
        "a_sos_adj_wp": away_feats.get("sos_adj_wp", 0.5),
        "sos_adj_wp_diff": home_feats.get("sos_adj_wp", 0.5) - away_feats.get("sos_adj_wp", 0.5),
        # Rest quality (quadratic: optimal rest = 1 day for NBA)
        "rest_quality_h": home_feats.get("rest_quality", 0.0),
        "rest_quality_a": away_feats.get("rest_quality", 0.0),
        "rest_quality_diff": home_feats.get("rest_quality", 0.0) - away_feats.get("rest_quality", 0.0),
        # Pythagorean regression risk (sample-size weighted)
        "h_pyth_regression": home_feats.get("pyth_regression_risk", 0.0),
        "a_pyth_regression": away_feats.get("pyth_regression_risk", 0.0),
        "pyth_regression_diff": home_feats.get("pyth_regression_risk", 0.0) - away_feats.get("pyth_regression_risk", 0.0),
        # Back-to-back with travel
        "h_b2b_travel": home_feats.get("b2b_travel", 0.0),
        "a_b2b_travel": away_feats.get("b2b_travel", 0.0),
        "b2b_travel_diff": home_feats.get("b2b_travel", 0.0) - away_feats.get("b2b_travel", 0.0),
        # Win streak momentum (weighted recent form)
        "h_streak_momentum": home_feats.get("streak_momentum", 0.0),
        "a_streak_momentum": away_feats.get("streak_momentum", 0.0),
        "streak_momentum_diff": home_feats.get("streak_momentum", 0.0) - away_feats.get("streak_momentum", 0.0),
        # Home/away split
        "h_ha_split": home_feats.get("home_away_split", 0.0),
        "a_ha_split": away_feats.get("home_away_split", 0.0),
        "ha_split_diff": home_feats.get("home_away_split", 0.0) - away_feats.get("home_away_split", 0.0),
        # Scoring variance ratio (offense vs defense consistency)
        "h_var_ratio": home_feats.get("variance_ratio", 1.0),
        "a_var_ratio": away_feats.get("variance_ratio", 1.0),
        "var_ratio_diff": home_feats.get("variance_ratio", 1.0) - away_feats.get("variance_ratio", 1.0),
    }


FEATURE_COLS = [
    "elo_prob", "elo_diff", "player_diff",
    "h_ppg", "h_papg", "h_win_pct", "h_margin",
    "a_ppg", "a_papg", "a_win_pct", "a_margin",
    "ppg_diff", "papg_diff", "win_pct_diff", "margin_diff",
    "off_diff", "def_diff",
    "h_rest", "a_rest", "rest_diff",
    "h_pyth", "a_pyth", "pyth_diff",
    "h_streak", "a_streak", "streak_diff",
    "h_consistency", "a_consistency",
    "h_trend", "a_trend", "trend_diff",
    "h_venue_wp", "a_venue_wp", "venue_wp_diff",
    "h_fatigue", "a_fatigue", "fatigue_diff",
    "h_clutch", "a_clutch", "clutch_diff",
    "h_score_vol", "a_score_vol",
    "dominance_diff",
    "day_of_week", "month",
    # New game-history features
    "is_b2b_h", "is_b2b_a",
    "road_trip_len_h", "road_trip_len_a",
    "homestand_len_h", "homestand_len_a",
    "season_game_num",
    "elo_rating_h", "elo_rating_a",
    "avg_opp_elo_h", "avg_opp_elo_a", "opp_elo_diff",
    "margin_trend_h", "margin_trend_a",
    "last3_wp_h", "last3_wp_a", "last3_wp_diff",
    "total_points_ou",
    # Player stat features
    "h_fg_pct", "a_fg_pct", "fg_pct_diff",
    "h_3p_pct", "a_3p_pct", "three_pct_diff",
    "h_reb", "a_reb", "reb_diff",
    "h_ast", "a_ast", "ast_diff",
    "h_tov", "a_tov", "tov_diff",
    "h_net_rtg", "a_net_rtg", "net_rtg_diff",
    # Advanced occult features
    "h_pyth_resid", "a_pyth_resid", "pyth_resid_diff",
    "h_decay_ppg", "a_decay_ppg", "decay_ppg_diff",
    "h_decay_papg", "a_decay_papg", "decay_papg_diff",
    "h_momentum", "a_momentum", "momentum_diff",
    "h_def_trend", "a_def_trend", "def_trend_diff",
    "h_off_trend", "a_off_trend", "off_trend_diff",
    "h_sos_adj_wp", "a_sos_adj_wp", "sos_adj_wp_diff",
    # Rest quality (quadratic: optimal rest = 1 day)
    "rest_quality_h", "rest_quality_a", "rest_quality_diff",
    # Pythagorean regression risk
    "h_pyth_regression", "a_pyth_regression", "pyth_regression_diff",
    # Back-to-back with travel
    "h_b2b_travel", "a_b2b_travel", "b2b_travel_diff",
    # Win streak momentum
    "h_streak_momentum", "a_streak_momentum", "streak_momentum_diff",
    # Home/away split
    "h_ha_split", "a_ha_split", "ha_split_diff",
    # Scoring variance ratio
    "h_var_ratio", "a_var_ratio", "var_ratio_diff",
]


def run_enhanced_backtest(csv_file=GAMES_FILE, min_train=200, retrain_every=50,
                          elo_weight=0.70, label="enhanced", time_decay=False):
    """
    Walk-forward backtest with XGBoost ensemble.
    - First min_train games: Elo-only predictions (XGB needs training data).
    - After that: XGBoost trained on accumulated features, ensembled with Elo.
    - Retrain XGBoost every retrain_every games.
    """
    if not os.path.exists(csv_file):
        print("ERROR: %s not found" % csv_file)
        return None

    settings = load_elo_settings()
    _elo_keys = {"base_rating", "k", "home_adv", "use_mov", "player_boost",
                 "rest_factor", "form_weight", "travel_factor", "sos_factor",
                 "pace_factor", "playoff_hca_factor", "division_factor", "mean_reversion",
                 "b2b_penalty", "road_trip_factor", "homestand_factor", "win_streak_factor",
                 "altitude_factor", "season_phase_factor", "scoring_consistency_factor",
                 "rest_advantage_cap"}
    model = NBAElo(**{k: v for k, v in settings.items() if k in _elo_keys})
    player_df = load_player_stats()
    if not player_df.empty:
        model.set_player_stats(player_df)

    # Load advanced stats and compute team-level stats for features
    try:
        from data_players import load_advanced_stats
        adv_df = load_advanced_stats()
    except Exception as e:
        logging.debug("Advanced stats not available: %s", e)
        adv_df = None
    team_stats = compute_team_stats(player_df, adv_df)

    games = pd.read_csv(csv_file)
    if "neutral_site" not in games.columns:
        games["neutral_site"] = False
    games["_date_parsed"] = pd.to_datetime(games["date"], errors="coerce")

    tracker = TeamTracker()
    feature_rows = []  # accumulated training data
    label_rows = []

    all_probs = []
    all_actuals = []
    correct = 0
    xgb_model = None
    xgb_predictions = 0
    _pred_buffer = np.zeros((1, len(FEATURE_COLS)), dtype=np.float32)

    for idx, row in tqdm(games.iterrows(), total=len(games),
                         desc="  Enhanced backtest", leave=True):
        game_date = row["_date_parsed"] if pd.notna(row["_date_parsed"]) else None
        home = row["home_team"]
        away = row["away_team"]
        neutral = bool(row["neutral_site"])
        home_actual = 1 if row["home_score"] > row["away_score"] else 0

        # Get Elo probability
        elo_prob = model.win_prob(home, away, team_a_home=True,
                                  neutral_site=neutral, calibrated=False,
                                  game_date=game_date)
        elo_diff = model.ratings[home] - model.ratings[away]
        player_diff = 0.0
        if model._player_scores and model.player_boost > 0:
            player_diff = (model._player_scores.get(home, 0.0) -
                           model._player_scores.get(away, 0.0))

        # Get rolling features
        home_feats = tracker.get_features(home, game_date, is_home=True)
        away_feats = tracker.get_features(away, game_date, is_home=False)
        dow = game_date.dayofweek if game_date is not None else 3
        mon = game_date.month if game_date is not None else 6
        game_features = build_game_features(home_feats, away_feats, elo_prob,
                                            elo_diff, player_diff,
                                            day_of_week=dow, month=mon,
                                            elo_rating_h=model.ratings[home],
                                            elo_rating_a=model.ratings[away],
                                            h_team_stats=team_stats.get(home),
                                            a_team_stats=team_stats.get(away))

        # Make prediction
        if xgb_model is not None and game_features is not None:
            for j, c in enumerate(FEATURE_COLS):
                _pred_buffer[0, j] = game_features.get(c, 0.0)
            xgb_prob = float(xgb_model.predict(xgb.DMatrix(_pred_buffer,
                             feature_names=FEATURE_COLS))[0])
            # Ensemble: weighted average of Elo and XGBoost
            if time_decay:
                # Transition from 95% Elo early to 70% Elo late as XGBoost accumulates data
                total_games = len(games)
                progress = min(1.0, idx / total_games) if total_games > 0 else 0
                ew = 0.95 - 0.25 * progress
            else:
                ew = elo_weight
            final_prob = ew * elo_prob + (1 - ew) * xgb_prob
            xgb_predictions += 1
        else:
            final_prob = elo_prob

        pred_winner = home if final_prob >= 0.5 else away
        actual_winner = home if home_actual == 1 else away
        all_probs.append(final_prob)
        all_actuals.append(home_actual)
        if pred_winner == actual_winner:
            correct += 1

        # Accumulate training data
        if game_features is not None:
            feature_rows.append([game_features[c] for c in FEATURE_COLS])
            label_rows.append(home_actual)

        # Retrain XGBoost periodically
        if (len(feature_rows) >= min_train and
                len(feature_rows) % retrain_every == 0):
            X = np.array(feature_rows)
            y = np.array(label_rows)
            dtrain = xgb.DMatrix(X, label=y, feature_names=FEATURE_COLS)
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": 4,
                "eta": 0.03,
                "subsample": 0.85,
                "colsample_bytree": 0.7,
                "min_child_weight": 5,
                "lambda": 3.0,
                "alpha": 0.5,
                "verbosity": 0,
            }
            xgb_model = xgb.train(params, dtrain, num_boost_round=300,
                                  verbose_eval=False)

        # Update trackers AFTER prediction (no leakage)
        model.update_game(home, away, row["home_score"], row["away_score"],
                          neutral_site=neutral, game_date=game_date)
        home_won = row["home_score"] > row["away_score"]
        tracker.update(home, row["home_score"], row["away_score"], home_won, game_date,
                       is_home=True, opp_elo=model.ratings[away])
        tracker.update(away, row["away_score"], row["home_score"], not home_won, game_date,
                       is_home=False, opp_elo=model.ratings[home])

    n = len(all_probs)
    if n == 0:
        return None

    acc_raw = correct / n * 100
    ll_raw = log_loss_binary(all_actuals, all_probs)
    brier_raw = brier_score_binary(all_actuals, all_probs)

    # Fit both calibrators on ensemble probs
    scaler = fit_platt_scaler(all_probs, all_actuals)
    cal_probs = [apply_platt(p, scaler) for p in all_probs]
    cal_correct = sum(1 for p, a in zip(cal_probs, all_actuals) if (p >= 0.5) == (a == 1))
    acc_cal = cal_correct / n * 100
    ll_cal = log_loss_binary(all_actuals, cal_probs)
    brier_cal = brier_score_binary(all_actuals, cal_probs)

    # Also fit isotonic (available for use when data grows)
    iso_scaler = fit_isotonic_scaler(all_probs, all_actuals)
    save_isotonic_scaler(iso_scaler)

    print("\n" + "=" * 60)
    print("  ENHANCED TEST: %s" % label)
    print("=" * 60)
    print("  Games tested:     %d" % n)
    print("  XGBoost predictions: %d (Elo-only for first %d)" % (xgb_predictions, min_train))
    if time_decay:
        print("  Ensemble weight:  TIME-DECAYED (Elo 95%%->70%%, XGB 5%%->30%%)")
    else:
        print("  Ensemble weight:  Elo=%.0f%% XGB=%.0f%%" % (elo_weight*100, (1-elo_weight)*100))
    print("  --- Raw (uncalibrated) ---")
    print("  Accuracy:         %.2f%%" % acc_raw)
    print("  Log Loss:         %.4f" % ll_raw)
    print("  Brier Score:      %.4f" % brier_raw)
    print("  --- Platt calibrated ---")
    print("  Accuracy:         %.2f%%" % acc_cal)
    print("  Log Loss:         %.4f" % ll_cal)
    print("  Brier Score:      %.4f" % brier_cal)
    print("=" * 60)

    result = {
        "label": label, "n": n,
        "acc_raw": acc_raw, "ll_raw": ll_raw, "brier_raw": brier_raw,
        "acc_cal": acc_cal, "ll_cal": ll_cal, "brier_cal": brier_cal,
        "xgb_model": xgb_model, "scaler": scaler,
        "feature_rows": feature_rows, "label_rows": label_rows,
        "elo_weight": elo_weight, "time_decay": time_decay,
    }

    # SHAP feature importance (XGBoost native, no extra deps)
    if xgb_model is not None and feature_rows:
        shap_feature_importance(xgb_model, feature_rows)

    return result


def shap_feature_importance(xgb_model=None, feature_rows=None):
    """SHAP feature importance using XGBoost native pred_contribs (no extra deps).
    Reveals which of the 20 features drive XGBoost predictions."""
    if xgb_model is None:
        # Try loading saved model + features
        xgb_model, _meta = load_enhanced_model()
        if xgb_model is None:
            print("  No trained XGBoost model found. Run 'enhanced' first.")
            return None
        if feature_rows is None and os.path.exists("nba_enhanced_features.npz"):
            data = np.load("nba_enhanced_features.npz")
            feature_rows = data["features"].tolist()
        if feature_rows is None:
            print("  No feature data found. Run 'enhanced' first.")
            return None

    X = np.array(feature_rows)
    dmatrix = xgb.DMatrix(X, feature_names=FEATURE_COLS)
    # pred_contribs returns [n_samples, n_features+1], last col = bias
    contribs = xgb_model.predict(dmatrix, pred_contribs=True)
    mean_abs_shap = np.mean(np.abs(contribs[:, :-1]), axis=0)

    importance = sorted(zip(FEATURE_COLS, mean_abs_shap), key=lambda x: -x[1])
    max_val = importance[0][1] if importance else 1

    print("\n" + "=" * 55)
    print("  SHAP FEATURE IMPORTANCE (XGBoost)")
    print("=" * 55)
    for feat, val in importance:
        bar_len = int(25 * val / max_val) if max_val > 0 else 0
        print("  %-16s  %.4f  %s" % (feat, val, "|" * bar_len))
    print("=" * 55)

    # Check if elo_prob dominates (making XGBoost redundant)
    elo_share = next((v for f, v in importance if f == "elo_prob"), 0) / max(sum(v for _, v in importance), 1e-8)
    if elo_share > 0.5:
        print("  NOTE: elo_prob contributes >50%% of SHAP signal — XGBoost may be mostly echoing Elo")
    else:
        print("  Rolling features contribute meaningful signal beyond Elo")
    return importance


def save_enhanced_model(result, filename=ENHANCED_MODEL_FILE):
    """Save XGBoost model, scaler, and feature data for live predictions + SHAP."""
    if result is None or result.get("xgb_model") is None:
        return
    result["xgb_model"].save_model("nba_xgb_model.json")
    payload = {
        "scaler": result["scaler"],
        "elo_weight": result.get("elo_weight", 0.8),
        "feature_cols": FEATURE_COLS,
        "trained_games": result["n"],
        "time_decay": result.get("time_decay", False),
        "metrics": {
            "acc_cal": result["acc_cal"],
            "ll_cal": result["ll_cal"],
            "brier_cal": result["brier_cal"],
        },
    }
    with open(filename, "w") as f:
        json.dump(payload, f, indent=2)
    # Save feature data for standalone SHAP analysis
    if result.get("feature_rows"):
        np.savez_compressed("nba_enhanced_features.npz",
                            features=np.array(result["feature_rows"]),
                            labels=np.array(result["label_rows"]))
    logging.info("Enhanced model saved -> %s + nba_xgb_model.json", filename)


def load_enhanced_model(filename=ENHANCED_MODEL_FILE):
    """Load XGBoost model for live predictions."""
    if not os.path.exists(filename) or not os.path.exists("nba_xgb_model.json"):
        return None, None
    with open(filename, "r") as f:
        meta = json.load(f)
    # Check feature count matches current FEATURE_COLS
    saved_cols = meta.get("feature_cols", [])
    if len(saved_cols) != len(FEATURE_COLS):
        logging.warning("Saved XGBoost model has %d features, current code expects %d. "
                        "Run 'enhanced' to retrain.", len(saved_cols), len(FEATURE_COLS))
        return None, None
    xgb_model = xgb.Booster()
    xgb_model.load_model("nba_xgb_model.json")
    return xgb_model, meta


if __name__ == "__main__":
    import sys
    ew = float(sys.argv[1]) if len(sys.argv) > 1 else 0.70
    result = run_enhanced_backtest(elo_weight=ew, label="elo_w=%.1f" % ew)
    if result and result.get("xgb_model"):
        save_enhanced_model(result)
