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
from elo_model import NHLElo
from data_players import load_player_stats
from metrics import log_loss_binary, brier_score_binary
from platt import (fit_platt_scaler, apply_platt,
                   fit_isotonic_scaler, apply_isotonic, save_isotonic_scaler)

ENHANCED_MODEL_FILE = "nhl_enhanced_model.json"
WINDOW = 10  # rolling window for team stats
PYTH_EXP = 2.05  # NHL Pythagorean exponent


class TeamTracker:
    """Tracks rolling stats per team from game-by-game data."""

    def __init__(self):
        self.points_scored = defaultdict(list)
        self.points_allowed = defaultdict(list)
        self.results = defaultdict(list)
        self.margins = defaultdict(list)
        self.streak = defaultdict(int)
        self.last_date = {}
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

    def get_features(self, team, game_date=None, is_home=True):
        scored = self.points_scored.get(team, [])
        allowed = self.points_allowed.get(team, [])
        results = self.results.get(team, [])
        margins = self.margins.get(team, [])

        n = len(scored)
        if n < 3:
            return None

        rpg = np.mean(scored[-WINDOW:])
        rapg = np.mean(allowed[-WINDOW:])
        ppg = rpg
        papg = rapg
        win_pct = np.mean(results[-WINDOW:])
        avg_margin = np.mean(margins[-WINDOW:])
        off_rating = ppg
        def_rating = papg

        # Pythagorean win expectation
        rs_exp = max(rpg, 0.1) ** PYTH_EXP
        ra_exp = max(rapg, 0.1) ** PYTH_EXP
        pyth = rs_exp / (rs_exp + ra_exp)

        # Margin consistency (lower std = more predictable)
        consistency = float(np.std(margins[-WINDOW:])) if len(margins) >= 3 else 2.0

        # Trend: recent 5 vs full window win%
        recent_5 = np.mean(results[-5:]) if len(results) >= 5 else win_pct
        trend = recent_5 - win_pct

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

        score_vol = float(np.std(scored[-WINDOW:])) if len(scored) >= 5 else 1.5

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

        # --- Prediction edge features ---

        # Pythagorean regression risk: sample-size weighted
        games_played = n
        pyth_regression_risk = abs(pyth_residual) * min(1.0, games_played / 40.0)

        # Win streak momentum: weighted recent form
        if len(results) >= 3:
            last3_m = [1.0 if r >= 0.5 else -1.0 for r in results[-3:]]
            last5_m = [1.0 if r >= 0.5 else -1.0 for r in results[-5:]] if len(results) >= 5 else last3_m
            streak_momentum = sum(last3_m) / 3.0 * 0.6 + sum(last5_m) / max(len(last5_m), 1) * 0.4
        else:
            streak_momentum = 0.0

        # Home/away split: performance differential
        ha_home_results = self.home_results.get(team, [])
        ha_away_results = self.away_results.get(team, [])
        ha_home_wp = sum(ha_home_results) / max(len(ha_home_results), 1)
        ha_away_wp = sum(ha_away_results) / max(len(ha_away_results), 1)
        home_away_split = ha_home_wp - ha_away_wp

        # Rest quality (quadratic): U-shaped curve, 2 days optimal for NHL
        rest_quality = 0.0
        if rest is not None:
            rest_quality = 1.0 - ((rest - 2.0) ** 2) / 4.0
            rest_quality = max(-1.0, min(1.0, rest_quality))

        return {
            "ppg": ppg,
            "papg": papg,
            "win_pct": win_pct,
            "avg_margin": avg_margin,
            "off_rating": off_rating,
            "def_rating": def_rating,
            "pyth": pyth,
            "streak": float(self.streak.get(team, 0)),
            "consistency": consistency,
            "trend": trend,
            "rest_days": rest,
            "games_played": n,
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
            # Prediction edge features
            "pyth_regression_risk": pyth_regression_risk,
            "streak_momentum": streak_momentum,
            "home_away_split": home_away_split,
            "rest_quality": rest_quality,
        }

    def update(self, team, pts_scored, pts_allowed, won, game_date=None,
               is_home=True, opp_elo=1500.0):
        self.points_scored[team].append(pts_scored)
        self.points_allowed[team].append(pts_allowed)
        self.results[team].append(1.0 if won else 0.0)
        self.margins[team].append(pts_scored - pts_allowed)
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
        # Track close games (1-goal margin) and blowouts (3+ goals)
        margin = abs(pts_scored - pts_allowed)
        if margin <= 1:
            self.close_results[team].append(1.0 if won else 0.0)
            if len(self.close_results[team]) > 15:
                self.close_results[team] = self.close_results[team][-15:]
        if margin >= 3:
            self.blowout_results[team].append(1.0 if won else 0.0)
            if len(self.blowout_results[team]) > 15:
                self.blowout_results[team] = self.blowout_results[team][-15:]
        if game_date is not None:
            self.last_date[team] = game_date


def compute_team_stats(player_df, adv_df=None):
    """Aggregate player-level stats to team-level averages (NHL-specific)."""
    from config import TEAM_ABBR
    abbr_to_full = {v: k for k, v in TEAM_ABBR.items()}
    team_stats = {}
    if player_df is None or player_df.empty:
        return team_stats
    for abbr in player_df["Tm"].unique():
        team_name = abbr_to_full.get(abbr, abbr)
        tp = player_df[player_df["Tm"] == abbr]
        stats = {}
        # Skater stats: PTS, G, A, +/-
        for col, key in [("PTS", "pts"), ("G", "goals"), ("A", "assists"), ("+/-", "plus_minus")]:
            if col in tp.columns:
                vals = pd.to_numeric(tp[col], errors="coerce").dropna()
                stats[key] = float(vals.mean()) if len(vals) > 0 else 0.0
            else:
                stats[key] = 0.0
        # Goalie stats: GAA, SV%
        for col, key in [("GAA", "gaa"), ("SV%", "save_pct")]:
            if col in tp.columns:
                vals = pd.to_numeric(tp[col], errors="coerce").dropna()
                stats[key] = float(vals.mean()) if len(vals) > 0 else 0.0
            else:
                stats[key] = 0.0
        # Goals per game (from team scoring data)
        if "G" in tp.columns:
            gvals = pd.to_numeric(tp["G"], errors="coerce").dropna()
            stats["goals_pg"] = float(gvals.sum()) / max(1, len(tp)) if len(gvals) > 0 else 0.0
        else:
            stats["goals_pg"] = 0.0
        team_stats[team_name] = stats
    return team_stats


def build_game_features(home_feats, away_feats, elo_prob, elo_diff,
                        player_diff=0.0, goalie_diff=0.0,
                        h_goalie_rating=0.0, a_goalie_rating=0.0,
                        day_of_week=3, month=1,
                        elo_rating_h=1500.0, elo_rating_a=1500.0,
                        h_team_stats=None, a_team_stats=None):
    if home_feats is None or away_feats is None:
        return None
    hs = h_team_stats or {}
    as_ = a_team_stats or {}
    return {
        "elo_prob": elo_prob,
        "elo_diff": elo_diff,
        "player_diff": player_diff,
        "goalie_diff": goalie_diff,
        "h_goalie": h_goalie_rating,
        "a_goalie": a_goalie_rating,
        "h_ppg": home_feats["ppg"],
        "h_papg": home_feats["papg"],
        "h_win_pct": home_feats["win_pct"],
        "h_margin": home_feats["avg_margin"],
        "a_ppg": away_feats["ppg"],
        "a_papg": away_feats["papg"],
        "a_win_pct": away_feats["win_pct"],
        "a_margin": away_feats["avg_margin"],
        "ppg_diff": home_feats["ppg"] - away_feats["ppg"],
        "papg_diff": home_feats["papg"] - away_feats["papg"],
        "win_pct_diff": home_feats["win_pct"] - away_feats["win_pct"],
        "margin_diff": home_feats["avg_margin"] - away_feats["avg_margin"],
        "off_diff": home_feats["off_rating"] - away_feats["off_rating"],
        "def_diff": home_feats["def_rating"] - away_feats["def_rating"],
        "h_rest": home_feats["rest_days"] if home_feats["rest_days"] is not None else 1.0,
        "a_rest": away_feats["rest_days"] if away_feats["rest_days"] is not None else 1.0,
        "rest_diff": (home_feats["rest_days"] or 1) - (away_feats["rest_days"] or 1),
        "h_pyth": home_feats["pyth"],
        "a_pyth": away_feats["pyth"],
        "pyth_diff": home_feats["pyth"] - away_feats["pyth"],
        "h_streak": home_feats["streak"],
        "a_streak": away_feats["streak"],
        "streak_diff": home_feats["streak"] - away_feats["streak"],
        "h_consistency": home_feats["consistency"],
        "a_consistency": away_feats["consistency"],
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
        "h_score_vol": home_feats.get("score_vol", 1.5),
        "a_score_vol": away_feats.get("score_vol", 1.5),
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
        # --- NHL-specific player stat features ---
        "h_save_pct": hs.get("save_pct", 0.0),
        "a_save_pct": as_.get("save_pct", 0.0),
        "save_pct_diff": hs.get("save_pct", 0.0) - as_.get("save_pct", 0.0),
        "h_goals_pg": hs.get("goals_pg", 0.0),
        "a_goals_pg": as_.get("goals_pg", 0.0),
        "goals_pg_diff": hs.get("goals_pg", 0.0) - as_.get("goals_pg", 0.0),
        "h_plus_minus": hs.get("plus_minus", 0.0),
        "a_plus_minus": as_.get("plus_minus", 0.0),
        "plus_minus_diff": hs.get("plus_minus", 0.0) - as_.get("plus_minus", 0.0),
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
        # Prediction edge features
        "h_pyth_regression": home_feats.get("pyth_regression_risk", 0.0),
        "a_pyth_regression": away_feats.get("pyth_regression_risk", 0.0),
        "pyth_regression_diff": home_feats.get("pyth_regression_risk", 0.0) - away_feats.get("pyth_regression_risk", 0.0),
        "h_streak_momentum": home_feats.get("streak_momentum", 0.0),
        "a_streak_momentum": away_feats.get("streak_momentum", 0.0),
        "streak_momentum_diff": home_feats.get("streak_momentum", 0.0) - away_feats.get("streak_momentum", 0.0),
        "h_ha_split": home_feats.get("home_away_split", 0.0),
        "a_ha_split": away_feats.get("home_away_split", 0.0),
        "ha_split_diff": home_feats.get("home_away_split", 0.0) - away_feats.get("home_away_split", 0.0),
        "h_rest_quality": home_feats.get("rest_quality", 0.0),
        "a_rest_quality": away_feats.get("rest_quality", 0.0),
        "rest_quality_diff": home_feats.get("rest_quality", 0.0) - away_feats.get("rest_quality", 0.0),
    }


FEATURE_COLS = [
    "elo_prob", "elo_diff", "player_diff",
    "goalie_diff", "h_goalie", "a_goalie",
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
    # NHL-specific player stat features
    "h_save_pct", "a_save_pct", "save_pct_diff",
    "h_goals_pg", "a_goals_pg", "goals_pg_diff",
    "h_plus_minus", "a_plus_minus", "plus_minus_diff",
    # Advanced occult features
    "h_pyth_resid", "a_pyth_resid", "pyth_resid_diff",
    "h_decay_ppg", "a_decay_ppg", "decay_ppg_diff",
    "h_decay_papg", "a_decay_papg", "decay_papg_diff",
    "h_momentum", "a_momentum", "momentum_diff",
    "h_def_trend", "a_def_trend", "def_trend_diff",
    "h_off_trend", "a_off_trend", "off_trend_diff",
    "h_sos_adj_wp", "a_sos_adj_wp", "sos_adj_wp_diff",
    # Prediction edge features
    "h_pyth_regression", "a_pyth_regression", "pyth_regression_diff",
    "h_streak_momentum", "a_streak_momentum", "streak_momentum_diff",
    "h_ha_split", "a_ha_split", "ha_split_diff",
    "h_rest_quality", "a_rest_quality", "rest_quality_diff",
]


def run_enhanced_backtest(csv_file=GAMES_FILE, min_train=200, retrain_every=50,
                          elo_weight=0.5, label="enhanced", time_decay=False):
    if not os.path.exists(csv_file):
        print("ERROR: %s not found" % csv_file)
        return None

    settings = load_elo_settings()
    _elo_keys = {"base_rating", "k", "home_adv", "use_mov", "player_boost",
                 "starter_boost", "rest_factor", "form_weight", "travel_factor",
                 "sos_factor", "pace_factor", "playoff_hca_factor",
                 "division_factor", "mean_reversion",
                 "b2b_penalty", "road_trip_factor", "homestand_factor",
                 "win_streak_factor", "altitude_factor", "season_phase_factor",
                 "scoring_consistency_factor", "rest_advantage_cap", "overtime_factor"}
    model = NHLElo(**{k: v for k, v in settings.items() if k in _elo_keys})
    from build_model import _calc_altitude_bonus
    model._altitude_bonus = _calc_altitude_bonus(csv_file)
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
    feature_rows = []
    label_rows = []

    all_probs = []
    all_actuals = []
    correct = 0
    xgb_model = None
    xgb_predictions = 0
    _pred_buffer = np.zeros((1, len(FEATURE_COLS)), dtype=np.float32)

    prev_season = None
    for idx, row in tqdm(games.iterrows(), total=len(games),
                         desc="  Enhanced backtest", leave=True):
        game_date = row["_date_parsed"] if pd.notna(row["_date_parsed"]) else None
        home = row["home_team"]
        away = row["away_team"]
        neutral = bool(row["neutral_site"])
        home_actual = 1 if row["home_score"] > row["away_score"] else 0

        # Detect season boundary and regress goalie ratings
        if game_date is not None:
            row_season = game_date.year + 1 if game_date.month >= 10 else game_date.year
            if prev_season is not None and row_season != prev_season:
                model.regress_goalie_ratings(factor=0.5)
            prev_season = row_season

        home_starter = str(row.get("home_starter", "") or "").strip()
        away_starter = str(row.get("away_starter", "") or "").strip()

        elo_prob = model.win_prob(home, away, team_a_home=True,
                                  neutral_site=neutral, calibrated=False,
                                  game_date=game_date, use_injuries=False,
                                  home_starter=home_starter,
                                  away_starter=away_starter)
        elo_diff = model.ratings[home] - model.ratings[away]
        player_diff = 0.0
        if model._player_scores and model.player_boost > 0:
            player_diff = (model._player_scores.get(home, 0.0) -
                           model._player_scores.get(away, 0.0))

        h_gr = model._goalie_ratings.get(home_starter, 0.0)
        a_gr = model._goalie_ratings.get(away_starter, 0.0)

        home_feats = tracker.get_features(home, game_date, is_home=True)
        away_feats = tracker.get_features(away, game_date, is_home=False)
        dow = game_date.dayofweek if game_date is not None else 3
        mon = game_date.month if game_date is not None else 1
        game_features = build_game_features(home_feats, away_feats, elo_prob,
                                            elo_diff, player_diff,
                                            h_gr - a_gr, h_gr, a_gr,
                                            day_of_week=dow, month=mon,
                                            elo_rating_h=model.ratings[home],
                                            elo_rating_a=model.ratings[away],
                                            h_team_stats=team_stats.get(home),
                                            a_team_stats=team_stats.get(away))

        if xgb_model is not None and game_features is not None:
            for j, c in enumerate(FEATURE_COLS):
                _pred_buffer[0, j] = game_features.get(c, 0.0)
            xgb_prob = float(xgb_model.predict(xgb.DMatrix(_pred_buffer,
                             feature_names=FEATURE_COLS))[0])
            if time_decay:
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

        if game_features is not None:
            feature_rows.append([game_features[c] for c in FEATURE_COLS])
            label_rows.append(home_actual)

        if (len(feature_rows) >= min_train and
                len(feature_rows) % retrain_every == 0):
            X = np.array(feature_rows)
            y = np.array(label_rows)
            dtrain = xgb.DMatrix(X, label=y, feature_names=FEATURE_COLS)
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": 3,
                "eta": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.6,
                "min_child_weight": 10,
                "lambda": 5.0,
                "alpha": 1.0,
                "verbosity": 0,
            }
            xgb_model = xgb.train(params, dtrain, num_boost_round=200,
                                  verbose_eval=False)

        model.update_game(home, away, row["home_score"], row["away_score"],
                          neutral_site=neutral, game_date=game_date,
                          home_starter=home_starter, away_starter=away_starter)
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

    scaler = fit_platt_scaler(all_probs, all_actuals)
    cal_probs = [apply_platt(p, scaler) for p in all_probs]
    cal_correct = sum(1 for p, a in zip(cal_probs, all_actuals) if (p >= 0.5) == (a == 1))
    acc_cal = cal_correct / n * 100
    ll_cal = log_loss_binary(all_actuals, cal_probs)
    brier_cal = brier_score_binary(all_actuals, cal_probs)

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

    if xgb_model is not None and feature_rows:
        shap_feature_importance(xgb_model, feature_rows)

    return result


def shap_feature_importance(xgb_model=None, feature_rows=None):
    if xgb_model is None:
        xgb_model, _meta = load_enhanced_model()
        if xgb_model is None:
            print("  No trained XGBoost model found. Run 'enhanced' first.")
            return None
        if feature_rows is None and os.path.exists("nhl_enhanced_features.npz"):
            data = np.load("nhl_enhanced_features.npz")
            feature_rows = data["features"].tolist()
        if feature_rows is None:
            print("  No feature data found. Run 'enhanced' first.")
            return None

    X = np.array(feature_rows)
    dmatrix = xgb.DMatrix(X, feature_names=FEATURE_COLS)
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

    elo_share = next((v for f, v in importance if f == "elo_prob"), 0) / max(sum(v for _, v in importance), 1e-8)
    if elo_share > 0.5:
        print("  NOTE: elo_prob contributes >50%% of SHAP signal -- XGBoost may be mostly echoing Elo")
    else:
        print("  Rolling features contribute meaningful signal beyond Elo")
    return importance


def save_enhanced_model(result, filename=ENHANCED_MODEL_FILE):
    if result is None or result.get("xgb_model") is None:
        return
    result["xgb_model"].save_model("nhl_xgb_model.json")
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
    if result.get("feature_rows"):
        np.savez_compressed("nhl_enhanced_features.npz",
                            features=np.array(result["feature_rows"]),
                            labels=np.array(result["label_rows"]))
    logging.info("Enhanced model saved -> %s + nhl_xgb_model.json", filename)


def load_enhanced_model(filename=ENHANCED_MODEL_FILE):
    if not os.path.exists(filename) or not os.path.exists("nhl_xgb_model.json"):
        return None, None
    with open(filename, "r") as f:
        meta = json.load(f)
    # Check for feature count mismatch (model was trained with different features)
    saved_cols = meta.get("feature_cols", [])
    if saved_cols and len(saved_cols) != len(FEATURE_COLS):
        logging.warning(
            "XGBoost feature mismatch: saved model has %d features, current code has %d. "
            "Run 'enhanced' to retrain.", len(saved_cols), len(FEATURE_COLS))
        return None, None
    xgb_model = xgb.Booster()
    xgb_model.load_model("nhl_xgb_model.json")
    return xgb_model, meta


if __name__ == "__main__":
    import sys
    ew = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    result = run_enhanced_backtest(elo_weight=ew, label="elo_w=%.1f" % ew)
    if result and result.get("xgb_model"):
        save_enhanced_model(result)
