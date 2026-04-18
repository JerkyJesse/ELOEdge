"""Backtest, grid search, genetic optimization, and advanced validation methods."""

import os
import json
import logging
from itertools import product, combinations
from collections import defaultdict
from math import comb as math_comb

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import norm as norm_dist
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *a, **kw):
        return iterable if iterable is not None else range(0)
    tqdm.write = print

from config import GAMES_FILE, load_elo_settings, save_elo_settings
from color_helpers import cok, cwarn, cdim, chi, div
from elo_model import NBAElo
from data_players import load_player_stats, build_league_player_scores
from metrics import (log_loss_binary, brier_score_binary, calibration_table,
                     ece_score, mce_score, brier_skill_score,
                     conformal_prediction_set)
from platt import (fit_platt_scaler, save_platt_scaler, load_platt_scaler,
                   apply_platt, regress_ratings_to_mean)
from build_model import _calc_altitude_bonus

# All Elo constructor keys
_ELO_KEYS = {"base_rating", "k", "home_adv", "use_mov", "player_boost",
             "rest_factor", "form_weight", "travel_factor", "sos_factor",
             "playoff_hca_factor", "pace_factor", "division_factor", "mean_reversion",
             "b2b_penalty", "road_trip_factor", "homestand_factor", "win_streak_factor",
             "altitude_factor", "season_phase_factor", "scoring_consistency_factor",
             "rest_advantage_cap"}


def _save_backtest_state(model, state_file):
    """Save full model state for incremental resume."""
    import time as _time
    state = {
        "ratings": dict(model.ratings),
        "pitcher_ratings": dict(getattr(model, "_pitcher_ratings", {})),
        "goalie_ratings": dict(getattr(model, "_goalie_ratings", {})),
        "player_scores": {k: float(v) for k, v in getattr(model, "_player_scores", {}).items()},
        "rest_days": {k: (str(v) if hasattr(v, 'isoformat') else v)
                      for k, v in getattr(model, "_rest_days", {}).items()},
        "recent_results": {k: list(v) for k, v in getattr(model, "_recent_results", {}).items()},
        "last_game_dates": {k: str(v) for k, v in getattr(model, "_last_game_dates", {}).items()},
        "games_played": dict(getattr(model, "_games_played", {})),
        "saved_at": _time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2, default=str)


def _restore_backtest_state(state, model):
    """Restore model internal state from saved dict (keeps existing settings)."""
    model.ratings = state.get("ratings", model.ratings)
    if state.get("pitcher_ratings"):
        model._pitcher_ratings = state["pitcher_ratings"]
    if state.get("goalie_ratings"):
        model._goalie_ratings = state["goalie_ratings"]
    if state.get("player_scores"):
        model._player_scores = state["player_scores"]
    if state.get("rest_days"):
        model._rest_days = state["rest_days"]
    if state.get("recent_results"):
        model._recent_results = {k: list(v) for k, v in state["recent_results"].items()}
    if state.get("last_game_dates"):
        model._last_game_dates = state["last_game_dates"]
    if state.get("games_played"):
        model._games_played = state["games_played"]
    return model


def backtest_model(csv_file=GAMES_FILE, output_csv="nba_backtest_predictions.csv",
                   calibration_csv="nba_calibration.csv", k=None, home_adv=None,
                   model=None, fit_platt=False, resume=False):
    """
    fit_platt=True: fit and save Platt scaler from this run's raw probs.
    Only pass fit_platt=True on the main/user-facing backtest call,
    not inside optimizer loops (leakage + speed).
    resume=True: load previous predictions + model state, only process new games.
    """
    if not os.path.exists(csv_file):
        return False, {}

    state_file = output_csv.replace("_predictions.csv", "_state.json")

    # --- Resume: load previous predictions + state ---
    old_preds, old_probs, old_actuals = [], [], []
    last_pred_date = None
    saved_state = None

    if resume and model is None and os.path.exists(output_csv) and os.path.exists(state_file):
        try:
            old_df = pd.read_csv(output_csv)
            if len(old_df) > 0:
                last_pred_date = str(old_df.iloc[-1]["date"])
                old_preds = old_df.to_dict("records")
                old_probs = old_df["home_win_prob"].tolist()
                old_actuals = [1 if r["actual_winner"] == r["home_team"] else 0
                               for r in old_preds]
            with open(state_file) as f:
                saved_state = json.load(f)
        except Exception as e:
            logging.warning("Resume load failed, running full backtest: %s", e)
            old_preds, old_probs, old_actuals = [], [], []
            last_pred_date = None
            saved_state = None

    if model is None:
        settings = load_elo_settings()
        model = NBAElo(**{k_: v for k_, v in settings.items() if k_ in _ELO_KEYS})
        if k is not None:
            model.k = k
        if home_adv is not None:
            model.home_adv = home_adv
        model._altitude_bonus = _calc_altitude_bonus(csv_file)
        player_df = load_player_stats()
        if not player_df.empty:
            model.set_player_stats(player_df)

    # Restore saved state if resuming
    if saved_state and last_pred_date:
        _restore_backtest_state(saved_state, model)

    games = pd.read_csv(csv_file)
    if "neutral_site" not in games.columns:
        games["neutral_site"] = False
    # Parse dates for rest-day calculations
    games["_date_parsed"] = pd.to_datetime(games["date"], errors="coerce")

    # Filter to new games only if resuming
    if last_pred_date and saved_state:
        cutoff = pd.to_datetime(last_pred_date)
        games = games[games["_date_parsed"] > cutoff].copy()
        if len(games) == 0:
            print("  No new games since %s (resume)" % last_pred_date)
            # Return metrics from old predictions
            if old_probs:
                brier_val = brier_score_binary(old_actuals, old_probs)
                ll_val = log_loss_binary(old_actuals, old_probs)
                acc_val = sum(1 for p in old_preds if p["correct"]) / len(old_preds) * 100
                return True, {"accuracy": acc_val, "log_loss": ll_val,
                              "brier": brier_val, "n_games": len(old_preds)}
            return False, {}
        print("  Resuming: %d old + %d new games" % (len(old_preds), len(games)))

    predictions, probs, actuals = [], [], []
    for _, row in games.iterrows():
        try:
            game_date = row["_date_parsed"] if pd.notna(row["_date_parsed"]) else None
            home_win_prob = model.win_prob(
                row["home_team"], row["away_team"],
                team_a_home=True, neutral_site=bool(row["neutral_site"]),
                calibrated=False, game_date=game_date, use_injuries=False,
            )
            pred_winner   = row["home_team"] if home_win_prob >= 0.5 else row["away_team"]
            home_actual   = 1 if row["home_score"] > row["away_score"] else 0
            actual_winner = row["home_team"] if home_actual == 1 else row["away_team"]
            predictions.append({
                "date": row["date"], "home_team": row["home_team"],
                "away_team": row["away_team"], "home_score": row["home_score"],
                "away_score": row["away_score"], "neutral_site": bool(row["neutral_site"]),
                "home_win_prob": home_win_prob, "pred_winner": pred_winner,
                "actual_winner": actual_winner, "correct": int(pred_winner == actual_winner),
            })
            probs.append(home_win_prob)
            actuals.append(home_actual)
            model.update_game(
                row["home_team"], row["away_team"],
                row["home_score"], row["away_score"],
                neutral_site=bool(row["neutral_site"]),
                game_date=game_date,
            )
        except Exception as e:
            logging.warning("Backtest row error: %s", e)
    if not predictions and not old_preds:
        return False, {}

    # Merge old + new predictions
    all_predictions = old_preds + predictions
    all_probs = old_probs + probs
    all_actuals = old_actuals + actuals

    pred_df = pd.DataFrame(all_predictions)
    pred_df.to_csv(output_csv, index=False)
    calibration_table(all_probs, all_actuals).to_csv(calibration_csv, index=False)

    # Save model state for future resume
    _save_backtest_state(model, state_file)

    # Use merged lists for metrics below
    probs = all_probs
    actuals = all_actuals

    if fit_platt and len(probs) >= 100:
        scaler     = fit_platt_scaler(probs, actuals)
        save_platt_scaler(scaler)
        model._platt_scaler = scaler
        cal_probs  = [apply_platt(p, scaler) for p in probs]
        brier_raw  = brier_score_binary(actuals, probs)
        brier_cal  = brier_score_binary(actuals, cal_probs)
        ll_cal     = log_loss_binary(actuals, cal_probs)
        raw_s      = cwarn("%.4f" % brier_raw)
        cal_s      = cok("%.4f" % brier_cal)
        print("\n  %s" % chi("Platt scaler fitted and saved."))
        print("  Brier (raw):        %s" % raw_s)
        print("  Brier (calibrated): %s  <- use this going forward" % cal_s)
        print("  LogLoss (cal):      %.4f" % ll_cal)
        # ECE, MCE, BSS
        ece = ece_score(cal_probs, actuals)
        mce = mce_score(cal_probs, actuals)
        home_rate = float(np.mean(actuals))
        bss_50 = brier_skill_score(cal_probs, actuals)
        bss_home = brier_skill_score(cal_probs, actuals, [home_rate] * len(actuals))
        print("  ECE:                %.4f%s" % (ece, "  (excellent)" if ece < 0.03 else "  (needs work)" if ece > 0.08 else ""))
        print("  MCE:                %.4f" % mce)
        print("  BSS vs 50%%:         %.4f  |  BSS vs home-rate (%.0f%%): %.4f"
              % (bss_50, home_rate * 100, bss_home))
    else:
        brier_cal = brier_score_binary(actuals, probs)
        ll_cal    = log_loss_binary(actuals, probs)

    metrics = {
        "accuracy": float(pred_df["correct"].mean() * 100.0),
        "log_loss": ll_cal,
        "brier":    brier_cal,
        "n_games":  int(len(pred_df)),
    }
    return True, metrics


_OPT_KEYS = ("k", "home_adv", "player_boost", "rest_factor",
             "travel_factor", "sos_factor", "playoff_hca_factor", "pace_factor",
             "form_weight", "division_factor", "mean_reversion",
             "b2b_penalty", "road_trip_factor", "homestand_factor",
             "win_streak_factor", "altitude_factor", "season_phase_factor",
             "scoring_consistency_factor", "rest_advantage_cap")


# ── Fast Numpy-Based Evaluation for Optimizer Inner Loop ─────────────
#
# Pre-computes all game data into numpy arrays once, then evaluates
# any parameter set via a tight sequential loop with direct array
# indexing -- no NBAElo object creation, no pandas iteration, no dict
# lookups inside the hot loop.
#
# Usage:
#   precomp = precompute_games(csv_file, player_scores, altitude_bonus)
#   metrics = fast_evaluate(precomp, k=8.5, home_adv=32.0, ...)
#
# Returns dict with 'accuracy', 'log_loss', 'brier', 'n_games'.

def precompute_games(csv_file, player_scores=None, altitude_bonus=None,
                     base_rating=1500.0):
    """Load all games into numpy arrays for fast_evaluate.

    Returns a dict of pre-computed arrays and lookup tables.
    Called ONCE before any optimizer loop.
    """
    import math as _math
    from config import same_division

    games = pd.read_csv(csv_file)
    if "neutral_site" not in games.columns:
        games["neutral_site"] = False
    games["_date_parsed"] = pd.to_datetime(games["date"], errors="coerce")
    n_games = len(games)

    # --- Build team ID mapping ---
    all_teams = sorted(set(games["home_team"].unique()) | set(games["away_team"].unique()))
    team_to_id = {t: i for i, t in enumerate(all_teams)}
    n_teams = len(all_teams)

    # --- Pre-compute timezone map (team_id -> tz offset) ---
    from elo_model import TEAM_TIMEZONE
    tz_map = np.full(n_teams, -6.0, dtype=np.float64)  # default Central
    for t, tz in TEAM_TIMEZONE.items():
        if t in team_to_id:
            tz_map[team_to_id[t]] = float(tz)

    # --- Pre-compute division pairs ---
    same_div = np.zeros((n_teams, n_teams), dtype=np.bool_)
    for i, t1 in enumerate(all_teams):
        for j, t2 in enumerate(all_teams):
            if i != j and same_division(t1, t2):
                same_div[i, j] = True

    # --- Pre-compute player scores per team ---
    player_score_arr = np.zeros(n_teams, dtype=np.float64)
    if player_scores:
        for t, sc in player_scores.items():
            if t in team_to_id:
                player_score_arr[team_to_id[t]] = float(sc)

    # --- Pre-compute altitude bonus per team ---
    alt_bonus_arr = np.zeros(n_teams, dtype=np.float64)
    if altitude_bonus:
        for t, ab in altitude_bonus.items():
            if t in team_to_id:
                alt_bonus_arr[team_to_id[t]] = float(ab)

    # --- Build per-game arrays ---
    home_id = np.empty(n_games, dtype=np.int32)
    away_id = np.empty(n_games, dtype=np.int32)
    home_score = np.empty(n_games, dtype=np.float64)
    away_score = np.empty(n_games, dtype=np.float64)
    neutral = np.zeros(n_games, dtype=np.bool_)
    home_win = np.empty(n_games, dtype=np.float64)  # 1.0 or 0.0
    mov_log = np.empty(n_games, dtype=np.float64)  # log(|margin|+1)

    # Date ordinals for rest-day calculation (NaT -> -9999)
    date_ordinal = np.full(n_games, -9999, dtype=np.int64)

    # Playoff indicator
    is_playoff = np.zeros(n_games, dtype=np.bool_)

    for idx in range(n_games):
        row = games.iloc[idx]
        h = team_to_id[row["home_team"]]
        a = team_to_id[row["away_team"]]
        home_id[idx] = h
        away_id[idx] = a
        hs = float(row["home_score"])
        as_ = float(row["away_score"])
        home_score[idx] = hs
        away_score[idx] = as_
        neutral[idx] = bool(row["neutral_site"])
        home_win[idx] = 1.0 if hs > as_ else 0.0
        margin = abs(hs - as_)
        mov_log[idx] = _math.log(max(1.0, margin) + 1.0)
        dt = row["_date_parsed"]
        if pd.notna(dt):
            date_ordinal[idx] = dt.toordinal()
            m, d = dt.month, dt.day
            if (m == 4 and d >= 15) or m in (5, 6):
                is_playoff[idx] = True

    return {
        "n_games": n_games,
        "n_teams": n_teams,
        "team_to_id": team_to_id,
        "all_teams": all_teams,
        "home_id": home_id,
        "away_id": away_id,
        "home_score": home_score,
        "away_score": away_score,
        "neutral": neutral,
        "home_win": home_win,
        "mov_log": mov_log,
        "date_ordinal": date_ordinal,
        "is_playoff": is_playoff,
        "tz_map": tz_map,
        "same_div": same_div,
        "player_score_arr": player_score_arr,
        "alt_bonus_arr": alt_bonus_arr,
        "base_rating": base_rating,
    }


def fast_evaluate(pc, k=8.23, home_adv=34.0, use_mov=True,
                  player_boost=35.0, rest_factor=25.0, form_weight=10.0,
                  travel_factor=28.2, sos_factor=0.0,
                  playoff_hca_factor=0.57, pace_factor=35.0,
                  division_factor=10.0, mean_reversion=0.0,
                  b2b_penalty=0.0, road_trip_factor=3.47,
                  homestand_factor=0.0, win_streak_factor=0.0,
                  altitude_factor=0.0, season_phase_factor=0.0,
                  scoring_consistency_factor=0.0, rest_advantage_cap=2.97):
    """Evaluate a parameter set using pre-computed numpy arrays.

    Much faster than building NBAElo + calling backtest_model() because:
    - No object construction overhead
    - No pandas row iteration
    - No dict lookups in hot loop (direct array indexing)
    - No string comparisons
    - Pre-allocated output arrays

    Parameters
    ----------
    pc : dict
        Pre-computed data from precompute_games().
    All other params match NBAElo constructor.

    Returns
    -------
    dict with 'accuracy', 'log_loss', 'brier', 'n_games', or None on failure.
    """
    n_games = pc["n_games"]
    n_teams = pc["n_teams"]
    base = pc["base_rating"]
    home_id = pc["home_id"]
    away_id = pc["away_id"]
    home_score_arr = pc["home_score"]
    away_score_arr = pc["away_score"]
    neutral_arr = pc["neutral"]
    home_win_arr = pc["home_win"]
    mov_log_arr = pc["mov_log"]
    date_ord = pc["date_ordinal"]
    is_playoff = pc["is_playoff"]
    tz_map = pc["tz_map"]
    same_div_mat = pc["same_div"]
    player_sc = pc["player_score_arr"]
    alt_bonus = pc["alt_bonus_arr"]

    if n_games == 0:
        return None

    rest_cap = int(rest_advantage_cap) if rest_advantage_cap > 0 else 3

    # --- Per-team state arrays (reset each evaluation) ---
    elo = np.full(n_teams, base, dtype=np.float64)

    # Last game date ordinal per team (-9999 = unknown)
    last_date = np.full(n_teams, -9999, dtype=np.int64)

    # Last game location (team_id of venue home team, -1 = unknown)
    last_location = np.full(n_teams, -1, dtype=np.int32)

    # Recent results: circular buffer [n_teams, 10]
    recent_results = np.full((n_teams, 10), -1.0, dtype=np.float64)
    recent_count = np.zeros(n_teams, dtype=np.int32)

    # Opponent Elo history: circular buffer [n_teams, 10]
    opp_elos = np.full((n_teams, 10), base, dtype=np.float64)
    opp_count = np.zeros(n_teams, dtype=np.int32)

    # Team scores: circular buffer [n_teams, 10, 2]  (pts_for, pts_against)
    team_scores = np.zeros((n_teams, 10, 2), dtype=np.float64)
    score_count = np.zeros(n_teams, dtype=np.int32)

    # Last margin per team
    last_margin = np.zeros(n_teams, dtype=np.float64)
    has_last_margin = np.zeros(n_teams, dtype=np.bool_)

    # Consecutive home/away
    consec_home = np.zeros(n_teams, dtype=np.int32)
    consec_away = np.zeros(n_teams, dtype=np.int32)

    # Games played per team (for season_phase)
    games_played = np.zeros(n_teams, dtype=np.int32)

    # --- Output arrays ---
    preds = np.empty(n_games, dtype=np.float64)

    # --- Main loop ---
    for g in range(n_games):
        h = home_id[g]
        a = away_id[g]
        is_neutral = neutral_arr[g]
        is_po = is_playoff[g]
        g_date_ord = date_ord[g]

        ra = elo[h]
        rb = elo[a]

        # Home court advantage (reduced in playoffs)
        if not is_neutral:
            hca = home_adv * playoff_hca_factor if is_po else home_adv
            ra += hca
            # Altitude bonus for home team
            ab = alt_bonus[h]
            if ab > 0.0:
                ra += ab * altitude_factor if altitude_factor > 0.0 else ab

        # Player boost
        if player_boost > 0.0:
            ra += player_sc[h] * player_boost
            rb += player_sc[a] * player_boost

        # Rest adjustment
        if rest_factor != 0.0 and g_date_ord > 0:
            ld_h = last_date[h]
            ld_a = last_date[a]
            if ld_h > 0:
                rd_h = g_date_ord - ld_h
                rd_h = min(rd_h, rest_cap)
                ra += rest_factor * (rd_h - 1)
            if ld_a > 0:
                rd_a = g_date_ord - ld_a
                rd_a = min(rd_a, rest_cap)
                rb += rest_factor * (rd_a - 1)

        # Form adjustment (recent win%)
        if form_weight != 0.0:
            rc_h = recent_count[h]
            if rc_h >= 5:
                n_use = min(rc_h, 10)
                # Sum the most recent n_use results
                start_h = rc_h % 10  # next write position = oldest in window
                win_sum_h = 0.0
                for ri in range(n_use):
                    idx = (start_h - n_use + ri) % 10
                    val = recent_results[h, idx]
                    if val >= 0:
                        win_sum_h += val
                ra += form_weight * (win_sum_h / n_use - 0.5)

            rc_a = recent_count[a]
            if rc_a >= 5:
                n_use = min(rc_a, 10)
                start_a = rc_a % 10
                win_sum_a = 0.0
                for ri in range(n_use):
                    idx = (start_a - n_use + ri) % 10
                    val = recent_results[a, idx]
                    if val >= 0:
                        win_sum_a += val
                rb += form_weight * (win_sum_a / n_use - 0.5)

        # Travel adjustment
        if travel_factor > 0.0 and not is_neutral:
            venue = h  # home team's city is the venue
            ll_h = last_location[h]
            if ll_h >= 0:
                tz_diff_h = abs(tz_map[ll_h] - tz_map[venue])
                if tz_diff_h > 0:
                    ra -= travel_factor * tz_diff_h
            ll_a = last_location[a]
            if ll_a >= 0:
                tz_diff_a = abs(tz_map[ll_a] - tz_map[venue])
                if tz_diff_a > 0:
                    rb -= travel_factor * tz_diff_a

        # SOS adjustment
        if sos_factor != 0.0:
            oc_h = opp_count[h]
            if oc_h >= 5:
                n_use = min(oc_h, 10)
                opp_sum_h = 0.0
                for ri in range(n_use):
                    opp_sum_h += opp_elos[h, (oc_h - n_use + ri) % 10]
                ra += sos_factor * (opp_sum_h / n_use - base) / 100.0

            oc_a = opp_count[a]
            if oc_a >= 5:
                n_use = min(oc_a, 10)
                opp_sum_a = 0.0
                for ri in range(n_use):
                    opp_sum_a += opp_elos[a, (oc_a - n_use + ri) % 10]
                rb += sos_factor * (opp_sum_a / n_use - base) / 100.0

        # Pace adjustment
        if pace_factor != 0.0:
            sc_h = score_count[h]
            sc_a = score_count[a]
            if sc_h >= 5 and sc_a >= 5:
                n_h = min(sc_h, 10)
                n_a = min(sc_a, 10)
                pace_h = 0.0
                for ri in range(n_h):
                    idx = (sc_h - n_h + ri) % 10
                    pace_h += team_scores[h, idx, 0] + team_scores[h, idx, 1]
                pace_h /= (2.0 * n_h)
                pace_a = 0.0
                for ri in range(n_a):
                    idx = (sc_a - n_a + ri) % 10
                    pace_a += team_scores[a, idx, 0] + team_scores[a, idx, 1]
                pace_a /= (2.0 * n_a)
                pace_diff = pace_h - pace_a
                ra -= pace_factor * pace_diff / 10.0
                rb += pace_factor * pace_diff / 10.0

        # Division rivalry
        if division_factor != 0.0 and same_div_mat[h, a]:
            diff = elo[h] - elo[a]
            adj = division_factor * diff / 100.0
            ra -= adj
            rb += adj

        # Mean reversion
        if mean_reversion != 0.0:
            if has_last_margin[h] and abs(last_margin[h]) > 15:
                ra -= mean_reversion * last_margin[h] / 100.0
            if has_last_margin[a] and abs(last_margin[a]) > 15:
                rb -= mean_reversion * last_margin[a] / 100.0

        # B2B penalty
        if b2b_penalty != 0.0 and g_date_ord > 0:
            ld_h2 = last_date[h]
            if ld_h2 > 0:
                rd2 = g_date_ord - ld_h2
                if rd2 <= 1:
                    scale = 1.0 if rd2 == 0 else 0.5
                    ra -= b2b_penalty * scale
            ld_a2 = last_date[a]
            if ld_a2 > 0:
                rd2 = g_date_ord - ld_a2
                if rd2 <= 1:
                    scale = 1.0 if rd2 == 0 else 0.5
                    rb -= b2b_penalty * scale

        # Road trip / homestand
        if road_trip_factor != 0.0:
            ca = consec_away[h]
            if ca >= 3:
                ra -= road_trip_factor * min(ca - 2, 5)
            ca2 = consec_away[a]
            if ca2 >= 3:
                rb -= road_trip_factor * min(ca2 - 2, 5)

        if homestand_factor != 0.0:
            ch = consec_home[h]
            if ch >= 3:
                ra += homestand_factor * min(ch - 2, 5)
            ch2 = consec_home[a]
            if ch2 >= 3:
                rb += homestand_factor * min(ch2 - 2, 5)

        # Win streak momentum
        if win_streak_factor != 0.0:
            for team_idx, rating_add in ((h, True), (a, False)):
                rc = recent_count[team_idx]
                if rc >= 2:
                    pos = (rc - 1) % 10
                    last_r = recent_results[team_idx, pos]
                    if last_r >= 0:
                        streak = 1
                        for si in range(2, min(rc, 6)):
                            prev_pos = (rc - si) % 10
                            if recent_results[team_idx, prev_pos] == last_r:
                                streak += 1
                            else:
                                break
                        streak = min(streak, 5)
                        ws_adj = win_streak_factor * streak / 5.0
                        if last_r < 0.5:
                            ws_adj = -ws_adj
                        if rating_add:
                            ra += ws_adj
                        else:
                            rb += ws_adj

        # Scoring consistency
        if scoring_consistency_factor != 0.0:
            for team_idx, is_home in ((h, True), (a, False)):
                sc = score_count[team_idx]
                if sc >= 5:
                    n_use = min(sc, 10)
                    pf_vals_local = np.empty(n_use)
                    for ri in range(n_use):
                        pf_vals_local[ri] = team_scores[team_idx, (sc - n_use + ri) % 10, 0]
                    std = float(np.std(pf_vals_local))
                    adj = -scoring_consistency_factor * (std - 8.0) / 10.0
                    if is_home:
                        ra += adj
                    else:
                        rb += adj

        # Season phase dampener
        if season_phase_factor != 0.0:
            avg_gn = (games_played[h] + games_played[a]) / 2.0
            game_frac = min(avg_gn / 82.0, 1.0)
            if game_frac < 0.20:
                ph_adj = season_phase_factor * (0.20 - game_frac)
                diff = ra - rb
                ra -= ph_adj * diff / 100.0
                rb += ph_adj * diff / 100.0

        # --- Win probability ---
        prob = 1.0 / (1.0 + 10.0 ** (-(ra - rb) / 400.0))
        preds[g] = prob

        # --- Update state (must happen AFTER prediction) ---
        # Opponent Elo tracking
        opp_elos[h, opp_count[h] % 10] = elo[a]
        opp_count[h] += 1
        opp_elos[a, opp_count[a] % 10] = elo[h]
        opp_count[a] += 1

        # Score tracking
        hs = home_score_arr[g]
        as_ = away_score_arr[g]
        team_scores[h, score_count[h] % 10, 0] = hs
        team_scores[h, score_count[h] % 10, 1] = as_
        score_count[h] += 1
        team_scores[a, score_count[a] % 10, 0] = as_
        team_scores[a, score_count[a] % 10, 1] = hs
        score_count[a] += 1

        # Elo update
        ea = 1.0 / (1.0 + 10.0 ** ((elo[a] - elo[h] - (0.0 if neutral_arr[g] else home_adv)) / 400.0))
        hw = home_win_arr[g]
        sa = hw
        sb = 1.0 - hw
        mov = mov_log_arr[g] if use_mov else 1.0
        elo[h] += k * mov * (sa - ea)
        elo[a] += k * mov * (sb - (1.0 - ea))

        # Date tracking
        if g_date_ord > 0:
            last_date[h] = g_date_ord
            last_date[a] = g_date_ord

        # Location tracking
        last_location[h] = h
        last_location[a] = h  # away team was at home team's city

        # Recent results tracking
        recent_results[h, recent_count[h] % 10] = sa
        recent_count[h] += 1
        recent_results[a, recent_count[a] % 10] = sb
        recent_count[a] += 1

        # Margin tracking
        margin = hs - as_
        last_margin[h] = margin
        last_margin[a] = -margin
        has_last_margin[h] = True
        has_last_margin[a] = True

        # Consecutive home/away
        consec_home[h] += 1
        consec_away[h] = 0
        consec_away[a] += 1
        consec_home[a] = 0

        # Games played
        games_played[h] += 1
        games_played[a] += 1

    # --- Compute metrics ---
    outcomes = home_win_arr
    correct = ((preds > 0.5) == (outcomes > 0.5))
    accuracy = float(np.mean(correct) * 100.0)

    # Log loss
    preds_clipped = np.clip(preds, 1e-15, 1.0 - 1e-15)
    ll = -float(np.mean(outcomes * np.log(preds_clipped) +
                        (1.0 - outcomes) * np.log(1.0 - preds_clipped)))

    # Brier score
    brier = float(np.mean((preds - outcomes) ** 2))

    return {"accuracy": accuracy, "log_loss": ll, "brier": brier, "n_games": n_games}


def _fast_eval_from_dict(pc, params_dict, use_mov=True):
    """Thin wrapper: call fast_evaluate from a params dict (optimizer convenience)."""
    return fast_evaluate(
        pc, use_mov=use_mov,
        k=params_dict.get("k", 8.23),
        home_adv=params_dict.get("home_adv", 34.0),
        player_boost=params_dict.get("player_boost", 35.0),
        rest_factor=params_dict.get("rest_factor", 25.0),
        form_weight=params_dict.get("form_weight", 10.0),
        travel_factor=params_dict.get("travel_factor", 28.2),
        sos_factor=params_dict.get("sos_factor", 0.0),
        playoff_hca_factor=params_dict.get("playoff_hca_factor", 0.57),
        pace_factor=params_dict.get("pace_factor", 35.0),
        division_factor=params_dict.get("division_factor", 10.0),
        mean_reversion=params_dict.get("mean_reversion", 0.0),
        b2b_penalty=params_dict.get("b2b_penalty", 0.0),
        road_trip_factor=params_dict.get("road_trip_factor", 3.47),
        homestand_factor=params_dict.get("homestand_factor", 0.0),
        win_streak_factor=params_dict.get("win_streak_factor", 0.0),
        altitude_factor=params_dict.get("altitude_factor", 0.0),
        season_phase_factor=params_dict.get("season_phase_factor", 0.0),
        scoring_consistency_factor=params_dict.get("scoring_consistency_factor", 0.0),
        rest_advantage_cap=params_dict.get("rest_advantage_cap", 2.97),
    )


def _apply_best_settings(best_params, csv_file):
    """Save best params to settings, rebuild model, refit Platt."""
    settings = load_elo_settings()
    for key in _OPT_KEYS:
        if key in best_params:
            settings[key] = best_params[key]
    save_elo_settings(settings)

    # Refit Platt scaler with the winning params
    model = NBAElo(**{k: v for k, v in settings.items() if k in _ELO_KEYS})
    model._altitude_bonus = _calc_altitude_bonus(csv_file)
    player_df = load_player_stats()
    if not player_df.empty:
        model.set_player_stats(player_df)
    _, metrics = backtest_model(csv_file, model=model, fit_platt=True)

    print("\n  %s" % cok("Best settings saved to nba_elo_settings.json"))
    core = ("  K=%.1f  HomeAdv=%.1f  PBoost=%.1f  Rest=%.1f  Travel=%.1f  SOS=%.1f  PlayoffHCA=%.1f  Pace=%.1f"
            % (settings["k"], settings["home_adv"],
               settings.get("player_boost", 0), settings.get("rest_factor", 0),
               settings.get("travel_factor", 0), settings.get("sos_factor", 0),
               settings.get("playoff_hca_factor", 1.0), settings.get("pace_factor", 0)))
    print(core)
    # Print non-zero secondary params
    secondary_parts = []
    for key, label in [("form_weight", "Form"), ("division_factor", "Div"),
                       ("mean_reversion", "MeanRev"), ("b2b_penalty", "B2B"),
                       ("road_trip_factor", "RoadTrip"), ("homestand_factor", "Homestand"),
                       ("win_streak_factor", "WinStrk"), ("altitude_factor", "Alt"),
                       ("season_phase_factor", "SeasonPh"),
                       ("scoring_consistency_factor", "ScorCon"),
                       ("rest_advantage_cap", "RestCap")]:
        val = settings.get(key, 0)
        if val and abs(val) > 0.001:
            secondary_parts.append("%s=%.1f" % (label, val))
    if secondary_parts:
        print("  " + "  ".join(secondary_parts))
    return settings


def grid_search_optimization(csv_file=GAMES_FILE, output_file="nba_grid_search.csv"):
    logging.info("Running GRID SEARCH...")
    settings = load_elo_settings()
    base     = settings.get("base_rating", 1500.0)
    use_mov  = settings.get("use_mov", True)

    print("\nGRID SEARCH RANGES  (press Enter to use defaults)")
    print(cdim("  Objective: minimize Brier + LogLoss  (calibration-focused)"))
    div(48)

    def _prompt_range(label, default_lo, default_hi, default_step):
        try:
            lo   = input("  %s min   [%s]: " % (label, default_lo)).strip()
            hi   = input("  %s max   [%s]: " % (label, default_hi)).strip()
            step = input("  %s step  [%s]: " % (label, default_step)).strip()
            lo   = float(lo)   if lo   else default_lo
            hi   = float(hi)   if hi   else default_hi
            step = float(step) if step else default_step
        except ValueError:
            print("  Invalid input - using defaults for %s" % label)
            lo, hi, step = default_lo, default_hi, default_step
        return np.arange(lo, hi + step * 0.001, step)

    k_values            = _prompt_range("K",           10, 50,  4)
    print()
    home_adv_values     = _prompt_range("HomeAdv",     20, 80,  5)
    print()
    player_boost_values = _prompt_range("PlayerBoost",  0, 30,  5)
    print()
    rest_factor_values  = _prompt_range("RestFactor",   0, 30,  5)
    print()
    travel_values       = _prompt_range("TravelFactor", 0, 50, 10)
    print()
    pace_values         = _prompt_range("PaceFactor",   0, 50, 10)
    print()
    playoff_hca_values  = _prompt_range("PlayoffHCA",   0.4, 1.0, 0.2)
    print()

    player_df   = load_player_stats()
    has_players = not player_df.empty
    _prebuilt_scores = build_league_player_scores(player_df) if has_players else {}
    _alt_bonus = _calc_altitude_bonus(csv_file)

    # Pre-compute game arrays once for fast evaluation
    _pc = precompute_games(csv_file, _prebuilt_scores if has_players else None,
                           _alt_bonus, base)

    results    = []
    best_score = -1e9
    best_params = None
    total = (len(k_values) * len(home_adv_values) * len(player_boost_values)
             * len(rest_factor_values) * len(travel_values) * len(pace_values)
             * len(playoff_hca_values))

    div(120)
    print("  GRID SEARCH - %d combos   |   Objective: minimize Brier + LogLoss" % total)
    div(120)

    pbar = tqdm(
        enumerate(product(k_values, home_adv_values, player_boost_values,
                          rest_factor_values, travel_values, pace_values,
                          playoff_hca_values), 1),
        total=total, desc="  Grid search", leave=True)
    for i, (k, home_adv, player_boost, rest_factor, travel, pace, phca) in pbar:
        metrics = fast_evaluate(
            _pc, k=float(k), home_adv=float(home_adv), use_mov=use_mov,
            player_boost=float(player_boost), rest_factor=float(rest_factor),
            travel_factor=float(travel), pace_factor=float(pace),
            playoff_hca_factor=float(phca),
        )
        if metrics is None:
            continue
        # ACCURACY-FIRST objective (was: LogLoss*8 + Brier*40)
        score = -((100.0 - metrics["accuracy"]) + metrics["brier"] * 5.0)
        row   = {
            "k": float(k), "home_adv": float(home_adv),
            "player_boost": float(player_boost), "rest_factor": float(rest_factor),
            "travel_factor": float(travel), "pace_factor": float(pace),
            "playoff_hca_factor": float(phca),
            "accuracy": metrics["accuracy"], "log_loss": metrics["log_loss"],
            "brier": metrics["brier"], "score": score,
        }
        results.append(row)
        is_best = score > best_score
        if is_best:
            best_score  = score
            best_params = row
            tqdm.write("  %5d/%d  K=%.0f HA=%.0f PB=%.0f R=%.0f T=%.0f P=%.0f PHCA=%.1f  "
                       "Acc=%.2f%% LL=%.4f Br=%.4f %s"
                       % (i, total, k, home_adv, player_boost, rest_factor,
                          travel, pace, phca,
                          metrics["accuracy"], metrics["log_loss"], metrics["brier"],
                          cok("<- BEST")))

    div(120)
    if results:
        pd.DataFrame(results).to_csv(output_file, index=False)
    if best_params:
        acc_s = cok("%.2f%%" % best_params["accuracy"])
        print("\n%s K=%.1f HA=%.1f PB=%.1f R=%.1f T=%.1f P=%.1f PHCA=%.1f"
              % (cok("* GRID BEST:"), best_params["k"], best_params["home_adv"],
                 best_params["player_boost"], best_params["rest_factor"],
                 best_params["travel_factor"], best_params["pace_factor"],
                 best_params["playoff_hca_factor"]))
        print("  Acc=%s  LogLoss=%.4f  Brier=%.4f"
              % (acc_s, best_params["log_loss"], best_params["brier"]))
        _apply_best_settings(best_params, csv_file)
    return best_params


def genetic_optimization(csv_file=GAMES_FILE, output_file="nba_genetic_results.csv"):
    logging.info("Running GENETIC ALGORITHM...")
    settings    = load_elo_settings()
    base        = settings["base_rating"]
    use_mov     = settings.get("use_mov", True)
    player_df   = load_player_stats()
    has_players = not player_df.empty
    _prebuilt_scores = build_league_player_scores(player_df) if has_players else {}

    print("\nGENETIC OPTIMIZER SETTINGS  (press Enter to use defaults)")
    print(cdim("  Objective: minimize Brier + LogLoss  (calibration-focused)"))
    div(52)

    def _prompt_float(label, default):
        try:
            raw = input("  %s [%s]: " % (label, default)).strip()
            return float(raw) if raw else default
        except ValueError:
            return default

    def _prompt_bound(label, lo_default, hi_default):
        try:
            lo_raw = input("  %s min [%s]: " % (label, lo_default)).strip()
            hi_raw = input("  %s max [%s]: " % (label, hi_default)).strip()
            lo = float(lo_raw) if lo_raw else lo_default
            hi = float(hi_raw) if hi_raw else hi_default
        except ValueError:
            lo, hi = lo_default, hi_default
        return (lo, hi)

    k_bounds    = _prompt_bound("K",           5,  80)
    ha_bounds   = _prompt_bound("HomeAdv",     0, 150)
    pb_bounds   = _prompt_bound("PlayerBoost", 0,  50)
    rf_bounds   = _prompt_bound("RestFactor",  0,  50)
    tf_bounds   = _prompt_bound("TravelFactor",0,  60)
    pf_bounds   = _prompt_bound("PaceFactor",  0,  60)
    phca_bounds = _prompt_bound("PlayoffHCA",  0.0, 1.0)
    maxiter     = int(_prompt_float("Max generations (maxiter)", 80))
    popsize     = int(_prompt_float("Population size (popsize)", 40))
    print()

    bounds        = [k_bounds, ha_bounds, pb_bounds, rf_bounds,
                     tf_bounds, pf_bounds, phca_bounds]
    _gen_counter  = [0]
    _best_score   = [1e9]
    _best_params  = [None]
    _eval_counter = [0]
    _alt_bonus    = _calc_altitude_bonus(csv_file)

    # Pre-compute game arrays once for fast evaluation
    _pc = precompute_games(csv_file, _prebuilt_scores if has_players else None,
                           _alt_bonus, base)

    div(120)
    print("  GENETIC OPTIMIZER - maxiter=%d  popsize=%d  (7 params)" % (maxiter, popsize))
    print("  K:%s  HA:%s  PB:%s  Rest:%s  Travel:%s  Pace:%s  PHCA:%s"
          % (k_bounds, ha_bounds, pb_bounds, rf_bounds,
             tf_bounds, pf_bounds, phca_bounds))
    div(120)
    print("  %4s  %6s  %10s  %5s %5s %4s %4s %4s %4s %4s"
          % (chi("Gen"), chi("Evals"), chi("BestScore"),
             chi("K"), chi("HA"), chi("PB"), chi("R"),
             chi("T"), chi("P"), chi("PHCA")))
    div(90)

    def _progress_callback(xk, convergence):
        _gen_counter[0] += 1
        if _best_params[0] is not None:
            p = _best_params[0]
            best_s = cok("%10.4f" % (-_best_score[0]))
            print("  %4d  %6d  %s  %5.1f %5.1f %4.1f %4.1f %4.1f %4.1f %4.1f"
                  % (_gen_counter[0], _eval_counter[0], best_s,
                     p[0], p[1], p[2], p[3], p[4], p[5], p[6]),
                  flush=True)

    def objective(params):
        k, home_adv, player_boost, rest_factor, travel, pace, phca = params
        metrics = fast_evaluate(
            _pc, k=float(k), home_adv=float(home_adv), use_mov=use_mov,
            player_boost=float(player_boost), rest_factor=float(rest_factor),
            travel_factor=float(travel), pace_factor=float(pace),
            playoff_hca_factor=float(phca),
        )
        if metrics is None:
            return 1e9
        # ACCURACY-FIRST objective (was: LogLoss*8 + Brier*40)
        score = (100.0 - metrics["accuracy"]) + metrics["brier"] * 5.0
        _eval_counter[0] += 1
        if score < _best_score[0]:
            _best_score[0]  = score
            _best_params[0] = tuple(float(x) for x in params)
        return score

    result = differential_evolution(
        objective, bounds=bounds, maxiter=maxiter, popsize=popsize,
        workers=1, polish=True, callback=_progress_callback,
    )
    best_k, best_home, best_pb, best_rf, best_tf, best_pf, best_phca = result.x
    div(90)
    div(120)
    print("%s  K=%.1f HA=%.1f PB=%.1f R=%.1f T=%.1f P=%.1f PHCA=%.2f"
          % (cok("* GENETIC BEST:"), best_k, best_home, best_pb, best_rf,
             best_tf, best_pf, best_phca))
    print("  Convergence: %s  |  Total evaluations: %d" % (result.message, _eval_counter[0]))
    div(120)

    # Final backtest with best params
    fresh = NBAElo(
        base_rating=base, k=float(best_k), home_adv=float(best_home),
        use_mov=use_mov, player_boost=float(best_pb),
        rest_factor=float(best_rf), travel_factor=float(best_tf),
        pace_factor=float(best_pf), playoff_hca_factor=float(best_phca),
    )
    fresh._altitude_bonus = _alt_bonus
    if has_players:
        fresh._player_scores = _prebuilt_scores
    success, metrics = backtest_model(
        csv_file, "temp_genetic.csv", "temp_genetic_cal.csv", model=fresh,
    )
    row = {
        "k": float(best_k), "home_adv": float(best_home),
        "player_boost": float(best_pb), "rest_factor": float(best_rf),
        "travel_factor": float(best_tf), "pace_factor": float(best_pf),
        "playoff_hca_factor": float(best_phca),
        "accuracy": metrics.get("accuracy", np.nan),
        "log_loss": metrics.get("log_loss",  np.nan),
        "brier":    metrics.get("brier",     np.nan),
        # ACCURACY-FIRST objective (was: LogLoss*8 + Brier*40)
        "score":    -((100.0 - metrics.get("accuracy",0)) + metrics.get("brier",0)*5.0),
    }
    pd.DataFrame([row]).to_csv(output_file, index=False)
    for tmp in ["temp_genetic.csv", "temp_genetic_cal.csv"]:
        try: os.remove(tmp)
        except OSError: pass
    acc_s = cok("%.2f%%" % row["accuracy"])
    print("  FINAL METRICS:  Acc=%s  LogLoss=%.4f  Brier=%.4f  Score=%.4f"
          % (acc_s, row["log_loss"], row["brier"], row["score"]))

    # Auto-save best settings and refit Platt
    _apply_best_settings(row, csv_file)
    return row


def show_optimization_results():
    result_files = [
        "nba_grid_search.csv", "nba_genetic_results.csv", "nba_bayesian_results.csv",
        "nba_super_grid.csv", "nba_super_fine_grid.csv",
    ]
    found_any = False
    for f in result_files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            if not df.empty and "score" in df.columns:
                found_any = True
                best  = df.loc[df["score"].idxmax()]
                extras = ""
                for key, label in [("player_boost","PB"), ("rest_factor","R"),
                                   ("travel_factor","T"), ("pace_factor","P"),
                                   ("playoff_hca_factor","PHCA"),
                                   ("sos_factor","SOS"), ("form_weight","FW")]:
                    if key in best and pd.notna(best[key]):
                        extras += ", %s=%.1f" % (label, best[key])
                acc_s = cok("%.2f%%" % best["accuracy"]) if "accuracy" in best else "?"
                ll_s = "%.4f" % best["log_loss"] if "log_loss" in best else "?"
                br_s = "%.4f" % best["brier"] if "brier" in best else "?"
                print("  %s: BEST K=%.1f, HA=%.1f%s, Acc=%s, LL=%s, Br=%s"
                      % (chi(f), best.get("k"), best.get("home_adv"), extras,
                         acc_s, ll_s, br_s))
                print("    %d total trials" % len(df))
                # Deflated Sharpe Ratio
                dsr = deflated_sharpe_ratio(df["score"].values)
                if dsr is not None:
                    if dsr >= 1.96:
                        print("    DSR=%.2f %s (significant at 95%%)" % (dsr, cok("*")))
                    else:
                        print("    DSR=%.2f %s (not significant — best may be noise)"
                              % (dsr, cwarn("!")))
    if not found_any:
        print(cwarn("  No optimization results found. Run 'grid', 'genetic', 'bayesian', 'autoopt', or 'superopt' first."))


# ── Deflated Sharpe Ratio ────────────────────────────────────────────

def deflated_sharpe_ratio(scores):
    """Adjust best optimization score for multiple-testing bias.
    DSR < 1.96 -> best result not significant at 95% confidence."""
    if scores is None or len(scores) < 5:
        return None
    scores = np.array(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    if n < 5:
        return None
    mu = np.mean(scores)
    sigma = np.std(scores, ddof=1)
    if sigma < 1e-12:
        return None
    sr = mu / sigma
    z = (scores - mu) / sigma
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4)) - 3.0  # excess kurtosis
    # Expected max SR under null (EVT approximation)
    log_n = np.log(max(n, 2))
    sr_expected = np.sqrt(2 * log_n) - (np.log(np.pi) + np.log(log_n)) / (2 * np.sqrt(2 * log_n))
    # SE of SR (Lo 2002)
    se = np.sqrt((1 - skew * sr + ((kurt + 2) / 4) * sr ** 2) / max(n - 1, 1))
    if se < 1e-12:
        return None
    return float((sr - sr_expected) / se)


# ── Bayesian Hyperparameter Optimization ─────────────────────────────

def bayesian_optimization(csv_file=GAMES_FILE, output_file="nba_bayesian_results.csv"):
    """GP surrogate + Expected Improvement — more sample-efficient than grid search."""
    from scipy.stats.qmc import LatinHypercube

    settings = load_elo_settings()
    base = settings.get("base_rating", 1500.0)
    use_mov = settings.get("use_mov", True)
    player_df = load_player_stats()
    has_players = not player_df.empty
    _prebuilt_scores = build_league_player_scores(player_df) if has_players else {}
    _alt_bonus = _calc_altitude_bonus(csv_file)

    # Pre-compute game arrays once for fast evaluation
    _pc = precompute_games(csv_file, _prebuilt_scores if has_players else None,
                           _alt_bonus, base)

    print("\nBAYESIAN OPTIMIZER SETTINGS  (press Enter to use defaults)")
    print(cdim("  GP surrogate + Expected Improvement acquisition"))
    div(52)

    def _pf(label, default):
        try:
            raw = input("  %s [%s]: " % (label, default)).strip()
            return float(raw) if raw else default
        except ValueError:
            return default

    def _pb(label, lo_d, hi_d):
        try:
            lo = input("  %s min [%s]: " % (label, lo_d)).strip()
            hi = input("  %s max [%s]: " % (label, hi_d)).strip()
            return (float(lo) if lo else lo_d, float(hi) if hi else hi_d)
        except ValueError:
            return (lo_d, hi_d)

    bounds = [_pb("K", 5, 80), _pb("HomeAdv", 0, 150), _pb("PlayerBoost", 0, 50),
              _pb("RestFactor", 0, 50), _pb("TravelFactor", 0, 60),
              _pb("PaceFactor", 0, 60), _pb("PlayoffHCA", 0.0, 1.0)]
    n_initial = int(_pf("Initial samples", 20))
    n_iter = int(_pf("Optimization iterations", 60))
    print()

    lows = np.array([b[0] for b in bounds])
    highs = np.array([b[1] for b in bounds])
    ranges = highs - lows
    ranges[ranges < 1e-12] = 1.0  # prevent div by zero

    def objective(params):
        k, ha, pb, rf, tf, pf, phca = params
        metrics = fast_evaluate(
            _pc, k=float(k), home_adv=float(ha), use_mov=use_mov,
            player_boost=float(pb), rest_factor=float(rf), travel_factor=float(tf),
            pace_factor=float(pf), playoff_hca_factor=float(phca),
        )
        if metrics is None:
            return 1e9
        # ACCURACY-FIRST objective (was: LogLoss*8 + Brier*40)
        return (100.0 - metrics["accuracy"]) + metrics["brier"] * 5.0

    # Minimal GP surrogate
    class _GP:
        def __init__(self, ls=1.0, noise=1e-4):
            self.ls, self.noise = ls, noise
        def _kern(self, A, B):
            d = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2)
            return np.exp(-0.5 * d / self.ls ** 2)
        def fit(self, X, y):
            self.X, self.y = X, y
            K = self._kern(X, X) + self.noise * np.eye(len(X))
            try:
                self.L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K += 1e-3 * np.eye(len(X))
                self.L = np.linalg.cholesky(K)
            self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y))
        def predict(self, Xn):
            k = self._kern(Xn, self.X)
            mu = k @ self.alpha
            v = np.linalg.solve(self.L, k.T)
            var = 1.0 + self.noise - np.sum(v ** 2, axis=0)
            return mu, np.sqrt(np.maximum(var, 1e-12))

    # Latin Hypercube initial sampling
    sampler = LatinHypercube(d=7)
    X_init = lows + sampler.random(n=n_initial) * ranges

    div(120)
    print("  BAYESIAN OPTIMIZER - %d initial + %d iterations (7 params)" % (n_initial, n_iter))
    div(120)

    X_all, y_all = [], []
    best_score, best_params = 1e9, None

    for i, x in tqdm(enumerate(X_init), total=n_initial, desc="  Bayesian init", leave=True):
        s = objective(x)
        X_all.append(x)
        y_all.append(s)
        if s < best_score:
            best_score, best_params = s, x
            tqdm.write("  [%d/%d] Score=%.4f %s" % (i + 1, n_initial, s, cok("<- BEST")))

    gp = _GP()
    for it in tqdm(range(n_iter), desc="  Bayesian opt", leave=True):
        Xa = np.array(X_all)
        ya = np.array(y_all)
        X_norm = (Xa - lows) / ranges
        ym, ys = ya.mean(), max(ya.std(), 1e-8)
        y_norm = (ya - ym) / ys
        gp.fit(X_norm, y_norm)

        cands = np.random.rand(1000, 7)
        mu, sigma = gp.predict(cands)
        best_n = (best_score - ym) / ys
        z = (best_n - mu) / sigma
        ei = sigma * (z * norm_dist.cdf(z) + norm_dist.pdf(z))
        ei[sigma < 1e-10] = 0.0

        x_next = np.clip(lows + cands[np.argmax(ei)] * ranges, lows, highs)
        s = objective(x_next)
        X_all.append(x_next)
        y_all.append(s)
        is_best = s < best_score
        if is_best:
            best_score, best_params = s, x_next
            tqdm.write("  Iter %3d/%d  Score=%.4f  Best=%.4f %s"
                       % (it + 1, n_iter, s, best_score, cok("<- BEST")))

    for tmp in ["temp_bayes.csv", "temp_bayes_cal.csv"]:
        try: os.remove(tmp)
        except OSError: pass

    # Save results
    rows = []
    for x, s in zip(X_all, y_all):
        if s < 1e8:
            rows.append({"k": float(x[0]), "home_adv": float(x[1]),
                         "player_boost": float(x[2]), "rest_factor": float(x[3]),
                         "travel_factor": float(x[4]), "pace_factor": float(x[5]),
                         "playoff_hca_factor": float(x[6]), "score": -s})
    if rows:
        pd.DataFrame(rows).to_csv(output_file, index=False)

    div(120)
    if best_params is not None:
        bp = best_params
        m = NBAElo(base_rating=base, k=float(bp[0]), home_adv=float(bp[1]),
                   use_mov=use_mov, player_boost=float(bp[2]),
                   rest_factor=float(bp[3]), travel_factor=float(bp[4]),
                   pace_factor=float(bp[5]), playoff_hca_factor=float(bp[6]))
        m._altitude_bonus = _alt_bonus
        if has_players:
            m._player_scores = _prebuilt_scores
        _, met = backtest_model(csv_file, model=m)
        print("\n%s K=%.1f HA=%.1f PB=%.1f R=%.1f T=%.1f P=%.1f PHCA=%.2f"
              % (cok("* BAYESIAN BEST:"), bp[0], bp[1], bp[2], bp[3], bp[4], bp[5], bp[6]))
        acc_s = cok("%.2f%%" % met.get("accuracy", 0))
        print("  Acc=%s  LogLoss=%.4f  Brier=%.4f  (%d evaluations)"
              % (acc_s, met.get("log_loss", 0), met.get("brier", 0), len(y_all)))
        _apply_best_settings({"k": float(bp[0]), "home_adv": float(bp[1]),
                              "player_boost": float(bp[2]), "rest_factor": float(bp[3]),
                              "travel_factor": float(bp[4]), "pace_factor": float(bp[5]),
                              "playoff_hca_factor": float(bp[6])}, csv_file)
    return best_params


# ── Purged Walk-Forward Cross-Validation ─────────────────────────────

def purged_walk_forward_cv(csv_file=GAMES_FILE, k_folds=5, embargo_games=5):
    """k-fold CV with embargo gap to prevent Elo momentum leakage between folds."""
    if not os.path.exists(csv_file):
        return None
    settings = load_elo_settings()
    games = pd.read_csv(csv_file)
    n = len(games)
    fold_size = n // k_folds

    print("\nPURGED WALK-FORWARD CROSS-VALIDATION")
    print(cdim("  %d folds, ~%d games/fold, %d embargo games" % (k_folds, fold_size, embargo_games)))
    div(80)

    player_df = load_player_stats()
    _alt_bonus = _calc_altitude_bonus(csv_file)
    fold_metrics = []

    for fold in range(k_folds):
        test_start = fold * fold_size
        test_end = min((fold + 1) * fold_size, n)
        train_end = max(0, test_start - embargo_games)
        if train_end < 100:
            print("  Fold %d: skipping (insufficient training data)" % (fold + 1))
            continue

        train_games = games.iloc[:train_end]
        train_file = "temp_purged_train.csv"
        train_games.to_csv(train_file, index=False)

        model = NBAElo(**{k: v for k, v in settings.items() if k in _ELO_KEYS})
        model._altitude_bonus = _alt_bonus
        if not player_df.empty:
            model.set_player_stats(player_df)

        # Train: replay training games to build ratings
        backtest_model(train_file, "temp_purged_p.csv", "temp_purged_c.csv", model=model)

        # Predict test fold
        test_df = games.iloc[test_start:test_end].copy()
        if "neutral_site" not in test_df.columns:
            test_df["neutral_site"] = False
        test_df["_dp"] = pd.to_datetime(test_df["date"], errors="coerce")
        probs, actuals = [], []
        for _, row in test_df.iterrows():
            try:
                gd = row["_dp"] if pd.notna(row["_dp"]) else None
                p = model.win_prob(row["home_team"], row["away_team"],
                                   team_a_home=True, neutral_site=bool(row.get("neutral_site", False)),
                                   calibrated=False, game_date=gd, use_injuries=False)
                a = 1 if row["home_score"] > row["away_score"] else 0
                probs.append(p)
                actuals.append(a)
                model.update_game(row["home_team"], row["away_team"],
                                  row["home_score"], row["away_score"],
                                  neutral_site=bool(row.get("neutral_site", False)), game_date=gd)
            except Exception as e:
                logging.debug("Purged CV prediction error: %s", e)

        if probs:
            acc = sum(1 for p, a in zip(probs, actuals) if (p >= 0.5) == (a == 1)) / len(probs) * 100
            ll = log_loss_binary(actuals, probs)
            br = brier_score_binary(actuals, probs)
            fold_metrics.append({"fold": fold + 1, "accuracy": acc, "log_loss": ll, "brier": br, "n": len(probs)})
            print("  Fold %d: Acc=%.2f%%  LL=%.4f  Brier=%.4f  (%d games)"
                  % (fold + 1, acc, ll, br, len(probs)))

    for tmp in ["temp_purged_train.csv", "temp_purged_p.csv", "temp_purged_c.csv"]:
        try: os.remove(tmp)
        except OSError: pass

    if fold_metrics:
        accs = [m["accuracy"] for m in fold_metrics]
        lls = [m["log_loss"] for m in fold_metrics]
        brs = [m["brier"] for m in fold_metrics]
        div(80)
        print("  MEAN:  Acc=%.2f%% (+/-%.2f)  LL=%.4f (+/-%.4f)  Brier=%.4f (+/-%.4f)"
              % (np.mean(accs), np.std(accs), np.mean(lls), np.std(lls), np.mean(brs), np.std(brs)))
        if np.std(accs) > 3.0:
            print("  %s Accuracy varies >3%% across folds - model may be fragile" % cwarn("WARNING:"))
        else:
            print("  %s Accuracy variance within normal range" % cok("OK:"))
    return fold_metrics


# ── Combinatorial Purged Cross-Validation (CPCV) ────────────────────

def combinatorial_purged_cv(csv_file=GAMES_FILE, k_blocks=5, k_test=2):
    """All C(k, k_test) train/test combinations — tighter confidence intervals."""
    if not os.path.exists(csv_file):
        return None
    settings = load_elo_settings()
    games = pd.read_csv(csv_file)
    n = len(games)
    block_size = n // k_blocks
    n_combos = math_comb(k_blocks, k_test)

    print("\nCOMBINATORIAL PURGED CROSS-VALIDATION")
    print(cdim("  %d blocks, %d test, %d combinations" % (k_blocks, k_test, n_combos)))
    div(80)

    player_df = load_player_stats()
    _alt_bonus = _calc_altitude_bonus(csv_file)
    all_accs = []

    for ci, test_blocks in enumerate(combinations(range(k_blocks), k_test), 1):
        train_blocks = [b for b in range(k_blocks) if b not in test_blocks]
        train_idx = []
        for b in train_blocks:
            train_idx.extend(range(b * block_size, min((b + 1) * block_size, n)))
        test_idx = []
        for b in test_blocks:
            test_idx.extend(range(b * block_size, min((b + 1) * block_size, n)))
        if len(train_idx) < 100 or len(test_idx) < 20:
            continue
        train_idx.sort()

        train_file = "temp_cpcv.csv"
        games.iloc[train_idx].to_csv(train_file, index=False)
        model = NBAElo(**{k: v for k, v in settings.items() if k in _ELO_KEYS})
        model._altitude_bonus = _alt_bonus
        if not player_df.empty:
            model.set_player_stats(player_df)
        backtest_model(train_file, "temp_cpcv_p.csv", "temp_cpcv_c.csv", model=model)

        test_df = games.iloc[test_idx].copy()
        if "neutral_site" not in test_df.columns:
            test_df["neutral_site"] = False
        test_df["_dp"] = pd.to_datetime(test_df["date"], errors="coerce")
        correct, total = 0, 0
        for _, row in test_df.iterrows():
            try:
                gd = row["_dp"] if pd.notna(row["_dp"]) else None
                p = model.win_prob(row["home_team"], row["away_team"],
                                   team_a_home=True, neutral_site=bool(row.get("neutral_site", False)),
                                   calibrated=False, game_date=gd, use_injuries=False)
                a = 1 if row["home_score"] > row["away_score"] else 0
                if (p >= 0.5) == (a == 1):
                    correct += 1
                total += 1
            except Exception as e:
                logging.debug("CPCV prediction error: %s", e)
        if total > 0:
            acc = correct / total * 100
            all_accs.append(acc)
            if ci <= 10 or ci % 5 == 0 or ci == n_combos:
                print("  Combo %d/%d  test=%s  Acc=%.2f%%  (%d games)"
                      % (ci, n_combos, test_blocks, acc, total), flush=True)

    for tmp in ["temp_cpcv.csv", "temp_cpcv_p.csv", "temp_cpcv_c.csv"]:
        try: os.remove(tmp)
        except OSError: pass

    if all_accs:
        div(80)
        print("  CPCV RESULTS (%d paths):" % len(all_accs))
        print("  Mean=%.2f%%  Std=%.2f%%  Min=%.2f%%  Max=%.2f%%"
              % (np.mean(all_accs), np.std(all_accs), np.min(all_accs), np.max(all_accs)))
        above65 = sum(1 for a in all_accs if a > 65) / len(all_accs) * 100
        below55 = sum(1 for a in all_accs if a < 55) / len(all_accs) * 100
        print("  Paths >65%%: %.0f%%  |  Paths <55%%: %.0f%%" % (above65, below55))
        if below55 > 10:
            print("  %s Some paths <55%% - path-dependent overfitting risk" % cwarn("WARNING:"))
        else:
            print("  %s Model robust across combinatorial paths" % cok("OK:"))
    return all_accs


# ── Probability of Backtest Overfitting (PBO) ────────────────────────

def probability_of_backtest_overfitting(grid_file="nba_grid_search.csv"):
    """Check if grid search 'best' is likely overfit via symmetric cross-validation."""
    if not os.path.exists(grid_file):
        print(cwarn("  No grid search results found. Run 'grid' first."))
        return None
    df = pd.read_csv(grid_file)
    if "score" not in df.columns or len(df) < 10:
        print(cwarn("  Insufficient grid search data for PBO analysis."))
        return None

    scores = df["score"].values
    n = len(scores)
    half = n // 2

    print("\nPROBABILITY OF BACKTEST OVERFITTING")
    print(cdim("  Symmetric CV on %d grid search trials" % n))
    div(80)

    n_splits = min(1000, n * 10)
    overfit_count = 0
    logit_lambdas = []

    for _ in range(n_splits):
        perm = np.random.permutation(n)
        s1_idx, s2_idx = perm[:half], perm[half:2 * half]
        s1_scores = scores[s1_idx]
        # Best in s1 (by index into s1)
        best_s1_local = np.argmax(s1_scores)
        best_s1_global = s1_idx[best_s1_local]
        # Rank this trial's score among s2
        s2_sorted = np.sort(scores[s2_idx])
        rank = np.searchsorted(s2_sorted, scores[best_s1_global])
        pct = rank / len(s2_idx) if len(s2_idx) > 0 else 0.5
        # logit(pct) for lambda distribution
        pct_c = np.clip(pct, 0.01, 0.99)
        logit_lambdas.append(np.log(pct_c / (1 - pct_c)))
        if pct < 0.5:
            overfit_count += 1

    pbo = overfit_count / n_splits
    print("  PBO = %.3f  (%d/%d splits where IS-best ranks below OOS median)"
          % (pbo, overfit_count, n_splits))
    if pbo > 0.5:
        print("  %s PBO > 0.5 - optimization is likely overfit" % cwarn("WARNING:"))
    else:
        print("  %s PBO <= 0.5 - optimization results appear genuine" % cok("OK:"))
    return pbo


# ── Monte Carlo Permutation Testing ──────────────────────────────────

def monte_carlo_permutation_test(csv_file=GAMES_FILE, n_permutations=500):
    """Null distribution test: is the model's edge statistically significant?"""
    if not os.path.exists(csv_file):
        return None
    settings = load_elo_settings()
    player_df = load_player_stats()
    _alt_bonus = _calc_altitude_bonus(csv_file)

    # Real model performance
    model = NBAElo(**{k: v for k, v in settings.items() if k in _ELO_KEYS})
    model._altitude_bonus = _alt_bonus
    if not player_df.empty:
        model.set_player_stats(player_df)
    _, real_met = backtest_model(csv_file, "temp_mc_real.csv", "temp_mc_real_cal.csv", model=model)
    real_acc = real_met["accuracy"]
    real_brier = real_met["brier"]

    try:
        raw = input("\n  Permutations [%d]: " % n_permutations).strip()
        if raw:
            n_permutations = int(raw)
    except ValueError:
        pass

    print("\nMONTE CARLO PERMUTATION TEST")
    print(cdim("  Real: Acc=%.2f%% Brier=%.4f  |  %d permutations" % (real_acc, real_brier, n_permutations)))
    div(80)

    games = pd.read_csv(csv_file)
    perm_accs, perm_briers = [], []

    for i in tqdm(range(n_permutations), desc="  Permutations", leave=True):
        shuffled = games.copy()
        swap = np.random.random(len(shuffled)) < 0.5
        hs = shuffled["home_score"].values.copy()
        aws = shuffled["away_score"].values.copy()
        shuffled.loc[swap, "home_score"] = aws[swap]
        shuffled.loc[swap, "away_score"] = hs[swap]
        shuffled.to_csv("temp_mc_shuf.csv", index=False)

        pm = NBAElo(**{k: v for k, v in settings.items() if k in _ELO_KEYS})
        pm._altitude_bonus = _alt_bonus
        if not player_df.empty:
            pm.set_player_stats(player_df)
        _, met = backtest_model("temp_mc_shuf.csv", "temp_mc_p.csv", "temp_mc_c.csv", model=pm)
        if met:
            perm_accs.append(met["accuracy"])
            perm_briers.append(met["brier"])

    for tmp in ["temp_mc_real.csv", "temp_mc_real_cal.csv", "temp_mc_shuf.csv", "temp_mc_p.csv", "temp_mc_c.csv"]:
        try: os.remove(tmp)
        except OSError: pass

    if perm_accs and perm_briers:
        p_acc = sum(1 for a in perm_accs if a >= real_acc) / len(perm_accs)
        p_brier = sum(1 for b in perm_briers if b <= real_brier) / len(perm_briers)
        div(80)
        print("  Real Acc: %.2f%%  |  Null: %.2f%% (std %.2f%%)"
              % (real_acc, np.mean(perm_accs), np.std(perm_accs)))
        print("  Real Brier: %.4f  |  Null: %.4f (std %.4f)"
              % (real_brier, np.mean(perm_briers), np.std(perm_briers)))
        print("  p-value (acc): %.4f  |  p-value (Brier): %.4f" % (p_acc, p_brier))
        if p_acc < 0.05:
            print("  %s Model is statistically significant (p < 0.05)" % cok("SIGNIFICANT:"))
        else:
            print("  %s Model NOT significant (p >= 0.05)" % cwarn("NOT SIGNIFICANT:"))
        return {"p_acc": p_acc, "p_brier": p_brier}
    else:
        p_acc = 1.0
        p_brier = 1.0
        return {"p_acc": p_acc, "p_brier": p_brier}


# ── Rolling Origin Recalibration ─────────────────────────────────────

def rolling_origin_recalibration(csv_file=GAMES_FILE, chunk_size=50):
    """Expanding-window Platt refitting for truly out-of-sample calibrated metrics."""
    if not os.path.exists(csv_file):
        return None
    settings = load_elo_settings()
    model = NBAElo(**{k: v for k, v in settings.items() if k in _ELO_KEYS})
    model._altitude_bonus = _calc_altitude_bonus(csv_file)
    player_df = load_player_stats()
    if not player_df.empty:
        model.set_player_stats(player_df)

    games = pd.read_csv(csv_file)
    if "neutral_site" not in games.columns:
        games["neutral_site"] = False
    games["_dp"] = pd.to_datetime(games["date"], errors="coerce")

    print("\nROLLING ORIGIN RECALIBRATION")
    print(cdim("  Expanding-window Platt every %d games" % chunk_size))
    div(80)

    raw_probs, actuals, cal_probs = [], [], []
    scaler = None

    for _, row in games.iterrows():
        try:
            gd = row["_dp"] if pd.notna(row["_dp"]) else None
            p = model.win_prob(row["home_team"], row["away_team"],
                               team_a_home=True, neutral_site=bool(row["neutral_site"]),
                               calibrated=False, game_date=gd, use_injuries=False)
            a = 1 if row["home_score"] > row["away_score"] else 0
            raw_probs.append(p)
            actuals.append(a)
            cal_probs.append(apply_platt(p, scaler) if scaler else p)

            if len(raw_probs) >= 200 and len(raw_probs) % chunk_size == 0:
                scaler = fit_platt_scaler(raw_probs, actuals)
                cs = max(0, len(raw_probs) - chunk_size)
                cb = brier_score_binary(actuals[cs:], cal_probs[cs:])
                print("  Games 1-%d: refit Platt  |  chunk Brier=%.4f" % (len(raw_probs), cb))

            model.update_game(row["home_team"], row["away_team"],
                              row["home_score"], row["away_score"],
                              neutral_site=bool(row["neutral_site"]), game_date=gd)
        except Exception:
            pass

    if cal_probs:
        acc = sum(1 for p, a in zip(cal_probs, actuals) if (p >= 0.5) == (a == 1)) / len(actuals) * 100
        ll = log_loss_binary(actuals, cal_probs)
        br = brier_score_binary(actuals, cal_probs)
        ece = ece_score(cal_probs, actuals)
        # Compare with single-pass
        ss = fit_platt_scaler(raw_probs, actuals)
        sc = [apply_platt(p, ss) for p in raw_probs]
        br_single = brier_score_binary(actuals, sc)
        div(80)
        print("  ROLLING:     Acc=%.2f%%  LL=%.4f  Brier=%.4f  ECE=%.4f" % (acc, ll, br, ece))
        print("  SINGLE-PASS: Brier=%.4f (for comparison)" % br_single)
        if br < br_single:
            print("  %s Rolling is better by %.4f Brier" % (cok("BETTER:"), br_single - br))
        else:
            print("  %s Single-pass is better by %.4f Brier" % (cwarn("NOTE:"), br - br_single))
    return {"rolling_brier": br, "single_brier": br_single} if cal_probs else None


# ── Kelly Criterion Position Sizing Backtest ─────────────────────────

def kelly_criterion_backtest(predictions_csv="nba_backtest_predictions.csv",
                             kelly_fraction=0.25, initial_bankroll=1000.0):
    """Simulate Kelly-optimal position sizing on backtest predictions."""
    if not os.path.exists(predictions_csv):
        print(cwarn("  No backtest predictions found. Run 'backtest' first."))
        return None
    scaler = load_platt_scaler()
    df = pd.read_csv(predictions_csv)

    try:
        raw = input("\n  Kelly fraction (0.25 = quarter-Kelly) [%.2f]: " % kelly_fraction).strip()
        if raw:
            kelly_fraction = float(raw)
        raw = input("  Initial bankroll [%.0f]: " % initial_bankroll).strip()
        if raw:
            initial_bankroll = float(raw)
    except ValueError:
        pass

    print("\nKELLY CRITERION BACKTEST")
    print(cdim("  %.0f%%-Kelly  |  $%.0f starting bankroll" % (kelly_fraction * 100, initial_bankroll)))
    div(80)

    bankroll = initial_bankroll
    peak = bankroll
    max_dd = 0.0
    trades, wins = 0, 0
    history = [bankroll]

    for _, row in df.iterrows():
        raw_prob = row["home_win_prob"]
        prob = apply_platt(raw_prob, scaler) if scaler else raw_prob
        # Our edge: model prob vs fair-odds entry (0.50)
        entry_price = 0.50
        if prob >= 0.5:
            edge_prob = prob
            won = int(row.get("correct", 0)) == 1
        else:
            edge_prob = 1 - prob
            won = int(row.get("correct", 0)) == 1
        # Kelly: f = (p * b - q) / b where b = payout odds, q = 1-p
        payout = 1.0 - entry_price
        kelly_f = max(0, (edge_prob * payout - (1 - edge_prob) * entry_price) / payout) * kelly_fraction
        if kelly_f <= 0 or bankroll <= 0:
            history.append(bankroll)
            continue
        position = kelly_f * bankroll
        contracts = position / entry_price
        if won:
            profit = contracts * payout - contracts * 0.02
            wins += 1
        else:
            profit = -contracts * entry_price
        bankroll += profit
        bankroll = max(bankroll, 0)
        trades += 1
        history.append(bankroll)
        peak = max(peak, bankroll)
        dd = (peak - bankroll) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    if trades == 0:
        print(cwarn("  No trades taken (Kelly fraction too conservative)"))
        return None

    ret = (bankroll - initial_bankroll) / initial_bankroll * 100
    wr = wins / trades * 100
    returns = np.diff(history) / np.maximum(np.array(history[:-1]), 1e-8)
    nz = returns[returns != 0]
    sharpe = float(np.mean(nz) / max(np.std(nz), 1e-8) * np.sqrt(252)) if len(nz) > 1 else 0

    div(80)
    print("  KELLY RESULTS (%d trades, %d wins)" % (trades, wins))
    print("  Final bankroll:    $%.2f (%.1f%% return)" % (bankroll, ret))
    print("  Win rate:          %.1f%%" % wr)
    print("  Max drawdown:      %.1f%%" % (max_dd * 100))
    print("  Annualized Sharpe: %.2f" % sharpe)
    return {"final": bankroll, "return_pct": ret, "max_dd": max_dd, "sharpe": sharpe,
            "trades": trades, "win_rate": wr}


# ── Sliding Window vs Expanding Window Backtest ──────────────────────

def sliding_window_backtest(csv_file=GAMES_FILE, window_size=300):
    """Compare sliding (reset every N games) vs expanding window."""
    if not os.path.exists(csv_file):
        return None
    settings = load_elo_settings()
    _alt_bonus = _calc_altitude_bonus(csv_file)
    player_df = load_player_stats()

    try:
        raw = input("\n  Window size (games before heavy regression) [%d]: " % window_size).strip()
        if raw:
            window_size = int(raw)
    except ValueError:
        pass

    print("\nSLIDING WINDOW BACKTEST")
    print(cdim("  Window: %d games (heavy regression at boundary)" % window_size))
    div(80)

    games = pd.read_csv(csv_file)
    if "neutral_site" not in games.columns:
        games["neutral_site"] = False
    games["_dp"] = pd.to_datetime(games["date"], errors="coerce")

    model = NBAElo(**{k: v for k, v in settings.items() if k in _ELO_KEYS})
    model._altitude_bonus = _alt_bonus
    if not player_df.empty:
        model.set_player_stats(player_df)

    probs, actuals = [], []
    games_since_reset = 0

    for _, row in games.iterrows():
        try:
            gd = row["_dp"] if pd.notna(row["_dp"]) else None
            if games_since_reset >= window_size:
                model.ratings = defaultdict(
                    lambda: model.base_rating,
                    regress_ratings_to_mean(dict(model.ratings), factor=0.67))
                games_since_reset = 0

            p = model.win_prob(row["home_team"], row["away_team"],
                               team_a_home=True, neutral_site=bool(row["neutral_site"]),
                               calibrated=False, game_date=gd, use_injuries=False)
            a = 1 if row["home_score"] > row["away_score"] else 0
            probs.append(p)
            actuals.append(a)
            model.update_game(row["home_team"], row["away_team"],
                              row["home_score"], row["away_score"],
                              neutral_site=bool(row["neutral_site"]), game_date=gd)
            games_since_reset += 1
        except Exception:
            pass

    if not probs:
        return None
    acc_s = sum(1 for p, a in zip(probs, actuals) if (p >= 0.5) == (a == 1)) / len(probs) * 100
    ll_s = log_loss_binary(actuals, probs)
    br_s = brier_score_binary(actuals, probs)

    # Expanding window comparison
    model2 = NBAElo(**{k: v for k, v in settings.items() if k in _ELO_KEYS})
    model2._altitude_bonus = _alt_bonus
    if not player_df.empty:
        model2.set_player_stats(player_df)
    _, em = backtest_model(csv_file, "temp_sw.csv", "temp_sw_c.csv", model=model2)
    for tmp in ["temp_sw.csv", "temp_sw_c.csv"]:
        try: os.remove(tmp)
        except OSError: pass

    div(80)
    print("  SLIDING (reset/%d): Acc=%.2f%%  LL=%.4f  Brier=%.4f" % (window_size, acc_s, ll_s, br_s))
    print("  EXPANDING (std):    Acc=%.2f%%  LL=%.4f  Brier=%.4f"
          % (em["accuracy"], em["log_loss"], em["brier"]))
    if br_s < em["brier"]:
        print("  %s Sliding window is better - old data may hurt" % cok("RESULT:"))
    else:
        print("  %s Expanding window is better - full history helps" % cok("RESULT:"))
    return {"sliding_brier": br_s, "expanding_brier": em["brier"]}


# ── Elo Rating Convergence Analysis ──────────────────────────────────

def elo_convergence_analysis(csv_file=GAMES_FILE, chunk_size=100):
    """Identify burn-in period by chunking predictions into accuracy segments."""
    pred_file = "nba_backtest_predictions.csv"
    if not os.path.exists(pred_file):
        print(cwarn("  Run 'backtest' first to generate predictions."))
        return None

    try:
        raw = input("\n  Chunk size [%d]: " % chunk_size).strip()
        if raw:
            chunk_size = int(raw)
    except ValueError:
        pass

    df = pd.read_csv(pred_file)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    print("\nELO RATING CONVERGENCE ANALYSIS")
    print(cdim("  Chunk size: %d games" % chunk_size))
    div(80)

    chunks = []
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start:start + chunk_size]
        if len(chunk) < 20:
            continue
        acc = float(chunk["correct"].mean() * 100)
        dr = ""
        dates = chunk["date"].dropna()
        if len(dates) > 0:
            dr = " (%s to %s)" % (str(dates.iloc[0])[:10], str(dates.iloc[-1])[:10])
        chunks.append({"start": start, "n": len(chunk), "accuracy": acc, "dr": dr})
        print("  Games %4d-%4d: Acc=%.2f%% (%d games)%s"
              % (start + 1, start + len(chunk), acc, len(chunk), dr))

    if len(chunks) >= 2:
        accs = [c["accuracy"] for c in chunks]
        converged_at = 0
        for i in range(len(chunks)):
            if np.std(accs[i:]) < 2.0 and len(accs[i:]) >= 2:
                converged_at = i
                break
        burn_in = chunks[converged_at]["start"]
        steady = float(np.mean(accs[converged_at:]))
        div(80)
        if converged_at > 0:
            print("  Burn-in: first %d games (avg Acc=%.2f%%)"
                  % (burn_in, np.mean(accs[:converged_at])))
        print("  Steady state after game %d: avg Acc=%.2f%%" % (burn_in, steady))
        print("  Suggested burn-in exclusion: %d games" % burn_in)
    return chunks


# ── Conformal Prediction Analysis ────────────────────────────────────

def conformal_analysis(csv_file=GAMES_FILE):
    """Run conformal prediction set analysis on backtest predictions."""
    pred_file = "nba_backtest_predictions.csv"
    if not os.path.exists(pred_file):
        print(cwarn("  Run 'backtest' first."))
        return None

    df = pd.read_csv(pred_file)
    probs = df["home_win_prob"].values
    actuals = (df["home_score"] > df["away_score"]).astype(int).values

    scaler = load_platt_scaler()
    if scaler:
        probs = np.array([apply_platt(p, scaler) for p in probs])

    print("\nCONFORMAL PREDICTION ANALYSIS")
    div(80)

    for alpha in [0.05, 0.10, 0.20]:
        result = conformal_prediction_set(probs, actuals, alpha=alpha)
        if result:
            cov = result["actual_coverage"] * 100
            tgt = result["target_coverage"] * 100
            ok = cok("*") if abs(cov - tgt) < 3 else cwarn("!")
            print("  alpha=%.2f  Target=%.0f%%  Actual=%.1f%% %s  Avg set=%.2f  "
                  "Singleton=%.0f%%  Empty=%.0f%%  Both=%.0f%%"
                  % (alpha, tgt, cov, ok, result["avg_set_size"],
                     result["singleton_pct"], result["empty_pct"], result["both_pct"]))

    print(cdim("\n  Singleton = model is confident (only one outcome in set)"))
    print(cdim("  Both = model is uncertain (both outcomes in set)"))
    print(cdim("  Empty = model is overconfident (set excludes true outcome)"))
    return True


# ── Beta Calibration Command ─────────────────────────────────────────

def run_beta_calibration(csv_file=GAMES_FILE):
    """Fit beta calibration on backtest predictions and compare with Platt."""
    from platt import fit_beta_scaler, apply_beta, save_beta_scaler
    pred_file = "nba_backtest_predictions.csv"
    if not os.path.exists(pred_file):
        print(cwarn("  Run 'backtest' first."))
        return None

    df = pd.read_csv(pred_file)
    probs = df["home_win_prob"].values
    actuals = (df["home_score"] > df["away_score"]).astype(int).values

    print("\nBETA CALIBRATION")
    div(80)

    beta_sc = fit_beta_scaler(probs, actuals)
    save_beta_scaler(beta_sc)
    beta_cal = [apply_beta(p, beta_sc) for p in probs]
    br_beta = brier_score_binary(actuals, beta_cal)
    ll_beta = log_loss_binary(actuals, beta_cal)
    ece_beta = ece_score(beta_cal, actuals)

    # Compare with Platt
    platt_sc = fit_platt_scaler(probs, actuals)
    platt_cal = [apply_platt(p, platt_sc) for p in probs]
    br_platt = brier_score_binary(actuals, platt_cal)
    ll_platt = log_loss_binary(actuals, platt_cal)
    ece_platt = ece_score(platt_cal, actuals)

    print("  BETA (a=%.3f b=%.3f c=%.3f):  Brier=%.4f  LL=%.4f  ECE=%.4f"
          % (beta_sc["a"], beta_sc["b"], beta_sc["c"], br_beta, ll_beta, ece_beta))
    print("  PLATT (coef=%.3f int=%.3f):     Brier=%.4f  LL=%.4f  ECE=%.4f"
          % (platt_sc["coef"], platt_sc["intercept"], br_platt, ll_platt, ece_platt))

    if abs(beta_sc["a"] - beta_sc["b"]) > 0.3:
        print("  %s a != b — asymmetric miscalibration detected (beta is more appropriate)"
              % chi("NOTE:"))
    else:
        print("  %s a ~ b — calibration is symmetric (Platt is sufficient)" % cdim("NOTE:"))

    if br_beta < br_platt:
        print("  %s Beta calibration wins by %.4f Brier" % (cok("BETTER:"), br_platt - br_beta))
    else:
        print("  %s Platt calibration is equal or better" % cok("OK:"))
    return beta_sc


# ── Auto-Optimize ────────────────────────────────────────────────────

def auto_optimize(csv_file=GAMES_FILE):
    """Run grid -> genetic -> bayesian automatically, compare all, apply the best.

    No interactive prompts. Uses sensible defaults for each phase:
      Phase 1: Coarse grid (core params only) to find the neighborhood
      Phase 2: Genetic with ALL 19 params (tightened core + full secondary)
      Phase 3: Bayesian with ALL 19 params (GP surrogate + EI)
      Phase 4: Compare all three winners, apply the absolute best
    """
    import time
    from scipy.stats.qmc import LatinHypercube

    if not os.path.exists(csv_file):
        print(cwarn("  Game data not found: %s" % csv_file))
        return None

    settings = load_elo_settings()
    base = settings.get("base_rating", 1500.0)
    use_mov = settings.get("use_mov", True)
    player_df = load_player_stats()
    has_players = not player_df.empty
    _prebuilt = build_league_player_scores(player_df) if has_players else {}
    _alt = _calc_altitude_bonus(csv_file)

    # Pre-compute game arrays once for fast evaluation
    _pc = precompute_games(csv_file, _prebuilt if has_players else None, _alt, base)

    eval_count = [0]

    # All 19 prediction-relevant parameters
    # (name, grid_values_or_None, (lo, hi), half_width_for_tightening)
    PARAMS = [
        # --- Core (in grid) ---
        ("k",                      [10, 20, 30, 45],   (3, 80),     10),
        ("home_adv",               [25, 45, 65, 85],   (0, 150),    15),
        ("player_boost",           [0, 10, 25],         (0, 60),     10),
        ("rest_factor",            [0, 15, 30],         (0, 60),     10),
        ("form_weight",            [0, 10],             (0, 40),     10),
        ("travel_factor",          [0, 20],             (0, 60),     15),
        ("sos_factor",             [0, 10],             (0, 40),     10),
        ("playoff_hca_factor",     [0.5, 0.8, 1.0],    (0.0, 1.5),  0.3),
        ("pace_factor",            [0, 20],             (0, 60),     15),
        # --- Secondary (genetic/bayesian only) ---
        ("division_factor",        None,                (0, 40),     10),
        ("mean_reversion",         None,                (0, 20),     5),
        ("b2b_penalty",            None,                (0, 150),    30),
        ("road_trip_factor",       None,                (0, 20),     5),
        ("homestand_factor",       None,                (0, 20),     5),
        ("win_streak_factor",      None,                (0, 20),     5),
        ("altitude_factor",        None,                (0, 30),     8),
        ("season_phase_factor",    None,                (0, 20),     5),
        ("scoring_consistency_factor", None,            (0, 15),     4),
        ("rest_advantage_cap",     None,                (0, 10),     3),
    ]

    PARAM_NAMES = [p[0] for p in PARAMS]
    n_params = len(PARAMS)

    # Indices of params that have grid values (core)
    grid_indices = [i for i, p in enumerate(PARAMS) if p[1] is not None]
    # Default values for secondary params (midpoint of bounds)
    default_vals = [(p[2][0] + p[2][1]) / 2.0 if p[1] is None else 0.0 for p in PARAMS]

    def _eval(params):
        """Shared objective: minimize (100 - accuracy) + Brier*5. Accepts all 19 params."""
        kw = {name: float(params[i]) for i, name in enumerate(PARAM_NAMES)}
        met = fast_evaluate(_pc, use_mov=use_mov, **kw)
        eval_count[0] += 1
        if met is None:
            return 1e9, {}
        # ACCURACY-FIRST objective (was: LogLoss*8 + Brier*40)
        return (100.0 - met["accuracy"]) + met["brier"] * 5.0, met

    def _params_dict(p):
        return {name: float(p[i]) for i, name in enumerate(PARAM_NAMES)}

    def _fmt(p):
        # Always show core params
        parts = ["K=%.1f HA=%.1f PB=%.1f R=%.1f T=%.1f P=%.1f PHCA=%.2f SOS=%.1f FW=%.1f"
                 % (p[0], p[1], p[2], p[3], p[5], p[8], p[7], p[6], p[4])]
        # Show non-zero secondaries
        sec = []
        for i in range(9, n_params):
            if abs(p[i]) > 0.001:
                short = PARAM_NAMES[i].replace("_factor", "").replace("_", "")
                sec.append("%s=%.1f" % (short, p[i]))
        if sec:
            parts.append(" ".join(sec))
        return "  ".join(parts)

    winners = []  # [(score, params_array, source_name, metrics)]

    print("\n" + "=" * 80)
    print(chi("  AUTO-OPTIMIZE: Grid -> Genetic -> Bayesian (fully automatic, %d params)" % n_params))
    print("=" * 80)
    t_start = time.time()

    # ── Phase 1: Coarse Grid Search (core params only) ───────────────
    print(chi("\n  PHASE 1: Coarse Grid Search (%d core params)" % len(grid_indices)))
    div(80)

    grid_value_lists = [PARAMS[i][1] for i in grid_indices]
    total = 1
    for gv in grid_value_lists:
        total *= len(gv)
    print("  %d combinations" % total)

    grid_best_score, grid_best_params = 1e9, None
    grid_best_met = {}
    grid_rows = []
    t1 = time.time()

    for i, combo in tqdm(enumerate(product(*grid_value_lists), 1), total=total,
                         desc="  Autoopt grid", leave=True):
        # Build full param vector: grid values for core, 0 for secondary
        full_p = [0.0] * n_params
        for ci, gi in enumerate(grid_indices):
            full_p[gi] = float(combo[ci])
        score, met = _eval(full_p)
        if score < 1e8:
            row = _params_dict(full_p)
            row.update({"accuracy": met.get("accuracy", 0),
                        "log_loss": met.get("log_loss", 0),
                        "brier": met.get("brier", 0), "score": -score})
            grid_rows.append(row)
        is_best = score < grid_best_score
        if is_best:
            grid_best_score = score
            grid_best_params = np.array(full_p, dtype=float)
            grid_best_met = met
            short = "K=%.0f HA=%.0f PB=%.0f R=%.0f FW=%.0f T=%.0f SOS=%.0f PHCA=%.1f P=%.0f" % combo[:9] if len(combo) >= 9 else " ".join("%.0f" % v for v in combo)
            tqdm.write("  %4d/%d  Acc=%.2f%% LL=%.4f Br=%.4f  %s %s"
                       % (i, total, met.get("accuracy", 0), met.get("log_loss", 0),
                          met.get("brier", 0), short, cok("<- BEST")))

    if grid_rows:
        pd.DataFrame(grid_rows).to_csv("nba_grid_search.csv", index=False)

    t1_elapsed = time.time() - t1
    if grid_best_params is not None:
        winners.append((grid_best_score, grid_best_params, "Grid", grid_best_met))
        print("  Grid done: %.0fs  |  Best score=%.4f  Acc=%.2f%%"
              % (t1_elapsed, -grid_best_score, grid_best_met.get("accuracy", 0)))
    else:
        print(cwarn("  Grid search produced no valid results"))
        return None

    # ── Tighten bounds around grid best ──────────────────────────────
    def _tight(val, lo, hi, hw):
        tlo = max(lo, val - hw)
        thi = min(hi, val + hw)
        if tlo >= thi:
            tlo, thi = lo, hi
        return (tlo, thi)

    bp = grid_best_params
    bounds = []
    for i, (name, gv, (lo, hi), hw) in enumerate(PARAMS):
        bounds.append(_tight(bp[i], lo, hi, hw))
    lows = np.array([b[0] for b in bounds])
    highs = np.array([b[1] for b in bounds])
    ranges = highs - lows
    ranges[ranges < 1e-12] = 1.0

    print(cdim("  Tightened bounds for %d params" % n_params))

    # ── Phase 2: Genetic Optimization (ALL params) ───────────────────
    print(chi("\n  PHASE 2: Genetic Algorithm (tightened bounds, %d params)" % n_params))
    div(80)
    t2 = time.time()

    gen_best_score = [1e9]
    gen_best_params = [None]
    gen_counter = [0]

    def _gen_cb(xk, convergence):
        gen_counter[0] += 1
        if gen_counter[0] % 10 == 0 and gen_best_params[0] is not None:
            print("  Gen %d  best=%.4f" % (gen_counter[0], -gen_best_score[0]), flush=True)

    def _gen_obj(params):
        score, met = _eval(params)
        if score < gen_best_score[0]:
            gen_best_score[0] = score
            gen_best_params[0] = np.array(params)
        return score

    de_result = differential_evolution(
        _gen_obj, bounds=bounds, maxiter=50, popsize=25,
        workers=1, polish=True, callback=_gen_cb)

    gen_p = de_result.x
    gen_score, gen_met = _eval(gen_p)
    winners.append((gen_score, np.array(gen_p), "Genetic", gen_met))
    pd.DataFrame([{**_params_dict(gen_p), "score": -gen_score,
                    "accuracy": gen_met.get("accuracy", 0),
                    "log_loss": gen_met.get("log_loss", 0),
                    "brier": gen_met.get("brier", 0)}]).to_csv(
        "nba_genetic_results.csv", index=False)

    t2_elapsed = time.time() - t2
    print("  Genetic done: %.0fs  |  Best score=%.4f  Acc=%.2f%%"
          % (t2_elapsed, -gen_score, gen_met.get("accuracy", 0)))

    # ── Phase 3: Bayesian Optimization (ALL params) ──────────────────
    print(chi("\n  PHASE 3: Bayesian Optimization (GP surrogate, %d params)" % n_params))
    div(80)
    t3 = time.time()

    class _GP:
        def __init__(self, ls=1.0, noise=1e-4):
            self.ls, self.noise = ls, noise
        def _kern(self, A, B):
            d = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2)
            return np.exp(-0.5 * d / self.ls ** 2)
        def fit(self, X, y):
            self.X, self.y = X, y
            K = self._kern(X, X) + self.noise * np.eye(len(X))
            try:
                self.L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K += 1e-3 * np.eye(len(X))
                self.L = np.linalg.cholesky(K)
            self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y))
        def predict(self, Xn):
            k = self._kern(Xn, self.X)
            mu = k @ self.alpha
            v = np.linalg.solve(self.L, k.T)
            var = 1.0 + self.noise - np.sum(v ** 2, axis=0)
            return mu, np.sqrt(np.maximum(var, 1e-12))

    n_initial, n_iter = 15, 40
    sampler = LatinHypercube(d=n_params)
    X_init = lows + sampler.random(n=n_initial) * ranges

    X_all, y_all = [], []
    bay_best_score, bay_best_params = 1e9, None
    bay_best_met = {}

    for x in tqdm(X_init, desc="  Autoopt Bayes init", leave=True):
        s, met = _eval(x)
        X_all.append(x)
        y_all.append(s)
        if s < bay_best_score:
            bay_best_score, bay_best_params, bay_best_met = s, x, met

    gp = _GP()
    for it in tqdm(range(n_iter), desc="  Autoopt Bayes opt", leave=True):
        Xa, ya = np.array(X_all), np.array(y_all)
        Xn = (Xa - lows) / ranges
        ym, ys = ya.mean(), max(ya.std(), 1e-8)
        yn = (ya - ym) / ys
        gp.fit(Xn, yn)

        cands = np.random.rand(1000, n_params)
        mu, sigma = gp.predict(cands)
        best_n = (bay_best_score - ym) / ys
        z = (best_n - mu) / sigma
        ei = sigma * (z * norm_dist.cdf(z) + norm_dist.pdf(z))
        ei[sigma < 1e-10] = 0.0

        x_next = np.clip(lows + cands[np.argmax(ei)] * ranges, lows, highs)
        s, met = _eval(x_next)
        X_all.append(x_next)
        y_all.append(s)
        is_best = s < bay_best_score
        if is_best:
            bay_best_score, bay_best_params, bay_best_met = s, x_next, met
            tqdm.write("  Iter %2d/%d  score=%.4f  best=%.4f %s"
                       % (it + 1, n_iter, -s if s < 1e8 else 0, -bay_best_score, cok("<- BEST")))

    winners.append((bay_best_score, np.array(bay_best_params), "Bayesian", bay_best_met))
    bay_rows = []
    for x, s in zip(X_all, y_all):
        if s < 1e8:
            bay_rows.append({**_params_dict(x), "score": -s})
    if bay_rows:
        pd.DataFrame(bay_rows).to_csv("nba_bayesian_results.csv", index=False)

    t3_elapsed = time.time() - t3
    print("  Bayesian done: %.0fs  |  Best score=%.4f  Acc=%.2f%%"
          % (t3_elapsed, -bay_best_score, bay_best_met.get("accuracy", 0)))

    # ── Cleanup temp files ───────────────────────────────────────────
    for tmp in ["temp_auto.csv", "temp_auto_cal.csv"]:
        try: os.remove(tmp)
        except OSError: pass

    # ── Phase 4: Compare and apply best ──────────────────────────────
    total_time = time.time() - t_start
    total_evals = eval_count[0]

    print("\n" + "=" * 80)
    print(chi("  AUTO-OPTIMIZE RESULTS (%d params)" % n_params))
    print("=" * 80)
    print("  %-10s  %-8s  %-8s  %-8s  %-8s  %s"
          % (chi("Source"), chi("Score"), chi("Acc%"), chi("LogLoss"), chi("Brier"), chi("Parameters")))
    div(80)

    overall_best_score = 1e9
    overall_best = None
    for score, params, source, met in sorted(winners, key=lambda x: x[0]):
        is_winner = score < overall_best_score
        if is_winner:
            overall_best_score = score
            overall_best = (score, params, source, met)
        flag = cok(" ** WINNER") if is_winner and score == sorted(winners, key=lambda x: x[0])[0][0] else ""
        print("  %-10s  %8.4f  %7.2f%%  %8.4f  %8.4f  %s%s"
              % (source, -score, met.get("accuracy", 0), met.get("log_loss", 0),
                 met.get("brier", 0), _fmt(params), flag))

    div(80)
    print("  Total time: %.0f seconds  |  Total evaluations: %d" % (total_time, total_evals))

    # Apply the winner
    if overall_best:
        _, best_p, best_src, best_met = overall_best
        print("\n  %s  %s  %s" % (cok("WINNER:"), chi(best_src), _fmt(best_p)))
        print("  Acc=%s  LogLoss=%.4f  Brier=%.4f"
              % (cok("%.2f%%" % best_met.get("accuracy", 0)),
                 best_met.get("log_loss", 0), best_met.get("brier", 0)))
        _apply_best_settings(_params_dict(best_p), csv_file)

    print("=" * 80)
    return overall_best


# ── Super-Optimize ───────────────────────────────────────────────────

def super_optimize(csv_file=GAMES_FILE):
    """Exhaustive multi-round optimization across all 9 tunable parameters.

    Runs every optimization method with aggressive settings, validates the
    winner, and saves to nba_elo_settings.json (auto-loaded on every start).

    Phase 1: Broad grid search (all 9 params, ~6000+ combos)
    Phase 2: Genetic round 1 - wide bounds (50 gen x 15 pop)
    Phase 3: Bayesian round 1 - wide bounds, 30 initial + 80 iterations
    Phase 4: Genetic round 2 - tightened around overall best (30 gen x 10 pop)
    Phase 5: Bayesian round 2 - tightened, 20 initial + 60 iterations
    Phase 6: Fine grid - tiny steps around absolute best
    Phase 7: Validation - purgedcv + PBO + montecarlo on the winner
    """
    import time
    from scipy.stats.qmc import LatinHypercube

    if not os.path.exists(csv_file):
        print(cwarn("  Game data not found: %s" % csv_file))
        return None

    settings = load_elo_settings()
    base = settings.get("base_rating", 1500.0)
    use_mov = settings.get("use_mov", True)
    player_df = load_player_stats()
    has_players = not player_df.empty
    _prebuilt = build_league_player_scores(player_df) if has_players else {}
    _alt = _calc_altitude_bonus(csv_file)

    # Pre-compute game arrays once for fast evaluation
    _pc = precompute_games(csv_file, _prebuilt if has_players else None, _alt, base)

    eval_count = [0]

    # 9-param evaluation: K, HomeAdv, PlayerBoost, RestFactor, TravelFactor,
    #                      PaceFactor, PlayoffHCA, SOSFactor, FormWeight
    _PARAM_NAMES = ["K", "HA", "PB", "Rest", "Travel", "Pace", "PHCA", "SOS", "Form"]

    def _eval(params):
        k, ha, pb, rf, tf, pf, phca, sos, fw = params
        met = fast_evaluate(
            _pc, k=float(k), home_adv=float(ha), use_mov=use_mov,
            player_boost=float(pb), rest_factor=float(rf),
            travel_factor=float(tf), pace_factor=float(pf),
            playoff_hca_factor=float(phca), sos_factor=float(sos),
            form_weight=float(fw),
        )
        eval_count[0] += 1
        if met is None:
            return 1e9, {}
        # ACCURACY-FIRST objective (was: LogLoss*8 + Brier*40)
        return (100.0 - met.get("accuracy", 0)) + met["brier"] * 5.0, met

    def _params_dict(p):
        return {"k": float(p[0]), "home_adv": float(p[1]), "player_boost": float(p[2]),
                "rest_factor": float(p[3]), "travel_factor": float(p[4]),
                "pace_factor": float(p[5]), "playoff_hca_factor": float(p[6]),
                "sos_factor": float(p[7]), "form_weight": float(p[8])}

    def _fmt(p):
        return ("K=%.1f HA=%.1f PB=%.1f R=%.1f T=%.1f P=%.1f PHCA=%.2f SOS=%.1f FW=%.1f"
                % tuple(p[:9]))

    def _fmt_short(p):
        return ("K=%.1f HA=%.1f PB=%.1f R=%.1f T=%.1f P=%.1f"
                % tuple(p[:6]))

    winners = []  # [(score, params_array, source_name, metrics)]
    phase_times = []

    print("\n" + "=" * 100)
    print(chi("  SUPER-OPTIMIZE: Exhaustive multi-round optimization (9 params, all methods)"))
    print(chi("  No prompts. No shortcuts. Finds the absolute best settings."))
    print("=" * 100)
    t_start = time.time()

    # ── Phase 1: Broad Grid Search (9 params) ────────────────────────
    print(chi("\n  PHASE 1/7: Broad Grid Search (9 parameters)"))
    div(100)

    k_vals    = [8, 13, 18, 25, 35, 50]
    ha_vals   = [15, 30, 45, 60, 80]
    pb_vals   = [0, 8, 18, 30]
    rf_vals   = [0, 5, 15, 30]
    tf_vals   = [0, 15, 35]
    pf_vals   = [0, 15, 35]
    phca_vals = [0.4, 0.7, 1.0]
    sos_vals  = [0, 10, 25]
    fw_vals   = [0, 5]

    total = (len(k_vals) * len(ha_vals) * len(pb_vals) * len(rf_vals) *
             len(tf_vals) * len(pf_vals) * len(phca_vals) * len(sos_vals) * len(fw_vals))
    print("  %d combinations across 9 dimensions" % total)

    grid_best_score, grid_best_params, grid_best_met = 1e9, None, {}
    grid_rows = []
    t1 = time.time()

    for i, (k, ha, pb, rf, tf, pf, phca, sos, fw) in enumerate(
        product(k_vals, ha_vals, pb_vals, rf_vals, tf_vals, pf_vals,
                phca_vals, sos_vals, fw_vals), 1
    ):
        score, met = _eval([k, ha, pb, rf, tf, pf, phca, sos, fw])
        if score < 1e8:
            grid_rows.append({"k": k, "home_adv": ha, "player_boost": pb,
                              "rest_factor": rf, "travel_factor": tf,
                              "pace_factor": pf, "playoff_hca_factor": phca,
                              "sos_factor": sos, "form_weight": fw,
                              "accuracy": met.get("accuracy", 0),
                              "log_loss": met.get("log_loss", 0),
                              "brier": met.get("brier", 0), "score": -score})
        is_best = score < grid_best_score
        if is_best:
            grid_best_score = score
            grid_best_params = np.array([k, ha, pb, rf, tf, pf, phca, sos, fw], dtype=float)
            grid_best_met = met
        if is_best or i % 200 == 0 or i == total:
            flag = cok(" <- BEST") if is_best else ""
            print("  %5d/%d  Acc=%.2f%% Br=%.4f  %s%s"
                  % (i, total, met.get("accuracy", 0), met.get("brier", 0),
                     _fmt_short([k, ha, pb, rf, tf, pf]), flag), flush=True)

    if grid_rows:
        pd.DataFrame(grid_rows).to_csv("nba_super_grid.csv", index=False)

    t1_elapsed = time.time() - t1
    phase_times.append(("Grid (broad)", t1_elapsed))
    if grid_best_params is not None:
        winners.append((grid_best_score, grid_best_params, "Grid-Broad", grid_best_met))
        print("  Grid done: %.0fs  |  Best=%.4f  Acc=%.2f%%"
              % (t1_elapsed, -grid_best_score, grid_best_met.get("accuracy", 0)))
    else:
        print(cwarn("  Grid search failed"))
        return None

    # Track the overall best across all phases
    overall_best_score = grid_best_score
    overall_best_params = grid_best_params.copy()

    # ── Phase 2: Genetic Round 1 (wide bounds, big pop) ──────────────
    print(chi("\n  PHASE 2/7: Genetic Algorithm Round 1 (wide bounds, aggressive)"))
    div(100)
    t2 = time.time()

    wide_bounds = [
        (3, 80),      # K
        (0, 150),     # HomeAdv
        (0, 50),      # PlayerBoost
        (0, 50),      # RestFactor
        (0, 60),      # TravelFactor
        (0, 60),      # PaceFactor
        (0.0, 1.0),   # PlayoffHCA
        (0, 40),      # SOSFactor
        (0, 20),      # FormWeight
    ]

    gen1_best = [1e9]
    gen1_params = [None]
    gen1_met = [{}]
    gen1_counter = [0]

    def _gen1_cb(xk, convergence):
        gen1_counter[0] += 1
        if gen1_counter[0] % 20 == 0 and gen1_params[0] is not None:
            print("  Gen %d  best=%.4f" % (gen1_counter[0], -gen1_best[0]), flush=True)

    def _gen1_obj(params):
        score, met = _eval(params)
        if score < gen1_best[0]:
            gen1_best[0] = score
            gen1_params[0] = np.array(params)
            gen1_met[0] = met
        return score

    print("  50 generations, population 15, 9 parameters")
    de1 = differential_evolution(
        _gen1_obj, bounds=wide_bounds, maxiter=50, popsize=15,
        workers=1, polish=True, callback=_gen1_cb)

    gen1_p = de1.x
    gen1_s, gen1_m = _eval(gen1_p)
    winners.append((gen1_s, np.array(gen1_p), "Genetic-R1", gen1_m))

    t2_elapsed = time.time() - t2
    phase_times.append(("Genetic R1", t2_elapsed))
    print("  Genetic R1 done: %.0fs  |  Best=%.4f  Acc=%.2f%%"
          % (t2_elapsed, -gen1_s, gen1_m.get("accuracy", 0)))
    if gen1_s < overall_best_score:
        overall_best_score = gen1_s
        overall_best_params = np.array(gen1_p)

    # ── Phase 3: Bayesian Round 1 (wide bounds) ─────────────────────
    print(chi("\n  PHASE 3/7: Bayesian Optimization Round 1 (wide bounds)"))
    div(100)
    t3 = time.time()

    class _GP:
        def __init__(self, ls=1.0, noise=1e-4):
            self.ls, self.noise = ls, noise
        def _kern(self, A, B):
            d = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2)
            return np.exp(-0.5 * d / self.ls ** 2)
        def fit(self, X, y):
            self.X, self.y = X, y
            K = self._kern(X, X) + self.noise * np.eye(len(X))
            try:
                self.L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K += 1e-3 * np.eye(len(X))
                self.L = np.linalg.cholesky(K)
            self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y))
        def predict(self, Xn):
            k = self._kern(Xn, self.X)
            mu = k @ self.alpha
            v = np.linalg.solve(self.L, k.T)
            var = 1.0 + self.noise - np.sum(v ** 2, axis=0)
            return mu, np.sqrt(np.maximum(var, 1e-12))

    def _run_bayesian(bounds_list, n_initial, n_iter, label):
        lows = np.array([b[0] for b in bounds_list])
        highs = np.array([b[1] for b in bounds_list])
        ranges = highs - lows
        ranges[ranges < 1e-12] = 1.0

        sampler = LatinHypercube(d=9)
        X_init = lows + sampler.random(n=n_initial) * ranges
        X_all, y_all = [], []
        b_best_score, b_best_params, b_best_met = 1e9, None, {}

        print("  %s: %d initial + %d iterations" % (label, n_initial, n_iter))
        for idx, x in enumerate(X_init):
            s, met = _eval(x)
            X_all.append(x)
            y_all.append(s)
            if s < b_best_score:
                b_best_score, b_best_params, b_best_met = s, x, met
                if (idx + 1) % 5 == 0 or idx == 0:
                    print("  [%d/%d] score=%.4f %s" % (idx + 1, n_initial, -s, cok("BEST")), flush=True)
            elif (idx + 1) % 10 == 0:
                print("  [%d/%d] evaluated" % (idx + 1, n_initial), flush=True)

        gp = _GP()
        for it in range(n_iter):
            Xa, ya = np.array(X_all), np.array(y_all)
            Xn = (Xa - lows) / ranges
            ym, ys = ya.mean(), max(ya.std(), 1e-8)
            yn = (ya - ym) / ys
            gp.fit(Xn, yn)

            cands = np.random.rand(2000, 9)
            mu, sigma = gp.predict(cands)
            best_n = (b_best_score - ym) / ys
            z = (best_n - mu) / sigma
            ei = sigma * (z * norm_dist.cdf(z) + norm_dist.pdf(z))
            ei[sigma < 1e-10] = 0.0

            x_next = np.clip(lows + cands[np.argmax(ei)] * ranges, lows, highs)
            s, met = _eval(x_next)
            X_all.append(x_next)
            y_all.append(s)
            is_best = s < b_best_score
            if is_best:
                b_best_score, b_best_params, b_best_met = s, x_next, met
            if is_best or (it + 1) % 15 == 0:
                flag = cok(" <- BEST") if is_best else ""
                print("  Iter %3d/%d  best=%.4f%s" % (it + 1, n_iter, -b_best_score, flag), flush=True)

        return b_best_score, b_best_params, b_best_met, X_all, y_all

    bay1_s, bay1_p, bay1_m, bay1_X, bay1_y = _run_bayesian(wide_bounds, 30, 80, "Bayesian-R1")
    winners.append((bay1_s, np.array(bay1_p), "Bayesian-R1", bay1_m))

    t3_elapsed = time.time() - t3
    phase_times.append(("Bayesian R1", t3_elapsed))
    print("  Bayesian R1 done: %.0fs  |  Best=%.4f  Acc=%.2f%%"
          % (t3_elapsed, -bay1_s, bay1_m.get("accuracy", 0)))
    if bay1_s < overall_best_score:
        overall_best_score = bay1_s
        overall_best_params = np.array(bay1_p)

    # ── Tighten bounds around overall best ────────────────────────────
    def _tight(val, lo, hi, hw):
        tlo = max(lo, val - hw)
        thi = min(hi, val + hw)
        if tlo >= thi:
            tlo, thi = lo, hi
        return (tlo, thi)

    bp = overall_best_params
    tight_bounds = [
        _tight(bp[0], 3, 80, 6),       # K
        _tight(bp[1], 0, 150, 10),      # HomeAdv
        _tight(bp[2], 0, 50, 6),        # PlayerBoost
        _tight(bp[3], 0, 50, 6),        # RestFactor
        _tight(bp[4], 0, 60, 10),       # TravelFactor
        _tight(bp[5], 0, 60, 10),       # PaceFactor
        _tight(bp[6], 0.0, 1.0, 0.15),  # PlayoffHCA
        _tight(bp[7], 0, 40, 8),        # SOSFactor
        _tight(bp[8], 0, 20, 5),        # FormWeight
    ]
    print(cdim("\n  Tightened bounds around best-so-far: %s" % _fmt(bp)))

    # ── Phase 4: Genetic Round 2 (tight bounds) ─────────────────────
    print(chi("\n  PHASE 4/7: Genetic Algorithm Round 2 (tightened bounds)"))
    div(100)
    t4 = time.time()

    gen2_best = [1e9]
    gen2_params = [None]
    gen2_met = [{}]
    gen2_counter = [0]

    def _gen2_cb(xk, convergence):
        gen2_counter[0] += 1
        if gen2_counter[0] % 15 == 0 and gen2_params[0] is not None:
            print("  Gen %d  best=%.4f" % (gen2_counter[0], -gen2_best[0]), flush=True)

    def _gen2_obj(params):
        score, met = _eval(params)
        if score < gen2_best[0]:
            gen2_best[0] = score
            gen2_params[0] = np.array(params)
            gen2_met[0] = met
        return score

    print("  30 generations, population 10, 9 parameters")
    de2 = differential_evolution(
        _gen2_obj, bounds=tight_bounds, maxiter=30, popsize=10,
        workers=1, polish=True, callback=_gen2_cb)

    gen2_p = de2.x
    gen2_s, gen2_m = _eval(gen2_p)
    winners.append((gen2_s, np.array(gen2_p), "Genetic-R2", gen2_m))

    t4_elapsed = time.time() - t4
    phase_times.append(("Genetic R2", t4_elapsed))
    print("  Genetic R2 done: %.0fs  |  Best=%.4f  Acc=%.2f%%"
          % (t4_elapsed, -gen2_s, gen2_m.get("accuracy", 0)))
    if gen2_s < overall_best_score:
        overall_best_score = gen2_s
        overall_best_params = np.array(gen2_p)

    # ── Phase 5: Bayesian Round 2 (tight bounds) ────────────────────
    print(chi("\n  PHASE 5/7: Bayesian Optimization Round 2 (tightened bounds)"))
    div(100)
    t5 = time.time()

    bay2_s, bay2_p, bay2_m, _, _ = _run_bayesian(tight_bounds, 20, 60, "Bayesian-R2")
    winners.append((bay2_s, np.array(bay2_p), "Bayesian-R2", bay2_m))

    t5_elapsed = time.time() - t5
    phase_times.append(("Bayesian R2", t5_elapsed))
    print("  Bayesian R2 done: %.0fs  |  Best=%.4f  Acc=%.2f%%"
          % (t5_elapsed, -bay2_s, bay2_m.get("accuracy", 0)))
    if bay2_s < overall_best_score:
        overall_best_score = bay2_s
        overall_best_params = np.array(bay2_p)

    # ── Phase 6: Fine Grid around absolute best ─────────────────────
    print(chi("\n  PHASE 6/7: Fine Grid Search (tiny steps around best)"))
    div(100)
    t6 = time.time()

    bp = overall_best_params
    # Generate fine values: 3 points centered on best for each param (3^9 = 19683 max)
    def _fine_range(val, lo, hi, step):
        pts = [val - step, val, val + step]
        return sorted(set(max(lo, min(hi, round(v, 4))) for v in pts))

    fine_k    = _fine_range(bp[0], 3, 80, 0.8)
    fine_ha   = _fine_range(bp[1], 0, 150, 1.5)
    fine_pb   = _fine_range(bp[2], 0, 50, 1.0)
    fine_rf   = _fine_range(bp[3], 0, 50, 1.0)
    fine_tf   = _fine_range(bp[4], 0, 60, 1.5)
    fine_pf   = _fine_range(bp[5], 0, 60, 1.5)
    fine_phca = _fine_range(bp[6], 0.0, 1.0, 0.03)
    fine_sos  = _fine_range(bp[7], 0, 40, 1.5)
    fine_fw   = _fine_range(bp[8], 0, 20, 1.0)

    fine_total = (len(fine_k) * len(fine_ha) * len(fine_pb) * len(fine_rf) *
                  len(fine_tf) * len(fine_pf) * len(fine_phca) * len(fine_sos) * len(fine_fw))
    print("  %d fine combos (3^9 max, deduped)" % fine_total)

    fine_best_score, fine_best_params, fine_best_met = 1e9, None, {}
    fine_rows = []

    for i, (k, ha, pb, rf, tf, pf, phca, sos, fw) in enumerate(
        product(fine_k, fine_ha, fine_pb, fine_rf, fine_tf, fine_pf,
                fine_phca, fine_sos, fine_fw), 1
    ):
        score, met = _eval([k, ha, pb, rf, tf, pf, phca, sos, fw])
        if score < 1e8:
            fine_rows.append({"k": k, "home_adv": ha, "player_boost": pb,
                              "rest_factor": rf, "travel_factor": tf,
                              "pace_factor": pf, "playoff_hca_factor": phca,
                              "sos_factor": sos, "form_weight": fw,
                              "accuracy": met.get("accuracy", 0),
                              "log_loss": met.get("log_loss", 0),
                              "brier": met.get("brier", 0), "score": -score})
        is_best = score < fine_best_score
        if is_best:
            fine_best_score = score
            fine_best_params = np.array([k, ha, pb, rf, tf, pf, phca, sos, fw], dtype=float)
            fine_best_met = met
        if is_best or i % 500 == 0 or i == fine_total:
            flag = cok(" <- BEST") if is_best else ""
            print("  %5d/%d  Acc=%.2f%% Br=%.4f%s"
                  % (i, fine_total, met.get("accuracy", 0), met.get("brier", 0), flag), flush=True)

    if fine_rows:
        pd.DataFrame(fine_rows).to_csv("nba_super_fine_grid.csv", index=False)

    t6_elapsed = time.time() - t6
    phase_times.append(("Fine Grid", t6_elapsed))
    if fine_best_params is not None:
        winners.append((fine_best_score, fine_best_params, "Fine-Grid", fine_best_met))
        print("  Fine grid done: %.0fs  |  Best=%.4f  Acc=%.2f%%"
              % (t6_elapsed, -fine_best_score, fine_best_met.get("accuracy", 0)))
        if fine_best_score < overall_best_score:
            overall_best_score = fine_best_score
            overall_best_params = fine_best_params.copy()

    # ── Cleanup temp files ────────────────────────────────────────────
    for tmp in ["temp_super.csv", "temp_super_cal.csv"]:
        try: os.remove(tmp)
        except OSError: pass

    # ── Select the overall winner ─────────────────────────────────────
    winners.sort(key=lambda x: x[0])
    champion_score, champion_params, champion_src, champion_met = winners[0]

    # ── Apply winner BEFORE validation (so validation uses winning params)
    print("\n" + "=" * 100)
    print(chi("  SUPER-OPTIMIZE: Applying best settings before validation"))
    print("=" * 100)
    print("  %s from %s: %s" % (cok("CHAMPION"), chi(champion_src), _fmt(champion_params)))
    print("  Acc=%s  LogLoss=%.4f  Brier=%.4f"
          % (cok("%.2f%%" % champion_met.get("accuracy", 0)),
             champion_met.get("log_loss", 0), champion_met.get("brier", 0)))
    _apply_best_settings(_params_dict(champion_params), csv_file)

    # ── Phase 7: Validation ──────────────────────────────────────────
    print(chi("\n  PHASE 7/7: Validation (Purged CV + PBO + Monte Carlo)"))
    div(100)
    t7 = time.time()

    # 7a: Purged walk-forward CV
    print(chi("\n  7a. Purged Walk-Forward Cross-Validation"))
    try:
        purged_walk_forward_cv(csv_file, k_folds=5, embargo_games=5)
    except Exception as e:
        print(cwarn("  Purged CV error: %s" % e))

    # 7b: PBO (needs grid search CSV)
    print(chi("\n  7b. Probability of Backtest Overfitting"))
    try:
        pbo_file = "nba_super_grid.csv"
        if os.path.exists(pbo_file):
            probability_of_backtest_overfitting(pbo_file)
        elif os.path.exists("nba_grid_search.csv"):
            probability_of_backtest_overfitting()
        else:
            print(cdim("  No grid CSV for PBO — skipped"))
    except Exception as e:
        print(cwarn("  PBO error: %s" % e))

    # 7c: Monte Carlo permutation test (reduced to 200 for speed)
    print(chi("\n  7c. Monte Carlo Permutation Test (200 permutations)"))
    try:
        monte_carlo_permutation_test(csv_file, n_permutations=200)
    except Exception as e:
        print(cwarn("  Monte Carlo error: %s" % e))

    t7_elapsed = time.time() - t7
    phase_times.append(("Validation", t7_elapsed))

    # ── Final Summary ────────────────────────────────────────────────
    total_time = time.time() - t_start
    total_evals = eval_count[0]

    print("\n" + "=" * 100)
    print(chi("  SUPER-OPTIMIZE COMPLETE"))
    print("=" * 100)

    # Results table
    print("\n  %-14s  %-8s  %-8s  %-8s  %-8s  %s"
          % (chi("Phase"), chi("Score"), chi("Acc%"), chi("LogLoss"), chi("Brier"), chi("Parameters")))
    div(100)

    for score, params, source, met in winners:
        is_champ = (score == champion_score and source == champion_src)
        flag = cok(" ** WINNER") if is_champ else ""
        print("  %-14s  %8.4f  %7.2f%%  %8.4f  %8.4f  %s%s"
              % (source, -score, met.get("accuracy", 0), met.get("log_loss", 0),
                 met.get("brier", 0), _fmt(params), flag))

    div(100)
    print("\n  Phase Timing:")
    for name, elapsed in phase_times:
        print("    %-20s %6.0fs  (%d min)" % (name, elapsed, elapsed / 60))
    print("    %-20s %6.0fs  (%d min)" % ("TOTAL", total_time, total_time / 60))
    print("  Total evaluations: %d" % total_evals)

    # Confirm settings are saved
    print("\n  %s" % cok("Settings saved to nba_elo_settings.json (auto-loads on every start)"))
    print("  %s" % cok("Platt scaler refitted and saved to nba_platt_scaler.json"))
    print("\n  Winning parameters (%s):" % champion_src)
    print("    %s" % _fmt(champion_params))
    print("    Accuracy: %s  LogLoss: %.4f  Brier: %.4f"
          % (cok("%.2f%%" % champion_met.get("accuracy", 0)),
             champion_met.get("log_loss", 0), champion_met.get("brier", 0)))

    # DSR on the fine grid results
    if fine_rows:
        scores_arr = np.array([r["score"] for r in fine_rows])
        dsr = deflated_sharpe_ratio(scores_arr)
        if dsr is not None:
            if dsr >= 1.96:
                print("    DSR: %.2f %s" % (dsr, cok("(significant at 95%)")))
            else:
                print("    DSR: %.2f %s" % (dsr, cwarn("(not significant)")))

    print("=" * 100)
    return (champion_score, champion_params, champion_src, champion_met)
