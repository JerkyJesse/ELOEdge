"""Quick accuracy test -- run after each improvement to measure delta."""

import sys
import os
import numpy as np
import pandas as pd
from collections import defaultdict

from config import GAMES_FILE, load_elo_settings
from elo_model import NFLElo
from data_players import load_player_stats, build_league_player_scores
from build_model import _calc_altitude_bonus
from metrics import log_loss_binary, brier_score_binary, calibration_table
from platt import fit_platt_scaler, apply_platt


def run_accuracy_test(label="current", csv_file=GAMES_FILE):
    """Train-predict walk-forward backtest and report metrics."""
    if not os.path.exists(csv_file):
        print("ERROR: %s not found" % csv_file)
        return None

    settings = load_elo_settings()
    _elo_keys = {"base_rating", "k", "home_adv", "use_mov", "player_boost",
                 "rest_factor", "form_weight", "travel_factor", "sos_factor",
                 "playoff_hca_factor", "pace_factor", "division_factor", "mean_reversion",
                 "b2b_penalty", "road_trip_factor", "homestand_factor", "win_streak_factor",
                 "altitude_factor", "season_phase_factor", "scoring_consistency_factor",
                 "rest_advantage_cap", "bye_week_factor"}
    model = NFLElo(**{k: v for k, v in settings.items() if k in _elo_keys})
    model._altitude_bonus = _calc_altitude_bonus(csv_file)
    player_df = load_player_stats()
    if not player_df.empty:
        model.set_player_stats(player_df)

    games = pd.read_csv(csv_file)
    if "neutral_site" not in games.columns:
        games["neutral_site"] = False
    games["_date_parsed"] = pd.to_datetime(games["date"], errors="coerce")

    probs, actuals = [], []
    correct = 0
    for _, row in games.iterrows():
        try:
            game_date = row["_date_parsed"] if pd.notna(row["_date_parsed"]) else None
            home_win_prob = model.win_prob(
                row["home_team"], row["away_team"],
                team_a_home=True, neutral_site=bool(row["neutral_site"]),
                calibrated=False, game_date=game_date, use_injuries=False,
            )
            pred_winner = row["home_team"] if home_win_prob >= 0.5 else row["away_team"]
            home_actual = 1 if row["home_score"] > row["away_score"] else 0
            actual_winner = row["home_team"] if home_actual == 1 else row["away_team"]
            probs.append(home_win_prob)
            actuals.append(home_actual)
            if pred_winner == actual_winner:
                correct += 1
            model.update_game(
                row["home_team"], row["away_team"],
                row["home_score"], row["away_score"],
                neutral_site=bool(row["neutral_site"]),
                game_date=game_date,
            )
        except Exception as e:
            pass

    n = len(probs)
    if n == 0:
        print("ERROR: no predictions generated")
        return None

    # Raw metrics
    acc_raw = correct / n * 100
    ll_raw = log_loss_binary(actuals, probs)
    brier_raw = brier_score_binary(actuals, probs)

    # Fit Platt and get calibrated metrics
    scaler = fit_platt_scaler(probs, actuals)
    cal_probs = [apply_platt(p, scaler) for p in probs]
    cal_correct = sum(1 for p, a in zip(cal_probs, actuals) if (p >= 0.5) == (a == 1))
    acc_cal = cal_correct / n * 100
    ll_cal = log_loss_binary(actuals, cal_probs)
    brier_cal = brier_score_binary(actuals, cal_probs)

    print("\n" + "=" * 60)
    print("  ACCURACY TEST: %s" % label)
    print("=" * 60)
    print("  Games tested:     %d" % n)
    print("  --- Raw (uncalibrated) ---")
    print("  Accuracy:         %.2f%%" % acc_raw)
    print("  Log Loss:         %.4f" % ll_raw)
    print("  Brier Score:      %.4f" % brier_raw)
    print("  --- Platt calibrated ---")
    print("  Accuracy:         %.2f%%" % acc_cal)
    print("  Log Loss:         %.4f" % ll_cal)
    print("  Brier Score:      %.4f" % brier_cal)
    print("=" * 60)

    return {
        "label": label, "n": n,
        "acc_raw": acc_raw, "ll_raw": ll_raw, "brier_raw": brier_raw,
        "acc_cal": acc_cal, "ll_cal": ll_cal, "brier_cal": brier_cal,
    }


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    run_accuracy_test(label)
