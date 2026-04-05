"""Coordinate-descent (single-parameter-at-a-time) optimizer for NHL Elo.

Sweeps each parameter individually while holding all others fixed.
Repeats passes until no improvement or max passes reached.
Optimizes for prediction accuracy (%).

Usage:
    python single_param_opt.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd

from config import GAMES_FILE, load_elo_settings, save_elo_settings
from elo_model import NHLElo
from data_players import load_player_stats, build_league_player_scores
from build_model import _calc_altitude_bonus

# ---------------------------------------------------------------------------
# Parameters to optimize: (min, max, coarse_step)
# ---------------------------------------------------------------------------
PARAM_RANGES = {
    "k":                   (0.5, 100.0, 2.0),
    "home_adv":            (0.0, 200.0, 5.0),
    "player_boost":        (0.0, 200.0, 5.0),
    "starter_boost":       (0.0, 200.0, 5.0),
    "rest_factor":         (0.0, 200.0, 5.0),
    "form_weight":         (0.0, 200.0, 5.0),
    "travel_factor":       (0.0, 200.0, 5.0),
    "sos_factor":          (0.0, 200.0, 5.0),
    "pace_factor":         (0.0, 200.0, 5.0),
    "playoff_hca_factor":  (0.0,   2.0, 0.1),
    "division_factor":     (0.0, 200.0, 5.0),
    "mean_reversion":      (0.0, 200.0, 5.0),
    "b2b_penalty":         (0.0, 200.0, 5.0),
    "road_trip_factor":    (0.0, 200.0, 5.0),
    "homestand_factor":    (0.0, 200.0, 5.0),
    "win_streak_factor":   (0.0, 200.0, 5.0),
    "altitude_factor":     (0.0,  20.0, 1.0),
    "season_phase_factor": (0.0, 200.0, 5.0),
    "scoring_consistency_factor": (0.0, 200.0, 5.0),
    "rest_advantage_cap":  (0.0,  30.0, 1.0),
    "overtime_factor":     (0.0, 200.0, 5.0),
}

_ELO_KEYS = {
    "base_rating", "k", "home_adv", "use_mov", "player_boost",
    "starter_boost", "rest_factor", "form_weight", "travel_factor",
    "sos_factor", "pace_factor", "playoff_hca_factor",
    "division_factor", "mean_reversion",
    "b2b_penalty", "road_trip_factor", "homestand_factor",
    "win_streak_factor", "altitude_factor", "season_phase_factor",
    "scoring_consistency_factor", "rest_advantage_cap", "overtime_factor",
}

MAX_PASSES = 5


# ---------------------------------------------------------------------------
# Data loading (done once)
# ---------------------------------------------------------------------------
def _load_data(csv_file):
    """Load games DataFrame, player stats, altitude bonus, and starters."""
    if not os.path.exists(csv_file):
        print("ERROR: %s not found" % csv_file)
        sys.exit(1)

    games = pd.read_csv(csv_file)
    if "neutral_site" not in games.columns:
        games["neutral_site"] = False
    games["_date_parsed"] = pd.to_datetime(games["date"], errors="coerce")

    player_df = load_player_stats()
    alt_bonus = _calc_altitude_bonus(csv_file)

    # Pre-extract starters so we don't re-parse per evaluation
    starters = []
    for _, row in games.iterrows():
        starters.append((
            str(row.get("home_starter", "") or "").strip(),
            str(row.get("away_starter", "") or "").strip(),
        ))

    return games, player_df, alt_bonus, starters


# ---------------------------------------------------------------------------
# Evaluation function  (returns accuracy %)
# ---------------------------------------------------------------------------
def evaluate(settings, games, player_df, alt_bonus, starters):
    """Run walk-forward backtest with given settings, return accuracy %."""
    elo_kwargs = {k: v for k, v in settings.items() if k in _ELO_KEYS}
    model = NHLElo(**elo_kwargs)
    model._altitude_bonus = alt_bonus
    if not player_df.empty:
        model.set_player_stats(player_df)

    correct = 0
    total = 0
    prev_season = None

    for idx, (_, row) in enumerate(games.iterrows()):
        game_date = row["_date_parsed"] if pd.notna(row["_date_parsed"]) else None
        home_starter, away_starter = starters[idx]

        # Season boundary -> regress goalie ratings
        if game_date is not None:
            row_season = game_date.year + 1 if game_date.month >= 10 else game_date.year
            if prev_season is not None and row_season != prev_season:
                model.regress_goalie_ratings(factor=0.5)
            prev_season = row_season

        try:
            p = model.win_prob(
                row["home_team"], row["away_team"],
                team_a_home=True,
                neutral_site=bool(row["neutral_site"]),
                calibrated=False,
                game_date=game_date,
                use_injuries=False,
                home_starter=home_starter,
                away_starter=away_starter,
            )
            home_actual = 1 if row["home_score"] > row["away_score"] else 0
            if (p >= 0.5) == (home_actual == 1):
                correct += 1
            total += 1

            model.update_game(
                row["home_team"], row["away_team"],
                row["home_score"], row["away_score"],
                neutral_site=bool(row["neutral_site"]),
                game_date=game_date,
                home_starter=home_starter,
                away_starter=away_starter,
            )
        except Exception:
            pass

    if total == 0:
        return 0.0
    return correct / total * 100.0


# ---------------------------------------------------------------------------
# Single-parameter sweep (coarse + fine)
# ---------------------------------------------------------------------------
def sweep_param(param_name, settings, games, player_df, alt_bonus, starters,
                current_best_acc):
    """Sweep one parameter over its range. Returns (best_value, best_acc).

    First does a coarse sweep over the full range, then a fine sweep
    around the coarse winner with step/5 resolution and +/- 2*step range.
    """
    lo, hi, step = PARAM_RANGES[param_name]
    original_value = settings[param_name]

    # --- coarse sweep ---
    coarse_values = np.arange(lo, hi + step * 0.01, step)
    best_val = original_value
    best_acc = current_best_acc

    for val in coarse_values:
        val = round(float(val), 6)
        settings[param_name] = val
        acc = evaluate(settings, games, player_df, alt_bonus, starters)
        if acc > best_acc:
            best_acc = acc
            best_val = val
            print("    >> %.4f -> %.3f%%  (new best)" % (val, acc))

    # --- fine sweep around coarse winner ---
    fine_step = step / 5.0
    fine_lo = max(lo, best_val - 2.0 * step)
    fine_hi = min(hi, best_val + 2.0 * step)
    fine_values = np.arange(fine_lo, fine_hi + fine_step * 0.01, fine_step)

    for val in fine_values:
        val = round(float(val), 6)
        settings[param_name] = val
        acc = evaluate(settings, games, player_df, alt_bonus, starters)
        if acc > best_acc:
            best_acc = acc
            best_val = val
            print("    >> %.4f -> %.3f%%  (fine, new best)" % (val, acc))

    # Restore best value into settings
    settings[param_name] = best_val
    return best_val, best_acc


# ---------------------------------------------------------------------------
# Main coordinate-descent loop
# ---------------------------------------------------------------------------
def run_coordinate_descent(csv_file=GAMES_FILE):
    """Run coordinate descent optimization across all parameters."""
    print("=" * 62)
    print("  NHL ELO  --  COORDINATE DESCENT OPTIMIZER")
    print("=" * 62)

    # Load data once
    print("\n  Loading data...")
    t0 = time.time()
    games, player_df, alt_bonus, starters = _load_data(csv_file)
    print("  Loaded %d games in %.1fs" % (len(games), time.time() - t0))

    # Load current settings as starting point
    settings = load_elo_settings()
    param_names = list(PARAM_RANGES.keys())

    # Baseline accuracy
    print("\n  Computing baseline accuracy...")
    baseline_acc = evaluate(settings, games, player_df, alt_bonus, starters)
    print("  Baseline accuracy: %.3f%%\n" % baseline_acc)

    best_acc = baseline_acc

    for pass_num in range(1, MAX_PASSES + 1):
        print("-" * 62)
        print("  PASS %d / %d  (current best: %.3f%%)" % (
            pass_num, MAX_PASSES, best_acc))
        print("-" * 62)

        pass_start_acc = best_acc
        improved_any = False

        for param in param_names:
            old_val = settings[param]
            print("\n  Sweeping %-22s  (current=%.4f, acc=%.3f%%)" % (
                param, old_val, best_acc))

            new_val, new_acc = sweep_param(
                param, settings, games, player_df, alt_bonus, starters,
                best_acc,
            )

            if new_acc > best_acc:
                print("  * %-22s  %.4f -> %.4f  (acc %.3f%% -> %.3f%%)" % (
                    param, old_val, new_val, best_acc, new_acc))
                best_acc = new_acc
                improved_any = True
            else:
                # Restore original value since no improvement
                settings[param] = old_val
                print("  . %-22s  no improvement (kept %.4f)" % (
                    param, old_val))

        pass_delta = best_acc - pass_start_acc
        print("\n  Pass %d complete:  accuracy %.3f%%  (delta +%.3f%%)" % (
            pass_num, best_acc, pass_delta))

        if not improved_any:
            print("  No parameter improved this pass -- stopping early.")
            break

    # Summary
    total_gain = best_acc - baseline_acc
    print("\n" + "=" * 62)
    print("  OPTIMIZATION COMPLETE")
    print("  Baseline:  %.3f%%" % baseline_acc)
    print("  Final:     %.3f%%  (+%.3f%%)" % (best_acc, total_gain))
    print("  " + "-" * 40)
    for param in param_names:
        print("    %-22s = %.4f" % (param, settings[param]))
    print("=" * 62)

    # Save
    save_elo_settings(settings)
    print("  Settings saved to nhl_elo_settings.json")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_coordinate_descent()
