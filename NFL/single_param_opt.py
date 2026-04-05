"""Coordinate-descent (single-parameter-at-a-time) optimizer for NFL Elo model.

Sweeps each parameter independently through a range of values while holding
all others fixed, then does a fine-grained pass around the best found value.
Repeats until no parameter improves or max passes is reached.
"""

import os
import sys
import time

import numpy as np
import pandas as pd

from config import GAMES_FILE, load_elo_settings, save_elo_settings
from elo_model import NFLElo
from data_players import load_player_stats
from build_model import _calc_altitude_bonus

# ── Parameters to optimize: (min, max, coarse_step) ──────────────────────
PARAM_RANGES = {
    "k":                   (0.5, 100.0, 2.0),
    "home_adv":            (0.0, 200.0, 5.0),
    "player_boost":        (0.0, 200.0, 5.0),
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
    "bye_week_factor":     (0.0, 200.0, 5.0),
}

_ELO_KEYS = {
    "base_rating", "k", "home_adv", "use_mov", "player_boost",
    "rest_factor", "form_weight", "travel_factor", "sos_factor",
    "pace_factor", "playoff_hca_factor",
    "division_factor", "mean_reversion",
    "b2b_penalty", "road_trip_factor", "homestand_factor", "win_streak_factor",
    "altitude_factor", "season_phase_factor", "scoring_consistency_factor",
    "rest_advantage_cap", "bye_week_factor",
}

MAX_PASSES = 5


# ── Evaluation function ──────────────────────────────────────────────────
def evaluate(settings, games, player_df, alt_bonus):
    """Run walk-forward backtest and return accuracy percentage."""
    elo_args = {k: v for k, v in settings.items() if k in _ELO_KEYS}
    model = NFLElo(**elo_args)
    model._altitude_bonus = alt_bonus
    if not player_df.empty:
        model.set_player_stats(player_df)

    correct = 0
    total = 0
    for _, row in games.iterrows():
        game_date = row["_date_parsed"] if pd.notna(row["_date_parsed"]) else None
        try:
            p = model.win_prob(
                row["home_team"], row["away_team"],
                team_a_home=True,
                neutral_site=bool(row["neutral_site"]),
                calibrated=False,
                game_date=game_date,
                use_injuries=False,
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
            )
        except Exception:
            pass

    if total == 0:
        return 0.0
    return correct / total * 100.0


# ── Sweep helpers ─────────────────────────────────────────────────────────
def coarse_sweep(param_name, settings, games, player_df, alt_bonus, current_best_acc):
    """Sweep param_name over its full range; return (best_value, best_acc)."""
    lo, hi, step = PARAM_RANGES[param_name]
    values = np.arange(lo, hi + step * 0.01, step)

    best_val = settings[param_name]
    best_acc = current_best_acc

    for v in values:
        settings[param_name] = float(v)
        acc = evaluate(settings, games, player_df, alt_bonus)
        if acc > best_acc:
            best_acc = acc
            best_val = float(v)
            print("      -> %.4g  acc=%.3f%% (NEW BEST)" % (float(v), acc))

    # Restore to best found
    settings[param_name] = best_val
    return best_val, best_acc


def fine_sweep(param_name, settings, games, player_df, alt_bonus, coarse_best, current_best_acc):
    """Narrow sweep around coarse_best with step/5 resolution, +/- 2*step range."""
    lo_range, hi_range, step = PARAM_RANGES[param_name]
    fine_step = step / 5.0

    sweep_lo = max(lo_range, coarse_best - 2.0 * step)
    sweep_hi = min(hi_range, coarse_best + 2.0 * step)
    values = np.arange(sweep_lo, sweep_hi + fine_step * 0.01, fine_step)

    best_val = coarse_best
    best_acc = current_best_acc

    for v in values:
        settings[param_name] = float(v)
        acc = evaluate(settings, games, player_df, alt_bonus)
        if acc > best_acc:
            best_acc = acc
            best_val = float(v)
            print("      [fine] -> %.4g  acc=%.3f%%" % (float(v), acc))

    settings[param_name] = best_val
    return best_val, best_acc


# ── Main optimizer ────────────────────────────────────────────────────────
def run_coordinate_descent(csv_file=GAMES_FILE):
    """Run coordinate-descent optimization over all Elo parameters."""
    if not os.path.exists(csv_file):
        print("ERROR: %s not found. Run data download first." % csv_file)
        return

    # ── Pre-load data once ────────────────────────────────────────────────
    print("=" * 64)
    print("  NFL COORDINATE-DESCENT SINGLE-PARAM OPTIMIZER")
    print("=" * 64)
    print("\nLoading data...")

    games = pd.read_csv(csv_file)
    if "neutral_site" not in games.columns:
        games["neutral_site"] = False
    games["_date_parsed"] = pd.to_datetime(games["date"], errors="coerce")
    print("  Games loaded: %d rows" % len(games))

    player_df = load_player_stats()
    if player_df.empty:
        print("  Player stats: none (proceeding without)")
    else:
        print("  Player stats: %d rows" % len(player_df))

    alt_bonus = _calc_altitude_bonus(csv_file)
    print("  Altitude bonuses: %s" % (alt_bonus if alt_bonus else "none"))

    settings = load_elo_settings()

    # ── Baseline ──────────────────────────────────────────────────────────
    print("\nCalculating baseline accuracy...")
    baseline_acc = evaluate(settings, games, player_df, alt_bonus)
    print("  Baseline accuracy: %.3f%%\n" % baseline_acc)

    best_acc = baseline_acc
    param_names = list(PARAM_RANGES.keys())

    # ── Coordinate-descent passes ─────────────────────────────────────────
    for pass_num in range(1, MAX_PASSES + 1):
        pass_start_acc = best_acc
        print("-" * 64)
        print("  PASS %d / %d  (current best: %.3f%%)" % (pass_num, MAX_PASSES, best_acc))
        print("-" * 64)

        improved_any = False

        for param in param_names:
            old_val = settings[param]
            lo, hi, step = PARAM_RANGES[param]
            n_vals = int((hi - lo) / step) + 1
            print("\n  [%s] current=%.4g  sweeping %d values [%.3g .. %.3g]" % (
                param, old_val, n_vals, lo, hi))
            t0 = time.time()

            # Coarse sweep
            coarse_val, coarse_acc = coarse_sweep(
                param, settings, games, player_df, alt_bonus, best_acc)

            if coarse_acc > best_acc:
                # Fine sweep around the coarse best
                print("    Coarse best: %.4g (%.3f%%) -- running fine sweep..." % (
                    coarse_val, coarse_acc))
                fine_val, fine_acc = fine_sweep(
                    param, settings, games, player_df, alt_bonus, coarse_val, coarse_acc)
                best_acc = fine_acc
                settings[param] = fine_val
                improved_any = True
                elapsed = time.time() - t0
                print("    UPDATED %s: %.4g -> %.4g  acc=%.3f%%  (%.1fs)" % (
                    param, old_val, fine_val, best_acc, elapsed))
            else:
                # No improvement: restore original value
                settings[param] = old_val
                elapsed = time.time() - t0
                print("    No improvement. Keeping %s=%.4g  (%.1fs)" % (
                    param, old_val, elapsed))

        pass_improvement = best_acc - pass_start_acc
        print("\n  Pass %d complete. Accuracy: %.3f%% (improvement: %+.3f%%)" % (
            pass_num, best_acc, pass_improvement))

        if not improved_any:
            print("  No parameter improved this pass. Stopping early.")
            break

    # ── Summary and save ──────────────────────────────────────────────────
    total_improvement = best_acc - baseline_acc
    print("\n" + "=" * 64)
    print("  OPTIMIZATION COMPLETE")
    print("=" * 64)
    print("  Baseline accuracy:  %.3f%%" % baseline_acc)
    print("  Final accuracy:     %.3f%%" % best_acc)
    print("  Total improvement:  %+.3f%%" % total_improvement)
    print("\n  Final parameter values:")
    for param in param_names:
        print("    %-22s = %.4g" % (param, settings[param]))

    save_elo_settings(settings)
    print("\n  Settings saved to nfl_elo_settings.json")


if __name__ == "__main__":
    run_coordinate_descent()
