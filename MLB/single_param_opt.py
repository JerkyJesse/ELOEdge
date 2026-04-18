"""Coordinate descent (single-parameter-at-a-time) optimizer for MLB Elo model.

Sweeps each parameter individually while keeping all others fixed,
then refines with a fine-grained pass around the best value found.
Repeats until no parameter improves or max passes reached.
"""

import sys
import time
import copy
import logging

import numpy as np
import pandas as pd

from collections import defaultdict

from config import GAMES_FILE, load_elo_settings, save_elo_settings
from elo_model import MLBElo
from data_players import load_player_stats, load_fangraphs_pitching
from build_model import _calc_altitude_bonus
from platt import regress_ratings_to_mean


# ── Elo constructor keys ────────────────────────────────────────────
_ELO_KEYS = {
    "base_rating", "k", "home_adv", "use_mov", "player_boost",
    "starter_boost", "rest_factor", "form_weight", "travel_factor",
    "sos_factor", "pace_factor", "playoff_hca_factor",
    "division_factor", "mean_reversion",
    "pyth_factor", "home_road_factor", "mov_base",
    "b2b_penalty", "road_trip_factor", "homestand_factor", "win_streak_factor",
    "altitude_factor", "season_phase_factor", "scoring_consistency_factor",
    "rest_advantage_cap", "park_factor_weight",
    "mov_cap", "east_travel_penalty", "series_adaptation",
    "interleague_factor", "bullpen_factor", "opp_pitcher_factor",
    "k_decay", "surprise_k",
}

# ── Parameter search ranges: (min, max, coarse_step) ────────────────
PARAM_RANGES = {
    "k":                  (0.1, 100.0,  2.0),
    "home_adv":           (0.0, 200.0,  5.0),
    "player_boost":       (0.0, 200.0,  5.0),
    "starter_boost":      (0.0, 200.0,  5.0),
    "rest_factor":        (0.0, 200.0,  5.0),
    "form_weight":        (0.0, 200.0,  5.0),
    "travel_factor":      (0.0, 200.0,  5.0),
    "sos_factor":         (0.0, 200.0,  5.0),
    "pace_factor":        (0.0, 200.0,  5.0),
    "playoff_hca_factor": (0.0,   2.0,  0.1),
    "division_factor":    (0.0, 200.0,  5.0),
    "mean_reversion":     (0.0, 200.0,  5.0),
    "season_regress":     (0.0,   1.0,  0.05),
    "pyth_factor":        (0.0, 200.0,  5.0),
    "home_road_factor":   (0.0, 200.0,  5.0),
    "b2b_penalty":        (0.0, 200.0,  5.0),
    "road_trip_factor":   (0.0, 200.0,  5.0),
    "homestand_factor":   (0.0, 200.0,  5.0),
    "win_streak_factor":  (0.0, 200.0,  5.0),
    "altitude_factor":    (0.0,  20.0,  1.0),
    "season_phase_factor": (0.0, 200.0, 5.0),
    "scoring_consistency_factor": (0.0, 200.0, 5.0),
    "rest_advantage_cap": (0.0,  30.0,  1.0),
    "park_factor_weight": (0.0,  20.0,  1.0),
    "mov_cap":            (0.0,  20.0,  1.0),
    "east_travel_penalty": (0.0, 200.0, 5.0),
    "series_adaptation":  (0.0, 200.0,  5.0),
    "interleague_factor": (0.0, 200.0,  5.0),
    "bullpen_factor":     (0.0, 200.0,  5.0),
    "opp_pitcher_factor": (0.0, 200.0,  5.0),
    "k_decay":            (0.0,   5.0,  0.2),
    "surprise_k":         (0.0,  10.0,  0.5),
}

MAX_PASSES = 5


# ── Fast evaluation cache (shared with backtest.py) ────────────────
_fast_cache = None

def _preload_data(csv_file=GAMES_FILE):
    """Build fast evaluation cache once."""
    global _fast_cache
    from backtest import _get_fast_cache
    _fast_cache = _get_fast_cache(csv_file)


def evaluate(settings):
    """Run a fast walk-forward backtest and return accuracy percentage.

    Uses the numpy-backed FastEloCache for 10-100x speedup over the
    original row-by-row MLBElo approach.
    """
    if _fast_cache is None:
        raise RuntimeError("Call _preload_data() before evaluate()")
    met = _fast_cache.evaluate(settings)
    return met["accuracy"]


def _coarse_sweep(param, settings, current_best_acc):
    """Sweep the full range for *param* at coarse resolution.

    Returns (best_value, best_accuracy).
    """
    lo, hi, step = PARAM_RANGES[param]
    original_val = settings[param]
    best_val = original_val
    best_acc = current_best_acc

    candidates = np.arange(lo, hi + step * 0.01, step)
    n = len(candidates)

    for idx, val in enumerate(candidates):
        val = round(float(val), 4)
        if abs(val - original_val) < 1e-9:
            # Already evaluated as baseline
            continue
        settings[param] = val
        acc = evaluate(settings)
        tag = " ** NEW BEST" if acc > best_acc else ""
        print("    [{}/{}] {} = {:<8.4f}  acc = {:.4f}%{}".format(
            idx + 1, n, param, val, acc, tag))
        if acc > best_acc:
            best_acc = acc
            best_val = val

    # Restore best into settings
    settings[param] = best_val
    return best_val, best_acc


def _fine_sweep(param, settings, coarse_best_val, current_best_acc):
    """Refine around coarse_best_val with step/5 resolution, +/- 2*step."""
    lo, hi, step = PARAM_RANGES[param]
    fine_step = step / 5.0
    sweep_lo = max(lo, coarse_best_val - 2 * step)
    sweep_hi = min(hi, coarse_best_val + 2 * step)

    best_val = coarse_best_val
    best_acc = current_best_acc

    candidates = np.arange(sweep_lo, sweep_hi + fine_step * 0.01, fine_step)
    n = len(candidates)

    print("    Fine sweep: {} in [{:.4f}, {:.4f}] step {:.4f}  ({} pts)".format(
        param, sweep_lo, sweep_hi, fine_step, n))

    for idx, val in enumerate(candidates):
        val = round(float(val), 4)
        if abs(val - coarse_best_val) < 1e-9:
            continue
        settings[param] = val
        acc = evaluate(settings)
        tag = " ** NEW BEST" if acc > best_acc else ""
        print("    [{}/{}] {} = {:<8.4f}  acc = {:.4f}%{}".format(
            idx + 1, n, param, val, acc, tag))
        if acc > best_acc:
            best_acc = acc
            best_val = val

    settings[param] = best_val
    return best_val, best_acc


def run_coordinate_descent(csv_file=GAMES_FILE):
    """Run coordinate descent optimization."""
    print("=" * 65)
    print("  MLB Elo -- Coordinate Descent Parameter Optimizer")
    print("=" * 65)
    print()

    # Pre-load all data
    _preload_data(csv_file)
    print()

    # Load current settings
    settings = load_elo_settings()
    param_names = list(PARAM_RANGES.keys())

    # Baseline accuracy
    print("Evaluating baseline accuracy ...")
    baseline_acc = evaluate(settings)
    print("Baseline accuracy: {:.4f}%".format(baseline_acc))
    print("Current parameter values:")
    for p in param_names:
        print("  {:22s} = {}".format(p, settings[p]))
    print()

    global_best_acc = baseline_acc

    for pass_num in range(1, MAX_PASSES + 1):
        pass_start_acc = global_best_acc
        print("-" * 65)
        print("PASS {}/{}  (current best accuracy: {:.4f}%)".format(
            pass_num, MAX_PASSES, global_best_acc))
        print("-" * 65)

        improved_any = False

        for param in param_names:
            print()
            print("  Sweeping '{}'  (current value: {})".format(
                param, settings[param]))
            t0 = time.time()

            # Coarse sweep
            coarse_val, coarse_acc = _coarse_sweep(
                param, settings, global_best_acc)

            # Fine sweep around best coarse value
            fine_val, fine_acc = _fine_sweep(
                param, settings, coarse_val, coarse_acc)

            elapsed = time.time() - t0

            if fine_acc > global_best_acc:
                print("  >> '{}' improved: {} -> {}  (acc {:.4f}% -> {:.4f}%)  [{:.1f}s]".format(
                    param, settings.get(param), fine_val,
                    global_best_acc, fine_acc, elapsed))
                settings[param] = fine_val
                global_best_acc = fine_acc
                improved_any = True
            else:
                print("  >> '{}' unchanged at {}  [{:.1f}s]".format(
                    param, settings[param], elapsed))

        pass_improvement = global_best_acc - pass_start_acc
        print()
        print("Pass {} complete.  Accuracy: {:.4f}%  (improvement this pass: {:+.4f}%)".format(
            pass_num, global_best_acc, pass_improvement))

        if not improved_any:
            print("No parameter improved -- stopping early.")
            break

    # ── Summary ──────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  OPTIMIZATION COMPLETE")
    print("=" * 65)
    total_improvement = global_best_acc - baseline_acc
    print("Baseline accuracy:  {:.4f}%".format(baseline_acc))
    print("Final accuracy:     {:.4f}%".format(global_best_acc))
    print("Total improvement:  {:+.4f}%".format(total_improvement))
    print()
    print("Optimized parameters:")
    for p in param_names:
        print("  {:22s} = {}".format(p, settings[p]))
    print()

    # Save
    print("Saving best settings ...")
    save_elo_settings(settings)
    print("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    optimize()
