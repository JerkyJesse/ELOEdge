"""Coordinate-descent (single-parameter-at-a-time) optimizer for NBA Elo.

Sweeps each parameter individually while holding the rest fixed.
Coarse pass finds the best value, then a fine-grained pass refines it.
Repeats up to 5 full passes until no parameter improves.
"""

import sys
import os
import time
import numpy as np
import pandas as pd

from config import GAMES_FILE, load_elo_settings, save_elo_settings
from elo_model import NBAElo
from data_players import load_player_stats
from build_model import _calc_altitude_bonus

# --------------------------------------------------------------------------- #
#  Elo constructor keys (must match NBAElo.__init__ kwargs)
# --------------------------------------------------------------------------- #
_ELO_KEYS = {
    "base_rating", "k", "home_adv", "use_mov", "player_boost",
    "rest_factor", "form_weight", "travel_factor", "sos_factor",
    "pace_factor", "playoff_hca_factor",
    "division_factor", "mean_reversion",
    "b2b_penalty", "road_trip_factor", "homestand_factor", "win_streak_factor",
    "altitude_factor", "season_phase_factor", "scoring_consistency_factor",
    "rest_advantage_cap",
}

# --------------------------------------------------------------------------- #
#  Parameter ranges: (min, max, coarse_step)
# --------------------------------------------------------------------------- #
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
}

MAX_PASSES = 5


# --------------------------------------------------------------------------- #
#  Evaluation function (accuracy %)
# --------------------------------------------------------------------------- #
def evaluate(params_dict, games, player_df, alt_bonus):
    """Run walk-forward backtest with given params and return accuracy %."""
    elo_kwargs = {k: v for k, v in params_dict.items() if k in _ELO_KEYS}
    model = NBAElo(**elo_kwargs)
    model._altitude_bonus = alt_bonus
    if not player_df.empty:
        model.set_player_stats(player_df)

    correct = 0
    total = 0
    for _, row in games.iterrows():
        try:
            game_date = row["_date_parsed"] if pd.notna(row["_date_parsed"]) else None
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


# --------------------------------------------------------------------------- #
#  Coordinate-descent optimizer
# --------------------------------------------------------------------------- #
def run_coordinate_descent(csv_file=GAMES_FILE):
    if not os.path.exists(csv_file):
        print("ERROR: %s not found" % csv_file)
        return

    # ---- pre-load data once ------------------------------------------------
    print("Loading data...")
    games = pd.read_csv(csv_file)
    if "neutral_site" not in games.columns:
        games["neutral_site"] = False
    games["_date_parsed"] = pd.to_datetime(games["date"], errors="coerce")

    player_df = load_player_stats()
    alt_bonus = _calc_altitude_bonus(csv_file)

    n_games = len(games)
    print("  %d games loaded" % n_games)

    # ---- current settings as starting point --------------------------------
    settings = load_elo_settings()
    current = {p: float(settings.get(p, PARAM_RANGES[p][0]))
               for p in PARAM_RANGES}

    # ---- baseline accuracy -------------------------------------------------
    print("\nComputing baseline accuracy...")
    test_settings = dict(settings)
    test_settings.update(current)
    baseline_acc = evaluate(test_settings, games, player_df, alt_bonus)
    print("  Baseline accuracy: %.2f%%\n" % baseline_acc)

    best_overall_acc = baseline_acc
    param_order = list(PARAM_RANGES.keys())
    total_evals = 0
    t_start = time.time()

    # ---- coordinate descent passes -----------------------------------------
    for pass_num in range(1, MAX_PASSES + 1):
        pass_start_acc = best_overall_acc
        any_improved = False

        print("=" * 64)
        print("  PASS %d / %d   (current best: %.2f%%)" % (
            pass_num, MAX_PASSES, best_overall_acc))
        print("=" * 64)

        for param in param_order:
            lo, hi, step = PARAM_RANGES[param]
            old_val = current[param]

            # --- coarse sweep -----------------------------------------------
            candidates = np.arange(lo, hi + step * 0.01, step)
            best_val = old_val
            best_acc = best_overall_acc

            print("  %-22s  sweeping %.1f .. %.1f  (step %.2f, %d pts)" % (
                param, lo, hi, step, len(candidates)))

            for val in candidates:
                current[param] = float(val)
                test_settings = dict(settings)
                test_settings.update(current)
                acc = evaluate(test_settings, games, player_df, alt_bonus)
                total_evals += 1
                if acc > best_acc:
                    best_acc = acc
                    best_val = float(val)

            # --- fine sweep around coarse best ------------------------------
            fine_step = step / 5.0
            fine_lo = max(lo, best_val - 2.0 * step)
            fine_hi = min(hi, best_val + 2.0 * step)
            fine_candidates = np.arange(fine_lo, fine_hi + fine_step * 0.01, fine_step)

            for val in fine_candidates:
                current[param] = float(val)
                test_settings = dict(settings)
                test_settings.update(current)
                acc = evaluate(test_settings, games, player_df, alt_bonus)
                total_evals += 1
                if acc > best_acc:
                    best_acc = acc
                    best_val = float(val)

            # --- apply best value for this param ----------------------------
            current[param] = best_val
            if best_acc > best_overall_acc:
                best_overall_acc = best_acc
                any_improved = True
                print("    >>> IMPROVED: %-18s = %-8.3f  acc = %.2f%%" % (
                    param, best_val, best_acc))
            else:
                # restore old value if no improvement
                current[param] = old_val
                print("    --- no gain:  %-18s  kept %.3f" % (param, old_val))

        # --- end-of-pass summary --------------------------------------------
        pass_delta = best_overall_acc - pass_start_acc
        elapsed = time.time() - t_start
        print("\n  Pass %d complete: %.2f%% -> %.2f%%  (delta: %+.3f%%)  "
              "[%d evals, %.0fs elapsed]\n" % (
                  pass_num, pass_start_acc, best_overall_acc,
                  pass_delta, total_evals, elapsed))

        if not any_improved:
            print("  No parameter improved this pass -- stopping early.\n")
            break

    # ---- save best settings ------------------------------------------------
    total_gain = best_overall_acc - baseline_acc
    elapsed = time.time() - t_start

    print("=" * 64)
    print("  COORDINATE DESCENT COMPLETE")
    print("=" * 64)
    print("  Baseline accuracy: %.2f%%" % baseline_acc)
    print("  Final accuracy:    %.2f%%  (%+.3f%%)" % (
        best_overall_acc, total_gain))
    print("  Total evaluations: %d" % total_evals)
    print("  Elapsed time:      %.1f s" % elapsed)
    print()
    print("  Best parameters:")
    for param in param_order:
        old = float(settings.get(param, PARAM_RANGES[param][0]))
        new = current[param]
        marker = " *" if abs(new - old) > 1e-6 else ""
        print("    %-22s = %-8.3f  (was %.3f)%s" % (param, new, old, marker))

    settings.update(current)
    save_elo_settings(settings)
    print("\n  Settings saved to nba_elo_settings.json")


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    run_coordinate_descent()
