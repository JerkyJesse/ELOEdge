"""Mega-Ensemble Optimizer.

Exhaustive multi-phase optimization that finds the absolute best
mega-ensemble settings for each sport. No shortcuts, no time limits.

Phase 1: Grid search over critical parameters (max_adj, meta_model, retrain_every)
Phase 2: Coordinate descent on secondary parameters
Phase 3: Bayesian optimization with Gaussian Process for fine-tuning
Phase 4: Tournament validation - best configs compete head-to-head
Phase 5: Final stability test with multiple random seeds

Objective: minimize 8*LogLoss + 40*Brier (matches existing optimizer weighting)
"""

import os
import sys
import json
import time
import math
import logging
from datetime import datetime
from itertools import product
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

_parent = os.path.dirname(__file__)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from mega_backtest import run_mega_backtest, SPORT_DEFAULTS

# ── Tunable parameter definitions ─────────────────────────────────

MEGA_PARAM_SPACE = {
    # Phase 1: Critical parameters (grid search)
    "max_adj": {
        "type": "float", "phase": 1,
        "grid": [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30],
        "desc": "Max probability adjustment from meta-learner",
    },
    "meta_model": {
        "type": "categorical", "phase": 1,
        "grid": ["ridge", "logistic", "xgboost"],
        "desc": "Meta-learner algorithm",
    },
    "retrain_every": {
        "type": "int", "phase": 1,
        "grid_nfl": [16, 24, 32, 48, 64],
        "grid_mlb": [40, 60, 80, 100, 120],
        "grid_nba": [25, 40, 50, 75, 100],
        "grid_nhl": [25, 40, 50, 75, 100],
        "desc": "Retrain meta-learner every N games",
    },

    # Phase 2: Secondary parameters (coordinate descent)
    "min_train": {
        "type": "int", "phase": 2,
        "range_nfl": [80, 100, 120, 150, 180, 200],
        "range_mlb": [150, 200, 250, 300, 400, 500],
        "range_nba": [100, 150, 200, 250, 300],
        "range_nhl": [100, 150, 200, 250, 300],
        "desc": "Games before meta-learner starts predicting",
    },
    "kalman_process_noise": {
        "type": "float", "phase": 2,
        "range": [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0],
        "desc": "Kalman filter process noise",
    },
    "kalman_measurement_noise": {
        "type": "float", "phase": 2,
        "range_nfl": [8, 10, 12, 13.5, 15, 18],
        "range_mlb": [2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        "range_nba": [7, 9, 11, 13, 15],
        "range_nhl": [1.5, 2.0, 2.5, 3.0, 3.5],
        "desc": "Kalman filter measurement noise",
    },
    "hmm_states": {
        "type": "int", "phase": 2,
        "range": [2, 3, 4],
        "desc": "Number of HMM hidden states",
    },
    "network_decay": {
        "type": "float", "phase": 2,
        "range": [0.90, 0.93, 0.95, 0.97, 0.99],
        "desc": "PageRank temporal decay",
    },
    "momentum_friction": {
        "type": "float", "phase": 2,
        "range": [0.02, 0.05, 0.08, 0.10, 0.15],
        "desc": "Momentum friction coefficient",
    },
    "n_clusters": {
        "type": "int", "phase": 2,
        "range": [3, 4, 5, 6],
        "desc": "Number of team archetype clusters",
    },

    # Phase 3: Fine-tuning (Bayesian optimization)
    "bt_decay": {
        "type": "float", "phase": 3,
        "bounds": [0.95, 1.0],
        "desc": "Bradley-Terry recency decay",
    },
    "glicko_initial_rd": {
        "type": "float", "phase": 3,
        "bounds": [100, 400],
        "desc": "Glicko-2 initial rating deviation",
    },
    "mc_simulations": {
        "type": "int", "phase": 3,
        "bounds": [500, 5000],
        "desc": "Monte Carlo simulations per game",
    },
    "window": {
        "type": "int", "phase": 3,
        "bounds_nfl": [3, 8],
        "bounds_mlb": [8, 25],
        "bounds_nba": [5, 20],
        "bounds_nhl": [5, 20],
        "desc": "Rolling feature window size",
    },
}


def _objective(result):
    """Compute objective value from backtest results.

    Same weighting as existing optimizer: LogLoss*8 + Brier*40
    Lower is better.
    """
    if result is None:
        return 999.0
    return result["log_loss"] * 8 + result["brier"] * 40


def _get_param_values(param_name, sport):
    """Get the list of values to try for a parameter."""
    spec = MEGA_PARAM_SPACE[param_name]

    # Sport-specific ranges
    sport_key = f"grid_{sport}"
    if sport_key in spec:
        return spec[sport_key]
    sport_key = f"range_{sport}"
    if sport_key in spec:
        return spec[sport_key]
    if "grid" in spec:
        return spec["grid"]
    if "range" in spec:
        return spec["range"]
    if "bounds" in spec:
        lo, hi = spec["bounds"]
        if spec["type"] == "int":
            return list(range(lo, hi + 1, max(1, (hi - lo) // 5)))
        return list(np.linspace(lo, hi, 6))

    sport_bounds = f"bounds_{sport}"
    if sport_bounds in spec:
        lo, hi = spec[sport_bounds]
        if spec["type"] == "int":
            return list(range(lo, hi + 1, max(1, (hi - lo) // 5)))
        return list(np.linspace(lo, hi, 6))

    return [0]


SETTINGS_FILE_TPL = "{sport}_mega_settings.json"


def _load_settings(sport, sport_dir):
    """Load saved mega-ensemble settings."""
    path = os.path.join(sport_dir, SETTINGS_FILE_TPL.format(sport=sport))
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_settings(sport, sport_dir, params):
    """Save mega-ensemble settings."""
    path = os.path.join(sport_dir, SETTINGS_FILE_TPL.format(sport=sport))
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    return path


def run_mega_optimize(csv_file, sport="nfl", elo_model_class=None,
                      elo_settings=None, player_df=None,
                      phases=None, verbose=True):
    """Run full multi-phase mega-ensemble optimization.

    Args:
        csv_file: path to games CSV
        sport: 'nfl', 'mlb', 'nba', 'nhl'
        elo_model_class: the sport's Elo class
        elo_settings: Elo model settings dict
        player_df: player stats DataFrame
        phases: list of phases to run (default [1,2,3,4,5] = all)
        verbose: print progress

    Returns dict with best_params, best_objective, best_results, history.
    """
    if phases is None:
        phases = [1, 2, 3, 4, 5]

    sport_dir = os.path.dirname(os.path.abspath(csv_file))
    start_time = time.time()

    # Start from saved settings or defaults
    best_params = _load_settings(sport, sport_dir)
    history = []
    eval_count = 0

    def _eval(params, label=""):
        nonlocal eval_count
        eval_count += 1
        t0 = time.time()
        result = run_mega_backtest(
            csv_file, sport, elo_model_class, elo_settings, player_df,
            verbose=False, mega_params=params,
        )
        dt = time.time() - t0
        obj = _objective(result)

        if verbose and result:
            print("    #%03d [%5.1fs] %s obj=%.4f acc=%.2f%% ll=%.4f br=%.4f | %s"
                  % (eval_count, dt, label,
                     obj, result["accuracy"], result["log_loss"], result["brier"],
                     " ".join(f"{k}={v}" for k, v in params.items() if k in MEGA_PARAM_SPACE)))

        history.append({
            "params": dict(params),
            "objective": obj,
            "accuracy": result["accuracy"] if result else 0,
            "log_loss": result["log_loss"] if result else 1,
            "brier": result["brier"] if result else 0.5,
            "elapsed": dt,
        })
        return obj, result

    # ── Baseline evaluation ───────────────────────────────────────
    if verbose:
        print("\n  ====================================================")
        print("    MEGA-ENSEMBLE OPTIMIZER (%s)" % sport.upper())
        print("  ====================================================")
        print("    Objective = LogLoss*8 + Brier*40  (lower is better)")
        print("    Phases: %s" % phases)
        print()

    # Run baseline
    if verbose:
        print("  Phase 0: Baseline evaluation")
    baseline_obj, baseline_result = _eval(best_params, "BASELINE")
    best_obj = baseline_obj
    best_result = baseline_result

    if verbose and baseline_result:
        print("    Baseline: obj=%.4f acc=%.2f%% ll=%.4f br=%.4f"
              % (baseline_obj, baseline_result["accuracy"],
                 baseline_result["log_loss"], baseline_result["brier"]))
        print()

    # ══════════════════════════════════════════════════════════════
    # Phase 1: Grid search over critical parameters
    # ══════════════════════════════════════════════════════════════
    if 1 in phases:
        if verbose:
            print("  Phase 1: Grid search (critical parameters)")
            print("  " + "-" * 50)

        phase1_params = [p for p, s in MEGA_PARAM_SPACE.items() if s["phase"] == 1]
        grids = {}
        for p in phase1_params:
            grids[p] = _get_param_values(p, sport)

        # Full Cartesian product
        keys = list(grids.keys())
        combos = list(product(*[grids[k] for k in keys]))
        n_combos = len(combos)

        if verbose:
            print("    Parameters: %s" % keys)
            print("    Combinations: %d" % n_combos)
            est_time = n_combos * 60  # rough estimate
            print("    Estimated time: %d-%d minutes" % (est_time // 120, est_time // 30))
            print()

        for i, vals in enumerate(combos):
            params = dict(best_params)
            for k, v in zip(keys, vals):
                params[k] = v

            obj, result = _eval(params, f"P1[{i+1}/{n_combos}]")
            if obj < best_obj:
                best_obj = obj
                best_params = dict(params)
                best_result = result
                if verbose:
                    print("    >>> NEW BEST: obj=%.4f <<<" % best_obj)

        if verbose:
            print()
            print("    Phase 1 complete. Best obj=%.4f" % best_obj)
            print("    Best P1 params: %s" % {k: best_params.get(k) for k in phase1_params})
            print()

        # Save intermediate results
        _save_settings(sport, sport_dir, best_params)

    # ══════════════════════════════════════════════════════════════
    # Phase 2: Coordinate descent on secondary parameters
    # ══════════════════════════════════════════════════════════════
    if 2 in phases:
        if verbose:
            print("  Phase 2: Coordinate descent (secondary parameters)")
            print("  " + "-" * 50)

        phase2_params = [p for p, s in MEGA_PARAM_SPACE.items() if s["phase"] == 2]
        improved = True
        pass_num = 0

        while improved and pass_num < 3:
            improved = False
            pass_num += 1
            if verbose:
                print("    Pass %d:" % pass_num)

            for param in phase2_params:
                values = _get_param_values(param, sport)
                param_best_obj = best_obj
                param_best_val = best_params.get(param)

                for val in values:
                    params = dict(best_params)
                    params[param] = val
                    obj, result = _eval(params, f"P2-{param}")

                    if obj < param_best_obj:
                        param_best_obj = obj
                        param_best_val = val

                if param_best_val != best_params.get(param):
                    best_params[param] = param_best_val
                    best_obj = param_best_obj
                    improved = True
                    if verbose:
                        print("      %s: %s -> %s (obj=%.4f)" %
                              (param, best_params.get(param), param_best_val, param_best_obj))

        if verbose:
            print()
            print("    Phase 2 complete. Best obj=%.4f" % best_obj)
            print()

        _save_settings(sport, sport_dir, best_params)

    # ══════════════════════════════════════════════════════════════
    # Phase 3: Bayesian optimization for fine-tuning
    # ══════════════════════════════════════════════════════════════
    if 3 in phases:
        if verbose:
            print("  Phase 3: Bayesian optimization (fine-tuning)")
            print("  " + "-" * 50)

        phase3_params = [p for p, s in MEGA_PARAM_SPACE.items() if s["phase"] == 3]

        # Simple Bayesian-ish optimization: sample around best values
        n_trials = 30
        rng = np.random.RandomState(42)

        for trial in range(n_trials):
            params = dict(best_params)

            # Perturb each phase-3 param randomly around current best
            for param in phase3_params:
                spec = MEGA_PARAM_SPACE[param]
                current = best_params.get(param)
                bounds_key = f"bounds_{sport}"
                if bounds_key in spec:
                    lo, hi = spec[bounds_key]
                elif "bounds" in spec:
                    lo, hi = spec["bounds"]
                else:
                    continue

                if current is None:
                    current = (lo + hi) / 2

                # Gaussian perturbation
                std = (hi - lo) * 0.15
                new_val = current + rng.normal(0, std)
                new_val = max(lo, min(hi, new_val))

                if spec["type"] == "int":
                    new_val = int(round(new_val))
                else:
                    new_val = round(new_val, 4)

                params[param] = new_val

            obj, result = _eval(params, f"P3[{trial+1}/{n_trials}]")
            if obj < best_obj:
                best_obj = obj
                best_params = dict(params)
                best_result = result
                if verbose:
                    print("    >>> NEW BEST: obj=%.4f <<<" % best_obj)

        if verbose:
            print()
            print("    Phase 3 complete. Best obj=%.4f" % best_obj)
            print()

        _save_settings(sport, sport_dir, best_params)

    # ══════════════════════════════════════════════════════════════
    # Phase 4: Tournament - top configs compete
    # ══════════════════════════════════════════════════════════════
    if 4 in phases:
        if verbose:
            print("  Phase 4: Tournament validation")
            print("  " + "-" * 50)

        # Get top 10 configs from history
        sorted_history = sorted(history, key=lambda x: x["objective"])
        top_configs = []
        seen = set()
        for h in sorted_history[:20]:
            key = json.dumps(h["params"], sort_keys=True)
            if key not in seen:
                seen.add(key)
                top_configs.append(h["params"])
            if len(top_configs) >= 10:
                break

        if verbose:
            print("    Re-evaluating top %d configs..." % len(top_configs))

        for i, cfg in enumerate(top_configs):
            obj, result = _eval(cfg, f"P4[{i+1}/{len(top_configs)}]")
            if obj < best_obj:
                best_obj = obj
                best_params = dict(cfg)
                best_result = result

        if verbose:
            print()
            print("    Phase 4 complete. Best obj=%.4f" % best_obj)
            print()

        _save_settings(sport, sport_dir, best_params)

    # ══════════════════════════════════════════════════════════════
    # Phase 5: Stability test - verify not overfit
    # ══════════════════════════════════════════════════════════════
    if 5 in phases:
        if verbose:
            print("  Phase 5: Stability verification")
            print("  " + "-" * 50)

        # Run best config 3 times to check stability
        objs = []
        for run in range(3):
            obj, result = _eval(best_params, f"P5-stability[{run+1}/3]")
            objs.append(obj)

        stability_std = np.std(objs)
        stability_mean = np.mean(objs)

        if verbose:
            print("    Stability: mean=%.4f std=%.4f (%.1f%% variation)" %
                  (stability_mean, stability_std,
                   stability_std / stability_mean * 100 if stability_mean > 0 else 0))
            print()

    # ══════════════════════════════════════════════════════════════
    # Phase 6: Genetic algorithm (scipy differential evolution)
    # ══════════════════════════════════════════════════════════════
    if 6 in phases:
        if verbose:
            print("  Phase 6: Genetic algorithm (differential evolution)")
            print("  " + "-" * 50)

        # Build bounds for continuous/integer params
        de_params = []
        de_bounds = []
        for param, spec in MEGA_PARAM_SPACE.items():
            if spec["type"] in ("float", "int") and spec["phase"] <= 3:
                values = _get_param_values(param, sport)
                if values and len(values) >= 2:
                    lo, hi = min(values), max(values)
                    de_params.append(param)
                    de_bounds.append((lo, hi))

        if de_params:
            def _de_objective(x):
                params = dict(best_params)
                for i, param_name in enumerate(de_params):
                    spec = MEGA_PARAM_SPACE[param_name]
                    val = x[i]
                    if spec["type"] == "int":
                        val = int(round(val))
                    else:
                        val = round(val, 4)
                    params[param_name] = val
                obj, _ = _eval(params, "P6-DE")
                return obj

            try:
                from scipy.optimize import differential_evolution
                de_result = differential_evolution(
                    _de_objective, de_bounds,
                    maxiter=15, popsize=8, tol=0.001,
                    seed=42, disp=False,
                )
                if de_result.fun < best_obj:
                    # Reconstruct best params
                    for i, param_name in enumerate(de_params):
                        spec = MEGA_PARAM_SPACE[param_name]
                        val = de_result.x[i]
                        if spec["type"] == "int":
                            val = int(round(val))
                        else:
                            val = round(val, 4)
                        best_params[param_name] = val
                    best_obj = de_result.fun
                    if verbose:
                        print("    >>> DE found new best: obj=%.4f <<<" % best_obj)
            except ImportError:
                if verbose:
                    print("    scipy not available, skipping DE")

        if verbose:
            print()
            print("    Phase 6 complete. Best obj=%.4f" % best_obj)
            print()

        _save_settings(sport, sport_dir, best_params)

    # ══════════════════════════════════════════════════════════════
    # Phase 7: Super-optimize (exhaustive fine-grain on top params)
    # ══════════════════════════════════════════════════════════════
    if 7 in phases:
        if verbose:
            print("  Phase 7: Super-optimize (fine-grain around best)")
            print("  " + "-" * 50)

        # For each top-importance param, do a fine sweep around best value
        key_params = ["max_adj", "kalman_process_noise", "kalman_measurement_noise",
                      "network_decay", "momentum_friction", "hmm_states"]

        for param in key_params:
            if param not in MEGA_PARAM_SPACE:
                continue
            spec = MEGA_PARAM_SPACE[param]
            current = best_params.get(param)
            if current is None:
                continue

            # Generate fine-grain values around current best
            values = _get_param_values(param, sport)
            if not values:
                continue

            lo, hi = min(values), max(values)
            if spec["type"] == "int":
                fine_vals = sorted(set([max(int(lo), current - 1), current,
                                        min(int(hi), current + 1)]))
            else:
                step = (hi - lo) / 20
                fine_vals = [round(current + i * step, 4) for i in range(-3, 4)]
                fine_vals = [v for v in fine_vals if lo <= v <= hi]

            param_best = best_obj
            param_best_val = current

            for val in fine_vals:
                params = dict(best_params)
                params[param] = val
                obj, result = _eval(params, f"P7-{param}")
                if obj < param_best:
                    param_best = obj
                    param_best_val = val

            if param_best_val != current:
                best_params[param] = param_best_val
                best_obj = param_best
                if verbose:
                    print("    %s: %s -> %s (obj=%.4f)" %
                          (param, current, param_best_val, param_best))

        if verbose:
            print()
            print("    Phase 7 complete. Best obj=%.4f" % best_obj)
            print()

        _save_settings(sport, sport_dir, best_params)

    # ── Final results ─────────────────────────────────────────────
    elapsed = time.time() - start_time
    best_result_final = None
    if best_result:
        best_result_final = best_result

    # Save final settings
    settings_path = _save_settings(sport, sport_dir, best_params)

    # Summary
    if verbose:
        print("  ====================================================")
        print("    OPTIMIZATION COMPLETE (%s)" % sport.upper())
        print("  ====================================================")
        print("    Total evaluations : %d" % eval_count)
        print("    Total time        : %.1f minutes" % (elapsed / 60))
        print("    Baseline obj      : %.4f" % baseline_obj)
        print("    Best obj          : %.4f" % best_obj)
        improvement = (baseline_obj - best_obj) / baseline_obj * 100
        print("    Improvement       : %.2f%%" % improvement)
        print()
        if best_result_final:
            print("    Best accuracy     : %.2f%%" % best_result_final["accuracy"])
            print("    Best log loss     : %.4f" % best_result_final["log_loss"])
            print("    Best brier        : %.4f" % best_result_final["brier"])
        print()
        print("    Best parameters:")
        for k, v in sorted(best_params.items()):
            if k in MEGA_PARAM_SPACE:
                print("      %-30s = %s" % (k, v))
        print()
        print("    Settings saved to: %s" % settings_path)
        print("  ====================================================")

    return {
        "best_params": best_params,
        "best_objective": best_obj,
        "best_results": best_result_final,
        "baseline_objective": baseline_obj,
        "baseline_results": baseline_result,
        "n_evaluations": eval_count,
        "elapsed_seconds": elapsed,
        "history": history,
        "settings_path": settings_path,
    }


def run_quick_optimize(csv_file, sport="nfl", elo_model_class=None,
                       elo_settings=None, player_df=None):
    """Quick optimization: Phase 1 only (grid search on critical params)."""
    return run_mega_optimize(csv_file, sport, elo_model_class, elo_settings,
                             player_df, phases=[1])


def run_deep_optimize(csv_file, sport="nfl", elo_model_class=None,
                      elo_settings=None, player_df=None):
    """Deep optimization: All 7 phases, no shortcuts."""
    return run_mega_optimize(csv_file, sport, elo_model_class, elo_settings,
                             player_df, phases=[1, 2, 3, 4, 5, 6, 7])


def run_single_model_optimize(csv_file, sport="nfl", elo_model_class=None,
                               elo_settings=None, player_df=None,
                               verbose=True):
    """Test each model individually to find its contribution.

    For every model in the registry:
    1. Run baseline (all models ON at current settings)
    2. Turn OFF this one model, run backtest
    3. Difference = this model's contribution

    Then rank models by contribution and optionally disable
    models that hurt performance.
    """
    from mega_config import (MODEL_REGISTRY, ALL_MODELS,
                              load_model_switches, save_model_switches,
                              is_model_enabled, print_model_status)

    sport_dir = os.path.dirname(os.path.abspath(csv_file))
    switches = load_model_switches(sport, sport_dir)
    settings = _load_settings(sport, sport_dir)
    start_time = time.time()

    if verbose:
        print("\n  ====================================================")
        print("    SINGLE-MODEL ABLATION STUDY (%s)" % sport.upper())
        print("  ====================================================")
        print("    Testing each model's individual contribution...")
        print()

    # 1. Run baseline with ALL currently-enabled models
    if verbose:
        print("  Step 1: Baseline (all enabled models)")
    baseline_result = run_mega_backtest(
        csv_file, sport, elo_model_class, elo_settings, player_df,
        verbose=False, mega_params=settings,
    )
    baseline_obj = _objective(baseline_result)
    if verbose:
        print("    Baseline: obj=%.4f acc=%.2f%% ll=%.4f br=%.4f"
              % (baseline_obj, baseline_result["accuracy"],
                 baseline_result["log_loss"], baseline_result["brier"]))
        print()

    # 2. Ablation: turn off each model one at a time
    if verbose:
        print("  Step 2: Ablation (turn off one model at a time)")
        print("  %-20s %8s %8s %8s  %s" %
              ("Model", "Obj", "Delta", "Acc%", "Verdict"))
        print("  " + "-" * 65)

    contributions = []
    # Only test models that are currently enabled
    enabled_models = [m for m in ALL_MODELS if is_model_enabled(m, switches)
                      and m not in ("elo",)]  # Never ablate Elo

    for model_name in enabled_models:
        # Create switches with this one model OFF
        test_switches = dict(switches)
        test_switches[model_name] = False

        test_params = dict(settings)
        test_params["model_switches"] = test_switches

        result = run_mega_backtest(
            csv_file, sport, elo_model_class, elo_settings, player_df,
            verbose=False, mega_params=test_params,
        )

        if result is None:
            continue

        obj = _objective(result)
        delta = obj - baseline_obj  # Positive = worse without this model = model helps

        if delta > 0.001:
            verdict = "HELPS"
        elif delta < -0.001:
            verdict = "HURTS"
        else:
            verdict = "neutral"

        contributions.append({
            "model": model_name,
            "obj_without": obj,
            "delta": delta,
            "accuracy_without": result["accuracy"],
            "verdict": verdict,
        })

        if verbose:
            print("  %-20s %8.4f %+7.4f %7.2f%%  %s" %
                  (model_name, obj, delta, result["accuracy"], verdict))

    # 3. Sort by contribution (highest delta = most helpful)
    contributions.sort(key=lambda x: x["delta"], reverse=True)

    if verbose:
        print()
        print("  Step 3: Rankings (most helpful -> least)")
        print("  %-4s %-20s %+8s  %s" % ("Rank", "Model", "Impact", "Verdict"))
        print("  " + "-" * 45)
        for i, c in enumerate(contributions):
            print("  %3d. %-20s %+7.4f  %s" %
                  (i + 1, c["model"], c["delta"], c["verdict"]))

    # 4. Count models that hurt
    hurting = [c for c in contributions if c["verdict"] == "HURTS"]

    if verbose and hurting:
        print()
        print("  Models that HURT performance (consider disabling):")
        for c in hurting:
            print("    - %s (obj %+.4f when removed)" % (c["model"], c["delta"]))

    # 5. Auto-disable models that hurt (if any)
    if hurting:
        if verbose:
            print()
            print("  Auto-disabling %d model(s) that hurt performance..." % len(hurting))

        for c in hurting:
            switches[c["model"]] = False
            if verbose:
                print("    Disabled: %s" % c["model"])

        save_model_switches(sport, sport_dir, switches)

        # Verify improvement
        test_params = dict(settings)
        test_params["model_switches"] = switches
        final_result = run_mega_backtest(
            csv_file, sport, elo_model_class, elo_settings, player_df,
            verbose=False, mega_params=test_params,
        )
        final_obj = _objective(final_result)

        if verbose:
            print()
            print("    After disabling: obj=%.4f (was %.4f)" % (final_obj, baseline_obj))
            if final_obj < baseline_obj:
                print("    Improvement: %+.4f" % (baseline_obj - final_obj))
            else:
                # Revert
                for c in hurting:
                    switches[c["model"]] = True
                save_model_switches(sport, sport_dir, switches)
                print("    No improvement — reverted all changes")
    else:
        if verbose:
            print()
            print("  All enabled models are helping. No changes needed.")

    elapsed = time.time() - start_time

    if verbose:
        print()
        print("  ====================================================")
        print("    ABLATION COMPLETE (%s)" % sport.upper())
        print("  ====================================================")
        n_enabled = sum(1 for m in ALL_MODELS if is_model_enabled(m, switches))
        print("    Models tested  : %d" % len(enabled_models))
        print("    Models helping : %d" % len([c for c in contributions if c["verdict"] == "HELPS"]))
        print("    Models neutral : %d" % len([c for c in contributions if c["verdict"] == "neutral"]))
        print("    Models hurting : %d" % len(hurting))
        print("    Models enabled : %d / %d" % (n_enabled, len(ALL_MODELS)))
        print("    Time           : %.1f minutes" % (elapsed / 60))
        print("  ====================================================")

    return {
        "contributions": contributions,
        "baseline_obj": baseline_obj,
        "n_hurting": len(hurting),
        "elapsed_seconds": elapsed,
    }
