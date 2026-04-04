# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

NFL game prediction system combining Elo ratings with an XGBoost ensemble, with Predicts $1 contract trading ledger. Interactive CLI app -- no web server, no tests framework, no build system.

## Running

```bash
pip install -r requirements.txt
python main.py
```

First run auto-downloads game data (ESPN API), player stats, and injury reports (ESPN). Then runs a backtest to fit the Platt calibration scaler and enters the interactive CLI loop.

## Architecture

**Two-stage prediction pipeline:**
1. **Elo model** (`elo_model.py` -> `NFLElo` class) -- base team ratings adjusted for home field, altitude (Denver only at 5280ft), player strength, rest days (centered at 7 days, not 1), travel fatigue, pace mismatch, injuries, and strength of schedule
2. **XGBoost ensemble** (`enhanced_model.py`) -- 80% Elo / 20% XGBoost using 31 rolling features per team (5-game window via `TeamTracker`, includes Pythagorean win expectation, streaks, consistency, and trend)
3. **Platt calibration** (`platt.py`) -- logistic regression on raw probabilities for well-calibrated outputs

**Data flow:**
- `data_games.py` / `data_players.py` -> download from ESPN public API with 6-hour cache (`config.is_cache_stale`)
- `injuries.py` -> ESPN JSON API with 4-hour cache; position-based impact (QB out = -50 Elo)
- `build_model.py` -> constructs `NFLElo` from settings + game CSV, applies season regression, sets player scores
- `backtest.py` -> walk-forward backtest, also houses grid search and genetic (`scipy.optimize.differential_evolution`) optimizer
- `main.py` -> CLI entry point, `dispatch()` routes all commands, team name input triggers prediction flow

**NFL-specific design decisions:**
- K-factor = 35 (fewer games per season than NBA, each game matters more)
- Home advantage = 48 Elo (~57% NFL home win rate)
- Rest factor = 40, centered at 7 days (NFL teams play weekly, rest is huge)
- MOV formula: `log(max(1, abs(margin)) + 1)` (NFL-specific)
- Season spans two calendar years: `season = year if month >= 9 else year - 1`
- Only Denver Broncos get altitude bonus (5280ft)

**State files (all gitignored, generated at runtime):**
- `nfl_elo_settings.json` -- tunable parameters (K-factor, home advantage, etc.)
- `nfl_elo_ratings.json` -- current team Elo ratings
- `nfl_platt_scaler.json` / `nfl_isotonic_scaler.json` -- calibration scalers
- `nfl_enhanced_model.json` -- saved XGBoost model weights
- `predicts_lots.csv` -- trade ledger (positions, P&L)

**Key patterns:**
- All settings load/save through `config.py` (`load_elo_settings` / `save_elo_settings`)
- Color output uses wrapper functions in `color_helpers.py` (`cok`, `cerr`, `cwarn`, `chi`, `cdim`, `cbold`), not raw colorama
- The `NFLElo` class stores per-team state in dicts/defaultdicts on the instance (ratings, last game dates, recent results, player scores, etc.)
- Optimizer objective is `LogLoss * 8 + Brier * 40` -- this weighting is intentional
- `backtest_model(..., fit_platt=True)` should only be called for user-facing runs, never inside optimizer loops (leakage + speed)

## Backtesting & Optimization

### Walk-Forward Backtest (`backtest` command)

`backtest_model()` in `backtest.py` iterates through every game in `nfl_recent_games.csv` chronologically. For each game it:
1. Calls `model.win_prob()` with `calibrated=False` and `use_injuries=False` (raw Elo only -- no future leakage)
2. Records the prediction, then calls `model.update_game()` to update ratings *after* predicting
3. Outputs `nfl_backtest_predictions.csv` (per-game predictions) and `nfl_calibration.csv` (10-bin calibration table)

When `fit_platt=True` (user-facing runs only), it fits a Platt scaler on the full run's raw probabilities and saves to `nfl_platt_scaler.json`. This flag must **never** be used inside optimizer objective functions -- it causes data leakage and slows iteration.

Metrics reported: accuracy (%), log loss, Brier score. Calibration table bins predictions into 10 probability buckets and compares predicted vs actual win rates.

### XGBoost Enhanced Backtest (`enhanced` command)

`run_enhanced_backtest()` in `enhanced_model.py` is a separate walk-forward loop that builds an 80/20 Elo+XGBoost ensemble:
- First `min_train` games (default 200): Elo-only predictions while accumulating training features
- After that: XGBoost is trained on accumulated features and retrained every `retrain_every` games (default 50)
- `TeamTracker` maintains rolling 5-game windows (PPG, PAPG, win%, margins, rest days) per team
- Feature vector has 31 columns: elo_prob, elo_diff, player_diff, per-team rolling stats, differentials, rest, Pythagorean win expectation, streaks, consistency, and trend
- XGBoost params: `max_depth=5, eta=0.03, subsample=0.9, colsample_bytree=0.8, min_child_weight=3, 300 rounds`
- After the walk-forward, fits both Platt and isotonic calibrators on ensemble probabilities
- Saves the trained XGBoost booster to `nfl_xgb_model.json` and metadata to `nfl_enhanced_model.json`

### Grid Search (`grid` command)

`grid_search_optimization()` searches 7 parameters via Cartesian product: K, HomeAdv, PlayerBoost, RestFactor, TravelFactor, PaceFactor, PlayoffHCA. Each combo builds a fresh `NFLElo`, runs `backtest_model()` (without Platt fitting), and scores with:

```
score = -(LogLoss * 8 + Brier * 40)
```

Results saved to `nfl_grid_search.csv`. After completion, `_apply_best_settings()` saves the winning params and refits the Platt scaler.

### Genetic Optimizer (`genetic` command)

`genetic_optimization()` uses `scipy.optimize.differential_evolution` over the same 7 parameters. Same objective function as grid search. After convergence, saves settings and refits Platt.

### Bayesian Optimization (`bayesian` command)

GP surrogate + Expected Improvement acquisition. Finds optima in ~50-100 evaluations. Uses Latin Hypercube initial sampling, minimal Gaussian Process (squared exponential kernel), and EI acquisition on 1000 random candidates per iteration.

## Advanced Backtesting Methods

All 16 advanced backtesting techniques are implemented in `backtest.py`, `enhanced_model.py`, `platt.py`, and `metrics.py`. No additional dependencies beyond the existing stack (numpy, scipy, pandas, xgboost).

### Metrics (auto-reported by `backtest` command)

- **ECE / MCE** (`metrics.py`) -- Expected and Maximum Calibration Error
- **BSS** (`metrics.py`) -- Brier Skill Score vs 50% baseline and vs home-win-rate baseline
- **DSR** (`backtest.py`) -- Deflated Sharpe Ratio, adjusts for multiple-testing bias

### Cross-Validation (`purgedcv`, `cpcv` commands)

- **Purged Walk-Forward CV** -- k chronological folds with embargo gap
- **CPCV** -- All C(k, k_test) train/test combinations

### Overfitting Detection (`pbo`, `montecarlo` commands)

- **PBO** -- Symmetric cross-validation on grid search trial population
- **Monte Carlo Permutation** -- Shuffles outcomes, re-runs backtest N times, computes p-value

### Calibration (`rollingcal`, `betacal`, `conformal` commands)

- **Rolling Origin Recalibration** -- Expanding-window Platt fitting every 50 games
- **Beta Calibration** -- 3-parameter asymmetric calibration
- **Conformal Prediction** -- Distribution-free prediction sets at 90%, 95%, 80% coverage

### Ensemble Analysis (`enhanced decay`, `shap` commands)

- **Time-Decayed Weighting** -- Elo weight transitions 95% -> 70% over the season
- **SHAP Feature Importance** -- XGBoost native `pred_contribs`

### Structural Analysis (`sliding`, `convergence` commands)

- **Sliding Window** -- Heavy regression every N games vs standard expanding window
- **Convergence Analysis** -- Per-chunk accuracy to find burn-in period

### Position Sizing (`kelly` command)

**Kelly Criterion Backtest** -- Simulates fractional Kelly position sizing on `nfl_backtest_predictions.csv`. Default quarter-Kelly.

## Complete Model Validation Workflow (run in order)

**Phase 1: Baseline & Diagnostics**
1. `backtest` -- Establish baseline metrics
2. `convergence` -- Find burn-in period
3. `sliding` -- Check if old data helps or hurts

**Phase 2: Parameter Optimization**
4. `grid` -- Coarse search
5. `pbo` -- Check overfitting
6. `results` -- Check DSR significance
7. `genetic` or `bayesian` -- Fine-tune

**Phase 3: Cross-Validation**
8. `purgedcv` -- Fold stability
9. `cpcv` -- Combinatorial robustness
10. `montecarlo` -- Statistical significance

**Phase 4: Ensemble & Features**
11. `enhanced` -- Train XGBoost ensemble (SHAP auto-runs)
12. `shap` -- Standalone feature importance
13. `enhanced decay` -- Compare time-decayed vs fixed

**Phase 5: Calibration Refinement**
14. `rollingcal` -- Out-of-sample calibration
15. `betacal` -- Asymmetric calibration
16. `conformal` -- Coverage analysis

**Phase 6: P&L Simulation**
17. `kelly` -- Position sizing backtest

## Conventions

- Python 3.8+ compatibility (no walrus operators, no `match` statements)
- No test framework -- validation is via `backtest` command and `accuracy_test.py`
- All data files use `nfl_` prefix; temp files use `temp_` prefix
- Settings tuned via `set param=value` CLI command or optimizer; changes require `backtest` rerun
- Color output uses `color_helpers.py` wrappers, never raw colorama
- Optimizer objective is `LogLoss * 8 + Brier * 40` -- intentional, do not change casually
- Team lookup is fuzzy: `NFLElo.find_team()` accepts full names, abbreviations, partial matches
- NFL season detection: month >= 9 means current year, month <= 8 means previous year's season
- All 32 NFL teams defined in `config.py` TEAM_ABBR dict
