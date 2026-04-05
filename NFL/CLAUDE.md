# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

NFL game prediction system combining Elo ratings with a 35-model mega-ensemble, with Predicts $1 contract trading ledger. Interactive CLI app -- no web server, no test framework, no build system.

## Running

```bash
pip install -r requirements.txt
python main.py
```

First run auto-downloads game data (ESPN API), player stats, and injury reports (ESPN). Then runs a backtest to fit the Platt calibration scaler and enters the interactive CLI loop.

## Architecture

**Three-stage prediction pipeline:**
1. **Elo model** (`elo_model.py` -> `NFLElo` class) -- base team ratings adjusted for home field, altitude (Denver only at 5280ft), player strength, rest days (centered at 7 days), travel fatigue, pace mismatch, injuries, division rivalry, and strength of schedule
2. **XGBoost ensemble** (`enhanced_model.py`) -- 80% Elo / 20% XGBoost (default `elo_weight=0.8`) using 31 rolling features per team (5-game window via `TeamTracker`, includes Pythagorean win expectation, streaks, consistency, and trend)
3. **Mega-ensemble** (`mega_predictor.py` + `mega_backtest.py`) -- 35 base models stacked via a meta-learner (XGBoost, ridge, or logistic). Produces a bounded adjustment (+/- max_adj, default 0.10) on top of the Elo+XGBoost probability. Models span 7 tiers: Core (Elo, XGBoost), Proven (HMM, Kalman, PageRank, LightGBM, CatBoost, MLP, LSTM), Exotic (GARCH, Fourier/Wavelet, Survival, Copula), Info/Physics (Shannon Entropy, Momentum, Markov Chain, Clustering, Game Theory), Classical Ratings (Poisson, Glicko-2, Bradley-Terry, Monte Carlo, Random Forest), Sports-Specific (SRS, Colley Matrix, Log5, Pythagorean, Exponential Smoothing, Mean Reversion), Additional (SVM, Fibonacci, EVT, Benford), and Data Enrichment (Weather, Odds).
4. **Platt calibration** (`platt.py`) -- logistic regression on raw probabilities for well-calibrated outputs

**Data flow:**
- `data_games.py` / `data_players.py` -> download from ESPN public API with 6-hour cache (`config.is_cache_stale`)
- `advanced_stats.py` -> EPA, CPOE, success rate via nfl_data_py (nflverse play-by-play); referee tendencies; cached to `nfl_epa_stats.csv`
- `injuries.py` -> ESPN JSON API with 4-hour cache; position-based impact (QB out = -50 Elo)
- `weather.py` -> Open-Meteo API (free, no key) with 2-hour cache for outdoor games
- `odds_tracker.py` -> The Odds API (free tier, 500 req/month) for moneyline odds + CLV
- `kalshi.py` -> Kalshi public API for live contract prices
- `build_model.py` -> constructs `NFLElo` from settings + game CSV, applies season regression, sets player scores
- `backtest.py` -> walk-forward backtest, also houses grid search, genetic (`scipy.optimize.differential_evolution`), and Bayesian (GP + EI) optimizers
- `mega_backtest.py` -> walk-forward backtest for all 35 models with meta-learner training
- `mega_optimizer.py` -> 7-phase per-model optimization (solo tuning, tournament, ablation, DE fine-tuning, validation)
- `mega_predictor.py` -> live predictions using all 35 models (replays history, loads trained meta-learner)
- `main.py` -> CLI entry point, `dispatch()` routes all commands, team name input triggers prediction flow

**State files (all gitignored, generated at runtime):**
- `nfl_elo_settings.json` -- tunable Elo parameters (24 params: K-factor, home advantage, division factor, etc.)
- `nfl_elo_ratings.json` -- current team Elo ratings
- `nfl_platt_scaler.json` / `nfl_isotonic_scaler.json` / `nfl_beta_scaler.json` -- calibration scalers
- `nfl_enhanced_model.json` / `nfl_xgb_model.json` -- saved XGBoost model weights
- `nfl_mega_settings.json` -- mega-ensemble model switches, per-model hyperparameters, meta-learner config
- `nfl_meta_xgb.json` / `nfl_meta_learner.json` -- trained meta-learner weights
- `nfl_enhanced_features.npz` -- saved enhanced feature matrix for SHAP analysis
- `weather_cache.json` -- cached weather API responses (2-hour TTL)
- `predicts_lots.csv` -- trade ledger (positions, P&L)

**Key patterns:**
- All settings load/save through `config.py` (`load_elo_settings` / `save_elo_settings`)
- Color output uses wrapper functions in `color_helpers.py` (`cok`, `cerr`, `cwarn`, `chi`, `cdim`, `cbold`), not raw colorama
- The `NFLElo` class stores per-team state in dicts/defaultdicts on the instance (ratings, last game dates, recent results, player scores, etc.)
- Optimizer objective is `LogLoss * 8 + Brier * 40` -- this weighting is intentional
- `backtest_model(..., fit_platt=True)` should only be called for user-facing runs, never inside optimizer loops (leakage + speed)
- Mega-ensemble config managed through `mega_config.py` (model registry, switches, per-model hyperparams)
- Smart caching via `cache_utils.py` -- season-aware staleness checks (different refresh rates for games, players, odds, weather)

### Core Model Pipeline

1. **Data Download** (`data_games.py`, `data_players.py`)
   - Games: ESPN public API scoreboard -> 2 years of completed games
   - Players: ESPN passing, rushing, receiving leaders
   - Advanced: EPA, CPOE, success rate via nfl_data_py/nflverse play-by-play
   - Cache: 6-hour staleness check on all CSV files (season-aware via `cache_utils.py`)

2. **Elo Model** (`elo_model.py`)
   - `NFLElo` class with 32 NFL teams
   - Base rating 1500, configurable K-factor (default 28.36 for 17-game season)
   - Home field advantage (default 25.55 Elo, ~53.6% implied home win rate)
   - Margin of victory adjustment (logarithmic, `log(max(1, abs(margin)) + 1)`)
   - Player strength boost (team composite from passing/rushing/receiving leaders)
   - Rest days centered at 7 (NFL weekly schedule), bye week factor
   - B2B penalty for Thursday games after Sunday (short turnaround)
   - Division rivalry factor (31.33 default)
   - Altitude bonus for Denver Broncos only (5280 ft)
   - Travel fatigue via timezone-based distance
   - Season regression (33% pull toward mean at year boundary)
   - Playoff HCA factor (1.1 default)

3. **XGBoost Ensemble** (`enhanced_model.py`)
   - 31 rolling features per game (TeamTracker class, includes Pythagorean, streaks, consistency, trend)
   - Walk-forward training (no leakage): Elo-only for first 200 games
   - Default blend: 80% Elo / 20% XGBoost
   - Optional time-decay: transitions from 95% Elo early to 70% Elo late
   - SHAP feature importance via XGBoost native `pred_contribs`

4. **Mega-Ensemble** (`mega_backtest.py`, `mega_predictor.py`, `mega_config.py`, `mega_optimizer.py`, `meta_learner.py`)
   - 35 base models across 7 tiers (see MODEL_REGISTRY in `mega_config.py`)
   - Walk-forward backtest: first 200 games Elo-only, retrain every 50 games (NFL defaults)
   - Meta-learner stacking: XGBoost (default), ridge, or logistic regression
   - Bounded probability adjustment: meta prediction clamped to +/- max_adj (default 0.10)
   - Per-model on/off switches saved to `nfl_mega_settings.json`
   - Per-model hyperparameter tuning via 7-phase optimizer
   - Live predictions via `MegaPredictor` class (replays all history through 35 models at startup)

5. **Calibration** (`platt.py`)
   - Platt scaling: logistic regression on logit(raw_prob)
   - Isotonic regression: PAV algorithm (pure numpy, no sklearn)
   - Beta calibration: 3-parameter asymmetric (a, b, c)
   - Season regression: 33% pull toward mean at year boundaries

## File Map

### Core System
| File | Purpose |
|------|---------|
| `main.py` | CLI entry point, `dispatch()` routes all commands |
| `config.py` | Constants, 32 NFL teams, 8 divisions, settings I/O |
| `elo_model.py` | `NFLElo` class (ratings, predictions, 24+ adjusters) |
| `build_model.py` | Model training with season regression, altitude bonus |
| `data_games.py` | Game data download via ESPN public API |
| `data_players.py` | Player stats download + team scoring |
| `backtest.py` | All backtesting & optimization (~2100 lines) |
| `enhanced_model.py` | XGBoost ensemble + SHAP + TeamTracker |
| `platt.py` | Calibration (Platt, isotonic, beta, regression) |
| `metrics.py` | Log loss, Brier, ECE, MCE, BSS, conformal |
| `elo_set_handler.py` | Shared handler for `set param=value` commands |
| `single_param_opt.py` | Coordinate descent optimizer (one param at a time) |
| `cache_utils.py` | Season-aware smart caching for all API data |

### Mega-Ensemble (35 models)
| File | Purpose |
|------|---------|
| `mega_predictor.py` | `MegaPredictor` class: live predictions using all 35 models |
| `mega_backtest.py` | Walk-forward backtest for full mega-ensemble + meta-learner training |
| `mega_optimizer.py` | 7-phase per-model optimizer (solo, tournament, ablation, DE, validation) |
| `mega_config.py` | Model registry, on/off switches, per-model hyperparams, `mega set` handler |
| `meta_learner.py` | `MetaLearner` class: XGBoost/ridge/logistic stacking combiner |

### Individual Base Models
| File | Purpose |
|------|---------|
| `hmm_model.py` | Hidden Markov Model (hot/cold team states) |
| `kalman_model.py` | Kalman filter (latent strength estimation) |
| `network_model.py` | PageRank + HITS (network/graph analysis) |
| `gbm_models.py` | LightGBM + CatBoost gradient boosting |
| `nn_models.py` | MLP + LSTM neural networks (requires PyTorch) |
| `random_forest_model.py` | Random Forest (bagging diversity) |
| `volatility_model.py` | GARCH volatility + Lyapunov exponent + Hurst exponent |
| `signal_model.py` | Fourier / wavelet (cycle detection) |
| `survival_model.py` | Survival analysis (streak hazard rates) |
| `copula_model.py` | Copula (offense/defense joint dependency) |
| `information_theory_model.py` | Shannon entropy + KL divergence |
| `momentum_model.py` | Newtonian momentum / inertia model |
| `markov_chain_model.py` | Markov chain (transition matrices) |
| `clustering_model.py` | k-Means team archetypes |
| `game_theory_model.py` | Nash equilibrium + style matchups |
| `poisson_model.py` | Poisson / Dixon-Coles (score distributions) |
| `glicko_model.py` | Glicko-2 (uncertainty-aware ratings) |
| `bradley_terry_model.py` | Bradley-Terry MLE (paired comparison) |
| `monte_carlo_model.py` | Monte Carlo simulation (2000 sims default) |
| `classic_models.py` | SRS, Colley Matrix, Log5, Pythagorean, Exponential Smoothing, Mean Reversion |
| `svm_model.py` | SVM classifier (RBF kernel + Platt scaling) |
| `fibonacci_model.py` | Fibonacci retracement (EMA-smoothed support/resistance levels) |
| `evt_model.py` | Extreme Value Theory (Generalized Pareto tail risk) |
| `benford_model.py` | Benford's Law (chi-squared scoring anomaly detection) |

### Data Enrichment
| File | Purpose |
|------|---------|
| `advanced_stats.py` | EPA, CPOE, success rate, referee tendencies via nfl_data_py |
| `odds_tracker.py` | Live odds via The Odds API (free tier), CLV tracking |
| `weather.py` | Weather impact via Open-Meteo API (free, no key) -- outdoor games only |
| `kalshi.py` | Kalshi public API for live contract prices + auto-Kelly |
| `injuries.py` | ESPN injury report + position-based Elo impact scoring |

### Trading & Display
| File | Purpose |
|------|---------|
| `predict_ledger.py` | Contract ledger management (add, sell, resolve, mark, invert) |
| `live_scores.py` | Live NFL scores via ESPN API, 60s refresh loop |
| `auto_resolve.py` | Auto-settle finished trades against live final scores |
| `html_generator.py` | Blogger-ready HTML prediction tables |
| `help_system.py` | Help text for all commands |
| `accuracy_test.py` | Quick walk-forward accuracy test |
| `color_helpers.py` | Colorama terminal formatting (`cok`, `cerr`, `cwarn`, `chi`, `cdim`, `cbold`) |

### Utility Scripts (standalone)
| File | Purpose |
|------|---------|
| `run_optimize.py` | Standalone optimization runner |
| `accuracy_optimize.py` | Accuracy-focused optimization |
| `run_enhanced_all.py` | Run enhanced backtest standalone |
| `quick_optimizer.py` | Quick parameter sweep |
| `sweep_enhanced.py` | Enhanced parameter sweep |

## Data Files (generated at runtime, all gitignored)

### API Data Caches
- `nfl_recent_games.csv` -- Game history (2 years, 6-hour cache)
- `nfl_player_stats.csv` -- Player leaders (6-hour cache)
- `nfl_advanced_stats.csv` -- EPA/CPOE stats via nfl_data_py
- `nfl_epa_stats.csv` -- Cached EPA/CPOE per team
- `weather_cache.json` -- Open-Meteo weather API responses (2-hour TTL)

### Model State
- `nfl_elo_ratings.json` -- Current team Elo ratings (32 teams)
- `nfl_elo_settings.json` -- Tuned Elo parameters (24 params)
- `nfl_platt_scaler.json` -- Platt calibration coefficients
- `nfl_isotonic_scaler.json` -- Isotonic calibration mappings
- `nfl_beta_scaler.json` -- Beta calibration (a, b, c params)
- `nfl_enhanced_model.json` -- XGBoost ensemble metadata
- `nfl_xgb_model.json` -- XGBoost model weights
- `nfl_enhanced_features.npz` -- Saved feature matrix for SHAP analysis

### Mega-Ensemble State
- `nfl_mega_settings.json` -- Model switches, per-model hyperparams, meta-learner config
- `nfl_meta_xgb.json` -- Trained meta-learner XGBoost weights
- `nfl_meta_learner.json` -- Meta-learner state (feature names, importance)

### Optimization Results
- `nfl_grid_search.csv` -- Grid search scored parameter combos
- `nfl_bayesian_results.csv` -- Bayesian optimization trial history
- `nfl_genetic_results.csv` -- Genetic optimizer results
- `nfl_backtest_predictions.csv` -- Per-game backtest predictions
- `nfl_calibration.csv` -- 10-bin calibration table

### Trading & Display
- `predicts_lots.csv` -- Trading ledger (positions, P&L)
- `today_nfl_predictions.html` -- Blogger-ready HTML predictions
- `today_nfl_predictions.txt` -- Plain-text predictions

## Key NFL-Specific Design Choices

### Elo Parameters (defaults from `config.load_elo_settings`)
- **K-factor = 28.36**: High K because 17-game season means each game carries significant signal
- **Home advantage = 25.55 Elo**: Reflects ~53.6% NFL home win rate
- **player_boost = 24.61**: Team player strength boost from composite scoring
- **rest_factor = 0.68**: Rest days centered at 7 (NFL weekly schedule)
- **b2b_penalty = 5.45**: Thursday Night Football penalty (short turnaround after Sunday)
- **bye_week_factor = 0.0**: Bye week boost (currently disabled, tunable)
- **division_factor = 31.33**: Divisional rivalry familiarity adjustment
- **win_streak_factor = 20.0**: Hot/cold streak adjustment
- **homestand_factor = 20.0**: Extended home-game stretch bonus
- **season_phase_factor = 20.0**: Early/mid/late season adjustment
- **altitude_factor = 0.66**: Denver Broncos altitude bonus multiplier
- **playoff_hca_factor = 1.1**: Adjusted home advantage in playoffs
- **rest_advantage_cap = 3.32**: Maximum rest differential impact
- **season_regress = 0.33**: 33% pull toward mean at season boundaries
- **MOV formula**: `log(max(1, abs(margin)) + 1)` -- logarithmic dampening for NFL blowouts

### Mega-Ensemble NFL Defaults (from `mega_backtest.SPORT_DEFAULTS`)
- **window = 5**: Rolling feature window (narrow due to 17-game NFL season)
- **min_train = 200**: Games before meta-learner starts predicting
- **retrain_every = 50**: Retrain ML models every 50 games
- **kalman_process_noise = 1.0**: Kalman filter process noise
- **kalman_measurement_noise = 15.0**: Kalman filter measurement noise
- **hmm_min_games = 5**: Minimum games before HMM predictions
- **pyth_exp = 2.37**: Pythagorean exponent for point-based win estimation

### Other NFL-Specific Design
- **Season = two calendar years**: NFL runs Sep-Feb; `season = year if month >= 9 else year - 1`
- **17-game regular season**: Far fewer data points than NBA (82) or MLB (162), so higher K-factor and wider confidence intervals
- **Position-based injury impact**: QB out = -50 Elo, RB = -15, WR/TE = -12, DEF = -10, OL = -8, K/P = -5
- **Advanced stats**: EPA (Expected Points Added), CPOE (Completion % Over Expected), success rate from nfl_data_py/nflverse play-by-play data
- **Weather matters**: Open-Meteo weather for outdoor stadiums affects game predictions (wind, temperature, precipitation)
- **Only Denver gets altitude bonus**: 5280 ft Mile High Stadium is the only NFL venue at significant elevation
- **Bye week detection**: Games with 10+ day rest gap detected as post-bye
- **Division awareness**: `config.same_division()` for 8-division rivalry adjustments
- **Timezone tracking**: Per-team UTC offsets for travel fatigue calculations

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
- Feature vector has 31 columns (`FEATURE_COLS` in `enhanced_model.py`): elo_prob, elo_diff, player_diff, per-team rolling stats, differentials, rest, Pythagorean win expectation, streaks, consistency, and trend
- XGBoost params: `max_depth=5, eta=0.03, subsample=0.9, colsample_bytree=0.8, min_child_weight=3, 300 rounds`
- After the walk-forward, fits both Platt and isotonic calibrators on ensemble probabilities
- Saves the trained XGBoost booster to `nfl_xgb_model.json` and metadata to `nfl_enhanced_model.json`

The saved XGBoost model is loaded by `NFLElo._xgb_predict()` at runtime for live predictions, using the team's `_team_scores` deque to construct rolling features on the fly.

### Elo Optimization Commands

- **Grid search** (`grid`): Cartesian product over 7 Elo parameters with configurable ranges
- **Genetic** (`genetic`): `scipy.optimize.differential_evolution` over same 7 params
- **Bayesian** (`bayesian`): GP surrogate + Expected Improvement acquisition (~50-100 evaluations)
- **Auto-optimize** (`autoopt`): automatic grid->genetic->bayesian pipeline
- **Super-optimize** (`superopt`): exhaustive 7-phase optimization (9 params, takes hours)
- **Coordinate descent** (`singleopt`): one-param-at-a-time sweep with fine-grained refinement

All use the same objective: `score = -(LogLoss * 8 + Brier * 40)`. After completion, `_apply_best_settings()` saves the winning params and refits the Platt scaler.

### Mega-Ensemble Optimization (`mega optimize` command)

`mega_optimizer.py` runs a 7-phase per-model optimization:
- **Phase 0**: Elo-only baseline -- establish the floor
- **Phase 1**: Per-model solo optimization -- Elo + one model at a time, coordinate descent
- **Phase 2**: Head-to-head tournament -- pairs, triples, top-N combos
- **Phase 3**: Meta-learner + global tuning -- max_adj, meta_model, retrain_every, min_train, window
- **Phase 4**: Combined DE fine-tuning -- differential evolution over top numeric params
- **Phase 5**: Final ablation -- test each model's contribution with fully tuned params
- **Phase 6**: Validation -- run best config 5 times, report stability

### Calibration Methods (`platt.py`)

Three calibrators are available, all implemented without sklearn:
- **Platt scaling**: logistic regression on `logit(raw_prob)` via `scipy.optimize.minimize` (L-BFGS-B)
- **Isotonic regression**: Pool Adjacent Violators algorithm on binned probabilities (50 bins default)
- **Beta calibration**: 3-parameter `logit(p_cal) = c + a*log(p) - b*log(1-p)`

Season regression (`regress_ratings_to_mean`): pulls all team ratings 33% toward the league mean at season boundaries, applied during model building.

## Advanced Backtesting Methods

All 16 advanced backtesting techniques are implemented across `backtest.py`, `enhanced_model.py`, `platt.py`, and `metrics.py`. No additional dependencies beyond the existing stack (numpy, scipy, pandas, xgboost).

### Metrics (auto-reported by `backtest` command)

- **ECE / MCE** (`metrics.py: ece_score, mce_score`) -- Expected and Maximum Calibration Error. ECE < 0.03 is excellent; > 0.08 needs work.
- **BSS** (`metrics.py: brier_skill_score`) -- Brier Skill Score vs 50% baseline and vs home-win-rate baseline. BSS > 0 means model beats the reference.
- **DSR** (`backtest.py: deflated_sharpe_ratio`) -- Deflated Sharpe Ratio. Adjusts optimization "best" for multiple-testing bias. DSR < 1.96 means best is not significant at 95%.

### Cross-Validation (`purgedcv`, `cpcv` commands)

- **Purged Walk-Forward CV** (`purgedcv`) -- k chronological folds with embargo gap. Reports mean/std across folds. Accuracy variance >3% = fragile model.
- **CPCV** (`cpcv`) -- All C(k, k_test) train/test combinations. With k=5, k_test=2: 10 paths.

### Overfitting Detection (`pbo`, `montecarlo` commands)

- **PBO** (`pbo`) -- Symmetric cross-validation on grid search trial population. PBO > 0.5 = optimization is likely overfit.
- **Monte Carlo Permutation** (`montecarlo`) -- Shuffles scores randomly, re-runs backtest N times (default 500), computes p-value. p > 0.05 = model not statistically significant.

### Calibration (`rollingcal`, `betacal`, `conformal` commands)

- **Rolling Origin Recalibration** (`rollingcal`) -- Expanding-window Platt fitting every 50 games.
- **Beta Calibration** (`betacal`) -- 3-parameter asymmetric calibration.
- **Conformal Prediction** (`conformal`) -- Distribution-free prediction sets at 90%, 95%, 80% coverage.

### Ensemble Analysis (`enhanced decay`, `shap` commands)

- **Time-Decayed Weighting** (`enhanced decay`) -- Elo weight transitions 95% -> 70% over the season.
- **SHAP Feature Importance** (`shap`) -- XGBoost native `pred_contribs`.

### Structural Analysis (`sliding`, `convergence` commands)

- **Sliding Window** (`sliding`) -- Heavy regression (67%) every N games vs standard expanding window.
- **Convergence Analysis** (`convergence`) -- Chunks predictions by N-game windows, finds burn-in period.

### Position Sizing (`kelly` command)

**Kelly Criterion Backtest** -- Simulates fractional Kelly position sizing on `nfl_backtest_predictions.csv`. Default quarter-Kelly. Reports final bankroll, return %, max drawdown, annualized Sharpe.

### Complete Model Validation Workflow (run in order)

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

### Decision Framework After Validation

| Metric | Good | Marginal | Bad | Action if bad |
|--------|------|----------|-----|---------------|
| Accuracy | >66% | 63-66% | <63% | Revisit Elo adjusters, check data quality |
| ECE | <0.03 | 0.03-0.08 | >0.08 | Refit Platt, try beta calibration |
| BSS vs 50% | >0.08 | 0.04-0.08 | <0.04 | Model has weak discriminative power |
| PBO | <0.3 | 0.3-0.5 | >0.5 | Grid search overfit -- use wider ranges |
| DSR | >1.96 | 1.0-1.96 | <1.0 | Best params are noise -- simplify model |
| Monte Carlo p | <0.01 | 0.01-0.05 | >0.05 | Model has no statistically significant skill |
| Purged CV std | <2% | 2-3% | >3% | Model is fragile to training data selection |
| Kelly Sharpe | >1.0 | 0.5-1.0 | <0.5 | Edge too thin for real trading |
| Kelly max DD | <30% | 30-50% | >50% | Size down (use 10%-Kelly instead of 25%) |

### Trading Ledger (`predict_ledger.py`)
- CSV-based lot tracking with entry/exit fees (2%)
- Add, sell (partial), resolve, mark, invert positions
- Monthly P&L chart generation
- Live score matching for open positions

### Live Features
- `live_scores.py`: ESPN live scores, 60s refresh loop
- `auto_resolve.py`: Auto-settle finished games against open trades
- `injuries.py`: ESPN NFL injury API with position-based Elo impact scoring
- `html_generator.py`: Blogger-ready HTML prediction tables

## Mega-Ensemble

The `mega_predictor.py` module implements a 35-model mega-ensemble predictor. Each model runs independently on the same walk-forward game loop, producing a raw probability. A meta-learner (Ridge, Logistic, or XGBoost) combines all 35 outputs into a single calibrated adjustment that is clamped to +/- `max_adj` (default 0.10) around the Elo anchor probability.

**Key commands:**
- `mega` -- Run full mega-ensemble backtest with all enabled models
- `mega optimize` -- 7-phase per-model exhaustive optimization (54 hyperparameters)
- `mega quick` -- Quick mega optimization (Phases 0-1 only)
- `mega tune` -- Per-model solo optimization (same as mega quick)
- `mega tournament` -- Head-to-head model comparison (Phase 2 only)
- `mega ablation` -- Ablation study: test each model's contribution, auto-prune weak ones
- `mega models` -- Show all 35 models with ON/OFF status by tier
- `mega on <model>` / `mega off <model>` -- Enable/disable individual models (or `mega on all`)
- `mega settings` -- Show all mega parameter values
- `mega set <param>=<value>` -- Set mega parameters (e.g., `mega set adj=0.10`, `mega set meta=ridge`, `mega set mc_sims=3000`)

**Model tiers:** Tier 0 (Elo, XGBoost), Tier 1 (HMM, Kalman, PageRank, LightGBM, CatBoost, MLP, LSTM), Tier 2 (GARCH, Fourier, Survival, Copula), Tier 3 (InfoTheory, Momentum, Markov, Clustering, GameTheory), Tier 4 (Poisson, Glicko-2, Bradley-Terry, Monte Carlo, Random Forest), Tier 5 (SRS, Colley, Log5, Pythagorean, ExpSmoothing, MeanReversion), Tier 6 (Weather, Odds), Tier 7 (SVM, Fibonacci, EVT, Benford).

All 35 models run in parallel via `ThreadPoolExecutor`. Settings stored in `nfl_mega_settings.json`.

## Comprehensive Workflow

### Daily Prediction Workflow

1. **Launch**: `python main.py` -- auto-downloads fresh data (games, players, injuries), builds Elo model, runs baseline backtest, fits Platt scaler
2. **Check model**: `settings` to verify parameters, `all` to see team rankings, `injuries` to review injury impact
3. **Make predictions**: type a team name (e.g. `Chiefs`) -> enter opponent -> specify home/away (`a`/`b`/`n`) -> see calibrated win probability with injury impact and key player stats
4. **Log position**: type `y` -> enter contract count, price, optional notes -> position saved to `predicts_lots.csv`
5. **Publish**: `today` or `tomorrow` generates Blogger HTML (`today_nfl_predictions.html`) + plain text for all scheduled games
6. **Monitor**: `live` shows real-time scores for open positions; `mark` updates current market prices
7. **Settle**: `resolve` to manually settle (win/loss prompt), `autoresolve` to auto-settle from live final scores, `sell` to exit early at a price
8. **Review**: `predicts` shows full P&L ledger with realized/unrealized/marked totals, win rate, ROI; `chart` saves monthly P&L bar chart

### Model Tuning Workflow

1. **Baseline**: run `backtest` -- note accuracy, log loss, Brier, ECE, MCE, BSS (all auto-reported)
2. **Diagnostics**: run `convergence` to find burn-in period, `sliding` to check if old data helps
3. **Coarse search**: run `grid` with wide ranges and large steps to identify promising parameter regions
4. **Validate search**: run `pbo` to check overfitting, `results` to check DSR significance
5. **Fine-tune**: run `genetic` or `bayesian` with tighter bounds around the grid search sweet spot
6. **Cross-validate**: run `purgedcv` for fold stability, `cpcv` for combinatorial robustness
7. **Significance test**: run `montecarlo` -- if p > 0.05, model has no proven edge
8. **Train ensemble**: run `enhanced` (SHAP auto-runs), then `enhanced decay` to compare
9. **Calibration**: run `rollingcal` for OOS calibration, `betacal` for asymmetry, `conformal` for coverage
10. **P&L simulation**: run `kelly` to connect model quality to bankroll trajectory
11. **Mega-ensemble**: run `mega` to train 35-model stack, then `mega optimize` for full tuning
12. **Iterate**: adjust individual params with `set k=28`, `set home=25`, etc. -- always rerun `backtest` after to refit the Platt scaler

### Position Management

Predicts contracts are $1 binary options. The ledger tracks:
- **Entry**: contracts count, price per contract, 2% entry fee on potential payout ($1 per contract)
- **Mark-to-market**: `mark` updates current price; `predicts` shows unrealized P&L based on marks
- **Exit**: `sell` for partial/full exit at a price (with 2% exit fee); `resolve` for binary win/loss settlement
- **Invert**: `invert` flips a position's direction (swaps predicted winner, inverts probability) without changing cost basis
- **Auto-resolve**: matches open positions against today's final scores via ESPN API; toggle with `autoresolve on/off` (runs on startup when enabled)

### Data Refresh

- `refresh` deletes cached game/player/ratings/scaler files and re-downloads everything, then rebuilds the model
- Data caches expire after 6 hours for games/players, 4 hours for injuries, 2 hours for weather, 60 seconds for live scores
- Cache staleness checked by `config.is_cache_stale()` with sport-aware logic from `cache_utils.py`
- Files under 500 bytes are always considered stale (empty/corrupt stubs)

## CLI Commands

Enter a team name to start a prediction. Core commands:

**Data & Display**: `all`, `refresh`, `players`, `settings`, `injuries`, `today`/`html`, `tomorrow`

**Data Enrichment**: `odds`, `kalshi`, `weather`, `advstats`

**Backtesting**: `backtest`, `enhanced`, `enhanced decay`

**Elo Optimization**: `grid`, `genetic`, `bayesian`, `autoopt`, `superopt`, `singleopt`, `results`

**Validation**: `purgedcv`, `cpcv`, `pbo`, `montecarlo`, `convergence`, `sliding`

**Calibration**: `rollingcal`, `betacal`, `conformal`

**Analysis**: `shap`, `kelly`

**Mega-Ensemble (35 models)**:
- `mega` -- Run mega-ensemble backtest (all enabled models)
- `mega optimize` -- 7-phase per-model optimization (all phases, takes hours)
- `mega quick` -- Quick mega optimization (Phases 0-1 only)
- `mega tune` -- Per-model solo optimization (same as mega quick)
- `mega tournament` -- Head-to-head model tournament (Phase 2)
- `mega ablation` -- Test each model's individual contribution, auto-prune weak ones
- `mega models` -- Show all 35 models with ON/OFF status by tier
- `mega on <model>` / `mega off <model>` -- Enable/disable individual models (or `mega on all`)
- `mega settings` -- Show all mega parameter values
- `mega set <param>=<value>` -- Set mega parameter (e.g., `mega set max_adj=0.10`, `mega set meta=ridge`)

**Trading**: `predicts`, `balance`, `resolve`, `sell`, `mark`, `invert`, `chart`, `live`, `autoresolve`, `autoresolve on/off`

**Elo Settings** (24 params, type `set` to see all):
- `set k=28`, `set home=25`, `set boost=24`, `set rest=0.68`, `set b2b=5`
- `set travel=0`, `set pace=0`, `set div=31`, `set streak=20`
- `set altitude=0.66`, `set playoff=1.1`, `set phase=20`
- `set kelly=quarter`, `set balance=50`, `set autoresolve=true`

`help` for overview, `help <command>` for details, `help advanced` for all validation commands, `quit` to exit.

## Conventions

- Python 3.8+ compatibility (no walrus operators, no `match` statements)
- No test framework -- validation is via `backtest` command and `accuracy_test.py` (run manually: `python accuracy_test.py [label]`)
- All data files use `nfl_` prefix; temp files use `temp_` prefix (both gitignored)
- Settings are tuned via `set param=value` CLI command or optimizer; changes require `backtest` rerun to refit Platt scaler
- Color output uses `color_helpers.py` wrappers (`cok`, `cerr`, `cwarn`, `chi`, `cdim`, `cbold`), never raw colorama
- Optimizer objective is `LogLoss * 8 + Brier * 40` -- this weighting is intentional and should not be changed casually
- Team lookup is fuzzy: `NFLElo.find_team()` accepts full names, abbreviations, partial matches, and close matches via `difflib.get_close_matches`
- NFL season detection: month >= 9 means current year, month <= 8 means previous year's season
- All 32 NFL teams defined in `config.py` TEAM_ABBR dict
- Mega-ensemble model switches are per-sport and saved to `nfl_mega_settings.json`
- All data sources must be completely free (no paid APIs)

## Dependencies
- pandas, numpy, scipy, colorama, tqdm, xgboost, requests, matplotlib
- nfl_data_py (for EPA/CPOE advanced stats)
- Optional: torch (for MLP/LSTM), hmmlearn, filterpy, lightgbm, catboost, networkx
