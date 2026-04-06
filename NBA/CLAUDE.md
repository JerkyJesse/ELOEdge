# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

NBA game prediction system combining Elo ratings with a 35-model mega-ensemble, with Predicts $1 contract trading ledger. Interactive CLI app -- no web server, no test framework, no build system.

## Running

```bash
pip install -r requirements.txt
python main.py
```

First run auto-downloads game data (NBA API), player stats, and injury reports (ESPN). Then runs a backtest to fit the Platt calibration scaler and enters the interactive CLI loop.

## Architecture

**Three-stage prediction pipeline:**
1. **Elo model** (`elo_model.py` → `NBAElo` class) -- base team ratings adjusted for home court, altitude, player strength, rest days, travel fatigue, pace mismatch, injuries, and strength of schedule
2. **XGBoost ensemble** (`enhanced_model.py`) -- 80% Elo / 20% XGBoost (default `elo_weight=0.8`) using 102 rolling features per team (10-game window via `TeamTracker`, includes Pythagorean win expectation, streaks, consistency, and trend)
3. **Mega-ensemble** (`mega_predictor.py` + `mega_backtest.py`) -- 35 base models stacked via a meta-learner (XGBoost, ridge, or logistic). Produces a bounded adjustment (+/- max_adj, default 0.08) on top of the Elo+XGBoost probability. Models span 7 tiers: Core (Elo, XGBoost), Proven (HMM, Kalman, PageRank, LightGBM, CatBoost, MLP, LSTM), Exotic (GARCH, Fourier/Wavelet, Survival, Copula), Info/Physics (Shannon Entropy, Momentum, Markov Chain, Clustering, Game Theory), Classical Ratings (Poisson, Glicko-2, Bradley-Terry, Monte Carlo, Random Forest), Sports-Specific (SRS, Colley Matrix, Log5, PythagenPat, Exponential Smoothing, Mean Reversion), Additional (SVM, Fibonacci, EVT, Benford), and Data Enrichment (Weather, Odds).
4. **Platt calibration** (`platt.py`) -- logistic regression on raw probabilities for well-calibrated outputs

**Data flow:**
- `data_games.py` / `data_players.py` → download from NBA API with 6-hour cache (`config.is_cache_stale`)
- `injuries.py` → ESPN JSON API with 4-hour cache
- `weather.py` → Open-Meteo API (free, no key) with 2-hour cache
- `odds_tracker.py` → The Odds API (free tier, 500 req/month) for moneyline odds + CLV
- `kalshi.py` → Kalshi public API for live contract prices
- `build_model.py` → constructs `NBAElo` from settings + game CSV, applies season regression, sets player scores
- `backtest.py` → walk-forward backtest, also houses grid search, genetic (`scipy.optimize.differential_evolution`), and Bayesian (GP + EI) optimizers
- `mega_backtest.py` → walk-forward backtest for all 35 models with meta-learner training
- `mega_optimizer.py` → 7-phase per-model optimization (solo tuning, tournament, ablation, DE fine-tuning, validation)
- `mega_predictor.py` → live predictions using all 35 models (replays history, loads trained meta-learner)
- `main.py` → CLI entry point, `dispatch()` routes all commands, team name input triggers prediction flow

**State files (all gitignored, generated at runtime):**
- `nba_elo_settings.json` -- tunable Elo parameters (39 params: K-factor, home advantage, etc.)
- `nba_elo_ratings.json` -- current team Elo ratings
- `nba_platt_scaler.json` / `nba_isotonic_scaler.json` / `nba_beta_scaler.json` -- calibration scalers
- `nba_enhanced_model.json` / `nba_xgb_model.json` -- saved XGBoost model weights
- `nba_mega_settings.json` -- mega-ensemble model switches, per-model hyperparameters, meta-learner config
- `nba_meta_xgb.json` / `nba_meta_learner.json` -- trained meta-learner weights
- `nba_enhanced_features.npz` -- saved enhanced feature matrix for SHAP analysis
- `weather_cache.json` -- cached weather API responses (2-hour TTL)
- `predicts_lots.csv` -- trade ledger (positions, P&L)

**Key patterns:**
- All settings load/save through `config.py` (`load_elo_settings` / `save_elo_settings`)
- Color output uses wrapper functions in `color_helpers.py` (`cok`, `cerr`, `cwarn`, `chi`, `cdim`, `cbold`), not raw colorama
- The `NBAElo` class stores per-team state in dicts/defaultdicts on the instance (ratings, last game dates, recent results, player scores, etc.)
- Optimizer objective is `LogLoss * 8 + Brier * 40` -- this weighting is intentional
- `backtest_model(..., fit_platt=True)` should only be called for user-facing runs, never inside optimizer loops (leakage + speed)
- Mega-ensemble config managed through `mega_config.py` (model registry, switches, per-model hyperparams)
- Smart caching via `cache_utils.py` -- season-aware staleness checks (different refresh rates for games, players, odds, weather)

## File Map

### Core System
| File | Purpose |
|------|---------|
| `main.py` | CLI entry point, `dispatch()` routes all commands |
| `config.py` | Constants, 30 NBA teams, divisions, settings I/O |
| `elo_model.py` | `NBAElo` class (ratings, predictions, 24+ adjusters) |
| `build_model.py` | Model training with season regression |
| `data_games.py` | Game data download via nba_api |
| `data_players.py` | Player stats download + team scoring |
| `backtest.py` | All backtesting & optimization (~2100 lines) |
| `enhanced_model.py` | XGBoost ensemble + SHAP + TeamTracker |
| `platt.py` | Calibration (Platt, isotonic, beta, regression) |
| `metrics.py` | Log loss, Brier, ECE, MCE, BSS, conformal |
| `elo_set_handler.py` | Shared handler for `set param=value` commands (39 Elo params) |
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
| `classic_models.py` | SRS, Colley Matrix, Log5, PythagenPat, Exponential Smoothing, Mean Reversion |
| `svm_model.py` | SVM classifier (RBF kernel + Platt scaling) |
| `fibonacci_model.py` | Fibonacci retracement (EMA-smoothed support/resistance levels) |
| `evt_model.py` | Extreme Value Theory (Generalized Pareto tail risk) |
| `benford_model.py` | Benford's Law (chi-squared scoring anomaly detection) |

### Data Enrichment
| File | Purpose |
|------|---------|
| `odds_tracker.py` | Live odds via The Odds API (free tier), CLV tracking |
| `weather.py` | Weather impact via Open-Meteo API (free, no key) |
| `kalshi.py` | Kalshi public API for live contract prices + auto-Kelly |
| `injuries.py` | ESPN injury report + Elo impact scoring |

### Trading & Display
| File | Purpose |
|------|---------|
| `predict_ledger.py` | Contract ledger management (add, sell, resolve, mark, invert) |
| `live_scores.py` | Live NBA scores via NBA Stats API, 60s refresh loop |
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
- `nba_recent_games.csv` -- Game history (2 years, 6-hour cache)
- `nba_player_stats.csv` -- Player leaders (6-hour cache)
- `weather_cache.json` -- Open-Meteo weather API responses (2-hour TTL)

### Model State
- `nba_elo_ratings.json` -- Current team Elo ratings (30 teams)
- `nba_elo_settings.json` -- Tuned Elo parameters (39 params)
- `nba_platt_scaler.json` -- Platt calibration coefficients
- `nba_isotonic_scaler.json` -- Isotonic calibration mappings
- `nba_beta_scaler.json` -- Beta calibration (a, b, c params)
- `nba_enhanced_model.json` -- XGBoost ensemble metadata
- `nba_xgb_model.json` -- XGBoost model weights
- `nba_enhanced_features.npz` -- Saved feature matrix for SHAP analysis

### Mega-Ensemble State
- `nba_mega_settings.json` -- Model switches, per-model hyperparams, meta-learner config
- `nba_meta_xgb.json` -- Trained meta-learner XGBoost weights
- `nba_meta_learner.json` -- Meta-learner state (feature names, importance)

### Optimization Results
- `nba_grid_search.csv` -- Grid search scored parameter combos
- `nba_bayesian_results.csv` -- Bayesian optimization trial history
- `nba_genetic_results.csv` -- Genetic optimizer results
- `nba_backtest_predictions.csv` -- Per-game backtest predictions
- `nba_calibration.csv` -- 10-bin calibration table

### Trading & Display
- `predicts_lots.csv` -- Trading ledger (positions, P&L)
- `today_nba_predictions.html` -- Blogger-ready HTML predictions
- `today_nba_predictions.txt` -- Plain-text predictions

## Key NBA-Specific Design Choices

### Elo Parameters (defaults from `config.load_elo_settings`)
- **K-factor = 8.53**: Optimized K for 82-game season (balances responsiveness with stability)
- **Home advantage = 31.99 Elo**: Optimized NBA home court advantage (~58% implied home win rate)
- **player_boost = 48.0**: Team player strength boost from composite scoring (high -- NBA is star-driven)
- **rest_factor = 18.21**: Rest days impact (significant in NBA back-to-backs)
- **b2b_penalty = 0.0**: Back-to-back penalty (captured by rest_factor instead)
- **travel_factor = 12.20**: Cross-country travel fatigue
- **sos_factor = 2.0**: Strength of schedule adjustment
- **pace_factor = 26.53**: Pace mismatch adjustment (significant for tempo mismatches)
- **form_weight = 5.06**: Recent form (last 10 games) adjustment
- **division_factor = 10.0**: Divisional rivalry familiarity
- **altitude_factor = 0.04**: Denver Nuggets altitude bonus (near-zero -- optimizer found minimal impact)
- **playoff_hca_factor = 0.71**: Reduced home advantage in playoffs (~29% reduction)
- **road_trip_factor = 5.0**: Extended road trip penalty
- **homestand_factor = 15.0**: Extended homestand bonus
- **rest_advantage_cap = 2.47**: Maximum rest differential impact
- **season_regress = 0.33**: 33% pull toward mean at season boundaries

### Mega-Ensemble NBA Defaults (from `mega_backtest.SPORT_DEFAULTS`)
- **window = 10**: Rolling feature window (10-game for NBA's 82-game season)
- **min_train = 400**: Games before meta-learner starts predicting
- **retrain_every = 80**: Retrain ML models every 80 games
- **kalman_process_noise = 0.5**: Kalman filter process noise
- **kalman_measurement_noise = 8.0**: Kalman filter measurement noise
- **hmm_min_games = 10**: Minimum games before HMM predictions
- **pyth_exp = 13.91**: Pythagorean exponent for point-based win estimation

### Other NBA-Specific Design
- **Season = cross-year**: NBA runs Oct-Jun spanning two calendar years; season detection uses `year + 1 if month >= 10`
- **82-game season**: More data points than NFL (17) but fewer than MLB (162)
- **No pitcher/goalie system**: Unlike MLB/NHL, no individual player sub-ratings
- **Player scoring**: Points, assists, rebounds for scoring composite; offensive/defensive split
- **Data source**: `nba_api` package for game logs and player stats
- **Division awareness**: `config.same_division()` for divisional rivalry adjustments
- **Altitude**: Only Denver gets altitude bonus (5280 ft)

## Mega-Ensemble

The `mega_predictor.py` module implements a 35-model mega-ensemble predictor. Each model runs independently on the same walk-forward game loop, producing a raw probability. A meta-learner (Ridge, Logistic, or XGBoost) combines all 35 outputs into a single calibrated adjustment that is clamped to +/- `max_adj` (default 0.08) around the Elo anchor probability.

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
- `mega set <param>=<value>` -- Set mega parameters (e.g., `mega set max_adj=0.10`, `mega set meta=ridge`)

**Model tiers:** Tier 0 (Elo, XGBoost), Tier 1 (HMM, Kalman, PageRank, LightGBM, CatBoost, MLP, LSTM), Tier 2 (GARCH, Fourier, Survival, Copula), Tier 3 (InfoTheory, Momentum, Markov, Clustering, GameTheory), Tier 4 (Poisson, Glicko-2, Bradley-Terry, Monte Carlo, Random Forest), Tier 5 (SRS, Colley, Log5, Pythagorean, ExpSmoothing, MeanReversion), Tier 6 (Weather, Odds), Tier 7 (SVM, Fibonacci, EVT, Benford).

All 35 models run in parallel via `ThreadPoolExecutor`. Settings stored in `nba_mega_settings.json`.

## Backtesting & Optimization

### Walk-Forward Backtest (`backtest` command)

`backtest_model()` in `backtest.py` iterates through every game in `nba_recent_games.csv` chronologically. For each game it:
1. Calls `model.win_prob()` with `calibrated=False` and `use_injuries=False` (raw Elo only — no future leakage)
2. Records the prediction, then calls `model.update_game()` to update ratings *after* predicting
3. Outputs `nba_backtest_predictions.csv` (per-game predictions) and `nba_calibration.csv` (10-bin calibration table)

When `fit_platt=True` (user-facing runs only), it fits a Platt scaler on the full run's raw probabilities and saves to `nba_platt_scaler.json`. This flag must **never** be used inside optimizer objective functions — it causes data leakage and slows iteration.

Metrics reported: accuracy (%), log loss, Brier score. Calibration table bins predictions into 10 probability buckets and compares predicted vs actual win rates.

### XGBoost Enhanced Backtest (`enhanced` command)

`run_enhanced_backtest()` in `enhanced_model.py` is a separate walk-forward loop that builds a 80/20 Elo+XGBoost ensemble:
- First `min_train` games (default 200): Elo-only predictions while accumulating training features
- After that: XGBoost is trained on accumulated features and retrained every `retrain_every` games (default 50)
- `TeamTracker` maintains rolling 10-game windows (PPG, PAPG, win%, margins, rest days) per team
- Feature vector has 102 columns (`FEATURE_COLS` in `enhanced_model.py`): elo_prob, elo_diff, player_diff, per-team rolling stats, differentials, rest, Pythagorean win expectation, streaks, consistency, and trend
- XGBoost params: `max_depth=5, eta=0.03, subsample=0.9, colsample_bytree=0.8, min_child_weight=3, 300 rounds`
- After the walk-forward, fits both Platt and isotonic calibrators on ensemble probabilities
- Saves the trained XGBoost booster to `nba_xgb_model.json` and metadata to `nba_enhanced_model.json`

The saved XGBoost model is loaded by `NBAElo._xgb_predict()` at runtime for live predictions, using the team's `_team_scores` deque to construct rolling features on the fly.

### Grid Search (`grid` command)

`grid_search_optimization()` searches 7 parameters via Cartesian product: K, HomeAdv, PlayerBoost, RestFactor, TravelFactor, PaceFactor, PlayoffHCA. Interactive prompts let the user customize min/max/step for each dimension (defaults provided). Each combo builds a fresh `NBAElo`, runs `backtest_model()` (without Platt fitting), and scores with:

```
score = -(LogLoss * 8 + Brier * 40)
```

Player scores are precomputed once (`build_league_player_scores`) and altitude bonuses are cached to avoid redundant work across combos. Progress prints every 50 combos and on new bests. Results saved to `nba_grid_search.csv`. After completion, `_apply_best_settings()` saves the winning params and refits the Platt scaler.

### Genetic Optimizer (`genetic` command)

`genetic_optimization()` uses `scipy.optimize.differential_evolution` over the same 7 parameters. Interactive prompts configure bounds, `maxiter` (default 80), and `popsize` (default 40). Same objective function as grid search. Progress callback prints per-generation best. After convergence, does a final backtest with the best params, saves settings, and refits Platt.

The genetic optimizer is recommended for fine-tuning after a coarse grid search narrows the ranges.

### Standalone Accuracy Test

`python accuracy_test.py [label]` runs a quick walk-forward backtest with current settings, reports raw + Platt-calibrated metrics. Useful for measuring deltas after code changes without entering the full CLI.

### Calibration Methods (`platt.py`)

Two calibrators are available, both implemented without sklearn:
- **Platt scaling**: logistic regression on `logit(raw_prob)` via `scipy.optimize.minimize` (L-BFGS-B). Corrects overconfidence at 0.9+ and underconfidence at 0.1-0.2.
- **Isotonic regression**: Pool Adjacent Violators algorithm on binned probabilities (50 bins default), with linear interpolation for continuous mapping. Fitted during `enhanced` backtest but not currently used for live predictions.

Season regression (`regress_ratings_to_mean`): pulls all team ratings 33% toward the league mean at season boundaries, applied during model building.

## Advanced Backtesting Methods

All 16 advanced backtesting techniques are implemented across `backtest.py`, `enhanced_model.py`, `platt.py`, and `metrics.py`. No additional dependencies beyond the existing stack (numpy, scipy, pandas, xgboost).

### Metrics (auto-reported by `backtest` command)

- **ECE / MCE** (`metrics.py: ece_score, mce_score`) — Expected and Maximum Calibration Error. ECE < 0.03 is excellent; > 0.08 needs work. Reported automatically when `backtest` fits the Platt scaler.
- **BSS** (`metrics.py: brier_skill_score`) — Brier Skill Score vs 50% baseline and vs home-win-rate baseline. BSS > 0 means model beats the reference. Reported automatically by `backtest`.
- **DSR** (`backtest.py: deflated_sharpe_ratio`) — Deflated Sharpe Ratio. Adjusts optimization "best" for multiple-testing bias. DSR < 1.96 means best is not significant at 95%. Auto-shown by `results` command.

### Optimization (`bayesian` command)

**Bayesian optimization** (`backtest.py: bayesian_optimization`) — GP surrogate + Expected Improvement acquisition. Finds optima in ~50-100 evaluations vs thousands for grid search. Uses Latin Hypercube initial sampling, minimal Gaussian Process (squared exponential kernel), and EI acquisition on 1000 random candidates per iteration. Same 7 parameters, same objective, same `_apply_best_settings()` postprocessing.

### Cross-Validation (`purgedcv`, `cpcv` commands)

- **Purged Walk-Forward CV** (`backtest.py: purged_walk_forward_cv`) — k chronological folds with embargo gap. Resets Elo, replays training games, skips embargo, predicts test fold. Reports mean/std across folds. Accuracy variance >3% = fragile model.
- **CPCV** (`backtest.py: combinatorial_purged_cv`) — All C(k, k_test) train/test combinations. With k=5, k_test=2: 10 paths. Tighter confidence intervals. If >90% of paths show >65% accuracy, model is robust.

### Overfitting Detection (`pbo`, `montecarlo` commands)

- **PBO** (`backtest.py: probability_of_backtest_overfitting`) — Symmetric cross-validation on grid search trial population. PBO > 0.5 = optimization is likely overfit. Requires `grid` to have been run.
- **Monte Carlo Permutation** (`backtest.py: monte_carlo_permutation_test`) — Shuffles home/away scores randomly, re-runs backtest N times (default 500), computes p-value. p > 0.05 = model not statistically significant. Takes ~8 min for 500 permutations.

### Calibration (`rollingcal`, `betacal`, `conformal` commands)

- **Rolling Origin Recalibration** (`backtest.py: rolling_origin_recalibration`) — Expanding-window Platt fitting every 50 games. Gives truly out-of-sample calibrated metrics. Compares with single-pass calibration.
- **Beta Calibration** (`platt.py: fit_beta_scaler / apply_beta`, `backtest.py: run_beta_calibration`) — 3-parameter `logit(p_cal) = c + a*log(p) - b*log(1-p)`. If a != b, miscalibration is asymmetric and beta is better than Platt.
- **Conformal Prediction** (`metrics.py: conformal_prediction_set`, `backtest.py: conformal_analysis`) — Distribution-free prediction sets at 90%, 95%, 80% coverage. Singleton = confident, Both = uncertain, Empty = overconfident.

### Ensemble Analysis (`enhanced decay`, `shap` commands)

- **Time-Decayed Weighting** (`enhanced_model.py: run_enhanced_backtest(time_decay=True)`) — Elo weight transitions 95% → 70% over the season. Acknowledges XGBoost features are noisy early but reliable after 500+ games.
- **SHAP Feature Importance** (`enhanced_model.py: shap_feature_importance`) — XGBoost native `pred_contribs`. Shows mean absolute SHAP per feature. Reveals if XGBoost is echoing Elo or adding signal. Auto-runs after `enhanced`; also standalone `shap` command using saved features.

### Structural Analysis (`sliding`, `convergence` commands)

- **Sliding Window** (`backtest.py: sliding_window_backtest`) — Heavy regression (67%) every N games vs standard expanding window. Determines if old data helps or hurts. If sliding wins, consider more aggressive season regression.
- **Convergence Analysis** (`backtest.py: elo_convergence_analysis`) — Chunks predictions by N-game windows, reports per-chunk accuracy. Finds burn-in period where ratings stabilize. Suggested exclusion window for honest metrics.

### Position Sizing (`kelly` command)

**Kelly Criterion Backtest** (`backtest.py: kelly_criterion_backtest`) — Simulates fractional Kelly position sizing on `nba_backtest_predictions.csv`. Default quarter-Kelly. Reports final bankroll, return %, max drawdown, annualized Sharpe, win rate. Connects model quality directly to P&L.

### Complete Model Validation Workflow (run in order)

This is the recommended sequence for thoroughly validating and tuning the model. Each step builds on the previous one's output. Skip steps that aren't relevant to your current question, but follow the order for any steps you do run.

**Phase 1: Baseline & Diagnostics**

1. `backtest` — Establish baseline. Note accuracy, log loss, Brier, ECE, MCE, BSS. These are the numbers to beat. The Platt scaler is fitted here and all subsequent predictions use it.
2. `convergence` — How many games before Elo ratings stabilize? If burn-in is 200+ games, your headline accuracy is understated. Know this before interpreting any other metric.
3. `sliding` — Is old data helping or hurting? If sliding window wins, your season regression factor (currently 33%) may be too mild. If expanding wins (typical), full history is valuable and you can proceed with confidence.

**Phase 2: Parameter Optimization**

4. `grid` — Coarse search. Wide ranges, large steps. Produces `nba_grid_search.csv` with thousands of scored parameter combos. Don't trust the "best" yet — it hasn't been validated.
5. `pbo` — Immediately after grid search. Is the best overfit? PBO > 0.5 = yes, widen your ranges or accept that the data doesn't distinguish between many parameter combos. PBO < 0.3 = the optimum is real.
6. `results` — Check DSR (auto-shown). DSR < 1.96 = the best score isn't statistically distinguishable from random picks among your grid combos. If both PBO and DSR look bad, the grid search found noise, not signal.
7. `genetic` — Fine-tune with tighter bounds around the grid sweet spot. Or use `bayesian` if you want fewer evaluations (~80 vs thousands). Both auto-save best settings and refit Platt.

**Phase 3: Cross-Validation**

8. `purgedcv` — Does the optimized model hold up across 5 chronological folds with embargo? If accuracy std > 3%, the model is fragile — the params are fit to a specific slice of history. Consider more conservative settings.
9. `cpcv` — Combinatorial validation. All 10 paths through 5 blocks. If any path drops below 55% accuracy, there's path-dependent overfitting. The percentage of paths above 65% is your robustness score.
10. `montecarlo` — Is the model genuinely skilled? p-value < 0.05 = yes, the model's accuracy is statistically unlikely to arise from random chance. This is the definitive skill test. Takes ~8 minutes.

**Phase 4: Ensemble & Features**

11. `enhanced` — Train XGBoost on top of optimized Elo. SHAP analysis runs automatically at the end. Check: is `elo_prob` >50% of SHAP signal? If so, XGBoost is mostly echoing Elo and the 20% XGB weight may not be worth the complexity.
12. `shap` — Re-examine feature importance standalone if needed. Look for features with near-zero SHAP — they're candidates for removal to reduce overfitting.
13. `enhanced decay` — Compare time-decayed (Elo 95%→70%) vs fixed 80/20. If decay wins on Brier, use it — it means early-season XGBoost was adding noise.

**Phase 5: Calibration Refinement**

14. `rollingcal` — Is your single-pass Platt calibration honest? Rolling origin gives truly out-of-sample calibrated Brier. If rolling is worse, your Platt scaler is overfitting slightly to the full sample (common, usually by <0.005 Brier).
15. `betacal` — Is there asymmetric miscalibration? If `a != b` significantly (difference > 0.3), the model is overconfident in one direction but not the other. Beta calibration fixes this; Platt cannot. If a ≈ b, Platt is sufficient.
16. `conformal` — What's the model's actual confidence? Look at singleton percentage (games where only one outcome is in the prediction set). Higher singleton % with good coverage = model is both accurate and well-calibrated. High "both" % = model is uncertain about many games (maybe don't bet those).

**Phase 6: P&L Simulation**

17. `kelly` — What would optimal position sizing produce? The Sharpe ratio and max drawdown are the numbers that matter for real trading. If max drawdown > 50% even with quarter-Kelly, the edge is too thin for aggressive sizing. If Sharpe > 1.0, there's a tradeable edge.

### Decision Framework After Validation

After running the full workflow, use these thresholds to decide next steps:

| Metric | Good | Marginal | Bad | Action if bad |
|--------|------|----------|-----|---------------|
| Accuracy | >66% | 63-66% | <63% | Revisit Elo adjusters, check data quality |
| ECE | <0.03 | 0.03-0.08 | >0.08 | Refit Platt, try beta calibration |
| BSS vs 50% | >0.08 | 0.04-0.08 | <0.04 | Model has weak discriminative power |
| PBO | <0.3 | 0.3-0.5 | >0.5 | Grid search overfit — use wider ranges or fewer params |
| DSR | >1.96 | 1.0-1.96 | <1.0 | Best params are noise — simplify model |
| Monte Carlo p | <0.01 | 0.01-0.05 | >0.05 | Model has no statistically significant skill |
| Purged CV std | <2% | 2-3% | >3% | Model is fragile to training data selection |
| CPCV paths <55% | 0% | 1-10% | >10% | Path-dependent overfitting |
| Kelly Sharpe | >1.0 | 0.5-1.0 | <0.5 | Edge too thin for real trading |
| Kelly max DD | <30% | 30-50% | >50% | Size down (use 10%-Kelly instead of 25%) |

## Comprehensive Workflow

### Daily Prediction Workflow

1. **Launch**: `python main.py` — auto-downloads fresh data (games, players, injuries), builds Elo model, runs baseline backtest, fits Platt scaler
2. **Check model**: `settings` to verify parameters, `all` to see team rankings, `injuries` to review injury impact
3. **Make predictions**: type a team name (e.g. `Lakers`) → enter opponent → specify home/away (`a`/`b`/`n`) → see calibrated win probability with injury impact and key player stats
4. **Log position**: type `y` → enter contract count, price, optional notes → position saved to `predicts_lots.csv`
5. **Publish**: `today` or `tomorrow` generates Blogger HTML (`today_nba_predictions.html`) + plain text for all scheduled games
6. **Monitor**: `live` shows real-time scores for open positions; `mark` updates current market prices
7. **Settle**: `resolve` to manually settle (win/loss prompt), `autoresolve` to auto-settle from live final scores, `sell` to exit early at a price
8. **Review**: `predicts` shows full P&L ledger with realized/unrealized/marked totals, win rate, ROI; `chart` saves monthly P&L bar chart

### Model Tuning Workflow

1. **Baseline**: run `backtest` — note accuracy, log loss, Brier, ECE, MCE, BSS (all auto-reported)
2. **Diagnostics**: run `convergence` to find burn-in period, `sliding` to check if old data helps
3. **Coarse search**: run `grid` with wide ranges and large steps to identify promising parameter regions
4. **Validate search**: run `pbo` to check overfitting, `results` to check DSR significance
5. **Fine-tune**: run `genetic` or `bayesian` with tighter bounds around the grid search sweet spot
6. **Cross-validate**: run `purgedcv` for fold stability, `cpcv` for combinatorial robustness
7. **Significance test**: run `montecarlo` — if p > 0.05, model has no proven edge
8. **Train ensemble**: run `enhanced` (SHAP auto-runs), then `enhanced decay` to compare
9. **Calibration**: run `rollingcal` for OOS calibration, `betacal` for asymmetry, `conformal` for coverage
10. **P&L simulation**: run `kelly` to connect model quality to bankroll trajectory
11. **Iterate**: adjust individual params with `set k=8.53`, `set home=32`, etc. — always rerun `backtest` after to refit the Platt scaler

See "Complete Model Validation Workflow" above for detailed guidance on interpreting each step's output and what to do when metrics fall short.

After any parameter change, the Platt scaler must be refitted (the CLI warns about this). The optimizer auto-saves best settings and refits Platt after completion.

### Position Management

Predicts contracts are $1 binary options. The ledger tracks:
- **Entry**: contracts count, price per contract, 2% entry fee on potential payout ($1 per contract)
- **Mark-to-market**: `mark` updates current price; `predicts` shows unrealized P&L based on marks
- **Exit**: `sell` for partial/full exit at a price (with 2% exit fee); `resolve` for binary win/loss settlement
- **Invert**: `invert` flips a position's direction (swaps predicted winner, inverts probability) without changing cost basis
- **Auto-resolve**: matches open positions against today's final scores via NBA CDN; toggle with `autoresolve on/off` (runs on startup when enabled)

### Data Refresh

- `refresh` deletes cached game/player/ratings/scaler files and re-downloads everything, then rebuilds the model
- Data caches expire after 6 hours for games/players, 4 hours for injuries, 60 seconds for live scores
- Cache staleness checked by `config.is_cache_stale()` — files under 500 bytes are always considered stale

## CLI Commands

Enter a team name to start a prediction. Core commands:

**Data & Display**: `all`, `refresh`, `players`, `settings`, `injuries`, `today`/`html`, `tomorrow`

**Data Enrichment**: `odds`, `kalshi`, `weather`

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

**Elo Settings** (39 params, type `set` to see all):
- `set k=8.53`, `set home=32`, `set boost=48`, `set rest=18`, `set b2b=0`
- `set travel=12`, `set pace=27`, `set sos=2`, `set div=10`
- `set altitude=0.04`, `set playoff=0.71`
- `set kelly=quarter`, `set balance=1000`, `set autoresolve=true`

`help` for overview, `help <command>` for details, `help advanced` for all validation commands, `quit` to exit.

## Conventions

- Python 3.8+ compatibility (no walrus operators, no `match` statements)
- No test framework -- validation is via `backtest` command and `accuracy_test.py` (run manually: `python accuracy_test.py [label]`)
- All data files use `nba_` prefix; temp files use `temp_` prefix (both gitignored)
- Settings are tuned via `set param=value` CLI command or optimizer; changes require `backtest` rerun to refit Platt scaler
- Color output uses `color_helpers.py` wrappers (`cok`, `cerr`, `cwarn`, `chi`, `cdim`, `cbold`), never raw colorama
- Optimizer objective is `LogLoss * 8 + Brier * 40` -- this weighting is intentional and should not be changed casually
- Team lookup is fuzzy: `NBAElo.find_team()` accepts full names, abbreviations, partial matches, and close matches via `difflib.get_close_matches`
- Mega-ensemble model switches are per-sport and saved to `nba_mega_settings.json`
- All data sources must be completely free (no paid APIs)

## Dependencies
- pandas, numpy, scipy, colorama, tqdm, xgboost, requests, nba_api, matplotlib
- Optional: torch (for MLP/LSTM), hmmlearn, filterpy, lightgbm, catboost, networkx
