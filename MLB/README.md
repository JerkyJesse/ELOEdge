# MLB SharpStack -- 36-Model Mega-Ensemble Prediction System

MLB game prediction engine combining Elo ratings, XGBoost, and a 36-model mega-ensemble with meta-learner stacking. Includes a full contract trading ledger for Kalshi/$1 binary prediction markets.

---

## Quick Start

```bash
cd Claude/MLB
pip install -r requirements.txt
python main.py
```

On first launch the system automatically:
1. Downloads 2 years of game data from the MLB Stats API
2. Downloads batting and pitching leader stats
3. Fetches injury reports from ESPN
4. Builds the Elo model with current settings
5. Loads the Platt calibration scaler (if previously fitted)

Run `backtest` immediately after first launch to fit the calibration scaler. Without it, probabilities are uncalibrated.

---

## How Predictions Work

Type any team name (fuzzy matched) to start a prediction flow:

```
> Yankees
Opponent team: Red Sox
Home team? (a = first team home, b = second, n = neutral): a
```

The system returns a calibrated win probability incorporating Elo ratings, player strength, starting pitcher quality, injuries, park factors, weather, rest days, travel fatigue, and more.

Type `y` after the prediction to log a contract position in the trading ledger.

---

## Architecture -- Three Prediction Layers

### Layer 1: Elo Model (elo_model.py)

Base team ratings (starting at 1500) with 27+ adjustment factors:

- Home field advantage (default 37.63 Elo, ~55.3% implied)
- Starting pitcher quality (per-pitcher Elo, 700+ pitchers tracked, K_PITCHER=6)
- Player strength boost (z-scored team batting+pitching composite)
- Park factors (30 team-specific multipliers from FanGraphs 5-year data)
- Margin of victory (logarithmic, capped at mov_cap=19.9)
- Rest days, travel distance, altitude, back-to-back penalty
- Series adaptation, interleague factor, bullpen quality
- Opponent pitcher factor, east travel penalty, road trip/homestand effects
- Form (last 15 games), strength of schedule, pace/run-environment
- Season phase, mean reversion, K-decay, surprise-K adaptive learning
- Playoff detection (October) with reduced HCA

### Layer 2: XGBoost Ensemble (enhanced_model.py)

An 80/20 Elo/XGBoost blend built on 96 rolling features per game:

- TeamTracker maintains 15-game rolling windows per team
- Features: Elo prob, Elo diff, player diff, runs scored/allowed, win%, margins, rest, Pythagorean expectation, streaks, consistency, trend
- Walk-forward training with no leakage: Elo-only for the first 200 games while features accumulate
- Optional time-decay mode: transitions from 95% Elo early season to 70% Elo late season
- SHAP feature importance via XGBoost native pred_contribs

### Layer 3: 36-Model Mega-Ensemble (mega_predictor.py)

All 36 base models feed into a meta-learner (XGBoost, ridge, or logistic regression) that produces a bounded probability adjustment clamped to +/- max_adj (default 0.08) on top of the Elo+XGBoost probability.

Walk-forward backtest: first 300 games are Elo-only while the meta-learner accumulates training data. ML models retrain every 80 games.

### Platt Calibration (platt.py)

Final probabilities pass through a logistic regression calibrator (Platt scaling) fitted during backtest. Also supports isotonic regression and 3-parameter beta calibration. Season regression pulls all ratings 33% toward the mean at year boundaries.

---

## The 36 Models

### Tier 0 -- Core (always on)
| Model | Description |
|-------|-------------|
| Elo | Elo ratings with 24+ adjustment factors |
| XGBoost | XGBoost ensemble with 96 rolling features |

### Tier 1 -- Proven
| Model | Description |
|-------|-------------|
| HMM | Hidden Markov Model (hot/cold team states) |
| Kalman | Kalman Filter (latent strength estimation) |
| PageRank | PageRank + HITS network/graph analysis |
| LightGBM | Leaf-wise gradient boosting |
| CatBoost | Ordered gradient boosting |
| MLP | Multi-layer perceptron neural network |
| LSTM | Long Short-Term Memory sequential patterns |
| SVM | Support Vector Machine (RBF kernel + Platt scaling) |

### Tier 2 -- Exotic / Physics-Inspired
| Model | Description |
|-------|-------------|
| GARCH | Time-varying volatility + Lyapunov/Hurst exponents |
| Fourier | Fourier/wavelet cycle detection |
| Survival | Survival analysis (streak hazard rates) |
| Copula | Offense/defense joint dependency modeling |
| Fibonacci | Fibonacci retracement (EMA support/resistance levels) |
| EVT | Extreme Value Theory (Generalized Pareto tail risk) |

### Tier 3 -- Information, Physics, and Pattern Detection
| Model | Description |
|-------|-------------|
| InfoTheory | Shannon entropy + KL divergence |
| Momentum | Newtonian momentum/inertia model |
| Markov | Markov chain transition matrices |
| Clustering | k-Means team archetypes |
| GameTheory | Nash equilibrium + style matchups |
| Benford | Benford's Law chi-squared scoring anomaly detection |
| Moedim | Prophetic cycle timing system |

### Tier 4 -- Classical Rating Systems
| Model | Description |
|-------|-------------|
| Poisson | Poisson/Dixon-Coles score distributions |
| Glicko | Glicko-2 uncertainty-aware ratings |
| BradleyTerry | Bradley-Terry MLE paired comparison |
| MonteCarlo | Monte Carlo simulation (3000 sims default) |
| RandomForest | Random Forest bagging for diversity |

### Tier 5 -- Sports-Specific
| Model | Description |
|-------|-------------|
| SRS | Simple Rating System (margin + strength of schedule) |
| Colley | Colley Matrix bias-free ranking |
| Log5 | Bill James Log5 head-to-head formula |
| PythagenPat | PythagenPat with dynamic Pythagorean exponent |
| ExpSmoothing | Exponential smoothing trend tracking |
| MeanReversion | Mean reversion with Bollinger bands |

### Tier 6 -- Data Enrichment
| Model | Description |
|-------|-------------|
| Weather | Temperature, wind, precipitation impact |
| Odds | Market odds / closing line value tracking (off by default) |

---

## Commands Reference

### Predictions and Display

| Command | What it does |
|---------|-------------|
| `<team name>` | Start a prediction (fuzzy match on any team name or abbreviation) |
| `all` | Show current Elo ratings for all 30 teams |
| `today` / `html` | Generate Blogger-ready HTML prediction table for today's games |
| `tomorrow` | Generate prediction table for tomorrow's games |
| `players` | Show top batting/pitching leaders |
| `injuries` | Show ESPN injury report with Elo impact scoring |
| `injuries set <team> <players>` | Manually mark players as injured |
| `settings` | Show current Elo model parameters |

### Data

| Command | What it does |
|---------|-------------|
| `refresh` | Force re-download all game data, player stats, rebuild model |
| `odds` | Show current moneyline odds from The Odds API |
| `kalshi` | Toggle auto-Kalshi odds fetching during predictions |
| `kalshi odds` / `kalshi all` | Show all current Kalshi game markets |
| `kalshi on` / `kalshi off` | Enable/disable auto-Kalshi during prediction flow |
| `weather` | Show weather report and impact for a home team's stadium |
| `advstats` / `statcast` | Download and display FanGraphs/Statcast advanced stats |

### Backtesting

| Command | What it does |
|---------|-------------|
| `backtest` | Walk-forward backtest + fit Platt calibration scaler |
| `enhanced` | Run Elo+XGBoost ensemble backtest (80/20 blend) |
| `enhanced decay` | Enhanced backtest with time-decayed Elo weighting |
| `shap` | SHAP feature importance analysis on XGBoost model |

### Elo Optimization

| Command | What it does |
|---------|-------------|
| `grid` | Grid search over 7 Elo parameters |
| `genetic` | Differential evolution optimization |
| `bayesian` | Gaussian Process + Expected Improvement optimization |
| `autoopt` | Automatic grid -> genetic -> bayesian pipeline |
| `superopt` | Exhaustive 7-phase optimization (takes hours) |
| `singleopt` | Coordinate descent, one parameter at a time |
| `results` | Show best parameters found across all optimizers |

### Validation and Diagnostics

| Command | What it does |
|---------|-------------|
| `purgedcv` | Purged walk-forward cross-validation (k chronological folds) |
| `cpcv` | Combinatorial purged cross-validation (all fold combinations) |
| `pbo` | Probability of Backtest Overfitting (> 0.5 means likely overfit) |
| `montecarlo` | Monte Carlo permutation test (p > 0.05 means no proven edge) |
| `convergence` | Chunked accuracy analysis to find burn-in period |
| `sliding` | Sliding window backtest with heavy regression |
| `rollingcal` | Rolling origin Platt recalibration every 50 games |
| `conformal` | Distribution-free prediction sets at 80/90/95% coverage |
| `betacal` | 3-parameter asymmetric beta calibration |
| `kelly` | Kelly criterion bankroll simulation (default quarter-Kelly) |

### Mega-Ensemble

| Command | What it does |
|---------|-------------|
| `mega` | Run full mega-ensemble backtest with all enabled models |
| `mega optimize` | 7-phase per-model optimization (takes hours) |
| `mega quick` | Quick optimization (Phases 0-1 only: baseline + solo tuning) |
| `mega tune` | Same as `mega quick` |
| `mega tournament` | Head-to-head model tournament (Phase 2) |
| `mega ablation` | Test each model's individual contribution, auto-prune weak ones |
| `mega models` | Show all 36 models with ON/OFF status by tier |
| `mega on <model>` | Enable a model (or `mega on all`) |
| `mega off <model>` | Disable a model |
| `mega settings` | Show all mega-ensemble parameter values |
| `mega set <param>=<value>` | Set a mega parameter (e.g., `mega set max_adj=0.10`, `mega set meta=ridge`) |

### Elo Settings (39 tunable parameters)

Type `set` with no arguments to see all available parameters.

```
set k=2.62          K-factor (learning rate)
set home=37.63      Home field advantage (Elo points)
set starter=88.08   Starting pitcher quality boost
set player=19.53    Team player strength boost
set pace=41.87      Run environment / pace factor
set rest=2.80       Rest days impact
set travel=13.71    Travel fatigue factor
set b2b=26.51       Back-to-back fatigue penalty
set altitude=12.48  Coors Field altitude bonus
set division=6.64   Divisional familiarity
set form=7.08       Recent form weight (last 15 games)
set sos=2.45        Strength of schedule
set interleague=2.04  AL vs NL adjustment
set series=3.92     Multi-game series familiarity
set bullpen=6.43    Bullpen quality impact
set opp_pitcher=18  Opponent starting pitcher quality
set mean_reversion=34.23  Regression after extreme results
set kelly=quarter   Kelly fraction for position sizing
set autoresolve=true  Auto-settle trades on startup
```

Always run `backtest` after changing parameters to refit the Platt scaler.

### Trading Ledger

| Command | What it does |
|---------|-------------|
| `predicts` / `summary` | Show full contract ledger with realized/unrealized P&L |
| `balance` | Show current bankroll balance |
| `deposit` | Deposit cash to shared cross-sport bankroll |
| `withdraw` | Withdraw cash from bankroll |
| `resolve` | Manually settle a finished contract (win/loss) |
| `sell` | Exit a position early at a specified price |
| `mark` | Update mark-to-market price on open positions |
| `invert` | Flip direction of an open position |
| `chart` | Generate monthly realized P&L bar chart |
| `live` | Live score tracker with open trade status (60s refresh) |
| `autoresolve` | Manually trigger auto-settlement of finished games |
| `autoresolve on/off` | Toggle automatic resolution on startup |
| `portfolio` | Cross-sport portfolio overview |

---

## Key Files

### Core System
| File | Purpose |
|------|---------|
| `main.py` | CLI entry point; `dispatch()` routes all commands |
| `config.py` | Constants, 30 MLB teams, divisions, settings I/O |
| `elo_model.py` | `MLBElo` class with ratings, predictions, 24+ adjusters, park factors |
| `build_model.py` | Model construction with season regression |
| `data_games.py` | Game data download via MLB Stats API |
| `data_players.py` | Player stats download and team composite scoring |
| `backtest.py` | All backtesting and optimization (~2100 lines) |
| `enhanced_model.py` | XGBoost ensemble, SHAP, TeamTracker |
| `platt.py` | Calibration (Platt, isotonic, beta, season regression) |
| `metrics.py` | Log loss, Brier, ECE, MCE, BSS, conformal metrics |
| `elo_set_handler.py` | Handler for `set param=value` commands |
| `single_param_opt.py` | Coordinate descent optimizer |
| `cache_utils.py` | Season-aware smart caching for all API data |

### Mega-Ensemble System
| File | Purpose |
|------|---------|
| `mega_predictor.py` | `MegaPredictor` class for live predictions using all 36 models |
| `mega_backtest.py` | Walk-forward backtest for the full mega-ensemble + meta-learner training |
| `mega_optimizer.py` | 7-phase per-model optimizer (solo, tournament, ablation, DE, validation) |
| `mega_config.py` | Model registry, on/off switches, per-model hyperparameters |
| `meta_learner.py` | `MetaLearner` class: XGBoost/ridge/logistic stacking combiner |

### Individual Model Files
| File | Models |
|------|--------|
| `hmm_model.py` | Hidden Markov Model |
| `kalman_model.py` | Kalman Filter |
| `network_model.py` | PageRank + HITS |
| `gbm_models.py` | LightGBM + CatBoost |
| `nn_models.py` | MLP + LSTM (requires PyTorch) |
| `random_forest_model.py` | Random Forest |
| `svm_model.py` | SVM classifier |
| `volatility_model.py` | GARCH + Lyapunov + Hurst |
| `signal_model.py` | Fourier/wavelet |
| `survival_model.py` | Survival analysis |
| `copula_model.py` | Copula dependency model |
| `information_theory_model.py` | Shannon entropy + KL divergence |
| `momentum_model.py` | Newtonian momentum/inertia |
| `markov_chain_model.py` | Markov chain transitions |
| `clustering_model.py` | k-Means team archetypes |
| `game_theory_model.py` | Nash equilibrium |
| `poisson_model.py` | Poisson/Dixon-Coles |
| `glicko_model.py` | Glicko-2 |
| `bradley_terry_model.py` | Bradley-Terry MLE |
| `monte_carlo_model.py` | Monte Carlo simulation |
| `classic_models.py` | SRS, Colley, Log5, PythagenPat, ExpSmoothing, MeanReversion |
| `fibonacci_model.py` | Fibonacci retracement |
| `evt_model.py` | Extreme Value Theory |
| `benford_model.py` | Benford's Law |
| `moedim_model.py` | Moedim prophetic cycle timing |

### Data Enrichment
| File | Purpose |
|------|---------|
| `odds_tracker.py` | Live odds via The Odds API (free tier, 500 req/month) |
| `weather.py` | Weather impact via Open-Meteo API (free, no key needed) |
| `kalshi.py` | Kalshi public API for live contract prices and auto-Kelly |
| `advanced_stats.py` | Statcast/FanGraphs via pybaseball (xwOBA, barrel rate, xERA) |
| `injuries.py` | ESPN injury report with Elo impact scoring |

### Trading and Display
| File | Purpose |
|------|---------|
| `predict_ledger.py` | Contract ledger (add, sell, resolve, mark, invert) with 2% fees |
| `live_scores.py` | Live MLB scores via Stats API, 60s refresh loop |
| `auto_resolve.py` | Auto-settle finished trades against live final scores |
| `html_generator.py` | Blogger-ready HTML prediction tables |
| `help_system.py` | Help text for all commands |
| `color_helpers.py` | Terminal color formatting wrappers |
| `accuracy_test.py` | Quick walk-forward accuracy test |

### Standalone Utility Scripts
| File | Purpose |
|------|---------|
| `master_optimize.py` | 10-phase master optimization runner (all methods, hours) |
| `run_optimize.py` | Standalone optimization runner |
| `accuracy_optimize.py` | Accuracy-focused optimization |
| `run_enhanced_all.py` | Standalone enhanced backtest runner |
| `quick_optimizer.py` | Quick parameter sweep |
| `sweep_enhanced.py` | Enhanced parameter sweep |

---

## Optimization

### In-CLI Optimization

Six built-in Elo optimizers, all using the objective `score = -(LogLoss * 8 + Brier * 40)`:

1. `grid` -- Cartesian product over 7 parameters with configurable ranges
2. `genetic` -- `scipy.optimize.differential_evolution` over the same 7 parameters
3. `bayesian` -- Gaussian Process surrogate with Expected Improvement acquisition (50-100 evaluations)
4. `autoopt` -- Automatic pipeline: grid -> genetic -> bayesian, applies best result
5. `superopt` -- Exhaustive 7-phase optimization over 9 parameters (takes hours)
6. `singleopt` -- Coordinate descent, one parameter at a time with fine-grained refinement

After any optimizer completes, the winning parameters are saved to `mlb_elo_settings.json` and the Platt scaler is refit.

### master_optimize.py (Standalone, 10+ Phases)

The master optimizer runs every optimization method sequentially. Uses an accuracy-first objective: `(100 - accuracy) + brier * 5.0`.

```bash
python master_optimize.py                # run all phases
python master_optimize.py --skip 6       # skip phase 6
python master_optimize.py --only 1 2     # run only phases 1 and 2
```

| Phase | Method |
|-------|--------|
| 1 | Quick Optimizer (12-param DE, accuracy focus) |
| 2 | Accuracy Optimize (two-phase DE, 9 params) |
| 3 | Auto Optimize (grid -> genetic -> bayesian pipeline) |
| 4 | Super Optimize (exhaustive multi-round) |
| 5 | Coordinate Descent (single-param sweeps) |
| 6 | Mega Backtest (all 36 models, walk-forward) |
| 7 | Mega Quick Optimize (Phases 0-1) |
| 8 | Mega Full Optimize (all 7 mega phases) |
| 9 | Mega Ablation (test each model's contribution) |
| 10 | Final backtest with Platt fitting |
| 11 | Summary |

### Mega-Ensemble Optimization (mega_optimizer.py)

Seven-phase per-model optimization:

- Phase 0: Elo-only baseline to establish the floor
- Phase 1: Per-model solo optimization (Elo + one model at a time, coordinate descent)
- Phase 2: Head-to-head tournament (pairs, triples, top-N combinations)
- Phase 3: Meta-learner + global tuning (max_adj, meta_model, retrain_every, min_train, window)
- Phase 4: Combined DE fine-tuning (differential evolution over top numeric parameters)
- Phase 5: Final ablation (test each model's contribution with fully tuned parameters)
- Phase 6: Validation (run best config 5 times, report stability)

---

## Kalshi Integration

The system integrates with Kalshi prediction markets for real-time contract pricing and edge detection.

```
kalshi              Toggle auto-Kalshi on/off
kalshi on           Enable auto-fetch during prediction flow
kalshi off          Disable auto-fetch
kalshi odds         Show all current Kalshi MLB game markets
kalshi all          Same as kalshi odds
kalshi help         Show Kalshi subcommands
```

When `auto_kalshi` is enabled, the prediction flow automatically fetches Kalshi contract prices and compares them to the model's probability, flagging edge opportunities.

---

## Data Sources

All data sources are free. No paid API keys required for core functionality.

| Source | Data | Cache TTL |
|--------|------|-----------|
| MLB Stats API | Game results (2 years), live scores, schedules | 6 hours |
| MLB Stats API | Batting leaders (top 200), pitching leaders (top 150) | 6 hours |
| ESPN JSON API | Injury reports with status (Out, IL, Doubtful, etc.) | 4 hours |
| Open-Meteo API | Temperature, wind, precipitation for game stadiums | 2 hours |
| The Odds API | Moneyline odds from multiple sportsbooks (free tier, 500 req/month) | Per-request |
| Kalshi Public API | Live prediction market contract prices | Per-request |
| FanGraphs/Statcast (via pybaseball) | xwOBA, barrel rate, xERA, advanced stats | 6 hours |

---

## Configuration Files

### mlb_elo_settings.json

Stores all 39 tunable Elo parameters. Modified via `set param=value` commands or by running any optimizer. Key parameters:

- `k` (2.62): K-factor / learning rate
- `home_advantage` (37.63): Home field advantage in Elo points
- `starter_boost` (88.08): Starting pitcher quality impact
- `player_boost` (19.53): Team batting+pitching composite impact
- `pace_factor` (41.87): Run environment adjustment
- `mean_reversion` (34.23): Regression after extreme results
- `season_regress` (0.33): 33% pull toward mean at season boundaries

### mlb_mega_settings.json

Controls the mega-ensemble:

- `model_switches`: Per-model on/off toggles (36 models)
- Per-model hyperparameters (e.g., `mc_sims` for Monte Carlo, `kalman_process_noise`)
- Meta-learner configuration: model type (`xgboost`, `ridge`, `logistic`), `max_adj`, `retrain_every`, `min_train`, `window`

Modify via `mega set <param>=<value>` or `mega on/off <model>`.

---

## Dependencies

Required:
```
pandas, numpy, scipy, colorama, tqdm, xgboost, requests, MLB-StatsAPI, matplotlib
```

Optional:
```
torch          -- for MLP and LSTM neural network models
pybaseball     -- for Statcast/FanGraphs advanced stats
lightgbm       -- for LightGBM model
catboost       -- for CatBoost model
```

---

## Daily Workflow

1. `python main.py` -- auto-downloads fresh data, builds model
2. `backtest` -- fit calibration scaler (do this at least once)
3. Type a team name -- enter opponent, specify home/away, get calibrated probability
4. `y` to log a contract position
5. `today` to generate Blogger HTML for all scheduled games
6. `live` to monitor real-time scores on open positions
7. `resolve` or `autoresolve` to settle finished games
8. `predicts` and `chart` for P&L review

## Tuning Workflow

1. `backtest` -- establish baseline (accuracy, log loss, Brier)
2. `convergence` -- find burn-in period; `sliding` -- check if old data helps
3. `grid` or `autoopt` -- coarse parameter search
4. `pbo` -- check for overfitting; `results` -- check significance
5. `genetic` or `bayesian` -- fine-tune around best region
6. `purgedcv` and `cpcv` -- cross-validation stability
7. `montecarlo` -- statistical significance (p < 0.05 needed)
8. `enhanced` -- train XGBoost layer; `shap` for feature importance
9. `mega` -- train 36-model ensemble; `mega optimize` for full tuning
10. Always rerun `backtest` after parameter changes to refit Platt scaler
