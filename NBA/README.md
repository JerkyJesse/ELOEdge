# NBA Prediction System (SharpStack)

## Overview

NBA game prediction system that combines Elo ratings with a 36-model mega-ensemble to produce calibrated win probabilities for every NBA matchup. The system integrates a Predicts $1 binary contract trading ledger for position tracking and P&L analysis. It runs as an interactive CLI application -- no web server, no test framework, no build system.

NBA-specific design priorities: back-to-back game penalties, rest day differentials (critical in the 82-game grind), high player impact weighting (the NBA is star-driven -- a single player can swing a game), 10-game rolling feature windows, altitude factor for Denver (5,280 ft), pace mismatch adjustments for tempo differentials, and cross-year season handling (Oct--Jun).

---

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

On first launch the system will:
1. Download two years of game data via the `nba_api` package
2. Download player stats (points, assists, rebounds leaders)
3. Fetch injury reports from the ESPN API
4. Build Elo ratings for all 30 NBA teams
5. Run a walk-forward backtest to fit the Platt calibration scaler
6. Enter the interactive CLI loop

Type a team name (e.g. `Lakers`) to start a prediction. Type `help` for all commands.

---

## Commands

### Data and Display
| Command | Description |
|---------|-------------|
| `all` | Show all 30 teams ranked by Elo rating |
| `refresh` | Delete cached data and re-download everything, rebuild model |
| `players` | Show top player stats and composite scores |
| `settings` | Display current Elo parameter values |
| `injuries` | Show injury report with Elo impact per team |
| `injuries set <team> <player1>, <player2>` | Manually mark players as OUT |
| `today` / `html` | Generate Blogger-ready HTML predictions for today's games |
| `tomorrow` | Generate predictions for tomorrow's games |

### Data Enrichment
| Command | Description |
|---------|-------------|
| `odds` | Show live moneyline odds from The Odds API |
| `kalshi` | Show Kalshi prediction market contract prices |
| `weather` | Show weather report for today's games |

### Backtesting
| Command | Description |
|---------|-------------|
| `backtest` | Run walk-forward backtest, fit Platt scaler, report accuracy/log loss/Brier |
| `enhanced` | Run Elo + XGBoost ensemble backtest (80/20 blend) |
| `enhanced decay` | Run time-decayed ensemble (Elo 95% early to 70% late) |

### Elo Optimization
| Command | Description |
|---------|-------------|
| `grid` | Grid search over 7 Elo parameters |
| `genetic` | Differential evolution optimizer |
| `bayesian` | GP surrogate + Expected Improvement optimizer |
| `autoopt` | Automatic grid then genetic then bayesian pipeline |
| `superopt` | Exhaustive 7-phase optimization (takes hours) |
| `singleopt` | Coordinate descent (one parameter at a time) |
| `results` | Show optimization results with DSR significance |

### Validation
| Command | Description |
|---------|-------------|
| `purgedcv` | Purged walk-forward cross-validation (k folds with embargo) |
| `cpcv` | Combinatorial purged cross-validation (all path combinations) |
| `pbo` | Probability of backtest overfitting |
| `montecarlo` | Monte Carlo permutation test (500 shuffles, ~8 min) |
| `convergence` | Elo convergence analysis (find burn-in period) |
| `sliding` | Sliding window vs expanding window comparison |

### Calibration
| Command | Description |
|---------|-------------|
| `rollingcal` | Rolling-origin recalibration (out-of-sample Platt) |
| `betacal` | Beta calibration (3-parameter asymmetric) |
| `conformal` | Conformal prediction sets at 80/90/95% coverage |

### Analysis
| Command | Description |
|---------|-------------|
| `shap` | SHAP feature importance for XGBoost model |
| `kelly` | Kelly Criterion bankroll simulation (quarter-Kelly default) |

### Mega-Ensemble (36 Models)
| Command | Description |
|---------|-------------|
| `mega` | Run full mega-ensemble backtest |
| `mega optimize` | 7-phase per-model optimization (all phases, takes hours) |
| `mega quick` / `mega tune` | Quick optimization (Phases 0-1 only) |
| `mega tournament` | Head-to-head model comparison (Phase 2) |
| `mega ablation` | Ablation study, auto-prune weak models |
| `mega models` | Show all models with ON/OFF status by tier |
| `mega on <model>` / `mega off <model>` | Enable/disable individual models |
| `mega settings` | Show all mega parameter values |
| `mega set <param>=<value>` | Set mega parameter (e.g., `mega set max_adj=0.10`) |

### Trading
| Command | Description |
|---------|-------------|
| `predicts` / `summary` | Show full P&L ledger with win rate and ROI |
| `balance` | Show current account balance |
| `deposit` / `withdraw` | Add or remove cash from account |
| `resolve` | Manually settle open positions (win/loss) |
| `autoresolve` | Auto-settle finished trades from live final scores |
| `autoresolve on` / `autoresolve off` | Toggle auto-resolve on startup |
| `sell` | Exit a position early at a specified price |
| `mark` | Update mark-to-market prices on open positions |
| `invert` | Flip a position's direction without changing cost basis |
| `chart` | Generate monthly P&L bar chart |
| `live` | Show live scores for open positions (60s refresh) |
| `portfolio` | Cross-sport portfolio summary |

### Elo Settings
Type `set` to see all tunable parameters. Examples:
```
set k=8.53          set home=32         set boost=48
set rest=18         set b2b=0           set travel=12
set pace=27         set sos=2           set div=10
set altitude=0.04   set playoff=0.71    set form=5
set kelly=quarter   set balance=1000    set autoresolve=true
```

After any parameter change, run `backtest` to refit the Platt calibration scaler.

---

## Architecture

The system uses a three-layer prediction pipeline. Each layer adds refinement on top of the previous one.

### Layer 1: Elo Model

The `NBAElo` class in `elo_model.py` maintains Elo ratings for all 30 NBA teams. The base rating is 1500. Each game updates ratings using a configurable K-factor (default 8.53, tuned for the 82-game season). The model applies 24+ adjusters:

- **Home court advantage** (31.99 Elo, ~58% implied home win rate)
- **Player strength** (48.0 boost -- highest of any sport, NBA is star-driven)
- **Rest days** (18.21 factor -- critical for back-to-back detection)
- **Travel fatigue** (12.20 -- timezone-based distance calculation)
- **Pace mismatch** (26.53 -- significant for tempo mismatches)
- **Division rivalry** (10.0 -- familiarity adjustment)
- **Altitude** (0.04 -- Denver Nuggets at 5,280 ft; near-zero after optimization)
- **Strength of schedule** (2.0)
- **Recent form** (5.06 -- last 10 games weight)
- **Playoff HCA reduction** (0.71 -- ~29% reduced home advantage in playoffs)
- **Season regression** (33% pull toward mean at season boundaries)
- **Margin of victory** uses `log(max(1, abs(margin)) + 1)` for dampening

### Layer 2: Enhanced XGBoost Ensemble

The `enhanced_model.py` module builds an 80% Elo / 20% XGBoost blended model. The `TeamTracker` class maintains rolling 10-game windows per team, generating 102 features:

- Elo probability and rating differential
- Player strength differential
- Per-team rolling stats: PPG, PAPG, win%, margins, rest days
- Pythagorean win expectation (exponent 13.91)
- Win/loss streaks, scoring consistency, trend indicators

XGBoost is trained walk-forward (no leakage): Elo-only for the first 200 games, then XGBoost trained and retrained every 50 games. Parameters: `max_depth=5, eta=0.03, subsample=0.9, 300 rounds`.

### Layer 3: 36-Model Mega-Ensemble

The `mega_predictor.py` and `mega_backtest.py` modules implement a 36-model stacking ensemble. Each model produces a raw probability independently. A meta-learner (XGBoost, Ridge, or Logistic regression) combines all outputs into a bounded adjustment clamped to +/- `max_adj` (default 0.08) around the Elo anchor probability. All 36 models run in parallel via `ThreadPoolExecutor`.

### Calibration

`platt.py` applies Platt scaling (logistic regression on logit of raw probability) to produce well-calibrated final outputs. Isotonic regression and beta calibration are also available. The Platt scaler is fitted during the `backtest` command and must be refitted after any parameter change.

---

## The 36 Models

### Tier 0 -- Core
| Model | File | Description |
|-------|------|-------------|
| Elo | `elo_model.py` | Base team ratings with 24+ adjusters |
| XGBoost | `enhanced_model.py` | 102-feature gradient boosting ensemble |

### Tier 1 -- Proven
| Model | File | Description |
|-------|------|-------------|
| HMM | `hmm_model.py` | Hidden Markov Model (hot/cold team states) |
| Kalman | `kalman_model.py` | Kalman filter (latent strength estimation) |
| PageRank | `network_model.py` | PageRank + HITS network/graph analysis |
| LightGBM | `gbm_models.py` | LightGBM gradient boosting |
| CatBoost | `gbm_models.py` | CatBoost gradient boosting |
| MLP | `nn_models.py` | Multi-layer perceptron neural network |
| LSTM | `nn_models.py` | Long short-term memory recurrent network |

### Tier 2 -- Exotic
| Model | File | Description |
|-------|------|-------------|
| GARCH | `volatility_model.py` | GARCH volatility + Lyapunov + Hurst exponents |
| Fourier | `signal_model.py` | Fourier/wavelet cycle detection |
| Survival | `survival_model.py` | Survival analysis (streak hazard rates) |
| Copula | `copula_model.py` | Copula offense/defense joint dependency |

### Tier 3 -- Information and Physics
| Model | File | Description |
|-------|------|-------------|
| InfoTheory | `information_theory_model.py` | Shannon entropy + KL divergence |
| Momentum | `momentum_model.py` | Newtonian momentum/inertia model |
| Markov | `markov_chain_model.py` | Markov chain transition matrices |
| Clustering | `clustering_model.py` | k-Means team archetypes |
| GameTheory | `game_theory_model.py` | Nash equilibrium + style matchups |

### Tier 4 -- Classical Ratings
| Model | File | Description |
|-------|------|-------------|
| Poisson | `poisson_model.py` | Poisson/Dixon-Coles score distributions |
| Glicko-2 | `glicko_model.py` | Uncertainty-aware ratings |
| Bradley-Terry | `bradley_terry_model.py` | Maximum likelihood paired comparison |
| Monte Carlo | `monte_carlo_model.py` | Monte Carlo simulation (2000 sims default) |
| Random Forest | `random_forest_model.py` | Random Forest bagging diversity |

### Tier 5 -- Sports-Specific
| Model | File | Description |
|-------|------|-------------|
| SRS | `classic_models.py` | Simple Rating System |
| Colley | `classic_models.py` | Colley Matrix |
| Log5 | `classic_models.py` | Log5 matchup formula |
| PythagenPat | `classic_models.py` | Pythagorean win expectation |
| ExpSmoothing | `classic_models.py` | Exponential smoothing |
| MeanReversion | `classic_models.py` | Mean reversion model |

### Tier 6 -- Data Enrichment
| Model | File | Description |
|-------|------|-------------|
| Weather | `weather.py` | Open-Meteo weather impact |
| Odds | `odds_tracker.py` | The Odds API moneyline consensus |

### Tier 7 -- Additional
| Model | File | Description |
|-------|------|-------------|
| SVM | `svm_model.py` | SVM classifier (RBF kernel + Platt scaling) |
| Fibonacci | `fibonacci_model.py` | Fibonacci retracement (EMA-smoothed levels) |
| EVT | `evt_model.py` | Extreme Value Theory (Generalized Pareto tail risk) |
| Benford | `benford_model.py` | Benford's Law anomaly detection |

---

## Key Files

### Core System
| File | Purpose |
|------|---------|
| `main.py` | CLI entry point; `dispatch()` routes all commands |
| `config.py` | Constants, 30 NBA teams, 6 divisions, settings I/O |
| `elo_model.py` | `NBAElo` class -- ratings, predictions, 24+ adjusters |
| `build_model.py` | Model construction with season regression and altitude bonus |
| `data_games.py` | Game data download via `nba_api` |
| `data_players.py` | Player stats download + team scoring composite |
| `backtest.py` | Walk-forward backtest, grid/genetic/bayesian optimizers (~2100 lines) |
| `enhanced_model.py` | XGBoost ensemble + SHAP + TeamTracker (102 features) |
| `platt.py` | Platt scaling, isotonic regression, beta calibration |
| `metrics.py` | Log loss, Brier, ECE, MCE, BSS, conformal prediction |
| `elo_set_handler.py` | Shared handler for `set param=value` commands |
| `single_param_opt.py` | Coordinate descent optimizer (one param at a time) |
| `cache_utils.py` | Season-aware smart caching for all API data |
| `color_helpers.py` | Terminal color output wrappers (`cok`, `cerr`, `cwarn`, `chi`, `cdim`, `cbold`) |

### Mega-Ensemble
| File | Purpose |
|------|---------|
| `mega_predictor.py` | `MegaPredictor` class -- live predictions using all 36 models |
| `mega_backtest.py` | Walk-forward backtest for full mega-ensemble + meta-learner training |
| `mega_optimizer.py` | 7-phase per-model optimizer (solo, tournament, ablation, DE, validation) |
| `mega_config.py` | Model registry, on/off switches, per-model hyperparameters |
| `meta_learner.py` | `MetaLearner` class -- XGBoost/Ridge/Logistic stacking combiner |

### Individual Model Files
| File | Models |
|------|--------|
| `hmm_model.py` | Hidden Markov Model |
| `kalman_model.py` | Kalman Filter |
| `network_model.py` | PageRank + HITS |
| `gbm_models.py` | LightGBM + CatBoost |
| `nn_models.py` | MLP + LSTM |
| `random_forest_model.py` | Random Forest |
| `volatility_model.py` | GARCH + Lyapunov + Hurst |
| `signal_model.py` | Fourier + Wavelet |
| `survival_model.py` | Survival Analysis |
| `copula_model.py` | Copula |
| `information_theory_model.py` | Shannon Entropy + KL Divergence |
| `momentum_model.py` | Newtonian Momentum |
| `markov_chain_model.py` | Markov Chain |
| `clustering_model.py` | k-Means Clustering |
| `game_theory_model.py` | Nash Equilibrium |
| `poisson_model.py` | Poisson / Dixon-Coles |
| `glicko_model.py` | Glicko-2 |
| `bradley_terry_model.py` | Bradley-Terry MLE |
| `monte_carlo_model.py` | Monte Carlo Simulation |
| `classic_models.py` | SRS, Colley, Log5, PythagenPat, ExpSmoothing, MeanReversion |
| `svm_model.py` | SVM Classifier |
| `fibonacci_model.py` | Fibonacci Retracement |
| `evt_model.py` | Extreme Value Theory |
| `benford_model.py` | Benford's Law |
| `moedim_model.py` | Additional model |

### Data Enrichment
| File | Purpose |
|------|---------|
| `odds_tracker.py` | Live odds via The Odds API (free tier, 500 req/month), CLV tracking |
| `weather.py` | Weather impact via Open-Meteo API (free, no key) |
| `kalshi.py` | Kalshi public API for live contract prices + auto-Kelly |
| `injuries.py` | ESPN injury report + Elo impact scoring |

### Trading and Display
| File | Purpose |
|------|---------|
| `predict_ledger.py` | Contract ledger (add, sell, resolve, mark, invert positions) |
| `live_scores.py` | Live NBA scores via NBA Stats API (60s refresh loop) |
| `auto_resolve.py` | Auto-settle finished trades against live final scores |
| `html_generator.py` | Blogger-ready HTML prediction tables |
| `help_system.py` | CLI help text for all commands |
| `accuracy_test.py` | Standalone quick walk-forward accuracy test |

### Utility Scripts
| File | Purpose |
|------|---------|
| `run_optimize.py` | Standalone optimization runner |
| `accuracy_optimize.py` | Accuracy-focused optimization |
| `run_enhanced_all.py` | Run enhanced backtest standalone |
| `quick_optimizer.py` | Quick parameter sweep |
| `sweep_enhanced.py` | Enhanced parameter sweep |
| `master_optimize.py` | Cross-sport master optimization runner |
| `run_sport_workflow.py` | Automated sport workflow runner |

---

## Optimization

### Objective Function

All optimizers use the same scoring objective:

```
score = -(LogLoss * 8 + Brier * 40)
```

This weighting is intentional. It penalizes overconfident wrong predictions (log loss) while also rewarding calibration (Brier). Higher is better (scores are negative; closest to zero wins).

### master_optimize.py

The `master_optimize.py` script runs the full optimization pipeline for the NBA system. It is also invoked by the cross-sport master optimizer that coordinates optimization across NBA, NFL, NHL, and MLB simultaneously.

The optimization is accuracy-first: the pipeline targets improvements in raw predictive accuracy while maintaining calibration quality.

### Optimization Pipeline

1. **Grid search** (`grid`) -- Coarse sweep over 7 Elo parameters with configurable ranges
2. **Genetic optimizer** (`genetic`) -- `scipy.optimize.differential_evolution` for fine-tuning
3. **Bayesian optimizer** (`bayesian`) -- GP surrogate + Expected Improvement, ~50-100 evaluations
4. **Auto-optimize** (`autoopt`) -- Automated grid-to-genetic-to-bayesian pipeline
5. **Super-optimize** (`superopt`) -- Exhaustive 7-phase optimization (9 parameters, takes hours)
6. **Coordinate descent** (`singleopt`) -- One-param-at-a-time sweep with fine refinement
7. **Mega optimizer** (`mega optimize`) -- 7-phase per-model optimization for the 36-model ensemble

### Mega-Ensemble Optimizer Phases

The `mega_optimizer.py` runs 7 phases for per-model exhaustive tuning:

- **Phase 0**: Elo-only baseline -- establish the floor
- **Phase 1**: Per-model solo optimization -- Elo + one model at a time
- **Phase 2**: Head-to-head tournament -- pairs, triples, top-N combos
- **Phase 3**: Meta-learner + global tuning -- max_adj, meta_model, retrain_every
- **Phase 4**: Combined DE fine-tuning -- differential evolution over top numeric params
- **Phase 5**: Final ablation -- test each model's contribution, prune weak ones
- **Phase 6**: Validation -- run best config 5 times, report stability

### Validation After Optimization

After optimizing, validate with this sequence:
1. `pbo` -- Probability of backtest overfitting (PBO < 0.3 is good)
2. `results` -- Check Deflated Sharpe Ratio (DSR > 1.96 means significant)
3. `purgedcv` -- Fold stability (accuracy std < 2% is good)
4. `cpcv` -- Combinatorial robustness (>90% paths above 65% accuracy)
5. `montecarlo` -- Statistical significance (p < 0.05)

---

## Kalshi Integration

The `kalshi.py` module connects to the Kalshi public API to fetch live prediction market contract prices for NBA games. Features:

- Show current contract prices for today's games (`kalshi` command)
- Compare model probability against market price to find edges
- Auto-Kelly position sizing recommendation based on edge size
- Track closing line value (CLV) -- whether the line moved toward or away from your position

Kalshi contracts are $1 binary options. The system's trading ledger tracks entry, mark-to-market, and settlement of these contracts with 2% fees on entry and exit.

---

## Data Sources

All data sources are free. No paid API keys are required.

| Source | Data | Cache TTL |
|--------|------|-----------|
| `nba_api` Python package | Game logs (2 years), player stats | 6 hours |
| ESPN JSON API | Injury reports (status, position, impact) | 4 hours |
| Open-Meteo API | Weather conditions (temperature, wind, precipitation) | 2 hours |
| The Odds API (free tier) | Moneyline odds, 500 requests/month | Per-request |
| Kalshi public API | Prediction market contract prices | Live |
| NBA Stats CDN | Live scores for auto-resolve | 60 seconds |

Smart caching via `cache_utils.py` provides season-aware staleness checks. Different data types refresh at different rates. Files under 500 bytes are always treated as stale.

---

## Configuration

### Settings File: `nba_elo_settings.json`

Contains all tunable Elo parameters (39 total). Key NBA-specific defaults:

| Parameter | Default | NBA Significance |
|-----------|---------|------------------|
| `k` | 8.53 | Low K for 82-game season (stability over responsiveness) |
| `home_advantage` | 31.99 | ~58% implied home win rate |
| `player_boost` | 48.0 | Highest of any sport -- NBA is star-driven |
| `rest_factor` | 18.21 | Back-to-back games are critical in NBA |
| `b2b_penalty` | 0.0 | Captured by rest_factor instead |
| `travel_factor` | 12.20 | Cross-country travel fatigue |
| `pace_factor` | 26.53 | Significant for tempo mismatch games |
| `altitude_factor` | 0.04 | Denver only; optimizer found minimal impact |
| `season_regress` | 0.33 | 33% regression toward mean at season boundaries |
| `form_weight` | 5.06 | Recent form from last 10 games |
| `playoff_hca_factor` | 0.71 | ~29% reduction in home court advantage during playoffs |

### Mega-Ensemble Settings: `nba_mega_settings.json`

| Parameter | Default | Notes |
|-----------|---------|-------|
| `window` | 10 | Rolling feature window (10-game for 82-game season) |
| `min_train` | 400 | Games before meta-learner starts predicting |
| `retrain_every` | 80 | Retrain ML models every 80 games |
| `max_adj` | 0.08 | Maximum mega-ensemble adjustment (+/- from Elo anchor) |
| `meta_model` | xgboost | Meta-learner type (xgboost, ridge, or logistic) |
| `pyth_exp` | 13.91 | Pythagorean exponent for point-based win estimation |

### Other Generated Files

| File | Purpose |
|------|---------|
| `nba_elo_ratings.json` | Current Elo ratings for all 30 teams |
| `nba_platt_scaler.json` | Platt calibration coefficients |
| `nba_enhanced_model.json` | XGBoost ensemble metadata |
| `nba_xgb_model.json` | Saved XGBoost model weights |
| `nba_meta_xgb.json` | Trained meta-learner weights |
| `nba_backtest_predictions.csv` | Per-game backtest predictions |
| `nba_calibration.csv` | 10-bin calibration table |
| `predicts_lots.csv` | Trading ledger (positions, P&L) |
| `weather_cache.json` | Cached weather API responses |

### Cross-Sport Shared Files

The wallet and portfolio are shared across all sport systems (NBA, NFL, NHL, MLB):

| File | Location | Purpose |
|------|----------|---------|
| `portfolio_settings.json` | Parent directory | Shared portfolio configuration |
| `cash_transactions.csv` | Parent directory | Shared cash deposit/withdrawal log |

---

## Dependencies

**Required:**
- pandas, numpy, scipy, colorama, tqdm, xgboost, requests, nba_api, matplotlib

**Optional (for advanced models):**
- torch (MLP, LSTM neural networks)
- hmmlearn (Hidden Markov Model)
- filterpy (Kalman Filter)
- lightgbm (LightGBM gradient boosting)
- catboost (CatBoost gradient boosting)
- networkx (PageRank network model)

---

## NBA-Specific Design Notes

- **Season spans two calendar years**: Oct--Jun. Season detection uses `year + 1 if month >= 10`.
- **82-game season**: More data points than NFL (17 games) but fewer than MLB (162). The 10-game rolling window balances signal vs noise.
- **No pitcher/goalie system**: Unlike MLB and NHL, the NBA does not track individual starting player sub-ratings. Player impact is captured through the team-level composite score (points, assists, rebounds).
- **Player impact is the highest of any sport**: `player_boost = 48.0` reflects that a single star player has an outsized effect on NBA outcomes compared to other team sports.
- **Back-to-back games are uniquely punishing**: NBA teams playing the second night of a back-to-back show measurably worse performance. The `rest_factor` captures this.
- **Denver altitude**: The Nuggets' home court at 5,280 ft creates a measurable (though small) advantage. The optimizer found `altitude_factor = 0.04`, indicating the effect is real but modest.
- **Pace mismatch matters**: When a fast-paced team faces a slow-paced team, the pace differential affects scoring variance and predictability.
- **Fuzzy team lookup**: `NBAElo.find_team()` accepts full names, abbreviations, partial matches, and close matches via `difflib.get_close_matches`. Typing `lakers`, `LAL`, or `Los Angeles Lakers` all work.
- **6 divisions**: Atlantic, Central, Southeast, Northwest, Pacific, Southwest. Division rivalry factor of 10.0 adjusts for familiarity.
- **Python 3.8+ compatible**: No walrus operators, no `match` statements.
- **All data files use `nba_` prefix**: Settings, ratings, caches, and backtest outputs are all namespaced.
