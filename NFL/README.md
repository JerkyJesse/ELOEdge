# NFL Prediction System (SharpStack)

## Overview

NFL game prediction system that combines Elo ratings with a 36-model mega-ensemble to produce calibrated win probabilities for every NFL matchup. The system integrates a Predicts $1 binary contract trading ledger for position tracking and P&L analysis. It runs as an interactive CLI application -- no web server, no test framework, no build system.

NFL-specific design priorities: weekly game cadence with rest days centered at 7, bye week detection and boost, Thursday Night Football short-turnaround penalties, smaller dataset (~272 regular-season games per year across 32 teams), win streak momentum as a strong signal, high home field advantage variance across venues, advanced EPA/CPOE stats via nflverse play-by-play data, weather impact for outdoor stadiums, and position-based injury scoring (QB injury = -50 Elo).

---

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

On first launch the system will:
1. Download two years of game data via the ESPN public API
2. Download player stats (passing, rushing, receiving leaders)
3. Fetch injury reports from the ESPN API
4. Build Elo ratings for all 32 NFL teams
5. Run a walk-forward backtest to fit the Platt calibration scaler
6. Enter the interactive CLI loop

Type a team name (e.g. `Chiefs`) to start a prediction. Type `help` for all commands.

---

## Commands

### Data and Display
| Command | Description |
|---------|-------------|
| `all` | Show all 32 teams ranked by Elo rating |
| `refresh` | Delete cached data and re-download everything, rebuild model |
| `players` | Show top player stats and composite scores |
| `settings` | Display current Elo parameter values |
| `injuries` | Show injury report with position-based Elo impact per team |
| `injuries set <team> <player1>, <player2>` | Manually mark players as OUT |
| `today` / `html` | Generate Blogger-ready HTML predictions for today's games |
| `tomorrow` | Generate predictions for tomorrow's games |

### Data Enrichment
| Command | Description |
|---------|-------------|
| `odds` | Show live moneyline odds from The Odds API |
| `kalshi` | Show Kalshi prediction market contract prices |
| `weather` | Show weather report for outdoor games |
| `advstats` | Show EPA/CPOE/success rate advanced stats rankings |

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
set k=17.3          set home=18         set boost=27
set rest=30         set b2b=4           set travel=25
set pace=4          set div=31          set streak=32
set altitude=0.67   set playoff=2.0     set phase=32
set kelly=quarter   set balance=50      set autoresolve=true
```

After any parameter change, run `backtest` to refit the Platt calibration scaler.

---

## Architecture

The system uses a three-layer prediction pipeline. Each layer adds refinement on top of the previous one.

### Layer 1: Elo Model

The `NFLElo` class in `elo_model.py` maintains Elo ratings for all 32 NFL teams. The base rating is 1500. Each game updates ratings using a configurable K-factor (default 17.30 -- higher than NBA's 8.53 because the 17-game season demands faster responsiveness). The model applies 24+ adjusters:

- **Home field advantage** (18.12 Elo, ~52.6% implied home win rate)
- **Player strength** (26.52 boost -- lower than NBA; football is more team-dependent)
- **Rest days centered at 7** (30.0 factor -- the weekly NFL schedule makes rest deviations highly impactful)
- **B2B penalty** (3.93 -- Thursday Night Football short turnaround after Sunday)
- **Bye week factor** (0.0 -- currently disabled, tunable)
- **Division rivalry** (30.85 -- much higher than NBA; NFL division games are grinder matchups)
- **Win streak factor** (32.0 -- momentum matters more in NFL with few games per season)
- **Season phase factor** (32.0 -- early/mid/late season adjustment)
- **Altitude** (0.67 -- Denver Broncos at Mile High, 5,280 ft)
- **Travel fatigue** (25.0 -- timezone-based distance)
- **Strength of schedule** (20.0 -- much higher than NBA)
- **Mean reversion** (17.21 -- regression after extreme results)
- **Playoff HCA factor** (2.0 -- amplified home advantage in playoffs)
- **Season regression** (33% pull toward mean at season boundaries)
- **Margin of victory** uses `log(max(1, abs(margin)) + 1)` for dampening NFL blowouts

### Layer 2: Enhanced XGBoost Ensemble

The `enhanced_model.py` module builds an 80% Elo / 20% XGBoost blended model. The `TeamTracker` class maintains rolling 5-game windows per team (narrower than NBA's 10-game window due to the 17-game season), generating 93 features:

- Elo probability and rating differential
- Player strength differential
- Per-team rolling stats: PPG, PAPG, win%, margins, rest days
- Pythagorean win expectation (exponent 2.37 -- lower than NBA due to lower-scoring games)
- Win/loss streaks, scoring consistency, trend indicators

XGBoost is trained walk-forward (no leakage): Elo-only for the first 200 games, then XGBoost trained and retrained every 50 games. Parameters: `max_depth=5, eta=0.03, subsample=0.9, 300 rounds`.

### Layer 3: 36-Model Mega-Ensemble

The `mega_predictor.py` and `mega_backtest.py` modules implement a 36-model stacking ensemble. Each model produces a raw probability independently. A meta-learner (XGBoost, Ridge, or Logistic regression) combines all outputs into a bounded adjustment clamped to +/- `max_adj` (default 0.10) around the Elo anchor probability. All 36 models run in parallel via `ThreadPoolExecutor`.

### Calibration

`platt.py` applies Platt scaling (logistic regression on logit of raw probability) to produce well-calibrated final outputs. Isotonic regression and beta calibration are also available. The Platt scaler is fitted during the `backtest` command and must be refitted after any parameter change.

---

## The 36 Models

### Tier 0 -- Core
| Model | File | Description |
|-------|------|-------------|
| Elo | `elo_model.py` | Base team ratings with 24+ adjusters |
| XGBoost | `enhanced_model.py` | 93-feature gradient boosting ensemble |

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
| Pythagorean | `classic_models.py` | Pythagorean win expectation |
| ExpSmoothing | `classic_models.py` | Exponential smoothing |
| MeanReversion | `classic_models.py` | Mean reversion model |

### Tier 6 -- Data Enrichment
| Model | File | Description |
|-------|------|-------------|
| Weather | `weather.py` | Open-Meteo weather impact (outdoor games only) |
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
| `config.py` | Constants, 32 NFL teams, 8 divisions, settings I/O |
| `elo_model.py` | `NFLElo` class -- ratings, predictions, 24+ adjusters |
| `build_model.py` | Model construction with season regression and altitude bonus |
| `data_games.py` | Game data download via ESPN public API |
| `data_players.py` | Player stats download (passing, rushing, receiving) + team scoring |
| `backtest.py` | Walk-forward backtest, grid/genetic/bayesian optimizers (~2100 lines) |
| `enhanced_model.py` | XGBoost ensemble + SHAP + TeamTracker (93 features) |
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
| `classic_models.py` | SRS, Colley, Log5, Pythagorean, ExpSmoothing, MeanReversion |
| `svm_model.py` | SVM Classifier |
| `fibonacci_model.py` | Fibonacci Retracement |
| `evt_model.py` | Extreme Value Theory |
| `benford_model.py` | Benford's Law |
| `moedim_model.py` | Additional model |

### Data Enrichment
| File | Purpose |
|------|---------|
| `advanced_stats.py` | EPA, CPOE, success rate, referee tendencies via nfl_data_py |
| `odds_tracker.py` | Live odds via The Odds API (free tier, 500 req/month), CLV tracking |
| `weather.py` | Weather impact via Open-Meteo API (free, no key) -- outdoor games only |
| `kalshi.py` | Kalshi public API for live contract prices + auto-Kelly |
| `injuries.py` | ESPN injury report + position-based Elo impact scoring |

### Trading and Display
| File | Purpose |
|------|---------|
| `predict_ledger.py` | Contract ledger (add, sell, resolve, mark, invert positions) |
| `live_scores.py` | Live NFL scores via ESPN API (60s refresh loop) |
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

The `master_optimize.py` script runs the full optimization pipeline for the NFL system. It is also invoked by the cross-sport master optimizer that coordinates optimization across NBA, NFL, NHL, and MLB simultaneously.

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

The `kalshi.py` module connects to the Kalshi public API to fetch live prediction market contract prices for NFL games. Features:

- Show current contract prices for this week's games (`kalshi` command)
- Compare model probability against market price to find edges
- Auto-Kelly position sizing recommendation based on edge size
- Track closing line value (CLV) -- whether the line moved toward or away from your position

Kalshi contracts are $1 binary options. The system's trading ledger tracks entry, mark-to-market, and settlement of these contracts with 2% fees on entry and exit.

---

## Data Sources

All data sources are free. No paid API keys are required.

| Source | Data | Cache TTL |
|--------|------|-----------|
| ESPN public API | Game scores and schedules (2 years) | 6 hours |
| ESPN public API | Player stats (passing, rushing, receiving) | 6 hours |
| ESPN JSON API | Injury reports (status, position, impact) | 4 hours |
| nfl_data_py / nflverse | EPA, CPOE, success rate play-by-play data | Cached to CSV |
| Open-Meteo API | Weather for outdoor stadiums (wind, temp, precip) | 2 hours |
| The Odds API (free tier) | Moneyline odds, 500 requests/month | Per-request |
| Kalshi public API | Prediction market contract prices | Live |
| ESPN live scoreboard | Live scores for auto-resolve | 60 seconds |

Smart caching via `cache_utils.py` provides season-aware staleness checks. Different data types refresh at different rates. Files under 500 bytes are always treated as stale.

---

## Configuration

### Settings File: `nfl_elo_settings.json`

Contains all tunable Elo parameters (24 total). Key NFL-specific defaults:

| Parameter | Default | NFL Significance |
|-----------|---------|------------------|
| `k` | 17.30 | Higher K than NBA (8.53) -- 17-game season demands faster adaptation |
| `home_advantage` | 18.12 | ~52.6% implied home win rate (lower than NBA) |
| `player_boost` | 26.52 | Lower than NBA -- football is more team-dependent |
| `rest_factor` | 30.0 | Centered at 7-day weekly schedule; deviations are major |
| `b2b_penalty` | 3.93 | Thursday Night Football short turnaround penalty |
| `bye_week_factor` | 0.0 | Bye week boost (currently disabled, tunable) |
| `division_factor` | 30.85 | Very high -- NFL division games are grinder matchups |
| `win_streak_factor` | 32.0 | Momentum matters more with few games per season |
| `season_phase_factor` | 32.0 | Early/mid/late season adjustment |
| `altitude_factor` | 0.67 | Denver Broncos Mile High (5,280 ft) |
| `travel_factor` | 25.0 | Cross-country travel fatigue |
| `sos_factor` | 20.0 | Strength of schedule (much higher than NBA) |
| `mean_reversion` | 17.21 | Regression after extreme results |
| `playoff_hca_factor` | 2.0 | Amplified home advantage in playoffs |
| `season_regress` | 0.33 | 33% regression toward mean at season boundaries |

### Mega-Ensemble Settings: `nfl_mega_settings.json`

| Parameter | Default | Notes |
|-----------|---------|-------|
| `window` | 5 | Rolling feature window (narrow for 17-game season) |
| `min_train` | 200 | Games before meta-learner starts predicting |
| `retrain_every` | 50 | Retrain ML models every 50 games |
| `max_adj` | 0.10 | Maximum mega-ensemble adjustment (+/- from Elo anchor) |
| `meta_model` | xgboost | Meta-learner type (xgboost, ridge, or logistic) |
| `pyth_exp` | 2.37 | Pythagorean exponent (low due to lower-scoring NFL games) |

### Other Generated Files

| File | Purpose |
|------|---------|
| `nfl_elo_ratings.json` | Current Elo ratings for all 32 teams |
| `nfl_platt_scaler.json` | Platt calibration coefficients |
| `nfl_enhanced_model.json` | XGBoost ensemble metadata |
| `nfl_xgb_model.json` | Saved XGBoost model weights |
| `nfl_meta_xgb.json` | Trained meta-learner weights |
| `nfl_backtest_predictions.csv` | Per-game backtest predictions |
| `nfl_calibration.csv` | 10-bin calibration table |
| `nfl_epa_stats.csv` | Cached EPA/CPOE per-team stats |
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
- pandas, numpy, scipy, colorama, tqdm, xgboost, requests, matplotlib
- nfl_data_py (for EPA/CPOE advanced stats from nflverse play-by-play)

**Optional (for advanced models):**
- torch (MLP, LSTM neural networks)
- hmmlearn (Hidden Markov Model)
- filterpy (Kalman Filter)
- lightgbm (LightGBM gradient boosting)
- catboost (CatBoost gradient boosting)
- networkx (PageRank network model)

---

## NFL-Specific Design Notes

- **Season spans two calendar years**: Sep--Feb. Season detection uses `year if month >= 9 else year - 1`.
- **17-game regular season**: Far fewer data points than NBA (82) or MLB (162). This drives several design choices: higher K-factor (17.30 vs NBA's 8.53), narrower rolling window (5 games vs NBA's 10), and wider confidence intervals on all predictions.
- **Weekly cadence with 7-day rest baseline**: Unlike daily sports, NFL rest is centered at 7 days. Deviations from this baseline (Thursday games after Sunday = 4 days, Monday-to-Sunday = 6 days, bye weeks = 14 days) are all highly predictive.
- **Thursday Night Football penalty**: The `b2b_penalty = 3.93` captures the measurable drop in performance when teams play on short turnaround. This is the NFL equivalent of NBA back-to-backs.
- **Position-based injury impact**: QB injuries are catastrophic (-50 Elo), while kicker injuries are minor (-5 Elo). The full injury tier: QB -50, RB -15, WR/TE -12, DEF -10, OL -8, K/P -5.
- **Win streaks matter more**: With only 17 games, a 3-game winning streak represents 18% of the season. The `win_streak_factor = 32.0` is the highest of any parameter.
- **Home field advantage varies greatly**: NFL stadiums range from domes (no weather, no altitude) to outdoor venues at altitude (Denver) in extreme weather (Green Bay, Buffalo). The model captures this through altitude factor, weather impact, and the base home advantage.
- **Advanced stats via nflverse**: EPA (Expected Points Added), CPOE (Completion Percentage Over Expected), and success rate provide play-level efficiency metrics not available in basic box scores. These come from `nfl_data_py` which wraps the nflverse play-by-play dataset.
- **Weather is a real factor**: Unlike indoor sports, NFL outdoor games are affected by wind, temperature, and precipitation. The weather model sources data from Open-Meteo and adjusts predictions for outdoor stadiums.
- **Denver is the only altitude team**: Mile High Stadium at 5,280 ft is the only NFL venue at significant elevation. `altitude_factor = 0.67`.
- **8 divisions**: AFC East/North/South/West, NFC East/North/South/West. Division rivalry factor of 30.85 is among the highest parameters.
- **Fuzzy team lookup**: `NFLElo.find_team()` accepts full names, abbreviations, partial matches, and close matches. Typing `chiefs`, `KC`, or `Kansas City Chiefs` all work.
- **Python 3.8+ compatible**: No walrus operators, no `match` statements.
- **All data files use `nfl_` prefix**: Settings, ratings, caches, and backtest outputs are all namespaced.
