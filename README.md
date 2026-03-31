# Sports Prediction Mega-Ensemble

Four-sport prediction system (NFL, MLB, NBA, NHL) combining **31 independent prediction models** into a single meta-learned ensemble. Every prediction method from Elo ratings to neural networks to chaos theory, all running together.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU PyTorch

# Run any sport
cd NFLClaude && python main.py
cd MLBClaude && python main.py
cd NBAClaude && python main.py
cd NHLClaude && python main.py
```

First run auto-downloads game data, player stats, and injury reports from free public APIs. No API keys required for core functionality.

---

## The 31 Models

Every historical prediction method, converted to sports:

| Tier | Model | Origin | Year |
|------|-------|--------|------|
| **Core** | Elo Ratings (24+ adjusters) | Arpad Elo | 1960 |
| | XGBoost Ensemble (31 features) | Tianqi Chen | 2014 |
| **Proven** | Hidden Markov Model | Baum-Welch | 1966 |
| | Kalman Filter | Rudolf Kalman / NASA | 1960 |
| | PageRank + HITS | Google / Kleinberg | 1998 |
| | LightGBM | Microsoft Research | 2017 |
| | CatBoost | Yandex | 2017 |
| | MLP Neural Network | Rumelhart | 1986 |
| | LSTM | Hochreiter & Schmidhuber | 1997 |
| **Exotic** | GARCH Volatility | Bollerslev | 1986 |
| | Fourier + Wavelet | Fourier / Morlet | 1807/1984 |
| | Survival Analysis | Actuarial science | 1700s |
| | Copula Dependencies | Sklar | 1959 |
| **Info/Physics** | Shannon Entropy + KL Divergence | Shannon | 1948 |
| | Newtonian Momentum | Newton | 1687 |
| | Markov Chains | Markov | 1906 |
| | k-Means Team Archetypes | Lloyd | 1957 |
| | Nash Equilibrium + Style Matchups | von Neumann / Nash | 1944/1950 |
| **Classical** | Poisson / Dixon-Coles | Poisson / Dixon & Coles | 1837/1997 |
| | Glicko-2 | Glickman | 1995 |
| | Bradley-Terry MLE | Bradley & Terry | 1952 |
| | Monte Carlo Simulation | Ulam & von Neumann | 1946 |
| | Random Forest | Breiman | 2001 |
| **Sports-Specific** | Simple Rating System (SRS) | Pro sports analytics | - |
| | Colley Matrix | Colley (BCS) | 2001 |
| | Log5 (Bill James) | Bill James | 1981 |
| | PythagenPat | Davenport & Woolner | 2000 |
| | Exponential Smoothing | Holt | 1957 |
| | Mean Reversion (Bollinger) | Bollinger | 1983 |
| **Data** | Weather Impact | Open-Meteo | - |
| | Market Odds / CLV | The Odds API | - |

---

## Architecture

```
Game Data --> [32 Base Models] --> Feature Matrix --> Meta-Learner --> Bounded Adj --> Final Prob
                                                                         |
                                                              Elo probability (anchor)
                                                              + meta adjustment (+/- max_adj)
```

The system uses an **Elo-anchored bounded adjustment**: the Elo probability is the foundation, and the meta-learner (trained on all models' features) adjusts it by a bounded amount. This prevents overfitting while letting exotic signals contribute at the margins.

---

## Complete Command Reference

### Predictions & Ratings

| Command | Description |
|---------|-------------|
| `<team name>` | Predict a matchup (fuzzy match: `Chiefs`, `pack`, `DAL`, `eagles`) |
| `all` | Show all team Elo ratings, ranked |
| `players` | Show top performers (league-wide player stats) |
| `settings` | Display all current model parameters and Platt scaler status |
| `injuries` | Show injury report with Elo impact estimates |
| `injuries set <team> <p1>,<p2>` | Manually mark players as OUT |

### Game Predictions & Publishing

| Command | Description |
|---------|-------------|
| `today` | Generate predictions for today's games (also: `html`, `blog`, `blogger`) |
| `tomorrow` | Generate predictions for tomorrow's games |

### Data Management

| Command | Description |
|---------|-------------|
| `refresh` | Re-download all data (games, players, injuries), rebuild model |
| `odds` | Show today's odds from 50+ bookmakers (needs API key) |
| `weather` | Weather impact report for a game venue |
| `advstats` | Advanced analytics rankings (NFL: EPA/CPOE, also: `epa`) |
| `statcast` | Statcast rankings (MLB: xwOBA/xERA, also: `advstats`) |
| `officials` | Referee tendency analysis (NFL only) |

### Backtesting

| Command | Description |
|---------|-------------|
| `backtest` | Walk-forward Elo backtest with Platt scaler fitting |
| `enhanced` | XGBoost ensemble backtest (80/20 Elo+XGBoost blend) + SHAP |
| `enhanced decay` | Time-decayed ensemble (Elo weight 95% early to 70% late) |
| `shap` | SHAP feature importance for XGBoost ensemble |

### Elo Optimization

| Command | Description |
|---------|-------------|
| `grid` | Interactive grid search (7 params, customizable ranges) |
| `genetic` | Genetic algorithm with scipy differential evolution |
| `bayesian` | Gaussian Process + Expected Improvement optimization |
| `autoopt` | Automatic grid + genetic + bayesian pipeline (~15-30 min) |
| `superopt` | Exhaustive 7-phase optimization, all 9 params (2-4 hours) |
| `singleopt` | Coordinate descent single-parameter optimization (also: `coord`) |
| `results` | Show best parameters from all optimizer runs + DSR significance |

### Validation & Statistical Testing

| Command | Description |
|---------|-------------|
| `purgedcv` | Purged walk-forward cross-validation (k-fold with embargo gap) |
| `cpcv` | Combinatorial purged CV (all C(k, k_test) train/test paths) |
| `pbo` | Probability of backtest overfitting (symmetric CV) |
| `montecarlo` | Monte Carlo permutation test (p-value, ~500 iterations) |
| `rollingcal` | Rolling origin Platt recalibration (out-of-sample metrics) |
| `kelly` | Kelly criterion position sizing backtest |
| `sliding` | Sliding vs expanding window comparison |
| `convergence` | Elo rating convergence / burn-in analysis |
| `conformal` | Conformal prediction intervals (coverage guarantees) |
| `betacal` | Beta calibration (3-parameter asymmetric) |
| `platt` | Refit Platt calibration scaler (also: `calibrate`) |

### Mega-Ensemble

| Command | Description |
|---------|-------------|
| `mega` | Run full mega-ensemble backtest (26+ models) (also: `megabacktest`) |
| `mega optimize` | Exhaustive 5-phase optimization - finds best settings (also: `megaopt`) |
| `mega quick` | Quick grid search, Phase 1 only (also: `megaquick`) |
| `mega ablation` | Single-model contribution test, auto-prunes bad models (also: `megasingle`) |
| `mega models` | Show all 32 models with ON/OFF status (also: `models`) |
| `mega on <model>` | Enable a specific model (e.g., `mega on lstm`) |
| `mega off <model>` | Disable a specific model (e.g., `mega off monte_carlo`) |
| `mega on all` | Enable all models |
| `mega enable <model>` | Same as `mega on` |
| `mega disable <model>` | Same as `mega off` |
| `mega settings` | Show all mega-ensemble parameter values (also: `mega params`, `mega config`) |
| `mega set <param>=<value>` | Set a mega parameter (e.g., `mega set adj=0.10`) |

### Elo Settings

| Command | Description |
|---------|-------------|
| `set` | Show all 39 Elo parameters with current values |
| `set <param>=<value>` | Set any Elo parameter (e.g., `set k=10`, `set home=48`) |

### Trading Ledger (Predicts / Kalshi)

| Command | Description |
|---------|-------------|
| `predicts` | Show full contract ledger with P&L (also: `summary`) |
| `balance` | View bankroll status (starting, current, P&L, ROI) |
| `resolve` | Settle finished contracts (win/loss prompt) |
| `sell` | Exit a position early at a price (partial or full) |
| `mark` | Update current market marks on open lots |
| `invert` | Flip an open trade's direction |
| `chart` | Generate monthly realized P&L bar chart |
| `live` | Live scores for open positions (60s auto-refresh) |
| `autoresolve` | Auto-resolve finished trades from live scores |
| `autoresolve on` | Enable auto-resolve on startup |
| `autoresolve off` | Disable auto-resolve on startup |

### Utility

| Command | Description |
|---------|-------------|
| `help` | Show help overview |
| `help <command>` | Detailed help for a specific command |
| `quit` | Save and exit (also: `q`, `exit`) |

---

## Settable Parameters

### Elo Parameters (39 total, `set param=value`)

| Category | Parameter | Aliases | Description |
|----------|-----------|---------|-------------|
| **Core** | `k` | k_factor | K-factor (learning rate per game) |
| | `base_rating` | base, rating | Starting Elo rating |
| | `home_adv` | home, hca | Home court/field advantage (Elo points) |
| | `use_mov` | mov, margin | Use margin of victory (true/false) |
| **Player** | `player_boost` | boost, player | Team-level player strength boost |
| | `starter_boost` | starter, sp_boost | Starting pitcher/QB quality |
| | `bullpen_factor` | bullpen, reliever | Bullpen/reliever quality factor |
| | `opp_pitcher_factor` | opp_pitcher, opp_sp | Opponent pitcher adjustment |
| **MOV** | `mov_base` | mov_mult, mov_constant | MOV log curve shift constant |
| | `mov_cap` | movcap, margin_cap | Maximum MOV adjustment cap |
| **Rest** | `rest_factor` | rest | Rest days advantage factor |
| | `rest_advantage_cap` | restcap, rest_cap | Maximum rest advantage multiplier |
| | `b2b_penalty` | b2b, back_to_back | Back-to-back game penalty |
| **Travel** | `travel_factor` | travel | Elo penalty per timezone crossed |
| | `east_travel_penalty` | east_travel, eastbound | Extra penalty for eastbound travel |
| | `road_trip_factor` | roadtrip, road_trip | Extended road trip penalty |
| | `homestand_factor` | homestand | Extended homestand bonus |
| | `altitude_factor` | altitude, alt | Altitude bonus (Denver/Colorado) |
| | `park_factor_weight` | parkfactor, park_factor | Park factor weight |
| **Form** | `form_weight` | form | Recent form weight |
| | `win_streak_factor` | streak, win_streak | Win/loss streak momentum |
| | `mean_reversion` | reversion, regress | Regression after extreme results |
| | `season_regress` | season_regression, regress_pct | Season boundary regression % |
| **Matchup** | `sos_factor` | sos | Strength of schedule weight |
| | `division_factor` | division, div | Divisional game confidence reducer |
| | `interleague_factor` | interleague, il_factor | Interleague game adjustment |
| | `series_adaptation` | series, adaptation | Series/rematch adaptation factor |
| **Scoring** | `pace_factor` | pace, tempo | Run/scoring environment mismatch |
| | `pyth_factor` | pyth, pythagorean | Pythagorean expected W% adjustment |
| | `scoring_consistency_factor` | consistency | Penalty for volatile scoring |
| | `home_road_factor` | home_road, split | Home/road split bonus |
| **Season** | `playoff_hca_factor` | playoff, playoff_hca | Playoff home advantage multiplier |
| | `season_phase_factor` | phase, season_phase | Early-season dampener |
| **K-factor** | `k_decay` | kdecay, k_reduction | K-factor decay over season |
| | `surprise_k` | surprise, upset_k | Extra K for surprise results |
| **NFL** | `bye_week_factor` | bye, bye_week | Bye week advantage (NFL only) |
| **Account** | `kelly_fraction` | kelly | Kelly fraction (quarter/half/0.25/0.50) |
| | `starting_balance` | balance, bankroll | Starting account balance |
| | `autoresolve_enabled` | autoresolve, auto_resolve | Auto-settle finished trades (true/false) |

### Mega Parameters (14 total, `mega set param=value`)

| Parameter | Aliases | Description |
|-----------|---------|-------------|
| `max_adj` | adj, maxadj, adjustment | Max meta-learner probability adjustment |
| `meta_model` | meta, metalearner, stacker | Meta-learner algorithm (ridge/logistic/xgboost) |
| `retrain_every` | retrain, retrain_interval | Retrain meta-learner every N games |
| `min_train` | mintrain, min_games, warmup | Games before meta-learner starts predicting |
| `kalman_process_noise` | kalman_pn, process_noise, pn | Kalman filter process noise |
| `kalman_measurement_noise` | kalman_mn, measurement_noise, mn | Kalman filter measurement noise |
| `hmm_states` | hmm_n, n_states, states | Number of HMM hidden states |
| `network_decay` | net_decay, pagerank_decay, decay | PageRank temporal decay (0-1) |
| `momentum_friction` | friction, mom_friction | Momentum friction coefficient |
| `n_clusters` | clusters, k_clusters, nclusters | Number of team archetype clusters |
| `glicko_initial_rd` | glicko_rd, initial_rd, rd | Glicko-2 initial rating deviation |
| `bt_decay` | bt_recency, bradley_decay | Bradley-Terry recency decay (0-1) |
| `mc_simulations` | mc_sims, simulations, n_sims, sims | Monte Carlo simulations per game |
| `window` | rolling_window, feat_window | Rolling feature window size (games) |

---

## Optimization System

### `mega optimize` - Exhaustive 5-Phase Search

1. **Grid Search** - 150 combinations of max_adj x meta_model x retrain_every
2. **Coordinate Descent** - 8 secondary params swept one-at-a-time, 3 passes
3. **Bayesian Optimization** - 30 Gaussian perturbation trials around best config
4. **Tournament** - Top 10 configs re-compete head-to-head
5. **Stability Check** - Verify best config isn't overfit (3 re-runs)

Objective: `LogLoss * 8 + Brier * 40` (lower is better, same as Elo optimizer)

### `mega ablation` - Single-Model Contribution Test

1. Runs baseline with all enabled models
2. Turns OFF each model one at a time, re-runs backtest
3. Measures each model's individual contribution (delta in objective)
4. Ranks all models: HELPS / neutral / HURTS
5. Auto-disables models that hurt performance
6. Verifies improvement, reverts if no gain

### Elo Optimization Pipeline

| Phase | Command | What It Does | Time |
|-------|---------|-------------|------|
| 1 | `backtest` | Baseline metrics | 5s |
| 2 | `convergence` | Find burn-in period | 10s |
| 3 | `grid` | Coarse parameter search | 5-30m |
| 4 | `pbo` | Check overfitting risk | 10m |
| 5 | `genetic` | Fine-tune with evolution | 5-15m |
| 6 | `bayesian` | GP-based optimization | 5-10m |
| 7 | `autoopt` | Automatic pipeline (3+4+5) | 15-30m |
| 8 | `superopt` | Exhaustive all params | 2-4h |
| 9 | `purgedcv` | Cross-validation stability | 5m |
| 10 | `cpcv` | Combinatorial robustness | 10m |
| 11 | `montecarlo` | Statistical significance | 8m |
| 12 | `enhanced` | Train XGBoost ensemble | 30s |
| 13 | `rollingcal` | Out-of-sample calibration | 10s |
| 14 | `conformal` | Coverage analysis | 10s |
| 15 | `kelly` | Position sizing backtest | 5s |

---

## API Setup (Optional Features)

### The Odds API (for `odds` command)

Free tier: 500 requests/month (enough for daily use):

1. Go to https://the-odds-api.com
2. Sign up for a free account
3. Copy your API key from the dashboard
4. Set environment variable:

```bash
# Windows
set ODDS_API_KEY=your_api_key

# Mac/Linux
export ODDS_API_KEY=your_api_key
```

### Weather (for `weather` command)

**No API key needed.** Uses Open-Meteo which is completely free and open. Just works out of the box.

### Sport Data APIs (core functionality)

**No API keys needed.** All core data sources are free public APIs:

| API | Package | Data | Rate Limits |
|-----|---------|------|-------------|
| ESPN Public API | `requests` | NFL/NHL scores, schedules, injuries | No key needed |
| MLB Stats API | `MLB-StatsAPI` | MLB scores, player stats | No key needed |
| NBA Stats API | `nba_api` | NBA game logs, player stats | No key needed |
| nfl_data_py | `nfl_data_py` | NFL play-by-play, EPA, CPOE | No key needed |
| Statcast | `pybaseball` | MLB xwOBA, xERA, barrel rate | No key needed |
| Open-Meteo | `requests` | Weather forecasts (62 venues) | No key needed |

---

## Data Sources (All Free)

| Source | Data | Sports | API Key |
|--------|------|--------|---------|
| ESPN API | Scores, schedules, injuries | NFL, NHL | No |
| MLB Stats API | Scores, player stats | MLB | No |
| NBA API | Game logs, player stats | NBA | No |
| nfl_data_py | EPA/play, CPOE, success rate | NFL | No |
| pybaseball | Statcast xwOBA, xERA, barrel rate | MLB | No |
| Open-Meteo | Weather forecasts (62 venues) | NFL, MLB | No |
| The Odds API | 50+ bookmaker odds, CLV | All | Free (500/mo) |

---

## File Structure

```
Claude/
|-- README.md                    # This file
|-- requirements.txt             # All dependencies
|-- LICENSE                      # MIT License
|-- .gitignore                   # Git ignore rules
|
|-- # Shared prediction models (used by all sports)
|-- elo_set_handler.py           # Unified set command handler (39 params)
|-- mega_config.py               # Model on/off switches + mega set commands
|-- mega_backtest.py             # Walk-forward engine (orchestrates all models)
|-- mega_optimizer.py            # 5-phase optimization + single-model ablation
|-- meta_learner.py              # Super-stacker (XGBoost/ridge/logistic)
|-- hmm_model.py                 # Hidden Markov Model
|-- kalman_model.py              # Kalman Filter
|-- network_model.py             # PageRank + HITS + Eigenvector
|-- gbm_models.py                # LightGBM + CatBoost
|-- nn_models.py                 # MLP + LSTM (PyTorch)
|-- volatility_model.py          # GARCH + Lyapunov + Hurst
|-- signal_model.py              # Fourier + Wavelet
|-- survival_model.py            # Streak survival analysis
|-- copula_model.py              # Joint offense/defense distributions
|-- information_theory_model.py  # Shannon entropy + KL divergence
|-- momentum_model.py            # Newtonian momentum/inertia
|-- markov_chain_model.py        # Score transition matrices
|-- clustering_model.py          # k-Means team archetypes
|-- game_theory_model.py         # Nash equilibrium + style matchups
|-- poisson_model.py             # Poisson / Dixon-Coles score prediction
|-- glicko_model.py              # Glicko-2 uncertainty-aware ratings
|-- bradley_terry_model.py       # Bradley-Terry MLE paired comparison
|-- monte_carlo_model.py         # Monte Carlo simulation
|-- random_forest_model.py       # Random Forest (bagging diversity)
|-- classic_models.py            # SRS, Colley, Log5, PythagenPat, ExpSmoothing, MeanReversion
|-- odds_tracker.py              # Bookmaker odds + CLV tracking
|-- weather.py                   # Weather impact (62 venues)
|
|-- NFLClaude/                   # NFL-specific
|   |-- main.py                  # CLI entry point
|   |-- elo_model.py             # NFLElo class (32 teams)
|   |-- config.py                # Team definitions, settings I/O
|   |-- backtest.py              # 16 backtest methods (~2100 lines)
|   |-- enhanced_model.py        # XGBoost ensemble + SHAP
|   |-- advanced_stats.py        # EPA, CPOE, success rate, referee data
|   |-- build_model.py           # Model construction with season regression
|   |-- data_games.py            # ESPN game download
|   |-- data_players.py          # Player stats download
|   |-- injuries.py              # ESPN injury reports
|   |-- platt.py                 # Calibration (Platt, isotonic, beta)
|   |-- metrics.py               # Log loss, Brier, ECE, MCE, BSS
|   |-- predict_ledger.py        # Trading ledger (CSV-based lots)
|   |-- live_scores.py           # Live score tracking (60s refresh)
|   |-- auto_resolve.py          # Auto-settle finished trades
|   |-- html_generator.py        # Blogger HTML prediction tables
|   |-- help_system.py           # Interactive help text
|   |-- color_helpers.py         # Terminal color formatting
|   |-- accuracy_test.py         # Quick accuracy test
|   |-- run_optimize.py          # Optimization launcher
|   |-- quick_optimizer.py       # Rapid parameter sweep
|   |-- sweep_enhanced.py        # XGBoost weight sweep
|   |-- single_param_opt.py      # Coordinate descent optimizer
|   +-- requirements.txt         # NFL-specific dependencies
|
|-- MLBClaude/                   # MLB-specific (same structure + advanced_stats.py for Statcast)
|-- NBAClaude/                   # NBA-specific (same structure)
+-- NHLClaude/                   # NHL-specific (same structure)
```

---

## Backtest Results

Results on real game data with default settings (no optimization):

| Sport | Games | Baseline Elo Acc | Mega Log Loss vs Baseline |
|-------|-------|-----------------|--------------------------|
| NFL | 570 | 69.5% | -1.2% (better calibration) |
| NBA | 1,119 | 69.2% | +0.6% (needs `mega optimize`) |
| MLB | 3,162 | 56.7% | needs `mega optimize` |
| NHL | 2,982 | 57.9% | needs `mega optimize` |

Run `mega optimize` per sport to tune for your dataset. Run `mega ablation` to auto-prune models that don't help for that sport.

---

## Requirements

- Python 3.8+
- ~2GB disk for PyTorch CPU
- No GPU required
- No paid APIs
- Windows / Mac / Linux

## Disclaimer

This software is for educational and research purposes. Sports betting involves risk. Past performance does not guarantee future results. The authors are not responsible for any financial losses.
