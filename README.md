# Sports Prediction Mega-Ensemble

Four-sport prediction system (NFL, MLB, NBA, NHL) combining **31 independent prediction models** into a single meta-learned ensemble. Every prediction method from Elo ratings (1960) to neural networks (2017) to chaos theory (1892), all running together with multithreaded training and optional GPU acceleration.

**41,000+ lines of Python** | **31 models** | **53 tunable parameters** | **7-phase optimizer** | **Season-aware smart caching** | **No paid APIs**

---

## Table of Contents

- [Quick Start](#quick-start)
- [The 31 Models](#the-31-models)
- [Architecture](#architecture)
- [Complete Command Reference](#complete-command-reference)
- [All Settable Parameters](#all-settable-parameters)
- [Optimization System](#optimization-system)
- [API Setup](#api-setup-optional-features)
- [Smart Caching](#smart-caching)
- [Performance](#performance)
- [File Structure](#file-structure)
- [Backtest Results](#backtest-results)
- [Requirements](#requirements)

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install PyTorch (CPU - works everywhere)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. (Optional) Install PyTorch with GPU acceleration
pip install torch  # Auto-detects CUDA

# 4. Run any sport
cd NFLClaude && python main.py
cd MLBClaude && python main.py
cd NBAClaude && python main.py
cd NHLClaude && python main.py
```

First run auto-downloads 2 years of game data, player stats, and injury reports from free public APIs. No API keys required for core functionality.

### First-Time Workflow
```
python main.py          # Auto-downloads data, builds model, runs backtest
> backtest              # Verify baseline accuracy
> mega models           # See all 31 models with ON/OFF status
> mega                  # Run mega-ensemble backtest
> mega optimize         # Find best settings (takes hours, finds optimal)
> mega ablation         # Auto-disable models that hurt performance
```

---

## The 31 Models

Every major historical prediction method, spanning 340 years of mathematics and science, converted to sports:

### Tier 0: Core (Always On)
| Model | Year | Method | What It Does |
|-------|------|--------|-------------|
| **Elo** | 1960 | Rating system | 24+ contextual adjusters (home, rest, travel, injuries, etc.) |
| **XGBoost** | 2014 | Gradient boosting | 31 rolling features per team, walk-forward trained |

### Tier 1: Proven ML Models
| Model | Year | Method | What It Does |
|-------|------|--------|-------------|
| **HMM** | 1966 | Hidden Markov Model | Detects hot/normal/cold latent team states |
| **Kalman Filter** | 1960 | State estimation (NASA) | Optimal strength estimation with uncertainty tracking |
| **PageRank** | 1998 | Network analysis (Google) | Team importance from win graph + HITS hub/authority |
| **LightGBM** | 2017 | Leaf-wise boosting (Microsoft) | Fast gradient boosting, auto-scales for sample size |
| **CatBoost** | 2017 | Ordered boosting (Yandex) | Anti-overfitting gradient boosting |
| **MLP** | 1986 | Neural network | Deep feature interactions, GPU-accelerated |
| **LSTM** | 1997 | Recurrent neural net | Sequential game-by-game pattern detection |

### Tier 2: Exotic / Physics-Inspired
| Model | Year | Method | What It Does |
|-------|------|--------|-------------|
| **GARCH** | 1986 | Volatility modeling (finance) | Time-varying performance volatility |
| **Fourier/Wavelet** | 1807/1984 | Signal processing | Periodic performance cycle detection |
| **Survival** | 1700s | Hazard functions (actuarial) | Win/loss streak durability analysis |
| **Copula** | 1959 | Joint distributions (Sklar) | Offense-defense collapse probability |

### Tier 3: Information Theory & Physics
| Model | Year | Method | What It Does |
|-------|------|--------|-------------|
| **Shannon Entropy** | 1948 | Information theory | Team predictability scoring, KL divergence |
| **Momentum** | 1687 | Newtonian physics | Performance velocity, impulse, jerk tracking |
| **Markov Chains** | 1906 | Stochastic processes | Score transition matrices + steady state |
| **Clustering** | 1957 | k-Means | Team archetype classification + matchup matrix |
| **Game Theory** | 1950 | Nash equilibrium | Style matchups + head-to-head adaptation |

### Tier 4: Classical Rating Systems
| Model | Year | Method | What It Does |
|-------|------|--------|-------------|
| **Poisson** | 1837/1997 | Score distribution (Dixon-Coles) | Predicts exact score distributions |
| **Glicko-2** | 1995 | Uncertainty-aware ratings | Ratings with confidence intervals |
| **Bradley-Terry** | 1952 | MLE paired comparison | Globally optimal strength via maximum likelihood |
| **Monte Carlo** | 1946 | Simulation (Manhattan Project) | 2000+ game simulations per matchup |
| **Random Forest** | 2001 | Bagging (Breiman) | Ensemble diversity via bootstrap aggregation |

### Tier 5: Sports-Specific Classical Models
| Model | Year | Method | What It Does |
|-------|------|--------|-------------|
| **SRS** | - | Simple Rating System | Margin-adjusted strength of schedule |
| **Colley Matrix** | 2001 | BCS ranking | Bias-free ranking from linear system |
| **Log5** | 1981 | Bill James formula | Head-to-head probability from win rates |
| **PythagenPat** | 2000 | Dynamic Pythagorean | Dynamic exponent based on run environment |
| **Exp Smoothing** | 1957 | Holt double smoothing | Trend-aware performance forecasting |
| **Mean Reversion** | 1983 | Bollinger bands | Z-score signals for regression to mean |

### Tier 6: Data Enrichment (Optional)
| Model | Key Required | What It Does |
|-------|-------------|-------------|
| **Weather** | No | Temperature, wind, precipitation impact (62 venues) |
| **Market Odds** | Yes (free) | 50+ bookmaker odds, closing line value tracking |

---

## Architecture

```
                    +------------------+
                    |   Game Data      |  (ESPN, MLB Stats API, NBA API)
                    +--------+---------+
                             |
              +--------------+--------------+
              |              |              |
         [Elo Model]   [31 Base Models]  [Data Enrichment]
         (24+ adjusters) (HMM, Kalman,   (Weather, Odds)
                          PageRank, etc)
              |              |              |
              +--------------+--------------+
                             |
                    +--------+---------+
                    | Feature Matrix   |  (100+ features per game)
                    +--------+---------+
                             |
                    +--------+---------+
                    | Meta-Learner     |  (XGBoost / Ridge / Logistic)
                    +--------+---------+
                             |
                    +--------+---------+
                    | Bounded Adjust   |  Elo + clamp(meta_adj, +/- max_adj)
                    +--------+---------+
                             |
                    +--------+---------+
                    | Final Probability|  (calibrated, constrained)
                    +------------------+
```

**Key design**: Elo-anchored bounded adjustment. The Elo probability is always the foundation. The meta-learner can only adjust it by `+/- max_adj` (default 8%). This prevents overfitting while letting exotic signals contribute at the margins.

**Multithreaded training**: LightGBM, CatBoost, Random Forest, and MLP train in parallel threads during the retrain cycle.

**GPU acceleration**: MLP and LSTM automatically use CUDA when a GPU is detected.

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
| `odds` | Show today's odds from 50+ bookmakers (needs free API key, 15-min cache) |
| `weather` | Weather impact report for a game venue (free, no key needed) |
| `advstats` | Advanced analytics rankings (NFL: EPA/CPOE; also: `epa`) |
| `statcast` | Statcast rankings (MLB: xwOBA/xERA; also: `advstats`) |
| `officials` | Referee tendency analysis (NFL only) |

### Backtesting

| Command | Description |
|---------|-------------|
| `backtest` | Walk-forward Elo backtest with Platt scaler fitting |
| `enhanced` | XGBoost ensemble backtest (80/20 Elo+XGBoost blend) + SHAP |
| `enhanced decay` | Time-decayed ensemble (Elo weight 95% early to 70% late) |
| `shap` | SHAP feature importance for XGBoost ensemble |

### Elo Optimization

| Command | Description | Time |
|---------|-------------|------|
| `grid` | Interactive grid search (7 params, customizable ranges) | 5-30m |
| `genetic` | Genetic algorithm with scipy differential evolution | 5-15m |
| `bayesian` | Gaussian Process + Expected Improvement optimization | 5-10m |
| `autoopt` | Automatic grid + genetic + bayesian pipeline | 15-30m |
| `superopt` | Exhaustive 7-phase optimization, all 9 params | 2-4h |
| `singleopt` | Coordinate descent single-parameter optimization | 10-30m |
| `results` | Show best parameters from all optimizer runs + DSR significance | instant |

### Validation & Statistical Testing

| Command | Description | Time |
|---------|-------------|------|
| `purgedcv` | Purged walk-forward cross-validation (k-fold with embargo gap) | 5m |
| `cpcv` | Combinatorial purged CV (all C(k, k_test) train/test paths) | 10m |
| `pbo` | Probability of backtest overfitting (symmetric CV) | 10m |
| `montecarlo` | Monte Carlo permutation test (p-value, 500 iterations) | 8m |
| `rollingcal` | Rolling origin Platt recalibration (out-of-sample metrics) | 10s |
| `kelly` | Kelly criterion position sizing backtest | 5s |
| `sliding` | Sliding vs expanding window comparison | 30s |
| `convergence` | Elo rating convergence / burn-in analysis | 10s |
| `conformal` | Conformal prediction intervals (coverage guarantees) | 10s |
| `betacal` | Beta calibration (3-parameter asymmetric) | 5s |
| `platt` | Refit Platt calibration scaler (also: `calibrate`) | 5s |

### Mega-Ensemble

| Command | Description | Time |
|---------|-------------|------|
| `mega` | Run full mega-ensemble backtest (26+ models) | 1-5m |
| `mega optimize` | Exhaustive 7-phase optimization (also: `megaopt`) | 2-8h |
| `mega quick` | Quick grid search, Phase 1 only | 30-90m |
| `mega ablation` | Single-model contribution test + auto-prune | 30-60m |
| `mega models` | Show all 31 models with ON/OFF status (also: `models`) | instant |
| `mega on <model>` | Enable a model (e.g., `mega on lstm`) | instant |
| `mega off <model>` | Disable a model (e.g., `mega off monte_carlo`) | instant |
| `mega on all` | Enable all models | instant |
| `mega enable <model>` | Same as `mega on` | instant |
| `mega disable <model>` | Same as `mega off` | instant |
| `mega settings` | Show all mega parameter values (also: `mega params`) | instant |
| `mega set <p>=<v>` | Set a mega parameter (e.g., `mega set adj=0.10`) | instant |

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

## All Settable Parameters

### Elo Parameters (39 total) - `set param=value`

| Category | Parameter | Aliases | Type | Description |
|----------|-----------|---------|------|-------------|
| **Core** | `k` | k_factor | float | K-factor (learning rate per game) |
| | `base_rating` | base, rating | float | Starting Elo rating |
| | `home_adv` | home, hca | float | Home court/field advantage (Elo points) |
| | `use_mov` | mov, margin | bool | Use margin of victory (true/false) |
| **Player** | `player_boost` | boost, player | float | Team-level player strength boost |
| | `starter_boost` | starter, sp_boost | float | Starting pitcher/QB quality |
| | `bullpen_factor` | bullpen, reliever | float | Bullpen/reliever quality factor |
| | `opp_pitcher_factor` | opp_pitcher, opp_sp | float | Opponent pitcher adjustment |
| **MOV** | `mov_base` | mov_mult, mov_constant | float | MOV log curve shift constant |
| | `mov_cap` | movcap, margin_cap | float | Maximum MOV adjustment cap |
| **Rest** | `rest_factor` | rest | float | Rest days advantage factor |
| | `rest_advantage_cap` | restcap, rest_cap | float | Maximum rest advantage multiplier |
| | `b2b_penalty` | b2b, back_to_back | float | Back-to-back game penalty |
| **Travel** | `travel_factor` | travel | float | Elo penalty per timezone crossed |
| | `east_travel_penalty` | east_travel, eastbound | float | Extra penalty for eastbound travel |
| | `road_trip_factor` | roadtrip, road_trip | float | Extended road trip penalty |
| | `homestand_factor` | homestand | float | Extended homestand bonus |
| | `altitude_factor` | altitude, alt | float | Altitude bonus (Denver/Colorado) |
| | `park_factor_weight` | parkfactor, park_factor | float | Park factor weight |
| **Form** | `form_weight` | form | float | Recent form weight |
| | `win_streak_factor` | streak, win_streak | float | Win/loss streak momentum |
| | `mean_reversion` | reversion, regress | float | Regression after extreme results |
| | `season_regress` | season_regression, regress_pct | float | Season boundary regression % |
| **Matchup** | `sos_factor` | sos | float | Strength of schedule weight |
| | `division_factor` | division, div | float | Divisional game confidence reducer |
| | `interleague_factor` | interleague, il_factor | float | Interleague game adjustment |
| | `series_adaptation` | series, adaptation | float | Series/rematch adaptation factor |
| **Scoring** | `pace_factor` | pace, tempo | float | Run/scoring environment mismatch |
| | `pyth_factor` | pyth, pythagorean | float | Pythagorean expected W% adjustment |
| | `scoring_consistency_factor` | consistency | float | Penalty for volatile scoring |
| | `home_road_factor` | home_road, split | float | Home/road split bonus |
| **Season** | `playoff_hca_factor` | playoff, playoff_hca | float | Playoff home advantage multiplier |
| | `season_phase_factor` | phase, season_phase | float | Early-season dampener |
| **K-factor** | `k_decay` | kdecay, k_reduction | float | K-factor decay over season |
| | `surprise_k` | surprise, upset_k | float | Extra K for surprise results |
| **NFL** | `bye_week_factor` | bye, bye_week | float | Bye week advantage (NFL only) |
| **Account** | `kelly_fraction` | kelly | special | Kelly fraction (quarter/half/0.25/0.50) |
| | `starting_balance` | balance, bankroll | float | Starting account balance |
| | `autoresolve_enabled` | autoresolve, auto_resolve | bool | Auto-settle finished trades |

### Mega Parameters (14 total) - `mega set param=value`

| Parameter | Aliases | Type | Description |
|-----------|---------|------|-------------|
| `max_adj` | adj, maxadj, adjustment | float | Max meta-learner probability adjustment |
| `meta_model` | meta, metalearner, stacker | str | Meta-learner algorithm (ridge/logistic/xgboost) |
| `retrain_every` | retrain, retrain_interval | int | Retrain meta-learner every N games |
| `min_train` | mintrain, min_games, warmup | int | Games before meta-learner starts predicting |
| `kalman_process_noise` | kalman_pn, process_noise, pn | float | Kalman filter process noise |
| `kalman_measurement_noise` | kalman_mn, measurement_noise, mn | float | Kalman filter measurement noise |
| `hmm_states` | hmm_n, n_states, states | int | Number of HMM hidden states |
| `network_decay` | net_decay, pagerank_decay, decay | float | PageRank temporal decay (0-1) |
| `momentum_friction` | friction, mom_friction | float | Momentum friction coefficient |
| `n_clusters` | clusters, k_clusters, nclusters | int | Number of team archetype clusters |
| `glicko_initial_rd` | glicko_rd, initial_rd, rd | float | Glicko-2 initial rating deviation |
| `bt_decay` | bt_recency, bradley_decay | float | Bradley-Terry recency decay (0-1) |
| `mc_simulations` | mc_sims, simulations, n_sims, sims | int | Monte Carlo simulations per game |
| `window` | rolling_window, feat_window | int | Rolling feature window size (games) |

---

## Optimization System

### Elo Optimization (per-sport backtest.py)

| Phase | Command | Params | Method | Time |
|-------|---------|--------|--------|------|
| 1 | `grid` | 7 | Cartesian product grid search | 5-30m |
| 2 | `genetic` | 7 | scipy differential_evolution | 5-15m |
| 3 | `bayesian` | 7 | GP surrogate + Expected Improvement | 5-10m |
| 4 | `autoopt` | 7 | Automatic grid + genetic + bayesian | 15-30m |
| 5 | `superopt` | 9 | Exhaustive 7-phase all-parameter search | 2-4h |
| 6 | `singleopt` | All | Coordinate descent one-at-a-time | 10-30m |

Objective: `LogLoss * 8 + Brier * 40` (calibration-focused)

### Mega Optimization (`mega optimize`)

7-phase exhaustive search over all mega-ensemble parameters:

| Phase | Method | What It Searches | Evaluations |
|-------|--------|-----------------|-------------|
| 1 | Grid search | max_adj x meta_model x retrain_every | ~150 |
| 2 | Coordinate descent | 8 secondary params (Kalman, HMM, network, etc.) | ~50-100 |
| 3 | Bayesian | Random perturbation around best config | 30 |
| 4 | Tournament | Top 10 configs re-compete head-to-head | 10 |
| 5 | Stability | Verify best isn't overfit (3 re-runs) | 3 |
| 6 | Genetic (DE) | scipy differential_evolution on all continuous params | ~120 |
| 7 | Super-optimize | Fine-grain sweep around best values | ~40 |

### `mega ablation` - Single-Model Contribution Test

1. Runs baseline with all enabled models
2. Turns OFF each model one at a time, re-runs backtest
3. Measures each model's individual contribution (delta in objective)
4. Ranks all models: **HELPS** / **neutral** / **HURTS**
5. Auto-disables models that hurt performance
6. Verifies improvement, reverts if no gain

### Recommended Optimization Workflow

```bash
# Step 1: Optimize Elo parameters first
> autoopt                    # 15-30 min

# Step 2: Run mega-ensemble with default settings
> mega                       # See baseline

# Step 3: Find best mega-ensemble settings
> mega optimize              # 2-8 hours (all 7 phases)

# Step 4: Auto-prune unhelpful models
> mega ablation              # 30-60 min

# Step 5: Verify final performance
> mega                       # Should be better than Step 2
```

---

## API Setup (Optional Features)

### The Odds API (for `odds` command)

Free tier: 500 requests/month (smart 15-min cache makes this last all month):

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

Or create a file `.odds_api_key` in the project root with just the key.

### Weather (for `weather` command)

**No API key needed.** Uses Open-Meteo, completely free and open. Works out of the box for all 62 NFL/MLB venues.

### Sport Data APIs (core functionality)

**No API keys needed.** All core data comes from free public APIs:

| API | Package | Data | Rate Limits |
|-----|---------|------|-------------|
| ESPN Public API | `requests` | NFL/NHL scores, schedules, injuries | No key, 0.5s delay |
| MLB Stats API | `MLB-StatsAPI` | MLB scores, player stats | No key |
| NBA Stats API | `nba_api` | NBA game logs, player stats | No key, 1s delay |
| nfl_data_py | `nfl_data_py` | NFL play-by-play, EPA, CPOE | No key |
| Statcast | `pybaseball` | MLB xwOBA, xERA, barrel rate | No key |
| Open-Meteo | `requests` | Weather forecasts (62 venues) | No key |
| The Odds API | `requests` | 50+ bookmaker odds | Free key (500/mo) |

---

## Smart Caching

The system uses season-aware caching (`cache_utils.py`) to minimize API calls:

| Data Type | Offseason | Game Day (in-season) | Non-Game Day |
|-----------|-----------|---------------------|--------------|
| **Games** | 1 week | 4 hours | 12 hours |
| **Players** | 30 days | 48 hours | 48 hours |
| **Injuries** | 30 days | 2 hours | 6 hours |
| **Odds** | Never fetch | 15 minutes | 4 hours |
| **Weather** | Never fetch | 2 hours | 12 hours |
| **Advanced stats** | 30 days | 24 hours | 24 hours |

Season calendars: NFL (Sep-Feb), MLB (Mar-Oct), NBA (Oct-Jun), NHL (Oct-Jun).

---

## Performance

### Multithreading

Model training runs in parallel threads during the retrain cycle:
- LightGBM, CatBoost, Random Forest, and MLP train simultaneously
- `ThreadPoolExecutor` with up to 4 workers
- Reduces retrain time by ~60% on multi-core systems

### GPU Acceleration

PyTorch models (MLP, LSTM) automatically detect and use CUDA GPUs:
- Install `torch` with CUDA support: `pip install torch` (auto-detects)
- Or CPU-only: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- GPU status shown at startup when detected
- Falls back to CPU transparently when no GPU available

### Backtest Results

Results on real game data with default settings (no optimization):

| Sport | Games | Baseline Elo Acc | Baseline LL | Mega LL | Mega vs Baseline |
|-------|-------|-----------------|-------------|---------|-----------------|
| **NFL** | 570 | 69.5% | 0.6297 | 0.6229 | **-1.1% better** |
| NBA | 1,119 | 69.2% | 0.6032 | 0.6069 | needs `mega optimize` |
| MLB | 3,162 | 56.7% | 0.6840 | - | needs `mega optimize` |
| NHL | 2,982 | 57.9% | 0.7088 | - | needs `mega optimize` |

Run `mega optimize` per sport to tune for your dataset. Run `mega ablation` to auto-prune models that don't help for that specific sport.

---

## File Structure

```
Claude/
|
|-- README.md                    # This file
|-- requirements.txt             # All dependencies
|-- LICENSE                      # MIT License
|-- .gitignore                   # Git ignore rules
|
|-- # Infrastructure
|-- cache_utils.py               # Smart season-aware API caching
|-- elo_set_handler.py           # Unified set command handler (39 params, 116 aliases)
|-- mega_config.py               # Model on/off switches + mega set commands (14 params)
|-- mega_backtest.py             # Walk-forward engine (orchestrates all 31 models)
|-- mega_optimizer.py            # 7-phase optimization + single-model ablation
|-- meta_learner.py              # Super-stacker (XGBoost / Ridge / Logistic meta-learner)
|
|-- # Tier 1: Proven ML Models
|-- hmm_model.py                 # Hidden Markov Model (hot/cold state detection)
|-- kalman_model.py              # Kalman Filter (uncertainty-aware strength)
|-- network_model.py             # PageRank + HITS + Eigenvector centrality
|-- gbm_models.py                # LightGBM + CatBoost (auto-scales for sample size)
|-- nn_models.py                 # MLP + LSTM with GPU auto-detection
|
|-- # Tier 2: Exotic / Physics-Inspired
|-- volatility_model.py          # GARCH + Lyapunov exponents + Hurst exponent
|-- signal_model.py              # Fourier FFT + Wavelet decomposition
|-- survival_model.py            # Streak survival / hazard rate analysis
|-- copula_model.py              # Joint offense-defense dependency modeling
|
|-- # Tier 3: Information Theory & Physics
|-- information_theory_model.py  # Shannon entropy + KL divergence + conditional entropy
|-- momentum_model.py            # Newtonian momentum, impulse, angular momentum, jerk
|-- markov_chain_model.py        # Transition matrices + steady-state distributions
|-- clustering_model.py          # k-Means team archetypes + matchup advantage matrix
|-- game_theory_model.py         # Nash equilibrium, style matchups, H2H adaptation
|
|-- # Tier 4: Classical Rating Systems
|-- poisson_model.py             # Poisson / Dixon-Coles exact score distributions
|-- glicko_model.py              # Glicko-2 with rating deviation tracking
|-- bradley_terry_model.py       # Bradley-Terry MLE (globally optimal strengths)
|-- monte_carlo_model.py         # Bootstrap + kernel density simulation (2000+ games)
|-- random_forest_model.py       # Pure-numpy Random Forest (no sklearn needed)
|
|-- # Tier 5: Sports-Specific Classical
|-- classic_models.py            # SRS, Colley Matrix, Log5, PythagenPat,
|                                # Exponential Smoothing, Mean Reversion
|
|-- # Tier 6: Data Enrichment
|-- odds_tracker.py              # Bookmaker odds + CLV tracking (15-min cache)
|-- weather.py                   # Open-Meteo weather impact (62 venues, no key)
|
|-- NFLClaude/                   # NFL-specific (32 teams)
|   |-- main.py                  # CLI entry point + all command dispatch
|   |-- elo_model.py             # NFLElo class (K=20, HomeAdv=48, bye week)
|   |-- config.py                # Team definitions, smart cache, settings I/O
|   |-- backtest.py              # 16 backtest + 6 optimization methods (~2100 lines)
|   |-- enhanced_model.py        # XGBoost ensemble + SHAP (31 features, 5-game window)
|   |-- advanced_stats.py        # nfl_data_py: EPA, CPOE, success rate, referees
|   |-- build_model.py           # Model construction with season regression
|   |-- data_games.py            # ESPN scoreboard API (incremental download)
|   |-- data_players.py          # ESPN byathlete API (passing/rushing/receiving)
|   |-- injuries.py              # ESPN injuries API (4h cache, position-based impact)
|   |-- platt.py                 # Calibration (Platt, isotonic, beta, conformal)
|   |-- metrics.py               # Log loss, Brier, ECE, MCE, BSS, DSR
|   |-- predict_ledger.py        # CSV trading ledger with P&L tracking
|   |-- live_scores.py           # Live score polling (60s refresh)
|   |-- auto_resolve.py          # Auto-settle trades from live scores
|   |-- html_generator.py        # Blogger-ready HTML prediction tables
|   |-- help_system.py           # Interactive help text for all commands
|   |-- color_helpers.py         # Terminal color formatting (cok, cerr, cwarn, etc.)
|   +-- requirements.txt
|
|-- MLBClaude/                   # MLB-specific (30 teams, K=4, starting pitcher Elo)
|-- NBAClaude/                   # NBA-specific (30 teams, K=8.38, Pythagorean exp=14)
+-- NHLClaude/                   # NHL-specific (32 teams, goalie tracking)
```

---

## Requirements

### System
- Python 3.8+
- ~2GB disk for PyTorch
- Multi-core CPU recommended (multithreaded training)
- GPU optional (CUDA for MLP/LSTM acceleration)
- Windows / Mac / Linux

### Core Dependencies
```
pandas>=1.5        numpy>=1.24       scipy>=1.10
colorama>=0.4      xgboost>=2.0      requests>=2.28
matplotlib>=3.7
```

### Sport Data APIs
```
nfl_data_py>=0.3   MLB-StatsAPI>=1.7  nba_api>=1.1
pybaseball>=2.3
```

### Prediction Models
```
hmmlearn>=0.3      filterpy>=1.4      lightgbm>=4.0
catboost>=1.2      networkx>=3.0      torch>=2.0
```

### Optional
```
nolds              # Lyapunov exponents, Hurst exponent
PyWavelets         # Wavelet transforms
```

---

## Disclaimer

This software is for educational and research purposes. Sports betting involves significant financial risk. Past performance does not guarantee future results. The authors are not responsible for any financial losses incurred through use of this software.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
