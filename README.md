# Sports Prediction Mega-Ensemble

Four-sport prediction system (NFL, MLB, NBA, NHL) combining **32 independent prediction models** into a single meta-learned ensemble. Every prediction method from Elo ratings to neural networks to chaos theory, all running together.

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

## The 32 Models

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
| | Reddit Sentiment (VADER) | Hutto & Gilbert | 2014 |
| | Market Odds / CLV | The Odds API | - |

## Architecture

```
Game Data --> [32 Base Models] --> Feature Matrix --> Meta-Learner --> Bounded Adj --> Final Prob
                                                                         |
                                                              Elo probability (anchor)
                                                              + meta adjustment (+/- max_adj)
```

The system uses an **Elo-anchored bounded adjustment**: the Elo probability is the foundation, and the meta-learner (trained on all models' features) adjusts it by a bounded amount. This prevents overfitting while letting exotic signals contribute at the margins.

## Commands

### Predictions
```
chiefs              # Predict: enter team name, then opponent, then home/away
today               # Show today's games with predictions
tomorrow            # Show tomorrow's games
```

### Mega-Ensemble
```
mega                # Run full mega-ensemble backtest (26+ models)
mega optimize       # Exhaustive 5-phase optimization (hours, finds best settings)
mega quick          # Quick grid search (Phase 1 only)
mega ablation       # Test each model's individual contribution, auto-prune bad ones
mega models         # Show all 32 models with ON/OFF status
mega on <model>     # Enable a model (e.g., mega on lstm)
mega off <model>    # Disable a model (e.g., mega off monte_carlo)
mega on all         # Enable all models
mega settings       # Show all mega-ensemble parameter values
mega set adj=0.10   # Set a mega parameter (see full list below)
```

### Elo Settings
```
set                 # Show all 39 Elo parameters with current values
set k=10            # Set K-factor
set home=48         # Set home advantage
set rest=20         # Set rest factor
set b2b=75          # Set back-to-back penalty
settings            # Show current Elo settings
```

### Backtesting & Optimization
```
backtest            # Walk-forward Elo backtest
enhanced            # XGBoost ensemble backtest
grid                # Grid search optimization
genetic             # Genetic algorithm (scipy DE)
bayesian            # Bayesian optimization (GP + EI)
autoopt             # Full auto-optimization pipeline
superopt            # Exhaustive 9-parameter search
```

### Validation
```
purgedcv            # Purged walk-forward cross-validation
cpcv                # Combinatorial purged CV
pbo                 # Probability of backtest overfitting
montecarlo          # Monte Carlo permutation test
convergence         # Elo convergence analysis
sliding             # Sliding window backtest
conformal           # Conformal prediction sets
```

### Trading
```
predicts            # View open positions
buy                 # Add a position
sell                # Sell a position
resolve             # Resolve a finished game
live                # Live score tracking (60s refresh)
autoresolve         # Auto-settle finished trades
kelly               # Kelly criterion backtest
chart               # P&L chart
```

### Data
```
all                 # Show all team ratings
refresh             # Re-download all data
players             # Show player stats
injuries            # Show injury report
odds                # Today's odds from 50+ bookmakers
weather             # Weather impact for game venues
advstats / epa      # Advanced stats (EPA, CPOE, xwOBA, etc.)
```

## Settable Parameters

### Elo Parameters (39 total, `set param=value`)

| Category | Parameter | Aliases | Description |
|----------|-----------|---------|-------------|
| Core | `k` | k_factor | K-factor (learning rate) |
| | `base_rating` | base, rating | Starting Elo |
| | `home_adv` | home, hca | Home advantage (Elo pts) |
| | `use_mov` | mov, margin | Margin of victory (true/false) |
| Player | `player_boost` | boost, player | Team player strength |
| | `starter_boost` | starter, sp_boost | Starting pitcher quality |
| | `bullpen_factor` | bullpen, reliever | Bullpen quality |
| | `opp_pitcher_factor` | opp_pitcher | Opponent pitcher |
| MOV | `mov_base` | mov_mult | MOV log curve shift |
| | `mov_cap` | movcap, margin_cap | Maximum MOV cap |
| Rest | `rest_factor` | rest | Rest days advantage |
| | `rest_advantage_cap` | restcap | Max rest multiplier |
| | `b2b_penalty` | b2b | Back-to-back penalty |
| Travel | `travel_factor` | travel | Per-timezone penalty |
| | `east_travel_penalty` | east_travel | Eastbound extra penalty |
| | `road_trip_factor` | roadtrip | Extended road trip penalty |
| | `homestand_factor` | homestand | Homestand bonus |
| | `altitude_factor` | altitude, alt | Denver/Colorado bonus |
| | `park_factor_weight` | parkfactor | Park factor weight |
| Form | `form_weight` | form | Recent form weight |
| | `win_streak_factor` | streak | Streak momentum |
| | `mean_reversion` | reversion | Regression to mean |
| | `season_regress` | regress_pct | Season boundary regression |
| Matchup | `sos_factor` | sos | Strength of schedule |
| | `division_factor` | division | Divisional game adjust |
| | `interleague_factor` | interleague | Interleague adjust |
| | `series_adaptation` | series | Rematch adaptation |
| Scoring | `pace_factor` | pace, tempo | Scoring environment |
| | `pyth_factor` | pyth | Pythagorean W% adjust |
| | `scoring_consistency_factor` | consistency | Volatility penalty |
| | `home_road_factor` | home_road | Home/road split bonus |
| Season | `playoff_hca_factor` | playoff | Playoff HCA multiplier |
| | `season_phase_factor` | phase | Early-season dampener |
| K-factor | `k_decay` | kdecay | K decay over season |
| | `surprise_k` | surprise | Extra K for upsets |
| Account | `kelly_fraction` | kelly | Kelly sizing (quarter/half) |
| | `starting_balance` | balance, bankroll | Starting bankroll |
| | `autoresolve_enabled` | autoresolve | Auto-settle trades |

### Mega Parameters (14 total, `mega set param=value`)

| Parameter | Aliases | Description |
|-----------|---------|-------------|
| `max_adj` | adj, maxadj | Max meta-learner adjustment |
| `meta_model` | meta, stacker | Meta type (ridge/logistic/xgboost) |
| `retrain_every` | retrain | Retrain interval (games) |
| `min_train` | mintrain, warmup | Warmup games before predicting |
| `kalman_process_noise` | kalman_pn, pn | Kalman process noise |
| `kalman_measurement_noise` | kalman_mn, mn | Kalman measurement noise |
| `hmm_states` | states, hmm_n | HMM hidden states |
| `network_decay` | decay, net_decay | PageRank temporal decay |
| `momentum_friction` | friction | Momentum friction coefficient |
| `n_clusters` | clusters | Team archetype clusters |
| `glicko_initial_rd` | rd, glicko_rd | Glicko-2 initial RD |
| `bt_decay` | bt_recency | Bradley-Terry recency |
| `mc_simulations` | sims, mc_sims | Monte Carlo sims per game |
| `window` | rolling_window | Rolling feature window |

## Optimization System

### `mega optimize` - Exhaustive 5-Phase Search

1. **Grid Search** - 150 combinations of max_adj x meta_model x retrain_every
2. **Coordinate Descent** - 8 secondary params swept one-at-a-time
3. **Bayesian Optimization** - 30 Gaussian perturbation trials
4. **Tournament** - Top 10 configs re-compete head-to-head
5. **Stability Check** - Verify best config isn't overfit

### `mega ablation` - Single-Model Contribution Test

Tests each model individually by turning it off and measuring the impact. Ranks all models from most helpful to least, and auto-disables any that hurt performance.

## Data Sources (All Free)

| Source | Data | Sports |
|--------|------|--------|
| ESPN API | Scores, schedules, injuries | NFL, NHL |
| MLB Stats API | Scores, player stats | MLB |
| NBA API | Game logs, player stats | NBA |
| nfl_data_py | EPA, CPOE, success rate | NFL |
| pybaseball | Statcast xwOBA, xERA | MLB |
| Open-Meteo | Weather forecasts | NFL, MLB |
| Reddit API | Fan sentiment | All |
| The Odds API | Bookmaker odds (free tier) | All |

No paid APIs. No API keys required for core functionality (Reddit and Odds need free API keys for those optional features).

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
|-- sentiment.py                 # Reddit sentiment (VADER)
|
|-- NFLClaude/                   # NFL-specific
|   |-- main.py                  # CLI entry point
|   |-- elo_model.py             # NFLElo class
|   |-- config.py                # 32 NFL teams, settings I/O
|   |-- backtest.py              # 16 backtest methods (~2100 lines)
|   |-- enhanced_model.py        # XGBoost ensemble + SHAP
|   |-- advanced_stats.py        # EPA, CPOE, referee data
|   |-- data_games.py            # ESPN game download
|   |-- data_players.py          # Player stats
|   |-- injuries.py              # ESPN injury reports
|   |-- platt.py                 # Calibration (Platt, isotonic, beta)
|   |-- metrics.py               # Log loss, Brier, ECE, BSS
|   |-- predict_ledger.py        # Trading ledger
|   |-- live_scores.py           # Live score tracking
|   |-- auto_resolve.py          # Auto-settle trades
|   |-- html_generator.py        # Blogger HTML tables
|   +-- requirements.txt
|
|-- MLBClaude/                   # MLB-specific (same structure)
|-- NBAClaude/                   # NBA-specific (same structure)
+-- NHLClaude/                   # NHL-specific (same structure)
```

## Backtest Results

Results on real game data with default settings (no optimization):

| Sport | Games | Baseline Elo Acc | Mega Log Loss vs Baseline |
|-------|-------|-----------------|--------------------------|
| NFL | 570 | 69.5% | -1.2% (better) |
| NBA | 1,119 | 69.2% | +0.6% (needs optimization) |
| MLB | 3,162 | 56.7% | needs optimization |
| NHL | 2,982 | 57.9% | needs optimization |

Run `mega optimize` per sport to tune for your dataset. Run `mega ablation` to auto-prune models that don't help.

## Requirements

- Python 3.8+
- ~2GB disk for PyTorch CPU
- No GPU required
- No paid APIs
- Windows/Mac/Linux

## Disclaimer

This software is for educational and research purposes. Sports betting involves risk. Past performance does not guarantee future results. The authors are not responsible for any financial losses.
