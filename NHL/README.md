# NHL Prediction System (SharpStack)

## Overview

NHL game prediction system built on an Elo rating engine with Platt-scaled calibration to produce win probabilities for every NHL matchup. Integrates a Predicts $1 binary contract trading ledger for position tracking and P&L analysis. Runs as an interactive CLI application -- no web server, no build system.

NHL-specific design priorities: per-goalie cumulative Elo sub-ratings (the system's most distinctive feature -- a starting goaltender can single-handedly win or lose a hockey game), overtime factor for OT/shootout outcomes, frequent back-to-back games in the 82-game schedule, two altitude teams (Colorado at 5,280 ft and Utah at 4,226 ft with proportional bonuses), goalie-weighted injury impact (star goalie out = -35 Elo), and cross-year season handling (Oct--Jun).

---

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

On first launch the system will:
1. Download two years of game data via the ESPN public API
2. Download player stats (skater points/goals/assists + goalie GAA/SVP/wins/shutouts)
3. Fetch injury reports from the ESPN API
4. Build Elo ratings for all 32 NHL teams
5. Run a walk-forward backtest to fit the Platt calibration scaler
6. Enter the interactive CLI loop

Type a team name (e.g. `Bruins`) to start a prediction. Type `help` for all commands.

---

## Commands

### Data and Display
| Command | Description |
|---------|-------------|
| `all` | Show all 32 teams ranked by Elo rating |
| `refresh` | Delete cached data and re-download everything, rebuild model |
| `players` | Show top player stats and composite scores |
| `settings` | Display current Elo parameter values |
| `injuries` | Show injury report with goalie-weighted Elo impact per team |
| `injuries set <team> <player1>, <player2>` | Manually mark players as OUT |
| `today` / `html` | Generate Blogger-ready HTML predictions for today's games |
| `tomorrow` | Generate predictions for tomorrow's games |

### Data Enrichment
| Command | Description |
|---------|-------------|
| `odds` | Show live moneyline odds from The Odds API |
| `kalshi` | Show Kalshi prediction market contract prices |
| `weather` | Show weather data (less relevant for indoor arenas) |

### Backtesting
| Command | Description |
|---------|-------------|
| `backtest` | Run walk-forward backtest, fit Platt scaler, report accuracy/log loss/Brier |

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
| `kelly` | Kelly Criterion bankroll simulation (quarter-Kelly default) |

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
| `live` | Show live scores with period display (P1, P2, P3, OT, SO) |
| `portfolio` | Cross-sport portfolio summary |

### Elo Settings
Type `set` to see all tunable parameters. Examples:
```
set k=5.16          set home=41         set boost=36
set starter=42      set rest=39         set b2b=18
set travel=17       set sos=32          set pace=7
set div=29          set altitude=4      set playoff=0.94
set form=8          set ot=5            set streak=2
set kelly=quarter   set balance=100     set autoresolve=true
```

After any parameter change, run `backtest` to refit the Platt calibration scaler.

---

## Architecture

The system uses a three-layer prediction pipeline. Each layer adds refinement on top of the previous one.

### Layer 1: Elo Model

The `NHLElo` class in `elo_model.py` maintains Elo ratings for all 32 NHL teams. The base rating is 1500. Each game updates ratings using a configurable K-factor (default 5.16 -- the lowest of any sport in the system, because the 82-game NHL season demands gradual, stable rating changes). The model applies 20+ adjusters:

- **Home ice advantage** (41.43 Elo, ~56% implied home win rate)
- **Starting goalie quality** (41.87 starter boost -- the most impactful single adjuster in the NHL system)
- **Per-goalie cumulative Elo** (K_GOALIE=6, 50% season regression -- tracks individual goalie performance)
- **Player strength** (36.36 boost, composite: 55% skaters / 45% goalies)
- **Rest days** (39.03 factor -- major for travel-heavy schedule with frequent back-to-backs)
- **Back-to-back penalty** (18.0 -- significant in hockey; much higher than NBA's 0.0)
- **Travel fatigue** (16.79 -- timezone-based distance)
- **Strength of schedule** (32.41 -- strong signal in NHL)
- **Division rivalry** (29.25 -- strong in NHL divisional play)
- **Mean reversion** (37.79 -- regression after extreme results)
- **Overtime factor** (5.0 -- adjustment for OT/shootout game outcomes)
- **Altitude** (4.0 multiplier -- Colorado Avalanche at 5,280 ft gets full bonus; Utah Hockey Club at 4,226 ft gets proportional bonus)
- **Pace mismatch** (7.36)
- **Season phase** (2.5 -- early/mid/late adjustment)
- **Playoff HCA factor** (0.94 -- slightly reduced playoff home advantage)
- **Season regression** (33% pull toward mean at season boundaries)
- **Margin of victory** uses `log(max(1.0, abs(goal_diff)) + 1.0)` -- log compression for hockey's lower-scoring games

### The Goaltender System

The NHL system's most distinctive feature is its per-goalie cumulative Elo sub-rating system. Because a starting goaltender can single-handedly determine a hockey game's outcome:

- **K_GOALIE = 6**: Each goalie maintains their own Elo rating, updated with a dedicated K-factor after every start
- **Starter boost = 41.87**: When a confirmed starting goalie is known, the team receives a large Elo adjustment based on that goalie's individual rating
- **50% season regression**: Goalie ratings regress more aggressively than team ratings (33%) at season boundaries, reflecting year-to-year goaltender volatility
- **Player composite weighting**: Skaters 55% / goalies 45% in overall player score -- goalies are nearly half a team's value
- **Backfill tool**: `backfill_goalies.py` (NHL-unique) retroactively fills starting goalie data into the game CSV using NHL API boxscore endpoints

### Calibration

`platt.py` applies Platt scaling (logistic regression on logit of raw probability) to produce well-calibrated final outputs. Isotonic regression and beta calibration are also available. The Platt scaler is fitted during the `backtest` command and must be refitted after any parameter change.

---

## Key Files

### Core System
| File | Purpose |
|------|---------|
| `main.py` | CLI entry point; `dispatch()` routes all commands |
| `config.py` | Constants, 32 NHL teams, 4 divisions, settings I/O, `NHL_API_ABBR` mapping |
| `elo_model.py` | `NHLElo` class -- ratings, goalie sub-ratings (K_GOALIE=6), 20+ adjusters |
| `build_model.py` | Model construction with season regression and altitude bonus |
| `data_games.py` | Game data download via ESPN public API |
| `data_players.py` | Player stats download (skater + goalie stats) + team scoring |
| `backtest.py` | Walk-forward backtest, grid/genetic/bayesian optimizers |
| `platt.py` | Platt scaling, isotonic regression, beta calibration |
| `metrics.py` | Log loss, Brier, ECE, MCE, BSS, conformal prediction |
| `elo_set_handler.py` | Shared handler for `set param=value` commands |
| `single_param_opt.py` | Coordinate descent optimizer (one param at a time) |
| `cache_utils.py` | Season-aware smart caching for all API data |
| `color_helpers.py` | Terminal color output wrappers (`cok`, `cerr`, `cwarn`, `chi`, `cdim`, `cbold`) |

### NHL-Unique Files
| File | Purpose |
|------|---------|
| `backfill_goalies.py` | Backfill starting goalie data into game CSV using NHL API boxscores |

### Data Enrichment
| File | Purpose |
|------|---------|
| `odds_tracker.py` | Live odds via The Odds API (free tier, 500 req/month), CLV tracking |
| `weather.py` | Weather data via Open-Meteo API (less relevant for indoor arenas) |
| `kalshi.py` | Kalshi public API for live contract prices + auto-Kelly |
| `injuries.py` | ESPN injury report + goalie-weighted Elo impact scoring |

### Trading and Display
| File | Purpose |
|------|---------|
| `predict_ledger.py` | Contract ledger (add, sell, resolve, mark, invert positions) |
| `live_scores.py` | Live NHL scores via ESPN API with period display (P1, P2, P3, OT, SO) |
| `auto_resolve.py` | Auto-settle finished trades against live final scores |
| `html_generator.py` | Blogger-ready HTML prediction tables |
| `help_system.py` | CLI help text for all commands |
| `accuracy_test.py` | Standalone quick walk-forward accuracy test |

### Utility Scripts
| File | Purpose |
|------|---------|
| `run_optimize.py` | Standalone optimization runner |
| `accuracy_optimize.py` | Accuracy-focused optimization |
| `quick_optimizer.py` | Quick parameter sweep |
| `master_optimize.py` | Cross-sport master optimization runner (ELO phases 1-5, 10) |

---

## Optimization

### Objective Function

All optimizers use the same scoring objective:

```
score = -(LogLoss * 8 + Brier * 40)
```

This weighting is intentional. It penalizes overconfident wrong predictions (log loss) while also rewarding calibration (Brier). Higher is better (scores are negative; closest to zero wins).

### master_optimize.py

The `master_optimize.py` script runs the full optimization pipeline for the NHL system. It is also invoked by the cross-sport master optimizer that coordinates optimization across NBA, NFL, NHL, and MLB simultaneously.

The optimization is accuracy-first: the pipeline targets improvements in raw predictive accuracy while maintaining calibration quality.

### Optimization Pipeline

1. **Grid search** (`grid`) -- Coarse sweep over 7 Elo parameters with configurable ranges (NHL-specific: K range 4-20, HomeAdv range 10-50)
2. **Genetic optimizer** (`genetic`) -- `scipy.optimize.differential_evolution` for fine-tuning (NHL bounds: K 3-40)
3. **Bayesian optimizer** (`bayesian`) -- GP surrogate + Expected Improvement, ~50-100 evaluations
4. **Auto-optimize** (`autoopt`) -- Automated grid-to-genetic-to-bayesian pipeline
5. **Super-optimize** (`superopt`) -- Exhaustive 7-phase optimization (9 parameters, takes hours)
6. **Coordinate descent** (`singleopt`) -- One-param-at-a-time sweep with fine refinement

### Validation After Optimization

After optimizing, validate with this sequence:
1. `pbo` -- Probability of backtest overfitting (PBO < 0.3 is good)
2. `results` -- Check Deflated Sharpe Ratio (DSR > 1.96 means significant)
3. `purgedcv` -- Fold stability (accuracy std < 2% is good)
4. `cpcv` -- Combinatorial robustness (>90% paths above 65% accuracy)
5. `montecarlo` -- Statistical significance (p < 0.05)

---

## Kalshi Integration

The `kalshi.py` module connects to the Kalshi public API to fetch live prediction market contract prices for NHL games. Features:

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
| ESPN public API | Game scores and schedules (2 years), 0.5s between calls | 6 hours |
| ESPN public API | Player stats (skater + goalie) | 6 hours |
| ESPN JSON API | Injury reports (status, position, goalie-weighted impact) | 4 hours |
| NHL API | Boxscore data for goalie backfill | On-demand |
| Open-Meteo API | Weather data (less relevant for indoor arenas) | 2 hours |
| The Odds API (free tier) | Moneyline odds, 500 requests/month | Per-request |
| Kalshi public API | Prediction market contract prices | Live |
| ESPN live scoreboard | Live scores with period display (P1-P3, OT, SO) | 60 seconds |

Smart caching via `cache_utils.py` provides season-aware staleness checks. Different data types refresh at different rates. Files under 500 bytes are always treated as stale.

---

## Configuration

### Settings File: `nhl_elo_settings.json`

Contains all tunable Elo parameters. Key NHL-specific defaults:

| Parameter | Default | NHL Significance |
|-----------|---------|------------------|
| `k` | 5.16 | Lowest K of any sport -- 82-game season demands gradual changes |
| `home_advantage` | 41.43 | ~56% implied home win rate (higher than NFL) |
| `player_boost` | 36.36 | Significant player quality differential impact |
| `starter_boost` | 41.87 | Known starting goalie advantage -- the most impactful NHL adjuster |
| `rest_factor` | 39.03 | Major factor for travel-heavy NHL schedule |
| `b2b_penalty` | 18.0 | Significant back-to-back penalty (much higher than NBA) |
| `travel_factor` | 16.79 | Cross-country travel fatigue |
| `sos_factor` | 32.41 | Strong strength-of-schedule signal |
| `division_factor` | 29.25 | Strong divisional rivalry adjustment |
| `mean_reversion` | 37.79 | Regression after extreme results |
| `overtime_factor` | 5.0 | OT/shootout outcome adjustment |
| `altitude_factor` | 4.0 | Colorado (full), Utah (proportional at 4,226 ft) |
| `playoff_hca_factor` | 0.94 | Slightly reduced playoff home advantage |
| `form_weight` | 7.71 | Recent form from last 10 games |
| `season_regress` | 0.33 | 33% team regression; 50% goalie regression at season boundary |

### Other Generated Files

| File | Purpose |
|------|---------|
| `nhl_elo_ratings.json` | Current Elo ratings for all 32 teams |
| `nhl_platt_scaler.json` | Platt calibration coefficients |
| `nhl_backtest_predictions.csv` | Per-game backtest predictions |
| `nhl_calibration.csv` | 10-bin calibration table |
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
- pandas, numpy, scipy, colorama, tqdm, requests, matplotlib

**Optional (for advanced models):**
- torch (MLP, LSTM neural networks)
- hmmlearn (Hidden Markov Model)
- filterpy (Kalman Filter)
- lightgbm (LightGBM gradient boosting)
- catboost (CatBoost gradient boosting)
- networkx (PageRank network model)

---

## NHL-Specific Design Notes

- **Season spans two calendar years**: Oct--Jun. Season detection uses `year + 1 if month >= 10`.
- **82-game season**: Same as NBA, providing ample data for model training. The 10-game rolling window matches NBA's window size.
- **No ties**: NHL games always produce a winner through overtime or shootout. The `overtime_factor = 5.0` adjusts for OT outcomes.
- **Goalie is everything**: The per-goalie Elo sub-rating system is the NHL system's most distinctive feature. Unlike any other sport in the SharpStack family, individual goaltender performance is tracked with its own K-factor (K_GOALIE=6) and more aggressive season regression (50% vs 33% for teams).
- **Goalie injury tiers**: Star goaltender (20+ wins or .910+ SV%) = -35 Elo, starting goaltender (10+ wins) = -25 Elo, backup goaltender = -15 Elo. A star goalie injury is roughly 50% of a team's total value.
- **Two altitude teams**: Colorado Avalanche at 5,280 ft gets the full altitude bonus; Utah Hockey Club at 4,226 ft (relocated from Arizona in 2024) gets a proportional bonus scaled by `(elevation - 4000) / (5280 - 4000)`.
- **Back-to-backs are common and impactful**: NHL teams frequently play back-to-back games, and the `b2b_penalty = 18.0` is much higher than NBA's 0.0 (where rest_factor handles it instead). In hockey, the goaltender rarely plays both ends of a back-to-back, making starter identification even more critical.
- **Indoor sport**: Weather has minimal direct impact on NHL games since all arenas are indoor. The weather model is available but less relevant than for NFL outdoor stadiums.
- **Player stats split**: Skater stats (points, goals, assists) weighted at 55%, goalie stats (GAA, SVP, wins, shutouts) weighted at 45% in the composite player score.
- **Injury statuses**: Out, Day-to-Day, IR (Injured Reserve), LTIR (Long-Term Injured Reserve).
- **Backfill tool**: `backfill_goalies.py` is unique to the NHL system. It retroactively fills starting goalie data into the game CSV using NHL API boxscore endpoints, enabling the goalie sub-rating system to train on historical data.
- **Two abbreviation mappings**: `TEAM_ABBR` for internal use and `NHL_API_ABBR` for NHL API compatibility (e.g., internal `LA` maps to NHL API `LAK`).
- **4 divisions**: Atlantic, Metropolitan, Central, Pacific. Division rivalry factor of 29.25.
- **ESPN API rate limiting**: All ESPN calls use 0.5s sleep between sequential requests.
- **Fuzzy team lookup**: `NHLElo.find_team()` accepts full names, abbreviations, partial matches, and close matches. Typing `bruins`, `BOS`, or `Boston Bruins` all work.
- **Python 3.8+ compatible**: No walrus operators, no `match` statements.
- **All data files use `nhl_` prefix**: Settings, ratings, caches, and backtest outputs are all namespaced.
