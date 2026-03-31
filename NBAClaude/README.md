# NBA Moneyball v1.0 - Elo + XGBoost Prediction Engine

NBA game prediction system combining Elo ratings with an XGBoost ensemble, integrated with a Predicts/Kalshi $1 contract trading ledger. Built-in Kelly criterion position sizing, live score tracking, auto-settlement, and Blogger HTML publishing.

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Daily Workflow](#daily-workflow)
- [Prediction Flow](#prediction-flow)
- [Kelly Criterion Position Sizing](#kelly-criterion-position-sizing)
- [Bankroll Management](#bankroll-management)
- [Commands Reference](#commands-reference)
- [Model Architecture](#model-architecture)
- [XGBoost Ensemble](#xgboost-ensemble)
- [Calibration Methods](#calibration-methods)
- [Optimization Guide](#optimization-guide)
- [Validation Workflow](#validation-workflow)
- [Trading Ledger (Predicts)](#trading-ledger-predicts)
- [HTML Publishing](#html-publishing)
- [Data Sources](#data-sources)
- [File Structure](#file-structure)
- [Configuration & Settings](#configuration--settings)
- [Advanced Topics](#advanced-topics)
- [Requirements](#requirements)
- [Disclaimer](#disclaimer)

---

## Features

### Prediction Engine
- Elo rating system with margin-of-victory updates and 33% season regression
- XGBoost ensemble (80% Elo / 20% XGBoost) with 31 rolling features per game
- Platt calibration for well-calibrated probability outputs
- Beta calibration (3-parameter) for asymmetric miscalibration correction
- Isotonic regression calibration (Pool Adjacent Violators)

### 8 Elo Adjustment Layers
1. **Home court advantage** - configurable, reduced during playoffs
2. **Altitude bonus** - Denver Nuggets (5,280 ft), Utah Jazz, and other high-altitude arenas
3. **Player roster strength** - top 10 players, box score + advanced stats composite, z-scored
4. **Rest day adjustment** - back-to-back penalty, extra rest bonus (0-3+ days)
5. **Travel fatigue** - great-circle distance between arenas using lat/lon coordinates
6. **Pace mismatch** - slower team gets tempo control bonus
7. **Strength of schedule** - rolling opponent quality from recent games
8. **Injury penalties** - ESPN API integration, live Out/Doubtful player impact on Elo

### Kelly Criterion Position Sizing
- Automatic Kelly criterion calculation during every prediction
- Prompts for actual market odds in cents (prediction market contract price)
- Calculates edge, full Kelly fraction, and adjusted Kelly lots
- Quarter-Kelly and half-Kelly modes (default: half-Kelly)
- Uses live bankroll tracking for accurate position sizing
- Kelly-suggested contract count becomes the default when logging a trade

### 5 Optimization Methods
- Grid search (interactive or automatic, 7-9 parameters)
- Genetic algorithm (differential evolution via SciPy)
- Bayesian optimization (GP surrogate + Expected Improvement acquisition)
- Auto-optimize (`autoopt`): runs grid + genetic + bayesian automatically
- Super-optimize (`superopt`): exhaustive 7-phase optimization across all 9 parameters

### 16 Validation & Analysis Methods
- Purged walk-forward cross-validation (k-fold with embargo gap)
- Combinatorial purged CV (all C(k, k_test) paths)
- Probability of backtest overfitting (symmetric CV on trial population)
- Monte Carlo permutation test (null distribution, p-value)
- Rolling origin Platt recalibration (expanding window, OOS metrics)
- Kelly criterion position sizing backtest (fractional Kelly bankroll simulation)
- Sliding vs expanding window comparison
- Elo convergence / burn-in analysis
- Conformal prediction intervals (distribution-free coverage guarantees)
- Beta calibration analysis (asymmetric miscalibration detection)
- SHAP feature importance (XGBoost native, no extra dependencies)
- Time-decayed ensemble weighting (Elo 95% -> 70% over season)
- ECE / MCE (Expected / Maximum Calibration Error)
- Brier Skill Score (vs 50% baseline and home-win-rate baseline)
- Deflated Sharpe Ratio (multiple-testing bias adjustment)

### Trading & Publishing
- Full Predicts/Kalshi ledger for tracking $1 moneyline contracts with P&L
- Starting balance tracking with bankroll status display
- Live score tracking against open positions via NBA CDN
- Auto-resolve finished trades from live final scores
- Blogger HTML export for publishing daily predictions
- Monthly realized P&L bar chart generation
- Position inversion (flip direction without changing cost basis)

---

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

On first run, the system automatically:
1. Downloads game data from NBA API (last 2 seasons, ~2,400 games)
2. Downloads player stats (per-game + advanced metrics)
3. Fetches injury reports from ESPN
4. Builds the Elo model and replays all games to establish ratings
5. Runs a walk-forward backtest and fits the Platt calibration scaler
6. Prompts you to set your starting bankroll for Kelly criterion sizing
7. Enters the interactive CLI

No API keys required. All data sources are free public APIs.

---

## Daily Workflow

```
1. Launch: python main.py
   -> Auto-downloads fresh data (games, players, injuries)
   -> Builds model, runs baseline backtest, fits Platt scaler
   -> Auto-resolves finished trades (if enabled)
   -> Shows starting balance prompt (first run only)

2. Predict: type a team name (e.g. "Lakers", "celt", "GSW")
   -> Enter opponent team
   -> Specify home/away (a = first team home, b = second, n = neutral)
   -> See calibrated win probability, Elo ratings, injury impact, key players

3. Size: enter market odds in cents when prompted (e.g. "62" for $0.62)
   -> See Kelly criterion recommendation (edge, Kelly %, suggested lots)

4. Trade: type "y" to log the position
   -> Kelly-suggested contracts are the default quantity
   -> Enter actual price paid, optional notes

5. Monitor: "live" for real-time scores, "mark" to update market prices

6. Settle: "resolve" for manual settlement, "autoresolve" for automatic,
           "sell" for early exit at a price

7. Review: "predicts" for full P&L ledger, "chart" for monthly P&L chart,
           "balance" for bankroll status

8. Publish: "today" or "tomorrow" for Blogger HTML tables
```

---

## Prediction Flow

When you type a team name, the full prediction flow is:

```
> lakers
Opponent team: celtics
Home team? (a = first team home, b = second, n = neutral): a

   Los Angeles Lakers - 42.3% win probability (calibrated)
    Los Angeles Lakers Elo: 1523  |  Boston Celtics Elo: 1612
    Site: Los Angeles Lakers home
    Boston Celtics injuries: Kristaps Porzingis (-35 Elo)

  KEY PLAYERS (season averages):
    Los Angeles Lakers:
      LeBron James      25.1 PPG  7.2 RPG  7.8 APG
      Anthony Davis      24.8 PPG  12.1 RPG  3.2 APG
      ...
    Boston Celtics:
      Jayson Tatum       27.3 PPG  8.5 RPG  4.9 APG
      ...

Actual trade odds in cents (e.g. 62 for $0.62, or Enter to skip): 38
------------------------------------------------------------
  KELLY CRITERION SIZING
    Model prob  : 42.3%
    Market price: 38c (38.0% implied)
    Edge        : +4.3%
    Full Kelly  : 6.9%
    50% Kelly   : 3.5%
    Balance     : $100.00
    Suggested   : 9 contracts @ 38c = $3.42
------------------------------------------------------------

Log this moneyline pick as a Predicts position? (y/n): y
--- LOG MONEYLINE POSITION ---
Number of contracts (default 9 [Kelly]):
Price paid per contract (e.g. 0.62): 0.38
Notes (optional): Lakers at home, LeBron healthy

  Logged lot #15
  9x Los Angeles Lakers @ $0.38
  Entry fee: $0.1800   Total cost: $3.6000
```

---

## Kelly Criterion Position Sizing

The system uses the Kelly criterion formula optimized for prediction market $1 contracts:

```
Edge         = Model Probability - Market Price
Full Kelly % = Edge / (1 - Market Price)
Adjusted     = Full Kelly % x Kelly Fraction (quarter or half)
Wager ($)    = Adjusted % x Current Bankroll
Contracts    = floor(Wager / Market Price)
```

### How It Works

1. After the model shows its win probability, you're prompted for the actual market price in cents
2. The system calculates your edge (model prob vs market implied prob)
3. It applies the Kelly fraction (default half-Kelly) to determine optimal position size
4. The suggested number of contracts becomes the default when logging the trade

### Kelly Fraction Options

| Setting | Command | Risk Level | Description |
|---------|---------|------------|-------------|
| Quarter Kelly | `set kelly=quarter` | Conservative | 25% of full Kelly - lower variance, slower growth |
| Half Kelly | `set kelly=half` | **Default** | 50% of full Kelly - good balance of growth vs risk |

Half-Kelly is the default because full Kelly is theoretically optimal but assumes perfect probability estimates. Half-Kelly provides ~75% of the growth rate with significantly less variance and drawdown risk.

### Example Calculation

```
Model says: Lakers 65% to win
Market price: 58 cents ($0.58 per contract)
Your bankroll: $200

Edge         = 0.65 - 0.58 = 0.07 (7%)
Full Kelly   = 0.07 / (1 - 0.58) = 16.7%
Half Kelly   = 16.7% x 0.50 = 8.3%
Wager        = $200 x 0.083 = $16.67
Contracts    = floor($16.67 / $0.58) = 28 contracts
Total cost   = 28 x $0.58 = $16.24
```

If the model probability is LESS than the market price, the edge is negative and Kelly suggests 0 contracts (no bet).

---

## Bankroll Management

### Setting Your Balance

On first startup, you'll be prompted to set your starting bankroll:

```
  SET STARTING BALANCE
  Enter your starting bankroll for Kelly criterion sizing.
  Starting balance ($): 100
  Starting balance set to $100.00
```

### Viewing Bankroll Status

```
> balance
--- BANKROLL STATUS ---
  Starting balance : $100.00
  Current balance  : $112.35
  P&L              : +$12.35 (+12.4%)
  Kelly fraction   : 50%
```

The current balance is calculated as: `Starting Balance - Total Entry Costs + Total Realized Cash`

### Updating Balance

Run `balance` at any time to update your starting balance. You can also adjust the Kelly fraction:

```
> set kelly=quarter    # Conservative: 25% Kelly
> set kelly=half       # Default: 50% Kelly
```

---

## Commands Reference

### Predictions & Data

| Command | Description |
|---------|-------------|
| `<team name>` | Predict a matchup (fuzzy match: `Lakers`, `celt`, `GSW`, `thunder`) |
| `all` | Show all team Elo ratings, ranked |
| `players` | Show top scorers (league-wide or per team) |
| `injuries` | Show NBA injury report with Elo impact estimates |
| `injuries set <team> <p1>,<p2>` | Manually mark players as OUT |
| `refresh` | Re-download all data (games, players, injuries), rebuild model |
| `settings` | Display all current model parameters and Platt status |
| `today` / `html` / `blogger` | Generate Blogger HTML for today's games |
| `tomorrow` | Generate Blogger HTML for tomorrow's games |

### Backtesting

| Command | Description |
|---------|-------------|
| `backtest` | Walk-forward backtest, fit Platt scaler, report ECE/MCE/BSS |
| `enhanced` | Train XGBoost ensemble (80/20 blend) + SHAP feature importance |
| `enhanced decay` | Time-decayed ensemble (Elo 95% -> 70% over season) |

### Optimization

| Command | Description |
|---------|-------------|
| `grid` | Interactive grid search (7 params, customizable ranges/steps) |
| `genetic` | Genetic algorithm with interactive bounds (7 params) |
| `bayesian` | Gaussian Process + Expected Improvement (7 params) |
| `autoopt` | Automatic grid + genetic + bayesian pipeline (~15-30 min) |
| `superopt` | Exhaustive 7-phase optimization, all 9 params (2-4 hours) |
| `results` | Show best parameters from all optimizer runs + DSR significance |

### Validation & Analysis

| Command | Description |
|---------|-------------|
| `purgedcv` | Purged walk-forward CV (k-fold with embargo gap) |
| `cpcv` | Combinatorial purged CV (all C(k, k_test) paths) |
| `pbo` | Probability of backtest overfitting |
| `montecarlo` | Monte Carlo permutation test (p-value, ~8 min) |
| `rollingcal` | Rolling origin Platt recalibration (OOS metrics) |
| `kelly` | Kelly criterion position sizing backtest |
| `sliding` | Sliding vs expanding window comparison |
| `convergence` | Elo rating convergence / burn-in analysis |
| `conformal` | Conformal prediction intervals (coverage guarantees) |
| `betacal` | Beta calibration (3-param, asymmetric miscalibration) |
| `shap` | SHAP feature importance for XGBoost ensemble |

### Trading (Predicts)

| Command | Description |
|---------|-------------|
| `predicts` / `summary` | Show full contract ledger with P&L |
| `balance` | View bankroll status (starting, current, P&L, ROI) |
| `resolve` | Settle finished contracts (win/loss prompt) |
| `sell` | Exit a position early at a price (partial or full) |
| `mark` | Update current market marks on open lots |
| `chart` | Generate monthly realized P&L bar chart |
| `live` | Live scores for open positions |
| `invert` | Flip an open trade's direction |
| `autoresolve` | Auto-resolve finished trades from live scores |
| `autoresolve on/off` | Toggle auto-resolve on startup |

### Mega-Ensemble (32 Models)

| Command | Description |
|---------|-------------|
| `mega` | Run full mega-ensemble backtest (26+ models) |
| `mega optimize` | Exhaustive 5-phase optimization (finds best settings) |
| `mega quick` | Quick grid search (Phase 1 only) |
| `mega ablation` | Test each model's contribution, auto-prune bad ones |
| `mega models` | Show all 32 models with ON/OFF status |
| `mega on <model>` | Enable a model (e.g., `mega on lstm`) |
| `mega off <model>` | Disable a model (e.g., `mega off monte_carlo`) |
| `mega on all` | Enable all models |
| `mega settings` | Show all mega-ensemble parameter values |
| `mega set adj=0.10` | Set mega parameter (see root README for full list) |

### Settings (39 Elo Parameters)

| Command | Description |
|---------|-------------|
| `set` | Show all 39 parameters with current values |
| `set k=4` | Elo K-factor (rating change per game) |
| `set home=24` | Home field advantage (Elo points) |
| `set boost=20` | Player roster strength weight |
| `set starter=30` | Starting pitcher quality |
| `set rest=10` | Rest day factor |
| `set travel=8` | Travel fatigue factor |
| `set b2b=75` | Back-to-back penalty |
| `set pace=19` | Pace mismatch factor |
| `set sos=2` | Strength of schedule weight |
| `set streak=0.5` | Win streak momentum |
| `set reversion=0` | Mean reversion after extremes |
| `set playoff=0.7` | Playoff home field multiplier |
| `set kelly=quarter` | Use quarter-Kelly sizing (25%) |
| `set balance=1000` | Starting bankroll |
| `set autoresolve=true` | Auto-settle finished trades |
| `help [command]` | Help overview or detailed command help |
| `quit` | Save and exit |

---

## Model Architecture

### Prediction Pipeline

```
Team A vs Team B (with home/away, date)
    |
    v
[Elo Ratings] --- base team strength from ~2,400 game history
    + Home court advantage (reduced in April+ playoffs)
    + Altitude bonus (Denver 80 Elo, Utah 40 Elo)
    + Player roster strength (z-scored composite of top 10)
    + Rest day adjustment (back-to-back penalty / extra rest bonus)
    + Travel fatigue (great-circle distance between arenas)
    + Pace mismatch (slower team gets tempo control bonus)
    + Strength of schedule (rolling opponent quality)
    + Injury penalties (ESPN API, Out/Doubtful players)
    |
    v
[Raw Elo Probability] --- logistic function on adjusted rating difference
    |
    v
[XGBoost Ensemble] --- 80% Elo + 20% XGBoost (31 rolling features)
    |
    v
[Platt Calibration] --- logistic regression on logit(raw probability)
    |
    v
Final calibrated win probability
    |
    v
[Kelly Criterion] --- optimal position sizing vs market odds
```

### Elo System Details

- **Base rating**: 1500 for all teams
- **K-factor**: Configurable (default 6.0), determines how quickly ratings change
- **Margin of victory**: Logarithmic MOV adjustment, capped to prevent outlier distortion
- **Season regression**: 33% pull toward league mean at season boundaries (Oct/Nov)
- **Win probability**: `P(A wins) = 1 / (1 + 10^((RatingB - RatingA) / 400))`

### Tunable Parameters (9)

| Parameter | Default | CLI | Description |
|-----------|---------|-----|-------------|
| K-factor | 6.0 | `set k=` | Rating volatility per game |
| Home advantage | 22.0 | `set home=` | Elo points for home team |
| Player boost | 25.0 | `set boost=` | Roster strength weight |
| Rest factor | 12.0 | `set rest=` | Elo per rest day deviation |
| Travel factor | 32.0 | `set travel=` | Distance-based travel penalty |
| Pace factor | 39.0 | `set pace=` | Tempo mismatch weight |
| Playoff HCA | 0.56 | `set playoff=` | Home advantage multiplier in playoffs |
| SOS factor | 0.0 | `set sos=` | Strength of schedule weight |
| Form weight | 0.0 | `set form=` | Recent form (last 10 games) adjustment |

---

## XGBoost Ensemble

The enhanced model combines Elo with gradient-boosted trees using 31 rolling features:

### Features (20 per game)

| Feature | Description |
|---------|-------------|
| `elo_prob` | Raw Elo win probability |
| `elo_diff` | Elo rating differential |
| `player_diff` | Player composite score differential |
| `h_ppg` / `a_ppg` | Home/away rolling PPG (10-game window) |
| `h_papg` / `a_papg` | Rolling points allowed per game |
| `h_win_pct` / `a_win_pct` | Rolling win percentage |
| `h_margin` / `a_margin` | Rolling point margin |
| `ppg_diff` / `papg_diff` | PPG and PAPG differentials |
| `win_pct_diff` / `margin_diff` | Win% and margin differentials |
| `off_diff` / `def_diff` | Offensive/defensive differentials |
| `h_rest` / `a_rest` / `rest_diff` | Rest days and differential |

### XGBoost Parameters

```
max_depth=5, eta=0.03, subsample=0.9,
colsample_bytree=0.8, min_child_weight=3, 300 rounds
```

### Walk-Forward Training

- First 200 games: Elo-only predictions while accumulating features
- After 200 games: XGBoost trained and blended at 80/20 Elo/XGBoost
- Retrained every 50 games with expanding training window
- Optional time-decay: Elo weight transitions 95% -> 70% over the season

---

## Calibration Methods

### Platt Scaling (Default)
Logistic regression on `logit(raw_prob)` via `scipy.optimize.minimize` (L-BFGS-B). Corrects systematic overconfidence at 0.9+ and underconfidence at 0.1-0.2. Fitted during `backtest` command.

### Isotonic Regression
Pool Adjacent Violators algorithm on 50 bins with linear interpolation. Non-parametric, can capture any monotone miscalibration pattern. Fitted during `enhanced` backtest.

### Beta Calibration
3-parameter model: `logit(p_cal) = c + a*log(p) - b*log(1-p)`. If `a != b`, the miscalibration is asymmetric (e.g., overconfident on favorites but accurate on underdogs). Platt scaling cannot detect this.

### Season Regression
All team ratings are pulled 33% toward the league mean at season boundaries. Prevents runaway ratings from carrying over to a new roster year.

---

## Optimization Guide

### Quick Optimization (15-30 minutes)
```
> autoopt
```
Runs coarse grid search (768 combos) -> genetic algorithm (50 gen x 25 pop) -> Bayesian GP (15+40 iter). Compares all three, applies the best, refits Platt.

### Exhaustive Optimization (2-4 hours)
```
> superopt
```
Seven phases:
1. Broad grid search (~6,000+ combos, 7 params)
2. Genetic round 1 (wide bounds, 100 gen x 50 pop)
3. Bayesian round 1 (wide bounds, 30 initial + 80 iter)
4. Genetic round 2 (tightened bounds, 80 gen x 40 pop)
5. Bayesian round 2 (tightened, 20 initial + 60 iter)
6. Fine grid (tiny steps around absolute best)
7. Validation (purged CV + PBO + Monte Carlo)

### Manual Optimization
```
> grid       # Set your own ranges and step sizes for each param
> genetic    # Set your own bounds and population/generations
> bayesian   # Set your own bounds and iterations
> results    # Compare all optimizer results + DSR significance
```

### Optimizer Objective
```
Score = -(LogLoss * 8 + Brier * 40)
```
This weighting is calibration-focused, penalizing poor probability estimates more than raw accuracy.

---

## Validation Workflow

Run in order after optimization for thorough model validation:

| Step | Command | What to Check | Red Flag |
|------|---------|---------------|----------|
| 1 | `backtest` | Accuracy, log loss, Brier, ECE, MCE, BSS | Accuracy < 63% |
| 2 | `convergence` | Games before Elo stabilizes | Burn-in > 200 games |
| 3 | `sliding` | Does old data help or hurt | Sliding wins = regression too mild |
| 4 | `purgedcv` | Accuracy std across folds | Std > 3% = fragile |
| 5 | `cpcv` | % of paths above 65% accuracy | Any path < 55% |
| 6 | `pbo` | Probability of overfitting | PBO > 0.5 = overfit |
| 7 | `montecarlo` | Statistical significance | p > 0.05 = no skill |
| 8 | `enhanced` | SHAP: XGBoost adding signal? | elo_prob > 50% SHAP |
| 9 | `rollingcal` | Is single-pass Platt honest? | Rolling much worse |
| 10 | `betacal` | Asymmetric miscalibration? | a != b by > 0.3 |
| 11 | `conformal` | Prediction set coverage | High "both" % = uncertain |
| 12 | `kelly` | Sharpe ratio, max drawdown | Sharpe < 0.5, DD > 50% |

### Decision Thresholds

| Metric | Good | Marginal | Bad |
|--------|------|----------|-----|
| Accuracy | > 66% | 63-66% | < 63% |
| ECE | < 0.03 | 0.03-0.08 | > 0.08 |
| BSS vs 50% | > 0.08 | 0.04-0.08 | < 0.04 |
| PBO | < 0.3 | 0.3-0.5 | > 0.5 |
| DSR | > 1.96 | 1.0-1.96 | < 1.0 |
| Monte Carlo p | < 0.01 | 0.01-0.05 | > 0.05 |
| Kelly Sharpe | > 1.0 | 0.5-1.0 | < 0.5 |

---

## Trading Ledger (Predicts)

### How Predicts/Kalshi Contracts Work

Predicts and Kalshi contracts are $1 binary options:
- Buy a contract for $0.01-$0.99 (the "price in cents")
- If your prediction is correct: contract pays out $1.00
- If wrong: contract expires worthless
- The price reflects implied probability (62 cents = 62% implied)
- 2% fee on potential payout ($1 per contract) charged at entry and exit

### Position Lifecycle

```
1. ENTRY:   Buy contracts at a price (e.g., 10x @ $0.55)
            Entry fee: 2% x $1 x contracts = $0.20
            Total cost: 10 x $0.55 + $0.20 = $5.70

2. MONITOR: "mark" updates current market price
            "live" shows real-time game scores
            "predicts" shows unrealized P&L based on marks

3. EXIT (choose one):
   a. RESOLVE: Game ends, contract settles at $1.00 (win) or $0.00 (loss)
      Win:  10 x $1.00 = $10.00 received, profit = $10.00 - $5.70 = $4.30
      Loss: $0.00 received, loss = -$5.70

   b. SELL: Exit early at current market price
      Sell 10x @ $0.72 -> 10 x $0.72 = $7.20 gross
      Exit fee: 2% x $1 x 10 = $0.20
      Net proceeds: $7.00, profit = $7.00 - $5.70 = $1.30

   c. INVERT: Flip direction (swap predicted winner) without changing cost basis
```

### Ledger Columns

The CSV ledger (`predicts_lots.csv`) tracks 19 fields per position:
- Identity: lot_id, opened_at, date, market
- Teams: home_team, away_team, predicted_winner, model_prob
- Position: contracts_open, contracts_original, avg_entry_price
- Costs: entry_fee_total, entry_cost_total
- P&L: realized_cash, realized_exit_fees, realized_profit
- Status: status (pending/partial/sold/settled), last_mark_price, notes

### P&L Summary

```
> predicts

  PREDICTS MONEYLINE LEDGER
  lot_id  date        market              predicted_winner  ...  status
  1       2026-01-15  Celtics @ Lakers    Los Angeles Lakers  ...  settled
  2       2026-01-16  Suns @ Nuggets      Denver Nuggets      ...  settled
  ...

  POSITIONS: 12 | CLOSED WIN RATE: 66.7%
  ENTRY FEES: $0.4800 | EXIT FEES: $0.1200
  REALIZED P&L: +$3.2400 | UNREALIZED: +$0.8500 | MARKED P&L: +$4.0900
  ROI REALIZED: +27.0% | ROI MARKED: +34.1%
```

---

## HTML Publishing

Generate prediction tables for blog publishing:

```
> today     # Generate HTML for today's scheduled games
> tomorrow  # Generate HTML for tomorrow's games
```

Outputs:
- `today_nba_predictions.html` - styled HTML table, copy-paste ready for Blogger
- Plain-text summary printed to terminal

The HTML includes: matchup, win probabilities, Elo ratings, injury notes, calibration status, and a dark-themed design.

---

## Data Sources

| Data | Source | Cache Duration | API Key |
|------|--------|----------------|---------|
| Game results (2 seasons) | NBA API (`leaguegamelog`) | 6 hours | None |
| Player stats (per-game + advanced) | NBA API | 6 hours | None |
| Injury reports | ESPN public JSON API | 4 hours | None |
| Live scores | NBA CDN (`todaysScoreboard`) | 60 seconds | None |
| Game schedule | NBA CDN (`scheduleLeagueV2`) | Per request | None |

All data is fetched automatically on startup and cached locally. Files under 500 bytes are always considered stale and re-downloaded.

---

## File Structure

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point, command dispatcher, prediction flow |
| `elo_model.py` | NBAElo class - ratings, 8 adjustments, predictions |
| `enhanced_model.py` | XGBoost ensemble, TeamTracker, SHAP analysis |
| `backtest.py` | Walk-forward backtest, 5 optimizers, 12 validation methods |
| `build_model.py` | Model construction, season regression, rating initialization |
| `platt.py` | Platt, isotonic, and beta calibration |
| `metrics.py` | Log loss, Brier, ECE, MCE, BSS, conformal sets |
| `config.py` | Settings I/O, team abbreviations, cache management |
| `data_games.py` | NBA API game data download |
| `data_players.py` | NBA API player stats, team scoring composite |
| `injuries.py` | ESPN injury API, Elo impact calculation |
| `live_scores.py` | NBA CDN live scores and schedule |
| `predict_ledger.py` | Trading ledger, Kelly sizing, bankroll management |
| `auto_resolve.py` | Automatic settlement of finished trades |
| `html_generator.py` | Blogger HTML and plain-text table generation |
| `color_helpers.py` | Terminal color formatting wrappers |
| `help_system.py` | In-app CLI help documentation |
| `quick_optimizer.py` | 9-parameter differential evolution optimizer for Elo tuning |
| `accuracy_test.py` | Standalone walk-forward accuracy test script |

### Generated Data Files (gitignored)

| File | Contents |
|------|----------|
| `nba_recent_games.csv` | ~2,400 games (2 seasons) |
| `nba_player_stats.csv` | Per-game player statistics |
| `nba_advanced_stats.csv` | Advanced player metrics |
| `nba_elo_ratings.json` | Current team Elo ratings |
| `nba_elo_settings.json` | Tuned parameters + balance + kelly fraction |
| `nba_platt_scaler.json` | Platt calibration coefficients |
| `nba_isotonic_scaler.json` | Isotonic calibration bins |
| `nba_enhanced_model.json` | XGBoost metadata |
| `nba_xgb_model.json` | XGBoost model weights |
| `nba_injuries.json` | Cached ESPN injury data |
| `predicts_lots.csv` | Trading ledger (all positions + P&L) |
| `nba_backtest_predictions.csv` | Per-game backtest predictions |
| `nba_calibration.csv` | 10-bin calibration table |
| `nba_grid_search.csv` | Grid search results |
| `nba_genetic_results.csv` | Genetic optimizer results |
| `nba_bayesian_results.csv` | Bayesian optimizer results |
| `predicts_pnl.png` | Monthly P&L bar chart |

---

## Configuration & Settings

All settings are stored in `nba_elo_settings.json` and persist between sessions:

```json
{
  "base_rating": 1500.0,
  "k": 6.0,
  "home_adv": 22.0,
  "use_mov": true,
  "player_boost": 25.0,
  "rest_factor": 12.0,
  "travel_factor": 32.0,
  "sos_factor": 0.0,
  "pace_factor": 39.0,
  "playoff_hca_factor": 0.56,
  "form_weight": 0.0,
  "autoresolve_enabled": false,
  "starting_balance": 100.0,
  "kelly_fraction": 0.50
}
```

Modify at runtime with `set <param>=<value>`. After changing model parameters, always rerun `backtest` to refit the Platt scaler.

---

## Advanced Topics

### Player Scoring Composite

Each team's player strength is calculated from the top 10 players' box score and advanced stats:
- Per-game stats: PPG, RPG, APG, SPG, BPG, FG%, 3P%, FT%, minutes
- Advanced: PER, TS%, usage rate, win shares
- Combined into a single composite, z-scored across the league
- Applied as an Elo adjustment proportional to `player_boost` setting

### Injury Impact Estimation

When a player is marked Out/Doubtful (from ESPN or manually):
- The system estimates their contribution to the team's Elo rating
- Impact is weighted by the player's usage rate and minutes share
- The Elo adjustment is subtracted from the team's effective rating for that prediction
- Visible in prediction output: `Boston Celtics injuries: Porzingis (-35 Elo)`

### Travel Distance Calculation

Travel fatigue uses the Haversine formula (great-circle distance) between arena coordinates:
- All 30 NBA arenas have hardcoded (latitude, longitude) coordinates
- Distance is converted to a penalty proportional to `travel_factor`
- Cross-country trips (e.g., LAL -> BOS) incur larger penalties than division games

### Conformal Prediction Sets

Distribution-free prediction intervals at configurable coverage levels:
- At 90% coverage: if the prediction set contains only one team, the model is confident
- If both teams are in the set: the model is uncertain about this game
- Empty set: model is overconfident (rare, indicates calibration issues)
- High singleton percentage with good coverage = well-calibrated confident model

---

## Requirements

- Python 3.8+
- Windows, macOS, or Linux

```bash
pip install -r requirements.txt
```

Dependencies: `pandas`, `numpy`, `scipy`, `xgboost`, `requests`, `nba_api`, `colorama`, `matplotlib`

---

## Trade on Kalshi

If you want to trade prediction market contracts, you can sign up for Kalshi using this referral link:

**[Sign up for Kalshi](https://kalshi.com/sign-up/?referral=e3da1362-7a60-453c-a63e-60f402ec22ab)**

Kalshi is a regulated prediction market exchange where you can trade $1 binary contracts on sports, politics, economics, and more.

---

## Disclaimer

This software is for entertainment, educational, and analytical purposes only. It is not financial advice. Past model performance does not guarantee future results. Prediction market trading involves risk of loss. Always trade responsibly and only with money you can afford to lose.
