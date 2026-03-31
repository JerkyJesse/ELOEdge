# MLB Moneyball v1.0 - Elo + XGBoost Prediction Engine

MLB game prediction system combining Elo ratings with an XGBoost ensemble, integrated with a Predicts/Kalshi $1 contract trading ledger. Built-in Kelly criterion position sizing, live score tracking, auto-settlement, and Blogger HTML publishing. Tuned for baseball's 162-game season with pitcher-weighted player scoring.

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
- [MLB-Specific Design Choices](#mlb-specific-design-choices)
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

### 9 Elo Adjustment Layers
1. **Home field advantage** - configurable (default 24 Elo, ~54% implied), reduced in October playoffs
2. **Altitude bonus** - Colorado Rockies (Coors Field, 5,280 ft) get altitude advantage
3. **Player roster strength** - 55% batting / 45% pitching composite, z-scored per team
4. **Starting pitcher quality** - per-pitcher cumulative Elo ratings (700+ pitchers tracked), updated after each start with margin-of-victory adjustment, 50% season regression
5. **Rest day adjustment** - off-day bonus, travel day penalty
6. **Travel fatigue** - great-circle distance between ballparks using lat/lon coordinates
7. **Pace / run environment** - run-scoring environment mismatch adjustment
8. **Strength of schedule** - rolling opponent quality from recent games
9. **Injury penalties** - ESPN API integration, IL designations (60-Day IL, 15-Day IL, Out, Doubtful)

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
- Live score tracking against open positions via MLB Stats API
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
1. Downloads game data from MLB Stats API (last 2 seasons, ~4,800 games)
2. Downloads batting leaders (top 200) and pitching leaders (top 150)
3. Fetches injury reports from ESPN (includes IL designations)
4. Builds the Elo model and replays all games to establish ratings
5. Runs a walk-forward backtest and fits the Platt calibration scaler
6. Prompts you to set your starting bankroll for Kelly criterion sizing
7. Enters the interactive CLI

No API keys required. All data sources are free public APIs.

---

## Daily Workflow

```
1. Launch: python main.py
   -> Auto-downloads fresh data (games, batting, pitching, injuries)
   -> Builds model, runs baseline backtest, fits Platt scaler
   -> Auto-resolves finished trades (if enabled)
   -> Shows starting balance prompt (first run only)

2. Predict: type a team name (e.g. "Yankees", "dodgers", "STL")
   -> Enter opponent team
   -> Specify home/away (a = first team home, b = second, n = neutral)
   -> See calibrated win probability, Elo ratings, injury impact, key players

3. Size: enter market odds in cents when prompted (e.g. "55" for $0.55)
   -> See Kelly criterion recommendation (edge, Kelly %, suggested lots)

4. Trade: type "y" to log the position
   -> Kelly-suggested contracts are the default quantity
   -> Enter actual price paid, optional notes

5. Monitor: "live" for real-time scores (innings), "mark" to update market prices

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
> yankees
Opponent team: red sox
Home team? (a = first team home, b = second, n = neutral): a

   New York Yankees - 58.2% win probability (calibrated)
    New York Yankees Elo: 1537  |  Boston Red Sox Elo: 1498
    Site: New York Yankees home
    Boston Red Sox injuries: Chris Sale (15-Day IL, -22 Elo)

  KEY PLAYERS (season stats):
    New York Yankees:
      Aaron Judge        .301 AVG  38 HR  89 RBI
      Juan Soto          .288 AVG  31 HR  78 RBI
      ...
    Boston Red Sox:
      Rafael Devers      .279 AVG  27 HR  82 RBI
      ...

Actual trade odds in cents (e.g. 62 for $0.62, or Enter to skip): 52
------------------------------------------------------------
  KELLY CRITERION SIZING
    Model prob  : 58.2%
    Market price: 52c (52.0% implied)
    Edge        : +6.2%
    Full Kelly  : 12.9%
    50% Kelly   : 6.5%
    Balance     : $200.00
    Suggested   : 25 contracts @ 52c = $13.00
------------------------------------------------------------

Log this moneyline pick as a Predicts position? (y/n): y
--- LOG MONEYLINE POSITION ---
Number of contracts (default 25 [Kelly]):
Price paid per contract (e.g. 0.62): 0.52
Notes (optional): Yankees at home, Judge hot streak

  Logged lot #8
  25x New York Yankees @ $0.52
  Entry fee: $0.5000   Total cost: $13.5000
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

Half-Kelly is the default because full Kelly is theoretically optimal but assumes perfect probability estimates. Half-Kelly provides ~75% of the growth rate with significantly less variance and drawdown risk. This is especially important in MLB where the inherent randomness of baseball means even the best models have more variance than NBA.

### Example Calculation

```
Model says: Dodgers 62% to win
Market price: 57 cents ($0.57 per contract)
Your bankroll: $500

Edge         = 0.62 - 0.57 = 0.05 (5%)
Full Kelly   = 0.05 / (1 - 0.57) = 11.6%
Half Kelly   = 11.6% x 0.50 = 5.8%
Wager        = $500 x 0.058 = $29.07
Contracts    = floor($29.07 / $0.57) = 50 contracts
Total cost   = 50 x $0.57 = $28.50
```

If the model probability is LESS than the market price, the edge is negative and Kelly suggests 0 contracts (no bet).

---

## Bankroll Management

### Setting Your Balance

On first startup, you'll be prompted to set your starting bankroll:

```
  SET STARTING BALANCE
  Enter your starting bankroll for Kelly criterion sizing.
  Starting balance ($): 200
  Starting balance set to $200.00
```

### Viewing Bankroll Status

```
> balance
--- BANKROLL STATUS ---
  Starting balance : $200.00
  Current balance  : $231.50
  P&L              : +$31.50 (+15.8%)
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
| `<team name>` | Predict a matchup (fuzzy match: `Yankees`, `dodgers`, `STL`, `cubs`) |
| `all` | Show all team Elo ratings, ranked |
| `players` | Show batting/pitching leaders |
| `injuries` | Show MLB injury report with Elo impact + IL designations |
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
| `live` | Live scores for open positions (with innings) |
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
[Elo Ratings] --- base team strength from ~4,800 game history (2 seasons)
    + Home field advantage (reduced in October playoffs)
    + Altitude bonus (Colorado Rockies / Coors Field only)
    + Player roster strength (55% batting + 45% pitching composite)
    + Rest day adjustment (off-day bonus / travel day penalty)
    + Travel fatigue (great-circle distance between ballparks)
    + Run environment mismatch (pace factor)
    + Strength of schedule (rolling opponent quality)
    + Injury penalties (ESPN API, IL designations)
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

- **Base rating**: 1500 for all 30 MLB teams
- **K-factor**: Default 4.0 (lower than NBA's ~8 because 162 games provides more signal per team)
- **Margin of victory**: Logarithmic MOV adjustment, capped to prevent blowout distortion
- **Season regression**: 33% pull toward league mean at calendar year boundaries
- **Win probability**: `P(A wins) = 1 / (1 + 10^((RatingB - RatingA) / 400))`

---

## MLB-Specific Design Choices

This system is specifically tuned for baseball, not a port of an NBA model:

| Design Choice | MLB Value | NBA Value | Rationale |
|---------------|-----------|-----------|-----------|
| K-factor | 4.0 | 8.38 | 162-game season = more data per team, less rating volatility needed |
| Home advantage | 24 Elo | 28 Elo | MLB ~54% home win rate vs NBA ~60% |
| Altitude | Rockies only | DEN + UTA | Only Coors Field (5,280 ft) has meaningful baseball impact |
| Season boundary | Calendar year | Oct cross-year | MLB runs April-October within one year |
| Playoff detection | October | April+ | October games get reduced HCA (factor 0.70) |
| Player scoring | 55% bat / 45% pitch | Box + advanced | Baseball is pitcher-dependent; separate scoring |
| Batting composite | `HR*2 + RBI + R*0.5 + SB*0.5 + AVG*100` | PPG/RPG/APG | Emphasizes power + run production |
| Pitching composite | `(4.50-ERA)*10 + K*0.5 + W*3` | N/A | 4.50 = league avg ERA baseline |
| IL statuses | Out, Doubtful, 15-Day IL, 60-Day IL | Out, Doubtful | Baseball has specific IL designations |

### Player Scoring Formula

**Batting (55% weight):**
```
HR * 2 + RBI + Runs * 0.5 + SB * 0.5 + AVG * 100
```

**Pitching (45% weight):**
```
(4.50 - ERA) * 10 + Strikeouts * 0.5 + Wins * 3
```
Where 4.50 is the league-average ERA baseline. Pitchers with ERA below 4.50 contribute positively; above 4.50 contributes negatively.

Both components are z-scored across the league and combined into a single team composite used for the `player_boost` Elo adjustment.

---

## XGBoost Ensemble

The enhanced model combines Elo with gradient-boosted trees using 31 rolling features:

### Features (20 per game)

| Feature | Description |
|---------|-------------|
| `elo_prob` | Raw Elo win probability |
| `elo_diff` | Elo rating differential |
| `player_diff` | Player composite score differential |
| `h_ppg` / `a_ppg` | Home/away rolling runs per game (10-game window) |
| `h_papg` / `a_papg` | Rolling runs allowed per game |
| `h_win_pct` / `a_win_pct` | Rolling win percentage |
| `h_margin` / `a_margin` | Rolling run margin |
| `ppg_diff` / `papg_diff` | Run scoring and allowing differentials |
| `win_pct_diff` / `margin_diff` | Win% and margin differentials |
| `off_diff` / `def_diff` | Offensive/defensive differentials |
| `h_rest` / `a_rest` / `rest_diff` | Rest days and differential |

### XGBoost Parameters

```
max_depth=5, eta=0.03, subsample=0.9,
colsample_bytree=0.8, min_child_weight=3, 300 rounds
```

### Walk-Forward Training

- First 200 games: Elo-only predictions while accumulating training features
- After 200 games: XGBoost trained and blended at 80/20 Elo/XGBoost
- Retrained every 50 games with expanding training window
- Optional time-decay: Elo weight transitions 95% -> 70% over the season

---

## Calibration Methods

### Platt Scaling (Default)
Logistic regression on `logit(raw_prob)` via `scipy.optimize.minimize` (L-BFGS-B). Corrects systematic over/underconfidence. Fitted during `backtest` command.

### Isotonic Regression
Pool Adjacent Violators algorithm on 50 bins with linear interpolation. Non-parametric. Fitted during `enhanced` backtest.

### Beta Calibration
3-parameter model: `logit(p_cal) = c + a*log(p) - b*log(1-p)`. If `a != b`, miscalibration is asymmetric. Particularly useful in MLB where favorites and underdogs may be miscalibrated differently.

### Season Regression
All team ratings pulled 33% toward league mean at year boundaries. Prevents carry-over of inflated/deflated ratings to new rosters.

---

## Optimization Guide

### Quick Optimization (15-30 minutes)
```
> autoopt
```
Runs coarse grid search -> genetic algorithm -> Bayesian GP. Compares all three, applies the best, refits Platt.

### Exhaustive Optimization (2-4 hours)
```
> superopt
```
Seven phases across all 9 parameters, ending with validation (purged CV + PBO + Monte Carlo).

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
Calibration-focused weighting that penalizes poor probability estimates more than raw accuracy.

### Tunable Parameters (10)

| Parameter | Default | CLI | Description |
|-----------|---------|-----|-------------|
| K-factor | 4.0 | `set k=` | Rating volatility per game |
| Home advantage | 24.0 | `set home=` | Home field Elo bonus |
| Player boost | 20.0 | `set boost=` | Roster strength weight |
| Starter boost | 70.0 | `set starter=` | Starting pitcher quality weight |
| Rest factor | 10.0 | `set rest=` | Rest day bonus/penalty |
| Travel factor | 20.0 | `set travel=` | Distance-based penalty |
| Pace factor | 25.0 | `set pace=` | Run environment weight |
| Playoff HCA | 0.70 | `set playoff=` | October HCA multiplier |
| SOS factor | 0.0 | `set sos=` | Strength of schedule weight |
| Form weight | 0.0 | `set form=` | Recent form adjustment |

---

## Validation Workflow

Run in order after optimization for thorough model validation:

| Step | Command | What to Check | Red Flag |
|------|---------|---------------|----------|
| 1 | `backtest` | Accuracy, log loss, Brier, ECE, MCE, BSS | Accuracy < 56% |
| 2 | `convergence` | Games before Elo stabilizes | Burn-in > 300 games |
| 3 | `sliding` | Does old data help or hurt | Sliding wins = regression too mild |
| 4 | `purgedcv` | Accuracy std across folds | Std > 3% = fragile |
| 5 | `cpcv` | % of paths above threshold | Any path < 52% |
| 6 | `pbo` | Probability of overfitting | PBO > 0.5 = overfit |
| 7 | `montecarlo` | Statistical significance | p > 0.05 = no skill |
| 8 | `enhanced` | SHAP: XGBoost adding signal? | elo_prob > 50% SHAP |
| 9 | `rollingcal` | Is single-pass Platt honest? | Rolling much worse |
| 10 | `betacal` | Asymmetric miscalibration? | a != b by > 0.3 |
| 11 | `conformal` | Prediction set coverage | High "both" % = uncertain |
| 12 | `kelly` | Sharpe ratio, max drawdown | Sharpe < 0.5, DD > 50% |

**Note:** MLB accuracy thresholds are lower than NBA because baseball has inherently more randomness. A 57-60% accuracy in MLB is strong; in NBA you'd expect 65%+.

---

## Trading Ledger (Predicts)

### How Predicts/Kalshi Contracts Work

Predicts and Kalshi contracts are $1 binary options:
- Buy a contract for $0.01-$0.99 (the "price in cents")
- If your prediction is correct: contract pays out $1.00
- If wrong: contract expires worthless
- The price reflects implied probability (55 cents = 55% implied)
- 2% fee on potential payout ($1 per contract) charged at entry and exit

### Position Lifecycle

```
1. ENTRY:   Buy contracts at a price (e.g., 25x @ $0.52)
            Entry fee: 2% x $1 x contracts = $0.50
            Total cost: 25 x $0.52 + $0.50 = $13.50

2. MONITOR: "mark" updates current market price
            "live" shows real-time game scores (innings)
            "predicts" shows unrealized P&L based on marks

3. EXIT (choose one):
   a. RESOLVE: Game ends, contract settles at $1.00 (win) or $0.00 (loss)
      Win:  25 x $1.00 = $25.00 received, profit = $25.00 - $13.50 = $11.50
      Loss: $0.00 received, loss = -$13.50

   b. SELL: Exit early at current market price
      Sell 25x @ $0.68 -> 25 x $0.68 = $17.00 gross
      Exit fee: 2% x $1 x 25 = $0.50
      Net proceeds: $16.50, profit = $16.50 - $13.50 = $3.00

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
  lot_id  date        market                   predicted_winner     ...  status
  1       2026-04-10  Red Sox @ Yankees        New York Yankees     ...  settled
  2       2026-04-11  Cubs @ Dodgers           Los Angeles Dodgers  ...  settled
  ...

  POSITIONS: 15 | CLOSED WIN RATE: 60.0%
  ENTRY FEES: $0.6000 | EXIT FEES: $0.2000
  REALIZED P&L: +$8.4500 | UNREALIZED: +$1.2000 | MARKED P&L: +$9.6500
  ROI REALIZED: +18.3% | ROI MARKED: +20.9%
```

---

## HTML Publishing

Generate prediction tables for blog publishing:

```
> today     # Generate HTML for today's scheduled MLB games
> tomorrow  # Generate HTML for tomorrow's games
```

Outputs:
- `today_mlb_predictions.html` / `tomorrow_mlb_predictions.html` - styled HTML, copy-paste ready for Blogger
- Plain-text summary printed to terminal

Uses MLB Stats API (`statsapi.schedule()`) for accurate game schedules including doubleheaders and postponements.

---

## Data Sources

| Data | Source | Cache Duration | API Key |
|------|--------|----------------|---------|
| Game results (2 seasons) | MLB Stats API (`statsapi.schedule`) | 6 hours | None |
| Batting leaders (top 200) | MLB Stats API (stats leaders) | 6 hours | None |
| Pitching leaders (top 150) | MLB Stats API (stats leaders) | 6 hours | None |
| Injury reports (with IL) | ESPN public JSON API | 4 hours | None |
| Live scores (innings) | MLB Stats API (live scoreboard) | 60 seconds | None |
| Game schedule | MLB Stats API | Per request | None |

All data is fetched automatically on startup and cached locally. Files under 500 bytes are always considered stale and re-downloaded.

---

## File Structure

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point, command dispatcher, prediction flow |
| `elo_model.py` | MLBElo class - ratings, 8 adjustments, predictions |
| `enhanced_model.py` | XGBoost ensemble, TeamTracker, SHAP analysis |
| `backtest.py` | Walk-forward backtest, 5 optimizers, 12 validation methods |
| `build_model.py` | Model construction, season regression, rating initialization |
| `platt.py` | Platt, isotonic, and beta calibration |
| `metrics.py` | Log loss, Brier, ECE, MCE, BSS, conformal sets |
| `config.py` | Settings I/O, 30 MLB team abbreviations, cache management |
| `data_games.py` | MLB Stats API game data download |
| `data_players.py` | MLB Stats API batting + pitching stats, team scoring |
| `injuries.py` | ESPN MLB injury API, IL designations, Elo impact |
| `live_scores.py` | MLB Stats API live scores (innings) |
| `predict_ledger.py` | Trading ledger, Kelly sizing, bankroll management |
| `auto_resolve.py` | Automatic settlement of finished trades |
| `html_generator.py` | Blogger HTML and plain-text table generation |
| `color_helpers.py` | Terminal color formatting wrappers |
| `help_system.py` | In-app CLI help documentation |
| `quick_optimizer.py` | 10-parameter differential evolution optimizer for Elo tuning |
| `accuracy_test.py` | Standalone quick accuracy test script |

### Generated Data Files (gitignored)

| File | Contents |
|------|----------|
| `mlb_recent_games.csv` | ~4,800 games (2 seasons) |
| `mlb_player_stats.csv` | Batting leaders |
| `mlb_advanced_stats.csv` | Pitching leaders |
| `mlb_elo_ratings.json` | Current team Elo ratings |
| `mlb_elo_settings.json` | Tuned parameters + balance + kelly fraction |
| `mlb_platt_scaler.json` | Platt calibration coefficients |
| `mlb_isotonic_scaler.json` | Isotonic calibration bins |
| `mlb_beta_scaler.json` | Beta calibration parameters |
| `mlb_enhanced_model.json` | XGBoost metadata |
| `mlb_xgb_model.json` | XGBoost model weights |
| `mlb_injuries.json` | Cached ESPN injury data |
| `predicts_lots.csv` | Trading ledger (all positions + P&L) |
| `mlb_backtest_predictions.csv` | Per-game backtest predictions |
| `mlb_calibration.csv` | 10-bin calibration table |
| `predicts_pnl.png` | Monthly P&L bar chart |

---

## Configuration & Settings

All settings are stored in `mlb_elo_settings.json` and persist between sessions:

```json
{
  "base_rating": 1500.0,
  "k": 4.0,
  "home_adv": 24.0,
  "use_mov": true,
  "player_boost": 20.0,
  "rest_factor": 10.0,
  "travel_factor": 20.0,
  "sos_factor": 0.0,
  "pace_factor": 25.0,
  "playoff_hca_factor": 0.70,
  "form_weight": 0.0,
  "autoresolve_enabled": false,
  "starting_balance": 200.0,
  "kelly_fraction": 0.50
}
```

Modify at runtime with `set <param>=<value>`. After changing model parameters, always rerun `backtest` to refit the Platt scaler.

---

## Advanced Topics

### Player Scoring and Injury Impact

Each team's roster strength is scored from MLB Stats API batting and pitching leaders:
- **Batting (55%)**: Power + run production composite (`HR*2 + RBI + R*0.5 + SB*0.5 + AVG*100`)
- **Pitching (45%)**: ERA deviation from league average + strikeouts + wins (`(4.50-ERA)*10 + K*0.5 + W*3`)
- Z-scored across all teams for fair comparison
- Applied as an Elo adjustment proportional to `player_boost` setting

When a player is on the IL or marked as Out:
- Their contribution to the team's composite is estimated
- An Elo penalty is applied during predictions
- Impact shown in output: `Boston Red Sox injuries: Chris Sale (15-Day IL, -22 Elo)`

### Travel Distance Calculation

Uses Haversine formula (great-circle distance) between all 30 MLB ballpark coordinates:
- Cross-country trips (e.g., SEA -> MIA) incur larger penalties
- Division rivals with nearby ballparks get minimal travel penalty
- Factor is configurable via `set travel=`

### Conformal Prediction Sets

Distribution-free prediction intervals at configurable coverage levels (90%, 95%, 80%):
- **Singleton set** (one team): model is confident in this pick
- **Both teams**: model is uncertain - consider skipping this game for trading
- **Empty set**: model is overconfident (calibration issue)
- Particularly useful in MLB where many games are close to 50/50

### Why MLB Predictions Are Harder

Baseball has inherently more randomness than basketball:
- In NBA, the better team wins ~70%+ of the time; in MLB it's ~55-58%
- A single game in baseball is heavily influenced by starting pitcher, bullpen usage, and sequencing luck
- 162-game seasons mean more data but each game carries less signal
- Accuracy expectations should be calibrated accordingly: 57% in MLB is roughly equivalent to 67% in NBA

---

## Requirements

- Python 3.8+
- Windows, macOS, or Linux

```bash
pip install -r requirements.txt
```

Dependencies: `pandas`, `numpy`, `scipy`, `xgboost`, `requests`, `MLB-StatsAPI`, `colorama`, `matplotlib`

---

## Trade on Kalshi

If you want to trade prediction market contracts, you can sign up for Kalshi using this referral link:

**[Sign up for Kalshi](https://kalshi.com/sign-up/?referral=e3da1362-7a60-453c-a63e-60f402ec22ab)**

Kalshi is a regulated prediction market exchange where you can trade $1 binary contracts on sports, politics, economics, and more.

---

## Disclaimer

This software is for entertainment, educational, and analytical purposes only. It is not financial advice. Past model performance does not guarantee future results. Prediction market trading involves risk of loss. Always trade responsibly and only with money you can afford to lose.
