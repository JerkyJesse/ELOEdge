# MLB Moneyball + Predicts $1 Contract Tracker

## Overview
MLB game prediction system using Elo ratings + XGBoost ensemble, with Predicts $1 binary contract trading ledger. Built for moneyline (win/loss) predictions.

## Architecture

### Core Model Pipeline
1. **Data Download** (`data_games.py`, `data_players.py`)
   - Games: MLB Stats API (`statsapi.schedule()`) → 2 years of completed games
   - Batting: Stats leaders (battingAverage, homeRuns, runsBattedIn) → top 200
   - Pitching: Stats leaders (earnedRunAverage, strikeouts, wins) → top 150
   - Cache: 6-hour staleness check on all CSV files

2. **Elo Model** (`elo_model.py`)
   - `MLBElo` class with 30 MLB teams
   - Base rating 1500, configurable K-factor (default 4.0 for 162-game season)
   - Home field advantage (default 24 Elo, ~54% implied home win rate)
   - Margin of victory adjustment (logarithmic, capped)
   - Player strength boost (z-scored team batting+pitching composite)
   - Starting pitcher quality (per-pitcher cumulative Elo ratings, 700+ tracked, K_PITCHER=6, 50% season regression)
   - Rest days, travel distance, form, SOS, run-environment (pace), altitude adjustments
   - Playoff detection (October) with reduced HCA factor
   - Injury-aware predictions via ESPN API

3. **XGBoost Ensemble** (`enhanced_model.py`)
   - 31 rolling features per game (TeamTracker class, includes Pythagorean, streaks, consistency, trend)
   - Walk-forward training (no leakage): Elo-only for first 200 games
   - Default blend: 80% Elo / 20% XGBoost
   - Optional time-decay: transitions from 95% Elo early to 70% Elo late
   - SHAP feature importance via XGBoost native `pred_contribs`

4. **Calibration** (`platt.py`)
   - Platt scaling: logistic regression on logit(raw_prob)
   - Isotonic regression: PAV algorithm (pure numpy, no sklearn)
   - Beta calibration: 3-parameter asymmetric (a, b, c)
   - Season regression: 33% pull toward mean at year boundaries

5. **Backtesting & Optimization** (`backtest.py`)
   - Walk-forward backtest with Platt fitting
   - Grid search, genetic (scipy DE), Bayesian (GP + EI) optimization
   - `autoopt`: automatic grid→genetic→bayesian pipeline
   - `superopt`: exhaustive 7-phase optimization (9 params, hours)
   - Purged CV, CPCV, PBO, Monte Carlo permutation test
   - Kelly criterion, sliding window, convergence analysis
   - Conformal prediction, beta calibration comparison
   - Deflated Sharpe Ratio for multiple-testing adjustment

### Trading Ledger (`predict_ledger.py`)
- CSV-based lot tracking with entry/exit fees (2%)
- Add, sell (partial), resolve, mark, invert positions
- Monthly P&L chart generation
- Live score matching for open positions

### Live Features
- `live_scores.py`: MLB Stats API live scores, 60s refresh loop
- `auto_resolve.py`: Auto-settle finished games against open trades
- `injuries.py`: ESPN MLB injury API with Elo impact scoring
- `html_generator.py`: Blogger-ready HTML prediction tables

## File Map
| File | Purpose |
|------|---------|
| `main.py` | CLI entry point, command dispatch |
| `config.py` | Constants, 30 MLB teams, settings I/O |
| `elo_model.py` | MLBElo class (ratings, predictions, adjustments) |
| `build_model.py` | Model training with season regression |
| `data_games.py` | Game data download via MLB Stats API |
| `data_players.py` | Player stats download + team scoring |
| `backtest.py` | All backtesting & optimization (~2100 lines) |
| `enhanced_model.py` | XGBoost ensemble + SHAP |
| `platt.py` | Calibration (Platt, isotonic, beta, regression) |
| `metrics.py` | Log loss, Brier, ECE, MCE, BSS, conformal |
| `predict_ledger.py` | Contract ledger management |
| `live_scores.py` | Live MLB scores + open trade display |
| `auto_resolve.py` | Auto-settle finished trades |
| `injuries.py` | ESPN injury report + impact calculation |
| `html_generator.py` | Blogger HTML table generation |
| `help_system.py` | Help text for all commands |
| `accuracy_test.py` | Quick walk-forward accuracy test |
| `color_helpers.py` | Colorama terminal formatting |

## Data Files (generated at runtime)
- `mlb_recent_games.csv` - Game history (2 years)
- `mlb_player_stats.csv` - Batting leaders
- `mlb_advanced_stats.csv` - Pitching leaders
- `mlb_elo_ratings.json` - Saved Elo ratings
- `mlb_elo_settings.json` - Tuned parameters
- `mlb_platt_scaler.json` - Platt calibration coefficients
- `mlb_isotonic_scaler.json` - Isotonic calibration
- `mlb_beta_scaler.json` - Beta calibration
- `mlb_enhanced_model.json` - XGBoost metadata
- `mlb_xgb_model.json` - XGBoost model weights
- `predicts_lots.csv` - Trading ledger

## Key MLB-Specific Design Choices
- **K-factor = 4.0**: Lower than NBA (8.38) because 162-game season provides more signal per team
- **Home advantage = 24 Elo**: Reflects ~54% MLB home win rate (vs ~60% NBA)
- **Altitude bonus**: Only Colorado Rockies (Coors Field, 5280 ft)
- **Season = calendar year**: MLB runs April-October within one year (unlike NBA cross-year)
- **Playoff detection**: October games get reduced HCA (factor 0.70)
- **Player scoring**: Top 10 batters by composite (HR*2 + RBI + AVG*100), scored as HR*2 + RBI + R*0.5 + SB*0.5 + AVG*100. Batting 45% / pitching 55% (pitching-heavy, reflects MLB reality)
- **Pitching score**: Top 8 pitchers by composite ((4.50-ERA)*5 + K*0.5), scored as (4.50-ERA)*10 + K*0.5 + W*3. Scaled to match batting magnitude before blending
- **Rolling window**: 15 games (wider than NBA's 10 due to higher game-to-game variance in baseball)

## Dependencies
- pandas, numpy, scipy, colorama, xgboost, requests, MLB-StatsAPI, matplotlib
