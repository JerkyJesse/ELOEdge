# NBA Moneyball -- 31-Model Mega-Ensemble

A production-grade NBA game prediction system that combines a deeply-tuned Elo rating engine with 31 independent statistical, machine learning, and physics-inspired models. All 31 model outputs feed a meta-learner (Ridge/Logistic/XGBoost) that produces a single calibrated win probability for every game. Includes a full Predicts $1 contract trading ledger with mark-to-market, auto-resolve, Kelly criterion sizing, and P&L tracking.

Every data source is **completely free** -- no paid APIs, no subscriptions, no keys required for core functionality. The system pulls game logs and player stats from the official NBA API (`nba_api`), injury reports from ESPN, weather from Open-Meteo, and optional betting odds from The Odds API (free tier, 500 requests/month).

The interactive CLI handles the entire workflow: download data, build ratings, backtest, optimize parameters across 7 phases, validate with 16 statistical tests, generate daily prediction tables, log trades, and track P&L -- all from a single `python main.py` session.

---

## Quick Start

### Install

```bash
cd NBAClaude
pip install -r requirements.txt
python main.py
```

### First-Time Workflow

```
1. Launch              python main.py
                       (auto-downloads 2+ seasons of NBA games, player stats, injuries)
                       (auto-runs baseline backtest, fits Platt calibration scaler)

2. Check ratings       all            -- see all 30 teams ranked by Elo
                       settings       -- verify K-factor, home advantage, etc.
                       injuries       -- review current injury impact by team

3. Make a prediction   Lakers         -- type any team name (fuzzy match)
                       -> Celtics     -- enter opponent
                       -> a           -- who is home? (a=first team, b=second, n=neutral)
                       -> See calibrated win probability with injury adjustments

4. Log a trade         y              -- after seeing prediction
                       -> 5           -- number of $1 contracts
                       -> 0.62        -- price per contract
                       -> notes       -- optional notes

5. Run mega-ensemble   mega           -- 31-model ensemble backtest
                       mega tune      -- per-model solo optimization
                       mega optimize  -- 7-phase per-model optimization (54 hyperparams)

6. Publish             today          -- generate HTML table for all today's games
                       tomorrow       -- generate tomorrow's slate

7. Settle              resolve        -- manually settle (win/loss)
                       autoresolve    -- auto-settle from live final scores
```

### Environment Setup

```bash
# Core install (CPU-only PyTorch for neural network models)
pip install -r requirements.txt

# For GPU-accelerated PyTorch (optional, speeds up MLP/LSTM training)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "import nba_api; import xgboost; import lightgbm; print('All good')"
```

---

## The 31 Models

Every model produces an independent win probability for each game. The meta-learner stacks them into one final prediction.

### Tier 0 -- Core (Always On)

| # | Model | Year | Method | Description |
|---|-------|------|--------|-------------|
| 1 | **Elo** | 1960 | Rating system | 24+ adjusters: home court, player strength, rest, travel, pace, altitude, injuries, SOS, streaks, back-to-back penalty, playoff HCA, division familiarity, scoring consistency, mean reversion, season phase |
| 2 | **XGBoost** | 2016 | Gradient boosting | 31 rolling features per team (10-game window): Elo prob, Elo diff, player diff, PPG, PAPG, win%, margins, rest days, Pythagorean expectation, streaks, consistency, trend. 80% Elo / 20% XGBoost blend. Walk-forward retrained every 50 games |

### Tier 1 -- Proven Models

| # | Model | Year | Method | Description |
|---|-------|------|--------|-------------|
| 3 | **HMM** | 1966 | Hidden Markov Model | Detects hot/cold team states from observable game outcomes. Teams transition between latent performance regimes (2-4 states). Probability conditioned on current hidden state |
| 4 | **Kalman** | 1960 | Kalman Filter | Recursive Bayesian estimation of true team strength. Separates signal (real ability) from noise (game variance). Process noise and measurement noise are tunable |
| 5 | **PageRank** | 1998 | Network analysis | Builds directed win graph, applies PageRank + HITS authority scores. Captures transitive strength: beating teams that beat good teams matters. Temporal decay weights recent games more |
| 6 | **LightGBM** | 2017 | Leaf-wise boosting | Microsoft's gradient boosting with leaf-wise tree growth (faster, often more accurate than XGBoost on tabular data). Uses same feature set as XGBoost for diversity |
| 7 | **CatBoost** | 2017 | Ordered boosting | Yandex's gradient boosting with ordered target statistics (reduces prediction shift). Handles categorical features natively. Third independent GBM vote |
| 8 | **MLP** | 1986 | Neural network | Multi-layer perceptron with dropout and batch normalization. Captures non-linear feature interactions that tree models miss. PyTorch implementation |
| 9 | **LSTM** | 1997 | Recurrent neural net | Long Short-Term Memory network for sequential game patterns. Models temporal dependencies across a team's season arc. Off by default (slow to train) |

### Tier 2 -- Exotic / Physics-Inspired

| # | Model | Year | Method | Description |
|---|-------|------|--------|-------------|
| 10 | **GARCH** | 1986 | Volatility modeling | Generalized Autoregressive Conditional Heteroskedasticity. Models time-varying performance variance. High-volatility teams are harder to predict -- GARCH captures this |
| 11 | **Fourier** | 1807 | Cycle detection | Fourier transforms + wavelet decomposition on scoring time series. Detects periodic patterns (weekly rhythms, monthly form cycles). Signal processing for sports |
| 12 | **Survival** | 1972 | Hazard analysis | Cox proportional hazards on streak durations. Models "time until streak ends" -- a winning streak's hazard rate increases with length. Captures regression to mean |
| 13 | **Copula** | 1959 | Joint dependency | Models the joint distribution of offense and defense performance. Captures asymmetric dependency: elite offense + poor defense behaves differently than the marginals suggest |

### Tier 3 -- Information & Physics

| # | Model | Year | Method | Description |
|---|-------|------|--------|-------------|
| 14 | **Info Theory** | 1948 | Shannon entropy | Shannon entropy of scoring distribution + KL divergence between team pairs. Low-entropy teams are predictable; high KL divergence means style mismatch |
| 15 | **Momentum** | 1687 | Newtonian physics | Models team form as physical momentum (mass x velocity). Friction coefficient controls decay. A team on a 7-game win streak has high momentum; friction bleeds it off |
| 16 | **Markov** | 1906 | Transition matrices | Builds state transition matrix from game outcomes. States: strong-win, close-win, close-loss, blowout-loss. Predicts next-game state from current state distribution |
| 17 | **Clustering** | 1957 | k-Means archetypes | Clusters teams into archetypes (e.g., defensive grinders, pace-and-space offenses). Matchup probabilities are archetype-vs-archetype historical records |
| 18 | **Game Theory** | 1950 | Nash equilibrium | Models games as strategic interactions. Computes mixed-strategy Nash equilibria for style matchups. Identifies exploitable tendencies in team strategies |

### Tier 4 -- Classical Rating Systems

| # | Model | Year | Method | Description |
|---|-------|------|--------|-------------|
| 19 | **Poisson** | 1898 | Score distributions | Dixon-Coles Poisson model for score prediction. Estimates team attack/defense parameters, simulates score probabilities. Basketball adaptation uses high-scoring Poisson |
| 20 | **Glicko-2** | 2001 | Uncertainty-aware | Mark Glickman's rating system with rating deviation (RD) and volatility. Teams with few recent games have high RD = less certain predictions. More principled than raw Elo |
| 21 | **Bradley-Terry** | 1952 | Paired comparison | Maximum likelihood estimation of team strengths from head-to-head results. Clean mathematical framework. Recency decay weights recent matchups more heavily |
| 22 | **Monte Carlo** | 1946 | Stochastic simulation | 3,000 game simulations per prediction using historical scoring distributions. Produces full probability distributions, not just point estimates |
| 23 | **Random Forest** | 2001 | Bagging ensemble | Hundreds of decorrelated decision trees. Provides diversity against the GBM models (XGBoost, LightGBM, CatBoost) through bootstrap aggregating |

### Tier 5 -- Classical Sports Models

| # | Model | Year | Method | Description |
|---|-------|------|--------|-------------|
| 24 | **SRS** | ~1980 | Simple Rating System | Margin-of-victory plus strength-of-schedule. Iterative solve: team_rating = avg_margin + avg_opponent_rating. The foundation of many power ranking systems |
| 25 | **Colley** | 2001 | Bias-free ranking | Colley Matrix method: solves Ax=b where A encodes the schedule and b encodes results. Produces rankings with no bias toward conference strength or schedule difficulty |
| 26 | **Log5** | ~1981 | Bill James formula | Log5 head-to-head formula: P(A beats B) = (pA - pA*pB) / (pA + pB - 2*pA*pB). Uses team win percentages. Simple but remarkably effective baseline |
| 27 | **PythagenPat** | 2005 | Dynamic exponent | Pythagorean expectation with dynamically computed exponent based on scoring environment. For NBA, base exponent is 14 (Morey's research). Converts points scored/allowed to expected win% |
| 28 | **Exp Smoothing** | 1957 | Trend tracking | Exponential smoothing (Holt-Winters family) on team performance metrics. Separates level, trend, and seasonality components. Adapts faster than simple moving averages |
| 29 | **Mean Reversion** | ~1990 | Bollinger bands | Tracks team performance relative to historical mean. When a team's rolling metrics exceed 2 standard deviations (Bollinger band breakout), regression is likely |

### Tier 6 -- Data Enrichment (Off by Default)

| # | Model | Year | Method | Description |
|---|-------|------|--------|-------------|
| 30 | **Weather** | -- | Environmental | Temperature, humidity, wind impact on outdoor venues. Limited applicability for NBA (indoor arenas) but included for completeness. Free via Open-Meteo |
| 31 | **Odds** | -- | Market consensus | Ingests live betting lines from The Odds API. Tracks CLV (Closing Line Value). Market odds represent the wisdom of crowds -- a powerful independent signal |

---

## Architecture

```
                         DATA LAYER
    +------------------+------------------+------------------+
    |   nba_api        |   ESPN JSON      |   Open-Meteo     |
    |   Game Logs      |   Injuries       |   Weather        |
    |   Player Stats   |   (4hr cache)    |   (2hr cache)    |
    |   (6hr cache)    |                  |   (no key)       |
    +--------+---------+--------+---------+--------+---------+
             |                  |                  |
             v                  v                  v
    +--------------------------------------------------------+
    |              FEATURE ENGINEERING                        |
    |  TeamTracker: 10-game rolling window per team           |
    |  31 features: PPG, PAPG, win%, margins, rest,           |
    |  Pythagorean exp, streaks, consistency, trend           |
    +----------------------------+---------------------------+
                                 |
              +------------------+------------------+
              |                                     |
              v                                     v
    +-------------------+             +---------------------------+
    |    ELO ENGINE     |             |   30 INDEPENDENT MODELS   |
    |    (NBAElo)       |             |                           |
    |                   |             |  T1: HMM, Kalman,         |
    |  24+ adjusters:   |             |      PageRank, LightGBM,  |
    |  - Home court     |             |      CatBoost, MLP, LSTM  |
    |  - Player str.    |             |  T2: GARCH, Fourier,      |
    |  - Rest days      |             |      Survival, Copula     |
    |  - B2B penalty    |             |  T3: Info Theory,         |
    |  - Travel fatigue |             |      Momentum, Markov,    |
    |  - Altitude       |             |      Clustering, GameThry |
    |  - Pace mismatch  |             |  T4: Poisson, Glicko,     |
    |  - Injuries       |             |      Bradley-Terry, MC,   |
    |  - SOS            |             |      Random Forest        |
    |  - Streaks        |             |  T5: SRS, Colley, Log5,   |
    |  - Division       |             |      PythagenPat, ExpSm,  |
    |  - Playoff HCA    |             |      Mean Reversion       |
    |  - Season phase   |             |  T6: Weather, Odds        |
    +--------+----------+             +------------+--------------+
             |                                     |
             |    +------+------+------+------+    |
             +--->| p_elo| p_hmm|p_kalm| ...  |<--+
                  +------+------+------+------+
                  |    31 probability columns   |
                  +-------------+---------------+
                                |
                                v
                  +----------------------------+
                  |      META-LEARNER          |
                  |  (Ridge / Logistic / XGB)  |
                  |                            |
                  |  Elo-anchored bounded      |
                  |  adjustment:               |
                  |                            |
                  |  p_final = p_elo +         |
                  |    clamp(meta_adj,         |
                  |          -max_adj,         |
                  |          +max_adj)         |
                  |                            |
                  |  Retrained every N games   |
                  |  Walk-forward (no leakage) |
                  +-------------+--------------+
                                |
                                v
                  +----------------------------+
                  |    PLATT CALIBRATION       |
                  |  Logistic regression on     |
                  |  logit(raw_prob)            |
                  |  Fixes over/under-conf.     |
                  +-------------+--------------+
                                |
                                v
                  +----------------------------+
                  |   FINAL CALIBRATED         |
                  |   WIN PROBABILITY          |
                  |   (0.00 - 1.00)            |
                  +----------------------------+
```

### Elo-Anchored Bounded Adjustment

The meta-learner does NOT replace Elo -- it nudges it. The final probability is:

```
p_final = p_elo + clamp(meta_learner_adjustment, -max_adj, +max_adj)
```

With `max_adj = 0.20` (default), the meta-learner can shift Elo's probability by at most +/- 20 percentage points. This prevents the meta-learner from overriding Elo with wild predictions when it encounters novel situations. Elo remains the anchor; the 30 other models provide bounded corrections.

### Multithreaded Training

Models in Tiers 1-5 are trained independently, so they run in parallel via `ThreadPoolExecutor`. On a 4-core machine, this cuts mega-ensemble backtest time by roughly 60%. Each model thread gets its own copy of the feature matrix and writes probabilities to a shared array.

### GPU Acceleration

PyTorch-based models (MLP, LSTM) automatically detect and use CUDA GPUs when available. The GBM models (XGBoost, LightGBM, CatBoost) also support GPU training via their respective `device` parameters. CPU fallback is seamless -- no configuration needed.

---

## NBA-Specific Design Choices

Every parameter in this system was chosen for basketball-specific reasons. Here is why each value is what it is.

### K-Factor = 8.0

The NBA regular season has 82 games per team (1,230 total). A low K-factor prevents overreaction to individual results in such a long season. Compare:
- NFL K ~ 20 (only 17 games, each one matters enormously)
- MLB K ~ 1.0 (162 games, very low K keeps ratings stable)
- NBA K ~ 8.0 (82 games, balanced between NFL and MLB)

K = 8.0 was found via grid search + genetic optimization over multiple seasons. Higher K causes ratings to whipsaw; lower K makes them too sluggish to capture mid-season improvement.

### Home Advantage = 34.0 Elo Points (~60% Home Win Rate)

NBA home court advantage translates to roughly 3.2 points on the spread and approximately 60% home team win rate historically. The 34.0 Elo point value maps to this through the Elo expected score formula:

```
E = 1 / (1 + 10^(-34.0/400)) = ~0.549
```

Combined with other home-related factors (rest, travel, altitude), the effective home win rate reaches the observed ~60%.

### Back-to-Back Penalty

The NBA is the sport where back-to-back games matter most. Players cover enormous court distances at high intensity, and recovery time is critical. Teams playing their second game in two nights show measurably worse performance:
- Historically ~3-5% lower win rate on the second night of a B2B
- Effect is amplified on road B2Bs (travel + fatigue compounding)
- The `b2b_penalty` parameter reduces Elo expectations for B2B teams

### 10-Game Rolling Window

The `TeamTracker` uses a 10-game window for all rolling statistics. Why 10?
- 10 games = roughly 2 weeks of NBA schedule
- Large enough to smooth single-game variance
- Small enough to capture genuine form changes (injuries, trades, lineup experiments)
- NBA teams play 3-4 games per week, so 10 games reflects ~3 weeks of real time

### Pythagorean Exponent = 14

The Pythagorean win expectation formula is:

```
Expected Win% = PtsFor^exp / (PtsFor^exp + PtsAgainst^exp)
```

The exponent of 14 comes from Daryl Morey's basketball-specific research (later refined by Dean Oliver and others). Basketball's high-scoring nature and relatively low variance in final scores justify a much higher exponent than baseball (1.83) or football (~2.37). At exp=14, small scoring advantages translate to large win probability differences, reflecting basketball's relatively deterministic outcomes.

### Altitude Factor -- Denver Nuggets Only

Ball Arena in Denver sits at 5,280 feet (1,609 meters). Visiting teams experience measurably reduced aerobic capacity, affecting:
- Fast-break execution and transition defense
- Late-game stamina and free-throw shooting
- Back-to-back performance when Denver is the second game

The `altitude_factor` parameter gives Denver a small Elo bonus at home, similar to the well-documented effect in other sports (NFL Broncos, MLS Rapids, Liga MX clubs in Mexico City).

### No Park Factor, No Pitcher Concepts

Unlike MLB, basketball has no park factors (all courts are the same dimensions), no starting pitcher matchup (it is a full-team game), no bullpen management, and no interleague play distinction. The Elo parameters `park_factor_weight`, `starter_boost`, `bullpen_factor`, `opp_pitcher_factor`, and `interleague_factor` are present in the shared codebase but unused (zeroed out) for NBA.

### Season Detection: Cross-Year Calendar

The NBA season runs October through June, crossing the calendar year boundary. Season detection uses:

```python
season_year = current_year + 1  if month >= 10  else current_year
```

So games in October 2025 through June 2026 are all "2025-26 season." Season regression (33% toward league mean) is applied at detected season boundaries during model building.

### 6 Divisions, 2 Conferences

```
EASTERN CONFERENCE              WESTERN CONFERENCE
+------------------+            +-------------------+
| Atlantic         |            | Northwest         |
|  BOS BKN NYK     |            |  DEN MIN OKC      |
|  PHI TOR         |            |  POR UTA          |
+------------------+            +-------------------+
| Central          |            | Pacific           |
|  CHI CLE DET     |            |  GSW LAC LAL      |
|  IND MIL         |            |  PHX SAC          |
+------------------+            +-------------------+
| Southeast        |            | Southwest         |
|  ATL CHA MIA     |            |  DAL HOU MEM      |
|  ORL WAS         |            |  NOP SAS          |
+------------------+            +-------------------+
```

The `division_factor` parameter reduces prediction confidence for division matchups where teams are highly familiar with each other's tendencies, making outcomes harder to predict.

---

## Complete Command Reference

### Predictions & Data

| Command | Description | Time |
|---------|-------------|------|
| `<team name>` | Start prediction for any team. Fuzzy match: `Lakers`, `celtics`, `GSW`, `knicks`, `celt`, `warriors` all work | instant |
| `all` | Show all 30 teams ranked by current Elo rating | instant |
| `players` | Show top scorers league-wide (composite of PTS, REB, AST, etc.) | instant |
| `injuries` | Show current injury report with team-level impact estimates | instant |
| `injuries set <team> <player1>, <player2>` | Manually mark players as OUT for a team | instant |
| `refresh` | Force re-download of all cached data (games, players, ratings, scaler) and rebuild model | 30-60s |
| `today` / `html` / `blog` / `blogger` | Generate HTML prediction table for all of today's games | 5-10s |
| `tomorrow` | Generate HTML prediction table for tomorrow's games | 5-10s |
| `odds` | Show current betting odds for today's NBA games (requires Odds API key) | 2-5s |
| `weather` | Show weather report for a specific arena (limited relevance for indoor NBA) | 2-5s |
| `settings` | Show all current Elo parameters and Platt scaler status | instant |
| `advstats` | Show advanced analytics | 5-10s |

### Backtesting

| Command | Description | Time |
|---------|-------------|------|
| `backtest` | Walk-forward backtest on full history. Fits Platt calibration scaler. Reports accuracy, log loss, Brier, ECE, MCE, BSS | 15-30s |
| `enhanced` | XGBoost ensemble backtest (80/20 Elo/XGB blend, 31 features, walk-forward retrained). Auto-runs SHAP analysis | 30-60s |
| `enhanced decay` | Same as `enhanced` but with time-decayed weighting (95% Elo early season to 70% Elo late season) | 30-60s |
| `shap` | Standalone SHAP feature importance analysis for the saved XGBoost model | 5-10s |

### Elo Optimization

| Command | Description | Time |
|---------|-------------|------|
| `grid` | Cartesian grid search over 7 parameters (K, HomeAdv, PlayerBoost, Rest, Travel, Pace, PlayoffHCA). Interactive prompts for ranges | 10-60min |
| `genetic` | Differential evolution optimizer over same 7 parameters. Configurable generations and population size | 5-30min |
| `bayesian` | Gaussian Process surrogate + Expected Improvement acquisition. Finds optima in ~50-100 evaluations | 5-15min |
| `autoopt` | Fully automatic: grid (768 combos) then genetic (50 gen) then bayesian (40 iter). No prompts, applies best result | 15-30min |
| `superopt` | Exhaustive 7-phase optimization over all 9 parameters. Runs for hours. The definitive optimization | 2-4hr |
| `singleopt` / `coord` | Coordinate descent: one parameter at a time, accuracy-focused. Max 5 passes until convergence | 15-30min |
| `results` | Show best parameters from all optimizers, with DSR (Deflated Sharpe Ratio) significance test | instant |

### Validation & Statistical Testing

| Command | Description | Time |
|---------|-------------|------|
| `purgedcv` | Purged walk-forward CV: k chronological folds with embargo gap. Reports mean/std accuracy across folds | 1-3min |
| `cpcv` | Combinatorial purged CV: all C(k, k_test) train/test combinations (10 paths with k=5, k_test=2) | 2-5min |
| `pbo` | Probability of backtest overfitting. Requires `grid` to have been run first. PBO > 0.5 = overfit | 1-2min |
| `montecarlo` | Monte Carlo permutation test. Shuffles outcomes 500+ times, computes p-value. Definitive skill test | ~8min |
| `rollingcal` | Rolling origin Platt recalibration (expanding window, truly out-of-sample) | 1-2min |
| `betacal` | Beta calibration (3-parameter, detects asymmetric miscalibration). If a != b, use beta over Platt | 1-2min |
| `conformal` | Conformal prediction sets at 80%, 90%, 95% coverage. Singleton% = model confidence quality | 1-2min |
| `convergence` | Elo burn-in analysis: per-chunk accuracy to find where ratings stabilize | 30s |
| `sliding` | Sliding vs expanding window comparison. Determines if old data helps or hurts | 1-2min |
| `kelly` | Kelly criterion position sizing backtest. Simulates fractional Kelly on historical predictions | 30s |

### Mega-Ensemble

| Command | Description | Time |
|---------|-------------|------|
| `mega` | Run full 31-model mega-ensemble backtest | 3-10min |
| `mega optimize` / `mega opt` | 7-phase per-model exhaustive mega-ensemble optimization (54 hyperparameters) | 1-4hr |
| `mega tune` | Per-model solo optimization (Phase 1 only) | 15-30min |
| `mega tournament` | Head-to-head model tournament (Phase 2 only) | 15-30min |
| `mega quick` | Quick grid search (Phase 1 only) | 15-30min |
| `mega ablation` | Single-model ablation study -- tests each model's individual contribution | 30-60min |
| `mega models` | Show all 31 models with ON/OFF status, organized by tier | instant |
| `mega on <model>` | Enable a specific model (e.g., `mega on lstm`) | instant |
| `mega off <model>` | Disable a specific model | instant |
| `mega on all` | Enable all 31 models | instant |
| `mega settings` / `mega params` | Show all mega-ensemble parameter values | instant |
| `mega set <param>=<value>` | Set a mega parameter (e.g., `mega set adj=0.10`, `mega set meta=ridge`) | instant |

### Trading Ledger

| Command | Description | Time |
|---------|-------------|------|
| `predicts` / `summary` | Show full contract ledger with realized/unrealized/marked P&L, win rate, ROI | instant |
| `balance` | Show current account balance | instant |
| `resolve` | Settle a finished contract (win/loss prompt) | interactive |
| `sell` | Exit an open position at a specified price (partial or full) | interactive |
| `mark` | Update current market price on open positions for MTM P&L | interactive |
| `invert` | Flip direction of an open position (swaps predicted winner, inverts probability) | interactive |
| `live` | Show real-time scores for games with open positions | 5-10s |
| `chart` | Generate monthly realized P&L bar chart (saved as image) | 2-5s |
| `autoresolve` | Manually trigger auto-settlement from live final scores | 5-10s |
| `autoresolve on` / `off` | Toggle automatic resolve on startup | instant |

### Settings

| Command | Description | Time |
|---------|-------------|------|
| `set k=8.38` | Set Elo K-factor | instant |
| `set home=28` | Set home court advantage | instant |
| `set boost=28` | Set player strength boost | instant |
| `set rest=16` | Set rest days factor | instant |
| `set b2b=30` | Set back-to-back penalty | instant |
| `set travel=33` | Set travel fatigue factor | instant |
| `set pace=40` | Set pace mismatch factor | instant |
| `set sos=5` | Set strength of schedule weight | instant |
| `set streak=5` | Set win/loss streak momentum | instant |
| `set playoff=0.56` | Set playoff home advantage multiplier | instant |
| `set kelly=quarter` | Set Kelly fraction (quarter/half/full/0.25/0.50) | instant |
| `set balance=1000` | Set starting account balance | instant |
| `set autoresolve=true` | Toggle auto-resolve | instant |

### Utility

| Command | Description | Time |
|---------|-------------|------|
| `help` | Show command overview | instant |
| `help <command>` | Detailed help for a specific command | instant |
| `help advanced` | Show all 16 advanced validation commands | instant |
| `platt` / `calibrate` | Show Platt scaler status (fitted date, samples, coefficients) | instant |
| `quit` / `q` | Save ratings and settings, then exit | instant |

---

## All Settable Parameters

### Elo Parameters (39 Total)

Parameters are set with `set <param>=<value>` or `set <alias>=<value>`. After any change, run `backtest` to refit the Platt scaler.

#### Core

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `k` | `k_factor` | float | 8.38 | Elo K-factor (learning rate per game). Lower = more stable, higher = more reactive |
| `base_rating` | `base`, `rating` | float | 1500.0 | Starting Elo rating for all teams |
| `home_adv` | `home`, `hca`, `home_advantage` | float | 28.05 | Home court advantage in Elo points (~60% win rate) |
| `use_mov` | `mov`, `margin` | bool | true | Use margin of victory adjustment (log-scaled) |

#### Player Strength

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `player_boost` | `boost`, `player` | float | 28.0 | Team-level composite player strength (PTS + REB + AST composite) |

*Note: `starter_boost`, `bullpen_factor`, `opp_pitcher_factor` exist in the shared codebase but are unused (zeroed) for NBA -- there are no pitcher concepts in basketball.*

#### Margin of Victory

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `mov_base` | `mov_mult`, `mov_constant` | float | (default) | MOV multiplier constant (log curve shift) |
| `mov_cap` | `movcap`, `margin_cap` | float | (default) | Maximum MOV adjustment cap |

#### Rest & Schedule

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `rest_factor` | `rest` | float | 16.02 | Rest days advantage factor |
| `rest_advantage_cap` | `restcap`, `rest_cap` | float | 0.0 | Maximum rest advantage multiplier |
| `b2b_penalty` | `b2b`, `back_to_back` | float | 0.0 | Back-to-back game penalty (critical for NBA) |
| `road_trip_factor` | `roadtrip`, `road_trip` | float | 0.0 | Extended road trip penalty (3+ consecutive away games) |
| `homestand_factor` | `homestand` | float | 0.0 | Extended homestand bonus |

#### Travel & Venue

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `travel_factor` | `travel` | float | 33.16 | Elo penalty per timezone crossed (jet lag, circadian disruption) |
| `east_travel_penalty` | `east_travel`, `eastbound` | float | (default) | Extra penalty for eastbound travel (harder on circadian rhythm) |
| `altitude_factor` | `altitude`, `alt` | float | 0.0 | Denver altitude bonus multiplier (5,280 ft elevation) |

*Note: `park_factor_weight` exists but is unused for NBA -- all courts are identical dimensions.*

#### Form & Momentum

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `form_weight` | `form` | float | 0.0 | Recent form weight (hot/cold streaks) |
| `win_streak_factor` | `streak`, `win_streak` | float | 0.0 | Win/loss streak momentum factor |
| `mean_reversion` | `reversion`, `regress` | float | 0.0 | Mean reversion after extreme results |
| `season_regress` | `season_regression`, `regress_pct` | float | 0.33 | Season boundary regression fraction (33% toward mean) |

#### Matchup Adjustments

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `sos_factor` | `sos`, `strength_of_schedule` | float | 0.0 | Strength of schedule weight |
| `division_factor` | `division`, `div` | float | 0.0 | Divisional game confidence reducer |
| `series_adaptation` | `series`, `adaptation` | float | (default) | Series adaptation factor (rematches) |

*Note: `interleague_factor` exists but is unused for NBA -- no interleague concept.*

#### Scoring Model

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `pace_factor` | `pace`, `tempo` | float | 39.97 | Run/scoring environment mismatch penalty |
| `pyth_factor` | `pyth`, `pythagorean` | float | (default) | Pythagorean expected W% adjustment (exp = 14) |
| `scoring_consistency_factor` | `consistency`, `scoring_consistency` | float | 0.0 | Penalty for volatile scoring (high game-to-game variance) |
| `home_road_factor` | `home_road`, `split` | float | (default) | Team-specific home/road split bonus |

#### Season & Phase

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `playoff_hca_factor` | `playoff`, `playoff_hca`, `postseason` | float | 0.56 | Playoff home advantage multiplier (reduced from regular season) |
| `season_phase_factor` | `phase`, `season_phase` | float | 0.0 | Early-season dampener (less confident when ratings are immature) |

#### K-Factor Variants

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `k_decay` | `kdecay`, `k_reduction` | float | (default) | K-factor decay over season (reduce learning rate as season progresses) |
| `surprise_k` | `surprise`, `upset_k` | float | (default) | Extra K for surprise results (big upsets update ratings more) |

#### Account & Trading

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `kelly_fraction` | `kelly` | special | 0.50 | Kelly criterion fraction. Accepts: `quarter`, `half`, `full`, or any decimal 0-1 |
| `starting_balance` | `balance`, `bankroll` | float | 0.0 | Starting account balance for P&L tracking |
| `autoresolve_enabled` | `autoresolve`, `auto_resolve` | bool | false | Auto-resolve finished trades on startup |

### Mega-Ensemble Parameters (14 Total)

Set with `mega set <param>=<value>`. Changes take effect on next `mega` run.

| Parameter | Aliases | Type | Description |
|-----------|---------|------|-------------|
| `max_adj` | `maxadj`, `adj`, `adjustment` | float | Max meta-learner adjustment (+/- probability). Default ~0.10 |
| `meta_model` | `meta`, `metalearner`, `stacker` | str | Meta-learner type: `ridge`, `logistic`, `xgboost` |
| `retrain_every` | `retrain`, `retrain_interval` | int | Retrain meta-learner every N games |
| `min_train` | `mintrain`, `min_games`, `warmup` | int | Games before meta-learner starts predicting |
| `kalman_process_noise` | `kalman_pn`, `process_noise`, `pn` | float | Kalman filter process noise |
| `kalman_measurement_noise` | `kalman_mn`, `measurement_noise`, `mn` | float | Kalman filter measurement noise |
| `hmm_states` | `hmm_n`, `n_states`, `states` | int | Number of HMM hidden states (2-4) |
| `network_decay` | `net_decay`, `pagerank_decay`, `decay` | float | PageRank temporal decay (0-1, higher = longer memory) |
| `momentum_friction` | `friction`, `mom_friction` | float | Momentum friction coefficient (higher = faster decay) |
| `n_clusters` | `clusters`, `k_clusters`, `nclusters` | int | Number of team archetype clusters for k-Means |
| `glicko_initial_rd` | `glicko_rd`, `initial_rd`, `rd` | float | Glicko-2 initial rating deviation |
| `bt_decay` | `bt_recency`, `bradley_decay` | float | Bradley-Terry recency decay (0-1) |
| `mc_simulations` | `mc_sims`, `simulations`, `n_sims`, `sims` | int | Monte Carlo simulations per game (default 3000) |
| `window` | `rolling_window`, `feat_window` | int | Rolling feature window size in games |

---

## Optimization System

### Elo Optimization

Six optimization methods are available, each suited to different stages of the tuning process.

#### Grid Search (`grid`)
Cartesian product sweep over 7 parameters. Interactive prompts let you set min/max/step for each dimension. Best for initial exploration when you have no idea where the optimum lies.

- **Parameters**: K, HomeAdv, PlayerBoost, RestFactor, TravelFactor, PaceFactor, PlayoffHCA
- **Objective**: `-(LogLoss * 8 + Brier * 40)`
- **Output**: `nba_grid_search.csv` with all scored combinations
- **Post-processing**: Auto-saves best settings, refits Platt scaler

#### Genetic Optimizer (`genetic`)
Differential evolution via `scipy.optimize.differential_evolution`. Configurable generations (default 80) and population size (default 40). More efficient than grid search for fine-tuning.

#### Bayesian Optimizer (`bayesian`)
Gaussian Process surrogate with Expected Improvement acquisition function. Latin Hypercube initial sampling, squared exponential kernel. Finds optima in ~50-100 evaluations vs thousands for grid search.

#### Auto-Optimize (`autoopt`)
Fully automatic pipeline, no interactive prompts:

```
Phase 1: Coarse grid (768 combos)    -> identify promising region
Phase 2: Genetic (50 gen, pop 25)    -> refine with tightened bounds
Phase 3: Bayesian (15 init + 40 iter) -> fine-tune with GP surrogate
Phase 4: Compare all three winners   -> apply absolute best
```

#### Super-Optimize (`superopt`)
The definitive optimization. Runs for hours with no shortcuts:

```
Phase 1: Broad grid search (9 params, ~6000+ combos)
Phase 2: Genetic round 1 (wide bounds, 100 gen x 50 pop)
Phase 3: Bayesian round 1 (wide bounds, 30 initial + 80 iter)
Phase 4: Genetic round 2 (tightened bounds, 80 gen x 40 pop)
Phase 5: Bayesian round 2 (tightened, 20 initial + 60 iter)
Phase 6: Fine grid (tiny steps around absolute best)
Phase 7: Validation (purgedcv + PBO + Monte Carlo)
```

All 9 Elo parameters are searched. Each round narrows the search space. Takes 2-4 hours depending on data size. Combined with 19 tunable Elo parameters and 54 per-model hyperparameters, the system has 73+ total tunable parameters.

#### Coordinate Descent (`singleopt`)
Optimizes one parameter at a time for accuracy (not LogLoss/Brier). Conservative local search:

```
1. Load current settings as starting point
2. For each parameter: coarse sweep across full range
3. Fine-grained sweep around best coarse value
4. Accept new value only if accuracy improves
5. Repeat passes until no parameter improves (max 5 passes)
```

### Mega-Ensemble Optimization (`mega optimize`)

Per-model 7-phase exhaustive optimization over 54 per-model hyperparameters:

```
Phase 1: Solo test each model individually to find per-model optimal settings
Phase 2: Tournament -- top configs compete head-to-head
Phase 3: Meta-learner tuning (max_adj, meta_model, retrain_every, min_train)
Phase 4: Differential Evolution (DE) over all continuous params
Phase 5: Ablation -- prune models that hurt ensemble accuracy
Phase 6: Validation (purged CV + stability test)
Phase 7: Apply best and save
```

54 per-model hyperparameters are tunable, including: `max_adj` (default 0.20), `meta_model`, `retrain_every`, `min_train`, `kalman_process_noise`, `kalman_measurement_noise`, `hmm_states`, `network_decay`, `momentum_friction`, `n_clusters`, `glicko_initial_rd`, `bt_decay`, `mc_simulations`, and per-model enable/disable flags.

Use `mega tune` for per-model solo optimization (Phase 1 only) and `mega tournament` for head-to-head model comparison (Phase 2 only).

### Mega Ablation (`mega ablation`)

Tests each model's individual contribution by disabling it and measuring the accuracy drop. Models that hurt ensemble performance are automatically flagged. Use this to prune weak models and speed up predictions.

### Recommended Optimization Workflow

```
1. backtest              -- establish baseline (19 Elo params)
2. autoopt               -- quick automatic Elo optimization (15-30 min)
3. backtest              -- verify improvement
4. mega                  -- run mega-ensemble with current Elo settings
5. mega tune             -- per-model solo optimization (finds each model's best settings)
6. mega tournament       -- head-to-head model comparison
7. mega optimize         -- full 7-phase optimization (54 per-model hyperparameters, 1-4 hr)
8. mega ablation         -- prune underperforming models
9. mega                  -- verify final ensemble performance
10. superopt             -- (optional) exhaustive Elo re-optimization with validated params
```

---

## API Setup

### Data Sources

| API | Data | Key Required | Free Tier | Cache Duration |
|-----|------|-------------|-----------|----------------|
| **nba_api** | Game logs, box scores, player stats, team info | No key needed | Unlimited | 6 hours (games), 24 hours (players) |
| **ESPN JSON** | Injury reports (status, return dates) | No key needed | Unlimited | 4 hours |
| **Open-Meteo** | Temperature, humidity, wind, precipitation | No key needed | Unlimited | 2 hours (game days) |
| **The Odds API** | Moneyline, spread, totals from 15+ sportsbooks | Free key needed | 500 req/month | 15 min (game days) |

### The Odds API Setup (Optional)

The Odds API provides live betting lines for value detection and CLV tracking. The free tier gives 500 requests per month, which is sufficient for daily use.

1. Sign up at [the-odds-api.com](https://the-odds-api.com)
2. Copy your API key from the dashboard
3. Set the environment variable:

```bash
# Windows
set ODDS_API_KEY=your_key_here

# Linux/Mac
export ODDS_API_KEY=your_key_here

# Permanent (add to .bashrc or .zshrc)
echo 'export ODDS_API_KEY=your_key_here' >> ~/.bashrc
```

4. Verify: `odds` command in the CLI should show today's lines

### nba_api Notes

The `nba_api` package wraps the official NBA stats API (stats.nba.com). It provides:
- Game logs for current and recent seasons
- Player statistics (per-game, advanced)
- Team information and schedule data

No authentication is needed, but the NBA API has rate limits. The smart cache system (see below) minimizes API calls by only fetching when data is likely to have changed.

---

## Smart Caching

The caching system adapts to the NBA calendar. During the active season (October-June), data is refreshed more frequently. During the offseason (July-September), caches last much longer because the underlying data rarely changes.

### Cache Durations

| Data Type | In-Season (Game Day) | In-Season (Off Day) | Offseason |
|-----------|---------------------|---------------------|-----------|
| Game scores | 6 hours | 6 hours | 168 hours (1 week) |
| Player stats | 24 hours | 48 hours | 168 hours (1 week) |
| Injuries | 4 hours | 8 hours | 168 hours (1 week) |
| Odds | 15 minutes | 2 hours | N/A |
| Weather | 2 hours | 6 hours | N/A |
| Advanced stats | 24 hours | 48 hours | 168 hours (1 week) |

### Season Calendar

```
NBA Season: October - June (cross-year)

     Oct   Nov   Dec   Jan   Feb   Mar   Apr   May   Jun   Jul   Aug   Sep
     [===========REGULAR SEASON===========][PLAYOFFS][=======OFFSEASON======]
     ^                                      ^         ^
     Season start                           Postseason  Offseason
     (preseason in Oct)                     begins      (caches go long)
```

Cache staleness is checked by `config.is_cache_stale()`. Files under 500 bytes are always considered stale (likely corrupt or empty). Force refresh with the `refresh` command to bypass all caching.

---

## Performance

### Multithreading

The mega-ensemble uses `ThreadPoolExecutor` to train models in parallel. On a 4-core CPU:

```
Sequential (1 thread):   ~8-12 minutes for full mega backtest
Parallel   (4 threads):  ~3-5 minutes for full mega backtest
Parallel   (8 threads):  ~2-3 minutes (diminishing returns from GIL)
```

GBM models (XGBoost, LightGBM, CatBoost) have their own internal parallelism via `nthread` parameters. The system avoids oversubscription by coordinating thread counts.

### GPU Acceleration

| Model | CPU Time | GPU Time | Speedup |
|-------|----------|----------|---------|
| MLP (PyTorch) | ~45s | ~8s | ~5.6x |
| LSTM (PyTorch) | ~120s | ~15s | ~8x |
| XGBoost | ~20s | ~5s | ~4x |
| LightGBM | ~15s | ~4s | ~3.8x |
| CatBoost | ~25s | ~6s | ~4.2x |

GPU is auto-detected. To install PyTorch with CUDA support:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Typical Backtest Results

Results vary by season and parameter tuning. Representative ranges:

```
Baseline accuracy:         ~68.93%
Platt-calibrated accuracy: ~68.93%
Mega-ensemble (31 models): 67-71%

Log Loss:   ~0.5896
Brier:      ~0.2016
ECE:        ~0.028
```

---

## Complete Model Validation Workflow

A 6-phase process to thoroughly validate and tune the prediction model. Each phase builds on the previous one. Run phases in order; skip steps that are not relevant to your current question.

### Phase 1: Baseline & Diagnostics

```
1. backtest      -- Establish baseline accuracy, log loss, Brier, ECE, MCE, BSS
2. convergence   -- How many games before Elo ratings stabilize?
3. sliding       -- Does old data help or hurt? (expanding vs sliding window)
```

### Phase 2: Parameter Optimization

```
4. grid          -- Coarse search with wide ranges (produces nba_grid_search.csv)
5. pbo           -- Is the grid search best overfit? (PBO > 0.5 = yes)
6. results       -- Check DSR significance (DSR < 1.96 = not statistically significant)
7. genetic       -- Fine-tune with tighter bounds (or use bayesian)
```

### Phase 3: Cross-Validation

```
8. purgedcv      -- Does optimized model hold up across 5 chronological folds?
9. cpcv          -- Combinatorial validation: all 10 paths through 5 blocks
10. montecarlo   -- Is the model genuinely skilled? (p < 0.05 = yes)
```

### Phase 4: Ensemble & Features

```
11. enhanced     -- Train XGBoost on optimized Elo (SHAP auto-runs)
12. shap         -- Re-examine feature importance standalone
13. enhanced decay -- Compare time-decayed vs fixed 80/20 blend
```

### Phase 5: Calibration Refinement

```
14. rollingcal   -- Is single-pass Platt calibration honest? (OOS comparison)
15. betacal      -- Asymmetric miscalibration? (If a != b, use beta over Platt)
16. conformal    -- Prediction set quality (singleton% = model confidence)
```

### Phase 6: P&L Simulation

```
17. kelly        -- Optimal position sizing simulation (Sharpe, max drawdown)
```

### Decision Framework

| Metric | Good | Marginal | Bad | Action if Bad |
|--------|------|----------|-----|---------------|
| Accuracy | >66% | 63-66% | <63% | Revisit Elo adjusters, check data quality |
| ECE | <0.03 | 0.03-0.08 | >0.08 | Refit Platt, try beta calibration |
| BSS vs 50% | >0.08 | 0.04-0.08 | <0.04 | Model has weak discriminative power |
| PBO | <0.3 | 0.3-0.5 | >0.5 | Grid search overfit -- use wider ranges or fewer params |
| DSR | >1.96 | 1.0-1.96 | <1.0 | Best params are noise -- simplify model |
| Monte Carlo p | <0.01 | 0.01-0.05 | >0.05 | Model has no statistically significant skill |
| Purged CV std | <2% | 2-3% | >3% | Model is fragile to training data selection |
| CPCV paths <55% | 0% | 1-10% | >10% | Path-dependent overfitting |
| Kelly Sharpe | >1.0 | 0.5-1.0 | <0.5 | Edge too thin for real trading |
| Kelly max DD | <30% | 30-50% | >50% | Size down (use 10%-Kelly instead of 25%) |

---

## Daily Prediction Workflow

```
MORNING (pre-game)
  1. Launch        python main.py
                   -> auto-downloads fresh games, players, injuries
                   -> runs baseline backtest, fits Platt scaler
                   -> auto-resolves yesterday's finished trades (if enabled)

  2. Check         injuries       -> review who is OUT / Doubtful
                   all            -> see current team rankings
                   odds           -> check market lines (if Odds API configured)

  3. Predict       Lakers         -> enter opponent -> home/away
                   (repeat for each game of interest)
                   today          -> generate HTML table for ALL today's games

  4. Log           y              -> after each prediction, log contract
                   -> enter count, price, optional notes

  5. Publish       today          -> generates today_nba_predictions.html
                   Copy HTML -> paste into Blogger/WordPress -> publish

EVENING (in-game & post-game)
  6. Monitor       live           -> real-time scores for open positions
                   mark           -> update current market prices

  7. Settle        autoresolve    -> auto-settle from final scores
                   resolve        -> manually settle (win/loss prompt)
                   sell           -> exit early at a price

  8. Review        predicts       -> full P&L ledger
                   chart          -> monthly P&L bar chart
                   balance        -> current account balance
```

---

## Trading Ledger

The Predicts system tracks $1 binary option contracts. Each contract pays $1 on a correct prediction and $0 on an incorrect one.

### Contract Structure

```
Buy 10 contracts at $0.62 each:
  Cost basis:  10 x $0.62 = $6.20
  Entry fee:   10 x $1.00 x 2% = $0.20  (2% of potential payout)
  Total cost:  $6.40

If prediction is CORRECT (win):
  Payout:      10 x $1.00 = $10.00
  Profit:      $10.00 - $6.40 = $3.60

If prediction is WRONG (loss):
  Payout:      $0.00
  Loss:        -$6.40
```

### Operations

| Operation | Command | Description |
|-----------|---------|-------------|
| **Buy** | `y` after prediction | Log new position with count, price, notes |
| **Mark-to-Market** | `mark` | Update current market price on open positions |
| **Sell (partial/full)** | `sell` | Exit at specified price. 2% exit fee on sale proceeds |
| **Resolve** | `resolve` | Binary settlement: $1 per contract (win) or $0 (loss) |
| **Auto-Resolve** | `autoresolve` | Match open positions against final scores from NBA CDN |
| **Invert** | `invert` | Flip position direction (swap predicted winner) without changing cost basis |
| **Summary** | `predicts` | Full ledger: realized P&L, unrealized P&L (from marks), win rate, ROI |
| **Chart** | `chart` | Monthly realized P&L bar chart saved as image |

### Auto-Resolve

When `autoresolve` is enabled (`autoresolve on`), the system automatically matches open positions against today's final game scores on startup. Games are matched by team names and date. Resolved positions are settled at $1 (win) or $0 (loss) with the fee structure above.

---

## File Structure

```
NBAClaude/
|
|-- main.py                    Main CLI entry point, dispatch() routes all commands
|-- config.py                  Constants, team abbreviations, settings I/O, season detection
|-- elo_model.py               NBAElo class: Elo ratings with 24+ adjusters
|-- build_model.py             Constructs NBAElo from settings + game CSV, applies season regression
|-- backtest.py                Walk-forward backtest, grid/genetic/bayesian optimizers, all validation tests
|-- enhanced_model.py          XGBoost ensemble: TeamTracker, 31 features, walk-forward training
|-- platt.py                   Platt scaling + isotonic regression + beta calibration
|-- metrics.py                 ECE, MCE, BSS, log loss, Brier score, conformal prediction
|-- single_param_opt.py        Coordinate descent optimizer (one param at a time)
|
|-- data_games.py              Download NBA game logs from nba_api with smart caching
|-- data_players.py            Download player stats, compute composite scores
|-- injuries.py                ESPN injury report fetcher + team impact calculator
|-- cache_utils.py             Smart season-aware cache staleness logic
|
|-- mega_backtest.py           31-model mega-ensemble backtest engine (multithreaded)
|-- mega_config.py             Model registry (31 models), on/off switches, mega parameter set/get
|-- mega_optimizer.py          7-phase mega-ensemble optimization (grid, coord descent, Bayesian, tournament)
|-- meta_learner.py            Meta-learner: Ridge, Logistic, or XGBoost stacker
|
|-- hmm_model.py               Hidden Markov Model (hot/cold states)
|-- kalman_model.py            Kalman filter (strength estimation)
|-- network_model.py           PageRank + HITS (graph analysis via NetworkX)
|-- gbm_models.py              LightGBM + CatBoost gradient boosting
|-- nn_models.py               MLP + LSTM neural networks (PyTorch)
|-- volatility_model.py        GARCH volatility modeling
|-- signal_model.py            Fourier transforms + wavelet cycle detection
|-- survival_model.py          Cox proportional hazards (streak analysis)
|-- copula_model.py            Copula joint dependency modeling
|-- information_theory_model.py Shannon entropy + KL divergence
|-- momentum_model.py          Newtonian momentum / inertia
|-- markov_chain_model.py      Markov chain transition matrices
|-- clustering_model.py        k-Means team archetype clustering
|-- game_theory_model.py       Nash equilibrium + style matchups
|-- poisson_model.py           Dixon-Coles Poisson score model
|-- glicko_model.py            Glicko-2 rating system
|-- bradley_terry_model.py     Bradley-Terry MLE paired comparison
|-- monte_carlo_model.py       Monte Carlo simulation (3000 sims/game)
|-- random_forest_model.py     Random Forest bagging ensemble
|-- classic_models.py          SRS, Colley, Log5, PythagenPat, ExpSmoothing, MeanReversion
|
|-- odds_tracker.py            The Odds API integration + CLV tracking
|-- weather.py                 Open-Meteo weather data (limited use for indoor NBA)
|
|-- predict_ledger.py          $1 contract ledger: buy, sell, mark, resolve, P&L
|-- live_scores.py             Real-time score tracker for open trades
|-- auto_resolve.py            Auto-settle finished trades from NBA CDN final scores
|-- html_generator.py          Generate HTML prediction tables for Blogger/WordPress
|-- help_system.py             CLI help display system
|-- color_helpers.py           Colorama wrapper functions (cok, cerr, cwarn, chi, cdim, cbold)
|-- elo_set_handler.py         Shared handler for 'set param=value' (39 params with aliases)
|
|-- run_optimize.py            Standalone optimization runner
|-- run_enhanced_all.py        Standalone enhanced backtest runner
|-- accuracy_optimize.py       Accuracy-focused optimizer
|-- accuracy_test.py           Quick standalone accuracy test
|-- quick_optimizer.py         Quick optimization runner
|-- sweep_enhanced.py          Enhanced model sweep utility
|
|-- requirements.txt           All dependencies
|
|-- [Generated at runtime, gitignored:]
|   |-- nba_elo_settings.json        Tunable Elo parameters
|   |-- nba_elo_ratings.json         Current team Elo ratings
|   |-- nba_platt_scaler.json        Platt calibration scaler
|   |-- nba_isotonic_scaler.json     Isotonic calibration scaler
|   |-- nba_enhanced_model.json      Saved XGBoost model metadata
|   |-- nba_xgb_model.json           XGBoost booster weights
|   |-- nba_mega_settings.json       Mega-ensemble parameters + model switches
|   |-- nba_recent_games.csv         Cached game data (2+ seasons)
|   |-- nba_player_stats.csv         Cached player statistics
|   |-- nba_advanced_stats.csv       Cached advanced statistics
|   |-- nba_backtest_predictions.csv Per-game backtest predictions
|   |-- nba_calibration.csv          10-bin calibration table
|   |-- nba_grid_search.csv          Grid search results
|   |-- nba_enhanced_features.npz    Saved feature matrix
|   |-- nba_power_ratings.json       Power ratings cache
|   |-- nba_stacked_scaler.json      Stacked model scaler
|   |-- nba_meta_xgb.json            Meta-learner XGBoost model
|   |-- predicts_lots.csv            Trading ledger
|   |-- today_nba_predictions.html   Today's prediction table
|   |-- today_nba_predictions.txt    Today's predictions (plain text)
```

---

## Requirements

### System Requirements

- Python 3.8+
- 4 GB RAM minimum (8 GB recommended for mega-ensemble)
- Internet connection for initial data download (subsequent runs use cache)
- Optional: CUDA-capable GPU for accelerated neural network and GBM training

### Dependencies

#### Core (Required)

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | >= 1.5 | Data manipulation and CSV I/O |
| numpy | >= 1.24 | Numerical computing |
| scipy | >= 1.10 | Optimization (differential evolution, L-BFGS-B), statistics |
| colorama | >= 0.4 | Colored terminal output |
| xgboost | >= 2.0 | Gradient boosting (Tier 0 model + meta-learner option) |
| requests | >= 2.28 | HTTP requests for APIs (ESPN injuries, Odds, Weather) |
| matplotlib | >= 3.7 | P&L charts and visualization |

#### NBA Data APIs

| Package | Version | Purpose |
|---------|---------|---------|
| nba_api | >= 1.1 | Official NBA stats API wrapper (game logs, player stats, schedules) |

#### Prediction Models

| Package | Version | Purpose |
|---------|---------|---------|
| hmmlearn | >= 0.3 | Hidden Markov Models (Tier 1) |
| filterpy | >= 1.4 | Kalman filters (Tier 1) |
| lightgbm | >= 4.0 | LightGBM gradient boosting (Tier 1) |
| catboost | >= 1.2 | CatBoost gradient boosting (Tier 1) |
| networkx | >= 3.0 | PageRank and HITS graph analysis (Tier 1) |

#### Neural Networks

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >= 2.0 | MLP + LSTM neural networks (Tier 1). CPU-only by default |

#### Optional (Enhanced Features)

| Package | Purpose |
|---------|---------|
| nolds | Lyapunov exponents, Hurst exponent (chaos theory metrics) |
| PyWavelets | Wavelet transforms for signal processing model |
| openmeteo-requests | Weather data helper (free, no API key) |

### Install Everything

```bash
pip install -r requirements.txt

# CPU-only PyTorch (default, smaller download)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# GPU PyTorch (CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Disclaimer

This software is provided for **educational and research purposes only**. It is not financial advice. Sports prediction models are inherently uncertain and past performance does not guarantee future results.

- No prediction model can consistently beat efficient markets over the long term
- Always practice responsible bankroll management
- Never risk money you cannot afford to lose
- Verify all predictions independently before making any financial decisions
- The authors assume no liability for financial losses incurred through use of this software

---

## License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
