# SharpStack-NBA -- 35-Model Mega-Ensemble for NBA Game Prediction

A production-grade NBA game prediction system combining a deeply-tuned Elo rating engine with 35 independent statistical, machine learning, and physics-inspired models. All 35 model outputs feed a meta-learner (Ridge/Logistic/XGBoost) that produces a single calibrated win probability for every game. Includes a full Predicts $1 contract trading ledger with mark-to-market, auto-resolve, Kelly criterion sizing, and P&L tracking.

**30 NBA teams. 35 models. 90+ tunable parameters (24 Elo + 58 per-model). 68.74% accuracy on 1,158 games tested.**

Every data source is **completely free** -- no paid APIs, no subscriptions, no keys required for core functionality. The system pulls game logs and player stats from the official NBA API (`nba_api`), injury reports from ESPN, weather from Open-Meteo, and optional betting odds from The Odds API (free tier, 500 requests/month).

The interactive CLI handles the entire workflow: download data, build ratings, backtest, optimize parameters across 7 phases, validate with 16 statistical tests, generate daily prediction tables, log trades, and track P&L -- all from a single `python main.py` session.

---

## Quick Start

### Install

```bash
git clone https://github.com/JerkyJesse/SharpStack-NBA.git
cd SharpStack-NBA
pip install -r requirements.txt
python main.py
```

### First-Time Workflow

```
1. Launch              python main.py
                       (auto-downloads 2+ seasons of NBA games, player stats, injuries)
                       (auto-runs baseline backtest, fits Platt calibration scaler)

2. Check ratings       all            -- see all 30 teams ranked by Elo
                       settings       -- verify K-factor, home court advantage, etc.
                       injuries       -- review current injury impact by team

3. Make a prediction   Lakers         -- type any team name (fuzzy match)
                       -> Celtics     -- enter opponent
                       -> a           -- who is home? (a=first team, b=second, n=neutral)
                       -> See calibrated win probability with injury adjustments

4. Log a trade         y              -- after seeing prediction
                       -> 5           -- number of $1 contracts
                       -> 0.62        -- price per contract
                       -> notes       -- optional notes

5. Run mega-ensemble   mega           -- 35-model ensemble backtest
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

## The 35 Models

Every model produces an independent win probability for each game. The meta-learner stacks them into one final prediction.

### Tier 0 -- Core (Always On)

| # | Model | Year | Method | NBA-Specific Description |
|---|-------|------|--------|--------------------------|
| 1 | **Elo** | 1960 | Rating system | 24+ adjusters: home court, player strength (composite of PTS/REB/AST), rest days, travel fatigue (timezone crossings), pace mismatch, altitude (Denver/Utah), injuries, SOS, back-to-back penalty, playoff HCA reduction, division familiarity, scoring consistency, mean reversion, season phase |
| 2 | **XGBoost** | 2016 | Gradient boosting | 31 rolling features per team (10-game window via TeamTracker): Elo prob, Elo diff, player diff, PPG, PAPG, win%, margins, rest days, Pythagorean expectation (exp=14), streaks, consistency, trend. 80% Elo / 20% XGBoost blend (default). Walk-forward retrained every 50 games |

### Tier 1 -- Proven Models

| # | Model | Year | Method | NBA-Specific Description |
|---|-------|------|--------|--------------------------|
| 3 | **HMM** | 1966 | Hidden Markov Model | Detects hot/cold team states from observable game outcomes. NBA teams swing between performance regimes mid-season due to injuries, trades, and lineup experiments. 2-4 hidden states capture these transitions |
| 4 | **Kalman** | 1960 | Kalman Filter | Recursive Bayesian estimation of true team strength. Separates signal (real ability) from noise (high game-to-game variance in NBA scoring). Process noise and measurement noise are tunable |
| 5 | **PageRank** | 1998 | Network analysis | Builds directed win graph across the 30-team NBA, applies PageRank + HITS authority scores. Captures transitive strength: beating teams that beat good teams matters. Temporal decay weights recent games more |
| 6 | **LightGBM** | 2017 | Leaf-wise boosting | Microsoft's gradient boosting with leaf-wise tree growth. Uses same 31-feature set as XGBoost for ensemble diversity. Often more accurate on tabular NBA stat data |
| 7 | **CatBoost** | 2017 | Ordered boosting | Yandex's gradient boosting with ordered target statistics (reduces prediction shift). Third independent GBM vote alongside XGBoost and LightGBM |
| 8 | **MLP** | 1986 | Neural network | Multi-layer perceptron with dropout and batch normalization. Captures non-linear feature interactions (e.g., rest x travel x altitude compounding) that tree models miss. PyTorch implementation |
| 9 | **LSTM** | 1997 | Recurrent neural net | Long Short-Term Memory network for sequential game patterns. Models temporal dependencies across a team's 82-game season arc. Captures momentum shifts after trades and lineup changes |

### Tier 2 -- Exotic / Physics-Inspired

| # | Model | Year | Method | NBA-Specific Description |
|---|-------|------|--------|--------------------------|
| 10 | **GARCH** | 1986 | Volatility modeling | Models time-varying performance variance. High-volatility NBA teams (inconsistent scoring, young rosters) are harder to predict -- GARCH captures this uncertainty |
| 11 | **Fourier** | 1807 | Cycle detection | Fourier transforms + wavelet decomposition on scoring time series. Detects periodic patterns in NBA scheduling (weekly rhythms, monthly form cycles, All-Star break effects) |
| 12 | **Survival** | 1972 | Hazard analysis | Cox proportional hazards on streak durations. Models "time until win streak ends" -- in the NBA, a 7-game win streak's hazard rate increases with length. Captures regression to the mean |
| 13 | **Copula** | 1959 | Joint dependency | Models the joint distribution of offensive and defensive performance. Elite offense + poor defense behaves differently than the marginals suggest. Captures asymmetric dependencies in NBA team profiles |

### Tier 3 -- Information & Physics

| # | Model | Year | Method | NBA-Specific Description |
|---|-------|------|--------|--------------------------|
| 14 | **Info Theory** | 1948 | Shannon entropy | Shannon entropy of scoring distribution + KL divergence between team pairs. Low-entropy teams (consistent scorers) are predictable; high KL divergence means style mismatch (e.g., pace-and-space vs grind-it-out) |
| 15 | **Momentum** | 1687 | Newtonian physics | Models team form as physical momentum (mass x velocity). Friction coefficient controls decay. A team riding a 7-game win streak has high momentum; friction bleeds it off as regression approaches |
| 16 | **Markov** | 1906 | Transition matrices | Builds state transition matrix from game outcomes. States: blowout-win, close-win, close-loss, blowout-loss. Predicts next-game state from current state distribution. NBA's high game volume makes transitions statistically robust |
| 17 | **Clustering** | 1957 | k-Means archetypes | Clusters the 30 NBA teams into archetypes (e.g., defensive grinders, pace-and-space offenses, balanced contenders). Matchup probabilities are archetype-vs-archetype historical records |
| 18 | **Game Theory** | 1950 | Nash equilibrium | Models games as strategic interactions. Computes mixed-strategy Nash equilibria for style matchups. Identifies exploitable tendencies (e.g., a team that always switches defensively vs one that exploits isolation plays) |

### Tier 4 -- Classical Rating Systems

| # | Model | Year | Method | NBA-Specific Description |
|---|-------|------|--------|--------------------------|
| 19 | **Poisson** | 1898 | Score distributions | Dixon-Coles Poisson model adapted for basketball's high-scoring environment. Estimates team attack/defense parameters, simulates score probabilities. NBA scores (90-130 range) use a scaled Poisson distribution |
| 20 | **Glicko-2** | 2001 | Uncertainty-aware | Mark Glickman's rating system with rating deviation (RD) and volatility. Teams with few recent games (e.g., post-All-Star break) have high RD = less certain predictions. More principled than raw Elo |
| 21 | **Bradley-Terry** | 1952 | Paired comparison | Maximum likelihood estimation of team strengths from head-to-head results. Recency decay weights recent matchups more heavily. Clean mathematical framework across the 30-team league |
| 22 | **Monte Carlo** | 1946 | Stochastic simulation | 3,000 game simulations per prediction using historical NBA scoring distributions. Produces full probability distributions, not just point estimates. Captures tail-risk upsets |
| 23 | **Random Forest** | 2001 | Bagging ensemble | Hundreds of decorrelated decision trees. Provides diversity against the GBM models (XGBoost, LightGBM, CatBoost) through bootstrap aggregating. Different inductive bias than boosting |

### Tier 5 -- Classical Sports Models

| # | Model | Year | Method | NBA-Specific Description |
|---|-------|------|--------|--------------------------|
| 24 | **SRS** | ~1980 | Simple Rating System | Margin-of-victory plus strength-of-schedule. Iterative solve: team_rating = avg_margin + avg_opponent_rating. The foundation of many NBA power ranking systems (Basketball-Reference uses SRS) |
| 25 | **Colley** | 2001 | Bias-free ranking | Colley Matrix method: solves Ax=b where A encodes the schedule and b encodes results. Produces rankings with no bias toward conference strength or schedule difficulty. Used in BCS-era college rankings |
| 26 | **Log5** | ~1981 | Bill James formula | Log5 head-to-head formula: P(A beats B) = (pA - pA*pB) / (pA + pB - 2*pA*pB). Uses team win percentages. Simple but remarkably effective NBA baseline |
| 27 | **PythagenPat** | 2005 | Dynamic exponent | Pythagorean expectation with dynamically computed exponent. NBA base exponent is 14 (Daryl Morey's research). Converts points scored/allowed to expected win%. Higher exponent than any other sport because basketball outcomes are relatively deterministic |
| 28 | **Exp Smoothing** | 1957 | Trend tracking | Exponential smoothing (Holt-Winters family) on team performance metrics. Separates level, trend, and seasonality components. Adapts faster than simple moving averages to mid-season roster changes |
| 29 | **Mean Reversion** | ~1990 | Bollinger bands | Tracks team performance relative to historical mean. When a team's rolling metrics exceed 2 standard deviations (Bollinger band breakout), regression is likely. Useful for detecting unsustainable hot/cold streaks |

### Tier 6 -- Data Enrichment

| # | Model | Year | Method | NBA-Specific Description |
|---|-------|------|--------|--------------------------|
| 30 | **Weather** | -- | Environmental | Temperature, humidity, wind impact. Limited applicability for NBA (all 30 teams play in indoor arenas) but included for completeness and consistency with multi-sport framework. Free via Open-Meteo |
| 31 | **Odds** | -- | Market consensus | Ingests live betting lines from The Odds API (off by default). Tracks CLV (Closing Line Value). Market odds represent the wisdom of crowds -- a powerful independent signal for NBA |

### Tier 7 -- Novel / Experimental

| # | Model | Year | Method | NBA-Specific Description |
|---|-------|------|--------|--------------------------|
| 32 | **SVM** | 2026 | Support Vector Machine | RBF kernel with Platt scaling, maximum-margin classifier |
| 33 | **Fibonacci** | 2026 | Fibonacci Retracement | EMA-smoothed performance swings with support/resistance levels |
| 34 | **EVT** | 2026 | Extreme Value Theory | Generalized Pareto distribution for tail risk analysis |
| 35 | **Benford** | 2026 | Benford's Law | Chi-squared scoring pattern anomaly detection |

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
    |  31 features: PPG, PAPG, win%, margins, rest,          |
    |  Pythagorean exp (exp=14), streaks, consistency, trend  |
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
                  |    35 probability columns   |
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
                  |  Logistic regression on    |
                  |  logit(raw_prob)           |
                  |  Fixes over/under-conf.    |
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

Every parameter in this system was chosen for basketball-specific reasons.

### K-Factor = 8.23

The NBA regular season has 82 games per team (1,230 total). A moderate K-factor balances stability with responsiveness in such a long season. Compare:
- NFL K ~ 20 (only 17 games, each one matters enormously)
- MLB K ~ 1.0 (162 games, very low K keeps ratings stable)
- NBA K ~ 8.23 (82 games, balanced between NFL and MLB)

K = 8.23 was found via genetic optimization (differential evolution) over multiple seasons. Higher K causes ratings to whipsaw after upsets; lower K makes them too sluggish to capture mid-season improvement from trades and lineup changes.

### Home Court Advantage = 34.0 Elo Points (~58% Home Win Rate)

NBA home court advantage translates to roughly 3 points on the spread and approximately 58% home team win rate in recent seasons (trending down from the historical ~60%). The 34.0 Elo point value maps to this through the Elo expected score formula:

```
E = 1 / (1 + 10^(-34.0/400)) = ~0.549
```

Combined with other home-related factors (rest, travel, altitude), the effective home win rate reaches the observed ~58%.

### Pace Factor = 35.0

Unique to basketball among the four sports. The pace factor models scoring tempo mismatch:
- Estimated pace = (points_for + points_against) / 2 per game
- League average pace is approximately 112 points per game
- When a fast-paced team (e.g., 118 PPG combined) faces a slow-paced team (e.g., 106 PPG combined), the slower team gets a small Elo bonus (they control tempo)
- The pace differential adjustment: `adj = -pace_factor * pace_diff / 10.0`

This captures the NBA reality that tempo control is a genuine strategic advantage. Grind-it-out defensive teams benefit from slowing pace-and-space offenses.

### Back-to-Back Penalty

The NBA is the sport where back-to-back games matter most. Players cover enormous court distances at high intensity, and recovery time is critical. Teams playing their second game in two nights show measurably worse performance:
- Historically ~3-5% lower win rate on the second night of a B2B
- Effect is amplified on road B2Bs (travel + fatigue compounding)
- The `b2b_penalty` parameter reduces Elo expectations for B2B teams
- With 82 games in roughly 170 days, B2B situations occur frequently throughout the NBA season

### Altitude Factor -- Denver & Utah

Ball Arena in Denver sits at 5,280 feet (1,609 meters) and Delta Center in Salt Lake City sits at 4,226 feet (1,288 meters). Visiting teams experience measurably reduced aerobic capacity, affecting:
- Fast-break execution and transition defense
- Late-game stamina and free-throw shooting under fatigue
- Back-to-back performance when altitude is the second game

The `altitude_factor` parameter gives Denver and Utah a small Elo bonus at home (scaled by relative elevation above a 4,000 ft threshold), similar to the well-documented effect in other sports.

### 10-Game Rolling Window

The `TeamTracker` uses a 10-game window for all rolling statistics:
- 10 games = roughly 2-3 weeks of the NBA schedule
- Large enough to smooth single-game variance (NBA scores range from ~90 to ~140)
- Small enough to capture genuine form changes (injuries, trades, lineup experiments)
- NBA teams play 3-4 games per week, so 10 games reflects real-time form

### Pythagorean Exponent = 14

The Pythagorean win expectation formula:

```
Expected Win% = PtsFor^exp / (PtsFor^exp + PtsAgainst^exp)
```

The exponent of 14 comes from Daryl Morey's basketball-specific research (later refined by Dean Oliver). Basketball's high-scoring nature and relatively low variance in final scores justify a much higher exponent than baseball (1.83) or football (~2.37). At exp=14, small scoring advantages translate to large win probability differences, reflecting basketball's relatively deterministic outcomes.

### Playoff Home Court Advantage Reduction

The `playoff_hca_factor` (default 0.57) multiplies the regular-season home court advantage during playoff games. Playoff home advantage is reduced because:
- Teams prepare more extensively for specific opponents in a 7-game series
- Travel fatigue is reduced (same two cities)
- Better teams disproportionately have home court, compressing the effective advantage

Playoff detection: month=4 (after April 15), month=5, month=6.

### No Park Factor, No Pitcher Concepts

Unlike MLB, basketball has no park factors (all courts are the same dimensions), no starting pitcher matchup (it is a full-team game), no bullpen management, and no interleague play distinction. The `player_boost` parameter covers all NBA players via a composite score (PTS + REB + AST + defensive metrics) rather than separating starters from bench players the way MLB separates pitchers from position players.

### Season Detection: Cross-Year Calendar

The NBA season runs October through June, crossing the calendar year boundary:

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

The `division_factor` parameter reduces prediction confidence for division matchups where teams are highly familiar with each other's tendencies (4 games per year against division rivals), making outcomes harder to predict.

### Travel Factor = 28.2

NBA travel spans the full continental US and Toronto. The system tracks timezone crossings between consecutive games and applies an Elo penalty per timezone crossed:

```
travel_adj = -travel_factor * abs(prev_timezone - curr_timezone)
```

A team flying from Portland (UTC-8) to Miami (UTC-5) crosses 3 timezones, receiving a penalty of -84.6 Elo points. This captures the documented effect of circadian disruption on NBA performance, especially for West-to-East travel.

---

## Complete Command Reference

### Predictions & Data

| Command | Description | Time |
|---------|-------------|------|
| `<team name>` | Start prediction for any team. Fuzzy match: `Lakers`, `celtics`, `GSW`, `knicks`, `celt`, `warriors` all work | instant |
| `all` | Show all 30 NBA teams ranked by current Elo rating | instant |
| `players` | Show top scorers league-wide (composite of PTS, REB, AST, etc.) | instant |
| `injuries` | Show current injury report with team-level impact estimates | instant |
| `injuries set <team> <player1>, <player2>` | Manually mark players as OUT for a team | instant |
| `refresh` | Force re-download of all cached data (games, players, ratings, scaler) and rebuild model | 30-60s |
| `today` / `html` / `blog` / `blogger` | Generate HTML prediction table for all of today's NBA games | 5-10s |
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
| `mega` | Run full 35-model mega-ensemble backtest | 3-10min |
| `mega optimize` / `mega opt` | 7-phase per-model exhaustive mega-ensemble optimization (54 hyperparameters) | 1-4hr |
| `mega tune` | Per-model solo optimization (Phase 1 only) | 15-30min |
| `mega tournament` | Head-to-head model tournament (Phase 2 only) | 15-30min |
| `mega quick` | Quick grid search (Phase 1 only) | 15-30min |
| `mega ablation` | Single-model ablation study -- tests each model's individual contribution | 30-60min |
| `mega models` | Show all 35 models with ON/OFF status, organized by tier | instant |
| `mega on <model>` | Enable a specific model (e.g., `mega on lstm`) | instant |
| `mega off <model>` | Disable a specific model | instant |
| `mega on all` | Enable all 35 models | instant |
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
| `set k=8.23` | Set Elo K-factor | instant |
| `set home=34` | Set home court advantage | instant |
| `set boost=35` | Set player strength boost | instant |
| `set rest=25` | Set rest days factor | instant |
| `set b2b=18` | Set back-to-back penalty | instant |
| `set travel=28.2` | Set travel fatigue factor | instant |
| `set pace=35` | Set pace mismatch factor | instant |
| `set sos=5` | Set strength of schedule weight | instant |
| `set streak=5` | Set win/loss streak momentum | instant |
| `set playoff=0.57` | Set playoff home court advantage multiplier | instant |
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

### Elo Parameters (24 Primary)

Parameters are set with `set <param>=<value>` or `set <alias>=<value>`. After any change, run `backtest` to refit the Platt scaler.

#### Core

| Parameter | Aliases | Type | Default | Range | Description |
|-----------|---------|------|---------|-------|-------------|
| `k` | `k_factor` | float | 8.23 | 1-30 | Elo K-factor (learning rate per game). Lower = more stable, higher = more reactive. 82-game NBA season needs moderate K |
| `base_rating` | `base`, `rating` | float | 1500.0 | 1000-2000 | Starting Elo rating for all 30 teams |
| `home_adv` | `home`, `hca`, `home_advantage` | float | 34.0 | 0-100 | Home court advantage in Elo points. 34.0 maps to ~55% home win rate before other factors |
| `use_mov` | `mov`, `margin` | bool | true | true/false | Use margin of victory adjustment (log-scaled). Critical for NBA where blowouts are common |

#### Player Strength

| Parameter | Aliases | Type | Default | Range | Description |
|-----------|---------|------|---------|-------|-------------|
| `player_boost` | `boost`, `player` | float | 35.0 | 0-100 | Team-level composite player strength. Computed from PTS + REB + AST + defensive metrics for all rostered players. No pitcher/starter distinction -- NBA is a full-team game |

#### Rest & Schedule

| Parameter | Aliases | Type | Default | Range | Description |
|-----------|---------|------|---------|-------|-------------|
| `rest_factor` | `rest` | float | 25.0 | 0-80 | Rest days advantage factor. B2B = -rest_factor, normal = 0, extra rest = +bonus |
| `rest_advantage_cap` | `restcap`, `rest_cap` | float | 2.97 | 0-10 | Maximum rest days counted. Prevents 5-day rest from being worth 5x a 1-day rest |
| `b2b_penalty` | `b2b`, `back_to_back` | float | 0.0 | 0-50 | Additional back-to-back penalty beyond rest_factor. Critical for NBA's compressed schedule |
| `road_trip_factor` | `roadtrip`, `road_trip` | float | 3.47 | 0-30 | Extended road trip penalty (3+ consecutive away games). NBA road trips can span 4-5 games |
| `homestand_factor` | `homestand` | float | 0.0 | 0-30 | Extended homestand bonus (3+ consecutive home games) |

#### Travel & Venue

| Parameter | Aliases | Type | Default | Range | Description |
|-----------|---------|------|---------|-------|-------------|
| `travel_factor` | `travel` | float | 28.2 | 0-80 | Elo penalty per timezone crossed. East-West travel across the US (up to 3 zones) is significant |
| `altitude_factor` | `altitude`, `alt` | float | 0.0 | 0-50 | Altitude bonus multiplier for Denver (5,280 ft) and Utah (4,226 ft) home games |

#### Form & Momentum

| Parameter | Aliases | Type | Default | Range | Description |
|-----------|---------|------|---------|-------|-------------|
| `form_weight` | `form` | float | 10.0 | 0-50 | Recent form weight based on last 10 games win%. Hot/cold streaks matter in the NBA |
| `win_streak_factor` | `streak`, `win_streak` | float | 0.0 | 0-20 | Win/loss streak momentum factor |
| `mean_reversion` | `reversion`, `regress` | float | 0.0 | 0-20 | Mean reversion after extreme results (blowout correction) |
| `season_regress` | `season_regression`, `regress_pct` | float | 0.33 | 0-1.0 | Season boundary regression fraction. 0.33 = pull ratings 33% toward league mean between seasons |

#### Matchup Adjustments

| Parameter | Aliases | Type | Default | Range | Description |
|-----------|---------|------|---------|-------|-------------|
| `sos_factor` | `sos`, `strength_of_schedule` | float | 0.0 | 0-20 | Strength of schedule weight. Teams facing tougher opponents get battle-tested bonus |
| `division_factor` | `division`, `div` | float | 10.0 | 0-30 | Divisional game confidence reducer. Division rivals play 4 times/year, increasing unpredictability |

#### Scoring & Pace

| Parameter | Aliases | Type | Default | Range | Description |
|-----------|---------|------|---------|-------|-------------|
| `pace_factor` | `pace`, `tempo` | float | 35.0 | 0-80 | Pace mismatch adjustment. Slower team gets bonus when facing a faster team (controls tempo). NBA-unique feature |
| `scoring_consistency_factor` | `consistency`, `scoring_consistency` | float | 0.0 | 0-20 | Penalty for volatile scoring teams (high game-to-game point variance) |

#### Season & Playoffs

| Parameter | Aliases | Type | Default | Range | Description |
|-----------|---------|------|---------|-------|-------------|
| `playoff_hca_factor` | `playoff`, `playoff_hca`, `postseason` | float | 0.57 | 0-1.0 | Playoff home court advantage multiplier. 0.57 = home court is 57% of regular-season value in the playoffs |
| `season_phase_factor` | `phase`, `season_phase` | float | 0.0 | 0-10 | Early-season dampener. Reduces prediction confidence when Elo ratings are still immature (first 20 games) |

#### Account & Trading

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `kelly_fraction` | `kelly` | special | 0.50 | Kelly criterion fraction. Accepts: `quarter`, `half`, `full`, or any decimal 0-1 |
| `starting_balance` | `balance`, `bankroll` | float | 0.0 | Starting account balance for P&L tracking |
| `autoresolve_enabled` | `autoresolve`, `auto_resolve` | bool | false | Auto-resolve finished trades on startup |

### Mega-Ensemble Parameters (54 Per-Model Hyperparameters)

Set with `mega set <param>=<value>`. Changes take effect on next `mega` run.

#### Meta-Learner Controls

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `max_adj` | `maxadj`, `adj`, `adjustment` | float | 0.20 | Max meta-learner adjustment (+/- probability). Controls how far the ensemble can deviate from Elo |
| `meta_model` | `meta`, `metalearner`, `stacker` | str | ridge | Meta-learner type: `ridge`, `logistic`, or `xgboost` |
| `retrain_every` | `retrain`, `retrain_interval` | int | 50 | Retrain meta-learner every N games during walk-forward |
| `min_train` | `mintrain`, `min_games`, `warmup` | int | 200 | Games before meta-learner starts predicting (warmup period) |
| `window` | `rolling_window`, `feat_window` | int | 10 | Rolling feature window size in games |

#### Per-Model Hyperparameters

| Parameter | Model | Type | Default | Search Range | Description |
|-----------|-------|------|---------|-------------|-------------|
| `kalman_process_noise` | Kalman | float | 0.3 | 0.1-2.0 | Kalman filter process noise |
| `kalman_measurement_noise` | Kalman | float | 2.5 | 1.5-5.0 | Kalman filter measurement noise |
| `hmm_states` | HMM | int | 3 | 2-5 | Number of hidden states (hot/cold regimes) |
| `hmm_covariance_type` | HMM | str | diag | diag/full/spherical | HMM covariance structure |
| `network_decay` | PageRank | float | 0.95 | 0.90-0.99 | PageRank temporal decay |
| `pagerank_damping` | PageRank | float | 0.85 | 0.75-0.95 | PageRank damping factor |
| `momentum_friction` | Momentum | float | 0.05 | 0.02-0.15 | Momentum friction coefficient |
| `n_clusters` | Clustering | int | 4 | 3-8 | Number of team archetype clusters |
| `glicko_initial_rd` | Glicko-2 | float | 200 | 100-400 | Initial rating deviation |
| `bt_decay` | Bradley-Terry | float | 0.99 | 0.95-1.0 | Recency decay factor |
| `mc_simulations` | Monte Carlo | int | 2000 | 1000-5000 | Simulations per game prediction |
| `garch_alpha` | GARCH | float | 0.10 | 0.05-0.30 | GARCH alpha (short-term shock weight) |
| `garch_beta` | GARCH | float | 0.80 | 0.60-0.90 | GARCH beta (persistence) |
| `poisson_home_adv` | Poisson | float | 1.2 | 1.0-1.5 | Poisson home scoring multiplier |
| `poisson_decay` | Poisson | float | 0.98 | 0.95-1.0 | Poisson recency decay |
| `markov_n_states` | Markov | int | 4 | 3-6 | Markov chain state count |
| `gt_ema_alpha` | Game Theory | float | 0.05 | 0.02-0.15 | Game theory EMA smoothing |
| `it_n_bins` | Info Theory | int | 5 | 3-10 | Information theory histogram bins |

#### GBM Model Hyperparameters

| Parameter | Model | Type | Default | Search Range |
|-----------|-------|------|---------|-------------|
| `lgbm_num_leaves` | LightGBM | int | 31 | 8-63 |
| `lgbm_learning_rate` | LightGBM | float | 0.03 | 0.01-0.10 |
| `lgbm_n_rounds` | LightGBM | int | 300 | 100-500 |
| `lgbm_lambda_l1` | LightGBM | float | 0.1 | 0.0-2.0 |
| `lgbm_lambda_l2` | LightGBM | float | 0.1 | 0.0-2.0 |
| `cb_iterations` | CatBoost | int | 300 | 100-500 |
| `cb_learning_rate` | CatBoost | float | 0.05 | 0.01-0.10 |
| `cb_depth` | CatBoost | int | 6 | 3-8 |
| `cb_l2_leaf_reg` | CatBoost | float | 3.0 | 1.0-10.0 |

#### Neural Network Hyperparameters

| Parameter | Model | Type | Default | Search Range |
|-----------|-------|------|---------|-------------|
| `mlp_hidden` | MLP | str | 64,32 | 32,16 to 128,64,32 |
| `mlp_lr` | MLP | float | 0.001 | 0.0005-0.01 |
| `mlp_epochs` | MLP | int | 100 | 50-200 |
| `mlp_dropout` | MLP | float | 0.3 | 0.1-0.5 |
| `lstm_hidden_dim` | LSTM | int | 64 | 32-128 |
| `lstm_n_layers` | LSTM | int | 2 | 1-3 |
| `lstm_lr` | LSTM | float | 0.001 | 0.0005-0.003 |
| `lstm_epochs` | LSTM | int | 80 | 40-120 |
| `lstm_dropout` | LSTM | float | 0.3 | 0.1-0.5 |

#### Meta-XGBoost Hyperparameters (when meta_model=xgboost)

| Parameter | Type | Default | Search Range |
|-----------|------|---------|-------------|
| `meta_xgb_max_depth` | int | 4 | 3-6 |
| `meta_xgb_eta` | float | 0.05 | 0.01-0.12 |
| `meta_xgb_subsample` | float | 0.8 | 0.6-0.9 |
| `meta_xgb_min_child_weight` | int | 5 | 3-12 |
| `meta_xgb_alpha` | float | 0.5 | 0.1-2.0 |
| `meta_xgb_lambda` | float | 1.0 | 0.5-5.0 |
| `meta_xgb_num_boost_round` | int | 200 | 100-500 |

---

## Data Sources

All data sources are **completely free** -- no paid APIs, no subscriptions required.

| API | Data | Key Required | Free Tier | Cache Duration |
|-----|------|-------------|-----------|----------------|
| **nba_api** | Game logs, box scores, player stats, team info | No key needed | Unlimited | 6 hours (games), 24 hours (players) |
| **ESPN JSON** | Injury reports (status, return dates) | No key needed | Unlimited | 4 hours |
| **Open-Meteo** | Temperature, humidity, wind, precipitation | No key needed | Unlimited | 2 hours (game days) |
| **The Odds API** | Moneyline, spread, totals from 15+ sportsbooks | Free key needed | 500 req/month | 15 min (game days) |

### Smart Caching

The caching system adapts to the NBA calendar. During the active season (October-June), data is refreshed more frequently. During the offseason (July-September), caches last much longer.

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

Cache staleness is checked by `config.is_cache_stale()`. Files under 500 bytes are always considered stale (likely corrupt or empty). Force refresh with the `refresh` command.

### The Odds API Setup (Optional)

1. Sign up at [the-odds-api.com](https://the-odds-api.com)
2. Copy your API key from the dashboard
3. Set the environment variable:

```bash
# Windows
set ODDS_API_KEY=your_key_here

# Linux/Mac
export ODDS_API_KEY=your_key_here
```

4. Verify: `odds` command in the CLI should show today's lines

### nba_api Notes

The `nba_api` package wraps the official NBA stats API (stats.nba.com). It provides game logs for current and recent seasons, player statistics (per-game and advanced), and team schedule data. No authentication is needed, but the NBA API has rate limits. The smart cache system minimizes API calls by only fetching when data is likely to have changed.

---

## Optimization System

### Elo Optimization (6 Methods)

#### Grid Search (`grid`)
Cartesian product sweep over 7 parameters. Interactive prompts let you set min/max/step for each dimension. Best for initial exploration.

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

#### Coordinate Descent (`singleopt`)
Optimizes one parameter at a time for accuracy (not LogLoss/Brier):

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

### Recommended Optimization Workflow

```
1. backtest              -- establish baseline (24 Elo params)
2. autoopt               -- quick automatic Elo optimization (15-30 min)
3. backtest              -- verify improvement
4. mega                  -- run mega-ensemble with current Elo settings
5. mega tune             -- per-model solo optimization
6. mega tournament       -- head-to-head model comparison
7. mega optimize         -- full 7-phase optimization (54 hyperparams, 1-4 hr)
8. mega ablation         -- prune underperforming models
9. mega                  -- verify final ensemble performance
10. superopt             -- (optional) exhaustive Elo re-optimization
```

---

## 6-Phase Model Validation Workflow

A 6-phase process to thoroughly validate and tune the prediction model. Each phase builds on the previous one. Run phases in order.

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

### Kelly Criterion Position Sizing

The `kelly` command simulates fractional Kelly on historical predictions:
- Default quarter-Kelly (kelly_fraction = 0.50, then halved for safety)
- Reports final bankroll, return %, max drawdown, annualized Sharpe, win rate
- Connects model quality directly to P&L

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

## File Structure

```
NBA/
|
|-- main.py                    Main CLI entry point, dispatch() routes all 60+ commands
|-- config.py                  Constants, 30 team abbreviations, settings I/O, season detection
|-- elo_model.py               NBAElo class: Elo ratings with 24+ adjusters
|-- build_model.py             Constructs NBAElo from settings + game CSV, applies season regression
|-- backtest.py                Walk-forward backtest, grid/genetic/bayesian optimizers, all validation tests
|-- enhanced_model.py          XGBoost ensemble: TeamTracker (10-game window), 31 features, walk-forward
|-- platt.py                   Platt scaling + isotonic regression + beta calibration
|-- metrics.py                 ECE, MCE, BSS, log loss, Brier score, conformal prediction
|-- single_param_opt.py        Coordinate descent optimizer (one param at a time)
|
|-- data_games.py              Download NBA game logs from nba_api with smart caching
|-- data_players.py            Download player stats, compute composite scores (PTS+REB+AST)
|-- injuries.py                ESPN injury report fetcher + team impact calculator
|-- cache_utils.py             Smart season-aware cache staleness logic
|
|-- mega_backtest.py           35-model mega-ensemble backtest engine (multithreaded)
|-- mega_config.py             Model registry (35 models), on/off switches, 54 hyperparameters
|-- mega_optimizer.py          7-phase mega-ensemble optimization
|-- mega_predictor.py          Live mega-ensemble predictor (replays history, loads meta-learner)
|-- meta_learner.py            Meta-learner: Ridge, Logistic, or XGBoost stacker
|
|-- hmm_model.py               Hidden Markov Model (hot/cold team states)
|-- kalman_model.py            Kalman filter (true strength estimation)
|-- network_model.py           PageRank + HITS (win graph analysis via NetworkX)
|-- gbm_models.py              LightGBM + CatBoost gradient boosting
|-- nn_models.py               MLP + LSTM neural networks (PyTorch)
|-- random_forest_model.py     Random Forest bagging ensemble
|-- glicko_model.py            Glicko-2 uncertainty-aware rating system
|-- bradley_terry_model.py     Bradley-Terry MLE paired comparison
|-- poisson_model.py           Dixon-Coles Poisson model (adapted for high-scoring NBA)
|-- monte_carlo_model.py       Monte Carlo simulation (2000+ sims/game)
|-- clustering_model.py        k-Means team archetype clustering
|-- game_theory_model.py       Nash equilibrium + style matchups
|-- markov_chain_model.py      Markov chain transition matrices
|-- information_theory_model.py Shannon entropy + KL divergence
|-- momentum_model.py          Newtonian momentum / inertia
|-- volatility_model.py        GARCH volatility modeling
|-- signal_model.py            Fourier transforms + wavelet cycle detection
|-- survival_model.py          Cox proportional hazards (streak analysis)
|-- copula_model.py            Copula joint dependency modeling
|-- classic_models.py          SRS, Colley, Log5, PythagenPat, ExpSmoothing, MeanReversion
|-- svm_model.py               SVM classifier (RBF kernel + Platt scaling)
|-- fibonacci_model.py         Fibonacci retracement analysis
|-- evt_model.py               Extreme Value Theory tail risk
|-- benford_model.py           Benford's Law anomaly detection
|
|-- odds_tracker.py            The Odds API integration + CLV tracking
|-- weather.py                 Open-Meteo weather data (limited use for indoor NBA)
|-- kalshi.py                  Kalshi market integration
|
|-- predict_ledger.py          $1 contract ledger: buy, sell, mark, resolve, P&L
|-- live_scores.py             Real-time score tracker for open trades (NBA CDN)
|-- auto_resolve.py            Auto-settle finished trades from live final scores
|-- html_generator.py          Generate HTML prediction tables for Blogger/WordPress
|-- help_system.py             CLI help display system
|-- color_helpers.py           Colorama wrapper functions (cok, cerr, cwarn, chi, cdim, cbold)
|-- elo_set_handler.py         Shared handler for 'set param=value' (24+ params with aliases)
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
|   |-- nba_elo_ratings.json         Current team Elo ratings (all 30 teams)
|   |-- nba_platt_scaler.json        Platt calibration scaler
|   |-- nba_isotonic_scaler.json     Isotonic calibration scaler
|   |-- nba_enhanced_model.json      Saved XGBoost model metadata
|   |-- nba_xgb_model.json           XGBoost booster weights
|   |-- nba_mega_settings.json       Mega-ensemble parameters + model switches
|   |-- nba_mega_predictions.csv     Mega-ensemble prediction log
|   |-- nba_meta_learner.json        Trained meta-learner state
|   |-- nba_meta_xgb.json            Meta-learner XGBoost model (when meta=xgboost)
|   |-- nba_recent_games.csv         Cached game data (2+ NBA seasons)
|   |-- nba_player_stats.csv         Cached player statistics
|   |-- nba_advanced_stats.csv       Cached advanced statistics
|   |-- nba_injuries.json            Cached ESPN injury report
|   |-- nba_backtest_predictions.csv Per-game backtest predictions
|   |-- nba_calibration.csv          10-bin calibration table
|   |-- nba_grid_search.csv          Grid search results
|   |-- nba_bayesian_results.csv     Bayesian optimization results
|   |-- nba_genetic_results.csv      Genetic optimization results
|   |-- nba_enhanced_features.npz    Saved feature matrix
|   |-- nba_power_ratings.json       Power ratings cache
|   |-- nba_stacked_scaler.json      Stacked model scaler
|   |-- nba_optimization_log.txt     Optimization run log
|   |-- predicts_lots.csv            Trading ledger (positions, P&L)
|   |-- today_nba_predictions.html   Today's prediction table (HTML)
|   |-- today_nba_predictions.txt    Today's predictions (plain text)
|   |-- tomorrow_nba_predictions.html Tomorrow's prediction table
```

---

## Performance

### Multithreading

The mega-ensemble uses `ThreadPoolExecutor` to train models in parallel:

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

### Smoke Test Results

```
Baseline accuracy:         68.74%  (best of all 4 sports)
Log Loss:                  0.5858
Brier:                     0.2002
Games tested:              1,158
```

---

## Dependencies

### System Requirements

- Python 3.8+ (no walrus operators, no match statements)
- 4 GB RAM minimum (8 GB recommended for mega-ensemble)
- Internet connection for initial data download (subsequent runs use cache)
- Optional: CUDA-capable GPU for accelerated neural network and GBM training

### Core (Required)

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | >= 1.5 | Data manipulation and CSV I/O |
| numpy | >= 1.24 | Numerical computing |
| scipy | >= 1.10 | Optimization (differential evolution, L-BFGS-B), statistics |
| colorama | >= 0.4 | Colored terminal output |
| xgboost | >= 2.0 | Gradient boosting (Tier 0 model + meta-learner option) |
| requests | >= 2.28 | HTTP requests for APIs (ESPN injuries, Odds, Weather) |
| tqdm | >= 4.60 | Progress bars for optimization and backtesting |
| matplotlib | >= 3.7 | P&L charts and visualization |

### NBA Data APIs

| Package | Version | Purpose |
|---------|---------|---------|
| nba_api | >= 1.1 | Official NBA stats API wrapper (game logs, player stats, schedules) |

### Prediction Models

| Package | Version | Purpose |
|---------|---------|---------|
| hmmlearn | >= 0.3 | Hidden Markov Models (Tier 1) |
| filterpy | >= 1.4 | Kalman filters (Tier 1) |
| lightgbm | >= 4.0 | LightGBM gradient boosting (Tier 1) |
| catboost | >= 1.2 | CatBoost gradient boosting (Tier 1) |
| networkx | >= 3.0 | PageRank and HITS graph analysis (Tier 1) |

### Neural Networks

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >= 2.0 | MLP + LSTM neural networks (Tier 1). CPU-only by default |

### Optional (Enhanced Features)

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
