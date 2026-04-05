# NFL Moneyball -- 31-Model Mega-Ensemble

A production-grade NFL game prediction system that fuses 31 independent models -- spanning Elo ratings, gradient boosting, Hidden Markov Models, Kalman filters, PageRank, neural networks, survival analysis, information theory, game theory, and classical football analytics -- into a single calibrated probability through a walk-forward meta-learner. Every model trains on real NFL data pulled from completely free APIs (ESPN public API, nfl_data_py/nflverse play-by-play, ESPN injuries, Open-Meteo weather). The system includes a full Predicts $1 binary contract trading ledger with Kelly criterion position sizing, live score tracking, auto-settlement, and monthly P&L charting. All 17-game-season parameters are tuned through a 7-phase exhaustive optimizer with multithreaded backtesting and optional GPU acceleration.

**32 NFL teams** | **35 models** | **74+ tunable parameters** (20 Elo + 54 per-model) | **7-phase per-model optimizer** | **No paid APIs**

---

## Quick Start

```bash
# 1. Clone and enter directory
cd NFL

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Install PyTorch for neural network models (CPU-only)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4. Launch
python main.py
```

**First-time workflow:**

```
1. System auto-downloads 2 years of NFL game data via ESPN API
2. System auto-downloads passing/rushing/receiving leaders, injury reports
3. Baseline backtest runs automatically (fits Platt calibration scaler)
4. Enter starting balance when prompted (for contract tracking)
5. Type a team name (e.g. "Chiefs") to make your first prediction
6. Run 'mega' for the full 35-model ensemble backtest
7. Run 'mega tune' to solo-test each model's optimal settings
8. Run 'mega optimize' for full 7-phase per-model optimization
```

On startup, the system downloads and caches all required data, builds the Elo model with season regression, runs a baseline backtest with Platt calibration, and drops you into the interactive command loop. No API keys are needed for core functionality -- the ESPN public API, nfl_data_py/nflverse, ESPN injuries, and Open-Meteo weather are all completely free.

---

## The 31 Models

Every model runs independently on the same game-by-game walk-forward loop. Their raw outputs feed into the meta-learner, which produces a single calibrated adjustment bounded by `max_adj`.

### Tier 0 -- Core (Always On)

| # | Model | Year | Method | Description |
|---|-------|------|--------|-------------|
| 1 | **Elo** | 1960 | Paired comparison rating | 24+ adjusters: home field advantage, MOV, rest/bye week, travel, altitude (Denver), form, SOS, divisional rivalry, conference, playoff detection. All 32 NFL teams tracked. K=28.36 tuned for the 17-game season with logarithmic MOV dampening. |
| 2 | **XGBoost** | 2016 | Gradient boosted trees | 31 rolling features per game (win%, Pythagorean expectation, streaks, scoring consistency, trend, rest days, travel). Walk-forward training with 80/20 Elo/XGBoost blend. SHAP feature importance built in. |

### Tier 1 -- Proven Models

| # | Model | Year | Method | Description |
|---|-------|------|--------|-------------|
| 3 | **HMM** | 1966 | Hidden Markov Model | Detects latent hot/cold team states from win/loss sequences. Forward-backward algorithm estimates state probabilities. Captures momentum shifts invisible to pure ratings. |
| 4 | **Kalman** | 1960 | Kalman Filter | Treats true team strength as a hidden state with process noise. Bayesian updates after each game. Provides uncertainty estimates alongside point predictions. |
| 5 | **PageRank** | 1998 | Network analysis | Builds directed win graph, runs PageRank + HITS authority scores. Teams that beat strong teams get more credit. Temporal decay weights recent results. |
| 6 | **LightGBM** | 2017 | Leaf-wise gradient boosting | Microsoft's fast GBM with leaf-wise splits. Handles categorical features natively. Lower memory than XGBoost with comparable accuracy. |
| 7 | **CatBoost** | 2017 | Ordered gradient boosting | Yandex's ordered boosting prevents target leakage during training. Handles categorical features with target statistics. Robust to overfitting. |
| 8 | **MLP** | 1986 | Multi-layer perceptron | PyTorch feedforward neural network with batch normalization and dropout. Learns nonlinear feature interactions that tree models miss. |
| 9 | **LSTM** | 1997 | Long Short-Term Memory | Recurrent neural network that models sequential game patterns. Captures long-range dependencies in team performance trajectories. Off by default (slow). |

### Tier 2 -- Exotic / Physics-Inspired

| # | Model | Year | Method | Description |
|---|-------|------|--------|-------------|
| 10 | **GARCH** | 1986 | Volatility modeling | Generalized AutoRegressive Conditional Heteroskedasticity. Models time-varying volatility in scoring. High-variance teams are harder to predict. |
| 11 | **Fourier** | 1822 | Cycle detection | Fourier transforms + wavelet analysis on scoring time series. Detects periodic patterns and seasonal rhythms in team performance across a 17-game schedule. |
| 12 | **Survival** | 1958 | Hazard modeling | Cox proportional hazards applied to win/loss streaks. Models the probability that a streak ends given its length and covariates. |
| 13 | **Copula** | 1959 | Joint dependency | Models the dependency structure between offensive and defensive performance using copula functions. Captures teams where offense/defense move together vs independently. |

### Tier 3 -- Information & Physics

| # | Model | Year | Method | Description |
|---|-------|------|--------|-------------|
| 14 | **Info Theory** | 1948 | Shannon entropy + KL divergence | Measures predictability of each team's scoring distribution. High-entropy teams are chaotic; low-entropy teams are predictable. KL divergence quantifies matchup asymmetry. |
| 15 | **Momentum** | 1687 | Newtonian mechanics analogy | Treats team strength as a physical object with mass (games played) and velocity (recent trend). Friction coefficient controls decay. Captures inertia in form. |
| 16 | **Markov Chain** | 1906 | Transition matrices | Models sequences of outcomes (W/L/close-W/blowout-W) as Markov transitions. Stationary distribution gives long-run expected state probabilities. |
| 17 | **Clustering** | 1957 | k-Means archetypes | Groups teams into archetypes (e.g., high-offense/low-defense, balanced, defense-dominant). Matchup predictions based on how archetype pairs historically perform. |
| 18 | **Game Theory** | 1950 | Nash equilibrium | Models strategic matchups: power run vs pass-heavy, offense vs defense. Computes Nash equilibrium strategies and style-based advantages. |

### Tier 4 -- Classical Rating Systems

| # | Model | Year | Method | Description |
|---|-------|------|--------|-------------|
| 19 | **Poisson** | 1898 | Dixon-Coles score distribution | Models point scoring as Poisson-distributed. Dixon-Coles correction for low-scoring games. Produces full score probability matrix for each matchup. |
| 20 | **Glicko-2** | 2001 | Uncertainty-aware ratings | Extends Elo with rating deviation (confidence interval) and volatility. Teams with fewer recent games have wider uncertainty. More principled than fixed-K Elo. |
| 21 | **Bradley-Terry** | 1952 | Maximum likelihood paired comparison | MLE estimation of team strengths from pairwise outcomes. Recency-weighted decay ensures recent games matter more. Clean probabilistic framework. |
| 22 | **Monte Carlo** | 1940s | Stochastic simulation | Runs 3,000 game simulations per matchup using historical scoring distributions. Produces win probability from simulation outcomes. |
| 23 | **Random Forest** | 2001 | Bagged decision trees | Ensemble of decorrelated decision trees. Provides diversity to the meta-learner -- different inductive bias from boosted methods. |

### Tier 5 -- Classical Football / Sports Models

| # | Model | Year | Method | Description |
|---|-------|------|--------|-------------|
| 24 | **SRS** | ~1980s | Simple Rating System | Average margin of victory adjusted for strength of schedule. Iterative convergence. The backbone of many newspaper power rankings. |
| 25 | **Colley** | 2001 | Colley Matrix | Bias-free ranking using only wins and losses. Solves a linear system -- no preseason assumptions, no margin of victory. Used by the BCS. |
| 26 | **Log5** | 1981 | Bill James formula | The original sabermetric head-to-head formula: P(A beats B) = (pA - pA*pB) / (pA + pB - 2*pA*pB). Elegant and theoretically grounded. |
| 27 | **Pythagorean** | ~1980s | Pythagorean expected win% | Points-scored / points-allowed formula with exponent ~2.37 for NFL. Converts scoring margins into expected win rates. A cornerstone of football analytics. |
| 28 | **Exp Smoothing** | 1957 | Exponential smoothing | Holt-Winters style smoothing on team performance metrics. Captures level, trend, and seasonality in scoring. Simple but effective trend tracker. |
| 29 | **Mean Reversion** | ~1990s | Bollinger band analog | Identifies teams performing above/below their "true" level using a z-score band approach. Teams far from the mean are expected to regress. |

### Tier 6 -- Data Enrichment

| # | Model | Year | Method | Description |
|---|-------|------|--------|-------------|
| 30 | **Weather** | -- | Environmental impact | Temperature, wind speed, humidity, precipitation probability. Open-Meteo API (free, no key). Adjusts predictions for extreme weather -- critical for outdoor NFL stadiums in winter. Off by default. |
| 31 | **Odds** | -- | Market consensus | Ingests moneyline odds from The Odds API. Closing Line Value (CLV) tracking. Markets are efficient -- odds provide a strong independent signal. Off by default (requires free API key). |

---

## Architecture

```
                         ESPN PUBLIC API           NFL_DATA_PY / NFLVERSE
                        (games, scores)         (EPA, CPOE, success rate)
                             |                         |
                   ESPN INJURIES API             OPEN-METEO WEATHER
                    (IR, Doubtful, Out)          (temp, wind, precip)
                             |                         |
                   +---------+---------+---------------+
                   |                                   |
                   v                                   v
            +------+-------+                  +--------+--------+
            |  DATA LAYER  |                  | ADVANCED STATS  |
            | data_games   |                  | advanced_stats  |
            | data_players |                  | (EPA/CPOE/Refs) |
            +--------------+                  +-----------------+
                   |                                   |
                   +-----------------------------------+
                   |
                   v
  +=====================================+
  |         ELO ENGINE (Tier 0)         |
  |  NFLElo class -- 32 teams           |
  |  24 adjustment factors              |
  |  Season regression (33%)            |
  |  Platt/Isotonic/Beta calibration    |
  +=====================================+
                   |
                   | Elo probability (anchor)
                   |
  +=====================================+
  |    31 BASE MODEL PREDICTIONS        |
  |                                     |
  |  [Tier 0] Elo, XGBoost             |
  |  [Tier 1] HMM, Kalman, PageRank,   |
  |           LightGBM, CatBoost,       |
  |           MLP, LSTM                 |
  |  [Tier 2] GARCH, Fourier,          |
  |           Survival, Copula          |
  |  [Tier 3] InfoTheory, Momentum,    |
  |           Markov, Clustering,       |
  |           GameTheory                |
  |  [Tier 4] Poisson, Glicko, B-T,    |
  |           MonteCarlo, RandomForest  |
  |  [Tier 5] SRS, Colley, Log5,       |
  |           Pythagorean, ExpSmooth,   |
  |           MeanReversion             |
  |  [Tier 6] Weather, Odds            |
  |                                     |
  |  All models run in PARALLEL via     |
  |  ThreadPoolExecutor                 |
  +=====================================+
                   |
                   | Vector of 31 probabilities
                   v
  +=====================================+
  |       META-LEARNER (Stacker)        |
  |                                     |
  |  Ridge / Logistic / XGBoost         |
  |  Walk-forward retrain every N games |
  |  min_train warmup period            |
  |  Trains on base model outputs only  |
  +=====================================+
                   |
                   | Raw adjustment delta
                   v
  +=====================================+
  |   ELO-ANCHORED BOUNDED ADJUSTMENT   |
  |                                     |
  |  final = elo_prob + clamp(          |
  |    meta_adjustment, -max_adj,       |
  |    +max_adj)                        |
  |                                     |
  |  Elo is ALWAYS the anchor.          |
  |  Meta-learner can only nudge the    |
  |  probability within +/- max_adj     |
  |  (default 0.10 = 10 percentage      |
  |   points).                          |
  +=====================================+
                   |
                   v
         FINAL CALIBRATED PROBABILITY
                   |
                   v
       +---------------------+
       |  PREDICTION OUTPUT  |
       |  + Trading Ledger   |
       |  + HTML Export       |
       |  + Live Scores      |
       +---------------------+
```

### Elo-Anchored Bounded Adjustment

The Elo model serves as the anchor probability. The meta-learner (trained on all 31 base model outputs) produces an adjustment that is **clamped** to `+/- max_adj` (default 0.10). This means even if all exotic models disagree with Elo, the final probability can shift at most 10 percentage points. This design prevents catastrophic predictions from untested models while allowing proven signal to improve accuracy.

### Multithreaded Training

All 31 base models run inside a `ThreadPoolExecutor`. On a typical 8-core machine, the mega-ensemble backtest completes 3-5x faster than sequential execution. Each model receives the same game-by-game data and produces an independent probability estimate.

### GPU Acceleration

XGBoost, LightGBM, and CatBoost automatically detect CUDA-capable GPUs. If available, tree construction runs on GPU (`tree_method='gpu_hist'` for XGBoost, `device='gpu'` for LightGBM/CatBoost). PyTorch models (MLP, LSTM) also move to GPU when `torch.cuda.is_available()`. CPU fallback is always automatic and silent.

---

## NFL-Specific Design Choices

Every parameter in this system was chosen with the specific structure of the National Football League in mind. Here is why each default is what it is:

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| **K-factor** | 28.36 | NFL plays only 17 regular season games -- far fewer than MLB (162) or NBA (82). A much higher K means each individual game moves ratings significantly. Every win and loss matters enormously in a short season, so the system must react quickly. |
| **Home advantage** | 25.55 Elo (~53.6%) | NFL home teams historically win about 53-57% of games. 25.55 Elo points in the standard Elo formula yields approximately 53.6% expected win rate. Combined with other home-related factors (rest, SOS, division), the effective home win rate reaches observed levels. |
| **Player scoring weight** | Passing-heavy (24.61) | Quarterback play dominates football outcomes more than any single position in other sports. The player scoring composite weights passing stats (yards, TDs, passer rating) heavily, with rushing and receiving as secondary signals. A QB marked Out costs the team approximately -50 Elo points. |
| **Rest factor** | 0.68 (centered at 7 days) | NFL teams play weekly, so rest is centered at 7 days instead of 1 day like daily sports. Extra rest (bye week = 14 days, Thursday-to-Sunday = 10 days) or short rest (Sunday-to-Thursday = 4 days) creates significant advantages and disadvantages. The rest advantage is capped at 3.32 to prevent extreme values. |
| **B2B penalty** | 5.45 | Thursday Night Football and other short-week games (3-4 days rest vs the standard 7) produce measurably worse performance. The 5.45 Elo point penalty captures the fatigue, reduced preparation time, and increased injury risk of short-turnaround games. |
| **Bye week factor** | 0.0 (disabled) | NFL teams get one bye week per season (a week off with no game). Despite the conventional wisdom that teams play better after a bye, optimization found no statistically significant advantage once rest days are already accounted for by the rest_factor parameter. Currently zeroed out. |
| **Rolling window** | 5 games | Much narrower than MLB's 15-game window because NFL teams play only 17 games per season. A 5-game window represents nearly a third of the season and captures meaningful form changes without excessive noise. |
| **Altitude factor** | 0.66 (Denver only) | Only the Denver Broncos play at significant altitude (Empower Field at Mile High, 5,280 ft). The thin air affects passing distance, kicking range, and player stamina. No other NFL stadium has meaningful altitude effects. The bonus is computed from the excess home win rate at altitude vs league average. |
| **Weather impact** | Critical (outdoor stadiums) | Unlike indoor sports, most NFL games are played outdoors. Snow, rain, wind, and extreme cold dramatically affect passing accuracy, kicking distance, and ball handling. Weather is the most impactful environmental factor in NFL prediction. Open-Meteo provides free forecasts for all 30 outdoor stadiums. |
| **Playoff HCA factor** | 1.1 | Optimization found that playoff home advantage is slightly amplified compared to regular season. Higher-seeded teams earned homefield; the combination of familiar stadium, rested roster, and playoff atmosphere produces a 10% boost to the standard home advantage. January/February games are auto-detected as postseason. |
| **Division factor** | 31.33 | Divisional opponents play each other twice per year and have deep familiarity. Games between division rivals (8 divisions: AFC East/North/South/West, NFC East/North/South/West) tend to be closer than talent gaps suggest, reducing the predictive edge of pure ratings. |
| **Season regression** | 33% | At the start of each new season (September), all ratings regress 33% toward 1500. This accounts for roster turnover, free agency, coaching changes, the NFL Draft, and the reality that last year's team is not this year's team. Detected by month >= 9 in the calendar. |
| **MOV formula** | log(max(1, abs(margin)) + 1) | Point margins in football follow a roughly logarithmic value curve -- the difference between a 3-point win (field goal) and a 10-point win is much more informative than between a 35-point win and a 42-point win. The log transform prevents garbage-time blowouts from having outsized influence on ratings. |
| **Season calendar** | September-February (cross-year) | Unlike MLB which runs within one calendar year, the NFL season crosses the new year boundary. If month >= 9, it is the current year's season. If month <= 8, it is the previous year's season. This affects season detection, regression timing, and data partitioning. |
| **Pythagorean exponent** | ~2.37 | The Pythagorean theorem for football uses an exponent of approximately 2.37 (per Pro Football Reference), reflecting the scoring environment in NFL games. This is much lower than NBA (~14) because football scores are lower and margins are tighter. Used in both the XGBoost feature pipeline and the Pythagorean model. |
| **Win streak factor** | 20.0 | Momentum matters in football. A team on a 5-game winning streak carries psychological and tactical advantages that pure ratings may not capture. The win_streak_factor adds a momentum signal based on recent consecutive outcomes. |
| **Homestand factor** | 20.0 | Consecutive home games provide compounding advantages: no travel, familiar surroundings, fan energy, and sleep in own beds. The homestand factor rewards teams playing multiple consecutive home games. |
| **Season phase factor** | 20.0 | Early-season games (Weeks 1-4) are less predictable because rosters are still gelling, new schemes are being installed, and ratings have not yet converged. This factor dampens confidence in early-season predictions and increases it as the season progresses. |

---

## Complete Command Reference

### Predictions & Data

| Command | Description | Time |
|---------|-------------|------|
| `<team name>` | Start prediction for any team (fuzzy match: `Chiefs`, `pack`, `DAL`, `eagles`) | ~2s |
| `today` / `html` / `blogger` | Generate HTML prediction table for today's games | ~5s |
| `tomorrow` | Generate HTML prediction table for tomorrow's games | ~5s |
| `all` | Show current Elo ratings for all 32 teams, sorted by rating | ~1s |
| `players` | Show top players by composite score (league-wide or per team) | ~1s |
| `injuries` | Show current NFL injury report from ESPN with Elo impact | ~3s |
| `injuries set <team> <player1>, <player2>` | Manually mark players as OUT for a team | instant |
| `refresh` | Force redownload of all game + player + injury data | ~30s |
| `odds` | Show today's moneyline odds from The Odds API | ~3s |
| `weather` | Show weather forecast for a home team's stadium | ~2s |
| `advstats` / `epa` | Show EPA, CPOE, success rate team rankings via nfl_data_py | ~15s |
| `officials` | Show referee tendency analysis (home win%, penalty rates) | ~10s |

### Backtesting

| Command | Description | Time |
|---------|-------------|------|
| `backtest` | Walk-forward backtest + fit Platt calibration scaler | ~10s |
| `enhanced` | XGBoost ensemble backtest (80/20 Elo/XGB blend, 31 features) | ~20s |
| `enhanced decay` | Time-decayed ensemble (95% Elo early -> 70% Elo late season) | ~20s |
| `shap` | SHAP feature importance analysis for XGBoost features | ~10s |
| `sliding` | Sliding vs expanding window comparison | ~20s |
| `convergence` | Elo rating convergence / burn-in analysis | ~10s |
| `platt` / `calibrate` | Show Platt calibration scaler status and coefficients | instant |

### Elo Optimization

| Command | Description | Time |
|---------|-------------|------|
| `grid` | Grid search over K, HomeAdv, PlayerBoost (768+ combos) | ~5-10m |
| `genetic` | Genetic algorithm optimization (scipy differential evolution) | ~10-20m |
| `bayesian` | Bayesian optimization with GP surrogate + Expected Improvement | ~10-15m |
| `autoopt` | Automatic pipeline: grid -> genetic -> bayesian, apply best | ~30-45m |
| `superopt` | Exhaustive 7-phase optimization, all 9 params (hours) | ~2-4h |
| `singleopt` | Coordinate descent, one param at a time (accuracy-focused) | ~15-30m |
| `results` | Show best parameters found across all optimizers + DSR | instant |

### Validation & Statistical Testing

| Command | Description | Time |
|---------|-------------|------|
| `purgedcv` | Purged walk-forward cross-validation (k-fold with embargo gap) | ~2m |
| `cpcv` | Combinatorial purged CV (all C(k, k_test) train/test paths) | ~5m |
| `pbo` | Probability of backtest overfitting (requires `grid` first) | ~1m |
| `montecarlo` | Monte Carlo permutation test (500 shuffles, p-value) | ~8m |
| `rollingcal` | Rolling origin Platt recalibration (expanding OOS window) | ~2m |
| `conformal` | Conformal prediction intervals (coverage at 80/90/95%) | ~1m |
| `betacal` | Beta calibration (3-param asymmetric, compare to Platt) | ~1m |
| `kelly` | Kelly criterion position sizing backtest (bankroll sim) | ~1m |

### Mega-Ensemble

| Command | Description | Time |
|---------|-------------|------|
| `mega` | Run full mega-ensemble backtest (all enabled models) | ~3-10m |
| `mega optimize` | 7-phase per-model exhaustive mega optimization (54 hyperparameters) | ~1-3h |
| `mega tune` | Per-model solo optimization (Phase 1 only) | ~15-30m |
| `mega tournament` | Head-to-head model tournament (Phase 2 only) | ~15-30m |
| `mega quick` | Quick grid search only (Phase 1) | ~20-40m |
| `mega ablation` | Ablation study: test each model's individual contribution | ~30-60m |
| `mega models` | Show all 35 models with ON/OFF status and tier | instant |
| `mega on <model>` | Enable a specific model (e.g., `mega on lstm`) | instant |
| `mega off <model>` | Disable a specific model (e.g., `mega off weather`) | instant |
| `mega on all` | Enable all 35 models | instant |
| `mega settings` | Show all mega parameter current values | instant |
| `mega set <param>=<value>` | Set a mega parameter (e.g., `mega set adj=0.10`) | instant |

### Trading Ledger

| Command | Description | Time |
|---------|-------------|------|
| `predicts` / `summary` | Show full contract ledger with P&L summary | instant |
| `balance` | Show current account balance | instant |
| `resolve` | Settle a finished contract (win/loss outcome) | instant |
| `sell` | Sell partial or full open position at market price | instant |
| `mark` | Update current market price on open lots | instant |
| `invert` | Flip side of an open position (no accounting change) | instant |
| `chart` | Generate monthly realized P&L bar chart (matplotlib) | ~2s |
| `live` | Live score tracker + open trade status (60s auto-refresh) | ongoing |
| `autoresolve` | Manually run auto-settle on today's finished games | ~5s |
| `autoresolve on` / `off` | Toggle automatic resolution during `live` tracking | instant |

### Settings

| Command | Description | Time |
|---------|-------------|------|
| `settings` | Show all current Elo parameters + Platt scaler status | instant |
| `set <param>=<value>` | Change any Elo parameter (see table below) | instant |
| `set` (no args) | List all available parameters with current values | instant |

### Utility

| Command | Description | Time |
|---------|-------------|------|
| `help` | Show full command list | instant |
| `help <command>` | Detailed help for a specific command | instant |
| `quit` | Save state and exit | instant |

---

## All Settable Parameters

### Elo Parameters (20 tunable via optimizer)

Type `set <param>=<value>` or `set <alias>=<value>`. Example: `set k=20`, `set home=48`, `set bye=1.05`.

#### Core

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `k` | `k_factor` | float | 28.36 | Elo K-factor (learning rate per game). High because only 17 games per season. |
| `base_rating` | `base`, `rating` | float | 1500.0 | Starting Elo rating for all 32 teams |
| `home_adv` | `home`, `hca`, `home_advantage` | float | 25.55 | Home field advantage in Elo points (~53.6% implied) |
| `use_mov` | `mov`, `margin` | bool | true | Use margin of victory adjustment: log(max(1, abs(margin)) + 1) |

#### Player Strength

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `player_boost` | `boost`, `player` | float | 24.61 | Team-level player strength boost (passing-heavy composite) |

#### Margin of Victory

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `mov_base` | `mov_mult`, `mov_constant` | float | 0.8 | MOV multiplier constant (log curve shift) |
| `mov_cap` | `movcap`, `margin_cap` | float | 0.0 | Maximum MOV adjustment cap (0 = uncapped) |

#### Rest / Schedule

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `rest_factor` | `rest` | float | 0.68 | Rest days advantage factor (centered at 7 days, not 1) |
| `rest_advantage_cap` | `restcap`, `rest_cap` | float | 3.32 | Maximum rest advantage multiplier |
| `bye_week_factor` | `bye`, `bye_week`, `bye_factor` | float | 0.0 | Bye week rest advantage bonus (currently disabled) |
| `b2b_penalty` | `b2b`, `back_to_back` | float | 5.45 | Short-week game penalty (Thursday Night Football, etc.) |
| `road_trip_factor` | `roadtrip`, `road_trip` | float | 0.0 | Extended road trip penalty |
| `homestand_factor` | `homestand` | float | 20.0 | Extended homestand bonus (consecutive home games) |

#### Travel / Venue

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `travel_factor` | `travel` | float | 0.0 | Elo penalty per timezone crossed (32 teams span 4 US timezones) |
| `east_travel_penalty` | `east_travel`, `eastbound` | float | 0.0 | Extra penalty for eastbound travel (jet lag asymmetry) |
| `altitude_factor` | `altitude`, `alt` | float | 0.66 | Altitude bonus multiplier (Denver Broncos at 5,280 ft only) |

#### Form / Momentum

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `form_weight` | `form` | float | 20.0 | Recent form weight (last 5 games in a 17-game season) |
| `win_streak_factor` | `streak`, `win_streak` | float | 20.0 | Win/loss streak momentum factor |
| `mean_reversion` | `reversion`, `regress` | float | 0.0 | Mean reversion after extreme results |
| `season_regress` | `season_regression`, `regress_pct` | float | 0.33 | Season boundary regression fraction toward 1500 |

#### Matchup Adjustments

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `sos_factor` | `sos`, `strength_of_schedule` | float | 0.0 | Strength of schedule weight |
| `division_factor` | `division`, `div` | float | 31.33 | Divisional game confidence reducer (AFC/NFC divisions) |
| `conference_factor` | `conference`, `conf` | float | 0.0 | Conference (AFC vs NFC) game adjustment |
| `series_adaptation` | `series`, `adaptation` | float | 0.0 | Series adaptation factor (rematches within the season) |

#### Scoring Model

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `pace_factor` | `pace`, `tempo` | float | 0.0 | Scoring environment mismatch adjustment |
| `pyth_factor` | `pyth`, `pythagorean` | float | 0.0 | Pythagorean expected W% adjustment (exponent ~2.37) |
| `scoring_consistency_factor` | `consistency`, `scoring_consistency` | float | 0.0 | Penalty for volatile scoring patterns |
| `home_road_factor` | `home_road`, `split` | float | 0.0 | Team-specific home/road split bonus |

#### Season / Phase

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `playoff_hca_factor` | `playoff`, `playoff_hca`, `postseason` | float | 1.1 | Playoff home advantage multiplier (Jan-Feb games auto-detected) |
| `season_phase_factor` | `phase`, `season_phase` | float | 20.0 | Early-season dampener (Weeks 1-4 less predictable) |

#### K-Factor Variants

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `k_decay` | `kdecay`, `k_reduction` | float | 0.0 | K-factor decay over the season (reduces reactivity late) |
| `surprise_k` | `surprise`, `upset_k` | float | 0.0 | Extra K for surprise/upset results |

#### Account / Trading

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `kelly_fraction` | `kelly` | special | 0.50 | Kelly criterion fraction (`quarter`/`half`/`full` or 0.25/0.50/1.0) |
| `starting_balance` | `balance`, `bankroll` | float | 0.0 | Starting account balance |
| `autoresolve_enabled` | `autoresolve`, `auto_resolve` | bool | false | Auto-resolve finished trades |

### Mega-Ensemble Parameters (14 total)

Type `mega set <param>=<value>` or `mega set <alias>=<value>`. Example: `mega set adj=0.10`, `mega set meta=ridge`.

| Parameter | Aliases | Type | Description |
|-----------|---------|------|-------------|
| `max_adj` | `maxadj`, `adj`, `adjustment` | float | Max meta-learner adjustment (+/- probability, default ~0.10) |
| `meta_model` | `meta`, `metalearner`, `stacker` | str | Meta-learner type: `ridge`, `logistic`, or `xgboost` |
| `retrain_every` | `retrain`, `retrain_interval` | int | Retrain meta-learner every N games |
| `min_train` | `mintrain`, `min_games`, `warmup` | int | Games before meta-learner starts predicting |
| `kalman_process_noise` | `kalman_pn`, `process_noise`, `pn` | float | Kalman filter process noise |
| `kalman_measurement_noise` | `kalman_mn`, `measurement_noise`, `mn` | float | Kalman filter measurement noise |
| `hmm_states` | `hmm_n`, `n_states`, `states` | int | Number of HMM hidden states |
| `network_decay` | `net_decay`, `pagerank_decay`, `decay` | float | PageRank temporal decay (0-1) |
| `momentum_friction` | `friction`, `mom_friction` | float | Momentum friction coefficient |
| `n_clusters` | `clusters`, `k_clusters`, `nclusters` | int | Number of team archetype clusters |
| `glicko_initial_rd` | `glicko_rd`, `initial_rd`, `rd` | float | Glicko-2 initial rating deviation |
| `bt_decay` | `bt_recency`, `bradley_decay` | float | Bradley-Terry recency decay (0-1) |
| `mc_simulations` | `mc_sims`, `simulations`, `n_sims`, `sims` | int | Monte Carlo simulations per game |
| `window` | `rolling_window`, `feat_window` | int | Rolling feature window size (games) |

---

## Optimization System

### Elo Optimization (6 methods)

| Phase | Command | Method | Params | Time |
|-------|---------|--------|--------|------|
| 1 | `grid` | Exhaustive grid search | K, HomeAdv, PlayerBoost (768+ combos) | ~5-10m |
| 2 | `genetic` | Differential evolution (scipy) | 7 params, 50 gen x 25 pop | ~10-20m |
| 3 | `bayesian` | Gaussian Process + Expected Improvement | 7 params, 15 initial + 40 iter | ~10-15m |
| 4 | `autoopt` | Automatic pipeline (grid -> genetic -> bayesian) | 7 params, best of all three | ~30-45m |
| 5 | `superopt` | Exhaustive 7-phase multi-round optimization | 9 params, hours of search | ~2-4h |
| 6 | `singleopt` | Coordinate descent (one param at a time) | All params, accuracy-focused | ~15-30m |

**Objective function**: `-(LogLoss * 8 + Brier * 40)`. This weighting is intentional -- the Brier component penalizes miscalibration more heavily than raw discrimination, producing probabilities that are well-calibrated rather than just accurate.

**`superopt` 7-phase detail:**

1. **Phase 1 -- Broad Grid Search**: 9 params, ~6,000+ combinations. Establishes the promising region of parameter space.
2. **Phase 2 -- Genetic Round 1**: Wide bounds, 100 generations x 50 population. Differential evolution explores the full space.
3. **Phase 3 -- Bayesian Round 1**: Wide bounds, 30 initial points + 80 iterations. GP surrogate models the objective surface.
4. **Phase 4 -- Genetic Round 2**: Tightened bounds around best-so-far, 80 generations x 40 population. Intensifies search in the best region.
5. **Phase 5 -- Bayesian Round 2**: Tightened bounds, 20 initial + 60 iterations. Fine-grained exploitation of the GP model.
6. **Phase 6 -- Fine Grid**: Tiny step sizes around the absolute best parameters found. Ensures no nearby optimum was missed.
7. **Phase 7 -- Validation**: Runs purgedcv + PBO + Monte Carlo on the winning parameters to confirm they are not overfit.

### Mega-Ensemble Optimization (7 phases, 54 per-model hyperparameters)

| Phase | Method | Description |
|-------|--------|-------------|
| 1 | Solo test | Test each model individually to find per-model optimal settings. |
| 2 | Tournament | Top configs compete head-to-head on held-out data. |
| 3 | Meta-learner tuning | Optimize `max_adj`, `meta_model`, `retrain_every`, `min_train`. |
| 4 | Differential Evolution (DE) | Fine-tune all continuous params with genetic optimization. |
| 5 | Ablation | Prune models that hurt ensemble accuracy. |
| 6 | Validation | Purged CV + stability test with multiple random seeds. |
| 7 | Apply best | Save winning parameters. |

Use `mega tune` for per-model solo optimization (Phase 1 only) and `mega tournament` for head-to-head model comparison (Phase 2 only).

**Mega ablation** (`mega ablation`): Disables each model one at a time and measures the accuracy change. Models that hurt overall accuracy are automatically flagged for pruning. This identifies which of the 35 models are contributing positive signal and which are adding noise.

### Recommended Optimization Workflow

```
Step 1:  python main.py                     # Baseline backtest runs automatically (20 Elo params)
Step 2:  grid                               # Find promising region (~5-10m)
Step 3:  genetic                            # Refine with evolution (~10-20m)
Step 4:  bayesian                           # Fine-tune with GP (~10-15m)
Step 5:  results                            # Compare all optimizer outputs
Step 6:  backtest                           # Refit Platt with best params
Step 7:  mega                               # Run mega-ensemble with Elo base
Step 8:  mega tune                          # Per-model solo optimization
Step 9:  mega tournament                    # Head-to-head model comparison
Step 10: mega optimize                      # Full 7-phase optimization (54 per-model hyperparams, ~1-3h)
Step 11: mega ablation                      # Prune bad models (~30-60m)
Step 12: purgedcv -> pbo -> montecarlo      # Validate (not overfit)
Step 13: kelly                              # Size positions optimally
```

Or use the fully automated shortcut:
```
autoopt                                     # Steps 2-4 automated (~30-45m)
superopt                                    # Steps 2-6 + validation (~2-4h)
```

---

## API Setup

### Data Sources

All data sources are **completely free**. No paid APIs.

| Source | Package / URL | Data Provided | API Key? | Rate Limit |
|--------|--------------|---------------|----------|------------|
| **ESPN Public API** | `requests` (pip) | Game scores, schedules, rosters, live scores | None needed | Unlimited |
| **nfl_data_py / nflverse** | `nfl_data_py` (pip) | EPA, CPOE, success rate, play-by-play, officials | None needed | Unlimited (pre-compiled CSVs from nflverse) |
| **ESPN Injuries** | ESPN public API | Injury reports: IR, Doubtful, Questionable, Out | None needed | Unlimited |
| **Open-Meteo Weather** | `open-meteo.com` REST API | Temperature, wind, humidity, precipitation | None needed | 10,000/day |
| **The Odds API** | `the-odds-api.com` | Moneyline odds from major sportsbooks | Free key (500 req/month) | 500/month |

### The Odds API Setup (Optional)

The Odds API provides real-time moneyline odds. It is free for up to 500 requests per month.

1. Go to [https://the-odds-api.com](https://the-odds-api.com)
2. Sign up for a free account
3. Copy your API key from the dashboard
4. Set environment variable: `set ODDS_API_KEY=your_key_here` (Windows) or `export ODDS_API_KEY=your_key_here` (Linux/Mac)
5. Run `odds` in the CLI to see today's lines

500 requests per month is plenty for daily use -- each `odds` call uses 1 request. During the NFL season (September-February), one call per game day uses roughly 20-25/month (5% of your quota).

### Weather (No Setup Required)

Open-Meteo provides free weather forecasts with no API key. The `weather` command automatically geolocates NFL stadiums and pulls temperature, wind speed, wind direction, humidity, and precipitation probability. Weather is especially important for NFL -- outdoor stadiums in cold-weather cities (Green Bay, Buffalo, Chicago, Denver) can see snow, sub-zero temperatures, and heavy winds that dramatically affect game outcomes.

### Advanced Stats via nfl_data_py (No Setup Required)

The nfl_data_py package provides access to the entire nflverse data ecosystem -- pre-compiled CSVs with play-by-play data, EPA (Expected Points Added), CPOE (Completion Percentage Over Expected), success rates, and officials assignments. No API key, no rate limits, completely free. Install with `pip install nfl_data_py`.

---

## Smart Caching

The caching system adapts to the NFL season calendar (September-February) and whether today is a game day. This minimizes unnecessary API calls while keeping data fresh when it matters.

### Cache Duration Table

| Data Type | Offseason | Game Day (In-Season) | Non-Game Day (In-Season) |
|-----------|-----------|---------------------|--------------------------|
| **Games** (scores, results) | 7 days (168h) | 4 hours | 12 hours |
| **Players** (passing/rushing/receiving leaders) | 30 days (720h) | 48 hours | 48 hours |
| **Injuries** (ESPN IR/Doubtful/Out) | 30 days (720h) | 2 hours | 6 hours |
| **Odds** (moneyline from The Odds API) | Never fetched | 15 minutes | 4 hours |
| **Weather** (Open-Meteo forecast) | Never fetched | 2 hours | 12 hours |
| **Advanced Stats** (nfl_data_py EPA/CPOE) | 30 days (720h) | 24 hours | 24 hours |

### NFL Season Calendar

```
Jan  Feb  Mar  Apr  May  Jun  Jul  Aug  Sep  Oct  Nov  Dec
[PLAYOFFS][------OFF------][--PRESEASON--][--REGULAR SEASON--]
     ^                                    ^                  ^
     Super Bowl                    Season starts     Regular season
     (early Feb)                   (Week 1, Sep)     ends (early Jan)
```

NFL plays games primarily on Sundays, with Monday Night Football and Thursday Night Football adding additional game days. The regular season runs from early September through early January (18 weeks for 17 games plus a bye), followed by playoffs through the Super Bowl in early February. The caching system treats Sunday/Monday/Thursday as potential game days during the active season.

### 32 NFL Teams (8 Divisions)

```
AFC East:  Buffalo Bills, Miami Dolphins, New England Patriots, New York Jets
AFC North: Baltimore Ravens, Cincinnati Bengals, Cleveland Browns, Pittsburgh Steelers
AFC South: Houston Texans, Indianapolis Colts, Jacksonville Jaguars, Tennessee Titans
AFC West:  Denver Broncos, Kansas City Chiefs, Las Vegas Raiders, Los Angeles Chargers

NFC East:  Dallas Cowboys, New York Giants, Philadelphia Eagles, Washington Commanders
NFC North: Chicago Bears, Detroit Lions, Green Bay Packers, Minnesota Vikings
NFC South: Atlanta Falcons, Carolina Panthers, New Orleans Saints, Tampa Bay Buccaneers
NFC West:  Arizona Cardinals, Los Angeles Rams, San Francisco 49ers, Seattle Seahawks
```

---

## Performance

### Multithreading

The mega-ensemble uses `concurrent.futures.ThreadPoolExecutor` to run all 31 base models in parallel. On typical hardware:

- **4-core machine**: ~2-3x speedup over sequential
- **8-core machine**: ~3-5x speedup over sequential
- **16-core machine**: ~5-8x speedup over sequential

The GIL is not a bottleneck because most models spend time in C extensions (numpy, scipy, xgboost, lightgbm, catboost) which release the GIL.

### GPU Acceleration

| Library | GPU Backend | Speedup | Detection |
|---------|------------|---------|-----------|
| XGBoost | CUDA (`gpu_hist`) | 3-10x on tree construction | Automatic if CUDA available |
| LightGBM | CUDA (`device='gpu'`) | 2-5x on tree construction | Automatic if CUDA available |
| CatBoost | CUDA (`task_type='GPU'`) | 3-8x on tree construction | Automatic if CUDA available |
| PyTorch (MLP/LSTM) | CUDA | 5-20x on neural network training | `torch.cuda.is_available()` |

GPU is entirely optional. All models fall back to CPU silently. No configuration needed.

### Typical Backtest Results

Results from smoke testing on ~570 games (approximately 2 NFL seasons):

| Metric | Value |
|--------|-------|
| **Games tested** | 570 |
| **Accuracy** | 65.61% |
| **Log Loss** | 0.6181 |
| **Brier Score** | 0.2138 |
| **ECE** (calibration error) | ~0.039 |
| **BSS vs 50%** | ~0.145 |

Football is more predictable than baseball (best NFL teams win ~75-80% of games, worst teams win ~20-25%), so accuracy in the 65-70% range on moneyline picks represents solid performance. The NFL's smaller sample size (272 regular season games per year vs 2,430 in MLB) means confidence intervals are wider. A model consistently above 60% accuracy across multiple seasons demonstrates genuine predictive signal.

---

## Complete Model Validation Workflow

A rigorous 6-phase workflow ensures your model is genuinely predictive and not overfit to historical data.

### Phase 1: Baseline & Diagnostics

```
backtest          # Walk-forward accuracy, LogLoss, Brier, ECE
convergence       # How many games before Elo ratings stabilize?
sliding           # Does old data help or hurt? (sliding vs expanding window)
```

Establishes baseline metrics. The `convergence` command identifies the burn-in period (typically 100-200 games for NFL -- about 1 full season). `sliding` determines whether the model benefits from full history or performs better with a shorter memory.

### Phase 2: Parameter Optimization

```
grid              # Broad search over K, HomeAdv, PlayerBoost
pbo               # Is the grid search overfit? (PBO > 0.5 = overfit)
results           # Compare all optimizer outputs + Deflated Sharpe Ratio
genetic           # Refine with differential evolution
```

The Probability of Backtest Overfitting (PBO) test is critical here. If PBO > 0.5, your grid search likely found parameters that are overfit to this specific data window. The Deflated Sharpe Ratio (DSR) adjusts for multiple comparisons.

### Phase 3: Cross-Validation

```
purgedcv          # k-fold with embargo gap (prevents Elo momentum leakage)
cpcv              # All C(k, k_test) paths for tighter confidence intervals
montecarlo        # 500 permutation shuffles -> p-value for significance
```

Purged CV adds an embargo gap between train and test folds to prevent Elo momentum from leaking across boundaries. CPCV produces many more backtest paths for tighter confidence. Monte Carlo gives a p-value: if < 0.05, the model's edge is statistically significant.

### Phase 4: Ensemble & Features

```
enhanced          # XGBoost ensemble (31 features, 80/20 blend)
shap              # Which features are driving XGBoost predictions?
enhanced decay    # Time-decayed weighting (XGB gets more weight over season)
```

SHAP analysis reveals whether XGBoost is adding genuine signal beyond Elo or just echoing it. If the top SHAP features are all Elo-derived, the ensemble may not be adding value.

### Phase 5: Calibration

```
rollingcal        # Expanding-window Platt recalibration (truly OOS)
betacal           # 3-parameter beta calibration (handles asymmetry)
conformal         # Distribution-free prediction intervals with coverage
```

Rolling calibration gives truly out-of-sample calibrated metrics. Beta calibration fixes asymmetric miscalibration (e.g., overconfident on favorites but well-calibrated on underdogs). Conformal prediction provides coverage guarantees without distributional assumptions.

### Phase 6: P&L Simulation

```
kelly             # Kelly criterion bankroll simulation on backtest
```

Simulates optimal position sizing over the backtest period. Reports final bankroll, maximum drawdown, Sharpe ratio, and win rate. Uses fractional Kelly (default 50%) for practical sizing.

### Decision Framework

| Metric | Good | Marginal | Bad |
|--------|------|----------|-----|
| **Accuracy** | > 66% | 63-66% | < 63% |
| **ECE** (calibration error) | < 0.02 | 0.02-0.05 | > 0.05 |
| **BSS** (Brier Skill Score vs 50%) | > 0.04 | 0.01-0.04 | < 0.01 |
| **PBO** (Prob Backtest Overfit) | < 0.30 | 0.30-0.50 | > 0.50 |
| **DSR** (Deflated Sharpe Ratio) | > 2.0 | 1.0-2.0 | < 1.0 |
| **Monte Carlo p-value** | < 0.05 | 0.05-0.10 | > 0.10 |
| **Purged CV std** | < 3% | 3-5% | > 5% |
| **CPCV paths > 63%** | > 90% | 70-90% | < 70% |
| **Kelly Sharpe** | > 1.0 | 0.5-1.0 | < 0.5 |
| **Kelly max drawdown** | < 20% | 20-40% | > 40% |

**Interpretation**: If most metrics are "Good", the model has genuine predictive power. If PBO is "Bad" or Monte Carlo p > 0.10, the model's apparent edge is likely noise. Do not trade a model with "Bad" validation metrics.

---

## Daily Prediction Workflow

Step-by-step workflow for making daily predictions during the NFL season:

```
1. LAUNCH
   python main.py
   -> Auto-downloads latest games, players, injuries
   -> Runs baseline backtest (fits Platt scaler)
   -> Shows baseline accuracy

2. CHECK MODEL STATUS
   settings            # Verify parameters are tuned
   platt               # Confirm calibration scaler is fitted
   mega models         # Check which models are enabled

3. MAKE PREDICTIONS
   today               # Generate HTML table for all today's games
   chiefs              # Individual matchup prediction (fuzzy search)
   -> Enter opponent, home team, see calibrated probability
   -> 'y' to log as Predicts contract

4. LOG POSITIONS
   balance             # Check account balance
   -> Enter contracts through prediction flow
   predicts            # Review all open positions

5. PUBLISH (optional)
   today               # Generates today_nfl_predictions.html
   blogger             # Same as today -- copy HTML to Blogger

6. MONITOR
   live                # Live score tracker with open trade status
                       # Auto-refreshes every 60 seconds
   odds                # Check latest odds for CLV comparison

7. SETTLE
   autoresolve         # Auto-settle finished games
   resolve             # Manually settle a specific contract
   sell                # Exit early at market price

8. REVIEW
   predicts            # Full P&L summary
   chart               # Monthly P&L bar chart
   kelly               # Was sizing optimal?
```

---

## Trading Ledger

The Predicts $1 contract tracking system models binary outcome contracts (similar to prediction market contracts) where each contract settles at $1.00 (win) or $0.00 (loss).

### How It Works

1. **Entry**: Buy a contract at the model's implied probability (e.g., buy KC at $0.68)
2. **Entry fee**: 2% of entry price deducted at purchase
3. **Settlement**: Contract resolves to $1.00 (team wins) or $0.00 (team loses)
4. **Profit/Loss**: Settlement value minus entry price minus fees
5. **Exit fee**: 2% deducted if you sell before settlement

### Commands

- `predicts` / `summary` -- Show all lots with entry price, current mark, P&L
- `balance` -- Current account balance
- `resolve` -- Settle a contract (enter W or L outcome)
- `sell` -- Exit a position early at current market price
- `mark` -- Update the current market price of open positions
- `invert` -- Flip the side of a position (e.g., bought YES -> now SHORT NO)
- `chart` -- Monthly realized P&L bar chart
- `autoresolve` -- Auto-settle using today's game results from the ESPN API
- `live` -- Watch live scores with real-time P&L on open trades

### Mark-to-Market

Open positions can be marked to current market prices at any time using `mark`. This updates the unrealized P&L without closing the position. The `predicts` summary shows both unrealized (mark-to-market) and realized (settled) P&L.

### Auto-Resolve

When `autoresolve on` is active, the system automatically settles contracts when final game scores are detected during `live` tracking. You can also manually trigger `autoresolve` to batch-settle all finished games.

### Kelly Criterion Position Sizing

The `kelly` command simulates optimal position sizing over the backtest period. Fractional Kelly (default 50%) is used for practical sizing -- full Kelly is mathematically optimal but produces uncomfortable drawdowns. The Kelly recommendation shown during prediction flow tells you the optimal bet size based on edge and implied probability.

---

## File Structure

```
NFL/
|
|-- main.py                     # CLI entry point, command dispatch loop
|-- config.py                   # Constants, 32 NFL teams, 8 divisions, settings I/O
|-- elo_model.py                # NFLElo class (ratings, predictions, 24 adjusters)
|-- build_model.py              # Model training pipeline with season regression
|
|-- data_games.py               # Game data download via ESPN public API
|-- data_players.py             # Passing + rushing + receiving leaders via ESPN API
|-- advanced_stats.py           # nfl_data_py: EPA, CPOE, success rate, referees
|
|-- backtest.py                 # All backtesting & optimization (~2100 lines)
|-- enhanced_model.py           # XGBoost ensemble (31 features, 5-game window) + SHAP
|-- single_param_opt.py         # Coordinate descent optimizer
|
|-- platt.py                    # Calibration (Platt, isotonic, beta, regression)
|-- metrics.py                  # LogLoss, Brier, ECE, MCE, BSS, conformal
|
|-- predict_ledger.py           # Predicts $1 contract ledger management
|-- live_scores.py              # Live NFL scores + open trade display
|-- auto_resolve.py             # Auto-settle finished trades from live scores
|
|-- injuries.py                 # ESPN injury report + Elo impact (QB out = -50 Elo)
|-- html_generator.py           # Blogger HTML prediction table generation
|-- help_system.py              # Help text for all commands
|-- color_helpers.py            # Colorama terminal formatting utilities
|-- cache_utils.py              # Smart season-aware API caching
|-- elo_set_handler.py          # 'set param=value' command handler (39 params)
|
|-- hmm_model.py                # Hidden Markov Model (hot/cold states)
|-- kalman_model.py             # Kalman Filter (strength estimation)
|-- network_model.py            # PageRank + HITS (network analysis)
|-- gbm_models.py               # LightGBM + CatBoost gradient boosting
|-- nn_models.py                # MLP + LSTM neural networks (PyTorch)
|-- volatility_model.py         # GARCH volatility + Lyapunov/Hurst
|-- signal_model.py             # Fourier + wavelet (cycle detection)
|-- survival_model.py           # Survival analysis (streak hazards)
|-- copula_model.py             # Copula (offense/defense dependency)
|-- information_theory_model.py # Shannon entropy + KL divergence
|-- momentum_model.py           # Newtonian momentum / inertia
|-- markov_chain_model.py       # Markov chain transition matrices
|-- clustering_model.py         # k-Means team archetypes
|-- game_theory_model.py        # Nash equilibrium + style matchups
|-- poisson_model.py            # Poisson / Dixon-Coles score distribution
|-- glicko_model.py             # Glicko-2 uncertainty-aware ratings
|-- bradley_terry_model.py      # Bradley-Terry MLE paired comparison
|-- monte_carlo_model.py        # Monte Carlo simulation (3000 sims)
|-- random_forest_model.py      # Random Forest (bagging diversity)
|-- classic_models.py           # SRS, Colley, Log5, Pythagorean, ExpSmooth, MeanReversion
|
|-- odds_tracker.py             # The Odds API integration + CLV tracking
|-- weather.py                  # Open-Meteo weather impact calculation
|-- kalshi.py                   # Kalshi prediction market integration
|
|-- meta_learner.py             # Ridge/Logistic/XGBoost meta-learner stacker
|-- mega_backtest.py            # Mega-ensemble walk-forward backtest engine
|-- mega_predictor.py           # MegaPredictor class (35-model runtime)
|-- mega_optimizer.py           # 7-phase mega-ensemble optimization
|-- mega_config.py              # Per-model on/off switches + mega params (54 hyperparams)
|
|-- run_optimize.py             # Batch optimization runner
|-- quick_optimizer.py          # Quick optimization utilities
|-- sweep_enhanced.py           # Enhanced model sweep runner
|-- run_enhanced_all.py         # Batch enhanced model runner
|-- accuracy_optimize.py        # Accuracy-focused optimization utilities
|-- accuracy_test.py            # Quick walk-forward accuracy test
|
|-- requirements.txt            # Python dependencies
|-- CLAUDE.md                   # Claude Code agent instructions
|-- README.md                   # This file
|
|-- nfl_elo_settings.json       # [generated] Tuned Elo parameters
|-- nfl_mega_settings.json      # [generated] Mega-ensemble settings (model switches + hyperparams)
|-- nfl_recent_games.csv        # [generated] 2 years of game history from ESPN
|-- nfl_player_stats.csv        # [generated] Passing/rushing/receiving leaders
|-- nfl_advanced_stats.csv      # [generated] EPA, CPOE, success rate from nflverse
|-- nfl_elo_ratings.json        # [generated] Saved Elo ratings for all 32 teams
|-- nfl_platt_scaler.json       # [generated] Platt calibration coefficients
|-- nfl_enhanced_model.json     # [generated] XGBoost metadata
|-- nfl_xgb_model.json          # [generated] XGBoost model weights
|-- nfl_injuries.json           # [generated] Cached injury report (4-hour TTL)
|-- predicts_lots.csv           # [generated] Trading ledger (positions, P&L)
```

55 Python files total. All generated data files use the `nfl_` prefix and are gitignored.

---

## Requirements

### System Requirements

- **Python**: 3.9 or higher
- **OS**: Windows, macOS, or Linux
- **RAM**: 4 GB minimum, 8 GB recommended (mega-ensemble holds all 35 models in memory)
- **Disk**: ~500 MB for cached data + model files
- **Internet**: Required for API data downloads (can run offline with cached data)
- **GPU**: Optional (CUDA-capable NVIDIA GPU for XGBoost/LightGBM/CatBoost/PyTorch acceleration)

### Python Dependencies

#### Core (Required)

```
pandas>=1.5
numpy>=1.24
scipy>=1.10
colorama>=0.4
tqdm>=4.60
xgboost>=2.0
requests>=2.28
matplotlib>=3.7
```

#### NFL Data APIs

```
nfl_data_py>=0.3              # nflverse play-by-play data (EPA, CPOE, officials, rosters)
```

#### Prediction Models

```
hmmlearn>=0.3             # Hidden Markov Models
filterpy>=1.4             # Kalman filters
lightgbm>=4.0             # LightGBM gradient boosting
catboost>=1.2             # CatBoost gradient boosting
networkx>=3.0             # PageRank / HITS graph analysis
```

#### Neural Networks (CPU or GPU)

```
torch>=2.0                # MLP + LSTM (PyTorch)
```

CPU-only install (smaller download):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

GPU install (requires CUDA):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

#### Optional (Enhanced Features)

```
nolds                     # Lyapunov exponents, Hurst exponent (chaos theory metrics)
PyWavelets                # Wavelet transforms (signal processing)
openmeteo-requests        # Weather data helper (not strictly required, plain requests works)
```

### Quick Install

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Recent Changes

- **Fixed `_rolling()` rest_days bug**: Away team was incorrectly using the home team's rest days in XGBoost features. Each team now correctly uses its own rest day calculation.
- **Fixed broad exception handling in `backtest.py`**: Changed bare `except Exception` to `except OSError` so that programming errors are no longer silently swallowed.
- **Added NaN guard for momentum autocorrelation in `enhanced_model.py`**: Prevents NaN values from propagating through the feature pipeline when autocorrelation cannot be computed.
- **Made `season_regress` configurable via settings**: Was previously hardcoded to 0.33; now loaded from `nfl_elo_settings.json` like all other parameters.
- **Optimized Elo parameters from bayesian optimization results**: K=28.36, home_adv=25.55, player_boost=24.61, rest_factor=0.68, form_weight=20.0, division_factor=31.33, win_streak_factor=20.0, homestand_factor=20.0, season_phase_factor=20.0, playoff_hca_factor=1.1, b2b_penalty=5.45, altitude_factor=0.66, rest_advantage_cap=3.32. Zeroed out: travel_factor, sos_factor, pace_factor, mean_reversion, road_trip_factor, scoring_consistency_factor, bye_week_factor.
- **Enabled previously disabled adjusters**: Travel, pace, altitude, homestand, b2b, form, win streak, mean reversion, SOS, division, and season phase factors are now active with optimized values instead of defaulting to 0.

---

## Disclaimer

This software is for **educational and research purposes only**. It is not financial advice. Sports prediction models are inherently uncertain -- even the best models are wrong 30-35% of the time for NFL moneyline picks. No model can guarantee profits. Past backtest performance does not predict future results. Always gamble responsibly and never risk money you cannot afford to lose.

The prediction probabilities produced by this system are statistical estimates, not certainties. The Predicts $1 contract ledger is a paper-trading simulation tool, not a connection to any real prediction market or sportsbook.

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
