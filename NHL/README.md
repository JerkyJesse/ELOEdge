# SharpStack-NHL -- 35-Model Mega-Ensemble

A production-grade NHL game prediction system that fuses 35 independent models -- spanning Elo ratings, gradient boosting, Hidden Markov Models, Kalman filters, PageRank, neural networks, survival analysis, information theory, game theory, and classical hockey analytics -- into a single calibrated probability through a walk-forward meta-learner. Every model trains on real NHL data pulled from completely free APIs (ESPN public API for scores, schedules, player stats, goalie stats, and injuries; Open-Meteo for weather). The system includes a full Predicts $1 binary contract trading ledger with Kelly criterion position sizing, live score tracking with period display (P1, P2, P3, OT, SO), auto-settlement, and monthly P&L charting. All 82-game-season parameters are tuned through a 7-phase exhaustive optimizer with multithreaded backtesting and optional GPU acceleration.

The NHL system's most distinctive feature is its **per-goalie cumulative Elo sub-rating system**. Because a starting goaltender can single-handedly win or lose a hockey game, the system tracks individual goalie Elo ratings (K_GOALIE=6) with 50% season regression, and weights goaltender contributions at 45% of the overall player composite score. A star goaltender injury costs approximately 35 Elo points -- roughly 50% of a team's total value.

**32 NHL teams** | **35 models** | **90+ tunable parameters** (22 Elo + 74 per-model) | **7-phase per-model optimizer** | **No paid APIs**

---

## Quick Start

```bash
# 1. Clone and enter directory
git clone https://github.com/JerkyJesse/SharpStack-NHL.git
cd SharpStack-NHL

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Install PyTorch for neural network models (CPU-only)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4. Launch
python main.py
```

**First-time workflow:**

```
1. System auto-downloads 2 years of NHL game data via ESPN API
2. System auto-downloads skater leaders, goalie leaders, injury reports
3. Baseline backtest runs automatically (fits Platt calibration scaler)
4. Enter starting balance when prompted (for contract tracking)
5. Type a team name (e.g. "Bruins") to make your first prediction
6. Run 'mega' for the full 35-model ensemble backtest
7. Run 'mega tune' to solo-test each model's optimal settings
8. Run 'mega optimize' for full 7-phase per-model optimization
```

On startup, the system downloads and caches all required data, builds the Elo model with 33% season regression, runs a baseline backtest with Platt calibration, and drops you into the interactive command loop. No API keys are needed for core functionality -- the ESPN public API, Open-Meteo weather, and all data sources are completely free. No special NHL package is required -- all data comes through the `requests` library hitting ESPN endpoints directly.

---

## The 35 Models

Every model runs independently on the same game-by-game walk-forward loop. Their raw outputs feed into the meta-learner, which produces a single calibrated adjustment bounded by `max_adj`.

### Tier 0 -- Core (Always On)

| # | Model | Year | Method | Description |
|---|-------|------|--------|-------------|
| 1 | **Elo** | 1960 | Paired comparison rating | 24+ adjusters: home ice advantage, MOV (log-compressed for hockey's low scoring), per-goalie cumulative Elo sub-ratings (K_GOALIE=6), rest days, back-to-back penalty, travel fatigue, altitude (Colorado 5280 ft, Utah 4226 ft), form, SOS, overtime detection, playoff detection. 32 teams, 82-game season, K=5.0 with logarithmic MOV. |
| 2 | **XGBoost** | 2016 | Gradient boosted trees | 96 rolling features per game (goals for/against, save percentage, plus/minus, Pythagorean win%, streaks, consistency, scoring trend, rest, travel, back-to-back flags, road trip length, SOS-adjusted win%). Walk-forward training with 80/20 Elo/XGBoost blend. SHAP feature importance built in. |

### Tier 1 -- Proven Models

| # | Model | Year | Method | Description |
|---|-------|------|--------|-------------|
| 3 | **HMM** | 1966 | Hidden Markov Model | Detects latent hot/cold team states from win/loss sequences. Forward-backward algorithm estimates state probabilities. Captures momentum shifts invisible to pure ratings -- critical in hockey where teams go on extended hot/cold streaks. |
| 4 | **Kalman** | 1960 | Kalman Filter | Treats true team strength as a hidden state with process noise. Bayesian updates after each game. Provides uncertainty estimates alongside point predictions. Handles the high game-to-game variance inherent in low-scoring hockey. |
| 5 | **PageRank** | 1998 | Network analysis | Builds directed win graph, runs PageRank + HITS authority scores. Teams that beat strong teams get more credit. Temporal decay weights recent results. Naturally captures strength of schedule in the NHL's unbalanced schedule. |
| 6 | **LightGBM** | 2017 | Leaf-wise gradient boosting | Microsoft's fast GBM with leaf-wise splits. Handles categorical features natively. Lower memory than XGBoost with comparable accuracy. |
| 7 | **CatBoost** | 2017 | Ordered gradient boosting | Yandex's ordered boosting prevents target leakage during training. Handles categorical features with target statistics. Robust to overfitting. |
| 8 | **MLP** | 1986 | Multi-layer perceptron | PyTorch feedforward neural network with batch normalization and dropout. Learns nonlinear feature interactions that tree models miss. |
| 9 | **LSTM** | 1997 | Long Short-Term Memory | Recurrent neural network that models sequential game patterns. Captures long-range dependencies in team performance trajectories. Off by default (slow). |

### Tier 2 -- Exotic / Physics-Inspired

| # | Model | Year | Method | Description |
|---|-------|------|--------|-------------|
| 10 | **GARCH** | 1986 | Volatility modeling | Generalized AutoRegressive Conditional Heteroskedasticity. Models time-varying volatility in goal scoring. High-variance teams are harder to predict -- especially relevant in hockey where a single period can swing a game. |
| 11 | **Fourier** | 1822 | Cycle detection | Fourier transforms + wavelet analysis on scoring time series. Detects periodic patterns (weekly, monthly) and seasonal rhythms in team performance across the 82-game schedule. |
| 12 | **Survival** | 1958 | Hazard modeling | Cox proportional hazards applied to win/loss streaks. Models the probability that a streak ends given its length and covariates. |
| 13 | **Copula** | 1959 | Joint dependency | Models the dependency structure between offensive and defensive performance using copula functions. Captures teams where offense/defense move together vs independently -- distinguishes goaltending-dominant teams from balanced ones. |

### Tier 3 -- Information & Physics

| # | Model | Year | Method | Description |
|---|-------|------|--------|-------------|
| 14 | **Info Theory** | 1948 | Shannon entropy + KL divergence | Measures predictability of each team's goal scoring distribution. High-entropy teams are chaotic; low-entropy teams are predictable. KL divergence quantifies matchup asymmetry. |
| 15 | **Momentum** | 1687 | Newtonian mechanics analogy | Treats team strength as a physical object with mass (games played) and velocity (recent trend). Friction coefficient controls decay. Captures inertia in form. |
| 16 | **Markov Chain** | 1906 | Transition matrices | Models sequences of outcomes (W/L/close-W/blowout-W) as Markov transitions. Stationary distribution gives long-run expected state probabilities. Blowout threshold is 3+ goals (NHL-specific). |
| 17 | **Clustering** | 1957 | k-Means archetypes | Groups teams into archetypes (e.g., high-offense/low-defense, balanced, goaltending-dominant). Matchup predictions based on how archetype pairs historically perform. |
| 18 | **Game Theory** | 1950 | Nash equilibrium | Models strategic matchups: power vs finesse, offense vs defense. Computes Nash equilibrium strategies and style-based advantages. |

### Tier 4 -- Classical Rating Systems

| # | Model | Year | Method | Description |
|---|-------|------|--------|-------------|
| 19 | **Poisson** | 1898 | Dixon-Coles score distribution | Models goal scoring as Poisson-distributed. Dixon-Coles correction for low-scoring games. Produces full score probability matrix for each matchup -- naturally suited to hockey's discrete, low-scoring outcomes. |
| 20 | **Glicko-2** | 2001 | Uncertainty-aware ratings | Extends Elo with rating deviation (confidence interval) and volatility. Teams with fewer recent games have wider uncertainty. More principled than fixed-K Elo. |
| 21 | **Bradley-Terry** | 1952 | Maximum likelihood paired comparison | MLE estimation of team strengths from pairwise outcomes. Recency-weighted decay ensures recent games matter more. Clean probabilistic framework. |
| 22 | **Monte Carlo** | 1940s | Stochastic simulation | Runs 2,000+ game simulations per matchup using historical goal scoring distributions. Produces win probability from simulation outcomes. |
| 23 | **Random Forest** | 2001 | Bagged decision trees | Ensemble of decorrelated decision trees. Provides diversity to the meta-learner -- different inductive bias from boosted methods. |

### Tier 5 -- Classical Hockey / Sports Models

| # | Model | Year | Method | Description |
|---|-------|------|--------|-------------|
| 24 | **SRS** | ~1980s | Simple Rating System | Average goal margin adjusted for strength of schedule. Iterative convergence. The backbone of many hockey power rankings. |
| 25 | **Colley** | 2001 | Colley Matrix | Bias-free ranking using only wins and losses. Solves a linear system -- no preseason assumptions, no margin of victory. Used by the BCS. |
| 26 | **Log5** | 1981 | Bill James formula | The original sabermetric head-to-head formula: P(A beats B) = (pA - pA*pB) / (pA + pB - 2*pA*pB). Elegant and theoretically grounded. |
| 27 | **PythagenPat** | 2005 | Dynamic Pythagorean exponent | Extends the Pythagorean expected win% formula with a dynamic exponent based on the goal environment (goals per game). Better than fixed-exponent Pythagorean for hockey where scoring levels vary across eras. Default PYTH_EXP = 2.05 for NHL. |
| 28 | **Exp Smoothing** | 1957 | Exponential smoothing | Holt-Winters style smoothing on team performance metrics. Captures level, trend, and seasonality in goal scoring. Simple but effective trend tracker. |
| 29 | **Mean Reversion** | ~1990s | Bollinger band analog | Identifies teams performing above/below their "true" level using a z-score band approach. Teams far from the mean are expected to regress. |

### Tier 6 -- Data Enrichment

| # | Model | Year | Method | Description |
|---|-------|------|--------|-------------|
| 30 | **Weather** | -- | Environmental impact | Temperature, wind speed, humidity, precipitation probability. Open-Meteo API (free, no key). Adjusts predictions for extreme weather conditions at outdoor venues. Off by default. |
| 31 | **Odds** | -- | Market consensus | Ingests moneyline odds from The Odds API. Closing Line Value (CLV) tracking. Markets are efficient -- odds provide a strong independent signal. Off by default (requires free API key). |

### Tier 7 -- Novel / Experimental

| # | Model | Year | Method | Description |
|---|-------|------|--------|-------------|
| 32 | **SVM** | 2026 | Support Vector Machine | RBF kernel with Platt scaling, maximum-margin classifier |
| 33 | **Fibonacci** | 2026 | Fibonacci Retracement | EMA-smoothed performance swings with support/resistance levels |
| 34 | **EVT** | 2026 | Extreme Value Theory | Generalized Pareto distribution for tail risk analysis |
| 35 | **Benford** | 2026 | Benford's Law | Chi-squared scoring pattern anomaly detection |

---

## Architecture

```
                          ESPN PUBLIC API            ESPN PUBLIC API
                        (games, scores,             (player stats,
                         schedules)                  goalie stats)
                              |                         |
                    ESPN INJURIES API             OPEN-METEO WEATHER
                    (Out, DTD, IR, LTIR)          (temp, wind, precip)
                              |                         |
                    +---------+---------+---------------+
                    |                                   |
                    v                                   v
             +------+-------+                  +--------+--------+
             |  DATA LAYER  |                  |  PLAYER STATS   |
             | data_games   |                  |  data_players   |
             | data_players |                  |  (ESPN skaters  |
             +--------------+                  |   + goalies)    |
                    |                          +-----------------+
                    +-----------------------------------+
                    |
                    v
   +=====================================+
   |         ELO ENGINE (Tier 0)         |
   |  NHLElo class -- 32 teams           |
   |  Per-goalie cumulative Elo ratings  |
   |  (K_GOALIE=6, 50% season regress)  |
   |  24 adjustment factors              |
   |  Season regression (33%)            |
   |  Platt/Isotonic/Beta calibration    |
   +=====================================+
                    |
                    | Elo probability (anchor)
                    |
   +=====================================+
   |    35 BASE MODEL PREDICTIONS        |
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
   |           PythagenPat, ExpSmooth,   |
   |           MeanReversion             |
   |  [Tier 6] Weather, Odds            |
   |                                     |
   |  All models run in PARALLEL via     |
   |  ThreadPoolExecutor                 |
   +=====================================+
                    |
                    | Vector of 35 probabilities
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
   |  (default 0.20 = 20 percentage      |
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
        |    (P1/P2/P3/OT/SO) |
        +---------------------+
```

### Elo-Anchored Bounded Adjustment

The Elo model serves as the anchor probability. The meta-learner (trained on all 35 base model outputs) produces an adjustment that is **clamped** to `+/- max_adj` (default 0.20). This means even if all exotic models disagree with Elo, the final probability can shift at most 20 percentage points. This design prevents catastrophic predictions from untested models while allowing proven signal to improve accuracy.

### Multithreaded Training

All 35 base models run inside a `ThreadPoolExecutor`. On a typical 8-core machine, the mega-ensemble backtest completes 3-5x faster than sequential execution. Each model receives the same game-by-game data and produces an independent probability estimate.

### GPU Acceleration

XGBoost, LightGBM, and CatBoost automatically detect CUDA-capable GPUs. If available, tree construction runs on GPU (`tree_method='gpu_hist'` for XGBoost, `device='gpu'` for LightGBM/CatBoost). PyTorch models (MLP, LSTM) also move to GPU when `torch.cuda.is_available()`. CPU fallback is always automatic and silent.

---

## NHL-Specific Design Choices

Every parameter in this system was chosen with the specific structure of the National Hockey League in mind. Here is why each default is what it is:

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| **K-factor** | 5.0 | NHL plays 82 games per season -- fewer than MLB (162) but more than NFL (17). K=5.0 was found via bayesian optimizer to minimize calibration error. Moderate K balances responsiveness with stability, which works well for hockey's high game-to-game variance due to low scoring and goaltender influence. |
| **Home ice advantage** | 26.0 Elo (~53.7%) | NHL home teams historically win about 54-55% of games. 26.0 Elo points in the standard Elo formula yields approximately 53.7% expected win rate. Home ice advantage comes from last change (line matching), familiar ice surface, and crowd energy. |
| **Player scoring weight** | 55% skaters / 45% goalies | Goaltending is disproportionately important in hockey -- a single goaltender can steal or lose a game. The 45% goalie weight reflects the reality that goaltending dominates low-scoring outcomes. Skater stats tracked: Points, Goals, Assists. Goalie stats tracked: GAA, Save Percentage, Wins, Shutouts. |
| **Goaltender Elo tracking** | Per-goalie cumulative Elo (K_GOALIE=6) | NHL is unique among major sports: the starting goaltender identity has massive impact on game outcomes. The system tracks individual goalie Elo ratings that update after every start using `K_GOALIE * (outcome - 0.5) * log_mov`. Goalie ratings regress 50% toward zero at season boundaries. A starter_boost of 5.0 scales the goalie's cumulative rating into the team's Elo. Goalie injuries are the most impactful in all of sports -- star goaltender out = -35 Elo, starter out = -25 Elo, backup out = -15 Elo. Approximately 50% of a team's value resides in its goaltending. |
| **Back-to-back penalty** | 18.0 Elo | Back-to-back games are common in the NHL schedule (3-4 per month per team). Fatigue from consecutive-day games is significant, especially for goaltenders who must maintain explosive reflexes. The 18-point penalty reflects reduced performance from consecutive-day games -- much higher than NFL's equivalent because B2Bs actually occur in hockey. |
| **Rest factor** | 12.0 Elo | Higher than NFL due to the frequency of NHL games (3-4 per week). Rest advantage is computed as `rest_factor * (min(rest_days, cap) - 1)`. Extra rest beyond the cap (default 1.5 days, effectively rounded to 1) provides diminishing returns. |
| **Travel factor** | 8.0 Elo per timezone | Active in NHL (unlike some sports where it is zeroed out). Computed as `-travel_factor * timezone_crossings` based on the team's last game location. A Pacific-to-Eastern cross-country trip (3 timezone difference) costs 24 Elo points. |
| **Overtime factor** | 5.0 | NHL-specific: all games have a winner (no ties since 2005). Games tied after regulation go to 5-minute 3-on-3 overtime, then a shootout if needed. Overtime/shootout results carry different informational value than regulation wins -- a team that frequently goes to OT may be evenly matched rather than dominant. |
| **Altitude factor** | 4.0 | Two altitude teams: Colorado Avalanche (Ball Arena, 5,280 ft) and Utah Hockey Club (Delta Center, 4,226 ft -- relocated from Arizona in 2024). Altitude bonus is computed from excess home win rate, scaled by relative elevation above 4,000 ft. The thin air at altitude affects player fatigue and puck behavior. |
| **MOV formula** | log(max(1.0, abs(goal_diff)) + 1.0) | Goal margins in hockey follow a logarithmic value curve -- a 1-goal game is much more informative than the difference between a 5-goal and 6-goal blowout. The log compression is essential for hockey's typically low-scoring games (average ~3 goals per team). |
| **Rolling window** | 10 games | Narrower than MLB's 15-game window because hockey has a shorter 82-game season. A 10-game window captures meaningful form over ~2-3 weeks while staying responsive to genuine changes in team quality. |
| **Pythagorean exponent** | 2.05 (dynamic via PythagenPat) | The Pythagorean theorem for hockey uses an exponent based on the goal environment. PythagenPat computes the exponent dynamically from goals per game rather than using a fixed value, improving accuracy across different scoring eras. The base PYTH_EXP of 2.05 is tuned for the modern NHL. |
| **Season regression** | 33% | At the start of each new season, all team ratings regress 33% toward 1500. This accounts for roster turnover, free agency, trades, coaching changes, and the reality that last year's team is not this year's team. Goalie ratings regress 50% separately. |
| **Season calendar** | October-June (cross-year) | Unlike MLB which runs within a single calendar year, the NHL season spans two calendar years. The system detects season boundaries using `year + 1 if month >= 10`, meaning October 2025 through June 2026 is the 2025-26 season. |
| **No ties** | Always a winner | NHL games cannot end in a tie. If regulation ends tied, the game goes to overtime (5-minute 3-on-3) and then a shootout if needed. The final result is always a win or loss, simplifying the prediction to a pure binary outcome. |
| **Blowout threshold** | 3+ goals | Mean reversion triggers after a 3+ goal margin (NHL-specific). A 3-goal game in hockey is a dominant performance; the model expects regression toward the mean after such results. |
| **Playoff HCA factor** | 1.0 | Playoff games use full home ice advantage. NHL playoffs run mid-April through June. The system detects playoff games by date (April 15+ through June). |
| **Injury statuses** | Out, Day-to-Day, IR, LTIR | NHL injury designations tracked from ESPN: Out (not playing), Day-to-Day (questionable), IR (Injured Reserve, minimum 7 days), LTIR (Long-Term Injured Reserve, minimum 24 days). Goaltender injuries are weighted at approximately 50% of team value. |
| **Divisions** | 4 divisions, 32 teams | Atlantic (8), Metropolitan (8), Central (8 -- includes Utah Hockey Club, relocated from Arizona in 2024), Pacific (8). Division factor reduces prediction confidence for divisional matchups where familiarity creates parity. |

---

## Complete Command Reference

### Predictions & Data

| Command | Description | Time |
|---------|-------------|------|
| `<team name>` | Start prediction for any team (fuzzy match: `Bruins`, `rangers`, `COL`, `caps`) | ~2s |
| `today` / `html` / `blogger` | Generate HTML prediction table for today's games | ~5s |
| `tomorrow` | Generate HTML prediction table for tomorrow's games | ~5s |
| `all` | Show current Elo ratings for all 32 teams, sorted by rating | ~1s |
| `players` | Show top skaters by composite score (league-wide or per team) | ~1s |
| `injuries` | Show current NHL injury report from ESPN with Elo impact | ~3s |
| `injuries set <team> <player1>, <player2>` | Manually mark players as OUT for a team | instant |
| `refresh` | Force redownload of all game + player + injury data | ~30s |
| `odds` | Show today's moneyline odds from The Odds API | ~3s |
| `weather` | Show weather forecast for a home team's arena | ~2s |

### Backtesting

| Command | Description | Time |
|---------|-------------|------|
| `backtest` | Walk-forward backtest + fit Platt calibration scaler | ~15s |
| `enhanced` | XGBoost ensemble backtest (80/20 Elo/XGB blend, 96 features) | ~30s |
| `enhanced decay` | Time-decayed ensemble (95% Elo early -> 70% Elo late season) | ~30s |
| `shap` | SHAP feature importance analysis for XGBoost features | ~10s |
| `sliding` | Sliding vs expanding window comparison | ~30s |
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
| `live` | Live score tracker + open trade status (60s auto-refresh, shows P1/P2/P3/OT/SO) | ongoing |
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

### Elo Parameters (21 tunable via optimizer)

Type `set <param>=<value>` or `set <alias>=<value>`. Example: `set k=8.0`, `set home=25`, `set boost=20`.

#### Core

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `k` | `k_factor` | float | 5.0 | Elo K-factor (learning rate per game). Range 3-40 for NHL's 82-game season. |
| `base_rating` | `base`, `rating` | float | 1500.0 | Starting Elo rating for all teams |
| `home_adv` | `home`, `hca`, `home_advantage` | float | 26.0 | Home ice advantage in Elo points (~54% win rate) |
| `use_mov` | `mov`, `margin` | bool | true | Use margin of victory adjustment (log-compressed for hockey) |

#### Player / Goaltender

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `player_boost` | `boost`, `player` | float | 10.0 | Team-level player strength boost (55% skaters / 45% goalies) |
| `starter_boost` | `starter`, `goalie_boost` | float | 5.0 | Starting goaltender quality boost (scales per-goalie Elo into team rating) |

#### Margin of Victory

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `mov_base` | `mov_mult`, `mov_constant` | float | 0.8 | MOV multiplier constant (log curve shift) |
| `mov_cap` | `movcap`, `margin_cap` | float | 0.0 | Maximum MOV adjustment cap |

#### Rest / Schedule

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `rest_factor` | `rest` | float | 12.0 | Rest days advantage factor (higher than NFL due to frequent games) |
| `rest_advantage_cap` | `restcap`, `rest_cap` | float | 1.5 | Maximum rest advantage multiplier |
| `b2b_penalty` | `b2b`, `back_to_back` | float | 18.0 | Back-to-back game penalty (common in NHL, 3-4 per month) |
| `road_trip_factor` | `roadtrip`, `road_trip` | float | 2.5 | Extended road trip penalty (3+ consecutive away games) |
| `homestand_factor` | `homestand` | float | 3.0 | Extended homestand bonus |

#### Travel / Venue

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `travel_factor` | `travel` | float | 8.0 | Elo penalty per timezone crossed (active in NHL) |
| `east_travel_penalty` | `east_travel`, `eastbound` | float | 0.0 | Extra penalty for eastbound travel |
| `altitude_factor` | `altitude`, `alt` | float | 4.0 | Altitude bonus multiplier (Colorado 5280 ft, Utah 4226 ft) |

#### Form / Momentum

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `form_weight` | `form` | float | 4.0 | Recent form weight (last 10 games win%) |
| `win_streak_factor` | `streak`, `win_streak` | float | 2.0 | Win/loss streak momentum factor |
| `mean_reversion` | `reversion`, `regress` | float | 2.5 | Mean reversion after extreme results (3+ goal blowouts) |
| `season_regress` | `season_regression`, `regress_pct` | float | 0.33 | Season boundary regression fraction toward 1500 |

#### Matchup Adjustments

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `sos_factor` | `sos`, `strength_of_schedule` | float | 10.0 | Strength of schedule weight (active in NHL) |
| `division_factor` | `division`, `div` | float | 5.0 | Divisional game confidence reducer (4 divisions: Atlantic, Metro, Central, Pacific) |
| `series_adaptation` | `series`, `adaptation` | float | 0.0 | Series adaptation factor (rematches) |
| `overtime_factor` | `overtime`, `ot` | float | 5.0 | Overtime/shootout result adjustment (NHL-specific, all games have a winner) |

#### Scoring Model

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `pace_factor` | `pace`, `tempo` | float | 10.0 | Goal environment mismatch adjustment (high-scoring vs low-scoring teams) |
| `pyth_factor` | `pyth`, `pythagorean` | float | 0.0 | Pythagorean expected W% adjustment (exponent 2.05 for NHL) |
| `scoring_consistency_factor` | `consistency`, `scoring_consistency` | float | 1.5 | Penalty for volatile goal scoring patterns |
| `home_road_factor` | `home_road`, `split` | float | 0.0 | Team-specific home/road split bonus |

#### Season / Phase

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `playoff_hca_factor` | `playoff`, `playoff_hca`, `postseason` | float | 1.0 | Playoff home ice advantage multiplier (playoffs mid-April through June) |
| `season_phase_factor` | `phase`, `season_phase` | float | 2.5 | Early-season dampener |

#### K-Factor Variants

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `k_decay` | `kdecay`, `k_reduction` | float | 0.0 | K-factor decay over the season |
| `surprise_k` | `surprise`, `upset_k` | float | 0.0 | Extra K for surprise/upset results |

#### Account / Trading

| Parameter | Aliases | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `kelly_fraction` | `kelly` | special | 0.25 | Kelly criterion fraction (`quarter`/`half`/`full` or 0.25/0.50/1.0) |
| `starting_balance` | `balance`, `bankroll` | float | 50.0 | Starting account balance |
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

### Per-Model Hyperparameters (54 total)

Set via `mega set <param>=<value>`. These are tuned automatically by `mega optimize` (Phase 1 solo testing).

| Model | Hyperparameters | Defaults |
|-------|----------------|----------|
| **Meta XGBoost** | max_depth, eta, subsample, min_child_weight, alpha, lambda, num_boost_round | 4, 0.05, 0.8, 5, 0.5, 1.0, 200 |
| **Meta Ridge** | alpha_scale | 0.5 |
| **Meta Logistic** | l2 | 0.01 |
| **LightGBM** | num_leaves, learning_rate, n_rounds, lambda_l1, lambda_l2 | 31, 0.03, 300, 0.1, 0.1 |
| **CatBoost** | iterations, learning_rate, depth, l2_leaf_reg | 300, 0.05, 6, 3.0 |
| **MLP** | hidden layers, lr, epochs, dropout | 64/32, 0.001, 100, 0.3 |
| **LSTM** | hidden_dim, n_layers, lr, epochs, dropout | 64, 2, 0.001, 80, 0.3 |
| **Random Forest** | n_trees, max_depth, min_samples_leaf | 100, 5, 10 |
| **HMM** | states, covariance_type, n_iter | 3, diag, 100 |
| **Kalman** | process_noise, measurement_noise | 0.3, 2.5 |
| **PageRank** | network_decay, damping | 0.95, 0.85 |
| **GARCH** | alpha, beta | 0.10, 0.80 |
| **Poisson** | home_adv, decay | 1.2, 0.98 |
| **Glicko** | initial_rd, initial_vol | 200, 0.06 |
| **Bradley-Terry** | decay, max_iterations | 0.99, 100 |
| **Monte Carlo** | simulations, kde_bandwidth | 2000, 0.3 |
| **Momentum** | friction, velocity_window, impulse_window | 0.05, 5, 3 |
| **Clustering** | n_clusters | 4 |
| **Markov** | n_states | 4 |
| **Game Theory** | ema_alpha | 0.05 |
| **Info Theory** | n_bins | 5 |

---

## Data Sources

| Source | Data | Cost | Cache |
|--------|------|------|-------|
| ESPN API | Game scores, schedules | Free | 6 hours |
| ESPN API | Player stats (skaters + goalies) | Free | 6 hours |
| NHL API | Goalie boxscore data (for backfill) | Free | On demand |
| ESPN API | Injury reports | Free | 4 hours |
| Open-Meteo | Weather conditions | Free, no key | 2 hours |
| The Odds API | Moneyline odds, CLV | Free tier (500 req/mo) | 1 hour |
| Kalshi | Prediction market prices | Free public API | Real-time |

### Smart Caching

Cache staleness is season-aware via `cache_utils.py`:
- **In-season** (October-June): Games/players refresh every 6 hours, injuries every 4 hours, weather every 2 hours
- **Off-season**: All caches extend to 24+ hours
- **Stale detection**: Files under 500 bytes treated as corrupt stubs
- **Manual refresh**: `refresh` command deletes all caches and re-downloads

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

NHL-specific optimizer bounds: K range 3-40 (lower than NBA's 5-80), HomeAdv range 10-50 (lower than NBA's 20-80), reflecting hockey's tighter rating distributions and lower-scoring games.

**Optimizer objective function:**

```
score = -(LogLoss * 8 + Brier * 40)
```

This weighting is intentional -- LogLoss penalizes confident wrong predictions while Brier rewards calibration. The 8:40 ratio prioritizes calibration over raw accuracy.

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
Step 1:  python main.py                     # Baseline backtest runs automatically (21 Elo params)
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

## The Goaltender System

The goaltender tracking system is the most distinctive NHL feature in this codebase, and it reflects the outsized impact that goaltending has on hockey outcomes compared to any single player in other major sports.

### How Goaltender Elo Works

Each goaltender in the league has a **cumulative Elo sub-rating** that tracks their individual performance across starts:

1. **Initial rating**: All goalies start at 0.0 (neutral).
2. **Update formula**: After each start, the goalie's rating updates by `K_GOALIE * (outcome - 0.5) * log_mov`, where K_GOALIE = 6, outcome is 1.0 (win) or 0.0 (loss), and log_mov is the log-compressed goal differential.
3. **Season regression**: At each season boundary, all goalie ratings regress 50% toward zero (more aggressive than the 33% team regression, reflecting year-to-year goalie volatility).
4. **Integration into team Elo**: The `starter_boost` parameter (default 5.0) scales the goalie's cumulative rating and adds it to the team's effective Elo: `team_elo += starter_boost * goalie_rating / 100.0`.

### Goaltender Injury Impact

Goaltender injuries carry the heaviest weight of any position in any sport modeled by this system:

| Goalie Tier | Elo Impact | Example |
|------------|-----------|---------|
| Star goaltender out | -35 Elo | Losing a Vezina-caliber goalie |
| Starting goaltender out | -25 Elo | Losing a reliable #1 goalie |
| Backup goaltender out | -15 Elo | Losing depth in goal |

For context, -35 Elo represents roughly 50% of a typical team's advantage over an average opponent, or approximately a 5 percentage point swing in win probability. This is why the player composite weight allocates 45% to goaltending despite goalies being just 1-2 players on a roster.

### Goalie Data Pipeline

- **backfill_goalies.py**: NHL-unique utility that fetches historical starting goalie data from the NHL API (`api-web.nhle.com/v1`) by looking up boxscores and identifying the goalie with the most time on ice for each game.
- **data_players.py**: Fetches current goalie statistics (GAA, Save Percentage, Wins, Shutouts) from the ESPN API.
- **injuries.py**: Tracks goalie injury status (Out, Day-to-Day, IR, LTIR) from ESPN. Goalie injuries are flagged as high-impact.

### Smoke Test Results

From the most recent validation run: 3,003 games tested, **202 goalies tracked** individually, 56.11% accuracy, 0.6867 log loss, 0.2459 Brier score.

---

## API Setup

### Data Sources

All data sources are **completely free**. No paid APIs. No special NHL package required.

| Source | Package / URL | Data Provided | API Key? | Rate Limit |
|--------|--------------|---------------|----------|------------|
| **ESPN Public API** | `requests` (pip) -- `site.api.espn.com` | Game scores, schedules, player stats, goalie stats, live scores | None needed | Unlimited (0.5s sleep between calls) |
| **ESPN Injuries** | `requests` (pip) -- ESPN public API | Injury reports: Out, Day-to-Day, IR, LTIR designations | None needed | Unlimited |
| **NHL API** | `requests` (pip) -- `api-web.nhle.com/v1` | Historical boxscores for goalie backfill | None needed | Unlimited |
| **Open-Meteo Weather** | `open-meteo.com` REST API | Temperature, wind, humidity, precipitation | None needed | 10,000/day |
| **The Odds API** | `the-odds-api.com` | Moneyline odds from major sportsbooks | Free key (500 req/month) | 500/month |

**Important**: NHL data uses NO special sport-specific package. All game scores, schedules, player statistics, goalie statistics, and injury data come from the ESPN public JSON API (`site.api.espn.com`) accessed directly via the standard `requests` library with a 0.5-second sleep between sequential calls to be respectful. This is unlike MLB (which uses `MLB-StatsAPI` and `pybaseball`) or NBA (which uses `nba_api`).

### The Odds API Setup (Optional)

The Odds API provides real-time moneyline odds. It is free for up to 500 requests per month.

1. Go to [https://the-odds-api.com](https://the-odds-api.com)
2. Sign up for a free account
3. Copy your API key from the dashboard
4. Set environment variable: `set ODDS_API_KEY=your_key_here` (Windows) or `export ODDS_API_KEY=your_key_here` (Linux/Mac)
5. Run `odds` in the CLI to see today's lines

500 requests per month is plenty for daily use -- each `odds` call uses 1 request. At one call per day, that is 30/month (6% of your quota).

### Weather (No Setup Required)

Open-Meteo provides free weather forecasts with no API key. The `weather` command automatically geolocates NHL arenas and pulls temperature, wind speed, wind direction, humidity, and precipitation probability.

---

## Smart Caching

The caching system adapts to the NHL season calendar (October-June) and whether today is a game day. This minimizes unnecessary API calls while keeping data fresh when it matters.

### Cache Duration Table

| Data Type | Offseason | Game Day (In-Season) | Non-Game Day (In-Season) |
|-----------|-----------|---------------------|--------------------------|
| **Games** (scores, results) | 7 days (168h) | 4 hours | 12 hours |
| **Players** (skater/goalie leaders) | 30 days (720h) | 48 hours | 48 hours |
| **Injuries** (ESPN IR/LTIR/DTD/Out) | 30 days (720h) | 2 hours | 6 hours |
| **Odds** (moneyline from The Odds API) | Never fetched | 15 minutes | 4 hours |
| **Weather** (Open-Meteo forecast) | Never fetched | 2 hours | 12 hours |

Files under 500 bytes are always treated as stale (empty/corrupt stubs). Cache staleness is checked via `config.is_cache_stale()`, which delegates to the smart season-aware logic in `cache_utils.py` when available.

### NHL Season Calendar

```
Jan  Feb  Mar  Apr  May  Jun  Jul  Aug  Sep  Oct  Nov  Dec
[------REGULAR SEASON------][PLAYOFFS][---OFFSEASON---][SEASON]
                                      ^                ^
                                 Cup Finals         Season starts
                              (reduced schedule)    (preseason/opening)
```

The NHL season spans two calendar years: October through June. The system uses `year + 1 if month >= 10` for season boundary detection (e.g., October 2025 games belong to the 2025-26 season). Games are played most days during the regular season (October-April), with playoffs running April through June. The caching system treats every day as a potential game day during the active season.

---

## Performance

### Multithreading

The mega-ensemble uses `concurrent.futures.ThreadPoolExecutor` to run all 35 base models in parallel. On typical hardware:

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

Results vary by season and parameter tuning. Typical ranges from the most recent smoke test (3,003 games, 202 goalies tracked):

- **Baseline accuracy**: ~56-57%
- **Platt-calibrated accuracy**: ~56-57%
- **Full mega-ensemble**: 57-60%
- **LogLoss**: ~0.6867
- **Brier score**: ~0.2459
- **ECE** (calibration error): ~0.013
- **BSS vs 50%**: ~0.031

Hockey is inherently one of the hardest major sports to predict. Low-scoring games (average ~3 goals per team), goaltender variance, overtime/shootout randomness, and puck luck mean even the best teams only win ~60-62% of games. The NHL has the highest home-field parity of any major sport. Accuracy above 57% on moneyline picks represents strong performance for hockey.

---

## Complete Model Validation Workflow

A rigorous 6-phase workflow ensures your model is genuinely predictive and not overfit to historical data.

### Phase 1: Baseline & Diagnostics

```
backtest          # Walk-forward accuracy, LogLoss, Brier, ECE
convergence       # How many games before Elo ratings stabilize?
sliding           # Does old data help or hurt? (sliding vs expanding window)
```

Establishes baseline metrics. The `convergence` command identifies the burn-in period (typically 200-400 games for NHL). `sliding` determines whether the model benefits from full history or performs better with a shorter memory. If `sliding` wins, your 33% season regression may be too mild.

### Phase 2: Parameter Optimization

```
grid              # Broad search over K, HomeAdv, PlayerBoost
pbo               # Is the grid search overfit? (PBO > 0.5 = overfit)
results           # Compare all optimizer outputs + Deflated Sharpe Ratio
genetic           # Refine with differential evolution
```

The Probability of Backtest Overfitting (PBO) test is critical here. If PBO > 0.5, your grid search likely found parameters that are overfit to this specific data window. The Deflated Sharpe Ratio (DSR) adjusts for multiple comparisons. NHL-specific grid search uses K range 3-40 and HomeAdv range 10-50.

### Phase 3: Cross-Validation

```
purgedcv          # k-fold with embargo gap (prevents Elo momentum leakage)
cpcv              # All C(k, k_test) paths for tighter confidence intervals
montecarlo        # 500 permutation shuffles -> p-value for significance
```

Purged CV adds an embargo gap between train and test folds to prevent Elo momentum from leaking across boundaries. CPCV produces many more backtest paths for tighter confidence. Monte Carlo gives a p-value: if < 0.05, the model's edge is statistically significant. The permutation test shuffles home/away scores randomly and re-runs the backtest 500 times (~8 minutes).

### Phase 4: Ensemble & Features

```
enhanced          # XGBoost ensemble (96 features, 80/20 blend)
shap              # Which features are driving XGBoost predictions?
enhanced decay    # Time-decayed weighting (XGB gets more weight over season)
```

SHAP analysis reveals whether XGBoost is adding genuine signal beyond Elo or just echoing it. If the top SHAP features are all Elo-derived, the ensemble may not be adding value. NHL-specific features include save percentage differential, goals per game, plus/minus, and goalie Elo differential.

### Phase 5: Calibration

```
rollingcal        # Expanding-window Platt recalibration (truly OOS)
betacal           # 3-parameter beta calibration (handles asymmetry)
conformal         # Distribution-free prediction intervals with coverage
```

Rolling calibration gives truly out-of-sample calibrated metrics. Beta calibration fixes asymmetric miscalibration (e.g., overconfident on favorites but well-calibrated on underdogs). Conformal prediction provides coverage guarantees without distributional assumptions -- singleton prediction sets indicate confident picks, while "both" sets indicate uncertain games.

### Phase 6: P&L Simulation

```
kelly             # Kelly criterion bankroll simulation on backtest
```

Simulates optimal position sizing over the backtest period. Reports final bankroll, maximum drawdown, Sharpe ratio, and win rate. Uses fractional Kelly (default 25%) for practical sizing.

### Decision Framework

| Metric | Good | Marginal | Bad |
|--------|------|----------|-----|
| **Accuracy** | > 58% | 55-58% | < 55% |
| **ECE** (calibration error) | < 0.02 | 0.02-0.05 | > 0.05 |
| **BSS** (Brier Skill Score vs 50%) | > 0.02 | 0.00-0.02 | < 0.00 |
| **PBO** (Prob Backtest Overfit) | < 0.30 | 0.30-0.50 | > 0.50 |
| **DSR** (Deflated Sharpe Ratio) | > 2.0 | 1.0-2.0 | < 1.0 |
| **Monte Carlo p-value** | < 0.05 | 0.05-0.10 | > 0.10 |
| **Purged CV std** | < 2% | 2-4% | > 4% |
| **CPCV paths > 55%** | > 90% | 70-90% | < 70% |
| **Kelly Sharpe** | > 1.0 | 0.5-1.0 | < 0.5 |
| **Kelly max drawdown** | < 20% | 20-40% | > 40% |

**Interpretation**: If most metrics are "Good", the model has genuine predictive power. If PBO is "Bad" or Monte Carlo p > 0.10, the model's apparent edge is likely noise. Do not trade a model with "Bad" validation metrics.

---

## Daily Prediction Workflow

Step-by-step workflow for making daily predictions:

```
1. LAUNCH
   python main.py
   -> Auto-downloads latest games, players, injuries from ESPN
   -> Builds Elo model with per-goalie ratings
   -> Runs baseline backtest (fits Platt scaler)
   -> Shows baseline accuracy

2. CHECK MODEL STATUS
   settings            # Verify parameters are tuned
   platt               # Confirm calibration scaler is fitted
   mega models         # Check which models are enabled
   injuries            # Check goaltender injuries (highest impact)

3. MAKE PREDICTIONS
   today               # Generate HTML table for all today's games
   bruins              # Individual matchup prediction (fuzzy search)
   -> Enter opponent, home team, see calibrated probability
   -> Goaltender impact and injury adjustments shown
   -> 'y' to log as Predicts contract

4. LOG POSITIONS
   balance             # Check account balance
   -> Enter contracts through prediction flow
   predicts            # Review all open positions

5. PUBLISH (optional)
   today               # Generates today_nhl_predictions.html
   blogger             # Same as today -- copy HTML to Blogger

6. MONITOR
   live                # Live score tracker with open trade status
                       # Auto-refreshes every 60 seconds
                       # Shows periods: P1, P2, P3, OT, SO
   odds                # Check latest odds for CLV comparison

7. SETTLE
   autoresolve         # Auto-settle finished games via ESPN scoreboard
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

1. **Entry**: Buy a contract at the model's implied probability (e.g., buy BOS at $0.62)
2. **Entry fee**: 2% of entry price deducted at purchase
3. **Settlement**: Contract resolves to $1.00 (team wins) or $0.00 (team loses). NHL games always have a winner (OT/SO if needed).
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
- `autoresolve` -- Auto-settle using today's game results from the ESPN NHL scoreboard API
- `live` -- Watch live scores with real-time P&L on open trades (shows P1, P2, P3, OT, SO periods)

### Mark-to-Market

Open positions can be marked to current market prices at any time using `mark`. This updates the unrealized P&L without closing the position. The `predicts` summary shows both unrealized (mark-to-market) and realized (settled) P&L.

### Auto-Resolve

When `autoresolve on` is active, the system automatically settles contracts when final game scores are detected during `live` tracking. You can also manually trigger `autoresolve` to batch-settle all finished games. All NHL games have a definitive winner (overtime and shootout produce a final result), so every contract settles cleanly.

---

## File Structure

```
NHL/
|
|-- main.py                     # CLI entry point, command dispatch loop
|-- config.py                   # Constants, 32 NHL teams, 4 divisions, settings I/O
|-- elo_model.py                # NHLElo class (ratings, predictions, 24 adjusters, per-goalie Elo)
|-- build_model.py              # Model training pipeline with season regression + altitude calc
|
|-- data_games.py               # Game data download via ESPN public API
|-- data_players.py             # Skater + goalie leaders download + team scoring
|-- backfill_goalies.py         # NHL-UNIQUE: backfill historical starting goalie data (NHL API)
|
|-- backtest.py                 # All backtesting & optimization (~2100 lines)
|-- enhanced_model.py           # XGBoost ensemble (96 features) + SHAP
|-- single_param_opt.py         # Coordinate descent optimizer
|
|-- platt.py                    # Calibration (Platt, isotonic, beta, regression)
|-- metrics.py                  # LogLoss, Brier, ECE, MCE, BSS, conformal
|
|-- predict_ledger.py           # Predicts $1 contract ledger management
|-- live_scores.py              # Live NHL scores + open trade display (P1/P2/P3/OT/SO)
|-- auto_resolve.py             # Auto-settle finished trades from live scores
|
|-- injuries.py                 # ESPN injury report + Elo impact (goaltender-weighted)
|-- html_generator.py           # Blogger HTML prediction table generation
|-- help_system.py              # Help text for all commands
|-- color_helpers.py            # Colorama terminal formatting utilities
|-- cache_utils.py              # Smart season-aware API caching
|-- elo_set_handler.py          # 'set param=value' command handler (37 params)
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
|-- poisson_model.py            # Poisson / Dixon-Coles goal distribution
|-- glicko_model.py             # Glicko-2 uncertainty-aware ratings
|-- bradley_terry_model.py      # Bradley-Terry MLE paired comparison
|-- monte_carlo_model.py        # Monte Carlo simulation (2000+ sims)
|-- random_forest_model.py      # Random Forest (bagging diversity)
|-- classic_models.py           # SRS, Colley, Log5, PythagenPat, ExpSmooth, MeanReversion
|-- svm_model.py                # SVM classifier (RBF kernel + Platt scaling)
|-- fibonacci_model.py          # Fibonacci retracement analysis
|-- evt_model.py                # Extreme Value Theory tail risk
|-- benford_model.py            # Benford's Law anomaly detection
|
|-- odds_tracker.py             # The Odds API integration + CLV tracking
|-- weather.py                  # Open-Meteo weather impact calculation
|-- kalshi.py                   # Kalshi prediction market integration
|
|-- meta_learner.py             # Ridge/Logistic/XGBoost meta-learner stacker
|-- mega_predictor.py           # Live 35-model ensemble predictor
|-- mega_backtest.py            # Mega-ensemble walk-forward backtest engine
|-- mega_optimizer.py           # 7-phase mega-ensemble optimization
|-- mega_config.py              # Per-model on/off switches + mega params (54 hyperparameters)
|
|-- run_optimize.py             # Optimization runner utilities
|-- quick_optimizer.py          # Quick optimization routines
|-- run_enhanced_all.py         # Batch enhanced model runner
|-- sweep_enhanced.py           # Enhanced model parameter sweeps
|-- accuracy_optimize.py        # Accuracy-focused optimization utilities
|-- accuracy_test.py            # Quick walk-forward accuracy test
|
|-- requirements.txt            # Python dependencies
|-- README.md                   # This file
|-- CLAUDE.md                   # Claude Code agent instructions
|
|-- nhl_elo_settings.json       # [generated] Tuned Elo parameters
|-- nhl_mega_settings.json      # [generated] Mega-ensemble settings + model switches
|-- nhl_recent_games.csv        # [generated] 2 years of game history
|-- nhl_player_stats.csv        # [generated] Skater + goalie leaders
|-- nhl_elo_ratings.json        # [generated] Saved Elo ratings (32 teams)
|-- nhl_platt_scaler.json       # [generated] Platt calibration coefficients
|-- nhl_enhanced_model.json     # [generated] XGBoost metadata
|-- nhl_xgb_model.json          # [generated] XGBoost model weights
|-- nhl_injuries.json           # [generated] Cached injury report (4h TTL)
|-- predicts_lots.csv           # [generated] Trading ledger
```

**Total: 59 Python files** across core engine, 35 model implementations, data pipeline, optimization, trading, and display.

---

## Requirements

### System Requirements

- **Python**: 3.9 or higher (3.8+ compatible -- no walrus operators, no `match` statements)
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

#### NHL Data APIs

```
# No special package needed!
# All NHL data comes from ESPN public API via the standard 'requests' library.
# Game scores, schedules, player stats, goalie stats, injuries -- all from ESPN.
# Goalie backfill uses api-web.nhle.com/v1 (also via requests).
# 0.5s sleep between sequential API calls to be respectful.
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

## Disclaimer

This software is for **educational and research purposes only**. It is not financial advice. Sports prediction models are inherently uncertain -- even the best models are wrong 40-45% of the time for NHL moneyline picks. No model can guarantee profits. Past backtest performance does not predict future results. Always gamble responsibly and never risk money you cannot afford to lose.

The prediction probabilities produced by this system are statistical estimates, not certainties. The Predicts $1 contract ledger is a paper-trading simulation tool, not a connection to any real prediction market or sportsbook.

---

## Recent Changes

- **Fixed `_rolling()` rest_days bug in `elo_model.py`**: The away team was incorrectly using the home team's rest days when constructing XGBoost features. Both teams now use their own rest day values.
- **Fixed broad exception handling in `backtest.py`**: Changed bare `except Exception` to `except OSError` to avoid silently swallowing unexpected errors.
- **Added NaN guard for momentum autocorrelation in `enhanced_model.py`**: Prevents NaN values from propagating through the feature pipeline when autocorrelation is undefined.
- **Made `season_regress` configurable via settings**: The season regression fraction was previously hardcoded to 0.33; it can now be tuned via `set season_regress=<value>` like any other parameter.
- **Optimized Elo parameters from bayesian optimization results**: Updated defaults based on bayesian optimization: K 3.01->5.0, home_adv 29.3->26.0, player_boost 15.43->10.0, starter_boost 4.89->5.0, rest_factor 0.0->12.0, form_weight 6.94->4.0, travel_factor 0.0->8.0, sos_factor 12.48->10.0, playoff_hca_factor 1.3->1.0, pace_factor 30.0->10.0, division_factor 0.0->5.0, mean_reversion 0.0->2.5, b2b_penalty 29.46->18.0, altitude_factor 0.237->4.0, overtime_factor 0.0->5.0.
- **Enabled previously disabled adjusters**: Rest, travel, division, mean_reversion, and overtime factors were previously set to 0.0 (disabled). Bayesian optimization found non-zero values that improve calibration, so these adjusters are now active by default.
- **Fixed `rest_days` default from 1 to 2 in `elo_model.py`**: The `_get_rest_days()` method was defaulting to 1 day of rest when no prior game was found. Changed to 2 days, which better reflects NHL scheduling (teams typically play every other day).
- **Added Utah Hockey Club to ALTITUDE_TEAMS**: Utah Hockey Club (formerly Arizona Coyotes, relocated 2024) plays in Salt Lake City at 4,226 ft. Altitude bonus is now scaled by relative elevation above 4,000 ft.

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
