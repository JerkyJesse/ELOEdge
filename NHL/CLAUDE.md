# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

NHL game prediction system combining Elo ratings with an XGBoost ensemble, with Predicts $1 contract trading ledger. Interactive CLI app — no web server, no tests framework, no build system.

## Running

```bash
pip install -r requirements.txt
python main.py
```

First run auto-downloads game data (ESPN API), player stats, and injury reports (ESPN). Then runs a backtest to fit the Platt calibration scaler and enters the interactive CLI loop.

## File Map

**Core pipeline:**
- `main.py` -- CLI entry point, `dispatch()` routes all commands
- `config.py` -- constants, file paths, settings load/save, 32-team TEAM_ABBR dict, NHL_API_ABBR mapping
- `elo_model.py` -- `NHLElo` class: ratings, adjusters, goalie sub-ratings (K_GOALIE=6), win_prob(), update_game()
- `build_model.py` -- constructs NHLElo from settings + game CSV, season regression, altitude bonus calculation
- `backtest.py` -- walk-forward backtest, grid/genetic/bayesian optimizers, advanced validation
- `enhanced_model.py` -- XGBoost ensemble (80/20 Elo+XGB), TeamTracker, SHAP
- `platt.py` -- Platt scaling + isotonic regression + beta calibration
- `metrics.py` -- ECE, MCE, BSS, conformal prediction

**Data & features:**
- `data_games.py` -- ESPN game data download + cache
- `data_players.py` -- ESPN player stats download (skater + goalie stats)
- `injuries.py` -- ESPN injury reports, goalie-weighted Elo impact (star goalie out = -35 Elo)
- `backfill_goalies.py` -- NHL-unique: backfills starting goalie data into game CSV using NHL API boxscores
- `weather.py` -- weather data (less relevant for NHL indoor arenas, but available)
- `odds_tracker.py` -- odds/line tracking

**Mega-ensemble:**
- `mega_predictor.py` -- 35-model mega-ensemble runner (MegaPredictor class)
- `mega_config.py` -- mega settings, model registry, tier definitions
- `mega_backtest.py` -- mega-ensemble walk-forward backtest
- `mega_optimizer.py` -- 7-phase per-model optimization
- `meta_learner.py` -- Ridge/Logistic/XGBoost meta-learner for combining model outputs

**Individual models (mega-ensemble components):**
- `gbm_models.py` -- LightGBM, CatBoost
- `nn_models.py` -- MLP, LSTM neural network models
- `random_forest_model.py` -- Random Forest
- `hmm_model.py` -- Hidden Markov Model
- `kalman_model.py` -- Kalman Filter
- `network_model.py` -- PageRank network model
- `volatility_model.py` -- GARCH volatility
- `signal_model.py` -- Fourier signal decomposition
- `survival_model.py` -- Survival analysis
- `copula_model.py` -- Copula dependency model
- `information_theory_model.py` -- Information theory (entropy)
- `momentum_model.py` -- Momentum/trend model
- `markov_chain_model.py` -- Markov chain transitions
- `clustering_model.py` -- Team clustering
- `game_theory_model.py` -- Game theory (Nash equilibrium)
- `poisson_model.py` -- Poisson goal model
- `glicko_model.py` -- Glicko-2 rating system
- `bradley_terry_model.py` -- Bradley-Terry paired comparisons
- `monte_carlo_model.py` -- Monte Carlo simulation
- `classic_models.py` -- SRS, Colley, Log5, Pythagorean, ExpSmoothing, MeanReversion

**Infrastructure:**
- `color_helpers.py` -- colored output wrappers (cok, cerr, cwarn, chi, cdim, cbold)
- `cache_utils.py` -- unified cache management, staleness checks
- `elo_set_handler.py` -- Elo settings import/export/backup
- `kalshi.py` -- Kalshi prediction market integration
- `live_scores.py` -- ESPN live scoreboard
- `auto_resolve.py` -- auto-settle positions from final scores
- `predict_ledger.py` -- Predicts $1 contract trading ledger
- `html_generator.py` -- Blogger HTML output for predictions
- `help_system.py` -- CLI help text
- `accuracy_test.py` -- standalone quick backtest script

**Optimization utilities:**
- `run_optimize.py`, `quick_optimizer.py`, `single_param_opt.py`, `accuracy_optimize.py`, `sweep_enhanced.py`, `run_enhanced_all.py` -- various optimization scripts

## Architecture

**Two-stage prediction pipeline:**
1. **Elo model** (`elo_model.py` → `NHLElo` class) — base team ratings adjusted for home ice, altitude, player strength, starting goalie quality (per-goalie cumulative Elo, K_GOALIE=6, 50% season regression), rest days, travel fatigue, pace mismatch, injuries, and strength of schedule
2. **XGBoost ensemble** (`enhanced_model.py`) — 80% Elo / 20% XGBoost using 31 rolling features per team (10-game window via `TeamTracker`, includes Pythagorean win expectation, streaks, consistency, and trend)
3. **Platt calibration** (`platt.py`) — logistic regression on raw probabilities for well-calibrated outputs

**Data flow:**
- `data_games.py` / `data_players.py` → download from ESPN API with 6-hour cache (`config.is_cache_stale`), 0.5s sleep between API calls
- `injuries.py` → ESPN JSON API with 4-hour cache; goaltender injuries have outsized impact (~50% of team value)
- `build_model.py` → constructs `NHLElo` from settings + game CSV, applies season regression, sets player scores
- `backtest.py` → walk-forward backtest, also houses grid search and genetic (`scipy.optimize.differential_evolution`) optimizer
- `main.py` → CLI entry point, `dispatch()` routes all commands, team name input triggers prediction flow

**State files (all gitignored, generated at runtime):**
- `nhl_elo_settings.json` — tunable parameters (K-factor, home advantage, etc.)
- `nhl_elo_ratings.json` — current team Elo ratings
- `nhl_platt_scaler.json` / `nhl_isotonic_scaler.json` — calibration scalers
- `nhl_enhanced_model.json` — saved XGBoost model weights
- `predicts_lots.csv` — trade ledger (positions, P&L)

**Key patterns:**
- All settings load/save through `config.py` (`load_elo_settings` / `save_elo_settings`)
- Color output uses wrapper functions in `color_helpers.py` (`cok`, `cerr`, `cwarn`, `chi`, `cdim`, `cbold`), not raw colorama
- The `NHLElo` class stores per-team state in dicts/defaultdicts on the instance (ratings, last game dates, recent results, player scores, etc.)
- Optimizer objective is `LogLoss * 8 + Brier * 40` — this weighting is intentional
- `backtest_model(..., fit_platt=True)` should only be called for user-facing runs, never inside optimizer loops (leakage + speed)
- MOV (margin of victory) uses `log(max(1.0, abs(goal_diff)) + 1.0)` — log compression appropriate for hockey's lower-scoring games
- Two altitude teams: Colorado Avalanche (5280 ft) and Utah Hockey Club (4226 ft, formerly Arizona Coyotes, relocated 2024). Altitude bonus scales by `(elevation - 4000) / (5280 - 4000)` so Utah gets a partial bonus relative to Denver.
- Overtime factor = 5.0 (adjustment for OT game outcomes)
- NHL season spans two calendar years (Oct–Jun); season boundary detection: `year + 1 if month >= 10`

## Backtesting & Optimization

### Walk-Forward Backtest (`backtest` command)

`backtest_model()` in `backtest.py` iterates through every game in `nhl_recent_games.csv` chronologically. For each game it:
1. Calls `model.win_prob()` with `calibrated=False` and `use_injuries=False` (raw Elo only — no future leakage)
2. Records the prediction, then calls `model.update_game()` to update ratings *after* predicting
3. Outputs `nhl_backtest_predictions.csv` (per-game predictions) and `nhl_calibration.csv` (10-bin calibration table)

When `fit_platt=True` (user-facing runs only), it fits a Platt scaler on the full run's raw probabilities and saves to `nhl_platt_scaler.json`. This flag must **never** be used inside optimizer objective functions — it causes data leakage and slows iteration.

Metrics reported: accuracy (%), log loss, Brier score. Calibration table bins predictions into 10 probability buckets and compares predicted vs actual win rates.

### XGBoost Enhanced Backtest (`enhanced` command)

`run_enhanced_backtest()` in `enhanced_model.py` is a separate walk-forward loop that builds an 80/20 Elo+XGBoost ensemble:
- First `min_train` games (default 200): Elo-only predictions while accumulating training features
- After that: XGBoost is trained on accumulated features and retrained every `retrain_every` games (default 50)
- `TeamTracker` maintains rolling 10-game windows (GF/GA per game, win%, margins, rest days) per team
- Feature vector has 31 columns (`FEATURE_COLS` in `enhanced_model.py`): elo_prob, elo_diff, player_diff, per-team rolling stats, differentials, rest, Pythagorean win expectation, streaks, consistency, and trend
- XGBoost params: `max_depth=5, eta=0.03, subsample=0.9, colsample_bytree=0.8, min_child_weight=3, 300 rounds`
- After the walk-forward, fits both Platt and isotonic calibrators on ensemble probabilities
- Saves the trained XGBoost booster to `nhl_xgb_model.json` and metadata to `nhl_enhanced_model.json`

The saved XGBoost model is loaded by `NHLElo._xgb_predict()` at runtime for live predictions, using the team's `_team_scores` deque to construct rolling features on the fly.

### Grid Search (`grid` command)

`grid_search_optimization()` searches 7 parameters via Cartesian product: K, HomeAdv, PlayerBoost, RestFactor, TravelFactor, PaceFactor, PlayoffHCA. Interactive prompts let the user customize min/max/step for each dimension (defaults provided). Each combo builds a fresh `NHLElo`, runs `backtest_model()` (without Platt fitting), and scores with:

```
score = -(LogLoss * 8 + Brier * 40)
```

NHL-specific grid defaults: K range 4-20 (vs NBA 10-50), HomeAdv range 10-50 (vs NBA 20-80). Player scores are precomputed once (`build_league_player_scores`) and altitude bonuses are cached to avoid redundant work across combos. Progress prints every 50 combos and on new bests. Results saved to `nhl_grid_search.csv`. After completion, `_apply_best_settings()` saves the winning params and refits the Platt scaler.

### Genetic Optimizer (`genetic` command)

`genetic_optimization()` uses `scipy.optimize.differential_evolution` over the same 7 parameters. Interactive prompts configure bounds, `maxiter` (default 80), and `popsize` (default 40). NHL-specific bounds: K 3-40 (vs NBA 5-80). Same objective function as grid search. Progress callback prints per-generation best. After convergence, does a final backtest with the best params, saves settings, and refits Platt.

The genetic optimizer is recommended for fine-tuning after a coarse grid search narrows the ranges.

### Standalone Accuracy Test

`python accuracy_test.py [label]` runs a quick walk-forward backtest with current settings, reports raw + Platt-calibrated metrics. Useful for measuring deltas after code changes without entering the full CLI.

### Calibration Methods (`platt.py`)

Two calibrators are available, both implemented without sklearn:
- **Platt scaling**: logistic regression on `logit(raw_prob)` via `scipy.optimize.minimize` (L-BFGS-B). Corrects overconfidence at 0.9+ and underconfidence at 0.1-0.2.
- **Isotonic regression**: Pool Adjacent Violators algorithm on binned probabilities (50 bins default), with linear interpolation for continuous mapping. Fitted during `enhanced` backtest but not currently used for live predictions.

Season regression (`regress_ratings_to_mean`): pulls all team ratings 33% toward the league mean at season boundaries, applied during model building.

## Advanced Backtesting Methods

All 16 advanced backtesting techniques are implemented across `backtest.py`, `enhanced_model.py`, `platt.py`, and `metrics.py`. No additional dependencies beyond the existing stack (numpy, scipy, pandas, xgboost).

### Metrics (auto-reported by `backtest` command)

- **ECE / MCE** (`metrics.py: ece_score, mce_score`) — Expected and Maximum Calibration Error. ECE < 0.03 is excellent; > 0.08 needs work. Reported automatically when `backtest` fits the Platt scaler.
- **BSS** (`metrics.py: brier_skill_score`) — Brier Skill Score vs 50% baseline and vs home-win-rate baseline. BSS > 0 means model beats the reference. Reported automatically by `backtest`.
- **DSR** (`backtest.py: deflated_sharpe_ratio`) — Deflated Sharpe Ratio. Adjusts optimization "best" for multiple-testing bias. DSR < 1.96 means best is not significant at 95%. Auto-shown by `results` command.

### Optimization (`bayesian` command)

**Bayesian optimization** (`backtest.py: bayesian_optimization`) — GP surrogate + Expected Improvement acquisition. Finds optima in ~50-100 evaluations vs thousands for grid search. Uses Latin Hypercube initial sampling, minimal Gaussian Process (squared exponential kernel), and EI acquisition on 1000 random candidates per iteration. Same 7 parameters, same objective, same `_apply_best_settings()` postprocessing. NHL-specific bounds: K 3-40 (vs NBA 5-80).

### Cross-Validation (`purgedcv`, `cpcv` commands)

- **Purged Walk-Forward CV** (`backtest.py: purged_walk_forward_cv`) — k chronological folds with embargo gap. Resets Elo, replays training games, skips embargo, predicts test fold. Reports mean/std across folds. Accuracy variance >3% = fragile model.
- **CPCV** (`backtest.py: combinatorial_purged_cv`) — All C(k, k_test) train/test combinations. With k=5, k_test=2: 10 paths. Tighter confidence intervals. If >90% of paths show >65% accuracy, model is robust.

### Overfitting Detection (`pbo`, `montecarlo` commands)

- **PBO** (`backtest.py: probability_of_backtest_overfitting`) — Symmetric cross-validation on grid search trial population. PBO > 0.5 = optimization is likely overfit. Requires `grid` to have been run.
- **Monte Carlo Permutation** (`backtest.py: monte_carlo_permutation_test`) — Shuffles home/away scores randomly, re-runs backtest N times (default 500), computes p-value. p > 0.05 = model not statistically significant. Takes ~8 min for 500 permutations.

### Calibration (`rollingcal`, `betacal`, `conformal` commands)

- **Rolling Origin Recalibration** (`backtest.py: rolling_origin_recalibration`) — Expanding-window Platt fitting every 50 games. Gives truly out-of-sample calibrated metrics. Compares with single-pass calibration.
- **Beta Calibration** (`platt.py: fit_beta_scaler / apply_beta`, `backtest.py: run_beta_calibration`) — 3-parameter `logit(p_cal) = c + a*log(p) - b*log(1-p)`. If a != b, miscalibration is asymmetric and beta is better than Platt.
- **Conformal Prediction** (`metrics.py: conformal_prediction_set`, `backtest.py: conformal_analysis`) — Distribution-free prediction sets at 90%, 95%, 80% coverage. Singleton = confident, Both = uncertain, Empty = overconfident.

### Ensemble Analysis (`enhanced decay`, `shap` commands)

- **Time-Decayed Weighting** (`enhanced_model.py: run_enhanced_backtest(time_decay=True)`) — Elo weight transitions 95% -> 70% over the season. Acknowledges XGBoost features are noisy early but reliable after 500+ games.
- **SHAP Feature Importance** (`enhanced_model.py: shap_feature_importance`) — XGBoost native `pred_contribs`. Shows mean absolute SHAP per feature. Reveals if XGBoost is echoing Elo or adding signal. Auto-runs after `enhanced`; also standalone `shap` command using saved features.

### Structural Analysis (`sliding`, `convergence` commands)

- **Sliding Window** (`backtest.py: sliding_window_backtest`) — Heavy regression (67%) every N games vs standard expanding window. Determines if old data helps or hurts. If sliding wins, consider more aggressive season regression.
- **Convergence Analysis** (`backtest.py: elo_convergence_analysis`) — Chunks predictions by N-game windows, reports per-chunk accuracy. Finds burn-in period where ratings stabilize. Suggested exclusion window for honest metrics.

### Position Sizing (`kelly` command)

**Kelly Criterion Backtest** (`backtest.py: kelly_criterion_backtest`) — Simulates fractional Kelly position sizing on `nhl_backtest_predictions.csv`. Default quarter-Kelly. Reports final bankroll, return %, max drawdown, annualized Sharpe, win rate. Connects model quality directly to P&L.

### Complete Model Validation Workflow (run in order)

This is the recommended sequence for thoroughly validating and tuning the model. Each step builds on the previous one's output. Skip steps that aren't relevant to your current question, but follow the order for any steps you do run.

**Phase 1: Baseline & Diagnostics**

1. `backtest` — Establish baseline. Note accuracy, log loss, Brier, ECE, MCE, BSS. These are the numbers to beat. The Platt scaler is fitted here and all subsequent predictions use it.
2. `convergence` — How many games before Elo ratings stabilize? If burn-in is 200+ games, your headline accuracy is understated. Know this before interpreting any other metric.
3. `sliding` — Is old data helping or hurting? If sliding window wins, your season regression factor (currently 33%) may be too mild. If expanding wins (typical), full history is valuable and you can proceed with confidence.

**Phase 2: Parameter Optimization**

4. `grid` — Coarse search. Wide ranges, large steps. Produces `nhl_grid_search.csv` with thousands of scored parameter combos. Don't trust the "best" yet — it hasn't been validated.
5. `pbo` — Immediately after grid search. Is the best overfit? PBO > 0.5 = yes, widen your ranges or accept that the data doesn't distinguish between many parameter combos. PBO < 0.3 = the optimum is real.
6. `results` — Check DSR (auto-shown). DSR < 1.96 = the best score isn't statistically distinguishable from random picks among your grid combos. If both PBO and DSR look bad, the grid search found noise, not signal.
7. `genetic` — Fine-tune with tighter bounds around the grid sweet spot. Or use `bayesian` if you want fewer evaluations (~80 vs thousands). Both auto-save best settings and refit Platt.

**Phase 3: Cross-Validation**

8. `purgedcv` — Does the optimized model hold up across 5 chronological folds with embargo? If accuracy std > 3%, the model is fragile — the params are fit to a specific slice of history. Consider more conservative settings.
9. `cpcv` — Combinatorial validation. All 10 paths through 5 blocks. If any path drops below 55% accuracy, there's path-dependent overfitting. The percentage of paths above 65% is your robustness score.
10. `montecarlo` — Is the model genuinely skilled? p-value < 0.05 = yes, the model's accuracy is statistically unlikely to arise from random chance. This is the definitive skill test. Takes ~8 minutes.

**Phase 4: Ensemble & Features**

11. `enhanced` — Train XGBoost on top of optimized Elo. SHAP analysis runs automatically at the end. Check: is `elo_prob` >50% of SHAP signal? If so, XGBoost is mostly echoing Elo and the 20% XGB weight may not be worth the complexity.
12. `shap` — Re-examine feature importance standalone if needed. Look for features with near-zero SHAP — they're candidates for removal to reduce overfitting.
13. `enhanced decay` — Compare time-decayed (Elo 95%->70%) vs fixed 80/20. If decay wins on Brier, use it — it means early-season XGBoost was adding noise.

**Phase 5: Calibration Refinement**

14. `rollingcal` — Is your single-pass Platt calibration honest? Rolling origin gives truly out-of-sample calibrated Brier. If rolling is worse, your Platt scaler is overfitting slightly to the full sample (common, usually by <0.005 Brier).
15. `betacal` — Is there asymmetric miscalibration? If `a != b` significantly (difference > 0.3), the model is overconfident in one direction but not the other. Beta calibration fixes this; Platt cannot. If a ~ b, Platt is sufficient.
16. `conformal` — What's the model's actual confidence? Look at singleton percentage (games where only one outcome is in the prediction set). Higher singleton % with good coverage = model is both accurate and well-calibrated. High "both" % = model is uncertain about many games (maybe don't bet those).

**Phase 6: P&L Simulation**

17. `kelly` — What would optimal position sizing produce? The Sharpe ratio and max drawdown are the numbers that matter for real trading. If max drawdown > 50% even with quarter-Kelly, the edge is too thin for aggressive sizing. If Sharpe > 1.0, there's a tradeable edge.

### Decision Framework After Validation

After running the full workflow, use these thresholds to decide next steps:

| Metric | Good | Marginal | Bad | Action if bad |
|--------|------|----------|-----|---------------|
| Accuracy | >66% | 63-66% | <63% | Revisit Elo adjusters, check data quality |
| ECE | <0.03 | 0.03-0.08 | >0.08 | Refit Platt, try beta calibration |
| BSS vs 50% | >0.08 | 0.04-0.08 | <0.04 | Model has weak discriminative power |
| PBO | <0.3 | 0.3-0.5 | >0.5 | Grid search overfit — use wider ranges or fewer params |
| DSR | >1.96 | 1.0-1.96 | <1.0 | Best params are noise — simplify model |
| Monte Carlo p | <0.01 | 0.01-0.05 | >0.05 | Model has no statistically significant skill |
| Purged CV std | <2% | 2-3% | >3% | Model is fragile to training data selection |
| CPCV paths <55% | 0% | 1-10% | >10% | Path-dependent overfitting |
| Kelly Sharpe | >1.0 | 0.5-1.0 | <0.5 | Edge too thin for real trading |
| Kelly max DD | <30% | 30-50% | >50% | Size down (use 10%-Kelly instead of 25%) |

## Comprehensive Workflow

### Daily Prediction Workflow

1. **Launch**: `python main.py` — auto-downloads fresh data (games, players, injuries), builds Elo model, runs baseline backtest, fits Platt scaler
2. **Check model**: `settings` to verify parameters, `all` to see team rankings, `injuries` to review injury impact
3. **Make predictions**: type a team name (e.g. `Bruins`) → enter opponent → specify home/away (`a`/`b`/`n`) → see calibrated win probability with injury impact and key player stats
4. **Log position**: type `y` → enter contract count, price, optional notes → position saved to `predicts_lots.csv`
5. **Publish**: `today` or `tomorrow` generates Blogger HTML (`today_nhl_predictions.html`) + plain text for all scheduled games
6. **Monitor**: `live` shows real-time scores for open positions; `mark` updates current market prices
7. **Settle**: `resolve` to manually settle (win/loss prompt), `autoresolve` to auto-settle from live final scores, `sell` to exit early at a price
8. **Review**: `predicts` shows full P&L ledger with realized/unrealized/marked totals, win rate, ROI; `chart` saves monthly P&L bar chart

### Model Tuning Workflow

1. **Baseline**: run `backtest` — note accuracy, log loss, Brier, ECE, MCE, BSS (all auto-reported)
2. **Diagnostics**: run `convergence` to find burn-in period, `sliding` to check if old data helps
3. **Coarse search**: run `grid` with wide ranges and large steps to identify promising parameter regions
4. **Validate search**: run `pbo` to check overfitting, `results` to check DSR significance
5. **Fine-tune**: run `genetic` or `bayesian` with tighter bounds around the grid search sweet spot
6. **Cross-validate**: run `purgedcv` for fold stability, `cpcv` for combinatorial robustness
7. **Significance test**: run `montecarlo` — if p > 0.05, model has no proven edge
8. **Train ensemble**: run `enhanced` (SHAP auto-runs), then `enhanced decay` to compare
9. **Calibration**: run `rollingcal` for OOS calibration, `betacal` for asymmetry, `conformal` for coverage
10. **P&L simulation**: run `kelly` to connect model quality to bankroll trajectory
11. **Iterate**: adjust individual params with `set k=8.0`, `set home=25.0`, etc. — always rerun `backtest` after to refit the Platt scaler

See "Complete Model Validation Workflow" above for detailed guidance on interpreting each step's output and what to do when metrics fall short.

After any parameter change, the Platt scaler must be refitted (the CLI warns about this). The optimizer auto-saves best settings and refits Platt after completion.

### Position Management

Predicts contracts are $1 binary options. The ledger tracks:
- **Entry**: contracts count, price per contract, 2% entry fee on potential payout ($1 per contract)
- **Mark-to-market**: `mark` updates current price; `predicts` shows unrealized P&L based on marks
- **Exit**: `sell` for partial/full exit at a price (with 2% exit fee); `resolve` for binary win/loss settlement
- **Invert**: `invert` flips a position's direction (swaps predicted winner, inverts probability) without changing cost basis
- **Auto-resolve**: matches open positions against today's final scores via ESPN NHL scoreboard API; toggle with `autoresolve on/off` (runs on startup when enabled)

### Data Refresh

- `refresh` deletes cached game/player/ratings/scaler files and re-downloads everything, then rebuilds the model
- Data caches expire after 6 hours for games/players, 4 hours for injuries, 60 seconds for live scores
- Cache staleness checked by `config.is_cache_stale()` — files under 500 bytes are always considered stale

## NHL-Specific Notes

- **32 NHL teams** with full name-to-abbreviation mapping in `config.py`; NHL API abbreviations mapped separately via `NHL_API_ABBR`
- **Season structure**: Oct-Jun, spanning two calendar years; season detection uses `year + 1 if month >= 10`
- **No ties**: NHL games always have a winner (overtime/shootout if needed); final score is what matters
- **Injury statuses**: Out, Day-to-Day, IR (Injured Reserve), LTIR (Long-Term Injured Reserve)
- **Player stats**: Points, Goals, Assists for skaters; GAA, SVP, Wins, Shutouts for goalies; composite weight: skaters 55% / goalies 45%
- **Live scores**: ESPN public scoreboard API; period display (P1, P2, P3, OT, SO) instead of quarters
- **ESPN API**: All data from ESPN public JSON API (`site.api.espn.com`); 0.5s sleep between API calls

### Goaltender System

The NHL system's most distinctive feature is its per-goalie cumulative Elo sub-rating system. Because a starting goaltender can single-handedly win or lose a hockey game:

- **K_GOALIE = 6**: Per-goalie Elo update constant (class constant on `NHLElo`)
- **Starter boost = 5.0**: Additional Elo adjustment when a confirmed starter is known
- **50% season regression**: Goalie ratings regress more aggressively than team ratings (33%) at season boundaries, reflecting year-to-year goalie volatility
- **Player composite**: skaters 55% / goalies 45% weight in overall player score
- **Goalie injury impact tiers** (in `injuries.py`):
  - Star goaltender (20+ wins or .910+ SV%): -35 Elo
  - Starting goaltender (10+ wins): -25 Elo
  - Backup goaltender: -15 Elo
- **Backfill tool**: `backfill_goalies.py` (NHL-unique) retroactively fills starting goalie data into `nhl_recent_games.csv` using the NHL API boxscore endpoints

### NHL Default Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| K-factor | 5.0 | Low K for 82-game season (gradual rating changes) |
| Home advantage | 26.0 | ~52% NHL home win rate |
| Player boost | 10.0 | Impact of player quality differential |
| Starter boost | 5.0 | Known starting goalie advantage |
| Rest factor | 12.0 | Rest days impact |
| B2B penalty | 18.0 | Back-to-back game penalty (significant in hockey) |
| Travel factor | 8.0 | Cross-country travel fatigue |
| SOS factor | 10.0 | Strength of schedule adjustment |
| Pace factor | 10.0 | Pace mismatch adjustment |
| Division factor | 5.0 | Divisional rivalry familiarity |
| Overtime factor | 5.0 | OT game outcome adjustment |
| Road trip factor | 2.5 | Extended road trip penalty |
| Homestand factor | 3.0 | Extended homestand bonus |
| Win streak factor | 2.0 | Hot/cold streak adjustment |
| Season phase factor | 2.5 | Early/mid/late season adjustment |
| Altitude factor | 4.0 | Multiplier for altitude bonus (Colorado, Utah) |
| Season regression | 33% | Pull ratings toward mean at season boundaries |

## CLI Commands

Enter a team name to start a prediction. Core commands:

**Data & Display**: `all`, `refresh`, `players`, `settings`, `injuries`, `today`/`html`, `tomorrow`

**Backtesting**: `backtest`, `enhanced`, `enhanced decay`

**Optimization**: `grid`, `genetic`, `bayesian`, `autoopt`, `superopt`, `results`

**Validation**: `purgedcv`, `cpcv`, `pbo`, `montecarlo`, `convergence`, `sliding`

**Calibration**: `rollingcal`, `betacal`, `conformal`

**Analysis**: `shap`, `kelly`

**Trading**: `predicts`, `resolve`, `sell`, `mark`, `live`, `chart`, `invert`, `autoresolve`

**Settings**: `set k=8.0`, `set home=25`, `set boost=20`, `set rest=15`, `set travel=20`, `set sos=10`, `set pace=20`, `set playoff=0.6`, `set form=0`

`help` for overview, `help <command>` for details, `help advanced` for all validation commands, `quit` to exit.

## Mega-Ensemble

The `mega_predictor.py` module implements a 35-model mega-ensemble predictor. Each model runs independently on the same walk-forward game loop, producing a raw probability. A meta-learner (Ridge, Logistic, or XGBoost) combines all 31 outputs into a single calibrated adjustment that is clamped to +/- `max_adj` (default 0.10) around the Elo anchor probability.

**Key commands:**
- `mega` -- Run full mega-ensemble backtest with all enabled models
- `mega optimize` -- 7-phase per-model exhaustive optimization (54 hyperparameters)
- `mega tune` -- Per-model solo optimization (Phase 1 only)
- `mega tournament` -- Head-to-head model comparison (Phase 2 only)
- `mega ablation` -- Ablation study: test each model's contribution, auto-prune bad ones
- `mega models` -- Show all 35 models with ON/OFF status
- `mega on/off <model>` -- Enable/disable individual models
- `mega settings` -- Show all mega parameter values
- `mega set <param>=<value>` -- Set mega parameters (e.g., `mega set adj=0.10`)

**Model tiers:** Tier 0 (Elo, XGBoost), Tier 1 (HMM, Kalman, PageRank, LightGBM, CatBoost, MLP, LSTM), Tier 2 (GARCH, Fourier, Survival, Copula), Tier 3 (InfoTheory, Momentum, Markov, Clustering, GameTheory), Tier 4 (Poisson, Glicko-2, Bradley-Terry, Monte Carlo, Random Forest), Tier 5 (SRS, Colley, Log5, Pythagorean, ExpSmoothing, MeanReversion), Tier 6 (Weather, Odds).

All 35 models run in parallel via `ThreadPoolExecutor`. Settings stored in `nhl_mega_settings.json`.

## Conventions

- Python 3.8+ compatibility (no walrus operators, no `match` statements)
- No test framework — validation is via `backtest` command and `accuracy_test.py` (run manually: `python accuracy_test.py [label]`)
- All data files use `nhl_` prefix; temp files use `temp_` prefix (both gitignored)
- Settings are tuned via `set param=value` CLI command or optimizer; changes require `backtest` rerun to refit Platt scaler
- Color output uses `color_helpers.py` wrappers (`cok`, `cerr`, `cwarn`, `chi`, `cdim`, `cbold`), never raw colorama
- Optimizer objective is `LogLoss * 8 + Brier * 40` — this weighting is intentional and should not be changed casually
- Team lookup is fuzzy: `NHLElo.find_team()` accepts full names, abbreviations, partial matches, and close matches via `difflib.get_close_matches`
- All ESPN API calls use `requests` library with 0.5s sleep between sequential requests
