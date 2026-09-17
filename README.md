# ELOEdge

> Pure Elo. Four sports. Calibrated. Open.

ELOEdge is an Elo-only prediction engine for the four major North American sports leagues. No XGBoost ensembles, no LightGBM stacks, no neural-net add-ons. Just Arpad Elo (1959) plus 22+ sport-specific adjusters, gated by walk-forward block-bootstrap p-values, squashed through Platt scaling for honest win probabilities.

**Live site:** [jerkyjesse.github.io/ELOEdge](https://jerkyjesse.github.io/ELOEdge/)

## What's inside

| Sport | K | Home Adv | Adjusters | Distinctive |
|-------|---|----------|-----------|-------------|
| **NBA** | ~6.2 | 34.4 | 22 | Altitude (Denver), B2B penalty, pace mismatch |
| **NFL** | ~8.9 | 14.0 | 24 | Weather (wind/temp/precip), bye-week recovery, QB injury |
| **MLB** | ~1.5 | 32.4 | 31 | Starting pitcher Elo, park factors, Pythagorean blend |
| **NHL** | ~4.4 | 27.8 | 23 | Per-goalie Elo (K=6), 50% season regression, win streaks |

Plus a shared `benchmarks/` module that wraps the production Elo model in a `BenchmarkRating` ABC and runs head-to-head ablations against Elo++, Glicko-2, OpenSkill, and WHR (NBA-only at v1, full sport rollout planned).

## Layout

```
ELOEdge/
├── NBA/          # Self-contained NBA module (data, model, CLI, tests)
├── NFL/          # Self-contained NFL module
├── MLB/          # Self-contained MLB module
├── NHL/          # Self-contained NHL module
├── benchmarks/   # Cross-system rating harness (Classic, Elo++, Glicko-2, OpenSkill, WHR)
├── tests/        # Cross-cutting integration tests
├── docs/         # GitHub Pages site (sports-themed)
└── portfolio.py  # Cross-sport bankroll + Kelly sizing
```

## Quick start

```bash
git clone https://github.com/JerkyJesse/ELOEdge.git
cd ELOEdge/NBA   # pick a sport
pip install -r requirements.txt
python main.py
```

Then in the interactive CLI:

```
> refresh        # pull two seasons of games + injuries
> backtest       # walk-forward train + Platt fit
> today          # generate today's predictions HTML
> all            # ranked teams by Elo
> grid           # 7-param grid search
> genetic        # differential evolution
```

Non-interactive optimization:

```bash
python run_optimize.py        # auto: grid → genetic → super-optimize
python master_optimize.py     # everything sequentially
```

## Benchmarks (NBA v1)

```bash
cd ELOEdge
python -m Claude.benchmarks.cli --model classic_elo --tier 1 --json
python -m Claude.benchmarks.cli ablation --tier2-adjusters --tier2-mode addition --bootstrap 10000
```

§7 gate keeps an adjuster only if `Δ log-loss ≤ -0.003` AND `p < 0.05`. Otherwise it gets cut. Receipts or bust.

## Why Elo-only

The mega-ensemble experiment (Elo + XGBoost + meta-learner + SHAP) was stripped because the gradient-boost layers were not beating raw Platt-calibrated Elo on out-of-sample log-loss after honest walk-forward CPCV. The added complexity bought zero edge but doubled training time, broke reproducibility, and hid which adjusters were actually load-bearing.

ELOEdge is the version that survived the cuts.

## License

Dual-licensed: [AGPL-3.0-or-later](LICENSE), or a [commercial license](COMMERCIAL.md) for use without the AGPL's source-disclosure obligations. Contact: mechapip@mechapip.com

## Credits

- Arpad Elo (1959) -- original rating system
- John Platt (1999) -- probabilistic calibration via logistic regression
- nba_api, sportsdataverse, MLB Stats API -- data sources
