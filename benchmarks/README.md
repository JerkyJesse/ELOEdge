# Benchmarks module

Reference rating-system harness for NBA Elo evaluation (design v3,
`~/.cavestack/projects/PredictMarkets1.0/jerky-unknown-design-20260418-151757.md`).

**NBA-only, v1.** MLB/NFL/NHL are explicit Phase 2.

## What's here

| File | Role |
|------|------|
| `base.py` | `BenchmarkRating` ABC (`fit` / `predict`) |
| `scoring.py` | log_loss, brier, crps, ece, mce, sharpness, bss |
| `bootstrap.py` | `assign_blocks`, `block_bootstrap_delta` (ISO-week / season blocks) |
| `data_adapter.py` | `load_nba_games()` canonicalizes `Claude/NBA/nba_recent_games.csv` |
| `splits.py` | frozen train 2015-2022 / test 2023-2024 split, min-games floor |
| `classic_elo.py` | Tier 0 Arpad + Tier 1 `+home_adv` |
| `elopp.py` | Elo++ (Sismanis 2010) SGD + L2 + early-stop |
| `glicko2.py` | Glicko-2 via `riix` |
| `openskill_rate.py` | OpenSkill (Weng-Lin / Plackett-Luce) via `openskill.py` |
| `whr.py` | WHR batch-refit-weekly (approximation, not online MAP) |
| `ablation.py` | per-adjuster block-bootstrap harness + §7 gate (addition or leave-one-out) |
| `nba_tier2.py` | wraps production `NBAElo` as a `BenchmarkRating`; 22-adjuster variant factories |
| `cli.py` | `python -m Claude.benchmarks.cli` entry point |

## Running

```bash
# Single system, human-readable summary
python -m Claude.benchmarks.cli --model classic_elo --tier 1

# Machine-readable summary
python -m Claude.benchmarks.cli --model elopp --json

# Two-system block-bootstrap comparison
python -m Claude.benchmarks.cli ablation --baseline classic_elo --variant elopp --bootstrap 10000

# Full 22-adjuster NBA Tier 2 ablation (canonical §7 gate)
python -m Claude.benchmarks.cli ablation --tier2-adjusters --tier2-mode addition --bootstrap 10000

# Leave-one-out form (Tier 2 full vs Tier 2 minus X)
python -m Claude.benchmarks.cli ablation --tier2-adjusters --tier2-mode leave_one_out

# Single adjuster quick check
python -m Claude.benchmarks.cli ablation --tier2-adjusters --only use_mov,pace_factor
```

Outputs the ablation CSV at `Claude/benchmarks/reports/ablation-nba-YYYYMMDD.csv`.

## How to add a new rating system

1. Subclass `BenchmarkRating` in a new file under `Claude/benchmarks/`.
2. Implement `fit(games)` (batch, sorted by date) and `predict(home, away, date)`.
   The `games` frame has columns `date, season, home, away, home_score, away_score`.
3. Register the factory in `cli._make_model`.
4. Add a smoke test to `tests/test_rating_systems_smoke.py` (valid probabilities,
   rating reflects training outcomes).
5. If the system has sport-structural parameters (home advantage, starter/goalie
   splits), expose them as `__init__` kwargs — Tier 1 is defined by what you
   add to Tier 0; the production Tier 2 (22 adjusters) is layered separately.

## Interpreting `block_bootstrap_delta`

```
center, ci_lo, ci_hi, p_value = block_bootstrap_delta(probs_a, probs_b, y, block_ids)
```

- **Δ log-loss** (`center`) = log_loss(a) − log_loss(b). **Sign convention: Δ > 0
  means model a is WORSE** (higher log-loss). For ablation (baseline vs variant),
  negative Δ means variant beats baseline.
- **95% CI** `[ci_lo, ci_hi]` from the percentile bootstrap over 10k resamples
  (default). Resamples whole ISO-week-within-season blocks with replacement,
  preserving intra-block autocorrelation.
- **p-value** is a two-sided percentile-at-zero p. Degenerate A≡B returns p=1
  (cannot reject no-difference). Otherwise p ≈ twice the bootstrap-tail mass on
  the opposite side of 0 from `center`.
- Block boundary: no block ever spans seasons. ISO week within season.

**§7 gate for an adjuster:** KEEP iff `delta_logloss ≤ −0.003` AND `p_value < 0.05`
(variant strictly better, with significance). Otherwise CUT — adjuster logged
with `_ablation_verdict` in `nba_elo_settings.json`, NOT code-deleted (see design
§3 / §10).

## Install

```bash
pip install -r Claude/benchmarks/requirements-bench.txt
```

Does NOT touch top-level `requirements.txt` (design §5).

## Coverage caveat (2026-04-18)

`Claude/NBA/nba_recent_games.csv` currently holds only the 2025-26 season. The
design split (train 2015-2022 / test 2023-2024) therefore yields zero rows; the
CLI falls back to an 80/20 time-based within-data split and prints a warning.

## Populating historical games from Basketball-Reference

For each season you need (eg. 2015-16 through 2023-24 = seasons 2015..2023),
open the Basketball-Reference schedule page, then **Share & Export → Get
table as CSV → download**:

| Season | URL |
|--------|-----|
| 2015-16 | https://www.basketball-reference.com/leagues/NBA_2016_games.html |
| 2016-17 | https://www.basketball-reference.com/leagues/NBA_2017_games.html |
| ...    | ...                                                             |
| 2023-24 | https://www.basketball-reference.com/leagues/NBA_2024_games.html |
| 2024-25 | https://www.basketball-reference.com/leagues/NBA_2025_games.html |

(URL pattern: `NBA_{ENDING_YEAR}_games.html`. Grab **every month tab** in the
schedule — BR paginates by month.)

Save the CSVs anywhere (eg. `~/Downloads/nba_sched/NBA_2023_games.csv`) then:

```bash
# Dry run to verify parsing + row counts before writing
python -m Claude.benchmarks.cli import-bref \
    ~/Downloads/nba_sched/*.csv --dry-run

# Write into Claude/NBA/nba_recent_games.csv (deduped, preserves existing rows)
python -m Claude.benchmarks.cli import-bref \
    ~/Downloads/nba_sched/*.csv
```

Historical team-rename aliases (Charlotte Bobcats → Hornets, New Orleans
Hornets → Pelicans, Seattle SuperSonics → Oklahoma City Thunder, New Jersey
Nets → Brooklyn Nets) are applied automatically so BR's older spellings merge
cleanly with the current-franchise names in the upstream CSV.

Once historical games are in place the design split (train 2015-2022 /
test 2023-2024) will yield real rows and the canonical ablation run is:

```bash
python -m Claude.benchmarks.cli ablation --tier2-adjusters --tier2-mode addition --bootstrap 10000
```
