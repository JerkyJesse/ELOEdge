"""Command-line entry point (design §4, §8.1).

Usage
-----
python -m Claude.benchmarks.cli --model classic_elo --tier 0 [--json]
python -m Claude.benchmarks.cli --model classic_elo --tier 1
python -m Claude.benchmarks.cli --model elopp
python -m Claude.benchmarks.cli --model glicko2
python -m Claude.benchmarks.cli --model openskill
python -m Claude.benchmarks.cli --model whr
python -m Claude.benchmarks.cli --ablation two-sys --baseline classic_elo --variant elopp

Output: one-line human-readable summary + --json flag for machine-readable.
See `--help` for full option list.

Reads the canonical games frame from Claude/NBA/nba_recent_games.csv. If the
frame does not cover the design train/test seasons, the CLI emits a warning
and runs on whatever coverage exists (the harness is still exercised).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd

from .base import BenchmarkRating
from .bootstrap import assign_blocks, block_bootstrap_delta
from .classic_elo import ClassicEloTier0, ClassicEloTier1
from .data_adapter import load_nba_games
from .elopp import EloPlusPlus
from .glicko2 import Glicko2Bench
from .openskill_rate import OpenSkillBench
from .scoring import brier, crps, ece, log_loss, mce, sharpness
from .splits import train_test_split
from .whr import WHRBatchRefitWeekly


def _make_model(name: str, tier: int) -> BenchmarkRating:
    name = name.lower()
    if name == "classic_elo":
        return ClassicEloTier1() if tier == 1 else ClassicEloTier0()
    if name == "elopp":
        return EloPlusPlus()
    if name == "glicko2":
        return Glicko2Bench()
    if name == "openskill":
        return OpenSkillBench()
    if name == "whr":
        return WHRBatchRefitWeekly()
    raise ValueError(f"unknown model: {name}")


def _predict_on_frame(model: BenchmarkRating, frame: pd.DataFrame) -> np.ndarray:
    preds = np.empty(len(frame), dtype=float)
    for i, row in enumerate(frame.itertuples(index=False)):
        preds[i] = float(model.predict(row.home, row.away, row.date))
    return np.clip(preds, 1e-9, 1.0 - 1e-9)


def _eval_summary(model_name: str, preds: np.ndarray, frame: pd.DataFrame) -> Dict[str, Any]:
    y = (frame["home_score"].to_numpy() > frame["away_score"].to_numpy()).astype(float)
    return {
        "model": model_name,
        "n_games": int(len(preds)),
        "log_loss": round(log_loss(y, preds), 6),
        "brier": round(brier(y, preds), 6),
        "crps": round(crps(y, preds), 6),
        "ece": round(ece(y, preds), 6),
        "mce": round(mce(y, preds), 6),
        "sharpness": round(sharpness(preds), 6),
    }


def _load_and_split(json_mode: bool):
    games = load_nba_games(verbose=not json_mode)
    try:
        train, test, _ = train_test_split(games, verbose=not json_mode)
    except KeyError as e:
        raise SystemExit(f"data_adapter returned bad frame: {e}")

    if len(train) == 0 or len(test) == 0:
        if not json_mode:
            print(
                "[cli] design train/test split empty for this CSV coverage; "
                "falling back to time-based 80/20 within-data split.",
                file=sys.stderr,
            )
        seasons = sorted(games["season"].unique().tolist())
        if len(seasons) >= 2:
            test_season = seasons[-1]
            train = games[games["season"] < test_season].reset_index(drop=True)
            test = games[games["season"] == test_season].reset_index(drop=True)
        else:
            cutoff = int(len(games) * 0.8)
            train = games.iloc[:cutoff].reset_index(drop=True)
            test = games.iloc[cutoff:].reset_index(drop=True)
    return train, test


def _cmd_single(args: argparse.Namespace) -> int:
    train, test = _load_and_split(args.json)

    model = _make_model(args.model, args.tier)
    model.fit(train)
    preds = _predict_on_frame(model, test)
    summary = _eval_summary(f"{args.model}_t{args.tier}", preds, test)
    summary["train_rows"] = int(len(train))
    summary["test_rows"] = int(len(test))

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"[{summary['model']}] n={summary['n_games']} "
            f"log_loss={summary['log_loss']:.4f} brier={summary['brier']:.4f} "
            f"ece={summary['ece']:.4f} sharpness={summary['sharpness']:.4f}"
        )
    return 0


def _cmd_ablation(args: argparse.Namespace) -> int:
    from .ablation import run_ablation, write_ablation_csv

    train, test = _load_and_split(args.json)

    if args.tier2_adjusters:
        from .nba_tier2 import (
            NBATier2,
            addition_variant_factories,
            one_off_variant_factories,
            tier1_baseline_factory,
        )
        if args.tier2_mode == "addition":
            baseline_factory = tier1_baseline_factory()
            factories = addition_variant_factories()
            direction = "addition"
        else:
            baseline_factory = lambda: NBATier2()
            factories = one_off_variant_factories()
            direction = "leave_one_out"
        if args.only:
            wanted = set(args.only.split(","))
            factories = {k: v for k, v in factories.items() if k in wanted}
            if not factories:
                print(f"no adjusters match --only {args.only}", file=sys.stderr)
                return 4
        df = run_ablation(
            train, test,
            baseline_factory=baseline_factory,
            variants=factories,
            n_bootstrap=args.bootstrap,
            direction=direction,
        )
    else:
        baseline = lambda: _make_model(args.baseline, tier=1)
        variant = lambda: _make_model(args.variant, tier=1)
        df = run_ablation(
            train, test, baseline_factory=baseline,
            variants={args.variant: variant},
            n_bootstrap=args.bootstrap,
        )
    suffix = args.tier2_mode if args.tier2_adjusters else None
    path = write_ablation_csv(df, sport="nba", mode_suffix=suffix)
    if args.json:
        print(json.dumps({"csv_path": path, "rows": df.to_dict(orient="records")}, indent=2))
    else:
        print(df.to_string(index=False))
        print(f"\nwrote {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m Claude.benchmarks.cli",
        description=(
            "Benchmark harness CLI. Runs rating systems + scoring + ablation "
            "against the canonical NBA games frame. See design sections 4 and 8."
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=False)

    p.add_argument("--model", default="classic_elo",
                   choices=["classic_elo", "elopp", "glicko2", "openskill", "whr"],
                   help="rating system to eval")
    p.add_argument("--tier", type=int, default=1, choices=[0, 1],
                   help="tier: 0=pure Arpad, 1=+home_adv (see design sec.3)")
    p.add_argument("--json", action="store_true", help="emit JSON summary")

    abl = sub.add_parser(
        "ablation",
        help="two-system block-bootstrap comparison (design sec.7)",
    )
    abl.add_argument("--baseline", default="classic_elo")
    abl.add_argument("--variant", default="elopp")
    abl.add_argument("--bootstrap", type=int, default=10000)
    abl.add_argument("--json", action="store_true")
    abl.add_argument(
        "--tier2-adjusters", action="store_true",
        help="run all NBA Tier 2 adjusters (design sec.8.4)",
    )
    abl.add_argument(
        "--tier2-mode", default="addition", choices=["addition", "leave_one_out"],
        help="addition: Tier1 baseline vs Tier1+X variant (canonical sec.7 gate). "
             "leave_one_out: Tier2-full vs Tier2-minus-X.",
    )
    abl.add_argument(
        "--only", default=None,
        help="comma-separated adjuster names to run (default: all)",
    )

    imp = sub.add_parser(
        "import-bref",
        help="import basketball-reference schedule CSVs -> nba_recent_games.csv",
    )
    imp.add_argument("files", nargs="+", help="one or more Basketball-Reference CSV paths")
    imp.add_argument(
        "--output", default=None,
        help="target upstream CSV (default: Claude/NBA/nba_recent_games.csv)",
    )
    imp.add_argument(
        "--dry-run", action="store_true",
        help="parse + summarize only, do not write",
    )

    bf = sub.add_parser(
        "backfill-nba-api",
        help="fetch historical games via stats.nba.com LeagueGameLog (uses Claude/NBA/nba_http.py shim)",
    )
    bf.add_argument(
        "--start-year", type=int, default=2015,
        help="first season starting year (2015 = 2015-16 season)",
    )
    bf.add_argument(
        "--end-year", type=int, default=2024,
        help="last season starting year INCLUSIVE (2024 = 2024-25 season)",
    )
    bf.add_argument("--pause", type=float, default=1.0, help="seconds between fetches")
    bf.add_argument(
        "--output", default=None,
        help="target upstream CSV (default: Claude/NBA/nba_recent_games.csv)",
    )
    bf.add_argument(
        "--dry-run", action="store_true",
        help="fetch + summarize only, do not write",
    )

    return p


def _cmd_import_bref(args: argparse.Namespace) -> int:
    from .bref_import import (
        merge_with_existing,
        parse_many,
        write_upstream_csv,
    )
    from .data_adapter import NBA_GAMES_CSV

    target = args.output or NBA_GAMES_CSV
    parsed = parse_many(args.files)
    print(
        f"bref_import: parsed total {len(parsed)} games, "
        f"seasons {sorted(parsed['season'].unique().tolist())}",
        file=sys.stderr,
    )
    if args.dry_run:
        print("[dry-run] not writing. Pass --output or omit --dry-run to persist.")
        return 0

    combined = merge_with_existing(parsed, target)
    print(
        f"bref_import: merged {len(combined)} total games after dedup; "
        f"writing to {target}",
        file=sys.stderr,
    )
    write_upstream_csv(combined, target)
    return 0


def _cmd_backfill_nba_api(args: argparse.Namespace) -> int:
    from .bref_import import merge_with_existing, write_upstream_csv
    from .data_adapter import NBA_GAMES_CSV
    from .nba_api_backfill import backfill

    seasons = list(range(args.start_year, args.end_year + 1))
    fetched = backfill(seasons=seasons, pause_seconds=args.pause)
    if len(fetched) == 0:
        print("nba_api_backfill: fetched zero games — aborting write.", file=sys.stderr)
        return 4

    # Normalize to bref_import's column set so merge_with_existing works
    fetched = fetched.rename(columns={"home_team": "home", "away_team": "away"})
    fetched["season"] = fetched["date"].dt.year.where(
        fetched["date"].dt.month >= 10, fetched["date"].dt.year - 1
    ).astype(int)

    target = args.output or NBA_GAMES_CSV
    if args.dry_run:
        print(
            f"[dry-run] fetched {len(fetched)} games, seasons "
            f"{sorted(fetched['season'].unique().tolist())} — not writing."
        )
        return 0

    combined = merge_with_existing(fetched, target)
    print(
        f"nba_api_backfill: merged {len(combined)} total games after dedup; "
        f"writing to {target}",
        file=sys.stderr,
    )
    write_upstream_csv(combined, target)
    return 0


def main(argv=None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    if args.cmd == "ablation":
        return _cmd_ablation(args)
    if args.cmd == "import-bref":
        return _cmd_import_bref(args)
    if args.cmd == "backfill-nba-api":
        return _cmd_backfill_nba_api(args)
    return _cmd_single(args)


if __name__ == "__main__":
    raise SystemExit(main())
