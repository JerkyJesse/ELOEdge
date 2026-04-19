"""Canonical §7 ablation with tuned + Platt-calibrated Tier 1 baseline.

NBATier2-zeroed with K=8 home_adv=60 (from tier1_report grid search) is
structurally Classic-Elo-Tier-1; we then wrap every factory in Platt so the
baseline and each variant share the same calibration pipeline. Variants use
the same tuned K + home_adv, zero all other adjusters except the one under
test (addition framing, design §7).

Writes to reports/ablation-nba-{YYYYMMDD}-addition_tuned_platt.csv.
"""

from __future__ import annotations

import sys
from typing import Any, Callable, Dict

import numpy as np

from .ablation import run_ablation, write_ablation_csv
from .calibrate import PlattCalibrated
from .data_adapter import load_nba_games
from .nba_tier2 import (
    ALL_ADJUSTERS,
    NBATier2,
    _tier1_settings,
    addition_variant_factories,
    load_base_settings,
)
from .splits import train_test_split


TUNED_K = 8.0
TUNED_HOME_ADV = 60.0


def _tuned_base() -> Dict[str, Any]:
    base = load_base_settings()
    base["k"] = TUNED_K
    base["home_adv"] = TUNED_HOME_ADV
    return base


def _wrap_platt(factory: Callable) -> Callable:
    def _build(f=factory):
        return PlattCalibrated(f())
    return _build


def run() -> None:
    games = load_nba_games(verbose=True)
    train, test, _ = train_test_split(games, verbose=True)
    if len(train) == 0 or len(test) == 0:
        raise SystemExit("no rows in split — populate historical games first")

    base = _tuned_base()
    tier1_settings = _tier1_settings(base)

    baseline_factory = _wrap_platt(
        lambda s=dict(tier1_settings): NBATier2(base_settings=s),
    )
    raw_variants = addition_variant_factories(base_settings=base)
    variants: Dict[str, Callable] = {
        name: _wrap_platt(raw) for name, raw in raw_variants.items()
    }

    print(
        f"\nrun_tuned_platt_ablation: train={len(train)} test={len(test)} "
        f"K={TUNED_K} home_adv={TUNED_HOME_ADV} variants={len(variants)}",
        file=sys.stderr,
    )

    df = run_ablation(
        train, test,
        baseline_factory=baseline_factory,
        variants=variants,
        n_bootstrap=5000,
        direction="addition",
    )
    path = write_ablation_csv(df, sport="nba", mode_suffix="addition_tuned_platt")
    print(df.to_string(index=False))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    run()
