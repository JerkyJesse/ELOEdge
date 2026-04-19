"""Leave-one-adjuster-out ablation harness (design §7, §8.4).

Generic: takes a BASELINE rating class and a dict of `variant_name -> factory`
callables producing alternate-configured rating systems. For each variant:
    1. fit(train) on both baseline and variant
    2. predict held-out test rows -> probability vector
    3. run block_bootstrap_delta(variant, baseline, y, weeks) per design §6
    4. apply the §7 gate: Δlog-loss ≥ 0.003 AND p<0.05 ⇒ KEEP else CUT

Output: DataFrame with columns (adjuster, delta_logloss, ci_lo, ci_hi, p_value,
verdict), optionally written to CSV at `Claude/benchmarks/reports/`.

This module is GENERIC by design. Plug in the production NBAElo class with
one-adjuster-zeroed variants via the `variants` mapping to reproduce §8.4. The
Tier 2 code itself remains read-only from this module (design §3 / §10).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Iterable

import numpy as np
import pandas as pd

from .bootstrap import assign_blocks, block_bootstrap_delta
from .scoring import log_loss


# Gate thresholds from design §7 (stated once there; referenced here).
GATE_DELTA_THRESHOLD = 0.003
GATE_P_THRESHOLD = 0.05


@dataclass
class AblationRow:
    adjuster: str
    delta_logloss: float
    ci_lo: float
    ci_hi: float
    p_value: float
    verdict: str
    baseline_logloss: float
    variant_logloss: float


def _predict_all(model, test: pd.DataFrame) -> np.ndarray:
    preds = np.empty(len(test), dtype=float)
    for i, row in enumerate(test.itertuples(index=False)):
        preds[i] = float(model.predict(row.home, row.away, row.date))
    return np.clip(preds, 1e-9, 1.0 - 1e-9)


def _verdict(delta: float, p_value: float, direction: str = "addition") -> str:
    """Gate verdict per design sec.7 thresholds.

    direction = "addition": baseline + adjuster vs baseline. KEEP iff variant
        is significantly better (Δ ≤ -0.003 AND p < 0.05).
    direction = "leave_one_out": baseline-full vs baseline-minus-adjuster.
        KEEP iff removing the adjuster makes things significantly WORSE
        (Δ ≥ +0.003 AND p < 0.05) — i.e. adjuster earns its keep.
    """
    if direction == "addition":
        if delta <= -GATE_DELTA_THRESHOLD and p_value < GATE_P_THRESHOLD:
            return "KEEP"
        return "CUT"
    if direction == "leave_one_out":
        if delta >= GATE_DELTA_THRESHOLD and p_value < GATE_P_THRESHOLD:
            return "KEEP"
        return "CUT"
    raise ValueError(f"unknown direction: {direction}")


def run_ablation(
    train: pd.DataFrame,
    test: pd.DataFrame,
    baseline_factory: Callable[[], Any],
    variants: Dict[str, Callable[[], Any]],
    n_bootstrap: int = 10000,
    seed: int = 20260418,
    direction: str = "addition",
) -> pd.DataFrame:
    """Run per-adjuster ablation per design sec.7.

    Parameters
    ----------
    train : canonical games frame (see base.REQUIRED_COLUMNS).
    test : canonical games frame; used for the held-out prediction + bootstrap.
    baseline_factory : zero-arg callable producing a fresh BenchmarkRating.
    variants : {adjuster_name -> zero-arg callable} producing rating systems
        to compare against baseline.
    n_bootstrap : resamples for block_bootstrap_delta.
    seed : bootstrap seed.
    direction : "addition" (baseline + adjuster vs baseline) or "leave_one_out"
        (baseline-full vs baseline-minus-adjuster). Affects verdict only.

    Returns a DataFrame sorted by delta_logloss ascending (best variant first).
    """
    if len(test) == 0:
        raise ValueError("test set is empty")

    # Baseline fit + predict
    baseline = baseline_factory()
    baseline.fit(train)
    base_preds = _predict_all(baseline, test)

    y = (test["home_score"].to_numpy() > test["away_score"].to_numpy()).astype(float)
    blocks = assign_blocks(test["date"], test["season"])["block_id"].to_numpy()
    base_ll = log_loss(y, base_preds)

    rows = []
    for name, factory in variants.items():
        variant = factory()
        variant.fit(train)
        var_preds = _predict_all(variant, test)
        var_ll = log_loss(y, var_preds)
        # block_bootstrap_delta(a, b): Δ = LL(a) - LL(b). Put variant as 'a'
        # so Δ<0 means variant better than baseline.
        center, lo, hi, p_val = block_bootstrap_delta(
            var_preds, base_preds, y, blocks, n=n_bootstrap, seed=seed
        )
        rows.append(
            AblationRow(
                adjuster=name,
                delta_logloss=float(center),
                ci_lo=float(lo),
                ci_hi=float(hi),
                p_value=float(p_val),
                verdict=_verdict(center, p_val, direction=direction),
                baseline_logloss=float(base_ll),
                variant_logloss=float(var_ll),
            )
        )

    df = pd.DataFrame([r.__dict__ for r in rows]).sort_values(
        "delta_logloss"
    ).reset_index(drop=True)
    return df


def write_ablation_csv(
    df: pd.DataFrame,
    sport: str = "nba",
    reports_dir: str | None = None,
    mode_suffix: str | None = None,
) -> str:
    """Write the ablation table to
    Claude/benchmarks/reports/ablation-{sport}-{YYYYMMDD}[-{mode}].csv.

    Filename date = today (run date) per design §8.4. `mode_suffix` (e.g.
    "addition" or "leave_one_out") optional — distinguishes multiple runs on
    the same date.
    """
    if reports_dir is None:
        reports_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    suffix = f"-{mode_suffix}" if mode_suffix else ""
    path = os.path.join(reports_dir, f"ablation-{sport}-{stamp}{suffix}.csv")
    df.to_csv(path, index=False)
    return path
