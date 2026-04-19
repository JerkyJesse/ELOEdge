"""Paired block-bootstrap Δ log-loss (design §6, §4b).

Blocks = ISO calendar week (Mon–Sun) within the same season. Season boundary is
a HARD block boundary: no block spans seasons. Resamples whole blocks with
replacement, preserving intra-block autocorrelation.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def assign_blocks(dates, seasons) -> pd.DataFrame:
    """Map each game to its (season, ISO week) block.

    Returns a DataFrame aligned row-wise with `dates`/`seasons`, columns:
      block_id (int 0..K-1, dense) | season (int) | iso_year (int) | iso_week (int)

    Composite key = "{season}|{iso_year}|W{iso_week:02d}". iso_year is captured
    because late-Dec dates can belong to ISO year+1; without it, `season` alone
    still disambiguates but debug output loses the ISO label.
    """
    d = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    s = pd.Series(seasons).reset_index(drop=True).astype(int)
    if len(d) != len(s):
        raise ValueError("dates and seasons must align")
    iso = d.dt.isocalendar()
    iso_year = iso["year"].astype(int)
    iso_week = iso["week"].astype(int)
    key = (
        s.astype(str)
        + "|"
        + iso_year.astype(str)
        + "|W"
        + iso_week.astype(str).str.zfill(2)
    )
    codes, _ = pd.factorize(key, sort=True)
    return pd.DataFrame(
        {
            "block_id": codes.astype(int),
            "season": s.values,
            "iso_year": iso_year.values,
            "iso_week": iso_week.values,
        }
    )


def _logloss_per_game(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def block_bootstrap_delta(
    probs_a,
    probs_b,
    y,
    block_ids,
    n: int = 10000,
    seed: int = 20260418,
) -> Tuple[float, float, float, float]:
    """Paired block-bootstrap of Δ log-loss = logloss(a) − logloss(b).

    Sign convention: Δ > 0 ⇒ model a has HIGHER log-loss ⇒ model a WORSE.
    Sample size-K bootstrap draws K blocks with replacement (K = number of unique
    block ids) and recomputes mean Δ on the concatenated resample.

    Returns (center, ci_lo_95, ci_hi_95, p_value_two_sided).
    p-value uses the percentile-at-zero convention:
        if center ≥ 0: p = 2·P(Δ* ≤ 0)
        else:          p = 2·P(Δ* ≥ 0)
    clipped to [1/n, 1]. For degenerate A=B (all Δ=0) returns p = 1.
    """
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    pa = np.asarray(probs_a, dtype=float).reshape(-1)
    pb = np.asarray(probs_b, dtype=float).reshape(-1)
    bids = np.asarray(block_ids, dtype=int).reshape(-1)
    if not (len(y_arr) == len(pa) == len(pb) == len(bids)):
        raise ValueError(
            f"length mismatch: y={len(y_arr)} a={len(pa)} b={len(pb)} blocks={len(bids)}"
        )
    if len(y_arr) == 0:
        raise ValueError("empty input")

    la = _logloss_per_game(y_arr, pa)
    lb = _logloss_per_game(y_arr, pb)
    diff = la - lb

    order = np.argsort(bids, kind="stable")
    sorted_bids = bids[order]
    change = np.where(np.diff(sorted_bids) != 0)[0] + 1
    boundaries = np.r_[0, change, len(sorted_bids)]
    k = len(boundaries) - 1
    block_to_idx = [order[boundaries[i] : boundaries[i + 1]] for i in range(k)]

    rng = np.random.default_rng(seed)
    deltas = np.empty(n, dtype=float)
    for i in range(n):
        picks = rng.integers(0, k, size=k)
        sample_idx = np.concatenate([block_to_idx[p] for p in picks])
        deltas[i] = diff[sample_idx].mean()

    center = float(diff.mean())
    ci_lo = float(np.quantile(deltas, 0.025))
    ci_hi = float(np.quantile(deltas, 0.975))

    if np.allclose(deltas, 0.0) and center == 0.0:
        p_val = 1.0
    else:
        if center >= 0:
            p_val = 2.0 * float((deltas <= 0).mean())
        else:
            p_val = 2.0 * float((deltas >= 0).mean())
        p_val = min(1.0, max(1.0 / n, p_val))

    return center, ci_lo, ci_hi, p_val
