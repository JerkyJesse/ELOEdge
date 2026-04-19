"""Scoring metrics — the ruler (design §6).

Proper scoring rules: log_loss, brier, crps.
Calibration diagnostics: ece, mce.
Skill comparison: sharpness, brier_skill_score (per-season home-win-rate reference).

CRPS for binary forecasts has closed form (p - y)^2, which equals Brier.
Parity-tested against properscoring.crps_ensemble to 1e-6 (see tests/).
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd


def _as_arrays(y_true, p_pred):
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(p_pred, dtype=float).reshape(-1)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y={y.shape} p={p.shape}")
    return y, p


def log_loss(y_true, p_pred, eps: float = 1e-12) -> float:
    y, p = _as_arrays(y_true, p_pred)
    p = np.clip(p, eps, 1.0 - eps)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def brier(y_true, p_pred) -> float:
    y, p = _as_arrays(y_true, p_pred)
    return float(((p - y) ** 2).mean())


def crps(y_true, p_pred) -> float:
    """Binary CRPS. Closed form for Bernoulli(p) vs binary obs y: (p - y)^2.
    Equals Brier; kept distinct for API clarity and future non-binary extension."""
    y, p = _as_arrays(y_true, p_pred)
    return float(((p - y) ** 2).mean())


def sharpness(p_pred) -> float:
    """std(predictions). Higher = sharper. Distinguishes sharp-accurate from dull-accurate."""
    p = np.asarray(p_pred, dtype=float).reshape(-1)
    if p.size == 0:
        return 0.0
    return float(p.std(ddof=0))


def _bin_edges(bins: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, bins + 1)


def ece(y_true, p_pred, bins: int = 10) -> float:
    """Expected Calibration Error: weighted average per-bin |mean_pred − mean_actual|."""
    y, p = _as_arrays(y_true, p_pred)
    n = len(p)
    if n == 0:
        return 0.0
    edges = _bin_edges(bins)
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        nb = int(mask.sum())
        if nb == 0:
            continue
        total += (nb / n) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(total)


def mce(y_true, p_pred, bins: int = 10) -> float:
    """Maximum Calibration Error: worst per-bin gap."""
    y, p = _as_arrays(y_true, p_pred)
    if len(p) == 0:
        return 0.0
    edges = _bin_edges(bins)
    worst = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        worst = max(worst, abs(float(p[mask].mean()) - float(y[mask].mean())))
    return float(worst)


def season_home_win_rate(games: pd.DataFrame) -> Mapping[int, float]:
    """Per-season home-win rate from the canonical games frame.
    Used to build the reference vector for brier_skill_score (design §6)."""
    for col in ("season", "home_score", "away_score"):
        if col not in games.columns:
            raise KeyError(f"games missing column: {col}")
    rates: dict[int, float] = {}
    for season, sub in games.groupby("season"):
        y = (sub["home_score"] > sub["away_score"]).astype(float)
        rates[int(season)] = float(y.mean())
    return rates


def brier_skill_score(
    y_true,
    p_pred,
    reference_probs: Optional[Sequence[float]] = None,
) -> float:
    """BSS = 1 − Brier_model / Brier_reference. BSS > 0 ⇒ model beats reference.

    Design §6: default reference is per-season home-win-rate, supplied by caller
    (`reference_probs` aligned row-wise to y_true). If None, falls back to 0.5
    constant (no-skill); callers wanting the §6 standard MUST pass the per-season
    vector built via season_home_win_rate()."""
    y, p = _as_arrays(y_true, p_pred)
    brier_m = brier(y, p)
    if reference_probs is None:
        ref = np.full_like(p, 0.5)
    else:
        ref = np.asarray(reference_probs, dtype=float).reshape(-1)
        if ref.shape != y.shape:
            raise ValueError(f"reference_probs shape {ref.shape} != y shape {y.shape}")
    brier_r = float(((ref - y) ** 2).mean())
    if brier_r < 1e-12:
        return 0.0
    return float(1.0 - brier_m / brier_r)
