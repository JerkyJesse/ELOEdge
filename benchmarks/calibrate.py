"""Probability calibration wrappers: Platt (logistic) + isotonic.

Standard Elo ratings produce overconfident probabilities — ECE typically
0.05-0.15 even on well-tuned K/home_adv. Fitting a monotone mapping from
raw_p to calibrated_p on a hold-out slice of train, then applying at predict
time, collapses overconfidence without changing rank ordering.

Usage:
    from .classic_elo import ClassicEloTier1
    from .calibrate import PlattCalibrated
    model = PlattCalibrated(ClassicEloTier1(k=8, home_adv=100))
    model.fit(train)          # internally: time-based split, fit Elo on head,
                              #   fit Platt on tail, refit Elo on all train
    model.predict("Boston Celtics", "LA Lakers", date)

Two flavors:
- PlattCalibrated: two-param sigmoid A * logit(p) + B. Fast, parametric,
  robust on small calibration sets.
- IsotonicCalibrated: monotone piecewise-constant step function via
  sklearn.isotonic.IsotonicRegression. Flexible, needs more data.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .base import BenchmarkRating


def _logit(p: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


def fit_platt(p_raw: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Platt scaling: find (A, B) minimizing log-loss of sigmoid(A*logit(p)+B)."""
    x = _logit(np.asarray(p_raw, dtype=float))
    y = np.asarray(y, dtype=float)

    def nll(params):
        a, b = params
        p = _sigmoid(a * x + b)
        p = np.clip(p, 1e-12, 1.0 - 1e-12)
        return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())

    res = minimize(nll, x0=[1.0, 0.0], method="Nelder-Mead")
    return float(res.x[0]), float(res.x[1])


def apply_platt(p_raw: float | np.ndarray, a: float, b: float) -> np.ndarray:
    x = _logit(np.asarray(p_raw, dtype=float))
    return _sigmoid(a * x + b)


class PlattCalibrated(BenchmarkRating):
    """Wrap any BenchmarkRating with Platt calibration fit on the tail of train.

    The inner model is trained twice: once on `head_frac` of train to produce
    calibration predictions against the tail, once on the full train for
    deployment. Platt params (A, B) are fit on (p_tail, y_tail)."""

    name = "calibrated_platt"
    tier = 1

    def __init__(self, inner: BenchmarkRating, tail_frac: float = 0.20) -> None:
        self.inner = inner
        self.tail_frac = float(tail_frac)
        self.a = 1.0
        self.b = 0.0
        self._fitted = False

    def fit(self, games: pd.DataFrame) -> None:
        self._check_schema(games)
        g = games.sort_values("date").reset_index(drop=True)
        n_tail = max(50, int(len(g) * self.tail_frac))
        head = g.iloc[: len(g) - n_tail].reset_index(drop=True)
        tail = g.iloc[len(g) - n_tail :].reset_index(drop=True)

        self.inner.fit(head)
        p_tail = np.array(
            [self.inner.predict(r.home, r.away, r.date) for r in tail.itertuples(index=False)]
        )
        y_tail = (tail["home_score"].to_numpy() > tail["away_score"].to_numpy()).astype(float)
        self.a, self.b = fit_platt(p_tail, y_tail)

        # Refit on full train so prediction-time ratings reflect everything
        self.inner.fit(g)
        self._fitted = True

    def predict(self, home: str, away: str, date: Any = None) -> float:
        raw = self.inner.predict(home, away, date)
        return float(apply_platt(raw, self.a, self.b))
