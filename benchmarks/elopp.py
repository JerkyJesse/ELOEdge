"""Elo++ (Sismanis 2010). SGD over (home, away, outcome) with L2 regularization
toward a global mean rating, early-stopping on a held-out tail validation split.

Defaults (design §3.2): lr=0.01, epochs=20, reg=0.02, early-stop patience=3.

Ref: Sismanis, "How I won the Kaggle chess ratings competition", 2010.
    https://www.kaggle.com/competitions/chess/discussion/253
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from .base import BenchmarkRating


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


class EloPlusPlus(BenchmarkRating):
    """Elo++ with Tier 1 home-adv term by default."""

    name = "elopp"
    tier = 1

    def __init__(
        self,
        lr: float = 0.01,
        epochs: int = 20,
        reg: float = 0.02,
        home_adv: float = 100.0,
        initial: float = 1500.0,
        scale: float = 400.0,
        val_frac: float = 0.10,
        patience: int = 3,
        seed: int = 20260418,
    ) -> None:
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.reg = float(reg)
        self.home_adv = float(home_adv)
        self.initial = float(initial)
        self.scale = float(scale)
        self.val_frac = float(val_frac)
        self.patience = int(patience)
        self.seed = int(seed)
        self.ratings: Dict[str, float] = {}
        self._epochs_run = 0

    def _get(self, t: str) -> float:
        return self.ratings.get(t, self.initial)

    def _p_home(self, h: str, a: str) -> float:
        rh, ra = self._get(h), self._get(a)
        return float(_sigmoid((rh + self.home_adv - ra) / self.scale * np.log(10.0)))

    def fit(self, games: pd.DataFrame) -> None:
        self._check_schema(games)
        g = games.sort_values("date").reset_index(drop=True)
        n = len(g)
        if n == 0:
            self.ratings = {}
            self._epochs_run = 0
            return

        n_val = max(1, int(self.val_frac * n))
        train = g.iloc[: n - n_val].reset_index(drop=True)
        val = g.iloc[n - n_val :].reset_index(drop=True)

        teams = set(g["home"]).union(set(g["away"]))
        self.ratings = {t: self.initial for t in teams}

        rng = np.random.default_rng(self.seed)
        best_val = float("inf")
        plateau = 0
        best_ratings = dict(self.ratings)

        train_home = train["home"].to_numpy()
        train_away = train["away"].to_numpy()
        train_y = (train["home_score"].to_numpy() > train["away_score"].to_numpy()).astype(float)
        t_log10 = np.log(10.0)

        for epoch in range(self.epochs):
            idx = rng.permutation(len(train))
            for i in idx:
                h = train_home[i]
                a = train_away[i]
                y = train_y[i]
                rh = self.ratings[h]
                ra = self.ratings[a]
                p = _sigmoid((rh + self.home_adv - ra) / self.scale * t_log10)
                err = y - p
                # Gradient on rh: +scale*err; on ra: -scale*err. L2 toward initial.
                self.ratings[h] = rh + self.lr * (
                    self.scale * err - self.reg * (rh - self.initial)
                )
                self.ratings[a] = ra + self.lr * (
                    -self.scale * err - self.reg * (ra - self.initial)
                )
            val_ll = self._val_logloss(val)
            if val_ll + 1e-6 < best_val:
                best_val = val_ll
                plateau = 0
                best_ratings = dict(self.ratings)
            else:
                plateau += 1
                if plateau >= self.patience:
                    break
            self._epochs_run = epoch + 1
        self.ratings = best_ratings

    def _val_logloss(self, val: pd.DataFrame) -> float:
        if len(val) == 0:
            return float("inf")
        p = np.array(
            [self._p_home(row.home, row.away) for row in val.itertuples(index=False)]
        )
        y = (val["home_score"].to_numpy() > val["away_score"].to_numpy()).astype(float)
        p = np.clip(p, 1e-12, 1.0 - 1e-12)
        return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())

    def predict(self, home: str, away: str, date: Any = None) -> float:
        return float(self._p_home(home, away))
