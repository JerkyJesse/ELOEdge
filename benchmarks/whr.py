"""WHR batch-refit-weekly (design §3.5).

NOT the MAP Whole-History-Rating of Coulom 2008. Faithful batch variant:
for each week W in the fit set, re-run Elo++ from scratch on all games with
date < W_start and snapshot the resulting ratings. Prediction for a game in
week W uses the snapshot captured at the start of W.

Between weekly snapshots the ratings are frozen — no live MAP update. This
is the "stable ratings by period" model design §3.5 explicitly asked for.

Ref: Coulom, "Whole-History Rating", 2008, https://www.remi-coulom.fr/WHR/WHR.pdf
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from .base import BenchmarkRating
from .elopp import EloPlusPlus


def _week_key(dates: pd.Series) -> np.ndarray:
    d = pd.to_datetime(dates)
    iso = d.dt.isocalendar()
    return (iso["year"].astype(int) * 100 + iso["week"].astype(int)).to_numpy()


class WHRBatchRefitWeekly(BenchmarkRating):
    name = "whr_batch_weekly"
    tier = 1

    def __init__(
        self,
        home_adv: float = 100.0,
        lr: float = 0.01,
        epochs: int = 10,
        reg: float = 0.02,
        initial: float = 1500.0,
    ) -> None:
        self.home_adv = float(home_adv)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.reg = float(reg)
        self.initial = float(initial)
        self._snapshots: Dict[int, Dict[str, float]] = {}
        self._snapshot_weeks: np.ndarray = np.empty(0, dtype=int)
        self._all_games: pd.DataFrame = pd.DataFrame()

    def _fit_snapshot(self, subset: pd.DataFrame) -> Dict[str, float]:
        m = EloPlusPlus(
            lr=self.lr,
            epochs=self.epochs,
            reg=self.reg,
            home_adv=self.home_adv,
            initial=self.initial,
            val_frac=0.10,
        )
        m.fit(subset)
        return dict(m.ratings)

    def fit(self, games: pd.DataFrame) -> None:
        self._check_schema(games)
        g = games.sort_values("date").reset_index(drop=True).copy()
        g["_wk"] = _week_key(g["date"])
        self._all_games = g
        unique_weeks = np.unique(g["_wk"].to_numpy())
        self._snapshots = {}
        for wk in unique_weeks:
            prior = g[g["_wk"] < wk]
            if len(prior) == 0:
                self._snapshots[int(wk)] = {}
            else:
                self._snapshots[int(wk)] = self._fit_snapshot(prior)
        self._snapshot_weeks = unique_weeks

    def _lookup_week(self, date: Any) -> int:
        wk = int(_week_key(pd.Series([date]))[0])
        # Find greatest snapshot_week <= wk
        candidates = self._snapshot_weeks[self._snapshot_weeks <= wk]
        if len(candidates) == 0:
            # If no prior week, fall back to earliest snapshot (initial ratings).
            return int(self._snapshot_weeks[0]) if len(self._snapshot_weeks) else -1
        return int(candidates.max())

    def predict(self, home: str, away: str, date: Any = None) -> float:
        if date is None or len(self._snapshot_weeks) == 0:
            return 0.5
        wk = self._lookup_week(date)
        if wk < 0:
            return 0.5
        snap = self._snapshots.get(wk, {})
        rh = snap.get(home, self.initial)
        ra = snap.get(away, self.initial)
        diff = (rh + self.home_adv - ra) / 400.0 * np.log(10.0)
        return float(1.0 / (1.0 + np.exp(-np.clip(diff, -500.0, 500.0))))
