"""OpenSkill (Weng-Lin / Plackett-Luce) via openskill.py. 1v1 wrapper.

Design §3.4 wanted riix.PlackettLuce; riix 0.0.6 exposes the Bradley-Terry /
Thurstone-Mosteller variants via `riix.models.weng_lin.WengLin(model=...)`,
not Plackett-Luce. We use openskill.py directly for Plackett-Luce which is
lighter weight and faithful to Weng-Lin 2011 JMLR.

Ref: Weng & Lin 2011, JMLR, https://jmlr.org/papers/v12/weng11a.html
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from openskill.models import PlackettLuce

from .base import BenchmarkRating


class OpenSkillBench(BenchmarkRating):
    name = "openskill"
    tier = 0

    def __init__(
        self,
        mu: float = 25.0,
        sigma: float = 25.0 / 3.0,
        beta: float = 25.0 / 6.0,
    ) -> None:
        self.mu0 = float(mu)
        self.sigma0 = float(sigma)
        self.beta = float(beta)
        self._model = PlackettLuce(mu=self.mu0, sigma=self.sigma0, beta=self.beta)
        self._ratings: Dict[str, Any] = {}

    def _team(self, name: str):
        if name not in self._ratings:
            self._ratings[name] = self._model.rating(name=name)
        return self._ratings[name]

    def fit(self, games: pd.DataFrame) -> None:
        self._check_schema(games)
        g = games.sort_values("date").reset_index(drop=True)
        self._ratings = {}
        for row in g.itertuples(index=False):
            ra = self._team(row.home)
            rb = self._team(row.away)
            home_won = row.home_score > row.away_score
            # Pass in WIN-ORDER: [[winner], [loser]]
            if home_won:
                new = self._model.rate([[ra], [rb]])
                self._ratings[row.home] = new[0][0]
                self._ratings[row.away] = new[1][0]
            else:
                new = self._model.rate([[rb], [ra]])
                self._ratings[row.away] = new[0][0]
                self._ratings[row.home] = new[1][0]

    def predict(self, home: str, away: str, date: Any = None) -> float:
        ra = self._team(home)
        rb = self._team(away)
        probs = self._model.predict_win([[ra], [rb]])
        return float(probs[0])
