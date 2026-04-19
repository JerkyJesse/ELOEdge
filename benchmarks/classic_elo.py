"""Classic Elo (Arpad 1978). Tier 0 canonical + Tier 1 +home_adv (design §3).

Tier 0: K, c=400, logistic. No home_adv, no MOV. Immovable floor.
Tier 1: Tier 0 plus a single additive home-court term (Elo units). NBA has no
starter/goalie split to model, so home_adv IS the sport-structural term.

Reference: Arpad E. Elo, *The Rating of Chess Players, Past and Present*, Arco, 1978.
Logistic form (c=400) per FIDE.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from .base import BenchmarkRating


def _expected_home(r_home: float, r_away: float, home_adv: float = 0.0,
                   c: float = 400.0) -> float:
    exponent = (r_away - r_home - home_adv) / c
    return 1.0 / (1.0 + 10.0 ** exponent)


class ClassicEloTier0(BenchmarkRating):
    """Arpad canonical: K, c=400, logistic. No home_adv. No MOV. No decay."""

    name = "classic_elo_t0"
    tier = 0

    def __init__(self, k: float = 20.0, c: float = 400.0, initial: float = 1500.0) -> None:
        self.k = float(k)
        self.c = float(c)
        self.initial = float(initial)
        self.home_adv = 0.0
        self.ratings: Dict[str, float] = {}
        self._fitted_through: Any = None

    def _get(self, team: str) -> float:
        return self.ratings.get(team, self.initial)

    def _update_single(self, home: str, away: str, home_won: float) -> None:
        rh = self._get(home)
        ra = self._get(away)
        exp_h = _expected_home(rh, ra, home_adv=self.home_adv, c=self.c)
        self.ratings[home] = rh + self.k * (home_won - exp_h)
        self.ratings[away] = ra + self.k * ((1.0 - home_won) - (1.0 - exp_h))

    def fit(self, games: pd.DataFrame) -> None:
        self._check_schema(games)
        self.ratings = {}
        g = games.sort_values("date")
        for row in g.itertuples(index=False):
            home_won = 1.0 if row.home_score > row.away_score else 0.0
            self._update_single(row.home, row.away, home_won)
        self._fitted_through = g["date"].max() if len(g) else None

    def predict(self, home: str, away: str, date: Any = None) -> float:
        """P(home wins) from current ratings. Caller enforces no-leakage by
        fitting ONLY on games with date < `date` before calling."""
        return float(_expected_home(self._get(home), self._get(away),
                                    home_adv=self.home_adv, c=self.c))

    def predict_sequence(self, games: pd.DataFrame) -> np.ndarray:
        """Walk games in date order. For each game, predict using ratings
        reflecting ALL prior games only, then update from the observed outcome.
        Returns p_home array aligned to the date-sorted input.

        This is the standard online-Elo no-leakage eval path. Use for computing
        held-out probabilities without an external evaluation loop.
        """
        self._check_schema(games)
        self.ratings = {}
        g = games.sort_values("date").reset_index(drop=True)
        preds = np.empty(len(g), dtype=float)
        for i, row in enumerate(g.itertuples(index=False)):
            preds[i] = float(_expected_home(self._get(row.home), self._get(row.away),
                                            home_adv=self.home_adv, c=self.c))
            home_won = 1.0 if row.home_score > row.away_score else 0.0
            self._update_single(row.home, row.away, home_won)
        self._fitted_through = g["date"].max() if len(g) else None
        return preds


class ClassicEloTier1(ClassicEloTier0):
    """Tier 0 plus a single home_adv term (Elo units). Sport-structural Tier 1."""

    name = "classic_elo_t1"
    tier = 1

    def __init__(self, k: float = 20.0, c: float = 400.0, initial: float = 1500.0,
                 home_adv: float = 100.0) -> None:
        super().__init__(k=k, c=c, initial=initial)
        self.home_adv = float(home_adv)
