"""Glicko-2 (Glickman 2013) via riix. Thin wrapper handling name↔index mapping
and ISO-week periods so ratings update on natural time cadence.

riix 0.0.6 API is index-based: competitors list at init; batched_update takes
(matchups[N,2] int indices, outcomes[N] float in {0, 0.5, 1}, time_step: int).

Note: design §5 pinned `riix>=0.2.0,<0.3`; current PyPI max is 0.0.6, so the
bench requirements file reflects reality. Wrapper is one-file-blast-radius
per design §9.

Ref: Glickman 2013, http://www.glicko.net/glicko/glicko2.pdf
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from riix.models.glicko2 import Glicko2 as _RiixGlicko2

from .base import BenchmarkRating


class Glicko2Bench(BenchmarkRating):
    """Glicko-2. No home_adv in core algorithm (Tier 0)."""

    name = "glicko2"
    tier = 0

    def __init__(
        self,
        tau: float = 0.5,
        initial_rating: float = 1500.0,
        initial_rd: float = 350.0,
        initial_sigma: float = 0.06,
    ) -> None:
        self.tau = float(tau)
        self.initial_rating = float(initial_rating)
        self.initial_rd = float(initial_rd)
        self.initial_sigma = float(initial_sigma)
        self._model = None
        self._team_to_idx: Dict[str, int] = {}
        self._idx_to_team: List[str] = []

    def _build_model(self, teams) -> None:
        self._idx_to_team = sorted(set(teams))
        self._team_to_idx = {t: i for i, t in enumerate(self._idx_to_team)}
        self._model = _RiixGlicko2(
            competitors=self._idx_to_team,
            initial_rating=self.initial_rating,
            initial_rd=self.initial_rd,
            initial_sigma=self.initial_sigma,
            tau=self.tau,
            update_method="batched",
        )

    def fit(self, games: pd.DataFrame) -> None:
        self._check_schema(games)
        g = games.sort_values("date").reset_index(drop=True)
        teams = set(g["home"]).union(set(g["away"]))
        self._build_model(teams)
        if len(g) == 0:
            return
        d = pd.to_datetime(g["date"])
        iso = d.dt.isocalendar()
        period_key = (iso["year"].astype(int) * 100 + iso["week"].astype(int)).to_numpy()
        unique_periods = np.unique(period_key)
        period_to_step = {int(p): i for i, p in enumerate(unique_periods)}

        matchups = np.array(
            [
                [self._team_to_idx[h], self._team_to_idx[a]]
                for h, a in zip(g["home"], g["away"])
            ],
            dtype=np.int64,
        )
        outcomes = (
            g["home_score"].to_numpy() > g["away_score"].to_numpy()
        ).astype(np.float64)

        for p in unique_periods:
            mask = period_key == p
            step = period_to_step[int(p)]
            self._model.batched_update(
                matchups=matchups[mask],
                outcomes=outcomes[mask],
                time_step=step,
            )

    def predict(self, home: str, away: str, date: Any = None) -> float:
        if (
            self._model is None
            or home not in self._team_to_idx
            or away not in self._team_to_idx
        ):
            return 0.5
        matchups = np.array(
            [[self._team_to_idx[home], self._team_to_idx[away]]], dtype=np.int64
        )
        probs = self._model.predict(matchups=matchups)
        return float(probs[0])
