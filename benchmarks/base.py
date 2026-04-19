"""BenchmarkRating ABC. Every rating system implements fit(games) then predict(home, away, date).

No-leakage rule (design §4b): predict for game g must only use games with date < g.date.
Callers (splits.py, ablation.py) enforce; implementations trust the contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = ("date", "season", "home", "away", "home_score", "away_score")


class BenchmarkRating(ABC):
    """Abstract rating system.

    Canonical games frame (see design §4b):
      columns = date (datetime64 or ISO str), season (int),
                home (str), away (str), home_score (int), away_score (int)
      sorted ascending by date.
    """

    name: str = "benchmark_rating"
    tier: int = 0

    @abstractmethod
    def fit(self, games: pd.DataFrame) -> None:
        """Fit ratings on `games`. Must validate schema via `_check_schema`."""

    @abstractmethod
    def predict(self, home: str, away: str, date: Any) -> float:
        """Return P(home wins) at `date`. Uses only data with date < `date`."""

    @staticmethod
    def _check_schema(games: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in games.columns]
        if missing:
            raise KeyError(f"games missing required columns: {missing}")
