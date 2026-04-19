"""Train/test split + min-games floor (design §4b).

Frozen config (seed=20260418):
  TRAIN_SEASONS   = 2015..2022 (inclusive, starting-year convention)
  TEST_SEASONS    = 2023, 2024
  MIN_GAMES_FLOOR = 10

Min-games floor semantics: teams with fewer than `min_games` appearances IN THE
TRAIN split are excluded from BOTH splits. Any game whose home OR away team is
excluded is dropped BEFORE block assembly (design §4b) so block sizes stay
uniform at "one ISO calendar week within a season".

If seed changes, design §4b forbids silent re-split: bump SPLITS_VERSION and
ablation CSV gets a new date suffix.
"""

from __future__ import annotations

import sys
from typing import Tuple

import pandas as pd


SPLITS_SEED = 20260418
SPLITS_VERSION = 1
TRAIN_SEASONS = (2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022)
TEST_SEASONS = (2023, 2024)
MIN_GAMES_FLOOR = 10


def _counts_per_team(df: pd.DataFrame) -> pd.Series:
    return pd.concat([df["home"], df["away"]], ignore_index=True).value_counts()


def train_test_split(
    games: pd.DataFrame,
    min_games: int = MIN_GAMES_FLOOR,
    train_seasons: Tuple[int, ...] = TRAIN_SEASONS,
    test_seasons: Tuple[int, ...] = TEST_SEASONS,
    verbose: bool = True,
):
    """Split canonical games frame -> (train_df, test_df, stats_dict).

    Min-games floor applied to TRAIN counts only; both splits are then filtered
    to games whose home AND away teams clear the floor. Dropped rows never
    reach block assembly.
    """
    for col in ("date", "season", "home", "away"):
        if col not in games.columns:
            raise KeyError(f"games missing column: {col}")

    train_raw = games[games["season"].isin(train_seasons)].copy()
    test_raw = games[games["season"].isin(test_seasons)].copy()

    counts = _counts_per_team(train_raw)
    kept_teams = set(counts[counts >= min_games].index)
    dropped_teams = set(counts[counts < min_games].index)
    # Teams that appear ONLY in test (never in train) are also dropped:
    test_only_teams = (set(test_raw["home"]) | set(test_raw["away"])) - set(counts.index)
    dropped_teams |= test_only_teams

    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        m = df["home"].isin(kept_teams) & df["away"].isin(kept_teams)
        return df[m].sort_values("date").reset_index(drop=True)

    train = _filter(train_raw)
    test = _filter(test_raw)

    stats = {
        "seed": SPLITS_SEED,
        "version": SPLITS_VERSION,
        "train_seasons": list(train_seasons),
        "test_seasons": list(test_seasons),
        "min_games": int(min_games),
        "train_before": int(len(train_raw)),
        "train_after": int(len(train)),
        "test_before": int(len(test_raw)),
        "test_after": int(len(test)),
        "teams_kept": sorted(kept_teams),
        "teams_dropped": sorted(dropped_teams),
    }

    if verbose:
        dropped_str = sorted(dropped_teams) if dropped_teams else "(none)"
        print(
            f"splits.train_test_split: train {stats['train_before']}->{stats['train_after']} "
            f"| test {stats['test_before']}->{stats['test_after']} "
            f"| min_games={min_games} drops teams: {dropped_str}",
            file=sys.stderr,
        )

    return train, test, stats
