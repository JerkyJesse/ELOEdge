"""Alternate data sources for NBA data fetchers.

Each adapter returns data normalized to the canonical on-disk schema so
swapping sources does not require downstream changes in backtest.py,
build_model.py, or the Elo engine.
"""

# Canonical column contract for GAMES_FILE (nba_recent_games.csv).
# Verified against downstream consumers (backtest.py, build_model.py,
# accuracy_test.py, single_param_opt.py) on 2026-04-18.
# backtest.py defaults neutral_site=False when missing, so it is effectively
# optional but we always emit it for consistency.
GAMES_CANONICAL_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "neutral_site",
]


class FallbackUnavailable(RuntimeError):
    """Raised by an alternate source when it cannot serve a request
    (missing API key, rate-limit exhausted, schema mismatch, etc.).

    The caller should log and either try the next source in the chain
    or return stale cache.
    """
