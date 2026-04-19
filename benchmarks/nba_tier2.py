"""Tier 2 shim: wraps production `Claude/NBA/elo_model.NBAElo` as a
`BenchmarkRating` (design sec.3, sec.7).

Tier 2 code stays READ-ONLY from this module (design sec.3, sec.10). This shim
only imports `NBAElo`, reads `nba_elo_settings.json`, and calls existing public
methods. Adjuster "leave-one-out" variants work by passing override kwargs into
`NBAElo.__init__` — setting one adjuster to 0.0 (or False for bool toggles)
disables it without touching production code.

Load cost: importing `NBAElo` pulls `config`, `color_helpers`, `platt` from
`Claude/NBA/`. We add that dir to `sys.path` at import time. No network side
effects from those imports (verified 2026-04-18).

Coverage caveat: adjusters that depend on external signals — `player_boost`
(needs `set_player_stats`), `altitude_factor` (needs `_altitude_bonus` dict),
`sos_factor` (builds from opponent ratings during fit — works), `b2b_penalty`
(uses `game_date` — works), injuries (disabled via `use_injuries=False`) —
will be inert in the ablation unless the caller populates those dicts before
`fit`. Document gaps in the ablation CSV downstream.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import BenchmarkRating


_NBA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "NBA"))
if _NBA_DIR not in sys.path:
    sys.path.insert(0, _NBA_DIR)

# Import deferred until needed so tests can skip cleanly if NBA deps missing.
NBAElo = None  # type: ignore
_ELO_SETTINGS_PATH = os.path.join(_NBA_DIR, "nba_elo_settings.json")


# Adjusters enumerated from nba_elo_settings.json (design sec.7: 22 adjusters).
# Split into numeric (zero-disables) and boolean (False-disables).
NUMERIC_ADJUSTERS: List[str] = [
    "player_boost",
    "rest_factor",
    "form_weight",
    "travel_factor",
    "sos_factor",
    "playoff_hca_factor",
    "pace_factor",
    "division_factor",
    "mean_reversion",
    "b2b_penalty",
    "road_trip_factor",
    "homestand_factor",
    "win_streak_factor",
    "altitude_factor",
    "season_phase_factor",
    "scoring_consistency_factor",
    "rest_advantage_cap",
]
BOOLEAN_ADJUSTERS: List[str] = ["use_mov"]
ALL_ADJUSTERS: List[str] = NUMERIC_ADJUSTERS + BOOLEAN_ADJUSTERS


def _import_nbaelo():
    global NBAElo
    if NBAElo is None:
        from elo_model import NBAElo as _NBAElo  # type: ignore
        NBAElo = _NBAElo
    return NBAElo


def load_base_settings(path: str = _ELO_SETTINGS_PATH) -> Dict[str, Any]:
    """Load production Elo settings JSON. Keys not understood by NBAElo
    (`starting_balance`, `kelly_fraction`, `auto_kalshi`, etc.) are filtered
    out so the kwargs pass cleanly."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing {path}")
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    nba_init_params = {
        "base_rating", "k", "home_adv", "use_mov", "player_boost", "rest_factor",
        "form_weight", "travel_factor", "sos_factor", "playoff_hca_factor",
        "pace_factor", "division_factor", "mean_reversion", "b2b_penalty",
        "road_trip_factor", "homestand_factor", "win_streak_factor",
        "altitude_factor", "season_phase_factor", "scoring_consistency_factor",
        "rest_advantage_cap",
    }
    return {k: v for k, v in raw.items() if k in nba_init_params}


class NBATier2(BenchmarkRating):
    """Production NBA Elo (22 adjusters) wrapped as a `BenchmarkRating`.

    Parameters
    ----------
    adjuster_override : dict, optional
        Maps adjuster name -> override value. Zero (or False for bool toggles)
        disables the adjuster — that's how leave-one-out variants are formed.
        Example: `{"player_boost": 0.0}` disables player boost only.
    base_settings : dict, optional
        Starting config. Defaults to `load_base_settings()` reading
        `Claude/NBA/nba_elo_settings.json`.
    calibrated : bool
        Pass-through to `NBAElo.win_prob(calibrated=...)`. Default False for
        ablation (benchmark against raw Elo, not Platt-scaled output).
    use_injuries : bool
        Pass-through. Default False — historical games don't have live injury
        data; enabling would leak future state.
    """

    name = "nba_tier2"
    tier = 2

    def __init__(
        self,
        adjuster_override: Optional[Dict[str, Any]] = None,
        base_settings: Optional[Dict[str, Any]] = None,
        calibrated: bool = False,
        use_injuries: bool = False,
    ) -> None:
        _import_nbaelo()
        self._override = dict(adjuster_override or {})
        self._base = dict(base_settings) if base_settings is not None else load_base_settings()
        self._base.update(self._override)
        self.calibrated = bool(calibrated)
        self.use_injuries = bool(use_injuries)
        self._model = None

    def _fresh_model(self):
        return NBAElo(**self._base)

    def fit(self, games: pd.DataFrame) -> None:
        self._check_schema(games)
        self._model = self._fresh_model()
        g = games.sort_values("date").reset_index(drop=True)
        for row in g.itertuples(index=False):
            self._model.update_game(
                home_team=row.home,
                away_team=row.away,
                home_score=float(row.home_score),
                away_score=float(row.away_score),
                neutral_site=False,
                game_date=pd.Timestamp(row.date).to_pydatetime(),
            )

    def predict(self, home: str, away: str, date: Any = None) -> float:
        if self._model is None:
            return 0.5
        game_date = pd.Timestamp(date).to_pydatetime() if date is not None else None
        p = self._model.win_prob(
            team_a=home,
            team_b=away,
            team_a_home=True,
            neutral_site=False,
            calibrated=self.calibrated,
            game_date=game_date,
            use_injuries=self.use_injuries,
        )
        return float(np.clip(p, 1e-9, 1.0 - 1e-9))


def _tier1_settings(base_settings: Dict[str, Any]) -> Dict[str, Any]:
    """Settings with all 22 adjusters neutralized. Keeps structural params
    (base_rating, k, home_adv). Equivalent to Tier 1 built on production NBAElo."""
    zeroed = {k: 0.0 for k in NUMERIC_ADJUSTERS}
    zeroed["use_mov"] = False
    out = dict(base_settings)
    out.update(zeroed)
    return out


def tier1_baseline_factory(
    base_settings: Optional[Dict[str, Any]] = None,
) -> Callable[[], NBATier2]:
    """Factory for the Tier 1 baseline used by design sec.7 gate: NBAElo with
    structural params from production settings, ALL 22 adjusters disabled."""
    if base_settings is None:
        base_settings = load_base_settings()
    settings = _tier1_settings(base_settings)
    return lambda settings=settings: NBATier2(base_settings=settings)


def one_off_variant_factories(
    base_settings: Optional[Dict[str, Any]] = None,
    adjusters: Optional[List[str]] = None,
) -> Dict[str, Callable[[], NBATier2]]:
    """{adjuster_name -> factory} for LEAVE-ONE-OUT ablation.

    Each factory returns a fresh `NBATier2` with the named adjuster zeroed
    (False for bool toggles) and all others at production values. Use with
    baseline = full NBATier2. Verdict flips: variant is Tier2-minus-X, so
    Δ_logloss > 0 means removing X hurts → adjuster earns its keep. Use
    `run_ablation(..., direction="leave_one_out")`.
    """
    if base_settings is None:
        base_settings = load_base_settings()
    if adjusters is None:
        adjusters = list(ALL_ADJUSTERS)

    out: Dict[str, Callable[[], NBATier2]] = {}
    for adj in adjusters:
        override = {adj: False if adj in BOOLEAN_ADJUSTERS else 0.0}

        def _factory(override=override, bs=dict(base_settings)):
            return NBATier2(adjuster_override=override, base_settings=bs)

        out[adj] = _factory
    return out


def apply_verdicts_to_settings(
    ablation_csv_path: str,
    settings_path: str = _ELO_SETTINGS_PATH,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """Read an ablation CSV, attach `_ablation_verdict` per adjuster to the
    production settings JSON (design sec.12 / DoD). Does NOT alter any of the
    runtime adjuster values — only adds the verdict annotation.

    dry_run=True (default) returns the new settings dict without writing.
    Pass dry_run=False to persist.

    The writer is intentionally non-destructive: existing keys are preserved,
    only new `_ablation_verdict__<adjuster>` keys are added (prefix chosen to
    avoid collision with any future legit setting). Users should review the
    CSV first, then run with dry_run=False.
    """
    df = pd.read_csv(ablation_csv_path)
    with open(settings_path, "r", encoding="utf-8") as fh:
        settings = json.load(fh)

    for _, row in df.iterrows():
        key = f"_ablation_verdict__{row['adjuster']}"
        settings[key] = {
            "verdict": str(row["verdict"]),
            "delta_logloss": float(row["delta_logloss"]),
            "p_value": float(row["p_value"]),
            "source_csv": os.path.basename(ablation_csv_path),
        }
    if not dry_run:
        with open(settings_path, "w", encoding="utf-8") as fh:
            json.dump(settings, fh, indent=2)
    return settings


def addition_variant_factories(
    base_settings: Optional[Dict[str, Any]] = None,
    adjusters: Optional[List[str]] = None,
) -> Dict[str, Callable[[], NBATier2]]:
    """{adjuster_name -> factory} for ADDITION-STYLE ablation per design sec.7.

    Each factory returns NBATier2 with ONLY the named adjuster at its
    production value; all other 21 adjusters are zeroed. Paired with
    `tier1_baseline_factory()` this is the canonical sec.7 gate: baseline =
    Tier 1, variant = Tier 1 + one adjuster. Standard verdict applies:
    Δ_logloss < -0.003 AND p < 0.05 → KEEP (adjuster earns its place).
    """
    if base_settings is None:
        base_settings = load_base_settings()
    if adjusters is None:
        adjusters = list(ALL_ADJUSTERS)

    zero_base = _tier1_settings(base_settings)

    out: Dict[str, Callable[[], NBATier2]] = {}
    for adj in adjusters:
        # Override = keep this one adjuster at its production value
        if adj in BOOLEAN_ADJUSTERS:
            prod_value = bool(base_settings.get(adj, True))
        else:
            prod_value = float(base_settings.get(adj, 0.0))
        override = {adj: prod_value}

        def _factory(override=override, bs=dict(zero_base)):
            return NBATier2(adjuster_override=override, base_settings=bs)

        out[adj] = _factory
    return out
