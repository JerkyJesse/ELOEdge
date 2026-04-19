"""Grid search over Tier 1 Classic Elo hyperparams (K, home_adv).

To avoid test-set leakage, tunes on a val split carved from the END of train
(time-based). Returns the full log-loss matrix + best params. Honest usage:
    * build train/test per splits.train_test_split
    * tune on train only via this function (internal val split)
    * fit a fresh Tier 1 with best params on ALL of train
    * predict test — test set never touched during tuning.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from .classic_elo import ClassicEloTier1
from .scoring import log_loss


DEFAULT_K_GRID = (4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0)
DEFAULT_HOME_ADV_GRID = (40.0, 60.0, 80.0, 100.0, 120.0)


def _predict_all(model, frame: pd.DataFrame) -> np.ndarray:
    preds = np.empty(len(frame), dtype=float)
    for i, row in enumerate(frame.itertuples(index=False)):
        preds[i] = float(model.predict(row.home, row.away, row.date))
    return np.clip(preds, 1e-9, 1.0 - 1e-9)


def grid_search_tier1(
    train: pd.DataFrame,
    k_grid: Iterable[float] = DEFAULT_K_GRID,
    home_adv_grid: Iterable[float] = DEFAULT_HOME_ADV_GRID,
    val_frac: float = 0.20,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """Time-based val split (last val_frac of train by date). Fit on head,
    score val. Returns (matrix, best) where matrix is a DataFrame with
    columns k, home_adv, val_log_loss sorted ascending by loss."""
    g = train.sort_values("date").reset_index(drop=True)
    n_val = max(1, int(len(g) * val_frac))
    head = g.iloc[: len(g) - n_val].reset_index(drop=True)
    val = g.iloc[len(g) - n_val :].reset_index(drop=True)
    y_val = (val["home_score"].to_numpy() > val["away_score"].to_numpy()).astype(float)

    rows = []
    for k in k_grid:
        for h in home_adv_grid:
            model = ClassicEloTier1(k=float(k), home_adv=float(h))
            model.fit(head)
            preds = _predict_all(model, val)
            ll = log_loss(y_val, preds)
            rows.append({"k": float(k), "home_adv": float(h), "val_log_loss": float(ll)})

    df = pd.DataFrame(rows).sort_values("val_log_loss").reset_index(drop=True)
    best = df.iloc[0].to_dict()
    if verbose:
        print(
            f"tune_tier1: val rows={len(val)} head rows={len(head)} "
            f"| best K={best['k']} home_adv={best['home_adv']} "
            f"val_logloss={best['val_log_loss']:.5f}"
        )
    return df, best
