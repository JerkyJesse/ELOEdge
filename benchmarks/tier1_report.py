"""End-to-end Tier 1 report: grid-search K+home_adv, evaluate raw + calibrated
on held-out test. Writes JSON + CSVs to `reports/`.

Usage: `python -m Claude.benchmarks.tier1_report` (or via CLI `tier1-report`
subcmd). Reads games via data_adapter, splits via splits.train_test_split,
tunes via tune_tier1.grid_search_tier1, evaluates on test.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import numpy as np

from .calibrate import PlattCalibrated
from .classic_elo import ClassicEloTier1
from .data_adapter import load_nba_games
from .scoring import brier, crps, ece, log_loss, mce, sharpness
from .splits import train_test_split
from .tune_tier1 import grid_search_tier1


def _predict_all(model, frame):
    preds = np.empty(len(frame), dtype=float)
    for i, row in enumerate(frame.itertuples(index=False)):
        preds[i] = float(model.predict(row.home, row.away, row.date))
    return np.clip(preds, 1e-9, 1.0 - 1e-9)


def _summary(label, preds, frame):
    y = (frame["home_score"].to_numpy() > frame["away_score"].to_numpy()).astype(float)
    return {
        "label": label,
        "log_loss": float(log_loss(y, preds)),
        "brier": float(brier(y, preds)),
        "crps": float(crps(y, preds)),
        "ece": float(ece(y, preds)),
        "mce": float(mce(y, preds)),
        "sharpness": float(sharpness(preds)),
    }


def run() -> dict:
    games = load_nba_games(verbose=True)
    train, test, _ = train_test_split(games, verbose=True)
    if len(train) == 0 or len(test) == 0:
        raise SystemExit("tier1_report: empty split — populate historical games first")

    print(f"\n[tier1_report] grid search on train ({len(train)} rows) ...")
    matrix, best = grid_search_tier1(train, verbose=True)

    # Evaluate defaults on test
    default_model = ClassicEloTier1()
    default_model.fit(train)
    default_preds = _predict_all(default_model, test)
    default_summary = _summary("tier1_defaults", default_preds, test)

    # Evaluate tuned-raw on test
    tuned_model = ClassicEloTier1(k=best["k"], home_adv=best["home_adv"])
    tuned_model.fit(train)
    tuned_preds = _predict_all(tuned_model, test)
    tuned_summary = _summary("tier1_tuned_raw", tuned_preds, test)

    # Evaluate tuned + Platt on test
    cal_model = PlattCalibrated(
        ClassicEloTier1(k=best["k"], home_adv=best["home_adv"]),
    )
    cal_model.fit(train)
    cal_preds = _predict_all(cal_model, test)
    cal_summary = _summary("tier1_tuned_platt", cal_preds, test)
    cal_summary["platt_a"] = float(cal_model.a)
    cal_summary["platt_b"] = float(cal_model.b)

    out = {
        "run_date": datetime.now().strftime("%Y-%m-%d"),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "best_params": best,
        "test_summaries": [default_summary, tuned_summary, cal_summary],
    }

    reports_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    matrix_path = os.path.join(reports_dir, f"tier1-tuning-nba-{stamp}.csv")
    matrix.to_csv(matrix_path, index=False)
    report_path = os.path.join(reports_dir, f"tier1-report-nba-{stamp}.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print("\n[tier1_report] test-set evaluation (lower log_loss better):")
    print(f"{'label':<25} {'log_loss':>10} {'brier':>8} {'ece':>8} {'sharpness':>10}")
    for s in out["test_summaries"]:
        print(
            f"{s['label']:<25} {s['log_loss']:>10.5f} "
            f"{s['brier']:>8.4f} {s['ece']:>8.4f} {s['sharpness']:>10.4f}"
        )
    print(f"\nwrote {matrix_path}\nwrote {report_path}")
    return out


if __name__ == "__main__":
    run()
