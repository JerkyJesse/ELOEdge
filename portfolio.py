"""Cross-sport portfolio view for PredictMarkets trading system.

Shows unified balance, per-sport P&L, and cash flow summary
across all 4 sports (NBA, NFL, MLB, NHL) using the shared bankroll.

Usage:
    python portfolio.py          (standalone)
    > portfolio                  (from any sport CLI)
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime

# ── Import color helpers from first available sport ──
_BASE = os.path.dirname(os.path.abspath(__file__))
for _d in ["NBA", "NFL", "MLB", "NHL"]:
    _p = os.path.join(_BASE, _d)
    if os.path.isdir(_p) and os.path.exists(os.path.join(_p, "color_helpers.py")):
        sys.path.insert(0, _p)
        break
try:
    from color_helpers import cok, cerr, cwarn, chi, cdim, div, hdr, cbold, cyel
except ImportError:
    def _passthrough(t): return str(t)
    cok = cerr = cwarn = chi = cdim = cbold = cyel = _passthrough
    def div(n=80): print("-" * n)
    def hdr(t): print("\n  " + t)

# ── Constants ──
SPORT_DIRS = ["NBA", "NFL", "MLB", "NHL"]
PORTFOLIO_SETTINGS_FILE = os.path.join(_BASE, "portfolio_settings.json")
CASH_TRANSACTIONS_FILE = os.path.join(_BASE, "cash_transactions.csv")


def calc_fee_from_payout(potential_payout_dollars):
    return round(float(potential_payout_dollars) * 0.02, 4)


def load_portfolio_settings():
    defaults = {"starting_balance": 0.0}
    if os.path.exists(PORTFOLIO_SETTINGS_FILE):
        try:
            with open(PORTFOLIO_SETTINGS_FILE, "r") as f:
                saved = json.load(f)
            defaults.update(saved)
        except Exception:
            pass
    return defaults


def load_cash_transactions():
    cols = ["txn_id", "timestamp", "type", "amount", "note"]
    if os.path.exists(CASH_TRANSACTIONS_FILE):
        try:
            df = pd.read_csv(CASH_TRANSACTIONS_FILE)
            for c in cols:
                if c not in df.columns:
                    df[c] = ""
            return df[cols]
        except Exception:
            pass
    return pd.DataFrame(columns=cols)


def load_sport_data(sport):
    """Load trading data for a single sport. Returns dict with P&L metrics."""
    sport_dir = os.path.join(_BASE, sport)
    predicts_path = os.path.join(sport_dir, "predicts_lots.csv")

    result = {
        "name": sport,
        "has_trades": False,
        "open_count": 0,
        "closed_count": 0,
        "entry_cost_total": 0.0,
        "realized_cash": 0.0,
        "realized_profit": 0.0,
        "unrealized_profit": 0.0,
        "total_pnl": 0.0,
        "win_rate": 0.0,
    }

    if not os.path.exists(predicts_path):
        return result

    try:
        df = pd.read_csv(predicts_path)
    except Exception:
        return result

    if df.empty:
        return result

    result["has_trades"] = True
    numeric_cols = [
        "contracts_open", "contracts_original", "avg_entry_price",
        "entry_fee_total", "entry_cost_total", "realized_cash",
        "realized_exit_fees", "realized_profit", "last_mark_price",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        else:
            df[c] = 0.0

    result["entry_cost_total"] = float(df["entry_cost_total"].sum())
    result["realized_cash"] = float(df["realized_cash"].sum())
    result["realized_profit"] = float(df["realized_profit"].sum())

    # Open / closed counts
    open_mask = df["contracts_open"] > 0
    result["open_count"] = int(open_mask.sum())
    closed = df[df["status"].isin(["sold", "settled"])] if "status" in df.columns else df[~open_mask]
    result["closed_count"] = len(closed)

    # Win rate
    wins = closed[closed["realized_profit"] > 0] if not closed.empty else pd.DataFrame()
    result["win_rate"] = (len(wins) / len(closed) * 100.0) if len(closed) > 0 else 0.0

    # Unrealized P&L
    open_df = df[open_mask]
    if not open_df.empty:
        unrealized_value = (
            open_df["contracts_open"] * open_df["last_mark_price"]
            - open_df["contracts_open"].apply(lambda x: calc_fee_from_payout(x * 1.0))
        )
        remaining_cost = (
            open_df["contracts_open"] * open_df["avg_entry_price"]
            + (open_df["entry_fee_total"] * open_df["contracts_open"]
               / open_df["contracts_original"].replace(0, np.nan)).fillna(0)
        )
        result["unrealized_profit"] = float((unrealized_value - remaining_cost).sum())

    result["total_pnl"] = result["realized_profit"] + result["unrealized_profit"]
    return result


def show_portfolio():
    """Display cross-sport portfolio summary."""
    settings = load_portfolio_settings()
    starting = float(settings.get("starting_balance", 0))

    # Cash flows
    txn_df = load_cash_transactions()
    total_deposited = 0.0
    total_withdrawn = 0.0
    if not txn_df.empty:
        txn_df["amount"] = pd.to_numeric(txn_df["amount"], errors="coerce").fillna(0)
        total_deposited = txn_df[txn_df["type"] == "deposit"]["amount"].sum()
        total_withdrawn = txn_df[txn_df["type"] == "withdraw"]["amount"].sum()
    net_flow = total_deposited - total_withdrawn
    funded = starting + net_flow

    if starting <= 0 and net_flow <= 0:
        print(cwarn("\n  No bankroll set up. Use 'deposit' or 'balance' from any sport CLI.\n"))
        return

    # Load all sports
    sports_data = [load_sport_data(s) for s in SPORT_DIRS]

    # Compute global balance
    total_spent = sum(s["entry_cost_total"] for s in sports_data)
    total_received = sum(s["realized_cash"] for s in sports_data)
    current_balance = starting + net_flow - total_spent + total_received

    total_realized = sum(s["realized_profit"] for s in sports_data)
    total_unrealized = sum(s["unrealized_profit"] for s in sports_data)
    total_pnl = total_realized + total_unrealized
    total_open = sum(s["open_count"] for s in sports_data)
    total_closed = sum(s["closed_count"] for s in sports_data)

    hdr("CROSS-SPORT PORTFOLIO")
    div(100)
    print("  %-6s | %6s | %6s | %12s | %12s | %12s | %8s"
          % (chi("Sport"), chi("Open"), chi("Closed"),
             chi("Realized"), chi("Unrealized"), chi("Total P&L"), chi("Win %")))
    div(100)

    for s in sports_data:
        if not s["has_trades"]:
            print("  %-6s | %6s | %6s | %12s | %12s | %12s | %8s"
                  % (cdim(s["name"]), cdim("-"), cdim("-"),
                     cdim("$0.00"), cdim("$0.00"), cdim("$0.00"), cdim("-")))
        else:
            rp = s["realized_profit"]
            up = s["unrealized_profit"]
            tp = s["total_pnl"]
            rp_s = cok("$%+.2f" % rp) if rp >= 0 else cerr("$%+.2f" % rp)
            up_s = cok("$%+.2f" % up) if up >= 0 else cerr("$%+.2f" % up)
            tp_s = cok("$%+.2f" % tp) if tp >= 0 else cerr("$%+.2f" % tp)
            wr_s = cok("%.0f%%" % s["win_rate"]) if s["win_rate"] >= 50 else cwarn("%.0f%%" % s["win_rate"])
            print("  %-6s | %6d | %6d | %12s | %12s | %12s | %8s"
                  % (chi(s["name"]), s["open_count"], s["closed_count"],
                     rp_s, up_s, tp_s,
                     wr_s if s["closed_count"] > 0 else cdim("-")))

    div(100)

    # Totals
    bal_s = cok("$%.2f" % current_balance)
    tr_s = cok("$%+.2f" % total_realized) if total_realized >= 0 else cerr("$%+.2f" % total_realized)
    tu_s = cok("$%+.2f" % total_unrealized) if total_unrealized >= 0 else cerr("$%+.2f" % total_unrealized)
    tp_s = cok("$%+.2f" % total_pnl) if total_pnl >= 0 else cerr("$%+.2f" % total_pnl)
    roi = (total_pnl / funded * 100.0) if funded > 0 else 0.0
    roi_s = cok("%+.1f%%" % roi) if roi >= 0 else cerr("%+.1f%%" % roi)

    print("  BALANCE: %s  |  OPEN: %s  |  CLOSED: %s"
          % (bal_s, cok(str(total_open)), cdim(str(total_closed))))
    print("  REALIZED: %s  |  UNREALIZED: %s  |  TOTAL P&L: %s  |  ROI: %s"
          % (tr_s, tu_s, tp_s, roi_s))

    # Cash flow summary
    if total_deposited > 0 or total_withdrawn > 0:
        print("  STARTING: %s  |  DEPOSITS: %s  |  WITHDRAWALS: %s  |  NET FLOW: %s"
              % (chi("$%.2f" % starting),
                 cok("$%.2f" % total_deposited),
                 cerr("$%.2f" % total_withdrawn),
                 cok("$%+.2f" % net_flow) if net_flow >= 0 else cerr("$%+.2f" % net_flow)))
    div(100)

    # Recent transactions
    if not txn_df.empty:
        recent = txn_df.tail(5)
        print("\n  RECENT TRANSACTIONS:")
        for _, r in recent.iterrows():
            t = str(r["type"])
            a = float(r["amount"])
            ts = str(r["timestamp"])
            n = str(r.get("note", ""))
            t_s = cok("DEPOSIT ") if t == "deposit" else cerr("WITHDRAW")
            print("    %s  %s  $%.2f  %s" % (cdim(ts), t_s, a, cdim(n) if n else ""))
    print()


def main():
    show_portfolio()


if __name__ == "__main__":
    main()
