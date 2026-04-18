"""Predicts contract ledger management."""

import os
from datetime import datetime

import numpy as np
import pandas as pd
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

import math

from config import (PREDICTS_FILE, BACKTEST_FILE, CASH_TRANSACTIONS_FILE,
                    PORTFOLIO_SETTINGS_FILE, SPORT_DIRS, _PARENT_DIR,
                    get_team_abbr, current_timestamp,
                    load_elo_settings, save_elo_settings,
                    load_portfolio_settings, save_portfolio_settings)
from color_helpers import (
    cok, cerr, cwarn, chi, cdim, cyel, div, hdr, cbold,
)
from live_scores import fetch_live_nfl_scores


def calc_fee_from_payout(potential_payout_dollars):
    return round(float(potential_payout_dollars) * 0.02, 4)


def odds_input_to_prob(prompt="  Implied prob or American odds (e.g. 62 or -145): "):
    raw = input(prompt).strip()
    if not raw:
        return None, ""
    try:
        val = float(raw)
        if 0 <= val <= 100:
            return val / 100.0, raw
        if val > 0:
            return 100.0 / (val + 100.0), raw
        return -val / (-val + 100.0), raw
    except ValueError:
        print(cwarn("  Couldn't parse input — no edge calculated"))
        return None, raw


def get_current_balance():
    settings = load_portfolio_settings()
    starting = float(settings.get("starting_balance", 0))
    # Net cash flows from deposits/withdrawals
    txn_df = load_cash_transactions()
    net_cash = 0.0
    if not txn_df.empty:
        txn_df["amount"] = pd.to_numeric(txn_df["amount"], errors="coerce").fillna(0)
        net_cash = (txn_df[txn_df["type"] == "deposit"]["amount"].sum()
                    - txn_df[txn_df["type"] == "withdraw"]["amount"].sum())
    if starting <= 0 and net_cash <= 0:
        return 0.0
    # Sum entry costs and realized cash across ALL sports
    total_spent = 0.0
    total_received = 0.0
    for sport in SPORT_DIRS:
        pf = os.path.join(_PARENT_DIR, sport, "predicts_lots.csv")
        if os.path.exists(pf):
            try:
                sdf = pd.read_csv(pf)
                for c in ["entry_cost_total", "realized_cash"]:
                    if c in sdf.columns:
                        sdf[c] = pd.to_numeric(sdf[c], errors="coerce").fillna(0)
                    else:
                        sdf[c] = 0.0
                total_spent += sdf["entry_cost_total"].sum()
                total_received += sdf["realized_cash"].sum()
            except Exception:
                pass
    return starting + net_cash - total_spent + total_received


def load_backtest_accuracy():
    if not os.path.exists(BACKTEST_FILE):
        return None
    try:
        df = pd.read_csv(BACKTEST_FILE)
        if "correct" not in df.columns or df.empty:
            return None
        return float(pd.to_numeric(df["correct"], errors="coerce").dropna().mean())
    except Exception:
        return None


def calc_kelly_lots(model_prob, market_price_cents, balance=None, kelly_frac=None, backtest_acc=None):
    if balance is None:
        balance = get_current_balance()
    if kelly_frac is None:
        settings = load_elo_settings()
        kelly_frac = float(settings.get("kelly_fraction", 0.25))
    if balance <= 0 or market_price_cents <= 0 or market_price_cents >= 100:
        return 0, 0.0, 0.0, 0.0
    market_price = market_price_cents / 100.0
    prob_for_edge = backtest_acc if backtest_acc is not None else model_prob
    edge = prob_for_edge - market_price
    if edge <= 0:
        return 0, 0.0, edge, 0.0
    kelly_full = edge / (1.0 - market_price)
    kelly_adj = kelly_full * kelly_frac
    wager = kelly_adj * balance
    contracts = max(0, math.floor(wager / market_price))
    return contracts, kelly_full, edge, kelly_adj


def show_kelly_recommendation(model_prob, market_price_cents):
    settings = load_elo_settings()
    balance = get_current_balance()
    kelly_frac = float(settings.get("kelly_fraction", 0.25))
    backtest_acc = load_backtest_accuracy()
    contracts, kelly_full, edge, kelly_adj = calc_kelly_lots(
        model_prob, market_price_cents, balance, kelly_frac, backtest_acc
    )
    market_price = market_price_cents / 100.0
    div(60)
    print("  %s" % cbold("KELLY CRITERION SIZING"))
    print("    Model prob  : %s" % cok("%.1f%%" % (model_prob * 100)))
    if backtest_acc is not None:
        print("    Backtest acc: %s (used for edge)" % chi("%.1f%%" % (backtest_acc * 100)))
    else:
        print("    Backtest acc: %s (using model prob)" % cwarn("unavailable"))
    print("    Market price: %s (%s implied)" % (
        chi("%d c" % market_price_cents),
        cdim("%.1f%%" % (market_price * 100))))
    print("    Edge        : %s" % (
        cok("%+.1f%%" % (edge * 100)) if edge > 0 else cerr("%+.1f%%" % (edge * 100))))
    print("    Full Kelly  : %s" % cdim("%.1f%%" % (kelly_full * 100)))
    print("    %.0f%% Kelly   : %s" % (kelly_frac * 100, cok("%.1f%%" % (kelly_adj * 100))))
    if balance > 0:
        print("    Balance     : %s" % cok("$%.2f" % balance))
        print("    Suggested   : %s contracts @ %dc = $%.2f" % (
            cok(str(contracts)), market_price_cents,
            contracts * market_price))
    else:
        print("    Balance     : %s (set with 'balance' command)" % cwarn("not set"))
    div(60)
    return contracts


def prompt_balance():
    settings = load_portfolio_settings()
    current_starting = float(settings.get("starting_balance", 0))
    if current_starting > 0:
        print("  Current starting balance: %s" % cok("$%.2f" % current_starting))
        change = input("  Update balance? (enter new amount or press Enter to keep): ").strip()
        if not change:
            return current_starting
        try:
            new_bal = float(change)
            if new_bal > 0:
                settings["starting_balance"] = new_bal
                save_portfolio_settings(settings)
                print(cok("  Starting balance set to $%.2f (shared across all sports)" % new_bal))
                return new_bal
        except ValueError:
            print(cwarn("  Invalid amount, keeping current balance."))
            return current_starting
    else:
        print("\n  %s" % chi("SET STARTING BALANCE"))
        print("  Enter your starting bankroll (shared across all 4 sports).")
        try:
            bal_str = input("  Starting balance ($): ").strip()
            if bal_str:
                new_bal = float(bal_str)
                if new_bal > 0:
                    settings["starting_balance"] = new_bal
                    save_portfolio_settings(settings)
                    print(cok("  Starting balance set to $%.2f (shared across all sports)" % new_bal))
                    return new_bal
        except ValueError:
            pass
        print(cwarn("  No balance set. Kelly sizing will be disabled until you run 'balance'."))
        return 0.0


def show_balance():
    settings = load_portfolio_settings()
    starting = float(settings.get("starting_balance", 0))
    kelly_frac = float(load_elo_settings().get("kelly_fraction", 0.25))
    txn_df = load_cash_transactions()
    total_deposited = 0.0
    total_withdrawn = 0.0
    if not txn_df.empty:
        txn_df["amount"] = pd.to_numeric(txn_df["amount"], errors="coerce").fillna(0)
        total_deposited = txn_df[txn_df["type"] == "deposit"]["amount"].sum()
        total_withdrawn = txn_df[txn_df["type"] == "withdraw"]["amount"].sum()
    net_flow = total_deposited - total_withdrawn
    if starting <= 0 and net_flow <= 0:
        print(cwarn("  No starting balance set. Use 'balance' or 'deposit' to fund your account."))
        return
    current = get_current_balance()
    funded = starting + net_flow
    pnl = current - funded
    pnl_s = cok("$%+.2f" % pnl) if pnl >= 0 else cerr("$%+.2f" % pnl)
    roi_s = cok("%+.1f%%" % (pnl / funded * 100)) if pnl >= 0 else cerr("%+.1f%%" % (pnl / funded * 100)) if funded > 0 else cdim("N/A")
    hdr("BANKROLL STATUS (SHARED ACROSS ALL SPORTS)")
    print("  Starting balance : %s" % chi("$%.2f" % starting))
    if total_deposited > 0 or total_withdrawn > 0:
        print("  Deposits         : %s" % cok("$%.2f" % total_deposited))
        print("  Withdrawals      : %s" % cerr("$%.2f" % total_withdrawn))
        net_s = cok("$%+.2f" % net_flow) if net_flow >= 0 else cerr("$%+.2f" % net_flow)
        print("  Net cash flow    : %s" % net_s)
        print("  Total funded     : %s" % chi("$%.2f" % funded))
    print("  Current balance  : %s" % cok("$%.2f" % current))
    print("  P&L              : %s (%s)" % (pnl_s, roi_s))
    print("  Kelly fraction   : %s" % cdim("%.0f%%" % (kelly_frac * 100)))


def load_cash_transactions(filename=CASH_TRANSACTIONS_FILE):
    cols = ["txn_id", "timestamp", "type", "amount", "note"]
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename)
            for c in cols:
                if c not in df.columns:
                    df[c] = ""
            return df[cols]
        except Exception:
            pass
    return pd.DataFrame(columns=cols)


def save_cash_transactions(df, filename=CASH_TRANSACTIONS_FILE):
    df.to_csv(filename, index=False)


def deposit_cash():
    hdr("DEPOSIT CASH")
    current = get_current_balance()
    if current > 0:
        print("  Current balance: %s" % cok("$%.2f" % current))
    try:
        amt_str = input("  Deposit amount ($): ").strip()
        if not amt_str:
            print(cwarn("  Cancelled."))
            return
        amount = float(amt_str)
        if amount <= 0:
            print(cerr("  Amount must be positive."))
            return
    except ValueError:
        print(cerr("  Invalid amount."))
        return
    note = input("  Note (optional): ").strip()
    df = load_cash_transactions()
    txn_id = int(pd.to_numeric(df["txn_id"], errors="coerce").max()) + 1 if not df.empty and not df["txn_id"].dropna().empty else 1
    new_row = pd.DataFrame([{
        "txn_id": txn_id,
        "timestamp": current_timestamp(),
        "type": "deposit",
        "amount": round(amount, 2),
        "note": note,
    }])
    df = pd.concat([df, new_row], ignore_index=True) if not df.empty else new_row
    save_cash_transactions(df)
    new_balance = get_current_balance()
    print(cok("  Deposited $%.2f. New balance: $%.2f" % (amount, new_balance)))


def withdraw_cash():
    hdr("WITHDRAW CASH")
    current = get_current_balance()
    if current <= 0:
        print(cerr("  No funds available to withdraw. Balance: $%.2f" % current))
        return
    print("  Current balance: %s" % cok("$%.2f" % current))
    try:
        amt_str = input("  Withdraw amount ($): ").strip()
        if not amt_str:
            print(cwarn("  Cancelled."))
            return
        amount = float(amt_str)
        if amount <= 0:
            print(cerr("  Amount must be positive."))
            return
        if amount > current:
            print(cerr("  Insufficient funds. Available: $%.2f" % current))
            return
    except ValueError:
        print(cerr("  Invalid amount."))
        return
    note = input("  Note (optional): ").strip()
    df = load_cash_transactions()
    txn_id = int(pd.to_numeric(df["txn_id"], errors="coerce").max()) + 1 if not df.empty and not df["txn_id"].dropna().empty else 1
    new_row = pd.DataFrame([{
        "txn_id": txn_id,
        "timestamp": current_timestamp(),
        "type": "withdraw",
        "amount": round(amount, 2),
        "note": note,
    }])
    df = pd.concat([df, new_row], ignore_index=True) if not df.empty else new_row
    save_cash_transactions(df)
    new_balance = get_current_balance()
    print(cok("  Withdrew $%.2f. New balance: $%.2f" % (amount, new_balance)))


def load_predict_lots(filename=PREDICTS_FILE):
    cols = [
        "lot_id","opened_at","date","market","home_team","away_team",
        "predicted_winner","model_prob","contracts_open","contracts_original",
        "avg_entry_price","entry_fee_total","entry_cost_total","realized_cash",
        "realized_exit_fees","realized_profit","status","last_mark_price","notes",
    ]
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        return df[cols]
    df = pd.DataFrame(columns=cols)
    df.to_csv(filename, index=False)
    return df


def save_predict_lots(df, filename=PREDICTS_FILE):
    df.to_csv(filename, index=False)


def next_lot_id(df):
    if df.empty or "lot_id" not in df.columns or df["lot_id"].dropna().empty:
        return 1
    return int(pd.to_numeric(df["lot_id"], errors="coerce").max()) + 1


def add_predict_contract(home_team, away_team, predicted_winner, model_prob, kelly_suggested=0):
    hdr("LOG MONEYLINE POSITION")
    date = datetime.now().strftime("%Y-%m-%d")
    try:
        default_qty = kelly_suggested if kelly_suggested > 0 else 1
        contracts_str = input("Number of contracts (default %d%s): " % (
            default_qty, " [Kelly]" if kelly_suggested > 0 else "")).strip()
        contracts     = int(contracts_str) if contracts_str else default_qty
        price_str     = input("Price paid per contract (e.g. 0.62): ").strip()
        price         = float(price_str) if price_str else 0.50
    except ValueError:
        print(cerr("Invalid input — position not logged."))
        return
    if contracts <= 0 or price <= 0 or price > 1:
        print(cerr("Invalid quantity or price (must be >0 and price <= $1.00)."))
        return
    fee        = calc_fee_from_payout(contracts * 1.0)
    total_cost = round(contracts * price + fee, 4)
    notes      = input("Notes (optional): ").strip()
    df         = load_predict_lots()
    lot_id     = next_lot_id(df)
    default_note = "Moneyline -> %s @ %.1f%%" % (predicted_winner, model_prob * 100)
    new_row = pd.DataFrame([{
        "lot_id": lot_id, "opened_at": current_timestamp(), "date": date,
        "market": "%s @ %s" % (away_team, home_team),
        "home_team": home_team, "away_team": away_team,
        "predicted_winner": predicted_winner, "model_prob": round(model_prob, 4),
        "contracts_open": contracts, "contracts_original": contracts,
        "avg_entry_price": round(price, 4), "entry_fee_total": fee,
        "entry_cost_total": total_cost, "realized_cash": 0.0,
        "realized_exit_fees": 0.0, "realized_profit": 0.0,
        "status": "pending", "last_mark_price": round(price, 4),
        "notes": notes or default_note,
    }])
    df = pd.concat([df, new_row], ignore_index=True) if not df.empty else new_row
    save_predict_lots(df)
    print("\n  %s #%d" % (cok("Logged lot"), lot_id))
    print("  %dx %s @ %s" % (contracts, chi(predicted_winner), cok("$%.2f" % price)))
    print("  Entry fee: %s   Total cost: %s" % (cwarn("$%.4f" % fee), cyel("$%.4f" % total_cost)))
    if notes:
        print("  Note: %s" % cdim(notes))


def _fetch_live_game_status(home_team, away_team):
    """Fetch live game status for an NFL matchup using ESPN scoreboard."""
    try:
        games     = fetch_live_nfl_scores()
        home_abbr = get_team_abbr(home_team).upper()
        away_abbr = get_team_abbr(away_team).upper()
        for g in games:
            ht = g.get("home_abbr", "").upper()
            at = g.get("away_abbr", "").upper()
            if {ht, at} != {home_abbr, away_abbr}:
                continue
            status_text = str(g.get("status", "")).strip()
            is_final = "final" in status_text.lower()
            period = g.get("period", 0)
            hs = g.get("home_score", 0)
            a_s = g.get("away_score", 0)
            if is_final:
                return "FINAL %s %s-%s %s" % (at, a_s, hs, ht)
            elif period > 0:
                return "Q%s | %s %s-%s %s" % (period, at, a_s, hs, ht)
            else:
                return "SCHEDULED %s" % status_text
        return None
    except Exception:
        return None


def _game_status_label(date_str, home_team="", away_team=""):
    try:
        game_date = pd.to_datetime(date_str).date()
        today     = datetime.now().date()
        delta     = (game_date - today).days
        if delta == 0:
            live = _fetch_live_game_status(home_team, away_team) if home_team and away_team else None
            return live if live else "TODAY"
        elif delta > 0:
            return "IN %dD" % delta
        else:
            return "%dD AGO" % abs(delta)
    except Exception:
        return "?"


def show_open_lots(df):
    open_df = df[df["contracts_open"].fillna(0).astype(float) > 0].copy()
    if open_df.empty:
        print(cwarn("No open moneyline positions."))
        return open_df
    print("\nOPEN MONEYLINE POSITIONS:")
    div(90)
    print("  %3s | %6s | %-10s | %-8s | %-35s | %-18s | %6s | %7s | %7s"
          % (chi("#"), chi("Lot #"), chi("Date"), chi("Status"), chi("Matchup"),
             chi("Pick"), chi("Open"), chi("Avg $"), chi("Mark $")))
    div(90)
    for i, (_, r) in enumerate(open_df.iterrows()):
        gs      = _game_status_label(str(r["date"]), str(r.get("home_team","")), str(r.get("away_team","")))
        matchup = "%s @ %s" % (r["away_team"], r["home_team"])
        gs_str  = gs if gs else "?"
        if "FINAL" in gs_str:
            gs_col = cok("%-8s" % gs_str)
        elif "TODAY" in gs_str or "Q" in gs_str:
            gs_col = cyel("%-8s" % gs_str)
        else:
            gs_col = cdim("%-8s" % gs_str)
        print("  %3d | %6d | %-10s | %s | %-35s | %-18s | %6d | %7.2f | %7.2f"
              % (i, int(r["lot_id"]), str(r["date"]), gs_col, matchup,
                 chi(str(r["predicted_winner"])), int(r["contracts_open"]),
                 float(r["avg_entry_price"]), float(r["last_mark_price"])))
    div(90)
    print("Total open positions: %s" % cok(len(open_df)))
    return open_df


def mark_pending_positions():
    df      = load_predict_lots()
    open_df = show_open_lots(df)
    if open_df.empty:
        return
    try:
        idx        = int(input("\nWhich lot to mark? Enter number: ").strip())
        real_idx   = open_df.index[idx]
        mark_price = float(input("Current market price per contract: ").strip())
    except (ValueError, IndexError):
        print(cerr("Invalid selection"))
        return
    df.loc[real_idx, "last_mark_price"] = round(mark_price, 4)
    save_predict_lots(df)
    print(cok(" Updated mark price"))


def invert_open_trade():
    df      = load_predict_lots()
    open_df = show_open_lots(df)
    if open_df.empty:
        return
    try:
        idx      = int(input("\nWhich lot to invert? Enter number: ").strip())
        real_idx = open_df.index[idx]
    except (ValueError, IndexError):
        print(cerr("Invalid selection"))
        return
    row        = df.loc[real_idx]
    old_pred   = str(row["predicted_winner"])
    home_full  = str(row["home_team"])
    away_full  = str(row["away_team"])
    old_prob   = float(row["model_prob"]) if not pd.isna(row["model_prob"]) else 0.5
    new_prob   = round(1.0 - old_prob, 4)
    pred_lower = old_pred.lower()
    if pred_lower in home_full.lower() or home_full.lower() in pred_lower:
        new_pred = away_full
    elif pred_lower in away_full.lower() or away_full.lower() in pred_lower:
        new_pred = home_full
    else:
        new_pred = away_full if old_pred == home_full else home_full
    df.loc[real_idx, "predicted_winner"] = new_pred
    df.loc[real_idx, "model_prob"]       = new_prob
    note = str(row["notes"] or "")
    df.loc[real_idx, "notes"] = (
        "%s | inverted: %s -> %s (prob %.2f%% -> %.2f%%)"
        % (note, old_pred, new_pred, old_prob * 100, new_prob * 100)
    )
    save_predict_lots(df)
    print("\n  %s Lot #%d" % (cok("Inverted"), int(row["lot_id"])))
    print("  %s (%.1f%%)  ->  %s (%.1f%%)"
          % (cwarn(old_pred), old_prob*100, cok(new_pred), new_prob*100))
    print(cdim("  Accounting / cost basis unchanged."))


def sell_predict_contract():
    df      = load_predict_lots()
    open_df = show_open_lots(df)
    if open_df.empty:
        return
    try:
        idx      = int(input("\nWhich lot to sell? Enter number: ").strip())
        real_idx = open_df.index[idx]
    except (ValueError, IndexError):
        print(cerr("Invalid selection"))
        return
    try:
        contracts_to_sell = int(input("Contracts to sell (partial allowed): ").strip())
        sell_price        = float(input("Sell price per contract: ").strip())
    except ValueError:
        print(cerr("Invalid input"))
        return
    contracts_open = int(df.loc[real_idx, "contracts_open"])
    if contracts_to_sell <= 0 or contracts_to_sell > contracts_open:
        print(cerr("Invalid quantity"))
        return
    avg_entry           = float(df.loc[real_idx, "avg_entry_price"])
    entry_fee_total     = float(df.loc[real_idx, "entry_fee_total"])
    original_contracts  = int(df.loc[real_idx, "contracts_original"])
    allocated_entry_fee = round(entry_fee_total * contracts_to_sell / original_contracts, 4)
    cost_basis          = round(contracts_to_sell * avg_entry + allocated_entry_fee, 4)
    exit_fee            = calc_fee_from_payout(contracts_to_sell * 1.0)
    proceeds            = round(contracts_to_sell * sell_price - exit_fee, 4)
    realized_profit     = round(proceeds - cost_basis, 4)
    df.loc[real_idx, "contracts_open"]     = contracts_open - contracts_to_sell
    df.loc[real_idx, "realized_cash"]      = round(float(df.loc[real_idx, "realized_cash"])      + proceeds,        4)
    df.loc[real_idx, "realized_exit_fees"] = round(float(df.loc[real_idx, "realized_exit_fees"]) + exit_fee,        4)
    df.loc[real_idx, "realized_profit"]    = round(float(df.loc[real_idx, "realized_profit"])    + realized_profit, 4)
    df.loc[real_idx, "last_mark_price"]    = round(sell_price, 4)
    note = str(df.loc[real_idx, "notes"])
    df.loc[real_idx, "notes"]  = "%s | sold %dx@%.2f" % (note, contracts_to_sell, sell_price)
    df.loc[real_idx, "status"] = "sold" if int(df.loc[real_idx, "contracts_open"]) == 0 else "partial"
    save_predict_lots(df)
    sign  = "+" if realized_profit >= 0 else ""
    pnl_s = cok("%s$%.4f" % (sign, realized_profit)) if realized_profit >= 0 else cerr("$%.4f" % realized_profit)
    print("  Sold %dx @ %s | Exit fee: %s | Realized: %s"
          % (contracts_to_sell, cok("$%.2f" % sell_price), cwarn("$%.4f" % exit_fee), pnl_s))


def resolve_predict_contracts():
    df      = load_predict_lots()
    open_df = show_open_lots(df)
    if open_df.empty:
        return
    try:
        idx      = int(input("\nWhich lot to settle? Enter number: ").strip())
        real_idx = open_df.index[idx]
    except (ValueError, IndexError):
        print(cerr("Invalid selection"))
        return
    row    = df.loc[real_idx]
    winner = str(row["predicted_winner"])
    print("\nSettling Lot #%d: Predicted winner = %s" % (int(row["lot_id"]), chi(winner)))
    correct = input("Did %s WIN? (y/n): " % cok(winner)).strip().lower() == "y"
    contracts_open      = int(row["contracts_open"])
    avg_entry           = float(row["avg_entry_price"])
    entry_fee_total     = float(row.get("entry_fee_total", 0))
    original_contracts  = int(row.get("contracts_original", contracts_open))
    allocated_entry_fee = round(entry_fee_total * contracts_open / max(original_contracts, 1), 4)
    remaining_cost_basis = round(contracts_open * avg_entry + allocated_entry_fee, 4)
    if correct:
        winnings        = float(contracts_open) * 1.0
        realized_profit = round(winnings - remaining_cost_basis, 4)
        df.loc[real_idx, "realized_cash"]   = round(float(df.loc[real_idx, "realized_cash"]   or 0) + winnings,        4)
        df.loc[real_idx, "realized_profit"] = round(float(df.loc[real_idx, "realized_profit"] or 0) + realized_profit, 4)
        print(cok("  WIN! +$%.4f" % realized_profit))
    else:
        realized_profit = round(-remaining_cost_basis, 4)
        df.loc[real_idx, "realized_profit"] = round(float(df.loc[real_idx, "realized_profit"] or 0) + realized_profit, 4)
        print(cerr("  LOSS"))
    df.loc[real_idx, "contracts_open"] = 0
    df.loc[real_idx, "status"]         = "settled"
    note    = str(row["notes"] or "")
    outcome = "WIN" if correct else "LOSS"
    df.loc[real_idx, "notes"] = "%s | manually resolved: %s %s" % (note, winner, outcome)
    save_predict_lots(df)
    print(cok("  Lot settled and saved."))


def summarize_predict_lots():
    df = load_predict_lots()
    if df.empty:
        print(cwarn("No contracts logged yet!"))
        return {}
    work = df.copy()
    numeric_cols = [
        "contracts_open","contracts_original","avg_entry_price","entry_fee_total",
        "entry_cost_total","realized_cash","realized_exit_fees","realized_profit","last_mark_price",
    ]
    for c in numeric_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0)
    work["unrealized_value_net"] = (
        work["contracts_open"] * work["last_mark_price"]
        - work["contracts_open"].apply(lambda x: calc_fee_from_payout(x * 1.0))
    )
    work["remaining_entry_cost_basis"] = (
        work["contracts_open"] * work["avg_entry_price"]
        + (work["entry_fee_total"] * work["contracts_open"]
           / work["contracts_original"].replace(0, np.nan)).fillna(0)
    )
    work["unrealized_profit"] = work["unrealized_value_net"] - work["remaining_entry_cost_basis"]
    work["total_pnl_marked"]  = work["realized_profit"] + work["unrealized_profit"]
    hdr("PREDICTS MONEYLINE LEDGER")
    display_cols = [
        "lot_id","date","market","predicted_winner","contracts_open","contracts_original",
        "avg_entry_price","entry_cost_total","realized_cash","realized_profit",
        "last_mark_price","unrealized_profit","total_pnl_marked","status",
    ]
    print(work[display_cols].to_string(index=False))
    div(110)
    closed       = work[work["status"].isin(["sold","settled"])]
    wins         = closed[closed["realized_profit"] > 0]
    win_rate     = (len(wins) / len(closed) * 100.0) if len(closed) else 0.0
    total_entry  = float(work["entry_cost_total"].sum())
    realized_pnl = float(work["realized_profit"].sum())
    unrealized   = float(work["unrealized_profit"].sum())
    marked_pnl   = float(work["total_pnl_marked"].sum())
    roi_realized = (realized_pnl / total_entry * 100.0) if total_entry else 0.0
    roi_marked   = (marked_pnl   / total_entry * 100.0) if total_entry else 0.0
    rp_s  = cok("$%+.4f" % realized_pnl)  if realized_pnl >= 0 else cerr("$%+.4f" % realized_pnl)
    up_s  = cok("$%+.4f" % unrealized)    if unrealized  >= 0 else cerr("$%+.4f" % unrealized)
    mp_s  = cok("$%+.4f" % marked_pnl)   if marked_pnl  >= 0 else cerr("$%+.4f" % marked_pnl)
    rr_s  = cok("%+.1f%%" % roi_realized) if roi_realized >= 0 else cerr("%+.1f%%" % roi_realized)
    rm_s  = cok("%+.1f%%" % roi_marked)   if roi_marked  >= 0 else cerr("%+.1f%%" % roi_marked)
    print("POSITIONS: %s | CLOSED WIN RATE: %s" % (cok(len(work)), cok("%.1f%%" % win_rate)))
    print("ENTRY FEES: %s | EXIT FEES: %s" % (cwarn("$%.4f" % work["entry_fee_total"].sum()), cwarn("$%.4f" % work["realized_exit_fees"].sum())))
    print("REALIZED P&L: %s | UNREALIZED: %s | MARKED P&L: %s" % (rp_s, up_s, mp_s))
    print("ROI REALIZED: %s | ROI MARKED: %s" % (rr_s, rm_s))
    div(110)
    return {
        "df": work, "win_rate": win_rate,
        "realized_profit": realized_pnl, "unrealized_profit": unrealized,
        "total_pnl_marked": marked_pnl, "roi_realized": roi_realized, "roi_marked": roi_marked,
    }


def plot_pnl_chart(output_file="predicts_pnl.png"):
    if not HAS_MPL:
        print("matplotlib not installed -- chart unavailable")
        return
    summary = summarize_predict_lots()
    if not summary:
        return
    df         = summary["df"].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df         = df.dropna(subset=["date"]).copy()
    if df.empty:
        print(cwarn("No dated contract data for chart."))
        return
    df["month"] = df["date"].dt.to_period("M").astype(str)
    monthly     = df.groupby("month")["realized_profit"].sum().reset_index()
    colors      = ["#2ecc71" if v >= 0 else "#e74c3c" for v in monthly["realized_profit"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(monthly["month"], monthly["realized_profit"], color=colors)
    ax.axhline(0, color="white", linewidth=0.8, linestyle="--")
    ax.set_title("Predicts - Monthly Realized P&L", color="white", fontsize=14, pad=12)
    ax.set_xlabel("Month", color="white")
    ax.set_ylabel("Profit ($)", color="white")
    ax.tick_params(colors="white", rotation=30)
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#16213e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(cok("P&L chart saved -> %s" % output_file))
