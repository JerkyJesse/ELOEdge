#!/usr/bin/env python3
"""
NBA ELOEdge -- Elo-only predictions + Predicts $1 contract tracker.
Rest days, back-to-back penalties, injury awareness, Platt-calibrated.
"""

import os
import logging

try:
    from colorama import Fore, Back, Style
except ImportError:
    class _Dummy:
        def __getattr__(self, _): return ""
    Fore = Back = Style = _Dummy()

from color_helpers import cok, cerr, cwarn, chi, cdim, cbold, div
from config import (
    GAMES_FILE, PLAYER_STATS_FILE, RATINGS_FILE, PLATT_SCALER_FILE, SETTINGS_FILE,
    load_elo_settings, save_elo_settings,
)
from platt import load_platt_scaler
from data_games import download_recent_games
from data_players import load_player_stats, download_player_stats, download_advanced_stats, show_player_metrics
from build_model import build_model
from backtest import (backtest_model, grid_search_optimization, genetic_optimization,
                      show_optimization_results, bayesian_optimization,
                      purged_walk_forward_cv, combinatorial_purged_cv,
                      probability_of_backtest_overfitting, monte_carlo_permutation_test,
                      rolling_origin_recalibration, kelly_criterion_backtest,
                      sliding_window_backtest, elo_convergence_analysis,
                      conformal_analysis, run_beta_calibration, auto_optimize,
                      super_optimize)
from predict_ledger import (
    load_predict_lots, save_predict_lots, add_predict_contract,
    show_open_lots, mark_pending_positions, invert_open_trade,
    sell_predict_contract, resolve_predict_contracts,
    summarize_predict_lots, plot_pnl_chart,
    prompt_balance, show_balance, show_kelly_recommendation,
    deposit_cash, withdraw_cash,
)
from live_scores import show_live_scores_for_open_trades
from auto_resolve import auto_resolve_finished_trades
from html_generator import generate_today_predictions_html, generate_tomorrow_predictions_html
from help_system import show_help
from injuries import show_injury_report, get_team_injuries, calc_injury_impact, manual_set_injuries, fetch_injury_report
from single_param_opt import run_coordinate_descent

import sys
try:
    from odds_tracker import show_odds_table, get_today_odds, find_game_odds
    from weather import show_weather_report, get_game_weather, compute_weather_impact
    pass  # sentiment module removed
    HAS_DATA_MODULES = True
except ImportError as _e:
    HAS_DATA_MODULES = False
    logging.debug("Optional data modules not available: %s", _e)

try:
    from kalshi import find_kalshi_odds, show_kalshi_odds
    HAS_KALSHI = True
except ImportError:
    HAS_KALSHI = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def dispatch(cmd, model, csv_file):
    if cmd == "all":
        model.show_all_teams()
    elif cmd == "refresh":
        for f in [GAMES_FILE, PLAYER_STATS_FILE, RATINGS_FILE, PLATT_SCALER_FILE]:
            try:
                if os.path.exists(f):
                    os.remove(f)
                    logging.info("Deleted %s to force refresh", f)
            except Exception as e:
                logging.warning("Could not delete %s: %s", f, e)
        new_file = download_recent_games()
        download_player_stats()
        if new_file:
            csv_file = new_file
            model    = build_model(csv_file)
            print(cok("Refresh complete -> games + players updated, model rebuilt"))
            print(cwarn("Run 'backtest' to refit the Platt calibration scaler."))
        else:
            print(cerr("Refresh failed - could not download recent games"))
    elif cmd in ("backtest", "backtest resume"):
        use_resume = "resume" in cmd
        _, metrics = backtest_model(csv_file, fit_platt=True, resume=use_resume)
        model._platt_scaler = load_platt_scaler()
        acc_s = cok("%.2f%%" % metrics.get("accuracy", 0))
        label = " (resume)" if use_resume else ""
        print("  Backtest%s accuracy: %s | LogLoss: %.4f | Brier: %.4f"
              % (label, acc_s, metrics.get("log_loss", 0), metrics.get("brier", 0)))
    elif cmd in ("platt", "calibrate"):
        if model._platt_scaler:
            ps = model._platt_scaler
            print("\n  Platt scaler: %s" % cok("ACTIVE"))
            print("  Fitted   : %s" % cdim(ps.get("fitted_at","?")))
            print("  Samples  : %s" % cok(ps.get("n_samples","?")))
            print("  Coef     : %.4f  Intercept: %.4f" % (ps.get("coef",0), ps.get("intercept",0)))
            print("  To refit : run %s" % chi("backtest"))
        else:
            print(cwarn("  No Platt scaler fitted yet. Run 'backtest' to fit one."))
    elif cmd == "grid":
        grid_search_optimization(csv_file)
        model = build_model(csv_file)
        model._platt_scaler = load_platt_scaler()
    elif cmd == "genetic":
        genetic_optimization(csv_file)
        model = build_model(csv_file)
        model._platt_scaler = load_platt_scaler()
    elif cmd == "results":
        show_optimization_results()
    elif cmd == "players":
        show_player_metrics(load_player_stats(), None, 10)
    elif cmd == "settings":
        model.show_settings()
    elif cmd == "live":
        show_live_scores_for_open_trades(model)
    elif cmd == "balance":
        show_balance()
    elif cmd == "deposit":
        deposit_cash()
    elif cmd == "withdraw":
        withdraw_cash()
    elif cmd == "portfolio":
        try:
            import importlib.util
            _pf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "portfolio.py")
            if os.path.exists(_pf_path):
                _spec = importlib.util.spec_from_file_location("portfolio", _pf_path)
                _pmod = importlib.util.module_from_spec(_spec)
                _spec.loader.exec_module(_pmod)
                _pmod.main()
            else:
                print(cerr("  Portfolio viewer not found at %s" % _pf_path))
        except Exception as e:
            print(cerr("  Portfolio error: %s" % e))
    elif cmd in ("predicts","summary"):
        summarize_predict_lots()
    elif cmd == "chart":
        plot_pnl_chart()
    elif cmd == "resolve":
        resolve_predict_contracts()
    elif cmd == "sell":
        sell_predict_contract()
    elif cmd == "mark":
        mark_pending_positions()
    elif cmd == "invert":
        invert_open_trade()
    elif cmd in ("today","html","blog","blogger"):
        generate_today_predictions_html(model)
    elif cmd == "tomorrow":
        generate_tomorrow_predictions_html(model)
    elif cmd in ("autoresolve on","autoresolve off"):
        enabled  = cmd.endswith("on")
        settings = load_elo_settings()
        settings["autoresolve_enabled"] = enabled
        save_elo_settings(settings)
        print(cok("  Auto-resolve is now %s. Saved to %s." % ("ENABLED" if enabled else "DISABLED", SETTINGS_FILE)))
    elif cmd == "autoresolve":
        count = auto_resolve_finished_trades(model, verbose=True)
        if count == 0:
            print(cwarn("  No lots auto-resolved (no matching final games found in today's feed)."))
        else:
            print(cok("  Auto-resolved %d lot(s)." % count))
    elif cmd == "bayesian":
        bayesian_optimization(csv_file)
        model = build_model(csv_file)
        model._platt_scaler = load_platt_scaler()
    elif cmd == "purgedcv":
        purged_walk_forward_cv(csv_file)
    elif cmd == "cpcv":
        combinatorial_purged_cv(csv_file)
    elif cmd == "pbo":
        probability_of_backtest_overfitting()
    elif cmd == "montecarlo":
        monte_carlo_permutation_test(csv_file)
    elif cmd == "rollingcal":
        rolling_origin_recalibration(csv_file)
    elif cmd == "kelly":
        kelly_criterion_backtest()
    elif cmd == "sliding":
        sliding_window_backtest(csv_file)
    elif cmd == "convergence":
        elo_convergence_analysis(csv_file)
    elif cmd == "conformal":
        conformal_analysis(csv_file)
    elif cmd == "betacal":
        run_beta_calibration(csv_file)
    elif cmd in ("autoopt", "auto-optimize", "auto optimize"):
        auto_optimize(csv_file)
        model = build_model(csv_file)
        model._platt_scaler = load_platt_scaler()
    elif cmd in ("superopt", "super-optimize", "super optimize", "super"):
        super_optimize(csv_file)
        model = build_model(csv_file)
        model._platt_scaler = load_platt_scaler()
    elif cmd in ("singleopt", "single-opt", "single opt", "coorddescent", "coord"):
        run_coordinate_descent(csv_file)
        model = build_model(csv_file)
        model._platt_scaler = load_platt_scaler()
    elif cmd == "injuries":
        show_injury_report(model)
    elif cmd.startswith("injuries set "):
        # injuries set <team> <player1>, <player2>
        rest = cmd[len("injuries set "):].strip()
        parts = rest.split(None, 1)
        if len(parts) >= 2:
            team = model.find_team(parts[0])
            if team:
                players = [p.strip() for p in parts[1].split(",")]
                manual_set_injuries(team, players)
                print(cok("  Marked %d player(s) OUT for %s: %s" % (len(players), team, ", ".join(players))))
            else:
                print(cerr("  Team not found: %s" % parts[0]))
        else:
            print(cwarn("  Usage: injuries set <team> <player1>, <player2>"))
    elif cmd.startswith("set "):
        try:
            from elo_set_handler import handle_elo_set, print_elo_settings
            args = cmd[4:].strip()
            if not args or args in ("help", "?", "list"):
                print_elo_settings(model, load_elo_settings)
            else:
                ok, msg = handle_elo_set(args, model, load_elo_settings,
                                         lambda s: save_elo_settings(s))
                if ok:
                    print(cok("  %s" % msg))
                    print(cwarn("  Tip: run 'backtest' to refit Platt scaler with new params."))
                else:
                    print(cerr("  %s" % msg))
        except ImportError:
            print(cerr("  Set handler not available. Check elo_set_handler.py"))
    # ── Odds / Weather commands ─────────────────────────────────────────
    elif cmd == "odds":
        if HAS_DATA_MODULES:
            show_odds_table("nba")
        else:
            print(cerr("Odds module not available. pip install requests"))
    elif cmd == "kalshi" or cmd.startswith("kalshi "):
        sub = cmd[6:].strip() if len(cmd) > 6 else ""
        if sub in ("help", "?"):
            print("\n  " + cbold("KALSHI SUBCOMMANDS"))
            print("  kalshi          Toggle auto_kalshi on/off")
            print("  kalshi on       Enable auto_kalshi (auto-fetch odds during predictions)")
            print("  kalshi off      Disable auto_kalshi")
            print("  kalshi odds     Show all current Kalshi game markets")
            print("  kalshi all      Same as 'kalshi odds'")
            print("  kalshi help     Show this help")
        elif sub in ("odds", "all"):
            if HAS_KALSHI:
                show_kalshi_odds("nba")
            else:
                print(cerr("Kalshi module not available. Check kalshi.py"))
        elif sub == "on":
            settings = load_elo_settings()
            settings["auto_kalshi"] = True
            save_elo_settings(settings)
            print(cok("  auto_kalshi is now ENABLED. Kalshi odds will auto-fetch during predictions."))
        elif sub == "off":
            settings = load_elo_settings()
            settings["auto_kalshi"] = False
            save_elo_settings(settings)
            print(cwarn("  auto_kalshi is now DISABLED."))
        elif sub == "":
            # Toggle auto_kalshi on/off
            settings = load_elo_settings()
            current = settings.get("auto_kalshi", False)
            settings["auto_kalshi"] = not current
            save_elo_settings(settings)
            state = "ENABLED" if not current else "DISABLED"
            print(cok("  auto_kalshi toggled -> %s" % state) if not current else cwarn("  auto_kalshi toggled -> %s" % state))
        else:
            print(cwarn("  Unknown kalshi subcommand: '%s'. Try 'kalshi help'." % sub))
    elif cmd == "weather":
        if HAS_DATA_MODULES:
            team = input(chi("  Home team: ")).strip()
            if team:
                found = model.find_team(team)
                if found:
                    show_weather_report(found, "nba")
                else:
                    print(cerr("  Team not found: %s" % team))
        else:
            print(cerr("Weather module not available."))
    elif cmd == "sentiment":
        print(cerr("Sentiment module has been removed."))
    elif cmd.startswith("help"):
        parts = cmd.split(None, 1)
        show_help(parts[1] if len(parts) > 1 else "")
    else:
        print(cwarn("Unknown command: '%s'" % cmd))
    return model, csv_file


def main():
    print(Back.GREEN + Fore.BLACK + Style.BRIGHT
          + "  NBA SHARPSTACK  [ELO-only]  "
          + Style.RESET_ALL)
    div(80)
    print("""
WORKFLOW:
1. Run %s FIRST -> fits calibration scaler (better probabilities)
2. Type team name (e.g. Lakers) -> opponent -> home? (a/b/n)?
3. See calibrated prediction + key player metrics
4. Type 'y' to log moneyline contract  |  'resolve' after game

PREDICTIONS: <teamname> | today | tomorrow | all | players | injuries | settings

DATA:        refresh | odds | kalshi | weather

BACKTEST:    backtest

ELO OPT:    grid | genetic | bayesian | autoopt | superopt | singleopt | results

VALIDATE:    purgedcv | cpcv | pbo | montecarlo | convergence | sliding
             rollingcal | conformal | betacal | kelly

SETTINGS (39 Elo params, type 'set' to see all):
  set k=8.38 | set home=50 | set boost=20 | set rest=10 | set b2b=30
  set travel=15 | set pace=35 | set sos=5 | set streak=5
  set kelly=quarter | set balance=1000 | set autoresolve=true

TRADING:     predicts | balance | resolve | sell | mark | invert | chart
             live | autoresolve | autoresolve on/off

             help [command] | quit
""" % chi("backtest"))

    csv_file = download_recent_games()
    download_player_stats()
    download_advanced_stats()
    injuries = fetch_injury_report()
    if injuries:
        out_count = sum(1 for i in injuries if i.get("status") in ("Out", "Doubtful"))
        print(cdim("  Injuries loaded: %d players (%d Out/Doubtful)" % (len(injuries), out_count)))
    if not csv_file:
        logging.error("Could not obtain game data. Exiting.")
        return

    model = build_model(csv_file)

    model._platt_scaler = load_platt_scaler()
    if model._platt_scaler:
        print(cdim("  Platt scaler loaded from previous backtest"))
    else:
        print(cwarn("  No Platt scaler found -- run 'backtest' to fit calibration"))

    if load_elo_settings().get("autoresolve_enabled", False):
        auto_resolve_finished_trades(model, verbose=True)

    from config import load_portfolio_settings
    psettings = load_portfolio_settings()
    if float(psettings.get("starting_balance", 0)) <= 0:
        prompt_balance()

    KNOWN_CMDS = {
        "all","refresh","backtest","grid","genetic",
        "bayesian","results","players","settings","predicts","summary","chart",
        "resolve","sell","mark","invert","live","help","autoresolve","balance","deposit","withdraw","portfolio",
        "autoresolve on","autoresolve off","today","html","blog","blogger",
        "tomorrow","platt","calibrate","injuries","purgedcv","cpcv",
        "pbo","montecarlo","rollingcal","kelly","sliding","convergence",
        "conformal","betacal","autoopt","auto-optimize","auto optimize",
        "superopt","super-optimize","super optimize","super",
        "singleopt","single-opt","single opt","coorddescent","coord",
        "kalshi","kalshi on","kalshi off","kalshi odds","kalshi all","kalshi help",
    }

    while True:
        try:
            cmd = input("\n%s " % cok(">")).strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd in ("quit","q"):
            model.metadata["settings"] = model.settings_dict()
            model.save()
            save_elo_settings(model.settings_dict())
            break

        if cmd in ("today","html","blog","blogger"):
            print("\n" + chi("Generating today's NBA predictions HTML..."))
            generate_today_predictions_html(model)
            continue

        if cmd == "tomorrow":
            print("\n" + chi("Generating tomorrow's NBA predictions HTML..."))
            generate_tomorrow_predictions_html(model)
            continue

        if cmd not in KNOWN_CMDS and not cmd.startswith("set ") and cmd.strip():
            team_a = model.find_team(cmd)
            if team_a:
                team_b_input = input("Opponent team: ").strip()
                team_b       = model.find_team(team_b_input)
                if not team_b:
                    print(cerr("Team not found: '%s'" % team_b_input))
                    continue
                home_input = input(
                    "Home team? (a = first team home, b = second, n = neutral): "
                ).strip().lower()
                if home_input == "a":
                    team_a_home, neutral = True, False
                elif home_input == "b":
                    team_a_home, neutral = False, False
                else:
                    team_a_home, neutral = None, True
                winner, prob = model.pick_winner(
                    team_a, team_b, team_a_home=team_a_home, neutral_site=neutral,
                )
                cal_label = cdim(" (calibrated)") if model._platt_scaler else ""
                prob_s    = cok("%.1f%%" % (prob * 100))
                print("\n   %s - %s win probability%s" % (cok(winner), prob_s, cal_label))
                print("    %s Elo: %s  |  %s Elo: %s"
                      % (chi(team_a), cok("%.0f" % model.ratings[team_a]),
                         chi(team_b), cok("%.0f" % model.ratings[team_b])))
                if neutral:
                    site_s = "Neutral"
                elif team_a_home:
                    site_s = "%s home" % chi(team_a)
                else:
                    site_s = "%s home" % chi(team_b)
                print("    Site: %s" % site_s)

                # Show injury impact if any
                for t in (team_a, team_b):
                    out = get_team_injuries(t)
                    if out:
                        impact = calc_injury_impact(t, out)
                        print("    %s injuries: %s %s" % (
                            cwarn(t), ", ".join(out),
                            cwarn("(%.0f Elo)" % impact)))

                player_df = load_player_stats()
                if not player_df.empty:
                    print("\n  " + cbold("KEY PLAYERS (season averages):"))
                    show_player_metrics(player_df, team_a, 3)
                    show_player_metrics(player_df, team_b, 3)
                kelly_contracts = 0
                market_cents = None
                # Auto-fetch from Kalshi if enabled
                settings_now = load_elo_settings()
                if HAS_KALSHI and settings_now.get("auto_kalshi"):
                    h_team = team_a if team_a_home else team_b
                    a_team = team_b if team_a_home else team_a
                    kalshi = find_kalshi_odds(h_team, a_team, "nba")
                    if kalshi and (kalshi.get("home_midpoint") or kalshi.get("away_midpoint")):
                        h_mid = kalshi.get("home_midpoint")
                        a_mid = kalshi.get("away_midpoint")
                        # Show both sides
                        h_s = "%s: %d\u00a2 (%d%%)" % (h_team, h_mid, h_mid) if h_mid else "%s: ?" % h_team
                        a_s = "%s: %d\u00a2 (%d%%)" % (a_team, a_mid, a_mid) if a_mid else "%s: ?" % a_team
                        print("\n   %s  %s  |  %s" % (cok("KALSHI"), chi(h_s), chi(a_s)))
                        # Use the predicted winner's side for Kelly sizing
                        if winner == h_team and h_mid:
                            market_cents = float(h_mid)
                        elif winner == a_team and a_mid:
                            market_cents = float(a_mid)
                        elif h_mid:
                            market_cents = float(h_mid)
                    else:
                        print(cdim("\n   Kalshi: no matching market found, enter manually"))
                if market_cents is None:
                    odds_input = input(
                        "\nActual trade odds in cents (e.g. 62 for $0.62, or Enter to skip): "
                    ).strip()
                    if odds_input:
                        try:
                            market_cents = float(odds_input)
                            if not (1 <= market_cents <= 99):
                                print(cwarn("  Odds must be between 1-99 cents."))
                                market_cents = None
                        except ValueError:
                            print(cwarn("  Couldn't parse odds input."))
                if market_cents and 1 <= market_cents <= 99:
                    kelly_contracts = show_kelly_recommendation(prob, market_cents)
                log_choice = input(
                    "\nLog this moneyline pick as a Predicts position? (y/n): "
                ).strip().lower()
                if log_choice == "y":
                    h = team_a if team_a_home else team_b
                    a = team_b if team_a_home else team_a
                    add_predict_contract(h, a, winner, prob, kelly_contracts)
                    try:
                        df = load_predict_lots()
                        if not df.empty:
                            last_idx   = df.index[-1]
                            clean_note = "Model %.1f%% - %s" % (prob * 100, winner)
                            if model._platt_scaler:
                                clean_note += " (calibrated)"
                            df.loc[last_idx, "notes"] = clean_note
                            save_predict_lots(df)
                            print(cdim("  Note updated: %s" % clean_note))
                    except Exception as e:
                        print(cwarn("  Could not update note: %s" % e))
                continue

        model, csv_file = dispatch(cmd, model, csv_file)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nFATAL ERROR: %s" % e)
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
    except KeyboardInterrupt:
        pass
