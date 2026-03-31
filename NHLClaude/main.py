#!/usr/bin/env python3
"""
NHL MONEYBALL + PREDICTS $1 CONTRACT TRACKER
v3: Elo + XGBoost ensemble, rest days, injury awareness
"""

import os
import logging

from colorama import Fore, Back, Style

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
)
from live_scores import show_live_scores_for_open_trades
from auto_resolve import auto_resolve_finished_trades
from html_generator import generate_today_predictions_html, generate_tomorrow_predictions_html
from help_system import show_help
from enhanced_model import (run_enhanced_backtest, save_enhanced_model,
                            load_enhanced_model, shap_feature_importance)
from injuries import show_injury_report, get_team_injuries, calc_injury_impact, manual_set_injuries, fetch_injury_report
from single_param_opt import run_coordinate_descent

try:
    from mega_backtest import run_mega_backtest
    HAS_MEGA_BACKTEST = True
except ImportError as _e:
    HAS_MEGA_BACKTEST = False
    logging.debug("Mega backtest not available: %s", _e)

try:
    from mega_optimizer import (run_mega_optimize, run_quick_optimize,
                                 run_deep_optimize, run_single_model_optimize)
    HAS_MEGA_OPTIMIZER = True
except ImportError as _e:
    HAS_MEGA_OPTIMIZER = False
    logging.debug("Mega optimizer not available: %s", _e)

try:
    from mega_config import (load_model_switches, save_model_switches,
                              is_model_enabled, print_model_status, ALL_MODELS,
                              handle_mega_set, print_mega_settings)
    HAS_MEGA_CONFIG = True
except ImportError:
    HAS_MEGA_CONFIG = False

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from odds_tracker import show_odds_table, get_today_odds, find_game_odds
    from weather import show_weather_report, get_game_weather, compute_weather_impact
    from sentiment import show_sentiment_report, fetch_all_team_sentiment, get_sentiment_features
    HAS_MEGA_DATA = True
except ImportError as _e:
    HAS_MEGA_DATA = False
    logging.debug("Mega-ensemble data modules not available: %s", _e)

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
    elif cmd == "backtest":
        _, metrics = backtest_model(csv_file, fit_platt=True)
        model._platt_scaler = load_platt_scaler()
        acc_s = cok("%.2f%%" % metrics.get("accuracy", 0))
        print("  Backtest accuracy: %s | LogLoss: %.4f | Brier: %.4f"
              % (acc_s, metrics.get("log_loss", 0), metrics.get("brier", 0)))
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
    elif cmd == "enhanced":
        print(chi("  Running enhanced backtest (Elo + XGBoost ensemble)..."))
        result = run_enhanced_backtest(csv_file, elo_weight=0.8, label="enhanced")
        if result and result.get("xgb_model"):
            save_enhanced_model(result)
            print(cok("  Enhanced model saved."))
    elif cmd == "enhanced decay":
        print(chi("  Running enhanced backtest with TIME-DECAYED weighting..."))
        result = run_enhanced_backtest(csv_file, elo_weight=0.8, label="enhanced-decay", time_decay=True)
        if result and result.get("xgb_model"):
            save_enhanced_model(result)
            print(cok("  Enhanced model saved (time-decayed)."))
    elif cmd == "shap":
        shap_feature_importance()
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
    # ── Mega-ensemble commands ─────────────────────────────────────────
    elif cmd in ("mega", "megabacktest", "mega backtest"):
        if HAS_MEGA_BACKTEST:
            from elo_model import NHLElo
            settings = load_elo_settings()
            player_df_fresh = load_player_stats()
            print(cdim("  Running mega-ensemble backtest (12+ models)..."))
            print(cdim("  This may take a few minutes."))
            result = run_mega_backtest(
                csv_file, sport="nhl",
                elo_model_class=NHLElo,
                elo_settings=settings,
                player_df=player_df_fresh,
            )
            if result:
                print(cok("\n  Mega-ensemble: %.2f%% accuracy | LogLoss: %.4f | Brier: %.4f"
                          % (result["accuracy"], result["log_loss"], result["brier"])))
        else:
            print(cerr("Mega backtest not available. Check imports."))
    elif cmd in ("mega optimize", "megaopt", "mega opt", "megaoptimize"):
        if HAS_MEGA_OPTIMIZER:
            from elo_model import NHLElo
            settings = load_elo_settings()
            player_df_fresh = load_player_stats()
            print(cdim("  Starting mega-ensemble optimization (all phases)..."))
            print(cdim("  This will take a LONG time. No shortcuts."))
            print()
            result = run_deep_optimize(
                csv_file, sport="nhl",
                elo_model_class=NHLElo,
                elo_settings=settings,
                player_df=player_df_fresh,
            )
            if result and result.get("best_results"):
                br = result["best_results"]
                print(cok("\n  Optimized: %.2f%% accuracy | LogLoss: %.4f | Brier: %.4f"
                          % (br["accuracy"], br["log_loss"], br["brier"])))
                print(cok("  Settings saved. Run 'mega' to use them."))
        else:
            print(cerr("Mega optimizer not available. Check imports."))
    elif cmd in ("mega quick", "megaquick"):
        if HAS_MEGA_OPTIMIZER:
            from elo_model import NHLElo
            settings = load_elo_settings()
            player_df_fresh = load_player_stats()
            print(cdim("  Quick mega optimization (grid search only)..."))
            result = run_quick_optimize(
                csv_file, sport="nhl",
                elo_model_class=NHLElo,
                elo_settings=settings,
                player_df=player_df_fresh,
            )
            if result and result.get("best_results"):
                br = result["best_results"]
                print(cok("\n  Optimized: %.2f%% accuracy | LogLoss: %.4f | Brier: %.4f"
                          % (br["accuracy"], br["log_loss"], br["brier"])))
        else:
            print(cerr("Mega optimizer not available. Check imports."))
    elif cmd in ("mega ablation", "mega ablate", "mega single", "megasingle"):
        if HAS_MEGA_OPTIMIZER:
            from elo_model import NHLElo
            settings = load_elo_settings()
            player_df_fresh = load_player_stats()
            print(cdim("  Running single-model ablation study..."))
            print(cdim("  Tests each model's individual contribution."))
            print()
            run_single_model_optimize(
                csv_file, sport="nhl",
                elo_model_class=NHLElo,
                elo_settings=settings,
                player_df=player_df_fresh,
            )
        else:
            print(cerr("Mega optimizer not available. Check imports."))
    elif cmd in ("mega models", "mega status", "models"):
        if HAS_MEGA_CONFIG:
            sport_dir = os.path.dirname(os.path.abspath(csv_file))
            switches = load_model_switches("nhl", sport_dir)
            print_model_status(switches)
        else:
            print(cerr("Mega config not available."))
    elif cmd.startswith("mega on ") or cmd.startswith("mega enable "):
        if HAS_MEGA_CONFIG:
            model_name = cmd.split()[-1].strip().lower()
            sport_dir = os.path.dirname(os.path.abspath(csv_file))
            switches = load_model_switches("nhl", sport_dir)
            if model_name in ALL_MODELS:
                switches[model_name] = True
                save_model_switches("nhl", sport_dir, switches)
                print(cok("  Enabled: %s" % model_name))
            elif model_name == "all":
                for m in ALL_MODELS:
                    switches[m] = True
                save_model_switches("nhl", sport_dir, switches)
                print(cok("  All models enabled"))
            else:
                print(cerr("  Unknown model: %s" % model_name))
                print(cdim("  Available: %s" % ", ".join(ALL_MODELS)))
        else:
            print(cerr("Mega config not available."))
    elif cmd.startswith("mega set "):
        if HAS_MEGA_CONFIG:
            sport_dir = os.path.dirname(os.path.abspath(csv_file))
            args = cmd[9:]  # everything after "mega set "
            ok, msg = handle_mega_set(args, "nhl", sport_dir)
            print(("  " + cok(msg)) if ok else ("  " + cerr(msg)))
            if ok:
                print(cwarn("  Tip: run 'mega' to use new settings."))
        else:
            print(cerr("Mega config not available."))
    elif cmd in ("mega settings", "mega params", "mega config"):
        if HAS_MEGA_CONFIG:
            sport_dir = os.path.dirname(os.path.abspath(csv_file))
            print_mega_settings("nhl", sport_dir)
        else:
            print(cerr("Mega config not available."))
    elif cmd.startswith("mega off ") or cmd.startswith("mega disable "):
        if HAS_MEGA_CONFIG:
            model_name = cmd.split()[-1].strip().lower()
            sport_dir = os.path.dirname(os.path.abspath(csv_file))
            switches = load_model_switches("nhl", sport_dir)
            if model_name in ALL_MODELS:
                switches[model_name] = False
                save_model_switches("nhl", sport_dir, switches)
                print(cwarn("  Disabled: %s" % model_name))
            else:
                print(cerr("  Unknown model: %s" % model_name))
        else:
            print(cerr("Mega config not available."))
    # ── Odds / Weather / Sentiment commands ─────────────────────────────
    elif cmd == "odds":
        if HAS_MEGA_DATA:
            show_odds_table("nhl")
        else:
            print(cerr("Odds module not available. pip install requests"))
    elif cmd == "weather":
        if HAS_MEGA_DATA:
            team = input(chi("  Home team: ")).strip()
            if team:
                found = model.find_team(team)
                if found:
                    show_weather_report(found, "nhl")
                else:
                    print(cerr("  Team not found: %s" % team))
        else:
            print(cerr("Weather module not available."))
    elif cmd == "sentiment":
        if HAS_MEGA_DATA:
            print(cdim("  Fetching Reddit sentiment (requires Reddit API credentials)..."))
            show_sentiment_report("nhl")
        else:
            print(cerr("Sentiment module not available."))
    elif cmd.startswith("help"):
        parts = cmd.split(None, 1)
        show_help(parts[1] if len(parts) > 1 else "")
    else:
        print(cwarn("Unknown command: '%s'" % cmd))
    return model, csv_file


def main():
    print(Back.GREEN + Fore.BLACK + Style.BRIGHT
          + "  NHL MONEYBALL + PREDICTS $1 CONTRACT TRACKER  [v3 - Elo+XGBoost ensemble]  "
          + Style.RESET_ALL)
    div(80)
    print("""
WORKFLOW:
1. Run %s FIRST -> fits calibration scaler (better probabilities)
2. Type team name (e.g. Bruins) -> opponent -> home? (a/b/n)?
3. See calibrated prediction + key player metrics
4. Type 'y' to log moneyline contract
5. After game: type 'resolve' | to sell early: type 'sell'
6. Type 'mark' to update current market marks | 'chart' for P&L chart

COMMANDS: teamname | all | refresh | backtest | enhanced | platt | grid | genetic
          bayesian | players | predicts | resolve | sell | mark | chart | settings
          results | live | invert | autoresolve | autoresolve on | autoresolve off
          balance        Set/view starting bankroll for Kelly criterion sizing
          injuries       Show NHL injury report with Elo impact estimates
          injuries set <team> <p1>,<p2>   Manually mark players as OUT
          today / html / blogger   Generate HTML table for today's games
          tomorrow                Generate HTML table for tomorrow's games
          set k=8 | set home=25 | set boost=0 | set rest=5
          set kelly=quarter | set kelly=half   Kelly sizing fraction | quit

OPTIMIZE: autoopt (auto grid+genetic+bayesian) | superopt (exhaustive, all 9 params)
          singleopt (coordinate descent, one param at a time)

ADVANCED: purgedcv | cpcv | pbo | montecarlo | convergence | sliding | rollingcal
          conformal | betacal | kelly | shap | enhanced decay
""" % chi("backtest"))

    csv_file = download_recent_games()
    download_player_stats()
    download_advanced_stats()
    injuries = fetch_injury_report()
    if injuries:
        out_count = sum(1 for i in injuries if i.get("status") in ("Out", "Day-to-Day"))
        print(cdim("  Injuries loaded: %d players (%d Out/Day-to-Day)" % (len(injuries), out_count)))
    if not csv_file:
        logging.error("Could not obtain game data. Exiting.")
        return

    model = build_model(csv_file)

    print("\n" + chi("BASELINE PERFORMANCE:"))
    _, baseline = backtest_model(csv_file, model=model, fit_platt=True)
    model._platt_scaler = load_platt_scaler()
    acc_s = cok("%.2f%%" % baseline.get("accuracy", 0))
    print("  Baseline accuracy: %s | LogLoss: %.4f | Brier: %.4f"
          % (acc_s, baseline.get("log_loss", 0), baseline.get("brier", 0)))

    if load_elo_settings().get("autoresolve_enabled", False):
        auto_resolve_finished_trades(model, verbose=True)

    settings = load_elo_settings()
    if float(settings.get("starting_balance", 0)) <= 0:
        prompt_balance()

    KNOWN_CMDS = {
        "all","refresh","backtest","enhanced","enhanced decay","grid","genetic",
        "bayesian","results","players","settings","predicts","summary","chart",
        "resolve","sell","mark","invert","live","help","autoresolve","balance",
        "autoresolve on","autoresolve off","today","html","blog","blogger",
        "tomorrow","platt","calibrate","injuries","shap","purgedcv","cpcv",
        "pbo","montecarlo","rollingcal","kelly","sliding","convergence",
        "conformal","betacal","autoopt","auto-optimize","auto optimize",
        "superopt","super-optimize","super optimize","super",
        "singleopt","single-opt","single opt","coorddescent","coord",
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
            print("\n" + chi("Generating today's NHL predictions HTML..."))
            generate_today_predictions_html(model)
            continue

        if cmd == "tomorrow":
            print("\n" + chi("Generating tomorrow's NHL predictions HTML..."))
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
                    print("\n  " + cbold("KEY PLAYERS (season stats):"))
                    show_player_metrics(player_df, team_a, 3)
                    show_player_metrics(player_df, team_b, 3)
                kelly_contracts = 0
                odds_input = input(
                    "\nActual trade odds in cents (e.g. 62 for $0.62, or Enter to skip): "
                ).strip()
                if odds_input:
                    try:
                        market_cents = float(odds_input)
                        if 1 <= market_cents <= 99:
                            kelly_contracts = show_kelly_recommendation(prob, market_cents)
                        else:
                            print(cwarn("  Odds must be between 1-99 cents."))
                    except ValueError:
                        print(cwarn("  Couldn't parse odds input."))
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
