"""Live NHL scores fetching and open-trade tracking display."""

import os
import json
import time
import logging
import urllib.request
from datetime import datetime, timedelta

from config import get_team_abbr, load_elo_settings
from color_helpers import cok, cerr, cwarn, chi, cdim, cyel, div

ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"


def _fetch_espn_scoreboard(date_str=None):
    """Fetch NHL scoreboard from ESPN public API for a given date or today."""
    try:
        url = ESPN_SCOREBOARD_URL
        if date_str:
            url = "%s?dates=%s" % (ESPN_SCOREBOARD_URL, date_str)
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        results = []
        for event in data.get("events", []):
            for comp in event.get("competitions", []):
                status_obj = comp.get("status", {})
                status_type = status_obj.get("type", {})
                status_text = status_type.get("shortDetail", "") or status_type.get("detail", "")
                completed = status_type.get("completed", False)
                period = int(status_obj.get("period", 0))

                home_team = away_team = ""
                home_abbr = away_abbr = ""
                home_score = away_score = 0

                for competitor in comp.get("competitors", []):
                    team_info = competitor.get("team", {})
                    team_name = team_info.get("displayName", "")
                    team_tri = team_info.get("abbreviation", "")
                    score = int(competitor.get("score", 0) or 0)
                    if competitor.get("homeAway") == "home":
                        home_team = team_name
                        home_abbr = team_tri
                        home_score = score
                    else:
                        away_team = team_name
                        away_abbr = team_tri
                        away_score = score

                if home_team and away_team:
                    if completed:
                        display_status = "Final"
                        if period > 3:
                            display_status = "Final/OT" if period == 4 else "Final/SO"
                    else:
                        display_status = status_text or "Scheduled"

                    results.append({
                        "home_team_city": home_team,
                        "home_abbr": home_abbr,
                        "away_team_city": away_team,
                        "away_abbr": away_abbr,
                        "home_score": home_score,
                        "away_score": away_score,
                        "status": display_status,
                        "period": period,
                    })
        return results
    except Exception as e:
        logging.warning("ESPN scoreboard fetch failed: %s", e)
        return []


def fetch_today_nhl_games():
    """Fetch today's scheduled NHL games from ESPN."""
    return _fetch_espn_scoreboard()


def fetch_tomorrow_nhl_games():
    """Fetch tomorrow's scheduled NHL games from ESPN."""
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y%m%d")
    return _fetch_espn_scoreboard(tomorrow)


def fetch_live_nhl_scores():
    """Fetch live NHL scores from ESPN scoreboard."""
    return _fetch_espn_scoreboard()


def match_abbr_to_full(abbr, model):
    abbr_upper = abbr.upper()
    for full_name in model.team_names:
        if get_team_abbr(full_name) == abbr_upper:
            return full_name
    return None


def _print_live_scores_once(model):
    from predict_ledger import load_predict_lots
    df      = load_predict_lots()
    open_df = df[df["contracts_open"].fillna(0).astype(float) > 0].copy()
    live    = fetch_live_nhl_scores()
    div(100)
    print("  " + chi("LIVE NHL SCORES - TODAY'S GAMES"))
    div(100)
    if not live:
        print(cwarn("  No live scores available right now."))
    else:
        print("  %-38s %-15s %-20s" % (chi("MATCHUP"), chi("SCORE"), chi("STATUS")))
        div(74)
        for g in live:
            matchup = "%s (%s) @ %s (%s)" % (
                g["away_team_city"], g["away_abbr"],
                g["home_team_city"], g["home_abbr"],
            )
            score  = "%s - %s" % (g["away_score"], g["home_score"])
            is_fin = "final" in g["status"].lower()
            print("  %-38s %-15s %-20s"
                  % (matchup,
                     cok(score)        if is_fin else cyel(score),
                     cok(g["status"])  if is_fin else cwarn(g["status"])))
    print()
    div(100)
    print("  " + chi("OPEN MONEYLINE POSITIONS - LIVE STATUS"))
    div(100)
    if open_df.empty:
        print(cwarn("  No open positions to track."))
        div(100)
        return
    matched_any = False
    for _, row in open_df.iterrows():
        home_full   = str(row["home_team"])
        away_full   = str(row["away_team"])
        pred        = str(row["predicted_winner"])
        contracts   = int(row["contracts_open"])
        entry_price = float(row["avg_entry_price"])
        mark_price  = float(row["last_mark_price"])
        print("\n  Lot #%d | %s @ %s" % (int(row["lot_id"]), chi(away_full), chi(home_full)))
        print("    Pick       : %s" % chi(pred))
        print("    Contracts  : %dx @ %s  |  Current mark: %s"
              % (contracts, cok("$%.2f" % entry_price), cwarn("$%.2f" % mark_price)))
        matched_game = None
        lot_abbrs    = {get_team_abbr(home_full), get_team_abbr(away_full)}
        for g in live:
            if {g["home_abbr"], g["away_abbr"]} == lot_abbrs:
                matched_game = g
                break
            live_home = (match_abbr_to_full(g["home_abbr"], model) or g["home_team_city"]).lower()
            live_away = (match_abbr_to_full(g["away_abbr"], model) or g["away_team_city"]).lower()
            lot_names = {home_full.lower(), away_full.lower()}
            if (any(n in live_home or live_home in n for n in lot_names) and
                    any(n in live_away or live_away in n for n in lot_names)):
                matched_game = g
                break
        if matched_game:
            matched_any = True
            hs          = matched_game["home_score"]
            a_s         = matched_game["away_score"]
            lead_team   = home_full if hs > a_s else (away_full if a_s > hs else "TIED")
            margin      = abs(hs - a_s)
            is_winning  = pred.lower() in lead_team.lower() if lead_team != "TIED" else False
            if is_winning:
                icon = cok("[WINNING]")
            elif lead_team == "TIED":
                icon = cwarn("[TIED]")
            else:
                icon = cerr("[LOSING]")
            print("    Live score : %s %d - %d %s"
                  % (matched_game["away_abbr"], a_s, hs, matched_game["home_abbr"]))
            print("    Status     : %s  %s" % (matched_game["status"], icon))
            if is_winning:
                print("    -> Your pick (%s) is currently WINNING" % cok(pred))
            elif lead_team == "TIED":
                print("    -> Game is currently TIED")
            else:
                print("    -> Your pick is currently LOSING -- %s leads by %d"
                      % (cerr(lead_team), margin))
        else:
            print(cdim("    Live score : Not found in today's feed"))
    if not matched_any and live:
        print(cwarn("\n  None of your open positions match any game in today's live feed."))
    div(100)


def show_live_scores_for_open_trades(model):
    from auto_resolve import auto_resolve_finished_trades
    REFRESH_SECONDS = 60
    print("\n  " + chi("Live score tracker started") + " - refreshing every 60 seconds.")
    print("  Press  Ctrl+C  to stop and return to the prompt.\n")
    iteration = 0
    try:
        while True:
            try:
                iteration += 1
                if iteration > 1:
                    os.system("cls" if os.name == "nt" else "clear")
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(cdim("  [Refresh #%d  -  %s  -  next update in %ds  |  Ctrl+C to stop]"
                            % (iteration, now_str, REFRESH_SECONDS)))
                _print_live_scores_once(model)
                if load_elo_settings().get("autoresolve_enabled", False):
                    _resolved = auto_resolve_finished_trades(model, verbose=True)
                    if _resolved:
                        print(cok("  [auto-resolve] %d lot(s) settled this refresh." % _resolved))
                for remaining in range(REFRESH_SECONDS, 0, -1):
                    time.sleep(1)
                    print("\r  %s" % cdim("Next refresh in %2ds ...   " % remaining), end="", flush=True)
                print()
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logging.error("Live scores error: %s", e)
                print(f"  \u26a0 Error refreshing scores: {e}. Retrying...")
                time.sleep(10)
                continue
    except KeyboardInterrupt:
        print("\n\n  " + cwarn("Live tracker stopped. Returning to main prompt.\n"))
