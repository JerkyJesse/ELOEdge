"""Live NFL scores fetching and open-trade tracking display via ESPN API."""

import os
import json
import time
import logging
from datetime import datetime, timedelta

import requests

from config import get_team_abbr, load_elo_settings
from color_helpers import cok, cerr, cwarn, chi, cdim, cyel, div


ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"


def _parse_espn_scoreboard(data):
    """Parse ESPN scoreboard JSON into a list of game dicts."""
    results = []
    for event in data.get("events", []):
        for comp in event.get("competitions", []):
            competitors = comp.get("competitors", [])
            if len(competitors) != 2:
                continue
            home_team = away_team = ""
            home_abbr = away_abbr = ""
            home_score = away_score = 0
            for c in competitors:
                team_info = c.get("team", {})
                name = team_info.get("displayName", "")
                abbr = team_info.get("abbreviation", "")
                score = int(c.get("score", 0) or 0)
                if c.get("homeAway") == "home":
                    home_team = name
                    home_abbr = abbr
                    home_score = score
                else:
                    away_team = name
                    away_abbr = abbr
                    away_score = score
            status_obj = comp.get("status", {})
            status_type = status_obj.get("type", {})
            status_text = status_type.get("shortDetail", "")
            period = int(status_obj.get("period", 0))
            completed = status_type.get("completed", False)

            if completed:
                status_text = "Final"

            results.append({
                "home_team_city": home_team,
                "home_abbr": home_abbr,
                "away_team_city": away_team,
                "away_abbr": away_abbr,
                "home_score": home_score,
                "away_score": away_score,
                "status": status_text,
                "period": period,
            })
    return results


def _fetch_espn_scoreboard(date_str=None):
    """Fetch NFL scoreboard from ESPN. date_str format: YYYYMMDD or None for today."""
    try:
        url = ESPN_SCOREBOARD_URL
        if date_str:
            url += "?dates=%s" % date_str
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return _parse_espn_scoreboard(resp.json())
    except Exception as e:
        logging.warning("ESPN scoreboard fetch failed: %s", e)
        return []


def fetch_today_nfl_games():
    """Fetch today's scheduled NFL games from ESPN."""
    return _fetch_espn_scoreboard()


def fetch_tomorrow_nfl_games():
    """Fetch tomorrow's scheduled NFL games from ESPN."""
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y%m%d")
    return _fetch_espn_scoreboard(tomorrow)


def fetch_live_nfl_scores():
    """Fetch live NFL scores for today."""
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
    live    = fetch_live_nfl_scores()
    div(100)
    print("  " + chi("LIVE NFL SCORES - TODAY'S GAMES"))
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
        print("\n\n  " + cwarn("Live tracker stopped. Returning to main prompt.\n"))
