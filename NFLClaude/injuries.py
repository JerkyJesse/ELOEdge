"""NFL injury report via ESPN public API + impact scoring."""

import os
import json
import logging
from datetime import datetime

import requests
import pandas as pd

from config import current_timestamp, get_team_abbr, TEAM_ABBR
from data_players import load_player_stats, load_advanced_stats

INJURY_CACHE_FILE = "nfl_injuries.json"
INJURY_CACHE_MAX_HOURS = 4
ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"


def fetch_injury_report():
    """Fetch NFL injury report from ESPN public API.
    Returns list of {player, team, status, detail}.
    Uses cache if fresh enough.
    """
    cached = _load_cache()
    if cached is not None:
        return cached

    injuries = _fetch_from_espn()
    if injuries:
        _save_cache(injuries)
        return injuries

    return []


def _fetch_from_espn():
    """Fetch structured injury data from ESPN's public JSON API."""
    try:
        resp = requests.get(ESPN_INJURIES_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logging.warning("ESPN injury fetch failed: %s", e)
        return None

    injuries = []
    for team_block in data.get("injuries", []):
        team_name = team_block.get("displayName", "")

        for entry in team_block.get("injuries", []):
            status = entry.get("status", "")
            athlete = entry.get("athlete", {})
            player = athlete.get("displayName", "")
            detail = entry.get("shortComment", "")
            position = athlete.get("position", {}).get("abbreviation", "")

            if player and status:
                injuries.append({
                    "player": player,
                    "team": team_name,
                    "status": status,
                    "detail": detail,
                    "position": position,
                })

    logging.info("ESPN injuries: %d entries across %d teams",
                 len(injuries), len(data.get("injuries", [])))
    return injuries


def _save_cache(injuries):
    cache = {"fetched_at": current_timestamp(), "injuries": injuries}
    try:
        with open(INJURY_CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


def _load_cache():
    if not os.path.exists(INJURY_CACHE_FILE):
        return None
    try:
        with open(INJURY_CACHE_FILE, "r") as f:
            cache = json.load(f)
        fetched = datetime.strptime(cache["fetched_at"], "%Y-%m-%d %H:%M:%S")
        age = (datetime.now() - fetched).total_seconds() / 3600
        if age > INJURY_CACHE_MAX_HOURS:
            return None
        return cache.get("injuries", [])
    except Exception:
        return None


def manual_set_injuries(team_name, out_players):
    """Manually mark players as out. Merges with existing cache."""
    injuries = _load_cache() or []
    for player in out_players:
        injuries.append({
            "player": player.strip(),
            "team": team_name,
            "status": "Out",
            "detail": "manual",
            "position": "",
        })
    _save_cache(injuries)
    return injuries


def get_team_injuries(team_name, injuries=None):
    """Get list of OUT/DOUBTFUL/IR players for an NFL team."""
    if injuries is None:
        injuries = fetch_injury_report()
    if not injuries:
        return []
    abbr = get_team_abbr(team_name)
    team_lower = team_name.lower()
    out = []
    for inj in injuries:
        inj_team = str(inj.get("team", "")).strip()
        status = str(inj.get("status", "")).strip()
        # NFL injury statuses that mean the player is unavailable
        if status not in ("Out", "Doubtful", "Injured Reserve",
                          "Physically Unable to Perform"):
            continue
        if (inj_team.upper() == abbr or
                inj_team.lower() == team_lower or
                team_lower in inj_team.lower() or
                inj_team.lower() in team_lower):
            out.append(inj["player"])
    return out


def calc_injury_impact(team_name, out_players):
    """Calculate Elo adjustment for missing NFL players.
    Returns negative value (penalty) for the injured team.

    NFL injury impact by position:
    - Star QB out: -50 Elo
    - Good QB out: -35 Elo
    - Star RB out: -15 Elo
    - Star WR/TE out: -12 Elo
    - Key defensive player out: -10 Elo
    - Kicker: -5 Elo
    """
    if not out_players:
        return 0.0

    # Try to get injury details with position info
    injuries = fetch_injury_report()
    injury_map = {}
    for inj in (injuries or []):
        injury_map[inj.get("player", "").lower()] = inj.get("position", "")

    total_impact = 0.0
    for player_name in out_players:
        position = injury_map.get(player_name.lower(), "")

        if position == "QB":
            # QBs have massive impact in NFL
            impact = -50.0
        elif position == "RB":
            impact = -15.0
        elif position in ("WR", "TE"):
            impact = -12.0
        elif position in ("CB", "S", "LB", "DE", "DT"):
            impact = -10.0
        elif position == "K" or position == "P":
            impact = -5.0
        elif position in ("OL", "OT", "OG", "C"):
            impact = -8.0
        else:
            # Unknown position, default moderate impact
            impact = -10.0

        total_impact += impact

    return total_impact


def show_injury_report(model=None):
    """Display current NFL injury report with impact estimates."""
    from color_helpers import cok, cwarn, cdim, chi, div, cred

    injuries = fetch_injury_report()
    if not injuries:
        print(cwarn("  No injury data available."))
        return

    by_team = {}
    for inj in injuries:
        team = inj.get("team", "Unknown")
        if team not in by_team:
            by_team[team] = []
        by_team[team].append(inj)

    div(80)
    print("  %s  (%d players listed)" % (chi("NFL INJURY REPORT"), len(injuries)))
    div(80)
    for team in sorted(by_team.keys()):
        players = by_team[team]
        out_names = [p["player"] for p in players
                     if p.get("status") in ("Out", "Doubtful", "Injured Reserve",
                                             "Physically Unable to Perform")]
        impact = calc_injury_impact(team, out_names) if out_names else 0
        impact_s = cred("%.0f Elo" % impact) if impact < -20 else cwarn("%.0f Elo" % impact) if impact < 0 else ""
        print("  %s %s" % (chi(team), impact_s))
        for p in players:
            status = p.get("status", "?")
            pos = p.get("position", "")
            pos_s = cdim(" (%s)" % pos) if pos else ""
            if status in ("Out", "Injured Reserve"):
                ss = cred(status)
            elif status in ("Doubtful", "Physically Unable to Perform"):
                ss = cwarn(status)
            elif status == "Questionable":
                ss = cdim(status)
            else:
                ss = cdim(status)
            print("    %-25s %s%s" % (p["player"], ss, pos_s))
    div(80)
