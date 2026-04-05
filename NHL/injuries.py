"""NHL injury report via ESPN public API + impact scoring."""

import os
import json
import logging
from datetime import datetime

import requests
import pandas as pd

from config import current_timestamp, get_team_abbr, TEAM_ABBR
from data_players import load_player_stats

INJURY_CACHE_FILE = "nhl_injuries.json"
INJURY_CACHE_MAX_HOURS = 4
ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/injuries"

# ESPN team names that differ from ours
ESPN_TEAM_MAP = {
}


def fetch_injury_report():
    """Fetch NHL injury report from ESPN public API."""
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
        espn_team = team_block.get("displayName", "")
        team_name = ESPN_TEAM_MAP.get(espn_team, espn_team)

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
    except Exception as e:
        logging.warning("Injury cache save failed: %s", e)


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
    except Exception as e:
        logging.debug("Injury cache load failed: %s", e)
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
    """Get list of OUT/Day-to-Day/IR/LTIR players for a team."""
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
        if status not in ("Out", "Day-to-Day", "Injured Reserve", "Long-Term Injured Reserve"):
            continue
        if (inj_team.upper() == abbr or
                inj_team.lower() == team_lower or
                team_lower in inj_team.lower() or
                inj_team.lower() in team_lower):
            out.append(inj["player"])
    return out


def calc_injury_impact(team_name, out_players):
    """Calculate Elo adjustment for missing players.
    NHL-specific: goaltender injuries have massive impact.

    Impact scale:
    - Star goaltender out: -35 Elo
    - Star forward (30+ goals): -20 Elo
    - Good forward (20+ goals): -12 Elo
    - Star defenseman (Norris candidate): -15 Elo
    - Regular player: -5 Elo
    """
    if not out_players:
        return 0.0

    player_df = load_player_stats()
    if player_df.empty:
        return -10.0 * len(out_players)

    abbr = get_team_abbr(team_name)
    team_players = player_df[player_df["Tm"] == abbr].copy()
    if team_players.empty:
        return -10.0 * len(out_players)

    total_impact = 0.0
    for player_name in out_players:
        matches = team_players[team_players["Player"].str.contains(
            player_name, case=False, na=False)]
        if matches.empty:
            for part in player_name.split():
                if len(part) > 3:
                    matches = team_players[team_players["Player"].str.contains(
                        part, case=False, na=False)]
                    if not matches.empty:
                        break
        if matches.empty:
            total_impact -= 8.0
            continue

        row = matches.iloc[0]
        goals = float(row.get("G", 0) or 0)
        pts = float(row.get("PTS", 0) or 0)
        gaa = float(row.get("GAA", 0) or 0)
        svp = float(row.get("SVP", 0) or 0)
        wins = float(row.get("W", 0) or 0)

        # Goaltender detection: has GAA or SVP stats
        is_goalie = gaa > 0 or svp > 0

        if is_goalie:
            # Goalie is ~50% of the team in hockey
            if wins >= 20 or svp >= 0.910:
                impact = -35.0  # Star goaltender
            elif wins >= 10:
                impact = -25.0  # Starting goaltender
            else:
                impact = -15.0  # Backup
        else:
            # Skater impact
            if goals >= 30:
                impact = -20.0  # Star forward
            elif goals >= 20:
                impact = -12.0  # Good forward
            elif pts >= 40:
                impact = -15.0  # Star defenseman / playmaker
            elif pts >= 20:
                impact = -8.0
            else:
                impact = -5.0

        total_impact += impact

    return total_impact


def show_injury_report(model=None):
    """Display current injury report with impact estimates."""
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

    div(70)
    print("  %s  (%d players listed)" % (chi("NHL INJURY REPORT"), len(injuries)))
    div(70)
    for team in sorted(by_team.keys()):
        players = by_team[team]
        out_names = [p["player"] for p in players
                     if p.get("status") in ("Out", "Day-to-Day", "Injured Reserve", "Long-Term Injured Reserve")]
        impact = calc_injury_impact(team, out_names) if out_names else 0
        impact_s = cred("%.0f Elo" % impact) if impact < -20 else cwarn("%.0f Elo" % impact) if impact < 0 else ""
        print("  %s %s" % (chi(team), impact_s))
        for p in players:
            status = p.get("status", "?")
            pos = p.get("position", "")
            pos_s = " (%s)" % pos if pos else ""
            if status == "Out":
                ss = cred(status)
            elif status in ("Injured Reserve", "Long-Term Injured Reserve"):
                ss = cred(status)
            elif status == "Day-to-Day":
                ss = cwarn(status)
            else:
                ss = cdim(status)
            print("    %-25s %s%s" % (p["player"], ss, cdim(pos_s)))
    div(70)
