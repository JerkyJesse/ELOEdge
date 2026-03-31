"""NFL player stats download, loading, scoring, and display via ESPN API."""

import os
import time
import logging

import numpy as np
import pandas as pd
import requests

from config import (
    PLAYER_STATS_FILE, ADVANCED_STATS_FILE, TEAM_ABBR,
    get_current_season_year, get_team_abbr, is_cache_stale,
)
from color_helpers import cok, cwarn, cdim, cbold, chi, cyel, cblu, cgrn


ESPN_STATS_URL = ("https://site.web.api.espn.com/apis/common/v3/sports/football/nfl"
                  "/statistics/byathlete")


def download_player_stats(csv_file=PLAYER_STATS_FILE):
    """Download NFL player stats from ESPN byathlete API."""
    if not is_cache_stale(csv_file):
        logging.info("Using cached player stats %s", csv_file)
        return csv_file
    logging.info("Downloading NFL player stats via ESPN stats API...")
    try:
        season = get_current_season_year()
        resp = requests.get(ESPN_STATS_URL,
                            params={"season": season, "seasontype": 2, "limit": 200},
                            timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Build label maps from top-level categories
        label_map = {}  # {cat_name: [label1, label2, ...]}
        for cat in data.get("categories", []):
            label_map[cat.get("name", "")] = cat.get("labels", [])

        rows = []
        for entry in data.get("athletes", []):
            ath = entry.get("athlete", {})
            player_name = ath.get("displayName", "")
            team_abbr = ath.get("teamShortName", "")
            pos = ath.get("position", {}).get("abbreviation", "")
            if not player_name:
                continue

            player = {"Player": player_name, "Tm": team_abbr, "Pos": pos}
            for cat in entry.get("categories", []):
                cat_name = cat.get("name", "")
                labels = label_map.get(cat_name, [])
                values = cat.get("values", [])
                # Prefix labels with category to avoid collisions (e.g. rushing YDS vs receiving YDS)
                prefix = ""
                if cat_name in ("rushing", "receiving", "passing"):
                    prefix = cat_name[:4].upper() + "_"
                for i, label in enumerate(labels):
                    if i < len(values) and values[i] is not None:
                        try:
                            player[prefix + label] = float(values[i])
                        except (ValueError, TypeError):
                            pass
            rows.append(player)
        time.sleep(0.5)

        if not rows:
            raise ValueError("No player data from ESPN")

        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)
        logging.info("Downloaded %d player entries -> %s", len(df), csv_file)
        return csv_file
    except Exception as e:
        logging.warning("ESPN player stats failed (%s).", e)
        return csv_file if os.path.exists(csv_file) else None


def download_advanced_stats(csv_file=ADVANCED_STATS_FILE):
    """Placeholder for advanced stats -- NFL doesn't have a clean public advanced stats API."""
    if not is_cache_stale(csv_file):
        logging.info("Using cached advanced stats %s", csv_file)
        return csv_file
    # For NFL, advanced stats are limited from public APIs
    # Create a minimal placeholder
    try:
        df = pd.DataFrame(columns=["Player", "Tm", "QBR", "EPA"])
        df.to_csv(csv_file, index=False)
        return csv_file
    except Exception:
        return csv_file if os.path.exists(csv_file) else None


def load_player_stats(filename=PLAYER_STATS_FILE):
    if not os.path.exists(filename):
        return pd.DataFrame()
    try:
        df = pd.read_csv(filename)
        return df
    except Exception:
        return pd.DataFrame()


def load_advanced_stats(filename=ADVANCED_STATS_FILE):
    if not os.path.exists(filename):
        return pd.DataFrame()
    try:
        df = pd.read_csv(filename)
        return df
    except Exception:
        return pd.DataFrame()


# ESPN team abbreviation mapping (ESPN abbr -> our config abbr)
_ESPN_ABBR_MAP = {
    "WSH": "WAS", "JAX": "JAX", "LAR": "LAR",
}


def _normalize_team_abbr(espn_abbr):
    """Map ESPN abbreviation to our TEAM_ABBR values."""
    mapped = _ESPN_ABBR_MAP.get(espn_abbr, espn_abbr)
    # Check if it's one of our known abbreviations
    all_abbrs = set(TEAM_ABBR.values())
    if mapped in all_abbrs:
        return mapped
    return espn_abbr


def calc_team_player_score(player_df, team_full_name, top_n=10):
    """Calculate a composite player strength score for an NFL team.

    Passing: YDS * 0.04 + TD * 4
    Rushing: YDS * 0.1 + TD * 6
    Receiving: YDS * 0.1 + TD * 6
    Blend: passing 50% / rushing+receiving 50%
    """
    if player_df is None or player_df.empty:
        return 0.0
    abbr = get_team_abbr(team_full_name)
    team_df = player_df[player_df["Tm"].apply(_normalize_team_abbr) == abbr].copy()
    if team_df.empty:
        team_df = player_df[player_df["Tm"] == abbr].copy()
    if team_df.empty:
        return 0.0

    for c in team_df.columns:
        if c not in ("Player", "Tm", "Pos"):
            team_df[c] = pd.to_numeric(team_df[c], errors="coerce").fillna(0.0)

    pass_score = 0.0
    rush_rec_score = 0.0
    for _, row in team_df.iterrows():
        pass_score += row.get("PASS_YDS", 0) * 0.04 + row.get("PASS_TD", 0) * 4
        rush_rec_score += (row.get("RUSH_YDS", 0) * 0.1 + row.get("RUSH_TD", 0) * 6
                           + row.get("RECE_YDS", 0) * 0.1 + row.get("RECE_TD", 0) * 6)

    return float(pass_score * 0.5 + rush_rec_score * 0.5)


def build_league_player_scores(player_df):
    """Build z-score normalized player strength scores for all teams."""
    if player_df is None or player_df.empty:
        return {}
    teams = list(TEAM_ABBR.keys())
    raw = {t: calc_team_player_score(player_df, t) for t in teams}
    values = [v for v in raw.values() if v != 0.0]
    if not values:
        return {}
    mean = float(np.mean(values))
    std = float(np.std(values)) or 1.0
    return {t: (v - mean) / std for t, v in raw.items()}


def show_player_metrics(player_df, team=None, top_n=5):
    """Display top NFL players (passing, rushing, receiving leaders)."""
    if player_df.empty:
        print("   " + cwarn("No player stats loaded."))
        return
    if team:
        abbr = get_team_abbr(team)
        df_sub = player_df[player_df["Tm"].apply(_normalize_team_abbr) == abbr].copy()
        if df_sub.empty:
            df_sub = player_df[player_df["Tm"] == abbr].copy()
        if df_sub.empty:
            print("   " + cwarn("No stats found for %s (%s)" % (team, abbr)))
            return
        df = df_sub.head(top_n)
        label = "Top %d for %s (%s)" % (top_n, team, abbr)
    else:
        # Sort by a composite: passing yards + rushing yards + receiving yards
        player_df = player_df.copy()
        for c in ["PASS_YDS", "RUSH_YDS", "RECE_YDS"]:
            if c not in player_df.columns:
                player_df[c] = 0.0
            player_df[c] = pd.to_numeric(player_df[c], errors="coerce").fillna(0.0)
        player_df["_total"] = player_df["PASS_YDS"] + player_df["RUSH_YDS"] + player_df["RECE_YDS"]
        df = player_df.sort_values("_total", ascending=False).head(top_n)
        label = "NFL LEADERS (top %d)" % top_n

    print("\n   %s:" % chi(label))
    sep = cdim("|")
    for i, (_, row) in enumerate(df.iterrows(), 1):
        name_s = cbold("%-22s" % str(row.get("Player", "-"))[:22])
        team_s = cdim("%-4s" % str(row.get("Tm", "?")))
        pos_s = cdim("%-3s" % str(row.get("Pos", "?")))
        parts = []
        # Show position-relevant stats
        pass_yds = float(row.get("PASS_YDS", 0) or 0)
        pass_td = float(row.get("PASS_TD", 0) or 0)
        rush_yds = float(row.get("RUSH_YDS", 0) or 0)
        rush_td = float(row.get("RUSH_TD", 0) or 0)
        rec_yds = float(row.get("RECE_YDS", 0) or 0)
        rec_td = float(row.get("RECE_TD", 0) or 0)
        if pass_yds > 0:
            parts.append("PaYD:%s" % cok("%.0f" % pass_yds))
            parts.append("PaTD:%s" % cok("%.0f" % pass_td))
        if rush_yds > 0:
            parts.append("RuYD:%s" % cok("%.0f" % rush_yds))
        if rec_yds > 0:
            parts.append("ReYD:%s" % cok("%.0f" % rec_yds))
        if rush_td + rec_td > 0:
            parts.append("TD:%s" % cok("%.0f" % (rush_td + rec_td)))
        stats_str = (" %s " % sep).join(parts) if parts else cdim("no stats")
        print("     %s %s %s %s %s %s" % (cdim("%d." % i), name_s, team_s, pos_s, sep, stats_str))
