"""Player stats download, loading, scoring, and display via ESPN API."""

import os
import time
import logging

import numpy as np
import pandas as pd
import requests

from config import (
    PLAYER_STATS_FILE, ADVANCED_STATS_FILE, TEAM_ABBR,
    get_season_label, get_team_abbr, is_cache_stale,
)
from color_helpers import cok, cwarn, cdim, cbold, chi, cyel, cblu, cgrn

ESPN_STATS_URL = ("https://site.web.api.espn.com/apis/common/v3/sports/hockey/nhl"
                  "/statistics/byathlete")


def download_player_stats(csv_file=PLAYER_STATS_FILE):
    if not is_cache_stale(csv_file):
        logging.info("Using cached player stats %s", csv_file)
        return csv_file
    logging.info("Downloading NHL player stats via ESPN stats API...")
    try:
        from config import get_current_season_year
        season = get_current_season_year()
        resp = requests.get(ESPN_STATS_URL,
                            params={"season": season, "seasontype": 2,
                                    "limit": 200, "isQualified": "true"},
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
                totals = cat.get("totals", [])
                values = cat.get("values", [])
                for i, label in enumerate(labels):
                    if i < len(values):
                        try:
                            player[label] = float(values[i])
                        except (ValueError, TypeError):
                            pass
            rows.append(player)

        if not rows:
            logging.warning("No player data from ESPN")
            return csv_file if os.path.exists(csv_file) else None

        df = pd.DataFrame(rows)
        # Map to our standard column names
        col_map = {"PTS": "PTS", "G": "G", "A": "A", "+/-": "PM",
                   "GP": "GP", "GAA": "GAA", "SV%": "SVP", "W": "W", "SO": "SO"}
        for espn_col, our_col in col_map.items():
            if espn_col in df.columns and espn_col != our_col:
                df[our_col] = df[espn_col]
        for c in ["PTS", "G", "A", "PM", "GP", "GAA", "SVP", "W", "SO"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
            else:
                df[c] = 0.0

        df.to_csv(csv_file, index=False)
        logging.info("Downloaded %d players -> %s", len(df), csv_file)
        return csv_file
    except Exception as e:
        logging.warning("ESPN player stats failed (%s).", e)
        return csv_file if os.path.exists(csv_file) else None


def download_advanced_stats(csv_file=ADVANCED_STATS_FILE):
    """Placeholder for advanced stats - ESPN leaders gives us what we need."""
    if not is_cache_stale(csv_file):
        logging.info("Using cached advanced stats %s", csv_file)
        return csv_file
    # NHL advanced stats are handled by the main player download
    return csv_file if os.path.exists(csv_file) else None


def load_player_stats(filename=PLAYER_STATS_FILE):
    if not os.path.exists(filename):
        return pd.DataFrame()
    df = pd.read_csv(filename)
    for c in ["PTS", "G", "A", "PM", "GP", "GAA", "SVP", "W", "SO"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_advanced_stats(filename=ADVANCED_STATS_FILE):
    if not os.path.exists(filename):
        return pd.DataFrame()
    df = pd.read_csv(filename)
    return df


def calc_team_player_score(player_df, team_full_name, top_n=10):
    """Calculate composite team score from skater + goaltender stats.

    Blend: skater_score * 0.55 + goalie_score * 0.45
    """
    if player_df is None or player_df.empty:
        return 0.0
    abbr = get_team_abbr(team_full_name)
    team_df = player_df[player_df["Tm"] == abbr].copy()
    if team_df.empty:
        return 0.0
    for c in ["PTS", "G", "A", "PM", "GP", "GAA", "SVP", "W", "SO"]:
        if c not in team_df.columns:
            team_df[c] = 0.0
        team_df[c] = pd.to_numeric(team_df[c], errors="coerce").fillna(0.0)

    # Identify goaltenders (have GAA > 0 or SVP > 0) vs skaters
    goalies = team_df[(team_df["GAA"] > 0) | (team_df["SVP"] > 0)].copy()
    skaters = team_df[~team_df.index.isin(goalies.index)].copy()

    # Skater score: top 6 forwards + top 4 defensemen by points
    top_skaters = skaters.nlargest(min(top_n, len(skaters)), "PTS") if not skaters.empty else pd.DataFrame()
    skater_score = 0.0
    if not top_skaters.empty:
        skater_score = (
            top_skaters["PTS"] * 1.0 + top_skaters["G"] * 0.5 + top_skaters["A"] * 0.3
        ).sum()

    # Goalie score: starting goalie (most wins)
    goalie_score = 0.0
    if not goalies.empty:
        starter = goalies.nlargest(1, "W")
        for _, g in starter.iterrows():
            svp = float(g.get("SVP", 0))
            gaa = float(g.get("GAA", 3.0))
            # Normalize: better save% and lower GAA = higher score
            goalie_score = (svp - 0.900) * 500 + (3.0 - gaa) * 10 + float(g.get("SO", 0)) * 5

    return float(skater_score * 0.55 + goalie_score * 0.45)


def build_league_player_scores(player_df):
    if player_df is None or player_df.empty:
        return {}
    teams  = list(TEAM_ABBR.keys())
    raw    = {t: calc_team_player_score(player_df, t) for t in teams}
    values = [v for v in raw.values() if v != 0.0]
    if not values:
        return {}
    mean = float(np.mean(values))
    std  = float(np.std(values)) or 1.0
    return {t: (v - mean) / std for t, v in raw.items()}


def show_player_metrics(player_df, team=None, top_n=5):
    if player_df.empty:
        print("   " + cwarn("No player stats loaded."))
        return
    if team:
        abbr   = get_team_abbr(team)
        df_sub = player_df[player_df["Tm"] == abbr].copy()
        if df_sub.empty:
            print("   " + cwarn("No stats found for %s (%s)" % (team, abbr)))
            return
        # Show skaters sorted by points
        skaters = df_sub[(df_sub["GAA"] == 0) & (df_sub["SVP"] == 0)].copy()
        if skaters.empty:
            skaters = df_sub.copy()
        df = skaters.sort_values("PTS", ascending=False).head(top_n)
        label = "Top %d for %s (%s)" % (top_n, team, abbr)
    else:
        df = player_df.sort_values("PTS", ascending=False).head(top_n)
        label = "LEAGUE TOP %d SCORERS" % top_n
    print("\n   %s:" % chi(label))
    for i, (_, row) in enumerate(df.iterrows(), 1):
        pts_s = cok("%5.0f" % float(row.get("PTS", 0) or 0))
        g_s   = cyel("%4.0f" % float(row.get("G", 0) or 0))
        a_s   = cblu("%4.0f" % float(row.get("A", 0) or 0))
        pm_s  = cgrn("%+4.0f" % float(row.get("PM", 0) or 0))
        sep   = cdim("|")
        name_s = cbold("%-22s" % str(row["Player"])[:22])
        print("     %s %s %s PTS:%s %s G:%s %s A:%s %s +/-:%s"
              % (cdim("%d." % i), name_s, sep, pts_s, sep, g_s, sep, a_s, sep, pm_s))
