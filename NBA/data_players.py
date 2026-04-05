"""Player stats download, loading, scoring, and display."""

import os
import time
import logging

import numpy as np
import pandas as pd

from config import (
    PLAYER_STATS_FILE, ADVANCED_STATS_FILE, TEAM_ABBR,
    get_season_label, get_team_abbr, is_cache_stale,
)
from color_helpers import cok, cwarn, cdim, cbold, chi, cyel, cblu, cgrn


def download_player_stats(csv_file=PLAYER_STATS_FILE):
    if not is_cache_stale(csv_file, data_type="players"):
        logging.info("Using cached player stats %s", csv_file)
        return csv_file
    logging.info("Downloading NBA player per-game stats via nba_api...")
    try:
        from nba_api.stats.endpoints import leaguedashplayerstats
        time.sleep(1)
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=get_season_label(), season_type_all_star="Regular Season",
            per_mode_detailed="PerGame",
        )
        df = stats.get_data_frames()[0]
        df = df.rename(columns={
            "PLAYER_NAME":"Player","TEAM_ABBREVIATION":"Tm","GP":"G","MIN":"MP",
            "FG_PCT":"FG%","FG3_PCT":"3P%","FT_PCT":"FT%","REB":"TRB",
            "AST":"AST","STL":"STL","BLK":"BLK","TOV":"TOV","PTS":"PTS",
        })
        keep = ["Player","Tm","G","MP","FG%","3P%","FT%","TRB","AST","STL","BLK","TOV","PTS"]
        df = df[[c for c in keep if c in df.columns]]
        for c in ["G","MP","FG%","3P%","FT%","TRB","AST","STL","BLK","TOV","PTS"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["Player","Tm"])
        if len(df) < 150:
            raise ValueError("Player table too small: %d" % len(df))
        df.to_csv(csv_file, index=False)
        logging.info("Downloaded %d players -> %s", len(df), csv_file)
        return csv_file
    except Exception as e:
        logging.warning("nba_api player stats failed (%s).", e)
        return csv_file if os.path.exists(csv_file) else None


def download_advanced_stats(csv_file=ADVANCED_STATS_FILE):
    if not is_cache_stale(csv_file, data_type="players"):
        logging.info("Using cached advanced stats %s", csv_file)
        return csv_file
    logging.info("Downloading NBA advanced player stats via nba_api...")
    try:
        from nba_api.stats.endpoints import leaguedashplayerstats
        time.sleep(1)
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=get_season_label(), season_type_all_star="Regular Season",
            per_mode_detailed="PerGame", measure_type_detailed_defense="Advanced",
        )
        df = stats.get_data_frames()[0]
        df = df.rename(columns={
            "PLAYER_NAME":"Player","TEAM_ABBREVIATION":"Tm","GP":"G","MIN":"MP",
        })
        keep = ["Player","Tm","G","MP","NET_RATING","OFF_RATING","DEF_RATING","PIE","USG_PCT","TS_PCT"]
        df = df[[c for c in keep if c in df.columns]]
        for c in ["G","MP","NET_RATING","OFF_RATING","DEF_RATING","PIE","USG_PCT","TS_PCT"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["Player","Tm"])
        df.to_csv(csv_file, index=False)
        logging.info("Downloaded %d advanced rows -> %s", len(df), csv_file)
        return csv_file
    except Exception as e:
        logging.warning("Advanced stats download failed (%s).", e)
        return csv_file if os.path.exists(csv_file) else None


def load_player_stats(filename=PLAYER_STATS_FILE):
    if not os.path.exists(filename):
        return pd.DataFrame()
    df = pd.read_csv(filename)
    for c in ["PTS","AST","TRB","MP"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_advanced_stats(filename=ADVANCED_STATS_FILE):
    if not os.path.exists(filename):
        return pd.DataFrame()
    df = pd.read_csv(filename)
    for c in ["NET_RATING","OFF_RATING","DEF_RATING","PIE","USG_PCT","TS_PCT","MP"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def calc_team_player_score(player_df, team_full_name, top_n=10):
    if player_df is None or player_df.empty:
        return 0.0
    abbr    = get_team_abbr(team_full_name)
    team_df = player_df[player_df["Tm"] == abbr].copy()
    if team_df.empty:
        return 0.0
    for c in ["PTS","AST","TRB","STL","BLK","TOV","MP"]:
        if c not in team_df.columns:
            team_df[c] = 0.0
        team_df[c] = pd.to_numeric(team_df[c], errors="coerce").fillna(0.0)
    top = team_df.nlargest(top_n, "PTS")
    box_score = (
        top["PTS"]*1.0 + top["AST"]*0.75 + top["TRB"]*0.5
        + top["STL"]*1.0 + top["BLK"]*0.75 - top["TOV"]*0.75
    ).sum()
    adv_df = load_advanced_stats()
    if adv_df.empty:
        return float(box_score)
    team_adv = adv_df[adv_df["Tm"] == abbr].copy()
    if team_adv.empty:
        return float(box_score)
    merged = top.merge(
        team_adv[["Player","NET_RATING","PIE","USG_PCT","TS_PCT"]], on="Player", how="left",
    )
    for c in ["NET_RATING","PIE","USG_PCT","TS_PCT"]:
        merged[c] = pd.to_numeric(merged.get(c), errors="coerce").fillna(0.0)
    adv_score = (
        merged["NET_RATING"]*0.5 + merged["PIE"]*20.0
        + (merged["USG_PCT"]*merged["TS_PCT"])*10.0
    ).sum()
    return float(box_score*0.55 + adv_score*0.45)


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
    adv_df = load_advanced_stats()
    if team:
        abbr   = get_team_abbr(team)
        df_sub = player_df[player_df["Tm"] == abbr].copy()
        if df_sub.empty:
            print("   " + cwarn("No stats found for %s (%s)" % (team, abbr)))
            return
        df    = df_sub.sort_values("PTS", ascending=False).head(top_n)
        label = "Top %d for %s (%s)" % (top_n, team, abbr)
    else:
        df    = player_df.sort_values("PTS", ascending=False).head(top_n)
        label = "LEAGUE TOP %d SCORERS" % top_n
    print("\n   %s:" % chi(label))
    for i, (_, row) in enumerate(df.iterrows(), 1):
        net_str = ""
        pie_str = ""
        if not adv_df.empty:
            adv_row = adv_df[adv_df["Player"] == row["Player"]]
            if not adv_row.empty:
                net = adv_row.iloc[0].get("NET_RATING", float("nan"))
                pie = adv_row.iloc[0].get("PIE",        float("nan"))
                if not pd.isna(net):
                    net_str = " %s NET:%s" % (cdim("|"), cok("%+.1f" % net))
                if not pd.isna(pie):
                    pie_str = " %s PIE:%s" % (cdim("|"), cgrn("%.3f" % pie))
        pts_s  = cok("%5.1f" % float(row.get("PTS", 0) or 0))
        ast_s  = cyel("%4.1f" % float(row.get("AST", 0) or 0))
        reb_s  = cblu("%4.1f" % float(row.get("TRB", 0) or 0))
        mp_s   = cdim("%5.1f" % float(row.get("MP",  0) or 0))
        sep    = cdim("|")
        name_s = cbold("%-22s" % str(row["Player"])[:22])
        print("     %s %s %s PTS:%s %s AST:%s %s REB:%s %s MP:%s%s%s"
              % (cdim("%d." % i), name_s, sep, pts_s, sep, ast_s, sep, reb_s, sep, mp_s,
                 net_str, pie_str))
