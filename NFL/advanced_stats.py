"""NFL Advanced Stats via nfl_data_py.

Pulls play-by-play EPA, CPOE, success rate, and officials data.
All data is 100% free from nflverse (pre-compiled CSVs, no API key).
"""

import os
import json
import logging
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd

ADVANCED_CACHE_FILE = "nfl_advanced_cache.json"
CACHE_MAX_AGE_HOURS = 12
EPA_STATS_FILE = "nfl_epa_stats.csv"
OFFICIALS_FILE = "nfl_officials.csv"


def _is_cache_stale(filepath, max_hours=CACHE_MAX_AGE_HOURS):
    try:
        from cache_utils import smart_cache_stale
        return smart_cache_stale(filepath, "nfl", "advanced", max_age_hours=max_hours)
    except ImportError:
        pass
    if not os.path.exists(filepath):
        return True
    age = datetime.now().timestamp() - os.path.getmtime(filepath)
    return age > max_hours * 3600


def download_pbp_data(seasons=None):
    """Download play-by-play data from nflverse.

    Returns DataFrame with EPA, CPOE, success, and game context columns.
    """
    import nfl_data_py as nfl

    if seasons is None:
        current_year = datetime.now().year
        month = datetime.now().month
        season_year = current_year if month >= 9 else current_year - 1
        seasons = [season_year - 1, season_year]

    logging.info("Downloading NFL play-by-play for seasons %s...", seasons)

    try:
        pbp = nfl.import_pbp_data(seasons, downcast=True, cache=True)
        logging.info("Downloaded %d plays across %d seasons", len(pbp), len(seasons))
        return pbp
    except Exception as e:
        logging.error("Failed to download PBP data: %s", e)
        return None


def compute_team_epa(pbp):
    """Compute rolling team EPA stats from play-by-play data.

    Returns DataFrame with per-team, per-game EPA metrics:
        team, season, week, game_id,
        off_epa_play, def_epa_play, pass_epa, rush_epa,
        cpoe, success_rate, turnover_rate,
        red_zone_success, third_down_rate,
        explosive_play_rate
    """
    if pbp is None or len(pbp) == 0:
        return pd.DataFrame()

    # Filter to real plays (exclude penalties, timeouts, etc.)
    plays = pbp[pbp["play_type"].isin(["pass", "run", "qb_kneel", "qb_spike"])].copy()

    results = []

    # Group by game
    for (game_id, season, week), game_plays in plays.groupby(
        ["game_id", "season", "week"], observed=True
    ):
        for is_offense in [True, False]:
            if is_offense:
                team_col = "posteam"
                epa_col = "epa"
            else:
                team_col = "defteam"
                epa_col = "epa"

            for team in game_plays[team_col].dropna().unique():
                if is_offense:
                    team_plays = game_plays[game_plays["posteam"] == team]
                else:
                    team_plays = game_plays[game_plays["defteam"] == team]
                    if len(team_plays) == 0:
                        continue

                n_plays = len(team_plays)
                if n_plays < 5:
                    continue

                if is_offense:
                    epa_per_play = team_plays["epa"].mean() if "epa" in team_plays.columns else 0
                    pass_plays = team_plays[team_plays["play_type"] == "pass"]
                    rush_plays = team_plays[team_plays["play_type"] == "run"]
                    pass_epa = pass_plays["epa"].mean() if len(pass_plays) > 0 else 0
                    rush_epa = rush_plays["epa"].mean() if len(rush_plays) > 0 else 0

                    # CPOE (completion percentage over expected)
                    cpoe = team_plays["cpoe"].dropna().mean() if "cpoe" in team_plays.columns else 0

                    # Success rate (play gained positive EPA)
                    success = (team_plays["epa"] > 0).mean() if "epa" in team_plays.columns else 0.5

                    # Turnover rate
                    turnovers = 0
                    if "interception" in team_plays.columns:
                        turnovers += team_plays["interception"].sum()
                    if "fumble_lost" in team_plays.columns:
                        turnovers += team_plays["fumble_lost"].sum()
                    turnover_rate = turnovers / n_plays

                    # Red zone (inside opponent 20)
                    rz = team_plays[team_plays.get("yardline_100", pd.Series(dtype=float)) <= 20] \
                        if "yardline_100" in team_plays.columns else pd.DataFrame()
                    rz_success = (rz["epa"] > 0).mean() if len(rz) > 3 else 0.5

                    # Third down
                    third = team_plays[team_plays.get("down", pd.Series(dtype=float)) == 3] \
                        if "down" in team_plays.columns else pd.DataFrame()
                    third_rate = (third.get("third_down_converted", pd.Series(dtype=float)) == 1).mean() \
                        if len(third) > 0 and "third_down_converted" in third.columns else 0.4

                    # Explosive plays (20+ yards)
                    explosive = (team_plays.get("yards_gained", pd.Series(dtype=float)) >= 20).mean() \
                        if "yards_gained" in team_plays.columns else 0.05

                    results.append({
                        "team": team, "season": season, "week": week,
                        "game_id": game_id, "side": "offense",
                        "epa_per_play": round(epa_per_play, 4),
                        "pass_epa": round(pass_epa, 4),
                        "rush_epa": round(rush_epa, 4),
                        "cpoe": round(cpoe, 2) if not np.isnan(cpoe) else 0,
                        "success_rate": round(success, 4),
                        "turnover_rate": round(turnover_rate, 4),
                        "red_zone_success": round(rz_success, 4),
                        "third_down_rate": round(third_rate, 4),
                        "explosive_play_rate": round(explosive, 4),
                        "n_plays": n_plays,
                    })
                else:
                    def_epa = team_plays["epa"].mean() if "epa" in team_plays.columns else 0
                    results.append({
                        "team": team, "season": season, "week": week,
                        "game_id": game_id, "side": "defense",
                        "epa_per_play": round(def_epa, 4),
                        "n_plays": n_plays,
                    })

    df = pd.DataFrame(results)
    return df


def compute_rolling_epa(epa_df, window=5):
    """Compute rolling EPA averages per team over last N games.

    Returns dict: team -> {off_epa, def_epa, pass_epa, rush_epa, cpoe, success_rate, ...}
    """
    if epa_df is None or len(epa_df) == 0:
        return {}

    team_stats = {}

    for team in epa_df["team"].unique():
        team_off = epa_df[(epa_df["team"] == team) & (epa_df["side"] == "offense")]
        team_def = epa_df[(epa_df["team"] == team) & (epa_df["side"] == "defense")]

        # Sort by season + week
        team_off = team_off.sort_values(["season", "week"]).tail(window)
        team_def = team_def.sort_values(["season", "week"]).tail(window)

        stats = {
            "off_epa": team_off["epa_per_play"].mean() if len(team_off) > 0 else 0,
            "def_epa": team_def["epa_per_play"].mean() if len(team_def) > 0 else 0,
            "pass_epa": team_off["pass_epa"].mean() if len(team_off) > 0 and "pass_epa" in team_off.columns else 0,
            "rush_epa": team_off["rush_epa"].mean() if len(team_off) > 0 and "rush_epa" in team_off.columns else 0,
            "cpoe": team_off["cpoe"].mean() if len(team_off) > 0 and "cpoe" in team_off.columns else 0,
            "success_rate": team_off["success_rate"].mean() if len(team_off) > 0 and "success_rate" in team_off.columns else 0.45,
            "turnover_rate": team_off["turnover_rate"].mean() if len(team_off) > 0 and "turnover_rate" in team_off.columns else 0.03,
            "explosive_rate": team_off["explosive_play_rate"].mean() if len(team_off) > 0 and "explosive_play_rate" in team_off.columns else 0.05,
            "games_in_window": len(team_off),
        }

        # Net EPA (offense - defense, higher is better)
        stats["net_epa"] = stats["off_epa"] - stats["def_epa"]

        team_stats[team] = {k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()}

    return team_stats


def download_officials_data(seasons=None):
    """Download NFL officials/referee data from nflverse.

    Returns DataFrame with game_id, referee name, and other officials.
    """
    import nfl_data_py as nfl

    if seasons is None:
        current_year = datetime.now().year
        month = datetime.now().month
        season_year = current_year if month >= 9 else current_year - 1
        seasons = [season_year - 1, season_year]

    try:
        officials = nfl.import_officials(seasons)
        logging.info("Downloaded officials data: %d records", len(officials))
        return officials
    except Exception as e:
        logging.error("Failed to download officials data: %s", e)
        return None


def compute_referee_tendencies(officials_df, pbp_df):
    """Compute referee tendencies from historical game data.

    Returns dict: referee_name -> {
        games, home_win_rate, avg_total_score,
        avg_penalties, home_penalty_bias
    }
    """
    if officials_df is None or pbp_df is None:
        return {}

    # Get unique games with referees
    if "game_id" not in officials_df.columns:
        return {}

    # Filter to head referee only
    ref_col = None
    for col in ["referee", "official_name", "name"]:
        if col in officials_df.columns:
            ref_col = col
            break

    if ref_col is None:
        logging.warning("No referee column found in officials data")
        return {}

    # Get game outcomes from PBP
    game_results = {}
    for game_id, game in pbp_df.groupby("game_id", observed=True):
        if "home_team" in game.columns and "home_score" in game.columns:
            home = game["home_team"].iloc[0]
            away = game["away_team"].iloc[0] if "away_team" in game.columns else ""
            home_score = game["home_score"].max() if "home_score" in game.columns else 0
            away_score = game["away_score"].max() if "away_score" in game.columns else 0
            game_results[game_id] = {
                "home": home, "away": away,
                "home_score": home_score, "away_score": away_score,
                "home_win": 1 if home_score > away_score else 0,
                "total": home_score + away_score,
            }

    # Compute per-referee stats
    ref_stats = defaultdict(lambda: {"games": 0, "home_wins": 0, "total_scores": []})

    for _, row in officials_df.iterrows():
        gid = row.get("game_id", "")
        ref = row.get(ref_col, "")
        if not ref or gid not in game_results:
            continue

        gr = game_results[gid]
        ref_stats[ref]["games"] += 1
        ref_stats[ref]["home_wins"] += gr["home_win"]
        ref_stats[ref]["total_scores"].append(gr["total"])

    result = {}
    for ref, stats in ref_stats.items():
        if stats["games"] < 5:
            continue
        result[ref] = {
            "games": stats["games"],
            "home_win_rate": round(stats["home_wins"] / stats["games"], 4),
            "avg_total_score": round(np.mean(stats["total_scores"]), 1),
            "total_score_std": round(np.std(stats["total_scores"]), 1),
        }

    return result


def get_advanced_features(home_team, away_team, team_epa_stats):
    """Get advanced stat features for a matchup.

    Returns dict of features suitable for the meta-learner.
    """
    home_stats = team_epa_stats.get(home_team, {})
    away_stats = team_epa_stats.get(away_team, {})

    if not home_stats or not away_stats:
        return {}

    features = {
        # EPA differentials
        "home_off_epa": home_stats.get("off_epa", 0),
        "away_off_epa": away_stats.get("off_epa", 0),
        "home_def_epa": home_stats.get("def_epa", 0),
        "away_def_epa": away_stats.get("def_epa", 0),
        "net_epa_diff": home_stats.get("net_epa", 0) - away_stats.get("net_epa", 0),

        # Passing
        "home_pass_epa": home_stats.get("pass_epa", 0),
        "away_pass_epa": away_stats.get("pass_epa", 0),
        "cpoe_diff": home_stats.get("cpoe", 0) - away_stats.get("cpoe", 0),

        # Rushing
        "home_rush_epa": home_stats.get("rush_epa", 0),
        "away_rush_epa": away_stats.get("rush_epa", 0),

        # Efficiency
        "success_rate_diff": home_stats.get("success_rate", 0.45) - away_stats.get("success_rate", 0.45),
        "turnover_rate_diff": away_stats.get("turnover_rate", 0.03) - home_stats.get("turnover_rate", 0.03),  # Inverted: opponent turnovers help
        "explosive_diff": home_stats.get("explosive_rate", 0.05) - away_stats.get("explosive_rate", 0.05),
    }

    return features


def load_or_download_epa(force=False):
    """Load cached EPA stats or download fresh data.

    Returns (epa_df, rolling_stats_dict).
    """
    epa_file = EPA_STATS_FILE

    if not force and not _is_cache_stale(epa_file):
        try:
            epa_df = pd.read_csv(epa_file)
            rolling = compute_rolling_epa(epa_df)
            logging.info("Loaded cached EPA stats (%d records)", len(epa_df))
            return epa_df, rolling
        except Exception:
            pass

    # Download fresh
    pbp = download_pbp_data()
    if pbp is None:
        return pd.DataFrame(), {}

    epa_df = compute_team_epa(pbp)
    if len(epa_df) > 0:
        epa_df.to_csv(epa_file, index=False)
        logging.info("Saved EPA stats (%d records) to %s", len(epa_df), epa_file)

    rolling = compute_rolling_epa(epa_df)
    return epa_df, rolling


def show_epa_rankings(team_stats):
    """Display team EPA rankings table."""
    if not team_stats:
        print("  No EPA data available. Run 'advstats' to download.")
        return

    # Sort by net EPA
    sorted_teams = sorted(team_stats.items(), key=lambda x: x[1].get("net_epa", 0), reverse=True)

    print(f"\n  {'Rank':>4} {'Team':<25} {'Off EPA':>8} {'Def EPA':>8} {'Net EPA':>8} {'CPOE':>6} {'Succ%':>6}")
    print("  " + "-" * 70)

    for i, (team, stats) in enumerate(sorted_teams, 1):
        off = stats.get("off_epa", 0)
        defe = stats.get("def_epa", 0)
        net = stats.get("net_epa", 0)
        cpoe = stats.get("cpoe", 0)
        succ = stats.get("success_rate", 0)

        print(f"  {i:>4} {team:<25} {off:>+8.3f} {defe:>+8.3f} {net:>+8.3f} {cpoe:>+6.1f} {succ:>5.1%}")
