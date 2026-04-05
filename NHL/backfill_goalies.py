"""Backfill starting goalie data into nhl_recent_games.csv using NHL team schedule + boxscore API."""
import time
import logging
import requests
import pandas as pd
from config import GAMES_FILE, TEAM_ABBR, NHL_API_ABBR

logging.basicConfig(level=logging.INFO, format="%(message)s")

NHL_API = "https://api-web.nhle.com/v1"

_FULL_TO_ABBR = {full: abbr for full, abbr in TEAM_ABBR.items()}

# Reverse map: NHL API abbr -> our abbr
_API_TO_OURS = {v: k for k, v in NHL_API_ABBR.items()}
# Our abbr -> NHL API abbr (for schedule lookups)
_OURS_TO_API = dict(NHL_API_ABBR)


def _get_nhl_seasons(start_year, end_year):
    """Return season codes like '20232024', '20242025'."""
    seasons = set()
    for y in range(start_year - 1, end_year + 1):
        seasons.add("%d%d" % (y, y + 1))
    return sorted(seasons)


def _build_game_id_map(seasons, team_abbrs):
    """Fetch team schedules and build game_id lookup.
    Returns dict: (date_str, home_abbr, away_abbr) -> game_id"""
    game_map = {}
    seen_ids = set()
    for season in seasons:
        logging.info("  Fetching schedules for season %s...", season)
        for abbr in team_abbrs:
            try:
                url = "%s/club-schedule-season/%s/%s" % (NHL_API, abbr, season)
                resp = requests.get(url, timeout=15)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                for g in data.get("games", []):
                    gid = g.get("id")
                    if not gid or gid in seen_ids:
                        continue
                    state = g.get("gameState", "")
                    if state not in ("OFF", "FINAL"):
                        continue
                    seen_ids.add(gid)
                    gdate = str(g.get("gameDate", ""))[:10]
                    h = g.get("homeTeam", {}).get("abbrev", "")
                    a = g.get("awayTeam", {}).get("abbrev", "")
                    if gdate and h and a:
                        game_map[(gdate, h, a)] = gid
                time.sleep(0.1)
            except Exception as e:
                logging.debug("Schedule fetch failed for %s/%s: %s", abbr, season, e)
    logging.info("  Built game ID map: %d games", len(game_map))
    return game_map


def _get_starting_goalies(game_id):
    """Get starting goalies from boxscore (goalie with most TOI)."""
    try:
        url = "%s/gamecenter/%s/boxscore" % (NHL_API, game_id)
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return "", ""
        box = resp.json()
        pbgs = box.get("playerByGameStats", {})
        result = {}
        for side in ("homeTeam", "awayTeam"):
            goalies = pbgs.get(side, {}).get("goalies", [])
            best_toi = ""
            best_name = ""
            for g in goalies:
                toi = g.get("toi", "00:00")
                name = g.get("name", {}).get("default", "")
                if toi > best_toi and name:
                    best_toi = toi
                    best_name = name
            result[side] = best_name
        return result.get("homeTeam", ""), result.get("awayTeam", "")
    except Exception:
        return "", ""


def backfill_goalies(csv_file=GAMES_FILE):
    """Add starting goalie data to existing games CSV."""
    df = pd.read_csv(csv_file)
    if "home_starter" not in df.columns:
        df["home_starter"] = ""
    if "away_starter" not in df.columns:
        df["away_starter"] = ""

    df["home_starter"] = df["home_starter"].fillna("")
    df["away_starter"] = df["away_starter"].fillna("")

    needs_fill = (df["home_starter"].str.strip() == "") | (df["away_starter"].str.strip() == "")
    n_need = needs_fill.sum()
    logging.info("Games needing goalie data: %d / %d", n_need, len(df))
    if n_need == 0:
        return

    # Determine date range
    dates = pd.to_datetime(df["date"])
    start_year = dates.min().year
    end_year = dates.max().year

    # Get team abbreviations in NHL API format
    team_abbrs = sorted(set(
        _OURS_TO_API.get(a, a) for a in TEAM_ABBR.values()
    ))
    seasons = _get_nhl_seasons(start_year, end_year)
    logging.info("Seasons: %s, Teams: %d", seasons, len(team_abbrs))

    # Build game ID lookup from team schedules
    game_map = _build_game_id_map(seasons, team_abbrs)

    # Convert dates for matching
    df["_date_str"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    filled = 0
    not_found = 0
    for idx in df.index[needs_fill]:
        date_str = df.at[idx, "_date_str"]
        home_full = df.at[idx, "home_team"]
        away_full = df.at[idx, "away_team"]
        h_abbr = _FULL_TO_ABBR.get(home_full, "")
        a_abbr = _FULL_TO_ABBR.get(away_full, "")

        if not h_abbr or not a_abbr:
            not_found += 1
            continue

        # Convert to NHL API abbreviations for lookup
        h_api = _OURS_TO_API.get(h_abbr, h_abbr)
        a_api = _OURS_TO_API.get(a_abbr, a_abbr)

        # Try exact date, then ±1 day
        game_id = None
        for d_offset in [0, -1, 1]:
            try_date = (pd.Timestamp(date_str) + pd.Timedelta(days=d_offset)).strftime("%Y-%m-%d")
            game_id = game_map.get((try_date, h_api, a_api))
            if game_id:
                break

        if not game_id:
            not_found += 1
            continue

        hg, ag = _get_starting_goalies(game_id)
        time.sleep(0.1)
        if hg:
            df.at[idx, "home_starter"] = hg
        if ag:
            df.at[idx, "away_starter"] = ag
        if hg or ag:
            filled += 1

        if (filled + not_found) % 100 == 0:
            logging.info("  Progress: %d filled, %d not found, %d/%d processed",
                         filled, not_found, filled + not_found, n_need)

    df.drop(columns=["_date_str"], inplace=True)
    df.to_csv(csv_file, index=False)

    final_h = (df["home_starter"].str.strip() != "").sum()
    final_a = (df["away_starter"].str.strip() != "").sum()
    logging.info("\nDone! Filled %d games (%d not found in API).", filled, not_found)
    logging.info("Coverage: home=%d/%d (%.1f%%), away=%d/%d (%.1f%%)",
                 final_h, len(df), 100 * final_h / len(df),
                 final_a, len(df), 100 * final_a / len(df))


if __name__ == "__main__":
    backfill_goalies()
