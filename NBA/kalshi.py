"""Kalshi API integration for auto-fetching live contract prices.

Fetches game-level moneyline markets from Kalshi's public API (no auth needed).
Used for auto-Kelly sizing when making live predictions.

API: https://api.elections.kalshi.com/trade-api/v2
Rate limit: 20 req/sec (basic tier, read-only)
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from difflib import get_close_matches

import requests

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

# Series tickers for each sport's game-level markets
SPORT_SERIES = {
    "mlb": "KXMLBGAME",
    "nba": "KXNBAGAME",
    "nfl": "KXNFLGAME",
    "nhl": "KXNHLGAME",
}

# ── In-memory cache ──────────────────────────────────────────────
_cache = {}            # {sport: {"events": [...], "fetched_at": float}}
_cache_lock = threading.Lock()
CACHE_TTL_SECONDS = 60


def _is_cache_fresh(sport):
    with _cache_lock:
        entry = _cache.get(sport)
        if not entry:
            return False
        return (time.time() - entry["fetched_at"]) < CACHE_TTL_SECONDS


# ── Kalshi team name mapping ─────────────────────────────────────
# Kalshi uses short city names (e.g. "Chicago C" for Cubs, "New York M" for Mets).
# We build fuzzy-match targets from these.

# Maps Kalshi short names → common full names for matching.
# Keys are lowercase Kalshi names from yes_sub_title / event titles.
KALSHI_TEAM_MAP = {
    # MLB
    "arizona": "Arizona Diamondbacks", "atlanta": "Atlanta Braves",
    "baltimore": "Baltimore Orioles", "boston": "Boston Red Sox",
    "chicago c": "Chicago Cubs", "chicago w": "Chicago White Sox",
    "cincinnati": "Cincinnati Reds", "cleveland": "Cleveland Guardians",
    "colorado": "Colorado Rockies", "detroit": "Detroit Tigers",
    "houston": "Houston Astros", "kansas city": "Kansas City Royals",
    "los angeles a": "Los Angeles Angels", "los angeles d": "Los Angeles Dodgers",
    "miami": "Miami Marlins", "milwaukee": "Milwaukee Brewers",
    "minnesota": "Minnesota Twins", "new york m": "New York Mets",
    "new york y": "New York Yankees", "oakland": "Oakland Athletics",
    "sacramento": "Oakland Athletics",
    "philadelphia": "Philadelphia Phillies", "pittsburgh": "Pittsburgh Pirates",
    "san diego": "San Diego Padres", "san francisco": "San Francisco Giants",
    "seattle": "Seattle Mariners", "st. louis": "St. Louis Cardinals",
    "st louis": "St. Louis Cardinals",
    "tampa bay": "Tampa Bay Rays", "texas": "Texas Rangers",
    "toronto": "Toronto Blue Jays", "washington": "Washington Nationals",
    # NBA
    "los angeles l": "Los Angeles Lakers", "los angeles c": "Los Angeles Clippers",
    "golden state": "Golden State Warriors", "san antonio": "San Antonio Spurs",
    "new orleans": "New Orleans Pelicans", "oklahoma city": "Oklahoma City Thunder",
    "portland": "Portland Trail Blazers", "sacramento": "Sacramento Kings",
    "indiana": "Indiana Pacers", "memphis": "Memphis Grizzlies",
    "charlotte": "Charlotte Hornets", "orlando": "Orlando Magic",
    "brooklyn": "Brooklyn Nets", "utah": "Utah Jazz",
    "denver": "Denver Nuggets", "dallas": "Dallas Mavericks",
    "phoenix": "Phoenix Suns",
    "new york": "New York Knicks", "new york k": "New York Knicks",
    # NFL
    "green bay": "Green Bay Packers", "las vegas": "Las Vegas Raiders",
    "new england": "New England Patriots", "jacksonville": "Jacksonville Jaguars",
    "buffalo": "Buffalo Bills", "carolina": "Carolina Panthers",
    "los angeles r": "Los Angeles Rams",
    "tampa bay b": "Tampa Bay Buccaneers",
    # NHL
    "new york r": "New York Rangers", "new york i": "New York Islanders",
    "new jersey": "New Jersey Devils", "los angeles k": "Los Angeles Kings",
    "san jose": "San Jose Sharks", "st louis b": "St. Louis Blues",
    "tampa bay l": "Tampa Bay Lightning", "columbus": "Columbus Blue Jackets",
    "winnipeg": "Winnipeg Jets", "calgary": "Calgary Flames",
    "edmonton": "Edmonton Oilers", "vancouver": "Vancouver Canucks",
    "ottawa": "Ottawa Senators", "montreal": "Montreal Canadiens",
    "florida": "Florida Panthers",
}


def _normalize(name):
    """Lowercase and strip common suffixes for matching."""
    return name.lower().strip()


def _kalshi_to_full(kalshi_name):
    """Convert Kalshi short team name to full team name for matching."""
    key = _normalize(kalshi_name)
    if key in KALSHI_TEAM_MAP:
        return KALSHI_TEAM_MAP[key]
    # Try prefix match (e.g. "Chicago" matches "Chicago Cubs" if only one Chicago team)
    matches = [v for k, v in KALSHI_TEAM_MAP.items() if k.startswith(key)]
    if len(matches) == 1:
        return matches[0]
    return kalshi_name  # Return as-is if no match


def _teams_match(kalshi_name, model_team):
    """Check if a Kalshi team name matches a model team name."""
    full = _kalshi_to_full(kalshi_name)
    model_l = _normalize(model_team)
    full_l = _normalize(full)
    # Exact match
    if full_l == model_l:
        return True
    # One contains the other
    if full_l in model_l or model_l in full_l:
        return True
    # City match (first word or two)
    kalshi_city = _normalize(kalshi_name).split()[0]
    model_city = model_l.split()[0]
    if kalshi_city == model_city and len(kalshi_city) > 3:
        return True
    # Fuzzy match
    close = get_close_matches(full_l, [model_l], n=1, cutoff=0.6)
    return len(close) > 0


# ── API functions ────────────────────────────────────────────────

def fetch_kalshi_events(sport):
    """Fetch open game-level events for a sport from Kalshi.

    Args:
        sport: "mlb", "nba", "nfl", or "nhl"

    Returns:
        List of event dicts with nested markets, or empty list on failure.
    """
    sport_l = sport.lower()
    if _is_cache_fresh(sport_l):
        with _cache_lock:
            return _cache[sport_l]["events"]

    series = SPORT_SERIES.get(sport_l)
    if not series:
        logging.warning("No Kalshi series for sport: %s", sport)
        return []

    try:
        resp = requests.get(
            f"{BASE_URL}/events",
            params={
                "status": "open",
                "series_ticker": series,
                "limit": 50,
                "with_nested_markets": "true",
            },
            timeout=10,
        )
        resp.raise_for_status()
        events = resp.json().get("events", [])
        with _cache_lock:
            _cache[sport_l] = {"events": events, "fetched_at": time.time()}
        return events
    except requests.RequestException as e:
        logging.error("Kalshi API error: %s", e)
        # Return stale cache if available
        with _cache_lock:
            if sport_l in _cache:
                logging.info("Using cached Kalshi data (may be stale)")
                return _cache[sport_l]["events"]
        return []


def fetch_orderbook(ticker):
    """Fetch orderbook for a specific market ticker.

    Returns:
        (yes_bid_cents, yes_ask_cents) or (None, None) on failure.
        Prices are in cents (1-99).
    """
    try:
        resp = requests.get(
            f"{BASE_URL}/markets/{ticker}/orderbook",
            timeout=10,
        )
        resp.raise_for_status()
        ob = resp.json().get("orderbook_fp", {})

        yes_levels = ob.get("yes_dollars", [])
        no_levels = ob.get("no_dollars", [])

        # Best YES bid = highest price someone will buy YES at
        yes_bid = None
        if yes_levels:
            prices = [float(p) for p, q in yes_levels if float(p) >= 0.05]
            if prices:
                yes_bid = int(round(max(prices) * 100))

        # Best YES ask = 1 - highest NO bid price
        yes_ask = None
        if no_levels:
            no_prices = [float(p) for p, q in no_levels if float(p) >= 0.05]
            if no_prices:
                yes_ask = int(round((1.0 - max(no_prices)) * 100))

        return yes_bid, yes_ask

    except requests.RequestException as e:
        logging.error("Kalshi orderbook error for %s: %s", ticker, e)
        return None, None


def _compute_midpoint(bid, ask):
    """Compute midpoint from bid/ask, handling None values."""
    if bid and ask:
        return (bid + ask) // 2
    elif bid:
        return bid
    elif ask:
        return ask
    return None


def _complement_price(price):
    """Compute the NO side price: 100 - YES price."""
    if price is not None:
        return 100 - price
    return None


def find_kalshi_odds(home_team, away_team, sport):
    """Find Kalshi market odds for a specific game matchup.

    Handles both single-market events (YES/NO on one ticker) and
    two-market events (separate ticker per team).

    Args:
        home_team: Full team name (e.g. "New York Yankees")
        away_team: Full team name (e.g. "Boston Red Sox")
        sport: "mlb", "nba", "nfl", or "nhl"

    Returns:
        Dict with keys:
            home_yes_bid: int (cents) — best bid for home team YES
            home_yes_ask: int (cents) — best ask for home team YES
            home_midpoint: int (cents) — midpoint of home bid/ask
            away_yes_bid: int (cents) — best bid for away team YES
            away_yes_ask: int (cents) — best ask for away team YES
            away_midpoint: int (cents) — midpoint of away bid/ask
            home_ticker: str — market ticker for home team
            away_ticker: str — market ticker for away team
            event_title: str — event title
            yes_team: str — which team the YES side represents
            midpoint: int (cents) — midpoint for predicted winner (for Kelly)
        Or None if no matching market found.
    """
    events = fetch_kalshi_events(sport)
    if not events:
        return None

    for ev in events:
        markets = ev.get("markets", [])
        if not markets:
            continue

        # ── Two-market events: one ticker per team ──────────────
        if len(markets) >= 2:
            home_market = None
            away_market = None

            for m in markets:
                sub = m.get("yes_sub_title", "") or ""
                if _teams_match(sub, home_team):
                    home_market = m
                elif _teams_match(sub, away_team):
                    away_market = m

            if home_market and away_market:
                home_ticker = home_market.get("ticker", "")
                away_ticker = away_market.get("ticker", "")

                home_bid, home_ask = fetch_orderbook(home_ticker)
                away_bid, away_ask = fetch_orderbook(away_ticker)

                # If one side failed, derive from the other
                if home_bid is None and home_ask is None and (away_bid or away_ask):
                    home_bid = _complement_price(away_ask)
                    home_ask = _complement_price(away_bid)
                elif away_bid is None and away_ask is None and (home_bid or home_ask):
                    away_bid = _complement_price(home_ask)
                    away_ask = _complement_price(home_bid)

                if home_bid is None and home_ask is None and away_bid is None and away_ask is None:
                    return None

                home_mid = _compute_midpoint(home_bid, home_ask)
                away_mid = _compute_midpoint(away_bid, away_ask)

                return {
                    "home_yes_bid": home_bid,
                    "home_yes_ask": home_ask,
                    "home_midpoint": home_mid,
                    "away_yes_bid": away_bid,
                    "away_yes_ask": away_ask,
                    "away_midpoint": away_mid,
                    "home_ticker": home_ticker,
                    "away_ticker": away_ticker,
                    "event_title": ev.get("title", ""),
                    "yes_team": home_market.get("yes_sub_title", ""),
                    "midpoint": home_mid,  # backward compat: home side
                }

        # ── Single-market events: YES = one team, NO = the other ──
        if len(markets) == 1:
            m = markets[0]
            sub = m.get("yes_sub_title", "") or ""
            ticker = m.get("ticker", "")

            # Figure out which team is the YES side
            yes_is_home = _teams_match(sub, home_team)
            yes_is_away = _teams_match(sub, away_team)

            if not yes_is_home and not yes_is_away:
                # Also try matching from event title
                title = ev.get("title", "") or ""
                title_lower = title.lower()
                if _normalize(home_team) in title_lower or _normalize(away_team) in title_lower:
                    # Event matches our game but we can't tell the YES side
                    pass
                else:
                    continue

            if yes_is_home or yes_is_away:
                yes_bid, yes_ask = fetch_orderbook(ticker)
                if yes_bid is None and yes_ask is None:
                    return None

                no_bid = _complement_price(yes_ask)
                no_ask = _complement_price(yes_bid)

                if yes_is_home:
                    home_bid, home_ask = yes_bid, yes_ask
                    away_bid, away_ask = no_bid, no_ask
                else:
                    home_bid, home_ask = no_bid, no_ask
                    away_bid, away_ask = yes_bid, yes_ask

                home_mid = _compute_midpoint(home_bid, home_ask)
                away_mid = _compute_midpoint(away_bid, away_ask)

                return {
                    "home_yes_bid": home_bid,
                    "home_yes_ask": home_ask,
                    "home_midpoint": home_mid,
                    "away_yes_bid": away_bid,
                    "away_yes_ask": away_ask,
                    "away_midpoint": away_mid,
                    "home_ticker": ticker if yes_is_home else "",
                    "away_ticker": ticker if yes_is_away else "",
                    "event_title": ev.get("title", ""),
                    "yes_team": sub,
                    "midpoint": home_mid,  # backward compat
                }

    return None


def show_kalshi_odds(sport):
    """Display all available Kalshi game markets for a sport.

    Shows both teams' implied probabilities for each event.
    Returns list of events for further use.
    """
    try:
        from color_helpers import cok, cerr, cwarn, chi, cdim, cbold, div
    except ImportError:
        def cok(t): return str(t)
        def cerr(t): return str(t)
        def cwarn(t): return str(t)
        def chi(t): return str(t)
        def cdim(t): return str(t)
        def cbold(t): return str(t)
        def div(n=80): print("-" * n)

    events = fetch_kalshi_events(sport)
    if not events:
        print(cwarn("  No Kalshi markets found for %s" % sport.upper()))
        return []

    print("\n  " + cbold("KALSHI %s GAME MARKETS" % sport.upper()))
    div(70)

    for ev in events:
        title = ev.get("title", "?")
        markets = ev.get("markets", [])
        if not markets:
            continue

        if len(markets) >= 2:
            # Two-market event: each market is a team's YES
            team1_sub = markets[0].get("yes_sub_title", "") or markets[0].get("ticker", "")
            team2_sub = markets[1].get("yes_sub_title", "") or markets[1].get("ticker", "")

            ticker0 = markets[0].get("ticker", "")
            ticker1 = markets[1].get("ticker", "")

            bid0, ask0 = fetch_orderbook(ticker0)
            bid1, ask1 = fetch_orderbook(ticker1)

            # If one side has data but other doesn't, derive complement
            if (bid0 is None and ask0 is None) and (bid1 is not None or ask1 is not None):
                bid0 = _complement_price(ask1)
                ask0 = _complement_price(bid1)
            elif (bid1 is None and ask1 is None) and (bid0 is not None or ask0 is not None):
                bid1 = _complement_price(ask0)
                ask1 = _complement_price(bid0)

            mid0 = _compute_midpoint(bid0, ask0)
            mid1 = _compute_midpoint(bid1, ask1)

            if mid0 is not None or mid1 is not None:
                t1_s = "%s: %d\u00a2 (%d%%)" % (team1_sub, mid0, mid0) if mid0 else "%s: ?" % team1_sub
                t2_s = "%s: %d\u00a2 (%d%%)" % (team2_sub, mid1, mid1) if mid1 else "%s: ?" % team2_sub
                print("  %s  |  %s" % (cok(t1_s), cok(t2_s)))
            else:
                print("  %s  %s" % (chi(title), cdim("no orders")))

        elif len(markets) == 1:
            # Single-market event: YES = one team, NO = the other
            m = markets[0]
            yes_team = m.get("yes_sub_title", "") or m.get("ticker", "")
            ticker = m.get("ticker", "")

            bid, ask = fetch_orderbook(ticker)
            mid_yes = _compute_midpoint(bid, ask)

            if mid_yes is not None:
                mid_no = 100 - mid_yes
                # Try to extract the other team name from event title
                no_team = "Opponent"
                ev_title = ev.get("title", "")
                # Typical title: "Team A vs Team B" or similar
                for sep in [" vs ", " vs. ", " v ", " @ "]:
                    if sep in ev_title:
                        parts = ev_title.split(sep)
                        if len(parts) == 2:
                            # Figure out which part is the YES team
                            p0, p1 = parts[0].strip(), parts[1].strip()
                            if _teams_match(yes_team, p0) or _normalize(yes_team) in _normalize(p0):
                                no_team = p1
                            else:
                                no_team = p0
                        break

                t1_s = "%s: %d\u00a2 (%d%%)" % (yes_team, mid_yes, mid_yes)
                t2_s = "%s: %d\u00a2 (%d%%)" % (no_team, mid_no, mid_no)
                print("  %s  |  %s" % (cok(t1_s), cok(t2_s)))
            else:
                print("  %s  %s" % (chi(title), cdim("no orders")))

    div(70)
    return events
