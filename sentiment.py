"""Reddit sentiment analysis for sports predictions.

Uses PRAW (Reddit API) + VADER sentiment scoring.
Free with Reddit OAuth2 credentials (no payment required).

To set up:
1. Go to https://www.reddit.com/prefs/apps
2. Create a "script" application
3. Set env vars: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
   Or create .reddit_creds.json in Claude/ root with those fields.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from collections import defaultdict

SENTIMENT_CACHE_FILE = "sentiment_cache.json"
CACHE_MAX_AGE_HOURS = 4

# Subreddits per sport
SPORT_SUBREDDITS = {
    "nfl": ["nfl", "fantasyfootball"],
    "nba": ["nba"],
    "mlb": ["baseball"],
    "nhl": ["hockey"],
}

# Team name variants for Reddit search
NFL_TEAM_KEYWORDS = {
    "Arizona Cardinals": ["cardinals", "arizona"],
    "Atlanta Falcons": ["falcons", "atlanta falcons"],
    "Baltimore Ravens": ["ravens", "baltimore"],
    "Buffalo Bills": ["bills", "buffalo bills"],
    "Carolina Panthers": ["panthers", "carolina"],
    "Chicago Bears": ["bears", "chicago bears", "da bears"],
    "Cincinnati Bengals": ["bengals", "cincinnati"],
    "Cleveland Browns": ["browns", "cleveland"],
    "Dallas Cowboys": ["cowboys", "dallas"],
    "Denver Broncos": ["broncos", "denver"],
    "Detroit Lions": ["lions", "detroit"],
    "Green Bay Packers": ["packers", "green bay"],
    "Houston Texans": ["texans", "houston texans"],
    "Indianapolis Colts": ["colts", "indianapolis"],
    "Jacksonville Jaguars": ["jaguars", "jags"],
    "Kansas City Chiefs": ["chiefs", "kansas city"],
    "Las Vegas Raiders": ["raiders", "las vegas"],
    "Los Angeles Chargers": ["chargers", "la chargers"],
    "Los Angeles Rams": ["rams", "la rams"],
    "Miami Dolphins": ["dolphins", "miami dolphins"],
    "Minnesota Vikings": ["vikings", "minnesota"],
    "New England Patriots": ["patriots", "pats", "new england"],
    "New Orleans Saints": ["saints", "new orleans"],
    "New York Giants": ["giants", "ny giants"],
    "New York Jets": ["jets", "ny jets"],
    "Philadelphia Eagles": ["eagles", "philly eagles"],
    "Pittsburgh Steelers": ["steelers", "pittsburgh"],
    "San Francisco 49ers": ["49ers", "niners", "san francisco"],
    "Seattle Seahawks": ["seahawks", "seattle"],
    "Tampa Bay Buccaneers": ["buccaneers", "bucs", "tampa bay"],
    "Tennessee Titans": ["titans", "tennessee"],
    "Washington Commanders": ["commanders", "washington"],
}

MLB_TEAM_KEYWORDS = {
    "Arizona Diamondbacks": ["diamondbacks", "dbacks"],
    "Atlanta Braves": ["braves", "atlanta braves"],
    "Baltimore Orioles": ["orioles", "baltimore"],
    "Boston Red Sox": ["red sox", "redsox", "boston"],
    "Chicago Cubs": ["cubs", "chicago cubs"],
    "Chicago White Sox": ["white sox", "whitesox"],
    "Cincinnati Reds": ["reds", "cincinnati reds"],
    "Cleveland Guardians": ["guardians", "cleveland"],
    "Colorado Rockies": ["rockies", "colorado"],
    "Detroit Tigers": ["tigers", "detroit tigers"],
    "Houston Astros": ["astros", "houston astros"],
    "Kansas City Royals": ["royals", "kansas city"],
    "Los Angeles Angels": ["angels", "la angels", "halos"],
    "Los Angeles Dodgers": ["dodgers", "la dodgers"],
    "Miami Marlins": ["marlins", "miami marlins"],
    "Milwaukee Brewers": ["brewers", "milwaukee"],
    "Minnesota Twins": ["twins", "minnesota twins"],
    "New York Mets": ["mets", "ny mets"],
    "New York Yankees": ["yankees", "yanks"],
    "Oakland Athletics": ["athletics", "oakland"],
    "Philadelphia Phillies": ["phillies", "philly"],
    "Pittsburgh Pirates": ["pirates", "pittsburgh"],
    "San Diego Padres": ["padres", "san diego"],
    "San Francisco Giants": ["giants", "sf giants"],
    "Seattle Mariners": ["mariners", "seattle"],
    "St. Louis Cardinals": ["cardinals", "stl cardinals"],
    "Tampa Bay Rays": ["rays", "tampa bay rays"],
    "Texas Rangers": ["rangers", "texas rangers"],
    "Toronto Blue Jays": ["blue jays", "jays", "toronto"],
    "Washington Nationals": ["nationals", "nats"],
}

SPORT_TEAM_KEYWORDS = {
    "nfl": NFL_TEAM_KEYWORDS,
    "mlb": MLB_TEAM_KEYWORDS,
}


def _get_reddit_creds():
    """Load Reddit API credentials from env or file."""
    client_id = os.environ.get("REDDIT_CLIENT_ID", "")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "")
    user_agent = os.environ.get("REDDIT_USER_AGENT", "sports-sentiment-bot/1.0")

    if client_id and client_secret:
        return client_id, client_secret, user_agent

    creds_file = os.path.join(os.path.dirname(__file__), ".reddit_creds.json")
    if os.path.exists(creds_file):
        with open(creds_file, "r") as f:
            creds = json.load(f)
        return (creds.get("client_id", ""),
                creds.get("client_secret", ""),
                creds.get("user_agent", user_agent))

    return "", "", user_agent


def _get_reddit_instance():
    """Create a PRAW Reddit instance."""
    import praw

    client_id, client_secret, user_agent = _get_reddit_creds()
    if not client_id or not client_secret:
        logging.warning("Reddit credentials not configured. "
                        "Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET env vars, "
                        "or create .reddit_creds.json")
        return None

    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )
        return reddit
    except Exception as e:
        logging.error("Failed to create Reddit instance: %s", e)
        return None


def _cache_path(sport):
    """Get cache file path, checking both sport dir and current dir."""
    base = os.path.dirname(__file__)
    # Try sport-specific directory first
    sport_dir = os.path.join(base, sport.upper()[:3] + (sport[3:] if len(sport) > 3 else "") + "Claude")
    if not os.path.isdir(sport_dir):
        sport_dir = os.path.join(base, sport.upper() + "Claude")
    if not os.path.isdir(sport_dir):
        # Fallback: save in shared cache dir
        sport_dir = os.path.join(base, "sentiment_cache")
    os.makedirs(sport_dir, exist_ok=True)
    return os.path.join(sport_dir, SENTIMENT_CACHE_FILE)


def _load_cache(sport):
    cache_file = _cache_path(sport)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
            # Check freshness
            fetched = data.get("fetched_at", "")
            if fetched:
                dt = datetime.fromisoformat(fetched)
                age_hours = (datetime.now() - dt).total_seconds() / 3600
                if age_hours < CACHE_MAX_AGE_HOURS:
                    return data
                # Return stale data with a flag so callers can decide
                data["_stale"] = True
                return data
        except (json.JSONDecodeError, IOError, ValueError):
            pass
    return None


def _save_cache(sport, data):
    cache_file = _cache_path(sport)
    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)


def analyze_sentiment_text(text):
    """Analyze sentiment of text using VADER.

    Returns dict with compound, positive, negative, neutral scores.
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    return {
        "compound": scores["compound"],
        "positive": scores["pos"],
        "negative": scores["neg"],
        "neutral": scores["neu"],
    }


def fetch_team_sentiment(reddit, team, keywords, subreddits, limit=50):
    """Fetch and analyze Reddit sentiment for a team.

    Returns dict with sentiment scores and post count.
    """
    all_scores = []

    for sub_name in subreddits:
        try:
            subreddit = reddit.subreddit(sub_name)
            # Search for team-related posts from last 24 hours
            for keyword in keywords[:2]:  # Limit queries per team
                try:
                    for post in subreddit.search(keyword, time_filter="day", limit=limit):
                        text = (post.title or "") + " " + (post.selftext or "")[:500]
                        if len(text.strip()) > 10:
                            score = analyze_sentiment_text(text)
                            score["upvote_ratio"] = post.upvote_ratio
                            score["score"] = post.score
                            all_scores.append(score)
                except Exception:
                    continue
        except Exception as e:
            logging.debug("Error searching r/%s: %s", sub_name, e)
            continue

    if not all_scores:
        return {"compound": 0, "n_posts": 0, "confidence": 0}

    # Weight by upvotes (popular posts matter more)
    weights = [max(1, s.get("score", 1)) for s in all_scores]
    total_weight = sum(weights)

    weighted_compound = sum(s["compound"] * w for s, w in zip(all_scores, weights)) / total_weight
    avg_positive = sum(s["positive"] * w for s, w in zip(all_scores, weights)) / total_weight
    avg_negative = sum(s["negative"] * w for s, w in zip(all_scores, weights)) / total_weight

    return {
        "compound": round(weighted_compound, 4),
        "positive": round(avg_positive, 4),
        "negative": round(avg_negative, 4),
        "n_posts": len(all_scores),
        "confidence": min(1.0, len(all_scores) / 20),  # 20+ posts = full confidence
    }


def fetch_all_team_sentiment(sport, force=False):
    """Fetch sentiment for all teams in a sport.

    Returns dict: team_name -> sentiment_dict
    """
    sport_l = sport.lower()

    # Check cache
    if not force:
        cached = _load_cache(sport_l)
        if cached and not cached.get("_stale"):
            logging.info("Using cached sentiment data (%d teams)", len(cached.get("teams", {})))
            return cached.get("teams", {})

    reddit = _get_reddit_instance()
    if reddit is None:
        # No Reddit credentials — return stale cache if available
        if not force:
            cached = _load_cache(sport_l)
            if cached:
                logging.info("No Reddit creds, using stale sentiment cache")
                return cached.get("teams", {})
        return {}

    team_keywords = SPORT_TEAM_KEYWORDS.get(sport_l, {})
    subreddits = SPORT_SUBREDDITS.get(sport_l, [])

    if not team_keywords:
        logging.warning("No team keywords configured for sport: %s", sport)
        return {}

    results = {}
    total = len(team_keywords)

    for i, (team, keywords) in enumerate(team_keywords.items(), 1):
        if i % 8 == 0:
            logging.info("Sentiment progress: %d/%d teams", i, total)
        try:
            sentiment = fetch_team_sentiment(reddit, team, keywords, subreddits)
            results[team] = sentiment
        except Exception as e:
            logging.debug("Failed sentiment for %s: %s", team, e)
            results[team] = {"compound": 0, "n_posts": 0, "confidence": 0}

    # Cache results
    cache_data = {
        "fetched_at": datetime.now().isoformat(),
        "sport": sport_l,
        "teams": results,
    }
    _save_cache(sport_l, cache_data)

    return results


def get_sentiment_features(home_team, away_team, sentiment_data):
    """Get sentiment-based features for a matchup.

    Returns dict of features for the meta-learner.
    """
    home_sent = sentiment_data.get(home_team, {})
    away_sent = sentiment_data.get(away_team, {})

    home_compound = home_sent.get("compound", 0)
    away_compound = away_sent.get("compound", 0)
    home_conf = home_sent.get("confidence", 0)
    away_conf = away_sent.get("confidence", 0)

    return {
        "home_sentiment": home_compound,
        "away_sentiment": away_compound,
        "sentiment_diff": home_compound - away_compound,
        "sentiment_confidence": min(home_conf, away_conf),
        "home_sentiment_posts": home_sent.get("n_posts", 0),
        "away_sentiment_posts": away_sent.get("n_posts", 0),
    }


def show_sentiment_report(sport, sentiment_data=None):
    """Display sentiment rankings for all teams."""
    if sentiment_data is None:
        sentiment_data = fetch_all_team_sentiment(sport)

    if not sentiment_data:
        print("  No sentiment data available. Configure Reddit credentials first.")
        print("  Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET env vars.")
        return

    # Sort by compound sentiment
    sorted_teams = sorted(sentiment_data.items(),
                          key=lambda x: x[1].get("compound", 0), reverse=True)

    print(f"\n  {'Team':<28} {'Sentiment':>10} {'Posts':>6} {'Positive':>9} {'Negative':>9} {'Confidence':>11}")
    print("  " + "-" * 78)

    for team, sent in sorted_teams:
        compound = sent.get("compound", 0)
        n_posts = sent.get("n_posts", 0)
        pos = sent.get("positive", 0)
        neg = sent.get("negative", 0)
        conf = sent.get("confidence", 0)

        # Color indicator
        if compound > 0.1:
            indicator = "+"
        elif compound < -0.1:
            indicator = "-"
        else:
            indicator = " "

        print(f"  {team:<28} {indicator}{compound:>+9.3f} {n_posts:>6} {pos:>8.3f} {neg:>9.3f} {conf:>10.0%}")
