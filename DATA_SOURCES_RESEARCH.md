# Complete Free Sports Data Sources for Python Prediction Systems
## Research Date: 2026-03-30

---

## 1. OFFICIAL APIs & PYTHON WRAPPERS

### NBA

**nba_api**
- PyPI: `pip install nba_api`
- GitHub: https://github.com/swar/nba_api
- Data: Complete NBA.com stats API wrapper. Player stats, team stats, game logs, shot charts, player tracking (speed/distance), box scores, play-by-play, league leaders, draft combine stats, hustle stats, clutch stats, advanced stats
- Tracking endpoints: LeagueDashPtStats (speed/distance), BoxScorePlayerTrackV3 (per-game tracking)
- Rate limits: No API key needed. Unofficial API - be respectful with request frequency (add delays)
- Requires: Python 3.10+, requests, numpy
- Sports: NBA, WNBA, G-League
- Quality: 9/10 - most comprehensive free NBA data source

**py_ball**
- GitHub: https://github.com/basketballrelativity/py_ball
- Data: Alternative stats.nba.com wrapper focused on NBA/WNBA
- Install: `pip install py_ball`
- Quality: 6/10 - less actively maintained than nba_api

### MLB

**pybaseball**
- PyPI: `pip install pybaseball`
- GitHub: https://github.com/jldbc/pybaseball
- Data: Statcast pitch-level data (spin rate, exit velocity, launch angle), batting/pitching stats from Baseball Reference and FanGraphs, standings, Retrosheet game logs, park factors, amateur draft data, WAR calculations
- Rate limits: Baseball Savant limits to 30,000 rows per query; queries >5 days auto-split
- Earliest Statcast data: 2008 season
- Sports: MLB
- Quality: 10/10 - gold standard for baseball data

**MLB-StatsAPI**
- PyPI: `pip install MLB-StatsAPI`
- GitHub: https://github.com/toddrob99/MLB-StatsAPI
- Data: Official MLB Stats API wrapper. Schedule, standings, rosters, player stats, game data, team info, draft, awards, venues
- Rate limits: Free, no key needed. Official API at statsapi.mlb.com
- Sports: MLB, MiLB
- Quality: 8/10 - official API, very reliable

**python-mlb-statsapi**
- PyPI: `pip install python-mlb-statsapi`
- GitHub: https://github.com/zero-sum-seattle/python-mlb-statsapi
- Data: Same MLB Stats API but with Pydantic models for data validation
- Requires: Python 3.10+
- Quality: 7/10 - modern, but newer/less tested

### NFL

**nfl_data_py**
- PyPI: `pip install nfl_data_py`
- GitHub: https://github.com/nflverse/nfl_data_py
- Data: Play-by-play (back to 1999), weekly player stats, seasonal stats, rosters, win totals, scoring lines, officials, draft picks, draft pick values, combine results, schedules, team descriptive info, ID mappings across sites
- Rate limits: None - pulls from nflverse data repository (pre-compiled CSVs)
- Downcast option: reduces memory ~30% (float64 -> float32)
- Sports: NFL
- Quality: 10/10 - the definitive NFL data package, includes EPA/WPA/CPOE

**nflreadpy**
- GitHub: https://github.com/nflverse/nflreadpy
- Data: Python port of nflreadr R package for loading nflverse data
- Quality: 7/10 - newer alternative to nfl_data_py

**nfllivepy**
- PyPI: `pip install nfllivepy`
- GitHub: https://github.com/jlkazan/nfllivepy
- Data: Live NFL play-by-play data during games
- Quality: 5/10 - niche use for live data

### NHL

**hockey_scraper**
- PyPI: `pip install hockey_scraper`
- GitHub: https://github.com/HarryShomer/Hockey-Scraper
- Data: Play-by-play and shift data from NHL API and website. Preseason, regular season, playoffs from 2007-08 onward
- Rate limits: Be respectful; can save raw files locally for re-scraping (docs_dir=True)
- Sports: NHL
- Quality: 8/10 - reliable and well-documented

**nhl-api-py**
- PyPI: `pip install nhl-api-py`
- GitHub: https://github.com/coreyjs/nhl-api-py
- Data: NHL EDGE API wrapper. Teams, rosters, schedules, stats, standings, game data, EDGE tracking data (shot speed, skate speed), prospects, draft
- Base URL: https://api-web.nhle.com/v1/
- Rate limits: No API key needed
- Sports: NHL
- Quality: 8/10 - actively maintained, covers new NHL EDGE stats

**nhlpy**
- GitHub: https://github.com/jdel7/nhlpy
- Data: Thin wrapper mapping directly to NHL API endpoints
- Rate limits: No sign-up or API key required
- Quality: 6/10 - minimal but functional

**pyhockey**
- GitHub: https://github.com/hockey-stats/pyhockey
- Data: NHL data from MoneyPuck and NaturalStatTrick via DuckDB/Polars
- Quality: 7/10 - good for advanced analytics access

**nhldata**
- GitHub: https://github.com/TonyAllenPrice/nhldata
- Data: MoneyPuck shot data (1.7M+ shots from 2007-2023), team/player stats
- Quality: 7/10 - great for shot modeling

### Multi-Sport

**sportsipy / sportsreference**
- PyPI: `pip install sportsipy` or `pip install sportsreference`
- GitHub: https://github.com/roclark/sportsipy
- Data: Scrapes Sports-Reference sites. Team/player stats, box scores, schedules for MLB, NBA, NFL, NHL, NCAA FB, NCAA BB, soccer
- Rate limits: Scraping-based; add delays to avoid blocking
- Sports: MLB, NBA, NFL, NHL, NCAAF, NCAAB, Soccer
- Quality: 5/10 - can break when Sports-Reference changes HTML; maintenance inconsistent
- WARNING: May have issues with current site layouts

**sportsdataverse-py**
- PyPI: `pip install sportsdataverse`
- GitHub: https://github.com/sportsdataverse/sportsdataverse-py
- Data: ESPN play-by-play, box scores, schedules for multiple sports. Expected points and win probability for football
- Install full: `pip install sportsdataverse[all]`
- Sports: NFL, CFB, NBA, WNBA, NHL, MBB, WBB
- Quality: 7/10 - companion to popular R packages (cfbfastR, hoopR)

**BALLDONTLIE API**
- URL: https://www.balldontlie.io/
- Data: Stats, scores, schedules, rosters, injuries for multiple sports
- Rate limits: Free tier = 60 requests/minute. Full ALL-ACCESS = $499.99/month
- Sports: NBA, NFL, MLB, NHL, EPL, WNBA, NCAAF, NCAAB
- Quality: 7/10 - easy to use but limited free tier for non-NBA

**ESPN API (Undocumented)**
- Documentation: https://github.com/pseudo-r/Public-ESPN-API
- Gist: https://gist.github.com/nntrn/ee26cb2a0716de0947a0a4e9a157bc1c
- Data: Scores, schedules, standings, rosters, injuries, news, play-by-play across 20+ sports
- Rate limits: No authentication needed. Unknown rate limits. Can change/break without warning
- Example: https://site.web.api.espn.com/apis/site/v2/sports/basketball/nba/summary
- Sports: All major sports
- Quality: 6/10 - free and comprehensive but unstable/undocumented

**espn-api (Fantasy)**
- PyPI: `pip install espn-api`
- Data: ESPN Fantasy Football, Basketball, Hockey, Baseball league data
- Quality: 6/10 - specific to fantasy leagues

**TheSportsDB**
- URL: https://www.thesportsdb.com/free_sports_api
- Python wrapper: https://github.com/TralahM/thesportsdb
- Data: Team info, rosters, player info, event results, historical data, fanart
- Rate limits: Free = 30 requests/minute
- Sports: All major sports worldwide
- Quality: 5/10 - good for metadata, not for deep stats

**API-Sports**
- URL: https://api-sports.io/
- Data: Fixtures, lineups, standings, stats for NFL, NBA, MLB, NHL, soccer, and more
- Rate limits: Free tier = 100 requests/day
- Sports: 15+ sports
- Quality: 6/10 - decent free tier but daily limit is low

---

## 2. ODDS / BETTING DATA

**The Odds API**
- URL: https://the-odds-api.com/
- Python samples: https://github.com/the-odds-api/samples-python
- Data: Live and pre-game odds from 50+ bookmakers worldwide. Moneyline, spreads, totals, props
- Rate limits: Free = 500 requests/month (measured in credits: markets x regions per call). 30 req/sec hard limit. Historical data requires paid plan ($25+/month)
- Sports: NBA, MLB, NFL, NHL, soccer, tennis, etc.
- Quality: 8/10 - industry standard free odds API

**odds-api.io**
- URL: https://odds-api.io/
- Data: Sports betting odds, similar to The Odds API
- Rate limits: Free tier available with limited requests
- Quality: 5/10 - smaller provider

**SportsGameOdds**
- URL: https://sportsgameodds.com/
- Data: Historical odds, line movements, settlement data for backtesting
- Rate limits: Free tier with scaling pricing
- Quality: 6/10 - good for historical backtesting

**TheRundown**
- URL: https://therundown.io/
- Data: Real-time odds, scores, stats, player data, historical movement since 2020
- Rate limits: Free tier available
- Quality: 6/10

**OddsJam**
- URL: https://oddsjam.com/odds-api
- Data: Line movement, consensus odds, sharp book comparisons, closing lines, opening odds, live line changes
- Rate limits: Primarily commercial; limited free access
- Quality: 7/10 for data quality, but expensive for full access

**Kaggle Datasets (Historical)**
- Various datasets with historical odds data available for download
- Search: "sports betting odds" on kaggle.com
- Quality: 6/10 - static datasets, good for backtesting

---

## 3. ADVANCED ANALYTICS

### NBA Tracking / Shot Charts

**nba_api tracking endpoints**
- LeagueDashPtStats: Player speed, distance (offensive/defensive), touches, paint touches, elbow touches
- ShotChartDetail: Every shot with x,y coordinates, distance, zone, result
- BoxScorePlayerTrackV3: Per-game tracking metrics
- PlayerDashPtShots: Shot type breakdowns (pullup, catch-and-shoot, etc.)
- Install: `pip install nba_api`
- Quality: 9/10 - official NBA tracking data, free

**shot-chart**
- PyPI: `pip install shot-chart`
- Data: NBA shot chart visualization tools
- Quality: 5/10 - visualization focused

**sportypy**
- Docs: https://sportypy.sportsdataverse.org/
- Data: Court/field/rink plotting for overlay of tracking data
- Quality: 6/10 - visualization support

### MLB Pitch-Level / Statcast

**pybaseball (Statcast)**
- Pitch-level: Spin rate, spin axis, release point, velocity, movement, exit velocity, launch angle, sprint speed, catch probability, expected stats (xBA, xSLG, xwOBA, xERA)
- Functions: statcast(), statcast_batter(), statcast_pitcher(), pitching_stats(), batting_stats()
- Quality: 10/10 - the richest freely available pitch data

### NFL Advanced

**nfl_data_py advanced columns**
- EPA (Expected Points Added), WPA (Win Probability Added), CPOE (Completion % Over Expected)
- Air yards, YAC, target share, ADOT
- Pre-snap reads, formation data, personnel packages
- Quality: 10/10 - academic-grade analytics built in

### NHL Expected Goals (xG)

**MoneyPuck Data Downloads**
- URL: https://moneypuck.com/data.htm
- Data: All historical shot data (1.7M+ shots, 2007-present), expected goals models, team/player xG stats
- Format: CSV downloads, free with credit required
- Quality: 9/10 - industry-standard xG model

**Natural Stat Trick**
- URL: https://naturalstattrick.com/
- Data: Individual/on-ice stats, CF%, xGF%, HDCF%, line combos, zone entries/exits, shooting/scoring heatmaps
- Access: Website (scraping needed), some CSV exports
- Quality: 8/10 - excellent advanced metrics

**Evolving Hockey**
- URL: https://evolving-hockey.com/
- Data: Most detailed NHL analytics (GAR, xGAR, standings projections)
- Access: Patreon supporters get most detailed data
- Quality: 9/10 for data, but mostly behind paywall

---

## 4. WEATHER DATA

**Open-Meteo**
- URL: https://open-meteo.com/
- PyPI: `pip install open-meteo`
- Data: Current weather, hourly/daily forecasts, historical weather (back decades), wind speed/direction, precipitation, temperature, humidity, pressure
- Rate limits: Free for non-commercial use. <10,000 calls/day, <5,000/hour, <600/minute. No API key needed
- Requires: Python 3.11+
- Quality: 9/10 - best free weather API, no key needed

**OpenWeatherMap**
- URL: https://openweathermap.org/api
- Python wrapper: `pip install pyowm`
- Data: Current weather, 5-day forecast, historical weather (paid), air quality
- Rate limits: Free = 1,000 calls/day, 60 calls/minute. API key required (free registration)
- Quality: 7/10 - well-known, but historical data costs money

**WeatherAPI.com**
- URL: https://www.weatherapi.com/
- Data: Current, forecast (up to 14 days), historical, astronomical, time zone, sports weather
- Rate limits: Free = 1,000,000 calls/month. API key required
- Quality: 7/10 - generous free tier, includes sports weather endpoint

**Visual Crossing**
- URL: https://www.visualcrossing.com/weather-api/
- Data: Historical weather (back to 1970s), forecasts, astronomy data
- Rate limits: Free = 1,000 records/day. API key required
- Quality: 8/10 - excellent historical data

**Tomorrow.io**
- URL: https://www.tomorrow.io/weather-api/
- Data: Minute-by-minute precipitation, hourly/daily forecasts, air quality
- Rate limits: Free = 500 calls/day
- Quality: 7/10

**Application for predictions**: Temperature, wind speed/direction, precipitation, humidity all affect outdoor sports (NFL, MLB). Key for NFL outdoor games, MLB pitcher performance, home run probability.

---

## 5. SOCIAL MEDIA SENTIMENT

**PRAW (Python Reddit API Wrapper)**
- PyPI: `pip install praw`
- Data: Reddit posts and comments from any subreddit (r/nba, r/nfl, r/baseball, r/hockey, r/sportsbook)
- Rate limits: 60 requests/minute with OAuth2. Requires Reddit API credentials (free)
- Sports-relevant subreddits: r/nba, r/nfl, r/baseball, r/hockey, r/fantasyfootball, r/sportsbook, r/sportsbetting
- Quality: 8/10 - excellent for sentiment signals

**Sentiment Analysis Libraries (pair with PRAW)**

- **VADER**: `pip install vaderSentiment` - Specifically tuned for social media text. Best for Reddit/Twitter-style short text. Returns positive/negative/neutral/compound scores. Quality: 8/10
- **TextBlob**: `pip install textblob` - Polarity (-1 to +1) and subjectivity (0 to 1). Easy to use. Quality: 7/10
- **Flair**: `pip install flair` - Deep learning-based. More accurate but slower. Quality: 8/10
- **transformers (Hugging Face)**: `pip install transformers` - Pre-trained BERT models for sentiment. Highest accuracy but resource-intensive. Quality: 9/10
- **NLTK**: `pip install nltk` - Includes VADER and other NLP tools. Quality: 7/10

**Twitter/X API**
- Current status: Extremely restricted free tier since 2023. Free = 1,500 tweets/month write, minimal read
- Paid: Basic = $200/month for 10,000 tweets/month read
- Quality: 3/10 for free tier - effectively useless for data collection now
- Alternative: Use Nitter scrapers (legally gray) or Reddit instead

**News Sentiment**
- **newsapi**: `pip install newsapi-python` - Search news articles by keyword. Free = 100 requests/day, 1-month old articles only
- **GNews**: `pip install gnews` - Google News scraper, no API key needed
- Quality: 6/10

---

## 6. INJURY / ROSTER DATA

**ESPN Undocumented API**
- Injury endpoint: `sports.core.api.espn.com/v2/sports/{sport}/leagues/{league}/teams/{teamId}/injuries`
- Roster endpoint: `sports.core.api.espn.com/v2/sports/{sport}/leagues/{league}/teams/{teamId}/roster`
- No authentication needed
- Sports: NFL, NBA, MLB, NHL
- Quality: 7/10 - comprehensive but undocumented/unstable

**nfl_data_py**
- Functions: import_rosters(), import_injuries() (through nflverse)
- Data: Weekly injury reports, roster transactions, IR designations
- Quality: 9/10 for NFL

**BALLDONTLIE**
- Injury data available on paid tiers
- Free tier has basic roster data
- Quality: 6/10 free, 8/10 paid

**Scraping Options**
- Pro-Football-Reference / Basketball-Reference / Hockey-Reference / Baseball-Reference
- RotoWire injury pages (rotowire.com)
- CBS Sports injury reports
- Tools: `pip install beautifulsoup4 requests`
- Quality: 7/10 - requires building your own scraper

**sportsdataverse-py**
- Includes roster data from ESPN endpoints
- Quality: 6/10

---

## 7. HISTORICAL DATABASES

### Baseball

**Retrosheet**
- URL: https://www.retrosheet.org/
- Data: Play-by-play data back to 1921, game logs back to 1871, box scores, rosters, transactions, schedules
- Access: Free download of event files, game logs, box scores
- Python: pybaseball includes retrosheet functions; `pip install pyretrosheet` for dedicated package
- Quality: 10/10 - the definitive historical baseball database

**pyretrosheet**
- PyPI: `pip install pyretrosheet`
- Data: Load, analyze, and enrich Retrosheet MLB data
- Quality: 7/10

**baseball.computer**
- URL: https://baseball.computer/
- Data: Free, open-source database of historical baseball statistics and play-by-play
- Quality: 7/10 - modern interface to historical data

### Basketball

**basketball_reference_scraper**
- PyPI: `pip install basketball-reference-scraper`
- GitHub: https://github.com/vishaalagartha/basketball_reference_scraper
- Data: Player stats, team stats, box scores, schedules, standings from Basketball Reference
- Quality: 7/10

**BRScraper**
- GitHub: https://github.com/GabrielPastorello/BRScraper
- Data: Basketball Reference data including NBA, G League, international leagues
- Quality: 7/10 - covers international data too

**basketball_reference_web_scraper**
- PyPI: `pip install basketball_reference_web_scraper`
- Data: Season stats, box scores, schedules from Basketball Reference
- Quality: 6/10

### Hockey

**hockey_scraper** (see Section 1)
- Historical data from 2007-08 season onward

**MoneyPuck CSV Downloads**
- URL: https://moneypuck.com/data.htm
- All shot data from 2007-present
- Quality: 9/10

### Football

**nfl_data_py** (see Section 1)
- Play-by-play back to 1999
- Quality: 10/10

**Pro-Football-Reference**
- Scrape with beautifulsoup4 or use sportsipy
- Data back to 1920s

### Multi-Sport Historical

**Kaggle**
- URL: https://www.kaggle.com/
- Numerous free historical sports datasets
- Search by sport for curated datasets
- Quality: Variable (5-8/10)

---

## 8. REFEREE / UMPIRE DATA

### MLB Umpires

**UmpScorecards**
- URL: https://umpscorecards.com/
- Data: Umpire accuracy rates, consistency scores, favor metrics, historical scorecards (2015+). Built-in API for custom filtering
- Kaggle dataset: https://www.kaggle.com/datasets/mattop/mlb-baseball-umpire-scorecards-2015-2022
- Quality: 9/10 - the best umpire analytics source

**Swish Analytics Umpire Factors**
- URL: https://www.swishanalytics.com/mlb/mlb-umpire-factors
- Data: K Boost, run scoring tendencies per umpire
- Quality: 7/10

**pybaseball**
- Can pull zone data to build your own umpire strike zone models from Statcast
- Quality: 8/10 with custom analysis

### NBA Referees

**RefMetrics**
- URL: https://www.refmetrics.com/
- Data: Foul call distributions, technical foul tendencies, game management metrics
- Covers: NCAA, NBA, WNBA, G League
- Quality: 7/10

**Covers.com**
- URL: https://www.covers.com/sport/basketball/nba/referees
- Data: Referee ATS records, over/under stats, home court tendencies
- Quality: 6/10 - betting-focused

**NBAstuffer**
- URL: https://www.nbastuffer.com/nba-stats/referee/
- Data: Referee stats and tendencies
- Quality: 6/10

**nfl_data_py**
- Includes officials data for NFL games
- Function: import_officials()
- Quality: 8/10 for NFL

### Data Collection Approach
- nfl_data_py provides officials directly
- For NBA/NHL: Build scrapers for Covers.com or use nba_api (some referee data in game logs)
- For MLB: UmpScorecards Kaggle dataset + pybaseball for strike zone analysis

---

## 9. VENUE / TRAVEL DATA

**geopy**
- PyPI: `pip install geopy`
- Data: Geocoding (stadium name -> lat/long), geodesic distance calculations between venues, altitude data (when available from geocoder)
- Uses: Calculate team travel distances, identify back-to-back road game fatigue
- Geocoders: Nominatim (free, OpenStreetMap), Google Maps (key needed), Bing Maps
- Rate limits (Nominatim): 1 request/second
- Quality: 8/10

**Stadium Data Sources**
- Wikipedia: "List of current NFL/NBA/MLB/NHL stadiums" - turf type, dome/outdoor, capacity, elevation
- Pro Football Network: Detailed NFL stadium lists (grass vs turf, dome vs outdoor)
- nfl_data_py: import_team_desc() includes some venue info
- Manual compilation recommended - ~120 venues across 4 sports is manageable

**Key Venue Factors for Predictions**
- Altitude: Denver (5,280 ft) affects ball flight (MLB HRs, NFL passing)
- Dome vs outdoor: Weather immunity, noise levels
- Turf type: Injury rates, speed of play
- Travel distance: Back-to-back games, timezone crossings

**geodistpy**
- PyPI: `pip install geodistpy`
- Data: Same distance calculations as geopy but 78-142x faster
- Quality: 7/10 - speed optimization

---

## 10. ECONOMIC / DEMOGRAPHIC DATA

**Spotrac (Web Scraping)**
- URL: https://www.spotrac.com/
- Data: Team payrolls, salary caps, individual contracts, luxury tax for NFL, NBA, MLB, NHL
- Access: Web scraping (no API)
- Quality: 9/10 - the definitive salary/payroll source

**FanGraphs Payroll (Baseball)**
- URL: https://www.fangraphs.com/roster-resource/breakdowns/payroll
- Data: MLB team payroll breakdowns by contract type
- Quality: 8/10

**pybaseball**
- Can pull team-level financial data from FanGraphs/Baseball-Reference
- Quality: 7/10

**GitHub Salary Datasets**
- Various CSV datasets with historical NBA/NFL/MLB salary data
- Example: https://github.com/erikgregorywebb/datasets/blob/master/nba-salaries.csv
- Quality: 5/10 - may not be current

**Market Size / Demographics**
- US Census API: Free, Python wrapper `pip install census`
- Metropolitan Statistical Area populations correlate with ticket revenue and team spending
- Quality: 6/10 for prediction value

---

## 11. BIOMETRIC / WEARABLE DATA

**Public Availability: EXTREMELY LIMITED**

Biometric/wearable data (heart rate, sleep, recovery, player load) is almost entirely proprietary:
- WHOOP: Used by NFL/NBA players - data owned by players, not public
- Catapult: GPS tracking and load monitoring - team-internal only
- NBA prohibits wearables during games
- NFL allows WHOOP but data stays private

**Proxy Metrics Available for Free**
- Rest days between games: Calculate from schedule data (nfl_data_py, nba_api)
- Travel distance: Calculate with geopy from schedule
- Minutes played trends: nba_api, hockey_scraper
- Pitch counts / innings pitched: pybaseball
- Snap counts: nfl_data_py
- Time on ice: hockey_scraper
- Back-to-back game flags: Derivable from schedule data

**Quality: 2/10** for actual biometric data (not publicly available)
**Quality: 7/10** for fatigue proxies you can derive from other sources

---

## 12. VIDEO / COMPUTER VISION

**SportsLabKit**
- GitHub: https://github.com/AtomScott/SportsLabKit
- Data/Tools: Convert game footage to CSV tracking data. Includes SORT, DeepSORT, ByteTrack, TeamTrack for object tracking. YOLOv8 integration
- Install: `pip install sportslabkit`
- Quality: 7/10 - academic-grade toolkit

**Roboflow Sports**
- GitHub: https://github.com/roboflow/sports
- Data/Tools: Player detection/tracking, re-identification, camera calibration, speed/distance extraction
- Install: `pip install supervision` (Roboflow's core library)
- Quality: 7/10

**OpenCV**
- PyPI: `pip install opencv-python`
- Data/Tools: Image processing, video frame extraction, color-based team identification
- Quality: 9/10 as a foundation library

**YOLOv8 (Ultralytics)**
- PyPI: `pip install ultralytics`
- Data/Tools: Real-time object detection for players, ball, officials
- Pre-trained models available, fine-tunable for sports
- Quality: 9/10

**AlphaPose**
- GitHub: https://github.com/MVIG-SJTU/AlphaPose
- Data/Tools: Multi-person pose estimation from video. Body, hand, facial keypoints
- Quality: 7/10

**Annotation Tools**
- CVAT: Free, open-source video annotation
- LabelMe: Free polygon annotation
- Roboflow: Free tier for dataset creation

**Practical Notes**: Video analysis requires significant compute resources and game footage access. Most useful for specific research projects rather than daily prediction pipelines.

---

## 13. SCHEDULE STRENGTH

**Derivable from existing packages - no separate source needed**

**NFL**
- nfl_data_py: Full schedules + team records = SOS calculation
- Functions: import_schedules(), import_pbp_data()
- nflverse provides pre-calculated Elo ratings
- Quality: 9/10

**NBA**
- nba_api: LeagueSchedule, TeamSchedule endpoints
- Back-to-back detection, road trip length, rest advantage
- Quality: 8/10

**MLB**
- pybaseball: schedule_and_record()
- MLB-StatsAPI: statsapi.schedule()
- Quality: 8/10

**NHL**
- nhl-api-py: Schedule endpoints
- hockey_scraper: scrape_schedule()
- Quality: 8/10

**Calculation approach**: Pull all game results + schedules, compute opponent win% weighted by recency. Available data from all the core packages above makes this straightforward.

---

## 14. DRAFT / PROSPECT DATA

**NFL**
- nfl_data_py: import_draft_picks(), import_combine_data(), import_draft_values()
- Data: Draft pick history, combine measurements (40-yard, bench, vertical, etc.), draft pick trade values
- Quality: 9/10

**NBA**
- nba_api: DraftCombineStats, DraftHistory endpoints
- Data: Combine measurements, draft results
- GitHub: https://github.com/JasonG7234/NBA-Draft-Model (scrapes realgm, basketballreference, barttorvik, 247sports)
- Quality: 7/10

**MLB**
- pybaseball: amateur_draft(), amateur_draft_by_team()
- MLB-StatsAPI: Draft endpoint
- Data: Draft pick history, signing bonuses
- MiLB stats available through Baseball Reference
- Quality: 7/10

**NHL**
- nhl-api-py: Prospects endpoint
- Data: Draft history, prospect info
- EliteProspects (eliteprospects.com) for international prospects - web scraping needed
- Quality: 6/10

**Multi-Sport (R package)**
- ProSportsDraftData: R package with scouting data for NHL, MLB, NFL
- Can be used via rpy2 Python-R bridge if needed
- Quality: 7/10

---

## 15. COACHING DATA

**No dedicated free package exists - derive from play-by-play data**

**NFL Coaching Tendencies**
- nfl_data_py play-by-play includes: play_type, formation, personnel, shotgun/no_huddle flags, run/pass ratios by situation (down, distance, score, quarter)
- Build models for: run/pass tendencies, 4th down aggressiveness, blitz rates, timeout usage
- Quality: 9/10 - rich situational data for tendency analysis

**NBA Coaching Tendencies**
- nba_api: TeamDashboard endpoints for pace, offensive/defensive rating by lineup
- Lineup data for rotation patterns
- Timeout and challenge usage from play-by-play
- Quality: 7/10

**MLB Manager Tendencies**
- pybaseball: Pinch-hit rates, bullpen usage patterns, sacrifice bunt frequency, lineup construction
- Derivable from play-by-play and lineup data
- Quality: 7/10

**NHL Coaching Tendencies**
- hockey_scraper shift data: Line matching, deployment patterns, goalie pull timing
- MoneyPuck/NaturalStatTrick: Zone entry/exit strategies by team
- Quality: 7/10

**Key coaching metrics to derive:**
- 4th down go-for-it rate (NFL)
- Pace of play (NBA)
- Bullpen leverage usage (MLB)
- Empty net timing (NHL)
- Timeout patterns (all sports)
- Challenge/review usage (all sports)

---

## SUMMARY: TOP RECOMMENDED STACK

### Core Data Packages (install these first)
```
pip install nba_api
pip install pybaseball
pip install nfl_data_py
pip install hockey_scraper
pip install nhl-api-py
pip install MLB-StatsAPI
```

### Odds & Betting
```
pip install requests  # for The Odds API calls
```
Sign up at https://the-odds-api.com/ for free API key (500 req/month)

### Weather
```
pip install open-meteo  # no API key needed
```

### Sentiment
```
pip install praw vaderSentiment
```

### Venue/Travel
```
pip install geopy
```

### Supplementary
```
pip install sportsdataverse  # ESPN data
pip install beautifulsoup4   # web scraping fallback
pip install espn-api         # fantasy data
```

### Analytics & ML Support
```
pip install pandas numpy scikit-learn xgboost
pip install matplotlib seaborn  # visualization
```

### Video Analysis (optional, heavy)
```
pip install opencv-python ultralytics supervision
```

---

## DATA QUALITY RANKINGS (for prediction model input)

| Source | Sport | Quality | Reliability | Update Frequency |
|--------|-------|---------|-------------|------------------|
| nfl_data_py | NFL | 10/10 | Excellent | Weekly during season |
| pybaseball | MLB | 10/10 | Excellent | Daily during season |
| nba_api | NBA | 9/10 | Good* | Near real-time |
| hockey_scraper | NHL | 8/10 | Good | Daily during season |
| MoneyPuck | NHL | 9/10 | Excellent | Daily during season |
| The Odds API | All | 8/10 | Excellent | Real-time |
| Open-Meteo | Weather | 9/10 | Excellent | Hourly |
| UmpScorecards | MLB | 9/10 | Good | Daily during season |
| ESPN API | All | 6/10 | Unstable | Real-time |
| PRAW | Sentiment | 8/10 | Good | Real-time |

*nba_api can occasionally get rate-limited or blocked by NBA.com

---

## LEGAL NOTES

- Most scraping-based tools operate in a gray area. Sports-Reference has blocked scrapers before
- Official APIs (MLB Stats API, NHL API) are generally safe to use
- nba_api accesses undocumented NBA.com endpoints - not officially sanctioned
- Always respect rate limits and add delays between requests
- MoneyPuck requires credit/attribution when using their data
- Retrosheet data is free to use for any purpose
- The Odds API historical data requires paid plans
- Open-Meteo is free for non-commercial use only
