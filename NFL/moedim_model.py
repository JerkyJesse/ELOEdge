"""
Moedim System Model -- 36th mega-ensemble model (NFL)
Complete implementation of The Moedim System by S.W. Phipps (April 2026).

Implements all 6 fractal levels, elemental seasons, three zones, shock points,
hierarchy suppression, planetary quality, zone harmony, native scale weighting,
named patterns, and expanded feature set (~40+ features per team).

Uses 28 prophetic cycle numbers tested at 4 scales against team founding
dates to calculate convergence density -- the primary predictive signal.
"""
import logging
from datetime import date, datetime
from collections import defaultdict

# ---------------------------------------------------------------------------
# The 28 prophetic cycle numbers organized by planet
# ---------------------------------------------------------------------------
CYCLES = {
    "saturn":  [2520, 777, 400, 120],
    "jupiter": [1335, 490, 70, 1000],
    "mars":    [390, 40, 430, 21],
    "mercury": [483, 1290, 2300, 360],
    "venus":   [666, 1260, 144, 84],
    "sun":     [153, 28, 50, 365],
    "moon":    [19, 42, 75, 273],
}
ALL_CYCLES = [c for group in CYCLES.values() for c in group]

# Reverse lookup: cycle number -> planet name
CYCLE_TO_PLANET = {}
for planet, nums in CYCLES.items():
    for n in nums:
        CYCLE_TO_PLANET[n] = planet

# ---------------------------------------------------------------------------
# Zone scoring for Y-positions (competitive context)
# ---------------------------------------------------------------------------
ZONE_SCORE = {
    1: 1.0,   # Y1 SEED - fresh energy, breakthroughs
    2: 0.0,   # Y2 ECHO - mirrors Y1, neutral
    3: 1.0,   # Y3 LABOR - competitive grinding
    4: -1.0,  # Y4 TRIAL - vulnerability, collapse under pressure
    5: 2.0,   # Y5 HARVEST - strongest for victory
    6: -2.0,  # Y6 WINTER - maximum compression
    7: 0.0,   # Y7 SABBATH - completing, winding down
}

# ---------------------------------------------------------------------------
# Three Zones mapping
# ---------------------------------------------------------------------------
ZONE_MAP = {
    1: "creating",   # Y1-Y2: New ideas, fresh potential
    2: "creating",
    3: "sustaining",  # Y3-Y4: Building through resistance
    4: "sustaining",
    5: "completing",  # Y5-Y7: Gathering fruit, compression, release
    6: "completing",
    7: "completing",
}

# Numeric encoding for zones
ZONE_ENCODE = {"creating": 1.0, "sustaining": 0.0, "completing": -1.0}

# ---------------------------------------------------------------------------
# Scale weights per Part 4 Note 4
# ---------------------------------------------------------------------------
SCALE_WEIGHTS = {"years": 4, "months": 3, "weeks": 2, "days": 1}

# ---------------------------------------------------------------------------
# Planetary Quality per cycle (Part 4 Note 5)
# Each planet has a quality that affects how its cycles interact
# ---------------------------------------------------------------------------
PLANETARY_QUALITY = {
    "sun":     1.0,   # Illuminating -- sovereignty, clarity, new beginning
    "moon":    0.0,   # Rhythmic -- reflection, reception, neutral
    "mars":   -1.0,   # Warrior -- conflict, burden, formation through friction
    "mercury": 0.0,   # Testing -- precision, the crucible (mixed)
    "jupiter": 1.0,   # Expansive -- covenant, abundance, institutional force
    "venus":  -1.0,   # Compressive -- beauty, cost, the price of love
    "saturn":  1.0,   # Completing -- law, limitation, the teacher
}

# Each Y-position is governed by a planet
Y_PLANET = {
    1: "sun", 2: "moon", 3: "mars", 4: "mercury",
    5: "jupiter", 6: "venus", 7: "saturn",
}

# ---------------------------------------------------------------------------
# Native Scale Weighting (Part 4 Note 7)
# When cycle fires at its native scale, give 1.5x weight
# When cycle fires at non-native scale, it's a fractal echo (1.0x)
# ---------------------------------------------------------------------------
NATIVE_SCALES = {
    # Confirmed explicit in scripture at this scale:
    19: "years",     # Metonic cycle - astronomical 19-year return
    28: "years",     # Birkat Hachama - 28-year solar cycle
    42: "months",    # Rev 11:2, 13:5 - "forty-two months" explicit
    273: "days",     # Numbers 3:46 count + gestation tradition (3x91 days)
    21: "days",      # Daniel 10:2-3 - "three full weeks" = 21 days explicit
    390: "days",     # Ezekiel 4:5 - "390 days" explicit
    1260: "days",    # Rev 11:3, 12:6 - "1,260 days" explicit
    1335: "days",    # Daniel 12:12 - "1,335 days" explicit
    1290: "days",    # Daniel 12:11 - "1,290 days" explicit
    2520: "days",    # Calculated: 7 x 360 prophetic days
    2300: "days",    # Daniel 8:14 - "2,300 evenings and mornings"
    70: "years",     # Jeremiah 25:11 - "seventy years" explicit
    490: "years",    # Daniel 9:24 - 70 weeks of years = 490 years
    400: "years",    # Genesis 15:13 - "four hundred years" explicit
    483: "years",    # Daniel 9:25 - 69 x 7 = 483 years
    120: "years",    # Genesis 6:3 - "120 years" explicit
    1000: "years",   # Revelation 20:4 - "a thousand years" explicit
    430: "years",    # Exodus 12:40 - "430 years" explicit
    # Dual-native (both scales receive full native bonus):
    50: "both",      # Jubilee (years, Lev 25:10) AND Pentecost (days, Lev 23:16)
    40: "both",      # Wilderness (years, Num 14:33) AND Flood/Sinai (days, Gen 7:12/Ex 34:28)
    # These operate at native years scale:
    777: "years",    # Genesis 5:31 - Lamech's lifespan
    365: "years",    # Genesis 5:23 - Enoch's lifespan
    75: "years",     # Genesis 12:4 - Abraham left Haran at 75
    84: "years",     # Luke 2:37 - Anna the prophetess
}

# ---------------------------------------------------------------------------
# Scriptural Strength Tiers (Part 4 Note 8)
# Weight cycles by the strength of their scriptural attestation
# ---------------------------------------------------------------------------
SCRIPTURAL_STRENGTH = {
    # Tier 1: Appears EXPLICITLY as a prophetic time period in scripture (strongest)
    1260: 1.0, 1335: 1.0, 1290: 1.0, 2300: 1.0,
    390: 1.0, 40: 1.0, 430: 1.0, 21: 1.0, 42: 1.0,
    70: 1.0, 400: 1.0, 120: 1.0, 1000: 1.0, 50: 1.0,
    # Tier 2: Calculated from explicit scripture (strong)
    2520: 0.9, 490: 0.9, 483: 0.9, 360: 0.9, 75: 0.9,
    # Tier 3: Explicit in scripture but NOT as a time period (moderate)
    666: 0.8, 153: 0.8, 144: 0.8, 273: 0.8,
    777: 0.8, 365: 0.8, 84: 0.8,
    # Tier 4: Astronomical/traditional, weak/no scriptural attestation
    19: 0.7, 28: 0.7,
}


def _get_native_bonus(cycle_num, scale):
    """When cycle fires at its native scale, give 1.5x weight."""
    native = NATIVE_SCALES.get(cycle_num)
    if native is None:
        return 1.0
    if native == "both":
        # Dual-native: days AND years both get full bonus
        if scale in ("days", "years"):
            return 1.5
        return 1.0
    if native == scale:
        return 1.5
    return 1.0


def _calc_zone_harmony(cycle_planet, entity_y_position):
    """Zone Harmony Scoring (Part 4 Note 4).
    How well the cycle's planetary zone matches entity's Y-position zone.
    """
    entity_planet = Y_PLANET.get(entity_y_position, "moon")
    cycle_quality = PLANETARY_QUALITY.get(cycle_planet, 0.0)
    entity_quality = PLANETARY_QUALITY.get(entity_planet, 0.0)

    # Same zone = resonance (amplified)
    if cycle_quality == entity_quality:
        return 1.5
    # Opposite zones = friction
    if cycle_quality * entity_quality < 0:
        return 0.5
    # Neutral interaction
    return 1.0


# ---------------------------------------------------------------------------
# NFL team founding dates (franchise establishment year)
# Uses the earliest continuous franchise date for each current team
# ---------------------------------------------------------------------------
FOUNDING_DATES = {
    "ARI": "1898-01-01",   # Arizona Cardinals (Morgan Athletic Club, Chicago 1898)
    "ATL": "1965-06-30",   # Atlanta Falcons (expansion 1965)
    "BAL": "1996-02-09",   # Baltimore Ravens (expansion 1996)
    "BUF": "1960-10-28",   # Buffalo Bills (AFL charter 1960)
    "CAR": "1993-10-26",   # Carolina Panthers (expansion 1993)
    "CHI": "1920-09-17",   # Chicago Bears (Decatur Staleys 1920, charter NFL)
    "CIN": "1968-05-24",   # Cincinnati Bengals (AFL expansion 1968)
    "CLE": "1999-03-23",   # Cleveland Browns (expansion 1999)
    "DAL": "1960-01-28",   # Dallas Cowboys (NFL expansion 1960)
    "DEN": "1960-08-14",   # Denver Broncos (AFL charter 1960)
    "DET": "1930-07-12",   # Detroit Lions (Portsmouth Spartans 1930)
    "GB":  "1919-08-11",   # Green Bay Packers (1919, charter NFL 1921)
    "HOU": "2002-01-01",   # Houston Texans (expansion 2002)
    "IND": "1953-01-23",   # Indianapolis Colts (Baltimore Colts 1953)
    "JAX": "1995-01-01",   # Jacksonville Jaguars (expansion 1995)
    "KC":  "1960-08-14",   # Kansas City Chiefs (Dallas Texans, AFL 1960)
    "LV":  "1960-01-30",   # Las Vegas Raiders (Oakland Raiders, AFL 1960)
    "LAC": "1960-08-14",   # LA Chargers (LA Chargers, AFL 1960)
    "LAR": "1936-02-13",   # LA Rams (Cleveland Rams 1936)
    "MIA": "1966-08-16",   # Miami Dolphins (AFL expansion 1966)
    "MIN": "1961-01-28",   # Minnesota Vikings (NFL expansion 1961)
    "NE":  "1960-11-16",   # New England Patriots (Boston Patriots, AFL 1960)
    "NO":  "1966-11-01",   # New Orleans Saints (NFL expansion 1966)
    "NYG": "1925-08-01",   # New York Giants (1925)
    "NYJ": "1960-08-14",   # New York Jets (Titans, AFL 1960)
    "PHI": "1933-07-08",   # Philadelphia Eagles (1933)
    "PIT": "1933-07-08",   # Pittsburgh Steelers (Pirates 1933)
    "SF":  "1946-06-04",   # San Francisco 49ers (AAFC 1946)
    "SEA": "1976-06-04",   # Seattle Seahawks (NFL expansion 1976)
    "TB":  "1976-04-24",   # Tampa Bay Buccaneers (NFL expansion 1976)
    "TEN": "1960-08-14",   # Tennessee Titans (Houston Oilers, AFL 1960)
    "WSH": "1932-07-09",   # Washington Commanders (Boston Braves 1932)
}


# ---------------------------------------------------------------------------
# Default features dict (returned when founding date is missing or invalid)
# ---------------------------------------------------------------------------
def _default_features():
    """Return zeroed feature dict with all keys present."""
    return {
        # Convergence / hit counts
        "convergence": 0.0,
        "tight_hits": 0.0,
        "total_hits": 0.0,
        "weighted_score": 0.0,
        # 6 fractal Y-positions
        "jubilee_era": 1.0,
        "jubilee_y": 1.0,
        "shemitah_number": 1.0,
        "life_y": 4.0,
        "month_y": 4.0,
        "week_y": 4.0,
        "day_y": 4.0,
        # Zone scores
        "zone_score": 0.0,
        "life_zone": 0.0,
        # Elemental season
        "elemental_quarter": 1.0,
        "element_score": 0.0,
        # Shock points
        "shock_point_1_proximity": 0.0,
        "shock_point_2_proximity": 0.0,
        "near_shock_1": 0.0,
        "near_shock_2": 0.0,
        # Hierarchy
        "hierarchy_factor": 1.0,
        # Zone harmony & planetary
        "zone_harmony_avg": 1.0,
        "native_scale_bonus_total": 0.0,
        "planetary_quality_sum": 0.0,
        # Named patterns
        "in_harvest": 0.0,
        "in_winter": 0.0,
        "in_trial": 0.0,
        "double_harvest": 0.0,
        "containment": 0.0,
        "triple_trial": 0.0,
        "trial_collapse": 0.0,
        "sabbath_arc": 0.0,
        "winter_friction": 0.0,
        # Double positions
        "double_position_life_month": 0.0,
        "double_position_month_week": 0.0,
        "double_position_count": 0.0,
        # Weighted convergence (full scoring formula)
        "weighted_convergence": 0.0,
        # Scripture quality (avg scriptural strength of hitting cycles)
        "scripture_quality": 0.8,
    }


class TeamMoedim:
    """Tracks Moedim cycle state for a single team.

    Implements the complete 5-part Moedim System:
    - 6 fractal levels (Jubilee, Shemitah, Annual, Month, Week, Day)
    - Elemental seasons (Fire, Water, Earth, Air)
    - Three zones (Creating, Sustaining, Completing)
    - Shock point proximity
    - Hierarchy suppression rule
    - Planetary quality per cycle
    - Zone harmony scoring
    - Native scale weighting
    - All named patterns
    """

    def __init__(self, founding_date_str, tolerance=3):
        self.founding = None
        self.tolerance = tolerance
        if founding_date_str:
            try:
                self.founding = datetime.strptime(founding_date_str, "%Y-%m-%d").date()
            except (ValueError, TypeError):
                try:
                    self.founding = datetime.strptime(
                        str(founding_date_str)[:10], "%Y-%m-%d"
                    ).date()
                except Exception:
                    self.founding = None

    # ------------------------------------------------------------------
    # Interval calculation
    # ------------------------------------------------------------------
    def _calc_intervals(self, game_date):
        """Calculate all time intervals from founding to game date."""
        if not self.founding or not game_date:
            return None

        if isinstance(game_date, str):
            game_date = datetime.strptime(game_date[:10], "%Y-%m-%d").date()
        elif isinstance(game_date, datetime):
            game_date = game_date.date()

        if game_date <= self.founding:
            return None

        total_days = (game_date - self.founding).days
        total_weeks = total_days // 7
        total_months = (
            (game_date.year - self.founding.year) * 12
            + (game_date.month - self.founding.month)
            + 1
        )
        completed_years = game_date.year - self.founding.year
        if (game_date.month, game_date.day) < (
            self.founding.month,
            self.founding.day,
        ):
            completed_years -= 1

        return {
            "days": total_days,
            "weeks": total_weeks,
            "months": max(1, total_months),
            "years": max(0, completed_years),
            "game_date": game_date,
        }

    # ------------------------------------------------------------------
    # Level 1-2: Jubilee / Shemitah
    # ------------------------------------------------------------------
    def _calc_jubilee(self, completed_years):
        """Calculate Jubilee life arc and Shemitah number."""
        jubilee_cycle = (completed_years // 7) + 1   # Which 7-year cycle (C1, C2, C3...)
        jubilee_era = ((jubilee_cycle - 1) // 7) + 1  # Which 49-year Jubilee era
        shemitah_number = jubilee_cycle               # The Nth 7-year cycle of life
        jubilee_y = ((jubilee_cycle - 1) % 7) + 1    # Y-position at Jubilee scale
        return jubilee_cycle, jubilee_era, shemitah_number, jubilee_y

    # ------------------------------------------------------------------
    # Levels 3-6: Y-positions at all 4 measurable scales
    # ------------------------------------------------------------------
    def _calc_y_positions(self, intervals):
        """Calculate Y-positions at all 6 fractal levels."""
        completed_years = intervals["years"]

        # Level 1-2: Jubilee
        jubilee_cycle, jubilee_era, shemitah_number, jubilee_y = self._calc_jubilee(
            completed_years
        )

        # Level 3: Y-Position (Annual)
        life_y = (completed_years % 7) + 1

        # Level 4: Month Y-Position
        month_y = ((intervals["months"] - 1) % 7) + 1

        # Level 5: Week Y-Position
        week_y = (intervals["weeks"] % 7) + 1

        # Level 6: Day Y-Position
        day_y = ((intervals["days"] - 1) % 7) + 1

        return {
            "jubilee_cycle": jubilee_cycle,
            "jubilee_era": jubilee_era,
            "shemitah_number": shemitah_number,
            "jubilee_y": jubilee_y,
            "life_y": life_y,
            "month_y": month_y,
            "week_y": week_y,
            "day_y": day_y,
        }

    # ------------------------------------------------------------------
    # Elemental Seasons (Q1-Q4)
    # ------------------------------------------------------------------
    def _calc_elemental_season(self, game_date):
        """Calculate elemental quarter based on day within the current Y-year.

        The Y-year starts on the team's founding anniversary date, not Jan 1.
        """
        founding = self.founding

        # Find most recent birthday anniversary
        try:
            last_anniversary = date(game_date.year, founding.month, founding.day)
        except ValueError:
            # Handle Feb 29 founding dates
            last_anniversary = date(game_date.year, founding.month, min(founding.day, 28))

        if last_anniversary > game_date:
            try:
                last_anniversary = date(
                    game_date.year - 1, founding.month, founding.day
                )
            except ValueError:
                last_anniversary = date(
                    game_date.year - 1, founding.month, min(founding.day, 28)
                )

        day_of_y_year = (game_date - last_anniversary).days + 1

        # Q1 Fire (1-91), Q2 Water (92-182), Q3 Earth (183-273), Q4 Air (274-365)
        if day_of_y_year <= 91:
            quarter = 1
            element_score = 1.0    # Fire - hot, outward, aggressive
        elif day_of_y_year <= 182:
            quarter = 2
            element_score = -0.5   # Water - descent inward, hidden forces
        elif day_of_y_year <= 273:
            quarter = 3
            element_score = -1.0   # Earth - cold test, endurance
        else:
            quarter = 4
            element_score = 0.5    # Air - movement returns, liberation

        return quarter, element_score

    # ------------------------------------------------------------------
    # Shock Point Proximity
    # ------------------------------------------------------------------
    def _calc_shock_points(self, life_y, quarter, jubilee_y):
        """Calculate proximity to shock points at the annual level.

        Shock Point 1: Y4 -> Y5 (external help needed)
        Shock Point 2: Y7 -> Y1 (inner will needed)
        """
        # At shock point (late quarter of the transition year)
        at_shock_1 = 1.0 if (life_y == 4 and quarter >= 3) else 0.0
        at_shock_2 = 1.0 if (life_y == 7 and quarter >= 3) else 0.0

        # Near shock point (anywhere in the transition year)
        near_shock_1 = 1.0 if life_y == 4 else 0.0
        near_shock_2 = 1.0 if life_y == 7 else 0.0

        return at_shock_1, at_shock_2, near_shock_1, near_shock_2

    # ------------------------------------------------------------------
    # Hierarchy Suppression Rule
    # ------------------------------------------------------------------
    def _calc_hierarchy_factor(self, jubilee_y, life_y):
        """Higher levels suppress or amplify lower levels.

        A positive Y-position inside a negative Jubilee cycle is suppressed.
        A negative Y-position inside a positive Jubilee cycle is softened.
        """
        jubilee_zone_score = ZONE_SCORE.get(jubilee_y, 0.0)
        zone_score = ZONE_SCORE.get(life_y, 0.0)

        if jubilee_zone_score > 0 and zone_score < 0:
            return 0.5   # Negative softened by positive Jubilee
        if jubilee_zone_score < 0 and zone_score > 0:
            return 0.5   # Positive suppressed by negative Jubilee
        if jubilee_zone_score > 0 and zone_score > 0:
            return 1.5   # Double positive = amplified
        if jubilee_zone_score < 0 and zone_score < 0:
            return 1.5   # Double negative = amplified (heavy)
        return 1.0       # Neutral

    # ------------------------------------------------------------------
    # Cycle Testing with full scoring formula
    # ------------------------------------------------------------------
    def _test_cycles(self, intervals, life_y):
        """Test all 28 cycles at all 4 scales.

        Returns hit counts and detailed scoring with:
        - proximity_score x scale_weight x zone_harmony x native_bonus
        """
        exact_hits = 0       # +/-0
        tight_hits = 0       # +/-1
        total_hits = 0       # +/-tolerance
        weighted_score = 0.0

        # Detailed tracking for zone harmony and native scale
        zone_harmony_sum = 0.0
        zone_harmony_count = 0
        native_bonus_total = 0.0
        planetary_quality_sum = 0.0
        weighted_convergence = 0.0
        scriptural_strength_sum = 0.0
        scriptural_strength_count = 0

        for cycle_num in ALL_CYCLES:
            cycle_planet = CYCLE_TO_PLANET.get(cycle_num, "moon")

            for scale_name, scale_weight in SCALE_WEIGHTS.items():
                value = intervals.get(scale_name, 0)
                if value <= 0 or cycle_num <= 0:
                    continue
                # Skip if cycle exceeds entity lifespan at this scale (Part 4 Note 1)
                if value < cycle_num:
                    continue

                remainder = value % cycle_num
                # Check proximity to nearest multiple
                proximity = min(remainder, cycle_num - remainder)

                if proximity <= self.tolerance:
                    # Calculate proximity score
                    if proximity == 0:
                        proximity_score = 1.0
                        exact_hits += 1
                        tight_hits += 1
                        total_hits += 1
                    elif proximity <= 1:
                        proximity_score = 0.8
                        tight_hits += 1
                        total_hits += 1
                    else:
                        proximity_score = 0.5
                        total_hits += 1

                    # Zone harmony for this hit
                    zh = _calc_zone_harmony(cycle_planet, life_y)
                    zone_harmony_sum += zh
                    zone_harmony_count += 1

                    # Native scale bonus
                    nb = _get_native_bonus(cycle_num, scale_name)
                    native_bonus_total += nb

                    # Planetary quality of hitting cycle
                    pq = PLANETARY_QUALITY.get(cycle_planet, 0.0)
                    planetary_quality_sum += pq

                    # Scriptural strength of this cycle
                    ss = SCRIPTURAL_STRENGTH.get(cycle_num, 0.8)
                    scriptural_strength_sum += ss
                    scriptural_strength_count += 1

                    # Basic weighted score (backward compatible)
                    weighted_score += scale_weight * proximity_score

                    # Full scoring formula:
                    # proximity_score x scale_weight x zone_harmony x native_bonus x scriptural_strength
                    weighted_convergence += (
                        proximity_score * scale_weight * zh * nb * ss
                    )

        zone_harmony_avg = (
            zone_harmony_sum / zone_harmony_count if zone_harmony_count > 0 else 1.0
        )

        scripture_quality = (
            scriptural_strength_sum / scriptural_strength_count
            if scriptural_strength_count > 0 else 0.8
        )

        return {
            "exact_hits": exact_hits,
            "tight_hits": tight_hits,
            "total_hits": total_hits,
            "weighted_score": weighted_score,
            "zone_harmony_avg": zone_harmony_avg,
            "native_scale_bonus_total": native_bonus_total,
            "planetary_quality_sum": planetary_quality_sum,
            "weighted_convergence": weighted_convergence,
            "scripture_quality": scripture_quality,
        }

    # ------------------------------------------------------------------
    # Named Patterns (Part 3 Chapter 5)
    # ------------------------------------------------------------------
    def _calc_named_patterns(self, ypos, total_hits, jubilee_y):
        """Calculate all named patterns from Part 3 Chapter 5."""
        life_y = ypos["life_y"]
        month_y = ypos["month_y"]
        week_y = ypos["week_y"]
        day_y = ypos["day_y"]

        # Life + Month HARVEST Alignment (strongest dual-positive)
        double_harvest = 1.0 if (life_y == 5 and month_y == 5) else 0.0

        # Life + Month WINTER Containment (cannot win)
        containment = 1.0 if (life_y == 6 and month_y == 6) else 0.0

        # Y4 TRIAL + Strong Hits = Leads Then Collapses
        trial_collapse = 1.0 if (life_y == 4 and total_hits >= 3) else 0.0

        # Triple Sub-Level TRIAL (catastrophic)
        triple_trial = 1.0 if (month_y == 4 and week_y == 4 and day_y == 4) else 0.0

        # SABBATH Completing Arc (near-victory, runner-up)
        sabbath_arc = 1.0 if (life_y == 7 or jubilee_y == 7) else 0.0

        # Month + Week WINTER without Life WINTER = friction, NOT containment
        winter_friction = (
            1.0
            if (month_y == 6 and week_y == 6) and life_y != 6
            else 0.0
        )

        # Double position: same Y at two scales = amplified character
        double_position_life_month = 1.0 if life_y == month_y else 0.0
        double_position_month_week = 1.0 if month_y == week_y else 0.0
        double_position_count = double_position_life_month + double_position_month_week

        # Basic indicators
        in_harvest = 1.0 if life_y == 5 else 0.0
        in_winter = 1.0 if life_y == 6 else 0.0
        in_trial = 1.0 if life_y == 4 else 0.0

        return {
            "double_harvest": double_harvest,
            "containment": containment,
            "trial_collapse": trial_collapse,
            "triple_trial": triple_trial,
            "sabbath_arc": sabbath_arc,
            "winter_friction": winter_friction,
            "double_position_life_month": double_position_life_month,
            "double_position_month_week": double_position_month_week,
            "double_position_count": double_position_count,
            "in_harvest": in_harvest,
            "in_winter": in_winter,
            "in_trial": in_trial,
        }

    # ------------------------------------------------------------------
    # Composite zone score
    # ------------------------------------------------------------------
    def _calc_zone_score(self, life_y, patterns):
        """Calculate composite zone score incorporating patterns."""
        base_score = ZONE_SCORE.get(life_y, 0.0)
        composite = (
            base_score
            + patterns["double_harvest"] * 3.0
            - patterns["containment"] * 5.0
            - patterns["triple_trial"] * 5.0
        )
        return composite

    # ------------------------------------------------------------------
    # Main feature extraction
    # ------------------------------------------------------------------
    def get_features(self, game_date):
        """Get all Moedim features for this team on the given game date.

        Returns ~35 features per team covering all 10 system elements.
        """
        intervals = self._calc_intervals(game_date)
        if intervals is None:
            return _default_features()

        game_date_obj = intervals["game_date"]

        # 6 fractal Y-positions
        ypos = self._calc_y_positions(intervals)

        # Elemental season
        quarter, element_score = self._calc_elemental_season(game_date_obj)

        # Shock points
        at_shock_1, at_shock_2, near_shock_1, near_shock_2 = self._calc_shock_points(
            ypos["life_y"], quarter, ypos["jubilee_y"]
        )

        # Hierarchy suppression
        hierarchy_factor = self._calc_hierarchy_factor(
            ypos["jubilee_y"], ypos["life_y"]
        )

        # Cycle testing with full scoring
        cycle_results = self._test_cycles(intervals, ypos["life_y"])

        # Named patterns
        patterns = self._calc_named_patterns(
            ypos, cycle_results["total_hits"], ypos["jubilee_y"]
        )

        # Composite zone score
        zone_score = self._calc_zone_score(ypos["life_y"], patterns)

        # Three zones (encoded)
        life_zone_name = ZONE_MAP.get(ypos["life_y"], "sustaining")
        life_zone = ZONE_ENCODE.get(life_zone_name, 0.0)

        # Assemble all features
        return {
            # Convergence / hit counts
            "convergence": float(cycle_results["exact_hits"]),
            "tight_hits": float(cycle_results["tight_hits"]),
            "total_hits": float(cycle_results["total_hits"]),
            "weighted_score": float(cycle_results["weighted_score"]),
            # 6 fractal Y-positions
            "jubilee_era": float(ypos["jubilee_era"]),
            "jubilee_y": float(ypos["jubilee_y"]),
            "shemitah_number": float(ypos["shemitah_number"]),
            "life_y": float(ypos["life_y"]),
            "month_y": float(ypos["month_y"]),
            "week_y": float(ypos["week_y"]),
            "day_y": float(ypos["day_y"]),
            # Zone scores
            "zone_score": float(zone_score),
            "life_zone": float(life_zone),
            # Elemental season
            "elemental_quarter": float(quarter),
            "element_score": float(element_score),
            # Shock points
            "shock_point_1_proximity": float(at_shock_1),
            "shock_point_2_proximity": float(at_shock_2),
            "near_shock_1": float(near_shock_1),
            "near_shock_2": float(near_shock_2),
            # Hierarchy
            "hierarchy_factor": float(hierarchy_factor),
            # Zone harmony & planetary
            "zone_harmony_avg": float(cycle_results["zone_harmony_avg"]),
            "native_scale_bonus_total": float(cycle_results["native_scale_bonus_total"]),
            "planetary_quality_sum": float(cycle_results["planetary_quality_sum"]),
            # Named patterns
            "in_harvest": float(patterns["in_harvest"]),
            "in_winter": float(patterns["in_winter"]),
            "in_trial": float(patterns["in_trial"]),
            "double_harvest": float(patterns["double_harvest"]),
            "containment": float(patterns["containment"]),
            "triple_trial": float(patterns["triple_trial"]),
            "trial_collapse": float(patterns["trial_collapse"]),
            "sabbath_arc": float(patterns["sabbath_arc"]),
            "winter_friction": float(patterns["winter_friction"]),
            # Double positions
            "double_position_life_month": float(patterns["double_position_life_month"]),
            "double_position_month_week": float(patterns["double_position_month_week"]),
            "double_position_count": float(patterns["double_position_count"]),
            # Weighted convergence (full scoring formula)
            "weighted_convergence": float(cycle_results["weighted_convergence"]),
            # Scripture quality (avg scriptural strength of hitting cycles)
            "scripture_quality": float(cycle_results["scripture_quality"]),
        }


class LeagueMoedim:
    """League-wide Moedim System tracking for NFL."""

    def __init__(self, sport=None, moedim_tolerance=None, **kwargs):
        self.tolerance = moedim_tolerance if moedim_tolerance is not None else 3
        self.sport = sport or "NFL"
        self.founding_dates = FOUNDING_DATES
        self.teams = {}
        self.n_games = 0

    def _get_team(self, team_name):
        if team_name not in self.teams:
            founding = self.founding_dates.get(team_name)
            if founding:
                self.teams[team_name] = TeamMoedim(founding, self.tolerance)
            else:
                # Try common abbreviation variants
                for alt in [team_name.upper(), team_name[:3].upper()]:
                    if alt in self.founding_dates:
                        self.teams[team_name] = TeamMoedim(
                            self.founding_dates[alt], self.tolerance
                        )
                        break
                else:
                    self.teams[team_name] = TeamMoedim(None, self.tolerance)
                    logging.debug(
                        "Moedim: No founding date for team '%s'", team_name
                    )
        return self.teams[team_name]

    def add_game(self, team, *args, **kwargs):
        """No state to update -- Moedim is purely date-based."""
        self.n_games += 1

    def get_features(self, home_team, away_team, game_date=None, **kwargs):
        """Get differential Moedim features for a matchup.

        Returns diff features (home - away) for all ~35 per-team features,
        plus raw home/away convergence and zone scores.
        """
        h = self._get_team(home_team)
        a = self._get_team(away_team)

        h_feats = h.get_features(game_date)
        a_feats = a.get_features(game_date)

        result = {}
        for key in h_feats:
            result["moedim_{}_diff".format(key)] = h_feats[key] - a_feats[key]

        # Add raw home/away features (some are useful without differencing)
        result["moedim_home_convergence"] = h_feats["convergence"]
        result["moedim_away_convergence"] = a_feats["convergence"]
        result["moedim_home_zone"] = h_feats["zone_score"]
        result["moedim_away_zone"] = a_feats["zone_score"]
        result["moedim_home_weighted_convergence"] = h_feats["weighted_convergence"]
        result["moedim_away_weighted_convergence"] = a_feats["weighted_convergence"]
        result["moedim_home_hierarchy"] = h_feats["hierarchy_factor"]
        result["moedim_away_hierarchy"] = a_feats["hierarchy_factor"]

        return result
