"""Mega-Ensemble Configuration: per-model on/off switches + settings.

Controls which models are active in the mega-ensemble.
Supports individual model enable/disable and single-model optimization.

Settings file: {sport}_mega_settings.json in each sport directory.
"""

import os
import json

# ── Master model registry ──────────────────────────────────────────
# Every model in the mega-ensemble, with default on/off and description.

MODEL_REGISTRY = {
    # Tier 0: Core (always on by default)
    "elo":              {"default": True,  "tier": 0, "desc": "Elo ratings (24+ adjusters)"},
    "xgboost":          {"default": True,  "tier": 0, "desc": "XGBoost ensemble (31 rolling features)"},

    # Tier 1: Proven models
    "hmm":              {"default": True,  "tier": 1, "desc": "Hidden Markov Model (hot/cold states)"},
    "kalman":           {"default": True,  "tier": 1, "desc": "Kalman Filter (strength estimation)"},
    "pagerank":         {"default": True,  "tier": 1, "desc": "PageRank + HITS (network analysis)"},
    "lightgbm":         {"default": True,  "tier": 1, "desc": "LightGBM (leaf-wise boosting)"},
    "catboost":         {"default": True,  "tier": 1, "desc": "CatBoost (ordered boosting)"},
    "mlp":              {"default": True,  "tier": 1, "desc": "MLP neural network (deep features)"},
    "lstm":             {"default": False, "tier": 1, "desc": "LSTM (sequential patterns)"},

    # Tier 2: Exotic / physics-inspired
    "garch":            {"default": True,  "tier": 2, "desc": "GARCH volatility (time-varying)"},
    "fourier":          {"default": True,  "tier": 2, "desc": "Fourier / wavelet (cycle detection)"},
    "survival":         {"default": True,  "tier": 2, "desc": "Survival analysis (streak hazards)"},
    "copula":           {"default": True,  "tier": 2, "desc": "Copula (off/def joint dependency)"},

    # Tier 3: Information & physics
    "info_theory":      {"default": True,  "tier": 3, "desc": "Shannon entropy + KL divergence"},
    "momentum":         {"default": True,  "tier": 3, "desc": "Newtonian momentum / inertia"},
    "markov":           {"default": True,  "tier": 3, "desc": "Markov chain (transition matrices)"},
    "clustering":       {"default": True,  "tier": 3, "desc": "k-Means team archetypes"},
    "game_theory":      {"default": True,  "tier": 3, "desc": "Nash equilibrium + style matchups"},

    # Tier 4: Classical rating systems
    "poisson":          {"default": True,  "tier": 4, "desc": "Poisson / Dixon-Coles (score dist)"},
    "glicko":           {"default": True,  "tier": 4, "desc": "Glicko-2 (uncertainty-aware ratings)"},
    "bradley_terry":    {"default": True,  "tier": 4, "desc": "Bradley-Terry MLE (paired comparison)"},
    "monte_carlo":      {"default": True,  "tier": 4, "desc": "Monte Carlo simulation (3000 sims)"},
    "random_forest":    {"default": True,  "tier": 4, "desc": "Random Forest (bagging diversity)"},

    # Tier 5: Classical baseball/sports models
    "srs":              {"default": True,  "tier": 5, "desc": "Simple Rating System (margin + SOS)"},
    "colley":           {"default": True,  "tier": 5, "desc": "Colley Matrix (bias-free ranking)"},
    "log5":             {"default": True,  "tier": 5, "desc": "Log5 Bill James (h2h formula)"},
    "pythagenpat":      {"default": True,  "tier": 5, "desc": "PythagenPat (dynamic exponent)"},
    "exp_smoothing":    {"default": True,  "tier": 5, "desc": "Exponential smoothing (trend tracking)"},
    "mean_reversion":   {"default": True,  "tier": 5, "desc": "Mean reversion (Bollinger bands)"},

    # Data enrichment models
    "weather":          {"default": False, "tier": 6, "desc": "Weather impact (temperature, wind)"},
    "sentiment":        {"default": False, "tier": 6, "desc": "Reddit sentiment (VADER NLP)"},
    "odds":             {"default": False, "tier": 6, "desc": "Market odds / CLV tracking"},
}

# All model names in order
ALL_MODELS = list(MODEL_REGISTRY.keys())


def get_default_switches():
    """Get default on/off switches for all models."""
    return {name: spec["default"] for name, spec in MODEL_REGISTRY.items()}


def load_model_switches(sport, sport_dir):
    """Load model on/off switches from settings file."""
    path = os.path.join(sport_dir, f"{sport}_mega_settings.json")
    switches = get_default_switches()

    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            saved = data.get("model_switches", {})
            switches.update(saved)
        except Exception:
            pass

    return switches


def save_model_switches(sport, sport_dir, switches):
    """Save model on/off switches to settings file."""
    path = os.path.join(sport_dir, f"{sport}_mega_settings.json")

    # Load existing settings
    data = {}
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception:
            pass

    data["model_switches"] = switches

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def is_model_enabled(model_name, switches):
    """Check if a model is enabled."""
    return switches.get(model_name, MODEL_REGISTRY.get(model_name, {}).get("default", False))


def print_model_status(switches):
    """Print all models with their on/off status."""
    print("\n  %-20s %-5s  %-4s  %s" % ("Model", "Status", "Tier", "Description"))
    print("  " + "-" * 70)

    current_tier = -1
    for name, spec in MODEL_REGISTRY.items():
        if spec["tier"] != current_tier:
            current_tier = spec["tier"]
            tier_names = {0: "Core", 1: "Proven", 2: "Exotic",
                          3: "Info/Physics", 4: "Classical", 5: "Sports-specific",
                          6: "Data Enrichment"}
            print("  -- %s --" % tier_names.get(current_tier, f"Tier {current_tier}"))

        enabled = is_model_enabled(name, switches)
        status = " ON " if enabled else " OFF"
        marker = "[*]" if enabled else "[ ]"
        print("  %s %-17s %s   T%d    %s" % (marker, name, status, spec["tier"], spec["desc"]))

    n_on = sum(1 for n in ALL_MODELS if is_model_enabled(n, switches))
    print("\n  %d/%d models enabled" % (n_on, len(ALL_MODELS)))


# ── Mega parameter set/get ─────────────────────────────────────────

# All settable mega-ensemble parameters with aliases and descriptions
MEGA_PARAMS = {
    # Critical
    "max_adj":                 {"aliases": ["maxadj", "adj", "adjustment"], "type": "float",
                                "desc": "Max meta-learner adjustment (+/- probability)"},
    "meta_model":              {"aliases": ["meta", "metalearner", "stacker"], "type": "str",
                                "desc": "Meta-learner type (ridge, logistic, xgboost)"},
    "retrain_every":           {"aliases": ["retrain", "retrain_interval"], "type": "int",
                                "desc": "Retrain meta-learner every N games"},
    "min_train":               {"aliases": ["mintrain", "min_games", "warmup"], "type": "int",
                                "desc": "Games before meta-learner starts predicting"},
    # Kalman
    "kalman_process_noise":    {"aliases": ["kalman_pn", "process_noise", "pn"], "type": "float",
                                "desc": "Kalman filter process noise"},
    "kalman_measurement_noise":{"aliases": ["kalman_mn", "measurement_noise", "mn"], "type": "float",
                                "desc": "Kalman filter measurement noise"},
    # HMM
    "hmm_states":              {"aliases": ["hmm_n", "n_states", "states"], "type": "int",
                                "desc": "Number of HMM hidden states"},
    # Network
    "network_decay":           {"aliases": ["net_decay", "pagerank_decay", "decay"], "type": "float",
                                "desc": "PageRank temporal decay (0-1)"},
    # Momentum
    "momentum_friction":       {"aliases": ["friction", "mom_friction"], "type": "float",
                                "desc": "Momentum friction coefficient"},
    # Clustering
    "n_clusters":              {"aliases": ["clusters", "k_clusters", "nclusters"], "type": "int",
                                "desc": "Number of team archetype clusters"},
    # Glicko
    "glicko_initial_rd":       {"aliases": ["glicko_rd", "initial_rd", "rd"], "type": "float",
                                "desc": "Glicko-2 initial rating deviation"},
    # Bradley-Terry
    "bt_decay":                {"aliases": ["bt_recency", "bradley_decay"], "type": "float",
                                "desc": "Bradley-Terry recency decay (0-1)"},
    # Monte Carlo
    "mc_simulations":          {"aliases": ["mc_sims", "simulations", "n_sims", "sims"], "type": "int",
                                "desc": "Monte Carlo simulations per game"},
    # Window
    "window":                  {"aliases": ["rolling_window", "feat_window"], "type": "int",
                                "desc": "Rolling feature window size (games)"},
}

# Build reverse lookup: alias -> canonical name
_ALIAS_MAP = {}
for canonical, spec in MEGA_PARAMS.items():
    _ALIAS_MAP[canonical] = canonical
    for alias in spec["aliases"]:
        _ALIAS_MAP[alias] = canonical


def resolve_mega_param(name):
    """Resolve a parameter name/alias to its canonical name. Returns None if unknown."""
    return _ALIAS_MAP.get(name.lower().strip())


def load_mega_params(sport, sport_dir):
    """Load all mega params from settings file."""
    path = os.path.join(sport_dir, f"{sport}_mega_settings.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_mega_params(sport, sport_dir, params):
    """Save mega params to settings file (merges with existing)."""
    path = os.path.join(sport_dir, f"{sport}_mega_settings.json")
    existing = {}
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                existing = json.load(f)
        except Exception:
            pass
    existing.update(params)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


def handle_mega_set(cmd_args, sport, sport_dir):
    """Handle 'mega set param=value' command. Returns (success, message).

    Usage: mega set max_adj=0.10
           mega set kalman_pn=1.5
           mega set meta=ridge
           mega set mc_sims=3000
    """
    parts = cmd_args.split("=", 1)
    if len(parts) != 2:
        return False, "Usage: mega set <param>=<value>"

    raw_name = parts[0].strip()
    raw_value = parts[1].strip()

    canonical = resolve_mega_param(raw_name)
    if canonical is None:
        avail = ", ".join(sorted(MEGA_PARAMS.keys()))
        return False, "Unknown mega param: %s\n  Available: %s" % (raw_name, avail)

    spec = MEGA_PARAMS[canonical]

    # Parse value
    if spec["type"] == "str":
        value = raw_value
    elif spec["type"] == "int":
        try:
            value = int(float(raw_value))
        except ValueError:
            return False, "Invalid integer: %s" % raw_value
    elif spec["type"] == "float":
        try:
            value = round(float(raw_value), 6)
        except ValueError:
            return False, "Invalid number: %s" % raw_value
    else:
        value = raw_value

    # Save
    params = load_mega_params(sport, sport_dir)
    params[canonical] = value
    save_mega_params(sport, sport_dir, params)

    return True, "Set %s = %s  (%s)" % (canonical, value, spec["desc"])


def print_mega_settings(sport, sport_dir):
    """Print all mega-ensemble settings with current values."""
    params = load_mega_params(sport, sport_dir)

    print("\n  %-30s %-12s  %s" % ("Parameter", "Value", "Description"))
    print("  " + "-" * 75)

    for canonical, spec in MEGA_PARAMS.items():
        value = params.get(canonical, "(default)")
        aliases = ", ".join(spec["aliases"][:2])
        print("  %-30s %-12s  %s" % (canonical, value, spec["desc"]))
        if aliases:
            print("  %-30s              aliases: %s" % ("", aliases))

    # Also show model switches
    switches = load_model_switches(sport, sport_dir)
    n_on = sum(1 for m in ALL_MODELS if is_model_enabled(m, switches))
    print("\n  Models: %d/%d enabled (use 'mega models' to see full list)" % (n_on, len(ALL_MODELS)))
