"""Regression: no module imports the stripped mega/xgb/enhanced code paths."""

import importlib
import inspect

STRIPPED_TOKENS = (
    "from enhanced_model",
    "import enhanced_model",
    "from mega_backtest",
    "from mega_optimizer",
    "from mega_predictor",
    "from mega_config",
    "from meta_learner",
    "MegaPredictor",
    "run_enhanced_backtest",
    "load_enhanced_model",
    "run_mega_backtest",
)

KEEPER_MODULES = (
    "config", "platt", "data_games", "data_players", "elo_model",
    "build_model", "backtest", "predict_ledger", "html_generator",
    "live_scores", "auto_resolve", "injuries", "kalshi", "odds_tracker",
    "weather", "metrics", "cache_utils", "help_system", "elo_set_handler",
    "single_param_opt",
)


def test_keeper_modules_have_no_stripped_tokens():
    for mod_name in KEEPER_MODULES:
        mod = importlib.import_module(mod_name)
        src = inspect.getsource(mod)
        for token in STRIPPED_TOKENS:
            assert token not in src, f"{mod_name}.py still contains token '{token}'"
