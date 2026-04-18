"""Regression tests for build_model post-mega-strip."""

import inspect

from elo_model import NFLElo


def test_build_model_import_has_no_enhanced_model():
    import build_model
    source = inspect.getsource(build_model)
    for token in ("enhanced_model", "load_enhanced_model", "_load_mega_background",
                  "MegaPredictor", "_xgb_model", "_mega_predictor", "_mega_loading",
                  "threading"):
        assert token not in source, f"build_model.py still references '{token}'"


def test_elo_model_source_has_no_xgb_or_mega():
    import elo_model
    source = inspect.getsource(elo_model)
    for token in ("_xgb_predict", "_xgb_model", "_xgb_meta",
                  "_mega_predictor", "_mega_loading", "build_game_features",
                  "enhanced_model"):
        assert token not in source, f"elo_model.py still references '{token}'"


def test_nfl_elo_init_allows_full_kwargs():
    m = NFLElo(base_rating=1500, k=17.30, home_adv=18.12,
               use_mov=True, player_boost=26.52, rest_factor=30.0)
    assert m.base_rating == 1500
