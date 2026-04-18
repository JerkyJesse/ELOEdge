"""Regression tests for build_model post-mega-strip."""

import inspect

from elo_model import MLBElo


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


def test_mlb_elo_init_allows_full_kwargs():
    m = MLBElo(base_rating=1500, k=2.62, home_adv=37.63,
               use_mov=True, player_boost=19.53, rest_factor=2.80)
    assert m.base_rating == 1500
    assert m.k == 2.62
