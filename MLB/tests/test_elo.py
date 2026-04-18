"""Regression tests for ELO-only post-strip MLB model."""

import pytest

from elo_model import MLBElo


def _model():
    m = MLBElo(base_rating=1500, k=2.62, home_adv=37.63)
    m._platt_scaler = None
    return m


def test_init_no_stripped_fields():
    m = _model()
    for attr in ("_xgb_model", "_xgb_meta", "_mega_predictor", "_mega_loading"):
        assert not hasattr(m, attr), f"{attr} should be stripped from MLBElo"


def test_platt_scaler_field_preserved():
    m = _model()
    assert hasattr(m, "_platt_scaler"), "_platt_scaler should still exist"


def test_expected_score_higher_rating_wins():
    m = _model()
    p_higher = m.expected_score(1600, 1500)
    p_lower = m.expected_score(1500, 1600)
    assert p_higher > 0.5 > p_lower
    assert abs(p_higher + p_lower - 1.0) < 1e-9


def test_win_prob_calibrated_no_platt_returns_raw():
    m = _model()
    m.ratings["A"] = 1600
    m.ratings["B"] = 1500
    m._platt_scaler = None
    p = m.win_prob("A", "B", team_a_home=True, neutral_site=False, calibrated=True)
    assert 0.0 <= p <= 1.0


def test_win_prob_uncalibrated_returns_raw():
    m = _model()
    m.ratings["A"] = 1600
    m.ratings["B"] = 1500
    p = m.win_prob("A", "B", team_a_home=True, neutral_site=False, calibrated=False)
    assert 0.0 <= p <= 1.0


def test_pick_winner_returns_valid_tuple():
    m = _model()
    m.ratings["A"] = 1700
    m.ratings["B"] = 1500
    winner, prob = m.pick_winner("A", "B", team_a_home=True, neutral_site=False)
    assert winner in ("A", "B")
    assert 0.5 <= prob <= 1.0


def test_show_settings_no_nameerror(capsys):
    m = _model()
    m.show_settings()
    captured = capsys.readouterr()
    assert "XGBoost" not in captured.out, "XGBoost row should be stripped"
    assert "Mega-Ensemble" not in captured.out, "Mega-Ensemble row should be stripped"
    assert "Base Rating" in captured.out
