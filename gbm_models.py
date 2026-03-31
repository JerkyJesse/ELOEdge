"""LightGBM + CatBoost ensemble models.

Different gradient boosting implementations find different patterns.
LightGBM: faster training, leaf-wise growth.
CatBoost: native categorical support, ordered boosting (less overfitting).

Packages: lightgbm (pip install lightgbm), catboost (pip install catboost)
"""

import logging
import json
import os
import math

import numpy as np
import pandas as pd


LGBM_MODEL_FILE = "{sport}_lgbm_model.txt"
CATBOOST_MODEL_FILE = "{sport}_catboost_model.cbm"


class LightGBMPredictor:
    """LightGBM binary classifier for game outcome prediction."""

    def __init__(self, sport="nfl"):
        self.sport = sport
        self.model = None
        self._fitted = False
        self.feature_names = []

    def train(self, X, y, feature_names=None):
        """Train LightGBM on feature matrix X and binary labels y.

        Args:
            X: numpy array or DataFrame of features
            y: binary labels (1=home win, 0=away win)
            feature_names: list of feature column names
        """
        import lightgbm as lgb

        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
            X = X.values

        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]

        dtrain = lgb.Dataset(X, label=y, feature_name=self.feature_names)

        # Scale complexity by dataset size to prevent overfitting
        n_samples = X.shape[0]
        if n_samples < 300:
            # Small data (NFL): very conservative
            num_leaves = 8
            min_child = max(20, n_samples // 10)
            n_rounds = 50
            lr = 0.05
            l1, l2 = 1.0, 2.0
        elif n_samples < 1000:
            num_leaves = 15
            min_child = 15
            n_rounds = 150
            lr = 0.03
            l1, l2 = 0.5, 1.0
        else:
            num_leaves = 31
            min_child = 10
            n_rounds = 300
            lr = 0.03
            l1, l2 = 0.1, 0.1

        params = {
            "objective": "binary",
            "metric": ["binary_logloss", "auc"],
            "boosting_type": "gbdt",
            "num_leaves": num_leaves,
            "learning_rate": lr,
            "feature_fraction": 0.6,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": min_child,
            "lambda_l1": l1,
            "lambda_l2": l2,
            "verbose": -1,
            "seed": 42,
        }

        self.model = lgb.train(
            params, dtrain,
            num_boost_round=n_rounds,
            valid_sets=[dtrain],
            callbacks=[lgb.log_evaluation(0)],
        )
        self._fitted = True
        logging.info("LightGBM trained: %d rounds, %d features", 300, X.shape[1])

    def predict_proba(self, X):
        """Predict win probability for home team."""
        if not self._fitted or self.model is None:
            return np.full(len(X), 0.5)

        if isinstance(X, pd.DataFrame):
            X = X.values

        probs = self.model.predict(X)
        return np.clip(probs, 0.01, 0.99)

    def predict_single(self, features_dict):
        """Predict probability from a feature dictionary."""
        if not self._fitted:
            return 0.5

        X = np.array([[features_dict.get(f, 0) for f in self.feature_names]])
        return float(self.predict_proba(X)[0])

    def feature_importance(self, importance_type="gain"):
        """Get feature importance scores."""
        if not self._fitted:
            return {}
        imp = self.model.feature_importance(importance_type=importance_type)
        return dict(zip(self.feature_names, imp))

    def save(self, filepath=None):
        if filepath is None:
            filepath = LGBM_MODEL_FILE.format(sport=self.sport)
        if self.model:
            self.model.save_model(filepath)

    def load(self, filepath=None):
        import lightgbm as lgb
        if filepath is None:
            filepath = LGBM_MODEL_FILE.format(sport=self.sport)
        if os.path.exists(filepath):
            self.model = lgb.Booster(model_file=filepath)
            self._fitted = True
            return True
        return False


class CatBoostPredictor:
    """CatBoost binary classifier for game outcome prediction.

    CatBoost excels with categorical features (team IDs, divisions, venues)
    and uses ordered boosting to reduce overfitting.
    """

    def __init__(self, sport="nfl"):
        self.sport = sport
        self.model = None
        self._fitted = False
        self.feature_names = []
        self.cat_features = []

    def train(self, X, y, feature_names=None, cat_features=None):
        """Train CatBoost on feature matrix X and binary labels y.

        Args:
            X: numpy array or DataFrame of features
            y: binary labels
            feature_names: list of feature column names
            cat_features: list of categorical column indices or names
        """
        from catboost import CatBoostClassifier, Pool

        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
            if cat_features is None:
                cat_features = [i for i, c in enumerate(X.columns) if X[c].dtype == object]
            X = X.values

        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        self.cat_features = cat_features or []

        # Scale complexity by dataset size
        n_samples = X.shape[0]
        if n_samples < 300:
            iters, depth, l2_reg = 50, 3, 10
        elif n_samples < 1000:
            iters, depth, l2_reg = 150, 4, 5
        else:
            iters, depth, l2_reg = 300, 6, 3

        self.model = CatBoostClassifier(
            iterations=iters,
            learning_rate=0.05,
            depth=depth,
            l2_leaf_reg=l2_reg,
            border_count=64,
            bootstrap_type="Bernoulli",
            subsample=0.8,
            random_seed=42,
            verbose=0,
            eval_metric="Logloss",
            auto_class_weights="Balanced",
        )

        pool = Pool(X, y, feature_names=self.feature_names,
                    cat_features=self.cat_features if self.cat_features else None)

        self.model.fit(pool)
        self._fitted = True
        logging.info("CatBoost trained: 300 iterations, %d features", X.shape[1])

    def predict_proba(self, X):
        """Predict win probability for home team."""
        if not self._fitted or self.model is None:
            return np.full(len(X) if hasattr(X, '__len__') else 1, 0.5)

        if isinstance(X, pd.DataFrame):
            X = X.values

        probs = self.model.predict_proba(X)
        # CatBoost returns [p(0), p(1)] -- we want p(1) = home win
        if len(probs.shape) > 1:
            return np.clip(probs[:, 1], 0.01, 0.99)
        return np.clip(probs, 0.01, 0.99)

    def predict_single(self, features_dict):
        """Predict probability from a feature dictionary."""
        if not self._fitted:
            return 0.5

        X = np.array([[features_dict.get(f, 0) for f in self.feature_names]])
        return float(self.predict_proba(X)[0])

    def feature_importance(self):
        """Get feature importance scores."""
        if not self._fitted:
            return {}
        imp = self.model.get_feature_importance()
        return dict(zip(self.feature_names, imp))

    def save(self, filepath=None):
        if filepath is None:
            filepath = CATBOOST_MODEL_FILE.format(sport=self.sport)
        if self.model:
            self.model.save_model(filepath)

    def load(self, filepath=None):
        from catboost import CatBoostClassifier
        if filepath is None:
            filepath = CATBOOST_MODEL_FILE.format(sport=self.sport)
        if os.path.exists(filepath):
            self.model = CatBoostClassifier()
            self.model.load_model(filepath)
            self._fitted = True
            return True
        return False


class GBMEnsemble:
    """Combined LightGBM + CatBoost ensemble.

    Trains both models on the same features and averages their predictions.
    Diversity in implementation leads to better ensemble performance.
    """

    def __init__(self, sport="nfl"):
        self.sport = sport
        self.lgbm = LightGBMPredictor(sport)
        self.catboost = CatBoostPredictor(sport)
        self.lgbm_weight = 0.5
        self.catboost_weight = 0.5
        self._fitted = False

    def train(self, X, y, feature_names=None, cat_features=None):
        """Train both models."""
        self.lgbm.train(X, y, feature_names)
        self.catboost.train(X, y, feature_names, cat_features)
        self._fitted = True

    def predict_proba(self, X):
        """Weighted average of both models' predictions."""
        lgbm_p = self.lgbm.predict_proba(X)
        cat_p = self.catboost.predict_proba(X)
        return self.lgbm_weight * lgbm_p + self.catboost_weight * cat_p

    def predict_single(self, features_dict):
        """Predict from feature dict, returning both individual and combined probs."""
        lgbm_p = self.lgbm.predict_single(features_dict)
        cat_p = self.catboost.predict_single(features_dict)
        combined = self.lgbm_weight * lgbm_p + self.catboost_weight * cat_p
        return {
            "lgbm_prob": lgbm_p,
            "catboost_prob": cat_p,
            "gbm_ensemble_prob": combined,
        }

    def feature_importance(self):
        """Combined feature importance from both models."""
        lgbm_imp = self.lgbm.feature_importance()
        cat_imp = self.catboost.feature_importance()

        all_features = set(list(lgbm_imp.keys()) + list(cat_imp.keys()))
        combined = {}
        for f in all_features:
            l_val = lgbm_imp.get(f, 0)
            c_val = cat_imp.get(f, 0)
            # Normalize each to [0,1] range before combining
            l_max = max(lgbm_imp.values()) if lgbm_imp else 1
            c_max = max(cat_imp.values()) if cat_imp else 1
            combined[f] = (l_val / max(l_max, 1e-10) + c_val / max(c_max, 1e-10)) / 2

        return dict(sorted(combined.items(), key=lambda x: x[1], reverse=True))

    def save(self):
        self.lgbm.save()
        self.catboost.save()

    def load(self):
        l = self.lgbm.load()
        c = self.catboost.load()
        self._fitted = l or c
        return self._fitted
