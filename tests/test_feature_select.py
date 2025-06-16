import numpy as np
import pandas as pd
from src.training.tune_hyperparams import select_features_optuna
from src.utils.cv import PurgedWalkForwardCV
from src.models.lgbm_model import LGBMWrapper


def test_select_features_returns_subset():
    X = pd.DataFrame(np.random.randn(200, 5), columns=list('ABCDE'))
    y = pd.Series(np.random.randn(200))
    cv = PurgedWalkForwardCV(n_splits=3, test_size=20, gap=5)
    params = {"objective": "regression", "metric": "rmse"}
    cols = select_features_optuna(X, y, cv, "lgbm", params, n_trials=2)
    assert set(cols).issubset(set(X.columns))
