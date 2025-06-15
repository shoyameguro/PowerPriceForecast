#!/usr/bin/env python
"""XGBoost model wrapper compatible with training pipeline."""
from pathlib import Path
import joblib
import pandas as pd
import xgboost as xgb


class XGBWrapper:
    def __init__(self, params: dict):
        self.params = params.copy()
        self.model: xgb.XGBRegressor | None = None
        self.feature_names: list[str] | None = None
        self.categorical_cols: list[str] | None = None
        self.cat_maps: dict[str, list] | None = None

    # ---------- private ----------
    @staticmethod
    def _encode_categories(df: pd.DataFrame, cat_cols: list[str], cat_maps: dict[str, list]) -> pd.DataFrame:
        for c in cat_cols:
            codes = pd.Categorical(df[c], categories=cat_maps[c]).codes
            df[c] = codes
        return df

    # ---------- train ----------
    def fit(self, X, y, valid):
        obj_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        self.categorical_cols = obj_cols
        self.cat_maps = {}
        X = X.copy()
        X_val = valid[0].copy()
        for c in obj_cols:
            cats = pd.Categorical(pd.concat([X[c], X_val[c]], ignore_index=True)).categories
            self.cat_maps[c] = list(cats)
            X[c] = pd.Categorical(X[c], categories=cats).codes
            X_val[c] = pd.Categorical(X_val[c], categories=cats).codes

        early = self.params.pop("early_stopping_rounds", None)
        self.model = xgb.XGBRegressor(**self.params)
        eval_set = [(X_val, valid[1])]
        self.model.fit(X, y, eval_set=eval_set, early_stopping_rounds=early, verbose=100)

        self.feature_names = X.columns.tolist()

    # ---------- predict ----------
    def predict(self, X):
        X = X.reindex(columns=self.feature_names, copy=False)
        X = X.copy()
        if self.categorical_cols:
            X = self._encode_categories(X, self.categorical_cols, self.cat_maps)
        return self.model.predict(X)

    # ---------- persistence ----------
    def save(self, path: Path):
        obj = {
            "model_type": "xgb",
            "booster": self.model,
            "feature_names": self.feature_names,
            "categorical_cols": self.categorical_cols,
            "cat_maps": self.cat_maps,
        }
        joblib.dump(obj, path)

    def load(self, path: Path):
        d = joblib.load(path)
        self.model = d["booster"]
        self.feature_names = d["feature_names"]
        self.categorical_cols = d["categorical_cols"]
        self.cat_maps = d.get("cat_maps", {})

