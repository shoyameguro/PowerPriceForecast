from pathlib import Path
import joblib
import lightgbm as lgb
import pandas as pd


class LGBMWrapper:
    """LightGBM Booster wrapper (v4+) that stores feature / category info."""

    def __init__(self, params: dict):
        self.params = params.copy()
        self.model: lgb.Booster | None = None
        self.feature_names: list[str] | None = None
        self.categorical_cols: list[str] | None = None

    # ---------- private ----------
    @staticmethod
    def _to_category(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
        for c in cat_cols:
            if c in df.columns and not pd.api.types.is_categorical_dtype(df[c]):
                df[c] = df[c].astype("category")
        return df

    # ---------- train ----------
    def fit(self, X, y, valid):
        # 1) オブジェクト型 → category
        obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
        X = self._to_category(X.copy(), obj_cols)
        X_val = self._to_category(valid[0].copy(), obj_cols)

        # 2) LightGBM 学習
        train_set = lgb.Dataset(X, y)
        val_set   = lgb.Dataset(X_val, valid[1])

        callbacks = [lgb.log_evaluation(period=100)]
        early = self.params.pop("early_stopping_rounds", None)
        if early:
            callbacks.append(lgb.early_stopping(early))

        self.model = lgb.train(
            params=self.params,
            train_set=train_set,
            num_boost_round=1000,
            valid_sets=[val_set],
            callbacks=callbacks,
        )

        # 3) 後で合わせるために列情報を保持
        self.feature_names = X.columns.tolist()
        self.categorical_cols = obj_cols

    # ---------- predict ----------
    def predict(self, X):
        # 1) 列順を学習時に合わせる
        X = X.reindex(columns=self.feature_names, copy=False)
        # 2) dtype も同期
        X = self._to_category(X.copy(), self.categorical_cols)
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    # ---------- persistence ----------
    def save(self, path: Path):
        obj = {
            "model_type": "lgbm",
            "booster": self.model,
            "feature_names": self.feature_names,
            "categorical_cols": self.categorical_cols,
        }
        joblib.dump(obj, path)

    def load(self, path: Path):
        d = joblib.load(path)
        self.model = d["booster"]
        self.feature_names = d["feature_names"]
        self.categorical_cols = d["categorical_cols"]

    def get_importance(self, importance_type: str = "gain"):
        if self.model is None:
            raise ValueError("Model not trained")
        imps = self.model.feature_importance(importance_type=importance_type)
        return pd.Series(imps, index=self.feature_names)

