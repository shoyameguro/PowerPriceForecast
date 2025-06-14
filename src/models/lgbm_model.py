from pathlib import Path
import joblib
import lightgbm as lgb

class LGBMWrapper:
    """LightGBM Booster wrapper (v4+)."""

    def __init__(self, params: dict):
        self.params = params.copy()
        self.model: lgb.Booster | None = None

    def _ensure_category(self, X):
        obj_cols = X.select_dtypes(include=["object"]).columns
        if len(obj_cols):
            X[obj_cols] = X[obj_cols].astype("category")
        return X

    # ---------- train ----------
    def fit(self, X, y, valid):
        X = self._ensure_category(X.copy())
        X_val = self._ensure_category(valid[0].copy())
        y_val = valid[1]

        train_set = lgb.Dataset(X, y)
        val_set   = lgb.Dataset(X_val, y_val)

        callbacks = [lgb.log_evaluation(period=100)]
        early = self.params.pop("early_stopping_rounds", None)
        if early:
            callbacks.append(lgb.early_stopping(early))

        self.model = lgb.train(
            params=self.params,
            train_set=train_set,
            num_boost_round=5000,
            valid_sets=[val_set],
            callbacks=callbacks,
        )

    # ---------- predict ----------
    def predict(self, X):
        X = self._ensure_category(X.copy())
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    # ---------- persistence ----------
    def save(self, path: Path):
        joblib.dump(self.model, path)

    def load(self, path: Path):
        self.model = joblib.load(path)