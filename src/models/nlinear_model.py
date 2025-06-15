from pathlib import Path
import joblib
import pandas as pd
import numpy as np

try:
    from darts import TimeSeries
    from darts.dataprocessing.transformers import Scaler
    from darts.models import NLinearModel
except Exception as e:  # pragma: no cover - darts may be missing in env
    TimeSeries = None
    Scaler = None
    NLinearModel = None

class NLinearWrapper:
    """Minimal wrapper around Darts ``NLinearModel``.

    This adapter allows the NLinear model to be used inside the existing
    training pipeline which expects ``fit(X, y, valid)`` and ``predict(X)``
    methods similar to the LightGBM/XGBoost wrappers.
    """

    def __init__(self, params: dict):
        self.params = params.copy()
        self.model = None
        self.scaler_y = None
        self.scaler_cov = None
        self.feature_names: list[str] | None = None
        self._train_series = None
        self._train_cov = None
        self._history_len = 0

    # ------------------------------------------------------------------
    def _to_timeseries(self, y: pd.Series, X: pd.DataFrame, start: int = 0):
        idx = pd.RangeIndex(start, start + len(y))
        ts_y = TimeSeries.from_times_and_values(idx, y.values.astype(np.float32))
        ts_cov = TimeSeries.from_times_and_values(idx, X.values.astype(np.float32))
        return ts_y, ts_cov

    # ------------------------------------------------------------------
    def fit(self, X, y, valid):
        if TimeSeries is None:
            raise ImportError("darts library is required for NLinearWrapper")

        X_tr, y_tr = X.copy(), y.copy()
        X_val, y_val = valid

        self.feature_names = X_tr.columns.tolist()
        ts_y_tr, ts_cov_tr = self._to_timeseries(y_tr, X_tr)
        ts_y_val, ts_cov_val = self._to_timeseries(y_val, X_val, start=len(y_tr))

        self.scaler_y = Scaler()
        self.scaler_cov = Scaler()
        ts_y_tr = self.scaler_y.fit_transform(ts_y_tr)
        ts_cov_tr = self.scaler_cov.fit_transform(ts_cov_tr)
        ts_y_val = self.scaler_y.transform(ts_y_val)
        ts_cov_val = self.scaler_cov.transform(ts_cov_val)

        self.model = NLinearModel(**self.params)
        self.model.fit(
            series=ts_y_tr,
            past_covariates=ts_cov_tr,
            val_series=ts_y_val,
            val_past_covariates=ts_cov_val,
            verbose=True,
        )

        self._train_series = ts_y_tr
        self._train_cov = ts_cov_tr
        self._history_len = len(ts_y_tr)

    # ------------------------------------------------------------------
    def _predict_future(self, future_cov: "TimeSeries"):
        cov_all = self._train_cov.concatenate(future_cov)
        preds = self.model.predict(
            n=len(future_cov),
            series=self._train_series,
            past_covariates=cov_all,
        )
        self._train_series = self._train_series.concatenate(preds)
        self._train_cov = cov_all
        preds = self.scaler_y.inverse_transform(preds)
        return preds.values().flatten()

    def predict(self, X):
        if TimeSeries is None:
            raise ImportError("darts library is required for NLinearWrapper")
        X = X.reindex(columns=self.feature_names, copy=False)
        future_cov = TimeSeries.from_times_and_values(
            pd.RangeIndex(self._history_len, self._history_len + len(X)),
            X.values.astype(np.float32),
        )
        future_cov = self.scaler_cov.transform(future_cov)
        preds = self._predict_future(future_cov)
        self._history_len += len(X)
        return preds

    # ------------------------------------------------------------------
    def save(self, path: Path):
        obj = {
            "model_type": "nlinear",
            "model": self.model,
            "feature_names": self.feature_names,
            "scaler_y": self.scaler_y,
            "scaler_cov": self.scaler_cov,
            "train_series": self._train_series,
            "train_cov": self._train_cov,
            "history_len": self._history_len,
        }
        joblib.dump(obj, path)

    def load(self, path: Path):
        d = joblib.load(path)
        self.model = d["model"]
        self.feature_names = d["feature_names"]
        self.scaler_y = d.get("scaler_y")
        self.scaler_cov = d.get("scaler_cov")
        self._train_series = d.get("train_series")
        self._train_cov = d.get("train_cov")
        self._history_len = d.get("history_len", 0)

