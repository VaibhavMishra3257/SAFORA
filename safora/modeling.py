"""Training, evaluation, and inference for four-week demand forecasting."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from .features import make_supervised, validate_trends, latest_feature_rows
from .config import MODEL_PATH, METRICS_PATH, TRENDS_PATH

def _metrics(y_true, pred):
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "r2": float(r2_score(y_true, pred)) if len(y_true) > 1 else None
    }

def train_model(data_path: str | Path = TRENDS_PATH, horizon: int = 4,
                model_path: str | Path = MODEL_PATH,
                metrics_path: str | Path = METRICS_PATH) -> dict:
    raw = pd.read_csv(data_path)
    clean = validate_trends(raw)
    supervised, features = make_supervised(clean, horizon=horizon)
    if supervised.empty:
        raise ValueError("No trainable rows. Need regular time-series history for each title.")
    unique_dates = sorted(supervised["date"].unique())
    if len(unique_dates) < 10:
        raise ValueError("Too few distinct dates for a chronological evaluation.")
    cutoff_idx = max(1, int(len(unique_dates) * 0.8))
    cutoff = pd.Timestamp(unique_dates[cutoff_idx])
    train = supervised[supervised["target_date"] < cutoff]
    test = supervised[supervised["date"] >= cutoff]
    if len(train) < 30 or len(test) < 10:
        raise ValueError(
            f"Insufficient train/test rows (train={len(train)}, test={len(test)}). "
            "Collect more historical observations or more titles."
        )
    X_train, y_train = train[features], train["target"]
    X_test, y_test = test[features], test["target"]

    rf = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=3, max_features=0.9,
        random_state=42, n_jobs=-1
    )
    et = ExtraTreesRegressor(
        n_estimators=250, min_samples_leaf=3, max_features=0.9,
        random_state=42, n_jobs=-1
    )
    model = VotingRegressor([("rf", rf), ("extra_trees", et)])
    model.fit(X_train, y_train)
    pred = np.clip(model.predict(X_test), 0, 100)
    baseline = np.clip(test["lag_1"].to_numpy(), 0, 100)

    imp = permutation_importance(model, X_test, y_test, n_repeats=8, random_state=42, scoring="neg_mean_absolute_error")
    importance = {f: float(v) for f, v in sorted(zip(features, imp.importances_mean), key=lambda x: x[1], reverse=True)}
    metrics = {
        "model": "VotingRegressor(RandomForest + ExtraTrees)",
        "target": f"search interest {horizon} weekly steps ahead",
        "horizon_weeks": int(horizon),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "cutoff_date": str(cutoff.date()),
        "test_date_min": str(test["date"].min().date()),
        "test_date_max": str(test["date"].max().date()),
        "model_metrics": _metrics(y_test, pred),
        "naive_baseline_metrics": _metrics(y_test, baseline),
        "beats_baseline_on_mae": bool(mean_absolute_error(y_test, pred) < mean_absolute_error(y_test, baseline)),
        "feature_importance": importance,
        "warning": "Search interest is a proxy, not revenue. Metrics apply to this test split only."
    }
    model_path, metrics_path = Path(model_path), Path(metrics_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": features, "horizon": horizon}, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics

def load_model(model_path: str | Path = MODEL_PATH):
    return joblib.load(model_path)

def predict_latest(trends_path: str | Path = TRENDS_PATH, model_path: str | Path = MODEL_PATH) -> pd.DataFrame:
    raw = pd.read_csv(trends_path)
    bundle = load_model(model_path)
    latest = latest_feature_rows(raw, bundle["features"])
    if latest.empty:
        return latest
    X = latest[bundle["features"]]
    latest["forecast_interest"] = np.clip(bundle["model"].predict(X), 0, 100)
    latest["change_vs_latest"] = latest["forecast_interest"] - latest["latest_interest"]
    latest["direction"] = np.where(latest["change_vs_latest"] > 2, "Rising",
                           np.where(latest["change_vs_latest"] < -2, "Falling", "Stable"))
    return latest.sort_values("forecast_interest", ascending=False)
