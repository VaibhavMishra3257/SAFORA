"""Leakage-aware time-series feature engineering."""
from __future__ import annotations
import pandas as pd

BASE_LAGS = (1, 2, 3, 4, 8, 12)

def validate_trends(df: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "title", "search_interest"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Trends data missing required columns: {sorted(missing)}")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["search_interest"] = pd.to_numeric(out["search_interest"], errors="coerce")
    out["title"] = out["title"].astype(str).str.strip()
    out = out.dropna(subset=["date", "search_interest"])
    out = out[out["title"] != ""]
    out = out[(out["search_interest"] >= 0) & (out["search_interest"] <= 100)]
    out = out.sort_values(["title", "date"])
    # Duplicate observations can occur when exports are concatenated.
    out = out.drop_duplicates(["title", "date"], keep="last")
    return out

def make_supervised(df: pd.DataFrame, horizon: int = 4, frequency: str = "W-SUN") -> tuple[pd.DataFrame, list[str]]:
    """Create future target and past-only predictors for each title.

    Input should be regular weekly data. Missing periods are left missing rather
    than fabricated; rows with incomplete features/targets are removed later.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    clean = validate_trends(df)
    parts = []
    features = [f"lag_{lag}" for lag in BASE_LAGS]
    features += ["rolling_mean_4", "rolling_mean_8", "rolling_std_8", "momentum_4", "week_sin", "week_cos"]
    for title, group in clean.groupby("title"):
        group = group.sort_values("date").set_index("date")
        # Resample to weekly periods. Do not interpolate missing target observations.
        weekly = group["search_interest"].resample(frequency).mean().to_frame()
        weekly["title"] = title
        for lag in BASE_LAGS:
            weekly[f"lag_{lag}"] = weekly["search_interest"].shift(lag)
        shifted = weekly["search_interest"].shift(1)
        weekly["rolling_mean_4"] = shifted.rolling(4, min_periods=4).mean()
        weekly["rolling_mean_8"] = shifted.rolling(8, min_periods=8).mean()
        weekly["rolling_std_8"] = shifted.rolling(8, min_periods=8).std()
        weekly["momentum_4"] = weekly["lag_1"] - weekly["lag_4"]
        week = weekly.index.isocalendar().week.astype(int)
        weekly["week_sin"] = __import__("numpy").sin(2 * __import__("numpy").pi * week / 52.18)
        weekly["week_cos"] = __import__("numpy").cos(2 * __import__("numpy").pi * week / 52.18)
        weekly["target"] = weekly["search_interest"].shift(-horizon)
        weekly["target_date"] = weekly.index.to_series().shift(-horizon)
        weekly = weekly.reset_index(names="date")
        parts.append(weekly)
    if not parts:
        return pd.DataFrame(), features
    result = pd.concat(parts, ignore_index=True)
    result = result.dropna(subset=features + ["target", "target_date"])
    result["target_date"] = pd.to_datetime(result["target_date"])
    return result.sort_values("date"), features

def latest_feature_rows(df: pd.DataFrame, features: list[str], frequency: str = "W-SUN") -> pd.DataFrame:
    """Create latest past-only feature row per title for future prediction."""
    clean = validate_trends(df)
    rows = []
    import numpy as np
    for title, group in clean.groupby("title"):
        series = group.set_index("date")["search_interest"].resample(frequency).mean()
        series = series.dropna()
        if len(series) < 12:
            continue
        values = {}
        for lag in BASE_LAGS:
            values[f"lag_{lag}"] = float(series.iloc[-lag])
        hist = series.iloc[:-1]
        values["rolling_mean_4"] = float(hist.tail(4).mean())
        values["rolling_mean_8"] = float(hist.tail(8).mean())
        values["rolling_std_8"] = float(hist.tail(8).std())
        values["momentum_4"] = values["lag_1"] - values["lag_4"]
        week = int(series.index[-1].isocalendar().week)
        values["week_sin"] = float(np.sin(2 * np.pi * week / 52.18))
        values["week_cos"] = float(np.cos(2 * np.pi * week / 52.18))
        values["title"] = title
        values["last_date"] = series.index[-1]
        values["latest_interest"] = float(series.iloc[-1])
        rows.append(values)
    return pd.DataFrame(rows)
