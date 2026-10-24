"""Transparent opportunity scoring using observed signals only."""
from __future__ import annotations
import numpy as np
import pandas as pd

def minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    lo, hi = values.min(), values.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(np.nan, index=series.index)
    return (values - lo) / (hi - lo)

def opportunity_scores(forecasts: pd.DataFrame, metadata: pd.DataFrame | None = None) -> pd.DataFrame:
    """Create a relative ranking, not a trained commercial-success probability.

    Weights are explicit heuristic defaults and must be shown to users.
    """
    if forecasts.empty:
        return forecasts.copy()
    out = forecasts.copy()
    out["forecast_norm"] = minmax(out["forecast_interest"])
    out["momentum_norm"] = minmax(out["change_vs_latest"])
    # 70% forecast level, 30% momentum; reweight if one feature is unavailable.
    parts = []
    weights = []
    for col, weight in [("forecast_norm", 0.7), ("momentum_norm", 0.3)]:
        if out[col].notna().any():
            parts.append(out[col].fillna(out[col].median()) * weight)
            weights.append(weight)
    if parts:
        score = sum(parts) / sum(weights)
        out["opportunity_score"] = (score * 100).round(1)
    else:
        out["opportunity_score"] = np.nan
    out["score_type"] = "heuristic relative ranking; not probability"
    if metadata is not None and not metadata.empty and "title" in metadata:
        out = out.merge(metadata, on="title", how="left", suffixes=("", "_metadata"))
    return out.sort_values("opportunity_score", ascending=False)
