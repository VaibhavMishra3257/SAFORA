"""Model explanations that do not pretend to be causal explanations."""
from __future__ import annotations
import numpy as np
import pandas as pd

def top_global_features(metrics: dict, n: int = 8) -> pd.DataFrame:
    values = metrics.get("feature_importance", {})
    return pd.DataFrame(
        [{"feature": k, "importance": v} for k, v in list(values.items())[:n]]
    )

def explain_forecast(row: pd.Series, metrics: dict) -> list[str]:
    """Create readable evidence statements from observed features and model output."""
    statements = []
    latest = row.get("latest_interest")
    forecast = row.get("forecast_interest")
    if pd.notna(latest) and pd.notna(forecast):
        delta = float(forecast) - float(latest)
        if delta > 2:
            statements.append(f"Forecast is {delta:.1f} index points above the latest observed value.")
        elif delta < -2:
            statements.append(f"Forecast is {abs(delta):.1f} index points below the latest observed value.")
        else:
            statements.append("Forecast is close to the latest observed index.")
    important = top_global_features(metrics, 3)
    if not important.empty:
        statements.append(
            "Most influential features globally in the held-out test: "
            + ", ".join(important["feature"].astype(str).tolist())
            + ". These are predictive associations, not causal effects."
        )
    statements.append("Search interest is a normalized proxy; this is not a revenue forecast.")
    return statements
