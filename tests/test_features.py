import pandas as pd
from safora.features import validate_trends, make_supervised

def test_validate_trends_drops_invalid_and_duplicates():
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-01", "bad"],
        "title": ["A", "A", "B"],
        "search_interest": [20, 30, 40]
    })
    out = validate_trends(df)
    assert len(out) == 1
    assert out.iloc[0]["search_interest"] == 30

def test_make_supervised_uses_past_features():
    dates = pd.date_range("2023-01-01", periods=80, freq="W-SUN")
    df = pd.DataFrame({
        "date": list(dates),
        "title": ["A"] * len(dates),
        "search_interest": [float((i * 7) % 100) for i in range(len(dates))]
    })
    supervised, features = make_supervised(df, horizon=4)
    assert not supervised.empty
    assert "lag_1" in features
    assert "target" in supervised
    assert (supervised["target_date"] > supervised["date"]).all()
