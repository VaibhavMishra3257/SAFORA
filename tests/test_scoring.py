import pandas as pd
from safora.scoring import opportunity_scores

def test_opportunity_score_is_not_probability():
    df = pd.DataFrame({
        "title": ["A", "B"],
        "forecast_interest": [80, 20],
        "change_vs_latest": [10, -5]
    })
    result = opportunity_scores(df)
    assert result.iloc[0]["title"] == "A"
    assert result["opportunity_score"].between(0, 100).all()
    assert result["score_type"].str.contains("not probability").all()
