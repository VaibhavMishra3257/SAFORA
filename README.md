# SAFORA — Japanese Media Market Intelligence for India

SAFORA combines public catalogue metadata, search-interest time series, optional review sentiment, forecasting, regional signals, explainability, and scenario planning.

## Important scientific boundary
- AniList metadata is real catalogue data, **not Indian demand**.
- Google Trends values are normalized relative indices, not search counts.
- Search interest is a demand proxy, not revenue or streaming viewership.
- No predictions are generated until a model is trained on collected time-series observations.
- Scenario simulator outputs are assumption-based estimates, not causal effects.
- Regional outputs are shown only when real regional observations have been imported.

## Features
- AniList GraphQL metadata collector
- Google Trends collector (optional unofficial `pytrends` client; may be rate-limited)
- CSV import path for Google Trends exports
- Leakage-aware four-week demand forecasting
- Naive baseline comparison, chronological test split, rolling-origin evaluation
- Title-level forecast dashboard
- Permutation feature importance
- Transparent opportunity score based on observed signals
- Optional multilingual sentiment adapter (requires model/dependency)
- Regional heatmap from actual regional data
- What-if scenario panel with explicit assumptions
- Source and data coverage checks

## Quick start (Windows PowerShell)
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m safora collect-anilist --pages 5
streamlit run app.py
```

The dashboard opens even before demand data exists; it will clearly explain what is missing.

## Collect search interest
Optional automated collector:
```powershell
python -m safora collect-trends --titles "Naruto,One Piece,Demon Slayer,Jujutsu Kaisen,Attack on Titan" --geo IN --timeframe "today 5-y"
```
`pytrends` is unofficial and can stop working or be rate-limited. Alternative: export CSV files from https://trends.google.com/trends/ and import them using the dashboard's Data Import page. For time-series modeling, use consistent collection windows and document the query/window; independently normalized series are not automatically comparable.

Expected canonical `data/raw/trends.csv` columns:
- `date`: date/week of observation
- `title`: search term/title
- `search_interest`: Google Trends index (0–100)
- optional `geo`: geographic scope, e.g. `IN`
- optional `source`: source identifier

Expected regional CSV `data/raw/regional_trends.csv`:
- `title`, `region`, `search_interest`, optional `date`, `geo`
Do not invent regional values. Google Trends regional exports may have different shapes; normalize them to this schema before import.

## Train model
```powershell
python -m safora train --horizon 4
```
The trainer creates `models/demand_model.joblib` and `reports/model_metrics.json` when sufficient real observations exist. It uses a chronological split and compares the model with a last-observation baseline. Inspect metrics before using predictions.

## Run tests
```powershell
pytest -q
```

## Suggested Git branches
Create a remote repository and use the branch layout below. Git does not store branches as directories; branches are parallel versions of the same repository.
- `main`: stable integrated application
- `feature/data-ingestion`: AniList and Trends collectors
- `feature/ml-pipeline`: features, training, evaluation, forecast
- `feature/sentiment`: optional sentiment pipeline
- `feature/dashboard`: Streamlit UI and visualization
- `feature/region-simulator`: regional analysis and what-if simulator

Example:
```bash
git init
git add .
git commit -m "chore: initial SAFORA scaffold"
git branch feature/data-ingestion
git branch feature/ml-pipeline
git branch feature/sentiment
git branch feature/dashboard
git branch feature/region-simulator
```
Recommended team workflow: checkout one feature branch per contributor, commit work, then merge through pull requests after tests pass.

## Repository structure
```text
SAFORA/
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
├── data/
│   ├── raw/.gitkeep
│   └── processed/.gitkeep
├── models/.gitkeep
├── reports/.gitkeep
├── safora/
│   ├── __init__.py
│   ├── config.py
│   ├── data_sources.py
│   ├── features.py
│   ├── modeling.py
│   ├── explainability.py
│   ├── scoring.py
│   └── cli.py
└── tests/
    ├── test_features.py
    └── test_scoring.py
```

## Roadmap
1. Collect sufficient title-level weekly search observations.
2. Validate title/term matching and query consistency.
3. Add permitted review data and manually validate sentiment.
4. Add regional observations when source data supports them.
5. Expand to manga/music/gaming only after source coverage is verified.
6. Consider a database and scheduled ingestion after the prototype is stable.
