# NBA Win Probability Predictor — MVP v1

A working end-to-end pipeline: collect NBA data → merge betting odds →
engineer features → train a logistic regression model → explore results in
a Streamlit dashboard.

Every script in `src/` has been smoke-tested against synthetic data that
matches the real schema, so the logic (reshaping, rolling features with no
data leakage, time-based train/test split, dashboard explanations) is
verified. You just need to point it at real data.

## 1. Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Collect NBA game data

This calls `stats.nba.com` via `nba_api`, so it needs to run on your normal
machine/internet connection (not a sandboxed environment).

```bash
python src/collect_data.py --seasons 2021-22 2022-23 2023-24
```

This creates:
- `data/raw_team_game_logs.csv` — one row per team per game
- `data/games_one_row_per_game.csv` — one row per game, home vs away columns

If `nba_api` gets rate-limited (HTTP errors), just re-run — it saves whatever
it successfully fetched and prints which seasons failed.

## 3. Get historical betting odds

Free options:
- Search Kaggle for **"NBA betting odds"** or **"NBA odds dataset"** — several
  are already close to the schema below.
- [sportsbookreviewsonline.com/scoresoddsarchives/nba](https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba/) —
  free per-season Excel files (needs a bit more reshaping than a Kaggle CSV).

Get your odds file into this schema (rename columns, or edit `COLUMN_MAP` in
`src/prep_data.py`):

| column      | meaning                                  |
|-------------|-------------------------------------------|
| date        | game date                                 |
| home_team   | home team name or abbreviation            |
| away_team   | away team name or abbreviation            |
| home_ml     | home moneyline (American odds, e.g. -150) |
| away_ml     | away moneyline                            |
| spread      | home team spread (optional)               |
| total       | over/under total (optional)               |

Save it as `data/raw_odds.csv`, then:

```bash
python src/prep_data.py --odds data/raw_odds.csv
```

If team names don't match (you'll see a warning), add the missing aliases to
`ALIASES` in `src/team_names.py` and re-run.

**Skipping odds for now?** That's fine — `build_features.py` will fall back
to `games_one_row_per_game.csv` and you'll just lose the "market comparison"
panel in the dashboard. You can add odds later.

## 4. Build features

```bash
python src/build_features.py
```

Creates `data/model_ready.csv`. All rolling stats use `shift(1)` before the
rolling window, so a game's features only ever use *prior* games — no
leakage from the game being predicted.

## 5. Train the model

```bash
python src/train_model.py
```

Trains on earlier games, tests on later games (time-based split, not
random — random splits leak future information in a time-series problem
like this). Prints accuracy, log loss, ROC AUC, and a "always pick home
team" baseline your model needs to beat. Saves `models/win_probability_model.pkl`.

## 6. Run the dashboard

```bash
streamlit run app.py
```

Pick a game, see win probability, confidence, top 3 explanatory factors,
and recent form for both teams.

## Project structure

```
nba-predictor/
├── src/
│   ├── collect_data.py      # nba_api → raw game logs
│   ├── team_names.py        # name normalization across data sources
│   ├── prep_data.py         # merge games + odds
│   ├── build_features.py    # rolling stats, no data leakage
│   └── train_model.py       # logistic regression, time-based split
├── tests/
│   └── generate_synthetic_data.py   # sanity-check the pipeline offline
├── app.py                   # Streamlit dashboard
├── data/                    # (created by the scripts)
├── models/                  # (created by train_model.py)
└── requirements.txt
```

## What's next (v2)

Once this is working end to end, natural additions in order of effort:
1. **Momentum Score** — weighted recent win margin, not just win %
2. **Fatigue Index** — games played in last 5/7 days, back-to-backs
3. Swap logistic regression for **XGBoost/LightGBM** and compare
4. **SHAP values** for per-game feature importance (nicer than raw
   coefficients, and works with non-linear models)
5. Live/upcoming-game predictions via `nba_api`'s scoreboard endpoints

Don't add these until v1 is producing sane, validated predictions on
historical data — a stable pipeline is much easier to extend than to debug
while also adding complexity.
