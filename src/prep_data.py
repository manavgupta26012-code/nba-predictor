"""
prep_data.py
------------
Merges the reshaped NBA game data (from collect_data.py) with a historical
betting-odds CSV, and produces one clean file ready for feature engineering.

WHERE TO GET ODDS DATA (free options):
  1. Kaggle "NBA Betting Odds" datasets (search "nba odds kaggle") - usually
     already close to the schema expected below.
  2. sportsbookreviewsonline.com/scoresoddsarchives/nba/ - per-season Excel
     files (needs a bit more reshaping; see NOTE below).

EXPECTED INPUT ODDS CSV SCHEMA (rename your columns to match this before
running, or edit COLUMN_MAP below):
    date        - game date (any parseable format)
    home_team   - home team name or abbreviation
    away_team   - away team name or abbreviation
    home_ml     - home moneyline odds (American odds, e.g. -150)
    away_ml     - away moneyline odds
    spread      - home team spread (negative = home favored)
    total       - over/under total points

Usage:
    python src/prep_data.py --odds data/raw_odds.csv
"""

import argparse
import pandas as pd
from pathlib import Path

from team_names import to_abbrev

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# If your odds CSV has different column names, map them here:
COLUMN_MAP = {
    "home": "home_team",
    "away": "away_team",
    "moneyline_home": "home_ml",
    "moneyline_away": "away_ml",
}


def american_odds_to_prob(odds: float) -> float:
    """Convert American moneyline odds to implied win probability."""
    if pd.isna(odds):
        return None
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)


def load_odds(path: str) -> pd.DataFrame:
    odds = pd.read_csv(path)
    if COLUMN_MAP:
        odds = odds.rename(columns=COLUMN_MAP)

    required = {"date", "home_team", "away_team", "home_ml", "away_ml"}
    missing = required - set(odds.columns)
    if missing:
        raise ValueError(
            f"Odds CSV is missing columns {missing}. "
            f"Rename your columns or edit COLUMN_MAP in prep_data.py."
        )

    odds["date"] = pd.to_datetime(odds["date"])
    odds["HOME_ABBR"] = odds["home_team"].map(to_abbrev)
    odds["AWAY_ABBR"] = odds["away_team"].map(to_abbrev)

    unmatched = odds[odds["HOME_ABBR"].isna() | odds["AWAY_ABBR"].isna()]
    if len(unmatched):
        print(f"WARNING: {len(unmatched)} odds rows had unmatched team names. "
              f"Examples: {unmatched[['home_team','away_team']].drop_duplicates().head().to_dict('records')}")
        print("Add these to ALIASES in team_names.py, then re-run.")

    odds["home_win_prob_market"] = odds["home_ml"].apply(american_odds_to_prob)
    odds["away_win_prob_market"] = odds["away_ml"].apply(american_odds_to_prob)

    if "spread" in odds.columns:
        pass  # kept as-is
    return odds


def merge_games_and_odds(games: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    merged = games.merge(
        odds,
        left_on=["GAME_DATE", "HOME_TEAM_ABBREVIATION", "AWAY_TEAM_ABBREVIATION"],
        right_on=["date", "HOME_ABBR", "AWAY_ABBR"],
        how="inner",
    )
    print(f"Matched {len(merged)} / {len(games)} games to odds data "
          f"({len(merged) / max(len(games), 1):.1%})")
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", default=str(DATA_DIR / "games_one_row_per_game.csv"))
    parser.add_argument("--odds", required=True)
    parser.add_argument("--out", default=str(DATA_DIR / "games_with_odds.csv"))
    args = parser.parse_args()

    games = pd.read_csv(args.games, parse_dates=["GAME_DATE"])
    odds = load_odds(args.odds)
    merged = merge_games_and_odds(games, odds)
    merged.to_csv(args.out, index=False)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
