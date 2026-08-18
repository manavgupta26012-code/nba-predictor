"""
collect_data.py
----------------
Pulls historical NBA game logs and team box scores using nba_api.

IMPORTANT: This script talks to stats.nba.com. It must be run on your own
machine / normal internet connection (it will NOT work in a sandboxed
environment with restricted network access).

Usage:
    python src/collect_data.py --seasons 2019-20 2020-21 2021-22 2022-23 2023-24
"""

import argparse
import time
import pandas as pd
from pathlib import Path

from nba_api.stats.endpoints import leaguegamelog

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def fetch_season_game_log(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """
    Fetch team-level game log for a single season.
    season format: '2023-24'
    """
    print(f"Fetching {season} ({season_type})...")
    log = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star=season_type,
        player_or_team_abbreviation="T",  # T = team-level rows
    )
    df = log.get_data_frames()[0]
    df["SEASON"] = season
    time.sleep(1.0)  # be polite to the API, avoid rate-limit blocks
    return df


def build_game_dataset(seasons: list[str]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        try:
            frames.append(fetch_season_game_log(season))
        except Exception as e:
            print(f"  Failed to fetch {season}: {e}")
    if not frames:
        raise RuntimeError("No seasons were successfully fetched.")
    return pd.concat(frames, ignore_index=True)


def reshape_to_one_row_per_game(team_rows: pd.DataFrame) -> pd.DataFrame:
    """
    nba_api's leaguegamelog gives one row per TEAM per game (so 2 rows/game).
    This reshapes it into one row per game with home/away columns, which is
    much easier to build features and a model on.
    """
    df = team_rows.copy()

    # MATCHUP looks like "LAL vs. BOS" (home) or "LAL @ BOS" (away)
    df["IS_HOME"] = df["MATCHUP"].str.contains("vs.")

    home = df[df["IS_HOME"]].copy()
    away = df[~df["IS_HOME"]].copy()

    home = home.add_prefix("HOME_")
    away = away.add_prefix("AWAY_")

    merged = home.merge(
        away,
        left_on="HOME_GAME_ID",
        right_on="AWAY_GAME_ID",
        suffixes=("", ""),
    )

    merged = merged.rename(columns={
        "HOME_GAME_ID": "GAME_ID",
        "HOME_GAME_DATE": "GAME_DATE",
        "HOME_SEASON": "SEASON",
    })
    merged["GAME_DATE"] = pd.to_datetime(merged["GAME_DATE"])
    merged["HOME_WIN"] = (merged["HOME_WL"] == "W").astype(int)

    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", required=True,
                         help="e.g. --seasons 2021-22 2022-23 2023-24")
    args = parser.parse_args()

    team_rows = build_game_dataset(args.seasons)
    team_rows.to_csv(DATA_DIR / "raw_team_game_logs.csv", index=False)
    print(f"Saved raw team game logs -> {DATA_DIR / 'raw_team_game_logs.csv'}")

    games = reshape_to_one_row_per_game(team_rows)
    games.to_csv(DATA_DIR / "games_one_row_per_game.csv", index=False)
    print(f"Saved reshaped games -> {DATA_DIR / 'games_one_row_per_game.csv'}")
    print(f"Total games: {len(games)}")


if __name__ == "__main__":
    main()
