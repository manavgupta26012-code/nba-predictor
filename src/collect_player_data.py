"""
collect_player_data.py
-----------------------
Pulls historical NBA PLAYER-level game logs using nba_api -- one row per
player per game, needed to detect when a team's best scorer sits out.

Uses the same LeagueGameLog endpoint as collect_data.py, just with
player_or_team_abbreviation="P" instead of "T", so it's just as efficient
(one API call per season, not one call per game).

IMPORTANT: This script talks to stats.nba.com. It must be run on your own
machine / normal internet connection (it will NOT work in a sandboxed
environment with restricted network access).

Usage:
    python src/collect_player_data.py --seasons 2021-22 2022-23 2023-24
"""

import argparse
import time
import pandas as pd
from pathlib import Path

from nba_api.stats.endpoints import leaguegamelog

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def fetch_season_player_log(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    print(f"Fetching player game logs for {season} ({season_type})...")
    log = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star=season_type,
        player_or_team_abbreviation="P",  # P = player-level rows
    )
    df = log.get_data_frames()[0]
    df["SEASON"] = season
    time.sleep(1.0)  # be polite to the API, avoid rate-limit blocks
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", required=True,
                         help="e.g. --seasons 2021-22 2022-23 2023-24 (use the SAME seasons as collect_data.py)")
    args = parser.parse_args()

    frames = []
    for season in args.seasons:
        try:
            frames.append(fetch_season_player_log(season))
        except Exception as e:
            print(f"  Failed to fetch {season}: {e}")

    if not frames:
        raise RuntimeError("No seasons were successfully fetched.")

    player_rows = pd.concat(frames, ignore_index=True)
    out_path = DATA_DIR / "raw_player_game_logs.csv"
    player_rows.to_csv(out_path, index=False)
    print(f"Saved -> {out_path} ({len(player_rows)} player-game rows)")


if __name__ == "__main__":
    main()
