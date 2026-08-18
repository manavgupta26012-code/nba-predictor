"""
collect_schedule.py
--------------------
Pulls the official NBA season schedule (past AND future games) using
nba_api's ScheduleLeagueV2 endpoint. This is what powers the "Upcoming
Games" view -- unlike LeagueGameLog (collect_data.py), which only returns
games that have already been played, this endpoint includes the full
announced schedule.

IMPORTANT: Talks to stats.nba.com -- run on your own machine, not in a
sandboxed environment.

Usage:
    python src/collect_schedule.py --season 2026-27
"""

import argparse
import pandas as pd
from pathlib import Path

from nba_api.stats.endpoints import scheduleleaguev2

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def parse_schedule(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw ScheduleLeagueV2 response into the columns the rest of
    this project expects. Excludes Preseason games (different rosters/
    rotations, not representative of real-season form) but KEEPS everything
    else -- blank-labeled games are regular season, and Emirates NBA Cup
    group-play games count in the standings and are worth predicting too.
    """
    df = raw.copy()
    df = df[df["gameLabel"] != "Preseason"].copy()

    df["GAME_DATE"] = pd.to_datetime(df["gameDate"])
    out = pd.DataFrame({
        "GAME_ID": df["gameId"],
        "GAME_DATE": df["GAME_DATE"],
        "GAME_DATETIME_EST": pd.to_datetime(df["gameDateTimeEst"], errors="coerce"),
        "HOME_TEAM_ABBREVIATION": df["homeTeam_teamTricode"],
        "AWAY_TEAM_ABBREVIATION": df["awayTeam_teamTricode"],
        "HOME_TEAM_ID": df["homeTeam_teamId"],
        "AWAY_TEAM_ID": df["awayTeam_teamId"],
        "GAME_LABEL": df["gameLabel"].replace("", "Regular Season"),
        "GAME_STATUS": df["gameStatus"],  # 1 = scheduled, 2 = live, 3 = final
        "HOME_SCORE": df["homeTeam_score"],
        "AWAY_SCORE": df["awayTeam_score"],
        "ARENA": df["arenaName"],
    })
    return out.sort_values("GAME_DATE").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", required=True, help="e.g. --season 2026-27")
    args = parser.parse_args()

    print(f"Fetching schedule for {args.season}...")
    sched = scheduleleaguev2.ScheduleLeagueV2(league_id="00", season=args.season)
    frames = sched.get_data_frames()

    biggest = max(frames, key=len)
    if "gameLabel" in biggest.columns:
        print("--- Game type breakdown (gameLabel) ---")
        print(biggest["gameLabel"].value_counts(dropna=False).to_string())
        print()
    if "gameDate" in biggest.columns:
        dates = pd.to_datetime(biggest["gameDate"], errors="coerce")
        print(f"Date range in this response: {dates.min()} to {dates.max()}")
        print()

    raw_path = DATA_DIR / "raw_schedule.csv"
    biggest.to_csv(raw_path, index=False)

    clean = parse_schedule(biggest)
    clean_path = DATA_DIR / "schedule.csv"
    clean.to_csv(clean_path, index=False)
    print(f"Saved raw -> {raw_path} ({len(biggest)} rows)")
    print(f"Saved clean schedule -> {clean_path} ({len(clean)} rows, Preseason excluded)")

    total_games_expected = 1230  # roughly, for a full 30-team regular season
    if len(clean) < total_games_expected * 0.5:
        print(f"\nNote: only {len(clean)} games found -- this looks like a PARTIAL "
              f"schedule release (full season is ~{total_games_expected} games). "
              f"Re-run this script once the NBA publishes the complete schedule.")


if __name__ == "__main__":
    main()
