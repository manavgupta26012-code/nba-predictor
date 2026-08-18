"""
collect_current_rosters.py
----------------------------
Pulls each team's ACTUAL CURRENT roster using nba_api. This fixes a real
bug: without this, "who is this team's top scorer" was inferred purely
from historical box scores, which misattributes any player who has since
been traded, signed elsewhere, or retired -- e.g. after an offseason move,
the old logic would keep crediting a player to his former team.

Team IDs come from nba_api's bundled static data (no network call needed).
Only the per-team roster call hits stats.nba.com, so this needs to run on
your own machine.

Usage:
    python src/collect_current_rosters.py --season 2026-27
"""

import argparse
import time
import pandas as pd
from pathlib import Path

from nba_api.stats.static import teams as static_teams
from nba_api.stats.endpoints import commonteamroster

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", required=True, help="e.g. --season 2026-27")
    args = parser.parse_args()

    all_teams = static_teams.get_teams()  # bundled offline, no network call
    print(f"Found {len(all_teams)} teams. Fetching rosters for {args.season}...")

    frames = []
    for t in all_teams:
        try:
            roster = commonteamroster.CommonTeamRoster(team_id=t["id"], season=args.season)
            df = roster.get_data_frames()[0]
            df["TEAM_ABBREVIATION"] = t["abbreviation"]
            frames.append(df)
            print(f"  {t['abbreviation']}: {len(df)} players")
        except Exception as e:
            print(f"  {t['abbreviation']}: FAILED -- {e}")
        time.sleep(0.8)  # be polite to the API

    if not frames:
        raise RuntimeError("No rosters were successfully fetched.")

    all_rosters = pd.concat(frames, ignore_index=True)

    print(f"\nColumns returned: {all_rosters.columns.tolist()}")
    print(all_rosters.head(3).to_string())

    out_path = DATA_DIR / "current_rosters.csv"
    all_rosters.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path} ({len(all_rosters)} player-team rows)")


if __name__ == "__main__":
    main()
