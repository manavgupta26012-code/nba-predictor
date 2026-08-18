"""
diagnose_standings.py
----------------------
Checks whether the projected standings are being computed from a
reasonable amount of real game data, or from a too-small/unbalanced
sample -- the most likely cause if standings look way off.

Usage:
    python src/diagnose_standings.py
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    schedule = pd.read_csv(DATA_DIR / "schedule.csv", parse_dates=["GAME_DATE"])
    print(f"Total games in schedule.csv: {len(schedule)}")
    print(f"\nBreakdown by GAME_LABEL:")
    print(schedule["GAME_LABEL"].value_counts().to_string())

    # NOTE: Emirates NBA Cup games count toward standings (confirmed via
    # NBA.com) except the single Championship game. We don't yet have a
    # reliable way to identify that one specific game, so we include all
    # NBA Cup games here -- a small, harmless overcount for at most 2
    # teams, rather than excluding all 67 for all 30 teams.
    playable = schedule[schedule["GAME_LABEL"] != "Preseason"]
    print(f"\nGames remaining after excluding Preseason only "
          f"(this is what compute_standings actually uses): {len(playable)}")

    if len(playable) == 0:
        print("\n>>> ZERO games available for standings. This alone would make "
              "standings meaningless -- everything is a 0-0 placeholder or missing entirely.")
        return

    print(f"\nGame count per team in that filtered set:")
    home_counts = playable["HOME_TEAM_ABBREVIATION"].value_counts()
    away_counts = playable["AWAY_TEAM_ABBREVIATION"].value_counts()
    total_counts = home_counts.add(away_counts, fill_value=0).sort_values(ascending=False)
    print(total_counts.to_string())

    print(f"\nMost games for one team: {total_counts.max():.0f}")
    print(f"Fewest games for one team: {total_counts.min():.0f}")
    print(f"Teams with ZERO games in this set: {30 - len(total_counts)}")

    if total_counts.max() < 20:
        print("\n>>> LIKELY CAUSE FOUND: even the team with the most games has fewer than "
              "20 out of a real 82-game season. Standings computed from this few games per "
              "team will look extreme/unreliable -- a couple of close losses or wins can "
              "swing a tiny sample's win% dramatically. This isn't a bug in the model; it's "
              "simply too little schedule data to project a full season from yet.")
    elif total_counts.max() - total_counts.min() > 15:
        print("\n>>> LIKELY CAUSE FOUND: large imbalance in games per team "
              f"({total_counts.min():.0f} to {total_counts.max():.0f}). Teams with very few "
              "games will have standings driven by a handful of results rather than a full "
              "season's worth of signal.")
    else:
        print("\nGame counts look reasonably balanced -- if standings still look off, "
              "the issue is more likely in the model/prediction logic itself, not schedule "
              "completeness. Worth running the backtest-against-a-real-past-season check next.")


if __name__ == "__main__":
    main()
