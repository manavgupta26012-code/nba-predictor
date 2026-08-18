"""
collect_advanced_stats.py
--------------------------
Pulls two richer data sources than basic box scores:

1. LeagueDashTeamStats (Advanced) -- offensive/defensive rating, pace,
   effective FG%, true shooting%, rebound%, turnover% -- all normalized
   per-possession, which is how real analysts actually compare teams
   (raw points/rebounds are influenced by how fast a team plays; these
   aren't).

2. LeagueDashTeamShotLocations -- FG% and volume broken out by shot zone
   (restricted area, paint, mid-range, corner 3, above-the-break 3). This
   is what actually enables real "points in the paint" matchup analysis,
   which raw box scores can't give us.

Both support last_n_games, so a single call per endpoint gives every
team's CURRENT (last-10-games) form -- no per-game historical collection
needed, since this feeds the "current snapshot" matchup narratives, not
the trained model's historical features.

IMPORTANT: Talks to stats.nba.com -- run on your own machine.

Usage:
    python src/collect_advanced_stats.py
"""

import pandas as pd
from pathlib import Path

from nba_api.stats.endpoints import leaguedashteamstats, leaguedashteamshotlocations
from nba_api.stats.static import teams as static_teams

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

ADVANCED_COLS_TO_KEEP = [
    "TEAM_ID", "OFF_RATING", "DEF_RATING", "NET_RATING", "PACE",
    "EFG_PCT", "TS_PCT", "OREB_PCT", "DREB_PCT", "REB_PCT", "TM_TOV_PCT", "AST_PCT",
]

# Maps the raw nested zone names to short, code-friendly identifiers.
ZONE_NAME_MAP = {
    "Restricted Area": "RA",
    "In The Paint (Non-RA)": "PAINT_NONRA",
    "Mid-Range": "MIDRANGE",
    "Left Corner 3": "LC3",
    "Right Corner 3": "RC3",
    "Above the Break 3": "AB3",
    "Corner 3": "CORNER3",
}


def _team_id_to_abbrev():
    return {t["id"]: t["abbreviation"] for t in static_teams.get_teams()}


def main():
    id_to_abbrev = _team_id_to_abbrev()

    print("Fetching advanced team stats (last 10 games)...")
    adv = leaguedashteamstats.LeagueDashTeamStats(
        measure_type_detailed_defense="Advanced",
        last_n_games=10,
        per_mode_detailed="PerGame",
    )
    adv_df = adv.get_data_frames()[0]
    adv_df = adv_df[ADVANCED_COLS_TO_KEEP].copy()
    adv_df["TEAM_ABBREVIATION"] = adv_df["TEAM_ID"].map(id_to_abbrev)
    adv_path = DATA_DIR / "advanced_team_stats.csv"
    adv_df.to_csv(adv_path, index=False)
    print(f"Saved -> {adv_path} ({len(adv_df)} teams)")

    print("\nFetching shot location breakdown (last 10 games)...")
    shots = leaguedashteamshotlocations.LeagueDashTeamShotLocations(
        last_n_games=10,
        per_mode_detailed="PerGame",
    )
    shot_df = shots.get_data_frames()[0]

    # Flatten the MultiIndex columns (e.g. ('Restricted Area','FGM')) into
    # single flat names (RA_FGM) so this saves/reloads as a normal CSV.
    flat_cols = []
    for zone, stat in shot_df.columns:
        if zone == "":
            flat_cols.append(stat)  # TEAM_ID, TEAM_NAME
        else:
            short_zone = ZONE_NAME_MAP.get(zone, zone.upper().replace(" ", "_"))
            flat_cols.append(f"{short_zone}_{stat}")
    shot_df.columns = flat_cols

    shot_df["TEAM_ABBREVIATION"] = shot_df["TEAM_ID"].map(id_to_abbrev)

    # Combined "paint" = Restricted Area + non-RA paint, the two zones that
    # together make up "points in the paint."
    shot_df["PAINT_FGM"] = shot_df["RA_FGM"] + shot_df["PAINT_NONRA_FGM"]
    shot_df["PAINT_FGA"] = shot_df["RA_FGA"] + shot_df["PAINT_NONRA_FGA"]
    shot_df["PAINT_FG_PCT"] = (shot_df["PAINT_FGM"] / shot_df["PAINT_FGA"]).round(3)
    shot_df["PAINT_PTS"] = (shot_df["PAINT_FGM"] * 2).round(1)  # all paint shots are 2s

    shot_path = DATA_DIR / "team_shot_locations.csv"
    shot_df.to_csv(shot_path, index=False)
    print(f"Saved -> {shot_path} ({len(shot_df)} teams)")

    print("\nSample (Boston Celtics):")
    sample_cols = ["TEAM_ABBREVIATION", "RA_FG_PCT", "PAINT_NONRA_FG_PCT", "PAINT_FG_PCT", "PAINT_PTS", "MIDRANGE_FG_PCT", "CORNER3_FG_PCT"]
    print(shot_df[shot_df["TEAM_ABBREVIATION"] == "BOS"][sample_cols].to_string(index=False))


if __name__ == "__main__":
    main()
