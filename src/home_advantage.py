"""
home_advantage.py
-------------------
Computes each team's real home vs. away performance split -- win rate and
average point margin at home vs. on the road. Uses ALL collected games
(not just the last 10, unlike most other rolling stats in this project),
since home/away splits already have half the sample size of a team's
full record, and cutting to a shorter recent window would make the
per-team numbers noisier, not more current -- home-court advantage is a
fairly stable team characteristic across a season, not something that
swings week to week the way "hot streak" stats do.
"""

import pandas as pd


def compute_home_away_splits(team_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per team: HOME_WIN_PCT, AWAY_WIN_PCT, HOME_AVG_MARGIN,
    AWAY_AVG_MARGIN, and HOME_ADVANTAGE (the gap between home and away win
    rate -- the single number that answers "how much does playing at home
    actually help this specific team").
    """
    df = team_rows.copy()
    df["IS_HOME"] = df["MATCHUP"].str.contains("vs.", regex=False)
    df["WIN"] = (df["WL"] == "W").astype(int)

    rows = []
    for team, g in df.groupby("TEAM_ABBREVIATION"):
        home = g[g["IS_HOME"]]
        away = g[~g["IS_HOME"]]
        rows.append(dict(
            TEAM_ABBREVIATION=team,
            HOME_WIN_PCT=home["WIN"].mean() if len(home) else None,
            AWAY_WIN_PCT=away["WIN"].mean() if len(away) else None,
            HOME_AVG_MARGIN=home["PLUS_MINUS"].mean() if len(home) else None,
            AWAY_AVG_MARGIN=away["PLUS_MINUS"].mean() if len(away) else None,
            HOME_GAMES=len(home),
            AWAY_GAMES=len(away),
        ))

    out = pd.DataFrame(rows)
    out["HOME_ADVANTAGE"] = out["HOME_WIN_PCT"] - out["AWAY_WIN_PCT"]
    return out.set_index("TEAM_ABBREVIATION")
