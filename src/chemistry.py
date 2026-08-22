"""
chemistry.py
------------
"Chemistry" in the full sense -- lineup familiarity, on-court rapport,
who plays well next to whom -- isn't measurable with the data this
project collects (no lineup or play-by-play data). What IS measurable
and genuinely meaningful is BALL MOVEMENT: teams with better chemistry
tend to share the ball more (a higher share of made baskets are
assisted) and turn it over less (fewer miscommunications). This is a
standard, real analytics concept -- not a fabricated substitute -- but
it's honestly a proxy, not the full picture, and the UI says so.

Percentiles are computed against the rest of the league THIS season,
since a raw "68% of shots assisted" number is hard to interpret without
knowing whether that's typical or exceptional.
"""

import pandas as pd


def compute_chemistry_metrics(team_rows: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Returns one row per team with their CURRENT (last `window` games)
    assist rate and turnover rate, plus percentile ranks against the rest
    of the league for context.
    """
    df = team_rows.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])
    grouped = df.groupby("TEAM_ABBREVIATION", group_keys=False)

    # Inclusive of the most recent game (not shifted) -- same convention
    # as get_latest_team_snapshot in predict_matchup.py, since this
    # represents "current form" for a future game, not a leak-free
    # historical training feature.
    df["ROLL_AST"] = grouped["AST"].apply(lambda s: s.rolling(window, min_periods=3).mean())
    df["ROLL_FGM"] = grouped["FGM"].apply(lambda s: s.rolling(window, min_periods=3).mean())
    df["ROLL_TOV"] = grouped["TOV"].apply(lambda s: s.rolling(window, min_periods=3).mean())

    snapshot = df.sort_values("GAME_DATE").groupby("TEAM_ABBREVIATION").tail(1).copy()
    snapshot["AST_PER_FGM"] = snapshot["ROLL_AST"] / snapshot["ROLL_FGM"]

    # Percentile rank vs. the rest of the league (0-100). Higher assist
    # rate = better (more ball movement). Lower turnovers = better, so we
    # rank ascending and flip it (100 - pct) for turnovers specifically.
    snapshot["AST_PERCENTILE"] = (snapshot["AST_PER_FGM"].rank(pct=True) * 100).round(0)
    snapshot["TOV_PERCENTILE"] = (100 - snapshot["ROLL_TOV"].rank(pct=True) * 100).round(0)

    out = snapshot[["TEAM_ABBREVIATION", "AST_PER_FGM", "ROLL_TOV", "AST_PERCENTILE", "TOV_PERCENTILE"]]
    return out.set_index("TEAM_ABBREVIATION")
