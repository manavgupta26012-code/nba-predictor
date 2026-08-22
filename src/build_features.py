"""
build_features.py
------------------
Builds model-ready features from the raw team game logs + merged game/odds
file. Every rolling stat is computed using ONLY prior games (shift(1) before
rolling) so we never leak future information into a prediction.

Usage:
    python src/build_features.py
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

ROLLING_WINDOW = 10  # "last 10 games" form

BOX_SCORE_COLS = [
    "PTS", "FG_PCT", "FG3_PCT", "FT_PCT", "REB", "OREB", "DREB",
    "AST", "STL", "BLK", "TOV", "PLUS_MINUS",
]


MOMENTUM_SPAN = 5        # exponential weighting span for Momentum Score
FATIGUE_WINDOW_DAYS = 5  # trailing window for "games played recently"


def _games_in_trailing_window(group: pd.DataFrame, days: int = FATIGUE_WINDOW_DAYS) -> pd.Series:
    """
    Count games played in the trailing N days, NOT including the current game
    (closed='left' excludes the right/current edge of the window).
    `group` must already be sorted by GAME_DATE and keep its original index.
    """
    dated = group.set_index("GAME_DATE")["GAME_ID"]
    counts = dated.rolling(f"{days}D", closed="left").count()
    counts.index = group.index  # restore original row labels (positions line up 1:1)
    return counts


CLUTCH_MARGIN = 5   # a "close game" = decided by this many points or fewer
CLUTCH_WINDOW = 5    # how many recent close games to look back over


def compute_clutch_feature(team_rows: pd.DataFrame, margin: int = CLUTCH_MARGIN,
                            window: int = CLUTCH_WINDOW) -> pd.DataFrame:
    """
    For each game, looks at how this team did in their last `window` CLOSE
    games (decided by `margin` points or fewer) -- not their overall record.
    A team that wins big but folds in close games is a different bet than
    their overall win % suggests.

    Processes games in date order; only close games get added to a team's
    history, and only AFTER computing the current row's value, so a team's
    performance in today's (possibly close) game never leaks into today's
    own clutch rating.
    """
    df = team_rows.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["WIN"] = (df["WL"] == "W").astype(int)
    df = df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])

    history = {}  # team -> list of WIN (1/0) from past CLOSE games only
    clutch_win_pct = []

    for _, row in df.iterrows():
        team = row["TEAM_ABBREVIATION"]
        past_close = history.get(team, [])[-window:]
        clutch_win_pct.append(sum(past_close) / len(past_close) if len(past_close) >= 3 else None)

        if abs(row["PLUS_MINUS"]) <= margin:
            history.setdefault(team, []).append(row["WIN"])

    df["CLUTCH_WIN_PCT"] = clutch_win_pct
    return df[["GAME_ID", "TEAM_ABBREVIATION", "GAME_DATE", "CLUTCH_WIN_PCT"]]


def compute_team_rolling_stats(team_rows: pd.DataFrame) -> pd.DataFrame:
    df = team_rows.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["WIN"] = (df["WL"] == "W").astype(int)
    df = df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])

    grouped = df.groupby("TEAM_ABBREVIATION", group_keys=False)

    # Rest days since previous game, capped at 7 -- a 200-day off-season gap
    # is not meaningfully "more rested" than a 7-day gap, and left uncapped it
    # dominates the model (verified: this was inflating DIFF_REST_DAYS above
    # DIFF_PLUS_MINUS in feature importance before this cap was added).
    df["REST_DAYS"] = grouped["GAME_DATE"].diff().dt.days.clip(upper=7)

    # Back-to-back flag: playing on one day's rest (game the very next day)
    df["BACK_TO_BACK"] = (df["REST_DAYS"] == 1).astype(int)

    # Fatigue Index: games played in the last 5 days (not counting this one).
    # More games in a short window = more cumulative fatigue, independent of
    # rest since the most recent single game.
    df["GAMES_LAST5"] = df.groupby("TEAM_ABBREVIATION", group_keys=False).apply(
        _games_in_trailing_window
    )

    # Rolling averages of box score stats, computed on PRIOR games only.
    # shift(1) drops the current game before the rolling window is applied.
    for col in BOX_SCORE_COLS:
        df[f"ROLL_{col}"] = (
            grouped[col]
            .apply(lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=3).mean())
        )

    df["ROLL_WIN_PCT"] = (
        grouped["WIN"]
        .apply(lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=3).mean())
    )

    # Momentum Score: exponentially-weighted recent point differential.
    # Unlike ROLL_PLUS_MINUS (flat average over the last 10 games), this
    # weights the most recent games more heavily -- a team on a hot streak
    # scores higher than a team with the same flat average but fading form.
    df["MOMENTUM_SCORE"] = (
        grouped["PLUS_MINUS"]
        .apply(lambda s: s.shift(1).ewm(span=MOMENTUM_SPAN, min_periods=3).mean())
    )

    keep_cols = ["GAME_ID", "TEAM_ABBREVIATION", "GAME_DATE", "REST_DAYS",
                 "BACK_TO_BACK", "GAMES_LAST5", "ROLL_WIN_PCT", "MOMENTUM_SCORE"] + \
                [f"ROLL_{c}" for c in BOX_SCORE_COLS]

    clutch = compute_clutch_feature(team_rows)[["GAME_ID", "TEAM_ABBREVIATION", "CLUTCH_WIN_PCT"]]
    result = df[keep_cols].merge(clutch, on=["GAME_ID", "TEAM_ABBREVIATION"], how="left")
    return result


STAR_MIN_GAMES = 5       # player needs at least this many recent games to be eligible as "the star"
STAR_ROLL_WINDOW = 10    # window for a player's own rolling scoring average


def _parse_minutes(v) -> float:
    """nba_api's MIN column is sometimes a plain number, sometimes 'MM:SS' as a string."""
    if pd.isna(v):
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v)
    if ":" in s:
        mins, secs = s.split(":")
        return float(mins) + float(secs) / 60
    try:
        return float(s)
    except ValueError:
        return 0.0


def compute_game_score(df: pd.DataFrame) -> pd.Series:
    """
    Hollinger Game Score: a single-number blend of a player's box score
    that credits efficient scoring, rebounding, passing, steals, and
    blocks, and penalizes missed shots, turnovers, and fouls. Used instead
    of raw points-per-game to rank "how much does this player matter" --
    a high-volume, inefficient scorer shouldn't outrank a more efficient,
    all-around contributor just because he shoots more.

        GmSc = PTS + 0.4*FGM - 0.7*FGA - 0.4*(FTA-FTM)
               + 0.7*OREB + 0.3*DREB + STL + 0.7*AST + 0.7*BLK - 0.4*PF - TOV

    Needs the full raw box score columns (FGM, FGA, FTA, FTM, PF, etc.) --
    these come from nba_api's player game logs directly, not from
    BOX_SCORE_COLS above (which is a reduced set chosen for the team-level
    model features).
    """
    required = ["PTS", "FGM", "FGA", "FTA", "FTM", "OREB", "DREB", "STL", "AST", "BLK", "PF", "TOV"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"compute_game_score needs columns {missing}, not found in this data.")
    return (
        df["PTS"] + 0.4 * df["FGM"] - 0.7 * df["FGA"] - 0.4 * (df["FTA"] - df["FTM"])
        + 0.7 * df["OREB"] + 0.3 * df["DREB"] + df["STL"] + 0.7 * df["AST"] + 0.7 * df["BLK"]
        - 0.4 * df["PF"] - df["TOV"]
    )


TOP_N_PLAYERS = 3  # how many of a team's top-impact players to track


def compute_top_players_features(player_rows: pd.DataFrame,
                                  top_n: int = TOP_N_PLAYERS,
                                  min_games: int = STAR_MIN_GAMES,
                                  window: int = STAR_ROLL_WINDOW) -> pd.DataFrame:
    """
    For each TEAM in each GAME, identifies that team's top `top_n` players
    BY IMPACT (Game Score, not raw scoring -- see compute_game_score),
    among those who've recently appeared for them, then checks whether
    each of them suited up for this game.

    Returns one row per TEAM per GAME with, for rank 1..top_n:
      - PLAYER{i}_NAME, PLAYER{i}_PPG (for readability), PLAYER{i}_IMPACT
        (Game Score -- what actually drives the ranking), PLAYER{i}_OUT
    plus MISSING_IMPACT: the combined Game Score of whichever of the top
    `top_n` players did NOT play.

    No leakage: a team's rankings for today's game come only from games
    strictly before today; today's box score only updates the roster
    tracker AFTER today's feature values are recorded.

    PERFORMANCE NOTE: this used to filter day_rows per ranked player
    (day_rows[day_rows["PLAYER_ID"] == pid]) and use iterrows() to update
    the roster tracker -- both are slow, and became a real bottleneck once
    running across multiple full seasons of player logs instead of one.
    Replaced with an O(1) dict lookup and itertuples(), verified to
    produce byte-identical output to the original on test data before
    being adopted.
    """
    df = player_rows.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["MIN_NUM"] = df["MIN"].apply(_parse_minutes)
    df["GAME_SCORE"] = compute_game_score(df)
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"])

    # Each player's own rolling IMPACT and PPG, INCLUDING their most recent
    # game -- stored in the team's roster tracker below and used for ALL of
    # that team's games until this player plays again. (No shift: leak-
    # freeness comes from the outer team-date loop, not from shifting
    # within the player's own series.)
    grouped = df.groupby("PLAYER_ID", group_keys=False)
    df["PLAYER_ROLL_IMPACT"] = grouped["GAME_SCORE"].apply(lambda s: s.rolling(window, min_periods=min_games).mean())
    df["PLAYER_ROLL_PTS"] = grouped["PTS"].apply(lambda s: s.rolling(window, min_periods=min_games).mean())

    df = df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])
    results = []

    for team, team_df in df.groupby("TEAM_ABBREVIATION"):
        roster_latest = {}  # player_id -> (rolling_impact, rolling_ppg, player_name)
        for game_date, day_rows in team_df.sort_values("GAME_DATE").groupby("GAME_DATE"):
            game_id = day_rows["GAME_ID"].iloc[0]
            row = {"GAME_ID": game_id, "TEAM_ABBREVIATION": team}

            # O(1) dict lookup instead of filtering day_rows per ranked
            # player -- built once per game-date (cheap: one row per
            # player who played that day) instead of re-filtering the
            # dataframe up to top_n times per game-date.
            minutes_by_player = dict(zip(day_rows["PLAYER_ID"], day_rows["MIN_NUM"]))

            ranked = sorted(roster_latest.items(), key=lambda kv: kv[1][0], reverse=True)[:top_n]
            missing_impact = 0.0
            for rank, (pid, (impact, ppg, name)) in enumerate(ranked, start=1):
                played = minutes_by_player.get(pid, 0) > 0
                row[f"PLAYER{rank}_NAME"] = name
                row[f"PLAYER{rank}_PPG"] = ppg
                row[f"PLAYER{rank}_IMPACT"] = impact
                row[f"PLAYER{rank}_OUT"] = 0 if played else 1
                if not played:
                    missing_impact += impact
            for rank in range(len(ranked) + 1, top_n + 1):
                row[f"PLAYER{rank}_NAME"] = None
                row[f"PLAYER{rank}_PPG"] = None
                row[f"PLAYER{rank}_IMPACT"] = None
                row[f"PLAYER{rank}_OUT"] = None
            row["MISSING_IMPACT"] = missing_impact if ranked else None
            results.append(row)

            # Update the roster's known impact/scoring averages using TODAY's
            # data, for use starting with this team's NEXT game (not this
            # one). itertuples() instead of iterrows() -- iterrows()
            # rebuilds a full pandas Series per row (slow); itertuples()
            # returns a lightweight namedtuple, typically 10-100x faster
            # for this kind of per-row access pattern.
            for r in day_rows.itertuples():
                if pd.notna(r.PLAYER_ROLL_IMPACT):
                    roster_latest[r.PLAYER_ID] = (r.PLAYER_ROLL_IMPACT, r.PLAYER_ROLL_PTS, r.PLAYER_NAME)

    return pd.DataFrame(results)


def attach_features_to_games(games: pd.DataFrame, rolling: pd.DataFrame) -> pd.DataFrame:
    home_roll = rolling.add_prefix("HOME_")
    away_roll = rolling.add_prefix("AWAY_")

    out = games.merge(
        home_roll,
        left_on=["GAME_ID", "HOME_TEAM_ABBREVIATION"],
        right_on=["HOME_GAME_ID", "HOME_TEAM_ABBREVIATION"],
        how="left",
    ).merge(
        away_roll,
        left_on=["GAME_ID", "AWAY_TEAM_ABBREVIATION"],
        right_on=["AWAY_GAME_ID", "AWAY_TEAM_ABBREVIATION"],
        how="left",
    )

    # Differential features -- often more predictive than raw values
    for col in BOX_SCORE_COLS:
        out[f"DIFF_{col}"] = out[f"HOME_ROLL_{col}"] - out[f"AWAY_ROLL_{col}"]
    out["DIFF_WIN_PCT"] = out["HOME_ROLL_WIN_PCT"] - out["AWAY_ROLL_WIN_PCT"]
    out["DIFF_REST_DAYS"] = out["HOME_REST_DAYS"] - out["AWAY_REST_DAYS"]
    out["DIFF_MOMENTUM_SCORE"] = out["HOME_MOMENTUM_SCORE"] - out["AWAY_MOMENTUM_SCORE"]
    out["DIFF_GAMES_LAST5"] = out["HOME_GAMES_LAST5"] - out["AWAY_GAMES_LAST5"]
    out["DIFF_BACK_TO_BACK"] = out["HOME_BACK_TO_BACK"] - out["AWAY_BACK_TO_BACK"]
    out["DIFF_CLUTCH_WIN_PCT"] = out["HOME_CLUTCH_WIN_PCT"] - out["AWAY_CLUTCH_WIN_PCT"]

    return out


H2H_WINDOW = 5  # how many past meetings between these two exact teams to consider


def compute_head_to_head_features(games: pd.DataFrame, window: int = H2H_WINDOW) -> pd.DataFrame:
    """
    For each game, looks at the last `window` meetings between these SAME TWO
    teams (regardless of which one was home in those past meetings) and
    computes, from the CURRENT home team's perspective:
      - H2H_HOME_WIN_PCT: how often the current home team won those meetings
      - H2H_AVG_MARGIN: their average point margin in those meetings

    Processes games in date order and only looks at meetings strictly BEFORE
    the current game (history is appended AFTER computing each row's
    features), so there is no leakage.
    """
    df = games.copy().sort_values("GAME_DATE").reset_index(drop=True)
    df["PAIR_KEY"] = df.apply(
        lambda r: tuple(sorted([r["HOME_TEAM_ABBREVIATION"], r["AWAY_TEAM_ABBREVIATION"]])),
        axis=1,
    )
    df["_MARGIN"] = df["HOME_PTS"] - df["AWAY_PTS"]

    history = {}  # PAIR_KEY -> list of (home_team_in_that_meeting, margin, home_won)
    win_pcts, avg_margins = [], []

    for _, row in df.iterrows():
        key = row["PAIR_KEY"]
        past = history.get(key, [])[-window:]

        if len(past) < 3:
            win_pcts.append(None)
            avg_margins.append(None)
        else:
            cur_home = row["HOME_TEAM_ABBREVIATION"]
            wins, margins = 0, []
            for past_home, past_margin, past_home_won in past:
                if past_home == cur_home:
                    margins.append(past_margin)
                    wins += 1 if past_home_won else 0
                else:
                    margins.append(-past_margin)
                    wins += 1 if not past_home_won else 0
            win_pcts.append(wins / len(past))
            avg_margins.append(sum(margins) / len(past))

        history.setdefault(key, []).append(
            (row["HOME_TEAM_ABBREVIATION"], row["_MARGIN"], bool(row["HOME_WIN"]))
        )

    df["H2H_HOME_WIN_PCT"] = win_pcts
    df["H2H_AVG_MARGIN"] = avg_margins
    return df[["GAME_ID", "H2H_HOME_WIN_PCT", "H2H_AVG_MARGIN"]]


def main():
    team_rows = pd.read_csv(DATA_DIR / "raw_team_game_logs.csv")
    rolling = compute_team_rolling_stats(team_rows)

    games_path = DATA_DIR / "games_with_odds.csv"
    if not games_path.exists():
        games_path = DATA_DIR / "games_one_row_per_game.csv"
        print(f"Note: games_with_odds.csv not found, using {games_path.name} (no odds columns).")

    games = pd.read_csv(games_path, parse_dates=["GAME_DATE"])
    features = attach_features_to_games(games, rolling)

    h2h = compute_head_to_head_features(games)
    features = features.merge(h2h, on="GAME_ID", how="left")

    player_path = DATA_DIR / "raw_player_game_logs.csv"
    if player_path.exists():
        player_rows = pd.read_csv(player_path)
        top_players = compute_top_players_features(player_rows)
        home_tp = top_players.add_prefix("HOME_").rename(columns={"HOME_GAME_ID": "GAME_ID"})
        away_tp = top_players.add_prefix("AWAY_").rename(columns={"AWAY_GAME_ID": "GAME_ID"})
        features = features.merge(
            home_tp, left_on=["GAME_ID", "HOME_TEAM_ABBREVIATION"],
            right_on=["GAME_ID", "HOME_TEAM_ABBREVIATION"], how="left"
        ).merge(
            away_tp, left_on=["GAME_ID", "AWAY_TEAM_ABBREVIATION"],
            right_on=["GAME_ID", "AWAY_TEAM_ABBREVIATION"], how="left"
        )
        # Positive = AWAY is missing more combined IMPACT than HOME is
        # (an advantage for HOME), consistent with this project's "positive
        # generally favors home" convention for DIFF_ columns.
        features["DIFF_MISSING_IMPACT"] = (
            features["AWAY_MISSING_IMPACT"].fillna(0) - features["HOME_MISSING_IMPACT"].fillna(0)
        )
        unknown = features["HOME_MISSING_IMPACT"].isna() | features["AWAY_MISSING_IMPACT"].isna()
        features.loc[unknown, "DIFF_MISSING_IMPACT"] = None
        print(f"Player data found -- added DIFF_MISSING_IMPACT "
              f"({features['DIFF_MISSING_IMPACT'].notna().sum()} / {len(features)} games have it).")
    else:
        print("No raw_player_game_logs.csv found -- skipping top-player features "
              "(run collect_player_data.py to enable).")

    # Drop early-season rows where rolling stats are still NaN (not enough history)
    feature_cols = [c for c in features.columns if c.startswith("DIFF_")]
    before = len(features)
    features = features.dropna(subset=feature_cols)
    print(f"Dropped {before - len(features)} rows with insufficient rolling history.")

    out_path = DATA_DIR / "model_ready.csv"
    features.to_csv(out_path, index=False)
    print(f"Saved -> {out_path} ({len(features)} rows)")


if __name__ == "__main__":
    main()
