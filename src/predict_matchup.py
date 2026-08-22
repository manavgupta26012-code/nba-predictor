"""
predict_matchup.py
-------------------
Predicts a HYPOTHETICAL matchup between any two teams -- including games
that haven't been played yet. Uses each team's most current known form
(their latest rolling stats) rather than looking up an actual game.

Real injury status IS available (see collect_injury_report.py, which
pulls the NBA's official daily injury report) -- but it's only published
~1 day ahead of each game by league rule, so it's only reliable for
near-term games, not weeks out. For games beyond that window, this module
still lets the user OPTIONALLY simulate "what if this team's top scorer
sits out" as a manual scenario toggle.
"""

import re
import pandas as pd
import build_features as bf
import home_advantage as ha
import chemistry as chem

BOX_SCORE_COLS = bf.BOX_SCORE_COLS

# Injury statuses treated as "not playing" for MISSING_IMPACT purposes.
# Questionable/Probable are deliberately NOT included here -- those
# players often do play, and treating them as automatically out would
# overstate the impact more often than it's right.
OUT_STATUSES = {"Out", "Doubtful"}

DEFENSIVE_STAT_COLS = ["FG3_PCT", "FG_PCT", "PTS"]  # what opponents have done AGAINST this team


def _normalize_injury_name(name: str) -> str:
    """Converts the injury report's 'Last, First' format (e.g. 'Nance Jr., Larry')
    to the 'First Last' format used elsewhere (e.g. 'Larry Nance Jr.')."""
    if "," not in name:
        return name.strip()
    last, first = name.split(",", 1)
    return f"{first.strip()} {last.strip()}"


def load_injury_lookup(injury_report: pd.DataFrame) -> dict:
    """Builds a {player_name: status} lookup from collect_injury_report.py's output."""
    lookup = {}
    for _, row in injury_report.iterrows():
        name = _normalize_injury_name(row["PLAYER_NAME"])
        lookup[name] = row["STATUS"]
    return lookup


def _add_allowed_stats(team_rows: pd.DataFrame, stat_cols=DEFENSIVE_STAT_COLS) -> pd.DataFrame:
    """
    For each team-game, pulls the OPPONENT's stats from that same game and
    labels them ALLOWED_<stat> -- e.g. ALLOWED_FG3_PCT is what the other
    team shot from three against this team. Uses a self-join on GAME_ID
    (each game has exactly 2 team rows), so no new data collection needed.
    """
    keep = ["GAME_ID", "TEAM_ABBREVIATION"] + list(stat_cols)
    df = team_rows[keep].copy()
    opp = df.rename(columns={c: f"ALLOWED_{c}" for c in stat_cols})
    opp = opp.rename(columns={"TEAM_ABBREVIATION": "OPP_TEAM_ABBREVIATION"})

    merged = df[["GAME_ID", "TEAM_ABBREVIATION"]].merge(opp, on="GAME_ID")
    merged = merged[merged["TEAM_ABBREVIATION"] != merged["OPP_TEAM_ABBREVIATION"]]
    return merged.drop(columns="OPP_TEAM_ABBREVIATION")


def get_latest_team_snapshot(team_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Each team's CURRENT form: rolling stats computed INCLUSIVE of their most
    recent game (not shifted -- for an actual past game we exclude the
    current game to avoid leakage, but here we're projecting forward to a
    game that hasn't happened, so all known games are fair to use).
    Returns one row per team = their latest snapshot.
    """
    df = team_rows.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["WIN"] = (df["WL"] == "W").astype(int)

    allowed = _add_allowed_stats(team_rows)
    df = df.merge(allowed, on=["GAME_ID", "TEAM_ABBREVIATION"], how="left")

    df = df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])
    grouped = df.groupby("TEAM_ABBREVIATION", group_keys=False)

    for col in BOX_SCORE_COLS:
        df[f"ROLL_{col}"] = grouped[col].apply(lambda s: s.rolling(10, min_periods=3).mean())
    df["ROLL_WIN_PCT"] = grouped["WIN"].apply(lambda s: s.rolling(10, min_periods=3).mean())
    for col in DEFENSIVE_STAT_COLS:
        df[f"ROLL_ALLOWED_{col}"] = grouped[f"ALLOWED_{col}"].apply(lambda s: s.rolling(10, min_periods=3).mean())

    snapshot = df.sort_values("GAME_DATE").groupby("TEAM_ABBREVIATION").tail(1)
    return snapshot.set_index("TEAM_ABBREVIATION")


def get_latest_top_players(player_rows: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """
    DEPRECATED for season-ahead predictions -- kept only for reference.
    This attributes players to whichever team they were on IN THE HISTORICAL
    BOX SCORES, which is wrong once a player is traded, signs elsewhere, or
    retires over the offseason. Use get_current_top_players() instead.
    """
    top_players = bf.compute_top_players_features(player_rows, top_n=top_n)
    top_players = top_players.merge(
        player_rows[["GAME_ID", "TEAM_ABBREVIATION", "GAME_DATE"]].drop_duplicates(),
        on=["GAME_ID", "TEAM_ABBREVIATION"], how="left"
    )
    top_players["GAME_DATE"] = pd.to_datetime(top_players["GAME_DATE"])
    snapshot = top_players.sort_values("GAME_DATE").groupby("TEAM_ABBREVIATION").tail(1)
    return snapshot.set_index("TEAM_ABBREVIATION")


def get_current_top_players(player_rows: pd.DataFrame, current_rosters: pd.DataFrame,
                             top_n: int = 3, window: int = 10, min_games: int = 3) -> pd.DataFrame:
    """
    Ranks each team's top N players BY IMPACT (Game Score -- see
    build_features.compute_game_score, not raw points) using their ACTUAL
    CURRENT roster combined with each player's own full box score history.
    Correctly handles trades/free agency -- a player who moved teams over
    the offseason is credited to his NEW team, using stats earned on his
    OLD team.
    """
    df = player_rows.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["GAME_SCORE"] = bf.compute_game_score(df)
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"])
    grouped = df.groupby("PLAYER_ID", group_keys=False)
    df["PLAYER_ROLL_IMPACT"] = grouped["GAME_SCORE"].apply(lambda s: s.rolling(window, min_periods=min_games).mean())
    df["PLAYER_ROLL_PTS"] = grouped["PTS"].apply(lambda s: s.rolling(window, min_periods=min_games).mean())
    latest = (
        df.sort_values("GAME_DATE").groupby("PLAYER_ID").tail(1)
        [["PLAYER_ID", "PLAYER_NAME", "PLAYER_ROLL_IMPACT", "PLAYER_ROLL_PTS"]]
    )

    roster_cols = current_rosters.rename(columns={"PLAYER": "ROSTER_PLAYER_NAME"})
    roster_cols["PLAYER_ID"] = roster_cols["PLAYER_ID"].astype(latest["PLAYER_ID"].dtype, errors="ignore")
    merged = roster_cols[["TEAM_ABBREVIATION", "PLAYER_ID", "ROSTER_PLAYER_NAME"]].merge(
        latest, on="PLAYER_ID", how="left"
    )

    results = []
    for team, g in merged.groupby("TEAM_ABBREVIATION"):
        ranked = g.dropna(subset=["PLAYER_ROLL_IMPACT"]).sort_values("PLAYER_ROLL_IMPACT", ascending=False).head(top_n)
        row = {"TEAM_ABBREVIATION": team}
        for rank, (_, p) in enumerate(ranked.iterrows(), start=1):
            name = p["PLAYER_NAME"] if pd.notna(p.get("PLAYER_NAME")) else p["ROSTER_PLAYER_NAME"]
            row[f"PLAYER{rank}_NAME"] = name
            row[f"PLAYER{rank}_PPG"] = p["PLAYER_ROLL_PTS"]
            row[f"PLAYER{rank}_IMPACT"] = p["PLAYER_ROLL_IMPACT"]
        for rank in range(len(ranked) + 1, top_n + 1):
            row[f"PLAYER{rank}_NAME"] = None
            row[f"PLAYER{rank}_PPG"] = None
            row[f"PLAYER{rank}_IMPACT"] = None
        results.append(row)

    return pd.DataFrame(results).set_index("TEAM_ABBREVIATION")


def get_h2h_snapshot(games: pd.DataFrame, home_team: str, away_team: str, window: int = 5) -> dict:
    """Last `window` meetings between these two teams (any order, using ALL
    known history), from the perspective of `home_team` playing at home."""
    pair_games = games[
        ((games["HOME_TEAM_ABBREVIATION"] == home_team) & (games["AWAY_TEAM_ABBREVIATION"] == away_team)) |
        ((games["HOME_TEAM_ABBREVIATION"] == away_team) & (games["AWAY_TEAM_ABBREVIATION"] == home_team))
    ].sort_values("GAME_DATE").tail(window)

    if len(pair_games) < 3:
        return {"H2H_HOME_WIN_PCT": None, "H2H_AVG_MARGIN": None}

    wins, margins = 0, []
    for _, g in pair_games.iterrows():
        margin = g["HOME_PTS"] - g["AWAY_PTS"]
        if g["HOME_TEAM_ABBREVIATION"] == home_team:
            margins.append(margin)
            wins += 1 if g["HOME_WIN"] == 1 else 0
        else:
            margins.append(-margin)
            wins += 1 if g["HOME_WIN"] == 0 else 0
    return {"H2H_HOME_WIN_PCT": wins / len(pair_games), "H2H_AVG_MARGIN": sum(margins) / len(pair_games)}


def build_hypothetical_row(home_team: str, away_team: str, game_date, team_rows: pd.DataFrame = None,
                            player_rows: pd.DataFrame = None, games: pd.DataFrame = None,
                            assume_home_top_scorer_out: bool = False,
                            assume_away_top_scorer_out: bool = False,
                            team_snapshot: pd.DataFrame = None,
                            player_snapshot: pd.DataFrame = None,
                            advanced_stats: pd.DataFrame = None,
                            shot_locations: pd.DataFrame = None,
                            injury_lookup: dict = None,
                            home_away_splits: pd.DataFrame = None,
                            chemistry_metrics: pd.DataFrame = None) -> pd.Series:
    """
    Builds a single feature row for a hypothetical HOME_TEAM vs AWAY_TEAM
    matchup on GAME_DATE, using each team's most current known form.

    `injury_lookup` (from the NBA's real official injury report) takes
    priority over the manual "assume out" toggles when both are available.

    For predicting MANY games at once, pass precomputed `team_snapshot` /
    `player_snapshot` instead of raw `team_rows` / `player_rows`.

    `advanced_stats` / `shot_locations` / `home_away_splits` are optional,
    narrative-only enrichment -- NOT used by the trained model.
    """
    game_date = pd.to_datetime(game_date)
    snap = team_snapshot if team_snapshot is not None else get_latest_team_snapshot(team_rows)

    if home_team not in snap.index or away_team not in snap.index:
        missing = [t for t in (home_team, away_team) if t not in snap.index]
        raise ValueError(f"No data found for team(s): {missing}")

    h, a = snap.loc[home_team], snap.loc[away_team]
    row = {"HOME_TEAM_ABBREVIATION": home_team, "AWAY_TEAM_ABBREVIATION": away_team, "GAME_DATE": game_date}

    for col in BOX_SCORE_COLS:
        row[f"HOME_ROLL_{col}"] = h[f"ROLL_{col}"]
        row[f"AWAY_ROLL_{col}"] = a[f"ROLL_{col}"]
        row[f"DIFF_{col}"] = h[f"ROLL_{col}"] - a[f"ROLL_{col}"]
    row["HOME_ROLL_WIN_PCT"] = h["ROLL_WIN_PCT"]
    row["AWAY_ROLL_WIN_PCT"] = a["ROLL_WIN_PCT"]
    row["DIFF_WIN_PCT"] = h["ROLL_WIN_PCT"] - a["ROLL_WIN_PCT"]

    for col in DEFENSIVE_STAT_COLS:
        if f"ROLL_ALLOWED_{col}" in h.index:
            row[f"HOME_ROLL_ALLOWED_{col}"] = h[f"ROLL_ALLOWED_{col}"]
            row[f"AWAY_ROLL_ALLOWED_{col}"] = a[f"ROLL_ALLOWED_{col}"]

    home_rest = min((game_date - h["GAME_DATE"]).days, 7)
    away_rest = min((game_date - a["GAME_DATE"]).days, 7)
    row["HOME_REST_DAYS"], row["AWAY_REST_DAYS"] = home_rest, away_rest
    row["DIFF_REST_DAYS"] = home_rest - away_rest
    row["HOME_BACK_TO_BACK"] = 1 if home_rest == 1 else 0
    row["AWAY_BACK_TO_BACK"] = 1 if away_rest == 1 else 0
    row["DIFF_BACK_TO_BACK"] = row["HOME_BACK_TO_BACK"] - row["AWAY_BACK_TO_BACK"]

    if games is not None:
        row.update(get_h2h_snapshot(games, home_team, away_team))
    else:
        row["H2H_HOME_WIN_PCT"], row["H2H_AVG_MARGIN"] = None, None

    if player_rows is not None or player_snapshot is not None:
        players = player_snapshot if player_snapshot is not None else get_latest_top_players(player_rows)
        for prefix, team, assume_out in [("HOME", home_team, assume_home_top_scorer_out),
                                          ("AWAY", away_team, assume_away_top_scorer_out)]:
            if team in players.index:
                p = players.loc[team]
                missing_impact = 0.0
                for rank in [1, 2, 3]:
                    name, ppg = p.get(f"PLAYER{rank}_NAME"), p.get(f"PLAYER{rank}_PPG")
                    impact = p.get(f"PLAYER{rank}_IMPACT")
                    row[f"{prefix}_PLAYER{rank}_NAME"] = name
                    row[f"{prefix}_PLAYER{rank}_PPG"] = ppg
                    row[f"{prefix}_PLAYER{rank}_IMPACT"] = impact

                    injury_status = injury_lookup.get(name) if (injury_lookup and pd.notna(name)) else None
                    if injury_status in OUT_STATUSES:
                        is_out = 1
                    elif rank == 1 and assume_out:
                        is_out = 1
                    else:
                        is_out = 0
                    row[f"{prefix}_PLAYER{rank}_OUT"] = is_out
                    row[f"{prefix}_PLAYER{rank}_INJURY_STATUS"] = injury_status
                    if is_out and pd.notna(impact):
                        missing_impact += impact
                row[f"{prefix}_MISSING_IMPACT"] = missing_impact
        row["DIFF_MISSING_IMPACT"] = row.get("AWAY_MISSING_IMPACT", 0) - row.get("HOME_MISSING_IMPACT", 0)
    else:
        row["DIFF_MISSING_IMPACT"] = None

    if advanced_stats is not None:
        adv_fields = ["OFF_RATING", "DEF_RATING", "NET_RATING", "PACE", "EFG_PCT", "TS_PCT", "TM_TOV_PCT"]
        for prefix, team in [("HOME", home_team), ("AWAY", away_team)]:
            if team in advanced_stats.index:
                for f in adv_fields:
                    row[f"{prefix}_{f}"] = advanced_stats.loc[team, f]

    if shot_locations is not None:
        paint_fields = ["RA_FG_PCT", "PAINT_NONRA_FG_PCT", "PAINT_FG_PCT", "PAINT_PTS", "MIDRANGE_FG_PCT", "CORNER3_FG_PCT"]
        for prefix, team in [("HOME", home_team), ("AWAY", away_team)]:
            if team in shot_locations.index:
                for f in paint_fields:
                    row[f"{prefix}_{f}"] = shot_locations.loc[team, f]

    if home_away_splits is not None:
        if home_team in home_away_splits.index:
            hh = home_away_splits.loc[home_team]
            row["HOME_TEAM_HOME_WIN_PCT"] = hh["HOME_WIN_PCT"]
            row["HOME_TEAM_HOME_MARGIN"] = hh["HOME_AVG_MARGIN"]
            row["HOME_TEAM_HOME_ADVANTAGE"] = hh["HOME_ADVANTAGE"]
        if away_team in home_away_splits.index:
            aa = home_away_splits.loc[away_team]
            row["AWAY_TEAM_AWAY_WIN_PCT"] = aa["AWAY_WIN_PCT"]
            row["AWAY_TEAM_AWAY_MARGIN"] = aa["AWAY_AVG_MARGIN"]

    if chemistry_metrics is not None:
        for prefix, team in [("HOME", home_team), ("AWAY", away_team)]:
            if team in chemistry_metrics.index:
                c = chemistry_metrics.loc[team]
                row[f"{prefix}_AST_PER_FGM"] = c["AST_PER_FGM"]
                row[f"{prefix}_ROLL_TOV_CHEM"] = c["ROLL_TOV"]
                row[f"{prefix}_AST_PERCENTILE"] = c["AST_PERCENTILE"]
                row[f"{prefix}_TOV_PERCENTILE"] = c["TOV_PERCENTILE"]

    return pd.Series(row)
