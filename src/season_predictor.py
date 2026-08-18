"""
season_predictor.py
--------------------
Runs the win-probability model across an ENTIRE season's schedule (not one
manual matchup at a time), and projects final Eastern/Western Conference
standings by combining real results (for games already played) with model
predictions (for games that haven't happened yet).
"""

import pandas as pd
import predict_matchup as pm

TEAM_CONFERENCE = {
    "ATL": "East", "BOS": "East", "BKN": "East", "CHA": "East", "CHI": "East",
    "CLE": "East", "DET": "East", "IND": "East", "MIA": "East", "MIL": "East",
    "NYK": "East", "ORL": "East", "PHI": "East", "TOR": "East", "WAS": "East",
    "DAL": "West", "DEN": "West", "GSW": "West", "HOU": "West", "LAC": "West",
    "LAL": "West", "MEM": "West", "MIN": "West", "NOP": "West", "OKC": "West",
    "PHX": "West", "POR": "West", "SAC": "West", "SAS": "West", "UTA": "West",
}


def predict_full_season(schedule: pd.DataFrame, team_rows: pd.DataFrame, games: pd.DataFrame,
                         model, scaler, features,
                         player_rows: pd.DataFrame = None, current_rosters: pd.DataFrame = None) -> pd.DataFrame:
    """
    Runs a prediction for every game in `schedule`. Games already marked
    FINAL (GAME_STATUS == 3) use their real recorded result instead of a
    model prediction -- no point predicting something we already know.

    Returns one row per game with HOME_WIN_PROB (real 0/1 for final games,
    model probability for scheduled ones), PREDICTED_WINNER, and IS_FINAL.
    """
    team_snap = pm.get_latest_team_snapshot(team_rows)
    player_snap = None
    if player_rows is not None and current_rosters is not None:
        player_snap = pm.get_current_top_players(player_rows, current_rosters)

    results = []
    for _, g in schedule.iterrows():
        home, away = g["HOME_TEAM_ABBREVIATION"], g["AWAY_TEAM_ABBREVIATION"]
        is_final = g.get("GAME_STATUS") == 3

        if is_final and pd.notna(g.get("HOME_SCORE")) and g["HOME_SCORE"] > 0:
            home_win_prob = 1.0 if g["HOME_SCORE"] > g["AWAY_SCORE"] else 0.0
            results.append(dict(GAME_ID=g["GAME_ID"], GAME_DATE=g["GAME_DATE"],
                                 HOME_TEAM_ABBREVIATION=home, AWAY_TEAM_ABBREVIATION=away,
                                 HOME_WIN_PROB=home_win_prob, IS_FINAL=True))
            continue

        try:
            row = pm.build_hypothetical_row(
                home, away, g["GAME_DATE"], games=games,
                team_snapshot=team_snap, player_snapshot=player_snap,
            )
        except ValueError:
            continue

        missing_feats = [f for f in features if f not in row.index or pd.isna(row[f])]
        if missing_feats:
            continue

        X = pd.DataFrame([row[features].values], columns=features).astype(float)
        prob = model.predict_proba(scaler.transform(X))[0, 1]
        results.append(dict(GAME_ID=g["GAME_ID"], GAME_DATE=g["GAME_DATE"],
                             HOME_TEAM_ABBREVIATION=home, AWAY_TEAM_ABBREVIATION=away,
                             HOME_WIN_PROB=prob, IS_FINAL=False))

    out = pd.DataFrame(results)
    if len(out):
        out["PREDICTED_WINNER"] = out.apply(
            lambda r: r["HOME_TEAM_ABBREVIATION"] if r["HOME_WIN_PROB"] >= 0.5 else r["AWAY_TEAM_ABBREVIATION"],
            axis=1
        )
    return out


def compute_standings(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Projects each team's final win total by summing:
      - 1 real win/loss for every FINAL game
      - their WIN PROBABILITY (an expected-value contribution, not a coin
        flip) for every scheduled game

    This is a point estimate (expected wins), not a full Monte Carlo
    simulation of every possible season outcome -- simpler, and the
    standard first approach for this kind of projection.
    """
    records = {}
    for _, g in predictions.iterrows():
        home, away, prob = g["HOME_TEAM_ABBREVIATION"], g["AWAY_TEAM_ABBREVIATION"], g["HOME_WIN_PROB"]
        for team in (home, away):
            records.setdefault(team, {"WINS": 0.0, "LOSSES": 0.0, "GAMES": 0})
        records[home]["WINS"] += prob
        records[home]["LOSSES"] += (1 - prob)
        records[away]["WINS"] += (1 - prob)
        records[away]["LOSSES"] += prob
        records[home]["GAMES"] += 1
        records[away]["GAMES"] += 1

    rows = []
    for team, r in records.items():
        # Round to whole-number win/loss totals like a real standings table.
        # Round wins first, then derive losses from games-wins so they always
        # sum correctly (rounding both independently could occasionally be
        # off by one game).
        win_pct = r["WINS"] / r["GAMES"] if r["GAMES"] else 0
        wins_rounded = round(r["WINS"])
        losses_rounded = r["GAMES"] - wins_rounded
        rows.append(dict(
            TEAM=team, CONFERENCE=TEAM_CONFERENCE.get(team, "Unknown"),
            WINS=wins_rounded, LOSSES=losses_rounded,
            WIN_PCT=win_pct, GAMES=r["GAMES"],
        ))
    standings = pd.DataFrame(rows).sort_values("WIN_PCT", ascending=False)
    return standings
