import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from collect_data import reshape_to_one_row_per_game

np.random.seed(42)
TEAMS = ["LAL", "BOS", "GSW", "MIA", "DEN", "PHX", "MIL", "PHI"]
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

rows = []
game_id = 1000
dates = pd.date_range("2023-10-24", periods=120, freq="D")

for d in dates:
    np.random.shuffle(TEAMS)
    for i in range(0, len(TEAMS), 2):
        home, away = TEAMS[i], TEAMS[i + 1]
        gid = f"00{game_id}"
        game_id += 1
        home_pts = np.random.normal(112, 10)
        away_pts = np.random.normal(110, 10)
        home_win = home_pts > away_pts
        for team, opp, pts, is_home, win in [
            (home, away, home_pts, True, home_win),
            (away, home, away_pts, False, not home_win),
        ]:
            rows.append({
                "TEAM_ID": hash(team) % 10000,
                "TEAM_ABBREVIATION": team,
                "TEAM_NAME": team,
                "GAME_ID": gid,
                "GAME_DATE": d.strftime("%Y-%m-%d"),
                "MATCHUP": f"{team} vs. {opp}" if is_home else f"{team} @ {opp}",
                "WL": "W" if win else "L",
                "PTS": round(pts, 1),
                "FG_PCT": round(np.random.uniform(0.40, 0.52), 3),
                "FG3_PCT": round(np.random.uniform(0.30, 0.42), 3),
                "FT_PCT": round(np.random.uniform(0.70, 0.85), 3),
                "REB": round(np.random.normal(44, 5), 1),
                "OREB": round(np.random.normal(10, 3), 1),
                "DREB": round(np.random.normal(34, 4), 1),
                "AST": round(np.random.normal(25, 4), 1),
                "STL": round(np.random.normal(7, 2), 1),
                "BLK": round(np.random.normal(5, 2), 1),
                "TOV": round(np.random.normal(13, 3), 1),
                "PLUS_MINUS": round(home_pts - away_pts if is_home else away_pts - home_pts, 1),
                "SEASON": "2023-24",
            })

team_rows = pd.DataFrame(rows)
team_rows.to_csv(DATA_DIR / "raw_team_game_logs.csv", index=False)

games = reshape_to_one_row_per_game(team_rows)
games.to_csv(DATA_DIR / "games_one_row_per_game.csv", index=False)

print(f"Synthetic team rows: {len(team_rows)}")
print(f"Synthetic games: {len(games)}")
