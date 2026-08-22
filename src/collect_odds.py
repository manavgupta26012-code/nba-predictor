"""
collect_odds.py
----------------
Pulls live NBA moneyline odds from The Odds API's free tier -- 25
requests/day, no credit card required. One call returns the whole
upcoming slate, not per-game, so this comfortably fits the free tier for
a personal dashboard checked once or twice a day.

Get a free API key (no card): https://the-odds-api.com/
Then set it as an environment variable:
    Windows:   set ODDS_API_KEY=your_key_here
    Mac/Linux: export ODDS_API_KEY=your_key_here

Usage:
    python src/collect_odds.py
"""

import os
import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

BASE_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"

# The Odds API returns full official team names (e.g. "Los Angeles Lakers"),
# not abbreviations -- this maps them to the abbreviations used everywhere
# else in this project.
TEAM_NAME_TO_ABBREV = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP", "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR", "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


def american_odds_to_prob(odds: float) -> float:
    """Converts American moneyline odds to implied win probability."""
    if odds is None:
        return None
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)


def main():
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ODDS_API_KEY environment variable not set. Get a free key (no card) at "
            "https://the-odds-api.com/ and set it before running this script."
        )

    print("Fetching NBA moneyline odds...")
    resp = requests.get(BASE_URL, params={
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
    }, timeout=10)

    print(f"HTTP status: {resp.status_code}")
    print(f"Requests used today: {resp.headers.get('x-requests-used')} / "
          f"remaining: {resp.headers.get('x-requests-remaining')}")

    if resp.status_code != 200:
        print(f"Response body: {resp.text[:500]}")
        return

    data = resp.json()
    print(f"\nGames returned: {len(data)}")
    if not data:
        print("No games found -- may be an off day, or the season may not have started yet.")
        return

    print(f"\n--- Raw structure of first game (for verification) ---")
    print(data[0])

    rows = []
    for game in data:
        home_full, away_full = game.get("home_team"), game.get("away_team")
        home_abbrev = TEAM_NAME_TO_ABBREV.get(home_full)
        away_abbrev = TEAM_NAME_TO_ABBREV.get(away_full)
        if home_abbrev is None or away_abbrev is None:
            print(f"  WARNING: unmapped team name(s): {home_full} / {away_full}")
            continue

        # Convert EACH bookmaker's price to an implied probability first,
        # then average the probabilities -- NOT the raw American odds
        # directly. American odds aren't on a linear scale, and averaging
        # them raw (especially when books disagree on which side is
        # favored, e.g. one book has a team at -103 while others have the
        # same team at +105) produces a mathematically nonsensical blend.
        # Converting to probability space first, where averaging is valid,
        # is the correct way to build a consensus.
        home_probs, away_probs = [], []
        for bm in game.get("bookmakers", []):
            for market in bm.get("markets", []):
                if market["key"] != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    if outcome["name"] == home_full:
                        home_probs.append(american_odds_to_prob(outcome["price"]))
                    elif outcome["name"] == away_full:
                        away_probs.append(american_odds_to_prob(outcome["price"]))

        if not home_probs or not away_probs:
            continue

        avg_home_prob = sum(home_probs) / len(home_probs)
        avg_away_prob = sum(away_probs) / len(away_probs)
        rows.append(dict(
            GAME_DATETIME=game.get("commence_time"),
            HOME_TEAM_ABBREVIATION=home_abbrev,
            AWAY_TEAM_ABBREVIATION=away_abbrev,
            HOME_WIN_PROB_MARKET=avg_home_prob,
            AWAY_WIN_PROB_MARKET=avg_away_prob,
            NUM_BOOKS=len(home_probs),
        ))

    df = pd.DataFrame(rows)
    out_path = DATA_DIR / "live_odds.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path} ({len(df)} games)")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
