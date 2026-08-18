"""
team_names.py
-------------
NBA team names show up differently across data sources:
  - nba_api uses 3-letter abbreviations: 'LAL', 'BOS', 'GSW'
  - odds datasets often use full names: 'Los Angeles Lakers', 'Boston Celtics'
  - some use city only: 'LA Lakers', 'Golden State'

This module maps everything to a single canonical abbreviation so datasets
can be joined reliably. Extend ALIASES if your odds source uses a name not
listed here (print unmatched names and add them).
"""

CANONICAL = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}

# Reverse + common alternate spellings seen in odds datasets
ALIASES = {v.lower(): k for k, v in CANONICAL.items()}
ALIASES.update({
    "atl": "ATL", "bkn": "BKN", "bos": "BOS", "cha": "CHA", "chi": "CHI",
    "cle": "CLE", "dal": "DAL", "den": "DEN", "det": "DET", "gs": "GSW",
    "hou": "HOU", "ind": "IND", "lac": "LAC", "lal": "LAL", "mem": "MEM",
    "mia": "MIA", "mil": "MIL", "min": "MIN", "no": "NOP", "ny": "NYK",
    "okc": "OKC", "orl": "ORL", "phi": "PHI", "phx": "PHX", "por": "POR",
    "sa": "SAS", "sac": "SAC", "tor": "TOR", "utah": "UTA", "wsh": "WAS",
})
# also allow abbreviations already being correct (case-insensitive)
ALIASES.update({k.lower(): k for k in CANONICAL.keys()})


def to_abbrev(name: str) -> str | None:
    """Normalize any team name string to its 3-letter abbreviation, or None if unknown."""
    if not isinstance(name, str):
        return None
    key = name.strip().lower()
    return ALIASES.get(key)
