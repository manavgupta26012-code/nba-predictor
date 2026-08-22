"""
app.py
------
NBA GameIQ -- an NBA game intelligence platform. Analysis is the
centerpiece; the win prediction is presented as the conclusion of the
analysis, not the whole product.

Run with:
    streamlit run app.py
"""

import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import predict_matchup as pm
import season_predictor as sp
import home_advantage as ha
import chemistry as chem

DATA_DIR = Path(__file__).resolve().parent / "data"
MODEL_DIR = Path(__file__).resolve().parent / "models"

VALUE_ALERT_THRESHOLD = 0.08  # flag when model and market disagree by 8+ points

# Static team ID map (from nba_api's bundled static data) -- hardcoded here
# so app.py doesn't need nba_api as a runtime dependency, just for logos.
TEAM_IDS = {
    "ATL": 1610612737, "BKN": 1610612751, "BOS": 1610612738, "CHA": 1610612766,
    "CHI": 1610612741, "CLE": 1610612739, "DAL": 1610612742, "DEN": 1610612743,
    "DET": 1610612765, "GSW": 1610612744, "HOU": 1610612745, "IND": 1610612754,
    "LAC": 1610612746, "LAL": 1610612747, "MEM": 1610612763, "MIA": 1610612748,
    "MIL": 1610612749, "MIN": 1610612750, "NOP": 1610612740, "NYK": 1610612752,
    "OKC": 1610612760, "ORL": 1610612753, "PHI": 1610612755, "PHX": 1610612756,
    "POR": 1610612757, "SAC": 1610612758, "SAS": 1610612759, "TOR": 1610612761,
    "UTA": 1610612762, "WAS": 1610612764,
}
TEAM_FULLNAMES = {
    "ATL": "Atlanta", "BKN": "Brooklyn", "BOS": "Boston", "CHA": "Charlotte",
    "CHI": "Chicago", "CLE": "Cleveland", "DAL": "Dallas", "DEN": "Denver",
    "DET": "Detroit", "GSW": "Golden State", "HOU": "Houston", "IND": "Indiana",
    "LAC": "LA Clippers", "LAL": "LA Lakers", "MEM": "Memphis", "MIA": "Miami",
    "MIL": "Milwaukee", "MIN": "Minnesota", "NOP": "New Orleans", "NYK": "New York",
    "OKC": "Oklahoma City", "ORL": "Orlando", "PHI": "Philadelphia", "PHX": "Phoenix",
    "POR": "Portland", "SAC": "Sacramento", "SAS": "San Antonio", "TOR": "Toronto",
    "UTA": "Utah", "WAS": "Washington",
}


def team_logo_url(abbrev: str) -> str:
    team_id = TEAM_IDS.get(abbrev)
    return f"https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.svg" if team_id else ""


# Maps each model feature to its underlying home/away columns and how to
# describe it in plain English -- used by explain_prediction() for the
# quick "top factors by model weight" summary.
FEATURE_META = {
    "DIFF_PTS":          dict(label="Scoring", plain="points scored per game", home_col="HOME_ROLL_PTS", away_col="AWAY_ROLL_PTS", fmt="num1", unit="pts/game"),
    "DIFF_FG_PCT":       dict(label="Shooting accuracy", plain="the share of shots that go in", home_col="HOME_ROLL_FG_PCT", away_col="AWAY_ROLL_FG_PCT", fmt="pct", unit=""),
    "DIFF_FG3_PCT":      dict(label="3-point shooting", plain="the share of 3-point shots that go in", home_col="HOME_ROLL_FG3_PCT", away_col="AWAY_ROLL_FG3_PCT", fmt="pct", unit=""),
    "DIFF_FT_PCT":       dict(label="Free throw shooting", plain="the share of free throws made", home_col="HOME_ROLL_FT_PCT", away_col="AWAY_ROLL_FT_PCT", fmt="pct", unit=""),
    "DIFF_REB":          dict(label="Rebounding", plain="missed shots (either team's) they collect", home_col="HOME_ROLL_REB", away_col="AWAY_ROLL_REB", fmt="num1", unit="/game"),
    "DIFF_OREB":         dict(label="Offensive rebounding", plain="their own missed shots they collect for another try", home_col="HOME_ROLL_OREB", away_col="AWAY_ROLL_OREB", fmt="num1", unit="/game"),
    "DIFF_DREB":         dict(label="Defensive rebounding", plain="the opponent's missed shots they collect", home_col="HOME_ROLL_DREB", away_col="AWAY_ROLL_DREB", fmt="num1", unit="/game"),
    "DIFF_AST":          dict(label="Passing (assists)", plain="baskets they set up for a teammate", home_col="HOME_ROLL_AST", away_col="AWAY_ROLL_AST", fmt="num1", unit="/game"),
    "DIFF_STL":          dict(label="Steals", plain="times they take the ball away from the other team", home_col="HOME_ROLL_STL", away_col="AWAY_ROLL_STL", fmt="num1", unit="/game"),
    "DIFF_BLK":          dict(label="Shot blocking", plain="opponent shots they block", home_col="HOME_ROLL_BLK", away_col="AWAY_ROLL_BLK", fmt="num1", unit="/game"),
    "DIFF_TOV":          dict(label="Turnovers", plain="times they lose the ball to the other team (lower is usually better)", home_col="HOME_ROLL_TOV", away_col="AWAY_ROLL_TOV", fmt="num1", unit="/game"),
    "DIFF_PLUS_MINUS":   dict(label="Point differential", plain="points scored minus points allowed, on average", home_col="HOME_ROLL_PLUS_MINUS", away_col="AWAY_ROLL_PLUS_MINUS", fmt="signed1", unit="pts/game"),
    "DIFF_WIN_PCT":      dict(label="Recent win rate", plain="how often they've won lately", home_col="HOME_ROLL_WIN_PCT", away_col="AWAY_ROLL_WIN_PCT", fmt="pct", unit=""),
    "DIFF_REST_DAYS":    dict(label="Rest", plain="days off since their last game", home_col="HOME_REST_DAYS", away_col="AWAY_REST_DAYS", fmt="num0", unit="days"),
    "DIFF_MISSING_IMPACT": dict(label="Missing impact", plain="combined 'Impact Score' of any top-3 player assumed unavailable", home_col="HOME_MISSING_IMPACT", away_col="AWAY_MISSING_IMPACT", fmt="num1", unit=""),
}

st.set_page_config(page_title="NBA Predictor", layout="wide", page_icon="🏀")


def inject_theme():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap');

    :root {
        --accent: #C97D1E;
        --accent-soft: rgba(201, 125, 30, 0.10);
        --teal: #0F8A68;
        --win: #2F9E5B;
        --loss: #D64545;
        --surface: #F6F5F1;
        --border: #E4E2DB;
        --text-primary: #1A1A1A;
        --text-muted: #6B6D72;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--text-primary); }
    h1, h2, h3 { font-family: 'Oswald', sans-serif !important; letter-spacing: 0.01em; color: var(--text-primary); }

    /* Force the light background regardless of Streamlit's own theme
       setting or config.toml -- config.toml is only read at server
       startup and can be overridden by a user's personal dark/light
       choice in Streamlit's own settings menu. */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
    [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stMain"] {
        background-color: #FFFFFF !important;
    }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span,
    .stApp p, .stApp li, .stApp label {
        color: var(--text-primary) !important;
    }
    /* Safety net: any custom div that forgets to set its own text color
       (like .risk-item originally did) still falls back to readable text
       instead of invisible black-on-black. */
    [data-testid="stMarkdownContainer"] div {
        color: var(--text-primary);
    }
    [data-testid="stExpander"] {
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
    }
    [data-testid="stDataFrame"] { background-color: #FFFFFF !important; }

    .hero-banner {
        border-bottom: 3px solid var(--accent);
        padding-bottom: 14px;
        margin-bottom: 6px;
    }
    .hero-title {
        font-family: 'Oswald', sans-serif;
        font-weight: 700;
        font-size: 3.4rem;
        letter-spacing: 0.02em;
        margin: 0;
        color: var(--text-primary);
        text-transform: uppercase;
    }
    .hero-title span { color: var(--accent); }
    .hero-sub {
        color: var(--text-muted);
        font-size: 0.92rem;
        margin-top: 2px;
    }

    .eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--accent);
        border-left: 3px solid var(--accent);
        padding-left: 8px;
        margin: 22px 0 10px 0;
    }

    .scoreboard { display: flex; gap: 12px; margin: 8px 0 18px 0; }
    .scoreboard-tile {
        flex: 1;
        background: var(--surface);
        border: 1px solid var(--border);
        border-top: 3px solid var(--accent);
        border-radius: 6px;
        padding: 14px 16px;
    }
    .scoreboard-tile.win-tile { border-top-color: var(--win); }
    .scoreboard-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-muted);
    }
    .scoreboard-value {
        font-family: 'Oswald', sans-serif;
        font-weight: 600;
        font-size: 1.9rem;
        color: var(--text-primary);
        margin-top: 2px;
    }
    .scoreboard-value.accent-text { color: var(--accent); }
    .scoreboard-value.win-text { color: var(--win); }

    /* st.metric doesn't inherit our theme automatically (same issue as
       buttons/pills/dataframes before) -- without this, every metric
       scattered through Team Analysis and Matchup renders in Streamlit's
       default font, clashing with the Oswald/JetBrains Mono look
       everywhere else. */
    [data-testid="stMetricLabel"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.7rem !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-muted) !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Oswald', sans-serif !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
    }

    .player-chip {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        padding: 3px 0;
        color: var(--text-primary);
    }
    .dot-green { color: var(--win); }
    .dot-red { color: var(--loss); }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Oswald', sans-serif;
        font-size: 0.95rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    .stButton>button {
        font-family: 'Oswald', sans-serif;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        border-radius: 4px;
    }
    /* Buttons and st.pills both render as literal <button> elements --
       targeting that directly (rather than a Streamlit-internal class
       name that can change between versions) is what actually fixed the
       unreadable black-on-black button bug. */
    .stApp button {
        background-color: var(--surface) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
    }
    .stApp button:hover {
        border-color: var(--accent) !important;
        color: var(--accent) !important;
    }
    .stApp button p {
        color: inherit !important;
    }

    .game-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: var(--surface);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent);
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 6px;
    }
    .game-row-teams {
        font-family: 'Oswald', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        color: var(--text-primary);
    }
    .game-row-pick {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: var(--win);
    }
    .detail-nav-item {
        font-family: 'Oswald', sans-serif;
        font-size: 0.85rem;
        letter-spacing: 0.02em;
        padding: 8px 4px;
        color: var(--text-muted);
    }
    .detail-nav-item.active {
        color: var(--accent);
        font-weight: 600;
        border-left: 3px solid var(--accent);
        padding-left: 10px;
    }
    .detail-nav-subitem {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: var(--text-muted);
        padding: 3px 0 3px 16px;
    }
    .placeholder-box {
        background: var(--accent-soft);
        border: 1px dashed var(--border);
        border-radius: 6px;
        padding: 16px;
        color: var(--text-muted);
        font-size: 0.9rem;
    }
    .risk-item {
        border-left: 3px solid var(--loss);
        background: var(--surface);
        border-radius: 4px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.9rem;
        color: var(--text-primary);
    }
    .content-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 14px 16px;
        margin-bottom: 12px;
        color: var(--text-primary);
        font-size: 0.92rem;
        line-height: 1.5;
    }
    .content-card b { color: var(--text-primary); }
    .standings-table {
        width: 100%;
        border-collapse: collapse;
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 6px;
        overflow: hidden;
        font-family: 'Inter', sans-serif;
        font-size: 0.88rem;
    }
    .standings-table th {
        background: var(--surface);
        color: var(--text-muted);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        text-align: left;
        padding: 8px 12px;
        border-bottom: 2px solid var(--accent);
    }
    .standings-table td {
        padding: 7px 12px;
        color: var(--text-primary);
        border-bottom: 1px solid var(--border);
    }
    .standings-table tr:nth-child(even) td {
        background: var(--accent-soft);
    }
    .standings-table td.rank-col {
        color: var(--text-muted);
        font-family: 'JetBrains Mono', monospace;
        width: 30px;
    }
    .standings-table td.team-col {
        font-family: 'Oswald', sans-serif;
        font-weight: 600;
    }
    .standings-table td.num-col {
        font-family: 'JetBrains Mono', monospace;
        text-align: right;
    }
    .matchup-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 20px 22px;
        margin-bottom: 8px;
    }
    .matchup-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
    }
    .matchup-team-block {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .matchup-team-block.away-block { justify-content: flex-start; }
    .matchup-team-block.home-block { justify-content: flex-end; }
    .matchup-team-logo { width: 40px; height: 40px; object-fit: contain; }
    .matchup-team-name {
        font-family: 'Oswald', sans-serif;
        font-weight: 600;
        font-size: 1.05rem;
        color: var(--text-primary);
    }
    .matchup-countdown {
        text-align: center;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        background: var(--accent-soft);
        border-radius: 6px;
        padding: 6px 12px;
        white-space: nowrap;
    }
    .matchup-countdown b {
        display: block;
        font-size: 0.95rem;
        color: var(--accent);
        font-family: 'Oswald', sans-serif;
    }
    .matchup-prob-row {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .matchup-prob-pct {
        font-family: 'Oswald', sans-serif;
        font-weight: 700;
        font-size: 1.7rem;
        min-width: 78px;
    }
    .matchup-prob-pct.left { text-align: left; }
    .matchup-prob-pct.right { text-align: right; }
    .matchup-badge-hw {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.62rem;
        background: var(--border);
        color: var(--text-muted);
        border-radius: 50%;
        width: 16px; height: 16px;
        display: inline-flex; align-items: center; justify-content: center;
        margin-right: 2px;
    }
    .matchup-bar-outer {
        flex: 1;
        position: relative;
        height: 10px;
    }
    .matchup-bar-wrap {
        display: flex;
        height: 10px;
        border-radius: 5px;
        overflow: hidden;
        width: 100%;
    }
    .matchup-bar-marker {
        position: absolute;
        top: 50%;
        width: 22px; height: 22px;
        background: var(--surface);
        border: 2px solid var(--border);
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.7rem;
        transform: translate(-50%, -50%);
    }
    </style>
    """, unsafe_allow_html=True)


def _win_color(pct: float) -> str:
    """Green if this side is favored, orange if a moderate underdog,
    red if a heavy underdog (< 20%)."""
    if pct >= 50:
        return "var(--win)"
    elif pct >= 20:
        return "var(--accent)"
    return "var(--loss)"


def _format_countdown(game_datetime) -> str:
    if pd.isna(game_datetime):
        return "TBD"
    # The NBA schedule's datetime field can come through timezone-aware
    # (UTC) while pd.Timestamp.now() is naive -- strip tz info from both
    # sides so the subtraction always works regardless of which we got.
    if game_datetime.tzinfo is not None:
        game_datetime = game_datetime.tz_localize(None)
    now = pd.Timestamp.now()
    delta = game_datetime - now
    hours = delta.total_seconds() / 3600
    if hours < 0:
        return "Final" if hours < -3 else "Live"
    if hours < 24:
        return f"{round(hours)} Hours"
    return f"{round(hours / 24)} Days"


def render_matchup_card(away, home, prob, game_datetime=None):
    """Card-grid style summary: team logos, a live countdown, and a
    gradient win-probability bar with a marker at the split point."""
    away_pct, home_pct = (1 - prob) * 100, prob * 100
    away_color, home_color = _win_color(away_pct), _win_color(home_pct)
    countdown = _format_countdown(game_datetime) if game_datetime is not None else "TBD"

    st.markdown(f"""
    <div class="matchup-card">
        <div class="matchup-header">
            <div class="matchup-team-block away-block">
                <span class="matchup-team-name">{TEAM_FULLNAMES.get(away, away)}</span>
                <img class="matchup-team-logo" src="{team_logo_url(away)}" onerror="this.style.display='none'">
            </div>
            <div class="matchup-countdown">Starts In<b>{countdown}</b></div>
            <div class="matchup-team-block home-block">
                <img class="matchup-team-logo" src="{team_logo_url(home)}" onerror="this.style.display='none'">
                <span class="matchup-team-name">{TEAM_FULLNAMES.get(home, home)}</span>
            </div>
        </div>
        <div class="matchup-prob-row">
            <div class="matchup-prob-pct left" style="color:{away_color}">
                {away_pct:.1f}<span style="font-size:1rem">%</span>
            </div>
            <span class="matchup-badge-hw">A</span>
            <div class="matchup-bar-outer">
                <div class="matchup-bar-wrap">
                    <div style="width:{away_pct}%; background:{away_color}"></div>
                    <div style="width:{home_pct}%; background:{home_color}"></div>
                </div>
                <div class="matchup-bar-marker" style="left:{away_pct}%">🏆</div>
            </div>
            <span class="matchup-badge-hw">H</span>
            <div class="matchup-prob-pct right" style="color:{home_color}">
                {home_pct:.1f}<span style="font-size:1rem">%</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_scoreboard(tiles):
    """tiles: list of (label, value, css_class) tuples -> renders as scoreboard tiles."""
    html = '<div class="scoreboard">'
    for label, value, cls in tiles:
        html += (
            f'<div class="scoreboard-tile {cls if "tile" in cls else ""}">'
            f'<div class="scoreboard-label">{label}</div>'
            f'<div class="scoreboard-value {cls if "text" in cls else ""}">{value}</div>'
            f'</div>'
        )
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_standings_table(conf_table):
    """
    Custom HTML standings table -- built instead of st.dataframe because
    Streamlit's dataframe widget renders through its own internal theming
    (a canvas-based grid) that our light-theme CSS can't reach, so it was
    silently falling back to a dark, inconsistent look regardless of the
    rest of the site's styling. This gives full, direct control instead.
    """
    rows_html = ""
    for rank, row in enumerate(conf_table.itertuples(), start=1):
        rows_html += (
            f'<tr>'
            f'<td class="rank-col">{rank}</td>'
            f'<td class="team-col">{row.TEAM}</td>'
            f'<td class="num-col">{row.WINS}</td>'
            f'<td class="num-col">{row.LOSSES}</td>'
            f'<td class="num-col">{row.WIN_PCT:.1%}</td>'
            f'</tr>'
        )
    st.markdown(f"""
    <table class="standings-table">
        <thead>
            <tr><th></th><th>Team</th><th>W</th><th>L</th><th>Win %</th></tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)


def render_content_card(html_content):
    """Wraps plain text/markdown-style content in a consistently styled
    card, matching the visual language used everywhere else in the app
    (scoreboard tiles, matchup cards) instead of bare unstyled text."""
    st.markdown(f'<div class="content-card">{html_content}</div>', unsafe_allow_html=True)


def render_player_chip(rank, name, ppg, impact=None):
    impact_str = f' <span style="color: var(--text-muted)">· Impact {impact:.1f}</span>' if impact is not None else ""
    st.markdown(
        f'<div class="player-chip">{rank}. {name} — {ppg:.1f} pts/game{impact_str}</div>',
        unsafe_allow_html=True
    )


@st.cache_data
def load_raw_data():
    team_rows = pd.read_csv(DATA_DIR / "raw_team_game_logs.csv")
    games = pd.read_csv(DATA_DIR / "games_one_row_per_game.csv", parse_dates=["GAME_DATE"])
    player_path = DATA_DIR / "raw_player_game_logs.csv"
    player_rows = pd.read_csv(player_path) if player_path.exists() else None
    roster_path = DATA_DIR / "current_rosters.csv"
    current_rosters = pd.read_csv(roster_path) if roster_path.exists() else None
    return team_rows, player_rows, games, current_rosters


@st.cache_data
def load_advanced_stats():
    """
    Optional narrative-only enrichment from collect_advanced_stats.py.
    Degrades gracefully (returns None) rather than crashing the whole tab
    if the file is missing, stale, or malformed.
    """
    def _safe_load(path):
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path)
            if "TEAM_ABBREVIATION" not in df.columns:
                st.sidebar.caption(
                    f"⚠️ {path.name} is missing TEAM_ABBREVIATION — probably from an older "
                    f"run of collect_advanced_stats.py. Re-run it to fix."
                )
                return None
            return df.set_index("TEAM_ABBREVIATION")
        except Exception:
            return None

    advanced_stats = _safe_load(DATA_DIR / "advanced_team_stats.csv")
    shot_locations = _safe_load(DATA_DIR / "team_shot_locations.csv")
    return advanced_stats, shot_locations


@st.cache_data
def load_home_away_splits(_team_rows):
    """Underscore prefix on the param tells Streamlit's cache not to try
    hashing the whole dataframe -- same convention as elsewhere in this file."""
    return ha.compute_home_away_splits(_team_rows)


@st.cache_data
def load_chemistry_metrics(_team_rows):
    return chem.compute_chemistry_metrics(_team_rows)


@st.cache_data
def load_injury_lookup():
    """Real injury status from collect_injury_report.py, if it's been run.
    Returns None if not available (games far ahead won't have a report yet)."""
    path = DATA_DIR / "injury_report.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return pm.load_injury_lookup(df)
    except Exception:
        return None


@st.cache_data
def load_live_odds():
    """Live moneyline odds from collect_odds.py (The Odds API), if run.
    Returns None if not available -- sportsbooks don't post odds for most
    games until close to game time, so this will be sparse for games far
    out even when the file exists, which is expected, not a bug."""
    path = DATA_DIR / "live_odds.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        df["GAME_DATETIME"] = pd.to_datetime(df["GAME_DATETIME"], errors="coerce")
        if df["GAME_DATETIME"].dt.tz is not None:
            df["GAME_DATETIME"] = df["GAME_DATETIME"].dt.tz_localize(None)
        return df
    except Exception:
        return None


def get_market_odds_for_game(odds_df, home, away, game_date):
    """Finds the matching odds row for this specific game -- same team
    pair AND same calendar date, since the same two teams can play each
    other more than once in a season on different dates."""
    if odds_df is None:
        return None
    game_date = pd.Timestamp(game_date).normalize()
    match = odds_df[
        (odds_df["HOME_TEAM_ABBREVIATION"] == home) &
        (odds_df["AWAY_TEAM_ABBREVIATION"] == away) &
        (odds_df["GAME_DATETIME"].dt.normalize() == game_date)
    ]
    if len(match):
        return match.iloc[0].to_dict()
    return None


@st.cache_data
def load_schedule():
    path = DATA_DIR / "schedule.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["GAME_DATE"])
        if "GAME_DATETIME_EST" in df.columns:
            df["GAME_DATETIME_EST"] = pd.to_datetime(df["GAME_DATETIME_EST"], errors="coerce")
        return df
    return None


@st.cache_data
def load_calibration():
    path = DATA_DIR / "calibration_table.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_DIR / "win_probability_model.pkl")
    return bundle["model"], bundle["scaler"], bundle["features"]


def explain_prediction(row, model, features):
    """Return the top 3 plain-language factor names behind the pick (quick summary,
    no per-stat 'winner' claim)."""
    contributions = row[features].values * model.coef_[0]
    contrib_series = pd.Series(contributions, index=features).sort_values(key=abs, ascending=False)
    lines = []
    for feat in contrib_series.head(3).index:
        if feat in FEATURE_META:
            label = FEATURE_META[feat]["label"]
        elif feat.startswith("H2H"):
            label = "Head-to-head history"
        else:
            label = feat.replace("DIFF_", "").replace("_", " ").title()
        lines.append(f"- **{label}**")
    return lines


def _fmt_val(v, fmt):
    if pd.isna(v):
        return "n/a"
    if fmt == "pct":
        return f"{v:.1%}"
    if fmt == "num1":
        return f"{v:.1f}"
    if fmt == "num0":
        return f"{v:.0f}"
    if fmt == "signed1":
        return f"{v:+.1f}"
    return str(v)


def _matchup_category(label, team_a, a_off, a_allowed_by_b, team_b, b_off, b_allowed_by_a, threshold, fmt="pct"):
    """
    Checks both directions of a category at once (each team's offense vs.
    the OTHER team's defense) and picks the sentence that's actually true:
      - only one team clears the other's defense -> a real, one-sided edge
      - BOTH clear the other's defense -> not a "contest" with a winner,
        said as a joint fact
      - neither clears the other's defense -> nothing said
    """
    if pd.isna(a_off) or pd.isna(a_allowed_by_b) or pd.isna(b_off) or pd.isna(b_allowed_by_a):
        return None
    a_clears = (a_off - a_allowed_by_b) >= threshold
    b_clears = (b_off - b_allowed_by_a) >= threshold

    if a_clears and b_clears:
        return (f"**{label}:** Both teams have been strong here — {team_a} at {_fmt_val(a_off, fmt)}, "
                f"{team_b} at {_fmt_val(b_off, fmt)} — and neither defense has fully contained it lately "
                f"({team_b} allows {_fmt_val(a_allowed_by_b, fmt)}, {team_a} allows {_fmt_val(b_allowed_by_a, fmt)}). "
                f"Worth watching for both sides.")
    elif a_clears:
        return (f"**{label}:** {team_a} has been at {_fmt_val(a_off, fmt)}, better than the "
                f"{_fmt_val(a_allowed_by_b, fmt)} {team_b}'s defense has typically allowed — an edge for {team_a}.")
    elif b_clears:
        return (f"**{label}:** {team_b} has been at {_fmt_val(b_off, fmt)}, better than the "
                f"{_fmt_val(b_allowed_by_a, fmt)} {team_a}'s defense has typically allowed — an edge for {team_b}.")
    return None


def compute_matchup_edges(row):
    """Shared logic for the concrete offense-vs-opponent's-defense edges."""
    home, away = row["HOME_TEAM_ABBREVIATION"], row["AWAY_TEAM_ABBREVIATION"]
    edge_lines = [
        _matchup_category(
            "3-point shooting", home, row.get("HOME_ROLL_FG3_PCT"), row.get("AWAY_ROLL_ALLOWED_FG3_PCT"),
            away, row.get("AWAY_ROLL_FG3_PCT"), row.get("HOME_ROLL_ALLOWED_FG3_PCT"), 0.025
        ),
        _matchup_category(
            "Overall shooting", home, row.get("HOME_ROLL_FG_PCT"), row.get("AWAY_ROLL_ALLOWED_FG_PCT"),
            away, row.get("AWAY_ROLL_FG_PCT"), row.get("HOME_ROLL_ALLOWED_FG_PCT"), 0.02
        ),
    ]
    if pd.notna(row.get("HOME_OFF_RATING")) and pd.notna(row.get("AWAY_DEF_RATING")):
        edge_lines.append(_matchup_category(
            "Offensive vs. defensive rating", home, row.get("HOME_OFF_RATING"), row.get("AWAY_DEF_RATING"),
            away, row.get("AWAY_OFF_RATING"), row.get("HOME_DEF_RATING"), 3.0, fmt="num1"
        ))
    return [e for e in edge_lines if e]


def generate_narrative(row, model, features):
    """Full narrative explanation, used inside the Model Explanation expander."""
    home, away = row["HOME_TEAM_ABBREVIATION"], row["AWAY_TEAM_ABBREVIATION"]
    prob = row["PRED_HOME_WIN_PROB"]
    winner = home if prob >= 0.5 else away
    conf = row["CONFIDENCE"]
    chance_in_ten = round(prob * 10) if winner == home else round((1 - prob) * 10)

    if conf >= 0.5:
        conf_word = "a strong favorite"
    elif conf >= 0.3:
        conf_word = "clearly favored"
    elif conf >= 0.1:
        conf_word = "a slight favorite"
    else:
        conf_word = "only a slight favorite — this one's close to a coin flip"

    paragraphs = [
        f"**{winner}** is {conf_word} ({prob:.0%} for {home} at home, {1 - prob:.0%} for {away} on the road — "
        f"about {chance_in_ten} wins out of 10 in games like this)."
    ]

    paragraphs.append(
        f"**Recent form:** Over their last 10 games, {home} is {row['HOME_ROLL_WIN_PCT']:.0%} "
        f"({row['HOME_ROLL_PTS']:.1f} pts/game), and {away} is {row['AWAY_ROLL_WIN_PCT']:.0%} "
        f"({row['AWAY_ROLL_PTS']:.1f} pts/game)."
    )

    edge_lines = compute_matchup_edges(row)
    if edge_lines:
        paragraphs.append("**Matchup edges:**\n" + "\n".join(f"- {e}" for e in edge_lines))

    home_pace, away_pace = row.get("HOME_PACE"), row.get("AWAY_PACE")
    if pd.notna(home_pace) and pd.notna(away_pace):
        avg_pace = (home_pace + away_pace) / 2
        if avg_pace >= 101:
            tempo_note = "expect a faster-paced game with more possessions than usual"
        elif avg_pace <= 96:
            tempo_note = "expect a slower-paced, grind-it-out game"
        else:
            tempo_note = "a fairly average pace expected"
        paragraphs.append(
            f"**Pace:** {home} plays at {home_pace:.1f} possessions/game, {away} at {away_pace:.1f} — {tempo_note}."
        )

    if pd.notna(row.get("HOME_PAINT_FG_PCT")) and pd.notna(row.get("AWAY_PAINT_FG_PCT")):
        paragraphs.append(
            f"**Paint scoring:** {home} shoots {row['HOME_PAINT_FG_PCT']:.0%} in the paint "
            f"({row['HOME_PAINT_PTS']:.0f} pts/game from there), {away} shoots {row['AWAY_PAINT_FG_PCT']:.0%} "
            f"({row['AWAY_PAINT_PTS']:.0f} pts/game). Not adjusted for either team's specific interior defense."
        )

    home_rest, away_rest = row.get("HOME_REST_DAYS"), row.get("AWAY_REST_DAYS")
    if pd.notna(home_rest) and pd.notna(away_rest):
        rest_note = f"{home}: {home_rest:.0f} day(s) rest. {away}: {away_rest:.0f} day(s) rest."
        if row.get("HOME_BACK_TO_BACK") == 1:
            rest_note += f" {home} is on a back-to-back."
        if row.get("AWAY_BACK_TO_BACK") == 1:
            rest_note += f" {away} is on a back-to-back."
        paragraphs.append(f"**Rest:** {rest_note}")

    if pd.notna(row.get("H2H_HOME_WIN_PCT")):
        h2h_leader = home if row["H2H_HOME_WIN_PCT"] >= 0.5 else away
        paragraphs.append(
            f"**Head-to-head:** {home} won {row['H2H_HOME_WIN_PCT']:.0%} of their last 5 meetings, "
            f"outscoring {away} by {row['H2H_AVG_MARGIN']:+.1f} pts on average — favors **{h2h_leader}**."
        )

    if pd.notna(row.get("HOME_PLAYER1_NAME")) and pd.notna(row.get("AWAY_PLAYER1_NAME")):
        paragraphs.append(
            f"**Key players:** {home}'s {row.get('HOME_PLAYER1_NAME')} "
            f"({row.get('HOME_PLAYER1_PPG', 0):.1f} pts/game) vs {away}'s "
            f"{row.get('AWAY_PLAYER1_NAME')} ({row.get('AWAY_PLAYER1_PPG', 0):.1f} pts/game), "
            f"ranked by overall impact, not points alone."
        )

    return "\n\n".join(paragraphs)


def chemistry_takeaway(row):
    """
    Turns the raw Chemistry percentiles into an actual conclusion, the
    same way compute_matchup_edges does for shooting -- numbers alone
    don't help a decision, a stated takeaway does. Unlike the shooting
    matchup edges (which compare one team's OFFENSE against the OTHER
    team's DEFENSE), assist%/turnover% are the same-team, same-metric
    comparison on both sides, so a direct comparison here is safe and
    doesn't carry the "backwards logic" risk that a naive offense-vs-
    defense comparison would.
    """
    home, away = row["HOME_TEAM_ABBREVIATION"], row["AWAY_TEAM_ABBREVIATION"]
    h_ast, a_ast = row.get("HOME_AST_PERCENTILE"), row.get("AWAY_AST_PERCENTILE")
    h_tov, a_tov = row.get("HOME_TOV_PERCENTILE"), row.get("AWAY_TOV_PERCENTILE")
    if any(pd.isna(x) for x in [h_ast, a_ast, h_tov, a_tov]):
        return None

    ast_leader = home if h_ast > a_ast else away
    tov_leader = home if h_tov > a_tov else away  # higher percentile = fewer turnovers = better

    if ast_leader == tov_leader:
        other = away if ast_leader == home else home
        return f"**{ast_leader}** has the clearer chemistry edge here — better ball movement *and* better ball security than {other}."
    return (f"Mixed signal: **{ast_leader}** moves the ball better, but **{tov_leader}** takes better "
            f"care of it — no clean chemistry edge either way.")


def render_team_analysis_section(row, model, features):
    """
    Everything that's fundamentally 'about each team' on its own --
    recent form, advanced ratings, paint scoring, key players, home
    advantage, chemistry, and how the model itself weighs these factors.
    Matchup-specific content (shooting edges, pace, head-to-head, risks)
    lives in its own separate Matchup section instead, since that's this
    project's strongest, most differentiated content and earns its own
    spot rather than being buried here.
    """
    home, away = row["HOME_TEAM_ABBREVIATION"], row["AWAY_TEAM_ABBREVIATION"]

    st.markdown('<div class="eyebrow">Recent Form (Last 10 Games)</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for col, team, prefix in [(cols[0], away, "AWAY"), (cols[1], home, "HOME")]:
        with col:
            st.write(f"**{team}**")
            st.metric("Win %", f"{row.get(f'{prefix}_ROLL_WIN_PCT', 0):.0%}")
            st.metric("Points/game", f"{row.get(f'{prefix}_ROLL_PTS', 0):.1f}")

    if pd.notna(row.get("HOME_OFF_RATING")):
        st.markdown('<div class="eyebrow">Advanced Ratings</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for col, team, prefix in [(cols[0], away, "AWAY"), (cols[1], home, "HOME")]:
            with col:
                st.write(f"**{team}**")
                st.metric("Off. Rating", f"{row.get(f'{prefix}_OFF_RATING', 0):.1f}")
                st.metric("Def. Rating", f"{row.get(f'{prefix}_DEF_RATING', 0):.1f}")

    if pd.notna(row.get("HOME_PAINT_FG_PCT")):
        st.markdown('<div class="eyebrow">Paint Scoring</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for col, team, prefix in [(cols[0], away, "AWAY"), (cols[1], home, "HOME")]:
            with col:
                st.write(f"**{team}**")
                st.metric("Paint FG%", f"{row.get(f'{prefix}_PAINT_FG_PCT', 0):.0%}")
                st.metric("Paint pts/game", f"{row.get(f'{prefix}_PAINT_PTS', 0):.0f}")

    st.markdown('<div class="eyebrow">Key Players (Ranked by Impact, Not Just Points)</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for col, prefix, team in [(cols[0], "AWAY", away), (cols[1], "HOME", home)]:
        with col:
            st.write(f"**{team}**")
            for rank in [1, 2, 3]:
                name, ppg = row.get(f"{prefix}_PLAYER{rank}_NAME"), row.get(f"{prefix}_PLAYER{rank}_PPG")
                impact = row.get(f"{prefix}_PLAYER{rank}_IMPACT")
                status = row.get(f"{prefix}_PLAYER{rank}_INJURY_STATUS")
                if pd.notna(name):
                    render_player_chip(rank, name, ppg, impact)
                    if pd.notna(status) and status != "Available":
                        st.caption(f"⚠️ {status}")

    if pd.notna(row.get("HOME_TEAM_HOME_WIN_PCT")):
        st.markdown('<div class="eyebrow">Home Advantage</div>', unsafe_allow_html=True)
        st.caption(
            f"{home} wins {row['HOME_TEAM_HOME_WIN_PCT']:.0%} at home (avg margin "
            f"{row['HOME_TEAM_HOME_MARGIN']:+.1f} pts)"
            + (f" — {away} wins {row['AWAY_TEAM_AWAY_WIN_PCT']:.0%} on the road "
               f"(avg margin {row['AWAY_TEAM_AWAY_MARGIN']:+.1f} pts)."
               if pd.notna(row.get("AWAY_TEAM_AWAY_WIN_PCT")) else ".")
        )

    if pd.notna(row.get("HOME_AST_PER_FGM")):
        st.markdown('<div class="eyebrow">Chemistry (Ball Movement Proxy)</div>', unsafe_allow_html=True)
        st.caption(
            "True chemistry (lineup familiarity, on-court rapport) isn't measurable with this "
            "project's data — this is a real, standard proxy instead: how often made baskets are "
            "assisted, and how carefully the ball is protected."
        )
        takeaway = chemistry_takeaway(row)
        if takeaway:
            st.markdown(takeaway)
        cols = st.columns(2)
        for col, team, prefix in [(cols[0], away, "AWAY"), (cols[1], home, "HOME")]:
            with col:
                st.write(f"**{team}**")
                st.metric("Assisted baskets", f"{row.get(f'{prefix}_AST_PER_FGM', 0):.0%}",
                          help="Share of made field goals that came off an assist")
                st.caption(f"League percentile: {row.get(f'{prefix}_AST_PERCENTILE', 0):.0f}th")
                st.metric("Turnovers/game", f"{row.get(f'{prefix}_ROLL_TOV_CHEM', 0):.1f}")
                st.caption(f"League percentile: {row.get(f'{prefix}_TOV_PERCENTILE', 0):.0f}th (higher = fewer TOs)")

    st.markdown('<div class="eyebrow">Top Factors (By Model Weight)</div>', unsafe_allow_html=True)
    for line in explain_prediction(row, model, features):
        st.markdown(line)
    with st.expander("📖 Full explanation", expanded=False):
        st.markdown(generate_narrative(row, model, features))


def render_matchup_section(row):
    """
    This project's strongest, most differentiated content -- concrete
    offense-vs-opponent's-defense edges, not just two teams' raw stats
    side by side. Kept as its own section rather than folded into Team
    Analysis, since this is the closest thing to what a real analyst
    would actually tell you about this specific game. Also includes what
    could make this specific prediction wrong -- risk/caveat content fits
    naturally here since it's tied to this matchup's actual data (limited
    head-to-head sample, missing injury report, etc.).
    """
    home, away = row["HOME_TEAM_ABBREVIATION"], row["AWAY_TEAM_ABBREVIATION"]

    edge_lines = compute_matchup_edges(row)
    st.markdown('<div class="eyebrow">Matchup Edges</div>', unsafe_allow_html=True)
    if edge_lines:
        render_content_card("<br>".join(f"• {e}" for e in edge_lines))
    else:
        st.caption("No standout shooting/ratings mismatches found for this matchup — looks fairly even.")

    home_pace, away_pace = row.get("HOME_PACE"), row.get("AWAY_PACE")
    if pd.notna(home_pace) and pd.notna(away_pace):
        st.markdown('<div class="eyebrow">Pace</div>', unsafe_allow_html=True)
        st.caption(f"{home}: {home_pace:.1f} possessions/game — {away}: {away_pace:.1f} possessions/game")

    if pd.notna(row.get("H2H_HOME_WIN_PCT")):
        h2h_leader = home if row["H2H_HOME_WIN_PCT"] >= 0.5 else away
        st.markdown('<div class="eyebrow">Head-to-Head</div>', unsafe_allow_html=True)
        render_content_card(
            f"{home} won {row['H2H_HOME_WIN_PCT']:.0%} of their last 5 meetings, outscoring "
            f"{away} by {row['H2H_AVG_MARGIN']:+.1f} pts on average — favors <b>{h2h_leader}</b>."
        )

    st.markdown('<div class="eyebrow">Risks — What Could Make This Wrong</div>', unsafe_allow_html=True)
    risks = []

    if pd.isna(row.get("H2H_HOME_WIN_PCT")):
        risks.append("<b>Limited head-to-head history</b> — these two teams haven't met enough times "
                      "recently for that signal to be reliable here.")

    if pd.isna(row.get("HOME_PLAYER1_INJURY_STATUS")) and pd.isna(row.get("AWAY_PLAYER1_INJURY_STATUS")):
        risks.append("<b>No injury report available for this date</b> — the NBA only publishes reports "
                      "~1 day ahead of a game, so player availability this far out is an assumption, "
                      "not confirmed data.")

    conf = row.get("CONFIDENCE", 0)
    if conf < 0.15:
        risks.append("<b>This is a near-coin-flip prediction</b> — confidence is low, meaning small, "
                      "unmodeled factors (a hot shooting night, foul trouble, an off game) are more "
                      "likely to decide this one than anything in this analysis.")

    risks.append("<b>Model accuracy ceiling</b> — even this model's most confident predictions are "
                  "right roughly 80% of the time, not 100%. NBA games have real variance no model fully captures.")
    risks.append("<b>No live injury/trade news beyond the last data refresh</b> — roster and health "
                  "status reflect whenever this data was last collected, not necessarily right now.")

    for r in risks:
        st.markdown(f'<div class="risk-item">⚠️ {r}</div>', unsafe_allow_html=True)


def render_market_section(row):
    """Shows the model's probability next to the market's real implied
    probability (from The Odds API, averaged across bookmakers), and
    flags a Value Alert when they meaningfully disagree. Falls back to an
    honest explanation when no odds exist for this game -- sportsbooks
    only post most lines close to game time, so this will be sparse for
    games far out, which is expected, not a bug."""
    home, away = row["HOME_TEAM_ABBREVIATION"], row["AWAY_TEAM_ABBREVIATION"]
    market_home_prob = row.get("HOME_WIN_PROB_MARKET")

    if pd.isna(market_home_prob):
        st.markdown(
            '<div class="placeholder-box">📉 <b>No odds posted yet for this game.</b> Sportsbooks '
            'typically only release lines close to game time (a day or two out for most games, '
            'though marquee matchups sometimes get early lines). This isn\'t a data problem — '
            'check back closer to the game.</div>',
            unsafe_allow_html=True
        )
        return

    model_prob = row["PRED_HOME_WIN_PROB"]
    market_away_prob = row.get("AWAY_WIN_PROB_MARKET", 1 - market_home_prob)
    num_books = row.get("MARKET_NUM_BOOKS")
    overround = market_home_prob + market_away_prob

    st.markdown('<div class="eyebrow">Model vs. Market</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    with cols[0]:
        st.metric(f"{home} — Your Model", f"{model_prob:.1%}")
        st.metric(f"{home} — Market", f"{market_home_prob:.1%}")
    with cols[1]:
        st.metric(f"{away} — Your Model", f"{1 - model_prob:.1%}")
        st.metric(f"{away} — Market", f"{market_away_prob:.1%}")

    st.caption(
        f"Market odds averaged across {int(num_books) if pd.notna(num_books) else '?'} bookmaker(s). "
        f"Combined probability: {overround:.1%} — the amount above 100% is the sportsbooks' built-in "
        f"margin (vig), not a sign of a mispriced line."
    )

    gap = abs(model_prob - market_home_prob)
    if gap >= VALUE_ALERT_THRESHOLD:
        lean = "more toward " + home if model_prob > market_home_prob else "more toward " + away
        st.markdown(
            f'<div class="placeholder-box">⚡ <b>Value Alert:</b> Your model and the market disagree '
            f'by {gap:.1%} — your model leans {lean} than the market does. This doesn\'t mean the '
            f'market is wrong, just that it\'s worth a second look.</div>',
            unsafe_allow_html=True
        )
    else:
        st.caption(f"Model and market are within {gap:.1%} of each other — broadly in agreement.")


GAME_DETAIL_SECTIONS = [
    ("team_analysis", "Team Analysis"),
    ("matchup", "Matchup"),
    ("market", "Market"),
]


def build_day_predictions(day_games, team_rows, player_rows, games, current_rosters, model, scaler, features):
    """Computes a prediction for every game on a given day. Returns a list
    of dicts, one per game, reused by both the compact list view and the
    detail view so the row is only ever built once per game per rerun."""
    team_snap = pm.get_latest_team_snapshot(team_rows)
    player_snap = None
    if player_rows is not None and current_rosters is not None:
        player_snap = pm.get_current_top_players(player_rows, current_rosters)
    advanced_stats, shot_locations = load_advanced_stats()
    injury_lookup = load_injury_lookup()
    home_away_splits = load_home_away_splits(team_rows)
    chemistry_metrics = load_chemistry_metrics(team_rows)
    live_odds = load_live_odds()

    results = []
    for _, g in day_games.iterrows():
        home, away = g["HOME_TEAM_ABBREVIATION"], g["AWAY_TEAM_ABBREVIATION"]
        try:
            row = pm.build_hypothetical_row(
                home, away, g["GAME_DATE"], games=games,
                team_snapshot=team_snap, player_snapshot=player_snap,
                advanced_stats=advanced_stats, shot_locations=shot_locations,
                injury_lookup=injury_lookup, home_away_splits=home_away_splits,
                chemistry_metrics=chemistry_metrics,
            )
        except ValueError:
            continue
        missing_feats = [f for f in features if f not in row.index or pd.isna(row[f])]
        if missing_feats:
            continue
        X = pd.DataFrame([row[features].values], columns=features).astype(float)
        prob = model.predict_proba(scaler.transform(X))[0, 1]
        winner = home if prob >= 0.5 else away
        row["PRED_HOME_WIN_PROB"] = prob
        row["PREDICTED_WINNER"] = winner
        row["CONFIDENCE"] = abs(prob - 0.5) * 2

        market = get_market_odds_for_game(live_odds, home, away, g["GAME_DATE"])
        if market is not None:
            row["HOME_WIN_PROB_MARKET"] = market["HOME_WIN_PROB_MARKET"]
            row["AWAY_WIN_PROB_MARKET"] = market["AWAY_WIN_PROB_MARKET"]
            row["MARKET_NUM_BOOKS"] = market["NUM_BOOKS"]

        results.append(dict(
            game_id=g["GAME_ID"], home=home, away=away, row=row, prob=prob, winner=winner,
            game_datetime=g.get("GAME_DATETIME_EST"),
        ))
    return results


def render_game_detail(selected, model, features):
    """The per-game analysis page: a left nav and the section content on
    the right. The prediction itself lives in the header card above,
    which is always visible -- no separate 'Prediction' tab needed."""
    row = selected["row"]
    if st.button("← Back to games"):
        st.session_state["selected_game"] = None
        st.rerun()

    render_matchup_card(selected["away"], selected["home"], selected["prob"], game_datetime=selected["game_datetime"])

    calibration = load_calibration()
    if calibration is not None:
        conf = row["CONFIDENCE"]
        edges = [0.0, 0.10, 0.20, 0.30, 0.50, 1.01]
        labels = ["Toss-up (0-10%)", "Slight edge (10-20%)", "Moderate (20-30%)",
                  "Confident (30-50%)", "Very confident (50%+)"]
        bucket_idx = next((i for i, e in enumerate(edges[1:]) if conf <= e), len(labels) - 1)
        match = calibration[calibration["bucket"] == labels[bucket_idx]]
        if len(match):
            n, acc = int(match.iloc[0]["n_games"]), match.iloc[0]["actual_accuracy"]
            st.caption(f"📊 In the '{labels[bucket_idx]}' confidence range, this model has "
                       f"historically been right {acc:.0%} of the time (n={n} test games).")

    st.session_state.setdefault("game_section", "team_analysis")
    nav_col, content_col = st.columns([1, 3])

    with nav_col:
        st.markdown('<div class="eyebrow">Game</div>', unsafe_allow_html=True)
        for key, label in GAME_DETAIL_SECTIONS:
            if st.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state["game_section"] = key
                st.rerun()

    with content_col:
        section = st.session_state["game_section"]
        if section == "team_analysis":
            render_team_analysis_section(row, model, features)
        elif section == "matchup":
            render_matchup_section(row)
        elif section == "market":
            render_market_section(row)


def render_games_section(schedule, team_rows, player_rows, games, current_rosters, model, scaler, features):
    playable = schedule[schedule["GAME_LABEL"] != "Preseason"].copy()
    if not len(playable):
        st.info("No games found in the schedule (excluding preseason).")
        return
    scheduled = playable[playable["GAME_STATUS"] == 1]
    if not len(scheduled):
        st.warning("No upcoming (not-yet-played) games found in the schedule.")
        return

    available_dates = sorted(scheduled["GAME_DATE"].dt.date.unique())
    today = pd.Timestamp.now().date()
    tomorrow = today + pd.Timedelta(days=1)

    def _pill_label(d):
        if d == today:
            return "Today"
        if d == tomorrow:
            return "Tomorrow"
        return d.strftime("%b-%d")

    upcoming_dates = [d for d in available_dates if d >= today][:8]
    if not upcoming_dates:
        upcoming_dates = available_dates[:8]
    pill_labels = [_pill_label(d) for d in upcoming_dates]
    picked_label = st.pills("", pill_labels, default=pill_labels[0], label_visibility="collapsed", key="date_pills")
    picked_date = upcoming_dates[pill_labels.index(picked_label)] if picked_label else upcoming_dates[0]

    day_games = scheduled[scheduled["GAME_DATE"].dt.date == picked_date].sort_values("GAME_DATE")
    if not len(day_games):
        st.info(f"No games scheduled on {picked_date.strftime('%A, %B %d')}.")
        return

    predictions = build_day_predictions(
        day_games, team_rows, player_rows, games, current_rosters, model, scaler, features
    )

    if st.session_state.get("selected_game") is not None:
        selected = next((p for p in predictions if p["game_id"] == st.session_state["selected_game"]), None)
        if selected is None:
            st.session_state["selected_game"] = None
            st.rerun()
        else:
            render_game_detail(selected, model, features)
            return

    st.caption(f"{len(predictions)} game(s) on {picked_date.strftime('%A, %B %d, %Y')}")
    for p in predictions:
        favored_pct = p["prob"] if p["winner"] == p["home"] else 1 - p["prob"]
        row_cols = st.columns([5, 1])
        with row_cols[0]:
            st.markdown(
                f'<div class="game-row">'
                f'<span class="game-row-teams">{p["away"]} vs {p["home"]}</span>'
                f'<span class="game-row-pick">{favored_pct:.0%} {p["winner"]}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        with row_cols[1]:
            if st.button("View →", key=f"view_{p['game_id']}"):
                st.session_state["selected_game"] = p["game_id"]
                st.rerun()


def render_standings(schedule, team_rows, player_rows, games, current_rosters, model, scaler, features):
    # NOTE: Emirates NBA Cup games ARE included here -- per NBA rule, all
    # 67 group/knockout games count toward regular-season standings, only
    # the single Championship final doesn't. We don't yet have a reliable
    # way to identify that one specific game, so we accept a small,
    # harmless overcount (at most 1 extra game each for the 2 finalist
    # teams) rather than excluding all 67 games for all 30 teams.
    playable = schedule[schedule["GAME_LABEL"] != "Preseason"].copy()
    if not len(playable):
        st.info("No regular-season games found in the schedule yet.")
        return

    n_final = (playable["GAME_STATUS"] == 3).sum()
    n_scheduled = (playable["GAME_STATUS"] == 1).sum()
    st.caption(
        f"Projected from {n_final} completed game(s) and {n_scheduled} predicted game(s). "
        f"Win totals are expected values (sum of win probabilities), not a single simulated outcome."
    )

    if n_final + n_scheduled < 100:
        st.warning(
            "⚠️ The schedule looks incomplete (a full season is ~1,230 games). "
            "These standings will be unreliable until the full schedule is loaded — "
            "re-run `collect_schedule.py` once the NBA publishes it in full.",
            icon="⚠️"
        )

    with st.spinner("Predicting every remaining game... this can take a minute for a full season."):
        predictions = sp.predict_full_season(
            playable, team_rows, games, model, scaler, features,
            player_rows=player_rows, current_rosters=current_rosters,
        )

    if not len(predictions):
        st.warning("Could not generate any predictions from this schedule.")
        return

    standings = sp.compute_standings(predictions)

    col1, col2 = st.columns(2)
    for col, conf in [(col1, "East"), (col2, "West")]:
        with col:
            st.markdown(f'<div class="eyebrow">{conf}ern Conference</div>', unsafe_allow_html=True)
            conf_table = standings[standings["CONFERENCE"] == conf].reset_index(drop=True)
            render_standings_table(conf_table)


def main():
    inject_theme()
    st.session_state.setdefault("nav_section", "Games")
    st.session_state.setdefault("selected_game", None)

    st.markdown(
        '<div class="hero-banner">'
        '<p class="hero-title">🏀 NBA <span>PREDICTOR</span></p>'
        '<p class="hero-sub">NBA game intelligence — analysis first, prediction as the conclusion</p>'
        '</div>',
        unsafe_allow_html=True
    )

    top_nav_labels = ["Games", "Teams", "Insights"]
    picked = st.pills("", top_nav_labels, default=st.session_state["nav_section"],
                       label_visibility="collapsed", key="top_nav")
    if picked:
        st.session_state["nav_section"] = picked

    model, scaler, features = load_model()
    schedule = load_schedule()

    if schedule is None:
        st.info(
            "No schedule data found. Run:\n\n"
            "```\npython src/collect_schedule.py --season 2026-27\n```\n\n"
            "then reload this page.",
            icon="📆"
        )
        return

    team_rows, player_rows, games, current_rosters = load_raw_data()

    if current_rosters is None:
        st.warning(
            "No current-roster data found — top-scorer info may be inaccurate for any "
            "traded/signed players. Run `python src/collect_current_rosters.py --season 2026-27` "
            "for accurate player attribution.",
            icon="⚠️"
        )

    section = st.session_state["nav_section"]
    if section == "Games":
        render_games_section(schedule, team_rows, player_rows, games, current_rosters, model, scaler, features)
    elif section == "Teams":
        st.markdown(
            '<div class="placeholder-box">🏟️ <b>Coming soon.</b> Team profile pages — season summary, '
            'roster, trends for a single team, viewed independent of any specific matchup.</div>',
            unsafe_allow_html=True
        )
    elif section == "Insights":
        st.markdown('<div class="eyebrow">Projected Final Standings</div>', unsafe_allow_html=True)
        render_standings(schedule, team_rows, player_rows, games, current_rosters, model, scaler, features)


if __name__ == "__main__":
    main()
