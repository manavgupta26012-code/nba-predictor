"""
app.py
------
NBA Season Predictor -- predicts every scheduled game of the season with
a plain-English explanation, and projects final Eastern/Western Conference
standings from those predictions.

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
# describe it in plain English -- this is what powers the extensive
# narrative explanation, not just the short "favors HOME/AWAY" bullets.
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
    "DIFF_MOMENTUM_SCORE": dict(label="Recent trend", plain="a 'hot or cold streak' score that weighs their last few games more than older ones", home_col="HOME_MOMENTUM_SCORE", away_col="AWAY_MOMENTUM_SCORE", fmt="signed1", unit="pts"),
    "DIFF_GAMES_LAST5":  dict(label="Recent workload", plain="games played in the last 5 days", home_col="HOME_GAMES_LAST5", away_col="AWAY_GAMES_LAST5", fmt="num0", unit="games"),
    "DIFF_MISSING_IMPACT": dict(label="Missing impact", plain="combined 'Impact Score' of any top-3 player assumed unavailable — a blend of scoring, rebounding, passing, and defense, not just points", home_col="HOME_MISSING_IMPACT", away_col="AWAY_MISSING_IMPACT", fmt="num1", unit=""),
}

st.set_page_config(page_title="NBA Season Predictor", layout="wide", page_icon="🏀")


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

    /* Force the light background regardless of Streamlit's own theme
       setting or config.toml -- config.toml is only read at server
       startup and can also be overridden by a user's personal
       dark/light choice in Streamlit's own settings menu, so relying on
       it alone left the page background dark while these custom
       components assumed light, producing barely-readable text. */
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
    [data-testid="stExpander"] {
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
    }
    [data-testid="stDataFrame"] { background-color: #FFFFFF !important; }

    h1, h2, h3 { font-family: 'Oswald', sans-serif !important; letter-spacing: 0.01em; color: var(--text-primary); }

    .hero-banner {
        border-bottom: 3px solid var(--accent);
        padding-bottom: 14px;
        margin-bottom: 6px;
    }
    .hero-title {
        font-family: 'Oswald', sans-serif;
        font-weight: 700;
        font-size: 2.1rem;
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
    .matchup-bar-seg-1 { }
    .matchup-bar-seg-2 { }
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
    red if a heavy underdog (< 20%) -- matches how sportsbook-style UIs
    typically signal a lopsided game, not just binary win/lose color."""
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
    gradient win-probability bar with a marker at the split point --
    matches the target design. Full detail lives in the tabs below this."""
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
                    <div class="matchup-bar-seg-1" style="width:{away_pct}%; background:{away_color}"></div>
                    <div class="matchup-bar-seg-2" style="width:{home_pct}%; background:{home_color}"></div>
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
    if the file is missing, stale, or malformed -- this is supplementary
    data, not something the model depends on.
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
def load_injury_lookup():
    """Real injury status from collect_injury_report.py, if it's been run.
    Returns None if not available (games far ahead won't have a report yet
    since the NBA only publishes ~1 day out)."""
    path = DATA_DIR / "injury_report.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return pm.load_injury_lookup(df)
    except Exception:
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
    no per-stat 'winner' claim -- see generate_narrative for why)."""
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
        it means both are simply strong in this category (e.g. two good
        3-point shooting teams facing defenses that aren't great at
        stopping it) -- said as a joint fact, not two competing "edge for
        X" claims that read like a contradiction
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


def generate_narrative(row, model, features):
    """
    Builds a plain-English explanation combining the model's prediction
    with concrete, human-readable basketball analytics -- offense-vs-
    opponent's-defense matchup comparisons, not just team-vs-team stats or
    a description of the model's own math.
    """
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

    # Concrete matchup edges: this team's offense vs. the opponent's actual
    # recent defense in the same category -- not just two numbers side by
    # side, and not a description of the model's coefficients.
    # NOTE: the 4th arg is what the OTHER team's defense allows (the bar
    # this team's offense needs to clear), not this team's own defense --
    # HOME_ROLL_ALLOWED_* / AWAY_ROLL_ALLOWED_* must be cross-wired here.
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

    # Offensive/defensive rating (points per 100 possessions) -- the
    # standard advanced-stats way to compare team strength, since it's
    # already pace-adjusted (unlike raw points per game). Both ratings are
    # in the same units, so this is a clean "clears the bar" comparison
    # just like the shooting ones above.
    if pd.notna(row.get("HOME_OFF_RATING")) and pd.notna(row.get("AWAY_DEF_RATING")):
        edge_lines.append(_matchup_category(
            "Offensive vs. defensive rating", home, row.get("HOME_OFF_RATING"), row.get("AWAY_DEF_RATING"),
            away, row.get("AWAY_OFF_RATING"), row.get("HOME_DEF_RATING"), 3.0, fmt="num1"
        ))


    edge_lines = [e for e in edge_lines if e]
    if edge_lines:
        paragraphs.append("**Matchup edges:**\n" + "\n".join(f"- {e}" for e in edge_lines))

    # Pace context: informational, not an "edge" -- there's no opposing
    # team's pace to "beat," it just changes how the game is likely to play.
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

    # Paint scoring: shown as descriptive facts, not a "vs. opponent's
    # interior defense" edge -- that would need opponent-specific
    # shot-location data (who allows what, by zone) that isn't collected
    # here. What's shown is real (last-10-games shot zone breakdown), just
    # not opponent-adjusted the way the shooting matchup edges above are.
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

    # Real injury status (from the NBA's official report, when available)
    # for any of either team's top-3 players.
    injury_lines = []
    for team_label, prefix in [(home, "HOME"), (away, "AWAY")]:
        for rank in [1, 2, 3]:
            name = row.get(f"{prefix}_PLAYER{rank}_NAME")
            status = row.get(f"{prefix}_PLAYER{rank}_INJURY_STATUS")
            if pd.notna(name) and pd.notna(status) and status != "Available":
                injury_lines.append(f"{name} ({team_label}) — **{status}**")
    if injury_lines:
        paragraphs.append(
            "**Injury report:** " + "; ".join(injury_lines) +
            ". Per the NBA's most recently published official injury report."
        )
    elif pd.notna(row.get("HOME_PLAYER1_NAME")):
        paragraphs.append(
            "_No injury report was available for this game (the NBA only publishes reports "
            "~1 day ahead) — top players are assumed available unless manually flagged otherwise._"
        )

    return "\n\n".join(paragraphs)


def render_wins_tab(row, model, features):
    """Everything from before: probability recap, quick factors, full narrative, calibration."""
    render_scoreboard([
        ("Home Win Probability", f"{row['PRED_HOME_WIN_PROB']:.1%}", "accent-text"),
        ("Predicted Winner", row["PREDICTED_WINNER"], "win-tile win-text"),
        ("Confidence", f"{row['CONFIDENCE']:.1%}", ""),
    ])

    st.markdown('<div class="eyebrow">Why This Prediction</div>', unsafe_allow_html=True)
    for line in explain_prediction(row, model, features):
        st.markdown(line)
    with st.expander("📖 Full explanation", expanded=False):
        st.markdown(generate_narrative(row, model, features))

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


def render_season_stats_tab(row):
    """A cleaner, side-by-side stat-sheet view of the same underlying data --
    recent form, advanced ratings, paint scoring, and key players ranked by
    impact. Same data as the Wins tab's narrative, laid out for scanning
    rather than reading."""
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
                st.metric("Pace", f"{row.get(f'{prefix}_PACE', 0):.1f}")

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


def main():
    inject_theme()
    st.markdown(
        '<div class="hero-banner">'
        '<p class="hero-title">🏀 NBA <span>Season Predictor</span></p>'
        '<p class="hero-sub">Every game of the season, predicted with an explanation — plus projected final standings</p>'
        '</div>',
        unsafe_allow_html=True
    )

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

    tab1, tab2 = st.tabs(["🏀 Game Predictions", "🏆 Projected Final Standings"])

    with tab1:
        render_game_predictions(schedule, team_rows, player_rows, games, current_rosters, model, scaler, features)
    with tab2:
        render_standings(schedule, team_rows, player_rows, games, current_rosters, model, scaler, features)


def render_game_predictions(schedule, team_rows, player_rows, games, current_rosters, model, scaler, features):
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

    # Show a manageable window of pills (next 8 available dates from today)
    upcoming_dates = [d for d in available_dates if d >= today][:8]
    if not upcoming_dates:
        upcoming_dates = available_dates[:8]
    pill_labels = [_pill_label(d) for d in upcoming_dates]

    picked_label = st.pills("", pill_labels, default=pill_labels[0], label_visibility="collapsed")
    picked_date = upcoming_dates[pill_labels.index(picked_label)] if picked_label else upcoming_dates[0]

    day_games = scheduled[scheduled["GAME_DATE"].dt.date == picked_date].sort_values("GAME_DATE")
    if not len(day_games):
        st.info(f"No games scheduled on {picked_date.strftime('%A, %B %d')}.")
        return

    st.caption(f"{len(day_games)} game(s) on {picked_date.strftime('%A, %B %d, %Y')}")

    team_snap = pm.get_latest_team_snapshot(team_rows)
    player_snap = None
    if player_rows is not None and current_rosters is not None:
        player_snap = pm.get_current_top_players(player_rows, current_rosters)
    advanced_stats, shot_locations = load_advanced_stats()
    injury_lookup = load_injury_lookup()

    game_list = list(day_games.iterrows())
    for pair_start in range(0, len(game_list), 2):
        cols = st.columns(2)
        for col, (_, g) in zip(cols, game_list[pair_start:pair_start + 2]):
            with col:
                home, away = g["HOME_TEAM_ABBREVIATION"], g["AWAY_TEAM_ABBREVIATION"]
                try:
                    row = pm.build_hypothetical_row(
                        home, away, g["GAME_DATE"], games=games,
                        team_snapshot=team_snap, player_snapshot=player_snap,
                        advanced_stats=advanced_stats, shot_locations=shot_locations,
                        injury_lookup=injury_lookup,
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

                game_datetime = g.get("GAME_DATETIME_EST")
                render_matchup_card(away, home, prob, game_datetime=game_datetime)

                wins_tab, stats_tab = st.tabs(["Wins", "Season Stats"])
                with wins_tab:
                    render_wins_tab(row, model, features)
                with stats_tab:
                    render_season_stats_tab(row)



def render_standings(schedule, team_rows, player_rows, games, current_rosters, model, scaler, features):
    # NOTE: Emirates NBA Cup games ARE included here -- per NBA rule, all
    # 67 group/knockout games count toward regular-season standings, only
    # the single Championship final doesn't. We don't yet have a reliable
    # way to identify that one specific game in the schedule data, so we
    # accept a small, harmless overcount (at most 1 extra game each for
    # the 2 finalist teams) rather than the much larger error of excluding
    # all 67 games for all 30 teams, which is what an earlier version of
    # this code did.
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
            conf_table.index = conf_table.index + 1
            st.dataframe(
                conf_table[["TEAM", "WINS", "LOSSES", "WIN_PCT"]].rename(columns={
                    "TEAM": "Team", "WINS": "W", "LOSSES": "L", "WIN_PCT": "Win %"
                }).style.format({"Win %": "{:.1%}"}),
                use_container_width=True
            )


if __name__ == "__main__":
    main()
