"""
collect_injury_report.py (diagnostic step)
--------------------------------------------
Fetches the NBA's official injury report PDF -- published by league
mandate before every game day, listing every player's status (Out,
Doubtful, Questionable, Probable, Available) and the reason. This is the
real, free, official source real analysts use; it's a PDF, not a JSON
API, so this step downloads it and prints the raw extracted text so the
actual parser can be built against the real structure (PDF text
extraction is inconsistent across tools/report versions -- better to see
the real thing than guess).

Report URL pattern: https://ak-static.cms.nba.com/referee/injury/
    Injury-Report_YYYY-MM-DD_HH_MMAM.pdf (or PM)
Reports are published continuously (every 15 min) during active periods,
so this probes backward from "now" in 15-minute steps to find the most
recent one that exists.

IMPORTANT: Needs network access to ak-static.cms.nba.com -- run on your
own machine.

Usage:
    python src/collect_injury_report.py
"""

import re
import io
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pdfplumber
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


PLAYER_ROW_RE = re.compile(
    r"^([A-Za-z.\-' ]+, [A-Za-z.\-' ]+?)\s+(Out|Doubtful|Questionable|Probable|Available)\s+(.*)$"
)
GAME_HEADER_RE = re.compile(
    r"^(\d{2}/\d{2}/\d{4})\s+(\d{1,2}:\d{2})\s*\(ET\)\s+([A-Z]{2,3})@([A-Z]{2,3})"
)


def parse_injury_report(text: str) -> list:
    """
    Parses the injury report's extracted text into structured rows. The
    report's layout repeats a "spanned" header (game date/time/matchup,
    then team name) once per team block, followed by one row per player
    on that team -- neither the game info nor the team name repeat on
    each player line, so both need to be carried forward as state while
    scanning line by line.

    Returns a list of dicts: PLAYER_NAME, TEAM_NAME, STATUS, REASON,
    GAME_DATE, MATCHUP.
    """
    rows = []
    current_game_date, current_matchup = None, None
    current_team = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        game_match = GAME_HEADER_RE.match(line)
        if game_match:
            current_game_date = game_match.group(1)
            current_matchup = f"{game_match.group(3)}@{game_match.group(4)}"
            # Team name (and possibly the first player) may trail on this
            # same line after the matchup -- strip the matched prefix and
            # keep parsing whatever's left as a normal content line.
            line = line[game_match.end():].strip()
            if not line:
                continue

        player_match = PLAYER_ROW_RE.match(line)
        if player_match:
            name, status, reason = player_match.groups()
            rows.append(dict(
                PLAYER_NAME=name.strip(), TEAM_NAME=current_team,
                STATUS=status, REASON=reason.strip(),
                GAME_DATE=current_game_date, MATCHUP=current_matchup,
            ))
        else:
            # Not a player row and not a game header -- this is a team
            # name line (e.g. "Cleveland Cavaliers"), possibly with the
            # first player's row appended directly after it on the same
            # line, since the PDF layout doesn't reliably separate them.
            # Try splitting off a trailing player row; whatever's left
            # (or all of it, if no player row is found) is the team name.
            trailing_player = PLAYER_ROW_RE.match(line)
            if trailing_player:
                current_team = line[:trailing_player.start()].strip() or current_team
                name, status, reason = trailing_player.groups()
                rows.append(dict(
                    PLAYER_NAME=name.strip(), TEAM_NAME=current_team,
                    STATUS=status, REASON=reason.strip(),
                    GAME_DATE=current_game_date, MATCHUP=current_matchup,
                ))
            else:
                current_team = line

    return rows



BASE_URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{date}_{hour:02d}_{minute:02d}{ampm}.pdf"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def _candidate_urls(start: datetime, lookback_hours: int = 30):
    """Yields candidate report URLs, stepping backward in 15-min increments."""
    t = start
    end = start - timedelta(hours=lookback_hours)
    while t > end:
        hour12 = t.hour % 12
        hour12 = 12 if hour12 == 0 else hour12
        ampm = "PM" if t.hour >= 12 else "AM"
        url = BASE_URL.format(date=t.strftime("%Y-%m-%d"), hour=hour12, minute=t.minute, ampm=ampm)
        yield t, url
        t -= timedelta(minutes=15)


def find_latest_report():
    print("Searching for the most recent injury report (stepping back in 15-min increments)...")
    for timestamp, url in _candidate_urls(datetime.now()):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=8)
            if resp.status_code == 200 and resp.headers.get("Content-Type", "").startswith("application/pdf"):
                print(f"Found report: {timestamp} -> {url}")
                return timestamp, resp.content
        except requests.RequestException:
            continue
    return None, None


def main():
    timestamp, pdf_bytes = find_latest_report()
    if pdf_bytes is None:
        print("No report found in the last 30 hours. The season may not have started yet, "
              "or the URL pattern may need adjusting.")
        return

    raw_path = DATA_DIR / "raw_injury_report.pdf"
    raw_path.write_bytes(pdf_bytes)
    print(f"Saved raw PDF -> {raw_path}")

    all_text = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        print(f"Extracting text from {len(pdf.pages)} page(s)...")
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text.append(text)
    full_text = "\n".join(all_text)

    rows = parse_injury_report(full_text)
    if not rows:
        print("WARNING: 0 player rows parsed. The report's text layout may differ from what "
              "the parser expects -- share the raw text (data/raw_injury_report_text.txt, "
              "saved below) so the parser can be adjusted against the real format.")
        (DATA_DIR / "raw_injury_report_text.txt").write_text(full_text, encoding="utf-8")
        return

    df = pd.DataFrame(rows)
    df["REPORT_TIMESTAMP"] = timestamp
    out_path = DATA_DIR / "injury_report.csv"
    df.to_csv(out_path, index=False)
    print(f"\nParsed {len(df)} player rows -> {out_path}")
    print(f"\nStatus breakdown:\n{df['STATUS'].value_counts().to_string()}")
    print(f"\nSample:\n{df.head(5).to_string(index=False)}")


if __name__ == "__main__":
    main()
