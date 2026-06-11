"""
Official FIFA World Cup 2026 group draw and group-stage fixture list.

Hosts: USA / Canada / Mexico — 48 teams, 12 groups of 4, 72 group-stage matches.
Group composition and fixtures reflect the final draw (5 Dec 2025) and the
March 2026 play-off resolutions (Czechia, Bosnia, Turkey, Sweden, Iraq, DR Congo).

Team display names below use common short forms.  A few differ from the spelling
used in the historical results dataset / Elo table — `NAME_MAP` maps display name
-> Elo-table name so ratings resolve correctly.
"""
from __future__ import annotations

# Display name (here) -> name used in data/processed/current_elos.csv
NAME_MAP = {
    "Czechia": "Czech Republic",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Curacao": "Curaçao",
}


def elo_name(team: str) -> str:
    """Resolve a fixture display name to the name used in the Elo table."""
    return NAME_MAP.get(team, team)


# ── Group composition ──────────────────────────────────────────────────────────
GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["USA", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# Host nations get a (mild) home-field edge when playing in their own country.
HOST_NATIONS = {"USA", "Mexico", "Canada"}


# ── Group-stage fixtures (72 matches) ───────────────────────────────────────────
# Each: group, date (YYYY-MM-DD), local kickoff (HH:MM), venue, home, away.
FIXTURES = [
    {"group": "A", "date": "2026-06-11", "time": "13:00", "venue": "Estadio Azteca, Mexico City", "home": "Mexico", "away": "South Africa"},
    {"group": "A", "date": "2026-06-11", "time": "20:00", "venue": "Estadio Akron, Guadalajara", "home": "South Korea", "away": "Czechia"},
    {"group": "B", "date": "2026-06-12", "time": "15:00", "venue": "BMO Field, Toronto", "home": "Canada", "away": "Bosnia and Herzegovina"},
    {"group": "D", "date": "2026-06-12", "time": "18:00", "venue": "SoFi Stadium, Los Angeles", "home": "USA", "away": "Paraguay"},
    {"group": "B", "date": "2026-06-13", "time": "12:00", "venue": "Levi's Stadium, San Francisco Bay Area", "home": "Qatar", "away": "Switzerland"},
    {"group": "C", "date": "2026-06-13", "time": "18:00", "venue": "MetLife Stadium, New York/New Jersey", "home": "Brazil", "away": "Morocco"},
    {"group": "C", "date": "2026-06-13", "time": "21:00", "venue": "Gillette Stadium, Boston", "home": "Haiti", "away": "Scotland"},
    {"group": "D", "date": "2026-06-13", "time": "21:00", "venue": "BC Place, Vancouver", "home": "Australia", "away": "Turkey"},
    {"group": "E", "date": "2026-06-14", "time": "12:00", "venue": "NRG Stadium, Houston", "home": "Germany", "away": "Curacao"},
    {"group": "F", "date": "2026-06-14", "time": "15:00", "venue": "AT&T Stadium, Dallas", "home": "Netherlands", "away": "Japan"},
    {"group": "E", "date": "2026-06-14", "time": "19:00", "venue": "Lincoln Financial Field, Philadelphia", "home": "Ivory Coast", "away": "Ecuador"},
    {"group": "F", "date": "2026-06-14", "time": "20:00", "venue": "Estadio BBVA, Monterrey", "home": "Sweden", "away": "Tunisia"},
    {"group": "H", "date": "2026-06-15", "time": "12:00", "venue": "Mercedes-Benz Stadium, Atlanta", "home": "Spain", "away": "Cape Verde"},
    {"group": "G", "date": "2026-06-15", "time": "15:00", "venue": "Lumen Field, Seattle", "home": "Belgium", "away": "Egypt"},
    {"group": "H", "date": "2026-06-15", "time": "18:00", "venue": "Hard Rock Stadium, Miami", "home": "Saudi Arabia", "away": "Uruguay"},
    {"group": "G", "date": "2026-06-15", "time": "21:00", "venue": "SoFi Stadium, Los Angeles", "home": "Iran", "away": "New Zealand"},
    {"group": "I", "date": "2026-06-16", "time": "15:00", "venue": "MetLife Stadium, New York/New Jersey", "home": "France", "away": "Senegal"},
    {"group": "I", "date": "2026-06-16", "time": "18:00", "venue": "Gillette Stadium, Boston", "home": "Iraq", "away": "Norway"},
    {"group": "J", "date": "2026-06-16", "time": "20:00", "venue": "Arrowhead Stadium, Kansas City", "home": "Argentina", "away": "Algeria"},
    {"group": "J", "date": "2026-06-16", "time": "21:00", "venue": "Levi's Stadium, San Francisco Bay Area", "home": "Austria", "away": "Jordan"},
    {"group": "K", "date": "2026-06-17", "time": "12:00", "venue": "NRG Stadium, Houston", "home": "Portugal", "away": "DR Congo"},
    {"group": "L", "date": "2026-06-17", "time": "15:00", "venue": "AT&T Stadium, Dallas", "home": "England", "away": "Croatia"},
    {"group": "L", "date": "2026-06-17", "time": "19:00", "venue": "BMO Field, Toronto", "home": "Ghana", "away": "Panama"},
    {"group": "K", "date": "2026-06-17", "time": "20:00", "venue": "Estadio Azteca, Mexico City", "home": "Uzbekistan", "away": "Colombia"},
    {"group": "A", "date": "2026-06-18", "time": "12:00", "venue": "Mercedes-Benz Stadium, Atlanta", "home": "Czechia", "away": "South Africa"},
    {"group": "B", "date": "2026-06-18", "time": "12:00", "venue": "SoFi Stadium, Los Angeles", "home": "Switzerland", "away": "Bosnia and Herzegovina"},
    {"group": "B", "date": "2026-06-18", "time": "15:00", "venue": "BC Place, Vancouver", "home": "Canada", "away": "Qatar"},
    {"group": "A", "date": "2026-06-18", "time": "21:00", "venue": "Estadio Akron, Guadalajara", "home": "Mexico", "away": "South Korea"},
    {"group": "D", "date": "2026-06-19", "time": "12:00", "venue": "Lumen Field, Seattle", "home": "USA", "away": "Australia"},
    {"group": "C", "date": "2026-06-19", "time": "18:00", "venue": "Gillette Stadium, Boston", "home": "Scotland", "away": "Morocco"},
    {"group": "C", "date": "2026-06-19", "time": "21:00", "venue": "Lincoln Financial Field, Philadelphia", "home": "Brazil", "away": "Haiti"},
    {"group": "D", "date": "2026-06-19", "time": "21:00", "venue": "Levi's Stadium, San Francisco Bay Area", "home": "Turkey", "away": "Paraguay"},
    {"group": "F", "date": "2026-06-20", "time": "12:00", "venue": "NRG Stadium, Houston", "home": "Netherlands", "away": "Sweden"},
    {"group": "E", "date": "2026-06-20", "time": "16:00", "venue": "BMO Field, Toronto", "home": "Germany", "away": "Ivory Coast"},
    {"group": "E", "date": "2026-06-20", "time": "19:00", "venue": "Arrowhead Stadium, Kansas City", "home": "Ecuador", "away": "Curacao"},
    {"group": "F", "date": "2026-06-20", "time": "22:00", "venue": "Estadio BBVA, Monterrey", "home": "Tunisia", "away": "Japan"},
    {"group": "H", "date": "2026-06-21", "time": "12:00", "venue": "Mercedes-Benz Stadium, Atlanta", "home": "Spain", "away": "Saudi Arabia"},
    {"group": "G", "date": "2026-06-21", "time": "12:00", "venue": "SoFi Stadium, Los Angeles", "home": "Belgium", "away": "Iran"},
    {"group": "H", "date": "2026-06-21", "time": "18:00", "venue": "Hard Rock Stadium, Miami", "home": "Uruguay", "away": "Cape Verde"},
    {"group": "G", "date": "2026-06-21", "time": "18:00", "venue": "BC Place, Vancouver", "home": "New Zealand", "away": "Egypt"},
    {"group": "J", "date": "2026-06-22", "time": "12:00", "venue": "AT&T Stadium, Dallas", "home": "Argentina", "away": "Austria"},
    {"group": "I", "date": "2026-06-22", "time": "17:00", "venue": "Lincoln Financial Field, Philadelphia", "home": "France", "away": "Iraq"},
    {"group": "I", "date": "2026-06-22", "time": "20:00", "venue": "MetLife Stadium, New York/New Jersey", "home": "Norway", "away": "Senegal"},
    {"group": "J", "date": "2026-06-22", "time": "20:00", "venue": "Levi's Stadium, San Francisco Bay Area", "home": "Jordan", "away": "Algeria"},
    {"group": "K", "date": "2026-06-23", "time": "12:00", "venue": "NRG Stadium, Houston", "home": "Portugal", "away": "Uzbekistan"},
    {"group": "L", "date": "2026-06-23", "time": "16:00", "venue": "Gillette Stadium, Boston", "home": "England", "away": "Ghana"},
    {"group": "L", "date": "2026-06-23", "time": "19:00", "venue": "BMO Field, Toronto", "home": "Panama", "away": "Croatia"},
    {"group": "K", "date": "2026-06-23", "time": "20:00", "venue": "Estadio Akron, Guadalajara", "home": "Colombia", "away": "DR Congo"},
    {"group": "B", "date": "2026-06-24", "time": "12:00", "venue": "BC Place, Vancouver", "home": "Switzerland", "away": "Canada"},
    {"group": "B", "date": "2026-06-24", "time": "12:00", "venue": "Lumen Field, Seattle", "home": "Bosnia and Herzegovina", "away": "Qatar"},
    {"group": "C", "date": "2026-06-24", "time": "18:00", "venue": "Hard Rock Stadium, Miami", "home": "Scotland", "away": "Brazil"},
    {"group": "C", "date": "2026-06-24", "time": "18:00", "venue": "Mercedes-Benz Stadium, Atlanta", "home": "Morocco", "away": "Haiti"},
    {"group": "A", "date": "2026-06-24", "time": "19:00", "venue": "Estadio Azteca, Mexico City", "home": "Czechia", "away": "Mexico"},
    {"group": "A", "date": "2026-06-24", "time": "19:00", "venue": "Estadio BBVA, Monterrey", "home": "South Africa", "away": "South Korea"},
    {"group": "E", "date": "2026-06-25", "time": "16:00", "venue": "MetLife Stadium, New York/New Jersey", "home": "Ecuador", "away": "Germany"},
    {"group": "E", "date": "2026-06-25", "time": "16:00", "venue": "Lincoln Financial Field, Philadelphia", "home": "Curacao", "away": "Ivory Coast"},
    {"group": "F", "date": "2026-06-25", "time": "18:00", "venue": "AT&T Stadium, Dallas", "home": "Japan", "away": "Sweden"},
    {"group": "F", "date": "2026-06-25", "time": "18:00", "venue": "Arrowhead Stadium, Kansas City", "home": "Tunisia", "away": "Netherlands"},
    {"group": "D", "date": "2026-06-25", "time": "19:00", "venue": "SoFi Stadium, Los Angeles", "home": "Turkey", "away": "USA"},
    {"group": "D", "date": "2026-06-25", "time": "19:00", "venue": "Levi's Stadium, San Francisco Bay Area", "home": "Paraguay", "away": "Australia"},
    {"group": "I", "date": "2026-06-26", "time": "15:00", "venue": "Gillette Stadium, Boston", "home": "Norway", "away": "France"},
    {"group": "I", "date": "2026-06-26", "time": "15:00", "venue": "BMO Field, Toronto", "home": "Senegal", "away": "Iraq"},
    {"group": "H", "date": "2026-06-26", "time": "19:00", "venue": "NRG Stadium, Houston", "home": "Cape Verde", "away": "Saudi Arabia"},
    {"group": "H", "date": "2026-06-26", "time": "18:00", "venue": "Estadio Akron, Guadalajara", "home": "Uruguay", "away": "Spain"},
    {"group": "G", "date": "2026-06-26", "time": "20:00", "venue": "Lumen Field, Seattle", "home": "Egypt", "away": "Iran"},
    {"group": "G", "date": "2026-06-26", "time": "20:00", "venue": "BC Place, Vancouver", "home": "New Zealand", "away": "Belgium"},
    {"group": "L", "date": "2026-06-27", "time": "17:00", "venue": "MetLife Stadium, New York/New Jersey", "home": "Panama", "away": "England"},
    {"group": "L", "date": "2026-06-27", "time": "17:00", "venue": "Lincoln Financial Field, Philadelphia", "home": "Croatia", "away": "Ghana"},
    {"group": "K", "date": "2026-06-27", "time": "19:00", "venue": "Hard Rock Stadium, Miami", "home": "Colombia", "away": "Portugal"},
    {"group": "K", "date": "2026-06-27", "time": "19:00", "venue": "Mercedes-Benz Stadium, Atlanta", "home": "DR Congo", "away": "Uzbekistan"},
    {"group": "J", "date": "2026-06-27", "time": "21:00", "venue": "Arrowhead Stadium, Kansas City", "home": "Algeria", "away": "Austria"},
    {"group": "J", "date": "2026-06-27", "time": "21:00", "venue": "AT&T Stadium, Dallas", "home": "Jordan", "away": "Argentina"},
]


def host_advantage(home: str, away: str) -> bool:
    """A match is non-neutral only when a host nation plays at home."""
    return home in HOST_NATIONS
