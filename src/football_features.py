"""Pre-match context snapshots; update results only after every fixture on a date."""
from collections import defaultdict, deque

import pandas as pd

GROUPS = {
    "elo": ["home_elo", "away_elo", "elo_diff"],
    "venue": ["home_home_pts_5", "home_home_gd_5", "home_home_history_count_5", "away_away_pts_5", "away_away_gd_5", "away_away_history_count_5"],
    "standings": [f"{side}_{field}" for side in ("home", "away") for field in ("league_position", "league_played", "league_points", "league_gd", "league_gf")] + ["position_diff"],
    "rest": ["home_rest_days", "away_rest_days", "rest_days_diff", "home_rest_known", "away_rest_known"],
}
FOOTBALL_COLUMNS = [c for columns in GROUPS.values() for c in columns]
POLICY = {"elo_initial": 1500, "elo_k": 20, "elo_home_advantage": 65,
          "elo_boundary": "Carry across seasons within competition, no decay; unseen teams start at 1500; returning teams retain last observed rating.",
          "other_boundaries": "Venue form, league table, and rest reset each season.",
          "standings": "Points, goal difference, goals scored descending; exact ties share competition rank (1,1,3). No sanctions or head-to-head rules.",
          "rest": "Calendar days since last league match this season; unknown=0 with known=0 flag. Cups/Europe unavailable.",
          "timing": "Strictly earlier calendar dates; all same-day updates applied after feature snapshots."}


def add_football_features(matches):
    ordered = matches.sort_values(["date", "competition", "season", "home_team", "away_team"]).reset_index(drop=True)
    ratings = defaultdict(lambda: 1500.0)
    tables = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))  # played, points, GD, GF
    venues = defaultdict(lambda: deque(maxlen=5))
    last = {}
    # Season membership is known before play; no future scores or dates used here.
    for key, season in ordered.groupby(["competition", "season"]):
        for team in set(season.home_team) | set(season.away_team):
            tables[key][team]
    output = []
    for date, day in ordered.groupby("date", sort=True):
        deltas = defaultdict(float)
        for row in day.itertuples():
            key = (row.competition, row.season)
            table = tables[key]
            values = {}
            for side, team in (("home", row.home_team), ("away", row.away_team)):
                team_key = (*key, team)
                values[f"{side}_elo"] = ratings[(row.competition, team)]
                played, points, gd, gf = table[team]
                rank = 1 + sum(tuple(stats[1:]) > (points, gd, gf) for stats in table.values())
                for field, value in zip(("league_position", "league_played", "league_points", "league_gd", "league_gf"), (rank, played, points, gd, gf)):
                    values[f"{side}_{field}"] = value
                history = venues[(*team_key, side)]
                values[f"{side}_{side}_pts_5"] = sum(p for p, _ in history)
                values[f"{side}_{side}_gd_5"] = sum(g for _, g in history)
                values[f"{side}_{side}_history_count_5"] = len(history)
                values[f"{side}_rest_known"] = int(team_key in last)
                values[f"{side}_rest_days"] = (date - last[team_key]).days if team_key in last else 0
            for diff, left, right in (("elo_diff", "home_elo", "away_elo"), ("position_diff", "home_league_position", "away_league_position"), ("rest_days_diff", "home_rest_days", "away_rest_days")):
                values[diff] = values[left] - values[right]
            output.append(values)
            expected = 1 / (1 + 10 ** ((values["away_elo"] - values["home_elo"] - POLICY["elo_home_advantage"]) / 400))
            actual = 1 if row.home_goals > row.away_goals else 0.5 if row.home_goals == row.away_goals else 0
            delta = POLICY["elo_k"] * (actual - expected)
            deltas[(row.competition, row.home_team)] += delta
            deltas[(row.competition, row.away_team)] -= delta
        for row in day.itertuples():
            for side, team, gf, ga in (("home", row.home_team, row.home_goals, row.away_goals), ("away", row.away_team, row.away_goals, row.home_goals)):
                key = (row.competition, row.season, team)
                points = 3 if gf > ga else 1 if gf == ga else 0
                stats = tables[key[:2]][team]
                for i, increment in enumerate((1, points, gf - ga, gf)):
                    stats[i] += increment
                venues[(*key, side)].append((points, gf - ga))
                last[key] = date
        for key, delta in deltas.items():
            ratings[key] += delta
    return pd.concat([ordered, pd.DataFrame(output, columns=FOOTBALL_COLUMNS)], axis=1)
