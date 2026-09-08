"""Previous-five totals across home and away games, strictly before match date."""
from collections import defaultdict, deque

import pandas as pd

FEATURE_COLUMNS = [f"{side}_{metric}_5" for side in ("home", "away") for metric in ("form_pts", "gf", "ga", "gd", "history_count")]


def add_form_features(matches: pd.DataFrame) -> pd.DataFrame:
    ordered = matches.sort_values(["date", "competition", "season", "home_team", "away_team"]).reset_index(drop=True)
    history = defaultdict(lambda: deque(maxlen=5))
    features = []
    for _, day in ordered.groupby("date", sort=True):
        # Snapshot every fixture before adding any result from this date.
        for row in day.itertuples():
            values = {}
            for side in ("home", "away"):
                prior = history[(row.competition, row.season, getattr(row, f"{side}_team"))]
                pts, gf, ga = (sum(item[i] for item in prior) for i in range(3))
                for metric, value in zip(("form_pts", "gf", "ga", "gd", "history_count"), (pts, gf, ga, gf - ga, len(prior))):
                    values[f"{side}_{metric}_5"] = value
            features.append(values)
        for row in day.itertuples():
            for team, gf, ga in ((row.home_team, row.home_goals, row.away_goals), (row.away_team, row.away_goals, row.home_goals)):
                history[(row.competition, row.season, team)].append((3 if gf > ga else 1 if gf == ga else 0, gf, ga))
    return pd.concat([ordered, pd.DataFrame(features, columns=FEATURE_COLUMNS)], axis=1)
