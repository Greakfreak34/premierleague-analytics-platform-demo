"""Generate all demonstration data locally: python -m tools.build_demo."""
import hashlib
import json

import numpy as np
import pandas as pd

from src.load_matches import ROOT
from src.clean_matches import clean_matches
from src.features import add_form_features
from src.football_features import add_football_features

CLUBS = [f"{name} FC" for name in ("Amber", "Birch", "Cedar", "Dune", "Elm", "Flint", "Grove", "Harbor", "Indigo", "Juniper", "Kestrel", "Lantern", "Meadow", "Northstar", "Orchid", "Pine", "Quartz", "River", "Summit", "Willow")]
OBSERVED = "2026-09-07T12:00:00+00:00"
SIMULATION = "2026-09-08T12:00:00+00:00"


def save_json(path, obj):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def save_csv(path, frame):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, date_format="%Y-%m-%d", lineterminator="\n")


def schedule(year):
    ring = list(range(20))
    first = []
    for week in range(19):
        pairs = [(ring[i], ring[-i-1]) if week % 2 == 0 else (ring[-i-1], ring[i]) for i in range(10)]
        first.append(pairs)
        ring = [ring[0], ring[-1], *ring[1:-1]]
    rounds = first + [[(away, home) for home, away in pairs] for pairs in first]
    rows = []
    for week, pairs in enumerate(rounds):
        date = pd.Timestamp(year, 8, 22) + pd.Timedelta(days=week * 7)
        for home, away in pairs:
            rows.append({"date": date, "home_team": CLUBS[home], "away_team": CLUBS[away]})
    return rows


def build():
    rng = np.random.default_rng(20260908)
    manifest, seasons = [], []
    for year in range(2022, 2026):
        rows = pd.DataFrame(schedule(year))
        rows["home_goals"] = rng.poisson(1.6, len(rows))
        rows["away_goals"] = rng.poisson(1.2, len(rows))
        name = f"{year}-{str(year+1)[-2:]}.csv"
        save_csv("data/raw/" + name, rows)
        manifest.append({"season": name[:-4], "competition": "ENG-PL", "file": name,
                         "sha256": hashlib.sha256((ROOT / "data/raw" / name).read_bytes()).hexdigest(),
                         "date_format": "%Y-%m-%d", "expected_matches": 380, "expected_teams": 20,
                         "source": "synthetic seeded generator; no external source"})
        clean, _ = clean_matches(rows, "%Y-%m-%d", season=name[:-4])
        seasons.append(clean)
    save_json("data/raw/seasons.json", manifest)
    history = pd.concat(seasons, ignore_index=True)
    save_csv("data/processed/matches.csv", history)
    featured = add_football_features(add_form_features(history))
    save_csv("data/processed/model_ready.csv", featured)
    save_csv("data/processed/match_features.csv", featured)
    config = {"season": "2026-27", "synthetic": True, "clubs": CLUBS, "aliases": {c.removesuffix(" FC"): c for c in CLUBS}}
    save_json("data/raw/current/clubs.json", config)
    fixtures = []
    for index, row in enumerate(schedule(2026)):
        kickoff = row["date"].tz_localize("UTC") + pd.Timedelta(hours=14)
        finished = kickoff < pd.Timestamp(OBSERVED)
        fixtures.append({"id": index+1, "kickoff_time": kickoff.isoformat(), "team_h": CLUBS.index(row["home_team"])+1,
                         "team_a": CLUBS.index(row["away_team"])+1, "finished": finished,
                         "team_h_score": int(rng.poisson(1.6)) if finished else None,
                         "team_a_score": int(rng.poisson(1.2)) if finished else None})
    folder = "data/raw/availability/snapshots/synthetic"
    save_json(folder + "/fixtures.json", fixtures)
    save_json(folder + "/bootstrap.json", {"synthetic": True, "teams": [{"id": i+1, "name": c} for i,c in enumerate(CLUBS)]})
    save_json(folder + "/clubs.json", config)
    save_json(folder + "/manifest.json", {"snapshot_time": OBSERVED, "synthetic": True,
        "files": {name: hashlib.sha256((ROOT / folder / name).read_bytes()).hexdigest() for name in ["fixtures.json", "bootstrap.json", "clubs.json"]}})
    save_json("data/raw/availability/latest.json", {"snapshot_id": "synthetic"})
    save_csv("data/current/team_availability_features.csv", pd.DataFrame([
        {"team": team, "snapshot_time": OBSERVED, "synthetic": True, "squad_player_count": 25,
         "available_count": 22, "unavailable_count": 2, "unknown_count": 1,
         "unavailable_strength_share": .08, "coverage_state": "synthetic_example"} for team in CLUBS]))
    save_json("data/demo_manifest.json", {"synthetic": True, "simulation_time": SIMULATION,
        "seed": 20260908, "historical_matches": 1520, "real_player_records": 0,
        "description": "Entire dataset is fictional. Not a forecast or reconstruction of real football."})
    from src.prediction.champion import export
    export()
    print("Synthetic demo ready: 1,520 generated matches, fictional fixtures, demo model.")


if __name__ == "__main__":
    build()
