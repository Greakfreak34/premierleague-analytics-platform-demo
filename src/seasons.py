"""Load explicit, checksummed season snapshots; never fetch data at runtime."""
import hashlib
import json
from pathlib import Path

import pandas as pd

from .clean_matches import clean_matches
from .load_matches import ROOT, load_matches

DEFAULT_MANIFEST = ROOT / "data/raw/seasons.json"


def load_seasons(manifest: Path = DEFAULT_MANIFEST) -> tuple[pd.DataFrame, dict]:
    manifest = Path(manifest)
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    frames, reports, seen = [], [], set()
    for entry in entries:
        key = (entry["competition"], entry["season"])
        if key in seen:
            raise ValueError(f"Duplicate competition/season in manifest: {key}")
        seen.add(key)
        path = manifest.parent / entry["file"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"Raw checksum mismatch: {path.name}")
        clean, report = clean_matches(load_matches(path), entry["date_format"], season=entry["season"], competition=entry["competition"])
        validate_season(clean, entry["expected_matches"], entry["expected_teams"])
        frames.append(clean)
        reports.append({"season": entry["season"], "competition": entry["competition"], "file": entry["file"], "raw_sha256": digest, **report})
    if not frames:
        raise ValueError("Manifest contains no seasons")
    combined = pd.concat(frames, ignore_index=True).sort_values(["date", "competition", "season", "home_team", "away_team"]).reset_index(drop=True)
    if not combined.match_id.is_unique or combined.duplicated(["competition", "date", "home_team", "away_team"]).any():
        raise ValueError("Duplicate fixture across source files")
    return combined, {"clean_rows": len(combined), "form_policy": "reset each competition/season; strictly earlier dates only", "seasons": reports}


def validate_season(df: pd.DataFrame, expected_matches: int, expected_teams: int) -> None:
    if len(df) != expected_matches:
        raise ValueError(f"Expected {expected_matches} completed matches, got {len(df)}")
    teams = set(df.home_team) | set(df.away_team)
    if len(teams) != expected_teams:
        raise ValueError(f"Expected {expected_teams} teams, got {len(teams)}")
    if df.duplicated(["home_team", "away_team"]).any():
        raise ValueError("Repeated home/away fixture within season")
    if expected_matches == expected_teams * (expected_teams - 1):
        expected = {(home, away) for home in teams for away in teams if home != away}
        if set(zip(df.home_team, df.away_team)) != expected:
            raise ValueError("Incomplete double round-robin season")
