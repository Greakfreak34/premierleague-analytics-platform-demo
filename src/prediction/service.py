"""Upcoming-fixture forecasts using saved observations, without network or fitting."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..availability import fixture_rows
from ..current_sources import utc
from ..features import add_form_features
from ..football_features import add_football_features
from ..load_matches import ROOT
from .champion import Champion, FEATURES, sha


class PredictionInputError(ValueError):
    pass


class DataUnavailableError(RuntimeError):
    pass


def safe(value):
    """JSON-native nullable context."""
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe(v) for v in value]
    if value is None or pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def league_features(history, current, home, away, kickoff):
    target_date = utc(kickoff).tz_convert("Europe/London").tz_localize(None).normalize()
    rows = pd.concat([history, current], ignore_index=True)
    rows["date"] = pd.to_datetime(rows.date)
    rows = rows.loc[rows.date < target_date]
    if rows.match_id.duplicated().any():
        raise DataUnavailableError("Duplicate league match IDs")
    if not np.isfinite(rows[["home_goals", "away_goals"]].to_numpy(dtype=float)).all() or (rows[["home_goals", "away_goals"]] < 0).any().any():
        raise DataUnavailableError("Invalid observed league scores")
    target = {"match_id": "prediction_target", "competition": "ENG-PL", "season": "2026-27",
              "date": target_date, "home_team": home, "away_team": away, "home_goals": 0, "away_goals": 0}
    # Target placeholders are consumed only AFTER its pre-match snapshot, never as inputs.
    frame = pd.concat([rows, pd.DataFrame([target])], ignore_index=True)
    frame = add_football_features(add_form_features(frame))
    row = frame.loc[frame.match_id == "prediction_target"].iloc[0]
    return {name: float(row[name]) for name in FEATURES}


class PredictionService:
    def __init__(self, root=ROOT, now=None):
        self.root = Path(root)
        # The public demo deliberately uses a reproducible simulation clock.
        self.now = now or (lambda: utc(json.loads((self.root / "data/demo_manifest.json").read_bytes())["simulation_time"]))
        try:
            self.champion = Champion(self.root / "models/logistic_simple_elo.json")
            if self.champion.metadata.get("synthetic") is not True:
                raise DataUnavailableError("Public demo requires a synthetic model")
            history_path = self.root / "data/processed/matches.csv"
            if sha(history_path.read_bytes()) != self.champion.metadata["historical_matches_sha256"]:
                raise DataUnavailableError("Historical matches differ from champion history")
            self.history = pd.read_csv(history_path)
            raw = self.root / "data/raw/availability"
            capture_id = json.loads((raw / "latest.json").read_bytes())["snapshot_id"]
            if Path(capture_id).name != capture_id:
                raise DataUnavailableError("Invalid snapshot ID")
            folder = raw / "snapshots" / capture_id
            manifest = json.loads((folder / "manifest.json").read_bytes())
            if manifest.get("synthetic") is not True:
                raise DataUnavailableError("Public demo requires synthetic source data")
            files = {}
            for name in ("bootstrap.json", "fixtures.json", "clubs.json"):
                data = (folder / name).read_bytes()
                if sha(data) != manifest["files"][name]:
                    raise DataUnavailableError("Current source checksum mismatch")
                files[name] = json.loads(data)
            self.observed_at = utc(manifest["snapshot_time"])
            self.config = files["clubs.json"]
            self.fixtures = fixture_rows({"bootstrap": files["bootstrap.json"], "fixtures": files["fixtures.json"]}, self.config["aliases"], self.config["clubs"])
            self.capture_id = capture_id
            self.current = []
            fixture_map = self.fixtures.set_index("match_id")
            for f in files["fixtures.json"]:
                match_id = f"FPL:2026-27:{f['id']}"
                if f.get("finished") is not True or match_id not in fixture_map.index:
                    continue
                row = fixture_map.loc[match_id]
                if pd.isna(row.result):
                    raise DataUnavailableError("A completed league fixture is missing its scores")
                self.current.append({"match_id": match_id, "competition": "ENG-PL", "season": "2026-27",
                    "date": utc(row.kickoff).tz_convert("Europe/London").tz_localize(None).normalize(),
                    "home_team": row.home_team, "away_team": row.away_team,
                    "home_goals": f["team_h_score"], "away_goals": f["team_a_score"], "kickoff": row.kickoff})
            self.current = pd.DataFrame(self.current, columns=["match_id", "competition", "season", "date", "home_team", "away_team", "home_goals", "away_goals", "kickoff"])
            self.state_version = sha(json.dumps(manifest, sort_keys=True).encode())
        except (OSError, KeyError, ValueError) as exc:
            raise DataUnavailableError(f"Cannot load prediction state: {exc}") from exc

    def team(self, value):
        aliases = {**{t: t for t in self.config["clubs"]}, **self.config["aliases"]}
        result = {k.casefold(): v for k, v in aliases.items()}.get(value.strip().casefold()) if isinstance(value, str) else None
        if result not in self.config["clubs"]:
            raise PredictionInputError(f"Unknown Premier League club: {value}")
        return result

    def forecast_status(self, home, away, kickoff, now):
        """Operational freshness rule, not a confidence or calibration measure."""
        reasons = []
        if kickoff - self.observed_at > pd.Timedelta(hours=72):
            reasons.append("Kickoff is more than 72 hours after the saved data snapshot.")
        if now - self.observed_at > pd.Timedelta(hours=48):
            reasons.append("The saved data snapshot is more than 48 hours old.")
        times = pd.to_datetime(self.fixtures.kickoff, utc=True)
        involved = self.fixtures.home_team.isin([home, away]) | self.fixtures.away_team.isin([home, away])
        intervening = self.fixtures.loc[involved & (times >= self.observed_at) & (times < kickoff)]
        if len(intervening):
            reasons.append(f"{len(intervening)} intervening EPL fixture(s) are in the saved schedule.")
        return {"status": "PROVISIONAL" if reasons else "CURRENT_STATE",
                "reasons": reasons, "intervening_league_fixtures": len(intervening),
                "policy": "Within 72h of data snapshot, snapshot age <=48h, no intervening saved EPL fixtures. Current-state does not certify complete coverage."}

    def predict_match(self, home_team, away_team, kickoff=None):
        home, away = self.team(home_team), self.team(away_team)
        if home == away:
            raise PredictionInputError("Home and away teams must differ")
        now = utc(self.now())
        if self.observed_at > now:
            raise DataUnavailableError("Current snapshot was collected after the service clock")
        candidates = self.fixtures.loc[(self.fixtures.home_team == home) & (self.fixtures.away_team == away)]
        if kickoff is None:
            candidates = candidates.loc[pd.to_datetime(candidates.kickoff, utc=True) > now]
            if len(candidates) != 1:
                raise PredictionInputError("No unique upcoming fixture; provide a scheduled kickoff")
            kickoff = candidates.iloc[0].kickoff
        try:
            kickoff = utc(kickoff)
        except (ValueError, TypeError) as exc:
            raise PredictionInputError("Kickoff must be a valid timezone-aware timestamp") from exc
        if kickoff <= now:
            raise PredictionInputError("Only upcoming fixtures can be predicted")
        candidates = candidates.loc[pd.to_datetime(candidates.kickoff, utc=True) == kickoff]
        if len(candidates) != 1 or pd.notna(candidates.iloc[0].result):
            raise PredictionInputError("Match and kickoff do not identify an upcoming saved EPL fixture")
        # No future or same-day results enter the champion, even in malformed source data.
        current = self.current.loc[pd.to_datetime(self.current.kickoff, utc=True) < min(now, kickoff, self.observed_at)]
        try:
            features = league_features(self.history, current, home, away, kickoff)
            probabilities = self.champion.probabilities(features)
        except (KeyError, ValueError) as exc:
            raise DataUnavailableError(f"Cannot build valid model features: {exc}") from exc
        context = self.context(home, away, kickoff, now)
        return {"schema_version": "1.1", "match_id": candidates.iloc[0].match_id,
                "home_team": home, "away_team": away, "kickoff": kickoff.isoformat(),
                "home_win": probabilities["H"], "draw": probabilities["D"], "away_win": probabilities["A"],
                "model": "logistic_simple_elo_demo", "model_version": self.champion.metadata["version"],
                "synthetic": True, "simulation_time": now.isoformat(),
                "state_version": self.state_version, "data_as_of": self.observed_at.isoformat(),
                "features": features, "context": context,
                "forecast": self.forecast_status(home, away, kickoff, now),
                "warnings": ["SYNTHETIC DEMO: fictional clubs, generated results, fixed simulation clock. Not a real football forecast.",
                             "Future intervening matches are not simulated; forecast is provisional.",
                             "Availability, sentiment, and all-competition context do not affect probabilities."]}

    def context(self, home, away, kickoff, now):
        cutoff = min(now, kickoff)
        result = {"availability": {}, "sentiment": {}, "all_competition": {}}
        for team, side in ((home, "home"), (away, "away")):
            result["availability"][side] = {"status": "unknown", "data": None}
            result["sentiment"][side] = {"status": "unknown", "data": None}
            result["all_competition"][side] = {"status": "unknown", "rest_hours": None, "matches_last_7d": None, "matches_last_14d": None}
        # Optional data failures never substitute synthetic or post-kickoff context.
        base = self.root / "data/current"
        try:
            frame = pd.read_csv(base / "team_availability_features.csv")
            for team, side in ((home, "home"), (away, "away")):
                rows = frame.loc[frame.team == team].copy()
                times = pd.to_datetime(rows.snapshot_time, utc=True)
                rows = rows.loc[(times < cutoff) & (times >= cutoff - pd.Timedelta(hours=48))]
                if len(rows):
                    result["availability"][side] = {"status": "observed", "data": safe(rows.sort_values("snapshot_time").iloc[-1].to_dict())}
        except (OSError, ValueError, KeyError):
            pass
        try:
            from ..sentiment.aggregate import eligible, summary
            frame = pd.read_csv(base / "sentiment/scored.csv")
            if "synthetic" not in frame or not frame.synthetic.eq(False).all():
                raise ValueError("Sentiment must be explicitly non-synthetic")
            end = min(cutoff, kickoff - pd.Timedelta(hours=1))
            selected = eligible(frame, end)
            for team, side in ((home, "home"), (away, "away")):
                data = summary(selected.loc[selected.team == team], end)
                result["sentiment"][side] = {"status": data["coverage"], "data": safe(data) if data["sentiment_volume"] else None}
        except (OSError, ValueError, KeyError):
            pass
        try:
            coverage = json.loads((base / "coverage.json").read_bytes())
            collected = utc(coverage["retrieved_at"])
            if collected >= cutoff:
                raise ValueError("Context was not observed before cutoff")
            frame = pd.read_csv(base / "2026-27_all_comp_matches.csv")
            times = pd.to_datetime(frame.kickoff, utc=True)
            available = pd.to_datetime(frame.available_at, utc=True)
            frame = frame.loc[(times < cutoff) & (available < cutoff) & (available <= collected)]
            for team, side in ((home, "home"), (away, "away")):
                rows = frame.loc[(frame.home_team == team) | (frame.away_team == team)]
                times = pd.to_datetime(rows.kickoff, utc=True)
                result["all_competition"][side] = {"status": "source_limited", "data_as_of": collected.isoformat(),
                    "rest_hours": (kickoff - times.max()).total_seconds() / 3600 if len(times) else None,
                    "matches_last_7d": int((times >= kickoff - pd.Timedelta(days=7)).sum()),
                    "matches_last_14d": int((times >= kickoff - pd.Timedelta(days=14)).sum())}
        except (OSError, ValueError, KeyError):
            pass
        return result


def predict_match(home_team, away_team, kickoff=None):
    return PredictionService().predict_match(home_team, away_team, kickoff)
