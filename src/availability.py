"""Conservative FPL-listed squad availability; operational observations only."""
import hashlib
import json

import numpy as np
import pandas as pd

from .current_sources import utc
from .load_matches import ROOT

POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def normalize(payload, snapshot_time, clubs, aliases):
    snapshot_time = utc(snapshot_time)
    if not pd.Timestamp("2026-07-01", tz="UTC") <= snapshot_time < pd.Timestamp("2027-07-01", tz="UTC"):
        raise ValueError("Snapshot outside configured season")
    deadlines = [utc(event["deadline_time"]) for event in payload["events"]]
    if not deadlines or not all(pd.Timestamp("2026-07-01", tz="UTC") <= d < pd.Timestamp("2027-07-01", tz="UTC") for d in deadlines):
        raise ValueError("Availability feed is not the configured 2026-27 season")
    team_map = {t["id"]: aliases.get(t["name"], t["name"]) for t in payload["teams"]}
    if len(team_map) != 20 or set(team_map.values()) != set(clubs):
        raise ValueError("Availability roster does not match the 20 current clubs")
    rows = []
    for p in payload["elements"]:
        if p["element_type"] == 5:
            continue
        if p["element_type"] not in POSITIONS or p["team"] not in team_map:
            raise ValueError("Unknown player position or team")
        published = utc(p["news_added"]) if p.get("news_added") else None
        if published is not None and published > snapshot_time:
            raise ValueError("Provider publication timestamp is in the future")
        status = p.get("status") or "unknown"
        chance = p.get("chance_of_playing_next_round")
        if chance is not None and (not np.isfinite(float(chance)) or not 0 <= float(chance) <= 100):
            raise ValueError("Invalid availability chance")
        removed = p.get("removed") is True
        squad_status = "removed" if removed else "not_in_squad" if status == "n" else "listed"
        injury, suspension, available = "unknown", "unknown", None
        if status == "i":
            injury, available = "injured", False
        elif status == "s":
            suspension, available = "suspended", False
        elif status in ("u", "n") or removed:
            available = False
        elif status == "a" and (chance is None or float(chance) == 100):
            injury, suspension, available = "not_flagged", "not_flagged", True
        if removed:
            available = False
        rows.append({"season": "2026-27", "snapshot_time": snapshot_time.isoformat(),
                     "known_at": snapshot_time.isoformat(), "source_published_at": published.isoformat() if published is not None else None,
                     "player_id": int(p["code"]), "provider_player_id": int(p["id"]),
                     "team": team_map[p["team"]], "player": f"{p['first_name']} {p['second_name']}".strip(),
                     "position": POSITIONS[p["element_type"]], "squad_status": squad_status,
                     "injury_status": injury, "suspension_status": suspension, "status": status,
                     "available": available, "expected_available": "unknown" if available is None else "available" if available else "unavailable",
                     "reported_chance_next_round": chance, "source": "https://fantasy.premierleague.com/api/bootstrap-static/"})
    result = pd.DataFrame(rows).drop_duplicates()
    if result.empty or result.player_id.isna().any() or result.player_id.duplicated().any() or result.provider_player_id.duplicated().any():
        raise ValueError("Duplicate player identity or multiple current squads")
    result["available"] = result.available.astype("boolean")
    return result.sort_values(["team", "player_id"]).reset_index(drop=True)


def attach_strength(players, history):
    history = history[history.season == "2025-26"].copy()
    if history.empty or (pd.to_datetime(history.last_source_kickoff, utc=True) >= pd.Timestamp("2026-07-01", tz="UTC")).any():
        raise ValueError("Strength requires completed 2025-26 player history")
    if history.duplicated(["season", "player_id", "team", "position"]).any():
        raise ValueError("Duplicate historical player contributions")
    totals = history.groupby("player_id", sort=True)[["minutes", "bps"]].sum()
    if (totals.minutes <= 0).any() or not np.isfinite(totals).all().all():
        raise ValueError("Invalid prior player strength inputs")
    totals["strength"] = (90 * totals.bps / totals.minutes).clip(lower=0)
    roles = history.sort_values(["player_id", "minutes", "position"], ascending=[True, False, True]).drop_duplicates("player_id").set_index("player_id").position
    medians = totals.join(roles).groupby("position").strength.median()
    if set(medians.index) != set(POSITIONS.values()):
        raise ValueError("Missing prior-season position fallback")
    result = players.copy()
    result["prior_minutes"] = result.player_id.map(totals.minutes).fillna(0)
    result["strength"] = result.player_id.map(totals.strength)
    result["strength_imputed"] = result.strength.isna()
    result["strength"] = result.strength.fillna(result.position.map(medians))
    result["strength_source_season"] = "2025-26"
    result["strength_basis"] = "nonnegative prior BPS/90; missing=prior position median"
    return result


def aggregate(players, clubs, snapshot_time):
    snapshot_time = utc(snapshot_time)
    if players.player_id.duplicated().any():
        raise ValueError("Duplicate player across squads")
    if not np.isfinite(players.strength).all() or (players.strength < 0).any():
        raise ValueError("Strength must be finite and nonnegative")
    if (pd.to_datetime(players.known_at, utc=True) > snapshot_time).any():
        raise ValueError("Availability information not known at cutoff")
    rows = []
    for team in sorted(clubs):
        squad = players[(players.team == team) & (players.squad_status != "removed")]
        available = squad.expected_available.eq("available")
        unavailable = squad.expected_available.eq("unavailable")
        unknown = squad.expected_available.eq("unknown")
        if not (available | unavailable | unknown).all():
            raise ValueError("Unrecognized availability state")
        total = float(squad.strength.sum())
        absent = float(squad.loc[unavailable, "strength"].sum())
        uncertain = float(squad.loc[unknown, "strength"].sum())
        minutes = float(squad.prior_minutes.sum())
        top = squad.sort_values(["strength", "player_id"], ascending=[False, True]).head(5)
        rows.append({"team": team, "snapshot_time": snapshot_time.isoformat(), "squad_player_count": len(squad),
                     "available_count": int(available.sum()), "unavailable_count": int(unavailable.sum()), "unknown_count": int(unknown.sum()),
                     "strength_imputed_count": int(squad.strength_imputed.sum()),
                     "full_squad_strength": total if len(squad) else None,
                     "available_squad_strength": float(squad.loc[available, "strength"].sum()) if len(squad) else None,
                     "unavailable_strength": absent if len(squad) else None, "unknown_strength": uncertain if len(squad) else None,
                     "unavailable_strength_share": absent / total if total > 0 else None,
                     "unavailable_strength_share_upper": (absent + uncertain) / total if total > 0 else None,
                     "unavailable_share_of_minutes": float(squad.loc[unavailable, "prior_minutes"].sum()) / minutes if minutes > 0 else None,
                     "missing_top5_players": int(top.expected_available.eq("unavailable").sum()) if len(squad) else None,
                     "unknown_top5_players": int(top.expected_available.eq("unknown").sum()) if len(squad) else None,
                     **{f"missing_{name}_strength": float(squad.loc[unavailable & squad.position.isin(positions), "strength"].sum()) if len(squad) else None
                        for name, positions in (("attack", ["FWD"]), ("midfield", ["MID"]), ("defense", ["DEF", "GK"]))},
                     "coverage_state": "no_players" if squad.empty else "unknown_statuses" if unknown.any() else "observed_statuses",
                     "roster_complete": False})
    return pd.DataFrame(rows)


def fixture_rows(payload, aliases, clubs):
    teams = {t["id"]: aliases.get(t["name"], t["name"]) for t in payload["bootstrap"]["teams"]}
    rows = []
    for fixture in payload["fixtures"]:
        if fixture.get("kickoff_time") is None:
            continue
        kickoff = utc(fixture["kickoff_time"])
        if not pd.Timestamp("2026-07-01", tz="UTC") <= kickoff < pd.Timestamp("2027-07-01", tz="UTC"):
            raise ValueError("Fixture outside 2026-27")
        home, away = teams[fixture["team_h"]], teams[fixture["team_a"]]
        if home not in clubs or away not in clubs or home == away:
            raise ValueError("Invalid availability fixture teams")
        finished = fixture.get("finished") is True
        hg, ag = fixture.get("team_h_score"), fixture.get("team_a_score")
        result = ("H" if hg > ag else "D" if hg == ag else "A") if finished and hg is not None and ag is not None else None
        rows.append({"match_id": f"FPL:2026-27:{fixture['id']}", "kickoff": kickoff.isoformat(),
                     "home_team": home, "away_team": away, "result": result})
    frame = pd.DataFrame(rows)
    if frame.match_id.duplicated().any():
        raise ValueError("Duplicate fixture IDs")
    return frame.sort_values(["kickoff", "match_id"]).reset_index(drop=True)


def pre_match(fixtures, snapshots, as_of, max_age_hours=48):
    """Select only genuinely collected snapshots strictly before kickoff."""
    as_of = utc(as_of)
    rows = []
    for match in fixtures.itertuples():
        kickoff = utc(match.kickoff)
        eligible = [s for s in snapshots if utc(s["time"]) < kickoff and utc(s["time"]) <= as_of
                    and 0 <= (kickoff - utc(s["time"])).total_seconds() / 3600 <= max_age_hours]
        chosen = max(eligible, key=lambda s: (utc(s["time"]), s["id"])) if eligible else None
        row = {"match_id": match.match_id, "kickoff": kickoff.isoformat(), "home_team": match.home_team,
               "away_team": match.away_team, "record_state": "preview" if kickoff > as_of else "finalized",
               "availability_snapshot_id": chosen["id"] if chosen else None,
               "availability_snapshot_time": chosen["time"] if chosen else None,
               "availability_coverage": "observed" if chosen else "no_fresh_pre_kickoff_snapshot"}
        for side, team in (("home", match.home_team), ("away", match.away_team)):
            # Stable schema even when every historical fixture lacks a snapshot.
            if snapshots:
                row.update({f"{side}_{key}": None for key in snapshots[0]["teams"].columns if key not in ("team", "snapshot_time")})
            if chosen:
                team_rows = chosen["teams"].set_index("team")
                if team not in team_rows.index or not team_rows.index.is_unique:
                    raise ValueError("Snapshot team missing or duplicated")
                row.update({f"{side}_{key}": value for key, value in team_rows.loc[team].items() if key != "snapshot_time"})
        rows.append(row)
    return pd.DataFrame(rows)
