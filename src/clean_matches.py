"""Normalize results and reject ambiguous or invalid historical records."""
import hashlib
import json
import re

import pandas as pd


def clean_matches(raw: pd.DataFrame, date_format: str = "%a %b %d %Y", *, season: str = "2022-23", competition: str = "ENG-PL") -> tuple[pd.DataFrame, dict]:
    if not re.fullmatch(r"\d{4}-\d{2}", season) or int(season[-2:]) != (int(season[:4]) + 1) % 100:
        raise ValueError("Season must be consecutive years in YYYY-YY format")
    if not competition.strip():
        raise ValueError("Competition must not be blank")
    df = raw.copy()
    df.columns = [re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_") for c in df.columns]
    df = df.rename(columns={"team_1": "home_team", "team_2": "away_team", "hometeam": "home_team", "awayteam": "away_team", "fthg": "home_goals", "ftag": "away_goals"})
    # Source betting/statistics fields are unused; their normalized names may collide.
    df = df.loc[:, df.columns.isin(["date", "home_team", "away_team", "home_goals", "away_goals", "ft"])]
    if df.columns.duplicated().any():
        raise ValueError("Ambiguous duplicate column names")
    if "ft" in df and not {"home_goals", "away_goals"}.issubset(df.columns):
        scores = df["ft"].astype("string").str.strip()
        bad = scores.notna() & scores.ne("") & ~scores.str.fullmatch(r"\d+\s*-\s*\d+", na=False)
        if bad.any():
            raise ValueError("Malformed full-time score")
        df[["home_goals", "away_goals"]] = scores.str.extract(r"^(\d+)\s*-\s*(\d+)$")
    columns = ["date", "home_team", "away_team", "home_goals", "away_goals"]
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df[columns].copy()
    df["date"] = pd.to_datetime(df["date"], format=date_format, errors="raise").dt.normalize()
    if df["date"].isna().any():
        raise ValueError("Missing match date")
    start_year = int(season[:4])
    if not df.date.between(pd.Timestamp(start_year, 7, 1), pd.Timestamp(start_year + 1, 6, 30)).all():
        raise ValueError("Match date falls outside the declared season (July–June)")
    for column in ["home_team", "away_team"]:
        df[column] = df[column].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
        if (df[column].isna() | df[column].eq("")).any():
            raise ValueError("Missing team name")
    if df.home_team.eq(df.away_team).any():
        raise ValueError("A team cannot play itself")
    for column in ["home_goals", "away_goals"]:
        text_values = df[column].astype("string").str.strip()
        values = pd.to_numeric(text_values.mask(text_values.eq("")), errors="raise")
        if ((values.dropna() < 0) | (values.dropna() % 1 != 0)).any():
            raise ValueError("Goals must be nonnegative integers")
        df[column] = values
    report = {"raw_rows": len(df), "missing_score_rows_dropped": int(df[["home_goals", "away_goals"]].isna().any(axis=1).sum())}
    df = df.dropna(subset=["home_goals", "away_goals"])
    before = len(df)
    df = df.drop_duplicates()
    report["duplicate_rows_dropped"] = before - len(df)
    if df.duplicated(["date", "home_team", "away_team"]).any():
        raise ValueError("Conflicting scores for the same match")
    if df.empty:
        raise ValueError("No completed matches remain")
    df[["home_goals", "away_goals"]] = df[["home_goals", "away_goals"]].astype("int64")
    df["result"] = ["H" if h > a else "A" if h < a else "D" for h, a in zip(df.home_goals, df.away_goals)]
    df["season"] = season
    df["competition"] = competition.strip()
    df["match_id"] = [hashlib.sha256(json.dumps([competition.strip(), season, d.strftime("%Y-%m-%d"), h, a], ensure_ascii=False).encode()).hexdigest()[:20] for d, h, a in zip(df.date, df.home_team, df.away_team)]
    df = df.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    if not df.match_id.is_unique:
        raise ValueError("Duplicate match IDs")
    report["clean_rows"] = len(df)
    return df[["match_id", "competition", "season"] + columns + ["result"]], report
