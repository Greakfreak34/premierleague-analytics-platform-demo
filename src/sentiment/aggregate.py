"""As-of revision selection and non-overlapping sentiment windows."""
import pandas as pd

from ..current_sources import utc


def eligible(posts, cutoff):
    cutoff = utc(cutoff)
    if posts.empty:
        return posts.copy()
    # Latest revision actually observed BEFORE cutoff; late edits never rewrite history.
    observed = pd.to_datetime(posts.collected_at, utc=True)
    selected = posts.loc[observed < cutoff].sort_values(["collected_at", "capture_id", "post_id"])
    selected = selected.drop_duplicates("post_id", keep="last")
    published = pd.to_datetime(selected.published_at, utc=True)
    return selected.loc[(published < cutoff) & (published >= cutoff - pd.Timedelta(hours=72))]


def summary(posts, cutoff):
    cutoff = utc(cutoff)
    n = len(posts)
    result = {"coverage": "observed" if n else "unknown", "sentiment_volume": n,
              "sentiment_mean": None, "sentiment_median": None, "sentiment_std": None,
              "positive_share": None, "negative_share": None, "neutral_share": None,
              "sentiment_24h": None, "sentiment_72h": None, "sentiment_previous_48h": None,
              "sentiment_change": None, "volume_24h": 0, "volume_previous_48h": 0}
    if not n:
        return result
    scores = posts.sentiment_score
    result.update(sentiment_mean=float(scores.mean()), sentiment_median=float(scores.median()),
                  sentiment_std=float(scores.std(ddof=0)), sentiment_72h=float(scores.mean()))
    for label in ("positive", "negative", "neutral"):
        result[f"{label}_share"] = float((posts.sentiment_label == label).mean())
    recent = pd.to_datetime(posts.published_at, utc=True) >= cutoff - pd.Timedelta(hours=24)
    result["volume_24h"], result["volume_previous_48h"] = int(recent.sum()), int((~recent).sum())
    if recent.any():
        result["sentiment_24h"] = float(scores[recent].mean())
    if (~recent).any():
        result["sentiment_previous_48h"] = float(scores[~recent].mean())
    if recent.any() and (~recent).any():
        result["sentiment_change"] = result["sentiment_24h"] - result["sentiment_previous_48h"]
    return result


def teams(posts, clubs, cutoff):
    selected = eligible(posts, cutoff)
    return pd.DataFrame([{"team": team, "snapshot_time": utc(cutoff).isoformat(),
                          **summary(selected.loc[selected.team == team], cutoff)} for team in clubs])


def matches(posts, fixtures, as_of):
    rows = []
    for fixture in fixtures.to_dict("records"):
        cutoff = utc(fixture["kickoff"]) - pd.Timedelta(hours=1)
        end = min(cutoff, utc(as_of))
        selected = eligible(posts, end)
        row = {k: fixture[k] for k in ("match_id", "kickoff", "home_team", "away_team")}
        row.update(snapshot_time=end.isoformat(), cutoff_time=cutoff.isoformat(),
                   record_state="finalized" if utc(as_of) >= cutoff else "preview")
        for venue in ("home", "away"):
            subset = selected.loc[selected.team == fixture[f"{venue}_team"]]
            row.update({f"{venue}_{k}": v for k, v in summary(subset, end).items()})
            row[f"{venue}_post_refs"] = "|".join(sorted(subset.capture_id + ":" + subset.post_id + "@" + subset.revision))
        home, away = row["home_sentiment_mean"], row["away_sentiment_mean"]
        row["sentiment_diff"] = home - away if home is not None and away is not None else None
        row["volume_diff"] = row["home_sentiment_volume"] - row["away_sentiment_volume"]
        rows.append(row)
    return pd.DataFrame(rows)
