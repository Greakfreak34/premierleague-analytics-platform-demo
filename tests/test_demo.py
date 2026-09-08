import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from src.clean_matches import clean_matches
from src.features import add_form_features
from src.football_features import add_football_features
from src.load_matches import ROOT
from src.modeling import fit_models, split_data
from src.prediction.champion import Champion, FEATURES
from src.prediction.service import PredictionService, PredictionInputError
from src.predict import main
from src.seasons import load_seasons
from src.sentiment.aggregate import eligible, summary
from src.sentiment.normalize import club_for_text


@pytest.fixture(scope="session", autouse=True)
def setup_demo():
    if not (ROOT / "data/demo_manifest.json").exists():
        from tools.build_demo import build
        build()


@pytest.fixture(scope="module")
def service():
    return PredictionService()


def test_generated_manifest_and_seasons():
    data, _ = load_seasons()
    assert len(data) == 1520 and data.match_id.is_unique
    assert data.groupby("season").size().eq(380).all()
    assert data.result.isin(["H", "D", "A"]).all()
    assert json.loads((ROOT / "data/demo_manifest.json").read_bytes())["synthetic"] is True


def test_cleaning_duplicates_and_missing_scores():
    data = pd.DataFrame([{"date": "2026-08-22", "home_team": "Amber", "away_team": "Birch", "home_goals": 2, "away_goals": 0}])
    clean, _ = clean_matches(pd.concat([data, data]), "%Y-%m-%d", season="2026-27")
    assert len(clean) == 1 and clean.iloc[0].result == "H"


def test_no_current_or_future_result_leakage():
    data = pd.read_csv(ROOT / "data/processed/matches.csv").iloc[:30].copy()
    data.date = pd.to_datetime(data.date)
    before = add_football_features(add_form_features(data))
    data.loc[data.date == data.date.max(), "home_goals"] = 50
    after = add_football_features(add_form_features(data.sample(frac=1, random_state=1)))
    pd.testing.assert_frame_equal(before[FEATURES], after[FEATURES])
    assert before.iloc[0].home_form_pts_5 == 0 and before.iloc[0].home_elo == 1500


def test_scaler_train_only_and_disjoint_splits():
    parts = split_data(pd.read_csv(ROOT / "data/processed/model_ready.csv"))
    assert not set(parts["train"].match_id) & set(parts["validation"].match_id)
    assert not set(parts["train"].match_id) & set(parts["test"].match_id)
    model = fit_models(parts["train"], {"simple": FEATURES})["simple"]
    np.testing.assert_allclose(model["scaler"].mean_, parts["train"][FEATURES].mean())
    assert "result" not in FEATURES and "home_goals" not in FEATURES


def test_model_artifact_demo_only():
    assert Champion().metadata["synthetic"] is True
    with pytest.raises(ValueError):
        Champion().probabilities({})


def test_api_cli_parity_and_determinism(service, capsys):
    expected = service.predict_match("Amber", "Birch")
    assert expected == service.predict_match("Amber", "Birch")
    assert expected["synthetic"] is True and expected["model"].endswith("_demo")
    assert sum(expected[k] for k in ["home_win", "draw", "away_win"]) == pytest.approx(1)
    response = TestClient(app).post("/predict", json={"home_team": "Amber", "away_team": "Birch"})
    assert response.status_code == 200, response.text
    main(["Amber", "Birch", "--json"], service)
    assert response.json() == json.loads(capsys.readouterr().out) == expected
    assert TestClient(app).get("/health").json()["synthetic"] is True


def test_human_output(service, capsys):
    main(["Amber", "Birch"], service)
    text = capsys.readouterr().out
    assert "SYNTHETIC DEMO" in text and "Model pick:" in text and "PROVISIONAL" in text


@pytest.mark.parametrize("home,away,kickoff", [("Unknown", "Birch", None), ("Amber", "Amber", None), ("Amber", "Birch", "not a date"), ("Amber", "Birch", "2020-01-01T00:00:00Z")])
def test_invalid_requests(service, home, away, kickoff):
    with pytest.raises(PredictionInputError):
        service.predict_match(home, away, kickoff)


def test_missing_sentiment_is_unknown(service):
    result = service.predict_match("Amber", "Birch")
    assert result["context"]["sentiment"]["home"]["status"] == "unknown"
    assert result["context"]["availability"]["home"]["data"]["synthetic"] is True


def test_sentiment_cutoff_and_unknown():
    posts = pd.DataFrame([{"post_id": "fictional", "revision": "1", "capture_id": "demo", "team": "Amber FC",
        "published_at": "2026-09-07T11:00:00Z", "collected_at": "2026-09-07T13:00:00Z", "sentiment_score": .5,
        "sentiment_label": "positive"}])
    cutoff = "2026-09-07T12:00:00Z"
    filtered = eligible(posts, cutoff)
    assert filtered.empty and summary(filtered, cutoff)["sentiment_mean"] is None


def test_alias_context():
    config = {"clubs": ["Amber FC"], "aliases": {"Amber": "Amber FC"}}
    assert club_for_text("Amber FC scored a fantastic goal", config) == "Amber FC"
    assert club_for_text("Amber is my favorite color", config) is None
