"""Start with uvicorn app.main:app --reload."""
from functools import lru_cache
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.prediction.service import DataUnavailableError, PredictionInputError, PredictionService

app = FastAPI(title="Football analytics — SYNTHETIC DEMO", version="1.1.0")


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    home_team: str = Field(min_length=1, max_length=100)
    away_team: str = Field(min_length=1, max_length=100)
    kickoff: str | None = Field(default=None, max_length=50)


class ContextResponse(BaseModel):
    availability: dict[str, Any]
    sentiment: dict[str, Any]
    all_competition: dict[str, Any]


class ForecastResponse(BaseModel):
    status: Literal["PROVISIONAL", "CURRENT_STATE"]
    reasons: list[str]
    intervening_league_fixtures: int
    policy: str


class PredictResponse(BaseModel):
    schema_version: Literal["1.1"]
    match_id: str
    home_team: str
    away_team: str
    kickoff: str
    home_win: float = Field(ge=0, le=1)
    draw: float = Field(ge=0, le=1)
    away_win: float = Field(ge=0, le=1)
    model: Literal["logistic_simple_elo_demo"]
    synthetic: Literal[True]
    simulation_time: str
    model_version: str
    state_version: str
    data_as_of: str
    features: dict[str, float]
    context: ContextResponse
    forecast: ForecastResponse
    warnings: list[str]

    @model_validator(mode="after")
    def probability_sum(self):
        if abs(self.home_win + self.draw + self.away_win - 1) > 1e-8:
            raise ValueError("Probabilities must sum to one")
        return self


@lru_cache(maxsize=1)
def get_service():
    try:
        return PredictionService()
    except DataUnavailableError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.get("/health")
def health():
    try:
        service = get_service()
        return {"status": "ready", "schema_version": "1.1", "model": "logistic_simple_elo_demo", "synthetic": True,
                "model_version": service.champion.metadata["version"], "data_as_of": service.observed_at.isoformat(),
                "freshness": "saved_snapshot"}
    except HTTPException:
        return JSONResponse(status_code=503, content={"status": "unavailable", "schema_version": "1.1"})


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, service: PredictionService = Depends(get_service)):
    try:
        return service.predict_match(request.home_team, request.away_team, request.kickoff)
    except PredictionInputError as exc:
        raise HTTPException(422, str(exc)) from exc
    except DataUnavailableError as exc:
        raise HTTPException(503, str(exc)) from exc
