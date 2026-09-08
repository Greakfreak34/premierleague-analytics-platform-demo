# Football analytics platform — public engineering demo

**All data and predictions in this demo are synthetic.** Clubs, scores, squad
availability, and the simulation clock are fictional. The fitted demo model is
generated locally and is not the private project's evaluated champion.

This is a runnable demonstration of a football forecasting workflow: raw CSVs →
clean matches → leakage-safe rolling form and Elo → chronological logistic
baseline → reusable prediction service → FastAPI and human-readable CLI.

## Run locally

Python 3.10+ (tested on 3.10). From this repository root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-api.txt
python -m tools.build_demo
python -m pytest -q
python -m src.predict Amber Birch
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs. `GET /health` reports readiness and
`POST /predict` accepts `{"home_team":"Amber","away_team":"Birch"}`.
Kickoff is resolved from the generated schedule. The full response is available
from `python -m src.predict Amber Birch --json`.

Setup generates 1,520 historical matches across four seasons, a fictional current
schedule, availability examples, and a logistic model. A fixed September 2026
simulation clock makes the demo runnable later without changing its results.
The generated outputs are ignored by Git. No external football source is called.

## What this demonstrates

- Input validation, duplicate handling, and checksummed season manifests.
- Previous-five form using strictly earlier dates; season-reset policy.
- Elo updates after feature snapshots, with persistence across seasons.
- Train: 2022–24; validation: 2024–25; test: 2025–26, all synthetic.
- StandardScaler fitted on training data only; explicit feature allowlist.
- JSON model export with provenance, checksums, and deterministic inference.
- `/health`, `/predict`, CLI/API parity, ranked probabilities and provisional status.
- Context fields separated from model inputs; absent sentiment is unknown.

Optional experiments: `python -m src.pipeline --report-dir reports/generated`,
`python -m src.football_modeling --report-dir reports/generated`. All resulting
metrics describe the fictional generator, not real-world football performance.
`src/model_comparison.py` retains the model-family comparison implementation.

The optional `src/sentiment` code includes conservative club matching, as-of
aggregation and a pinned pretrained scorer. No live social collector is included;
no real fan posts or downloaded model weights are shipped. Its dependency file is
separate so the API demo needs neither PyTorch nor a model download.

## Scope and provenance

This repository reproduces **software behavior**, not the original private
research results. It contains no third-party match dumps, website snapshots,
player datasets, private fitted model, or old private Git history.
See [PUBLICATION.md](PUBLICATION.md) for the distribution boundary and
[DATA_SOURCES.md](DATA_SOURCES.md) for source/license considerations.

There is no affiliation with any football competition, club, or data provider.
The inherited `ENG-PL` and season identifiers are schema examples, not claims
that generated rows represent actual competition results.
