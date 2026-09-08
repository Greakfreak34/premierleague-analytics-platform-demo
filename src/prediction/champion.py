"""Export/load the frozen training-only champion as transparent JSON parameters."""
import hashlib
import json

import numpy as np
import pandas as pd

from ..load_matches import ROOT

FEATURES = ["home_form_pts_5", "away_form_pts_5", "home_gd_5", "away_gd_5",
            "home_elo", "away_elo", "elo_diff"]
ARTIFACT = ROOT / "models/logistic_simple_elo.json"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode()


def export():
    from ..modeling import fit_models, split_data
    from sklearn.metrics import log_loss
    import sklearn
    path = ROOT / "data/processed/model_ready.csv"
    parts = split_data(pd.read_csv(path))
    model = fit_models(parts["train"], {"simple": FEATURES})["simple"]
    scaler, classifier = model["scaler"], model["classifier"]
    validation = log_loss(parts["validation"].result, model.predict_proba(parts["validation"][FEATURES]))
    payload = {"model": "logistic_simple_elo_demo", "synthetic": True, "benchmark_tag": "synthetic-demo-only",
               "features": FEATURES, "classes": classifier.classes_.tolist(),
               "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
               "coefficients": classifier.coef_.tolist(), "intercept": classifier.intercept_.tolist(),
               "training_seasons": ["2022-23", "2023-24"], "training_rows": len(parts["train"]),
               "validation_log_loss": float(validation), "sklearn_version": sklearn.__version__,
               "historical_features_sha256": sha(path.read_bytes()),
               "historical_matches_sha256": sha((ROOT / "data/processed/matches.csv").read_bytes()),
               "policy": "Synthetic training seasons only; no real football model or accuracy claims. No validation/test refit."}
    payload["version"] = sha(encoded(payload))
    ARTIFACT.parent.mkdir(exist_ok=True)
    if ARTIFACT.exists() and ARTIFACT.read_bytes() != encoded(payload) + b"\n":
        raise ValueError("Refusing to overwrite different champion artifact")
    ARTIFACT.write_bytes(encoded(payload) + b"\n")
    print(f"Exported {payload['model']} {payload['version'][:12]}; validation LL {validation:.6f}")


class Champion:
    def __init__(self, path=ARTIFACT):
        self.metadata = json.loads(path.read_bytes())
        payload = dict(self.metadata)
        version = payload.pop("version")
        if sha(encoded(payload)) != version or payload["features"] != FEATURES or payload["classes"] != ["A", "D", "H"]:
            raise ValueError("Invalid champion artifact identity or feature schema")
        for name, shape in (("mean", (7,)), ("scale", (7,)), ("coefficients", (3, 7)), ("intercept", (3,))):
            value = np.asarray(payload[name], dtype=float)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError("Invalid champion parameter dimensions")
        if (np.asarray(payload["scale"]) <= 0).any():
            raise ValueError("Invalid scaler")

    def probabilities(self, features):
        if set(features) != set(FEATURES):
            raise ValueError("Missing or unexpected model features")
        x = np.array([features[k] for k in FEATURES], dtype=float)
        if not np.isfinite(x).all():
            raise ValueError("Model features must be finite")
        m = self.metadata
        logits = ((x - m["mean"]) / m["scale"]) @ np.array(m["coefficients"]).T + m["intercept"]
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        return dict(zip(m["classes"], probabilities.tolist()))


if __name__ == "__main__":
    export()
