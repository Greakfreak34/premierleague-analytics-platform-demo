"""Fixed chronological baseline: python -m src.modeling."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS
from .load_matches import ROOT

LABELS = ["A", "D", "H"]  # Alphabetical order also matches sklearn log_loss.
SPLITS = {"train": ["2022-23", "2023-24"], "validation": ["2024-25"], "test": ["2025-26"]}
FEATURE_SETS = {
    "simple": ["home_form_pts_5", "away_form_pts_5", "home_gd_5", "away_gd_5"],
    "full": list(FEATURE_COLUMNS),
}


def split_data(df, seasons=None):
    seasons = SPLITS if seasons is None else seasons
    assigned = [s for part in seasons.values() for s in part]
    if set(seasons) != set(SPLITS) or len(assigned) != len(set(assigned)):
        raise ValueError("Train/validation/test seasons must not overlap")
    required = {"match_id", "competition", "season", "date", "result", *FEATURE_COLUMNS}
    if required - set(df.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(df.columns))}")
    df = df.copy()
    if df.match_id.isna().any() or not df.match_id.is_unique:
        raise ValueError("Match IDs must be present and unique")
    if set(df.competition) != {"ENG-PL"} or set(df.season) != set(assigned):
        raise ValueError("Expected exactly the configured Premier League seasons")
    if not df.result.isin(LABELS).all():
        raise ValueError("Targets must be H/D/A")
    if not np.isfinite(df[FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Features must be finite numeric values")
    df["date"] = pd.to_datetime(df.date, format="%Y-%m-%d", errors="raise")
    if df.date.isna().any():
        raise ValueError("Missing dates")
    for season, rows in df.groupby("season"):
        year = int(season[:4])
        if not rows.date.between(pd.Timestamp(year, 7, 1), pd.Timestamp(year + 1, 6, 30)).all():
            raise ValueError("Date outside declared season")
    df = df.sort_values(["date", "match_id"]).reset_index(drop=True)
    parts = {name: df[df.season.isin(seasons[name])].copy() for name in SPLITS}
    if any(part.empty for part in parts.values()):
        raise ValueError("Every split must contain matches")
    if not parts["train"].date.max() < parts["validation"].date.min() or not parts["validation"].date.max() < parts["test"].date.min():
        raise ValueError("Splits must be strictly chronological")
    if set(parts["train"].result) != set(LABELS):
        raise ValueError("Training data must contain all three classes")
    return parts


def predictors(df, feature_set, feature_sets=None):
    # Explicit allowlist: metadata, targets, and final scores cannot enter a fit.
    sets = FEATURE_SETS if feature_sets is None else feature_sets
    return df.loc[:, sets[feature_set]].astype(float)


def fit_models(train, feature_sets=None):
    feature_sets = FEATURE_SETS if feature_sets is None else feature_sets
    models = {"naive": DummyClassifier(strategy="prior").fit(predictors(train, "simple"), train.result)}
    for name in feature_sets:
        model = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42))])
        model.fit(predictors(train, name, feature_sets), train.result)
        models[name] = model
    return models


def evaluate(model, data, feature_set, feature_sets=None):
    x = predictors(data, feature_set, feature_sets)
    prediction = model.predict(x)
    raw = model.predict_proba(x)
    probability = raw[:, [list(model.classes_).index(label) for label in LABELS]]
    metrics = {
        "accuracy": float(accuracy_score(data.result, prediction)),
        "macro_f1": float(f1_score(data.result, prediction, labels=LABELS, average="macro", zero_division=0)),
        "log_loss": float(log_loss(data.result, probability, labels=LABELS)),
        "confusion_matrix": confusion_matrix(data.result, prediction, labels=LABELS).tolist(),
        "per_class": {label: values for label, values in classification_report(data.result, prediction, labels=LABELS, output_dict=True, zero_division=0).items() if label in LABELS},
    }
    predictions = data[["match_id", "season", "date", "result"]].copy()
    predictions["prediction"] = prediction
    for index, label in enumerate(LABELS):
        predictions[f"prob_{label}"] = probability[:, index]
    return metrics, predictions


def run_experiment(df):
    parts = split_data(df)
    models = fit_models(parts["train"])
    report = {
        "sklearn_version": sklearn.__version__,
        "class_order": LABELS,
        "splits": {name: {"seasons": SPLITS[name], "rows": len(part), "first_date": part.date.min().strftime("%Y-%m-%d"), "last_date": part.date.max().strftime("%Y-%m-%d")} for name, part in parts.items()},
        "feature_sets": FEATURE_SETS,
        "protocol": "Fit only train; select lowest validation log loss (simple wins exact ties); evaluate naive and selected model on test without refitting.",
        "naive_probability_policy": "Predict training majority class; probabilities are training class frequencies (DummyClassifier strategy=prior).",
        "logistic_parameters": {"C": 1.0, "solver": "lbfgs", "max_iter": 1000, "random_state": 42, "class_weight": None},
        "validation": {}, "test": {},
    }
    prediction_frames = []
    for name, model in models.items():
        metrics, predictions = evaluate(model, parts["validation"], "simple" if name == "naive" else name)
        report["validation"][name] = metrics
        prediction_frames.append(predictions.assign(split="validation", model=name))
    selected = min(FEATURE_SETS, key=lambda name: report["validation"][name]["log_loss"])
    report["selected_feature_set"] = selected
    # Test is evaluated only after feature selection, never used to rank candidates.
    for name in ("naive", selected):
        metrics, predictions = evaluate(models[name], parts["test"], "simple" if name == "naive" else name)
        report["test"][name] = metrics
        prediction_frames.append(predictions.assign(split="test", model=name))
    return report, pd.concat(prediction_frames, ignore_index=True), models


def save_reports(report, predictions, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "model_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    predictions.to_csv(output / "model_predictions.csv", index=False, date_format="%Y-%m-%d", lineterminator="\n")
    panels = [(split, name, metrics) for split in ("validation", "test") for name, metrics in report[split].items()]
    maximum = max(max(map(max, metrics["confusion_matrix"])) for _, _, metrics in panels)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, (split, name, metrics) in zip(axes.flat, panels):
        ax.set_visible(True)
        matrix = np.array(metrics["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha="center", va="center", color="white" if value > maximum / 2 else "black")
        ax.set(xticks=range(3), yticks=range(3), xticklabels=LABELS, yticklabels=LABELS, xlabel="Predicted", ylabel="Actual", title=f"{split.title()} — {name}\nAccuracy {metrics['accuracy']:.3f}")
    fig.suptitle("Chronological baseline confusion matrices (match counts)")
    fig.savefig(output / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/model_ready.csv")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    report, predictions, _ = run_experiment(pd.read_csv(args.data))
    report["input_sha256"] = hashlib.sha256(args.data.read_bytes()).hexdigest()
    save_reports(report, predictions, args.report_dir)
    for split, values in report["splits"].items():
        print(f"{split.title()} seasons: {', '.join(values['seasons'])} ({values['rows']} matches)")
    print(f"Selected logistic features by validation log loss: {report['selected_feature_set']}")
    for split in ("validation", "test"):
        for name, metrics in report[split].items():
            print(f"{split.title()} / {name}: accuracy={metrics['accuracy']:.4f}, macro F1={metrics['macro_f1']:.4f}, log loss={metrics['log_loss']:.4f}")


if __name__ == "__main__":
    main()
