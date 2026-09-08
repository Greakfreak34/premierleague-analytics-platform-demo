"""Small chronological model-family comparison: python -m src.model_comparison."""
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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .football_modeling import ABLATIONS
from .load_matches import ROOT
from .modeling import LABELS as OLD_LABELS, SPLITS, evaluate, predictors, split_data

LABELS = ["H", "D", "A"]
FEATURES = {name: list(ABLATIONS[name]) for name in ("simple_elo", "combined")}
BASELINE = "logistic/simple_elo/0"
METRICS = ["accuracy", "macro_f1", "log_loss", "brier_score", "draw_precision",
           "draw_recall", "mean_predicted_draw_probability"]


def configurations():
    forest = [{"n_estimators": n, "max_depth": depth, "min_samples_leaf": leaf}
              for n in (200, 500) for depth in (4, 8, None) for leaf in (5, 10)]
    boosting = [dict(n_estimators=n, learning_rate=rate, max_depth=depth, min_samples_leaf=10)
                for n, rate, depth in ((100, .03, 2), (200, .03, 2), (100, .05, 3), (100, .1, 2))]
    return [{"id": f"{family}/{features}/{i}", "family": family, "features": features, "parameters": params}
            for family, grid in (("logistic", [{}]), ("random_forest", forest), ("gradient_boosting", boosting))
            for features in FEATURES for i, params in enumerate(grid)]


def estimator(config):
    if config["family"] == "logistic":
        return Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(
            C=1, solver="lbfgs", max_iter=1000, random_state=42))])
    if config["family"] == "random_forest":
        return RandomForestClassifier(random_state=42, n_jobs=1, **config["parameters"])
    if config["family"] == "gradient_boosting":
        return GradientBoostingClassifier(random_state=42, **config["parameters"])
    raise ValueError("Unknown model family")


def score(model, data, features):
    metrics, predictions = evaluate(model, data, features, FEATURES)
    order = [OLD_LABELS.index(label) for label in LABELS]
    metrics["confusion_matrix"] = np.array(metrics["confusion_matrix"])[np.ix_(order, order)].tolist()
    metrics["per_class"] = {label: metrics["per_class"][label] for label in LABELS}
    probability = predictions[[f"prob_{label}" for label in LABELS]].to_numpy()
    truth = np.column_stack([data.result.to_numpy() == label for label in LABELS])
    metrics.update(brier_score=float(np.square(probability - truth).sum(axis=1).mean()),
                   draw_precision=metrics["per_class"]["D"]["precision"],
                   draw_recall=metrics["per_class"]["D"]["recall"],
                   mean_predicted_draw_probability=float(predictions.prob_D.mean()))
    columns = [c for c in predictions if not c.startswith("prob_")]
    return metrics, predictions[columns + [f"prob_{label}" for label in LABELS]]


def select_candidates(train, validation, configs):
    """This selection boundary deliberately has no test-data argument."""
    results, fitted, frames = {}, {}, []
    for config in configs:
        key = config["id"]
        if key in fitted:
            raise ValueError("Candidate IDs must be unique")
        model = estimator(config)
        model.fit(predictors(train, config["features"], FEATURES), train.result)
        metrics, predictions = score(model, validation, config["features"])
        fitted[key], results[key] = model, metrics
        frames.append(predictions.assign(split="validation", candidate=key))
    selected = min(results, key=lambda key: results[key]["log_loss"])
    winners = {}
    for config in configs:
        group = f"{config['family']}/{config['features']}"
        key = config["id"]
        if group not in winners or results[key]["log_loss"] < results[winners[group]]["log_loss"]:
            winners[group] = key
    return selected, winners, results, fitted, frames


def run_comparison(df, configs=None):
    configs = configurations() if configs is None else configs
    if BASELINE not in [c["id"] for c in configs]:
        raise ValueError("Frozen logistic baseline must be present")
    parts = split_data(df)
    if not np.isfinite(df[FEATURES["combined"]].to_numpy(dtype=float)).all():
        raise ValueError("Historical features must be finite")
    selected, winners, validation, fitted, frames = select_candidates(parts["train"], parts["validation"], configs)
    report = {"class_order": LABELS, "sklearn_version": sklearn.__version__, "random_seed": 42,
              "splits": SPLITS, "split_match_ids": {k: v.match_id.tolist() for k, v in parts.items()},
              "rows": {k: len(v) for k, v in parts.items()}, "feature_sets": FEATURES,
              "configurations": configs, "selected": selected, "group_winners": winners,
              "baseline": BASELINE, "validation": validation, "test": {},
              "protocol": "All fits use training only. Select minimum validation log loss; candidate order breaks exact ties. Then evaluate only selected and frozen logistic baseline on test, without refitting. The test season was previously inspected in earlier milestones, not a fresh blind holdout.",
              "calibration": "Diagnostic only: ten fixed equal-width one-vs-rest bins, counts included; no calibration model, threshold changes, or class weighting.",
              "brier_definition": "Mean over matches of sum over H/D/A of (probability - one_hot_target)^2; unscaled multiclass range 0 to 2.",
              "importance": "Tree impurity reduction, training only; correlated features share importance. Not causal effects or proof of incremental value."}
    lookup = {c["id"]: c for c in configs}
    for key in dict.fromkeys([BASELINE, selected]):
        metrics, predictions = score(fitted[key], parts["test"], lookup[key]["features"])
        report["test"][key] = metrics
        frames.append(predictions.assign(split="test", candidate=key))
    return report, pd.concat(frames, ignore_index=True), fitted


def calibration_bins(predictions):
    rows = []
    for (split, candidate), frame in predictions.groupby(["split", "candidate"], sort=True):
        for label in LABELS:
            probabilities = frame[f"prob_{label}"].to_numpy()
            bins = np.minimum((probabilities * 10).astype(int), 9)
            for index in range(10):
                mask = bins == index
                rows.append({"split": split, "candidate": candidate, "class": label, "bin": index,
                             "count": int(mask.sum()),
                             "mean_probability": float(probabilities[mask].mean()) if mask.any() else None,
                             "observed_frequency": float((frame.result.to_numpy()[mask] == label).mean()) if mask.any() else None})
    return pd.DataFrame(rows)


def save_reports(report, predictions, fitted, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "model_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lookup = {c["id"]: c for c in report["configurations"]}
    rows = [{"split": split, "candidate": key, "model": lookup[key]["family"],
             "features": lookup[key]["features"], "group_winner": key in report["group_winners"].values(),
             "selected": key == report["selected"], **{name: values[name] for name in METRICS}}
            for split in ("validation", "test") for key, values in report[split].items()]
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "model_comparison.csv", index=False, lineterminator="\n")
    predictions.to_csv(output / "model_predictions.csv", index=False, date_format="%Y-%m-%d", lineterminator="\n")
    displayed = predictions[(predictions.split == "test") | predictions.candidate.isin(report["group_winners"].values())]
    bins = calibration_bins(displayed)
    bins.to_csv(output / "calibration_bins.csv", index=False, lineterminator="\n")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for i, split in enumerate(("validation", "test")):
        for j, label in enumerate(LABELS):
            ax = axes[i, j]
            ax.plot([0, 1], [0, 1], "--", color="gray", label="Ideal")
            subset = bins[(bins.split == split) & (bins["class"] == label) & (bins["count"] > 0)]
            for candidate, points in subset.groupby("candidate", sort=False):
                ax.plot(points.mean_probability, points.observed_frequency, "o-", label=candidate, markersize=3)
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed frequency", title=f"{split.title()} / {label}")
            ax.legend(fontsize=6)
    fig.suptitle("Probability calibration diagnostics (10 bins; sparse bins are noisy)")
    fig.savefig(output / "calibration.png", dpi=150)
    plt.close(fig)
    for split in ("validation", "test"):
        keys = report["group_winners"].values() if split == "validation" else report["test"]
        for key in keys:
            matrix = np.array(report[split][key]["confusion_matrix"])
            fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
            ax.imshow(matrix, cmap="Blues")
            for (i, j), value in np.ndenumerate(matrix):
                ax.text(j, i, str(value), ha="center", va="center", color="white" if value > matrix.max() / 2 else "black")
            ax.set(xticks=range(3), yticks=range(3), xticklabels=LABELS, yticklabels=LABELS,
                   xlabel="Predicted", ylabel="Actual", title=f"{split}: {key}")
            fig.savefig(output / f"confusion_matrix_{split}_{key.replace('/', '_')}.png", dpi=150)
            plt.close(fig)
    importance, coefficients = [], []
    for key in report["group_winners"].values():
        model, config = fitted[key], lookup[key]
        names = FEATURES[config["features"]]
        if config["family"] == "logistic":
            classifier = model.named_steps["classifier"]
            for label in LABELS:
                index = list(classifier.classes_).index(label)
                coefficients.extend({"candidate": key, "class": label, "feature": name,
                                     "standardized_coefficient": float(value), "intercept": float(classifier.intercept_[index])}
                                    for name, value in zip(names, classifier.coef_[index]))
        else:
            importance.extend({"candidate": key, "feature": name, "importance": float(value)}
                              for name, value in zip(names, model.feature_importances_))
    pd.DataFrame(coefficients).to_csv(output / "logistic_coefficients.csv", index=False, lineterminator="\n")
    importance = pd.DataFrame(importance)
    importance.to_csv(output / "feature_importance.csv", index=False, lineterminator="\n")
    if len(importance):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
        for ax in axes.flat:
            ax.set_visible(False)
        for ax, (key, values) in zip(axes.flat, importance.groupby("candidate", sort=False)):
            ax.set_visible(True)
            top = values.nlargest(10, "importance").sort_values("importance")
            ax.barh(top.feature, top.importance, color="#2274a5")
            ax.set(title=key, xlabel="Training impurity importance (top 10)")
        fig.savefig(output / "feature_importance.png", dpi=150)
        plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/model_comparison")
    args = parser.parse_args()
    if args.report_dir.resolve() == (ROOT / "reports").resolve():
        raise ValueError("Use a separate report directory to preserve frozen baseline files")
    path = ROOT / "data/processed/model_ready.csv"
    input_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    report, predictions, fitted = run_comparison(pd.read_csv(path))
    report.update(input_sha256=input_hash, synthetic=True,
                  interpretation="Synthetic model-family demonstration; no real predictive-performance claim.")
    summary = save_reports(report, predictions, fitted, args.report_dir)
    print("Train: 2022-23, 2023-24 | Validation: 2024-25 | Test: 2025-26")
    print(summary[(summary.split == "validation") & summary.group_winner].to_string(index=False))
    print("Selected by validation log loss:", report["selected"])
    print(summary[summary.split == "test"].to_string(index=False))


if __name__ == "__main__":
    main()
