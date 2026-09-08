"""Fixed-parameter feature ablation: python -m src.football_modeling."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .football_features import FOOTBALL_COLUMNS, GROUPS, POLICY
from .load_matches import ROOT
from .modeling import FEATURE_SETS, LABELS, SPLITS, evaluate, fit_models, split_data

ABLATIONS = {"simple": FEATURE_SETS["simple"], **{f"simple_{name}": FEATURE_SETS["simple"] + columns for name, columns in GROUPS.items()}, "combined": FEATURE_SETS["simple"] + FOOTBALL_COLUMNS}


def run_ablation(df):
    if not np.isfinite(df[FOOTBALL_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Football features must be finite")
    parts = split_data(df)
    models = fit_models(parts["train"], ABLATIONS)
    report = {"splits": SPLITS, "rows": {k: len(v) for k, v in parts.items()}, "feature_sets": ABLATIONS, "feature_policy": POLICY, "class_order": LABELS,
              "protocol": "Fixed Milestone 3 estimator/scaler, train only. Select by validation log loss; insertion order breaks exact ties. Test naive, simple benchmark, and selected candidate without refitting. Test is a previously inspected comparison period.", "validation": {}, "test": {}}
    frames = []
    for name, model in models.items():
        metrics, predictions = evaluate(model, parts["validation"], "simple" if name == "naive" else name, ABLATIONS)
        report["validation"][name] = metrics
        frames.append(predictions.assign(split="validation", model=name))
    selected = min(ABLATIONS, key=lambda name: report["validation"][name]["log_loss"])
    report["selected_feature_set"] = selected
    for name in dict.fromkeys(["naive", "simple", selected]):
        metrics, predictions = evaluate(models[name], parts["test"], "simple" if name == "naive" else name, ABLATIONS)
        report["test"][name] = metrics
        frames.append(predictions.assign(split="test", model=name))
    return report, pd.concat(frames, ignore_index=True), models


def save_ablation(report, predictions, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "football_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    predictions.to_csv(output / "football_predictions.csv", index=False, date_format="%Y-%m-%d", lineterminator="\n")
    rows = [{"split": split, "feature_set": name, **{metric: scores[metric] for metric in ("accuracy", "macro_f1", "log_loss")}, "draw_recall": scores["per_class"]["D"]["recall"]} for split in ("validation", "test") for name, scores in report[split].items()]
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "football_ablation.csv", index=False, lineterminator="\n")
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    validation = summary[summary.split == "validation"]
    ax.barh(validation.feature_set, validation.log_loss, color="#2274a5")
    for i, value in enumerate(validation.log_loss):
        ax.text(value + 0.005, i, f"{value:.4f}", va="center")
    ax.set(xlabel="Validation log loss (lower is better)", title="Football feature ablation — 2024–25", xlim=(0, validation.log_loss.max() + 0.13))
    ax.invert_yaxis()
    fig.savefig(output / "football_ablation.png", dpi=150)
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/model_ready.csv")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    report, predictions, _ = run_ablation(pd.read_csv(args.data))
    report["input_sha256"] = hashlib.sha256(args.data.read_bytes()).hexdigest()
    print(save_ablation(report, predictions, args.report_dir).to_string(index=False))
    print("Selected:", report["selected_feature_set"])


if __name__ == "__main__":
    main()
