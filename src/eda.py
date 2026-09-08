"""Produce three descriptive plots without fitting a model."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def make_plots(df: pd.DataFrame, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    counts = df.result.value_counts().reindex(["H", "D", "A"], fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot.bar(ax=ax, color=["#2274a5", "#999999", "#e09936"], rot=0)
    ax.set(title="Match outcomes", xlabel="Home win / Draw / Away win", ylabel="Matches")
    for i, value in enumerate(counts):
        ax.text(i, value, str(value), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output / "outcomes.png", dpi=150)
    plt.close(fig)
    goals = df.home_goals + df.away_goals
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(goals, bins=[x - 0.5 for x in range(int(goals.max()) + 2)], color="#2274a5", rwidth=0.85)
    ax.set(title="Total goals per match", xlabel="Goals", ylabel="Matches")
    fig.tight_layout()
    fig.savefig(output / "goals.png", dpi=150)
    plt.close(fig)
    full = df[(df.home_history_count_5 == 5) & (df.away_history_count_5 == 5)].copy()
    diff = full.home_form_pts_5 - full.away_form_pts_5
    full["form"] = diff.map(lambda x: "Home better" if x > 0 else "Away better" if x < 0 else "Equal")
    rates = pd.crosstab(full.form, full.result).reindex(index=["Home better", "Equal", "Away better"], columns=["H", "D", "A"], fill_value=0)
    sizes = rates.sum(axis=1)
    fractions = rates.div(sizes.replace(0, float("nan")), axis=0).fillna(0)
    fig, ax = plt.subplots(figsize=(8, 4))
    fractions.plot.bar(stacked=True, ax=ax, rot=0, color=["#2274a5", "#999999", "#e09936"])
    ax.set(title="Outcomes by previous-five points (both teams have five games)", xlabel="Pre-match form", ylabel="Share of matches", ylim=(0, 1))
    ax.set_xticklabels([f"{label}\n(n={sizes[label]})" for label in rates.index])
    ax.legend(title="Result", loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(output / "form_outcomes.png", dpi=150)
    plt.close(fig)
    return {"outcomes": counts.to_dict(), "mean_goals": float(goals.mean()), "full_history_matches": len(full), "form_groups": rates.to_dict(orient="index")}
