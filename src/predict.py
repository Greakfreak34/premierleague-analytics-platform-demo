"""Synthetic demo: python -m src.predict Amber Birch [--kickoff ISO_TIMESTAMP]."""
import argparse
import json

from .current_sources import utc

from .prediction.service import DataUnavailableError, PredictionInputError, PredictionService


def format_prediction(result):
    home, away = result["home_team"], result["away_team"]
    ranked = sorted([(home, result["home_win"]), ("Draw", result["draw"]),
                     (away, result["away_win"])], key=lambda item: -item[1])
    picks = [name for name, probability in ranked if probability == ranked[0][1]]
    inputs = result["features"]
    diff = inputs["elo_diff"]
    advantage = f"{home if diff > 0 else away} +{abs(diff):.0f}" if diff else "Level"
    lines = ["SYNTHETIC DEMO — not a real football prediction", f"{home} vs {away}", f"Kickoff: {utc(result['kickoff']).strftime('%Y-%m-%d %H:%M UTC')}",
             f"Forecast status: {result['forecast']['status']}",
             f"Data current through: {utc(result['data_as_of']).strftime('%Y-%m-%d %H:%M UTC')}",
             "", "Prediction"]
    for name, probability in ranked:
        label = "Draw" if name == "Draw" else f"{name} win"
        lines.append(f"{label}: {probability:.1%}")
    lines += ["", "Model pick: " + (picks[0] if len(picks) == 1 else "Tied: " + " / ".join(picks)),
              "", "Key model inputs", f"{home} Elo: {inputs['home_elo']:.0f}",
              f"{away} Elo: {inputs['away_elo']:.0f}", f"Elo difference: {advantage}",
              "", "Recent form (up to 5 completed league matches this season)",
              f"{home}: {inputs['home_form_pts_5']:.0f} pts; goal difference {inputs['home_gd_5']:+.0f}",
              f"{away}: {inputs['away_form_pts_5']:.0f} pts; goal difference {inputs['away_gd_5']:+.0f}"]
    if result["forecast"]["reasons"]:
        lines += ["", *result["forecast"]["reasons"], "Future intervening results are not simulated. Refresh nearer kickoff."]
    lines += ["", "Availability, sentiment, and cross-competition context do not affect these probabilities."]
    return "\n".join(lines)


def main(argv=None, service=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("home_team")
    parser.add_argument("away_team")
    parser.add_argument("--kickoff")
    parser.add_argument("--json", action="store_true", help="Print the complete structured API response")
    args = parser.parse_args(argv)
    try:
        result = (service or PredictionService()).predict_match(args.home_team, args.away_team, args.kickoff)
    except (PredictionInputError, DataUnavailableError) as exc:
        parser.exit(2, f"Prediction unavailable: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False) if args.json else format_prediction(result))
    return result


if __name__ == "__main__":
    main()
