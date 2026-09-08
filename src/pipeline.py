"""Run with python -m src.pipeline from the repository root."""
import argparse
import hashlib
import json
from pathlib import Path

from .clean_matches import clean_matches
from .eda import make_plots
from .features import FEATURE_COLUMNS, add_form_features
from .football_features import FOOTBALL_COLUMNS, POLICY, add_football_features
from .load_matches import ROOT, load_matches
from .seasons import DEFAULT_MANIFEST, load_seasons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--raw", type=Path)
    inputs.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--season", help="Required with --raw, e.g. 2023-24")
    parser.add_argument("--competition", default="ENG-PL")
    parser.add_argument("--date-format", default="%a %b %d %Y")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    if args.raw:
        if not args.season:
            parser.error("--season is required with --raw")
        clean, report = clean_matches(load_matches(args.raw), args.date_format, season=args.season, competition=args.competition)
        report["raw_sha256"] = hashlib.sha256(args.raw.read_bytes()).hexdigest()
    else:
        clean, report = load_seasons(args.manifest)
    featured = add_football_features(add_form_features(clean))
    report["football_feature_policy"] = POLICY
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.output_dir / "matches.csv", index=False, date_format="%Y-%m-%d", lineterminator="\n")
    featured.to_csv(args.output_dir / "match_features.csv", index=False, date_format="%Y-%m-%d", lineterminator="\n")
    model = featured[["match_id", "competition", "season", "date", "home_team", "away_team"] + FEATURE_COLUMNS + FOOTBALL_COLUMNS + ["result"]]
    model.to_csv(args.output_dir / "model_ready.csv", index=False, date_format="%Y-%m-%d", lineterminator="\n")
    report["eda"] = make_plots(featured, args.report_dir)
    if "seasons" in report:
        for season_report in report["seasons"]:
            subset = featured[(featured.season == season_report["season"]) & (featured.competition == season_report["competition"])]
            season_report["eda"] = make_plots(subset, args.report_dir / season_report["competition"] / season_report["season"])
    (args.report_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
