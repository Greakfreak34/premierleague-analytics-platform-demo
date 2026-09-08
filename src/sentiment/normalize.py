"""Conservative club matching and auditable Bluesky observations."""
import hashlib
import re
from collections import Counter

import pandas as pd

from ..current_sources import utc

ACRONYMS = {"LFC": "Liverpool", "MUFC": "Man United", "MCFC": "Man City",
            "THFC": "Tottenham", "NUFC": "Newcastle", "CPFC": "Crystal Palace"}
CONTEXT = re.compile(r"\b(football|soccer|premier league|epl|goal|goals|match|kickoff|"
                     r"striker|midfielder|defender|keeper|manager|squad|transfer|penalty|"
                     r"offside|fixture|fixtures|stadium|injury|injured|season|points|"
                     r"clean sheet|relegation|champions league)\b", re.I)
FIELDS = ["post_id", "revision", "published_at", "collected_at", "team", "text",
          "text_sha256", "source", "source_url", "raw_file", "raw_index"]


def club_for_text(text, config):
    aliases = {**{t: t for t in config["clubs"]}, **config["aliases"], **ACRONYMS}
    hits = {team for alias, team in aliases.items()
            if alias.lower() not in {"city", "united", "palace", "reds"}
            and re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text, re.I)}
    # One whole-post sentiment cannot safely attribute opposing opinions to two teams.
    if len(hits) != 1 or not CONTEXT.search(text):
        return None
    return hits.pop()


def normalize(pages, collected_at, config):
    collected_at = utc(collected_at)
    rows, excluded, seen = [], Counter(), {}
    for name, payload in sorted(pages.items()):
        for index, post in enumerate(payload["posts"]):
            record = post["record"]
            text = record.get("text", "")
            published = utc(record["createdAt"])
            if published > collected_at:
                excluded["future_publication"] += 1
                continue
            if "en" not in record.get("langs", []):
                excluded["english_not_declared"] += 1
                continue
            team = club_for_text(text, config)
            if team is None:
                excluded["unassigned_or_multiple_clubs_or_no_football_context"] += 1
                continue
            key = post["uri"]
            fingerprint = (post["cid"], text, published.isoformat())
            if key in seen:
                if seen[key] != fingerprint:
                    raise ValueError("Conflicting revisions within one collection")
                excluded["duplicate"] += 1
                continue
            seen[key] = fingerprint
            rows.append(dict(zip(FIELDS, [key, post["cid"], published.isoformat(),
                collected_at.isoformat(), team, text, hashlib.sha256(text.encode()).hexdigest(),
                "Bluesky", "https://bsky.app/profile/" + key.split("/")[2] + "/post/" + key.split("/")[-1],
                name, index])))
    return pd.DataFrame(rows, columns=FIELDS).sort_values(["post_id", "revision"]).reset_index(drop=True), dict(excluded)
