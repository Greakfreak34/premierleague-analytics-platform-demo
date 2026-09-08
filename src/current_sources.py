"""Timezone validation only. Public demo has no external data collector."""
import pandas as pd


def utc(value):
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("Timestamp must include an explicit timezone")
    return timestamp.tz_convert("UTC")
