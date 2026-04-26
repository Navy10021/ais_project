"""Smoke checks for EDA script helpers."""
from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.skipif(
    not Path("data/raw/ais_raw.csv").exists(),
    reason="AIS raw data file is not available in this environment.",
)
def test_ais_raw_has_minimum_rows() -> None:
    df = pd.read_csv("data/raw/ais_raw.csv")
    sample = df.sample(n=min(len(df), 50_000), random_state=42)
    assert not sample.empty
