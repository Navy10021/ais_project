"""
Tests for AIS Cleaner
====================
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.preprocessing.cleaner import AISCleaner


@pytest.fixture
def sample_ais_data():
    data = {
        "MMSI": [538005955, 367127950, 367005270, 100, 999_999_999],
        "BaseDateTime": [
            "2022-03-31T05:55:47",
            "2022-03-31T03:38:31",
            "2022-03-31T14:59:21",
            "2022-03-31T10:00:00",
            "2022-03-31T12:00:00",
        ],
        "LAT": [29.7475, 39.99972, 30.05855, 91.0, 45.0],
        "LON": [-95.10412, -73.69033, -91.27603, 181.0, 30.0],
        "SOG": [3.7, 7.3, 0.0, 102.3, 10.0],
        "COG": [90.0, 295.6, 240.1, 360.0, 100.0],
        "Heading": [302.0, 511.0, 314.0, 511.0, 90.0],
        "VesselName": ["NIPPON PRINCESS", "D & S EXPRESS", "C J BOYNE", None, "TEST"],
        "IMO": ["IMO9380673", None, "IMO8975859", None, "IMO0000000"],
        "CallSign": ["V7IK5", "WDD3256", "WDC3285", None, None],
        "VesselType": [80.0, 30.0, 31.0, None, 70.0],
        "Status": [5.0, 0.0, 0.0, None, 1.0],
        "Length": [228.0, 13.0, 21.0, None, 100.0],
        "Width": [42.0, None, 9.0, None, 20.0],
        "Draft": [14.8, None, None, None, 5.0],
        "Cargo": [80.0, None, 0.0, None, 70.0],
        "TransceiverClass": ["A", "B", "A", "B", "A"],
    }
    return pd.DataFrame(data)


class TestAISCleaner:
    def test_clean_mmsi_removes_invalid(self, sample_ais_data):
        cleaner = AISCleaner("dummy_input.csv", "dummy_output.parquet")
        result = cleaner.clean_mmsi(sample_ais_data.copy())

        assert len(result) <= len(sample_ais_data)
        assert result["MMSI"].between(200_000_000, 799_999_999).all()
        assert "mmsi_special_type" in result.columns

    def test_clean_coordinates_filters_invalid(self, sample_ais_data):
        cleaner = AISCleaner("dummy_input.csv", "dummy_output.parquet")
        result = cleaner.clean_coordinates(sample_ais_data.copy())

        assert (result["LAT"] != cleaner.INVALID_LAT).all()
        assert (result["LON"] != cleaner.INVALID_LON).all()
        assert result["LAT"].between(-90.0, 90.0).all()
        assert result["LON"].between(-180.0, 180.0).all()

    def test_clean_kinematics_handles_sentinels(self, sample_ais_data):
        cleaner = AISCleaner("dummy_input.csv", "dummy_output.parquet")
        result = cleaner.clean_kinematics(sample_ais_data.copy())

        assert result["SOG"].max() < cleaner.INVALID_SOG
        assert result["COG"].max() < cleaner.INVALID_COG
        assert result["Heading"].max() < cleaner.INVALID_HEADING
        assert "sog_implausible_flag" in result.columns

    def test_impute_missing_fills_na(self, sample_ais_data):
        cleaner = AISCleaner("dummy_input.csv", "dummy_output.parquet")
        result = cleaner.impute_missing(sample_ais_data.copy())

        assert result["VesselName"].notna().all()
        assert result["IMO"].notna().all()
        assert result["CallSign"].notna().all()
        assert result["Status"].notna().all()

    def test_special_mmsi_detection(self, sample_ais_data):
        cleaner = AISCleaner("dummy_input.csv", "dummy_output.parquet")
        sample = sample_ais_data.copy()
        
        sample.loc[sample["MMSI"] == 975_000_000, "MMSI"] = 975_000_000
        sample.loc[sample["MMSI"] == 111_000_000, "MMSI"] = 111_000_000
        sample.loc[sample["MMSI"] == 990_000_000, "MMSI"] = 990_000_000
        
        result = sample.copy()
        cleaner.clean_mmsi(result)
        
        has_special = result["mmsi_special_type"].notna()
        assert has_special.any(), "Special MMSI types should be detected in data"