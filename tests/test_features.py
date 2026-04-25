"""
Tests for AIS Feature Engineer
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.preprocessing.feature_engineer import AISFeatureEngineer


@pytest.fixture
def sample_clean_data():
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2022-03-31", periods=n, freq="h")
    data = {
        "MMSI": np.random.randint(200_000_000, 700_000_000, n),
        "BaseDateTime": dates,
        "LAT": np.random.uniform(25, 50, n),
        "LON": np.random.uniform(-80, 30, n),
        "SOG": np.random.uniform(0, 15, n),
        "COG": np.random.uniform(0, 360, n),
        "Heading": np.random.uniform(0, 360, n),
        "VesselType": np.random.choice([70, 80, 30], n),
        "Status": np.random.choice([0, 1, 5], n),
    }
    return pd.DataFrame(data)


class TestAISFeatureEngineer:
    def test_kinematic_features(self, sample_clean_data):
        engineer = AISFeatureEngineer()
        result = engineer.add_kinematic_features(sample_clean_data.copy())

        assert "speed_category" in result.columns
        assert "delta_sog" in result.columns
        assert "delta_cog" in result.columns
        assert "time_diff_sec" in result.columns
        assert "is_dark_ship" in result.columns
        assert result["speed_category"].notna().any()

    def test_geospatial_features(self, sample_clean_data):
        engineer = AISFeatureEngineer()
        result = engineer.add_geospatial_features(sample_clean_data.copy())

        assert "grid_cell" in result.columns
        assert "in_conflict_zone" in result.columns
        assert "conflict_zone_name" in result.columns
        assert "dist_hormuz_km" in result.columns

    def test_conflict_zones(self, sample_clean_data):
        engineer = AISFeatureEngineer()
        result = engineer.add_geospatial_features(sample_clean_data.copy())

        assert result["conflict_zone_name"].isin(engineer.CONFLICT_ZONES.keys()).any() or (
            result["conflict_zone_name"] == "none"
        ).all()

    def test_behavioral_features(self, sample_clean_data):
        engineer = AISFeatureEngineer()
        kin = engineer.add_kinematic_features(sample_clean_data.copy())
        geo = engineer.add_geospatial_features(kin)
        result = engineer.add_behavioral_features(geo)

        assert "rolling_sog_mean_12h" in result.columns
        assert "rolling_sog_std_12h" in result.columns
        assert "route_entropy" in result.columns
        assert "loitering_flag" in result.columns

    def test_temporal_aggregation(self, sample_clean_data):
        engineer = AISFeatureEngineer()
        sample_clean_data = sample_clean_data.copy()
        sample_clean_data = engineer.add_kinematic_features(sample_clean_data)
        sample_clean_data = engineer.add_geospatial_features(sample_clean_data)
        sample_clean_data = engineer.add_behavioral_features(sample_clean_data)
        df, agg = engineer.add_temporal_aggregation(sample_clean_data)

        assert "traffic_count" in agg.columns
        assert "dark_ship_ratio" in agg.columns
        assert "military_ratio" in agg.columns
        assert "tanker_ratio" in agg.columns

    def test_conflict_labels_defaults(self, sample_clean_data):
        engineer = AISFeatureEngineer()
        result = engineer.add_conflict_labels(sample_clean_data.copy(), None)

        assert "conflict_label" in result.columns
        assert "days_to_conflict" in result.columns
        assert "conflict_intensity" in result.columns
        assert (result["conflict_label"] == 0).all()

    def test_haversine(self, sample_clean_data):
        engineer = AISFeatureEngineer()
        dist = engineer._haversine(0, 0, 0, 1)
        assert dist.min() > 0

    def test_full_pipeline(self, sample_clean_data):
        engineer = AISFeatureEngineer()
        df = sample_clean_data.copy()
        
        df = engineer.add_kinematic_features(df)
        df = engineer.add_geospatial_features(df)
        df = engineer.add_behavioral_features(df)
        df, agg = engineer.add_temporal_aggregation(df)
        df = engineer.add_conflict_labels(df, None)

        assert len(df) == len(sample_clean_data)
        assert "grid_cell" in df.columns
        assert "in_conflict_zone" in df.columns
        assert "traffic_count" in agg.columns