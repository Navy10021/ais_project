"""
Network Analyzer
================
Analyze vessel route networks, port connections, and graph patterns.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
import yaml
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Set

logger = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)


@dataclass
class NetworkConfig:
    """Configuration for network analysis"""
    min_port_visits: int = 3
    max_distance_km: float = 50.0


def load_config(config_path: str = "./config/settings.yaml") -> NetworkConfig:
    """Load network analysis config"""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return NetworkConfig(
            min_port_visits=config.get("analysis", {}).get("min_port_visits", 3),
            max_distance_km=config.get("analysis", {}).get("max_distance_km", 50.0),
        )
    except Exception:
        return NetworkConfig()


class NetworkAnalyzer:
    PORT_LOCATIONS = {
        "rotterdam": (51.9, 4.5),
        "sanghai": (31.2, 121.5),
        "singapore": (1.3, 103.8),
        "los_angeles": (33.7, -118.2),
        "new_york": (40.7, -74.0),
        "dubai": (25.0, 55.3),
        "hamburg": (53.5, 10.0),
        "antwerp": (51.2, 4.4),
        "hong_kong": (22.3, 114.2),
        "busan": (35.1, 129.0),
    }

    def __init__(
        self,
        output_dir: str = "./outputs/tables",
        config: Optional[NetworkConfig] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or NetworkConfig()

    @staticmethod
    def _haversine(
        lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate distance in km between two points"""
        R = 6371.0
        lat1_rad, lat2_rad = np.radians(lat1), np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        )
        return R * 2 * np.arcsin(np.sqrt(a))

    def identify_port_visits(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Identify likely port visits from low speed near ports"""
        if "SOG" not in df.columns or "LAT" not in df.columns:
            return pd.DataFrame()

        slow_near_port = df[
            (df["SOG"] < 1.0)
            & (df["LAT"].between(0, 70))
        ]

        if len(slow_near_port) == 0:
            return pd.DataFrame()

        port_visits = (
            slow_near_port.groupby("MMSI")
            .agg(
                visit_count=("MMSI", "count"),
                mean_lat=("LAT", "mean"),
                mean_lon=("LON", "mean"),
                first_seen=("BaseDateTime", "min"),
                last_seen=("BaseDateTime", "max"),
            )
            .reset_index()
        )

        port_visits = port_visits[
            port_visits["visit_count"] >= self.config.min_port_visits
        ]

        return port_visits

    def build_vessel_routes(
        self,
        df: pd.DataFrame,
    ) -> List[Dict]:
        """Build route sequences for vessels"""
        if "MMSI" not in df.columns or "BaseDateTime" not in df.columns:
            return []

        routes = []
        for mmsi, vessel_df in df.groupby("MMSI"):
            vessel_df = vessel_df.sort_values("BaseDateTime")

            if len(vessel_df) < self.config.min_port_visits:
                continue

            waypoints = vessel_df[["LAT", "LON", "BaseDateTime"]].to_dict("records")
            routes.append({
                "MMSI": mmsi,
                "n_waypoints": len(waypoints),
                "waypoints": waypoints,
            })

        return routes

    def compute_zone_connectivity(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute connectivity between zones"""
        if "conflict_zone_name" not in df.columns:
            return pd.DataFrame()

        zones = df["conflict_zone_name"].unique()
        zones = [z for z in zones if z != "none"]

        if len(zones) < 2:
            return pd.DataFrame()

        connectivity = []
        for mmsi in df["MMSI"].unique():
            vessel_zones = df[df["MMSI"] == mmsi]["conflict_zone_name"].unique()
            vessel_zones = [z for z in vessel_zones if z != "none"]

            for i, z1 in enumerate(vessel_zones):
                for z2 in vessel_zones[i + 1 :]:
                    pair = tuple(sorted([z1, z2]))
                    connectivity.append({"zone_pair": pair})

        if not connectivity:
            return pd.DataFrame()

        conn_df = pd.DataFrame(connectivity)
        conn_counts = (
            conn_df["zone_pair"]
            .value_counts()
            .reset_index()
        )
        conn_counts.columns = ["zone_pair", "n_vessels"]
        conn_counts["z1"] = conn_counts["zone_pair"].apply(lambda x: x[0])
        conn_counts["z2"] = conn_counts["zone_pair"].apply(lambda x: x[1])
        conn_counts = conn_counts.drop("zone_pair", axis=1)

        return conn_counts

    def analyze_zone_transitions(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Analyze vessel transitions between zones"""
        if "conflict_zone_name" not in df.columns:
            return pd.DataFrame()

        transitions = []
        for mmsi, vessel_df in df.groupby("MMSI"):
            vessel_df = vessel_df.sort_values("BaseDateTime")
            zones = vessel_df["conflict_zone_name"].unique()

            for i in range(len(zones) - 1):
                transitions.append({
                    "MMSI": mmsi,
                    "from_zone": zones[i],
                    "to_zone": zones[i + 1],
                })

        if not transitions:
            return pd.DataFrame()

        trans_df = pd.DataFrame(transitions)
        trans_counts = (
            trans_df.groupby(["from_zone", "to_zone"])
            .size()
            .reset_index(name="count")
        )

        return trans_counts

    def compute_vessel_complexity(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute route complexity metrics per vessel"""
        if "MMSI" not in df.columns:
            return pd.DataFrame()

        rows = []
        for mmsi, group in df.groupby("MMSI"):
            n_recs = len(group)
            lats = group["LAT"].round(1).tolist()
            lons = group["LON"].round(1).tolist()
            n_unique = len(set(zip(lats, lons)))
            time_span = (group["BaseDateTime"].max() - group["BaseDateTime"].min()).total_seconds() / 86400
            rows.append({
                "MMSI": mmsi,
                "n_records": n_recs,
                "n_unique_locs": n_unique,
                "time_span_days": time_span,
            })

        complexity = pd.DataFrame(rows)
        complexity["records_per_day"] = complexity["n_records"] / complexity["time_span_days"].clip(lower=1)
        return complexity

    def identify_hub_vessels(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Identify vessels that visit multiple zones (potential hubs)"""
        if "conflict_zone_name" not in df.columns or "MMSI" not in df.columns:
            return pd.DataFrame()

        zone_visits = df.groupby("MMSI")["conflict_zone_name"].nunique()
        hub_vessels = zone_visits[zone_visits >= 2].reset_index()
        hub_vessels.columns = ["MMSI", "n_zones"]

        return hub_vessels.sort_values("n_zones", ascending=False)

    def analyze_temporal_patterns(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """Analyze temporal patterns in network"""
        if "BaseDateTime" not in df.columns:
            return {}

        df = df.copy()
        df["hour"] = df["BaseDateTime"].dt.hour
        df["dayofweek"] = df["BaseDateTime"].dt.dayofweek

        hourly_patterns = (
            df.groupby("hour")["MMSI"]
            .nunique()
            .reset_index()
        )
        hourly_patterns.columns = ["hour", "vessel_count"]

        daily_patterns = (
            df.groupby("dayofweek")["MMSI"]
            .nunique()
            .reset_index()
        )
        daily_patterns.columns = ["dayofweek", "vessel_count"]

        return {
            "hourly_patterns": hourly_patterns,
            "daily_patterns": daily_patterns,
        }

    def run_all(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Run all network analyses"""
        logger.info("Running network analysis...")

        results = {}

        port_visits = self.identify_port_visits(df)
        if not port_visits.empty:
            results["port_visits"] = port_visits
            port_visits.to_csv(
                self.output_dir / "port_visits.csv",
                index=False,
            )
            logger.info(f"Saved: {self.output_dir / 'port_visits.csv'}")

        connectivity = self.compute_zone_connectivity(df)
        if not connectivity.empty:
            results["zone_connectivity"] = connectivity
            connectivity.to_csv(
                self.output_dir / "zone_connectivity.csv",
                index=False,
            )
            logger.info(f"Saved: {self.output_dir / 'zone_connectivity.csv'}")

        transitions = self.analyze_zone_transitions(df)
        if not transitions.empty:
            results["zone_transitions"] = transitions
            transitions.to_csv(
                self.output_dir / "zone_transitions.csv",
                index=False,
            )
            logger.info(f"Saved: {self.output_dir / 'zone_transitions.csv'}")

        complexity = self.compute_vessel_complexity(df)
        if not complexity.empty:
            results["vessel_complexity"] = complexity

        hub_vessels = self.identify_hub_vessels(df)
        if not hub_vessels.empty:
            results["hub_vessels"] = hub_vessels

        temporal = self.analyze_temporal_patterns(df)
        for key, val in temporal.items():
            if not val.empty:
                results[key] = val

        logger.info("Network analysis complete!")
        return results


def main():
    parser = argparse.ArgumentParser(description="Network Analyzer")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="./outputs/tables")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df):,} records")

    config = load_config()
    analyzer = NetworkAnalyzer(args.output_dir, config)
    analyzer.run_all(df)


if __name__ == "__main__":
    main()