# CLAUDE.md — Maritime Conflict Intelligence System (MCIS)

## PROJECT OVERVIEW

**Mission**: Analyze AIS (Automatic Identification System) data to uncover statistically
significant correlations between maritime behavioral patterns and armed conflicts, and build
a predictive system capable of detecting conflict precursors before they materialize.

**Core Research Question**:
> "Do maritime vessel behavioral patterns — traffic density, speed, vessel type composition,
> and route entropy — exhibit statistically significant changes before and after the onset of
> armed conflicts? Can these signals be used to predict conflict events in advance?"

**Target Conflicts**:
| Conflict | Period | Key Maritime Zone |
|----------|--------|-------------------|
| Russia–Ukraine War | 2022-02-24 → | Black Sea, Sea of Azov |
| Red Sea / Houthi Crisis | 2023-11 → | Red Sea, Gulf of Aden, Bab-el-Mandeb |
| Taiwan Strait Tensions | 2022-08 (PLA drills) | Taiwan Strait |
| South China Sea Disputes | Ongoing | SCS, Paracel & Spratly Islands |
| Iran–Gulf Tensions | Ongoing | Strait of Hormuz |

---

## REPOSITORY STRUCTURE

```
ais-conflict-intelligence/
├── CLAUDE.md                         # This file — architecture specification
├── README.md
├── requirements.txt
├── config/
│   └── settings.yaml                 # Global config: paths, parameters, conflict zones
├── data/
│   ├── raw/
│   │   └── ais_raw.csv               # Source AIS data (read-only — never modify)
│   ├── processed/
│   │   ├── ais_clean.parquet         # Output of Phase 1: Cleaner
│   │   ├── ais_features.parquet      # Output of Phase 1: Feature Engineer
│   │   └── conflict_events.csv       # Conflict event labels
│   ├── external/
│   │   ├── acled_events.csv          # ACLED armed conflict database
│   │   ├── gdelt_events.csv          # GDELT news event data
│   │   └── world_ports.csv           # World port coordinates
│   └── geojson/
│       ├── conflict_zones.geojson    # Conflict zone polygons
│       └── chokepoints.geojson       # Strategic strait polygons
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── cleaner.py                # Step 1: Raw data cleaning
│   │   ├── validator.py              # Step 2: Schema & physical validation
│   │   ├── feature_engineer.py       # Step 3: Feature generation
│   │   └── anomaly_detector.py       # Step 4: Preprocessing-stage outlier removal
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── spatial_viz.py            # Geospatial visualization (folium, plotly)
│   │   ├── temporal_viz.py           # Time-series visualization
│   │   ├── statistical_viz.py        # Distribution & correlation plots
│   │   └── conflict_overlay.py       # Conflict event overlay on maritime maps
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── traffic_analyzer.py       # Maritime traffic volume analysis
│   │   ├── behavioral_analyzer.py    # Vessel behavior pattern analysis
│   │   ├── network_analyzer.py       # Route network graph analysis
│   │   └── correlation_analyzer.py   # Conflict correlation (Granger, DiD, ITS)
│   └── models/
│       ├── __init__.py
│       ├── baseline.py               # Statistical baselines (ARIMA, Prophet)
│       ├── anomaly_model.py          # Anomaly detection (Isolation Forest, VAE)
│       ├── conflict_predictor.py     # Conflict prediction (LSTM, Transformer)
│       └── evaluator.py              # Model evaluation & reporting
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_preprocessing_validation.ipynb
│   ├── 03_visualization.ipynb
│   ├── 04_conflict_correlation.ipynb
│   └── 05_model_development.ipynb
├── outputs/
│   ├── figures/                      # High-resolution figures for publication
│   ├── tables/                       # Statistical result tables
│   ├── models/                       # Saved model weights & configs
│   └── reports/                      # Final analysis report
├── tests/
│   ├── test_cleaner.py
│   ├── test_features.py
│   └── test_models.py
└── scripts/
    ├── run_pipeline.sh               # Full pipeline execution
    └── generate_report.py            # Publication-ready output generator
```

---

## DATA SCHEMA

### Raw AIS Column Specification (`ais_raw.csv`)

| Column | Type | Description | Valid Range | Notes |
|--------|------|-------------|-------------|-------|
| `MMSI` | int64 | Maritime Mobile Service Identity (9-digit) | 100000000–999999999 | Unique vessel ID |
| `BaseDateTime` | datetime | AIS signal reception timestamp (UTC) | — | ISO 8601 format |
| `LAT` | float32 | Latitude | −90.0 to 90.0 | 91.0 = not available |
| `LON` | float32 | Longitude | −180.0 to 180.0 | 181.0 = not available |
| `SOG` | float32 | Speed Over Ground (knots) | 0.0 to 102.2 | 102.3 = not available |
| `COG` | float32 | Course Over Ground (degrees) | 0.0 to 359.9 | 360.0 = not available |
| `Heading` | float32 | True heading (degrees) | 0–359 | 511 = not available |
| `VesselName` | str | Vessel name | — | Nullable |
| `IMO` | str | IMO number (prefix "IMO" + 7 digits) | — | Nullable |
| `CallSign` | str | Radio call sign | — | Nullable |
| `VesselType` | float32 | Vessel type code (ITU/IMO) | 0–99 | See mapping below |
| `Status` | float32 | Navigational status code | 0–15 | 0=underway, 1=anchor, 5=moored |
| `Length` | float32 | Overall length (meters) | 0–500 | — |
| `Width` | float32 | Beam width (meters) | 0–100 | — |
| `Draft` | float32 | Draft depth (meters) | 0–30 | — |
| `Cargo` | float32 | Cargo type code | — | Linked to VesselType |
| `TransceiverClass` | category | AIS transceiver class | A, B | A=SOLAS mandatory, B=small craft |

### VesselType & Navigation Status Mappings
```python
VESSEL_TYPE_MAP = {
    0:  "Not Available",
    30: "Fishing",
    31: "Towing",
    32: "Towing — Large",
    33: "Dredging",
    34: "Diving Operations",
    35: "Military Operations",       # KEY: conflict indicator
    36: "Sailing",
    37: "Pleasure Craft",
    50: "Pilot Vessel",
    51: "Search and Rescue Vessel",  # SAR surge = conflict indicator
    52: "Tug",
    55: "Law Enforcement",           # Naval patrol indicator
    60: "Passenger",
    70: "Cargo",                     # Primary commercial analysis target
    79: "Cargo — Other",
    80: "Tanker",                    # Energy security indicator
    89: "Tanker — Other",
    90: "Other",
}

NAV_STATUS_MAP = {
    0:  "Under Way Using Engine",
    1:  "At Anchor",
    2:  "Not Under Command",         # Distress / loss of control
    3:  "Restricted Maneuverability",
    4:  "Constrained by Draft",
    5:  "Moored",
    6:  "Aground",
    7:  "Engaged in Fishing",
    8:  "Under Way Sailing",
    15: "Not Defined",
}
```

---

## PHASE 1 — ADVANCED PREPROCESSING (`src/preprocessing/`)

### Pipeline Order
```
ais_raw.csv
  → [1. Cleaner]          MMSI / coordinate / kinematic / timestamp / duplicate
  → [2. Validator]        Schema enforcement, physical bounds assertion
  → [3. Feature Engineer] Kinematic, geospatial, behavioral, aggregation, labels
  → [4. Anomaly Detector] Preprocessing-stage outlier flagging (IQR + Z-score)
  → ais_features.parquet
```

---

### 1-1. `cleaner.py`

```python
"""
AIS Raw Data Cleaner
====================
Physical validity filtering + per-column missing value imputation.

Processing order:
  1. MMSI validity       — 9-digit civilian range; flag special MMSI types
  2. Coordinate validity — range check + AIS sentinel values (91 / 181)
  3. Kinematic sentinels — SOG=102.3, COG=360, Heading=511 → NaN
  4. Timestamps          — UTC parse, future / pre-2010 record removal
  5. Duplicate removal   — MMSI + BaseDateTime
  6. Missing imputation  — per-column strategy (see impute_missing)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AISCleaner:
    # AIS standard sentinel (invalid) values
    INVALID_LAT     = 91.0
    INVALID_LON     = 181.0
    INVALID_SOG     = 102.3
    INVALID_COG     = 360.0
    INVALID_HEADING = 511
    INVALID_IMO     = "IMO0000000"

    # Special-purpose MMSI ranges (flagged but not removed)
    SPECIAL_MMSI = {
        "coastal_station": (0,         99_999_999),
        "group_ship":      (970_000_000, 979_999_999),
        "sar_aircraft":    (111_000_000, 111_999_999),
        "mob_device":      (972_000_000, 972_999_999),
        "aton":            (990_000_000, 999_999_999),
    }

    DTYPE_MAP = {
        "MMSI": "int64", "LAT": "float32", "LON": "float32",
        "SOG": "float32", "COG": "float32", "Heading": "float32",
        "VesselType": "float32", "Status": "float32",
        "Length": "float32", "Width": "float32",
        "Draft": "float32", "Cargo": "float32",
        "TransceiverClass": "category",
    }

    def __init__(self, input_path: str, output_path: str):
        self.input_path  = Path(input_path)
        self.output_path = Path(output_path)
        self.report: dict = {}

    def load(self) -> pd.DataFrame:
        """Parquet preferred; CSV fallback with explicit dtypes."""
        if self.input_path.suffix == ".parquet":
            return pd.read_parquet(self.input_path)
        return pd.read_csv(
            self.input_path, dtype=self.DTYPE_MAP,
            parse_dates=["BaseDateTime"], low_memory=False,
        )

    def clean_mmsi(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        for name, (lo, hi) in self.SPECIAL_MMSI.items():
            df.loc[df["MMSI"].between(lo, hi), "mmsi_special_type"] = name
        df = df[df["MMSI"].between(200_000_000, 799_999_999)].copy()
        self.report["mmsi_removed"] = n - len(df)
        return df

    def clean_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        df = df[
            df["LAT"].between(-90.0, 90.0) &
            df["LON"].between(-180.0, 180.0) &
            (df["LAT"] != self.INVALID_LAT) &
            (df["LON"] != self.INVALID_LON) &
            df["LAT"].notna() & df["LON"].notna()
        ].copy()
        self.report["coord_removed"] = n - len(df)
        return df

    def clean_kinematics(self, df: pd.DataFrame) -> pd.DataFrame:
        df["SOG"]     = df["SOG"].where(df["SOG"]     < self.INVALID_SOG,     np.nan)
        df["COG"]     = df["COG"].where(df["COG"]     < self.INVALID_COG,     np.nan)
        df["Heading"] = df["Heading"].where(df["Heading"] < self.INVALID_HEADING, np.nan)
        df["sog_implausible_flag"] = (df["SOG"] > 50.0).astype("int8")
        return df

    def clean_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"], utc=True, errors="coerce")
        now    = pd.Timestamp.now(tz="UTC")
        cutoff = pd.Timestamp("2010-01-01", tz="UTC")
        df = df[df["BaseDateTime"].between(cutoff, now)].copy()
        return df.sort_values(["MMSI", "BaseDateTime"]).reset_index(drop=True)

    def clean_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        df = df.drop_duplicates(subset=["MMSI", "BaseDateTime"], keep="first")
        self.report["duplicates_removed"] = n - len(df)
        return df

    def impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Strategy per column:
          VesselName / IMO / CallSign  → "UNKNOWN"
          VesselType                   → per-MMSI mode, else 0
          Length / Width / Draft       → per-MMSI median → per-VesselType median
          Status                       → 0 (under way using engine)
        """
        for col in ["VesselName", "IMO", "CallSign"]:
            df[col] = df[col].fillna("UNKNOWN").replace("", "UNKNOWN")
        df.loc[df["IMO"] == self.INVALID_IMO, "IMO"] = "UNKNOWN"

        type_mode = df.groupby("MMSI")["VesselType"].transform(
            lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else 0)
        )
        df["VesselType"] = df["VesselType"].fillna(type_mode)

        for col in ["Length", "Width", "Draft"]:
            df[col] = df.groupby("MMSI")[col].transform(lambda x: x.fillna(x.median()))
            df[col] = df[col].fillna(df.groupby("VesselType")[col].transform("median"))

        df["Status"] = df["Status"].fillna(0).astype("int8")
        return df

    def run(self) -> pd.DataFrame:
        logger.info("Loading raw AIS data...")
        df = self.load()
        logger.info(f"Raw records: {len(df):,}")
        df = self.clean_mmsi(df)
        df = self.clean_coordinates(df)
        df = self.clean_kinematics(df)
        df = self.clean_timestamps(df)
        df = self.clean_duplicates(df)
        df = self.impute_missing(df)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.output_path, index=False, compression="snappy")
        logger.info(f"Clean records: {len(df):,} → {self.output_path}")
        logger.info(f"Report: {self.report}")
        return df
```

---

### 1-2. `feature_engineer.py`

```python
"""
AIS Feature Engineer
====================
Five feature categories for conflict detection and prediction.

  A. Kinematic    — motion & maneuver characteristics
  B. Geospatial   — spatial context & chokepoint proximity
  C. Behavioral   — rolling-window irregularity metrics
  D. Aggregation  — grid-cell × 6-hour traffic statistics
  E. Labels       — binary conflict label + regression target
"""
import pandas as pd
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class AISFeatureEngineer:

    # Bounding boxes [lon_min, lat_min, lon_max, lat_max]
    CONFLICT_ZONES = {
        "black_sea":       {"bbox": [27.0, 40.5, 41.0, 46.8], "conflict": "ukraine_war"},
        "azov_sea":        {"bbox": [33.5, 45.0, 39.5, 47.5], "conflict": "ukraine_war"},
        "kerch_strait":    {"bbox": [36.4, 45.1, 36.8, 45.5], "conflict": "ukraine_war"},
        "red_sea":         {"bbox": [32.0, 12.0, 43.5, 30.0], "conflict": "houthi_crisis"},
        "bab_el_mandeb":   {"bbox": [43.0, 11.5, 45.0, 12.5], "conflict": "houthi_crisis"},
        "taiwan_strait":   {"bbox": [119.0, 22.0, 122.0, 26.0], "conflict": "taiwan_tension"},
        "south_china_sea": {"bbox": [109.0,  3.0, 121.0, 22.0], "conflict": "scs_dispute"},
        "strait_hormuz":   {"bbox": [ 56.0, 25.5,  59.5, 27.0], "conflict": "iran_tension"},
    }

    CHOKEPOINTS = {
        "hormuz":     (56.5,  26.5),
        "malacca":    (103.8,  1.2),
        "bab_mandeb": (43.4,  12.5),
        "suez":       (32.5,  30.7),
        "panama":     (-79.9,  9.0),
        "gibraltar":  (-5.4,  36.0),
        "dover":      (1.3,   51.0),
    }

    # ------------------------------------------------------------------ A
    def add_kinematic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        speed_category       : anchored / drifting / slow / cruising / fast
        delta_sog            : |SOG_t − SOG_{t-1}| per MMSI
        delta_cog            : |COG change|, 360° wrap-corrected
        time_diff_sec        : seconds since previous fix
        turning_rate         : delta_cog / time_diff_sec  (deg / s)
        is_dark_ship         : AIS gap > 6 h  (possible deliberate blackout)
        moored_vs_drifting   : SOG < 0.3 kn with valid heading
        sog_z_score          : per-MMSI standardized speed
        """
        df = df.sort_values(["MMSI", "BaseDateTime"]).copy()

        df["speed_category"] = pd.cut(
            df["SOG"],
            bins=[-0.1, 0.5, 3.0, 8.0, 15.0, 102.2],
            labels=["anchored", "drifting", "slow", "cruising", "fast"],
        )

        grp = df.groupby("MMSI", sort=False)
        df["delta_sog"]     = grp["SOG"].diff().abs()
        df["delta_cog"]     = grp["COG"].diff().abs().apply(
            lambda x: min(x, 360.0 - x) if pd.notna(x) else np.nan
        )
        df["time_diff_sec"] = grp["BaseDateTime"].diff().dt.total_seconds()
        df["turning_rate"]  = df["delta_cog"] / df["time_diff_sec"].replace(0, np.nan)
        df["is_dark_ship"]  = (df["time_diff_sec"] > 21_600).astype("int8")
        df["moored_vs_drifting"] = (
            (df["SOG"] < 0.3) & df["Heading"].notna()
        ).astype("int8")
        df["sog_z_score"]   = grp["SOG"].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-6)
        )
        return df

    # ------------------------------------------------------------------ B
    def add_geospatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        grid_cell            : 0.5° × 0.5° cell ID  (spatial aggregation key)
        in_conflict_zone     : boolean — inside any defined conflict zone
        conflict_zone_name   : zone label or "none"
        dist_{cp}_km         : Haversine distance to each strategic chokepoint
        """
        df["grid_lat"]  = (df["LAT"] // 0.5) * 0.5
        df["grid_lon"]  = (df["LON"] // 0.5) * 0.5
        df["grid_cell"] = df["grid_lat"].astype(str) + "_" + df["grid_lon"].astype(str)

        df["in_conflict_zone"]   = False
        df["conflict_zone_name"] = "none"
        for zone, info in self.CONFLICT_ZONES.items():
            b = info["bbox"]
            mask = (
                df["LON"].between(b[0], b[2]) &
                df["LAT"].between(b[1], b[3])
            )
            df.loc[mask, "in_conflict_zone"]   = True
            df.loc[mask, "conflict_zone_name"] = zone

        for name, (cp_lon, cp_lat) in self.CHOKEPOINTS.items():
            df[f"dist_{name}_km"] = self._haversine(
                df["LAT"], df["LON"], cp_lat, cp_lon
            )
        return df

    # ------------------------------------------------------------------ C
    def add_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        rolling_sog_mean_12h : 12-hour rolling mean SOG per MMSI
        rolling_sog_std_12h  : 12-hour rolling std SOG per MMSI
        route_entropy        : Shannon entropy of daily COG distribution (36 bins)
                               high entropy → unpredictable / evasive routing
        loitering_flag       : slow speed + frequent direction changes in conflict zone
        zig_zag_index        : COG direction reversal count over 10-point window
        """
        df = df.sort_values(["MMSI", "BaseDateTime"]).set_index("BaseDateTime")

        roll = df.groupby("MMSI")["SOG"].rolling("12H", min_periods=3)
        df["rolling_sog_mean_12h"] = roll.mean().reset_index(level=0, drop=True)
        df["rolling_sog_std_12h"]  = roll.std().reset_index(level=0, drop=True)
        df = df.reset_index()

        def _entropy(s: pd.Series) -> float:
            bins = pd.cut(s, bins=36, labels=False)
            p = bins.value_counts(normalize=True) + 1e-10
            return float(stats.entropy(p))

        df["_date"] = df["BaseDateTime"].dt.date
        ent = (
            df.groupby(["MMSI", "_date"])["COG"]
            .apply(_entropy)
            .reset_index()
            .rename(columns={"COG": "route_entropy"})
        )
        df = df.merge(ent, on=["MMSI", "_date"], how="left").drop(columns="_date")

        df["loitering_flag"] = (
            (df["SOG"] < 3.0) &
            (df["delta_cog"] > 45.0) &
            df["in_conflict_zone"]
        ).astype("int8")

        df["_cog_sign"]     = np.sign(df["delta_cog"].fillna(0))
        df["zig_zag_index"] = (
            df.groupby("MMSI")["_cog_sign"]
            .transform(lambda x: (x != x.shift()).rolling(10, min_periods=3).sum())
        )
        df.drop(columns="_cog_sign", inplace=True)
        return df

    # ------------------------------------------------------------------ D
    def add_temporal_aggregation(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aggregation unit: grid_cell × 6-hour time bucket.

        traffic_count        : unique vessels
        dark_ship_ratio      : fraction with AIS gap > 6 h
        military_ratio       : fraction VesselType == 35
        tanker_ratio         : fraction VesselType in {80–89}
        sar_count            : SAR vessel count
        mean_sog / std_sog   : speed statistics
        loitering_density    : loitering flag sum
        """
        df["time_bucket"] = df["BaseDateTime"].dt.floor("6H")
        agg = (
            df.groupby(["grid_cell", "time_bucket"])
            .agg(
                traffic_count     =("MMSI",           "nunique"),
                dark_ship_count   =("is_dark_ship",    "sum"),
                military_count    =("VesselType",      lambda x: (x == 35).sum()),
                cargo_count       =("VesselType",      lambda x: x.isin(range(70, 80)).sum()),
                tanker_count      =("VesselType",      lambda x: x.isin(range(80, 90)).sum()),
                sar_count         =("VesselType",      lambda x: (x == 51).sum()),
                mean_sog          =("SOG",             "mean"),
                std_sog           =("SOG",             "std"),
                loitering_density =("loitering_flag",  "sum"),
            )
            .reset_index()
        )
        denom = agg["traffic_count"].clip(lower=1)
        agg["dark_ship_ratio"] = agg["dark_ship_count"] / denom
        agg["military_ratio"]  = agg["military_count"]  / denom
        agg["tanker_ratio"]    = agg["tanker_count"]    / denom
        self.agg_df = agg
        return df, agg

    # ------------------------------------------------------------------ E
    def add_conflict_labels(
        self, df: pd.DataFrame, conflict_events_path: str
    ) -> pd.DataFrame:
        """
        conflict_label       : 1 if conflict event within 30 days in same zone
        days_to_conflict     : signed days to nearest event (regression target)
        conflict_intensity   : ACLED fatality count (continuous severity proxy)
        """
        events = pd.read_csv(conflict_events_path, parse_dates=["event_date"])
        df["conflict_label"]     = 0
        df["days_to_conflict"]   = np.nan
        df["conflict_intensity"] = 0.0

        for _, ev in events.iterrows():
            zone_mask = df["conflict_zone_name"] == ev.get("zone", "")
            day_diff  = (
                ev["event_date"] - df["BaseDateTime"].dt.tz_localize(None)
            ).dt.days
            match = zone_mask & day_diff.between(-7, 30)
            df.loc[match, "conflict_label"]     = 1
            df.loc[match, "days_to_conflict"]   = day_diff[match]
            df.loc[match, "conflict_intensity"] = ev.get("fatalities", 0)
        return df

    # ------------------------------------------------------------------ util
    @staticmethod
    def _haversine(lat1, lon1, lat2: float, lon2: float) -> pd.Series:
        R  = 6_371.0
        p1, p2 = np.radians(lat1), np.radians(lat2)
        dp = np.radians(lat2 - lat1)
        dl = np.radians(lon2 - lon1)
        a  = np.sin(dp / 2)**2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2)**2
        return R * 2 * np.arcsin(np.sqrt(a))

    def run(self, df: pd.DataFrame, conflict_events_path: str) -> pd.DataFrame:
        logger.info("A. Kinematic features...")
        df = self.add_kinematic_features(df)
        logger.info("B. Geospatial features...")
        df = self.add_geospatial_features(df)
        logger.info("C. Behavioral features...")
        df = self.add_behavioral_features(df)
        logger.info("D. Temporal aggregation...")
        df, _ = self.add_temporal_aggregation(df)
        logger.info("E. Conflict labels...")
        df = self.add_conflict_labels(df, conflict_events_path)
        return df
```

---

## PHASE 2 — VISUALIZATION (`src/visualization/`)

### `spatial_viz.py`
```python
"""
Outputs → outputs/figures/spatial/

Figures:
  1. Global vessel density heatmap (Folium HeatMapWithTime — animated)
     Conflict zone polygons overlaid; 6-hour time steps
  2. Individual vessel trajectory (SOG color-mapped polyline per MMSI)
  3. Dark ship cluster map (DBSCAN clusters inside conflict zones)
  4. Chokepoint traffic flow animation
  5. AIS gap event markers (is_dark_ship == 1)
"""
class SpatialVisualizer:
    def plot_global_density_heatmap(self, df, time_col="time_bucket"): ...
    def plot_vessel_trajectory(self, df, mmsi: int): ...
    def plot_dark_ship_clusters(self, df): ...
    def animate_chokepoint_flow(self, df, chokepoint: str): ...
```

### `temporal_viz.py`
```python
"""
Outputs → outputs/figures/temporal/

Figures:
  1. Traffic volume vs. conflict events (line + axvline markers, 95% CI shading)
  2. Vessel type composition change — stacked area chart
     Military / SAR / Law Enforcement surge detection
  3. Speed distribution change — monthly violin plots
  4. Dark ship ratio time series per conflict zone
  5. Cross-Correlation Function (CCF): AIS indicators vs. conflict intensity
     Lead/lag axis ±30 days; significant lags highlighted
"""
class TemporalVisualizer:
    def plot_traffic_vs_conflict(self, agg_df, events): ...
    def plot_vessel_type_composition(self, df, zone: str): ...
    def plot_speed_violin_series(self, df, zone: str): ...
    def plot_ccf(self, ais_series, conflict_series, max_lag: int = 30): ...
```

### `statistical_viz.py`
```python
"""
Outputs → outputs/figures/statistical/

Figures:
  1. Pre/post conflict feature distributions — KDE + boxplot overlay
  2. Spearman correlation heatmap (AIS features × conflict indicators)
  3. Random Forest feature importance bar chart
  4. ROC-AUC curves — LSTM / XGBoost / baseline comparison
  5. Precision–Recall curves (imbalanced label evaluation)
  6. SHAP summary bubble plot (model interpretability)
  7. Confusion matrices — per conflict zone, T+7 and T+30 horizons
"""
class StatisticalVisualizer:
    def plot_pre_post_distributions(self, df, features: list, event_date: str): ...
    def plot_correlation_heatmap(self, df): ...
    def plot_roc_comparison(self, models: dict, X_test, y_test): ...
    def plot_shap_summary(self, model, X): ...
```

---

## PHASE 3 — ANALYSIS (`src/analysis/`)

### `correlation_analyzer.py`
```python
"""
Conflict Correlation Analyzer
==============================
Statistical framework for establishing AIS → conflict relationships.

Methods:
  1. Granger Causality     — does AIS anomaly lead conflict intensity?
  2. Cross-Correlation     — optimal lead-lag window (days)
  3. Difference-in-Differences — conflict zone vs. control zone
  4. Event Study Analysis  — ±30-day mean comparison (Mann–Whitney U)
  5. Interrupted Time Series — level & slope change at conflict onset
"""
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.stats import mannwhitneyu
import statsmodels.api as sm
import numpy as np, pandas as pd


class ConflictCorrelationAnalyzer:

    CONFLICT_ONSET = {
        "ukraine_war":      "2022-02-24",
        "houthi_crisis":    "2023-11-19",
        "pla_taiwan_drill": "2022-08-04",
        "kerch_bridge":     "2022-10-08",
    }

    def granger_causality(
        self,
        ais_series: pd.Series,
        conflict_series: pd.Series,
        max_lag: int = 30,
    ) -> dict:
        """
        H₀: AIS anomaly index does NOT Granger-cause conflict intensity.
        Reject at p < 0.05 → AIS provides a statistically significant
        leading signal.
        """
        df = pd.DataFrame({"ais": ais_series, "conflict": conflict_series}).dropna()
        raw = grangercausalitytests(
            df[["conflict", "ais"]], maxlag=max_lag, verbose=False
        )
        sig = {
            lag: res[0]["ssr_ftest"][1]
            for lag, res in raw.items()
            if res[0]["ssr_ftest"][1] < 0.05
        }
        return {
            "significant_lags": sig,
            "optimal_lead_days": min(sig, key=sig.get) if sig else None,
        }

    def event_study(
        self,
        df: pd.DataFrame,
        event_date: str,
        zone: str,
        window_days: int = 30,
    ) -> pd.DataFrame:
        """
        Borrows the Abnormal Return framework from finance —
        adapted here as Abnormal Traffic Volume (ATV).
        Compares pre/post means with Mann–Whitney U test.
        """
        ev = pd.Timestamp(event_date, tz="UTC")
        z  = df[df["conflict_zone_name"] == zone].set_index("BaseDateTime").sort_index()
        pre  = z[ev - pd.Timedelta(days=window_days): ev]
        post = z[ev: ev + pd.Timedelta(days=window_days)]

        metrics = [
            "traffic_count", "dark_ship_ratio", "military_ratio",
            "tanker_ratio", "mean_sog", "loitering_density",
        ]
        rows = []
        for m in metrics:
            pv = pre[m].dropna(); qv = post[m].dropna()
            _, p = mannwhitneyu(pv, qv, alternative="two-sided")
            rows.append({
                "metric":      m,
                "pre_mean":    pv.mean(),
                "post_mean":   qv.mean(),
                "pct_change":  (qv.mean() - pv.mean()) / (pv.mean() + 1e-10) * 100,
                "p_value":     p,
                "significant": p < 0.05,
            })
        return pd.DataFrame(rows)

    def difference_in_differences(
        self,
        df: pd.DataFrame,
        treatment_zone: str,
        control_zone: str,
        event_date: str,
        metric: str = "traffic_count",
    ) -> dict:
        """
        DiD estimator = (Post_T − Pre_T) − (Post_C − Pre_C).
        Isolates the conflict effect net of global maritime trends.
        """
        ev = pd.Timestamp(event_date, tz="UTC")
        res = {}
        for zone in [treatment_zone, control_zone]:
            z = df[df["conflict_zone_name"] == zone].set_index("BaseDateTime")
            res[zone] = {"pre": z.loc[:ev, metric].mean(),
                         "post": z.loc[ev:, metric].mean()}
        t, c = res[treatment_zone], res[control_zone]
        did  = (t["post"] - t["pre"]) - (c["post"] - c["pre"])
        return {"did_estimator": did, "zones": res}

    def interrupted_time_series(
        self, series: pd.Series, breakpoint: str
    ) -> dict:
        """
        OLS segmented regression:
          y = β₀ + β₁·t + β₂·D + β₃·(t × D) + ε
          D = 0 pre-conflict, D = 1 post-conflict
          β₂ = level shift,  β₃ = slope change
        """
        series = series.dropna().sort_index()
        t   = np.arange(len(series))
        bp  = series.index.searchsorted(pd.Timestamp(breakpoint, tz="UTC"))
        D   = (t >= bp).astype(int)
        X   = sm.add_constant(np.column_stack([t, D, t * D]))
        res = sm.OLS(series.values, X).fit()
        return {
            "level_shift":  res.params[2],
            "slope_change": res.params[3],
            "level_p":      res.pvalues[2],
            "slope_p":      res.pvalues[3],
            "r_squared":    res.rsquared,
        }
```

---

## PHASE 4 — MODELS (`src/models/`)

### `anomaly_model.py` — Unsupervised Detection
```python
"""
Maritime Anomaly Detection
==========================
Operates without conflict labels. Detects behavioral outliers
that correlate with pre-conflict maritime patterns.

Models:
  1. Isolation Forest         — multivariate point anomaly score
  2. Variational Autoencoder  — reconstruction error on learned normal pattern
  3. DBSCAN                   — spatial density-based cluster anomaly
  4. Local Outlier Factor     — neighborhood density deviation

Input features:
  SOG, delta_sog, delta_cog, turning_rate,
  rolling_sog_std_12h, route_entropy, zig_zag_index,
  loitering_flag, is_dark_ship, dist_*_km, traffic_count

Outputs:
  anomaly_score   float ∈ [0, 1]
  anomaly_label   binary
  anomaly_type    "dark_ship" | "loitering" | "zig_zag"
                  | "density_surge" | "speed_spike"
"""
import torch, torch.nn as nn


class MaritimeVAE(nn.Module):
    """Variational Autoencoder — learns the normal maritime behavior distribution."""
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),        nn.ReLU(),
        )
        self.fc_mu  = nn.Linear(32, latent_dim)
        self.fc_var = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 64),         nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def reparameterize(self, mu, log_var):
        return mu + torch.exp(0.5 * log_var) * torch.randn_like(log_var)

    def forward(self, x):
        h = self.encoder(x)
        mu, lv = self.fc_mu(h), self.fc_var(h)
        return self.decoder(self.reparameterize(mu, lv)), mu, lv
```

### `conflict_predictor.py` — Supervised Prediction
```python
"""
Conflict Prediction Model
=========================
Estimates P(conflict within T+N days) per grid cell × time bucket.

Architectures compared:
  1. Bidirectional LSTM + Multi-Head Attention  (primary)
  2. Temporal Fusion Transformer  (static + dynamic inputs)
  3. XGBoost  (interpretable gradient-boosted baseline)
  4. Random Forest  (feature importance baseline)

Training setup:
  Target       : conflict_label (binary); days_to_conflict (regression)
  Horizons     : T+3, T+7, T+14, T+30 days — evaluated separately
  Imbalance    : SMOTE + class_weight adjustment
  Validation   : temporal split — train ≤ 2022, val 2023-Q1, test 2023-Q2+
  Loss         : Binary Cross-Entropy + Focal loss term (γ = 2)

Evaluation metrics:
  AUROC, AUPRC        — primary (imbalanced labels)
  F2-Score            — recall-prioritized (missed conflicts costlier)
  Mean Lead Time      — average advance warning days before conflict
  False Alarm Rate    — false positives per 30-day window
"""
import torch, torch.nn as nn


class ConflictLSTM(nn.Module):
    """Bidirectional LSTM with multi-head self-attention."""
    def __init__(
        self,
        input_dim:  int,
        hidden_dim: int   = 128,
        num_layers: int   = 2,
        num_heads:  int   = 4,
        dropout:    float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True,
        )
        self.attention = nn.MultiheadAttention(
            hidden_dim * 2, num_heads=num_heads,
            batch_first=True, dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _  = self.lstm(x)
        attn_out, _  = self.attention(lstm_out, lstm_out, lstm_out)
        pooled       = self.norm(lstm_out + attn_out).mean(dim=1)
        return self.head(pooled).squeeze(-1)


class FocalBCELoss(nn.Module):
    """Focal binary cross-entropy for highly imbalanced conflict labels."""
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        bce = nn.functional.binary_cross_entropy(pred, target, reduction="none")
        pt  = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()
```

---

## EXTERNAL DATA SOURCES

| Dataset | Source | URL | Purpose |
|---------|--------|-----|---------|
| Armed conflict events | ACLED | acleddata.com | Conflict labels & intensity |
| News events | GDELT 2.0 | gdeltproject.org | Conflict intensity proxy |
| EEZ boundaries | MarineRegions | marineregions.org | Jurisdiction features |
| World port index | NGA (US) | msi.nga.mil | Port origin/destination inference |
| Sea area polygons | OpenSeaMap | openseamap.org | Conflict zone definition |
| Sanctioned vessels | OFAC SDN | sanctionssearch.ofac.treas.gov | Sanctions evasion detection |
| Historical AIS | MarineTraffic / AISHub | marinetraffic.com | Extended time-series |

---

## EXECUTION COMMANDS

```bash
# Environment setup
conda create -n mcis python=3.11 -y && conda activate mcis
pip install -r requirements.txt

# Step 1 — Clean
python -m src.preprocessing.cleaner \
    --input  ./data/raw/ais_raw.csv \
    --output ./data/processed/ais_clean.parquet

# Step 2 — Features
python -m src.preprocessing.feature_engineer \
    --input           ./data/processed/ais_clean.parquet \
    --conflict-events ./data/external/acled_events.csv \
    --output          ./data/processed/ais_features.parquet

# Step 3 — Visualization
python -m src.visualization.spatial_viz \
    --input ./data/processed/ais_features.parquet --output-dir ./outputs/figures/

# Step 4 — Correlation analysis
python -m src.analysis.correlation_analyzer \
    --input ./data/processed/ais_features.parquet --output ./outputs/tables/

# Step 5 — Train anomaly model
python -m src.models.anomaly_model \
    --input ./data/processed/ais_features.parquet --output ./outputs/models/anomaly/

# Step 6 — Train conflict predictor
python -m src.models.conflict_predictor \
    --input ./data/processed/ais_features.parquet \
    --mode train --output ./outputs/models/predictor/

# Full pipeline
bash scripts/run_pipeline.sh
```

---

## REQUIREMENTS

```
# requirements.txt
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0
torch>=2.0.0
statsmodels>=0.14.0
geopandas>=0.13.0
shapely>=2.0.0
folium>=0.14.0
plotly>=5.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyarrow>=12.0.0
xgboost>=1.7.0
lightgbm>=4.0.0
shap>=0.42.0
imbalanced-learn>=0.11.0
prophet>=1.1.4
pytorch-forecasting>=1.0.0
jupyter>=1.0.0
tqdm>=4.65.0
pyyaml>=6.0
```

---

## PAPER STRUCTURE

```
Title:
  "Maritime Traffic Anomaly Detection as a Precursor to Armed Conflict:
   Evidence from Global AIS Data in Active Hotspots (2022–2024)"

Abstract

1.  Introduction
    1.1  Motivation — why AIS data for conflict early warning?
    1.2  Research questions and hypotheses
    1.3  Contributions

2.  Background and Related Work
    2.1  AIS data characteristics and known limitations
    2.2  Prior work: maritime anomaly detection
    2.3  Prior work: conflict early-warning systems

3.  Data and Methodology
    3.1  AIS data collection and preprocessing pipeline
    3.2  Conflict event data — ACLED integration
    3.3  Feature engineering
    3.4  Analytical framework (Granger, DiD, ITS, Event Study)

4.  Empirical Results
    4.1  Black Sea (Russia–Ukraine War) — traffic collapse & dark ship surge
    4.2  Red Sea (Houthi Crisis) — tanker rerouting pattern detection
    4.3  Taiwan Strait — PLA exercise anomaly window
    4.4  Leading indicator analysis — Granger test results across all zones

5.  Predictive Modeling
    5.1  Model comparison — LSTM-Attention vs. TFT vs. XGBoost
    5.2  Zone-level prediction performance (AUROC, F2, Lead Time)
    5.3  SHAP analysis — dominant predictive features
    5.4  Horizon sensitivity — T+3 to T+30 day evaluation

6.  Discussion
    6.1  AIS data limitations (spoofing, Class B gaps, flag state masking)
    6.2  Policy implications — designing an operational early-warning system

7.  Conclusion

References
Appendix A — Supplementary Figures
Appendix B — Full Statistical Tables
```

---

## AGENTIC EXECUTION PLAN

```
[Task 1]  src/preprocessing/cleaner.py  +  tests/test_cleaner.py
          output → data/processed/ais_clean.parquet

[Task 2]  src/preprocessing/feature_engineer.py  +  tests/test_features.py
          output → data/processed/ais_features.parquet

[Task 3]  notebooks/01_EDA.ipynb
          output → outputs/figures/eda/

[Task 4]  src/visualization/  (spatial + temporal + statistical)
          output → outputs/figures/  (300 dpi PNG)

[Task 5]  src/analysis/correlation_analyzer.py
          Granger + DiD + ITS + Event Study per zone
          output → outputs/tables/

[Task 6]  src/models/anomaly_model.py
          Isolation Forest + VAE ensemble
          output → outputs/models/anomaly/

[Task 7]  src/models/conflict_predictor.py
          LSTM-Attention + XGBoost training & evaluation
          output → outputs/models/predictor/

[Task 8]  scripts/generate_report.py
          Compile all figures & tables → PDF
          output → outputs/reports/mcis_final_report.pdf
```

---

## CODING CONVENTIONS

```python
# 1. Type hints on all public functions
def process(df: pd.DataFrame, config: dict) -> pd.DataFrame: ...

# 2. Logging — no bare print() inside src/
import logging
logger = logging.getLogger(__name__)
logger.info(f"Records after MMSI filter: {len(df):,}")

# 3. No hardcoded paths or parameters — use config/settings.yaml

# 4. Intermediate data → Parquet only (never CSV for large files)
df.to_parquet(path, index=False, compression="snappy")

# 5. Reproducibility — set all seeds at module level
SEED = 42
import random, numpy as np, torch
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# 6. Publication-quality figure defaults
import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.dpi":        300,
    "figure.figsize":    (10, 6),
    "font.family":       "DejaVu Sans",
    "font.size":         12,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
})

# 7. All statistical outputs must include p-value and effect size
# 8. Every module must expose a CLI entry point via argparse
# 9. Long loops require tqdm progress bars
```
