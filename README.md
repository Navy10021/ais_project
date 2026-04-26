# 🛡️ Maritime Conflict Intelligence System (MCIS)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)
![AIS](https://img.shields.io/badge/Data-AIS%20%7C%20ACLED%20%7C%20GDELT-blue?style=flat-square)

**AIS-based Maritime Armed Conflict Early Warning System**

*Detecting conflict precursors through vessel behavioral pattern analysis*

[Overview](#overview) · [Architecture](#architecture) · [Quickstart](#quick-start) · [Modules](#modules) · [Data Schema](#data-schema) · [Results](#output-structure)

</div>

---

## Overview

MCIS analyzes Automatic Identification System (AIS) data to uncover statistically significant correlations between maritime behavioral patterns and armed conflicts — and builds a predictive system capable of detecting conflict precursors before they materialize.

### Core Research Question

> *"Do maritime vessel behavioral patterns — traffic density, speed, vessel type composition, and route entropy — exhibit statistically significant changes before and after the onset of armed conflicts? Can these signals be used to predict conflict events in advance?"*

### Target Conflict Zones

| Conflict | Period | Key Maritime Zone | Status |
|----------|--------|-------------------|--------|
| Russia–Ukraine War | 2022-02-24 → | Black Sea, Sea of Azov | 🔴 Active |
| Red Sea / Houthi Crisis | 2023-11 → | Red Sea, Gulf of Aden, Bab-el-Mandeb | 🔴 Active |
| Taiwan Strait Tensions | 2022-08 (PLA drills) | Taiwan Strait | 🟡 Monitoring |
| South China Sea Disputes | Ongoing | SCS, Paracel & Spratly Islands | 🟡 Monitoring |
| Iran–Gulf Tensions | Ongoing | Strait of Hormuz | 🟡 Monitoring |

---

## Architecture

```
                        ┌──────────────────────────────────┐
                        │         AIS Raw Data             │
                        │  (MMSI · Position · Speed · COG) │
                        └────────────────┬─────────────────┘
                                         │
                        ┌────────────────▼────────────────┐
                        │        Preprocessing            │
                        │  cleaner.py · feature_engineer  │
                        └────────────────┬────────────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              │                          │                          │
  ┌───────────▼──────────┐  ┌────────────▼───────────┐  ┌──────────▼──────────┐
  │   Traffic Analysis   │  │ Behavioral Analysis    │  │  Network Analysis   │
  │  volume · density    │  │ loitering · dark ships │  │  port · hub vessels │
  │  anomaly detection   │  │ zigzag · route entropy │  │  zone connectivity  │
  └───────────┬──────────┘  └────────────┬───────────┘  └──────────┬──────────┘
              │                          │                          │
              └──────────────────────────┼──────────────────────────┘
                                         │
                        ┌────────────────▼────────────────┐
                        │     Correlation Analysis        │
                        │  Granger · DiD · ITS · Event    │
                        └────────────────┬────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
        ┌───────────▼──────────┐  ┌─────▼──────────┐  ┌──────▼──────────────┐
        │   Anomaly Detection  │  │ Conflict Pred. │  │    Visualization    │
        │  IsoForest·LOF·DBSCAN│  │  RF·GBM·XGBoost│  │ spatial·temporal    │
        └──────────────────────┘  └────────────────┘  └─────────────────────┘
                                         │
                        ┌────────────────▼────────────────┐
                        │       MCIS Final Report         │
                        │    HTML · CSV · Figures         │
                        └─────────────────────────────────┘
```

---

## Project Structure

```
ais-project/
├── CLAUDE.md                    # Architecture specification (EN)
├── CLAUDE_KOR.md               # Architecture specification (KR)
├── README.md                   # This file
├── requirements.txt
├── config/
│   └── settings.yaml          # Global configuration
├── data/
│   ├── raw/                   # Source AIS data
│   ├── processed/             # Cleaned & feature-engineered
│   └── external/              # ACLED, GDELT, port databases
├── src/
│   ├── preprocessing/         # Cleaner, Feature Engineer
│   ├── analysis/              # Traffic, Behavioral, Network, Correlation
│   ├── models/                # Anomaly, Predictor, Baseline, Evaluator
│   └── visualization/         # Spatial, Temporal, Statistical
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_preprocessing_validation.ipynb
│   ├── 03_visualization.ipynb
│   ├── 04_conflict_correlation.ipynb
│   └── 05_model_development.ipynb
├── tests/
├── scripts/
│   ├── run_eda.py
│   ├── run_pipeline.py
│   └── generate_report.py
├── run_mcis.bat               # Windows launcher
└── outputs/
    ├── figures/               # PNG visualizations
    ├── tables/                # CSV analysis results
    ├── models/                # Trained models (joblib)
    └── reports/               # HTML / TXT reports
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/Navy10021/mcis
cd ais_project

conda create -n mcis python=3.11 -y
conda activate mcis

pip install -r requirements.txt
```

### Full Pipeline

```bash
# Windows
run_mcis.bat

# Python
python -m scripts.run_pipeline --step full
```

### Step-by-Step Execution

```bash
# 1. Data Cleaning
python -m src.preprocessing.cleaner \
    --input data/raw/ais_raw.csv \
    --output data/processed/ais_clean.parquet

# 2. Feature Engineering
python -m src.preprocessing.feature_engineer \
    --input data/processed/ais_clean.parquet \
    --output data/processed/ais_features.parquet

# 3. Visualization
python -m src.visualization.spatial_viz    --input data/processed/ais_features.parquet --output-dir outputs/figures/spatial
python -m src.visualization.temporal_viz   --input data/processed/ais_features.parquet --output-dir outputs/figures/temporal
python -m src.visualization.statistical_viz --input data/processed/ais_features.parquet --output-dir outputs/figures/statistical

# 4. Analysis
python -m src.analysis.correlation_analyzer --input data/processed/ais_features.parquet --output-dir outputs/tables
python -m src.analysis.traffic_analyzer     --input data/processed/ais_features.parquet --output-dir outputs/tables
python -m src.analysis.behavioral_analyzer  --input data/processed/ais_features.parquet --output-dir outputs/tables
python -m src.analysis.network_analyzer     --input data/processed/ais_features.parquet --output-dir outputs/tables

# 5. Anomaly Detection
python -m src.models.anomaly_model --input data/processed/ais_features.parquet --contamination 0.05

# 6. Conflict Prediction
python -m src.models.conflict_predictor --input data/processed/ais_features.parquet --test-size 0.2

# 7. Report Generation
python scripts/generate_report.py
```

---

## Modules

### Preprocessing · `src/preprocessing/`

| Module | Description |
|--------|-------------|
| `cleaner.py` | MMSI validation, coordinate validation, outlier removal, imputation |
| `feature_engineer.py` | Kinematic, geospatial & behavioral features; temporal aggregation |

### Analysis · `src/analysis/`

| Module | Description |
|--------|-------------|
| `correlation_analyzer.py` | Granger causality, event study, DiD, interrupted time series (ITS) |
| `traffic_analyzer.py` | Traffic volume, grid density, anomaly detection |
| `behavioral_analyzer.py` | Speed behavior, loitering detection, dark ships, zigzag patterns |
| `network_analyzer.py` | Port visits, zone connectivity, hub vessel identification |

### Models · `src/models/`

| Module | Algorithms |
|--------|-----------|
| `anomaly_model.py` | Isolation Forest, LOF, DBSCAN |
| `conflict_predictor.py` | Random Forest, Gradient Boosting, XGBoost, Logistic Regression |
| `baseline.py` | Naive baselines for benchmarking |
| `evaluator.py` | AUROC, AUPRC, F1/F2, calibration curves |

### Visualization · `src/visualization/`

| Module | Description |
|--------|-------------|
| `base.py` | Shared configuration, color schemes, style setup |
| `spatial_viz.py` | Density heatmaps, vessel trajectories, dark ship locations, chokepoints |
| `temporal_viz.py` | Traffic volume trends, vessel type composition, speed time series |
| `statistical_viz.py` | Correlation heatmaps, feature distributions, model evaluation charts |

---

## Data Schema

### AIS Input

| Column | Type | Description | Valid Range |
|--------|------|-------------|-------------|
| `MMSI` | int64 | Maritime Mobile Service Identity | 100000000–999999999 |
| `BaseDateTime` | datetime | AIS timestamp (UTC, ISO 8601) | — |
| `LAT` / `LON` | float32 | Position | ±90 / ±180 |
| `SOG` | float32 | Speed Over Ground (knots) | 0.0–102.2 |
| `COG` | float32 | Course Over Ground (degrees) | 0.0–359.9 |
| `Heading` | float32 | True heading (degrees) | 0–359 |
| `VesselType` | float32 | ITU/IMO vessel type code | 0–99 |
| `Status` | float32 | Navigational status | 0–15 |
| `Length` / `Width` / `Draft` | float32 | Vessel dimensions | Various |

### Engineered Features

| Feature | Description |
|---------|-------------|
| `speed_category` | Anchored · Drifting · Slow · Cruising · Fast |
| `delta_sog` / `delta_cog` | Rate of change in speed / course |
| `is_dark_ship` | AIS transmission gap > 6 hours |
| `loitering_flag` | Low speed + high turn rate in conflict zone |
| `route_entropy` | Shannon entropy of COG distribution |
| `in_conflict_zone` | Boolean — inside any defined conflict zone |
| `dist_*_km` | Haversine distance to strategic chokepoints |

---

## Output Structure

### `outputs/figures/`

| Directory | Contents |
|-----------|----------|
| `eda/` | Missing values, vessel types, speed distribution, geographic & temporal plots |
| `spatial/` | Density heatmaps, trajectory maps, dark ship positions, chokepoint overlays |
| `temporal/` | Traffic volume trends, vessel type composition, speed time series |
| `statistical/` | Correlation heatmaps, feature distributions, model comparison charts |

### `outputs/tables/`

| File | Description |
|------|-------------|
| `zone_statistics.csv` | Per-zone summary statistics |
| `*_event_study.csv` | Event study results by conflict |
| `*_its.csv` | Interrupted time series results |
| `traffic_*.csv` | Traffic analysis outputs |
| `behavioral_*.csv` | Behavioral analysis outputs |

### `outputs/models/`

| Directory | Files |
|-----------|-------|
| `anomaly/` | `isolation_forest.joblib` · `lof.joblib` · `scaler.joblib` · `anomaly_results.parquet` |
| `predictor/` | `random_forest.joblib` · `gradient_boosting.joblib` · `evaluation_results.csv` · `feature_importance.csv` |

---

## Configuration

`config/settings.yaml`:

```yaml
data:
  raw_dir: "./data/raw"
  processed_dir: "./data/processed"

preprocessing:
  seed: 42
  min_mmsi: 200000000
  max_mmsi: 799999999
  invalid_lat: 91.0
  invalid_sog: 102.3

features:
  grid_resolution: 0.5          # degrees
  time_bucket: "6h"
  rolling_window: "12h"
  dark_ship_threshold_seconds: 21600

conflict_zones:
  black_sea:
    bbox: [27.0, 40.5, 41.0, 46.8]
  red_sea:
    bbox: [32.0, 12.0, 43.5, 30.0]
  taiwan_strait:
    bbox: [119.0, 22.0, 122.5, 26.5]
  south_china_sea:
    bbox: [105.0, 5.0, 122.0, 23.0]
  strait_of_hormuz:
    bbox: [55.0, 22.0, 60.0, 27.0]

models:
  anomaly:
    contamination: 0.05
    n_estimators: 100
  prediction:
    test_size: 0.2
    n_estimators: 100
    max_depth: 10

visualization:
  dpi: 300
  figsize: [10, 6]
```

---

## Requirements

```
# Core
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0
statsmodels>=0.14.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyyaml>=6.0
joblib>=1.3.0

# Optional
xgboost
lightgbm
geopandas
shapely
prophet
torch
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Module-specific
pytest tests/test_cleaner.py -v
pytest tests/test_features.py -v
pytest tests/test_models.py -v
```

---

## Notebooks

```bash
jupyter notebook notebooks/
```

| Notebook | Purpose |
|----------|---------|
| `01_EDA.ipynb` | Exploratory Data Analysis |
| `02_preprocessing_validation.ipynb` | Data quality & validation |
| `03_visualization.ipynb` | Full visualization suite |
| `04_conflict_correlation.ipynb` | Conflict correlation analysis |
| `05_model_development.ipynb` | Anomaly detection & prediction |

---

## References

- [ACLED](https://acleddata.com) — Armed Conflict Location & Event Data Project
- [GDELT](https://www.gdeltproject.org) — Global Database of Events, Language, and Tone
- [MarineTraffic](https://www.marinetraffic.com) / [AISHub](https://www.aishub.net) — AIS Data Providers
- [MarineRegions](https://www.marineregions.org) — Maritime Boundary & Zone Definitions
- ITU/IMO Vessel Type Code Reference

---

## License

Distributed under the **MIT License**. See `LICENSE` for details.

---

<div align="center">
<sub>Built for maritime intelligence research · @Navy10021</sub>
</div>
