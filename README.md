# Maritime Conflict Intelligence System (MCIS)

AIS-based maritime armed conflict early warning system.

## Project Overview

Analyze AIS (Automatic Identification System) data to uncover statistically significant correlations between maritime behavioral patterns and armed conflicts, and build a predictive system capable of detecting conflict precursors before they materialize.

### Core Research Question

> "Do maritime vessel behavioral patterns — traffic density, speed, vessel type composition, and route entropy — exhibit statistically significant changes before and after the onset of armed conflicts? Can these signals be used to predict conflict events in advance?"

### Target Conflicts

| Conflict | Period | Key Maritime Zone |
|----------|--------|-------------------|
| Russia–Ukraine War | 2022-02-24 → | Black Sea, Sea of Azov |
| Red Sea / Houthi Crisis | 2023-11 → | Red Sea, Gulf of Aden, Bab-el-Mandeb |
| Taiwan Strait Tensions | 2022-08 (PLA drills) | Taiwan Strait |
| South China Sea Disputes | Ongoing | SCS, Paracel & Spratly Islands |
| Iran–Gulf Tensions | Ongoing | Strait of Hormuz |

## Project Structure

```
ais-project/
├── CLAUDE.md                    # Architecture specification
├── CLAUDE_KOR.md               # Korean translation
├── README.md                   # This file
├── requirements.txt             # Python dependencies
├── config/
│   └── settings.yaml          # Global configuration
├── data/
│   ├── raw/                   # Source AIS data
│   ├── processed/            # Cleaned & features
│   └── external/              # ACLED, GDELT, ports
├── src/
│   ├── preprocessing/        # Cleaner, Feature Engineer
│   ├── analysis/             # Traffic, Behavioral, Network, Correlation
│   ├── models/              # Anomaly, Predictor, Baseline, Evaluator
│   └── visualization/        # Spatial, Temporal, Statistical
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_preprocessing_validation.ipynb
│   ├── 03_visualization.ipynb
│   ├── 04_conflict_correlation.ipynb
│   └── 05_model_development.ipynb
├── tests/
├── scripts/
│   ├── run_eda.py           # EDA runner
│   ├── run_pipeline.py      # Full pipeline runner
│   └── generate_report.py   # Report generator
├── run_mcis.bat            # Windows launcher
└── outputs/
    ├── figures/             # Visualization outputs (PNG)
    ├── tables/             # Analysis results (CSV)
    ├── models/             # Trained models (joblib)
    └── reports/             # Final reports (HTML)
```

## Installation

```bash
# Clone repository
git clone <repo-url>
cd ais_project

# Create virtual environment
conda create -n mcis python=3.11 -y
conda activate mcis

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run Full Pipeline

```bash
# Windows
run_mcis.bat

# Or Python
python -m scripts.run_pipeline --step full
```

### Individual Steps

```bash
# Step 1: Data Cleaning
python -m src.preprocessing.cleaner --input data/raw/ais_raw.csv --output data/processed/ais_clean.parquet

# Step 2: Feature Engineering
python -m src.preprocessing.feature_engineer --input data/processed/ais_clean.parquet --output data/processed/ais_features.parquet

# Step 3: Visualizations
python -m src.visualization.spatial_viz --input data/processed/ais_features.parquet --output-dir outputs/figures/spatial
python -m src.visualization.temporal_viz --input data/processed/ais_features.parquet --output-dir outputs/figures/temporal
python -m src.visualization.statistical_viz --input data/processed/ais_features.parquet --output-dir outputs/figures/statistical

# Step 4: Analysis
python -m src.analysis.correlation_analyzer --input data/processed/ais_features.parquet --output-dir outputs/tables
python -m src.analysis.traffic_analyzer --input data/processed/ais_features.parquet --output-dir outputs/tables
python -m src.analysis.behavioral_analyzer --input data/processed/ais_features.parquet --output-dir outputs/tables
python -m src.analysis.network_analyzer --input data/processed/ais_features.parquet --output-dir outputs/tables

# Step 5: Anomaly Detection
python -m src.models.anomaly_model --input data/processed/ais_features.parquet --contamination 0.05

# Step 6: Conflict Prediction
python -m src.models.conflict_predictor --input data/processed/ais_features.parquet --test-size 0.2

# Step 7: Generate Report
python scripts/generate_report.py
```

## Modules

### Preprocessing (`src/preprocessing/`)
| Module | Description |
|--------|-------------|
| `cleaner.py` | Data cleaning, MMSI validation, coordinate validation, imputation |
| `feature_engineer.py` | Kinematic, geospatial, behavioral features, temporal aggregation |

### Analysis (`src/analysis/`)
| Module | Description |
|--------|-------------|
| `correlation_analyzer.py` | Granger causality, event study, DiD, interrupted time series |
| `traffic_analyzer.py` | Traffic volume, grid density, anomaly detection |
| `behavioral_analyzer.py` | Speed behavior, loitering, dark ships,zigzag patterns |
| `network_analyzer.py` | Port visits, zone connectivity, hub vessels |

### Models (`src/models/`)
| Module | Description |
|--------|-------------|
| `anomaly_model.py` | Isolation Forest, LOF, DBSCAN |
| `conflict_predictor.py` | Random Forest, Gradient Boosting, XGBoost, Logistic Regression |
| `baseline.py` | Simple baselines for comparison |
| `evaluator.py` | AUROC, AUPRC, F1, F2, calibration |

### Visualization (`src/visualization/`)
| Module | Description |
|--------|-------------|
| `base.py` | Shared configuration and setup |
| `spatial_viz.py` | Density heatmap, trajectories, dark ships, chokepoints |
| `temporal_viz.py` | Traffic volume, vessel types, speed distribution, dark ship ratio |
| `statistical_viz.py` | Correlation heatmap, feature distributions, model evaluation |

## Data Schema

### AIS Input Columns

| Column | Type | Description | Valid Range |
|--------|------|-------------|-------------|
| MMSI | int64 | Maritime Mobile Service Identity | 100000000–999999999 |
| BaseDateTime | datetime | AIS timestamp (UTC) | ISO 8601 |
| LAT | float32 | Latitude | -90.0 to 90.0 |
| LON | float32 | Longitude | -180.0 to 180.0 |
| SOG | float32 | Speed Over Ground (knots) | 0.0 to 102.2 |
| COG | float32 | Course Over Ground (degrees) | 0.0 to 359.9 |
| Heading | float32 | True heading (degrees) | 0–359 |
| VesselType | float32 | Vessel type code (ITU/IMO) | 0–99 |
| Status | float32 | Navigational status code | 0–15 |
| Length/Width/Draft | float32 | Vessel dimensions | Various |

### Engineered Features

| Feature | Description |
|---------|-------------|
| speed_category | Anchored, drifting, slow, cruising, fast |
| delta_sog/delta_cog | Speed and course changes |
| is_dark_ship | AIS gap > 6 hours |
| loitering_flag | Slow speed + high turn rate in conflict zone |
| route_entropy | Shannon entropy of COG distribution |
| in_conflict_zone | Boolean - inside any defined zone |
| dist_*_km | Haversine distance to strategic chokepoints |

## Output Structure

### `outputs/figures/`
| Directory | Contents |
|-----------|----------|
| `eda/` | EDA plots: missing values, vessel types, speed distribution, geographic, temporal |
| `spatial/` | Spatial plots: density heatmap, trajectories, dark ships, chokepoints |
| `temporal/` | Temporal plots: traffic volume, vessel type composition, speed time series |
| `statistical/` | Statistical plots: correlation heatmap, feature distributions, model comparison |

### `outputs/tables/`
- `zone_statistics.csv` - Per-zone summary statistics
- `*_event_study.csv` - Event study results
- `*_its.csv` - Interrupted time series results
- `traffic_*.csv` - Traffic analysis results
- `behavioral_*.csv` - Behavioral analysis results

### `outputs/models/`
| Directory | Files |
|-----------|------|
| `anomaly/` | `isolation_forest.joblib`, `lof.joblib`, `scaler.joblib`, `anomaly_results.parquet` |
| `predictor/` | `random_forest.joblib`, `gradient_boosting.joblib`, `scaler.joblib`, `evaluation_results.csv`, `feature_importance.csv` |
| `evaluator/` | Evaluation reports |
| `baseline/` | Baseline model results |

### `outputs/reports/`
- `mcis_final_report.html` - Interactive HTML report
- `mcis_final_report.txt` - Plain text summary
- `summary_figure.png` - Dashboard visualization

## Configuration

Edit `config/settings.yaml`:

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
  grid_resolution: 0.5
  time_bucket: "6h"
  rolling_window: "12h"
  dark_ship_threshold_seconds: 21600

conflict_zones:
  black_sea:
    bbox: [27.0, 40.5, 41.0, 46.8]
  red_sea:
    bbox: [32.0, 12.0, 43.5, 30.0]
  # ... more zones

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

## Requirements

Core (see `requirements.txt`):
- pandas>=2.0.0, numpy>=1.24.0, scipy>=1.11.0
- scikit-learn>=1.3.0, statsmodels>=0.14.0
- matplotlib>=3.7.0, seaborn>=0.12.0
- pyyaml>=6.0, joblib>=1.3.0

Optional:
- torch (deep learning models)
- xgboost, lightgbm (advanced boosting)
- geopandas, shapely (geospatial)
- prophet (time series forecasting)

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific modules
pytest tests/test_cleaner.py -v
pytest tests/test_features.py -v
pytest tests/test_models.py -v
```

## Notebooks

```bash
# Jupyter notebooks available:
jupyter notebook notebooks/
```

| Notebook | Purpose |
|----------|----------|
| 01_EDA.ipynb | Exploratory Data Analysis |
| 02_preprocessing_validation.ipynb | Data validation and quality |
| 03_visualization.ipynb | All visualization modules |
| 04_conflict_correlation.ipynb | Correlation analysis |
| 05_model_development.ipynb | Anomaly and prediction models |

## License

MIT License

## References

- ACLED (Armed Conflict Location & Event Data Project)
- GDELT Project
- MarineTraffic / AISHub
- MarineRegions
- ITU/IMO Vessel Type Codes