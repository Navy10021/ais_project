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
ais_project/
├── CLAUDE.md                    # Architecture specification
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── config/
│   └── settings.yaml         # Global configuration
├── data/
│   ├── raw/                  # Source AIS data
│   ├── processed/           # Cleaned & features
│   └── external/            # ACLED, GDELT, ports
├── src/
│   ├── preprocessing/       # Cleaner, Feature Engineer
│   ├── analysis/            # Correlation Analyzer
│   ├── models/              # Anomaly, Predictor
│   └── visualization/       # EDA, plots
├── notebooks/
│   └── 01_EDA.ipynb
├── tests/
├── scripts/
└── outputs/
    ├── figures/             # Visualization outputs
    ├── tables/              # Analysis results
    ├── models/             # Trained models
    └── reports/            # Final reports
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

## Usage

### Step 1: Data Cleaning

```bash
python -m src.preprocessing.cleaner \
    --input ./data/raw/ais_raw.csv \
    --output ./data/processed/ais_clean.parquet \
    --config ./config/settings.yaml
```

### Step 2: Feature Engineering

```bash
python -m src.preprocessing.feature_engineer \
    --input ./data/processed/ais_clean.parquet \
    --output ./data/processed/ais_features.parquet
```

### Step 3: Visualization

```bash
# Spatial visualization
python -m src.visualization.spatial_viz \
    --input ./data/processed/ais_features.parquet

# Temporal visualization
python -m src.visualization.temporal_viz \
    --input ./data/processed/ais_features.parquet

# Statistical visualization
python -m src.visualization.statistical_viz \
    --input ./data/processed/ais_features.parquet
```

### Step 4: Correlation Analysis

```bash
python -m src.analysis.correlation_analyzer \
    --input ./data/processed/ais_features.parquet
```

### Step 5: Anomaly Detection

```bash
python -m src.models.anomaly_model \
    --input ./data/processed/ais_features.parquet \
    --contamination 0.05
```

### Step 6: Conflict Prediction

```bash
python -m src.models.conflict_predictor \
    --input ./data/processed/ais_features.parquet
```

### Step 7: Generate Report

```bash
python scripts/generate_report.py
```

## Data Schema

### Raw AIS Columns

| Column | Type | Description | Valid Range |
|--------|------|-------------|-------------|
| MMSI | int64 | Maritime Mobile Service Identity | 100000000–999999999 |
| BaseDateTime | datetime | AIS signal timestamp (UTC) | — |
| LAT | float32 | Latitude | −90.0 to 90.0 |
| LON | float32 | Longitude | −180.0 to 180.0 |
| SOG | float32 | Speed Over Ground (knots) | 0.0 to 102.2 |
| COG | float32 | Course Over Ground (degrees) | 0.0 to 359.9 |
| Heading | float32 | True heading (degrees) | 0–359 |
| VesselName | str | Vessel name | — |
| IMO | str | IMO number | — |
| VesselType | float32 | Vessel type code | 0–99 |
| Status | float32 | Navigational status code | 0–15 |
| Length | float32 | Overall length (meters) | 0–500 |
| Width | float32 | Beam width (meters) | 0–100 |
| Draft | float32 | Draft depth (meters) | 0–30 |

## Analysis Methods

### Correlation Analysis
- **Granger Causality**: Does AIS anomaly lead conflict intensity?
- **Difference-in-Differences**: Isolating conflict effect vs control zone
- **Event Study**: Pre/post mean comparison (Mann-Whitney U)
- **Interrupted Time Series**: Level & slope change detection

### Prediction Models
- **Unsupervised**: Isolation Forest, Local Outlier Factor
- **Supervised**: Random Forest, Gradient Boosting, Logistic Regression

## Output

- **Figures**: 300 DPI PNG (EDA, spatial, temporal, statistical)
- **Tables**: CSV (correlation analysis results)
- **Models**: joblib (trained models)
- **Reports**: HTML/TXT (final summary)

## Configuration

Global parameters can be modified in `config/settings.yaml`:

```yaml
preprocessing:
  seed: 42
  min_mmsi: 200000000
  max_mmsi: 799999999
  invalid_sog: 102.3

features:
  grid_resolution: 0.5
  time_bucket: "6h"

models:
  anomaly:
    contamination: 0.05
  prediction:
    test_size: 0.2

visualization:
  dpi: 300
  figsize: [10, 6]
```

## Requirements

Core dependencies (see `requirements.txt`):
- pandas, numpy, scipy
- scikit-learn, statsmodels
- matplotlib, seaborn, plotly
- pyyaml, joblib

Optional:
- torch (for deep learning)
- xgboost, lightgbm (for advanced models)
- prophet (for time series)
- geopandas, shapely (for advanced geo features)

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_cleaner.py -v
```

## License

MIT License

## References

- ACLED (Armed Conflict Location & Event Data Project)
- GDELT Project
- MarineTraffic / AISHub
- MarineRegions