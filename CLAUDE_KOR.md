# CLAUDE.md — AIS Maritime Conflict Intelligence System (MCIS)

## PROJECT OVERVIEW

**Mission**: AIS(자동선박식별시스템) 데이터를 분석하여 전쟁·분쟁과의 상관관계를 도출하고,
분쟁 발생 전 해상 이상 징후를 탐지·예측하는 논문 수준의 분석 시스템 구축.

**Research Question**:
> "전쟁·분쟁 발생 이전/이후, 해상 선박의 행동 패턴(속도, 밀도, 항로, 선종 구성)은
> 통계적으로 유의미한 변화를 보이는가? 이를 통해 분쟁을 사전 예측할 수 있는가?"

**Target Conflicts for Analysis**:
| 분쟁 | 기간 | 핵심 해역 |
|------|------|-----------|
| Russia-Ukraine War | 2022-02-24 ~ | 흑해(Black Sea), 아조프해(Sea of Azov) |
| Red Sea / Houthi Crisis | 2023-11 ~ | 홍해, 아덴만, 바브엘만데브 해협 |
| Taiwan Strait Tensions | 2022-08 (PLA 훈련) | 대만해협 |
| South China Sea Disputes | 상시 | 남중국해, 파라셀·스프래틀리 군도 |

---

## REPOSITORY STRUCTURE

```
ais-conflict-intelligence/
├── CLAUDE.md                        # 이 파일 (아키텍처 명세)
├── README.md
├── requirements.txt
├── config/
│   └── settings.yaml                # 전역 설정 (경로, 파라미터, conflict zones)
├── data/
│   ├── raw/
│   │   └── ais_raw.csv              # 원본 AIS 데이터 (입력 전용, 절대 수정 금지)
│   ├── processed/
│   │   ├── ais_preprocessed.parquet # 전처리 완료 데이터 (Parquet 형식)
│   │   ├── ais_features.parquet     # 피처 엔지니어링 결과
│   │   └── conflict_events.csv      # 분쟁 이벤트 레이블 데이터
│   ├── external/
│   │   ├── acled_events.csv         # ACLED 분쟁 데이터베이스
│   │   ├── gdelt_events.csv         # GDELT 뉴스 이벤트 데이터
│   │   └── world_ports.csv          # 세계 주요 항구 좌표
│   └── geojson/
│       ├── conflict_zones.geojson   # 분쟁 수역 폴리곤
│       └── chokepoints.geojson      # 전략 해협 폴리곤
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── cleaner.py               # Step 1: 데이터 정제
│   │   ├── validator.py             # Step 2: 유효성 검증
│   │   ├── feature_engineer.py      # Step 3: 피처 생성
│   │   └── anomaly_detector.py      # Step 4: 전처리 단계 이상치 제거
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── spatial_viz.py           # 지리적 시각화 (folium, plotly)
│   │   ├── temporal_viz.py          # 시계열 시각화
│   │   ├── statistical_viz.py       # 통계 분포 시각화
│   │   └── conflict_overlay.py      # 분쟁 이벤트 오버레이
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── traffic_analyzer.py      # 해상 교통량 분석
│   │   ├── behavioral_analyzer.py   # 선박 행동 분석
│   │   ├── network_analyzer.py      # 항로 네트워크 분석
│   │   └── correlation_analyzer.py  # 분쟁 상관관계 분석
│   └── models/
│       ├── __init__.py
│       ├── baseline.py              # 통계적 베이스라인 (ARIMA, Prophet)
│       ├── anomaly_model.py         # 이상 탐지 (Isolation Forest, VAE)
│       ├── conflict_predictor.py    # 분쟁 예측 (LSTM, Transformer)
│       └── evaluator.py             # 모델 평가
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_preprocessing_validation.ipynb
│   ├── 03_visualization.ipynb
│   ├── 04_conflict_correlation.ipynb
│   └── 05_model_development.ipynb
├── outputs/
│   ├── figures/                     # 논문용 고해상도 figure
│   ├── tables/                      # 통계 결과 테이블
│   ├── models/                      # 저장된 모델 weights
│   └── reports/                     # 최종 분석 보고서
├── tests/
│   ├── test_cleaner.py
│   ├── test_features.py
│   └── test_models.py
└── scripts/
    ├── run_pipeline.sh              # 전체 파이프라인 실행
    └── generate_report.py           # 논문용 결과 생성
```

---

## DATA SCHEMA

### 원본 AIS 데이터 컬럼 명세 (`ais_raw.csv`)

| Column | Type | Description | Valid Range | Notes |
|--------|------|-------------|-------------|-------|
| `MMSI` | int64 | 해상이동업무식별번호 (9자리) | 100000000–999999999 | 선박 고유 식별자 |
| `BaseDateTime` | datetime | AIS 신호 수신 UTC 타임스탬프 | - | ISO 8601 형식 |
| `LAT` | float64 | 위도 | -90.0 ~ 90.0 | 91.0 = 미유효 |
| `LON` | float64 | 경도 | -180.0 ~ 180.0 | 181.0 = 미유효 |
| `SOG` | float64 | 대지속력 (Speed Over Ground, knots) | 0.0 ~ 102.2 | 102.3 = 미유효 |
| `COG` | float64 | 대지침로 (Course Over Ground, degrees) | 0.0 ~ 359.9 | 360.0 = 미유효 |
| `Heading` | float64 | 선수방위 (True Heading, degrees) | 0 ~ 359 | 511 = 미유효 |
| `VesselName` | str | 선박명 | - | 결측 허용 |
| `IMO` | str | IMO 번호 (IMO + 7자리) | - | 결측 허용 |
| `CallSign` | str | 무선국 호출부호 | - | 결측 허용 |
| `VesselType` | float64 | 선종 코드 (ITU/IMO 기준) | 0–99 | 아래 매핑 참조 |
| `Status` | float64 | 항법 상태 코드 | 0–15 | 0=항행, 1=정박, 5=계류 등 |
| `Length` | float64 | 선박 전장 (meters) | 0–500 | - |
| `Width` | float64 | 선박 폭 (meters) | 0–100 | - |
| `Draft` | float64 | 흘수 (meters) | 0–30 | - |
| `Cargo` | float64 | 화물 유형 코드 | - | VesselType과 연동 |
| `TransceiverClass` | str | AIS 송수신기 등급 | A, B | A=SOLAS 의무, B=소형선 |

### VesselType 주요 코드 매핑
```python
VESSEL_TYPE_MAP = {
    0:  "Not Available",
    20: "WIG (Ground Effect)",
    21: "WIG - Hazardous A",
    30: "Fishing",
    31: "Towing",
    32: "Towing (Large)",
    33: "Dredging",
    34: "Diving Ops",
    35: "Military Ops",       # ← 군사 작전 선박 (핵심)
    36: "Sailing",
    37: "Pleasure Craft",
    50: "Pilot Vessel",
    51: "SAR Vessel",         # ← 수색구조 (전시 급증 지표)
    52: "Tug",
    55: "Law Enforcement",    # ← 법집행 선박
    57: "Spare",
    60: "Passenger",
    69: "Passenger (Other)",
    70: "Cargo",              # ← 상업 화물 (주력 분석 대상)
    79: "Cargo (Other)",
    80: "Tanker",             # ← 유조선 (에너지 안보 지표)
    89: "Tanker (Other)",
    90: "Other",
}

# Navigation Status Codes
NAV_STATUS_MAP = {
    0:  "Under Way Using Engine",
    1:  "At Anchor",
    2:  "Not Under Command",    # ← 통제불능 (위험 상황)
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

## PHASE 1: ADVANCED PREPROCESSING (`src/preprocessing/`)

### 실행 순서
```
raw_data → [1. Cleaner] → [2. Validator] → [3. Feature Engineer] → [4. Anomaly Detector] → preprocessed_data
```

### 1-1. `cleaner.py` — 데이터 정제

```python
"""
AIS Raw Data Cleaner
====================
MMSI/좌표/속도 등 물리적 유효성 필터링 + 결측치 처리 전략
"""
import pandas as pd
import numpy as np
from pathlib import Path

class AISCleaner:
    """
    AIS 원본 데이터 정제 클래스.

    처리 순서:
      1. MMSI 유효성 검증 (9자리 양수, 특수 MMSI 제외)
      2. 좌표 유효성 검증 (범위 + AIS 무효값 91/181 필터)
      3. SOG/COG/Heading 무효값 처리 (102.3/360/511)
      4. 타임스탬프 파싱 및 정렬
      5. 결측치 전략별 처리
      6. 중복 레코드 제거
    """

    # AIS 표준 무효값
    INVALID_LAT = 91.0
    INVALID_LON = 181.0
    INVALID_SOG = 102.3
    INVALID_COG = 360.0
    INVALID_HEADING = 511
    INVALID_IMO_PREFIX = "IMO0000000"

    # 특수 목적 MMSI 범위 (제외 또는 별도 플래그)
    SPECIAL_MMSI = {
        "coastal_station": (0, 99999999),
        "group_ship": (970000000, 979999999),
        "sar_aircraft": (111000000, 111999999),
        "mob_device": (972000000, 972999999),
        "aton": (990000000, 999999999),  # Aids to Navigation
    }

    def __init__(self, input_path: str, output_path: str):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.cleaning_report = {}

    def load(self) -> pd.DataFrame:
        """Parquet 우선, CSV fallback 로드."""
        if self.input_path.suffix == ".parquet":
            return pd.read_parquet(self.input_path)
        dtype_map = {
            "MMSI": "int64", "LAT": "float32", "LON": "float32",
            "SOG": "float32", "COG": "float32", "Heading": "float32",
            "VesselType": "float32", "Status": "float32",
            "Length": "float32", "Width": "float32",
            "Draft": "float32", "Cargo": "float32",
            "TransceiverClass": "category",
        }
        df = pd.read_csv(self.input_path, dtype=dtype_map,
                         parse_dates=["BaseDateTime"], low_memory=False)
        return df

    def clean_mmsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """MMSI 유효성: 9자리, 양수, 일반 선박 범위."""
        n_before = len(df)
        # 기본 범위 필터
        mask = (df["MMSI"] >= 200000000) & (df["MMSI"] <= 799999999)
        # 특수 MMSI는 별도 플래그로 보존
        for name, (lo, hi) in self.SPECIAL_MMSI.items():
            special = (df["MMSI"] >= lo) & (df["MMSI"] <= hi)
            df.loc[special, "mmsi_special_type"] = name
        df = df[mask].copy()
        self.cleaning_report["mmsi_removed"] = n_before - len(df)
        return df

    def clean_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """좌표 유효성 검증."""
        n_before = len(df)
        df = df[
            (df["LAT"].between(-90.0, 90.0)) &
            (df["LON"].between(-180.0, 180.0)) &
            (df["LAT"] != self.INVALID_LAT) &
            (df["LON"] != self.INVALID_LON) &
            (~df["LAT"].isna()) & (~df["LON"].isna())
        ].copy()
        self.cleaning_report["coord_removed"] = n_before - len(df)
        return df

    def clean_kinematics(self, df: pd.DataFrame) -> pd.DataFrame:
        """SOG/COG/Heading 무효값 → NaN 처리."""
        df["SOG"] = df["SOG"].where(df["SOG"] < self.INVALID_SOG, np.nan)
        df["COG"] = df["COG"].where(df["COG"] < self.INVALID_COG, np.nan)
        df["Heading"] = df["Heading"].where(
            df["Heading"] < self.INVALID_HEADING, np.nan
        )
        # 물리적 상한 필터 (SOG: 상업선 최대 ~30kn, 군함 ~60kn)
        df["sog_anomaly_flag"] = df["SOG"] > 50.0
        return df

    def clean_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """타임스탬프 파싱, UTC 정규화, 미래/과거 이상값 제거."""
        df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"], utc=True, errors="coerce")
        # 미래 타임스탬프 제거
        now_utc = pd.Timestamp.now(tz="UTC")
        df = df[df["BaseDateTime"] <= now_utc].copy()
        # 과도하게 오래된 타임스탬프 (2010년 이전) 제거
        cutoff = pd.Timestamp("2010-01-01", tz="UTC")
        df = df[df["BaseDateTime"] >= cutoff].copy()
        df = df.sort_values(["MMSI", "BaseDateTime"]).reset_index(drop=True)
        return df

    def clean_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """동일 MMSI + 타임스탬프 중복 제거."""
        n_before = len(df)
        df = df.drop_duplicates(subset=["MMSI", "BaseDateTime"], keep="first")
        self.cleaning_report["duplicates_removed"] = n_before - len(df)
        return df

    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        결측치 처리 전략:
        - VesselName/IMO/CallSign: 빈 문자열 → "UNKNOWN"
        - VesselType: 최빈값 보간 (MMSI별)
        - Length/Width/Draft: MMSI별 중앙값 보간, 없으면 VesselType별 중앙값
        - Status: 0 (항행 중) 기본값 부여
        """
        str_cols = ["VesselName", "IMO", "CallSign"]
        for col in str_cols:
            df[col] = df[col].fillna("UNKNOWN").replace("", "UNKNOWN")

        # IMO 무효값 정제
        df.loc[df["IMO"] == self.INVALID_IMO_PREFIX, "IMO"] = "UNKNOWN"

        # VesselType: MMSI별 최빈값
        vessel_type_mode = df.groupby("MMSI")["VesselType"].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 0)
        )
        df["VesselType"] = df["VesselType"].fillna(vessel_type_mode)

        # 물리적 특성: MMSI별 중앙값
        for col in ["Length", "Width", "Draft"]:
            df[col] = df.groupby("MMSI")[col].transform(
                lambda x: x.fillna(x.median())
            )
            # 여전히 결측이면 VesselType별 중앙값
            type_median = df.groupby("VesselType")[col].transform("median")
            df[col] = df[col].fillna(type_median)

        df["Status"] = df["Status"].fillna(0)
        return df

    def run(self) -> pd.DataFrame:
        """전체 정제 파이프라인 실행."""
        print("[CLEANER] Loading raw data...")
        df = self.load()
        print(f"[CLEANER] Raw records: {len(df):,}")

        df = self.clean_mmsi(df)
        df = self.clean_coordinates(df)
        df = self.clean_kinematics(df)
        df = self.clean_timestamps(df)
        df = self.clean_duplicates(df)
        df = self.handle_missing(df)

        print(f"[CLEANER] Clean records: {len(df):,}")
        print(f"[CLEANER] Report: {self.cleaning_report}")
        df.to_parquet(self.output_path, index=False, compression="snappy")
        return df
```

### 1-2. `feature_engineer.py` — 피처 엔지니어링

```python
"""
AIS Feature Engineer
====================
분쟁 탐지 및 예측에 필요한 고급 피처 생성.

생성 피처 카테고리:
  A. 운동학적 피처 (Kinematic Features)
  B. 지리적 피처 (Geospatial Features)
  C. 행동 패턴 피처 (Behavioral Features)
  D. 네트워크 피처 (Network Features)
  E. 시계열 집계 피처 (Temporal Aggregation Features)
  F. 분쟁 레이블 (Conflict Labels)
"""
import pandas as pd
import numpy as np
from scipy import stats
from shapely.geometry import Point
import geopandas as gpd

class AISFeatureEngineer:

    # 전략적 해협 / 분쟁 수역 정의
    CONFLICT_ZONES = {
        "black_sea":      {"bbox": [27.0, 40.5, 41.0, 46.8], "conflict": "ukraine_war"},
        "azov_sea":       {"bbox": [33.5, 45.0, 39.5, 47.5], "conflict": "ukraine_war"},
        "red_sea":        {"bbox": [32.0, 12.0, 43.5, 30.0], "conflict": "houthi_crisis"},
        "bab_el_mandeb":  {"bbox": [43.0, 11.5, 45.0, 12.5], "conflict": "houthi_crisis"},
        "taiwan_strait":  {"bbox": [119.0, 22.0, 122.0, 26.0], "conflict": "taiwan_tension"},
        "south_china_sea":{"bbox": [109.0, 3.0, 121.0, 22.0], "conflict": "scs_dispute"},
        "strait_hormuz":  {"bbox": [56.0, 25.5, 59.5, 27.0], "conflict": "iran_tension"},
        "kerch_strait":   {"bbox": [36.4, 45.1, 36.8, 45.5], "conflict": "ukraine_war"},
    }

    def add_kinematic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        A. 운동학적 피처
        - speed_category: 정박/저속/순항/고속 분류
        - heading_sog_consistency: 침로-속도 일관성 (급격한 변화 = 이상 기동)
        - delta_sog: 전 포인트 대비 속도 변화량
        - delta_cog: 전 포인트 대비 침로 변화량 (급선회 탐지)
        - turning_rate: 초당 침로 변화율
        - is_dark_ship: AIS 신호 공백 구간 플래그
        """
        df = df.sort_values(["MMSI", "BaseDateTime"])

        # 속도 범주화
        df["speed_category"] = pd.cut(
            df["SOG"],
            bins=[-0.1, 0.5, 3.0, 8.0, 15.0, 50.0],
            labels=["anchored", "drifting", "slow", "cruising", "fast"]
        )

        # MMSI별 전처리 (diff 연산)
        grp = df.groupby("MMSI")
        df["delta_sog"] = grp["SOG"].diff().abs()
        df["delta_cog"] = grp["COG"].diff().abs()
        # 침로는 360° wrap-around 처리
        df["delta_cog"] = df["delta_cog"].apply(
            lambda x: min(x, 360 - x) if pd.notna(x) else np.nan
        )
        df["time_diff_sec"] = grp["BaseDateTime"].diff().dt.total_seconds()
        df["turning_rate"] = df["delta_cog"] / df["time_diff_sec"].replace(0, np.nan)

        # AIS 암흑선박 탐지 (신호 공백 > 6시간)
        df["is_dark_ship"] = df["time_diff_sec"] > 21600

        # SOG=0이지만 Heading≠511인 경우: 계류 vs 표류 구분
        df["moored_vs_drifting"] = (
            (df["SOG"] < 0.3) & (df["Heading"] != 511)
        ).astype(int)

        return df

    def add_geospatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        B. 지리적 피처
        - in_conflict_zone: 분쟁 수역 진입 여부
        - conflict_zone_name: 분쟁 수역명
        - distance_to_chokepoint: 주요 해협까지 거리 (km)
        - is_in_eez: EEZ 내 위치 여부
        - grid_cell: 0.5° × 0.5° 격자 셀 ID (공간 집계용)
        """
        # 격자 셀 (공간 해상도 0.5도)
        df["grid_lat"] = (df["LAT"] // 0.5) * 0.5
        df["grid_lon"] = (df["LON"] // 0.5) * 0.5
        df["grid_cell"] = df["grid_lat"].astype(str) + "_" + df["grid_lon"].astype(str)

        # 분쟁 수역 진입 검사
        df["in_conflict_zone"] = False
        df["conflict_zone_name"] = "none"
        for zone_name, zone_info in self.CONFLICT_ZONES.items():
            bbox = zone_info["bbox"]  # [lon_min, lat_min, lon_max, lat_max]
            mask = (
                (df["LON"] >= bbox[0]) & (df["LON"] <= bbox[2]) &
                (df["LAT"] >= bbox[1]) & (df["LAT"] <= bbox[3])
            )
            df.loc[mask, "in_conflict_zone"] = True
            df.loc[mask, "conflict_zone_name"] = zone_name

        # Haversine 거리 계산 유틸 (주요 해협까지 최단 거리)
        CHOKEPOINTS = {
            "strait_of_hormuz": (56.5, 26.5),
            "strait_of_malacca": (103.8, 1.2),
            "bab_el_mandeb": (43.4, 12.5),
            "suez_canal": (32.5, 30.7),
            "panama_canal": (-79.9, 9.0),
        }
        for cp_name, (cp_lon, cp_lat) in CHOKEPOINTS.items():
            df[f"dist_{cp_name}_km"] = self._haversine_vectorized(
                df["LAT"], df["LON"], cp_lat, cp_lon
            )

        return df

    def add_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        C. 행동 패턴 피처 (MMSI별 롤링 윈도우 통계)
        - rolling_sog_mean/std: 12h 롤링 평균/표준편차 속도
        - route_entropy: 침로 엔트로피 (경로 예측 불가능성)
        - loitering_score: 배회 점수 (저속 + 반복 구역)
        - zig_zag_index: 지그재그 항행 지수 (잠수함 회피 기동 등)
        """
        df = df.sort_values(["MMSI", "BaseDateTime"])

        # 12시간 롤링 윈도우 (시간 기반)
        df = df.set_index("BaseDateTime")
        rolling = df.groupby("MMSI")["SOG"].rolling("12H", min_periods=3)
        df["rolling_sog_mean"] = rolling.mean().reset_index(level=0, drop=True)
        df["rolling_sog_std"]  = rolling.std().reset_index(level=0, drop=True)
        df = df.reset_index()

        # 침로 엔트로피 (24시간 윈도우, 36방위 구간)
        def cog_entropy(series):
            bins = pd.cut(series, bins=36, labels=False)
            counts = bins.value_counts(normalize=True) + 1e-10
            return stats.entropy(counts)

        cog_entropy_map = (
            df.groupby(["MMSI", df["BaseDateTime"].dt.date])["COG"]
            .apply(cog_entropy)
            .reset_index()
            .rename(columns={"COG": "route_entropy"})
        )
        df["date"] = df["BaseDateTime"].dt.date
        df = df.merge(cog_entropy_map, on=["MMSI", "date"], how="left")

        # Loitering 점수: 저속(<3kn) + 공간 변화 미미한 구간
        df["loitering_flag"] = (
            (df["SOG"] < 3.0) &
            (df["delta_cog"] > 45) &  # 방향 자주 바꿈
            (df["in_conflict_zone"])   # 분쟁 수역 내
        ).astype(int)

        # Zig-zag 지수: delta_cog의 부호 변화 빈도
        df["cog_sign"] = np.sign(df["delta_cog"].fillna(0))
        df["zig_zag_index"] = (
            df.groupby("MMSI")["cog_sign"]
            .transform(lambda x: (x != x.shift()).rolling(10, min_periods=3).sum())
        )

        return df

    def add_temporal_aggregation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        D. 시공간 집계 피처 (분쟁 탐지 핵심 지표)
        - grid_traffic_density: 격자별 시간당 선박 수
        - vessel_type_ratio: 격자별 군사/화물/유조선 비율
        - dark_ship_ratio: 격자별 AIS 공백 비율
        - new_vessel_ratio: 격자에 처음 등장하는 선박 비율
        """
        # 시간 버킷 (6시간 단위)
        df["time_bucket"] = df["BaseDateTime"].dt.floor("6H")

        # 격자+시간버킷별 집계
        agg_df = df.groupby(["grid_cell", "time_bucket"]).agg(
            traffic_count=("MMSI", "nunique"),
            dark_ship_count=("is_dark_ship", "sum"),
            military_count=("VesselType", lambda x: (x == 35).sum()),
            cargo_count=("VesselType", lambda x: x.isin([70,71,72,73,74,75,76,77,78,79]).sum()),
            tanker_count=("VesselType", lambda x: x.isin([80,81,82,83,84,85,86,87,88,89]).sum()),
            sar_count=("VesselType", lambda x: (x == 51).sum()),
            mean_sog=("SOG", "mean"),
            std_sog=("SOG", "std"),
            loitering_sum=("loitering_flag", "sum"),
        ).reset_index()

        agg_df["dark_ship_ratio"] = (
            agg_df["dark_ship_count"] / agg_df["traffic_count"].clip(lower=1)
        )
        agg_df["military_ratio"] = (
            agg_df["military_count"] / agg_df["traffic_count"].clip(lower=1)
        )
        agg_df["tanker_ratio"] = (
            agg_df["tanker_count"] / agg_df["traffic_count"].clip(lower=1)
        )

        return df, agg_df

    def add_conflict_labels(self, df: pd.DataFrame,
                            conflict_events_path: str) -> pd.DataFrame:
        """
        E. 분쟁 레이블 생성
        - conflict_label: 분쟁 발생 여부 (이진)
        - days_to_conflict: 분쟁 발생까지 남은 일수 (회귀 타겟)
        - conflict_intensity: 분쟁 강도 (ACLED 이벤트 수 기반)
        """
        events = pd.read_csv(conflict_events_path, parse_dates=["event_date"])

        # 각 AIS 레코드에 가장 가까운 분쟁 이벤트 매핑
        # 공간: 분쟁 수역 내, 시간: ±30일 윈도우
        df["conflict_label"] = 0
        df["days_to_conflict"] = np.nan
        df["conflict_intensity"] = 0.0

        for _, event in events.iterrows():
            zone_mask = df["conflict_zone_name"] == event.get("zone", "")
            time_diff = (event["event_date"] - df["BaseDateTime"].dt.normalize())
            within_window = time_diff.dt.days.between(-7, 30)  # -7일 ~ +30일
            match_mask = zone_mask & within_window

            df.loc[match_mask, "conflict_label"] = 1
            df.loc[match_mask, "days_to_conflict"] = (
                time_diff[match_mask].dt.days
            )
            df.loc[match_mask, "conflict_intensity"] = event.get("fatalities", 0)

        return df

    @staticmethod
    def _haversine_vectorized(lat1, lon1, lat2, lon2):
        """벡터화 Haversine 거리 계산 (km)."""
        R = 6371.0
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
        return R * 2 * np.arcsin(np.sqrt(a))

    def run(self, df: pd.DataFrame, conflict_events_path: str) -> pd.DataFrame:
        print("[FEATURE] Adding kinematic features...")
        df = self.add_kinematic_features(df)
        print("[FEATURE] Adding geospatial features...")
        df = self.add_geospatial_features(df)
        print("[FEATURE] Adding behavioral features...")
        df = self.add_behavioral_features(df)
        df, self.agg_df = self.add_temporal_aggregation(df)
        print("[FEATURE] Adding conflict labels...")
        df = self.add_conflict_labels(df, conflict_events_path)
        return df
```

---

## PHASE 2: VISUALIZATION (`src/visualization/`)

### 시각화 모듈 목록

```python
# spatial_viz.py — 지리적 시각화
class SpatialVisualizer:
    """
    생성 시각화:
    1. 전지구 선박 밀도 히트맵 (Folium + HeatMap plugin)
       - 시간별 애니메이션 (HeatMapWithTime)
       - 분쟁 수역 폴리곤 오버레이
    2. 개별 선박 항적 시각화 (특정 MMSI 추적)
       - 속도 컬러맵 (SOG → 색상 그라디언트)
    3. 분쟁 수역 진출입 이벤트 마커
    4. AIS 암흑선박 클러스터 시각화

    출력: outputs/figures/spatial/
    """
    def plot_global_density_heatmap(self): ...
    def plot_vessel_trajectory(self, mmsi: int): ...
    def plot_dark_ship_clusters(self): ...
    def animate_traffic_flow(self, time_range: tuple): ...

# temporal_viz.py — 시계열 시각화
class TemporalVisualizer:
    """
    생성 시각화:
    1. 분쟁 수역별 월간 교통량 추이 (선 그래프)
       - 분쟁 발발일 수직선 표시 (plt.axvline)
       - 전/후 95% 신뢰구간 음영
    2. 선종별 구성비 변화 (스택 영역 차트)
       - 군사선/SAR/법집행선 급증 탐지
    3. 속도 분포 변화 (바이올린 플롯 시계열)
    4. AIS 암흑선박 비율 시계열
    5. 상호상관함수 (CCF): AIS 지표 vs 분쟁 강도

    출력: outputs/figures/temporal/
    """
    def plot_traffic_vs_conflict_timeline(self): ...
    def plot_vessel_type_composition_change(self): ...
    def plot_speed_distribution_change(self): ...
    def plot_ccf_analysis(self, lead_lag_days: int = 30): ...

# statistical_viz.py — 통계 시각화
class StatisticalVisualizer:
    """
    생성 시각화:
    1. 분쟁 전/후 피처 분포 비교 (KDE + 박스플롯)
    2. 피처 중요도 히트맵 (Random Forest feature importance)
    3. 상관 행렬 (Spearman, 분쟁 지표와 AIS 피처)
    4. ROC-AUC 곡선 (모델별 비교)
    5. Precision-Recall 곡선
    6. SHAP 값 버블 플롯 (모델 해석)

    출력: outputs/figures/statistical/
    """
    def plot_pre_post_conflict_distributions(self): ...
    def plot_correlation_heatmap(self): ...
    def plot_shap_summary(self, model, X): ...
```

---

## PHASE 3: ANALYSIS (`src/analysis/`)

### 3-1. 분쟁 상관관계 분석 (`correlation_analyzer.py`)

```python
"""
Conflict Correlation Analyzer
==============================
AIS 지표와 분쟁 사건 간 통계적 상관관계 분석.

분석 방법론:
  1. Granger Causality Test: AIS 이상 → 분쟁 인과성
  2. Cross-Correlation Analysis (CCF): 리드-래그 관계
  3. Difference-in-Differences (DiD): 분쟁 수역 vs 통제 수역
  4. Event Study Analysis: 분쟁 발발 ±30일 평균 비교
  5. Interrupted Time Series (ITS): 분쟁 전후 추세 단절 분석
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests, ccf
from statsmodels.tsa.api import VAR
from scipy.stats import mannwhitneyu, ks_2samp, ttest_ind

class ConflictCorrelationAnalyzer:

    CONFLICT_EVENTS = {
        "ukraine_war_start":    "2022-02-24",
        "houthi_first_attack":  "2023-11-19",
        "taiwan_pla_exercise":  "2022-08-04",
        "kerch_bridge_attack":  "2022-10-08",
    }

    def granger_causality_test(self, ais_series: pd.Series,
                                conflict_series: pd.Series,
                                max_lag: int = 30) -> dict:
        """
        귀무가설: AIS 이상 지표가 분쟁 강도를 Granger-cause하지 않는다.
        p < 0.05이면 기각 → AIS 선행 예측 가능성 확인.
        """
        df = pd.DataFrame({
            "ais_anomaly": ais_series,
            "conflict_intensity": conflict_series
        }).dropna()

        results = grangercausalitytests(df, maxlag=max_lag, verbose=False)
        significant_lags = {
            lag: test[0]["ssr_ftest"][1]  # p-value
            for lag, test in results.items()
            if test[0]["ssr_ftest"][1] < 0.05
        }
        return {
            "significant_lags_days": significant_lags,
            "min_p_lag": min(significant_lags, key=significant_lags.get)
            if significant_lags else None
        }

    def event_study_analysis(self, df: pd.DataFrame,
                              event_date: str,
                              zone: str,
                              window_days: int = 30) -> pd.DataFrame:
        """
        이벤트 스터디: 분쟁 발발일 기준 ±30일 AIS 지표 평균 비교.
        Abnormal Return 개념 차용 → Abnormal Traffic Volume (ATV).
        """
        event_dt = pd.Timestamp(event_date, tz="UTC")
        zone_df = df[df["conflict_zone_name"] == zone].copy()
        zone_df = zone_df.set_index("BaseDateTime").sort_index()

        pre_window  = zone_df[event_dt - pd.Timedelta(days=window_days):event_dt]
        post_window = zone_df[event_dt:event_dt + pd.Timedelta(days=window_days)]

        metrics = ["traffic_count", "dark_ship_ratio", "military_ratio",
                   "tanker_ratio", "mean_sog", "loitering_sum"]
        results = []
        for metric in metrics:
            pre_mean  = pre_window[metric].mean()
            post_mean = post_window[metric].mean()
            stat, p   = mannwhitneyu(
                pre_window[metric].dropna(),
                post_window[metric].dropna(),
                alternative="two-sided"
            )
            results.append({
                "metric": metric,
                "pre_mean": pre_mean,
                "post_mean": post_mean,
                "pct_change": (post_mean - pre_mean) / (pre_mean + 1e-10) * 100,
                "mannwhitney_p": p,
                "significant": p < 0.05
            })
        return pd.DataFrame(results)

    def difference_in_differences(self, treatment_zone: str,
                                   control_zone: str,
                                   event_date: str) -> dict:
        """
        DiD: 분쟁 수역(treatment) vs 비교 수역(control)
        DiD Estimator = (Post_T - Pre_T) - (Post_C - Pre_C)
        """
        ...

    def interrupted_time_series(self, series: pd.Series,
                                 breakpoint: str) -> dict:
        """
        ITS 회귀: 분쟁 전후 수준(level)과 추세(slope) 변화 추정.
        y = β0 + β1·t + β2·D + β3·t·D + ε
        (D=0: 분쟁전, D=1: 분쟁후)
        """
        ...
```

---

## PHASE 4: MODELS (`src/models/`)

### 4-1. `anomaly_model.py` — 이상 탐지 (비지도 학습)

```python
"""
Maritime Anomaly Detection
==========================
분쟁 전조 이상 행동 탐지 (레이블 없는 데이터에서도 작동).

모델:
  1. Isolation Forest: 다변량 이상치 탐지
  2. Variational Autoencoder (VAE): 정상 패턴 재현 오류 기반
  3. DBSCAN: 공간 밀도 기반 이상 클러스터 탐지
  4. Local Outlier Factor (LOF): 국소 밀도 이상 탐지

입력 피처:
  - SOG, delta_sog, delta_cog, turning_rate
  - rolling_sog_std, route_entropy, zig_zag_index
  - loitering_flag, is_dark_ship
  - dist_*_km (주요 해협까지 거리)
  - traffic_density (격자별)

출력:
  - anomaly_score: 연속값 이상도 점수 (0~1)
  - anomaly_label: 이진 이상 여부
  - anomaly_type: "dark_ship" | "loitering" | "zig_zag" | "density_surge"
"""
```

### 4-2. `conflict_predictor.py` — 분쟁 예측 (지도 학습)

```python
"""
Conflict Prediction Model
=========================
T+N일 후 분쟁 발생 확률 예측.

아키텍처 후보:
  1. LSTM-Attention (시계열 기반)
     - 입력: (batch, seq_len=30days, features=20)
     - 출력: 분쟁 확률 (0~1)
  2. Temporal Fusion Transformer (TFT)
     - 정적 컨텍스트 (구역명, 선종 구성) + 동적 시계열
  3. Random Forest (해석 가능 베이스라인)
  4. XGBoost (그라디언트 부스팅 베이스라인)

학습 설계:
  - 타겟: days_to_conflict ≤ 7이면 임박 위협 (이진)
  - 클래스 불균형: SMOTE + 클래스 가중치 조정
  - 검증: 시간 기반 분할 (train ≤ 2022, val 2023 Q1, test 2023 Q2~)
  - 리드타임 평가: T+3, T+7, T+14, T+30일 AUC 비교

평가 지표:
  - AUROC, AUPRC (불균형 데이터)
  - F1-Score (β=2, recall 우선)
  - Lead Time: 평균 예측 선행 일수
  - False Alarm Rate: 허위 경보율
"""

import torch
import torch.nn as nn

class ConflictLSTM(nn.Module):
    """
    LSTM + Scaled Dot-Product Attention 분쟁 예측 모델.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.attention = nn.MultiheadAttention(
            hidden_dim * 2, num_heads=4, batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)           # (B, T, H*2)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        pooled = attn_out.mean(dim=1)        # (B, H*2) — global avg pool
        return self.classifier(pooled).squeeze(-1)
```

---

## EXTERNAL DATA SOURCES

| 데이터 | 출처 | URL | 용도 |
|--------|------|-----|------|
| 분쟁 이벤트 | ACLED | acleddata.com | 분쟁 레이블 생성 |
| 뉴스 이벤트 | GDELT 2.0 | gdeltproject.org | 분쟁 강도 보조 지표 |
| EEZ 경계 | MarineRegions | marineregions.org | 해상 경계 피처 |
| 세계 항구 | World Port Index | msi.nga.mil | 출발/도착항 추론 |
| 해역 폴리곤 | OpenSeaMap | openseamap.org | 분쟁 수역 정의 |
| 제재 선박 목록 | OFAC SDN | sanctionssearch.ofac.treas.gov | 제재 위반 선박 탐지 |

---

## EXECUTION COMMANDS

```bash
# 환경 설정
conda create -n ais-mcis python=3.11
conda activate ais-mcis
pip install -r requirements.txt

# 전체 파이프라인 순차 실행
python -m src.preprocessing.cleaner \
    --input ./data/raw/ais_raw.csv \
    --output ./data/processed/ais_clean.parquet

python -m src.preprocessing.feature_engineer \
    --input ./data/processed/ais_clean.parquet \
    --conflict-events ./data/external/acled_events.csv \
    --output ./data/processed/ais_features.parquet

python -m src.visualization.spatial_viz \
    --input ./data/processed/ais_features.parquet \
    --output-dir ./outputs/figures/

python -m src.analysis.correlation_analyzer \
    --input ./data/processed/ais_features.parquet \
    --output ./outputs/tables/correlation_results.csv

python -m src.models.conflict_predictor \
    --input ./data/processed/ais_features.parquet \
    --mode train \
    --output ./outputs/models/

# 또는 전체 셸 스크립트
bash scripts/run_pipeline.sh

# 개별 노트북 실행
jupyter nbconvert --to notebook --execute notebooks/01_EDA.ipynb
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
pyarrow>=12.0.0          # Parquet I/O
xgboost>=1.7.0
lightgbm>=4.0.0
shap>=0.42.0             # 모델 해석
imbalanced-learn>=0.11.0 # SMOTE
prophet>=1.1.4           # 베이스라인 시계열
jupyter>=1.0.0
tqdm>=4.65.0
pyyaml>=6.0
```

---

## PAPER STRUCTURE (논문 구성 목표)

```
Title: "Maritime Traffic Anomaly Detection as a Precursor to Armed Conflict:
        Evidence from AIS Data in Global Hotspots (2022–2024)"

Abstract
1. Introduction
   1.1 Research Motivation (AIS + 분쟁 탐지 필요성)
   1.2 Research Questions & Hypotheses
   1.3 Contributions

2. Background & Related Work
   2.1 AIS 데이터 특성 및 한계
   2.2 해양 이상 탐지 선행연구
   2.3 분쟁 조기경보 시스템 선행연구

3. Data & Methodology
   3.1 AIS 데이터 수집 및 전처리
   3.2 분쟁 이벤트 데이터 (ACLED)
   3.3 피처 엔지니어링
   3.4 분석 방법론 (Granger, DiD, ITS)

4. Empirical Results
   4.1 흑해 (러-우 전쟁): 전/후 교통 패턴 변화
   4.2 홍해 (후티 위기): 유조선·화물선 우회 탐지
   4.3 대만해협: PLA 훈련기간 패턴 이상
   4.4 선행 지표 분석 (Granger 검정 결과)

5. Predictive Modeling
   5.1 모델 비교 (LSTM vs TFT vs XGBoost)
   5.2 분쟁 수역별 예측 성능 (AUC, F1)
   5.3 SHAP 분석: 핵심 예측 피처
   5.4 Lead Time 분석

6. Discussion
   6.1 AIS 데이터 한계 (spoofing, dark shipping)
   6.2 정책적 함의 (조기경보 시스템 설계)

7. Conclusion

References
Appendix: 보완 시각화 및 통계 테이블
```

---

## CLAUDE CODE AGENTIC EXECUTION PLAN

### Task 순서 (의존성 기반)

```
[Task 1] src/preprocessing/cleaner.py 구현 + 단위 테스트
    → output: data/processed/ais_clean.parquet

[Task 2] src/preprocessing/feature_engineer.py 구현
    → input: ais_clean.parquet + acled_events.csv
    → output: data/processed/ais_features.parquet

[Task 3] notebooks/01_EDA.ipynb 실행 (데이터 탐색)
    → output: outputs/figures/eda/

[Task 4] src/visualization/ 전체 구현
    → output: outputs/figures/ (논문용 300dpi PNG)

[Task 5] src/analysis/correlation_analyzer.py 구현
    → Granger, DiD, ITS, Event Study 분석
    → output: outputs/tables/

[Task 6] src/models/anomaly_model.py 구현 + 학습
    → Isolation Forest + VAE 앙상블

[Task 7] src/models/conflict_predictor.py 구현 + 학습
    → LSTM-Attention + XGBoost 비교

[Task 8] 논문용 결과물 생성
    → scripts/generate_report.py
    → outputs/reports/final_report.pdf
```

### 각 Task 실행 시 Claude에게 전달할 컨텍스트
- 이 CLAUDE.md 전체를 항상 참조
- 데이터 경로: `./data/ais_raw.csv` (실제 대용량 파일)
- 중간 결과는 Parquet으로 저장 (메모리 효율)
- 시각화는 `matplotlib.rcParams['figure.dpi'] = 300` 고정
- 모든 통계 결과는 p-value와 함께 출력

---

## CODING CONVENTIONS

```python
# 1. 타입 힌트 필수
def process(df: pd.DataFrame, config: dict) -> pd.DataFrame: ...

# 2. 로깅 (print 대신)
import logging
logger = logging.getLogger(__name__)
logger.info(f"[CLEANER] Records after MMSI filter: {len(df):,}")

# 3. 설정 파일 분리 (하드코딩 금지)
# config/settings.yaml에서 읽어옴

# 4. Parquet 저장 (CSV 금지, 대용량 데이터)
df.to_parquet(path, index=False, compression="snappy")

# 5. 재현성
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# 6. 논문용 시각화 설정
import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.dpi": 300,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
```
