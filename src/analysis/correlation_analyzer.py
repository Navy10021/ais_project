"""
Conflict Correlation Analyzer
==============================
Statistical framework for establishing AIS → conflict relationships.

Methods:
  - Granger Causality     — does AIS anomaly lead conflict intensity?
  - Cross-Correlation     — optimal lead-lag window
  - Difference-in-Diffs  — isolating conflict effect
  - Event Study Analysis — pre/post comparison
  - Interrupted Time Series — level & slope change

Coding Conventions:
  - Type hints
  - Logging via logger
  - Reproducibility with SEED
"""
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr
import statsmodels.api as sm
from pathlib import Path
import logging
import argparse
import yaml
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)


@dataclass
class AnalysisConfig:
    """Configuration for analysis modules"""
    max_lag_granger: int = 30
    event_study_window: int = 30
    did_alpha: float = 0.05


def load_config(config_path: str = "./config/settings.yaml") -> AnalysisConfig:
    """Load analysis configuration from YAML"""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        analysis = config.get("analysis", {})
        return AnalysisConfig(
            max_lag_granger=analysis.get("max_lag_granger", 30),
            event_study_window=analysis.get("event_study_window", 30),
            did_alpha=analysis.get("did_alpha", 0.05),
        )
    except Exception:
        logger.warning(
            f"Could not load config from {config_path}, using defaults"
        )
        return AnalysisConfig()


class ConflictCorrelationAnalyzer:
    CONFLICT_ONSET: Dict[str, str] = {
        "ukraine_war": "2022-02-24",
        "houthi_crisis": "2023-11-19",
        "pla_taiwan_drill": "2022-08-04",
        "kerch_bridge": "2022-10-08",
        "black_sea": "2022-02-24",
        "red_sea": "2023-11-19",
        "taiwan_strait": "2022-08-04",
    }

    CONFLICT_ZONE_ALIASES: Dict[str, str] = {
        "ukraine_war": "black_sea",
        "houthi_crisis": "red_sea",
        "taiwan_tension": "taiwan_strait",
        "iran_tension": "strait_hormuz",
    }

    def __init__(
        self,
        output_dir: str = "./outputs/tables",
        config: Optional[AnalysisConfig] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or AnalysisConfig()

    def granger_causality(
        self,
        ais_series: pd.Series,
        conflict_series: pd.Series,
        max_lag: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Test if AIS anomaly Granger-causes conflict intensity"""
        from statsmodels.tsa.stattools import grangercausalitytests

        max_lag = max_lag or self.config.max_lag_granger

        df = pd.DataFrame({"ais": ais_series, "conflict": conflict_series}).dropna()
        if len(df) < max_lag * 2:
            logger.warning("Insufficient data for Granger causality test")
            return {"significant_lags": {}, "optimal_lead_days": None}

        try:
            max_lag = min(max_lag, len(df) // 4)
            if max_lag < 2:
                return {"significant_lags": {}, "optimal_lead_days": None}

            raw = grangercausalitytests(
                df[["conflict", "ais"]],
                maxlag=max_lag,
                verbose=False,
            )

            sig = {}
            for lag, res in raw.items():
                p_value = res[0]["ssr_ftest"][1]
                if p_value < 0.05:
                    sig[lag] = p_value

            return {
                "significant_lags": sig,
                "optimal_lead_days": min(sig, key=sig.get) if sig else None,
                "tested_lags": max_lag,
            }
        except Exception as e:
            logger.warning(f"Granger test failed: {e}")
            return {"significant_lags": {}, "optimal_lead_days": None}

    def cross_correlation(
        self,
        ais_series: pd.Series,
        conflict_series: pd.Series,
        max_lag: int = 30,
    ) -> Dict[str, Any]:
        """Compute cross-correlation at various lags"""
        ais = ais_series.dropna()
        conflict = conflict_series.reindex(ais.index).dropna()
        ais = ais.reindex(conflict.index)

        if len(ais) < max_lag:
            return {"optimal_lag": 0, "max_corr": np.nan}

        correlations = []
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                a = ais.iloc[:lag]
                c = conflict.iloc[-lag:]
            elif lag > 0:
                a = ais.iloc[lag:]
                c = conflict.iloc[:-lag]
            else:
                a = ais
                c = conflict

            if len(a) > 10:
                corr, _ = spearmanr(a, c)
                correlations.append((lag, corr))

        if not correlations:
            return {"optimal_lag": 0, "max_corr": np.nan}

        best = max(correlations, key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0)

        return {
            "optimal_lag": best[0],
            "max_corr": best[1],
            "all_correlations": dict(correlations),
        }

    def event_study(
        self,
        df: pd.DataFrame,
        event_date: str,
        zone: str,
        window_days: Optional[int] = None,
    ) -> pd.DataFrame:
        """Event study analysis - pre/post conflict comparison"""
        window_days = window_days or self.config.event_study_window

        ev = pd.Timestamp(event_date, tz="UTC")

        zone_df = df[df["conflict_zone_name"] == zone].copy()
        if len(zone_df) == 0:
            logger.warning(f"No data for zone: {zone}")
            return pd.DataFrame()

        zone_df = zone_df.set_index("BaseDateTime").sort_index()

        valid_metrics = [
            "SOG",
            "delta_sog",
            "delta_cog",
            "time_diff_sec",
            "is_dark_ship",
            "loitering_flag",
            "traffic_count",
            "mean_sog",
        ]
        metrics = [m for m in valid_metrics if m in zone_df.columns]

        pre_start = ev - pd.Timedelta(days=window_days)
        pre_end = ev
        post_start = ev
        post_end = ev + pd.Timedelta(days=window_days)

        pre = zone_df[pre_start:pre_end]
        post = zone_df[post_start:post_end]

        rows = []
        for m in metrics:
            if m not in pre.columns or m not in post.columns:
                continue

            pv = pre[m].dropna()
            qv = post[m].dropna()

            if len(pv) < 5 or len(qv) < 5:
                continue

            try:
                stat, p = mannwhitneyu(pv, qv, alternative="two-sided")
            except:
                p = np.nan

            pre_mean = pv.mean()
            post_mean = qv.mean()
            pct_change = (
                (post_mean - pre_mean) / abs(pre_mean) * 100
                if pre_mean != 0 and not np.isnan(pre_mean)
                else np.nan
            )

            rows.append({
                "metric": m,
                "pre_mean": pre_mean,
                "post_mean": post_mean,
                "pct_change": pct_change,
                "p_value": p,
                "significant": p < 0.05 if not np.isnan(p) else False,
                "n_pre": len(pv),
                "n_post": len(qv),
            })

        return pd.DataFrame(rows)

    def difference_in_differences(
        self,
        df: pd.DataFrame,
        treatment_zone: str,
        control_zone: str,
        event_date: str,
        metric: str = "SOG",
    ) -> Dict[str, Any]:
        """DiD estimator for treatment vs control zones"""
        ev = pd.Timestamp(event_date, tz="UTC")
        res = {}

        for zone in [treatment_zone, control_zone]:
            zone_df = df[df["conflict_zone_name"] == zone].copy()
            if len(zone_df) == 0:
                continue

            zone_df = zone_df.set_index("BaseDateTime").sort_index()

            if metric not in zone_df.columns:
                continue

            pre_mean = zone_df.loc[:ev, metric].mean()
            post_mean = zone_df.loc[ev:, metric].mean()
            res[zone] = {"pre": pre_mean, "post": post_mean}

        if treatment_zone not in res or control_zone not in res:
            logger.warning("Insufficient zones for DiD")
            return {"did_estimator": np.nan, "zones": res}

        t, c = res[treatment_zone], res[control_zone]
        did = (t["post"] - t["pre"]) - (c["post"] - c["pre"])

        return {
            "did_estimator": did,
            "treatment_diff": t["post"] - t["pre"],
            "control_diff": c["post"] - c["pre"],
            "zones": res,
        }

    def interrupted_time_series(
        self,
        series: pd.Series,
        breakpoint: str,
    ) -> Dict[str, float]:
        """Interrupted time series - segmented regression"""
        series = series.dropna().sort_index()
        if len(series) < 20:
            logger.warning("Insufficient data for ITS")
            return {
                "level_shift": np.nan,
                "slope_change": np.nan,
                "level_p": np.nan,
                "slope_p": np.nan,
                "r_squared": np.nan,
            }

        try:
            t = np.arange(len(series))
            bp = series.index.searchsorted(pd.Timestamp(breakpoint, tz="UTC"))
            bp = max(5, min(bp, len(series) - 5))

            D = (t >= bp).astype(int)
            X = sm.add_constant(np.column_stack([t, D, t * D]))
            res = sm.OLS(series.values, X).fit()

            return {
                "level_shift": res.params[2],
                "slope_change": res.params[3],
                "level_p": res.pvalues[2],
                "slope_p": res.pvalues[3],
                "r_squared": res.rsquared,
            }
        except Exception as e:
            logger.warning(f"ITS failed: {e}")
            return {
                "level_shift": np.nan,
                "slope_change": np.nan,
                "level_p": np.nan,
                "slope_p": np.nan,
                "r_squared": np.nan,
            }

    def compute_zone_statistics(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute summary statistics per zone"""
        zones = df["conflict_zone_name"].unique()
        zones = [z for z in zones if z != "none" and pd.notna(z)]

        if not zones:
            return pd.DataFrame()

        rows = []
        for zone in zones:
            zone_df = df[df["conflict_zone_name"] == zone]

            row = {"zone": zone, "n_records": len(zone_df)}
            row["n_vessels"] = zone_df["MMSI"].nunique()

            for col in ["SOG", "delta_sog", "delta_cog"]:
                if col in zone_df.columns:
                    row[f"{col}_mean"] = zone_df[col].mean()
                    row[f"{col}_std"] = zone_df[col].std()

            if "is_dark_ship" in zone_df.columns:
                row["dark_ship_ratio"] = (
                    zone_df["is_dark_ship"].mean()
                )

            if "traffic_count" in zone_df.columns:
                row["traffic_count"] = zone_df["traffic_count"].mean()

            rows.append(row)

        return pd.DataFrame(rows)

    def run_all(
        self,
        df: pd.DataFrame,
        conflict_events_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run full correlation analysis"""
        logger.info("Running correlation analysis...")

        results: Dict[str, Any] = {}

        zone_stats = self.compute_zone_statistics(df)
        if not zone_stats.empty:
            results["zone_statistics"] = zone_stats
            zone_stats.to_csv(
                self.output_dir / "zone_statistics.csv",
                index=False,
            )
            logger.info(f"Saved: {self.output_dir / 'zone_statistics.csv'}")

        zones = df["conflict_zone_name"].unique()
        zones = [z for z in zones if z != "none" and pd.notna(z)]

        for zone in zones:
            zone_df = df[df["conflict_zone_name"] == zone]

            event_date = self.CONFLICT_ONSET.get(zone)
            if not event_date:
                onset = self.CONFLICT_ZONE_ALIASES.get(zone)
                event_date = self.CONFLICT_ONSET.get(onset)

            if not event_date:
                event_date = df["BaseDateTime"].min().strftime("%Y-%m-%d")

            event_results = self.event_study(
                zone_df, event_date, zone, window_days=7
            )
            if not event_results.empty:
                results[f"{zone}_event_study"] = event_results
                event_results.to_csv(
                    self.output_dir / f"{zone}_event_study.csv",
                    index=False,
                )
                logger.info(
                    f"Saved: {self.output_dir / f'{zone}_event_study.csv'}"
                )

            if "SOG" in zone_df.columns:
                sog_series = (
                    zone_df.set_index("BaseDateTime")["SOG"]
                    .sort_index()
                    .resample("D")
                    .mean()
                )
                if len(sog_series) > 20:
                    its_result = self.interrupted_time_series(
                        sog_series, event_date
                    )
                    results[f"{zone}_its"] = its_result
                    pd.DataFrame([its_result]).to_csv(
                        self.output_dir / f"{zone}_its.csv",
                        index=False,
                    )
                    logger.info(
                        f"Saved: {self.output_dir / f'{zone}_its.csv'}"
                    )

        logger.info("Correlation analysis complete!")
        return results


def main():
    parser = argparse.ArgumentParser(description="Conflict Correlation Analyzer")
    parser.add_argument("--input", required=True, help="Input Parquet path")
    parser.add_argument(
        "--conflict-events",
        default=None,
        help="Conflict events CSV path",
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs/tables",
        help="Output directory",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df):,} records")

    config = load_config()
    analyzer = ConflictCorrelationAnalyzer(args.output_dir, config)
    analyzer.run_all(df, args.conflict_events)


if __name__ == "__main__":
    main()