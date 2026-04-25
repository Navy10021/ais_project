"""
Conflict Correlation Analyzer
===============================
Statistical framework for establishing AIS → conflict relationships.

Methods:
  - Granger Causality - does AIS anomaly lead conflict intensity?
  - Cross-Correlation - optimal lead-lag window
  - Difference-in-Differences - isolating conflict effect
  - Event Study Analysis - pre/post comparison
  - Interrupted Time Series - level & slope change

Coding Conventions:
  - Type hints
  - Logging via logger
  - Reproducibility with SEED
"""
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import statsmodels.api as sm
from pathlib import Path
import logging
import argparse
from dataclasses import dataclass

logger = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)

import random
random.seed(SEED)


@dataclass
class AnalysisConfig:
    max_lag_granger: int = 30
    event_study_window: int = 30


class ConflictCorrelationAnalyzer:
    CONFLICT_ONSET = {
        "ukraine_war": "2022-02-24",
        "houthi_crisis": "2023-11-19",
        "pla_taiwan_drill": "2022-08-04",
        "kerch_bridge": "2022-10-08",
    }

    def __init__(self, output_dir: str = "./outputs/tables"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def granger_causality(self, ais_series: pd.Series, conflict_series: pd.Series, max_lag: int = 30) -> dict:
        from statsmodels.tsa.stattools import grangercausalitytests

        df = pd.DataFrame({"ais": ais_series, "conflict": conflict_series}).dropna()
        if len(df) < max_lag * 2:
            logger.warning("Insufficient data for Granger causality test")
            return {"significant_lags": {}, "optimal_lead_days": None}

        try:
            raw = grangercausalitytests(
                df[["conflict", "ais"]], maxlag=min(max_lag, len(df) // 4), verbose=False
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
        except Exception as e:
            logger.warning(f"Granger test failed: {e}")
            return {"significant_lags": {}, "optimal_lead_days": None}

    def event_study(
        self,
        df: pd.DataFrame,
        event_date: str,
        zone: str,
        window_days: int = 30,
    ) -> pd.DataFrame:
        ev = pd.Timestamp(event_date, tz="UTC")
        
        zone_df = df[df["conflict_zone_name"] == zone].copy()
        if len(zone_df) == 0:
            logger.warning(f"No data for zone: {zone}")
            return pd.DataFrame()

        zone_df = zone_df.set_index("BaseDateTime").sort_index()

        try:
            pre = zone_df[ev - pd.Timedelta(days=window_days):ev]
            post = zone_df[ev:ev + pd.Timedelta(days=window_days)]
        except Exception as e:
            logger.warning(f"Event study window error: {e}")
            return pd.DataFrame()

        metrics = [
            "SOG", "delta_sog", "delta_cog", 
            "is_dark_ship", "loitering_flag"
        ]
        for m in metrics:
            if m not in zone_df.columns:
                metrics.remove(m)

        rows = []
        for m in metrics:
            pv = pre[m].dropna() if m in pre.columns else pd.Series()
            qv = post[m].dropna() if m in post.columns else pd.Series()
            
            if len(pv) > 0 and len(qv) > 0:
                try:
                    _, p = mannwhitneyu(pv, qv, alternative="two-sided")
                except:
                    p = np.nan
                
                pct_change = (qv.mean() - pv.mean()) / (pv.mean() + 1e-10) * 100 if pv.mean() != 0 else np.nan
                
                rows.append({
                    "metric": m,
                    "pre_mean": pv.mean(),
                    "post_mean": qv.mean(),
                    "pct_change": pct_change,
                    "p_value": p,
                    "significant": p < 0.05 if not np.isnan(p) else False,
                })

        return pd.DataFrame(rows)

    def difference_in_differences(
        self,
        df: pd.DataFrame,
        treatment_zone: str,
        control_zone: str,
        event_date: str,
        metric: str = "SOG",
    ) -> dict:
        ev = pd.Timestamp(event_date, tz="UTC")
        res = {}

        for zone in [treatment_zone, control_zone]:
            zone_df = df[df["conflict_zone_name"] == zone].copy()
            if len(zone_df) == 0:
                continue
            zone_df = zone_df.set_index("BaseDateTime").sort_index()
            
            pre_mean = zone_df.loc[:ev, metric].mean() if metric in zone_df.columns else np.nan
            post_mean = zone_df.loc[ev:, metric].mean() if metric in zone_df.columns else np.nan
            res[zone] = {"pre": pre_mean, "post": post_mean}

        if treatment_zone not in res or control_zone not in res:
            logger.warning("Insufficient zones for DiD")
            return {"did_estimator": np.nan, "zones": res}

        t, c = res[treatment_zone], res[control_zone]
        did = (t["post"] - t["pre"]) - (c["post"] - c["pre"])
        
        if np.isnan(t["pre"]) or np.isnan(c["pre"]):
            did = np.nan

        return {"did_estimator": did, "zones": res}

    def interrupted_time_series(self, series: pd.Series, breakpoint: str) -> dict:
        series = series.dropna().sort_index()
        if len(series) < 10:
            logger.warning("Insufficient data for ITS")
            return {"level_shift": np.nan, "slope_change": np.nan, "level_p": np.nan, "slope_p": np.nan, "r_squared": np.nan}

        try:
            t = np.arange(len(series))
            bp = series.index.searchsorted(pd.Timestamp(breakpoint, tz="UTC"))
            bp = max(1, min(bp, len(series) - 2))
            
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
            return {"level_shift": np.nan, "slope_change": np.nan, "level_p": np.nan, "slope_p": np.nan, "r_squared": np.nan}

    def run_all(self, df: pd.DataFrame, conflict_events_path: str = None):
        logger.info("Running correlation analysis...")

        results = {}
        zones = df['conflict_zone_name'].unique()

        for zone in zones:
            if zone == "none":
                continue
            zone_df = df[df['conflict_zone_name'] == zone]
            
            if zone in self.CONFLICT_ONSET:
                event_date = self.CONFLICT_ONSET[zone]
            else:
                event_date = df['BaseDateTime'].min().strftime('%Y-%m-%d')
            
            event_results = self.event_study(zone_df, event_date, zone, window_days=7)
            if not event_results.empty:
                results[f"{zone}_event_study"] = event_results

            if 'SOG' in zone_df.columns:
                sog_series = zone_df.set_index('BaseDateTime')['SOG'].sort_index()
                its_result = self.interrupted_time_series(sog_series, event_date)
                results[f"{zone}_its"] = its_result

        for key, val in results.items():
            if isinstance(val, pd.DataFrame):
                val.to_csv(self.output_dir / f"{key}.csv", index=False)
                logger.info(f"Saved: {self.output_dir / f'{key}.csv'}")
            elif isinstance(val, dict):
                pd.DataFrame([val]).to_csv(self.output_dir / f"{key}.csv", index=False)
                logger.info(f"Saved: {self.output_dir / f'{key}.csv'}")

        logger.info("Correlation analysis complete!")
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="./outputs/tables")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df):,} records")

    analyzer = ConflictCorrelationAnalyzer(args.output_dir)
    analyzer.run_all(df)


if __name__ == "__main__":
    main()