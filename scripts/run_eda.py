"""
EDA Runner Script
=================
Executes EDA analysis and saves figures.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.rcParams.update({
    'figure.dpi': 150,
    'figure.figsize': (10, 6),
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})
sns.set_style('whitegrid')

DATA_DIR = Path('./data/processed')
OUTPUT_DIR = Path('./outputs/figures/eda')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_eda():
    print("=" * 50)
    print("EDA - Maritime Conflict Intelligence System")
    print("=" * 50)

    df = pd.read_parquet(DATA_DIR / 'ais_features.parquet')
    print(f"\n[1] Data loaded: {len(df):,} records, {len(df.columns)} columns")

    print("\n[2] Basic Info:")
    print(df.dtypes)

    print("\n[3] Missing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    print(missing[missing > 0].sort_values(ascending=False).head(10))

    print("\n[4] Vessel Types:")
    print(df['VesselType'].value_counts().head(10))

    print("\n[5] Speed Stats:")
    print(df['SOG'].describe())

    print("\n[6] Conflict Zones:")
    print(df['conflict_zone_name'].value_counts())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False).head(10)
    missing.plot(kind='barh', ax=axes[0], color='coral')
    axes[0].set_xlabel('Missing Count')
    axes[0].set_title('Missing Values')

    df['VesselType'].value_counts().head(10).plot(kind='bar', ax=axes[1], color='steelblue')
    axes[1].set_xlabel('Vessel Type')
    axes[1].set_title('Top 10 Vessel Types')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'missing_vessel_types.png', dpi=300)
    plt.close()
    print(f"\n[Saved] {OUTPUT_DIR / 'missing_vessel_types.png'}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    df['SOG'].hist(bins=50, ax=axes[0], color='teal', edgecolor='white')
    axes[0].set_xlabel('Speed (knots)')
    axes[0].set_title('Speed Distribution')

    df.boxplot(column='SOG', ax=axes[1], vert=True)
    axes[1].set_title('Speed Boxplot')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'speed_distribution.png', dpi=300)
    plt.close()
    print(f"[Saved] {OUTPUT_DIR / 'speed_distribution.png'}")

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df['LON'], df['LAT'], c=df['SOG'], cmap='viridis', alpha=0.5, s=10)
    plt.colorbar(scatter, label='Speed (knots)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Vessel Traffic Geographic Distribution')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'geographic_distribution.png', dpi=300)
    plt.close()
    print(f"[Saved] {OUTPUT_DIR / 'geographic_distribution.png'}")

    df['hour'] = df['BaseDateTime'].dt.hour
    df['date'] = df['BaseDateTime'].dt.date

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    hourly = df.groupby('hour')['MMSI'].nunique()
    hourly.plot(kind='bar', ax=axes[0], color='coral')
    axes[0].set_xlabel('Hour of Day')
    axes[0].set_title('Hourly Activity')

    daily = df.groupby('date')['MMSI'].nunique()
    daily.plot(ax=axes[1], marker='o', color='teal')
    axes[1].set_xlabel('Date')
    axes[1].set_title('Daily Activity')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'temporal_distribution.png', dpi=300)
    plt.close()
    print(f"[Saved] {OUTPUT_DIR / 'temporal_distribution.png'}")

    fig, ax = plt.subplots(figsize=(10, 5))
    df['conflict_zone_name'].value_counts().plot(kind='bar', ax=ax, color='steelblue')
    ax.set_xlabel('Conflict Zone')
    ax.set_title('Records by Conflict Zone')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'conflict_zones.png', dpi=300)
    plt.close()
    print(f"[Saved] {OUTPUT_DIR / 'conflict_zones.png'}")

    numeric_cols = ['SOG', 'COG', 'Heading', 'VesselType', 'Length', 'Width']
    if all(col in df.columns for col in numeric_cols):
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'correlation_matrix.png', dpi=300)
        plt.close()
        print(f"[Saved] {OUTPUT_DIR / 'correlation_matrix.png'}")

    print("\n" + "=" * 50)
    print("EDA Complete!")
    print("=" * 50)
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_eda()