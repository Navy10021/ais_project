import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path("./data/processed")
OUT = Path("./outputs/figures/eda")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(DATA / "ais_features.parquet")
print("Records:", len(df))

sample = df.sample(n=min(50000, len(df)), random_state=42)
fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(sample["LON"], sample["LAT"], c=sample["SOG"], cmap="viridis", alpha=0.5, s=10)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Geographic Distribution")
plt.tight_layout()
plt.savefig(OUT / "geo.png", dpi=150)
plt.close()
print("Saved geo.png")