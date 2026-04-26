"""
Shared Visualization Configuration
=======================
Common configuration and setup for all visualization modules.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class VizConfig:
    """Configuration for Visualizations"""
    dpi: int = 300
    figsize: tuple = (10, 6)
    font_family: str = "DejaVu Sans"
    font_size: int = 12
    style: str = "whitegrid"


def load_viz_config(config_path: str = "./config/settings.yaml") -> VizConfig:
    """Load visualization configuration from YAML"""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        v = config.get("visualization", {})
        return VizConfig(
            dpi=v.get("dpi", 300),
            figsize=tuple(v.get("figsize", [10, 6])),
            font_family=v.get("font_family", "DejaVu Sans"),
            font_size=v.get("font_size", 12),
            style=v.get("style", "whitegrid"),
        )
    except Exception:
        logger.warning(f"Could not load config from {config_path}, using defaults")
        return VizConfig()


def setup_matplotlib(config: Optional[VizConfig] = None) -> None:
    """Setup matplotlib with publication-quality defaults"""
    config = config or VizConfig()
    plt.rcParams.update({
        "figure.dpi": config.dpi,
        "figure.figsize": config.figsize,
        "font.family": config.font_family,
        "font.size": config.font_size,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    sns.set_style(config.style)


def ensure_output_dir(path: str) -> Path:
    """Create output directory if it doesn't exist"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p