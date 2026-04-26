"""
Visualization Module
====================
Spatial, temporal, and statistical visualizations for AIS data.
"""
from .base import VizConfig, load_viz_config, setup_matplotlib
from .spatial_viz import SpatialVisualizer
from .temporal_viz import TemporalVisualizer
from .statistical_viz import StatisticalVisualizer

__all__ = [
    "VizConfig",
    "load_viz_config",
    "setup_matplotlib",
    "SpatialVisualizer",
    "TemporalVisualizer",
    "StatisticalVisualizer",
]