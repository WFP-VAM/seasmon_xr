"""Seasonal Monitoring Algorithms."""
# isort: skip_file
from ._version import __version__

from .accessors import (
    Anomalies,
    IterativeAggregation,
    LabelMaker,
    PixelAlgorithms,
    WhittakerSmoother,
)

__all__ = (
    "Anomalies",
    "IterativeAggregation",
    "LabelMaker",
    "PixelAlgorithms",
    "WhittakerSmoother",
    "__version__",
)
