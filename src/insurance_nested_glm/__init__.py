"""insurance-nested-glm: Nested GLM with neural network entity embeddings
and spatially constrained territory clustering.

Implements the framework from Wang, Shi, Cao (NAAJ 2025).  The three stages —
entity embedding of high-cardinality categoricals, spatially constrained
territory clustering, and a final outer GLM — are orchestrated by
:class:`NestedGLMPipeline`.

Core classes:

    NestedGLMPipeline     — four-phase end-to-end pipeline
    EmbeddingNet          — PyTorch embedding + dense network (CANN-style)
    EmbeddingTrainer      — fit/transform wrapper around EmbeddingNet
    TerritoryClusterer    — SKATER-based spatially constrained clustering
    NestedGLM             — outer statsmodels GLM with embeddings + territory

Utility functions:

    build_adjacency       — Queen contiguity weights from GeoDataFrame
    plot_territory_map    — choropleth of territory labels (requires [plot])
    embedding_pca_plot    — 2-D PCA of embedding vectors (requires [plot])
    credibility_report    — exposure / claims summary per territory

Optional dependencies:

    Spatial clustering requires geopandas, libpysal, spopt.
    Install with: pip install insurance-nested-glm[spatial]

    Plotting requires matplotlib.
    Install with: pip install insurance-nested-glm[plot]

References:
    Wang R, Shi H, Cao J (2025). A Nested GLM Framework with Neural Network
    Encoding and Spatially Constrained Clustering in Non-Life Insurance
    Ratemaking. North American Actuarial Journal, 29(3).
"""

from __future__ import annotations

from ._utils import build_adjacency, credibility_report
from .embedding import EmbeddingNet, EmbeddingTrainer
from .glm import NestedGLM
from .pipeline import NestedGLMPipeline
from .territory import TerritoryClusterer

# Optional plotting — lazy import with helpful error
_PLOT_NAMES = ("plot_territory_map", "embedding_pca_plot")

try:
    from ._utils import embedding_pca_plot, plot_territory_map

    _plot_available = True
except ImportError:
    _plot_available = False


def __getattr__(name: str):
    if name in _PLOT_NAMES:
        if _plot_available:
            from ._utils import embedding_pca_plot, plot_territory_map

            if name == "plot_territory_map":
                return plot_territory_map
            if name == "embedding_pca_plot":
                return embedding_pca_plot
        raise ImportError(
            f"insurance_nested_glm.{name} requires matplotlib. "
            "Install with: pip install insurance-nested-glm[plot]"
        )
    raise AttributeError(f"module 'insurance_nested_glm' has no attribute {name!r}")


__all__ = [
    # Pipeline
    "NestedGLMPipeline",
    # Embedding
    "EmbeddingNet",
    "EmbeddingTrainer",
    # Territory
    "TerritoryClusterer",
    # GLM
    "NestedGLM",
    # Utils
    "build_adjacency",
    "credibility_report",
    "plot_territory_map",
    "embedding_pca_plot",
]

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("insurance-nested-glm")
except PackageNotFoundError:
    __version__ = "0.0.0"  # not installed
