"""Utility functions for the nested GLM pipeline.

These are used internally and exposed for users who want lower-level access.
Spatial and plotting functions require the ``[spatial]`` and ``[plot]``
optional dependency groups respectively.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def build_adjacency(
    gdf: "geopandas.GeoDataFrame",  # noqa: F821
) -> "libpysal.weights.W":  # noqa: F821
    """Build Queen contiguity spatial weights from a GeoDataFrame.

    Parameters
    ----------
    gdf:
        GeoDataFrame with polygon geometries.

    Returns
    -------
    libpysal.weights.W
        Queen contiguity weights object.
    """
    try:
        import libpysal
    except ImportError as exc:
        raise ImportError(
            "build_adjacency requires libpysal. "
            "Install with: pip install insurance-nested-glm[spatial]"
        ) from exc

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = libpysal.weights.Queen.from_dataframe(gdf, silence_warnings=True)
    return w


def plot_territory_map(
    gdf: "geopandas.GeoDataFrame",  # noqa: F821
    labels: pd.Series,
    title: str = "Territory Map",
    figsize: tuple = (10, 8),
    cmap: str = "tab20",
) -> "matplotlib.figure.Figure":  # noqa: F821
    """Plot a choropleth map of territory labels.

    Parameters
    ----------
    gdf:
        GeoDataFrame with one row per spatial unit.  Must have a valid
        geometry column.
    labels:
        Territory labels (integer series), aligned with *gdf* index.
    title:
        Figure title.
    figsize:
        Matplotlib figure size.
    cmap:
        Matplotlib colourmap name.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_territory_map requires matplotlib. "
            "Install with: pip install insurance-nested-glm[plot]"
        ) from exc

    gdf = gdf.copy()
    gdf["_territory"] = labels.values

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    gdf.plot(
        column="_territory",
        ax=ax,
        cmap=cmap,
        legend=True,
        legend_kwds={"label": "Territory"},
    )
    ax.set_title(title)
    ax.set_axis_off()
    return fig


def embedding_pca_plot(
    embeddings: np.ndarray,
    labels: Optional[Sequence] = None,
    title: str = "Embedding PCA",
    figsize: tuple = (8, 6),
) -> "matplotlib.figure.Figure":  # noqa: F821
    """Visualise embedding vectors in 2-D via PCA.

    Parameters
    ----------
    embeddings:
        Array of shape ``(n_categories, embedding_dim)``.
    labels:
        Category labels for annotation.  If None, points are unlabelled.
    title:
        Figure title.
    figsize:
        Matplotlib figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "embedding_pca_plot requires matplotlib. "
            "Install with: pip install insurance-nested-glm[plot]"
        ) from exc

    from sklearn.decomposition import PCA

    n_components = min(2, embeddings.shape[1])
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(embeddings)

    if coords.shape[1] == 1:
        # Single-component embedding — plot as 1-D strip
        coords = np.column_stack([coords, np.zeros(len(coords))])

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(coords[:, 0], coords[:, 1], alpha=0.7, s=20)

    if labels is not None:
        for i, lbl in enumerate(labels):
            ax.annotate(str(lbl), (coords[i, 0], coords[i, 1]), fontsize=6)

    var_explained = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} var.)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} var.)" if len(var_explained) > 1 else "")
    ax.set_title(title)
    return fig


def credibility_report(
    labels: pd.Series,
    exposure: pd.Series,
    claims: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Summarise exposure and claims per territory.

    Parameters
    ----------
    labels:
        Territory label per policy/spatial unit.
    exposure:
        Exposure per policy/spatial unit.
    claims:
        Claim count per policy/spatial unit.  If None, only exposure is
        summarised.

    Returns
    -------
    pd.DataFrame
        Columns: ``['territory', 'n_units', 'total_exposure']`` and
        optionally ``['total_claims', 'frequency']``.
        Sorted by total exposure descending.
    """
    df = pd.DataFrame({"territory": labels, "exposure": exposure})
    if claims is not None:
        df["claims"] = claims

    agg = df.groupby("territory").agg(
        n_units=("exposure", "count"),
        total_exposure=("exposure", "sum"),
    )

    if claims is not None:
        claim_agg = df.groupby("territory")["claims"].sum().rename("total_claims")
        agg = agg.join(claim_agg)
        agg["frequency"] = agg["total_claims"] / agg["total_exposure"].clip(lower=1e-10)

    return agg.sort_values("total_exposure", ascending=False).reset_index()
