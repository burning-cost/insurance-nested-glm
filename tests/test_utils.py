"""Tests for insurance_nested_glm._utils."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# credibility_report
# ---------------------------------------------------------------------------


def test_credibility_report_basic():
    from insurance_nested_glm._utils import credibility_report

    labels = pd.Series([1, 1, 2, 2, 3])
    exposure = pd.Series([1.0, 2.0, 3.0, 1.0, 4.0])

    result = credibility_report(labels, exposure)
    assert set(result.columns) == {"territory", "n_units", "total_exposure"}
    assert len(result) == 3
    # Sorted descending by total_exposure
    assert result["total_exposure"].is_monotonic_decreasing


def test_credibility_report_with_claims():
    from insurance_nested_glm._utils import credibility_report

    labels = pd.Series([1, 1, 2, 2])
    exposure = pd.Series([1.0, 1.0, 1.0, 1.0])
    claims = pd.Series([0.0, 2.0, 1.0, 1.0])

    result = credibility_report(labels, exposure, claims)
    assert "total_claims" in result.columns
    assert "frequency" in result.columns

    row1 = result[result["territory"] == 1].iloc[0]
    assert row1["total_claims"] == 2.0
    assert row1["frequency"] == pytest.approx(1.0)

    row2 = result[result["territory"] == 2].iloc[0]
    assert row2["frequency"] == pytest.approx(1.0)


def test_credibility_report_empty():
    from insurance_nested_glm._utils import credibility_report

    labels = pd.Series([], dtype=int)
    exposure = pd.Series([], dtype=float)
    result = credibility_report(labels, exposure)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# build_adjacency (requires libpysal)
# ---------------------------------------------------------------------------


def test_build_adjacency_returns_w():
    pytest.importorskip("libpysal")
    import geopandas as gpd
    from shapely.geometry import box

    from insurance_nested_glm._utils import build_adjacency

    # Three boxes in a row: 0-1-2, each touching the next
    geoms = [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)]
    gdf = gpd.GeoDataFrame({"geometry": geoms}, crs="EPSG:27700")

    w = build_adjacency(gdf)
    # Middle polygon (index 1) should have both neighbours (0 and 2)
    neighbours_of_1 = list(w.neighbors[1])
    assert 0 in neighbours_of_1
    assert 2 in neighbours_of_1
    # End polygon (index 0) should have only one neighbour (1)
    assert 1 in list(w.neighbors[0])


def test_build_adjacency_raises_without_libpysal():
    """Without libpysal installed, build_adjacency should raise ImportError with helpful message."""
    import sys

    # Only mock if libpysal is actually installed (otherwise test is moot)
    try:
        import libpysal  # noqa: F401
    except ImportError:
        pytest.skip("libpysal not installed")

    from insurance_nested_glm._utils import build_adjacency
    import geopandas as gpd
    from shapely.geometry import box

    geoms = [box(0, 0, 1, 1)]
    gdf = gpd.GeoDataFrame({"geometry": geoms})

    # Temporarily hide libpysal
    original = sys.modules.get("libpysal")
    sys.modules["libpysal"] = None  # type: ignore[assignment]
    try:
        with pytest.raises((ImportError, AttributeError)):
            build_adjacency(gdf)
    finally:
        if original is None:
            del sys.modules["libpysal"]
        else:
            sys.modules["libpysal"] = original


# ---------------------------------------------------------------------------
# embedding_pca_plot (requires matplotlib + sklearn)
# ---------------------------------------------------------------------------


def test_embedding_pca_plot_returns_figure():
    pytest.importorskip("matplotlib")

    from insurance_nested_glm._utils import embedding_pca_plot

    embeddings = np.random.default_rng(0).normal(size=(20, 4))
    labels = [f"cat_{i}" for i in range(20)]
    fig = embedding_pca_plot(embeddings, labels=labels, title="Test PCA")

    import matplotlib.figure
    assert isinstance(fig, matplotlib.figure.Figure)


def test_embedding_pca_plot_single_dim():
    """1-D embeddings should not crash."""
    pytest.importorskip("matplotlib")

    from insurance_nested_glm._utils import embedding_pca_plot

    embeddings = np.arange(10).reshape(-1, 1).astype(float)
    fig = embedding_pca_plot(embeddings, title="1D Test")
    import matplotlib.figure
    assert isinstance(fig, matplotlib.figure.Figure)


# ---------------------------------------------------------------------------
# plot_territory_map (requires matplotlib + geopandas)
# ---------------------------------------------------------------------------


def test_plot_territory_map_returns_figure():
    pytest.importorskip("matplotlib")
    pytest.importorskip("geopandas")
    import geopandas as gpd
    from shapely.geometry import box

    from insurance_nested_glm._utils import plot_territory_map

    geoms = [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)]
    gdf = gpd.GeoDataFrame({"geometry": geoms}, crs="EPSG:27700")
    labels = pd.Series([1, 2, 1])

    import matplotlib
    matplotlib.use("Agg")
    fig = plot_territory_map(gdf, labels)

    import matplotlib.figure
    assert isinstance(fig, matplotlib.figure.Figure)
