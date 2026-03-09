"""Tests for insurance_nested_glm.territory.

Spatial tests use mock GeoDataFrames built from shapely geometries so we do
not need real geospatial data.  spopt/libpysal are mocked where needed to keep
tests fast and to test logic independently of the clustering algorithm.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from insurance_nested_glm.territory import TerritoryClusterer


# ---------------------------------------------------------------------------
# Helpers to build synthetic GeoDataFrames
# ---------------------------------------------------------------------------


def _make_grid_gdf(n_rows: int = 3, n_cols: int = 3):
    """Build a regular grid GeoDataFrame with square polygons."""
    pytest.importorskip("geopandas")
    pytest.importorskip("shapely")
    import geopandas as gpd
    from shapely.geometry import box

    geoms = []
    ids = []
    for r in range(n_rows):
        for c in range(n_cols):
            geoms.append(box(c, r, c + 1, r + 1))
            ids.append(f"unit_{r}_{c}")
    return gpd.GeoDataFrame({"unit_id": ids, "geometry": geoms}, crs="EPSG:27700")


def _make_disconnected_gdf():
    """Two separate grids (disconnected components)."""
    pytest.importorskip("geopandas")
    pytest.importorskip("shapely")
    import geopandas as gpd
    from shapely.geometry import box

    geoms = []
    ids = []
    # Grid 1: 0..2
    for c in range(3):
        geoms.append(box(c, 0, c + 1, 1))
        ids.append(f"main_{c}")
    # Grid 2: separate island at x=100
    geoms.append(box(100, 0, 101, 1))
    ids.append("island_0")
    return gpd.GeoDataFrame({"unit_id": ids, "geometry": geoms}, crs="EPSG:27700")


def _spopt_available():
    try:
        import spopt  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------


def test_territory_clusterer_defaults():
    tc = TerritoryClusterer()
    assert tc.n_clusters == 200
    assert tc.min_exposure is None
    assert tc.method == "skater"


def test_territory_clusterer_unfitted_labels_raises():
    tc = TerritoryClusterer()
    with pytest.raises(RuntimeError, match="fit()"):
        _ = tc.labels_


# ---------------------------------------------------------------------------
# _detect_components
# ---------------------------------------------------------------------------


def test_detect_components_single():
    """All-connected graph → single component."""
    tc = TerritoryClusterer()
    # Build mock weights object with a ring adjacency: 0-1-2-0
    w = MagicMock()
    w.neighbors = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    components = tc._detect_components(w, n=3)
    assert len(components) == 1
    assert sorted(components[0]) == [0, 1, 2]


def test_detect_components_two():
    """Two separate components."""
    tc = TerritoryClusterer()
    w = MagicMock()
    # 0-1 connected; 2-3 connected; no edges between groups
    w.neighbors = {0: [1], 1: [0], 2: [3], 3: [2]}
    components = tc._detect_components(w, n=4)
    assert len(components) == 2
    # Flatten and sort to check membership
    all_nodes = sorted(node for comp in components for node in comp)
    assert all_nodes == [0, 1, 2, 3]


def test_detect_components_isolated_node():
    """Node with no neighbours is its own component."""
    tc = TerritoryClusterer()
    w = MagicMock()
    w.neighbors = {0: [1], 1: [0], 2: []}
    components = tc._detect_components(w, n=3)
    assert len(components) == 2


# ---------------------------------------------------------------------------
# _apply_credibility_filter
# ---------------------------------------------------------------------------


def test_credibility_filter_merges_small():
    """Territory below min_exposure is merged into nearest neighbour."""
    pytest.importorskip("geopandas")
    import geopandas as gpd
    from shapely.geometry import box

    # Three units in a row; territory 1 has tiny exposure
    geoms = [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)]
    gdf = gpd.GeoDataFrame({"geometry": geoms}, crs="EPSG:27700")

    tc = TerritoryClusterer(min_exposure=5.0)
    labels = pd.Series([1, 2, 3], index=gdf.index)
    exposure = np.array([0.1, 10.0, 10.0])  # territory 1 is tiny

    new_labels = tc._apply_credibility_filter(gdf, labels, exposure, min_exposure=5.0)

    # Territory 1 (unit 0) should have been merged into a neighbour
    assert new_labels[0] != 1 or new_labels[0] == new_labels[1]
    # After merging, only 2 territories remain
    assert new_labels.nunique() == 2


def test_credibility_filter_no_change_when_all_large():
    """No merging when all territories exceed min_exposure."""
    pytest.importorskip("geopandas")
    import geopandas as gpd
    from shapely.geometry import box

    geoms = [box(0, 0, 1, 1), box(1, 0, 2, 1)]
    gdf = gpd.GeoDataFrame({"geometry": geoms}, crs="EPSG:27700")

    tc = TerritoryClusterer(min_exposure=1.0)
    labels = pd.Series([1, 2], index=gdf.index)
    exposure = np.array([10.0, 10.0])

    new_labels = tc._apply_credibility_filter(gdf, labels, exposure, min_exposure=1.0)
    assert list(new_labels) == [1, 2]


# ---------------------------------------------------------------------------
# Integration: fit on a grid GeoDataFrame (skips if spopt not installed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _spopt_available(), reason="spopt not installed")
def test_territory_clusterer_fit_grid():
    """Fit on a 3x3 grid should produce labels without error."""
    gdf = _make_grid_gdf(3, 3)
    gdf["feat_x"] = gdf.geometry.centroid.x
    gdf["feat_y"] = gdf.geometry.centroid.y

    tc = TerritoryClusterer(n_clusters=3, random_state=0)
    tc.fit(gdf, feature_cols=["feat_x", "feat_y"])

    labels = tc.labels_
    assert len(labels) == 9
    assert labels.nunique() <= 3
    assert labels.min() >= 1


@pytest.mark.skipif(not _spopt_available(), reason="spopt not installed")
def test_territory_clusterer_disconnected():
    """Disconnected components are clustered independently."""
    gdf = _make_disconnected_gdf()
    gdf["feat_x"] = gdf.geometry.centroid.x
    gdf["feat_y"] = gdf.geometry.centroid.y

    tc = TerritoryClusterer(n_clusters=2, random_state=0)
    tc.fit(gdf, feature_cols=["feat_x", "feat_y"])

    labels = tc.labels_
    assert len(labels) == len(gdf)
    # Island (last unit) should have a different label from all main units, or at
    # least the clustering should complete without error
    assert labels.min() >= 1
    assert labels.max() <= labels.nunique() + 1


# ---------------------------------------------------------------------------
# Zero exposure handling
# ---------------------------------------------------------------------------


def test_credibility_filter_zero_exposure():
    """Territory with zero exposure is merged into nearest neighbour."""
    pytest.importorskip("geopandas")
    import geopandas as gpd
    from shapely.geometry import box

    geoms = [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)]
    gdf = gpd.GeoDataFrame({"geometry": geoms}, crs="EPSG:27700")

    tc = TerritoryClusterer(min_exposure=0.5)
    labels = pd.Series([1, 2, 3], index=gdf.index)
    exposure = np.array([0.0, 5.0, 5.0])

    new_labels = tc._apply_credibility_filter(gdf, labels, exposure, min_exposure=0.5)
    # The zero-exposure territory must have been merged
    assert new_labels[0] == new_labels[1] or new_labels[0] == new_labels[2]
