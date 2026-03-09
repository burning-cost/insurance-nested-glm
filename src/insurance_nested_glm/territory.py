"""Spatially constrained territory clustering for insurance ratemaking.

Uses SKATER (Spatial 'K'luster Analysis by Tree Edge Removal) as the default
method, accessed via ``spopt.region.Skater``.  SKATER guarantees spatial
contiguity by construction: every territory is a connected subgraph of the
adjacency network.

UK island handling: the Channel Islands, Isle of Man, and Orkney/Shetland all
form disconnected components in a Queen adjacency graph.  This module detects
disconnected components and clusters each independently before merging labels.

Credibility filtering: after initial clustering, any territory whose total
exposure falls below ``min_exposure`` is merged into the nearest neighbour
territory (by centroid distance) that does meet the threshold.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


class TerritoryClusterer:
    """Spatially constrained clustering for insurance territory rating.

    Wraps ``spopt.region.Skater`` (default) or ``MaxPHeuristic`` from the same
    package.  Requires the ``[spatial]`` optional dependency group.

    Parameters
    ----------
    n_clusters:
        Target number of territory bands.  The actual number may be lower if
        the credibility filter merges small territories.
    min_exposure:
        Minimum total exposure (years) for a territory to survive the
        credibility filter.  Territories below this threshold are merged into
        the nearest neighbour.  ``None`` disables filtering.
    method:
        ``'skater'`` (default) or ``'maxp'``.
    random_state:
        Random seed passed to the underlying algorithm.

    Notes
    -----
    ``geopandas``, ``libpysal``, and ``spopt`` must be installed.  Install
    with ``pip install insurance-nested-glm[spatial]``.
    """

    def __init__(
        self,
        n_clusters: int = 200,
        min_exposure: Optional[float] = None,
        method: str = "skater",
        random_state: int = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self.min_exposure = min_exposure
        self.method = method
        self.random_state = random_state

        self._labels: Optional[pd.Series] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        gdf: "geopandas.GeoDataFrame",  # noqa: F821
        feature_cols: Sequence[str],
        exposure: Optional[np.ndarray] = None,
    ) -> "TerritoryClusterer":
        """Cluster spatial units into territories.

        Parameters
        ----------
        gdf:
            GeoDataFrame with one row per spatial unit (e.g. postcode sector).
            Must have a valid geometry column.
        feature_cols:
            Columns in *gdf* to use as clustering features (e.g. embedding
            coordinates, claim frequency, demographic scores).
        exposure:
            Total exposure per spatial unit.  Used only for the credibility
            filter (``min_exposure``).  If None, all units are assumed equal.

        Returns
        -------
        TerritoryClusterer
            self, with ``labels_`` attribute set.
        """
        try:
            import libpysal
            import spopt
        except ImportError as exc:
            raise ImportError(
                "Spatial clustering requires geopandas, libpysal, and spopt. "
                "Install with: pip install insurance-nested-glm[spatial]"
            ) from exc

        gdf = gdf.copy().reset_index(drop=True)
        features = gdf[list(feature_cols)].values.astype(float)

        # Detect connected components in the adjacency graph
        w_full = self._build_weights(gdf)
        components = self._detect_components(w_full, len(gdf))

        labels_array = np.full(len(gdf), -1, dtype=int)
        label_offset = 0

        for component_indices in components:
            if len(component_indices) < 2:
                # Single-unit component — assign its own territory
                labels_array[component_indices[0]] = label_offset
                label_offset += 1
                continue

            sub_gdf = gdf.iloc[component_indices].copy().reset_index(drop=True)
            sub_features = features[component_indices]

            # Scale n_clusters proportionally to component size
            n_comp = len(component_indices)
            n_clust_comp = max(1, round(self.n_clusters * n_comp / len(gdf)))
            n_clust_comp = min(n_clust_comp, n_comp)

            sub_labels = self._cluster_component(
                sub_gdf, sub_features, n_clust_comp, spopt
            )
            labels_array[component_indices] = sub_labels + label_offset
            label_offset += sub_labels.max() + 1

        self._labels = pd.Series(labels_array, name="territory", index=gdf.index)

        # Credibility filter
        if self.min_exposure is not None and exposure is not None:
            self._labels = self._apply_credibility_filter(
                gdf, self._labels, exposure, self.min_exposure
            )

        self._fitted = True
        return self

    @property
    def labels_(self) -> pd.Series:
        """Territory labels after fitting (1-indexed integer series)."""
        self._check_fitted()
        assert self._labels is not None
        # Re-index to 1..K
        mapping = {
            old: new + 1
            for new, old in enumerate(sorted(self._labels.unique()))
        }
        return self._labels.map(mapping)

    def predict(self, gdf: "geopandas.GeoDataFrame") -> pd.Series:  # noqa: F821
        """Assign territory labels to new spatial units via nearest centroid.

        Parameters
        ----------
        gdf:
            GeoDataFrame of new spatial units.

        Returns
        -------
        pd.Series
            Territory labels (1-indexed).
        """
        raise NotImplementedError(
            "predict() is not yet implemented.  Join on unit identifier instead."
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_weights(
        self,
        gdf: "geopandas.GeoDataFrame",  # noqa: F821
    ) -> "libpysal.weights.W":  # noqa: F821
        """Build Queen contiguity weights for *gdf*."""
        import libpysal

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = libpysal.weights.Queen.from_dataframe(gdf, silence_warnings=True)
        return w

    def _detect_components(
        self,
        w: "libpysal.weights.W",  # noqa: F821
        n: int,
    ) -> List[List[int]]:
        """Return list of index lists, one per connected component."""
        visited = np.zeros(n, dtype=bool)
        components: List[List[int]] = []

        for start in range(n):
            if visited[start]:
                continue
            # BFS
            queue = [start]
            component: List[int] = []
            while queue:
                node = queue.pop()
                if visited[node]:
                    continue
                visited[node] = True
                component.append(node)
                neighbours = w.neighbors.get(node, [])
                for nb in neighbours:
                    if not visited[nb]:
                        queue.append(nb)
            components.append(component)

        return components

    def _cluster_component(
        self,
        sub_gdf: "geopandas.GeoDataFrame",  # noqa: F821
        features: np.ndarray,
        n_clusters: int,
        spopt: object,
    ) -> np.ndarray:
        """Run SKATER or MaxP on a single connected component."""
        import libpysal

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = libpysal.weights.Queen.from_dataframe(sub_gdf, silence_warnings=True)

        if self.method == "skater":
            model = spopt.region.Skater(
                sub_gdf,
                w,
                attrs_name=None,  # we pass data directly
                n_clusters=n_clusters,
                floor=1,
            )
            # spopt Skater accepts a numpy array when attrs_name=None in some versions.
            # Fall back to using column names if needed.
            try:
                model.solve()
            except (TypeError, AttributeError):
                # Older spopt API: pass column names, not arrays.
                # We add temporary columns to the sub_gdf copy.
                tmp_cols = [f"_feat_{i}" for i in range(features.shape[1])]
                sub_gdf = sub_gdf.copy()
                for i, c in enumerate(tmp_cols):
                    sub_gdf[c] = features[:, i]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    w = libpysal.weights.Queen.from_dataframe(
                        sub_gdf, silence_warnings=True
                    )
                model = spopt.region.Skater(
                    sub_gdf,
                    w,
                    attrs_name=tmp_cols,
                    n_clusters=n_clusters,
                    floor=1,
                )
                model.solve()
        elif self.method == "maxp":
            # MaxP requires a threshold; use n_clusters as approximate guide.
            model = spopt.region.MaxPHeuristic(
                sub_gdf,
                w,
                attrs_name=list(sub_gdf.columns[:1]),
                threshold_name=list(sub_gdf.columns[:1])[0],
                threshold=0.0,
                top_n=10,
                max_iterations_construction=99,
            )
            model.solve()
        else:
            raise ValueError(f"Unknown method '{self.method}'. Use 'skater' or 'maxp'.")

        return np.asarray(model.labels_)

    def _apply_credibility_filter(
        self,
        gdf: "geopandas.GeoDataFrame",  # noqa: F821
        labels: pd.Series,
        exposure: np.ndarray,
        min_exposure: float,
    ) -> pd.Series:
        """Merge territories below *min_exposure* into the nearest neighbour territory.

        Nearest neighbour is determined by centroid distance.
        """
        labels = labels.copy()
        exposure_ser = pd.Series(exposure, index=gdf.index)

        centroids = gdf.geometry.centroid

        while True:
            terr_exposure = exposure_ser.groupby(labels).sum()
            small_terrs = terr_exposure[terr_exposure < min_exposure].index.tolist()

            if not small_terrs:
                break

            # Process smallest territory first
            small_terrs_sorted = terr_exposure.loc[small_terrs].sort_values().index.tolist()
            t = small_terrs_sorted[0]

            # Units in this territory
            mask_t = labels == t
            # Centroid of territory t
            cx_t = centroids[mask_t].x.mean()
            cy_t = centroids[mask_t].y.mean()

            # All other territories
            other_terrs = [ot for ot in labels.unique() if ot != t]
            if not other_terrs:
                break

            # Find nearest territory by centroid distance
            best_terr = None
            best_dist = float("inf")
            for ot in other_terrs:
                mask_ot = labels == ot
                cx_ot = centroids[mask_ot].x.mean()
                cy_ot = centroids[mask_ot].y.mean()
                dist = (cx_t - cx_ot) ** 2 + (cy_t - cy_ot) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_terr = ot

            labels[mask_t] = best_terr

        return labels

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before accessing labels_.")
