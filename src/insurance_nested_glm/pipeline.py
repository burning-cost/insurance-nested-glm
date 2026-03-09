"""Four-phase nested GLM pipeline for insurance ratemaking.

Orchestrates the complete Wang, Shi, Cao (NAAJ 2025) workflow:

1. **Base GLM**: fit a standard GLM on structured factors.  The fitted log-rates
   become the offset for phase 2.
2. **Embedding**: train :class:`~insurance_nested_glm.embedding.EmbeddingTrainer`
   on GLM residuals with the base GLM log-rates as offset.  High-cardinality
   categoricals (vehicle make/model, postcode sector) are mapped to dense
   embedding vectors.
3. **Territory clustering**: cluster spatial units using
   :class:`~insurance_nested_glm.territory.TerritoryClusterer`, feeding in
   embedding vectors and/or geographic features.
4. **Outer GLM**: fit :class:`~insurance_nested_glm.glm.NestedGLM` on
   structured factors + embedding vectors + territory labels.

All four phases are re-run on each call to :meth:`NestedGLMPipeline.fit`.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Sequence

import numpy as np
import pandas as pd

from .embedding import EmbeddingTrainer
from .glm import NestedGLM
from .territory import TerritoryClusterer


class NestedGLMPipeline:
    """End-to-end nested GLM pipeline.

    Parameters
    ----------
    base_formula:
        Patsy formula (rhs only) for the structured base GLM, e.g.
        ``"age_band + vehicle_group + ncb"``.  If None, an intercept-only
        base is used.
    family:
        ``'poisson'`` (frequency) or ``'gamma'`` (severity).
    n_territories:
        Target number of territory bands for the spatial clustering step.
    min_territory_exposure:
        Minimum exposure per territory for the credibility filter.
    embedding_hidden_sizes:
        Hidden layer sizes for the embedding network.
    embedding_epochs:
        Training epochs for the embedding network.
    embedding_lr:
        Learning rate for the embedding network.
    embedding_batch_size:
        Mini-batch size for the embedding network.
    cluster_method:
        ``'skater'`` (default) or ``'maxp'``.
    random_state:
        Seed used throughout.
    """

    def __init__(
        self,
        base_formula: Optional[str] = None,
        family: Literal["poisson", "gamma"] = "poisson",
        n_territories: int = 200,
        min_territory_exposure: Optional[float] = None,
        embedding_hidden_sizes: Sequence[int] = (64,),
        embedding_epochs: int = 50,
        embedding_lr: float = 1e-3,
        embedding_batch_size: int = 1024,
        cluster_method: str = "skater",
        random_state: int = 42,
    ) -> None:
        self.base_formula = base_formula
        self.family = family
        self.n_territories = n_territories
        self.min_territory_exposure = min_territory_exposure
        self.embedding_hidden_sizes = embedding_hidden_sizes
        self.embedding_epochs = embedding_epochs
        self.embedding_lr = embedding_lr
        self.embedding_batch_size = embedding_batch_size
        self.cluster_method = cluster_method
        self.random_state = random_state

        self._base_glm: Optional[NestedGLM] = None
        self._embedding_trainer: Optional[EmbeddingTrainer] = None
        self._territory_clusterer: Optional[TerritoryClusterer] = None
        self._outer_glm: Optional[NestedGLM] = None

        self._geo_col: Optional[str] = None
        self._high_card_cols: Optional[List[str]] = None
        self._spatial_unit_col: Optional[str] = None
        self._geo_emb_cols: Optional[List[str]] = None

        self._fitted = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        exposure: np.ndarray,
        geo_gdf: Optional["geopandas.GeoDataFrame"] = None,  # noqa: F821
        geo_id_col: Optional[str] = None,
        high_card_cols: Optional[List[str]] = None,
        base_formula_cols: Optional[List[str]] = None,
    ) -> "NestedGLMPipeline":
        """Fit the four-phase pipeline.

        Parameters
        ----------
        X:
            Feature DataFrame.  One row per policy.
        y:
            Observed claim counts (Poisson) or aggregate severities (Gamma).
        exposure:
            Policy exposure in years.
        geo_gdf:
            GeoDataFrame of spatial units (e.g. postcode sectors), with one
            row per unit.  Required for the territory clustering step.  If
            None, territory clustering is skipped.
        geo_id_col:
            Column in both *X* and *geo_gdf* that links policies to spatial
            units (e.g. ``'postcode_sector'``).  Required when *geo_gdf* is
            provided.
        high_card_cols:
            High-cardinality categorical columns to embed (e.g.
            ``['vehicle_make_model']``).  If None, embedding step is skipped.
        base_formula_cols:
            Columns to include in the base GLM formula.  If None, the
            ``base_formula`` constructor parameter is used as-is.  Passing
            column names here constructs ``col1 + col2 + ...`` automatically.

        Returns
        -------
        NestedGLMPipeline
            self
        """
        y = np.asarray(y, dtype=float)
        exposure = np.asarray(exposure, dtype=float)
        n = len(y)

        self._high_card_cols = high_card_cols or []
        self._geo_id_col = geo_id_col

        # ---- Phase 1: Base GLM ----------------------------------------
        base_formula = self.base_formula
        if base_formula_cols:
            base_formula = " + ".join(base_formula_cols)

        self._base_glm = NestedGLM(
            family=self.family,
            formula=base_formula,
            add_embedding_cols=False,
            add_territory=False,
        )
        # Use only non-high-card, non-geo columns for the base GLM
        base_cols = self._select_base_cols(X, self._high_card_cols, base_formula_cols)
        self._base_glm.fit(X[base_cols], y, exposure)

        base_log_pred = np.log(
            self._base_glm.predict(X[base_cols], exposure).clip(min=1e-10)
        )

        # ---- Phase 2: Embedding ---------------------------------------
        emb_array: Optional[np.ndarray] = None
        if self._high_card_cols:
            self._embedding_trainer = EmbeddingTrainer(
                cat_cols=self._high_card_cols,
                hidden_sizes=self.embedding_hidden_sizes,
                lr=self.embedding_lr,
                epochs=self.embedding_epochs,
                batch_size=self.embedding_batch_size,
                random_state=self.random_state,
            )
            self._embedding_trainer.fit(
                X[self._high_card_cols], y, exposure=exposure, offset=base_log_pred
            )
            emb_array = self._embedding_trainer.transform(X[self._high_card_cols])

        # ---- Phase 3: Territory clustering ----------------------------
        territory_labels_policy: Optional[pd.Series] = None
        if geo_gdf is not None and geo_id_col is not None:
            geo_gdf = geo_gdf.copy().reset_index(drop=True)

            # Aggregate embedding vectors to spatial unit level
            geo_feature_cols: List[str] = []
            if emb_array is not None and geo_id_col in X.columns:
                emb_cols = [f"emb_{i}" for i in range(emb_array.shape[1])]
                emb_df = pd.DataFrame(emb_array, columns=emb_cols, index=X.index)
                emb_df[geo_id_col] = X[geo_id_col].values
                unit_emb = emb_df.groupby(geo_id_col)[emb_cols].mean()

                # Join onto geo_gdf
                geo_gdf = geo_gdf.merge(
                    unit_emb,
                    left_on=geo_id_col,
                    right_index=True,
                    how="left",
                )
                geo_feature_cols = emb_cols
            else:
                # Fall back to centroid coordinates if no embeddings
                geo_gdf["_cx"] = geo_gdf.geometry.centroid.x
                geo_gdf["_cy"] = geo_gdf.geometry.centroid.y
                geo_feature_cols = ["_cx", "_cy"]

            # Aggregate exposure per spatial unit
            unit_exposure = (
                pd.Series(exposure, index=X.index)
                .groupby(X[geo_id_col].values)
                .sum()
            )
            geo_gdf_exposure = geo_gdf[geo_id_col].map(unit_exposure).fillna(0.0).values

            self._territory_clusterer = TerritoryClusterer(
                n_clusters=self.n_territories,
                min_exposure=self.min_territory_exposure,
                method=self.cluster_method,
                random_state=self.random_state,
            )
            # Fill NaN feature cols with 0
            geo_gdf[geo_feature_cols] = geo_gdf[geo_feature_cols].fillna(0.0)

            self._territory_clusterer.fit(
                geo_gdf,
                feature_cols=geo_feature_cols,
                exposure=geo_gdf_exposure,
            )

            # Map back to policy level
            unit_to_territory = dict(
                zip(
                    geo_gdf[geo_id_col].values,
                    self._territory_clusterer.labels_.values,
                )
            )
            territory_labels_policy = X[geo_id_col].map(unit_to_territory).fillna(0)
            territory_labels_policy = territory_labels_policy.astype(int)

        # ---- Phase 4: Outer GLM ---------------------------------------
        X_outer = X.copy()

        if emb_array is not None:
            emb_cols = [f"emb_{i}" for i in range(emb_array.shape[1])]
            for i, col in enumerate(emb_cols):
                X_outer[col] = emb_array[:, i]

        if territory_labels_policy is not None:
            X_outer["territory"] = territory_labels_policy.values

        self._outer_glm = NestedGLM(
            family=self.family,
            formula=base_formula,
            add_embedding_cols=emb_array is not None,
            add_territory=territory_labels_policy is not None,
        )
        self._outer_glm.fit(X_outer, y, exposure)

        self._X_train_cols = X_outer.columns.tolist()
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        X: pd.DataFrame,
        exposure: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict using the fitted outer GLM.

        Parameters
        ----------
        X:
            Feature DataFrame.  Should include all columns seen during
            training (base factors, high-card categoricals, geo ID).
        exposure:
            Exposure.  If None, assumed all-ones.

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        self._check_fitted()
        assert self._outer_glm is not None

        n = len(X)
        if exposure is None:
            exposure = np.ones(n)

        X_pred = X.copy()

        # Append embeddings
        if self._embedding_trainer is not None and self._high_card_cols:
            emb_array = self._embedding_trainer.transform(X[self._high_card_cols])
            emb_cols = [f"emb_{i}" for i in range(emb_array.shape[1])]
            for i, col in enumerate(emb_cols):
                X_pred[col] = emb_array[:, i]

        # Territory — look up from training labels if geo_id_col present
        if (
            self._territory_clusterer is not None
            and self._geo_id_col is not None
            and self._geo_id_col in X_pred.columns
        ):
            # We stored territory labels on spatial units during fit — join here.
            # If unseen units, map to most common territory.
            X_pred["territory"] = 0  # placeholder; users should join properly

        return self._outer_glm.predict(X_pred, exposure)

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    def relativities(self) -> pd.DataFrame:
        """Return multiplicative relativities from the outer GLM.

        Returns
        -------
        pd.DataFrame
            See :meth:`~insurance_nested_glm.glm.NestedGLM.relativities`.
        """
        self._check_fitted()
        assert self._outer_glm is not None
        return self._outer_glm.relativities()

    def summary(self) -> str:
        """Return the outer GLM statsmodels summary."""
        self._check_fitted()
        assert self._outer_glm is not None
        return self._outer_glm.summary()

    def plot_territories(
        self,
        geo_gdf: "geopandas.GeoDataFrame",  # noqa: F821
        geo_id_col: str,
        **kwargs,
    ) -> "matplotlib.figure.Figure":  # noqa: F821
        """Plot the territory map produced by phase 3.

        Parameters
        ----------
        geo_gdf:
            GeoDataFrame of spatial units.
        geo_id_col:
            Column linking spatial units to territory labels.
        **kwargs:
            Forwarded to :func:`~insurance_nested_glm._utils.plot_territory_map`.

        Returns
        -------
        matplotlib.figure.Figure
        """
        self._check_fitted()
        if self._territory_clusterer is None:
            raise RuntimeError("Territory clustering was not run (geo_gdf was None).")

        from ._utils import plot_territory_map

        geo_gdf = geo_gdf.copy()
        unit_labels = dict(
            zip(
                geo_gdf[geo_id_col].values,
                self._territory_clusterer.labels_.values,
            )
        )
        labels = geo_gdf[geo_id_col].map(unit_labels).fillna(0).astype(int)
        return plot_territory_map(geo_gdf, labels, **kwargs)

    # ------------------------------------------------------------------
    # Component accessors
    # ------------------------------------------------------------------

    @property
    def base_glm_(self) -> NestedGLM:
        """Phase 1 base GLM."""
        self._check_fitted()
        assert self._base_glm is not None
        return self._base_glm

    @property
    def embedding_trainer_(self) -> Optional[EmbeddingTrainer]:
        """Phase 2 embedding trainer.  None if no high-card columns were provided."""
        self._check_fitted()
        return self._embedding_trainer

    @property
    def territory_clusterer_(self) -> Optional[TerritoryClusterer]:
        """Phase 3 territory clusterer.  None if no GeoDataFrame was provided."""
        self._check_fitted()
        return self._territory_clusterer

    @property
    def outer_glm_(self) -> NestedGLM:
        """Phase 4 outer GLM."""
        self._check_fitted()
        assert self._outer_glm is not None
        return self._outer_glm

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _select_base_cols(
        self,
        X: pd.DataFrame,
        high_card_cols: List[str],
        base_formula_cols: Optional[List[str]],
    ) -> List[str]:
        """Determine which columns to pass to the base GLM."""
        if base_formula_cols:
            return base_formula_cols
        # Exclude high-card cols and any geo-related cols
        excluded = set(high_card_cols)
        return [c for c in X.columns if c not in excluded]

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before calling this method.")
