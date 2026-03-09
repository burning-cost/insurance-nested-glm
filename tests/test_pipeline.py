"""Tests for insurance_nested_glm.pipeline (NestedGLMPipeline).

End-to-end pipeline tests.  The spatial phase is skipped unless geopandas and
spopt are installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_nested_glm.pipeline import NestedGLMPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_policy_data(n: int = 400, seed: int = 42):
    rng = np.random.default_rng(seed)
    makes = ["Ford", "Vauxhall", "Toyota", "BMW", "Audi", "Honda", "Nissan", "VW"]
    age_bands = ["17-25", "26-35", "36-50", "51-65", "66+"]
    ncb = [0, 1, 2, 3, 4, 5]

    df = pd.DataFrame(
        {
            "age_band": rng.choice(age_bands, n),
            "ncb": rng.choice(ncb, n),
            "vehicle_make": rng.choice(makes, n),
        }
    )
    exposure = rng.uniform(0.5, 1.5, n)
    lam = 0.08 * exposure
    y = rng.poisson(lam).astype(float)
    return df, y, exposure


def _make_geo_gdf(n_units: int = 9):
    """Make a simple grid GeoDataFrame for spatial testing."""
    pytest.importorskip("geopandas")
    import geopandas as gpd
    from shapely.geometry import box

    n_side = int(np.sqrt(n_units))
    geoms = []
    ids = []
    for r in range(n_side):
        for c in range(n_side):
            geoms.append(box(c, r, c + 1, r + 1))
            ids.append(f"sector_{r}_{c}")
    return gpd.GeoDataFrame({"sector_id": ids, "geometry": geoms}, crs="EPSG:27700")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_pipeline_defaults():
    p = NestedGLMPipeline()
    assert p.family == "poisson"
    assert p.n_territories == 200


def test_pipeline_unfitted_raises():
    p = NestedGLMPipeline()
    with pytest.raises(RuntimeError, match="fit()"):
        p.relativities()


# ---------------------------------------------------------------------------
# Phase 1 + 2: base GLM + embeddings (no spatial)
# ---------------------------------------------------------------------------


def test_pipeline_no_spatial():
    """Pipeline should run without geo_gdf (skips territory phase)."""
    df, y, exposure = _make_policy_data()
    p = NestedGLMPipeline(
        base_formula="age_band + ncb",
        embedding_epochs=2,
        embedding_hidden_sizes=(8,),
        random_state=0,
    )
    p.fit(
        df,
        y,
        exposure,
        high_card_cols=["vehicle_make"],
        base_formula_cols=["age_band", "ncb"],
    )

    assert p._fitted
    assert p.embedding_trainer_ is not None
    assert p.territory_clusterer_ is None


def test_pipeline_predict_no_spatial():
    df, y, exposure = _make_policy_data()
    p = NestedGLMPipeline(
        base_formula="age_band",
        embedding_epochs=2,
        embedding_hidden_sizes=(8,),
        random_state=0,
    )
    p.fit(df, y, exposure, high_card_cols=["vehicle_make"], base_formula_cols=["age_band"])
    preds = p.predict(df, exposure)
    assert preds.shape == (len(df),)
    assert np.all(np.isfinite(preds))
    assert np.all(preds >= 0)


def test_pipeline_relativities_no_spatial():
    df, y, exposure = _make_policy_data()
    p = NestedGLMPipeline(
        base_formula="age_band",
        embedding_epochs=2,
        embedding_hidden_sizes=(8,),
        random_state=0,
    )
    p.fit(df, y, exposure, high_card_cols=["vehicle_make"], base_formula_cols=["age_band"])
    rel = p.relativities()
    assert isinstance(rel, pd.DataFrame)
    assert "relativity" in rel.columns
    assert (rel["relativity"] > 0).all()


def test_pipeline_summary_no_spatial():
    df, y, exposure = _make_policy_data()
    p = NestedGLMPipeline(base_formula="age_band", embedding_epochs=2, random_state=0)
    p.fit(df, y, exposure, high_card_cols=["vehicle_make"], base_formula_cols=["age_band"])
    s = p.summary()
    assert isinstance(s, str)
    assert len(s) > 0


# ---------------------------------------------------------------------------
# No high-card cols (no embedding)
# ---------------------------------------------------------------------------


def test_pipeline_no_embeddings():
    df, y, exposure = _make_policy_data()
    p = NestedGLMPipeline(base_formula="age_band + ncb", random_state=0)
    p.fit(
        df,
        y,
        exposure,
        high_card_cols=None,
        base_formula_cols=["age_band", "ncb"],
    )
    assert p.embedding_trainer_ is None
    pred = p.predict(df, exposure)
    assert pred.shape == (len(df),)


# ---------------------------------------------------------------------------
# Spatial pipeline (requires geopandas + spopt)
# ---------------------------------------------------------------------------


def _spopt_available():
    try:
        import spopt  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _spopt_available(), reason="spopt not installed")
def test_pipeline_with_spatial():
    """Full four-phase pipeline with a small synthetic GeoDataFrame."""
    gdf = _make_geo_gdf(9)  # 3x3 grid → 9 sectors

    rng = np.random.default_rng(10)
    n = 200
    sectors = gdf["sector_id"].tolist()
    age_bands = ["A", "B", "C"]
    makes = ["Ford", "Vauxhall", "Toyota"]

    df = pd.DataFrame(
        {
            "age_band": rng.choice(age_bands, n),
            "vehicle_make": rng.choice(makes, n),
            "sector_id": rng.choice(sectors, n),
        }
    )
    exposure = rng.uniform(0.5, 1.5, n)
    y = rng.poisson(0.1 * exposure).astype(float)

    p = NestedGLMPipeline(
        base_formula="age_band",
        n_territories=3,
        embedding_epochs=2,
        embedding_hidden_sizes=(8,),
        random_state=0,
    )
    p.fit(
        df,
        y,
        exposure,
        geo_gdf=gdf,
        geo_id_col="sector_id",
        high_card_cols=["vehicle_make"],
        base_formula_cols=["age_band"],
    )

    assert p.territory_clusterer_ is not None
    labels = p.territory_clusterer_.labels_
    assert labels.nunique() <= 3
    assert labels.min() >= 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_pipeline_very_low_claims():
    """Very low but nonzero claims should not cause division errors."""
    rng = np.random.default_rng(99)
    df, _, exposure = _make_policy_data()
    # Very low frequency but not all-zero — avoids GLM perfect-separation issues
    y = rng.poisson(0.001 * exposure).astype(float)
    p = NestedGLMPipeline(base_formula="age_band", embedding_epochs=2, random_state=0)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p.fit(df, y, exposure, high_card_cols=["vehicle_make"], base_formula_cols=["age_band"])
    pred = p.predict(df, exposure)
    assert np.all(np.isfinite(pred))


def test_pipeline_component_accessors():
    df, y, exposure = _make_policy_data()
    p = NestedGLMPipeline(base_formula="age_band", embedding_epochs=2, random_state=0)
    p.fit(df, y, exposure, high_card_cols=["vehicle_make"], base_formula_cols=["age_band"])
    # Accessors should return the right types
    from insurance_nested_glm.glm import NestedGLM
    from insurance_nested_glm.embedding import EmbeddingTrainer

    assert isinstance(p.base_glm_, NestedGLM)
    assert isinstance(p.embedding_trainer_, EmbeddingTrainer)
    assert isinstance(p.outer_glm_, NestedGLM)
