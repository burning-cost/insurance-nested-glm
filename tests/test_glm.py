"""Tests for insurance_nested_glm.glm (NestedGLM)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_nested_glm.glm import NestedGLM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_glm_data(n: int = 300, seed: int = 42, add_territory: bool = True):
    rng = np.random.default_rng(seed)
    age_bands = ["17-25", "26-35", "36-50", "51-65", "66+"]
    veh_groups = ["1", "2", "3", "4", "5"]

    df = pd.DataFrame(
        {
            "age_band": rng.choice(age_bands, n),
            "vehicle_group": rng.choice(veh_groups, n),
            "emb_0": rng.normal(0, 1, n),
            "emb_1": rng.normal(0, 1, n),
        }
    )
    if add_territory:
        df["territory"] = rng.integers(1, 11, n)

    exposure = rng.uniform(0.5, 1.5, n)
    lam = 0.1 * exposure * np.exp(
        0.3 * (df["age_band"] == "17-25").astype(float)
        - 0.1 * df["emb_0"]
    )
    y = rng.poisson(lam).astype(float)
    return df, y, exposure


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_nested_glm_defaults():
    glm = NestedGLM()
    assert glm.family == "poisson"
    assert glm.add_embedding_cols is True
    assert glm.add_territory is True


def test_nested_glm_unfitted_raises():
    glm = NestedGLM()
    with pytest.raises(RuntimeError, match="fit()"):
        glm.summary()


# ---------------------------------------------------------------------------
# Fit / predict
# ---------------------------------------------------------------------------


def test_nested_glm_fit_poisson():
    df, y, exposure = _make_glm_data(add_territory=True)
    glm = NestedGLM(
        family="poisson",
        formula="age_band + vehicle_group",
        add_embedding_cols=True,
        add_territory=True,
    )
    glm.fit(df, y, exposure)
    assert glm._fitted
    assert glm.result_ is not None


def test_nested_glm_predict_shape():
    df, y, exposure = _make_glm_data()
    glm = NestedGLM(formula="age_band")
    glm.fit(df, y, exposure)
    pred = glm.predict(df, exposure)
    assert pred.shape == (len(df),)
    assert np.all(pred >= 0)


def test_nested_glm_predict_no_exposure():
    df, y, exposure = _make_glm_data()
    glm = NestedGLM(formula="age_band")
    glm.fit(df, y, exposure)
    pred = glm.predict(df)  # exposure=None
    assert pred.shape == (len(df),)


def test_nested_glm_no_territory():
    df, y, exposure = _make_glm_data(add_territory=False)
    glm = NestedGLM(formula="age_band", add_territory=False)
    glm.fit(df, y, exposure)
    pred = glm.predict(df, exposure)
    assert np.all(pred >= 0)


def test_nested_glm_gamma_family():
    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame({"age_band": rng.choice(["A", "B", "C"], n)})
    exposure = rng.uniform(0.5, 2, n)
    y = rng.gamma(shape=2.0, scale=500, size=n)
    glm = NestedGLM(family="gamma", formula="age_band", add_embedding_cols=False, add_territory=False)
    glm.fit(df, y, exposure)
    pred = glm.predict(df, exposure)
    assert np.all(pred > 0)


def test_nested_glm_no_embeddings():
    df, y, exposure = _make_glm_data()
    # Drop embedding columns
    df_base = df[["age_band", "vehicle_group", "territory"]]
    glm = NestedGLM(formula="age_band + vehicle_group", add_embedding_cols=False, add_territory=True)
    glm.fit(df_base, y, exposure)
    pred = glm.predict(df_base, exposure)
    assert pred.shape == (len(df),)


# ---------------------------------------------------------------------------
# relativities()
# ---------------------------------------------------------------------------


def test_nested_glm_relativities_structure():
    df, y, exposure = _make_glm_data(add_territory=False)
    glm = NestedGLM(formula="age_band", add_embedding_cols=True, add_territory=False)
    glm.fit(df, y, exposure)
    rel = glm.relativities()

    assert isinstance(rel, pd.DataFrame)
    expected_cols = {"term", "coefficient", "relativity", "std_err", "z", "p_value", "ci_lower", "ci_upper"}
    assert expected_cols.issubset(set(rel.columns))


def test_nested_glm_relativities_positive():
    df, y, exposure = _make_glm_data(add_territory=False)
    glm = NestedGLM(formula="age_band", add_embedding_cols=False, add_territory=False)
    glm.fit(df, y, exposure)
    rel = glm.relativities()
    # All relativities should be positive (exponentials)
    assert (rel["relativity"] > 0).all()
    # CI lower should be < CI upper
    assert (rel["ci_lower"] <= rel["ci_upper"]).all()


# ---------------------------------------------------------------------------
# AIC / BIC / deviance
# ---------------------------------------------------------------------------


def test_nested_glm_metrics():
    df, y, exposure = _make_glm_data(add_territory=False)
    glm = NestedGLM(formula="age_band", add_embedding_cols=False, add_territory=False)
    glm.fit(df, y, exposure)
    assert np.isfinite(glm.aic())
    assert np.isfinite(glm.bic())
    assert glm.deviance() >= 0


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------


def test_nested_glm_summary_string():
    df, y, exposure = _make_glm_data(add_territory=False)
    glm = NestedGLM(formula="age_band", add_embedding_cols=False, add_territory=False)
    glm.fit(df, y, exposure)
    s = glm.summary()
    assert isinstance(s, str)
    assert "age_band" in s.lower() or "Generalized" in s
