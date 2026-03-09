"""Outer GLM that combines structured factors, entity embeddings, and territory bands.

This is the final stage of the nested GLM pipeline.  It uses ``statsmodels``
GLM so that standard actuarial outputs — coefficient table, LRT, AIC, BIC,
and multiplicative relativities — are all available.

Poisson (frequency) and Gamma (severity) families are supported.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper


_Family = Literal["poisson", "gamma"]


class NestedGLM:
    """Final outer GLM in the nested GLM pipeline.

    Assembles a design matrix from:

    * Base structured factors (specified via Patsy formula or column list).
    * Embedding vectors from :class:`~insurance_nested_glm.embedding.EmbeddingTrainer`
      (entered as continuous regressors — one column per embedding dimension).
    * Territory factor from :class:`~insurance_nested_glm.territory.TerritoryClusterer`
      (entered as a categorical fixed effect).

    Then fits a log-link GLM using ``statsmodels``.

    Parameters
    ----------
    family:
        ``'poisson'`` for frequency models; ``'gamma'`` for severity.
    formula:
        Patsy formula for the structured base factors.  If None, a simple
        intercept-only model is used.  Do *not* include embedding columns or
        ``territory`` in the formula — they are appended automatically.
    add_embedding_cols:
        If True, embedding columns are appended as continuous regressors.
    add_territory:
        If True, the territory column is added as a categorical fixed effect.
    """

    def __init__(
        self,
        family: _Family = "poisson",
        formula: Optional[str] = None,
        add_embedding_cols: bool = True,
        add_territory: bool = True,
    ) -> None:
        self.family = family
        self.formula = formula
        self.add_embedding_cols = add_embedding_cols
        self.add_territory = add_territory

        self._result: Optional[GLMResultsWrapper] = None
        self._design_cols: List[str] = []
        self._emb_cols: List[str] = []
        self._fitted = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        exposure: Optional[np.ndarray] = None,
    ) -> "NestedGLM":
        """Fit the GLM.

        Parameters
        ----------
        X:
            Feature DataFrame.  Should contain the base structured columns,
            any embedding columns (named ``emb_*``), and optionally a column
            named ``territory``.
        y:
            Observed response (claim count for Poisson; average severity for
            Gamma).
        exposure:
            Policy exposure.  Used as GLM offset (log(exposure)).  If None,
            assumed all-ones.

        Returns
        -------
        NestedGLM
            self
        """
        X = X.copy()
        y = np.asarray(y, dtype=float)
        n = len(y)

        if exposure is None:
            exposure = np.ones(n)
        exposure = np.asarray(exposure, dtype=float)

        # Identify embedding columns
        self._emb_cols = [c for c in X.columns if c.startswith("emb_")]

        # Build formula
        formula_terms: List[str] = []

        if self.formula:
            # User-supplied base formula (rhs only, e.g. "age_band + vehicle_group")
            base_rhs = self.formula.strip()
        else:
            base_rhs = "1"

        formula_terms.append(base_rhs)

        if self.add_embedding_cols and self._emb_cols:
            formula_terms.append(" + ".join(self._emb_cols))

        if self.add_territory and "territory" in X.columns:
            # Ensure territory is treated as categorical
            X["territory"] = X["territory"].astype("category")
            formula_terms.append("C(territory)")

        full_rhs = " + ".join(formula_terms)
        full_formula = f"y ~ {full_rhs}"

        df_fit = X.copy()
        df_fit["y"] = y
        df_fit["log_exposure"] = np.log(exposure.clip(min=1e-10))

        family_obj = (
            sm.families.Poisson(link=sm.families.links.Log())
            if self.family == "poisson"
            else sm.families.Gamma(link=sm.families.links.Log())
        )

        model = smf.glm(
            formula=full_formula,
            data=df_fit,
            family=family_obj,
            offset=df_fit["log_exposure"],
        )
        self._result = model.fit(disp=False)
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
        """Return predicted mean response (frequency or severity).

        Parameters
        ----------
        X:
            Feature DataFrame with same columns as training data.
        exposure:
            Exposure.  If None, assumed all-ones.

        Returns
        -------
        np.ndarray
            Predicted values, shape ``(n,)``.
        """
        self._check_fitted()
        assert self._result is not None

        n = len(X)
        if exposure is None:
            exposure = np.ones(n)
        exposure = np.asarray(exposure, dtype=float)

        X = X.copy()
        if "territory" in X.columns:
            X["territory"] = X["territory"].astype("category")

        X["log_exposure"] = np.log(exposure.clip(min=1e-10))
        X["y"] = 0.0  # dummy — not used in predict

        return self._result.predict(X).to_numpy()

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return the statsmodels GLM summary as a string."""
        self._check_fitted()
        assert self._result is not None
        return str(self._result.summary())

    def relativities(self) -> pd.DataFrame:
        """Return multiplicative relativities for all model terms.

        Exponentiates the log-link coefficients.  The intercept is included
        as the base rate.

        Returns
        -------
        pd.DataFrame
            Columns: ``['term', 'coefficient', 'relativity', 'std_err',
            'z', 'p_value', 'ci_lower', 'ci_upper']``.
        """
        self._check_fitted()
        assert self._result is not None

        params = self._result.params
        bse = self._result.bse
        tvalues = self._result.tvalues
        pvalues = self._result.pvalues
        conf = self._result.conf_int()

        rows = []
        for term in params.index:
            coef = params[term]
            rows.append(
                {
                    "term": term,
                    "coefficient": coef,
                    "relativity": np.exp(coef),
                    "std_err": bse[term],
                    "z": tvalues[term],
                    "p_value": pvalues[term],
                    "ci_lower": np.exp(conf.loc[term, 0]),
                    "ci_upper": np.exp(conf.loc[term, 1]),
                }
            )

        return pd.DataFrame(rows)

    def aic(self) -> float:
        """Akaike Information Criterion."""
        self._check_fitted()
        assert self._result is not None
        return float(self._result.aic)

    def bic(self) -> float:
        """Bayesian Information Criterion."""
        self._check_fitted()
        assert self._result is not None
        return float(self._result.bic)

    def deviance(self) -> float:
        """Residual deviance."""
        self._check_fitted()
        assert self._result is not None
        return float(self._result.deviance)

    @property
    def result_(self) -> GLMResultsWrapper:
        """Underlying ``statsmodels`` GLMResultsWrapper."""
        self._check_fitted()
        assert self._result is not None
        return self._result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before calling this method.")
