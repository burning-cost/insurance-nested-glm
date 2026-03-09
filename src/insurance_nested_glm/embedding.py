"""Neural network entity embeddings for high-cardinality categorical variables.

Implements the CANN-style offset architecture from Wang, Shi, Cao (NAAJ 2025):
the base GLM log-prediction is passed as an offset into the network so the
neural net learns a correction rather than re-learning the base structure.

Embedding dimension default follows the rule of thumb from the entity embedding
literature: min(50, ceil(n_levels / 2)).
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder


def _default_embedding_dim(n_levels: int) -> int:
    """Return embedding dimension for a categorical with *n_levels* unique values.

    Uses the rule of thumb from the entity embedding literature:
    ``min(50, ceil(n_levels / 2))``.
    """
    return min(50, math.ceil(n_levels / 2))


class EmbeddingNet(nn.Module):
    """PyTorch network that maps high-cardinality categoricals to a log-frequency prediction.

    Architecture:

    * One ``nn.Embedding`` per high-cardinality column.
    * All embedding vectors concatenated, then passed through one or more dense
      layers with ReLU activations.
    * A skip connection carries the base GLM log-prediction (offset) directly
      to the output neuron — CANN style.  The network therefore learns a
      residual correction on top of the structured GLM, not a replacement.
    * Output is log-scale (i.e. the model predicts log(mu)).  Poisson deviance
      loss is used during training.

    Parameters
    ----------
    vocab_sizes:
        Mapping of column name to number of unique levels (post label-encoding).
    embedding_dims:
        Mapping of column name to embedding dimension.  Defaults to
        ``_default_embedding_dim(vocab_size)`` for each column.
    hidden_sizes:
        Sizes of hidden dense layers between the concatenated embeddings and the
        output.  Default is a single layer of size 64.
    dropout:
        Dropout probability applied after each hidden layer.  0 disables it.
    """

    def __init__(
        self,
        vocab_sizes: Dict[str, int],
        embedding_dims: Optional[Dict[str, int]] = None,
        hidden_sizes: Sequence[int] = (64,),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if not vocab_sizes:
            raise ValueError("vocab_sizes must contain at least one column.")

        self.col_names: List[str] = sorted(vocab_sizes.keys())

        # Build embedding tables
        if embedding_dims is None:
            embedding_dims = {
                col: _default_embedding_dim(vocab_sizes[col])
                for col in self.col_names
            }

        self.embeddings = nn.ModuleDict(
            {
                col: nn.Embedding(vocab_sizes[col], embedding_dims[col])
                for col in self.col_names
            }
        )

        total_emb_dim = sum(embedding_dims[col] for col in self.col_names)

        # Dense layers after concatenated embeddings
        layers: List[nn.Module] = []
        in_dim = total_emb_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h

        self.dense = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, 1)

        # Store dims for downstream use
        self.embedding_dims = embedding_dims
        self.vocab_sizes = vocab_sizes

    def forward(
        self,
        cat_inputs: Dict[str, torch.Tensor],
        offset: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        cat_inputs:
            Mapping of column name to 1-D integer tensor of label-encoded
            category indices.  Shape: ``(batch,)`` per column.
        offset:
            Base GLM log-prediction tensor of shape ``(batch,)``.  If provided,
            added to the network output (CANN skip connection).

        Returns
        -------
        torch.Tensor
            Log-scale predictions, shape ``(batch,)``.
        """
        emb_vecs = [
            self.embeddings[col](cat_inputs[col]) for col in self.col_names
        ]
        x = torch.cat(emb_vecs, dim=-1)  # (batch, total_emb_dim)
        x = self.dense(x)
        log_pred = self.output_layer(x).squeeze(-1)  # (batch,)

        if offset is not None:
            log_pred = log_pred + offset

        return log_pred

    def get_embedding_weights(self) -> Dict[str, np.ndarray]:
        """Return the trained embedding weight matrices as numpy arrays.

        Returns
        -------
        dict
            Mapping of column name to array of shape
            ``(n_levels, embedding_dim)``.
        """
        return {
            col: self.embeddings[col].weight.detach().cpu().numpy()
            for col in self.col_names
        }


def _poisson_deviance_loss(
    log_pred: torch.Tensor,
    y: torch.Tensor,
    exposure: torch.Tensor,
) -> torch.Tensor:
    """Poisson deviance loss with exposure offset.

    D = 2 * sum(y * log(y / mu) - (y - mu))  where mu = exposure * exp(log_pred).

    A numerically stable form is used that avoids log(0) when y == 0.
    """
    log_mu = log_pred + torch.log(exposure.clamp(min=1e-10))
    mu = torch.exp(log_mu)

    # 2*(y*log(y/mu) - (y - mu))
    # When y==0 the first term is 0.
    term1 = torch.where(y > 0, y * (torch.log(y.clamp(min=1e-10)) - log_mu), torch.zeros_like(y))
    term2 = y - mu
    return 2.0 * (term1 - term2).mean()


class EmbeddingTrainer:
    """Train entity embeddings on GLM residuals and expose the learned vectors.

    This class handles the full preprocessing-train-extract cycle:

    1. Label-encodes each high-cardinality categorical column.
    2. Trains :class:`EmbeddingNet` on target values ``y`` with optional
       ``offset`` (base GLM log-predictions) and ``exposure``.
    3. Exposes trained embedding vectors via :meth:`transform` and
       :meth:`get_embedding_frame`.

    Parameters
    ----------
    cat_cols:
        Names of high-cardinality categorical columns to embed.
    embedding_dims:
        Override embedding dimensions per column.  Defaults to
        ``min(50, ceil(n_levels / 2))`` per column.
    hidden_sizes:
        Hidden layer sizes for the dense network.
    dropout:
        Dropout rate.
    lr:
        Learning rate for Adam optimiser.
    epochs:
        Number of training epochs.
    batch_size:
        Mini-batch size.
    device:
        ``'cpu'`` or ``'cuda'``.  Defaults to ``'cpu'``.
    random_state:
        Seed for reproducibility.
    """

    def __init__(
        self,
        cat_cols: List[str],
        embedding_dims: Optional[Dict[str, int]] = None,
        hidden_sizes: Sequence[int] = (64,),
        dropout: float = 0.0,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 1024,
        device: str = "cpu",
        random_state: int = 42,
    ) -> None:
        self.cat_cols = cat_cols
        self.embedding_dims_override = embedding_dims
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.random_state = random_state

        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._vocab_sizes: Dict[str, int] = {}
        self._net: Optional[EmbeddingNet] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        exposure: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> "EmbeddingTrainer":
        """Fit label encoders and train embedding network.

        Parameters
        ----------
        X:
            Feature DataFrame.  Must contain all columns in ``cat_cols``.
        y:
            Observed claim counts (or severity).  1-D array, length n.
        exposure:
            Policy exposure in years.  If None, assumed to be all-ones.
        offset:
            Base GLM log-predictions (log-scale).  If None, no skip connection
            is used.

        Returns
        -------
        EmbeddingTrainer
            self
        """
        torch.manual_seed(self.random_state)
        n = len(y)

        if exposure is None:
            exposure = np.ones(n, dtype=np.float32)
        else:
            exposure = np.asarray(exposure, dtype=np.float32)

        y = np.asarray(y, dtype=np.float32)

        # Label-encode categoricals
        encoded: Dict[str, np.ndarray] = {}
        for col in self.cat_cols:
            le = LabelEncoder()
            encoded[col] = le.fit_transform(X[col].astype(str)).astype(np.int64)
            self._label_encoders[col] = le
            self._vocab_sizes[col] = len(le.classes_)

        # Build network
        self._net = EmbeddingNet(
            vocab_sizes=self._vocab_sizes,
            embedding_dims=self.embedding_dims_override,
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout,
        ).to(self.device)

        optimiser = torch.optim.Adam(self._net.parameters(), lr=self.lr)

        # Build tensors
        y_t = torch.tensor(y, device=self.device)
        exp_t = torch.tensor(exposure, device=self.device)
        offset_t = (
            torch.tensor(offset.astype(np.float32), device=self.device)
            if offset is not None
            else None
        )
        cat_tensors = {
            col: torch.tensor(encoded[col], device=self.device)
            for col in self.cat_cols
        }

        # Training loop
        self._net.train()
        idx = np.arange(n)
        rng = np.random.default_rng(self.random_state)

        for epoch in range(self.epochs):
            rng.shuffle(idx)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, self.batch_size):
                batch_idx = idx[start : start + self.batch_size]
                b_idx_t = torch.tensor(batch_idx, device=self.device)

                b_cat = {col: cat_tensors[col][b_idx_t] for col in self.cat_cols}
                b_y = y_t[b_idx_t]
                b_exp = exp_t[b_idx_t]
                b_off = offset_t[b_idx_t] if offset_t is not None else None

                optimiser.zero_grad()
                log_pred = self._net(b_cat, b_off)
                loss = _poisson_deviance_loss(log_pred, b_y, b_exp)
                loss.backward()
                optimiser.step()

                epoch_loss += loss.item()
                n_batches += 1

        self._net.eval()
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Map each row to its concatenated embedding vector.

        Unknown categories (unseen during fit) are mapped to index 0.

        Parameters
        ----------
        X:
            DataFrame containing the high-cardinality categorical columns.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_rows, total_embedding_dim)``.
        """
        self._check_fitted()
        encoded = self._encode(X)
        cat_tensors = {
            col: torch.tensor(encoded[col], device=self.device)
            for col in self.cat_cols
        }
        with torch.no_grad():
            emb_vecs = [
                self._net.embeddings[col](cat_tensors[col]).cpu().numpy()  # type: ignore[union-attr]
                for col in self._net.col_names  # type: ignore[union-attr]
            ]
        return np.concatenate(emb_vecs, axis=1)

    def get_embedding_frame(self) -> Dict[str, pd.DataFrame]:
        """Return one DataFrame per column mapping category labels to embedding coordinates.

        Returns
        -------
        dict
            Mapping of column name to DataFrame with columns
            ``['category', 'emb_0', 'emb_1', ...]``.
        """
        self._check_fitted()
        weights = self._net.get_embedding_weights()  # type: ignore[union-attr]
        frames: Dict[str, pd.DataFrame] = {}
        for col in self.cat_cols:
            le = self._label_encoders[col]
            w = weights[col]
            dim = w.shape[1]
            df = pd.DataFrame(w, columns=[f"emb_{i}" for i in range(dim)])
            df.insert(0, "category", le.classes_)
            frames[col] = df
        return frames

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _encode(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Label-encode X, mapping unseen categories to 0."""
        encoded: Dict[str, np.ndarray] = {}
        for col in self.cat_cols:
            le = self._label_encoders[col]
            vals = X[col].astype(str).to_numpy()
            known_mask = np.isin(vals, le.classes_)
            if not known_mask.all():
                n_unknown = (~known_mask).sum()
                warnings.warn(
                    f"Column '{col}': {n_unknown} unseen categories mapped to index 0.",
                    UserWarning,
                    stacklevel=3,
                )
                vals = vals.copy()
                vals[~known_mask] = le.classes_[0]
            encoded[col] = le.transform(vals).astype(np.int64)
        return encoded

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

    @property
    def embedding_dims(self) -> Dict[str, int]:
        """Embedding dimensions per column after fitting."""
        self._check_fitted()
        return self._net.embedding_dims  # type: ignore[union-attr]

    @property
    def total_embedding_dim(self) -> int:
        """Total concatenated embedding dimension after fitting."""
        return sum(self.embedding_dims.values())
