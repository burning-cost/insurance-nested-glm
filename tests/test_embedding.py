"""Tests for insurance_nested_glm.embedding."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from insurance_nested_glm.embedding import (
    EmbeddingNet,
    EmbeddingTrainer,
    _default_embedding_dim,
    _poisson_deviance_loss,
)


# ---------------------------------------------------------------------------
# _default_embedding_dim
# ---------------------------------------------------------------------------


def test_embedding_dim_small():
    assert _default_embedding_dim(4) == 2


def test_embedding_dim_cap():
    # Very large n_levels should be capped at 50
    assert _default_embedding_dim(1000) == 50


def test_embedding_dim_three():
    # ceil(3/2) = 2
    assert _default_embedding_dim(3) == 2


# ---------------------------------------------------------------------------
# EmbeddingNet — construction
# ---------------------------------------------------------------------------


def test_embedding_net_construction():
    net = EmbeddingNet(
        vocab_sizes={"make": 10, "model": 50},
        embedding_dims={"make": 3, "model": 8},
        hidden_sizes=(16,),
    )
    assert "make" in net.embeddings
    assert "model" in net.embeddings
    assert net.embedding_dims["make"] == 3
    assert net.embedding_dims["model"] == 8


def test_embedding_net_default_dims():
    net = EmbeddingNet(vocab_sizes={"make": 10})
    # min(50, ceil(10/2)) = 5
    assert net.embedding_dims["make"] == 5


def test_embedding_net_empty_vocab_raises():
    with pytest.raises(ValueError, match="vocab_sizes"):
        EmbeddingNet(vocab_sizes={})


# ---------------------------------------------------------------------------
# EmbeddingNet — forward pass
# ---------------------------------------------------------------------------


def test_embedding_net_forward_shape():
    torch.manual_seed(0)
    net = EmbeddingNet(
        vocab_sizes={"make": 5, "model": 8},
        embedding_dims={"make": 2, "model": 3},
        hidden_sizes=(16,),
    )
    batch = 12
    cat_inputs = {
        "make": torch.randint(0, 5, (batch,)),
        "model": torch.randint(0, 8, (batch,)),
    }
    out = net(cat_inputs)
    assert out.shape == (batch,)


def test_embedding_net_forward_with_offset():
    torch.manual_seed(0)
    net = EmbeddingNet(
        vocab_sizes={"make": 5},
        embedding_dims={"make": 2},
        hidden_sizes=(8,),
    )
    batch = 6
    cat_inputs = {"make": torch.zeros(batch, dtype=torch.long)}
    offset = torch.ones(batch) * 2.0
    out_no_offset = net(cat_inputs)
    out_with_offset = net(cat_inputs, offset=offset)
    # With offset, output should differ from without offset
    assert not torch.allclose(out_no_offset, out_with_offset)
    # Difference should be approximately 2.0 (the offset) since no nonlinearity after
    diff = out_with_offset - out_no_offset
    assert torch.allclose(diff, torch.ones(batch) * 2.0, atol=1e-5)


def test_embedding_net_get_embedding_weights():
    net = EmbeddingNet(
        vocab_sizes={"make": 5},
        embedding_dims={"make": 3},
        hidden_sizes=(8,),
    )
    weights = net.get_embedding_weights()
    assert "make" in weights
    assert weights["make"].shape == (5, 3)
    assert isinstance(weights["make"], np.ndarray)


# ---------------------------------------------------------------------------
# _poisson_deviance_loss
# ---------------------------------------------------------------------------


def test_poisson_deviance_loss_zero_y():
    # When y=0 the loss should be finite and non-negative
    log_pred = torch.zeros(5)
    y = torch.zeros(5)
    exp = torch.ones(5)
    loss = _poisson_deviance_loss(log_pred, y, exp)
    assert loss.item() >= 0
    assert torch.isfinite(loss)


def test_poisson_deviance_loss_positive():
    log_pred = torch.tensor([0.5, 1.0, -0.3])
    y = torch.tensor([1.0, 2.0, 0.5])
    exp = torch.tensor([1.0, 1.0, 1.0])
    loss = _poisson_deviance_loss(log_pred, y, exp)
    assert loss.item() >= 0
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# EmbeddingTrainer — fit / transform
# ---------------------------------------------------------------------------


def _make_trainer_data(n: int = 200, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    makes = ["Ford", "Vauxhall", "Toyota", "BMW", "Audi"]
    models = ["A", "B", "C", "D", "E", "F", "G", "H"]
    df = pd.DataFrame(
        {
            "make": rng.choice(makes, n),
            "model": rng.choice(models, n),
        }
    )
    exposure = rng.uniform(0.5, 1.5, n).astype(np.float32)
    # Poisson claims
    lam = 0.1 * exposure
    y = rng.poisson(lam).astype(np.float32)
    return df, y, exposure


def test_embedding_trainer_fit_transform():
    df, y, exposure = _make_trainer_data()
    trainer = EmbeddingTrainer(
        cat_cols=["make", "model"],
        hidden_sizes=(16,),
        epochs=3,
        batch_size=64,
        random_state=42,
    )
    trainer.fit(df, y, exposure=exposure)
    assert trainer._fitted

    emb = trainer.transform(df)
    assert emb.shape == (len(df), trainer.total_embedding_dim)
    assert np.all(np.isfinite(emb))


def test_embedding_trainer_with_offset():
    df, y, exposure = _make_trainer_data()
    offset = np.log(exposure * 0.1)
    trainer = EmbeddingTrainer(
        cat_cols=["make"],
        hidden_sizes=(8,),
        epochs=2,
        random_state=0,
    )
    trainer.fit(df, y, exposure=exposure, offset=offset)
    emb = trainer.transform(df)
    assert emb.shape[0] == len(df)


def test_embedding_trainer_get_embedding_frame():
    df, y, _ = _make_trainer_data()
    trainer = EmbeddingTrainer(
        cat_cols=["make"],
        hidden_sizes=(8,),
        epochs=2,
        random_state=0,
    )
    trainer.fit(df, y)
    frames = trainer.get_embedding_frame()
    assert "make" in frames
    frame = frames["make"]
    assert "category" in frame.columns
    assert frame.shape[0] == df["make"].nunique()


def test_embedding_trainer_unfitted_raises():
    trainer = EmbeddingTrainer(cat_cols=["make"])
    with pytest.raises(RuntimeError, match="fit()"):
        trainer.transform(pd.DataFrame({"make": ["Ford"]}))


def test_embedding_trainer_unseen_category_warns():
    df, y, exposure = _make_trainer_data()
    trainer = EmbeddingTrainer(
        cat_cols=["make"],
        hidden_sizes=(8,),
        epochs=2,
        random_state=0,
    )
    trainer.fit(df, y, exposure=exposure)

    unseen_df = pd.DataFrame({"make": ["UnknownBrand"] * 5})
    with pytest.warns(UserWarning, match="unseen"):
        trainer.transform(unseen_df)


def test_embedding_trainer_single_level():
    """Single-level categorical should not crash."""
    df = pd.DataFrame({"make": ["Ford"] * 50})
    y = np.ones(50) * 0.1
    trainer = EmbeddingTrainer(cat_cols=["make"], epochs=2, hidden_sizes=(4,))
    trainer.fit(df, y)
    emb = trainer.transform(df)
    # min(50, ceil(1/2)) = 1
    assert emb.shape == (50, 1)


def test_embedding_trainer_total_dim():
    df, y, _ = _make_trainer_data()
    trainer = EmbeddingTrainer(
        cat_cols=["make", "model"],
        embedding_dims={"make": 3, "model": 4},
        hidden_sizes=(8,),
        epochs=2,
    )
    trainer.fit(df, y)
    assert trainer.total_embedding_dim == 7
