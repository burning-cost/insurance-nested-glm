# insurance-nested-glm

GLM ratemaking is well understood. The problem is what to do with the variables that don't fit cleanly into it: vehicle make/model has thousands of levels, postcode sector has even more, and the standard GLM response — group them or drop them — throws away real signal.

This library implements the nested GLM framework from Wang, Shi, Cao (NAAJ 2025). The idea is a four-phase pipeline:

1. Fit a base GLM on the structured factors you trust (age band, NCD, vehicle group, etc.).
2. Train a shallow neural network with entity embeddings to encode the high-cardinality categoricals. The base GLM log-prediction enters as an offset — the network learns a correction, not a replacement.
3. Cluster spatial units (postcode sectors, output areas) into territory bands using spatially constrained clustering (SKATER), fed by the learned embedding coordinates. Every territory is geographically contiguous by construction.
4. Fit an outer GLM on the structured factors, the embedding vectors (as continuous regressors), and the territory fixed effect. The result is a GLM you can read — relativities table, deviance, AIC — not a black box.

The full cycle runs in a single `pipeline.fit()` call.

## Install

Core (requires PyTorch and statsmodels):

```bash
pip install insurance-nested-glm
```

With spatial clustering (geopandas, libpysal, spopt):

```bash
pip install insurance-nested-glm[spatial]
```

With plotting:

```bash
pip install insurance-nested-glm[plot]
```

Everything:

```bash
pip install insurance-nested-glm[all]
```

## Quick start

```python
import pandas as pd
import numpy as np
from insurance_nested_glm import NestedGLMPipeline

# policies: one row per policy
df = pd.read_parquet("policies.parquet")
y = df["claim_count"].to_numpy()
exposure = df["earned_exposure"].to_numpy()

pipeline = NestedGLMPipeline(
    base_formula="age_band + ncb + vehicle_group",
    family="poisson",
    n_territories=200,
    min_territory_exposure=500,
    embedding_epochs=50,
)

pipeline.fit(
    df,
    y,
    exposure,
    high_card_cols=["vehicle_make_model"],
    base_formula_cols=["age_band", "ncb", "vehicle_group"],
)

# Multiplicative relativities — readable like a standard GLM
print(pipeline.relativities())

# Predictions
pred = pipeline.predict(df, exposure)
```

### With spatial clustering

```python
import geopandas as gpd

# geo_gdf: one row per postcode sector with polygon geometries
geo_gdf = gpd.read_file("postcode_sectors.gpkg")

pipeline.fit(
    df,
    y,
    exposure,
    geo_gdf=geo_gdf,
    geo_id_col="postcode_sector",
    high_card_cols=["vehicle_make_model"],
    base_formula_cols=["age_band", "ncb"],
)

fig = pipeline.plot_territories(geo_gdf, geo_id_col="postcode_sector")
fig.savefig("territories.png", dpi=150)
```

## API

### NestedGLMPipeline

The main entry point. Parameters:

| Parameter | Default | Notes |
|---|---|---|
| `base_formula` | `None` | Patsy rhs formula for structured base GLM |
| `family` | `'poisson'` | `'poisson'` or `'gamma'` |
| `n_territories` | `200` | Target territory count |
| `min_territory_exposure` | `None` | Credibility filter: merge territories below this exposure |
| `embedding_epochs` | `50` | Training epochs for embedding network |
| `embedding_hidden_sizes` | `(64,)` | Dense layer sizes in embedding net |
| `embedding_lr` | `1e-3` | Adam learning rate |
| `cluster_method` | `'skater'` | `'skater'` or `'maxp'` |

### EmbeddingTrainer

If you want to use the embedding step in isolation:

```python
from insurance_nested_glm import EmbeddingTrainer

trainer = EmbeddingTrainer(
    cat_cols=["vehicle_make_model"],
    epochs=50,
    hidden_sizes=(64, 32),
)
trainer.fit(df, y, exposure=exposure, offset=base_log_pred)

# Dense vectors, shape (n, total_embedding_dim)
emb = trainer.transform(df)

# DataFrame: category → embedding coordinates, one per col
frames = trainer.get_embedding_frame()
print(frames["vehicle_make_model"].head())
```

Embedding dimension defaults to `min(50, ceil(n_levels / 2))` per column. Override with `embedding_dims={"vehicle_make_model": 20}`.

### TerritoryClusterer

```python
from insurance_nested_glm import TerritoryClusterer

tc = TerritoryClusterer(n_clusters=200, min_exposure=500, method="skater")
tc.fit(geo_gdf, feature_cols=["emb_0", "emb_1", ...], exposure=unit_exposure)

# pd.Series of 1-indexed territory labels, aligned with geo_gdf
print(tc.labels_)
```

Island handling: disconnected components in the adjacency graph (Channel Islands, Isle of Man, Orkney, Shetland) are detected automatically and clustered independently.

### NestedGLM

The outer GLM, available separately:

```python
from insurance_nested_glm import NestedGLM

glm = NestedGLM(family="poisson", formula="age_band + ncb")
glm.fit(X_with_embeddings_and_territory, y, exposure)

print(glm.relativities())
print(glm.aic(), glm.bic())
```

### Utility functions

```python
from insurance_nested_glm import credibility_report, build_adjacency

# Exposure / claims summary per territory
report = credibility_report(labels, exposure, claims)

# Build Queen contiguity weights directly
w = build_adjacency(geo_gdf)
```

## Design notes

**Why CANN-style offset?** The embedding network takes the base GLM log-prediction as an offset. This means the structured factors are not re-learned from scratch — the network only corrects what the GLM misses. It trains faster and is less prone to overfitting on the high-cardinality features.

**Why SKATER for territories?** SKATER (Spatial K-luster Analysis by Tree Edge Removal) builds a minimum spanning tree over the spatial units and prunes edges to form k subtrees. Every territory is a connected subgraph, which is a regulatory requirement in UK motor pricing. MaxP is available as an alternative for threshold-based approaches.

**Why statsmodels for the outer GLM?** Because pricing teams need the coefficient table, standard errors, LRT results, and AIC. A `sklearn` model gives you none of that. The outer GLM wraps `statsmodels.formula.api.glm` and exposes `relativities()` in multiplicative form.

**Embedding dimension heuristic.** `min(50, ceil(n_levels / 2))` follows the standard entity embedding rule of thumb from Guo and Berkhahn (2016). Override it if you have a reason to.

## References

Wang R, Shi H, Cao J (2025). A Nested GLM Framework with Neural Network Encoding and Spatially Constrained Clustering in Non-Life Insurance Ratemaking. *North American Actuarial Journal*, 29(3).

Guo C, Berkhahn F (2016). Entity Embeddings of Categorical Variables. *arXiv:1604.06737*.

Asselman D, Schelldorfer J, Wüthrich M V (2022). CANN: Combined Actuarial Neural Networks. *SSRN*.

## License

MIT. See [LICENSE](LICENSE).
