# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-nested-glm: Full workflow demo
# MAGIC
# MAGIC This notebook demonstrates the complete four-phase nested GLM pipeline on
# MAGIC synthetic UK motor insurance data.
# MAGIC
# MAGIC **Pipeline phases:**
# MAGIC 1. Base GLM on structured rating factors
# MAGIC 2. Entity embedding of high-cardinality categoricals (vehicle make/model)
# MAGIC 3. Spatially constrained territory clustering (SKATER)
# MAGIC 4. Outer GLM combining all features — readable relativities table
# MAGIC
# MAGIC Reference: Wang, Shi, Cao (NAAJ 2025)

# COMMAND ----------

# MAGIC %pip install insurance-nested-glm[all] --quiet

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from insurance_nested_glm import (
    NestedGLMPipeline,
    EmbeddingTrainer,
    TerritoryClusterer,
    NestedGLM,
    credibility_report,
    embedding_pca_plot,
)

print("insurance-nested-glm imported successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Generate synthetic UK motor data

# COMMAND ----------

rng = np.random.default_rng(2025)

N_POLICIES = 5_000

# Vehicle makes — realistic UK distribution (high cardinality)
makes_common = ["Ford", "Vauxhall", "Volkswagen", "BMW", "Mercedes", "Toyota",
                "Honda", "Nissan", "Audi", "Hyundai", "Kia", "Peugeot", "Renault",
                "Seat", "Skoda", "Volvo", "Mazda", "Mitsubishi", "Subaru", "Fiat"]
makes_rare = [f"Specialist_{i}" for i in range(30)]
all_makes = makes_common + makes_rare

make_probs = np.concatenate([
    rng.dirichlet(np.ones(len(makes_common)) * 3),
    rng.dirichlet(np.ones(len(makes_rare)) * 0.2),
])
make_probs /= make_probs.sum()

age_bands = ["17-24", "25-34", "35-49", "50-64", "65+"]
ncb_levels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
vehicle_groups = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

df = pd.DataFrame({
    "age_band":       rng.choice(age_bands, N_POLICIES, p=[0.08, 0.20, 0.35, 0.25, 0.12]),
    "ncb":            rng.choice(ncb_levels, N_POLICIES),
    "vehicle_group":  rng.choice(vehicle_groups, N_POLICIES),
    "vehicle_make":   rng.choice(all_makes, N_POLICIES, p=make_probs),
})

# True frequency model: age and make drive frequency
age_effect = {"17-24": 2.5, "25-34": 1.4, "35-49": 1.0, "50-64": 0.8, "65+": 1.1}
make_effect = {m: np.exp(rng.normal(0, 0.4)) for m in all_makes}
ncb_effect  = {n: np.exp(-0.12 * n) for n in ncb_levels}

base_freq = 0.07
lam = (
    base_freq
    * df["age_band"].map(age_effect)
    * df["ncb"].map(ncb_effect)
    * df["vehicle_make"].map(make_effect)
)

exposure = rng.uniform(0.3, 1.2, N_POLICIES)
y = rng.poisson(lam * exposure).astype(float)

print(f"Policies: {N_POLICIES:,}")
print(f"Total claims: {y.sum():.0f}")
print(f"Overall frequency: {y.sum() / exposure.sum():.4f}")
print(f"Vehicle makes: {df['vehicle_make'].nunique()} unique")
print(f"Zero-claim policies: {(y == 0).mean():.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Phase 1 + 2: Base GLM + entity embeddings (no spatial)

# COMMAND ----------

pipeline = NestedGLMPipeline(
    base_formula="age_band + ncb + vehicle_group",
    family="poisson",
    embedding_epochs=30,
    embedding_hidden_sizes=(64,),
    embedding_lr=1e-3,
    embedding_batch_size=512,
    random_state=42,
)

pipeline.fit(
    df,
    y,
    exposure,
    high_card_cols=["vehicle_make"],
    base_formula_cols=["age_band", "ncb", "vehicle_group"],
)

print("Pipeline fitted (phases 1 + 2 + 4).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Inspect the outer GLM relativities

# COMMAND ----------

rel = pipeline.relativities()
print(f"\nOuter GLM — {len(rel)} terms")
print(rel.to_string(index=False))

# COMMAND ----------

# Age band relativities
age_rel = rel[rel["term"].str.contains("age_band", na=False)].copy()
print("\nAge band relativities:")
print(age_rel[["term", "relativity", "ci_lower", "ci_upper"]].to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Base GLM vs Outer GLM: AIC comparison

# COMMAND ----------

base_aic = pipeline.base_glm_.aic()
outer_aic = pipeline.outer_glm_.aic()
base_deviance = pipeline.base_glm_.deviance()
outer_deviance = pipeline.outer_glm_.deviance()

print(f"Base GLM  — AIC: {base_aic:,.1f}  Deviance: {base_deviance:,.1f}")
print(f"Outer GLM — AIC: {outer_aic:,.1f}  Deviance: {outer_deviance:,.1f}")
print(f"AIC improvement: {base_aic - outer_aic:,.1f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Inspect the vehicle make embeddings

# COMMAND ----------

trainer = pipeline.embedding_trainer_
emb_frame = trainer.get_embedding_frame()["vehicle_make"]
print(f"Embedding frame shape: {emb_frame.shape}")
print(emb_frame.head(10).to_string(index=False))

# COMMAND ----------

# PCA visualisation
emb_matrix = emb_frame[[c for c in emb_frame.columns if c.startswith("emb_")]].values
labels_list = emb_frame["category"].tolist()

fig = embedding_pca_plot(
    emb_matrix,
    labels=labels_list,
    title="Vehicle Make Embeddings — PCA (2 components)",
)
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Credibility report

# COMMAND ----------

pred = pipeline.predict(df, exposure)
print(f"Prediction range: [{pred.min():.4f}, {pred.max():.4f}]")
print(f"Pearson correlation with actuals: {np.corrcoef(pred, y)[0,1]:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. EmbeddingTrainer in isolation — custom use

# COMMAND ----------

# Use EmbeddingTrainer standalone to encode vehicle make
standalone_trainer = EmbeddingTrainer(
    cat_cols=["vehicle_make"],
    embedding_dims={"vehicle_make": 8},  # override default
    hidden_sizes=(32,),
    epochs=20,
    random_state=0,
)

# Phase 1 log-predictions as offset
base_log_pred = np.log(pipeline.base_glm_.predict(
    df[["age_band", "ncb", "vehicle_group"]], exposure
).clip(min=1e-10))

standalone_trainer.fit(df, y, exposure=exposure, offset=base_log_pred)

emb_vectors = standalone_trainer.transform(df)
print(f"Embedding vectors shape: {emb_vectors.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. NestedGLM in isolation — outer GLM only

# COMMAND ----------

# Build design matrix manually
X_manual = df[["age_band", "ncb", "vehicle_group"]].copy()
emb_array = standalone_trainer.transform(df)
for i in range(emb_array.shape[1]):
    X_manual[f"emb_{i}"] = emb_array[:, i]

outer = NestedGLM(
    family="poisson",
    formula="age_band + ncb + vehicle_group",
    add_embedding_cols=True,
    add_territory=False,
)
outer.fit(X_manual, y, exposure)
print(f"Outer GLM AIC: {outer.aic():.1f}")
rel_manual = outer.relativities()
print(f"\nRelativities ({len(rel_manual)} terms):")
print(rel_manual[rel_manual["term"].str.contains("age", na=False)].to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Spatial clustering demo (synthetic grid)

# COMMAND ----------

try:
    import geopandas as gpd
    from shapely.geometry import box
    SPATIAL_AVAILABLE = True
except ImportError:
    SPATIAL_AVAILABLE = False
    print("geopandas not available — skipping spatial demo")

# COMMAND ----------

if SPATIAL_AVAILABLE:
    # Build a 10x10 grid of synthetic postcode sectors
    N_SIDE = 10
    geoms, sector_ids = [], []
    for r in range(N_SIDE):
        for c in range(N_SIDE):
            geoms.append(box(c, r, c + 1, r + 1))
            sector_ids.append(f"SW{r}{c}")

    geo_gdf = gpd.GeoDataFrame({"sector_id": sector_ids, "geometry": geoms}, crs="EPSG:27700")

    # Assign policies to sectors
    df2 = df.copy()
    df2["sector_id"] = rng.choice(sector_ids, N_POLICIES)

    spatial_pipeline = NestedGLMPipeline(
        base_formula="age_band + ncb",
        family="poisson",
        n_territories=8,
        min_territory_exposure=50,
        embedding_epochs=15,
        embedding_hidden_sizes=(32,),
        random_state=0,
    )

    spatial_pipeline.fit(
        df2,
        y,
        exposure,
        geo_gdf=geo_gdf,
        geo_id_col="sector_id",
        high_card_cols=["vehicle_make"],
        base_formula_cols=["age_band", "ncb"],
    )

    tc = spatial_pipeline.territory_clusterer_
    labels = tc.labels_
    print(f"Territory clustering complete. {labels.nunique()} territories formed.")

    from insurance_nested_glm import plot_territory_map
    fig = plot_territory_map(geo_gdf, labels, title=f"Territory Map — {labels.nunique()} territories")
    display(fig)
    plt.close()

    # Credibility report
    unit_exposure = (
        pd.Series(exposure, index=df2.index)
        .groupby(df2["sector_id"].values)
        .sum()
        .reset_index()
    )
    unit_exposure.columns = ["sector_id", "exposure"]
    unit_territory = pd.Series(
        dict(zip(geo_gdf["sector_id"].values, labels.values)),
        name="territory",
    )
    unit_exposure["territory"] = unit_exposure["sector_id"].map(unit_territory)

    report = credibility_report(unit_exposure["territory"], unit_exposure["exposure"])
    print("\nTerritory credibility report:")
    display(report)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Component | Role |
# MAGIC |---|---|
# MAGIC | `NestedGLMPipeline` | Four-phase end-to-end pipeline |
# MAGIC | `EmbeddingTrainer` | Embeds high-cardinality categoricals into dense vectors |
# MAGIC | `TerritoryClusterer` | Spatially contiguous territory bands via SKATER |
# MAGIC | `NestedGLM` | Outer statsmodels GLM — relativities, AIC, deviance |
# MAGIC | `credibility_report` | Exposure/claims summary per territory |
# MAGIC
# MAGIC The pipeline produces a fully interpretable GLM: relativities table, standard
# MAGIC errors, and AIC — while capturing vehicle make/model signal that a standard
# MAGIC GLM cannot handle without severe grouping.
