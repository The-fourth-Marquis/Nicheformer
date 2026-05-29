#!/usr/bin/env python3
"""
Convert Stereo-seq GEF (cellbin) files from GSE274447 to AnnData h5ad format.

GSE274447: Spatial Perturb-Seq — in vivo CRISPR in mouse brain (Stereo-seq).
3 samples from 3 mice, each injected with pooled CRISPR library targeting
neurodegenerative disease risk genes.

Output: one merged AnnData with:
  - .X: sparse CSR float32 raw counts
  - .obs: sample, x, y, perturbation (if available)
  - .var: gene symbols
  - .obsm['spatial']: spatial coordinates

Usage:
    /root/code/py38/bin/python convert_gef_to_h5ad.py
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix

warnings.filterwarnings("ignore")

# Use stereopy from py38 env
sys.path.insert(0, "/root/code/py38/lib/python3.8/site-packages")
import stereo as st

DATA_DIR = "/mnt/172/wh/25-12/spatial/raw/GSE274447_RAW"
OUT_DIR  = "/mnt/172/wh/25-12/spatial/preprocessed"
os.makedirs(OUT_DIR, exist_ok=True)

GEF_FILES = {
    "GSM8449354_C02943C3.adjusted.cellbin.gef": "C02943C3",
    "GSM9659883_B03018A2.adjusted.cellbin.gef": "B03018A2",
    "GSM9659884_A03599E2.adjusted.cellbin.gef": "A03599E2",
}

def load_gef_to_adata(gef_path, sample_id):
    """Load a GEF cellbin file and return an AnnData object."""
    print(f"Loading {sample_id}: {os.path.basename(gef_path)} ...")
    data = st.io.read_gef(gef_path, bin_type="cell_bins")

    n_cells, n_genes = data.shape
    print(f"  -> {n_cells} cells x {n_genes} genes")

    # Expression matrix (stereopy returns scipy csr_matrix directly)
    print("  Building sparse matrix ...")
    X = data.exp_matrix
    if hasattr(X, "toarray"):
        X = X.tocsr().astype(np.float32)
    elif hasattr(X, "todense"):
        X = csr_matrix(X.todense(), dtype=np.float32)
    else:
        X = csr_matrix(X, dtype=np.float32)

    # Gene names — deduplicate by summing counts for identical gene symbols
    gene_names = list(data.gene_names)
    gene_series = pd.Series(gene_names)
    if gene_series.duplicated().any():
        n_dup = gene_series.duplicated().sum()
        print(f"  Deduplicating {n_dup} duplicate gene names ...")
        # Build row-indexed COO for column aggregation
        X_coo = X.tocoo()
        gene_idx = X_coo.col
        # Map old col idx → unique gene idx
        unique_genes, inverse = np.unique(gene_names, return_inverse=True)
        new_col = inverse[gene_idx]
        from scipy.sparse import coo_matrix
        X_dedup = coo_matrix(
            (X_coo.data, (X_coo.row, new_col)),
            shape=(X.shape[0], len(unique_genes))
        ).tocsr()
        X = X_dedup
        gene_names = list(unique_genes)
        print(f"  After dedup: {len(gene_names)} unique genes")

    # Spatial coordinates (already numpy array from stereopy)
    spatial = data.spatial
    if hasattr(spatial, "toarray"):
        spatial = spatial.toarray()
    elif hasattr(spatial, "todense"):
        spatial = np.array(spatial.todense())
    elif isinstance(spatial, np.ndarray):
        spatial = spatial
    else:
        spatial = np.array(spatial)

    # Build obs DataFrame
    obs = pd.DataFrame({
        "sample": sample_id,
        "x": spatial[:, 0].astype(np.float32),
        "y": spatial[:, 1].astype(np.float32),
    })

    # Cell metadata from GEF
    cells_df = data.cells.to_df()
    if "dnbCount" in cells_df.columns:
        obs["dnbCount"] = cells_df["dnbCount"].values
    if "area" in cells_df.columns:
        obs["area"] = cells_df["area"].values

    adata = sc.AnnData(
        X=X,
        obs=obs,
        var=pd.DataFrame(index=gene_names),
    )
    adata.obsm["spatial"] = spatial.astype(np.float32)
    adata.obs_names = [f"{sample_id}_{i}" for i in range(n_cells)]

    print(f"  AnnData: {adata.shape}")
    return adata


# ── Load all 3 GEF files ─────────────────────────────────────────────────
adatas = []
for fname, sample_id in GEF_FILES.items():
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path):
        adata = load_gef_to_adata(path, sample_id)
        adatas.append(adata)
    else:
        print(f"[WARN] File not found: {path}")

# ── Merge by gene intersection (concatenate samples) ─────────────────────
print("\nMerging samples ...")
# Find common genes across all samples
common_genes = set(adatas[0].var_names)
for a in adatas[1:]:
    common_genes &= set(a.var_names)

n_common = len(common_genes)
print(f"Genes with full overlap: {n_common} / union: ~{len(set().union(*[set(a.var_names) for a in adatas]))}")

# Filter each to common genes
for i, a in enumerate(adatas):
    a = a[:, list(common_genes)]
    adatas[i] = a

# Concatenate
adata_merged = adatas[0].concatenate(adatas[1:], batch_key="sample_batch", index_unique=None)
print(f"Merged: {adata_merged.shape}")

# ── Save ─────────────────────────────────────────────────────────────────
OUT_PATH = os.path.join(OUT_DIR, "GSE274447_spatial_perturb_seq.h5ad")
adata_merged.write(OUT_PATH)
print(f"\nSaved: {OUT_PATH}")

# ── Summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
for sample_id in ["C02943C3", "B03018A2", "A03599E2"]:
    n = (adata_merged.obs["sample"] == sample_id).sum()
    print(f"  {sample_id}: {n} cells")
print(f"  Total: {adata_merged.n_obs} cells x {adata_merged.n_vars} genes")
print(f"\nNote: Perturbation labels not yet available in GEF files.")
print(f"      Check GEO GSE274447 or spatialperturbseq GitHub for cell-level metadata.")
