#!/usr/bin/env python3
"""
Convert perturbation map h5 data (GEO CosMx mouse) to AnnData h5ad format.

The combined h5 file contains:
  - X: (6363 cells, 1053 genes) — raw expression counts
  - gene: (1053,) — gene symbols
  - perturbation: (6363,) — per-cell perturbation label (None/KP/Tgfbr2/Ifngr2/Jak2/periphery)
  - pos: (2, 6363) — spatial coordinates
  - tissue: (6363,) — tissue type (normal/tumor)
  - batch: (4, 6363) — one-hot slide encoding

Output: AnnData with:
  - .X: sparse CSR float32 counts
  - .obs: perturbation, tissue, batch (slide ID), x, y
  - .var: gene symbols as index
  - .obsm['spatial']: (n_cells, 2) spatial coordinates

Usage:
    uv run python convert_perturb_h5_to_h5ad.py
"""

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix
import os

INPUT_PATH = "/mnt/172/wh/25-12/spatial/raw/pertubmap/perturb_map_data_all.h5"
OUTPUT_PATH = "/mnt/172/wh/25-12/spatial/raw/pertubmap/perturb_map_adata.h5ad"

SLIDE_LABELS = ["GSM5808054", "GSM5808055", "GSM5808056", "GSM5808057"]

print(f"Loading {INPUT_PATH} ...")
with h5py.File(INPUT_PATH, "r") as f:
    X = f["X"][:]  # (6363, 1053)
    gene_names = [g.decode() for g in f["gene"][:]]
    perturbation = np.array([p.decode() for p in f["perturbation"][:]])
    pos = f["pos"][:]  # (2, 6363)
    tissue = np.array([t.decode() for t in f["tissue"][:]])
    batch_onehot = f["batch"][:]  # (4, 6363)

n_cells, n_genes = X.shape
print(f"  Cells: {n_cells}, Genes: {n_genes}")

# Convert batch one-hot to slide labels
batch_idx = np.argmax(batch_onehot, axis=0)
batch_labels = np.array([SLIDE_LABELS[i] for i in batch_idx])

# Build AnnData
print("Building AnnData ...")
adata = sc.AnnData(
    X=csr_matrix(X, dtype=np.float32),
    obs=pd.DataFrame(
        {
            "perturbation": perturbation,
            "tissue_type_raw": tissue,  # renamed — 'tissue' conflicts with CELLxGENE reserved column
            "batch": batch_labels,
            "x": pos[0, :].astype(np.float64),
            "y": pos[1, :].astype(np.float64),
        }
    ),
    var=pd.DataFrame(index=gene_names),
)

adata.obsm["spatial"] = pos.T.astype(np.float64)

print(f"  AnnData: {adata.shape}")
print(f"  obs columns: {list(adata.obs.columns)}")
print(f"  Perturbation types: {sorted(adata.obs['perturbation'].unique())}")
print(f"  Tissue types: {sorted(adata.obs['tissue_type_raw'].unique())}")
print(f"  Batches: {sorted(adata.obs['batch'].unique())}")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
adata.write(OUTPUT_PATH)
print(f"\nSaved: {OUTPUT_PATH}")
