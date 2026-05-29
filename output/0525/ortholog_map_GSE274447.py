#!/usr/bin/env python3
"""
Ortholog-mapping for GSE274447 (Spatial Perturb-Seq, mouse brain Stereo-seq).
Maps mouse gene symbols → ENSMUSG → human ENSG orthologs → model-aligned AnnData.

Usage:
    uv run python ortholog_map_GSE274447.py
"""

import os, sys, warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.sparse import csr_matrix, coo_matrix

warnings.filterwarnings("ignore")

import scanpy as sc
import mygene

INPUT_PATH  = "/mnt/172/wh/25-12/spatial/preprocessed/GSE274447_spatial_perturb_seq.h5ad"
MODEL_PATH  = "/root/code/nicheformer/data/model_means/model.h5ad"
OUTPUT_PATH = "/mnt/172/wh/25-12/spatial/preprocessed/GSE274447_spatial_perturb_seq_model_aligned.h5ad"

# ── Load data ────────────────────────────────────────────────────────────
print("Loading data …")
adata = sc.read_h5ad(INPUT_PATH)
model = sc.read_h5ad(MODEL_PATH)
print(f"  Data: {adata.shape[0]} cells x {adata.shape[1]} genes")
print(f"  Model: {len(model.var_names)} genes")

# ── Identify gene types ──────────────────────────────────────────────────
vnames = list(adata.var_names)
is_ensmusg = np.array([v.startswith("ENSMUSG") for v in vnames])
is_symbol  = ~is_ensmusg

print(f"  Gene symbols: {is_symbol.sum()}, ENSMUSG: {is_ensmusg.sum()}")

# ── Step 1: Map gene symbols → ENSMUSG ───────────────────────────────────
print("\nStep 1: Mapping gene symbols → ENSMUSG …")
mg = mygene.MyGeneInfo()

symbols_only = [v for v, m in zip(vnames, is_symbol) if m]
print(f"  Querying {len(symbols_only)} gene symbols via mygene …")

symbol_to_ensmusg = {}
batch_size = 1000
for start in range(0, len(symbols_only), batch_size):
    batch = symbols_only[start : start + batch_size]
    r = mg.querymany(batch, scopes="symbol", fields="ensembl.gene",
                     species="mouse", returnall=True, as_dataframe=False)
    for res in r["out"]:
        q = res.get("query", "")
        ens = res.get("ensembl", {})
        if isinstance(ens, dict):
            mus = ens.get("gene", "")
        elif isinstance(ens, list) and ens:
            mus = ens[0].get("gene", "")
        else:
            mus = ""
        if mus:
            symbol_to_ensmusg[q] = mus
    if (start // batch_size) % 10 == 0:
        print(f"    {start}/{len(symbols_only)} …")

print(f"  Mapped: {len(symbol_to_ensmusg)}/{len(symbols_only)} symbols → ENSMUSG")

# ── Step 2: ENSMUSG → human ortholog ─────────────────────────────────────
print("\nStep 2: ENSMUSG → human ortholog (homologene) …")

# Collect all ENSMUSGs
all_ensmusg = set()
# From symbol mapping
all_ensmusg.update(symbol_to_ensmusg.values())
# From existing ENSMUSG IDs
all_ensmusg.update([v for v, m in zip(vnames, is_ensmusg) if m])

print(f"  Total unique ENSMUSG: {len(all_ensmusg)}")

# Query mygene for homologene
ensmusg_list = list(all_ensmusg)
ensmusg_to_human_ncbi = {}
mg2 = mygene.MyGeneInfo()

for start in range(0, len(ensmusg_list), 1000):
    batch = ensmusg_list[start : start + 1000]
    r = mg2.querymany(batch, scopes="ensembl.gene", fields="homologene",
                      species="mouse", returnall=True, as_dataframe=False)
    for res in r["out"]:
        mus_id = res.get("query", "")
        hom = res.get("homologene", {})
        if hom and "genes" in hom:
            for tax_id, gene_id in hom["genes"]:
                if tax_id == 9606:  # human
                    ensmusg_to_human_ncbi[mus_id] = str(gene_id)
                    break

print(f"  ENSMUSG with human ortholog: {len(ensmusg_to_human_ncbi)}/{len(ensmusg_list)}")

# ── Step 3: Human NCBI Gene IDs → ENSG ───────────────────────────────────
print("\nStep 3: Human NCBI Gene ID → ENSG …")
human_ncbi_ids = list(set(ensmusg_to_human_ncbi.values()))
print(f"  Unique human NCBI gene IDs: {len(human_ncbi_ids)}")

ncbi_to_ensg = {}
mg3 = mygene.MyGeneInfo()
for start in range(0, len(human_ncbi_ids), 1000):
    batch = human_ncbi_ids[start : start + 1000]
    r = mg3.querymany(batch, scopes="entrezgene", fields="ensembl.gene",
                      species="human", returnall=True, as_dataframe=False)
    for res in r["out"]:
        ens = res.get("ensembl", {})
        if isinstance(ens, dict):
            ensg = ens.get("gene", "")
        elif isinstance(ens, list) and ens:
            ensg = ens[0].get("gene", "")
        else:
            ensg = ""
        if ensg:
            ncbi_to_ensg[res.get("query", "")] = ensg

print(f"  Human NCBI → ENSG mapped: {len(ncbi_to_ensg)}/{len(human_ncbi_ids)}")

# ── Step 4: Build full mapping: original var_name → ENSG ──────────────────
print("\nStep 4: Building full var_name → ENSG mapping …")

# ENSMUSG → ENSG via NCBI
ensmusg_to_ensg = {}
for mus_id, ncbi in ensmusg_to_human_ncbi.items():
    ensg = ncbi_to_ensg.get(ncbi, "")
    if ensg:
        ensmusg_to_ensg[mus_id] = ensg

# Map every var_name to ENSG
old_to_new = {}
n_mapped = 0
n_failed = 0
for old_name in vnames:
    if old_name.startswith("ENSMUSG"):
        ensg = ensmusg_to_ensg.get(old_name, "")
        if ensg:
            old_to_new[old_name] = ensg
            n_mapped += 1
        else:
            n_failed += 1
    else:
        mus_id = symbol_to_ensmusg.get(old_name, "")
        if mus_id and mus_id in ensmusg_to_ensg:
            old_to_new[old_name] = ensmusg_to_ensg[mus_id]
            n_mapped += 1
        else:
            n_failed += 1

print(f"  Mapped: {n_mapped}/{len(vnames)} → human ENSG")
print(f"  Failed: {n_failed}/{len(vnames)}")

# Also ensure sgRNA genes are preserved (they won't be in ortholog map)
for g in adata.var_names:
    if "sgrna" in g.lower() and g not in old_to_new:
        old_to_new[g] = g  # preserve as-is

# ── Step 5: Rebuild expression matrix with new gene IDs ───────────────────
print("\nStep 5: Rebuilding expression matrix …")

# Group columns by new name, summing counts for duplicates
new_col_map = defaultdict(list)
orig_cols_assigned = 0
for old_idx, old_name in enumerate(vnames):
    new_name = old_to_new.get(old_name, "")
    if new_name:
        new_col_map[new_name].append(old_idx)
        orig_cols_assigned += 1

print(f"  Old columns assigned: {orig_cols_assigned}/{len(vnames)}")
print(f"  Unique new genes: {len(new_col_map)}")

# Build new sparse matrix
X_old = adata.X.tocsr()
new_genes = sorted(new_col_map.keys())
new_n_cols = len(new_genes)

# Build COO for the new matrix
rows, cols, data = [], [], []
for new_j, new_gene in enumerate(new_genes):
    old_indices = new_col_map[new_gene]
    if len(old_indices) == 1:
        old_j = old_indices[0]
        col_data = X_old[:, old_j].tocoo()
        rows.extend(col_data.row)
        cols.extend([new_j] * len(col_data.data))
        data.extend(col_data.data)
    else:
        # Sum over multiple old columns
        combined = X_old[:, old_indices].sum(axis=1).A1
        for i, val in enumerate(combined):
            if val > 0:
                rows.append(i)
                cols.append(new_j)
                data.append(val)

X_new = coo_matrix((data, (rows, cols)), shape=(adata.n_obs, new_n_cols)).tocsr()
print(f"  New matrix: {X_new.shape}")

# ── Step 6: Build new AnnData ────────────────────────────────────────────
print("\nStep 6: Building new AnnData …")
adata_new = sc.AnnData(
    X=X_new.astype(np.float32),
    obs=adata.obs.copy(),
    var=pd.DataFrame(index=new_genes),
)
adata_new.obsm["spatial"] = adata.obsm["spatial"].copy()
for key in adata.obsm:
    if key != "spatial":
        adata_new.obsm[key] = adata.obsm[key].copy()

print(f"  New shape: {adata_new.shape}")

# ── Step 7: Gene reordering via concat with model.h5ad ───────────────────
print("\nStep 7: Gene reordering …")
# Outer join: model genes + data genes
adata_new = adata_new.concatenate(
    [model], join="outer", axis=0, batch_key="_tmp_batch"
)
# Strip model rows, keep data rows
is_data = adata_new.obs["_tmp_batch"] == "0"
adata_new = adata_new[is_data].copy()
del adata_new.obs["_tmp_batch"]

print(f"  After reorder: {adata_new.shape[0]} cells x {adata_new.shape[1]} genes")

# ── Step 8: Overlap stats ────────────────────────────────────────────────
overlap_with_model = set(adata_new.var_names) & set(model.var_names)
print(f"\n  Final overlap with model: {len(overlap_with_model)} genes")

# Count expressed genes per cell
n_nonzero_per_cell = (adata_new.X > 0).sum(axis=1).A1
print(f"  Mean genes/cell: {n_nonzero_per_cell.mean():.1f}")
print(f"  Median genes/cell: {np.median(n_nonzero_per_cell):.1f}")

# ── Save ─────────────────────────────────────────────────────────────────
adata_new.write(OUTPUT_PATH)
print(f"\nSaved: {OUTPUT_PATH}")
print("Done.")
