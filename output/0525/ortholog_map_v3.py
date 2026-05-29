#!/usr/bin/env python3
"""Fast ortholog mapping — filters non-coding genes, batches properly, caches results."""

import os, re, pickle, warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.sparse import csr_matrix, coo_matrix

warnings.filterwarnings("ignore")
import scanpy as sc
import mygene

INPUT  = "/mnt/172/wh/25-12/spatial/preprocessed/GSE274447_spatial_perturb_seq.h5ad"
MODEL  = "/root/code/nicheformer/data/model_means/model.h5ad"
OUTPUT = "/mnt/172/wh/25-12/spatial/preprocessed/GSE274447_spatial_perturb_seq_model_aligned.h5ad"
CACHE  = "/root/code/nicheformer/output/0525/ortholog_cache.pkl"

VALID_ENSMUSG = re.compile(r"^ENSMUSG\d{11}$")

# Genes that won't have human orthologs — skip mygene query
SKIP_PATTERNS = [
    r"^Gm\d+$", r"^Olfr\d+$", r"^Rik$", r"^[0-9].*Rik$", r".*os$", r"[0-9]Rik$",
    r"^Vmn", r"^Tas", r"^AA[0-9]", r"^AW[0-9]", r"^AI[0-9]", r"^BC[0-9]",
    r"^sgrna_", r"^mt-", r"^Rp[sl]", r"^Rpl", r"^Rps", r"^Snor", r"^Snhg",
    r"^Mir", r"^U[0-9]", r"^Vault", r"^n-R5",
]
SKIP_RE = re.compile("|".join(SKIP_PATTERNS))

print("Loading data ...")
adata = sc.read_h5ad(INPUT)
model  = sc.read_h5ad(MODEL)

vnames = list(adata.var_names)
model_genes = set(model.var_names)

# ── Separate gene types ──────────────────────────────────────────────────
is_ensmusg = [v.startswith("ENSMUSG") for v in vnames]
is_sgrna   = ["sgrna" in v.lower() for v in vnames]
is_symbol  = [not (e or s) for e, s in zip(is_ensmusg, is_sgrna)]

symbols_only = [v for v, m in zip(vnames, is_symbol) if m]
direct_mus  = [v for v, m in zip(vnames, is_ensmusg) if m]

# Filter to queryable symbols (skip unannotated)
queryable = [s for s in symbols_only if not SKIP_RE.match(s)]
skip_sym  = [s for s in symbols_only if SKIP_RE.match(s)]

print(f"Genes: {len(vnames)} total = {len(symbols_only)} symbols + {len(direct_mus)} ENSMUSG + {sum(is_sgrna)} sgRNA")
print(f"  Queryable symbols: {len(queryable)}")
print(f"  Skipped (no ortholog expected): {len(skip_sym)}")

# ── Load or compute mapping ──────────────────────────────────────────────
if os.path.exists(CACHE):
    print(f"\nLoading cached orthologs from {CACHE} ...")
    with open(CACHE, "rb") as f:
        cache = pickle.load(f)
    symbol_to_mus = cache["symbol_to_mus"]
    mus_to_human_ncbi = cache["mus_to_human_ncbi"]
    print(f"  Loaded: {len(symbol_to_mus)} symbol→ENSMUSG, {len(mus_to_human_ncbi)} ENSMUSG→ortholog")
else:
    mg = mygene.MyGeneInfo()

    # Step 1: Query symbols → ENSMUSG + homologene
    print("\nStep 1: Querying gene symbols → ENSMUSG + ortholog ...")
    symbol_to_mus = {}
    mus_to_human_ncbi = {}

    BATCH = 800
    n_total = len(queryable)
    for start in range(0, n_total, BATCH):
        batch = queryable[start : start + BATCH]
        r = mg.querymany(batch, scopes="symbol",
                         fields="ensembl.gene,homologene",
                         species="mouse", returnall=True,
                         as_dataframe=False, verbose=False)

        for res in r.get("out", []):
            q = res.get("query", "")
            ens = res.get("ensembl", {})
            if isinstance(ens, dict):
                mus = ens.get("gene", "")
            elif isinstance(ens, list) and ens:
                mus = ens[0].get("gene", "")
            else:
                mus = ""
            if mus and VALID_ENSMUSG.match(mus):
                symbol_to_mus[q] = mus
            # homologene
            hom = res.get("homologene", {})
            if hom and "genes" in hom and mus:
                for tax_id, gene_id in hom["genes"]:
                    if tax_id == 9606:
                        mus_to_human_ncbi[mus] = str(gene_id)
                        break

        if (start // BATCH) % 5 == 0:
            pct = 100 * min(start + BATCH, n_total) / n_total
            print(f"  {min(start+BATCH, n_total)}/{n_total} ({pct:.0f}%) — "
                  f"{len(symbol_to_mus)} mapped, {len(mus_to_human_ncbi)} with ortholog")

    print(f"  Done: {len(symbol_to_mus)} symbols→ENSMUSG, {len(mus_to_human_ncbi)} have orthologs")

    # Step 2: Missing ENSMUSG IDs
    missing_mus = [m for m in direct_mus if VALID_ENSMUSG.match(m) and m not in mus_to_human_ncbi]
    print(f"\nStep 2: Querying {len(missing_mus)} missing ENSMUSG for orthologs ...")
    for start in range(0, len(missing_mus), BATCH):
        batch = missing_mus[start : start + BATCH]
        r = mg.querymany(batch, scopes="ensembl.gene",
                         fields="homologene", species="mouse",
                         returnall=True, as_dataframe=False, verbose=False)
        for res in r.get("out", []):
            mus_id = res.get("query", "")
            hom = res.get("homologene", {})
            if hom and "genes" in hom:
                for tax_id, gene_id in hom["genes"]:
                    if tax_id == 9606:
                        mus_to_human_ncbi[mus_id] = str(gene_id)
                        break
    print(f"  Total ENSMUSG→ortholog: {len(mus_to_human_ncbi)}")

    # Save cache
    with open(CACHE, "wb") as f:
        pickle.dump({"symbol_to_mus": symbol_to_mus, "mus_to_human_ncbi": mus_to_human_ncbi}, f)
    print(f"  Cache saved to {CACHE}")

# ── Step 3: Human NCBI → ENSG ────────────────────────────────────────────
print("\nStep 3: Human NCBI Gene ID → ENSG ...")
human_ncbi_ids = list(set(mus_to_human_ncbi.values()))
print(f"  Unique human NCBI IDs: {len(human_ncbi_ids)}")

ncbi_cache_file = CACHE.replace(".pkl", "_ncbi2ensg.pkl")
if os.path.exists(ncbi_cache_file):
    with open(ncbi_cache_file, "rb") as f:
        ncbi_to_ensg = pickle.load(f)
    print(f"  Loaded from cache: {len(ncbi_to_ensg)}")
else:
    ncbi_to_ensg = {}
    mg2 = mygene.MyGeneInfo()
    BATCH2 = 800
    for start in range(0, len(human_ncbi_ids), BATCH2):
        batch = human_ncbi_ids[start : start + BATCH2]
        r = mg2.querymany(batch, scopes="entrezgene",
                          fields="ensembl.gene", species="human",
                          returnall=True, as_dataframe=False, verbose=False)
        for res in r.get("out", []):
            ens = res.get("ensembl", {})
            if isinstance(ens, dict):
                ensg = ens.get("gene", "")
            elif isinstance(ens, list) and ens:
                ensg = ens[0].get("gene", "")
            else:
                ensg = ""
            if ensg and ensg.startswith("ENSG"):
                ncbi_to_ensg[res.get("query", "")] = ensg
    with open(ncbi_cache_file, "wb") as f:
        pickle.dump(ncbi_to_ensg, f)

print(f"  NCBI→ENSG: {len(ncbi_to_ensg)}")

# ── Step 4: ENSMUSG → ENSG ───────────────────────────────────────────────
mus_to_ensg = {}
for mus_id, ncbi in mus_to_human_ncbi.items():
    ensg = ncbi_to_ensg.get(ncbi, "")
    if ensg:
        mus_to_ensg[mus_id] = ensg
print(f"  ENSMUSG→ENSG: {len(mus_to_ensg)}")

# ── Step 5: Map var_names → new IDs ──────────────────────────────────────
print("\nStep 5: Building mapping ...")
old_to_new = {}
n_mapped = 0
for old_name in vnames:
    if "sgrna" in old_name.lower():
        old_to_new[old_name] = old_name; n_mapped += 1
    elif old_name.startswith("ENSMUSG"):
        ensg = mus_to_ensg.get(old_name, "")
        if ensg: old_to_new[old_name] = ensg; n_mapped += 1
    else:
        mus = symbol_to_mus.get(old_name, "")
        if mus and mus in mus_to_ensg:
            old_to_new[old_name] = mus_to_ensg[mus]; n_mapped += 1

print(f"  Mapped: {n_mapped}/{len(vnames)}")

# ── Step 6: Rebuild matrix ───────────────────────────────────────────────
print("\nStep 6: Rebuilding expression matrix ...")
new_col_map = defaultdict(list)
for old_idx, old_name in enumerate(vnames):
    nn = old_to_new.get(old_name, "")
    if nn:
        new_col_map[nn].append(old_idx)

new_genes = sorted(new_col_map.keys())
print(f"  New genes: {len(new_genes)}")

X_old = adata.X.tocsr()
rows, cols, data = [], [], []
for new_j, ng in enumerate(new_genes):
    old_indices = new_col_map[ng]
    if len(old_indices) == 1:
        col = X_old[:, old_indices[0]].tocoo()
        rows.extend(col.row); cols.extend([new_j]*len(col.data)); data.extend(col.data)
    else:
        combined = X_old[:, old_indices].sum(axis=1).A1
        for i, v in enumerate(combined):
            if v > 0:
                rows.append(i); cols.append(new_j); data.append(v)

X_new = coo_matrix((data, (rows, cols)), shape=(adata.n_obs, len(new_genes))).tocsr()
print(f"  Matrix: {X_new.shape}")

# ── Step 7: Align with model ─────────────────────────────────────────────
print("\nStep 7: Gene reordering ...")
adata_new = sc.AnnData(
    X=X_new.astype(np.float32), obs=adata.obs.copy(),
    var=pd.DataFrame(index=new_genes),
)
adata_new.obsm["spatial"] = adata.obsm["spatial"].copy()

adata_new = adata_new.concatenate([model], join="outer", axis=0, batch_key="_tmp")
is_data = adata_new.obs["_tmp"] == "0"
adata_new = adata_new[is_data].copy()
del adata_new.obs["_tmp"]

# Fill NaN
X_arr = np.nan_to_num(adata_new.X.toarray(), nan=0.0)
adata_new.X = csr_matrix(X_arr.astype(np.float32))

print(f"  Final: {adata_new.shape}")

# Stats
n_genes_cell = (adata_new.X > 0).sum(axis=1).A1
print(f"  Mean genes/cell: {n_genes_cell.mean():.1f}")
print(f"  Median genes/cell: {np.median(n_genes_cell):.1f}")

# Preserve sgrna
sgrna_count = len([g for g in adata_new.var_names if "sgrna" in g.lower()])
print(f"  sgRNA genes preserved: {sgrna_count}")

# Save
adata_new.write(OUTPUT)
print(f"\nSaved: {OUTPUT}")
print("Done!")
