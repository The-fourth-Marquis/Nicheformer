#!/usr/bin/env python3
"""
Visualize GSE274447 (Spatial Perturb-Seq) — each slide/sample separately.
Uses uv env for mygene compatibility.
"""

import os, warnings
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

warnings.filterwarnings("ignore")
sc.settings.set_figure_params(dpi=100, facecolor="white", frameon=False)

H5AD_PATH = "/mnt/172/wh/25-12/spatial/preprocessed/GSE274447_spatial_perturb_seq.h5ad"
OUT_DIR   = "/root/code/nicheformer/output/0525/visualize_slides"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading data …")
adata = sc.read_h5ad(H5AD_PATH)
print(f"Shape: {adata.shape}")
print(f"obs columns: {list(adata.obs.columns)}")

# ── Per-slide overview ───────────────────────────────────────────────────
SAMPLES = ["C02943C3", "B03018A2", "A03599E2"]

for sample in SAMPLES:
    mask = adata.obs["sample"] == sample
    sub = adata[mask].copy()
    n = sub.n_obs
    vc = sub.obs["perturbation"].value_counts()
    n_pert = sum(vc[vc.index != "Neg"])
    print(f"\n{sample}: {n} cells, {n_pert} perturbed ({100*n_pert/n:.2f}%)")
    for pert, count in vc.head(10).items():
        print(f"  {pert:20s}: {count:>6d}")

# ── Figure 1: Spatial scatter per slide, colored by perturbation ─────────
print("\nGenerating spatial scatter plots …")

# Perturbation color palette (19 categories = 18 sgRNAs + Neg)
PERT_COLORS = {
    "Neg":      "#D3D3D3",  # light grey
    "Clu":      "#E41A1C",
    "Fasn":     "#377EB8",
    "Olig2":    "#4DAF4A",
    "Stk39":    "#984EA3",
    "Trem2":    "#FF7F00",
    "Gfap":     "#FFFF33",
    "Sh3gl2":   "#A65628",
    "Rraga":    "#F781BF",
    "Flcn":     "#66C2A5",
    "Cfap410":  "#FC8D62",
    "Rbfox3":   "#8DA0CB",
    "Ndufaf2":  "#E78AC3",
    "Tbk1":     "#A6D854",
    "Msafe":     "#FFD92F",
    "Dpp5":     "#E5C494",
    "Srf":       "#B3B3B3",
    "C9orf72":  "#1B9E77",
    "Lrrk2":    "#D95F02",
}

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
for i, sample in enumerate(SAMPLES):
    ax = axes[i]
    mask = adata.obs["sample"] == sample
    xy = adata.obsm["spatial"][mask.values]
    pert = adata.obs["perturbation"][mask].values

    # Plot Neg first (background), then perturbations
    neg_mask = pert == "Neg"
    ax.scatter(xy[neg_mask, 0], xy[neg_mask, 1], c=PERT_COLORS["Neg"],
               s=0.5, alpha=0.3, rasterized=True, label="Neg")

    for p in sorted(set(pert)):
        if p == "Neg":
            continue
        p_mask = pert == p
        if p_mask.sum() > 0:
            ax.scatter(xy[p_mask, 0], xy[p_mask, 1],
                       c=PERT_COLORS.get(p, "#000000"),
                       s=8, alpha=0.9, edgecolors="black", linewidth=0.3,
                       rasterized=True, label=f"{p} ({p_mask.sum()})")

    ax.set_title(f"{sample}\n({mask.sum()} cells, {(pert != 'Neg').sum()} perturbed)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.legend(fontsize=5, markerscale=2, loc="upper right", frameon=True)

fig.suptitle("GSE274447 Spatial Perturb-Seq — per-slide spatial scatter", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "spatial_scatter_per_slide.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  → spatial_scatter_per_slide.png")

# ── Figure 2: Perturbation only (no Neg background) ──────────────────────
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
for i, sample in enumerate(SAMPLES):
    ax = axes[i]
    mask = adata.obs["sample"] == sample
    xy = adata.obsm["spatial"][mask.values]
    pert = adata.obs["perturbation"][mask].values
    pert_mask = pert != "Neg"
    if pert_mask.sum() == 0:
        ax.set_title(f"{sample}\nno perturbed cells")
        continue
    xy_p = xy[pert_mask]
    pert_p = pert[pert_mask]
    for p in sorted(set(pert_p)):
        p_mask = pert_p == p
        ax.scatter(xy_p[p_mask, 0], xy_p[p_mask, 1],
                   c=PERT_COLORS.get(p, "#000000"),
                   s=15, alpha=0.9, edgecolors="black", linewidth=0.5,
                   rasterized=True, label=f"{p} ({p_mask.sum()})")
    ax.set_title(f"{sample}\nperturbed cells only (n={pert_mask.sum()})",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, markerscale=1.5, loc="upper right", frameon=True)

fig.suptitle("GSE274447 — perturbed cells only", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "spatial_pert_only.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  → spatial_pert_only.png")

# ── Figure 3: Perturbation composition bar chart ─────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 7))
for i, sample in enumerate(SAMPLES):
    ax = axes[i]
    mask = adata.obs["sample"] == sample
    vc = adata.obs["perturbation"][mask].value_counts()
    vc_pert = vc[vc.index != "Neg"]
    colors = [PERT_COLORS.get(p, "#333") for p in vc_pert.index]
    bars = ax.bar(range(len(vc_pert)), vc_pert.values,
                  color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(vc_pert)))
    ax.set_xticklabels(vc_pert.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Number of cells")
    ax.set_title(f"{sample} — {vc_pert.sum()} perturbed cells", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, vc_pert.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(val), ha="center", fontsize=7, fontweight="bold")

fig.suptitle("GSE274447 — perturbation composition per slide", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "perturbation_composition.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  → perturbation_composition.png")

# ── Figure 4: sgRNA expression spatial heatmap (top sgRNAs per slide) ────
sgrna_genes = [g for g in adata.var_names if "sgrna" in g.lower()]
print(f"\n{sgrna_genes}")
print(f"sgRNA genes: {len(sgrna_genes)}")

for sample in SAMPLES:
    mask = adata.obs["sample"] == sample
    sub = adata[mask].copy()

    # Get sgRNA expression for all cells
    sgrna_idx = [list(sub.var_names).index(g) for g in sgrna_genes]
    sgrna_expr = sub.X[:, sgrna_idx].toarray()

    # Find sgRNAs with any expression in this sample
    expressed = sgrna_expr.sum(axis=0) > 0
    n_sgrna = expressed.sum()
    if n_sgrna == 0:
        print(f"  {sample}: no sgRNA detected")
        continue

    active_sgrnas = [sgrna_genes[j] for j in range(len(sgrna_genes)) if expressed[j]]
    active_expr = sgrna_expr[:, expressed]

    n_cols = min(6, n_sgrna)
    n_rows = int(np.ceil(n_sgrna / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 3.5, n_rows * 3.5))
    axes_flat = axes.flatten() if n_sgrna > 1 else [axes]

    xy = sub.obsm["spatial"]
    for j in range(n_sgrna):
        ax = axes_flat[j]
        expr_j = active_expr[:, j]
        pos_mask = expr_j > 0
        # Background: all cells
        ax.scatter(xy[:, 0], xy[:, 1], c="#F0F0F0", s=0.3, alpha=0.5, rasterized=True)
        if pos_mask.sum() > 0:
            ax.scatter(xy[pos_mask, 0], xy[pos_mask, 1],
                       c="red", s=10, alpha=0.9, edgecolors="darkred",
                       linewidth=0.3, rasterized=True,
                       label=f"n={pos_mask.sum()}")
        ax.set_title(active_sgrnas[j], fontsize=9, fontweight="bold")
        ax.set_aspect("equal")
        ax.axis("off")
        if pos_mask.sum() > 0:
            ax.legend(fontsize=6, loc="upper right")

    for j in range(n_sgrna, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"{sample} — sgRNA barcode expression (spatial)", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"sgrna_spatial_{sample}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → sgrna_spatial_{sample}.png ({n_sgrna} sgRNAs detected)")

# ── Figure 5: Sample-level UMAP colored by perturbation ───────────────────
print("\nGenerating UMAP per sample …")

for sample in SAMPLES:
    mask = adata.obs["sample"] == sample
    sub = adata[mask].copy()
    if sub.n_obs > 5000:
        sc.pp.subsample(sub, n_obs=5000)
    sc.pp.normalize_total(sub, target_sum=1e4)
    sc.pp.log1p(sub)
    sc.pp.pca(sub, n_comps=30)
    sc.pp.neighbors(sub, n_neighbors=15)
    sc.tl.umap(sub)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    # Left: cell density
    sc.pl.umap(sub, color="perturbation", ax=axes[0], show=False,
               palette=PERT_COLORS, title=f"{sample} — perturbation",
               legend_loc="right margin")
    # Right: sample clustering
    sc.pl.umap(sub, ax=axes[1], show=False,
               title=f"{sample} — UMAP")

    fig.suptitle(f"{sample} UMAP", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"umap_{sample}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → umap_{sample}.png")

# ── Summary ──────────────────────────────────────────────────────────────
print(f"\nDone. All figures saved to: {OUT_DIR}")
for f in sorted(os.listdir(OUT_DIR)):
    size = os.path.getsize(os.path.join(OUT_DIR, f))
    print(f"  {f} ({size/1024:.0f} KB)")
