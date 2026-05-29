#!/usr/bin/env python3
"""Build visualization ipynb for GSE274447 per-slide analysis."""
import json, os

OUT_DIR = "/root/code/nicheformer/output/0525"

cells = []

# Cell 0: Setup
cells.append("""import os, warnings
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sc.settings.set_figure_params(dpi=100, facecolor="white", frameon=False)
sns.set_style("ticks")

H5AD_PATH = "/mnt/172/wh/25-12/spatial/preprocessed/GSE274447_spatial_perturb_seq.h5ad"
OUT_DIR   = "/root/code/nicheformer/output/0525/visualize_slides"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading data ...")
adata = sc.read_h5ad(H5AD_PATH)
print(f"Shape: {adata.shape}")
print(f"obs columns: {list(adata.obs.columns)}")
print(f"var names (first 10): {list(adata.var_names[:10])}")
""")

# Cell 1: Per-slide overview
cells.append("""SAMPLES = ["C02943C3", "B03018A2", "A03599E2"]

PERT_COLORS = {
    "Neg":      "#D3D3D3",
    "Clu":      "#E41A1C", "Fasn":     "#377EB8", "Olig2":    "#4DAF4A",
    "Stk39":    "#984EA3", "Trem2":    "#FF7F00", "Gfap":     "#FFFF33",
    "Sh3gl2":   "#A65628", "Rraga":    "#F781BF", "Flcn":     "#66C2A5",
    "Cfap410":  "#FC8D62", "Rbfox3":   "#8DA0CB", "Ndufaf2":  "#E78AC3",
    "Tbk1":     "#A6D854", "Msafe":     "#FFD92F", "Dpp5":     "#E5C494",
    "Srf":      "#B3B3B3", "C9orf72":  "#1B9E77", "Lrrk2":    "#D95F02",
}

print("="*70)
print("Per-slide perturbation summary")
print("="*70)

for sample in SAMPLES:
    mask = adata.obs["sample"] == sample
    sub = adata[mask]
    n = sub.n_obs
    vc = sub.obs["perturbation"].value_counts()
    n_pert = vc[vc.index != "Neg"].sum()
    print(f"\\n{sample}: {n} cells, {n_pert} perturbed ({100*n_pert/n:.2f}%)")
    for pert, count in vc.head(10).items():
        bar = "█" * int(30 * count / vc.max())
        print(f"  {pert:20s}: {count:>6d}  {bar}")
""")

# Cell 2: Spatial scatter per slide
cells.append("""fig, axes = plt.subplots(1, 3, figsize=(24, 8))

for i, sample in enumerate(SAMPLES):
    ax = axes[i]
    mask = adata.obs["sample"] == sample.values
    xy = adata.obsm["spatial"][mask]
    pert = adata.obs["perturbation"][mask].values

    # Background: all cells in light grey
    ax.scatter(xy[:, 0], xy[:, 1], c="#F5F5F5", s=0.3, alpha=0.6, rasterized=True)

    # Perturbed cells
    pert_mask = pert != "Neg"
    xy_p = xy[pert_mask]
    pert_p = pert[pert_mask]

    for p in sorted(set(pert_p)):
        p_mask = pert_p == p
        ax.scatter(xy_p[p_mask, 0], xy_p[p_mask, 1],
                   c=PERT_COLORS.get(p, "#000000"), s=12, alpha=0.9,
                   edgecolors="black", linewidth=0.3, rasterized=True,
                   label=f"{p} ({p_mask.sum()})")

    n_total = len(pert)
    n_pert = pert_mask.sum()
    ax.set_title(f"{sample}\\n{n_total:,} cells, {n_pert} perturbed ({100*n_pert/n_total:.2f}%)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.legend(fontsize=6, markerscale=1.5, loc="upper right", frameon=True, ncol=2)

fig.suptitle("GSE274447 Spatial Perturb-Seq — spatial scatter per slide", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "spatial_scatter_per_slide.png"), dpi=150, bbox_inches="tight")
plt.show()
print("-> spatial_scatter_per_slide.png")
""")

# Cell 3: Perturbation composition bar chart
cells.append("""fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, sample in enumerate(SAMPLES):
    ax = axes[i]
    mask = adata.obs["sample"] == sample
    vc = adata.obs["perturbation"][mask].value_counts()
    vc_pert = vc[vc.index != "Neg"]

    colors = [PERT_COLORS.get(p, "#333") for p in vc_pert.index]
    bars = ax.bar(range(len(vc_pert)), vc_pert.values,
                  color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(vc_pert)))
    ax.set_xticklabels(vc_pert.index, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Number of cells")
    ax.set_title(f"{sample} — {vc_pert.sum()} perturbed cells", fontsize=11, fontweight="bold")
    for bar, val in zip(bars, vc_pert.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(val), ha="center", fontsize=7, fontweight="bold")

fig.suptitle("Perturbation composition per slide", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "perturbation_composition.png"), dpi=150, bbox_inches="tight")
plt.show()
print("-> perturbation_composition.png")
""")

# Cell 4: sgRNA spatial expression
cells.append("""sgrna_genes = [g for g in adata.var_names if "sgrna" in g.lower()]
print(f"sgRNA genes in panel: {len(sgrna_genes)}")
for g in sgrna_genes:
    print(f"  {g}")

for sample in SAMPLES:
    mask = adata.obs["sample"] == sample.values
    sub = adata[mask]
    xy = sub.obsm["spatial"]

    # Get sgRNA expression
    sgrna_idx = [list(sub.var_names).index(g) for g in sgrna_genes]
    sgrna_expr = adata.X[mask][:, sgrna_idx].toarray()

    # Find sgRNAs with expression in this sample
    total_per_sgrna = sgrna_expr.sum(axis=0)
    has_expr = total_per_sgrna > 0
    n_active = has_expr.sum()

    if n_active == 0:
        print(f"\\n{sample}: no sgRNA detected")
        continue

    active_genes = [sgrna_genes[j] for j in range(len(sgrna_genes)) if has_expr[j]]
    active_expr = sgrna_expr[:, has_expr]

    ncols = min(6, n_active)
    nrows = int(np.ceil(n_active / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5))
    axes_flat = axes.flatten() if n_active > 1 else [axes]

    for j in range(n_active):
        ax = axes_flat[j]
        expr_j = active_expr[:, j]
        pos = expr_j > 0
        ax.scatter(xy[:, 0], xy[:, 1], c="#F0F0F0", s=0.2, alpha=0.4, rasterized=True)
        if pos.sum() > 0:
            ax.scatter(xy[pos, 0], xy[pos, 1], c="red", s=15, alpha=0.9,
                       edgecolors="darkred", linewidth=0.3, rasterized=True,
                       label=f"n={pos.sum()}")
            ax.legend(fontsize=6, loc="upper right")
        ax.set_title(active_genes[j], fontsize=8, fontweight="bold")
        ax.set_aspect("equal")
        ax.axis("off")

    for j in range(n_active, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"{sample} — sgRNA barcode expression (spatial)", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"sgrna_spatial_{sample}.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print(f"-> sgrna_spatial_{sample}.png ({n_active} sgRNAs)")
""")

# Cell 5: Top perturbed genes expression (target genes, not sgRNAs)
cells.append("""
# Check expression of the 16 measured target genes in each sample
TARGET_GENES = ["C9orf72", "Cfap410", "Clu", "Fasn", "Flcn", "Gfap", "Lrrk2",
                "Ndufaf2", "Olig2", "Rbfox3", "Rraga", "Sh3gl2", "Srf",
                "Stk39", "Tbk1", "Trem2"]

present_targets = [g for g in TARGET_GENES if g in adata.var_names]
print(f"Target genes in panel: {len(present_targets)}/{len(TARGET_GENES)}")

# Per-sample mean expression of each target gene by perturbation
for sample in SAMPLES:
    mask = adata.obs["sample"] == sample.values
    sub = adata[mask]
    n_pert_types = len([p for p in sub.obs["perturbation"].unique() if p != "Neg"])

    # Build pseudobulk matrix: perturbation x target gene
    pert_list = sorted(sub.obs["perturbation"].unique(),
                       key=lambda x: (x=="Neg", -((sub.obs["perturbation"]==x).sum())))

    # Build heatmap matrix
    matrix = np.zeros((len(pert_list[:15]), len(present_targets)))
    for i, pert in enumerate(pert_list[:15]):
        p_mask = sub.obs["perturbation"] == pert
        for j, gene in enumerate(present_targets):
            idx = list(sub.var_names).index(gene)
            matrix[i, j] = sub.X[p_mask][:, idx].toarray().mean()

    fig, ax = plt.subplots(figsize=(max(12, len(present_targets)*0.8),
                                    max(4, len(pert_list[:15])*0.4)))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=present_targets, yticklabels=pert_list[:15],
                ax=ax, linewidths=0.5, cbar_kws={"label": "Mean raw count"})
    ax.set_title(f"{sample} — target gene expression by perturbation",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"target_expr_heatmap_{sample}.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print(f"-> target_expr_heatmap_{sample}.png")
""")

# Cell 6: Summary
cells.append("""print("="*70)
print("Visualization complete. Output files:")
print("="*70)
for f in sorted(os.listdir(OUT_DIR)):
    size = os.path.getsize(os.path.join(OUT_DIR, f))
    print(f"  {f} ({size/1024:.0f} KB)")
print(f"\\nTotal: {len(os.listdir(OUT_DIR))} files in {OUT_DIR}")
""")

# Build notebook
notebook = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": [],
}
for i, code in enumerate(cells):
    notebook["cells"].append({
        "cell_type": "code", "source": [code], "metadata": {},
        "outputs": [], "execution_count": None,
    })

ipynb_path = os.path.join(OUT_DIR, "visualize_GSE274447_slides.ipynb")
with open(ipynb_path, "w") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
print(f"Notebook written: {ipynb_path} ({len(notebook['cells'])} cells)")
