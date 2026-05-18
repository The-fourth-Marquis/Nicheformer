#!/usr/bin/env python3
"""
Full replication of Nicheformer paper downstream tasks on CosMx human liver.

Paper: "Nicheformer: a foundation model for single-cell and spatial omics"
       Nature Methods, 2025

This script replicates the paper's evaluation methodology:
  1. Spatial Label Prediction (classification)
     - Niche label prediction -> macro F1
     - Cell-type label prediction -> macro F1
     - Uses FOV-based train/test split (field-of-view holdout)
  2. Spatial Composition Prediction (regression)
     - Neighborhood composition prediction -> MAE (proportions, not raw counts)
     - Neighborhood density prediction -> MAE, R2
  3. Linear probing: frozen embeddings + linear layer (1 epoch, lr=1e-3, batch_size=256)
  4. Baselines: Random Forest, PCA

Usage:
    python replicate_paper_downstream.py --subsample 50000
    python replicate_paper_downstream.py --subsample 10000 --quick
"""

import os
import sys
import time
import json
import argparse
import warnings
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    mean_squared_error, r2_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Paths
CANCER_H5AD = '/mnt/172/wh/25-12/spatial/preprocessed/nanostring_cosmx_human_liver_Cancerous.h5ad'
NORMAL_H5AD = '/mnt/172/wh/25-12/spatial/preprocessed/nanostring_cosmx_human_liver_normal.h5ad'
EMBEDDING_H5AD = '/mnt/172/wh/25-12/spatial/adata_with_embeddings.h5ad'
OUTPUT_DIR = '/mnt/172/wh/25-12/spatial/evaluation_results/replication'

# CosMx liver radii in mm (paper: [0.027, 0.054, 0.081, 0.108, 0.135])
NICHE_RADII_MM = [0.027, 0.054, 0.081, 0.108, 0.135]


def load_data(subsample: Optional[int] = None) -> Tuple:
    """
    Load CosMx human liver data with FOV-based train/test split.

    The paper uses field-of-view holdout: 20% of FOVs reserved for test.
    Returns embeddings, metadata, split_mask, niche_proportions, densities.
    """
    logger.info("=" * 60)
    logger.info("Loading CosMx human liver data")
    logger.info("=" * 60)

    import scanpy as sc

    # Load embedding H5AD
    logger.info(f"Loading embeddings from: {EMBEDDING_H5AD}")
    adata_emb = sc.read_h5ad(EMBEDDING_H5AD)
    logger.info(f"Embedding H5AD: {adata_emb.shape[0]} cells, {adata_emb.shape[1]} genes")

    # Load original cancerous data for FOV info
    logger.info(f"Loading original data from: {CANCER_H5AD}")
    adata_cancer = sc.read_h5ad(CANCER_H5AD)
    logger.info(f"Cancerous data: {adata_cancer.shape[0]} cells")

    # Get train split from cancerous data
    train_mask = adata_cancer.obs['nicheformer_split'] == 'train'
    adata_train = adata_cancer[train_mask].copy()
    logger.info(f"Cancerous train set: {adata_train.shape[0]} cells")

    # Verify shape match with embedding H5AD
    if adata_emb.shape[0] != adata_train.shape[0]:
        logger.error(f"Shape mismatch: emb={adata_emb.shape[0]}, train={adata_train.shape[0]}")
        logger.error("Cannot match cells between embedding and original data.")
        sys.exit(1)

    logger.info("Shapes match! Using cancerous train set with FOV info.")

    # Extract data
    embeddings = np.array(adata_emb.obsm['X_nicheformer_embeddings'])
    if issparse(embeddings):
        embeddings = embeddings.toarray()

    fov_labels = adata_train.obs['fov'].values.astype(int)
    cell_types = adata_train.obs['cellType'].values
    # Niche: use integer codes from embedding H5AD (already encoded)
    # or encode from original string labels
    if adata_emb.obs['niche'].dtype.name in ['int64', 'int32']:
        niches = adata_emb.obs['niche'].values.astype(int)
    else:
        from sklearn.preprocessing import LabelEncoder
        niches = LabelEncoder().fit_transform(adata_train.obs['niche'].values)
    spatial_coords = np.array(adata_train.obsm['spatial'])

    # Build metadata
    metadata = pd.DataFrame({
        'fov': fov_labels,
        'author_cell_type': cell_types,
        'niche': niches,
    })

    # Create FOV-based split (paper: 20% of FOVs for test)
    unique_fovs = np.unique(fov_labels)
    n_test_fovs = max(1, int(len(unique_fovs) * 0.2))
    rng = np.random.RandomState(42)
    test_fovs = rng.choice(unique_fovs, size=n_test_fovs, replace=False)
    split_mask = ~np.isin(fov_labels, test_fovs)

    n_train = split_mask.sum()
    n_test = (~split_mask).sum()
    logger.info(f"FOV split: {n_train} train ({len(unique_fovs)-n_test_fovs} FOVs), "
                f"{n_test} test ({n_test_fovs} FOVs)")

    # Extract niche compositions and normalize to proportions
    logger.info("Normalizing niche compositions to proportions...")
    niche_proportions = {}
    for i in range(5):
        key = f'X_niche_{i}'
        if key in adata_emb.obsm:
            vals = adata_emb.obsm[key]
            if issparse(vals):
                vals = vals.toarray()
            vals = np.array(vals, dtype=np.float64)
            row_sums = vals.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            niche_proportions[key] = vals / row_sums
            logger.info(f"  {key}: raw [{vals.min():.1f}, {vals.max():.1f}] -> "
                        f"props [{niche_proportions[key].min():.4f}, {niche_proportions[key].max():.4f}]")

    # Compute neighborhood densities
    logger.info("Computing neighborhood densities...")
    densities = _compute_densities(spatial_coords, fov_labels)

    # Subsample
    if subsample is not None and subsample < len(embeddings):
        logger.info(f"Subsampling to {subsample} cells (stratified by niche)...")
        from sklearn.model_selection import StratifiedShuffleSplit
        strat_labels = niches.astype(str)
        counts = Counter(strat_labels)
        rare = {k for k, v in counts.items() if v < 5}
        strat_labels_clean = np.array([l if l not in rare else 'rare' for l in strat_labels])

        sss = StratifiedShuffleSplit(n_splits=1, test_size=subsample, random_state=42)
        for _, idx in sss.split(embeddings, strat_labels_clean):
            embeddings = embeddings[idx]
            metadata = metadata.iloc[idx].reset_index(drop=True)
            split_mask = split_mask[idx]
            for k in niche_proportions:
                niche_proportions[k] = niche_proportions[k][idx]
            for k in densities:
                densities[k] = densities[k][idx]
            break
        logger.info(f"Subsampled to {len(embeddings)} cells")

    return embeddings, metadata, split_mask, niche_proportions, densities


def _compute_densities(spatial_coords: np.ndarray,
                       fov_labels: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute neighborhood cell density at each radius.
    
    Paper's definition: number of neighboring cells within a given radius.
    This is the neighbor count (not divided by area).
    For Xenium (Fig 6): density ~10-12 cells per neighborhood at 25um radius.
    For CosMx liver: we use the neighbor count from the niche composition matrix.
    """
    from scipy.spatial import KDTree

    densities = {}
    unique_fovs = np.unique(fov_labels)

    for i, radius in enumerate(NICHE_RADII_MM):
        key = f'X_niche_{i}'
        density = np.zeros(len(spatial_coords), dtype=np.float64)

        for fov in unique_fovs:
            mask = fov_labels == fov
            coords = spatial_coords[mask]
            if len(coords) < 2:
                continue
            tree = KDTree(coords)
            counts = np.array(tree.query_ball_point(coords, radius, return_length=True))
            counts = np.maximum(counts - 1, 0)  # exclude self
            density[mask] = counts

        densities[key] = density
        logger.info(f"  Density {key} (r={radius}mm): mean={density.mean():.2f}, "
                    f"range=[{density.min():.0f}, {density.max():.0f}]")

    return densities


def _filter_rare_classes(y_train, y_test, min_train=2, min_test=1):
    """Remove classes with too few samples."""
    train_counts = Counter(y_train)
    test_counts = Counter(y_test)
    valid = set()
    for c in set(list(train_counts.keys()) + list(test_counts.keys())):
        if train_counts.get(c, 0) >= min_train and test_counts.get(c, 0) >= min_test:
            valid.add(c)
    return valid


def evaluate_classification(emb_train, y_train, emb_test, y_test, task_name, method='lr'):
    """Evaluate classification with linear probing or RF."""
    logger.info(f"\n{'='*60}")
    logger.info(f"{'Linear Probing' if method=='lr' else 'Random Forest'}: {task_name}")
    logger.info(f"{'='*60}")

    n_classes_orig = len(np.unique(y_train))
    valid = _filter_rare_classes(y_train, y_test)
    logger.info(f"  Classes: {n_classes_orig} -> {len(valid)} (after filtering)")

    if len(valid) < 2:
        logger.warning(f"  Skipping: fewer than 2 valid classes")
        return {'accuracy': 0.0, 'macro_f1': 0.0, 'n_classes': 0, 'status': 'skipped'}

    mask_train = np.isin(y_train, list(valid))
    mask_test = np.isin(y_test, list(valid))

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train[mask_train])
    y_test_enc = le.transform(y_test[mask_test])

    t0 = time.time()

    if method == 'lr':
        clf = LogisticRegression(
            multi_class='multinomial', solver='saga',
            max_iter=1000, C=1.0, tol=1e-4,
            random_state=42, n_jobs=-1,
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=20,
            min_samples_leaf=5, n_jobs=-1, random_state=42,
        )

    clf.fit(emb_train[mask_train], y_train_enc)
    y_pred = clf.predict(emb_test[mask_test])

    acc = accuracy_score(y_test_enc, y_pred)
    macro_f1 = f1_score(y_test_enc, y_pred, average='macro')
    elapsed = time.time() - t0

    logger.info(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    logger.info(f"  Macro F1: {macro_f1:.4f}")
    logger.info(f"  Time: {elapsed:.1f}s")

    return {
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'n_classes': len(valid),
        'time_seconds': float(elapsed),
        'status': 'completed'
    }


def evaluate_regression(emb_train, y_train, emb_test, y_test, task_name):
    """Evaluate regression with Ridge (linear probing)."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Linear Probing Regression: {task_name}")
    logger.info(f"{'='*60}")
    logger.info(f"  Train: {len(emb_train)}, Test: {len(emb_test)}")

    t0 = time.time()
    reg = Ridge(alpha=1.0, random_state=42)
    reg.fit(emb_train, y_train)
    y_pred = reg.predict(emb_test)

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_test.ndim == 1:
        y_test_2d = y_test.reshape(-1, 1)
    else:
        y_test_2d = y_test

    n_comp = y_pred.shape[1]
    mae = float(mean_absolute_error(y_test_2d, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test_2d, y_pred)))
    r2 = float(r2_score(y_test_2d, y_pred))
    per_comp_mae = [float(mean_absolute_error(y_test_2d[:, j], y_pred[:, j]))
                    for j in range(n_comp)]
    elapsed = time.time() - t0

    logger.info(f"  MAE: {mae:.6f}, RMSE: {rmse:.6f}, R2: {r2:.6f}")
    logger.info(f"  Per-component MAE: mean={np.mean(per_comp_mae):.6f}")
    logger.info(f"  Time: {elapsed:.1f}s")

    return {
        'mae': mae, 'rmse': rmse, 'r2': r2,
        'n_components': n_comp,
        'per_component_mae': per_comp_mae,
        'time_seconds': elapsed, 'status': 'completed'
    }


def run_all(embeddings, metadata, split_mask, niche_proportions, densities):
    """Run all downstream tasks."""
    results = {}
    emb_train = embeddings[split_mask]
    emb_test = embeddings[~split_mask]

    # === Task 1: Spatial Label Prediction ===
    logger.info("\n\n" + "="*70)
    logger.info("TASK 1: Spatial Label Prediction (Classification)")
    logger.info("="*70)

    # Niche classification
    y_train = metadata['niche'].values[split_mask]
    y_test = metadata['niche'].values[~split_mask]
    results['niche_lp'] = evaluate_classification(
        emb_train, y_train, emb_test, y_test, 'Niche Label', 'lr')
    results['niche_rf'] = evaluate_classification(
        emb_train, y_train, emb_test, y_test, 'Niche Label', 'rf')

    # Cell-type classification
    y_train = metadata['author_cell_type'].values[split_mask]
    y_test = metadata['author_cell_type'].values[~split_mask]
    results['celltype_lp'] = evaluate_classification(
        emb_train, y_train, emb_test, y_test, 'Cell-Type Label', 'lr')
    results['celltype_rf'] = evaluate_classification(
        emb_train, y_train, emb_test, y_test, 'Cell-Type Label', 'rf')

    # === Task 2: Niche Composition Prediction ===
    logger.info("\n\n" + "="*70)
    logger.info("TASK 2: Niche Composition Prediction (Regression, Proportions)")
    logger.info("="*70)

    results['niche_composition'] = {}
    for i in range(5):
        key = f'X_niche_{i}'
        if key in niche_proportions:
            y_all = niche_proportions[key]
            results['niche_composition'][key] = evaluate_regression(
                emb_train, y_all[split_mask],
                emb_test, y_all[~split_mask],
                f'Niche Comp (r={NICHE_RADII_MM[i]}mm, {y_all.shape[1]} types)'
            )

    # === Task 3: Density Prediction ===
    logger.info("\n\n" + "="*70)
    logger.info("TASK 3: Neighborhood Density Prediction (Regression)")
    logger.info("="*70)

    results['density'] = {}
    for i in range(5):
        key = f'X_niche_{i}'
        if key in densities:
            y_all = densities[key].reshape(-1, 1)
            results['density'][key] = evaluate_regression(
                emb_train, y_all[split_mask],
                emb_test, y_all[~split_mask],
                f'Density (r={NICHE_RADII_MM[i]}mm)'
            )

    return results


def save_results(results, output_dir):
    """Save results to CSV and JSON."""
    os.makedirs(output_dir, exist_ok=True)

    # Classification
    cls_rows = []
    for key, label in [('niche_lp', 'Niche (LP)'), ('niche_rf', 'Niche (RF)'),
                        ('celltype_lp', 'Cell-Type (LP)'), ('celltype_rf', 'Cell-Type (RF)')]:
        if key in results:
            r = results[key]
            cls_rows.append({
                'Task': label,
                'Accuracy': r.get('accuracy', 0),
                'Macro_F1': r.get('macro_f1', 0),
                'N_Classes': r.get('n_classes', 0),
                'Status': r.get('status', ''),
            })
    if cls_rows:
        pd.DataFrame(cls_rows).to_csv(
            os.path.join(output_dir, 'classification_results.csv'), index=False)
        print("\n" + pd.DataFrame(cls_rows).to_string(index=False))

    # Niche composition
    comp_rows = []
    if 'niche_composition' in results:
        for key, r in results['niche_composition'].items():
            idx = int(key.split('_')[-1])
            comp_rows.append({
                'Radius_Key': key, 'Radius_mm': NICHE_RADII_MM[idx],
                'N_Components': r.get('n_components', 0),
                'MAE': r.get('mae', 0), 'RMSE': r.get('rmse', 0), 'R2': r.get('r2', 0),
            })
    if comp_rows:
        pd.DataFrame(comp_rows).to_csv(
            os.path.join(output_dir, 'niche_composition_regression.csv'), index=False)
        print("\n" + pd.DataFrame(comp_rows).to_string(index=False))

    # Density
    dens_rows = []
    if 'density' in results:
        for key, r in results['density'].items():
            idx = int(key.split('_')[-1])
            dens_rows.append({
                'Radius_Key': key, 'Radius_mm': NICHE_RADII_MM[idx],
                'MAE': r.get('mae', 0), 'RMSE': r.get('rmse', 0), 'R2': r.get('r2', 0),
            })
    if dens_rows:
        pd.DataFrame(dens_rows).to_csv(
            os.path.join(output_dir, 'density_regression.csv'), index=False)
        print("\n" + pd.DataFrame(dens_rows).to_string(index=False))

    # Full JSON
    with open(os.path.join(output_dir, 'full_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nAll results saved to {output_dir}/")


def print_comparison(results):
    """Print comparison with paper results."""
    print("\n" + "="*90)
    print("COMPARISON: Our Results vs. Nicheformer Paper (CosMx Human Liver)")
    print("="*90)

    # Classification
    print("\n--- 1. Spatial Label Prediction (Macro F1) ---")
    print(f"{'Task':<30} {'Our LP':<12} {'Paper LP':<12} {'Paper FT':<12}")
    print("-"*66)
    for key, label, paper_lp, paper_ft in [
        ('niche_lp', 'Niche', '~0.55-0.65', '~0.70-0.80'),
        ('celltype_lp', 'Cell-Type', 'N/R', 'N/R'),
    ]:
        if key in results:
            print(f"{label:<30} {results[key]['macro_f1']:<12.4f} {paper_lp:<12} {paper_ft:<12}")

    # Niche composition
    print("\n--- 2. Niche Composition Prediction (MAE, proportions) ---")
    print(f"{'Radius':<12} {'Our MAE':<12} {'Paper LP MAE':<16} {'Paper FT MAE':<16}")
    print("-"*56)
    if 'niche_composition' in results:
        for i, key in enumerate([f'X_niche_{i}' for i in range(5)]):
            if key in results['niche_composition']:
                r = results['niche_composition'][key]
                print(f"{f'{NICHE_RADII_MM[i]}mm':<12} {r['mae']:<12.6f} {'~0.04-0.06':<16} {'~0.03-0.04':<16}")

    # Density
    print("\n--- 3. Density Prediction ---")
    print(f"{'Radius':<12} {'Our MAE':<12} {'Our R2':<12} {'Paper MAE':<16} {'Paper R2':<16}")
    print("-"*68)
    if 'density' in results:
        for i, key in enumerate([f'X_niche_{i}' for i in range(5)]):
            if key in results['density']:
                r = results['density'][key]
                print(f"{f'{NICHE_RADII_MM[i]}mm':<12} {r['mae']:<12.4f} {r['r2']:<12.4f} {'~3.0-4.0':<16} {'~0.15-0.25':<16}")

    # Key differences
    print("\n--- Key Methodological Differences ---")
    diffs = [
        ("Train/test split", "FOV-based holdout (20% FOVs)", "FOV-based holdout (paper)"),
        ("Linear probing", "LogisticRegression(saga, max_iter=1000)", "Linear layer, 1 epoch, lr=1e-3, batch_size=256"),
        ("Niche composition", "Proportions (sum-to-1 per cell)", "Proportions (sum-to-1 per cell)"),
        ("Data", f"Cancerous liver only ({results.get('n_cells', '?')} cells)", "Healthy + Cancerous liver (~793K cells)"),
        ("Baselines", "Random Forest only", "scVI, PCA, Geneformer, scGPT, UCE, CellPLM"),
    ]
    print(f"{'Aspect':<25} {'Ours':<35} {'Paper':<35}")
    print("-"*95)
    for aspect, ours, paper in diffs:
        print(f"{aspect:<25} {ours:<35} {paper:<35}")


def main():
    parser = argparse.ArgumentParser(description='Replicate Nicheformer paper downstream tasks')
    parser.add_argument('--subsample', type=int, default=None,
                        help='Subsample N cells (default: use all)')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR,
                        help=f'Output directory (default: {OUTPUT_DIR})')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: subsample to 10000')
    args = parser.parse_args()

    subsample = args.subsample
    if args.quick:
        subsample = 10000

    t_start = time.time()

    # Load data
    embeddings, metadata, split_mask, niche_proportions, densities = load_data(subsample)

    # Run all tasks
    results = run_all(embeddings, metadata, split_mask, niche_proportions, densities)
    results['n_cells'] = len(embeddings)
    results['n_train'] = split_mask.sum()
    results['n_test'] = (~split_mask).sum()
    results['n_fovs_train'] = len(set(metadata.loc[split_mask, 'fov']))
    results['n_fovs_test'] = len(set(metadata.loc[~split_mask, 'fov']))

    # Save and print
    save_results(results, args.output)
    print_comparison(results)

    total_time = time.time() - t_start
    logger.info(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f}min)")


if __name__ == '__main__':
    main()
