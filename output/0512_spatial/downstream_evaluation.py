#!/usr/bin/env python3
"""
Downstream Evaluation of Nicheformer Embeddings

Following the methodology from:
Nicheformer: a foundation model for single-cell and spatial omics
(Nature Methods 2025, Vol. 22, pp. 2525-2538)

Tasks:
  1. Spatial Label Prediction (Classification -> macro F1)
     - Cell-type, niche, region classification via linear probing
  2. Spatial Composition Prediction (Regression -> MAE)
     - Niche composition regression at multiple radii
     - Neighborhood density regression
  3. Fine-Tuning (Full model training with NicheformerFineTune)

Usage:
  python downstream_evaluation.py [--quick] [--finetune] [--subsample N]

  --quick:       Skip fine-tuning, only run linear probing tasks
  --finetune:    Include full model fine-tuning (requires GPU)
  --subsample N: Use N random cells for faster execution (default: 20000)
"""

import os
import sys
import argparse
import warnings
import time
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.sparse import issparse
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    r2_score, mean_squared_error, classification_report
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
DATA_PATH = "/mnt/172/wh/25-12/spatial/adata_with_embeddings.h5ad"
CHECKPOINT_PATH = "/mnt/172/wh/25-12/nicheformer/ckpt/nicheformer.ckpt"
TECH_MEAN_PATH = "/mnt/172/wh/25-12/nicheformer/data/model_means/cosmx_mean_script.npy"
OUTPUT_DIR = "/mnt/172/wh/25-12/spatial/evaluation_results"
FT_OUTPUT_DIR = "/mnt/172/wh/25-12/spatial/checkpoints/nicheformer_ft_logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NICHE_KEYS = ['X_niche_0', 'X_niche_1', 'X_niche_2', 'X_niche_3', 'X_niche_4']
RANDOM_STATE = 42
TEST_SIZE = 0.2
SUBSAMPLE_DEFAULT = 20000  # Use 20k cells by default for speed


# ============================================================================
# 0. Load Data
# ============================================================================
def load_data(subsample=None):
    """Load AnnData with Nicheformer embeddings, optionally subsample."""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    t0 = time.time()
    emb = sc.read_h5ad(DATA_PATH)
    print(f"  Cells: {emb.n_obs}")
    print(f"  Genes: {emb.n_vars}")
    print(f"  Embeddings: {emb.obsm['X_nicheformer_embeddings'].shape}")
    print(f"  Obs columns: {list(emb.obs.columns)}")
    print(f"  ObSm keys: {list(emb.obsm.keys())}")
    for col in ['author_cell_type', 'niche', 'region', 'batch']:
        n_unique = emb.obs[col].nunique()
        vals = sorted(emb.obs[col].unique())
        print(f"  {col}: {n_unique} unique -> {vals[:8]}{'...' if len(vals) > 8 else ''}")

    if subsample is not None and subsample < emb.n_obs:
        print(f"\n  Subsampling to {subsample} cells...")
        rng = np.random.RandomState(RANDOM_STATE)
        idx = rng.choice(emb.n_obs, subsample, replace=False)
        emb = emb[idx].copy()
        print(f"  After subsample: {emb.n_obs} cells")
    print(f"  Load time: {time.time() - t0:.1f}s")
    return emb


# ============================================================================
# 1. Spatial Label Prediction (Linear Probing)
# ============================================================================
def evaluate_linear_probing_classification(emb, label_col, task_name):
    """Linear probing: SGDClassifier (log-loss) on frozen embeddings.
    Uses SGD for memory efficiency on large datasets."""
    t0 = time.time()
    print(f"\n  [{task_name}] Preparing data...", end=' ', flush=True)
    X = emb.obsm['X_nicheformer_embeddings']
    y_raw = emb.obs[label_col].values

    # Encode labels
    if y_raw.dtype not in [np.int32, np.int64, np.float32, np.float64]:
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
    else:
        y = y_raw

    # Filter out rare classes (< 2 samples)
    unique, counts = np.unique(y, return_counts=True)
    rare = unique[counts < 2]
    if len(rare) > 0:
        mask = ~np.isin(y, rare)
        X = X[mask]
        y = y[mask]
        print(f"(removed {len(rare)} rare classes)", end=' ', flush=True)

    n_classes = len(np.unique(y))
    if n_classes < 2:
        print(f"SKIPPED (only {n_classes} class)")
        return {'task': task_name, 'accuracy': 0.0, 'macro_f1': 0.0, 'weighted_f1': 0.0, 'skipped': True}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Scaling...", end=' ', flush=True)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Use SGDClassifier (log-loss = logistic regression) for memory efficiency
    print(f"Training ({n_classes} classes, {X_train_scaled.shape[0]} samples)...", end=' ', flush=True)
    clf = SGDClassifier(
        loss='log_loss', penalty='l2', alpha=1e-4,
        max_iter=1000, tol=1e-3, random_state=RANDOM_STATE,
        n_jobs=-1, verbose=0
    )
    clf.fit(X_train_scaled, y_train)
    print(f"Predicting...", end=' ', flush=True)
    y_pred = clf.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')

    elapsed = time.time() - t0
    print(f"Done [{elapsed:.1f}s]")
    print(f"  Accuracy:    {acc:.4f}  |  Macro F1: {macro_f1:.4f}  |  Weighted F1: {weighted_f1:.4f}")

    return {'task': task_name, 'accuracy': acc, 'macro_f1': macro_f1, 'weighted_f1': weighted_f1}


def evaluate_rf_classification(emb, label_col, task_name):
    """Random Forest baseline."""
    t0 = time.time()
    print(f"\n  [{task_name}] Preparing data...", end=' ', flush=True)
    X = emb.obsm['X_nicheformer_embeddings']
    y_raw = emb.obs[label_col].values

    if y_raw.dtype not in [np.int32, np.int64, np.float32, np.float64]:
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
    else:
        y = y_raw

    # Filter rare classes
    unique, counts = np.unique(y, return_counts=True)
    rare = unique[counts < 2]
    if len(rare) > 0:
        mask = ~np.isin(y, rare)
        X = X[mask]
        y = y[mask]

    n_classes = len(np.unique(y))
    if n_classes < 2:
        print(f"SKIPPED (only {n_classes} class)")
        return {'task': task_name, 'accuracy': 0.0, 'macro_f1': 0.0, 'weighted_f1': 0.0, 'skipped': True}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Training RF ({X_train.shape[0]} samples, {n_classes} classes)...", end=' ', flush=True)
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    print(f"Predicting...", end=' ', flush=True)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')

    elapsed = time.time() - t0
    print(f"Done [{elapsed:.1f}s]")
    print(f"  Accuracy:    {acc:.4f}  |  Macro F1: {macro_f1:.4f}  |  Weighted F1: {weighted_f1:.4f}")

    return {'task': task_name, 'accuracy': acc, 'macro_f1': macro_f1, 'weighted_f1': weighted_f1}


def run_classification_tasks(emb):
    """Run all classification tasks (linear probing + RF baseline)."""
    print("\n" + "=" * 70)
    print("SECTION 1: SPATIAL LABEL PREDICTION")
    print("=" * 70)

    tasks = [
        ('author_cell_type', 'Cell-Type Classification'),
        ('niche', 'Niche Classification'),
        ('region', 'Region Classification'),
    ]

    lr_results = {}
    rf_results = {}

    for label_col, task_name in tasks:
        lr_results[task_name] = evaluate_linear_probing_classification(emb, label_col, task_name)
        rf_results[task_name] = evaluate_rf_classification(emb, label_col, task_name)

    # Summary table
    rows = []
    for task_name, _, _ in [(t[1],) + t for t in tasks]:
        # Hack to iterate properly
        pass

    for label_col, task_name in tasks:
        lr = lr_results[task_name]
        rf = rf_results[task_name]
        rows.append({'Task': task_name, 'Method': 'Linear Probing (LogReg)',
                     'Accuracy': f"{lr['accuracy']:.4f}", 'Macro F1': f"{lr['macro_f1']:.4f}"})
        rows.append({'Task': task_name, 'Method': 'Random Forest',
                     'Accuracy': f"{rf['accuracy']:.4f}", 'Macro F1': f"{rf['macro_f1']:.4f}"})

    summary_df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("CLASSIFICATION RESULTS SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "classification_results.csv"), index=False)
    print(f"\nSaved to: {OUTPUT_DIR}/classification_results.csv")

    # Bar plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, (label_col, task_name) in enumerate(tasks):
        lr = lr_results[task_name]
        rf = rf_results[task_name]
        axes[i].bar(['Linear Probing', 'Random Forest'],
                    [lr['macro_f1'], rf['macro_f1']],
                    color=['#4C72B0', '#DD8452'], alpha=0.8)
        axes[i].set_title(task_name)
        axes[i].set_ylabel('Macro F1')
        axes[i].set_ylim([0, 1])
        for j, v in enumerate([lr['macro_f1'], rf['macro_f1']]):
            axes[i].text(j, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    plt.suptitle('Spatial Label Prediction: Macro F1 Score', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "classification_macro_f1.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {OUTPUT_DIR}/classification_macro_f1.png")

    return lr_results, rf_results, summary_df


# ============================================================================
# 2. Spatial Composition Prediction (Regression)
# ============================================================================
def run_regression_tasks(emb):
    """Run niche composition and density regression tasks."""
    print("\n" + "=" * 70)
    print("SECTION 2: SPATIAL COMPOSITION PREDICTION")
    print("=" * 70)

    X = emb.obsm['X_nicheformer_embeddings']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 2a. Niche Composition Regression ---
    print("\n--- Niche Composition Regression ---")
    comp_results = []
    for niche_key in NICHE_KEYS:
        y = emb.obsm[niche_key]
        if issparse(y):
            y = y.toarray()

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        comp_results.append({
            'niche_key': niche_key, 'n_components': y.shape[1],
            'MAE': mae, 'RMSE': rmse, 'R2': r2
        })
        print(f"  {niche_key:12s} ({y.shape[1]:2d} comp) -> MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

    comp_df = pd.DataFrame(comp_results)
    comp_df.to_csv(os.path.join(OUTPUT_DIR, "niche_composition_regression.csv"), index=False)
    print(f"\nSaved to: {OUTPUT_DIR}/niche_composition_regression.csv")

    # --- 2b. Density Regression ---
    print("\n--- Neighborhood Density Regression ---")
    density_results = []
    for niche_key in NICHE_KEYS:
        y = emb.obsm[niche_key]
        if issparse(y):
            y = y.toarray()
        density = np.array(y.sum(axis=1)).flatten()

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, density, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        density_results.append({
            'niche_key': niche_key, 'MAE': mae, 'RMSE': rmse, 'R2': r2,
            'mean_density': float(np.mean(density)),
            'std_density': float(np.std(density))
        })
        print(f"  {niche_key:12s} -> Density MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f} "
              f"(mean={np.mean(density):.2f})")

    density_df = pd.DataFrame(density_results)
    density_df.to_csv(os.path.join(OUTPUT_DIR, "density_regression.csv"), index=False)
    print(f"\nSaved to: {OUTPUT_DIR}/density_regression.csv")

    # --- 2c. Per-component MAE plot ---
    fig, axes = plt.subplots(1, len(NICHE_KEYS), figsize=(5 * len(NICHE_KEYS), 4))
    for idx, niche_key in enumerate(NICHE_KEYS):
        y = emb.obsm[niche_key]
        if issparse(y):
            y = y.toarray()

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        per_comp_mae = np.mean(np.abs(y_test - y_pred), axis=0)
        axes[idx].bar(range(len(per_comp_mae)), per_comp_mae, color='steelblue', alpha=0.7)
        axes[idx].set_title(f'{niche_key}\n({y.shape[1]} comp)')
        axes[idx].set_xlabel('Component')
        axes[idx].set_ylabel('MAE')
        axes[idx].axhline(y=np.mean(per_comp_mae), color='red', linestyle='--',
                          label=f'Mean={np.mean(per_comp_mae):.4f}')
        axes[idx].legend(fontsize=8)

    plt.suptitle('Per-Component MAE for Niche Composition Prediction', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "niche_composition_per_component_mae.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {OUTPUT_DIR}/niche_composition_per_component_mae.png")

    return comp_df, density_df


# ============================================================================
# 3. Fine-Tuning (Full Model)
# ============================================================================
def run_finetuning(emb):
    """Run full model fine-tuning for niche classification."""
    print("\n" + "=" * 70)
    print("SECTION 3: FINE-TUNING")
    print("=" * 70)

    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from torch.utils.data import DataLoader
    import anndata as ad
    from nicheformer.models._nicheformer import Nicheformer
    from nicheformer.models._nicheformer_fine_tune import NicheformerFineTune
    from nicheformer.data.dataset import NicheformerDataset

    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Config
    config_ft = {
        'batch_size': 16, 'max_seq_len': 1500, 'aux_tokens': 30,
        'chunk_size': 1000, 'num_workers': 4, 'precision': 32,
        'max_epochs': 50, 'lr': 1e-4, 'warmup': 10,
        'gradient_clip_val': 1.0, 'accumulate_grad_batches': 10,
        'extract_layers': [11], 'function_layers': 'mean',
        'freeze': False, 'reinit_layers': None, 'extractor': False,
        'regress_distribution': False, 'pool': 'mean',
        'predict_density': False, 'ignore_zeros': False,
        'organ': 'liver', 'without_context': True,
    }

    # Create splits
    print("\nCreating train/val/test splits...")
    np.random.seed(42)
    adata_ft = ad.read_h5ad(DATA_PATH)
    n_cells = adata_ft.n_obs
    indices = np.random.permutation(n_cells)
    n_train = int(n_cells * 0.7)
    n_val = int(n_cells * 0.15)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    adata_ft.obs['nicheformer_split'] = 'train'
    adata_ft.obs.iloc[train_idx, adata_ft.obs.columns.get_loc('nicheformer_split')] = 'train'
    adata_ft.obs.iloc[val_idx, adata_ft.obs.columns.get_loc('nicheformer_split')] = 'val'
    adata_ft.obs.iloc[test_idx, adata_ft.obs.columns.get_loc('nicheformer_split')] = 'test'
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Load tech mean
    technology_mean = np.load(TECH_MEAN_PATH)
    print(f"  Technology mean shape: {technology_mean.shape}")

    # Create datasets
    print("Creating datasets...")
    metadata_fields = {'obs': ['niche', 'author_cell_type', 'region', 'modality', 'assay', 'specie']}
    train_dataset = NicheformerDataset(adata_ft, technology_mean, 'train',
                                       config_ft['max_seq_len'], config_ft['aux_tokens'],
                                       config_ft['chunk_size'], metadata_fields)
    val_dataset = NicheformerDataset(adata_ft, technology_mean, 'val',
                                     config_ft['max_seq_len'], config_ft['aux_tokens'],
                                     config_ft['chunk_size'], metadata_fields)
    test_dataset = NicheformerDataset(adata_ft, technology_mean, 'test',
                                      config_ft['max_seq_len'], config_ft['aux_tokens'],
                                      config_ft['chunk_size'], metadata_fields)
    print(f"  Dataset sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config_ft['batch_size'],
                              shuffle=True, num_workers=config_ft['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config_ft['batch_size'],
                            shuffle=False, num_workers=config_ft['num_workers'], pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config_ft['batch_size'],
                             shuffle=False, num_workers=config_ft['num_workers'], pin_memory=True)

    # Load model
    print("\nLoading pre-trained Nicheformer...")
    model = Nicheformer.load_from_checkpoint(checkpoint_path=CHECKPOINT_PATH, strict=False)

    n_classes = emb.obs['niche'].nunique()
    print(f"Creating fine-tune model (niche_classification, {n_classes} classes)...")
    fine_tune_model = NicheformerFineTune(
        backbone=model, supervised_task='niche_classification',
        extract_layers=config_ft['extract_layers'],
        function_layers=config_ft['function_layers'],
        lr=config_ft['lr'], warmup=config_ft['warmup'],
        max_epochs=config_ft['max_epochs'],
        dim_prediction=1, n_classes=n_classes,
        freeze=config_ft['freeze'], reinit_layers=config_ft['reinit_layers'],
        extractor=config_ft['extractor'], regress_distribution=False,
        pool=config_ft['pool'], predict_density=config_ft['predict_density'],
        ignore_zeros=config_ft['ignore_zeros'], organ=config_ft['organ'],
        label='niche', without_context=config_ft['without_context']
    )

    # Trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=FT_OUTPUT_DIR,
        filename='nicheformer-ft-niche-{epoch:02d}-{val_classification_loss:.4f}',
        monitor='val/classification_loss', mode='min', save_top_k=3
    )
    early_stop_callback = EarlyStopping(
        monitor='val/classification_loss', patience=10, mode='min'
    )
    trainer = pl.Trainer(
        max_epochs=config_ft['max_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1, default_root_dir=FT_OUTPUT_DIR,
        precision=config_ft['precision'],
        gradient_clip_val=config_ft['gradient_clip_val'],
        accumulate_grad_batches=config_ft['accumulate_grad_batches'],
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_progress_bar=True
    )

    # Train
    print("\nStarting fine-tuning...")
    trainer.fit(model=fine_tune_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("Fine-tuning complete!")

    # Test
    print("\nTesting...")
    test_results = trainer.test(model=fine_tune_model, dataloaders=test_loader)
    print(f"Test results: {test_results}")

    # Predictions
    print("Getting predictions...")
    predictions = trainer.predict(fine_tune_model, dataloaders=test_loader)
    all_preds = torch.cat([p[0] for p in predictions]).cpu().numpy()
    all_logits = torch.cat([p[1] for p in predictions]).cpu().numpy()

    test_indices = adata_ft.obs['nicheformer_split'] == 'test'
    true_labels = adata_ft.obs.loc[test_indices, 'niche'].values.astype(int)

    test_acc = accuracy_score(true_labels, all_preds)
    test_macro_f1 = f1_score(true_labels, all_preds, average='macro')

    print(f"\nTest Set Results (Fine-Tuned Niche Classification):")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Macro F1:  {test_macro_f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(true_labels, all_preds))

    return {'accuracy': test_acc, 'macro_f1': test_macro_f1, 'predictions': all_preds}


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Downstream evaluation of Nicheformer embeddings')
    parser.add_argument('--quick', action='store_true', help='Skip fine-tuning')
    parser.add_argument('--finetune', action='store_true', help='Include fine-tuning')
    parser.add_argument('--subsample', type=int, default=SUBSAMPLE_DEFAULT,
                        help=f'Number of cells to subsample (default: {SUBSAMPLE_DEFAULT})')
    args = parser.parse_args()

    print("=" * 70)
    print("NICHEFORMER DOWNSTREAM EVALUATION")
    print("Following Nature Methods 2025, Vol. 22, pp. 2525-2538")
    print("=" * 70)

    # Load data (with optional subsampling)
    emb = load_data(subsample=args.subsample)

    # Section 1: Classification
    lr_results, rf_results, summary_df = run_classification_tasks(emb)

    # Section 2: Regression
    comp_df, density_df = run_regression_tasks(emb)

    # Section 3: Fine-tuning (optional)
    ft_results = None
    if args.finetune:
        ft_results = run_finetuning(emb)
    elif not args.quick:
        print("\n" + "=" * 70)
        print("SECTION 3: FINE-TUNING (SKIPPED)")
        print("Use --finetune flag to include full model fine-tuning")
        print("=" * 70)

    # Summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print("\nClassification Results:")
    print(summary_df.to_string(index=False))

    print("\nNiche Composition Regression:")
    print(comp_df.to_string(index=False))

    print("\nDensity Regression:")
    print(density_df.to_string(index=False))

    if ft_results:
        print(f"\nFine-Tuning (Niche Classification):")
        print(f"  Accuracy: {ft_results['accuracy']:.4f}")
        print(f"  Macro F1: {ft_results['macro_f1']:.4f}")

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("Done!")


if __name__ == '__main__':
    main()
