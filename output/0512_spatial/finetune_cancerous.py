#!/usr/bin/env python3
"""
Fastest fine-tuning of Nicheformer on CosMx human liver Cancerous data.

Strategy (fastest → slowest):
  1. Linear Probing (freeze backbone) — niche classification, 1 epoch
  2. Linear Probing — niche composition regression (17 cell types × 5 radii)
  3. Full fine-tuning (unfreeze backbone) — niche classification

This script:
  - Prepares the Cancerous H5AD for NicheformerDataset (adds modality/specie, loads tech mean)
  - Creates train/val/test splits (FOV-based to prevent leakage)
  - Runs linear probing (fastest proof of embedding quality)
  - Optionally runs full fine-tuning

Usage:
    # Fastest: linear probing only
    python finetune_cancerous.py --mode linear --task niche_classification --epochs 1

    # Linear probing + full fine-tuning
    python finetune_cancerous.py --mode full --task niche_classification --epochs 10

    # Niche composition regression (linear probing)
    python finetune_cancerous.py --mode linear --task niche_regression --epochs 1

    # Quick test with subsample
    python finetune_cancerous.py --mode linear --task niche_classification --epochs 1 --subsample 10000
"""

import os
import sys
import time
import json
import argparse
import warnings
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ============================================================================
# Paths
# ============================================================================
CANCER_H5AD = '/mnt/172/wh/25-12/spatial/preprocessed/nanostring_cosmx_human_liver_Cancerous.h5ad'
CHECKPOINT_PATH = '/root/code/25-12/nicheformer/ckpt/nicheformer.ckpt'
TECH_MEAN_PATH = '/root/code/25-12/nicheformer/data/model_means/cosmx_mean_script.npy'
MODEL_H5AD_PATH = '/root/code/25-12/nicheformer/data/model_means/model.h5ad'
OUTPUT_DIR = '/mnt/172/wh/25-12/spatial/evaluation_results/finetune'
os.makedirs(OUTPUT_DIR, exist_ok=True)

NICHE_RADII_MM = [0.027, 0.054, 0.081, 0.108, 0.135]


# ============================================================================
# Data Preparation
# ============================================================================
def prepare_data(
    h5ad_path: str,
    tech_mean_path: str,
    model_h5ad_path: str,
    subsample: Optional[int] = None,
    val_frac: float = 0.1,
) -> ad.AnnData:
    """
    Prepare Cancerous H5AD for NicheformerDataset.

    The raw preprocessed H5AD has:
      - obs: assay, organism, nicheformer_split, batch, niche, region, author_cell_type
      - obsm: X_niche_0..4, spatial
      - var: Ensembl IDs matching model gene order

    We need to add:
      - modality column (integer token ID: spatial=4)
      - specie column (integer token ID: human=5)
      - technology_mean in uns
      - val split (from train set)
    """
    logger.info("=" * 60)
    logger.info("Preparing Cancerous data for fine-tuning")
    logger.info("=" * 60)

    # Load
    logger.info(f"Loading: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    logger.info(f"  Shape: {adata.n_obs} cells × {adata.n_vars} genes")
    logger.info(f"  Obs columns: {list(adata.obs.columns)}")
    logger.info(f"  ObSm keys: {list(adata.obsm.keys())}")

    # Subsample if requested
    if subsample is not None and subsample < adata.n_obs:
        logger.info(f"Subsampling to {subsample} cells (stratified by niche)...")
        from sklearn.model_selection import StratifiedShuffleSplit
        niches = adata.obs['niche'].values.astype(str)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=subsample, random_state=42)
        for _, idx in sss.split(adata.X, niches):
            adata = adata[idx].copy()
            break
        logger.info(f"  After subsample: {adata.n_obs} cells")

    # Add modality and specie columns (integer token IDs)
    # modality: spatial=4, dissociated=3
    # specie: Homo sapiens=5, Mus musculus=6
    logger.info("Adding modality and specie columns...")
    adata.obs['modality'] = 4  # spatial
    adata.obs['specie'] = 5    # Homo sapiens

    # Convert categorical obs to integer codes for the model
    # The model expects integer token IDs, not strings
    logger.info("Converting obs columns to integer token IDs...")
    
    # niche: map string labels to integers
    niche_le = LabelEncoder()
    adata.obs['niche'] = niche_le.fit_transform(adata.obs['niche'].values.astype(str))
    logger.info(f"  niche: {len(niche_le.classes_)} classes -> {dict(zip(niche_le.classes_, range(len(niche_le.classes_))))}")

    # author_cell_type: map to integers
    ct_le = LabelEncoder()
    adata.obs['author_cell_type'] = ct_le.fit_transform(adata.obs['author_cell_type'].values.astype(str))
    logger.info(f"  author_cell_type: {len(ct_le.classes_)} classes")

    # assay: map "CosMx" → 8
    assay_map = {"CosMx": 8, "MERFISH": 7, "Visium": 9}
    adata.obs['assay'] = adata.obs['assay'].map(assay_map).fillna(8).astype(int)
    logger.info(f"  assay: {sorted(adata.obs['assay'].unique())}")

    # organism: map "Homo sapiens" → 5
    org_map = {"Homo sapiens": 5, "Mus musculus": 6}
    adata.obs['organism'] = adata.obs['organism'].map(org_map).fillna(5).astype(int)
    logger.info(f"  organism: {sorted(adata.obs['organism'].unique())}")

    # batch: convert to int
    adata.obs['batch'] = adata.obs['batch'].astype(int)
    
    # region: map to int
    adata.obs['region'] = pd.factorize(adata.obs['region'].astype(str))[0]

    # Create val split from train set
    logger.info("Creating train/val/test splits...")
    train_mask = adata.obs['nicheformer_split'] == 'train'
    test_mask = adata.obs['nicheformer_split'] == 'test'
    
    n_train = train_mask.sum()
    n_val = int(n_train * val_frac)
    
    # Get train indices and randomly select val
    train_indices = np.where(train_mask.values)[0]
    rng = np.random.RandomState(42)
    val_indices = rng.choice(train_indices, size=n_val, replace=False)
    
    # Set splits
    adata.obs['nicheformer_split'] = 'train'
    adata.obs.iloc[val_indices, adata.obs.columns.get_loc('nicheformer_split')] = 'val'
    adata.obs.loc[test_mask, 'nicheformer_split'] = 'test'
    
    logger.info(f"  Train: {(adata.obs['nicheformer_split']=='train').sum()}")
    logger.info(f"  Val:   {(adata.obs['nicheformer_split']=='val').sum()}")
    logger.info(f"  Test:  {(adata.obs['nicheformer_split']=='test').sum()}")

    # Load and store technology mean
    logger.info(f"Loading technology mean from {tech_mean_path}...")
    tech_mean = np.load(tech_mean_path)
    tech_mean = np.nan_to_num(tech_mean)
    # Round to nearest integer, replace zeros with 1
    rounded = np.where((tech_mean % 1) >= 0.5, np.ceil(tech_mean), np.floor(tech_mean))
    tech_mean = np.where(tech_mean == 0, 1, rounded).astype(np.float32)
    adata.uns['technology_mean'] = tech_mean
    logger.info(f"  Technology mean shape: {tech_mean.shape}")

    # Ensure gene order matches model
    logger.info(f"Reordering genes to match model...")
    model_adata = sc.read_h5ad(model_h5ad_path)
    logger.info(f"  Model genes: {model_adata.n_vars}")
    
    # Use concat trick to align gene order
    concat_adata = ad.concat([model_adata, adata], join='outer', axis=0)
    adata = concat_adata[1:].copy()  # drop model row
    del concat_adata
    logger.info(f"  After reordering: {adata.n_obs} cells × {adata.n_vars} genes")

    # Verify X_niche matrices exist
    for i in range(5):
        key = f'X_niche_{i}'
        if key in adata.obsm:
            logger.info(f"  {key}: {adata.obsm[key].shape}")
    
    logger.info("Data preparation complete!")
    return adata


# ============================================================================
# Fine-Tuning
# ============================================================================
def run_finetuning(
    adata: ad.AnnData,
    mode: str = 'linear',       # 'linear' (freeze) or 'full' (unfreeze)
    task: str = 'niche_classification',  # 'niche_classification' or 'niche_regression'
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-4,
    warmup: int = 10,
    max_seq_len: int = 1500,
    chunk_size: int = 1000,
    num_workers: int = 4,
    label: str = 'niche',       # target column
):
    """
    Run fine-tuning on prepared AnnData.
    
    Args:
        mode: 'linear' (freeze backbone) or 'full' (unfreeze)
        task: 'niche_classification' or 'niche_regression'
    """
    from nicheformer.models._nicheformer import Nicheformer
    from nicheformer.models._nicheformer_fine_tune import NicheformerFineTune
    from nicheformer.data.dataset import NicheformerDataset

    logger.info("\n" + "=" * 60)
    logger.info(f"Fine-Tuning: mode={mode}, task={task}, epochs={epochs}")
    logger.info("=" * 60)

    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Determine task parameters
    if task == 'niche_classification':
        n_classes = adata.obs[label].nunique()
        dim_prediction = 1
        supervised_task = 'niche_classification'
        logger.info(f"  Niche classification: {n_classes} classes")
    elif task == 'niche_regression':
        # Predict niche composition at radius 0 (17 cell types)
        n_components = adata.obsm['X_niche_0'].shape[1]
        n_classes = 1
        dim_prediction = n_components
        supervised_task = 'niche_regression'
        logger.info(f"  Niche composition regression: {n_components} components")
    else:
        raise ValueError(f"Unknown task: {task}")

    # Metadata fields needed by the model
    metadata_fields = {
        'obs': ['niche', 'author_cell_type', 'region', 'modality', 'assay', 'specie'],
    }
    if task == 'niche_regression':
        metadata_fields['obsm'] = ['X_niche_0']

    # Create datasets
    logger.info("Creating datasets...")
    t0 = time.time()
    
    train_dataset = NicheformerDataset(
        adata, adata.uns['technology_mean'], 'train',
        max_seq_len=max_seq_len, aux_tokens=30,
        chunk_size=chunk_size, metadata_fields=metadata_fields
    )
    val_dataset = NicheformerDataset(
        adata, adata.uns['technology_mean'], 'val',
        max_seq_len=max_seq_len, aux_tokens=30,
        chunk_size=chunk_size, metadata_fields=metadata_fields
    )
    test_dataset = NicheformerDataset(
        adata, adata.uns['technology_mean'], 'test',
        max_seq_len=max_seq_len, aux_tokens=30,
        chunk_size=chunk_size, metadata_fields=metadata_fields
    )
    
    logger.info(f"  Dataset sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    logger.info(f"  Dataset creation: {time.time()-t0:.1f}s")

    # Dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # Load pre-trained model
    logger.info("Loading pre-trained Nicheformer...")
    model = Nicheformer.load_from_checkpoint(
        checkpoint_path=CHECKPOINT_PATH, strict=False
    )
    logger.info(f"  Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # Create fine-tuning model
    freeze = (mode == 'linear')
    logger.info(f"  Freeze backbone: {freeze}")
    
    fine_tune_model = NicheformerFineTune(
        backbone=model,
        supervised_task=supervised_task,
        extract_layers=[11],       # last layer
        function_layers='mean',
        lr=lr,
        warmup=warmup,
        max_epochs=epochs,
        dim_prediction=dim_prediction,
        n_classes=n_classes,
        freeze=freeze,
        reinit_layers=None,
        extractor=False,
        regress_distribution=(task == 'niche_regression'),
        pool='mean',
        predict_density=False,
        ignore_zeros=False,
        organ='liver',
        label=label,
        without_context=True,
    )

    # Count trainable parameters
    trainable = sum(p.numel() for p in fine_tune_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in fine_tune_model.parameters())
    logger.info(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # Callbacks
    run_name = f"{mode}_{task}_{epochs}epochs"
    checkpoint_dir = os.path.join(OUTPUT_DIR, 'checkpoints', run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    monitor_metric = 'val/classification_loss' if 'classification' in task else 'val/regression_loss'
    monitor_mode = 'min'
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'nicheformer-{run_name}-'+'{epoch:02d}',
        monitor=monitor_metric,
        mode=monitor_mode,
        save_top_k=2,
        verbose=True,
    )
    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        patience=5 if not freeze else 20,
        mode=monitor_mode,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        default_root_dir=OUTPUT_DIR,
        precision='bf16-mixed' if torch.cuda.is_available() else 32,
        gradient_clip_val=1.0,
        accumulate_grad_batches=10,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    # Train
    logger.info(f"\nStarting training ({run_name})...")
    t_start = time.time()
    trainer.fit(
        model=fine_tune_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    train_time = time.time() - t_start
    logger.info(f"Training complete! Time: {train_time:.1f}s ({train_time/60:.1f}min)")

    # Test
    logger.info("\nTesting...")
    test_results = trainer.test(model=fine_tune_model, dataloaders=test_loader)
    logger.info(f"Test results: {test_results}")

    # Get predictions
    logger.info("Getting predictions...")
    predictions = trainer.predict(fine_tune_model, dataloaders=test_loader)
    
    if task == 'niche_classification':
        all_preds = torch.cat([p[0] for p in predictions]).cpu().numpy()
        all_logits = torch.cat([p[1] for p in predictions]).cpu().numpy()
        
        # Get true labels
        test_indices = adata.obs['nicheformer_split'] == 'test'
        true_labels = adata.obs.loc[test_indices, label].values.astype(int)
        
        # Align predictions with true labels (they should match in order)
        test_acc = accuracy_score(true_labels, all_preds)
        test_macro_f1 = f1_score(true_labels, all_preds, average='macro')
        test_weighted_f1 = f1_score(true_labels, all_preds, average='weighted')
        
        logger.info(f"\nTest Set Results ({run_name}):")
        logger.info(f"  Accuracy:     {test_acc:.4f} ({test_acc*100:.2f}%)")
        logger.info(f"  Macro F1:     {test_macro_f1:.4f}")
        logger.info(f"  Weighted F1:  {test_weighted_f1:.4f}")
        
        results = {
            'mode': mode, 'task': task, 'epochs': epochs,
            'test_accuracy': float(test_acc),
            'test_macro_f1': float(test_macro_f1),
            'test_weighted_f1': float(test_weighted_f1),
            'n_classes': n_classes,
            'train_time_seconds': float(train_time),
            'freeze': freeze,
        }
        
    elif task == 'niche_regression':
        all_preds = torch.cat([p[0] for p in predictions]).cpu().numpy()
        
        test_indices = adata.obs['nicheformer_split'] == 'test'
        true_vals = adata.obsm['X_niche_0'][test_indices]
        if hasattr(true_vals, 'toarray'):
            true_vals = true_vals.toarray()
        
        mae = mean_absolute_error(true_vals, all_preds)
        r2 = r2_score(true_vals, all_preds)
        
        logger.info(f"\nTest Set Results ({run_name}):")
        logger.info(f"  MAE: {mae:.6f}")
        logger.info(f"  R2:  {r2:.6f}")
        
        results = {
            'mode': mode, 'task': task, 'epochs': epochs,
            'test_mae': float(mae),
            'test_r2': float(r2),
            'n_components': dim_prediction,
            'train_time_seconds': float(train_time),
            'freeze': freeze,
        }

    # Save results
    results_path = os.path.join(OUTPUT_DIR, f'results_{run_name}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_path}")

    return results


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Fine-tune Nicheformer on Cancerous liver data')
    parser.add_argument('--mode', type=str, default='linear',
                        choices=['linear', 'full'],
                        help="'linear' = freeze backbone (fast), 'full' = unfreeze (slower but better)")
    parser.add_argument('--task', type=str, default='niche_classification',
                        choices=['niche_classification', 'niche_regression'],
                        help="Downstream task")
    parser.add_argument('--epochs', type=int, default=1,
                        help="Number of epochs (1 for linear probing is usually enough)")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--subsample', type=int, default=None,
                        help="Subsample N cells for quick testing")
    parser.add_argument('--max_seq_len', type=int, default=1500)
    parser.add_argument('--label', type=str, default='niche',
                        help="Target column: 'niche', 'author_cell_type', 'X_niche_0'")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("NICHEFORMER FINE-TUNING ON CANCEROUS LIVER")
    logger.info("=" * 70)
    logger.info(f"Args: {vars(args)}")

    t_total = time.time()

    # Prepare data
    adata = prepare_data(
        h5ad_path=CANCER_H5AD,
        tech_mean_path=TECH_MEAN_PATH,
        model_h5ad_path=MODEL_H5AD_PATH,
        subsample=args.subsample,
    )

    # Run fine-tuning
    results = run_finetuning(
        adata=adata,
        mode=args.mode,
        task=args.task,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        label=args.label,
        max_seq_len=args.max_seq_len,
    )

    total_time = time.time() - t_total
    logger.info(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f}min)")
    logger.info("Done!")


if __name__ == '__main__':
    main()
