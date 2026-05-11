"""
Generalized tokenization script for spatial omics data → Nicheformer Parquet files.

This script is a generalized version of the tokenization logic in
cosmx_human_liver.ipynb, xenium_human_colon.ipynb, and merfish_mouse_brain.ipynb.
It handles:

1. Loading preprocessed H5AD (output from preprocess_spatial.py)
2. Loading model.h5ad for gene ordering alignment
3. Concatenating with model to ensure same gene ordering
4. Mapping categorical obs columns to integer IDs
5. Computing X_niche_{idx} niche composition matrices (if not precomputed)
6. Subsetting by nicheformer_split (train/test)
7. Tokenizing gene expression data
8. Writing Parquet files for MerlinDataModuleDistributed

Usage:
    python tokenize_spatial.py [--config CONFIG_PATH]

    If no config path is given, the script uses the CONFIG dictionary below.
    Edit the CONFIG dictionary for your dataset.

Example:
    python tokenize_spatial.py
    python tokenize_spatial.py --config my_tokenize_config.py
"""

import os
import sys
import math
import logging
import argparse
from typing import Optional, Dict, Any, List

import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import numba
from scipy.sparse import issparse
from sklearn.utils import sparsefuncs

import pyarrow.parquet as pq
import pyarrow
from os.path import join
from tqdm import tqdm

from nicheformer.data.tools.niche_compositions import niche_compositions

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION — Edit this dictionary for your dataset
# ============================================================================
#
# Instructions:
#   1. Set PREPROCESSED_H5AD to the output of preprocess_spatial.py
#   2. Set MODEL_H5AD path (gene ordering reference)
#   3. Set TECHNOLOGY_MEAN path (technology-specific mean expression)
#   4. Set OUT_PATH for Parquet output
#   5. Set label dictionaries (specie_dict, modality_dict, technology_dict, etc.)
#   6. Set NICHE_RADII for X_niche computation
#   7. Set MAX_SEQ_LEN and CHUNK_SIZE for tokenization
# ============================================================================

CONFIG: Dict[str, Any] = {

    # -----------------------------------------------------------------------
    # Paths
    # -----------------------------------------------------------------------
    "preprocessed_h5ad": "/mnt/172/wh/25-12/spatial/preprocessed/nanostring_cosmx_human_liver_normal.h5ad",
    "model_h5ad": "/root/code/25-12/nicheformer/data/model_means/model.h5ad",
    "technology_mean": "/root/code/25-12/nicheformer/data/model_means/cosmx_mean_script.npy",
    "out_path": "/mnt/172/wh/25-12/spatial/tokenized/cosmx_liver",

    # -----------------------------------------------------------------------
    # Dataset name (used for output filenames)
    # -----------------------------------------------------------------------
    "dataset_name": "cosmx_human_liver",

    # -----------------------------------------------------------------------
    # Label dictionaries (string → integer ID)
    # -----------------------------------------------------------------------
    # These map categorical obs values to integer IDs for tokenization.
    # The IDs must be consistent across all datasets in the Nicheformer corpus.
    #
    # Modality: 3 = dissociated, 4 = spatial
    "modality_dict": {
        "dissociated": 3,
        "spatial": 4,
    },

    # Species: 5 = human, 6 = mouse
    "specie_dict": {
        "human": 5,
        "Homo sapiens": 5,
        "Mus musculus": 6,
        "mouse": 6,
    },

    # Technology/assay: 7 = MERFISH, 8 = CosMx, 9 = Visium, etc.
    "technology_dict": {
        "merfish": 7,
        "MERFISH": 7,
        "cosmx": 8,
        "CosMx": 8,
        "NanoString digital spatial profiling": 8,
        "visium": 9,
        "10x 5' v2": 10,
        "10x 3' v3": 11,
        "10x 3' v2": 12,
        "10x 5' v1": 13,
        "10x 3' v1": 14,
        "10x 3' transcription profiling": 15,
        "10x transcription profiling": 15,
        "10x 5' transcription profiling": 16,
        "CITE-seq": 17,
        "Smart-seq v4": 18,
        "xenium": 19,
        "Xenium": 19,
    },

    # Niche labels (tissue zones / regions)
    "niche_label_dict": {
        "nan": 0,
        # Add your niche labels here, e.g.:
        # "Zone_1a/PV": 0,
        # "tumor": 8,
    },

    # Region labels
    "region_label_dict": {
        "nan": 0,
        # Add your region labels here
    },

    # Author cell type labels
    "author_cell_type_dict": {
        # Add your cell type labels here, e.g.:
        # "Hepatocytes": 0,
        # "T_cells": 1,
    },

    # -----------------------------------------------------------------------
    # Niche composition radii (X_niche_{idx} computation)
    # -----------------------------------------------------------------------
    # If X_niche_{idx} is already in the preprocessed H5AD obsm, set this to
    # an empty list []. Otherwise, provide radii to compute them here.
    # The radii should be in the same units as your spatial coordinates.
    #
    # For CosMx liver (coordinates in mm): [0.027, 0.054, 0.081, 0.108, 0.135]
    # For Xenium (coordinates in um):      [25, 50, 75, 100, 125]
    # For MERFISH (coordinates in um):     [50, 100, 150, 200, 250]
    "niche_radii": [0.027, 0.054, 0.081, 0.108, 0.135],

    # Number of X_niche matrices (must match len(niche_radii) or number in obsm)
    "n_niche": 5,

    # -----------------------------------------------------------------------
    # Tokenization parameters
    # -----------------------------------------------------------------------
    "max_seq_len": 4096,       # Max number of genes per cell
    "aux_tokens": 30,          # Reserved tokens for special tokens
    "chunk_size": 10_000,      # Cells per Parquet batch
    "row_group_size": 1024,    # Parquet row group size

    # -----------------------------------------------------------------------
    # Obs columns to include in Parquet output
    # -----------------------------------------------------------------------
    "parquet_columns": [
        "assay",
        "specie",
        "modality",
        "idx",
        "author_cell_type",
        "niche",
        "region",
    ],

    # -----------------------------------------------------------------------
    # Whether to shuffle cells within each Parquet batch
    # -----------------------------------------------------------------------
    "shuffle_within_batch": True,
}


# ============================================================================
# Tokenization functions (from cosmx_human_liver.ipynb)
# ============================================================================

def sf_normalize(X: np.ndarray) -> np.ndarray:
    """Size-factor normalize to 10,000 counts per cell."""
    X = X.copy()
    counts = np.array(X.sum(axis=1))
    counts += counts == 0.  # avoid zero division
    scaling_factor = 10000. / counts
    if issparse(X):
        sparsefuncs.inplace_row_scale(X, scaling_factor)
    else:
        np.multiply(X, scaling_factor.reshape((-1, 1)), out=X)
    return X


@numba.jit(nopython=True, nogil=True)
def _sub_tokenize_data(
    x: np.ndarray,
    max_seq_len: int = -1,
    aux_tokens: int = 30,
) -> np.ndarray:
    """
    Tokenize expression values to gene token IDs.

    For each cell, selects the top expressed genes (up to max_seq_len)
    and maps them to token IDs (gene_index + aux_tokens offset).
    """
    n_cells = x.shape[0]
    n_genes = x.shape[1]
    seq_len = max_seq_len if max_seq_len > 0 else n_genes
    scores_final = np.empty((n_cells, seq_len), dtype=np.int32)

    for i in range(n_cells):
        cell = x[i]
        nonzero_mask = np.nonzero(cell)[0]
        # Sort by expression value descending, take top max_seq_len
        sorted_indices = nonzero_mask[np.argsort(-cell[nonzero_mask])][:max_seq_len]
        sorted_indices = sorted_indices + aux_tokens  # offset for special tokens
        scores = np.zeros(seq_len, dtype=np.int32)
        scores[:len(sorted_indices)] = sorted_indices.astype(np.int32)
        scores_final[i, :] = scores

    return scores_final


def tokenize_data(
    x: np.ndarray,
    median_counts_per_gene: np.ndarray,
    max_seq_len: int = 4096,
) -> np.ndarray:
    """
    Tokenize gene expression matrix to token IDs.

    Steps:
    1. Replace NaN with 0
    2. Size-factor normalize to 10,000 counts
    3. Divide by median counts per gene (technology mean)
    4. Select top-expressed genes and map to token IDs

    Parameters
    ----------
    x
        Cells × Genes expression matrix.
    median_counts_per_gene
        Technology-specific median expression per gene (from model_means).
    max_seq_len
        Maximum number of genes to keep per cell.

    Returns
    -------
    Tokenized matrix of shape (n_cells, max_seq_len) with dtype int32.
    """
    x = np.nan_to_num(x)  # fill NaN with 0
    x = sf_normalize(x)
    median_counts_per_gene = median_counts_per_gene.copy()
    median_counts_per_gene += median_counts_per_gene == 0
    out = x / median_counts_per_gene.reshape((1, -1))
    scores_final = _sub_tokenize_data(out, max_seq_len, 30)
    return scores_final.astype('i4')


# ============================================================================
# Tokenization pipeline
# ============================================================================

def tokenize(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Run the full tokenization pipeline.

    Parameters
    ----------
    config
        Configuration dictionary. If None, uses CONFIG above.
    """
    if config is None:
        config = CONFIG

    out_path = config["out_path"]
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(join(out_path, "train"), exist_ok=True)
    os.makedirs(join(out_path, "test"), exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load model.h5ad (gene ordering reference)
    # -----------------------------------------------------------------------
    logger.info(f"Loading model from {config['model_h5ad']}")
    model = sc.read_h5ad(config["model_h5ad"])
    logger.info(f"Model: {model.n_vars} genes")

    # -----------------------------------------------------------------------
    # 2. Load technology mean
    # -----------------------------------------------------------------------
    logger.info(f"Loading technology mean from {config['technology_mean']}")
    tech_mean = np.load(config["technology_mean"])
    tech_mean = np.nan_to_num(tech_mean)
    rounded = np.where((tech_mean % 1) >= 0.5, np.ceil(tech_mean), np.floor(tech_mean))
    tech_mean = np.where(tech_mean == 0, 1, rounded)
    logger.info(f"Technology mean shape: {tech_mean.shape}")

    # -----------------------------------------------------------------------
    # 3. Load preprocessed data
    # -----------------------------------------------------------------------
    logger.info(f"Loading preprocessed data from {config['preprocessed_h5ad']}")
    adata = sc.read_h5ad(config["preprocessed_h5ad"])
    logger.info(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

    # -----------------------------------------------------------------------
    # 4. Concatenate with model for gene ordering alignment
    # -----------------------------------------------------------------------
    # This ensures the data has the same gene ordering as the model.
    # The model's first observation is dropped (it's a placeholder).
    logger.info("Concatenating with model for gene ordering alignment ...")
    combined = ad.concat([model, adata], join='outer', axis=0)
    adata = combined[1:].copy()  # drop model's first obs
    del combined
    logger.info(f"After alignment: {adata.n_obs} cells × {adata.n_vars} genes")

    # -----------------------------------------------------------------------
    # 5. Select and prepare obs columns
    # -----------------------------------------------------------------------
    # Keep only the columns needed for tokenization
    required_obs = [
        "assay", "organism", "nicheformer_split", "batch",
        "niche", "region", "author_cell_type"
    ]
    available_obs = [c for c in required_obs if c in adata.obs.columns]
    missing_obs = [c for c in required_obs if c not in adata.obs.columns]
    if missing_obs:
        logger.warning(f"Missing obs columns: {missing_obs}")
    adata.obs = adata.obs[available_obs].copy()

    # Add modality and specie columns
    adata.obs["modality"] = "spatial"
    adata.obs["specie"] = adata.obs["organism"].values

    # -----------------------------------------------------------------------
    # 6. Map categorical columns to integer IDs
    # -----------------------------------------------------------------------
    logger.info("Mapping categorical columns to integer IDs ...")
    adata.obs.replace({"specie": config["specie_dict"]}, inplace=True)
    adata.obs.replace({"modality": config["modality_dict"]}, inplace=True)
    adata.obs.replace({"assay": config["technology_dict"]}, inplace=True)
    adata.obs.replace({"author_cell_type": config["author_cell_type_dict"]}, inplace=True)
    adata.obs.replace({"niche": config["niche_label_dict"]}, inplace=True)
    adata.obs.replace({"region": config["region_label_dict"]}, inplace=True)

    # Check for any unmapped values
    for col in ["specie", "modality", "assay", "author_cell_type", "niche", "region"]:
        if col in adata.obs.columns:
            unmapped = adata.obs[col].isna().sum()
            if unmapped > 0:
                unique_vals = adata.obs[col].unique()
                logger.warning(
                    f"  {unmapped} unmapped values in '{col}'. "
                    f"Unique values: {unique_vals}"
                )
                # Fill unmapped with 0
                adata.obs[col] = adata.obs[col].fillna(0).astype(int)

    # Convert to int
    for col in ["specie", "modality", "assay", "author_cell_type", "niche", "region"]:
        if col in adata.obs.columns:
            adata.obs[col] = adata.obs[col].astype(int)

    logger.info(f"Obs columns after mapping: {list(adata.obs.columns)}")

    # -----------------------------------------------------------------------
    # 7. Compute X_niche compositions (if not already present)
    # -----------------------------------------------------------------------
    radii = config.get("niche_radii", [])
    n_niche = config.get("n_niche", 5)

    # Check if X_niche matrices already exist
    existing_niche = [k for k in adata.obsm.keys() if k.startswith("X_niche_")]
    if len(existing_niche) >= n_niche:
        logger.info(f"X_niche matrices already present: {existing_niche}")
    elif len(radii) > 0:
        logger.info(f"Computing X_niche compositions for radii: {radii} ...")
        # Ensure spatial coordinates are in obsm
        if "spatial" not in adata.obsm:
            if "x" in adata.obs.columns and "y" in adata.obs.columns:
                adata.obsm["spatial"] = np.array(
                    adata.obs[["x", "y"]]
                ).astype(np.float64)
            else:
                raise KeyError(
                    "Spatial coordinates not found in obsm or obs. "
                    "Run preprocess_spatial.py first."
                )
        adata = niche_compositions(adata, radii)
    else:
        logger.warning(
            "No X_niche matrices found and no radii provided. "
            "Tokenization will proceed without X_niche columns."
        )

    # -----------------------------------------------------------------------
    # 8. Tokenize each split (train, test)
    # -----------------------------------------------------------------------
    for split in ["train", "test"]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Tokenizing split: {split}")
        logger.info(f"{'='*60}")

        # Subset by nicheformer_split
        split_mask = adata.obs["nicheformer_split"] == split
        n_split = split_mask.sum()
        if n_split == 0:
            logger.warning(f"  No cells for split '{split}', skipping.")
            continue
        logger.info(f"  Cells: {n_split}")

        adata_split = adata[split_mask].copy()
        adata_split.obs.reset_index(drop=True, inplace=True)

        # Save intermediate H5AD (optional, for debugging)
        # adata_split.write(join(out_path, f"{config['dataset_name']}_{split}_ready_to_tokenize.h5ad"))

        # Prepare obs DataFrame
        obs_split = adata_split.obs.copy()

        # Calculate batches
        n_batches = math.ceil(obs_split.shape[0] / config["chunk_size"])
        batch_indices = np.array_split(obs_split.index, n_batches)
        chunk_len = len(batch_indices[0])
        logger.info(f"  N_BATCHES: {n_batches}, chunk_len: {chunk_len}")

        # Add idx column (original index)
        obs_split = obs_split.reset_index().rename(columns={"index": "idx"})
        obs_split["idx"] = obs_split["idx"].astype("i8")

        # Select Parquet columns
        parquet_cols = config["parquet_columns"]
        available_pq_cols = [c for c in parquet_cols if c in obs_split.columns]
        obs_split = obs_split[available_pq_cols].copy()

        # Tokenize in batches
        logger.info(f"  Tokenizing {split} ...")
        for batch in tqdm(range(n_batches)):
            start = batch * chunk_len
            end = chunk_len * (batch + 1)

            # Get batch obs
            obs_tokens = obs_split.iloc[start:end].copy()

            # Tokenize gene expression
            X_batch = adata_split.X[start:end]
            if issparse(X_batch):
                X_batch = X_batch.toarray()
            tokenized = tokenize_data(X_batch, tech_mean, config["max_seq_len"])
            obs_tokens["X"] = [tokenized[i, :] for i in range(tokenized.shape[0])]

            # Add X_niche columns
            for i in range(n_niche):
                niche_key = f"X_niche_{i}"
                if niche_key in adata_split.obsm:
                    niche_batch = adata_split.obsm[niche_key].toarray()[start:end]
                    obs_tokens[niche_key] = [niche_batch[j, :] for j in range(niche_batch.shape[0])]

            # Shuffle within batch (mixes spatial and dissociated data)
            if config.get("shuffle_within_batch", True):
                obs_tokens = obs_tokens.sample(frac=1)

            # Write Parquet
            table = pyarrow.Table.from_pandas(obs_tokens)
            pq.write_table(
                table,
                join(out_path, split, f"tokens-{batch}.parquet"),
                row_group_size=config["row_group_size"],
            )

        logger.info(f"  Done tokenizing {split}")

    logger.info(f"\nTokenization complete! Output in: {out_path}")
    logger.info(f"  Train: {join(out_path, 'train')}")
    logger.info(f"  Test:  {join(out_path, 'test')}")


# ============================================================================
# CLI entry point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenize spatial omics data for Nicheformer."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a Python file with a CONFIG dictionary (optional).",
    )
    parser.add_argument(
        "--preprocessed_h5ad",
        type=str,
        default=None,
        help="Override: path to preprocessed H5AD file.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=None,
        help="Override: output directory for Parquet files.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Override: dataset name for output filenames.",
    )
    args = parser.parse_args()

    # Load config from file if provided
    if args.config:
        import importlib.util
        spec = importlib.util.spec_from_file_location("user_config", args.config)
        user_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(user_config)
        config = user_config.CONFIG
        logger.info(f"Loaded config from {args.config}")
    else:
        config = dict(CONFIG)  # copy

    # Apply CLI overrides
    if args.preprocessed_h5ad:
        config["preprocessed_h5ad"] = args.preprocessed_h5ad
    if args.out_path:
        config["out_path"] = args.out_path
    if args.dataset_name:
        config["dataset_name"] = args.dataset_name

    tokenize(config)
