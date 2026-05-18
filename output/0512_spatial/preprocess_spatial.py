"""
Generalized preprocessing script for spatial omics data → Nicheformer tokenization.

This script is a generalized version of preprocess_cosmx_liver.py that works with
any spatial dataset (CosMx, Xenium, MERFISH, etc.). It handles:

 1. Load raw H5AD from any spatial technology
 2. Map gene symbols → Ensembl IDs (via mygene) if needed
 3. Set CELLxGENE schema columns (ontology term IDs)
 4. Pre-filter invalid genes before setting adata.raw
 5. Run CELLxGENE + Nicheformer validation
 6. Set nicheformer-specific fields (nicheformer_split, library_key, etc.)
 7. Compute X_niche_{idx} niche composition matrices (optional)
 8. Save the preprocessed H5AD
 9. Reorder genes to match model gene order (via concat with model.h5ad)
10. Load & process technology-specific mean vector (stored in uns)
11. Map obs columns to integer token IDs using dictionaries
12. Split by nicheformer_split and save ready-to-tokenize H5AD files

Usage:
    python preprocess_spatial.py --config-yaml config.yaml

    If no config path is given, the script uses the CONFIG dictionary below.
    Edit the CONFIG dictionary for your dataset.

Example:
    python preprocess_spatial.py --config-yaml preprocess_spatial_config.yaml
"""

import os
import logging
import argparse
from typing import Optional, Dict, Any, List

import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
from scipy.sparse import csr_matrix
import mygene
from cellxgene_schema.ontology import GeneChecker, SupportedOrganisms

from nicheformer.data.constants import (
    DefaultPaths,
    ObsConstants,
    UnsConstants,
    VarConstants,
    AssayOntologyTermId,
    SexOntologyTermId,
    OrganismOntologyTermId,
    TissueOntologyTermId,
    SuspensionTypeId,
)
from nicheformer.data.validate import validate
from nicheformer.data.tools.niche_compositions import niche_compositions

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION — Edit this dictionary for your dataset
# ============================================================================
#
# Instructions:
#   1. Set RAW_H5AD to the path of your raw input file
#   2. Set OUTPUT_NAME for the preprocessed output file
#   3. Set ontology constants (ASSAY, SEX, ORGANISM, TISSUE, SUSPENSION_TYPE)
#   4. Set metadata (DATASET_TITLE, DONOR_ID, CONDITION_ID)
#   5. Set spatial coordinate column names (SPATIAL_X_COL, SPATIAL_Y_COL)
#   6. Set LIBRARY_KEY value (how to group cells for spatial neighbors)
#   7. Set SPLIT_COLUMN for train/test split strategy
#   8. Set NICHE_RADII for X_niche computation (optional)
#   9. Set GENE_MAPPING config (symbol → Ensembl ID)
# ============================================================================

CONFIG: Dict[str, Any] = {

    # -----------------------------------------------------------------------
    # Input / Output paths
    # -----------------------------------------------------------------------
    "raw_h5ad": os.path.join(
        DefaultPaths.SPATIAL,
        "raw",
        "cosmx_liver_full_h5ad_export",
        "cosmx_NormalLiver_fullvis.h5ad",
    ),
    "output_name": "nanostring_cosmx_human_liver_normal.h5ad",
    "output_dir": os.path.join(DefaultPaths.SPATIAL, "preprocessed"),

    # -----------------------------------------------------------------------
    # Ontology constants (from nicheformer.data.constants)
    # -----------------------------------------------------------------------
    # Available AssayOntologyTermId values:
    #   COSMX_SPATIAL="EFO:0030029", XENIUM="EFO:0030046", MERFISH="EFO:0030062",
    #   VISIUM="EFO:0010961", SLIDE_SEQ="EFO:0030004", STAR_MAP="EFO:0030061",
    #   DISSOCIATED="unknown"
    "assay": str(AssayOntologyTermId.COSMX_SPATIAL.value),  # EFO:0030029

    # Available SexOntologyTermId values:
    #   MALE="PATO:0000384", FEMALE="PATO:0000383",
    #   HERMAPHRODITE="PATO:0001340", UNKNOWN="unknown"
    "sex": str(SexOntologyTermId.MALE.value),

    # Available OrganismOntologyTermId values:
    #   HUMAN="NCBITaxon:9606", MOUSE="NCBITaxon:10090"
    "organism": str(OrganismOntologyTermId.HUMAN.value),
    "organism_validator": "human",  # used by validate()

    # Available TissueOntologyTermId values:
    #   LIVER="UBERON:0002107", LUNG="UBERON:0002048", BRAIN="UBERON:0000955",
    #   COLON="UBERON:0001155", BREAST="UBERON:0000310", etc.
    "tissue": str(TissueOntologyTermId.LIVER.value),  # UBERON:0002107

    # Available SuspensionTypeId values:
    #   SPATIAL="na", DISSOCIATED="cell"
    "suspension_type": str(SuspensionTypeId.SPATIAL.value),

    "tissue_type": "tissue",

    # -----------------------------------------------------------------------
    # Dataset metadata
    # -----------------------------------------------------------------------
    "dataset_title": "nanostring_cosmx_human_liver_normal",
    "donor_id": "NormalLiver",
    "condition_id": "normal",

    # -----------------------------------------------------------------------
    # Spatial coordinate column names in adata.obs
    # -----------------------------------------------------------------------
    # These are the column names in your raw H5AD that contain spatial
    # coordinates. Common names:
    #   CosMx:  "x_slide_mm", "y_slide_mm"
    #   Xenium: "x_location", "y_location"
    #   MERFISH: "x", "y"  (or "global_x", "global_y")
    "spatial_x_col": "x_slide_mm",
    "spatial_y_col": "y_slide_mm",

    # -----------------------------------------------------------------------
    # Library key — how cells are grouped for spatial neighbor computation
    # -----------------------------------------------------------------------
    # This is used by squidpy.gr.spatial_neighbors(library_key=...).
    # It groups cells by tissue section/FOV so neighbors are only computed
    # within the same section. Common values:
    #   "section" — all cells in one group (simple case)
    #   "fov" — group by field of view
    #   "Run_Tissue_name" — group by tissue section (CosMx)
    #   "slide_ID_numeric" — group by slide (CosMx)
    #   A column name from adata.obs — use an existing column
    "library_key": "section",

    # -----------------------------------------------------------------------
    # Train/test split strategy
    # -----------------------------------------------------------------------
    # Best practice: split by physical sample/slide to avoid data leakage.
    # Set split_column to a column in adata.obs that groups cells by
    # tissue section (e.g., "Run_Tissue_name", "slide_ID_numeric").
    # If the column doesn't exist, falls back to random 10% cell-level split.
    # Set to None to use random cell-level split always.
    "split_column": "Run_Tissue_name",  # or None for random split
    "split_random_seed": 42,
    "split_test_frac": 0.1,  # fraction of cells/samples for test
    "split_test_samples_frac": 0.2,  # fraction of unique samples for test (slide-level)

    # -----------------------------------------------------------------------
    # Niche composition radii (X_niche_{idx} computation)
    # -----------------------------------------------------------------------
    # These radii are used by niche_compositions() to compute X_niche_{idx}
    # matrices. Set to None to skip X_niche computation (compute later in
    # tokenization notebook). The radii should be in the same units as your
    # spatial coordinates (typically mm for CosMx, um for Xenium).
    #
    # For CosMx liver (coordinates in mm): [0.027, 0.054, 0.081, 0.108, 0.135]
    # For Xenium (coordinates in um):      [25, 50, 75, 100, 125]
    # For MERFISH (coordinates in um):     [50, 100, 150, 200, 250]
    "niche_radii": None,  # e.g., [0.027, 0.054, 0.081, 0.108, 0.135]

    # -----------------------------------------------------------------------
    # Gene mapping configuration
    # -----------------------------------------------------------------------
    # If your H5AD has gene symbols as var_names (not Ensembl IDs), set
    # map_genes=True to map them via mygene. If it already has Ensembl IDs,
    # set map_genes=False.
    "map_genes": True,
    "gene_mapping_species": "human",
    # Column in adata.var that contains original gene symbols (if var_names
    # are already Ensembl IDs but symbols are in a var column)
    "gene_symbol_var_col": None,  # e.g., "gene_name" or None

    # -----------------------------------------------------------------------
    # Cell type column
    # -----------------------------------------------------------------------
    # Column in adata.obs that contains cell type annotations. This is used
    # for author_cell_type and for X_niche computation.
    "cell_type_col": "cellType",  # e.g., "cellType", "cluster", "cell_type"

    # -----------------------------------------------------------------------
    # Assay name for obs['assay'] column (used in tokenization for technology_dict mapping)
    # -----------------------------------------------------------------------
    # This is the string value that will be placed in the 'assay' obs column
    # before being mapped to an integer token ID via technology_dict.
    # Examples: "CosMx", "Xenium", "MERFISH", "Visium"
    "assay_name": "CosMx",

    # -----------------------------------------------------------------------
    # Niche label column (optional)
    # -----------------------------------------------------------------------
    # Column in adata.obs that contains niche/region labels (e.g., tumor vs
    # normal zones). Set to None if not available.
    "niche_col": "niche",

    # -----------------------------------------------------------------------
    # Batch column
    # -----------------------------------------------------------------------
    # Column in adata.obs to use as batch identifier. The tokenization
    # notebook selects 'batch' from obs. Common choices:
    #   "slide_ID_numeric", "Run_name", "fov"
    "batch_col": "slide_ID_numeric",

    # -----------------------------------------------------------------------
    # Layer to use as raw counts
    # -----------------------------------------------------------------------
    # If your H5AD has a specific layer with raw counts (e.g., "counts"),
    # set this. Otherwise uses adata.X.
    "counts_layer": None,  # e.g., "counts" or None

    # -----------------------------------------------------------------------
    # Additional obs columns to preserve
    # -----------------------------------------------------------------------
    # List of additional obs columns from the raw data to keep in the output.
    # These are preserved alongside the required Nicheformer columns.
    "preserve_obs_cols": [],

    # -----------------------------------------------------------------------
    # Model & gene ordering (post-validation)
    # -----------------------------------------------------------------------
    # Path to model.h5ad — used to align gene order via concat.
    # Set to null to skip gene reordering.
    "model_h5ad": os.path.join(DefaultPaths.MODEL_MEANS, "model.h5ad"),

    # -----------------------------------------------------------------------
    # Technology-specific mean (post-validation)
    # -----------------------------------------------------------------------
    # Path to the technology-specific mean .npy file.
    # Set to null to skip loading.
    "technology_mean_path": os.path.join(
        DefaultPaths.MODEL_MEANS, "cosmx_mean_script.npy"
    ),
    # Technology name used for assay mapping (e.g., "CosMx", "Xenium", "MERFISH")
    "technology_name": "CosMx",

    # -----------------------------------------------------------------------
    # Tokenization dictionaries (obs column → integer ID mapping)
    # -----------------------------------------------------------------------
    # These map string labels in obs columns to integer token IDs used by
    # the Nicheformer model. Edit these to match your dataset's labels.
    # Set to null for a column to skip mapping (keep original string values).
    "tokenization_dicts": {
        # modality_dict: 'dissociated' → 3, 'spatial' → 4
        "modality_dict": {"spatial": 4, "dissociated": 3},
        # specie_dict: 'Homo sapiens' → 5, 'Mus musculus' → 6
        "specie_dict": {"Homo sapiens": 5, "human": 5, "Mus musculus": 6, "mouse": 6},
        # technology_dict: 'merfish' → 7, 'cosmx' → 8, 'visium' → 9, etc.
        "technology_dict": {
            "merfish": 7, "MERFISH": 7,
            "cosmx": 8, "CosMx": 8,
            "NanoString digital spatial profiling": 8,
            "visium": 9, "Visium": 9,
            "10x 5' v2": 10, "10x 3' v3": 11, "10x 3' v2": 12,
            "10x 5' v1": 13, "10x 3' v1": 14,
            "10x 3' transcription profiling": 15,
            "10x transcription profiling": 15,
            "10x 5' transcription profiling": 16,
            "CITE-seq": 17, "Smart-seq v4": 18,
        },
        # author_cell_type_dict: map each unique cell type to an integer
        # Set to null to auto-generate from unique values in the data.
        "author_cell_type_dict": None,
        # niche_dict: map each niche label to an integer
        "niche_dict": None,
        # region_dict: map each region label to an integer
        "region_dict": None,
    },

    # -----------------------------------------------------------------------
    # Ready-to-tokenize output
    # -----------------------------------------------------------------------
    # Directory to save the split-ready H5AD files (one per split).
    # Set to null to skip the ready-to-tokenize step.
    "ready_to_tokenize_dir": None,
    # Obs columns to keep in the ready-to-tokenize output.
    "tokenize_obs_cols": [
        "assay", "organism", "nicheformer_split", "batch",
        "niche", "region", "author_cell_type",
    ],
}


# ============================================================================
# Helper functions
# ============================================================================

def map_genes_to_ensembl(
    adata: sc.AnnData,
    species: str = "human",
    symbol_var_col: Optional[str] = None,
) -> sc.AnnData:
    """
    Map gene symbols to Ensembl Gene IDs using mygene.

    Parameters
    ----------
    adata
        AnnData with gene symbols as var_names (or in symbol_var_col).
    species
        Species for mygene lookup ("human" or "mouse").
    symbol_var_col
        If var_names are already Ensembl IDs, specify the var column
        containing original gene symbols.

    Returns
    -------
    AnnData with Ensembl Gene IDs as var_names.
    """
    if symbol_var_col and symbol_var_col in adata.var.columns:
        gene_symbols = list(adata.var[symbol_var_col].values)
    else:
        gene_symbols = list(adata.var_names)

    logger.info(f"Mapping {len(gene_symbols)} gene symbols to Ensembl IDs via mygene ...")
    mg = mygene.MyGeneInfo()
    results = mg.querymany(
        gene_symbols,
        scopes="symbol",
        fields="ensembl.gene",
        species=species,
        returnall=True,
        as_dataframe=False,
    )

    # Build mapping: symbol -> ensembl_id
    symbol_to_ensembl = {}
    for r in results["out"]:
        query = r.get("query", "")
        if "ensembl" in r:
            if isinstance(r["ensembl"], list):
                ens = r["ensembl"][0].get("gene", "")
            else:
                ens = r["ensembl"].get("gene", "")
            if ens:
                symbol_to_ensembl[query] = ens

    mapped = sum(1 for s in gene_symbols if s in symbol_to_ensembl)
    unmapped = len(gene_symbols) - mapped
    logger.info(f"  Mapped: {mapped}/{len(gene_symbols)}  |  Unmapped: {unmapped}")

    # Build new var DataFrame with Ensembl IDs as index
    var_data = adata.var.copy()
    new_index = []
    for s in gene_symbols:
        if s in symbol_to_ensembl:
            new_index.append(symbol_to_ensembl[s])
        else:
            logger.warning(f"  No Ensembl ID for '{s}', keeping as fallback")
            new_index.append(s)

    var_data.index = new_index
    var_data.index.name = None
    var_data["gene_name"] = gene_symbols  # preserve original symbols
    adata.var = var_data

    logger.info(f"var index sample: {list(adata.var_names[:5])}")
    return adata


def prefilter_invalid_genes(
    adata: sc.AnnData,
    organism: str = "human",
) -> sc.AnnData:
    """
    Pre-filter genes that are not in the CELLxGENE whitelist.

    The validate() function filters adata.var but NOT adata.raw.var,
    so we must pre-filter before setting adata.raw.

    Parameters
    ----------
    adata
        AnnData with Ensembl IDs as var_names.
    organism
        "human" or "mouse".

    Returns
    -------
    Filtered AnnData.
    """
    organism_map = {
        "human": SupportedOrganisms.HOMO_SAPIENS,
        "mouse": SupportedOrganisms.MUS_MUSCULUS,
    }
    gene_checker = GeneChecker(organism_map.get(organism, SupportedOrganisms.HOMO_SAPIENS))
    valid_genes = [g for g in adata.var_names if gene_checker.is_valid_id(g)]
    invalid_count = adata.n_vars - len(valid_genes)
    if invalid_count > 0:
        logger.info(f"Pre-filtering {invalid_count} invalid genes before setting raw ...")
        adata = adata[:, valid_genes].copy()
        logger.info(f"Genes after pre-filtering: {adata.n_vars}")
    else:
        logger.info("All genes are valid (no pre-filtering needed).")
    return adata


def setup_nicheformer_split(
    adata: sc.AnnData,
    split_column: Optional[str] = None,
    test_frac: float = 0.1,
    test_samples_frac: float = 0.2,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Set up nicheformer_split column.

    Best practice: split by physical sample/slide to avoid data leakage.
    If split_column is provided and exists in adata.obs, uses slide-level split.
    Otherwise falls back to random cell-level split.

    Parameters
    ----------
    adata
        AnnData object.
    split_column
        Column in adata.obs that groups cells by tissue section.
    test_frac
        Fraction of cells for test set (random split fallback).
    test_samples_frac
        Fraction of unique samples for test set (slide-level split).
    random_seed
        Random seed for reproducibility.

    Returns
    -------
    Array of split labels ("train" or "test").
    """
    np.random.seed(random_seed)

    if split_column and split_column in adata.obs.columns:
        logger.info(f"Splitting by '{split_column}' (slide-level split) ...")
        unique_samples = adata.obs[split_column].unique()
        n_test = max(1, int(len(unique_samples) * test_samples_frac))
        test_samples = np.random.choice(unique_samples, size=n_test, replace=False)
        split = np.array(["train"] * len(adata))
        for sample in test_samples:
            mask = (adata.obs[split_column] == sample).values
            split[mask] = "test"
        logger.info(f"  Test samples ({n_test}/{len(unique_samples)}): {test_samples}")
    else:
        if split_column:
            logger.warning(f"  Split column '{split_column}' not found in obs.")
        logger.info(f"Using random {test_frac*100:.0f}% cell-level split ...")
        test_indices = np.random.choice(
            a=list(range(len(adata))),
            size=int(len(adata) * test_frac),
            replace=False,
        )
        split = np.array(["train"] * len(adata))
        split[test_indices] = "test"

    train_count = (split == "train").sum()
    test_count = (split == "test").sum()
    logger.info(f"  Train: {train_count} cells  |  Test: {test_count} cells")
    return split


def compute_niche_compositions(
    adata: sc.AnnData,
    radii: List[float],
    library_key: str = "library_key",
    spatial_key: str = "spatial",
    cell_type_col: str = "author_cell_type",
) -> sc.AnnData:
    """
    Compute X_niche_{idx} niche composition matrices.

    Parameters
    ----------
    adata
        AnnData with spatial coordinates and cell type annotations.
    radii
        List of radii for spatial neighbors.
    library_key
        obs column for grouping cells by section.
    spatial_key
        obsm key for spatial coordinates.
    cell_type_col
        obs column for cell type annotations.

    Returns
    -------
    AnnData with X_niche_{idx} in obsm.
    """
    logger.info(f"Computing X_niche compositions for radii: {radii} ...")

    # Ensure spatial coordinates are in obsm
    if spatial_key not in adata.obsm:
        if (ObsConstants.SPATIAL_X in adata.obs and
            ObsConstants.SPATIAL_Y in adata.obs):
            adata.obsm[spatial_key] = np.array(
                adata.obs[[ObsConstants.SPATIAL_X, ObsConstants.SPATIAL_Y]]
            ).astype(np.float64)
        else:
            raise KeyError(
                f"Spatial coordinates not found. Need '{spatial_key}' in obsm "
                f"or '{ObsConstants.SPATIAL_X}'/'{ObsConstants.SPATIAL_Y}' in obs."
            )

    adata = niche_compositions(adata, radii)
    logger.info(f"X_niche matrices computed: {[k for k in adata.obsm.keys() if 'X_niche' in k]}")
    return adata


# ============================================================================
# Post-preprocessing steps (gene reordering, technology mean, token ID mapping)
# ============================================================================


def reorder_genes_by_model(
    adata: sc.AnnData,
    model_h5ad_path: str,
) -> sc.AnnData:
    """
    Reorder genes in the AnnData object to match the model's gene order.

    Uses the same approach as the tokenization notebook:
    ``ad.concat([model, adata], join='outer', axis=0)`` then drops the first
    observation (the model row). This ensures genes are in the same order as
    the model expects, filling missing genes with NaN (which become zeros).

    Parameters
    ----------
    adata
        Preprocessed AnnData object.
    model_h5ad_path
        Path to the model.h5ad file.

    Returns
    -------
    AnnData with genes reordered to match the model.
    """
    logger.info(f"Loading model.h5ad from {model_h5ad_path} for gene reordering ...")
    model = sc.read_h5ad(model_h5ad_path)
    logger.info(f"  Model: {model.n_obs} cells × {model.n_vars} genes")

    logger.info("  Concatenating model + data (join='outer', axis=0) ...")
    concat_adata = ad.concat([model, adata], join="outer", axis=0)

    # Drop the first observation (the model row)
    reordered = concat_adata[1:].copy()
    del concat_adata

    logger.info(f"  After reordering: {reordered.n_obs} cells × {reordered.n_vars} genes")
    logger.info(f"  First 5 var_names: {list(reordered.var_names[:5])}")

    return reordered


def load_technology_mean(technology_mean_path: str) -> np.ndarray:
    """
    Load and process the technology-specific mean expression vector.

    Processing steps (matching the tokenization notebook):
    1. Load the .npy file
    2. Replace NaN/Inf with zeros (``np.nan_to_num``)
    3. Round to nearest integer
    4. Replace zeros with 1 (to avoid division by zero during normalization)

    Parameters
    ----------
    technology_mean_path
        Path to the .npy file containing technology-specific mean expression.

    Returns
    -------
    Processed mean expression array.
    """
    logger.info(f"Loading technology-specific mean from {technology_mean_path} ...")
    mean_vec = np.load(technology_mean_path)
    logger.info(f"  Raw mean shape: {mean_vec.shape}")

    # Process: nan_to_num → round → replace zeros with 1
    mean_vec = np.nan_to_num(mean_vec)
    rounded_values = np.where(
        (mean_vec % 1) >= 0.5, np.ceil(mean_vec), np.floor(mean_vec)
    )
    mean_vec = np.where(mean_vec == 0, 1, rounded_values)

    logger.info(f"  Processed mean shape: {mean_vec.shape}")
    return mean_vec


def map_obs_to_token_ids(
    adata: sc.AnnData,
    tokenization_dicts: Dict[str, Any],
) -> sc.AnnData:
    """
    Map obs columns to integer token IDs using the provided dictionaries.

    This replicates the tokenization notebook's approach:
    1. Select only the relevant obs columns
    2. Add 'modality' and 'specie' columns
    3. Replace string labels with integer IDs using the dictionaries

    Parameters
    ----------
    adata
        AnnData with obs columns like 'assay', 'organism', 'nicheformer_split',
        'batch', 'niche', 'region', 'author_cell_type'.
    tokenization_dicts
        Dictionary containing modality_dict, specie_dict, technology_dict,
        author_cell_type_dict, niche_dict, region_dict.

    Returns
    -------
    AnnData with obs columns mapped to integer token IDs.
    """
    logger.info("Mapping obs columns to integer token IDs ...")

    # Select only the relevant obs columns
    keep_cols = [
        "assay", "organism", "nicheformer_split", "batch",
        "niche", "region", "author_cell_type",
    ]
    available_cols = [c for c in keep_cols if c in adata.obs.columns]
    adata.obs = adata.obs[available_cols].copy()

    # Add modality and specie columns
    adata.obs["modality"] = "spatial"
    adata.obs["specie"] = adata.obs["organism"].values

    # Apply tokenization dictionaries
    modality_dict = tokenization_dicts.get("modality_dict", {})
    specie_dict = tokenization_dicts.get("specie_dict", {})
    technology_dict = tokenization_dicts.get("technology_dict", {})
    author_cell_type_dict = tokenization_dicts.get("author_cell_type_dict")
    niche_dict = tokenization_dicts.get("niche_dict")
    region_dict = tokenization_dicts.get("region_dict")

    # Map modality
    if modality_dict:
        adata.obs["modality"] = adata.obs["modality"].replace(modality_dict)

    # Map specie
    if specie_dict:
        adata.obs["specie"] = adata.obs["specie"].replace(specie_dict)

    # Map assay/technology
    if technology_dict:
        adata.obs["assay"] = adata.obs["assay"].replace(technology_dict)

    # Auto-generate author_cell_type_dict from unique values if not provided
    if author_cell_type_dict is None:
        unique_types = sorted(adata.obs["author_cell_type"].unique())
        author_cell_type_dict = {ct: i for i, ct in enumerate(unique_types)}
        logger.info(f"  Auto-generated author_cell_type_dict with {len(author_cell_type_dict)} entries")
    adata.obs["author_cell_type"] = adata.obs["author_cell_type"].replace(author_cell_type_dict)

    # Auto-generate niche_dict from unique values if not provided
    if niche_dict is None:
        unique_niches = sorted(adata.obs["niche"].unique())
        niche_dict = {n: i for i, n in enumerate(unique_niches)}
        logger.info(f"  Auto-generated niche_dict with {len(niche_dict)} entries")
    adata.obs["niche"] = adata.obs["niche"].replace(niche_dict)

    # Auto-generate region_dict from unique values if not provided
    if region_dict is None:
        unique_regions = sorted(adata.obs["region"].unique())
        region_dict = {r: i for i, r in enumerate(unique_regions)}
        logger.info(f"  Auto-generated region_dict with {len(region_dict)} entries")
    adata.obs["region"] = adata.obs["region"].replace(region_dict)

    logger.info(f"  Obs columns after mapping: {list(adata.obs.columns)}")
    return adata


def save_split_ready_to_tokenize(
    adata: sc.AnnData,
    output_dir: str,
    dataset_title: str,
) -> List[str]:
    """
    Split by nicheformer_split and save ready-to-tokenize H5AD files.

    For each split value ('train', 'test'), filters cells, resets index,
    and saves to ``{output_dir}/{dataset_title}_{split}_ready_to_tokenize.h5ad``.

    Parameters
    ----------
    adata
        AnnData with obs columns mapped to integer token IDs.
    output_dir
        Directory to save the split-ready H5AD files.
    dataset_title
        Title of the dataset (used in output filenames).

    Returns
    -------
    List of paths to the saved H5AD files.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    for split_value in ["train", "test"]:
        mask = adata.obs["nicheformer_split"] == split_value
        n_cells = mask.sum()
        if n_cells == 0:
            logger.warning(f"  No cells for split '{split_value}', skipping.")
            continue

        logger.info(f"Saving {split_value} split ({n_cells} cells) ...")
        split_adata = adata[mask].copy()
        split_adata.obs.reset_index(drop=True, inplace=True)

        out_path = os.path.join(
            output_dir, f"{dataset_title}_{split_value}_ready_to_tokenize.h5ad"
        )
        split_adata.write(out_path)
        saved_paths.append(out_path)
        logger.info(f"  Saved: {out_path}")

    return saved_paths


# ============================================================================
# Main preprocessing pipeline
# ============================================================================

def preprocess(config: Optional[Dict[str, Any]] = None) -> str:
    """
    Run the full preprocessing pipeline.

    Parameters
    ----------
    config
        Configuration dictionary. If None, uses CONFIG above.

    Returns
    -------
    Path to the saved preprocessed H5AD file.
    """
    if config is None:
        config = CONFIG

    # ------------------------------------------------------------------
    # 0. Setup paths
    # ------------------------------------------------------------------
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, config["output_name"])

    # ------------------------------------------------------------------
    # 1. Load raw data
    # ------------------------------------------------------------------
    raw_path = config["raw_h5ad"]
    logger.info(f"Loading raw data from {raw_path}")
    adata = sc.read_h5ad(raw_path)
    logger.info(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

    # ------------------------------------------------------------------
    # 2. Map gene symbols → Ensembl gene IDs (if needed)
    # ------------------------------------------------------------------
    if config.get("map_genes", True):
        adata = map_genes_to_ensembl(
            adata,
            species=config.get("gene_mapping_species", "human"),
            symbol_var_col=config.get("gene_symbol_var_col"),
        )

    # ------------------------------------------------------------------
    # 3. Set required obs columns (CELLxGENE schema)
    # ------------------------------------------------------------------
    n = adata.n_obs

    # Spatial coordinates
    x_col = config["spatial_x_col"]
    y_col = config["spatial_y_col"]
    if x_col in adata.obs.columns and y_col in adata.obs.columns:
        adata.obs[ObsConstants.SPATIAL_X] = adata.obs[x_col].values
        adata.obs[ObsConstants.SPATIAL_Y] = adata.obs[y_col].values
    else:
        logger.warning(
            f"Spatial columns '{x_col}'/'{y_col}' not found. "
            "Set spatial_x_col and spatial_y_col in config."
        )
        for alt_x, alt_y in [("x", "y"), ("X", "Y"), ("x_slide_mm", "y_slide_mm"),
                              ("x_FOV_px", "y_FOV_px"), ("x_location", "y_location")]:
            if alt_x in adata.obs.columns and alt_y in adata.obs.columns:
                logger.info(f"  Using '{alt_x}'/'{alt_y}' as spatial coordinates.")
                adata.obs[ObsConstants.SPATIAL_X] = adata.obs[alt_x].values
                adata.obs[ObsConstants.SPATIAL_Y] = adata.obs[alt_y].values
                break
        else:
            raise KeyError(
                f"Cannot find spatial coordinate columns. "
                f"Available obs columns: {list(adata.obs.columns[:20])}"
            )

    # Ontology columns
    adata.obs[ObsConstants.ASSAY_ONTOLOGY_TERM_ID] = pd.Categorical(
        [config["assay"]] * n
    )
    adata.obs[ObsConstants.SEX_ONTOLOGY_TERM_ID] = pd.Categorical(
        [config["sex"]] * n
    )
    adata.obs[ObsConstants.ORGANISM_ONTOLOGY_TERM_ID] = pd.Categorical(
        [config["organism"]] * n
    )
    adata.obs[ObsConstants.TISSUE_ONTOLOGY_TERM_ID] = pd.Categorical(
        [config["tissue"]] * n
    )
    adata.obs[ObsConstants.SUSPENSION_TYPE] = pd.Categorical(
        [config["suspension_type"]] * n
    )

    # Nicheformer-specific required columns
    adata.obs[ObsConstants.DONOR_ID] = pd.Categorical(
        [config["donor_id"]] * n
    )
    adata.obs[ObsConstants.CONDITION_ID] = pd.Categorical(
        [config["condition_id"]] * n
    )
    adata.obs[ObsConstants.TISSUE_TYPE] = pd.Categorical(
        [config["tissue_type"]] * n
    )

    # Library key (must be categorical for squidpy.gr.spatial_neighbors)
    lib_key = config["library_key"]
    if lib_key in adata.obs.columns:
        adata.obs[ObsConstants.LIBRARY_KEY] = pd.Categorical(
            adata.obs[lib_key].values
        )
    else:
        adata.obs[ObsConstants.LIBRARY_KEY] = pd.Categorical(
            [lib_key] * n
        )

    # ------------------------------------------------------------------
    # 4. Set uns metadata
    # ------------------------------------------------------------------
    adata.uns[UnsConstants.TITLE] = config["dataset_title"]

    # ------------------------------------------------------------------
    # 5. Convert X to CSR float32 & filter zero-count cells
    # ------------------------------------------------------------------
    counts_layer = config.get("counts_layer")
    if counts_layer and counts_layer in adata.layers:
        logger.info(f"Using '{counts_layer}' layer as primary matrix")
        adata.X = adata.layers[counts_layer].copy()
    elif "counts" in adata.layers:
        logger.info("Using 'counts' layer as primary matrix")
        adata.X = adata.layers["counts"].copy()
    elif "data" in adata.layers:
        logger.info("Using 'data' layer as primary matrix")
        adata.X = adata.layers["data"].copy()

    adata.X = csr_matrix(adata.X, dtype=np.float32)
    sc.pp.filter_cells(adata, min_genes=1)
    logger.info(f"Cells after filtering zero-count cells: {adata.n_obs}")

    # ------------------------------------------------------------------
    # 6. Pre-filter invalid Ensembl IDs & set adata.raw
    # ------------------------------------------------------------------
    adata = prefilter_invalid_genes(
        adata,
        organism=config.get("organism_validator", "human"),
    )

    # Set raw so the CELLxGENE validator finds raw counts in adata.raw.X
    adata.raw = adata

    # IMPORTANT: feature_is_filtered must be set AFTER adata.raw = adata,
    # otherwise it propagates to raw.var and the validator rejects it.
    adata.var[VarConstants.FEATURE_IS_FILTERED] = False

    # ------------------------------------------------------------------
    # 7. Run CELLxGENE + Nicheformer validation
    # ------------------------------------------------------------------
    logger.info("Running CELLxGENE validation ...")
    adata_output, valid, errors, is_seurat_convertible = validate(
        adata, organism=config.get("organism_validator", "human")
    )

    if not valid:
        logger.error("Validation FAILED!")
        for e in errors:
            logger.error(f"  {e}")
        raise SystemExit(1)

    logger.info("Validation passed!")

    # ------------------------------------------------------------------
    # 8. Add nicheformer-specific fields
    # ------------------------------------------------------------------
    # Assay name (used in tokenization for technology_dict mapping)
    adata_output.obs["assay"] = pd.Categorical(
        [config.get("assay_name", "CosMx")] * len(adata_output)
    )
    # Overwrite assay_ontology_term_id with the correct one
    adata_output.obs[ObsConstants.ASSAY_ONTOLOGY_TERM_ID] = pd.Categorical(
        [config["assay"]] * len(adata_output)
    )
    adata_output.obs[ObsConstants.DATASET] = config["dataset_title"]

    # nicheformer_split
    split = setup_nicheformer_split(
        adata_output,
        split_column=config.get("split_column"),
        test_frac=config.get("split_test_frac", 0.1),
        test_samples_frac=config.get("split_test_samples_frac", 0.2),
        random_seed=config.get("split_random_seed", 42),
    )
    adata_output.obs[ObsConstants.SPLIT] = pd.Categorical(split)

    # niche, author_cell_type, region
    niche_col = config.get("niche_col")
    if niche_col and niche_col in adata.obs.columns:
        adata_output.obs[ObsConstants.NICHE] = adata.obs.loc[
            adata_output.obs.index, niche_col
        ].values
    else:
        adata_output.obs[ObsConstants.NICHE] = "nan"

    ct_col = config.get("cell_type_col")
    if ct_col and ct_col in adata.obs.columns:
        adata_output.obs[ObsConstants.AUTHOR_CELL_TYPE] = adata.obs.loc[
            adata_output.obs.index, ct_col
        ].values
    else:
        adata_output.obs[ObsConstants.AUTHOR_CELL_TYPE] = "nan"

    adata_output.obs[ObsConstants.REGION] = "nan"

    # Preserve additional obs columns
    for col in config.get("preserve_obs_cols", []):
        if col in adata.obs.columns:
            adata_output.obs[col] = adata.obs.loc[adata_output.obs.index, col].values
            logger.info(f"  Preserved obs column: '{col}'")

    # ------------------------------------------------------------------
    # 9. Add batch column
    # ------------------------------------------------------------------
    batch_col = config.get("batch_col")
    if batch_col and batch_col in adata.obs.columns:
        batch_values = adata.obs.loc[adata_output.obs.index, batch_col].values
    else:
        batch_values = np.zeros(len(adata_output), dtype=int)
    adata_output.obs["batch"] = pd.Categorical(batch_values.astype(str))

    # ------------------------------------------------------------------
    # 10. Add spatial obsm & compute X_niche compositions (optional)
    # ------------------------------------------------------------------
    adata_output.obsm["spatial"] = np.array(
        adata_output.obs[[ObsConstants.SPATIAL_X, ObsConstants.SPATIAL_Y]]
    ).astype(np.float64)

    radii = config.get("niche_radii")
    if radii is not None and len(radii) > 0:
        adata_output = compute_niche_compositions(
            adata_output,
            radii=radii,
            library_key=ObsConstants.LIBRARY_KEY,
            spatial_key="spatial",
            cell_type_col=ObsConstants.AUTHOR_CELL_TYPE,
        )

    # ------------------------------------------------------------------
    # 11. Save preprocessed file
    # ------------------------------------------------------------------
    logger.info(f"Saving preprocessed data to {output_path}")
    adata_output.write(output_path)
    logger.info("Preprocessed file saved.")

    # ------------------------------------------------------------------
    # 12. Reorder genes to match model gene order (optional)
    # ------------------------------------------------------------------
    model_h5ad = config.get("model_h5ad")
    if model_h5ad and os.path.exists(model_h5ad):
        adata_output = reorder_genes_by_model(adata_output, model_h5ad)
    else:
        logger.info("  Skipping gene reordering (model_h5ad not found or not configured).")

    # ------------------------------------------------------------------
    # 13. Load & process technology-specific mean (optional)
    # ------------------------------------------------------------------
    technology_mean_path = config.get("technology_mean_path")
    if technology_mean_path and os.path.exists(technology_mean_path):
        technology_mean = load_technology_mean(technology_mean_path)
        # Store in uns so the tokenization notebook / dataset class can access it
        adata_output.uns["technology_mean"] = technology_mean
        logger.info("  Technology mean stored in adata.uns['technology_mean']")
    else:
        logger.info("  Skipping technology mean loading (path not found or not configured).")

    # ------------------------------------------------------------------
    # 14. Map obs columns to integer token IDs (optional)
    # ------------------------------------------------------------------
    tokenization_dicts = config.get("tokenization_dicts")
    if tokenization_dicts:
        adata_output = map_obs_to_token_ids(
            adata_output,
            tokenization_dicts=tokenization_dicts,
        )
    else:
        logger.info("  Skipping token ID mapping (tokenization_dicts not configured).")

    # ------------------------------------------------------------------
    # 15. Split by nicheformer_split & save ready-to-tokenize files (optional)
    # ------------------------------------------------------------------
    ready_to_tokenize_dir = config.get("ready_to_tokenize_dir")
    if ready_to_tokenize_dir:
        saved = save_split_ready_to_tokenize(
            adata_output,
            output_dir=ready_to_tokenize_dir,
            dataset_title=config.get("dataset_title", "dataset"),
        )
        logger.info(f"Ready-to-tokenize files saved: {saved}")
    else:
        logger.info("  Skipping ready-to-tokenize split (ready_to_tokenize_dir not configured).")

    logger.info("Done!")
    return output_path


# ============================================================================
# CLI entry point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess spatial omics data for Nicheformer tokenization."
    )
    parser.add_argument(
        "--config-yaml",
        type=str,
        default=None,
        help="Path to a YAML config file with configuration keys.",
    )
    args = parser.parse_args()

    # Load config 
    import yaml
    with open(args.config_yaml, "r") as f:
        yaml_config = yaml.safe_load(f)
    config = dict(CONFIG)  # start with defaults
    config.update(yaml_config)  # overlay YAML values
    logger.info(f"Loaded config from {args.config_yaml}")

    preprocess(config)
