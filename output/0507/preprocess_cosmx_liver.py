"""
Preprocess CosMx human liver (CancerousLiver) data for Nicheformer tokenization.

This script follows the same pattern as preprocess_xenium_lung.py but adapted for
CosMx data. The key differences are:
1. CosMx data uses gene symbols as var index (not Ensembl IDs) — we map them via mygene
2. CosMx data has spatial coordinates in x_slide_mm / y_slide_mm
3. CosMx data already has niche labels (tumor, non-malignant, interface, tumor subtype)
4. CosMx data has cellType annotations in obs

Usage:
    python preprocess_cosmx_liver.py [data_path]

If no data_path is given, defaults to DefaultPaths.SPATIAL (/mnt/172/wh/25-12/spatial).
"""

import os
import sys
import logging

import scanpy as sc
import pandas as pd
import numpy as np
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
if len(sys.argv) == 1:
    path = DefaultPaths.SPATIAL
else:
    path = sys.argv[1]

raw_path = f"{path}/raw"
preprocessed_path = f"{path}/preprocessed"

os.makedirs(preprocessed_path, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration — edit these for your dataset
# ---------------------------------------------------------------------------
# RAW_H5AD = os.path.join(
#     raw_path,
#     "cosmx_liver_full_h5ad_export",
#     "cosmx_CancerousLiver_fullvis.h5ad",
# )

# OUTPUT_NAME = "nanostring_cosmx_human_liver_cancer.h5ad"

# # Ontology constants
# ASSAY = str(AssayOntologyTermId.COSMX_SPATIAL.value)       # EFO:0030029
# SEX = str(SexOntologyTermId.FEMALE.value)                  # unknown
# ORGANISM = str(OrganismOntologyTermId.HUMAN.value)          # NCBITaxon:9606
# ORGANISM_VALIDATOR = "human"
# TISSUE = str(TissueOntologyTermId.LIVER.value)              # UBERON:0002107
# SUSPENSION_TYPE = str(SuspensionTypeId.SPATIAL.value)       # na
# TISSUE_TYPE = "tissue"

# # Metadata for this dataset
# DATASET_TITLE = "nanostring_cosmx_human_liver_cancer"
# DONOR_ID = "CancerousLiver"
# CONDITION_ID = "cancerous"
# LIBRARY_KEY = "section"

RAW_H5AD = os.path.join(
    raw_path,
    "cosmx_liver_full_h5ad_export",
    "cosmx_NormalLiver_fullvis.h5ad",
)

OUTPUT_NAME = "nanostring_cosmx_human_liver_normal.h5ad"

# Ontology constants
ASSAY = str(AssayOntologyTermId.COSMX_SPATIAL.value)       # EFO:0030029
SEX = str(SexOntologyTermId.MALE.value)                  # unknown
ORGANISM = str(OrganismOntologyTermId.HUMAN.value)          # NCBITaxon:9606
ORGANISM_VALIDATOR = "human"
TISSUE = str(TissueOntologyTermId.LIVER.value)              # UBERON:0002107
SUSPENSION_TYPE = str(SuspensionTypeId.SPATIAL.value)       # na
TISSUE_TYPE = "tissue"

# Metadata for this dataset
DATASET_TITLE = "nanostring_cosmx_human_liver_normal"
DONOR_ID = "NormalLiver"
CONDITION_ID = "normal"
LIBRARY_KEY = "section"

# ---------------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------------
logger.info(f"Loading raw data from {RAW_H5AD}")
adata = sc.read_h5ad(RAW_H5AD)
logger.info(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

# ---------------------------------------------------------------------------
# 2. Map gene symbols → Ensembl gene IDs
# ---------------------------------------------------------------------------
# The CosMx H5AD has gene symbols as var_names and no 'gene_ids' column.
# The CELLxGENE validator requires Ensembl Gene IDs as var_names.
logger.info("Mapping gene symbols to Ensembl IDs via mygene ...")
mg = mygene.MyGeneInfo()
gene_symbols = list(adata.var_names)
results = mg.querymany(
    gene_symbols,
    scopes="symbol",
    fields="ensembl.gene",
    species="human",
    returnall=True,
    as_dataframe=False,
)

# Build a mapping: symbol -> ensembl_id
symbol_to_ensembl = {}
for r in results["out"]:
    query = r.get("query", "")
    if "ensembl" in r:
        if isinstance(r["ensembl"], list):
            # Sometimes multiple ensembl entries; take the first 'gene' one
            ens = r["ensembl"][0].get("gene", "")
        else:
            ens = r["ensembl"].get("gene", "")
        if ens:
            symbol_to_ensembl[query] = ens

# Report mapping statistics
mapped = sum(1 for s in gene_symbols if s in symbol_to_ensembl)
unmapped = len(gene_symbols) - mapped
logger.info(f"  Mapped: {mapped}/{len(gene_symbols)} genes  |  Unmapped: {unmapped}")

# Build new var DataFrame with Ensembl IDs as index
var_data = adata.var.copy()
new_index = []
for s in gene_symbols:
    if s in symbol_to_ensembl:
        new_index.append(symbol_to_ensembl[s])
    else:
        # Fallback: keep the gene symbol (will be filtered by validator later)
        logger.warning(f"  No Ensembl ID for {s}, keeping symbol as fallback")
        new_index.append(s)

var_data.index = new_index
var_data.index.name = None
var_data["gene_name"] = gene_symbols  # preserve original symbols
adata.var = var_data

logger.info(f"var index now contains Ensembl IDs (first 5): {list(adata.var_names[:5])}")

# ---------------------------------------------------------------------------
# 3. Set required obs columns (CELLxGENE schema)
# ---------------------------------------------------------------------------
n = adata.n_obs

# Spatial coordinates
adata.obs[ObsConstants.SPATIAL_X] = adata.obs["x_slide_mm"].values
adata.obs[ObsConstants.SPATIAL_Y] = adata.obs["y_slide_mm"].values

# Ontology columns
adata.obs[ObsConstants.ASSAY_ONTOLOGY_TERM_ID] = pd.Categorical([ASSAY] * n)
adata.obs[ObsConstants.SEX_ONTOLOGY_TERM_ID] = pd.Categorical([SEX] * n)
adata.obs[ObsConstants.ORGANISM_ONTOLOGY_TERM_ID] = pd.Categorical([ORGANISM] * n)
adata.obs[ObsConstants.TISSUE_ONTOLOGY_TERM_ID] = pd.Categorical([TISSUE] * n)
adata.obs[ObsConstants.SUSPENSION_TYPE] = pd.Categorical([SUSPENSION_TYPE] * n)

# Nicheformer-specific required columns
adata.obs[ObsConstants.DONOR_ID] = pd.Categorical([DONOR_ID] * n)
adata.obs[ObsConstants.CONDITION_ID] = pd.Categorical([CONDITION_ID] * n)
adata.obs[ObsConstants.TISSUE_TYPE] = pd.Categorical([TISSUE_TYPE] * n)

# Library key — used by squidpy for spatial neighbors
adata.obs[ObsConstants.LIBRARY_KEY] = pd.Categorical([LIBRARY_KEY] * n)

# ---------------------------------------------------------------------------
# 4. Set uns metadata
# ---------------------------------------------------------------------------
adata.uns[UnsConstants.TITLE] = DATASET_TITLE

# ---------------------------------------------------------------------------
# 5. Convert X to CSR float32 & filter zero-count cells
# ---------------------------------------------------------------------------
# Use the 'counts' layer as raw counts if available
if "counts" in adata.layers:
    logger.info("Using 'counts' layer as primary matrix")
    adata.X = adata.layers["counts"].copy()
elif "data" in adata.layers:
    logger.info("Using 'data' layer as primary matrix")
    adata.X = adata.layers["data"].copy()

adata.X = csr_matrix(adata.X, dtype=np.float32)

# Filter cells with zero counts across all genes
sc.pp.filter_cells(adata, min_genes=1)
logger.info(f"Cells after filtering zero-count cells: {adata.n_obs}")

# ---------------------------------------------------------------------------
# 5b. Pre-filter invalid Ensembl IDs using the CELLxGENE GeneChecker
# ---------------------------------------------------------------------------
# mygene may return Ensembl IDs that are not in the CELLxGENE whitelist
# (e.g., pseudogenes, non-coding RNAs). The validate() function filters
# adata.var but NOT adata.raw.var, so we must pre-filter here.
gene_checker = GeneChecker(SupportedOrganisms.HOMO_SAPIENS)
valid_genes = [g for g in adata.var_names if gene_checker.is_valid_id(g)]
invalid_count = adata.n_vars - len(valid_genes)
if invalid_count > 0:
    logger.info(f"Pre-filtering {invalid_count} invalid genes before setting raw ...")
    adata = adata[:, valid_genes].copy()
    logger.info(f"Genes after pre-filtering: {adata.n_vars}")

# Set raw so the CELLxGENE validator finds raw counts in adata.raw.X
adata.raw = adata

# IMPORTANT: feature_is_filtered must be set AFTER adata.raw = adata,
# otherwise it propagates to raw.var and the validator rejects it.
adata.var[VarConstants.FEATURE_IS_FILTERED] = False

# ---------------------------------------------------------------------------
# 6. Run CELLxGENE + Nicheformer validation
# ---------------------------------------------------------------------------
logger.info("Running CELLxGENE validation ...")
adata_output, valid, errors, is_seurat_convertible = validate(
    adata, organism=ORGANISM_VALIDATOR
)

if not valid:
    logger.error("Validation FAILED!")
    for e in errors:
        logger.error(f"  {e}")
    sys.exit(1)

logger.info("Validation passed!")

# ---------------------------------------------------------------------------
# 7. Add nicheformer-specific fields (used during tokenization)
# ---------------------------------------------------------------------------
adata_output.obs["assay"] = pd.Categorical(["CosMx"] * len(adata_output))
# Overwrite assay_ontology_term_id with the CosMx-specific one
adata_output.obs[ObsConstants.ASSAY_ONTOLOGY_TERM_ID] = pd.Categorical(
    [ASSAY] * len(adata_output)
)

adata_output.obs[ObsConstants.DATASET] = DATASET_TITLE

# ---------------------------------------------------------------------------
# 7b. Set nicheformer_split — best practice: split by slide/sample
# ---------------------------------------------------------------------------
# The `nicheformer_split` column is used by:
#   - NicheformerDataset (dataset.py:81): filters cells by split
#   - Tokenization notebook: tokenizes train/test separately
#   - Downstream fine-tuning (downstream_fine_tune.py:158): test_mask
#
# Best practice: split by physical sample/slide to avoid data leakage.
# This mirrors the original CosMx liver and MERFISH data patterns.
#
# If you have a column like Run_Tissue_name or slide_ID_numeric that
# groups cells by tissue section, use that. Otherwise, fall back to
# a random cell-level split (as in xenium_human_colon.ipynb).
SPLIT_COLUMN = "Run_Tissue_name"  # or "slide_ID_numeric", "sample_name_test"
if SPLIT_COLUMN in adata.obs.columns:
    logger.info(f"Splitting by '{SPLIT_COLUMN}' (slide-level split) ...")
    unique_samples = adata.obs.loc[adata_output.obs.index, SPLIT_COLUMN].unique()
    np.random.seed(42)
    n_test = max(1, int(len(unique_samples) * 0.2))
    test_samples = np.random.choice(unique_samples, size=n_test, replace=False)
    split = np.array(["train"] * len(adata_output))
    for sample in test_samples:
        mask = (adata_output.obs[SPLIT_COLUMN] == sample).values
        split[mask] = "test"
    logger.info(f"  Test samples ({n_test}/{len(unique_samples)}): {test_samples}")
else:
    logger.info("No slide-level column found; using random 10% cell-level split ...")
    np.random.seed(42)
    test_indices = np.random.choice(
        a=list(range(len(adata_output))),
        size=int(len(adata_output) * 0.1),
        replace=False,
    )
    split = np.array(["train"] * len(adata_output))
    split[test_indices] = "test"

adata_output.obs[ObsConstants.SPLIT] = pd.Categorical(split)
train_count = (split == "train").sum()
test_count = (split == "test").sum()
logger.info(f"  Train: {train_count} cells  |  Test: {test_count} cells")

# Preserve existing niche labels from the raw data
if "niche" in adata.obs.columns:
    adata_output.obs[ObsConstants.NICHE] = adata.obs.loc[
        adata_output.obs.index, "niche"
    ].values
else:
    adata_output.obs[ObsConstants.NICHE] = "nan"

# Preserve existing cell type annotations
if "cellType" in adata.obs.columns:
    adata_output.obs[ObsConstants.AUTHOR_CELL_TYPE] = adata.obs.loc[
        adata_output.obs.index, "cellType"
    ].values
else:
    adata_output.obs[ObsConstants.AUTHOR_CELL_TYPE] = "nan"

adata_output.obs[ObsConstants.REGION] = "nan"

# ---------------------------------------------------------------------------
# 8. Add batch column (used by tokenization notebook)
# ---------------------------------------------------------------------------
# The tokenization notebook selects 'batch' from obs. We set it to the
# slide_ID_numeric or Run_name so cells from the same slide share a batch.
if "slide_ID_numeric" in adata.obs.columns:
    batch_values = adata.obs.loc[adata_output.obs.index, "slide_ID_numeric"].values
elif "Run_name" in adata.obs.columns:
    batch_values = adata.obs.loc[adata_output.obs.index, "Run_name"].values
else:
    batch_values = np.zeros(len(adata_output), dtype=int)
adata_output.obs["batch"] = pd.Categorical(batch_values.astype(str))

# ---------------------------------------------------------------------------
# 9. Add spatial obsm (used by tokenization notebook for X_niche computation)
# ---------------------------------------------------------------------------
adata_output.obsm["spatial"] = np.array(
    adata_output.obs[[ObsConstants.SPATIAL_X, ObsConstants.SPATIAL_Y]]
).astype(np.float64)

# ---------------------------------------------------------------------------
# 10. Save preprocessed file
# ---------------------------------------------------------------------------
output_path = os.path.join(preprocessed_path, OUTPUT_NAME)
logger.info(f"Saving preprocessed data to {output_path}")
adata_output.write(output_path)
logger.info("Done!")
