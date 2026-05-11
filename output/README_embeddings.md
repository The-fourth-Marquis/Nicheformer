# How to Use the Nicheformer Embeddings

This guide explains how to use the Nicheformer embeddings generated in [`nich_emb.ipynb`](nich_emb.ipynb) within the broader [`25-12/`](../../) project.

---

## 1. What the Notebook Does

[`nich_emb.ipynb`](nich_emb.ipynb) loads a pre-computed AnnData file containing Nicheformer embeddings:

```python
import scanpy as sc
emb = sc.read_h5ad("/mnt/172/wh/25-12/spatial/adata_with_embeddings.h5ad")
emb
```

The resulting AnnData object has **368,226 cells × 20,310 genes** with the following structure:

| Component | Description |
|-----------|-------------|
| `emb.obs` | Metadata: `assay`, `organism`, `nicheformer_split`, `batch`, `niche`, `region`, `author_cell_type`, `modality`, `specie` |
| `emb.obsm['X_nicheformer_embeddings']` | **The embeddings** — shape `(368226, 512)` — a 512-dimensional vector per cell |
| `emb.obsm['X_niche_0'..'X_niche_4']` | Niche composition vectors (from spatial neighborhood) |
| `emb.obsm['X_pca']`, `emb.obsm['X_umap']` | Standard dimensionality reductions |
| `emb.layers['counts']`, `emb.layers['data']` | Raw and normalized expression |

---

## 2. Accessing the Embeddings

```python
import scanpy as sc
import numpy as np

# Load the data
adata = sc.read_h5ad("/mnt/172/wh/25-12/spatial/adata_with_embeddings.h5ad")

# The embeddings are in obsm
embeddings = adata.obsm['X_nicheformer_embeddings']
print(f"Embeddings shape: {embeddings.shape}")  # (368226, 512)

# Each row is a 512-dim vector for one cell
cell_0_embedding = embeddings[0]
```

---

## 3. Downstream Use Cases

### 3.1 Dimensionality Reduction & Visualization

```python
import scanpy as sc

# UMAP on the Nicheformer embeddings
adata.obsm['X_emb_umap'] = sc.tl.umap(adata.obsm['X_nicheformer_embeddings'], copy=True).obsm['X_umap']

# Color by cell type
sc.pl.umap(adata, color='author_cell_type', 
           obsm='X_emb_umap', title='Nicheformer Embeddings (UMAP)')
```

### 3.2 Clustering

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=15, random_state=42, n_init=10)
adata.obs['nicheformer_cluster'] = kmeans.fit_predict(embeddings)

# Cross-tabulate with known cell types
pd.crosstab(adata.obs['nicheformer_cluster'], adata.obs['author_cell_type'])
```

### 3.3 Cell-Type Classification

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = embeddings
y = adata.obs['author_cell_type'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
```

### 3.4 Niche Composition Regression

Use the embeddings to predict spatial niche composition (the `X_niche_*` vectors):

```python
from sklearn.linear_model import Ridge

# Predict niche composition from embeddings
X = embeddings
y = adata.obsm['X_niche_0']  # or any niche dimension

model = Ridge(alpha=1.0)
model.fit(X, y)
predictions = model.predict(X)
```

### 3.5 Batch Correction / Integration

```python
import scanpy as sc

# Use embeddings as input for Harmony integration
sc.external.pp.harmony_integrate(
    adata, key='batch', 
    basis='X_nicheformer_embeddings', 
    adjusted_basis='X_emb_harmony'
)
```

### 3.6 Gene Expression Imputation

```python
from sklearn.linear_model import Ridge

# Predict expression of a specific gene from embeddings
gene_idx = 0  # first gene
gene_name = adata.var_names[gene_idx]
y_gene = adata[:, gene_idx].X.toarray().flatten() if hasattr(adata.X, 'toarray') else adata[:, gene_idx].X.flatten()

model = Ridge(alpha=1.0)
model.fit(embeddings, y_gene)
```

---

## 4. How These Embeddings Were Generated

The embeddings were produced using the **Nicheformer** model pipeline documented in [`nicheformer_pipeline_example.ipynb`](nicheformer_pipeline_example.ipynb). The pipeline follows these steps:

### Step 1: Tokenization
Raw expression data is tokenized into gene ID sequences:
- `sf_normalize` → scale to 10,000 counts per cell
- Divide by technology mean (per-gene normalization)
- `_sub_tokenize_data` → sort genes by expression, keep top-4096, add offset of 30 (auxiliary tokens)

### Step 2: Model Inference
The tokenized sequences are passed through the Nicheformer model (12-layer transformer, 512-dim embeddings, 16 attention heads):

```python
from nicheformer.models import Nicheformer

model = Nicheformer.load_from_checkpoint(
    checkpoint_path='/root/code/25-12/nicheformer/ckpt/nicheformer.ckpt',
    strict=False
)

emb = model.get_embeddings(batch, layer=-1, with_context=False)
```

Key details from [`model.get_embeddings()`](25-12/nicheformer/src/nicheformer/models/_nicheformer.py:272):
1. Context tokens (modality, assay, specie) are prepended to the sequence
2. The sequence passes through all 12 transformer layers (layer=-1 means last layer)
3. Context tokens (first 3 positions) are removed (`with_context=False`)
4. Remaining token embeddings are **mean-pooled** → 512-dim vector per cell

### Step 3: Storage
Embeddings are saved to `adata.obsm['X_nicheformer_embeddings']` and written to H5AD.

---

## 5. Generating Embeddings for New Data

To generate embeddings for your own data, use the pipeline from [`nicheformer_pipeline_example.ipynb`](nicheformer_pipeline_example.ipynb) (Part 2):

```python
import scanpy as sc
import numpy as np
import torch
from torch.utils.data import DataLoader
from nicheformer.models import Nicheformer
from nicheformer.data import NicheformerDataset

# 1. Load your preprocessed data
adata = sc.read_h5ad("your_data_ready_to_tokenize.h5ad")
technology_mean = np.load("path/to/technology_mean.npy")

# 2. Ensure required metadata columns
if 'modality' not in adata.obs.columns:
    adata.obs['modality'] = 4  # spatial
if 'specie' not in adata.obs.columns:
    adata.obs['specie'] = 5    # human
if 'assay' not in adata.obs.columns:
    adata.obs['assay'] = 8     # cosmx

# 3. Create dataset
dataset = NicheformerDataset(
    adata=adata,
    technology_mean=technology_mean,
    split='train',
    max_seq_len=1500,
    aux_tokens=30,
    chunk_size=1000,
    metadata_fields={'obs': ['modality', 'specie', 'assay']}
)

dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

# 4. Load model and extract embeddings
model = Nicheformer.load_from_checkpoint(
    checkpoint_path='/root/code/25-12/nicheformer/ckpt/nicheformer.ckpt',
    strict=False
)
model.eval()

embeddings = []
device = next(model.parameters()).device

with torch.no_grad():
    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        emb = model.get_embeddings(batch, layer=-1, with_context=False)
        embeddings.append(emb.cpu().numpy())

embeddings = np.concatenate(embeddings, axis=0)
adata.obsm['X_nicheformer_embeddings'] = embeddings
adata.write_h5ad("adata_with_embeddings.h5ad")
```

---

## 6. Available Technology Means

The project provides pre-computed technology means for normalization:

| File | Path |
|------|------|
| CosMx | [`data/model_means/cosmx_mean_script.npy`](../../data/model_means/cosmx_mean_script.npy) |
| MERFISH | [`data/model_means/merfish_mean_script.npy`](../../data/model_means/merfish_mean_script.npy) |
| Xenium | [`data/model_means/xenium_mean_script.npy`](../../data/model_means/xenium_mean_script.npy) |
| ISS | [`data/model_means/iss_mean_script.npy`](../../data/model_means/iss_mean_script.npy) |
| Dissociated (scRNA-seq) | [`data/model_means/dissociated_mean_script.npy`](../../data/model_means/dissociated_mean_script.npy) |

---

## 7. Context Token Reference

When generating embeddings, context tokens are prepended to identify the data modality:

| Token | ID | Meaning |
|-------|----|---------|
| Modality: spatial | 4 | Spatial transcriptomics |
| Modality: dissociated | 3 | scRNA-seq |
| Specie: human | 5 | Homo sapiens |
| Specie: mouse | 6 | Mus musculus |
| Assay: cosmx | 8 | NanoString CosMx |
| Assay: merfish | 7 | MERFISH |
| Assay: xenium | 9 | 10x Xenium |

These are stripped from the final embedding (via `with_context=False` in [`get_embeddings()`](25-12/nicheformer/src/nicheformer/models/_nicheformer.py:301)).
