# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Use `uv` for all Python operations:
```bash
uv run python script.py          # Run a script
uv run python -c "..."           # Run inline code
uv lock                          # Update lock after pyproject.toml changes
uv sync                          # Sync venv to lockfile
```

The venv is at `.venv/`. Do NOT use `uv pip install` — it bypasses the lockfile and changes get reverted on next `uv sync`/`uv run`.

## Project overview

Nicheformer is a masked-language-model (MLM) transformer pretrained on single-cell and spatial omics data. Genes are tokenized as integer IDs, and a 12-layer transformer encoder learns contextual gene representations. The pretrained model produces 512-dim cell embeddings used for downstream tasks.

## Build / Lint / Test

```bash
uv run ruff check src/            # Lint
uv run ruff format --check src/   # Format check (line-width 120)
uv run pytest                     # Run tests (importlib mode)
```

## Architecture

```
src/nicheformer/
  models/
    _nicheformer.py          # Core model (pl.LightningModule): 12-layer, 16-head, dim=512
    _nicheformer_fine_tune.py  # Fine-tune wrapper
    _fine_tune_model.py       # Older fine-tune variant
  data/
    dataset.py               # NicheformerDataset + tokenization (sf_normalize → ÷tech_mean → _sub_tokenize_data)
    datamodules.py            # MerlinDataModuleDistributed (Parquet-based distributed loading)
    constants.py              # Ontology enums, DefaultPaths, ObsConstants
    validate.py               # CELLxGENE schema validation
    tools/
      niche_compositions.py  # X_niche computation via squidpy spatial neighbors
```

## Key pipeline: raw data → embeddings

1. **Preprocessing** (`output/0512_spatial/preprocess_spatial.py`): gene symbols→Ensembl (mygene), CELLxGENE validation, gene reordering via `ad.concat([model.h5ad, adata], join='outer')`, tech-mean loading, token-ID encoding
2. **Tokenization** (`dataset.py:93-143`): `nan_to_num → sf_normalize(10K) → ÷technology_mean → _sub_tokenize_data` (sort genes by expression, keep top 4096, add +30 aux offset)
3. **Embedding extraction** (`_nicheformer.py:272-306`): `model.get_embeddings(batch, layer=-1, with_context=False)` — prepends context tokens → transformer → strips context → mean-pools → 512-dim cell embedding

## Token ID schema

| Range | Purpose |
|-------|---------|
| 0 | Padding |
| 2 | CLS |
| 3-29 | Auxiliary reserved |
| 30-20369 | Gene tokens (20340 genes, offset +30 from gene index) |
| 4 | Modality: spatial (3=dissociated) |
| 5 | Specie: human (6=mouse) |
| 7-18 | Assay: merfish=7, cosmx=8, visium=9, xenium=9, 10x variants=10-18 |

Token ID for gene at column index `i` in model-aligned AnnData = `i + 30`.

## Gene harmonization for mouse data

The model has 20,159 human ENSG genes serving as tokens for both species via orthology, and only 151 mouse-specific ENSMUSG (genes without human orthologs). For mouse data:
1. Map mouse gene symbols → ENSMUSG (mygene, `scopes='ensembl.gene'`, `species='mouse'`)
2. Get orthologs: ENSMUSG → human NCBI Gene ID (mygene `homologene` field, tax_id=9606)
3. Human NCBI Gene ID → ENSG (mygene, `scopes='entrezgene'`, `species='human'`)
4. Replace var_names with human ENSG, deduplicate by summing counts for genes mapping to the same ortholog
5. Then `ad.concat([model.h5ad, adata], join='outer')` — now genes align correctly

## Critical constraints in pyproject.toml

- `numpy<2` — numba/numpyro compatibility
- `zarr<3` — anndata compatibility
- `numba>=0.57.0,<0.61` — must be <0.61 with numpy 1.x
- `torch>=2.5.1` — torch 2.6 defaults to `weights_only=True` for `torch.load`; monkey-patch `lightning_fabric.utilities.cloud_io._load` or set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` when loading checkpoints
- `anndata<0.10.0` — API compatibility

## Common issues

1. **Gene count mismatch in tokenization**: `NicheformerDataset` requires gene count in AnnData to match `technology_mean` shape (20310). The concat-with-model step must expand to exactly 20310 genes. If AnnData has more, subset to `:20310`.

2. **`nicheformer_split` filtering**: `NicheformerDataset` line 81 filters by `adata.obs.nicheformer_split == split`. Set all cells to `'train'` if you want all cells processed.

3. **NaN in `ad.concat` output**: Outer join fills missing genes with NaN. Call `np.nan_to_num()` before compute to avoid NaN propagation.

4. **Non-unique var_names after ortholog mapping**: Multiple mouse genes can map to the same human ortholog. Group and sum counts, or the concat will fail with `InvalidIndexError`.

5. **Pre-computed technology means**: Five files in `data/model_means/` (cosmx, merfish, xenium, visium, dissociated). Shape is always (20310,). Processing: `nan_to_num → round → zeros→1`.

6. **Hardcoded paths in constants.py**: `DefaultPaths.SPATIAL` and other paths point to `/mnt/172/wh/25-12/` — override via config rather than modifying constants.
