library(Seurat)
library(Matrix)

DefaultAssay(obj) <- "RNA"

get_layer_safe <- function(obj, assay, layer_name) {
  tryCatch(
    GetAssayData(obj, assay = assay, layer = layer_name),
    error = function(e) {
      GetAssayData(obj, assay = assay, slot = layer_name)
    }
  )
}

export_group_to_folder <- function(obj, cells_use, out_dir) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  
  # metadata
  meta <- obj@meta.data[cells_use, , drop = FALSE]
  write.csv(meta, file.path(out_dir, "metadata.csv"))
  write.csv(data.frame(cell = cells_use), file.path(out_dir, "cells.csv"), row.names = FALSE)
  
  # RNA counts and data
  rna_counts <- get_layer_safe(obj, "RNA", "counts")[, cells_use, drop = FALSE]
  rna_data <- get_layer_safe(obj, "RNA", "data")[, cells_use, drop = FALSE]
  
  writeMM(rna_counts, file.path(out_dir, "rna_counts_gene_by_cell.mtx"))
  writeMM(rna_data, file.path(out_dir, "rna_data_gene_by_cell.mtx"))
  write.csv(data.frame(gene = rownames(rna_counts)), file.path(out_dir, "rna_genes.csv"), row.names = FALSE)
  
  # falsecode assay
  if ("falsecode" %in% Assays(obj)) {
    false_counts <- get_layer_safe(obj, "falsecode", "counts")[, cells_use, drop = FALSE]
    writeMM(false_counts, file.path(out_dir, "falsecode_counts_feature_by_cell.mtx"))
    write.csv(
      data.frame(feature = rownames(false_counts)),
      file.path(out_dir, "falsecode_features.csv"),
      row.names = FALSE
    )
  }
  
  # negprobes assay
  if ("negprobes" %in% Assays(obj)) {
    neg_counts <- get_layer_safe(obj, "negprobes", "counts")[, cells_use, drop = FALSE]
    writeMM(neg_counts, file.path(out_dir, "negprobes_counts_feature_by_cell.mtx"))
    write.csv(
      data.frame(feature = rownames(neg_counts)),
      file.path(out_dir, "negprobes_features.csv"),
      row.names = FALSE
    )
  }
  
  # dimensional reductions
  reds <- Reductions(obj)
  write.csv(data.frame(reduction = reds), file.path(out_dir, "reductions_available.csv"), row.names = FALSE)
  
  for (red in reds) {
    emb <- Embeddings(obj, reduction = red)
    emb <- emb[cells_use, , drop = FALSE]
    
    safe_red <- gsub("[^A-Za-z0-9_]+", "_", red)
    write.csv(emb, file.path(out_dir, paste0("reduction_", safe_red, ".csv")))
  }
  
  rm(meta, rna_counts, rna_data)
  gc()
}

out_root <- "cosmx_liver_full_h5ad_export"
dir.create(out_root, showWarnings = FALSE)

tissues <- unique(obj$Run_Tissue_name)

for (tissue in tissues) {
  message("Exporting tissue: ", tissue)
  
  cells_use <- rownames(obj@meta.data)[obj$Run_Tissue_name == tissue]
  safe_tissue <- gsub("[^A-Za-z0-9_]+", "_", tissue)
  out_dir <- file.path(out_root, safe_tissue)
  
  export_group_to_folder(obj, cells_use, out_dir)
}