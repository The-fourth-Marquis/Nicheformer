library(Seurat)

rds_path <- "/root/code/25-12/2023_nicheformer_data_anna.schaar/spatial/raw/LiverDataReleaseSeurat_newUMAP.RDS"
obj <- readRDS(rds_path)

obj
Assays(obj)
DefaultAssay(obj)
head(obj@meta.data)
colnames(obj@meta.data)