

library(reticulate)
# use_condaenv("gene2gene") # any envrionment have dill is OK
dill <- import("dill")
path <- "/ifs1/User/leafhi/agpgs_fp/spatial/run_CMAP/slide5_svm/predictions_k3.pkl"
bytes <- readBin(path, what = "raw", n = file.info(path)$size)
py_obj <- dill$loads(bytes)

# objectAtK3 <- qs2::qs_read("/ifs1/User/leafhi/agpgs_fp/spatial/run_CMAP/CMAP_8e62d04_For_mouse_CMAP_qsts_slide5.qs2_3/For_mouse_CMAP_qsts_slide5.qs2_3_PROFILEDRESULT_HMRF_8e62d04.qs2")
# Previous environ cache


pred_sc_svm <- py_obj$predictions_1F
names(pred_sc_svm) <- rownames(objectAtK3$test_set)

pred_st_svm <- py_obj$predictions_1T
names(pred_st_svm) <- rownames(objectAtK3$train_set)
library(Seurat)
map_cell_to_spot2 <- function(sc_norm,
                             sc_meta,
                             st_norm,
                             spatial_location,
                             batch = TRUE,
                             st_gene_list = NULL,
                             pca_method = "prcomp_irlba",
                             pred_sc_svm,
                             pred_st_svm,
                             python_path,
                             num_epochs = 2000L,
                             para_distance = 1.0,
                             para_density = 1.0){
  if(is.null(st_gene_list)){
    st_gene_list <- CMAP:::map_spot_genes(sc_norm = sc_norm, sc_meta = sc_meta,
                                   st_norm = st_norm, spatial_location = spatial_location)
    shvgs <- unique(unlist(st_gene_list))
  }
  cluster <- sort(unique(pred_sc_svm))
  cell_spot_map <- data.frame()
  CSP <- list()
  for (j in 1:length(cluster)) {
    cluster_id = cluster[j]
    single_cell_id <- names(pred_sc_svm[pred_sc_svm == cluster_id])
    spot_id <- names(pred_st_svm[pred_st_svm == cluster_id])
    sub_embedding <- CMAP:::data_to_transform(sc_norm = sc_norm[, single_cell_id, drop = FALSE],
                                       st_norm = st_norm[, spot_id, drop = FALSE],
                                       spatial_genes = unique(st_gene_list[[j]]),
                                       batch = batch,
                                       pca_method = pca_method)
    cellxgene <- as.data.frame(t(sub_embedding)[single_cell_id, , drop = FALSE])
    spotxgene <- as.data.frame(t(sub_embedding)[spot_id, , drop = FALSE])
    CSP[[paste0("Cluster_", j)]] <- list(cellxgene = cellxgene, spotxgene = spotxgene)
  }
  return(CSP)
}

dd <- map_cell_to_spot2(sc_norm=objectAtK3$sc_norm,sc_meta=objectAtK3$sc_meta,
                                  st_norm=objectAtK3$st_norm,spatial_location=objectAtK3$spatial_location,
                                  pred_sc_svm=pred_sc_svm, pred_st_svm=pred_st_svm,
                                  batch=TRUE,
                                  num_epochs=2000L,
                                  para_distance=1.0,
                                  para_density=1.0)

str(dd, max.level = 2)
library(arrow)

save_nested_list <- function(data, prefix = "data") {
  for(i in seq_along(data)) {
    cluster_name <- names(data)[i]
    for(j in seq_along(data[[i]])) {
      df_name <- names(data[[i]])[j]
      filename <- paste0(prefix, "_", cluster_name, "_", df_name, ".parquet")
      write_parquet(data[[i]][[j]], filename)
      cat("Saved:", filename, "\n")
    }
  }
}

# save_nested_list(dd, "/ifs1/User/leafhi/agpgs_fp/spatial/run_CMAP/CMAP_8e62d04_For_mouse_CMAP_qsts_slide5.qs2_3/step2OT_3_")
# PQ has not row names. So another copy in QS2 is necessary. 
# qs2::qs_save(dd, "/ifs1/User/leafhi/agpgs_fp/spatial/run_CMAP/CMAP_8e62d04_For_mouse_CMAP_qsts_slide5.qs2_3/step2OT_3_.qs2")

