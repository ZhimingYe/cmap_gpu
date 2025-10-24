

library(dplyr)
library(reticulate)
# use_condaenv("gene2gene") # Any env have dill
dill <- import("dill")
path <- "/ifs1/User/leafhi/agpgs_fp/spatial/run_CMAP/slide5_svm/root/predictions_Location_5_CLS3.pkl"
bytes <- readBin(path, what = "raw", n = file.info(path)$size)
py_obj <- dill$loads(bytes)

cluster_list <- qs2::qs_read("/ifs1/User/leafhi/agpgs_fp/spatial/run_CMAP/CMAP_8e62d04_For_mouse_CMAP_qsts_slide5.qs2_3/step2OT_3_.qs2")

mapping <- purrr::map2(cluster_list, py_obj, function(x,y){
  cellxgene <- x$cellxgene
  spotxgene <- x$spotxgene
  cell_spot <- y
  cell_spot_map_sub <- data.frame(Single_cell = rownames(cellxgene),
                                    Spot = rownames(spotxgene)[apply(cell_spot, 1, which.max)],
                                    Probability = apply(cell_spot, 1, max))
  cell_spot_map_sub
})

mapping <- do.call(rbind,mapping)
objectAtK3 <- qs2::qs_read("/ifs1/User/leafhi/agpgs_fp/spatial/run_CMAP/CMAP_8e62d04_For_mouse_CMAP_qsts_slide5.qs2_3/For_mouse_CMAP_qsts_slide5.qs2_3_PROFILEDRESULT_HMRF_8e62d04.qs2")
library(CMAP)
spot_neigh_list <- spatial_relation_all(objectAtK3$spatial_location,
                                        spatial_data_type=c('honeycomb'))
library(Seurat)
sc_meta_coord <- calculate_cell_location(cell_spot_map=mapping,
                                         st_meta =objectAtK3$spatial_location,
                                         sc_meta=objectAtK3$sc_meta,
                                         sc_norm=objectAtK3$sc_norm,
                                         st_norm=objectAtK3$st_norm,
                                         parallel = TRUE,                   
                                         batch = TRUE,
                                         spot_neigh_list=spot_neigh_list,
                                         radius = 1)

