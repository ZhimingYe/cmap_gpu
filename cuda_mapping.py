# %%
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import logging
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
import os,sys
sys.path.insert(1,os.path.abspath('.'))
import cmap_optimizer as mpt


logging.getLogger().setLevel(logging.INFO)


# %%

import pandas as pd
from pathlib import Path

def load_nested_list(path=".", prefix="step2OT_3"):
    path_obj = Path(path)
    files = list(path_obj.glob(f'{prefix}__*.parquet'))
    
    if not files:
        print(f"No files found in {path_obj.absolute()}")
        return {}
    
    result = {}
    
    for file in sorted(files):
        name_part = file.stem.replace(f'{prefix}__', '')
        
        # SPLIT: Cluster_1_cellxgene -> [Cluster, 1, cellxgene]
        parts = name_part.split('_')
        cluster = f"{parts[0]}_{parts[1]}"  # Cluster_1
        df_name = '_'.join(parts[2:])        # cellxgene
        
        if cluster not in result:
            result[cluster] = {}
        result[cluster][df_name] = pd.read_parquet(file)
        print(f"✓ {cluster}/{df_name}: {result[cluster][df_name].shape}")
    
    return result

dd = load_nested_list("/root/S3T1/3c2g", "step2OT_3")



# %%
def map_cell_to_spot(
    adata_sc,
    adata_sp,
    device="cuda" if torch.cuda.is_available() else "cpu",  # 自动检测CUDA
    learning_rate=0.1,
    num_epochs=1000,
    verbose=True,
    random_seed_set=None,
    para_distance=1.0,
    para_density=1.0,
):
    """
    Function: CMAP-OptimalSpot. Assign cells to optimal spots.

    Parameters:
        adata_sc: AnnData type. Expression matrix of single cells.
        adata_sp: AnnData type. Expression matrix of spots.
        device: 'cpu' or 'cuda'. Default: auto-detect CUDA availability.
        random_seed_set: Int(Default: None). Pass an int to reproduce the results.
        para_distance: The weight for the SSIM term.
        para_density: The weight for the entropy term.

    Return:
        a cell-by-spot corresponding matrix (AnnData type), containing the probability of mapping cell i to spot j.
    """

    if isinstance(adata_sc.X, csc_matrix) or isinstance(adata_sc.X, csr_matrix):
        C = np.array(adata_sc.X.toarray(), dtype="float32")
    elif isinstance(adata_sc.X, np.ndarray):
        C = np.array(adata_sc.X, dtype="float32")
    else:
        X_type = type(adata_sc.X)
        logging.error("AnnData X has unrecognized type: {}".format(X_type))
        raise NotImplementedError

    if isinstance(adata_sp.X, csc_matrix) or isinstance(adata_sp.X, csr_matrix):
        S = np.array(adata_sp.X.toarray(), dtype="float32")
    elif isinstance(adata_sp.X, np.ndarray):
        S = np.array(adata_sp.X, dtype="float32")
    else:
        X_type = type(adata_sp.X)
        logging.error("AnnData X has unrecognized type: {}".format(X_type))
        raise NotImplementedError

    if isinstance(device, str):
        device = torch.device(device)
    
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            logging.warning("CUDA is not available. Falling back to CPU.")
            device = torch.device('cpu')
        else:
            logging.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            logging.info(f"Initial CUDA memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    else:
        logging.info("Using CPU for computation.")

    if verbose:
        print_each = 100
    else:
        print_each = None

    logging.info(f"Begin mapping on device: {device}...")
    
    mapobject = mpt.Mapping(C=C, S=S, device=device, random_seed_set=random_seed_set)
    print(mapobject)
        
    mapping_matrix = mapobject.train(
        learning_rate=learning_rate, 
        num_epochs=num_epochs, 
        print_each=print_each, 
        para_distance=para_distance, 
        para_density=para_density,
    )

    if device.type == 'cuda':
        logging.info(f"Peak CUDA memory: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB")
        torch.cuda.empty_cache()
        logging.info("CUDA cache cleared.")

    logging.info("Saving results..")

    return mapping_matrix

# %%
import anndata as ad
results = {}

for cluster_name in dd.keys():
    print(f"Processing {cluster_name}...")
    
    results[cluster_name] = map_cell_to_spot(
        ad.AnnData(X=dd[cluster_name]['cellxgene'].values),
        ad.AnnData(X=dd[cluster_name]['spotxgene'].values),
        num_epochs=2000,
        para_distance=1.0,
        para_density=1.0
    )
    
    print(f"{cluster_name} completed!")


# %%
import dill
with open('predictions_Location_5_CLS3.pkl', 'wb') as f:
    dill.dump(results, f)

# %%
