# Potential acceleration of `SuoLab-GZLab/CMAP`. 


This repository addresses two issues with the CMAP package:

1. For spatial transcriptomics datasets with more than 10k spots or cross-mapping of single-cell data with over 10k cells, the current implementation is very slow.
  
2. Most users either do not have suitable local machines, or, when memory-intensive devices and GPU devices are separate, may prefer a split, step-by-step execution mode.


```mermaid
graph LR
    A[gpu_svm_predict.py] --> B[run_svm.py]
    B --> C[map_cell_preprocess.R]
    C --> D[cuda_mapping.py]
    D --> E[after_map_cell2loc.R]
    
    style A fill:#e1f5ff
    style B fill:#e1f5ff
    style C fill:#ffe1e1
    style D fill:#e1f5ff
    style E fill:#ffe1e1
```
