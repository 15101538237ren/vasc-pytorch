# -*- coding:utf-8 -*-
import os
import anndata
import pandas as pd
import scanpy as sc
import squidpy as sq
import numpy as np
from utils.config import get_args
from utils.util import mkdir, get_spatial_coords, get_squidpy_data, SPATIAL_LIBD_DATASETS, SPATIAL_N_FEATURE_MAX, VISIUM_DATASETS

def feature_with_pseudo_time(args, feature_dir, dataset, linear=True,  n_neighbors=30, root_idx= 50):
    spatials = [False, True]
    feature_with_pseudo_time_dir = os.path.join(feature_dir, "feature_with_pseudo_time")
    mkdir(feature_with_pseudo_time_dir)

    for sid, spatial in enumerate(spatials):
        name = args.dataset if not spatial else "%s_with_spatial" % dataset
        feature_fp = os.path.join(feature_dir, "%s.tsv" % name)
        adata = sc.read_csv(feature_fp, delimiter="\t", first_column_names=None)

        # Neighbor Graph
        sc.pp.neighbors(adata, n_neighbors=n_neighbors)
        sc.tl.umap(adata)
        adata.uns['iroot'] = root_idx
        sc.tl.dpt(adata)
        pseudo_times = adata.obs['dpt_pseudotime'].values.reshape(-1, 1)
        arr = np.concatenate((adata.X, pseudo_times), axis=1)
        fp = os.path.join(feature_with_pseudo_time_dir, "%s_feature_wt_pseudotime.tsv" % name)

        np.savetxt(fp, arr, header='', delimiter='\t', footer='', comments='')
        print("Saved %s succesful!" % fp)

if __name__ == "__main__":
    args = get_args()
    linears = [True, False]
    datasets = SPATIAL_LIBD_DATASETS
    for linear in linears:
        for dataset in datasets:
            args.dataset = dataset
            feature_suff = "features_linear" if linear else "features_switch"
            feature_dir = os.path.join(args.dataset_dir, feature_suff)
            feature_with_pseudo_time(args, feature_dir, args.dataset, linear=linear)
