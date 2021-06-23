# -*- coding:utf-8 -*-

import os, argparse
import scanpy as sc
import anndata as an
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from utils.util import mkdir
import scanorama
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser(description='Infer the spatial coordinate of single-cell data by mapping it with spatial data')

parser.add_argument('-data_dir', type=str, default="../../data",
                    help='The tissue to use in integration, default: Liver')

parser.add_argument('-tissue', type=str, default="Liver", choices=["Liver", "Kidney"],
                    help='The tissue to use in integration, default: Liver')

parser.add_argument('-sample', type=int, default=1, choices=[1, 2],
                    help='The sample id to use in single cell data, default: 1')

def preprocessing(adata, n_pcs=50):
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=1000, inplace=True)
    sc.pp.pca(adata, n_comps=n_pcs)
    return adata

def infer_spatial_coord(args, knn=5, sigma=3.5, topN=3):
    coordinates = pd.read_csv(os.path.join(args.data_dir,
                                           args.tissue,
                                           "%s.idx" % args.tissue),
                              index_col=0)
    counts = sc.read_csv(os.path.join(args.data_dir,
                                      args.tissue,
                                      "%s.count.csv" % args.tissue)
                         ).transpose()
    adata_sp = counts[coordinates.index, :]
    coordinates = coordinates.to_numpy()
    adata_sp.obsm["spatial"] = coordinates
    adata_sp = preprocessing(adata_sp)

    adata_sc = sc.read_csv(os.path.join(args.data_dir,
                                        args.tissue,
                                        "%s%d_rm_batch.txt" % (args.tissue, args.sample))
                           , delimiter=" ").transpose()
    adata_sc = preprocessing(adata_sc)
    adata_concat = adata_sp.concatenate(
        adata_sc,
        batch_key="dataset",
        batch_categories=["slide-seq", "scRNA-seq"],
        join="outer",
        uns_merge="first",
    )
    embedding_key = "X_scanorama"

    sc.external.pp.scanorama_integrate(adata_concat, "dataset", knn=knn, sigma=sigma)
    dist_sc_sp = cosine_similarity(
        adata_concat[adata_concat.obs["dataset"] == "scRNA-seq"].obsm[embedding_key],
        adata_concat[adata_concat.obs["dataset"] == "slide-seq"].obsm[embedding_key],
    )
    sorted_close_spatial_cell_indexs = np.argsort(dist_sc_sp, axis=1)
    closest_spatial_cell_indexs = np.fliplr(sorted_close_spatial_cell_indexs[:, -topN:])
    spatial_coords = np.zeros((closest_spatial_cell_indexs.shape[0], 2))
    for idx in range(closest_spatial_cell_indexs.shape[0]):
        for coord_idx in range(2):
            spatial_coords[idx, coord_idx] = np.average(np.array([coordinates[spidx, 1+coord_idx] for spidx in closest_spatial_cell_indexs[idx, :]]), weights=dist_sc_sp[idx, closest_spatial_cell_indexs[idx, :]] + 1.0)
    save_coord_fp = os.path.join(args.data_dir,
                            args.tissue,
                            "%s%d_coord.csv" % (args.tissue, args.sample))
    np.savetxt(save_coord_fp, spatial_coords, delimiter=",", fmt='%.2f')
    print("save coordinates successful!")
    
if __name__ == "__main__":
    args = parser.parse_args()
    infer_spatial_coord(args, topN=3)
