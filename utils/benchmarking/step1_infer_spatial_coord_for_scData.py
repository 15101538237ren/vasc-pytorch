# -*- coding:utf-8 -*-

import os, argparse
import scanpy as sc
import anndata as an
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scanorama
from sklearn.metrics.pairwise import cosine_distances

parser = argparse.ArgumentParser(description='Infer the spatial coordinate of single-cell data by mapping it with spatial data')

parser.add_argument('-data_dir', type=str, default="../../data",
                    help='The tissue to use in integration, default: Liver')

parser.add_argument('-tissue', type=str, default="Liver", choices=["Liver", "Kidney"],
                    help='The tissue to use in integration, default: Liver')

parser.add_argument('-sample', type=int, default=1, choices=[1, 2],
                    help='The sample id to use in single cell data, default: 1')

def preprocessing(adata):
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000, inplace=True)
    return adata

def infer_spatial_coord(args):
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
    adatas = [adata_sp, adata_sc]
    scanorama.integrate_scanpy(adatas)

    adata_concat = adata_sp.concatenate(
        adata_sc,
        batch_key="dataset",
        batch_categories=["slide-seq", "scRNA-seq"],
        join="outer",
        uns_merge="first",
    )
    embedding_key = "scanorama_embedding"
    adata_concat.obsm[embedding_key] = np.concatenate([item.obsm['X_scanorama'] for item in adatas])
    dist_sc_sp = 1 - cosine_distances(
        adata_concat[adata_concat.obs["dataset"] == "scRNA-seq"].obsm[embedding_key],
        adata_concat[adata_concat.obs["dataset"] == "slide-seq"].obsm[embedding_key],
    )
    closest_spatial_cell_indexs = dist_sc_sp.argmax(axis=1)
    save_coord_fp = os.path.join(args.data_dir,
                            args.tissue,
                            "%s%d_coord.csv" % (args.tissue, args.sample))
    spatial_coords = coordinates[closest_spatial_cell_indexs, 1:]
    np.savetxt(save_coord_fp, spatial_coords, delimiter=",", fmt='%.2f')
    print("save coordinates successful!")
if __name__ == "__main__":
    args = parser.parse_args()
    infer_spatial_coord(args)
