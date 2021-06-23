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
max_spatial_dist = 300.0
def plt_setting(fontsz = 10):
    plt.rc('font', family='Arial')
    plt.rc('xtick', labelsize=fontsz)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=fontsz)  # fontsize of the tick labels

def plot_umap(tissue, sample, n_neighbors=15, n_pcs=50, root_idx= 1000, VASC=True, spatial=False, feature_dist_thrs=6, spatial_dist_thrs=max_spatial_dist//2):
    fig_dir = os.path.join("../../figures", tissue)
    mkdir(fig_dir)

    plt_setting()

    if VASC:
        if spatial:
            data_fp = os.path.join("../../data/features/%s_%s.tsv" % (
            tissue, "f_%.1f_sp_%.0f" % (feature_dist_thrs, spatial_dist_thrs)))
        else:
            data_fp = os.path.join("../../data/features/%s_%d_non_sp.tsv" % (tissue, sample))
        adata = sc.read_csv(data_fp, delimiter="\t", first_column_names=None)
    else:
        # Read data
        data_fp = os.path.join("../../data/%s/%s%d_rm_batch.txt" % (tissue, tissue, sample))
        adata = sc.read_csv(data_fp, delimiter=" ")

        # Preprocess data
        sc.pp.recipe_zheng17(adata)
        sc.pp.pca(adata, n_comps=n_pcs)

    # Neighbor Graph
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    sc.tl.umap(adata)
    adata.uns['iroot'] = root_idx
    sc.tl.dpt(adata)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    plt.subplots_adjust(left=0.2, bottom=0.2)
    sc.pl.umap(adata, color='dpt_pseudotime', ax=ax, show=False)
    ax.set_title("%s %d" % (tissue, sample))
    vasc_suff = "_vasc" if VASC else ""
    if spatial:
        vasc_suff += "_spatial"
    fig_fp = os.path.join(fig_dir, "%s_%d_umap%s.pdf" % (tissue, sample, vasc_suff))
    plt.savefig(fig_fp, dpi=300)

if __name__ == "__main__":
    tissues = ["Kidney"] #, "Liver"
    samples = [2] #, 2
    for tid, tissue in enumerate(tissues):
        plot_umap(tissue, samples[tid], VASC=True, spatial=False)
        plot_umap(tissue, samples[tid], VASC=True, spatial=True)
