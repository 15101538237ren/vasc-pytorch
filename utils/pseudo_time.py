# -*- coding:utf-8 -*-
import os
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from utils.config import get_args
from utils.util import mkdir

def plt_setting(fontsz = 10):
    plt.rc('font', family='Arial')
    plt.rc('xtick', labelsize=fontsz)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=fontsz)  # fontsize of the tick labels

def get_pseudo_time(data_fp, dataset, root_idx= 0, VASC=True):
    fig_dir = os.path.join("../figures")
    mkdir(fig_dir)
    plt_setting()

    if VASC:
        adata = sc.read_csv(data_fp, delimiter="\t", first_column_names=None)
    else:
        # Read data
        adata = sc.read_csv(data_fp, delimiter="\t")

        # Preprocess data
        sc.pp.recipe_zheng17(adata)

        # PCA dimension reduction
        sc.tl.pca(adata, svd_solver='arpack')
        print(adata)

    # Neighbor Graph
    sc.pp.neighbors(adata, n_neighbors=50)

    # Denoise Graph
    sc.tl.diffmap(adata)

    # Clustering
    sc.tl.louvain(adata, resolution=1.0)

    # Trajactory
    sc.tl.paga(adata, groups='louvain')

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    plt.subplots_adjust(left=0.2, bottom=0.2)

    sc.pl.paga(adata, color=['louvain'], ax=ax, show=False)

    ax.set_title(dataset)
    fig_fp = os.path.join(fig_dir, "%s_paga_vasc.pdf" % dataset)
    plt.savefig(fig_fp, dpi=300)
    fa = sc.tl.draw_graph(adata, init_pos='paga')

    adata.uns['iroot'] = root_idx
    sc.tl.dpt(adata)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    plt.subplots_adjust(left=0.2, bottom=0.2)
    sc.pl.draw_graph(adata, color=['dpt_pseudotime'], layout=fa, legend_loc='on data', ax=ax, show=False)
    ax.set_title(dataset)
    fig_fp = os.path.join(fig_dir, "%s_pseudo_time_vasc.pdf" % dataset)
    plt.savefig(fig_fp, dpi=300)

    print("figure plotted successful!")
    return adata.obs['dpt_pseudotime']

def plot_umap(data_fp, dataset, root_idx= 0, VASC=True):
    fig_dir = os.path.join("../figures")
    mkdir(fig_dir)
    plt_setting()

    if VASC:
        adata = sc.read_csv(data_fp, delimiter="\t", first_column_names=None)
    else:
        # Read data
        adata = sc.read_csv(data_fp, delimiter="\t")

        # Preprocess data
        sc.pp.recipe_zheng17(adata)

    # Neighbor Graph
    sc.pp.neighbors(adata, n_neighbors=10)

    sc.tl.umap(adata)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    plt.subplots_adjust(left=0.2, bottom=0.2)

    adata.uns['iroot'] = root_idx
    sc.tl.dpt(adata)
    sc.pl.umap(adata, color='dpt_pseudotime', ax=ax, show=False)
    ax.set_title(dataset)
    fig_fp = os.path.join(fig_dir, "%s_umap_vasc.pdf" % dataset)
    plt.savefig(fig_fp, dpi=300)

if __name__ == "__main__":
    sc.settings.verbosity = 3
    sc.settings.set_figure_params(dpi=300, frameon=False, figsize=(4, 4), facecolor='white')
    args = get_args()

    feature_dir = os.path.join("../", args.dataset_dir, args.feature_dir)
    feature_fp = os.path.join(feature_dir, "%s.tsv" % args.dataset)
    #plot_umap(feature_fp, args.dataset, root_idx=300)
    get_pseudo_time(feature_fp, args.dataset)

    # expr_fp = os.path.join("../", args.dataset_dir, args.dataset, "%s.txt" % args.dataset)
    # get_pseudo_time(expr_fp, args.dataset, VASC=False)
