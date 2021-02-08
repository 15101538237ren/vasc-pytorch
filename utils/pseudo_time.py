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

def get_pseudo_time(feature_fp, dataset):
    adata = sc.read_csv(feature_fp, delimiter="\t", first_column_names=None)
    sc.pp.neighbors(adata, n_neighbors=10)
    sc.tl.louvain(adata, resolution=1.0)
    sc.tl.paga(adata)
    fa = sc.tl.draw_graph(adata, root=11)
    # adata.uns['iroot']
    plt_setting()
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    plt.subplots_adjust(left=0.2, bottom=0.2)
    sc.pl.draw_graph(adata, color=['paga_pseudotime'], layout=fa, legend_loc='on data', ax=ax, show=False)
    ax.set_title(dataset)
    fig_dir = os.path.join("../figures")
    mkdir(fig_dir)
    fig_fp = os.path.join(fig_dir, "%s_pseudo_time.pdf" % dataset)
    plt.savefig(fig_fp, dpi=300)
    print("figure plotted successful!")
if __name__ == "__main__":
    sc.settings.verbosity = 3
    sc.settings.set_figure_params(dpi=300, frameon=False, figsize=(4, 4), facecolor='white')
    args = get_args()
    feature_dir = os.path.join("../", args.dataset_dir, args.feature_dir)
    feature_fp = os.path.join(feature_dir, "%s.tsv" % args.dataset)
    get_pseudo_time(feature_fp, args.dataset)
