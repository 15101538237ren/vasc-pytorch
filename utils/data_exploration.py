# -*- coding:utf-8 -*-
import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.spatial import distance
from utils.util import mkdir
from utils.config import get_args
args = get_args()

def plt_setting(fontsz = 10):
    plt.rc('font', family='Arial')
    plt.rc('xtick', labelsize=fontsz)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=fontsz)  # fontsize of the tick labels

def plot_hist(dists, fig_fp, title, xlabel, ylabel, bins=50):
    plt_setting()
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    plt.subplots_adjust(left=0.2, bottom=0.2)
    ax.hist(dists, bins=bins, edgecolor='black', alpha=0.5, linewidth=0.5)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, weight='bold', fontsize=10)
    ax.set_ylabel(ylabel, weight='bold', fontsize=10)
    mkdir(os.path.dirname(fig_fp))
    plt.savefig(fig_fp)

def get_highly_variable_genes_and_expr(dataset_dir, dataset):
    if dataset == "drosophila":
        expr_fp = os.path.join("..", dataset_dir, dataset, "%s_all.txt" % dataset)
        adata = sc.read_csv(expr_fp, delimiter="\t")
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, flavor='seurat')
        highly_variable = adata.var.loc[:, "highly_variable"].values
        df = pd.read_csv(expr_fp, sep="\t", header=0, index_col=0)
        df = df.iloc[:, highly_variable]
        expr_highly_variable_fp = os.path.join("..", dataset_dir, dataset, "%s.txt" % dataset)
        df.to_csv(expr_highly_variable_fp, sep="\t")
        print("Save sucessful!")
def hist_pipeline():
    spatial_dists = np.load("../data/drosophila/drosophila_spatial_dist.npy")
    spatial_hist_fp = "../figures/drosophila_spatial_hist.pdf"
    plot_hist(spatial_dists[np.triu_indices(spatial_dists.shape[0])], spatial_hist_fp, "Drosophila Spatial Hist",
              "Spatial Distance", "Freq")

    features = pd.read_csv("../data/features/drosophila.tsv", sep="\t", header=0, index_col=0).values
    feature_dists = distance.cdist(features, features, 'euclidean')
    feature_hist_fp = "../figures/drosophila_feature_hist.pdf"
    plot_hist(feature_dists[np.triu_indices(feature_dists.shape[0])], feature_hist_fp, "Drosophila Feature Hist",
              "Feature Distance", "Freq")

if __name__ == "__main__":
    get_highly_variable_genes_and_expr(args.dataset_dir, args.dataset)
