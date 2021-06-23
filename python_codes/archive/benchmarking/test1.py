# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import pandas as pd
import scanpy as sc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance
from torch.utils.data import TensorDataset, DataLoader
from models.vasc import VASC
from utils.config import get_args
from utils.util import mkdir, get_data,get_labels , preprocess_data, train, evaluate, save_features, prepare_cuda
from python_codes.archive.visualize.visualization import plot_2d_features, plot_2d_features_pesudo_time
import matplotlib.pyplot as plt

max_spatial_dist = 300.0

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

def plot_feature_hist(tissue, sample, args, spatial=True, feature_dist_thrs=6, spatial_dist_thrs=max_spatial_dist//2):
    args.spatial = spatial
    args.dataset = tissue
    thrs_str = "f_%.1f_sp_%.0f" % (feature_dist_thrs, spatial_dist_thrs) if spatial else "%d_non_sp" % sample
    feature_fp = os.path.join(args.dataset_dir, args.feature_dir, "%s_%s.tsv" % (args.dataset, thrs_str))
    features = pd.read_csv(feature_fp, sep="\t", header=0, index_col=0).values
    feature_dists = distance.cdist(features, features, 'euclidean')
    fig_dir = "../../figures/%s" % tissue
    mkdir(fig_dir)
    feature_hist_fp ="%s/%s_%s_feature_hist.pdf" % (fig_dir, args.dataset, thrs_str)
    plot_hist(feature_dists[np.triu_indices(feature_dists.shape[0])], feature_hist_fp, "Drosophila Feature Hist",
              "Feature Distance", "Freq")

def preprocessing(adata):
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e6)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000, inplace=True)
    return adata

def read_expr_data(data_dir, tissue, sample):
    data_path = os.path.join(data_dir, tissue,"%s%d_rm_batch.txt" % (tissue, sample))
    adata_sc = sc.read_csv(data_path, delimiter=" ").transpose()
    adata_sc = preprocessing(adata_sc)
    expr = adata_sc.X[:, adata_sc.var['highly_variable']]
    samples = adata_sc.obs_names.tolist()
    genes = adata_sc.var_names.tolist()
    return expr, genes, samples

def read_spatial_coord(data_dir, tissue, sample):
    coordinates = pd.read_csv(os.path.join(data_dir,
                                           tissue,
                                           "%s%d_coord.csv" % (tissue, sample)), index_col=None, header=None).values
    return coordinates

def run_model(tissue, sample, spatial=True, feature_dist_thrs=6, spatial_dist_thrs=max_spatial_dist//2):
    args = get_args()
    args.spatial = spatial
    args.dataset = tissue
    thrs_str = "f_%.1f_sp_%.0f" % (feature_dist_thrs, spatial_dist_thrs) if spatial else "%d_non_sp" % sample
    model_fp = os.path.join("../../data", "models", "%s_%s.pt" % (args.dataset, thrs_str))
    print(model_fp)
    coordinates = read_spatial_coord(args.dataset_dir, tissue, sample)
    spatial_dists = euclidean_distances(coordinates, coordinates)
    print("Distance calculated")
    spatial_dists = torch.from_numpy((spatial_dists * max_spatial_dist/spatial_dists.max()))
    expr, genes, samples = read_expr_data(args.dataset_dir, tissue, sample)
    print("Expression data readed")
    n_cell, n_gene = expr.shape
    expr_t = torch.Tensor(expr)
    train_loader = DataLoader(dataset=TensorDataset(expr_t), batch_size=args.batch_size, shuffle=False)
    device = prepare_cuda(args)
    vasc = VASC(x_dim=n_gene, z_dim=args.z_dim, var=args.var, dropout=args.dropout, isTrain=args.train).to(device)
    optimizer = torch.optim.RMSprop(vasc.parameters(), lr=args.lr)
    if args.train:
        train(vasc, optimizer, train_loader, model_fp, spatial_dists, feature_dist_thrs, spatial_dist_thrs, args)
    reduced_reprs = evaluate(vasc, expr_t, model_fp, args)
    if args.save_features:
        feature_dir = os.path.join(args.dataset_dir, args.feature_dir)
        save_features(reduced_reprs, feature_dir, "%s_%s" % (args.dataset, thrs_str))

def get_pseudo_time_of_feaure(feature_fp, n_neighbors = 10, root_idx= 4000):
    adata = sc.read_csv(feature_fp, delimiter="\t", first_column_names=None)

    # Neighbor Graph
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    sc.tl.umap(adata)
    adata.uns['iroot'] = root_idx

    sc.tl.dpt(adata)
    return adata.obs['dpt_pseudotime']

def plot_pesudo_time(tissue, sample, feature_dist_thrs=6, spatial_dist_thrs=max_spatial_dist//2, fontsz=12, root_idx=4000):
    args = get_args()
    args.dataset = tissue
    lims = [0, 6000]
    plt_setting(fontsz=fontsz)
    fig_sz = (5 * 2, 4)
    fig, axs = plt.subplots(1, 2, figsize=fig_sz)
    plt.subplots_adjust(wspace=0.3)

    spatial_cords = read_spatial_coord(args.dataset_dir, tissue, sample)

    feature_dir = os.path.join(args.dataset_dir, args.feature_dir)

    non_spatial_model_feature_fp = os.path.join(feature_dir, "%s_%d_non_sp.tsv" % (tissue, sample))
    non_spatial_pseudotime = get_pseudo_time_of_feaure(non_spatial_model_feature_fp, root_idx=root_idx)

    spatial_model_feature_fp = os.path.join(feature_dir, "%s_%s.tsv" % (args.dataset, "f_%.1f_sp_%.0f" % (feature_dist_thrs, spatial_dist_thrs)))
    spatial_pseudotime = get_pseudo_time_of_feaure(spatial_model_feature_fp, root_idx=root_idx)

    pesudo_times = [non_spatial_pseudotime, spatial_pseudotime]
    titles = ["VASC: %s" % tissue, "VASC + SP: %s" % tissue]

    for spi, pseudotime in enumerate(pesudo_times):
        ax = axs[spi]
        ax.grid(False)
        st = ax.scatter(spatial_cords[:, 0], spatial_cords[:, 1], s=4, c=pseudotime)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(st, cax=cax)
        ax.set_aspect('equal', 'box')

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        if spi == 0:
            ax.set_ylabel("z Spatial Coord", fontsize=fontsz)
        ax.set_xlabel("x Spatial Coord", fontsize=fontsz)
        ax.set_title(titles[spi])

    fig_dir = os.path.join("../../figures", args.dataset)
    mkdir(fig_dir)
    fig_fp = os.path.join(fig_dir, "Pesudotime_VASC_VS_SP_Root_%d.jpg" % root_idx)
    plt.savefig(fig_fp, dpi=300)
    print("Save %s successful!" % fig_fp)

if __name__ == "__main__":
    args = get_args()
    root_idxs = [5 * i for i in range(1200)]
    tissues = ["Liver"]  # , "Liver""Kidney", "Liver",
    samples = [1]  # , 12, 1,
    spatials = [True]  # False, False,
    for tid, tissue in enumerate(tissues):
        for root_idx in root_idxs:
            # plot_feature_hist(tissue, samples[tid], args, spatial=False)
            # run_model(tissue, samples[tid], spatial=spatials[tid])
            plot_pesudo_time(tissue, samples[tid], root_idx=root_idx)

