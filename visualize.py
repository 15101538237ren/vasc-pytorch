# -*- coding:utf-8 -*-
import os
import anndata
import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.config import get_args
from utils.util import mkdir, get_spatial_coords, SQUIDPY_DATASETS, SPATIAL_N_FEATURE_MAX
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import distance

def plt_setting(fontsz = 10):
    plt.rc('font', family='Arial', weight='bold')
    plt.rc('xtick', labelsize=fontsz)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=fontsz)  # fontsize of the tick labels

def plot_spatial_cord_with_pseudo_time(figure_dir, feature_dir, spatial_cords, dataset, root_idx= 50, VASC=True, n_neighbors=10):
    fig_dir = os.path.join(figure_dir, dataset)
    mkdir(fig_dir)

    plt_setting()
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(wspace=0.4, hspace=0.5, bottom=0.2)
    spatials = [False, True]

    neigh = NearestNeighbors(n_neighbors=n_neighbors + 1)
    neigh.fit(spatial_cords)
    neigh_ind = neigh.kneighbors(spatial_cords, n_neighbors + 1, return_distance=False)
    pseudo_times = []
    titles = ["VASC", "VASC + SP"]
    marker_sz = 4 if dataset != "seqfish" else 0.5
    for sid, spatial in enumerate(spatials):
        ax = axs[sid]
        name = args.dataset if not spatial else "%s_with_spatial" % dataset
        title = titles[sid]
        feature_fp = os.path.join(feature_dir, "%s.tsv" % name)

        if VASC:
            adata = sc.read_csv(feature_fp, delimiter="\t", first_column_names=None)
        else:
            # Read data
            adata = sc.read_csv(feature_fp, delimiter="\t")
            # Preprocess data
            sc.pp.recipe_zheng17(adata)
        adata = adata[:, :]

        # Neighbor Graph
        sc.pp.neighbors(adata, n_neighbors=n_neighbors)
        sc.tl.umap(adata)
        adata.uns['iroot'] = root_idx
        sc.tl.dpt(adata)
        ax.grid(False)

        st = ax.scatter(spatial_cords[:, 0], spatial_cords[:, 1], s=marker_sz, c=adata.obs['dpt_pseudotime'])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        clb = fig.colorbar(st, cax=cax)
        clb.ax.set_ylabel("Norm. Pseudotime Dist",  labelpad=10, rotation=270, fontsize=8, weight='bold')
        ax.set_aspect('equal', 'box')
        ax.set_xlabel("X", weight='bold')
        ax.set_ylabel("Y", weight='bold')
        ax.set_title("%s" % title, fontsize=12, weight='bold')
        pseudo_times.append(adata.obs['dpt_pseudotime'].tolist())
    pseudo_time_neighbor_dists = []
    ax = axs[2]
    colors = ["#ee6c4d", "#293241"]
    bins = np.arange(0, 1.01, 0.1)
    for sid, spatial in enumerate(spatials):
        pseudo_time_neighbor_dists.append([])
        pseudo_time_vals = pseudo_times[sid]
        for cid, ind in enumerate(neigh_ind):
            pseudo_time_neighbor_dists[sid].append([])
            indexs = ind[1:]
            for index in indexs:
                neighbor_pseudo_time = pseudo_time_vals[index]
                psudo_time_dist = np.sqrt(np.power((pseudo_time_vals[cid] - neighbor_pseudo_time), 2))
                pseudo_time_neighbor_dists[sid][cid].append(psudo_time_dist)

        ax.hist(np.array(pseudo_time_neighbor_dists[sid]).flatten().tolist(), bins=bins, density=False, color=colors[sid], edgecolor='black', alpha=0.5,
                              linewidth=1, label=titles[sid])
    ax.set_title("Histogram of Neighboring Pseudo-time Distances", fontsize=12, weight='bold')
    ax.set_xlabel("Pseudo-time Distances of Neighbors", weight='bold')
    ax.set_ylabel("Frequency", weight='bold')
    ax.legend()
    fig_fp = os.path.join(fig_dir, "%s.pdf" % dataset)
    plt.savefig(fig_fp, dpi=300)


def plot_spatial_vs_feature_dist_colored_pseudo_time(figure_dir, feature_dir, spatial_cords, dataset, root_idx=50, n_neighbors=20, ncells=100):
    cm = plt.get_cmap('gist_rainbow')
    fig_dir = os.path.join(figure_dir, dataset)
    mkdir(fig_dir)
    coords_ind = np.random.choice(spatial_cords.shape[0], ncells, replace=False)
    coords = spatial_cords[coords_ind, :].astype(float)
    spatial_dists = distance.cdist(coords, coords, 'euclidean')
    spatial_dists = (spatial_dists / np.max(spatial_dists)) * SPATIAL_N_FEATURE_MAX

    print("Finish Spatial Dist Calculation of %s" % dataset)
    plt_setting()
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    plt.subplots_adjust(wspace=0.4, hspace=0.5, bottom=0.2)
    spatials = [False, True]

    pseudotime_dists_arr = []
    feature_dist_arr = []
    titles = ["VASC", "VASC + SP"]
    marker_sz = 4 if dataset != "seqfish" else 0.5
    for sid, spatial in enumerate(spatials):
        name = args.dataset if not spatial else "%s_with_spatial" % dataset
        feature_fp = os.path.join(feature_dir, "%s.tsv" % name)
        vals = pd.read_csv(feature_fp, sep="\t", header=None).values[coords_ind, :]
        adata = anndata.AnnData(X=vals)
        feature_dists = distance.cdist(adata.X, adata.X, 'euclidean')
        feature_dists = (feature_dists / np.max(feature_dists)) * SPATIAL_N_FEATURE_MAX
        feature_dist_arr.append(feature_dists)
        print("Finish Feature %d Dist Calculation" % sid)

        # Neighbor Graph
        sc.pp.neighbors(adata, n_neighbors=n_neighbors)
        sc.tl.umap(adata)
        adata.uns['iroot'] = root_idx
        sc.tl.dpt(adata)
        pseudotime = np.array(adata.obs['dpt_pseudotime'].tolist()).reshape((-1, 1))
        pseudotime_dists = distance.cdist(pseudotime, pseudotime, 'euclidean')
        pseudotime_dists = (pseudotime_dists / np.max(pseudotime_dists)) * SPATIAL_N_FEATURE_MAX
        pseudotime_dists_arr.append(pseudotime_dists)
    spatial_neighbor_dists = []
    feature_neighbor_dists = []
    pseudo_time_neighbor_dists = []
    for sid, spatial in enumerate(spatials):
        ax = axs[sid]
        ax.grid(False)
        spatial_neighbor_dists.append([])
        feature_neighbor_dists.append([])
        pseudo_time_neighbor_dists.append([])

        for ci in range(coords.shape[0]):
            for cj in range(ci + 1, coords.shape[0]):
                spatial_neighbor_dists[sid].append(spatial_dists[ci][cj])
                feature_neighbor_dists[sid].append(feature_dist_arr[sid][ci][cj])
                pseudo_time_neighbor_dists[sid].append(pseudotime_dists_arr[sid][ci][cj])

        st = ax.scatter(spatial_neighbor_dists[sid], feature_neighbor_dists[sid], s=marker_sz, c=pseudo_time_neighbor_dists[sid], cmap=cm)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        clb = fig.colorbar(st, cax=cax)
        clb.ax.set_ylabel("Norm. Pseudotime Dist",  labelpad=10, rotation=270, fontsize=8, weight='bold')
        ax.set_xlabel("Norm. Spatial Dist", weight='bold')
        ax.set_ylabel("Norm. Feature Dist", weight='bold')
        ax.set_title("%s" % titles[sid], fontsize=12, weight='bold')

    fig_fp = os.path.join(fig_dir, "%s_dist_scatter.pdf" % dataset)
    plt.savefig(fig_fp, dpi=300)


if __name__ == "__main__":
    mpl.use('macosx')
    args = get_args()

    datasets = [
                   "V1_Breast_Cancer_Block_A_Section_1", "V1_Breast_Cancer_Block_A_Section_2",
                   "V1_Human_Heart", "V1_Human_Lymph_Node", "V1_Mouse_Kidney", "V1_Mouse_Brain_Sagittal_Posterior",
                   "V1_Mouse_Brain_Sagittal_Posterior_Section_2", "V1_Mouse_Brain_Sagittal_Anterior",
                   "V1_Mouse_Brain_Sagittal_Anterior_Section_2", "V1_Human_Brain_Section_2",
                   "V1_Adult_Mouse_Brain_Coronal_Section_1", "V1_Adult_Mouse_Brain_Coronal_Section_2",
                   "Targeted_Visium_Human_Cerebellum_Neuroscience", "Parent_Visium_Human_Cerebellum",
                   "Targeted_Visium_Human_SpinalCord_Neuroscience", "Parent_Visium_Human_SpinalCord",
                   "Targeted_Visium_Human_Glioblastoma_Pan_Cancer", "Parent_Visium_Human_Glioblastoma",
                   "Targeted_Visium_Human_BreastCancer_Immunology", "Parent_Visium_Human_BreastCancer",
                   "Targeted_Visium_Human_OvarianCancer_Pan_Cancer", "Targeted_Visium_Human_OvarianCancer_Immunology",
                   "Parent_Visium_Human_OvarianCancer", "Targeted_Visium_Human_ColorectalCancer_GeneSignature",
                   "Parent_Visium_Human_ColorectalCancer"
               ] + SQUIDPY_DATASETS
    for dataset in datasets:
        args.dataset = dataset
        coords = get_spatial_coords(args)
        feature_dir = os.path.join(args.dataset_dir, args.feature_dir)
        plot_spatial_cord_with_pseudo_time(args.figure_dir, feature_dir, coords, args.dataset)
        plot_spatial_vs_feature_dist_colored_pseudo_time(args.figure_dir, feature_dir, coords, args.dataset)
