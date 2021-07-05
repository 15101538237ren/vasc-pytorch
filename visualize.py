# -*- coding:utf-8 -*-
import os
import anndata
import pandas as pd
import scanpy as sc
import squidpy as sq
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.config import get_args
from utils.util import mkdir, get_spatial_coords, get_squidpy_data, SPATIAL_LIBD_DATASETS, SPATIAL_N_FEATURE_MAX, SQUIDPY_DATASETS
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

def plot_umap_clustering(figure_dir, feature_dir, spatial_cords, dataset, n_neighbors=10, linear=True):
    fig_dir = os.path.join(figure_dir, dataset)
    mkdir(fig_dir)

    plt_setting()
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    plt.subplots_adjust(wspace=0.4, hspace=0.5, bottom=0.2)
    spatials = [False, True]

    titles = ["VASC", "VASC + SP"]
    for sid, spatial in enumerate(spatials):
        ax = axs[sid]
        name = args.dataset if not spatial else "%s_with_spatial" % dataset
        title = titles[sid]
        feature_fp = os.path.join(feature_dir, "%s.tsv" % name)
        adata = sc.read_csv(feature_fp, delimiter="\t", first_column_names=None)

        # Neighbor Graph
        sc.pp.neighbors(adata, n_neighbors=n_neighbors)
        sc.tl.leiden(adata)

        sc.tl.paga(adata)
        sc.pl.paga(adata, plot=False)  # remove `plot=False` if you want to see the coarse-grained graph

        sc.tl.umap(adata, init_pos='paga')
        sc.pl.umap(adata, color="leiden", ax=ax, show=False)
        ax.grid(False)

        ax.set_title("%s" % title, fontsize=12, weight='bold')

    suffix = "linear" if linear else "switch"
    fig_fp = os.path.join(fig_dir, "%s_%s.pdf" % (dataset, suffix))
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')

def plot_cluster_on_img(args,  feature_dir, spatial_cords, dataset, clustering_method="leiden", linear=True, scale= 0.045):
    dataset_dir = args.dataset_dir
    dataset = args.dataset
    expr_dir = os.path.join(dataset_dir, dataset)

    coord_fp = os.path.join(expr_dir, "spatial_coords.csv")
    spatial_cords = pd.read_csv(coord_fp).values.astype(float) * scale


    info_df = pd.read_csv(os.path.join(expr_dir, "spot_info.csv"))
    clusters = info_df["layer_guess_reordered"].values.astype(str)
    # SpatialDE_PCA_clusters = info_df["SpatialDE_UMAP"].values.astype(str)

    library_id = dataset.split("_")[-1]
    fig_dir = os.path.join(args.figure_dir, dataset)
    mkdir(fig_dir)
    plt_setting()

    cm = plt.get_cmap('Set1')
    img = plt.imread(os.path.join(dataset_dir, dataset, "%s_tissue_lowres_image.png" % library_id))
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    plt.subplots_adjust(wspace=0.4, hspace=0.5, bottom=0.2)
    ax = axs[0]
    ax.axis('off')
    ax.imshow(img)
    unique_clusters = np.unique(clusters)
    for cid, cluster in enumerate(unique_clusters[:-1]):
        color = cm(1. * cid / (len(unique_clusters) + 1))
        ind = clusters == cluster
        ax.scatter(spatial_cords[ind, 1], spatial_cords[ind, 0], s=1, color=color, label= cluster)
    ax.set_title("Ground Truth", fontsize=12, weight='bold')

    spatials = [False, True]
    titles = ["VASC", "VASC + SP"]
    labels_dir = os.path.join(feature_dir, "cluster_labels")
    for sid, spatial in enumerate(spatials):
        name = args.dataset if not spatial else "%s_with_spatial" % dataset
        label_fp = os.path.join(labels_dir, "%s_%s_label.tsv" % (name, clustering_method))
        clusters = pd.read_csv(label_fp, header=None).values.astype(int)

        ax = axs[sid + 1]
        ax.axis('off')
        ax.imshow(img)
        unique_clusters = np.unique(clusters)
        for cid, cluster in enumerate(unique_clusters[:-1]):
            color = cm(1. * cid / (len(unique_clusters) + 1))
            ind = (clusters == cluster).flatten()
            ax.scatter(spatial_cords[ind, 1], spatial_cords[ind, 0], s=1, color=color, label= cluster)
        ax.set_title("%s" % titles[sid], fontsize=12, weight='bold')

    suffix = "linear" if linear else "switch"
    fig_fp = os.path.join(fig_dir, "%s_cluster_on_img_%s.pdf" % (dataset, suffix))
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')

def plot_pseudo_time_on_img(args,  feature_dir, spatial_cords, dataset, clustering_method="leiden", linear=True, scale= 0.045, n_neighbors=10, root_idx= 50):
    dataset_dir = args.dataset_dir
    dataset = args.dataset
    expr_dir = os.path.join(dataset_dir, dataset)

    coord_fp = os.path.join(expr_dir, "spatial_coords.csv")
    spatial_cords = pd.read_csv(coord_fp).values.astype(float) * scale

    info_df = pd.read_csv(os.path.join(expr_dir, "spot_info.csv"))
    clusters = info_df["layer_guess_reordered"].values.astype(str)
    # SpatialDE_PCA_clusters = info_df["SpatialDE_UMAP"].values.astype(str)

    library_id = dataset.split("_")[-1]
    fig_dir = os.path.join(args.figure_dir, dataset)
    mkdir(fig_dir)
    plt_setting()

    cm = plt.get_cmap('Set1')
    img = plt.imread(os.path.join(dataset_dir, dataset, "%s_tissue_lowres_image.png" % library_id))
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    plt.subplots_adjust(wspace=0.4, hspace=0.5, bottom=0.2)
    ax = axs[0]
    ax.axis('off')
    ax.imshow(img)
    unique_clusters = np.unique(clusters)
    for cid, cluster in enumerate(unique_clusters[:-1]):
        color = cm(1. * cid / (len(unique_clusters) + 1))
        ind = clusters == cluster
        ax.scatter(spatial_cords[ind, 1], spatial_cords[ind, 0], s=1, color=color, label= cluster)
    ax.set_title("Ground Truth", fontsize=12, weight='bold')

    spatials = [False, True]
    titles = ["VASC", "VASC + SP"]
    for sid, spatial in enumerate(spatials):
        name = args.dataset if not spatial else "%s_with_spatial" % dataset
        # Neighbor Graph
        feature_fp = os.path.join(feature_dir, "%s.tsv" % name)
        adata = sc.read_csv(feature_fp, delimiter="\t", first_column_names=None)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors)
        sc.tl.umap(adata)
        adata.uns['iroot'] = root_idx
        sc.tl.dpt(adata)

        ax = axs[sid + 1]
        ax.axis('off')
        ax.imshow(img)
        ax.grid(False)

        st = ax.scatter(spatial_cords[:, 1], spatial_cords[:, 0], s=1, c=adata.obs['dpt_pseudotime'])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        clb = fig.colorbar(st, cax=cax)
        clb.ax.set_ylabel("pseudotime", labelpad=10, rotation=270, fontsize=8, weight='bold')
        ax.set_title("%s" % titles[sid], fontsize=12, weight='bold')

    suffix = "linear" if linear else "switch"
    fig_fp = os.path.join(fig_dir, "%s_pseudotime_on_img_%s.pdf" % (dataset, suffix))
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')


if __name__ == "__main__":
    mpl.use('macosx')
    args = get_args()
    linears = [True, False]#
    datasets = SPATIAL_LIBD_DATASETS
    for linear in linears:
        for dataset in datasets:
            args.dataset = dataset
            coords = []#get_spatial_coords(args)
            feature_suff = "features_linear" if linear else "features_switch"
            feature_dir = os.path.join(args.dataset_dir, feature_suff)
            #plot_spatial_cord_with_pseudo_time(args.figure_dir, feature_dir, coords, args.dataset)
            #plot_spatial_vs_feature_dist_colored_pseudo_time(args.figure_dir, feature_dir, coords, args.dataset)
            #plot_umap_clustering(args.figure_dir, feature_dir, coords, args.dataset, linear=linear)
            plot_cluster_on_img(args, feature_dir, coords, args.dataset, linear=linear)
            plot_pseudo_time_on_img(args, feature_dir, coords, args.dataset, linear=linear)
