# -*- coding:utf-8 -*-
import os
import anndata
import json
import pandas as pd
import scanpy as sc
import squidpy as sq
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.config import get_args
from utils.util import mkdir, get_spatial_coords, get_squidpy_data, get_expr_name, SPATIAL_LIBD_DATASETS, SPATIAL_N_FEATURE_MAX, VISIUM_DATASETS
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import distance
from sklearn import metrics

def plt_setting(fontsz = 10):
    plt.rc('font', family='Arial', weight='bold')
    plt.rc('xtick', labelsize=fontsz)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=fontsz)  # fontsize of the tick labels

def plot_spatial_cord_with_pseudo_time(figure_dir, feature_dir, expr_name, spatial_cords, dataset, root_idx= 50, VASC=True, n_neighbors=10):
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
        args.spatial = spatial
        args.expr_name = expr_name
        name = get_expr_name(args)
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
        xs, ys = adata.obsm["X_umap"][:, 0], adata.obsm["X_umap"][:, 1]
        distances = np.sqrt(np.power((xs - xs.min()), 2) + np.power((ys - ys.min()), 2))
        adata.uns['iroot'] = np.argmin(distances)
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

def plot_spatial_vs_feature_dist_colored_pseudo_time(figure_dir, feature_dir, expr_name, spatial_cords, dataset, root_idx=50, n_neighbors=20, ncells=100):
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
        args.spatial = spatial
        args.expr_name = expr_name
        name = get_expr_name(args)
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
        xs, ys = adata.obsm["X_umap"][:, 0], adata.obsm["X_umap"][:, 1]
        distances = np.sqrt(np.power((xs - xs.min()), 2) + np.power((ys - ys.min()), 2))
        adata.uns['iroot'] = np.argmin(distances)
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

def plot_umap_clustering(figure_dir, feature_dir, expr_name, dataset, n_neighbors=10, deep_method="VASC"):
    fig_dir = os.path.join(figure_dir, dataset, "umap")
    mkdir(fig_dir)
    method_name = "_%s" % args.arch if args.arch != "VASC" else ""
    fig_fp = os.path.join(fig_dir, "%s_%s%s.pdf" % (dataset, expr_name, method_name))

    if os.path.exists(fig_fp):
        print("%s exist: pass" % fig_fp)
        return
    plt_setting()
    ncol = 2 if deep_method =="VASC" else 1
    fig, axs = plt.subplots(1, ncol, figsize=(4 * ncol, 4))
    plt.subplots_adjust(wspace=0.4, hspace=0.5, bottom=0.2)
    spatials = [False, True] if deep_method =="VASC" else [False]

    titles = ["VASC", "VASC + SP"] if deep_method =="VASC" else [deep_method]
    for sid, spatial in enumerate(spatials):
        ax = axs[sid] if ncol > 1 else axs
        title = titles[sid]
        args.spatial = spatial
        args.expr_name = expr_name
        args.arch = deep_method
        name = get_expr_name(args)
        feature_fp = os.path.join(feature_dir, "%s.tsv" % name)
        adata = sc.read_csv(feature_fp, delimiter="\t", first_column_names=None)

        # Neighbor Graph
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')
        sc.tl.leiden(adata)

        sc.tl.paga(adata)
        sc.pl.paga(adata, plot=False)  # remove `plot=False` if you want to see the coarse-grained graph

        sc.tl.umap(adata, init_pos='paga')
        sc.pl.umap(adata, color="leiden", ax=ax, show=False)
        ax.grid(False)

        ax.set_title("%s" % title, fontsize=12, weight='bold')

    plt.savefig(fig_fp, dpi=300)
    plt.close('all')

def plot_cluster_on_img(args, feature_dir, expr_name, resolution="1.0", clustering_method= "leiden", deep_method="VASC"):
    dataset_dir = args.dataset_dir
    dataset = args.dataset
    expr_dir = os.path.join(dataset_dir, dataset)
    method_name = "_%s" % args.arch if args.arch != "VASC" else ""
    fig_dir = os.path.join(args.figure_dir, dataset, "%s_clustering_resolution_%s" % (clustering_method, resolution))
    mkdir(fig_dir)
    fig_fp = os.path.join(fig_dir, "%s_cluster_on_img_%s%s.pdf" % (dataset, expr_name, method_name))
    # if os.path.exists(fig_fp):
    #     print("%s exist: pass" % fig_fp)
    #     return
    plt_setting()
    cm = plt.get_cmap('Set1')
    ncol = 4 if dataset in SPATIAL_LIBD_DATASETS else 3
    if deep_method != "VASC":
        ncol -= 1
    fig, axs = plt.subplots(1, ncol, figsize=(ncol * 4, 4))
    plt.subplots_adjust(wspace=0.4, hspace=0.5, bottom=0.2)
    ax = axs[0]
    ax.axis('off')

    if dataset in VISIUM_DATASETS:
        scale_factor_fp = os.path.join(expr_dir, "spatial", "scalefactors_json.json")
        with open(scale_factor_fp, "r") as json_file:
            data_dict = json.load(json_file)
            scale = data_dict["tissue_lowres_scalef"]
        adata = sc.datasets.visium_sge(dataset)
        spatial_cords = adata.obsm['spatial'].astype(float) * scale
        x, y = spatial_cords[:, 0], spatial_cords[:, 1]
        img = plt.imread(os.path.join(expr_dir, "spatial", "tissue_lowres_image.png"))
        ax.imshow(img)
    else:
        scale = 0.045
        coord_fp = os.path.join(expr_dir, "spatial_coords.csv")
        spatial_cords = pd.read_csv(coord_fp).values.astype(float) * scale
        x, y = spatial_cords[:, 1], spatial_cords[:, 0]

        info_df = pd.read_csv(os.path.join(expr_dir, "spot_info.csv"))
        clusters = info_df["layer_guess_reordered"].values.astype(str)

        library_id = dataset.split("_")[-1]

        if dataset in SPATIAL_LIBD_DATASETS:
            img = plt.imread(os.path.join(dataset_dir, dataset, "%s_tissue_lowres_image.png" % library_id))
            ax.imshow(img)
            ax.set_title("Histology of %s" % library_id, fontsize=12, weight='bold')
            ax = axs[1]
        img = plt.imread(os.path.join(dataset_dir, dataset, "%s_tissue_lowres_image.png" % library_id))
        ax.imshow(img)

        unique_clusters = np.unique(clusters)
        for cid, cluster in enumerate(unique_clusters[:-1]):
            color = cm(1. * cid / (len(unique_clusters) + 1))
            ind = clusters == cluster
            ax.scatter(x[ind], y[ind], s=1, color=color, label= cluster)
        ax.set_title("Ground Truth", fontsize=12, weight='bold')

    spatials = [False, True]  if deep_method =="VASC" else [False]
    titles = ["VASC", "VASC + SP"]  if deep_method =="VASC" else [deep_method]
    labels_dir = os.path.join(feature_dir, "%s_labels_resolution_%s" % (clustering_method, resolution))
    try:
        for sid, spatial in enumerate(spatials):
            args.spatial = spatial
            args.expr_name = expr_name
            args.arch = deep_method
            name = get_expr_name(args)
            label_fp = os.path.join(labels_dir, "%s_label.tsv" % name)
            if os.path.exists(label_fp):
                clusters = pd.read_csv(label_fp, header=None).values.astype(int)

                offset = 2 if dataset in SPATIAL_LIBD_DATASETS else 1
                ax = axs[sid + offset]
                ax.axis('off')
                ax.imshow(img)
                unique_clusters = np.unique(clusters)
                for cid, cluster in enumerate(unique_clusters[:-1]):
                    color = cm(1. * cid / (len(unique_clusters) + 1))
                    ind = (clusters == cluster).flatten()
                    ax.scatter(x[ind], y[ind], s=1, color=color, label= cluster)
                if dataset in SPATIAL_LIBD_DATASETS:
                    info_df = pd.read_csv(os.path.join(expr_dir, "spot_info.csv"))
                    ground_truth_clusters = info_df["layer_guess_reordered"].values.astype(str)
                    clusters = pd.read_csv(label_fp, header=None).values.flatten().astype(str)
                    ari = metrics.adjusted_rand_score(ground_truth_clusters, clusters)
                    ax.set_title("%s\n ARI: %.2f" % (titles[sid], ari), fontsize=12, weight='bold')
                else:
                    ax.set_title("%s" % titles[sid], fontsize=12, weight='bold')
            else:
                return
        for ax in axs:
            if dataset in ["V1_Adult_Mouse_Brain_Coronal_Section_1"]:
                ax.invert_xaxis()
        plt.savefig(fig_fp, dpi=300)
        plt.close('all')
    except IndexError as e:
        print(e)
    except ValueError as e:
        print(e)

def plot_pseudo_time_on_img(args,  feature_dir, expr_name, n_neighbors=10, root_idx= 50, umap_selected_root=False, deep_method="VASC"):
    dataset_dir = args.dataset_dir
    dataset = args.dataset
    expr_dir = os.path.join(dataset_dir, dataset)
    fig_dir = os.path.join(args.figure_dir, dataset, "pseudotime")
    mkdir(fig_dir)
    root_suffix = "_umap_based_root" if umap_selected_root else "root_%d" % root_idx
    method_name = "_%s" % deep_method if deep_method != "VASC" else ""
    fig_fp = os.path.join(fig_dir, "%s_pseudotime_on_img_%s%s_%s.pdf" % (dataset, expr_name, method_name, root_suffix))
    # if os.path.exists(fig_fp):
    #     print("%s exist: pass" % fig_fp)
    #     return
    plt_setting()
    cm = plt.get_cmap('Set1')
    ncol = 3 if dataset != "drosophila" and deep_method == "VASC" else 2
    fig, axs = plt.subplots(1, ncol, figsize=(ncol * 4, 4))
    plt.subplots_adjust(wspace=0.4, hspace=0.5, bottom=0.2)
    ax = axs[0]
    if dataset != "drosophila":
        ax.axis('off')

    if dataset in VISIUM_DATASETS:
        scale_factor_fp = os.path.join(expr_dir, "spatial", "scalefactors_json.json")
        with open(scale_factor_fp, "r") as json_file:
            data_dict = json.load(json_file)
            scale = data_dict["tissue_lowres_scalef"]
        adata = sc.datasets.visium_sge(dataset)
        spatial_cords = adata.obsm['spatial'].astype(float) * scale
        x, y = spatial_cords[:, 0], spatial_cords[:, 1]
        img = plt.imread(os.path.join(expr_dir, "spatial", "tissue_lowres_image.png"))
        ax.imshow(img)
    elif dataset == "drosophila":
        coord_fp = os.path.join(expr_dir, "spatial_pred.h5ad")
        spatial_adata = sc.read_h5ad(coord_fp)
        spatial_cords = spatial_adata.obsm['spatial']
        x, y = spatial_cords[:, 0], spatial_cords[:, 2]
    else:
        scale = 0.045
        coord_fp = os.path.join(expr_dir, "spatial_coords.csv")
        spatial_cords = pd.read_csv(coord_fp).values.astype(float) * scale
        x, y = spatial_cords[:, 1], spatial_cords[:, 0]

        info_df = pd.read_csv(os.path.join(expr_dir, "spot_info.csv"))
        clusters = info_df["layer_guess_reordered"].values.astype(str)

        library_id = dataset.split("_")[-1]
        img = plt.imread(os.path.join(dataset_dir, dataset, "%s_tissue_lowres_image.png" % library_id))
        ax.imshow(img)

        unique_clusters = np.unique(clusters)
        for cid, cluster in enumerate(unique_clusters[:-1]):
            color = cm(1. * cid / (len(unique_clusters) + 1))
            ind = clusters == cluster
            ax.scatter(x[ind], y[ind], s=1, color=color, label= cluster)
        ax.set_title("Ground Truth", fontsize=12, weight='bold')

    spatials = [False, True] if deep_method == "VASC" else [False]
    titles = ["VASC", "VASC + SP"]  if deep_method =="VASC" else [deep_method]

    for sid, spatial in enumerate(spatials):
        args.spatial = spatial
        args.expr_name = expr_name
        args.arch = deep_method
        name = get_expr_name(args)
        feature_fp = os.path.join(feature_dir, "%s.tsv" % name)
        if os.path.exists(feature_fp):
            adata = sc.read_csv(feature_fp, delimiter="\t", first_column_names=None)
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')
            sc.tl.umap(adata)
            if umap_selected_root:
                xs, ys = adata.obsm["X_umap"][:, 0], adata.obsm["X_umap"][:, 1]
                distances = np.sqrt(np.power((xs - xs.min()), 2) + np.power((ys - ys.min()), 2))
                root = np.argmin(distances)
            else:
                root = root_idx
            adata.uns['iroot'] = root
            sc.tl.diffmap(adata, n_comps=10)
            sc.tl.dpt(adata)
            offset = 1 if dataset != "drosophila" else 0
            ax = axs[sid + offset]
            if dataset != "drosophila":
                ax.imshow(img)
                ax.axis('off')
            else:
                ax.set_aspect('equal', 'box')
                ax.set_xlim([-200, 200])
                ax.set_ylim([-100, 100])
            ax.grid(False)
            try:
                st = ax.scatter(x, y, s=1, c=adata.obs['dpt_pseudotime'])
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                clb = fig.colorbar(st, cax=cax)
                clb.ax.set_ylabel("pseudotime", labelpad=10, rotation=270, fontsize=8, weight='bold')
                ax.set_title("%s" % titles[sid], fontsize=12, weight='bold')
            except ValueError as e:
                print(e)
        else:
            return

    # for ax in axs:
    #     #if dataset in ["V1_Adult_Mouse_Brain_Coronal_Section_1"]:
    #     ax.invert_xaxis()
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')

if __name__ == "__main__":
    mpl.use('macosx')
    args = get_args()
    datasets = SPATIAL_LIBD_DATASETS #+ VISIUM_DATASETS #["drosophila"]#
    expr_names = ["default"]
    # ["5_penalty1", "50_penalty1", "500_penalty1", "5_penalty1_2_panelty2", "50_penalty1_20_panelty2",
    #  "500_penalty1_200_panelty2"]
    # expr_names = ["-100_penalty2", "-500_penalty2", "-1000_penalty2", "200_penalty1", "500_penalty1", "1000_penalty1", "500_penalty1_100_penalty2", "500_penalty1_-100_penalty2", "500_penalty1_-50_penalty2", "500_penalty1_200_penalty2"]
    # expr_names = ["L2_wt_KLD"]#["5_penalty1", "50_penalty1", "500_penalty1",  "5_penalty1_2_panelty2", "50_penalty1_20_panelty2", "500_penalty1_200_panelty2",]#["500_penalty1_200_penalty2", "500_penalty1_-200_penalty2" , "500_penalty1_100_penalty3", "500_penalty1_-100_penalty3", "500_penalty1_200_penalty2_-100_penalty3", "500_penalty1_200_penalty2_-250_penalty3_50p22_spc_0.35_ftf_0.6", "500_penalty1_200_penalty2_-250_penalty3_50p22_spc_0.25_ftf_0.6", "500_penalty1_200_penalty2_-250_penalty3_50p22_spc_0.25_ftf_0.6_ftc_0.25_-100p33"]#["500_penalty1"]##["500_penalty1_-100_penalty3", "500_penalty1_100_penalty3"]#["-100p11", "-100p22", "-100p33"]#["500_penalty1_-50_penalty2", "500_penalty1_-100_penalty2", "500_penalty1_100_penalty2"]
    # expr_names = ["-100_penalty1", "-100_penalty2", "-100_penalty3", "-500_penalty1", "-500_penalty2", "-500_penalty3",
    #               "-1000_penalty1", "-1000_penalty2", "-1000_penalty3", "100_penalty1",
    #               "500_penalty1_200_penalty2_-100_penalty3", "500_penalty1_200_penalty2_-250_penalty3",
    #               "500_penalty1_200_penalty2_100_penalty3", "500_penalty1_200_penalty2_250_penalty3",
    #               "500_penalty1_200_penalty2", "500_penalty1", "500_penalty2", "500_penalty3", "1000_penalty1",
    #               "500_penalty1_200_penalty2_-250_penalty3_20p11_50p22_spc_0.25_ftf_0.6_ftc_0.25",
    #               "500_penalty1_200_penalty2_-250_penalty3_-20p11_50p22_spc_0.25_ftf_0.6_ftc_0.25",
    #               "500_penalty1_200_penalty2_-250_penalty3_50p22_spc_0.25_ftf_0.6_ftc_0.25_-200p33",
    #               "500_penalty1_200_penalty2_-250_penalty3_50p22_spc_0.25_ftf_0.6_ftc_0.25_-50p33",
    #               "500_penalty1_200_penalty2_-250_penalty3_50p22_spc_0.25_ftf_0.6_ftc_0.25_-100p33",
    #               "500_penalty1_200_penalty2_-250_penalty3_-50p33", "500_penalty1_200_penalty2_-250_penalty3_50p33",
    #               "500_penalty1_200_penalty2_-250_penalty3_50p22_spc_0.35_ftf_0.5",
    #               "500_penalty1_200_penalty2_-250_penalty3_50p22_spc_0.25_ftf_0.5",
    #               "500_penalty1_200_penalty2_-250_penalty3_50p22_spc_0.35_ftf_0.6",
    #               "500_penalty1_200_penalty2_-250_penalty3_1000p33", "500_penalty1_200_penalty2_-250_penalty3_500p33",
    #               "500_penalty1_200_penalty2_-250_penalty3_200p11",
    #               "500_penalty1_200_penalty2_-250_penalty3_50p22_spc_0.25_ftf_0.7",
    #               "500_penalty1_200_penalty2_-250_penalty3_50p22_spc_0.25_ftf_0.6",
    #               "500_penalty1_200_penalty2_-250_penalty3_-500p33_spc_0.2_ftc_0.35",
    #               "500_penalty1_200_penalty2_-250_penalty3_-100p33", "500_penalty1_200_penalty2_-250_penalty3_-1000p33",
    #               "500_penalty1_200_penalty2_-250_penalty3_-5000p33", "500_penalty1_200_penalty2_-250_penalty3_20p22",
    #               "500_penalty1_200_penalty2_-250_penalty3_100p22", "500_penalty1_200_penalty2_-250_penalty3_200p22",
    #               "500_penalty1_200_penalty2_-250_penalty3_-500p33", "500_penalty1_200_penalty2_-250_penalty3_100p11",
    #               "500_penalty1_200_penalty2_-250_penalty3_50p22",
    #               "500_penalty1_200_penalty2_-250_penalty3_50p11_10p22_-500p33",
    #               "500_penalty1_200_penalty2_-250_penalty3_50p11_20p22_-1000p33",
    #               "500_penalty1_200_penalty2_-250_penalty3_100p11_20p22_-1000p33",
    #               "500_penalty1_200_penalty2_-1000_penalty3", "500_penalty1_200_penalty2_-600_penalty3",
    #               "500_penalty1_200_penalty2_50_penalty3", "500_penalty1_100_penalty2_50_penalty3",
    #               "500_penalty1_100_penalty2_-50_penalty3", "500_penalty1_-100_penalty2_-50_penalty3",
    #               "500_penalty1_-100_penalty2_-100_penalty3", "500_penalty1_100_penalty2_-100_penalty3",
    #               "500_penalty1_-50_penalty2_-250_penalty3", "500_penalty1_-100_penalty2_-250_penalty3",
    #               "500_penalty1_-100_penalty2_-500_penalty3"]
    deep_methods = ['DGI']#, 'DGI', 'VGAE']
    resolutions = ["0.7", "0.6", "0.5", "0.4"]#, "0.3", "0.2", "0.1"]
    clustering_methods = ["leiden"]  # , "louvain"
    UMAP_based_selections = [True]#, False
    n_neighbors = 50
    for deep_method in deep_methods:
        for dataset in datasets:
            args.dataset = dataset
            for expr_name in expr_names:
                for umap_based_selection in UMAP_based_selections:
                    coords = []#get_spatial_coords(args)
                    feature_dir = os.path.join(args.dataset_dir, "features")
                    # plot_spatial_cord_with_pseudo_time(args.figure_dir, feature_dir, expr_name, coords, args.dataset)
                    # plot_spatial_vs_feature_dist_colored_pseudo_time(args.figure_dir, feature_dir, expr_name, coords, args.dataset)
                    # plot_umap_clustering(args.figure_dir, feature_dir, expr_name,  args.dataset, deep_method=deep_method)
                    #plot_pseudo_time_on_img(args, feature_dir, expr_name, umap_selected_root=umap_based_selection, deep_method=deep_method, n_neighbors=n_neighbors)
                    for resolution in resolutions:
                        for method in clustering_methods:
                            plot_cluster_on_img(args, feature_dir, expr_name, resolution=resolution, clustering_method=method, deep_method=deep_method)
