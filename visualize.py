# -*- coding:utf-8 -*-
import os
import anndata
import loompy
import json
import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.config import get_args
from utils.util import mkdir, get_squidpy_data, get_expr_name, SPATIAL_LIBD_DATASETS, SPATIAL_N_FEATURE_MAX, VISIUM_DATASETS, SQUIDPY_DATASETS
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import distance, distance_matrix
from sklearn import metrics
from scipy.sparse.linalg.eigen.arpack.arpack import ArpackNoConvergence
from scipy.stats import norm as normal

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

        distances = distance_matrix(adata.X, adata.X)
        adata.uns['iroot'] = np.argmax(distances.sum(axis=1))
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

        distances = distance_matrix(adata.X, adata.X)
        adata.uns['iroot'] = np.argmax(distances.sum(axis=1))
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

    plt_setting()
    if deep_method == "VASC":
        spatials = [False, True]
        ncol = 2
    elif expr_name != "default":
        spatials = [True]
        ncol = 1
    else:
        spatials = [False]
        ncol = 1
    fig, axs = plt.subplots(1, ncol, figsize=(4 * ncol, 4))
    plt.subplots_adjust(wspace=0.4, hspace=0.5, bottom=0.2)

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

def plot_cluster_on_img(args, feature_dir, expr_name, clustering_method= "leiden", deep_method="VASC", n_neighbors=100):
    dataset_dir = args.dataset_dir
    dataset = args.dataset
    expr_dir = os.path.join(dataset_dir, dataset)
    method_name = deep_method#"_%s" % args.arch if args.arch != "VASC" else ""
    fig_dir = os.path.join(args.figure_dir, dataset, "%s_clustering" % (clustering_method))
    mkdir(fig_dir)
    fig_fp = os.path.join(fig_dir, "%s_cluster_on_img_%s%s_nneighbor_%d.pdf" % (dataset, expr_name, method_name, n_neighbor))
    print("Dealing with %s" % fig_fp)
    # if os.path.exists(fig_fp):
    #     print("%s exist: pass" % fig_fp)
    #     return
    plt_setting()
    cm = plt.get_cmap('Set1')
    ncol = 4 if dataset in SPATIAL_LIBD_DATASETS else 3
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
    elif dataset in SQUIDPY_DATASETS:
        adata = get_squidpy_data(dataset)
        coords = adata.obsm['spatial']
        annotation_dict = {"imc": "cell type", "seqfish": "celltype_mapped_refined", "slideseqv2": "cluster"}
        cell_types = adata.obs[annotation_dict[dataset]]
        n_cells = coords.shape[0]
        if args.max_cells < n_cells:
            expr_dir = os.path.join(dataset_dir, dataset)
            mkdir(expr_dir)
            indices_fp = os.path.join(expr_dir, "indices.npy")
            if os.path.exists(indices_fp):
                with open(indices_fp, 'rb') as f:
                    indices = np.load(f)
                    print("loaded indices successful!")
            x, y = coords[:, 0].take(indices), coords[:, 1].take(indices)
            cell_types = cell_types.take(indices)
        else:
            x, y = coords[:, 0], coords[:, 1]
        cell_type_strs = cell_types.cat.categories.astype(str)
        cell_type_ints = cell_types.values.codes
        cell_type_colors = list(adata.uns['%s_colors' % annotation_dict[dataset]].astype(str))
        colors = np.array([cell_type_colors[item] for item in cell_type_ints])
        for cid in range(len(cell_type_colors)):
            cit = cell_type_ints == cid
            ax.scatter(x[cit], y[cit], s=1, c=colors[cit], label=cell_type_strs[cid])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(cell_type_strs, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})
        ax.set_title("Annotations", fontsize=12, weight='bold')
        ax.set_xlabel("Spatial1", fontsize=8, weight='bold')
        ax.set_ylabel("Spatial2", fontsize=8, weight='bold')
        ax.axis('off')
        if dataset in SQUIDPY_DATASETS:
            ax.invert_yaxis()
    elif dataset in SPATIAL_LIBD_DATASETS:
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
        for cid, cluster in enumerate(unique_clusters):
            color = cm(1. * cid / (len(unique_clusters) + 1))
            ind = clusters == cluster
            ax.scatter(x[ind], y[ind], s=1, color=color, label= cluster)
        ax.set_title("Ground Truth", fontsize=12, weight='bold')
    else:
        region_colors = {
            "Layer 6": '#027fd0',
            "Pia Layer 1": '#de4726',
            "Layer 4": '#ffbb19',
            "Layer 3-4": '#8a55ff',
            "Layer 2-3 medial": '#28d1eb',
            "Hippocampus": '#004b71',
            "Internal Capsule Caudoputamen": '#9b067d',
            "White matter": '#4ec030',
            "Ventricle": '#fadf0c',
            "Layer 2-3 lateral": '#28a5eb',
            "Layer 5": '#d61420',
        }
        expr_dir = os.path.join(dataset_dir, dataset)
        expr_fp = os.path.join(expr_dir, "osmFISH_SScortex_mouse_all_cells.loom")
        ds = loompy.connect(expr_fp)

        # Make Panda's Dataframe with count data
        df_osmfish = pd.DataFrame(data=ds[:, :], columns=ds.ca['CellID'], index=ds.ra['Gene']).astype(int)

        # Include cells:
        # with more than 0 molecules
        include_mol = ds.ca.Total_molecules >= 0
        # larger than 25 um2
        include_size_min = ds.ca.size_um2 >= 5
        # smaller than 272 um2
        include_size_max = ds.ca.size_um2 <= 275
        # Not in the double imaged region
        include_position = ds.ca.X >= 0

        include = np.logical_and(np.logical_and(include_mol, include_size_min),
                                 np.logical_and(include_size_max, include_position))

        # include = np.logical_and(np.logical_and(ds.ca.Total_molecules > 20, ds.ca.size_pix > 2000), ds.ca.size_pix < 60000)
        df_osmfish = df_osmfish.loc[:, include]

        # normalize data by total number of molecules per cell and per gene.
        # df_osmfish_totmol = df_osmfish.divide(df_osmfish.sum(axis=1), axis=0) * df_osmfish.shape[
        #     0]  # Corrected for total molecules per gene
        df_osmfish_totmol = df_osmfish.divide(df_osmfish.sum(axis=0), axis=1) * df_osmfish.shape[
            1]  # Corrected for the total per cell
        # Replace NA and Nan with zero:
        df_osmfish_totmol = df_osmfish_totmol.fillna(0)
        non_zero_count_rows = np.where(df_osmfish_totmol.values.T.sum(axis=1) > 0)[0]

        # Load the cell coordinates into a Pandas Dataframe. Units are pixels
        coordinates = np.stack((ds.ca.X, ds.ca.Y))
        df_coordinates = pd.DataFrame(data=coordinates, index=['X', 'Y'], columns=ds.ca.CellID)
        df_coordinates = df_coordinates.loc[:, include].values.T
        df_coordinates = df_coordinates[non_zero_count_rows, :]
        x, y = df_coordinates[:, 0], df_coordinates[:, 1]
        ground_truth_clusters = ds.ca.Region[include]
        ground_truth_clusters = ground_truth_clusters[non_zero_count_rows]
        uniques = np.unique(ground_truth_clusters)
        for uniq in uniques:
            if uniq != "Excluded":
                idxs = ground_truth_clusters == uniq
                label = uniq
                if label == "Internal Capsule Caudoputamen":
                    label = "IC CP"
                elif label == "Layer 2-3 lateral":
                    label = "Layer 2-3 lat"
                elif label == "Layer 2-3 medial":
                    label = "Layer 2-3 med"
                ax.scatter(x[idxs], y[idxs], s=1,
                           color=region_colors[uniq], cmap='spectral', label= label)
        ax.set_aspect('equal')
        ax.set_axis_off()

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})

    spatials = [False, True]
    titles = ["%s SP-" % deep_method, "%s SP+" % deep_method]
    labels_dir = os.path.join(feature_dir, "%s_labels" % clustering_method)
    try:
        for sid, spatial in enumerate(spatials):
            args.spatial = spatial
            args.expr_name = expr_name if spatial else "default"
            args.arch = deep_method
            name = get_expr_name(args)
            label_fp = os.path.join(labels_dir, "%s_label_nNeighbor_%d.tsv" % (name, n_neighbors))
            if os.path.exists(label_fp):
                clusters = pd.read_csv(label_fp, header=None).values.astype(int)

                offset = 2 if dataset in SPATIAL_LIBD_DATASETS else 1
                ax = axs[sid + offset]
                ax.axis('off')
                if dataset in SPATIAL_LIBD_DATASETS + VISIUM_DATASETS:
                    ax.imshow(img)
                unique_clusters = np.unique(clusters)
                for cid, cluster in enumerate(unique_clusters):
                    color = cm(1. * cid / (len(unique_clusters) + 1))
                    ind = (clusters == cluster).flatten()
                    ax.scatter(x[ind], y[ind], s=1, color=color, label= cluster)
                if dataset in SPATIAL_LIBD_DATASETS:
                    info_df = pd.read_csv(os.path.join(expr_dir, "spot_info.csv"))
                    ground_truth_clusters = info_df["layer_guess_reordered"].values.astype(str)
                    clusters = pd.read_csv(label_fp, header=None).values.flatten().astype(str)
                    ari = metrics.adjusted_rand_score(ground_truth_clusters, clusters)
                    ax.set_title("%s\n ARI: %.2f" % (titles[sid], ari), fontsize=12, weight='bold')
                elif dataset == "osmFISH":
                    clusters = pd.read_csv(label_fp, header=None).values.flatten().astype(str)
                    ari = metrics.adjusted_rand_score(ground_truth_clusters, clusters)
                    ax.set_title("%s\n ARI: %.2f" % (titles[sid], ari), fontsize=12, weight='bold')
                else:
                    ax.set_title("%s" % titles[sid], fontsize=12, weight='bold')
                if dataset in SQUIDPY_DATASETS:
                    ax.invert_yaxis()
            else:
                return

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
    fig_fp = os.path.join(fig_dir, "%s_pseudotime_on_img_%s%s_%s_nneighbor_%d.pdf" % (dataset, expr_name, method_name, root_suffix, n_neighbors))
    # if os.path.exists(fig_fp):
    #     print("%s exist: pass" % fig_fp)
    #     return
    plt_setting()
    cm = plt.get_cmap('gist_rainbow')
    ncol = 2 if (dataset == "drosophila" or dataset in SQUIDPY_DATASETS) else 3
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
    elif dataset in SQUIDPY_DATASETS:
        adata = get_squidpy_data(dataset)
        coords = adata.obsm['spatial']
        annotation_dict = {"imc": "cell type", "seqfish": "celltype_mapped_refined", "slideseqv2": "cluster"}
        cell_types = adata.obs[annotation_dict[dataset]]
        n_cells = coords.shape[0]
        if args.max_cells < n_cells:
            expr_dir = os.path.join(dataset_dir, dataset)
            mkdir(expr_dir)
            indices_fp = os.path.join(expr_dir, "indices.npy")
            if os.path.exists(indices_fp):
                with open(indices_fp, 'rb') as f:
                    indices = np.load(f)
                    print("loaded indices successful!")
            x, y = coords[:, 0].take(indices), coords[:, 1].take(indices)
            cell_types = cell_types.take(indices)
        else:
            x, y = coords[:, 0], coords[:, 1]
        cell_type_strs = cell_types.cat.categories.astype(str)
        cell_type_ints = cell_types.values.codes
        cell_type_colors = list(adata.uns['%s_colors' % annotation_dict[dataset]].astype(str))
        colors = np.array([cell_type_colors[item] for item in cell_type_ints])
        for cid in range(len(cell_type_colors)):
            cit = cell_type_ints == cid
            ax.scatter(x[cit], y[cit], s=1, c=colors[cit], label=cell_type_strs[cid])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(cell_type_strs, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})
        ax.set_title("Annotations", fontsize=12, weight='bold')
        ax.set_xlabel("Spatial1", fontsize=8, weight='bold')
        ax.set_ylabel("Spatial2", fontsize=8, weight='bold')
        ax.axis('off')
        if dataset in SQUIDPY_DATASETS:
            ax.invert_yaxis()
    elif dataset == "osmFISH":
        region_colors = {
            "Layer 6": '#027fd0',
            "Pia Layer 1": '#de4726',
            "Layer 4": '#ffbb19',
            "Layer 3-4": '#8a55ff',
            "Layer 2-3 medial": '#28d1eb',
            "Hippocampus": '#004b71',
            "Internal Capsule Caudoputamen": '#9b067d',
            "White matter": '#4ec030',
            "Ventricle": '#fadf0c',
            "Layer 2-3 lateral": '#28a5eb',
            "Layer 5": '#d61420',
        }
        expr_dir = os.path.join(dataset_dir, dataset)
        expr_fp = os.path.join(expr_dir, "osmFISH_SScortex_mouse_all_cells.loom")
        ds = loompy.connect(expr_fp)

        # Make Panda's Dataframe with count data
        df_osmfish = pd.DataFrame(data=ds[:, :], columns=ds.ca['CellID'], index=ds.ra['Gene']).astype(int)

        # Include cells:
        # with more than 0 molecules
        include_mol = ds.ca.Total_molecules >= 0
        # larger than 25 um2
        include_size_min = ds.ca.size_um2 >= 5
        # smaller than 272 um2
        include_size_max = ds.ca.size_um2 <= 275
        # Not in the double imaged region
        include_position = ds.ca.X >= 0

        include = np.logical_and(np.logical_and(include_mol, include_size_min),
                                 np.logical_and(include_size_max, include_position))

        # include = np.logical_and(np.logical_and(ds.ca.Total_molecules > 20, ds.ca.size_pix > 2000), ds.ca.size_pix < 60000)
        df_osmfish = df_osmfish.loc[:, include]

        # normalize data by total number of molecules per cell and per gene.
        # df_osmfish_totmol = df_osmfish.divide(df_osmfish.sum(axis=1), axis=0) * df_osmfish.shape[
        #     0]  # Corrected for total molecules per gene
        df_osmfish_totmol = df_osmfish.divide(df_osmfish.sum(axis=0), axis=1) * df_osmfish.shape[
            1]  # Corrected for the total per cell
        # Replace NA and Nan with zero:
        df_osmfish_totmol = df_osmfish_totmol.fillna(0)
        non_zero_count_rows = np.where(df_osmfish_totmol.values.T.sum(axis=1) > 0)[0]

        # Load the cell coordinates into a Pandas Dataframe. Units are pixels
        coordinates = np.stack((ds.ca.X, ds.ca.Y))
        df_coordinates = pd.DataFrame(data=coordinates, index=['X', 'Y'], columns=ds.ca.CellID)
        df_coordinates = df_coordinates.loc[:, include].values.T
        df_coordinates = df_coordinates[non_zero_count_rows, :]
        x, y = df_coordinates[:, 0], df_coordinates[:, 1]
        ground_truth_clusters = ds.ca.Region[include]
        ground_truth_clusters = ground_truth_clusters[non_zero_count_rows]
        uniques = np.unique(ground_truth_clusters)
        for uniq in uniques:
            if uniq != "Excluded":
                label = uniq
                if label == "Internal Capsule Caudoputamen":
                    label = "IC CP"
                elif label == "Layer 2-3 lateral":
                    label = "Layer 2-3 lat"
                elif label == "Layer 2-3 medial":
                    label = "Layer 2-3 med"
                idxs = ground_truth_clusters == uniq
                ax.scatter(x[idxs], y[idxs], s=1,
                           color=region_colors[uniq], cmap='spectral', label=label)
        ax.set_aspect('equal')
        ax.set_axis_off()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})
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

    spatials = [False, True]
    titles = ["%s SP-" % deep_method, "%s SP+" % deep_method]

    for sid, spatial in enumerate(spatials):
        args.spatial = spatial
        args.expr_name = expr_name if spatial else "default"
        args.arch = deep_method
        name = get_expr_name(args)
        feature_fp = os.path.join(feature_dir, "%s.tsv" % name)
        if os.path.exists(feature_fp):
            print("Readed %s" % feature_fp)
            adata = sc.read_csv(feature_fp, delimiter="\t", first_column_names=None)
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')
            sc.tl.umap(adata)
            sc.tl.leiden(adata, resolution=.8)
            sc.tl.paga(adata)
            distances = distance_matrix(adata.X, adata.X)
            adata.uns['iroot'] = np.argmax(distances.sum(axis=1))
            try:
                sc.tl.diffmap(adata)
                sc.tl.dpt(adata)
                offset = 1 if dataset not in ["drosophila"] + SQUIDPY_DATASETS else 0
                ax = axs[sid + offset]
                if dataset != "drosophila":
                    ax.axis('off')
                else:
                    ax.set_aspect('equal', 'box')
                    ax.set_xlim([-200, 200])
                    ax.set_ylim([-100, 100])
                if dataset not in ["drosophila", "osmFISH"] + SQUIDPY_DATASETS:
                    ax.imshow(img)
                ax.grid(False)
                st = ax.scatter(x, y, s=1, c=adata.obs['dpt_pseudotime'], cmap=cm)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                clb = fig.colorbar(st, cax=cax)
                clb.ax.set_ylabel("pseudotime", labelpad=10, rotation=270, fontsize=8, weight='bold')
                ax.set_title("%s" % titles[sid], fontsize=12, weight='bold')
                if dataset in SQUIDPY_DATASETS:
                   ax.invert_yaxis()
            except ValueError as e:
                print(e)
            except ArpackNoConvergence as e:
                print(e)
        else:
            return

    # for ax in axs:
    #     #if dataset in ["V1_Adult_Mouse_Brain_Coronal_Section_1"]:
    #     ax.invert_xaxis()
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')
    print("Plotted %s" % fig_fp)


def sparsify_transition_matrix(pi_matrix, topN = 10):
    n = pi_matrix.shape[0]
    sparsed_matrix = np.zeros((n, n))
    for i in range(n):
        ind = pi_matrix[i,:].argsort()[-(topN+1):][::-1][1:]
        sparsed_matrix[i, ind] = pi_matrix[i, ind]

    sum_of_rows = sparsed_matrix.sum(axis=1)
    normalized_sparsed_matrix = sparsed_matrix / sum_of_rows[:, np.newaxis]
    return normalized_sparsed_matrix

def get_displacement_matrix(cords):
    n = cords.shape[0]
    x_displacement = np.zeros((n, n))
    y_displacement = np.zeros((n, n))
    for i in range(1, n):
        for j in range(i):
            disps = cords[j, :] - cords[i, :]
            R = np.linalg.norm(disps)
            x_disp, y_disp = disps/R
            x_displacement[i, j] = x_disp
            x_displacement[j, i] = x_disp
            y_displacement[i, j] = y_disp
            y_displacement[j, i] = y_disp
    return x_displacement, y_displacement

def get_weighted_displacements(displacements, sparsed_pi_mat):
    n = displacements.shape[0]
    disp = [0.0 for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if j != i and sparsed_pi_mat[i, j] > 1e-5:
                disp[i] += sparsed_pi_mat[i, j] * displacements[i, j]
    return np.array(disp)

def quiver_autoscale(X_emb, V_emb):
    import matplotlib.pyplot as pl

    scale_factor = np.abs(X_emb).max()  # just so that it handles very large values
    fig, ax = pl.subplots()
    Q = ax.quiver(
        X_emb[:, 0] / scale_factor,
        X_emb[:, 1] / scale_factor,
        V_emb[:, 0],
        V_emb[:, 1],
        angles="xy",
        scale_units="xy",
        scale=None,
    )
    Q._init()
    fig.clf()
    pl.close(fig)
    return Q.scale / scale_factor

def compute_velocity_on_grid(
    X_emb,
    V_emb,
    density=None,
    smooth=None,
    n_neighbors=None,
    min_mass=None,
    autoscale=True,
    adjust_for_stream=False,
    cutoff_perc=None,
):
    # remove invalid cells
    idx_valid = np.isfinite(X_emb.sum(1) + V_emb.sum(1))
    X_emb = X_emb[idx_valid]
    V_emb = V_emb[idx_valid]

    # prepare grid
    n_obs, n_dim = X_emb.shape
    density = 1 if density is None else density
    smooth = 0.5 if smooth is None else smooth

    grs = []
    for dim_i in range(n_dim):
        m, M = np.min(X_emb[:, dim_i]), np.max(X_emb[:, dim_i])
        m = m - 0.01 * np.abs(M - m)
        M = M + 0.01 * np.abs(M - m)
        gr = np.linspace(m, M, int(50 * density))
        grs.append(gr)

    meshes_tuple = np.meshgrid(*grs)
    X_grid = np.vstack([i.flat for i in meshes_tuple]).T

    # estimate grid velocities
    if n_neighbors is None:
        n_neighbors = int(n_obs / 50)
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(X_emb)
    dists, neighs = nn.kneighbors(X_grid)

    scale = np.mean([(g[1] - g[0]) for g in grs]) * smooth
    weight = normal.pdf(x=dists, scale=scale)
    p_mass = weight.sum(1)

    V_grid = (V_emb[neighs] * weight[:, :, None]).sum(1)
    V_grid /= np.maximum(1, p_mass)[:, None]
    if min_mass is None:
        min_mass = 1

    if adjust_for_stream:
        X_grid = np.stack([np.unique(X_grid[:, 0]), np.unique(X_grid[:, 1])])
        ns = int(np.sqrt(len(V_grid[:, 0])))
        V_grid = V_grid.T.reshape(2, ns, ns)

        mass = np.sqrt((V_grid ** 2).sum(0))
        min_mass = 10 ** (min_mass - 6)  # default min_mass = 1e-5
        min_mass = np.clip(min_mass, None, np.max(mass) * 0.9)
        cutoff = mass.reshape(V_grid[0].shape) < min_mass

        if cutoff_perc is None:
            cutoff_perc = 5
        length = np.sum(np.mean(np.abs(V_emb[neighs]), axis=1), axis=1).T
        length = length.reshape(ns, ns)
        cutoff |= length < np.percentile(length, cutoff_perc)

        V_grid[0][cutoff] = np.nan
    else:
        min_mass *= np.percentile(p_mass, 99) / 100
        X_grid, V_grid = X_grid[p_mass > min_mass], V_grid[p_mass > min_mass]

        if autoscale:
            V_grid /= 3 * quiver_autoscale(X_grid, V_grid)

    return X_grid, V_grid

def default_arrow(size):
    if isinstance(size, (list, tuple)) and len(size) == 3:
        head_l, head_w, ax_l = size
    elif isinstance(size, (int, np.integer, float)):
        head_l, head_w, ax_l = 12 * size, 10 * size, 8 * size
    else:
        head_l, head_w, ax_l = 12, 10, 8
    return head_l, head_w, ax_l

def plot_pseudo_time_with_arrows_on_img(args,  feature_dir, expr_name, n_neighbors=10, root_idx= 50, umap_selected_root=False, deep_method="VASC"):
    dataset_dir = args.dataset_dir
    dataset = args.dataset
    expr_dir = os.path.join(dataset_dir, dataset)
    fig_dir = os.path.join(args.figure_dir, dataset, "pseudotime")
    mkdir(fig_dir)
    root_suffix = "_umap_based_root" if umap_selected_root else "root_%d" % root_idx
    method_name = "_%s" % deep_method if deep_method != "VASC" else ""
    fig_fp = os.path.join(fig_dir, "%s_pseudotime_on_img_%s%s_%s_nneighbor_%d.pdf" % (dataset, expr_name, method_name, root_suffix, n_neighbors))
    # if os.path.exists(fig_fp):
    #     print("%s exist: pass" % fig_fp)
    #     return
    plt_setting()
    cm = plt.get_cmap('gist_rainbow')
    ncol = 2 if (dataset == "drosophila" or dataset in SQUIDPY_DATASETS) else 3
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
    elif dataset in SQUIDPY_DATASETS:
        adata = get_squidpy_data(dataset)
        coords = adata.obsm['spatial']
        annotation_dict = {"imc": "cell type", "seqfish": "celltype_mapped_refined", "slideseqv2": "cluster"}
        cell_types = adata.obs[annotation_dict[dataset]]
        n_cells = coords.shape[0]
        if args.max_cells < n_cells:
            expr_dir = os.path.join(dataset_dir, dataset)
            mkdir(expr_dir)
            indices_fp = os.path.join(expr_dir, "indices.npy")
            if os.path.exists(indices_fp):
                with open(indices_fp, 'rb') as f:
                    indices = np.load(f)
                    print("loaded indices successful!")
            x, y = coords[:, 0].take(indices), coords[:, 1].take(indices)
            cell_types = cell_types.take(indices)
        else:
            x, y = coords[:, 0], coords[:, 1]
        cell_type_strs = cell_types.cat.categories.astype(str)
        cell_type_ints = cell_types.values.codes
        cell_type_colors = list(adata.uns['%s_colors' % annotation_dict[dataset]].astype(str))
        colors = np.array([cell_type_colors[item] for item in cell_type_ints])
        for cid in range(len(cell_type_colors)):
            cit = cell_type_ints == cid
            ax.scatter(x[cit], y[cit], s=1, c=colors[cit], label=cell_type_strs[cid])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(cell_type_strs, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})
        ax.set_title("Annotations", fontsize=12, weight='bold')
        ax.set_xlabel("Spatial1", fontsize=8, weight='bold')
        ax.set_ylabel("Spatial2", fontsize=8, weight='bold')
        ax.axis('off')
        if dataset in SQUIDPY_DATASETS:
            ax.invert_yaxis()
    elif dataset == "osmFISH":
        region_colors = {
            "Layer 6": '#027fd0',
            "Pia Layer 1": '#de4726',
            "Layer 4": '#ffbb19',
            "Layer 3-4": '#8a55ff',
            "Layer 2-3 medial": '#28d1eb',
            "Hippocampus": '#004b71',
            "Internal Capsule Caudoputamen": '#9b067d',
            "White matter": '#4ec030',
            "Ventricle": '#fadf0c',
            "Layer 2-3 lateral": '#28a5eb',
            "Layer 5": '#d61420',
        }
        expr_dir = os.path.join(dataset_dir, dataset)
        expr_fp = os.path.join(expr_dir, "osmFISH_SScortex_mouse_all_cells.loom")
        ds = loompy.connect(expr_fp)

        # Make Panda's Dataframe with count data
        df_osmfish = pd.DataFrame(data=ds[:, :], columns=ds.ca['CellID'], index=ds.ra['Gene']).astype(int)

        # Include cells:
        # with more than 0 molecules
        include_mol = ds.ca.Total_molecules >= 0
        # larger than 25 um2
        include_size_min = ds.ca.size_um2 >= 5
        # smaller than 272 um2
        include_size_max = ds.ca.size_um2 <= 275
        # Not in the double imaged region
        include_position = ds.ca.X >= 0

        include = np.logical_and(np.logical_and(include_mol, include_size_min),
                                 np.logical_and(include_size_max, include_position))

        # include = np.logical_and(np.logical_and(ds.ca.Total_molecules > 20, ds.ca.size_pix > 2000), ds.ca.size_pix < 60000)
        df_osmfish = df_osmfish.loc[:, include]

        # normalize data by total number of molecules per cell and per gene.
        # df_osmfish_totmol = df_osmfish.divide(df_osmfish.sum(axis=1), axis=0) * df_osmfish.shape[
        #     0]  # Corrected for total molecules per gene
        df_osmfish_totmol = df_osmfish.divide(df_osmfish.sum(axis=0), axis=1) * df_osmfish.shape[
            1]  # Corrected for the total per cell
        # Replace NA and Nan with zero:
        df_osmfish_totmol = df_osmfish_totmol.fillna(0)
        non_zero_count_rows = np.where(df_osmfish_totmol.values.T.sum(axis=1) > 0)[0]

        # Load the cell coordinates into a Pandas Dataframe. Units are pixels
        coordinates = np.stack((ds.ca.X, ds.ca.Y))
        df_coordinates = pd.DataFrame(data=coordinates, index=['X', 'Y'], columns=ds.ca.CellID)
        df_coordinates = df_coordinates.loc[:, include].values.T
        df_coordinates = df_coordinates[non_zero_count_rows, :]
        x, y = df_coordinates[:, 0], df_coordinates[:, 1]
        ground_truth_clusters = ds.ca.Region[include]
        ground_truth_clusters = ground_truth_clusters[non_zero_count_rows]
        uniques = np.unique(ground_truth_clusters)
        for uniq in uniques:
            if uniq != "Excluded":
                label = uniq
                if label == "Internal Capsule Caudoputamen":
                    label = "IC CP"
                elif label == "Layer 2-3 lateral":
                    label = "Layer 2-3 lat"
                elif label == "Layer 2-3 medial":
                    label = "Layer 2-3 med"
                idxs = ground_truth_clusters == uniq
                ax.scatter(x[idxs], y[idxs], s=1,
                           color=region_colors[uniq], cmap='spectral', label=label)
        ax.set_aspect('equal')
        ax.set_axis_off()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})
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

    spatials = [False, True]
    titles = ["%s SP-" % deep_method, "%s SP+" % deep_method]

    stream_kwargs = {
        "linewidth": 1,
        "density": 1,
        "zorder": 3,
        "color": "k",
        "arrowsize": 1,
        "arrowstyle": "-|>",
        "maxlength": 4,
        "integration_direction": "both",
    }

    for sid, spatial in enumerate(spatials):
        args.spatial = spatial
        args.expr_name = expr_name if spatial else "default"
        args.arch = deep_method
        name = get_expr_name(args)
        feature_fp = os.path.join(feature_dir, "%s.tsv" % name)
        if os.path.exists(feature_fp):
            print("Readed %s" % feature_fp)
            adata = sc.read_csv(feature_fp, delimiter="\t", first_column_names=None)
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')
            sc.tl.umap(adata)
            sc.tl.leiden(adata, resolution=.8)
            sc.tl.paga(adata)
            distances = distance_matrix(adata.X, adata.X)
            adata.uns['iroot'] = np.argmax(distances.sum(axis=1))
            sc.tl.diffmap(adata)
            sc.tl.dpt(adata)
            # pseudotimes = adata.obs['dpt_pseudotime']
            # ncell =pseudotimes.shape[0]
            # pi_matrix = np.zeros((ncell, ncell))
            # for i in range(ncell):
            #     for j in range(ncell):
            #         if i != j:
            #             if pseudotimes[i] < pseudotimes[j]:
            #                 pi_matrix[i, j] = np.exp((pseudotimes[j] - pseudotimes[i]) * -5)

            pi_matrix = (np.corrcoef(adata.X) + 1) / 2.0
            # sparsed_pi_mat = sparsify_transition_matrix(pi_matrix, topN=50)
            # cords_vals = np.vstack((x, y)).T
            # delta_x, delta_y = get_displacement_matrix(cords_vals)
            # weighted_delta_x, weighted_delta_y = get_weighted_displacements(delta_x, sparsed_pi_mat), \
            #                                      get_weighted_displacements(delta_y, sparsed_pi_mat)
            # V_emb = np.vstack((weighted_delta_x, weighted_delta_y)).T
            # X_grid, V_grid = compute_velocity_on_grid(
            #     X_emb=cords_vals,
            #     V_emb=V_emb,
            #     density=.5,
            #     autoscale=True,
            #     smooth=1,
            #     n_neighbors=50,
            #     min_mass=None,
            # )

            try:
                offset = 1 if dataset not in ["drosophila"] + SQUIDPY_DATASETS else 0
                ax = axs[sid + offset]
                if dataset != "drosophila":
                    ax.axis('off')
                else:
                    ax.set_aspect('equal', 'box')
                    ax.set_xlim([-200, 200])
                    ax.set_ylim([-100, 100])
                if dataset not in ["drosophila", "osmFISH"] + SQUIDPY_DATASETS:
                    ax.imshow(img)
                ax.grid(False)
                st = ax.scatter(x, y, s=1, c=adata.obs['dpt_pseudotime'], cmap=cm)
                # ax.quiver(X_grid[:, 0], X_grid[:, 1], V_grid[:, 0], V_grid[:, 1], color="black")#, **stream_kwargs)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                clb = fig.colorbar(st, cax=cax)
                clb.ax.set_ylabel("pseudotime", labelpad=10, rotation=270, fontsize=8, weight='bold')
                ax.set_title("%s" % titles[sid], fontsize=12, weight='bold')
                if dataset in SQUIDPY_DATASETS:
                   ax.invert_yaxis()
            except ValueError as e:
                print(e)
            except ArpackNoConvergence as e:
                print(e)
        else:
            return

    # for ax in axs:
    #     #if dataset in ["V1_Adult_Mouse_Brain_Coronal_Section_1"]:
    #     ax.invert_xaxis()
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')
    print("Plotted %s" % fig_fp)


def visualize_SQUIDPY_DATASETS_annotation_with_pseudotime(args, expr_name, deep_method, feature_dir, n_neighbors=100, constraint_only=True):
    datasets = SQUIDPY_DATASETS
    annotation_dict = {"imc": "cell type", "seqfish": "celltype_mapped_refined", "slideseqv2":"cluster"}
    for dataset in datasets:
        args.dataset = dataset
        print("Processing %s" % dataset)
        adata = get_squidpy_data(dataset)
        cell_types = adata.obs[annotation_dict[dataset]]
        coords = adata.obsm['spatial']
        n_cells = coords.shape[0]
        if args.max_cells < n_cells:
            expr_dir = os.path.join(args.dataset_dir, dataset)
            mkdir(expr_dir)
            indices_fp = os.path.join(expr_dir, "indices.npy")
            if os.path.exists(indices_fp):
                with open(indices_fp, 'rb') as f:
                    indices = np.load(f)
                    print("loaded indices successful!")
            x, y = coords[:, 0].take(indices), coords[:, 1].take(indices)
            cell_types = cell_types.take(indices)
        else:
            x, y = coords[:, 0], coords[:, 1]

        cell_type_strs = cell_types.cat.categories.astype(str)
        cell_type_ints = cell_types.values.codes
        cell_type_colors = list(adata.uns['%s_colors'% annotation_dict[dataset]].astype(str))

        fig_dir = os.path.join(args.figure_dir, dataset, "annotation_vs_pseudotime")
        mkdir(fig_dir)
        method_name = "_%s" % deep_method
        fig_fp = os.path.join(fig_dir, "%s_pseudotime_on_img_%s%s_nneighbor_%d.pdf" % (
        dataset, expr_name, method_name, n_neighbors))
        plt_setting()
        nrow = 2 if constraint_only else 3
        ncol = len(cell_type_colors) + 1
        fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 4, nrow * 4))
        plt.subplots_adjust(wspace=0.4, hspace=0.5, bottom=0.2)
        for row in range(nrow):
            if row == 0:
                for col in range(ncol):
                    ax = axs[row][col]
                    if col == 0:
                        cells_in_this_category = np.array([True for _ in cell_type_ints])
                        title = "Annotations"
                    else:
                        cells_in_this_category = cell_type_ints == (col - 1)
                        title = cell_type_strs[col - 1]
                    colors = np.array([cell_type_colors[item] for item in cell_type_ints[cells_in_this_category]])
                    if col == 0:
                        for cid in range(len(cell_type_colors)):
                            cit = cell_type_ints == cid
                            ax.scatter(x[cit], y[cit], s=1, c=colors[cit], label=cell_type_strs[cid])
                        box = ax.get_position()
                        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                        ax.legend(cell_type_strs, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})
                    else:
                        ax.scatter(x[cells_in_this_category], y[cells_in_this_category], s=1, c=colors)

                    ax.set_title(title, fontsize=12, weight='bold')
                    ax.set_xlabel("Spatial1", fontsize=8, weight='bold')
                    ax.set_ylabel("Spatial2", fontsize=8, weight='bold')
                    ax.axis('off')
                    if dataset in SQUIDPY_DATASETS:
                        ax.invert_yaxis()

            else:
                spatials = [True] if constraint_only else [False, True]
                titles = ["%s SP+" % deep_method] if constraint_only else ["%s SP-" % deep_method, "%s SP+" % deep_method]
                args.spatial = spatials[row - 1]
                args.expr_name = expr_name if args.spatial else "default"
                args.arch = deep_method
                name = get_expr_name(args)
                feature_fp = os.path.join(feature_dir, "%s.tsv" % name)
                if os.path.exists(feature_fp):
                    print("Readed %s" % feature_fp)
                    adata = sc.read_csv(feature_fp, delimiter="\t", first_column_names=None)
                    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')
                    sc.tl.umap(adata)
                    distances = distance_matrix(adata.X, adata.X)
                    adata.uns['iroot'] = np.argmax(distances.sum(axis=1))
                    try:
                        sc.tl.diffmap(adata)
                        sc.tl.dpt(adata)
                        for col in range(ncol):
                            ax = axs[row][col]
                            if col == 0:
                                cells_in_this_category = np.array([True for _ in cell_type_ints])
                                title = "All Cells"
                            else:
                                cells_in_this_category = cell_type_ints == (col - 1)
                                title = cell_type_strs[col - 1]
                            st = ax.scatter(x[cells_in_this_category], y[cells_in_this_category], s=1, c=adata.obs['dpt_pseudotime'][cells_in_this_category])
                            ax.set_title(title, fontsize=12, weight='bold')
                            ax.axis('off')
                            divider = make_axes_locatable(ax)
                            cax = divider.append_axes("right", size="5%", pad=0.05)
                            clb = fig.colorbar(st, cax=cax)
                            clb.ax.set_ylabel("pseudotime", labelpad=10, rotation=270, fontsize=8, weight='bold')
                            if col == 0:
                                ax.set_ylabel(titles[row - 1], fontsize=8, weight='bold')
                            if dataset in SQUIDPY_DATASETS:
                                ax.invert_yaxis()
                    except ValueError as e:
                        print(e)
                    except ArpackNoConvergence as e:
                        print(e)
                else:
                    return

        plt.savefig(fig_fp, dpi=300)
        plt.close('all')
        print("Plotted %s" % fig_fp)
if __name__ == "__main__":
    mpl.use('macosx')
    args = get_args()
    # expr_name = "100_penalty1"
    # deep_methods = ["VASC", "DGI"]
    # feature_dir = os.path.join(args.dataset_dir, "features")
    # for deep_method in deep_methods:
    #     visualize_SQUIDPY_DATASETS_annotation_with_pseudotime(args, expr_name, deep_method, feature_dir, n_neighbors=100, constraint_only=False)
    datasets = SPATIAL_LIBD_DATASETS# + SPATIAL_LIBD_DATASETS#["seqfish"]#SQUIDPY_DATASETS#["osmFISH"]#
    expr_names = ["p1"]#, "250_penalty1", "500_penalty1", "1000_penalty1"]##["20_penalty1", "100_penalty1", "500_penalty1", "1000_penalty1"] #["-500_penalty2", "500_penalty1", "500_penalty1_200_penalty2", "500_penalty1_-100_penalty2"]#, "-100_penalty2", "-1000_penalty2","200_penalty1", "1000_penalty1", "500_penalty1_100_penalty2", "500_penalty1_-50_penalty2", ]#
    # # ["5_penalty1", "50_penalty1", "500_penalty1", "5_penalty1_2_panelty2", "50_penalty1_20_panelty2",
    # #  "500_penalty1_200_panelty2"]
    # # expr_names = ["-100_penalty2", "-500_penalty2", "-1000_penalty2", "200_penalty1", "500_penalty1", "1000_penalty1", "500_penalty1_100_penalty2", "500_penalty1_-100_penalty2", "500_penalty1_-50_penalty2", "500_penalty1_200_penalty2"]
    # # expr_names = ["L2_wt_KLD"]#["5_penalty1", "50_penalty1", "500_penalty1",  "5_penalty1_2_panelty2", "50_penalty1_20_panelty2", "500_penalty1_200_panelty2",]#["500_penalty1_200_penalty2", "500_penalty1_-200_penalty2" , "500_penalty1_100_penalty3", "500_penalty1_-100_penalty3", "500_penalty1_200_penalty2_-100_penalty3", "500_penalty1_200_penalty2_-250_penalty3_50p22_spc_0.35_ftf_0.6", "500_penalty1_200_penalty2_-250_penalty3_50p22_spc_0.25_ftf_0.6", "500_penalty1_200_penalty2_-250_penalty3_50p22_spc_0.25_ftf_0.6_ftc_0.25_-100p33"]#["500_penalty1"]##["500_penalty1_-100_penalty3", "500_penalty1_100_penalty3"]#["-100p11", "-100p22", "-100p33"]#["500_penalty1_-50_penalty2", "500_penalty1_-100_penalty2", "500_penalty1_100_penalty2"]
    # # expr_names = ["-100_penalty1", "-100_penalty2", "-100_penalty3", "-500_penalty1", "-500_penalty2", "-500_penalty3",
    # #               "-1000_penalty1", "-1000_penalty2", "-1000_penalty3", "100_penalty1",
    # #               "500_penalty1_200_penalty2_-100_penalty3", "500_penalty1_200_penalty2_-250_penalty3",
    # #               "500_penalty1_200_penalty2_100_penalty3", "500_penalty1_200_penalty2_250_penalty3",
    # #               "500_penalty1_200_penalty2", "500_penalty1", "500_penalty2", "500_penalty3", "1000_penalty1",
    # #               "500_penalty1_200_penalty2_-250_penalty3_20p11_50p22_spc_0.25_ftf_0.6_ftc_0.25",
    # #               "500_penalty1_200_penalty2_-250_penalty3_-20p11_50p22_spc_0.25_ftf_0.6_ftc_0.25",
    # #               "500_penalty1_200_penalty2_-250_penalty3_50p22_spc_0.25_ftf_0.6_ftc_0.25_-200p33",
    # #               "500_penalty1_200_penalty2_-250_penalty3_50p22_spc_0.25_ftf_0.6_ftc_0.25_-50p33",
    # #               "500_penalty1_200_penalty2_-250_penalty3_50p22_spc_0.25_ftf_0.6_ftc_0.25_-100p33",
    # #               "500_penalty1_200_penalty2_-250_penalty3_-50p33", "500_penalty1_200_penalty2_-250_penalty3_50p33",
    # #               "500_penalty1_200_penalty2_-250_penalty3_50p22_spc_0.35_ftf_0.5",
    # #               "500_penalty1_200_penalty2_-250_penalty3_50p22_spc_0.25_ftf_0.5",
    # #               "500_penalty1_200_penalty2_-250_penalty3_50p22_spc_0.35_ftf_0.6",
    # #               "500_penalty1_200_penalty2_-250_penalty3_1000p33", "500_penalty1_200_penalty2_-250_penalty3_500p33",
    # #               "500_penalty1_200_penalty2_-250_penalty3_200p11",
    # #               "500_penalty1_200_penalty2_-250_penalty3_50p22_spc_0.25_ftf_0.7",
    # #               "500_penalty1_200_penalty2_-250_penalty3_50p22_spc_0.25_ftf_0.6",
    # #               "500_penalty1_200_penalty2_-250_penalty3_-500p33_spc_0.2_ftc_0.35",
    # #               "500_penalty1_200_penalty2_-250_penalty3_-100p33", "500_penalty1_200_penalty2_-250_penalty3_-1000p33",
    # #               "500_penalty1_200_penalty2_-250_penalty3_-5000p33", "500_penalty1_200_penalty2_-250_penalty3_20p22",
    # #               "500_penalty1_200_penalty2_-250_penalty3_100p22", "500_penalty1_200_penalty2_-250_penalty3_200p22",
    # #               "500_penalty1_200_penalty2_-250_penalty3_-500p33", "500_penalty1_200_penalty2_-250_penalty3_100p11",
    # #               "500_penalty1_200_penalty2_-250_penalty3_50p22",
    # #               "500_penalty1_200_penalty2_-250_penalty3_50p11_10p22_-500p33",
    # #               "500_penalty1_200_penalty2_-250_penalty3_50p11_20p22_-1000p33",
    # #               "500_penalty1_200_penalty2_-250_penalty3_100p11_20p22_-1000p33",
    # #               "500_penalty1_200_penalty2_-1000_penalty3", "500_penalty1_200_penalty2_-600_penalty3",
    # #               "500_penalty1_200_penalty2_50_penalty3", "500_penalty1_100_penalty2_50_penalty3",
    # #               "500_penalty1_100_penalty2_-50_penalty3", "500_penalty1_-100_penalty2_-50_penalty3",
    # #               "500_penalty1_-100_penalty2_-100_penalty3", "500_penalty1_100_penalty2_-100_penalty3",
    # #               "500_penalty1_-50_penalty2_-250_penalty3", "500_penalty1_-100_penalty2_-250_penalty3",
    # #               "500_penalty1_-100_penalty2_-500_penalty3"]
    deep_methods = ["DGI"]#]#"DGI",["VASC"] #, 'VGAE', 'GAE'
    clustering_methods = ["kmeans"]#"leiden"
    n_neighbors = [50] #
    UMAP_based_selections = [True]#, False

    for deep_method in deep_methods:
        for dataset in datasets:
            args.dataset = dataset
            for expr_name in expr_names:
                for n_neighbor in n_neighbors:
                    for umap_based_selection in UMAP_based_selections:
                        feature_dir = os.path.join(args.dataset_dir, "features")
                        # plot_spatial_cord_with_pseudo_time(args.figure_dir, feature_dir, expr_name, coords, args.dataset)
                        # plot_spatial_vs_feature_dist_colored_pseudo_time(args.figure_dir, feature_dir, expr_name, coords, args.dataset)
                        # plot_umap_clustering(args.figure_dir, feature_dir, expr_name,  args.dataset, deep_method=deep_method)
                        #plot_pseudo_time_with_arrows_on_img(args, feature_dir, expr_name, umap_selected_root=umap_based_selection, deep_method=deep_method, n_neighbors=n_neighbor)
                        for method in clustering_methods:
                            plot_cluster_on_img(args, feature_dir, expr_name, clustering_method=method, deep_method=deep_method, n_neighbors=n_neighbor)
