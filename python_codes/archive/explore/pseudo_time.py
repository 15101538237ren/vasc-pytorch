# -*- coding:utf-8 -*-
import os
import scanpy as sc
import scvelo as scv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.config import get_args
from utils.util import mkdir
import networkx as nx
from scipy.sparse import csr_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

mpl.use('macosx')

def plt_setting(fontsz = 10):
    plt.rc('font', family='Arial')
    plt.rc('xtick', labelsize=fontsz)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=fontsz)  # fontsize of the tick labels

def get_pseudo_time(data_fp, dataset, root_idx= 300, VASC=True):
    fig_dir = os.path.join("../figures", dataset)
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

def plot_neibor_graph(data_fp, dataset, dist_fp, spatial_constrained=True, n_neighbors=5):
    sdists = np.load(dist_fp)
    sdists = sdists/np.max(sdists)
    fig_dir = os.path.join("../figures", dataset)
    mkdir(fig_dir)
    plt_setting()

    adata = sc.read_csv(data_fp, delimiter="\t", first_column_names=None)

    # Neighbor Graph
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    cmap = plt.get_cmap('gist_rainbow')
    fa = sc.tl.draw_graph(adata)
    edges = sc.Neighbors(adata).to_igraph().to_networkx().edges
    edge_colors = [cmap(sdists[e[0]][e[1]]) for e in edges]
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    plt.subplots_adjust(left=0.15, bottom=0.05)
    sc.pl.draw_graph(adata, layout=fa, ax=ax, show=False, edges= True, edges_color= edge_colors, color_map=cmap, size= 12, edges_width= 0.15)
    title = "Spatial Constrained\n(n_neighbor=%d)" % n_neighbor if spatial_constrained else "Spatial Free\n(n_neighbor=%d)" % n_neighbor
    ax.set_title(title,fontsize= 12)
    suffix = "sp_const" if spatial_constrained else "sp_free"
    fig_fp = os.path.join(fig_dir, "%s_n_neighbors_%d_%s.pdf" % (dataset, n_neighbors, suffix))
    plt.savefig(fig_fp, dpi=300)

    print("figure plotted successful!")

def plot_statistics(dataset, dist_fp, n_neighbors=5, type="weight", pernode= True):
    colors = ["blue", "red"]
    plt_setting()
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    plt.subplots_adjust(left=0.2, bottom=0.2)
    ax.grid(False)

    if pernode:
        xmax = 1.0 if type == "weight" else 250
        ymax = 500 if type == "weight" else 600
    else:
        xmax = 1.0 if type == "weight" else 300
        ymax = 2500 if type == "weight" else 2800

    bins = np.linspace(0.0, xmax, 21)

    fig_dir = os.path.join("../figures", dataset)
    mkdir(fig_dir)
    for sid, spatial_constraint in enumerate([False, True]):
        suffix = "_spatial" if spatial_constraint else "_non_spatial"
        data_fp = os.path.join(feature_dir, "%s%s.tsv" % (args.dataset, suffix))
        sdists = np.load(dist_fp)

        adata = sc.read_csv(data_fp, delimiter="\t", first_column_names=None)

        # Neighbor Graph
        sc.pp.neighbors(adata, n_neighbors=n_neighbors)
        if type == "weight":
            weight_matrix_csr = adata.obsp["connectivities"]
            if pernode:
                sum_of_weights_per_node = weight_matrix_csr.sum(axis=1).A1
                counts_of_weights_per_node = np.diff(weight_matrix_csr.indptr)
                avgs = sum_of_weights_per_node / counts_of_weights_per_node
            else:
                dense_mat = weight_matrix_csr.todense()
                avgs = np.asarray(dense_mat[dense_mat > 0.0]).reshape(-1)
        else: # spatial distances
            edges = sc.Neighbors(adata).to_igraph().to_networkx().edges
            spatial_mat = csr_matrix(sdists.shape, dtype=np.float64)
            for e in edges:
                spatial_mat[e[0], e[1]] = sdists[e[0]][e[1]]

            if pernode:
                sum_of_dists_per_node = spatial_mat.sum(axis=1).A1
                counts_of_dists_per_node = np.diff(spatial_mat.indptr)
                avgs = sum_of_dists_per_node / counts_of_dists_per_node
            else:
                dense_mat = spatial_mat.todense()
                avgs = np.asarray(dense_mat[dense_mat > 0.0]).reshape(-1)

        label = "sp_const" if spatial_constraint else "sp_free"
        ax.hist(avgs, bins=bins, edgecolor='black', facecolor=colors[sid], alpha=0.5, linewidth=0.5, label=label)
    ax.legend()
    ax.set_xlabel(type.capitalize(), weight='bold', fontsize=10)
    ax.set_ylabel("Freq.", weight='bold', fontsize=10)
    ax.set_xlim([0, xmax])
    ax.set_ylim([0, ymax])

    pernode_str = "pernode" if pernode else "all"
    fig_fp = os.path.join(fig_dir, "%s_%s_n_neighbors_%d.pdf" % (type, pernode_str, n_neighbors))
    mkdir(os.path.dirname(fig_fp))
    plt.savefig(fig_fp, dpi=300)

    print("figure plotted successful!")

def plot_umap(data_fp, dataset, root_idx= 300, VASC=True):
    fig_dir = os.path.join("../figures", dataset)
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
    adata.uns['iroot'] = root_idx
    sc.tl.dpt(adata)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    plt.subplots_adjust(left=0.2, bottom=0.2)
    sc.pl.umap(adata, color='dpt_pseudotime', ax=ax, show=False)
    ax.set_title(dataset)
    fig_fp = os.path.join(fig_dir, "%s_umap_vasc.pdf" % dataset)
    plt.savefig(fig_fp, dpi=300)

def plot_spatial_cord_with_pseudo_time(data_fp, spatial_cord_fp, dataset, root_idx= 300, VASC=True):
    fig_dir = os.path.join("../figures", dataset)
    mkdir(fig_dir)
    plt_setting()

    spatial_adata = sc.read_h5ad(spatial_cord_fp)
    spatial_cords = spatial_adata.obsm['spatial']
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
    ax.grid(False)
    st = ax.scatter(spatial_cords[:, 0], spatial_cords[:, 2], s=4, c=adata.obs['dpt_pseudotime'])
    fig.colorbar(st, ax=ax)
    ax.set_aspect('equal', 'box')
    ax.set_xlim([-200, 200])
    ax.set_ylim([-100, 100])
    ax.set_title("Axis: x, z")

    fig_fp = os.path.join(fig_dir, "%s_spatial_pseudo_time_vasc.pdf" % dataset)
    plt.savefig(fig_fp, dpi=300)

def plot_spatial_cord_with_pseudo_time_3d(data_fp, spatial_cord_fp, dataset, root_idx= 300, VASC=True):
    fig_dir = os.path.join("../figures", dataset)
    mkdir(fig_dir)
    plt_setting()

    spatial_adata = sc.read_h5ad(spatial_cord_fp)
    spatial_cords = spatial_adata.obsm['spatial']
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
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0.2, bottom=0.2)

    adata.uns['iroot'] = root_idx
    sc.tl.dpt(adata)
    ax.grid(False)
    st = ax.scatter(spatial_cords[:, 0], spatial_cords[:, 2], spatial_cords[:, 1], s=4, c=adata.obs['dpt_pseudotime'])
    fig.colorbar(st, ax=ax)
    plt.show()
    fig_fp = os.path.join(fig_dir, "%s_spatial_pseudo_time_vasc_3d.pdf" % dataset)
    #plt.savefig(fig_fp, dpi=300)

def plot_spatial_cord_with_pseudo_time_3d_comparison(feature_non_spatial_fp, feature_spatial_fp, spatial_cord_fp, dataset, root_idx= 300):
    fig_dir = os.path.join("../figures", dataset)
    mkdir(fig_dir)
    plt_setting()

    spatial_adata = sc.read_h5ad(spatial_cord_fp)
    spatial_cords = spatial_adata.obsm['spatial']

    adata_non_spatial = sc.read_csv(feature_non_spatial_fp, delimiter="\t", first_column_names=None)
    adata_spatial = sc.read_csv(feature_spatial_fp, delimiter="\t", first_column_names=None)

    # Neighbor Graph
    sc.pp.neighbors(adata_non_spatial, n_neighbors=10)
    sc.pp.neighbors(adata_spatial, n_neighbors=10)

    sc.tl.umap(adata_non_spatial)
    sc.tl.umap(adata_spatial)

    adata_non_spatial.uns['iroot'] = root_idx
    adata_spatial.uns['iroot'] = root_idx

    sc.tl.dpt(adata_non_spatial)
    sc.tl.dpt(adata_spatial)

    fig = plt.figure(figsize=(8, 4))
    plt.subplots_adjust(left=0.2, bottom=0.2)

    ax = fig.add_subplot(121, projection='3d')
    ax.grid(False)
    st = ax.scatter(spatial_cords[:, 0], spatial_cords[:, 2], spatial_cords[:, 1], s=4, c=adata_non_spatial.obs['dpt_pseudotime'])
    fig.colorbar(st, ax=ax)
    ax.set_title("No Spatial Constraint", fontsize=14)

    ax = fig.add_subplot(122, projection='3d')
    ax.grid(False)

    st = ax.scatter(spatial_cords[:, 0], spatial_cords[:, 2], spatial_cords[:, 1], s=4, c=adata_spatial.obs['dpt_pseudotime'])
    fig.colorbar(st, ax=ax)
    ax.set_title("Spatial Constraint", fontsize=14)

    plt.show()
    fig_fp = os.path.join(fig_dir, "%s_spatial_pseudo_time_vasc_3d.pdf" % dataset)

def get_pseudo_time_of_feaure(feature_fp, n_neighbors = 10, root_idx= 300):
    adata = sc.read_csv(feature_fp, delimiter="\t", first_column_names=None)

    # Neighbor Graph
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    sc.tl.umap(adata)
    adata.uns['iroot'] = root_idx

    sc.tl.dpt(adata)
    return adata.obs['dpt_pseudotime']

def plot_param_exploration_figures(fontsz=12):
    args = get_args()
    spatial_cord_fp = os.path.join("../", args.dataset_dir, args.dataset, "spatial_pred.h5ad")
    spatial_adata = sc.read_h5ad(spatial_cord_fp)
    spatial_cords = spatial_adata.obsm['spatial']

    n_sections = 10
    max_feature_dist = 3.
    feature_dist_thresholds = np.arange(max_feature_dist / n_sections, max_feature_dist * 1.05,
                                        max_feature_dist / n_sections)
    max_spatial_dist = 300.
    spatial_dist_thresholds = np.arange(max_spatial_dist / n_sections, max_spatial_dist * 1.05,
                                        max_spatial_dist / n_sections)
    cords = [[0, 2], [0, 1], [1, 2]]
    lims = [[-200, 200], [0, 80], [-100, 100]]
    labels = ["x", "y", "z"]
    plt_setting(fontsz=fontsz)
    fig_sz = (5 * (n_sections//2 + 1), 4 * len(cords))
    sc.settings.set_figure_params(dpi=300, frameon=False, figsize=fig_sz, facecolor='white')
    fig, axs = plt.subplots(len(cords), (n_sections//2) + 1, figsize=fig_sz)
    plt.subplots_adjust(wspace=0.3)
    feature_dir = os.path.join("../", args.dataset_dir, args.feature_dir)
    feature_non_spatial_fp = os.path.join(feature_dir, "%s_non_spatial.tsv" % args.dataset)
    pseudotime = get_pseudo_time_of_feaure(feature_non_spatial_fp)

    for li, cord in enumerate(cords):
        ax = axs[li, 0]
        ax.grid(False)
        xind, yind = cord
        st = ax.scatter(spatial_cords[:, xind], spatial_cords[:, yind], s=4, c=pseudotime)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(st, cax=cax)

        ax.set_aspect('equal', 'box')
        ax.set_xlim(lims[xind])
        ax.set_ylim(lims[yind])
        ax.set_ylabel("%s Spatial Coord" % labels[yind], fontsize=fontsz)
        ax.set_xlabel("%s Spatial Coord" % labels[xind], fontsize=fontsz)
        ax.set_title("NO Spatial Constraint")

    for ni in range(n_sections):
        if ni % 2 == 0:
            feature_dist_thrs = feature_dist_thresholds[ni]
            spatial_dist_thrs = spatial_dist_thresholds[ni]
            thrs_str = "f_%.1f_sp_%.0f" % (feature_dist_thrs, spatial_dist_thrs)
            feature_fp = os.path.join(feature_dir, "%s_%s.tsv" % (args.dataset, thrs_str))
            pseudotime = get_pseudo_time_of_feaure(feature_fp)
            for li, cord in enumerate(cords):
                ax = axs[li, ni//2 + 1]
                ax.grid(False)
                xind, yind = cord
                ax.scatter(spatial_cords[:, xind], spatial_cords[:, yind], s=4, c=pseudotime)
                ax.set_aspect('equal', 'box')
                ax.set_xlim(lims[xind])
                ax.set_ylim(lims[yind])
                ax.set_title("Feature thrs: %.1f, Spatial thrs: %d" % (feature_dist_thrs, spatial_dist_thrs))
                ax.set_xlabel("%s Spatial Coord" % labels[xind], fontsize=fontsz)

    fig_dir = os.path.join("../figures", args.dataset)
    mkdir(fig_dir)
    fig_fp = os.path.join(fig_dir, "param_comparison.pdf")
    plt.savefig(fig_fp, dpi=300)

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

def sparsify_transition_matrix(pi_matrix, topN = 10):
    n = pi_matrix.shape[0]
    sparsed_matrix = np.zeros((n, n))
    for i in range(n):
        ind = pi_matrix[i,:].argsort()[-(topN+1):][::-1][1:]
        sparsed_matrix[i, ind] = pi_matrix[i, ind]

    sum_of_rows = sparsed_matrix.sum(axis=1)
    normalized_sparsed_matrix = sparsed_matrix / sum_of_rows[:, np.newaxis]
    return normalized_sparsed_matrix

def get_weighted_displacements(displacements, sparsed_pi_mat):
    n = displacements.shape[0]
    disp = [0.0 for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if j != i and sparsed_pi_mat[i, j] > 1e-5:
                disp[i] += sparsed_pi_mat[i, j] * displacements[i, j]
    return np.array(disp)

def plot_param_exploration_scVelo_figures(fontsz=12, n_neighbors=10):
    args = get_args()
    spatial_cord_fp = os.path.join("../", args.dataset_dir, args.dataset, "spatial_pred.h5ad")
    spatial_adata = sc.read_h5ad(spatial_cord_fp)
    spatial_cords = spatial_adata.obsm['spatial']
    cm = plt.get_cmap('viridis')
    n_sections = 10
    max_feature_dist = 3.
    feature_dist_thresholds = np.arange(max_feature_dist / n_sections, max_feature_dist * 1.05,
                                        max_feature_dist / n_sections)
    max_spatial_dist = 300.
    spatial_dist_thresholds = np.arange(max_spatial_dist / n_sections, max_spatial_dist * 1.05,
                                        max_spatial_dist / n_sections)
    cords = [[0, 2], [0, 1], [1, 2]]
    lims = [[-200, 200], [0, 80], [-100, 100]]
    labels = ["x", "y", "z"]
    plt_setting(fontsz=fontsz)
    fig_sz = (5 * (n_sections//2 + 1), 4 * len(cords))
    sc.settings.set_figure_params(dpi=300, frameon=False, figsize=fig_sz, facecolor='white')
    fig, axs = plt.subplots(len(cords), (n_sections//2) + 1, figsize=fig_sz)
    plt.subplots_adjust(wspace=0.3)
    feature_dir = os.path.join("../", args.dataset_dir, args.feature_dir)
    feature_non_spatial_fp = os.path.join(feature_dir, "%s_non_spatial.tsv" % args.dataset)
    pseudotime = get_pseudo_time_of_feaure(feature_non_spatial_fp)
    adata = scv.read(feature_non_spatial_fp, delimiter="\t")
    pi_matrix = (np.corrcoef(adata.X) + 1)/2.0
    sparsed_pi_mat = sparsify_transition_matrix(pi_matrix, topN=n_neighbors)

    for li, cord in enumerate(cords):
        ax = axs[li, 0]
        ax.grid(False)
        xind, yind = cord
        cords_vals = spatial_cords[:, [xind, yind]]
        delta_x, delta_y = get_displacement_matrix(cords_vals)
        weighted_delta_x, weighted_delta_y = get_weighted_displacements(delta_x, sparsed_pi_mat), get_weighted_displacements(delta_y, sparsed_pi_mat)
        st = ax.quiver(cords_vals[:, 0], cords_vals[:, 1], weighted_delta_x, weighted_delta_y, pseudotime, cmap=cm)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(st, cax=cax)
        ax.set_aspect('equal', 'box')
        ax.set_xlim(lims[xind])
        ax.set_ylim(lims[yind])
        ax.set_ylabel("%s Spatial Coord" % labels[yind], fontsize=fontsz)
        ax.set_xlabel("%s Spatial Coord" % labels[xind], fontsize=fontsz)
        ax.set_title("NO Spatial Constraint")

    for ni in range(n_sections):
        if ni % 2 == 0:
            feature_dist_thrs = feature_dist_thresholds[ni]
            spatial_dist_thrs = spatial_dist_thresholds[ni]
            thrs_str = "f_%.1f_sp_%.0f" % (feature_dist_thrs, spatial_dist_thrs)
            feature_fp = os.path.join(feature_dir, "%s_%s.tsv" % (args.dataset, thrs_str))
            pseudotime = get_pseudo_time_of_feaure(feature_fp)
            adata = scv.read(feature_fp, delimiter="\t")
            pi_matrix = (np.corrcoef(adata.X) + 1) / 2.0
            sparsed_pi_mat = sparsify_transition_matrix(pi_matrix, topN=n_neighbors)
            for li, cord in enumerate(cords):
                ax = axs[li, ni//2 + 1]
                ax.grid(False)
                xind, yind = cord
                cords_vals = spatial_cords[:, [xind, yind]]
                delta_x, delta_y = get_displacement_matrix(cords_vals)
                weighted_delta_x, weighted_delta_y = get_weighted_displacements(delta_x, sparsed_pi_mat), get_weighted_displacements(
                    delta_y, sparsed_pi_mat)
                ax.quiver(cords_vals[:, 0], cords_vals[:, 1], weighted_delta_x, weighted_delta_y, pseudotime, cmap=cm)
                ax.set_aspect('equal', 'box')
                ax.set_xlim(lims[xind])
                ax.set_ylim(lims[yind])
                ax.set_title("Feature thrs: %.1f, Spatial thrs: %d" % (feature_dist_thrs, spatial_dist_thrs))
                ax.set_xlabel("%s Spatial Coord" % labels[xind], fontsize=fontsz)

    fig_dir = os.path.join("../figures", args.dataset)
    mkdir(fig_dir)
    fig_fp = os.path.join(fig_dir, "param_comparison_velocity_n_neighbor_%d.pdf" % n_neighbors)
    plt.savefig(fig_fp, dpi=300)

def plot_param_exploration_median_weights_and_spatial_dists(args, dist_fp, n_neighbors=5):
    plt_setting(fontsz=8)
    cm = plt.get_cmap('hsv')
    fig, axs = plt.subplots(1, 2, figsize=(4 * 2, 4))
    plt.subplots_adjust(left=0.2, bottom=0.2, wspace= 0.4)
    sdists = np.load(dist_fp)

    fig_dir = os.path.join("../figures", args.dataset)
    feature_dir = os.path.join("..", args.dataset_dir, args.feature_dir)
    mkdir(fig_dir)

    n_sections = 10
    max_feature_dist = 3.
    feature_dist_thresholds = np.arange(max_feature_dist / n_sections, max_feature_dist * 1.05,
                                        max_feature_dist / n_sections)
    max_spatial_dist = 300.
    spatial_dist_thresholds = np.arange(max_spatial_dist / n_sections, max_spatial_dist * 1.05,
                                        max_spatial_dist / n_sections)
    values = [[], []]
    labels = []
    for ni in range(n_sections + 1):
        if ni == 0:
            feature_fp = os.path.join(feature_dir, "%s%s.tsv" % (args.dataset, "_non_spatial"))
            labels.append("No Sp-Const")
        else:
            feature_dist_thrs = feature_dist_thresholds[ni - 1]
            spatial_dist_thrs = spatial_dist_thresholds[ni - 1]
            thrs_str = "f_%.1f_sp_%.0f" % (feature_dist_thrs, spatial_dist_thrs)
            feature_fp = os.path.join(feature_dir, "%s_%s.tsv" % (args.dataset, thrs_str))
            labels.append("F:%.1f, S:%d" % (feature_dist_thrs, spatial_dist_thrs))

        adata = sc.read_csv(feature_fp, delimiter="\t", first_column_names=None)
        # Neighbor Graph
        sc.pp.neighbors(adata, n_neighbors=n_neighbors)

        for i in range(2):
            if i == 1: # weights
                weight_matrix_csr = adata.obsp["connectivities"]
                sum_of_weights_per_node = weight_matrix_csr.sum(axis=1).A1
                counts_of_weights_per_node = np.diff(weight_matrix_csr.indptr)
                avgs = sum_of_weights_per_node / counts_of_weights_per_node
            else: # spatial distances
                edges = sc.Neighbors(adata).to_igraph().to_networkx().edges
                spatial_mat = csr_matrix(sdists.shape, dtype=np.float64)
                for e in edges:
                    spatial_mat[e[0], e[1]] = sdists[e[0]][e[1]]
                sum_of_dists_per_node = spatial_mat.sum(axis=1).A1
                counts_of_dists_per_node = np.diff(spatial_mat.indptr)
                avgs = sum_of_dists_per_node / counts_of_dists_per_node
            avgs = avgs[np.isfinite(avgs)]
            median_val = np.median(avgs)
            values[i].append(median_val)
    ind = np.arange(n_sections + 1)

    for ni in range(2):
        ax = axs[ni]
        ax.grid(False)
        colors = [cm(1. * idx / (n_sections + 1)) for idx in range(n_sections + 1)]
        ax.scatter(ind, tuple(values[ni]), s=16, c=colors)
        ax.set_xticks(ind)
        ax.set_xticklabels(labels, rotation=90)
        max_v = np.array(values[ni]).max()
        ax.set_ylim([0.6 * max_v, 1.05 * max_v])
        ylabel = "Median Norm. Weight" if ni == 0 else "Median Norm. Spatial Dist"
        ax.set_ylabel(ylabel, weight='bold', fontsize=12)

    fig_fp = os.path.join(fig_dir, "%s_n_neighb_%d.pdf" % ("Median_Weight_Spatial_Dist_Bars", n_neighbors))
    mkdir(os.path.dirname(fig_fp))
    plt.savefig(fig_fp, dpi=300)
    print("figure plotted successful!")

if __name__ == "__main__":
    n_neighbors_arr = [20, 40, 60, 100]
    for n_neighbor in n_neighbors_arr:
        plot_param_exploration_scVelo_figures(n_neighbors=n_neighbor)

    # plot_param_exploration_figures()
    # args = get_args()
    # dist_fp = "../data/drosophila/drosophila_spatial_dist.npy"
    # plot_param_exploration_median_weights_and_spatial_dists(args, dist_fp, n_neighbors=10)
    # sc.settings.verbosity = 3
    #
    # args = get_args()
    #
    # feature_dir = os.path.join("../", args.dataset_dir, args.feature_dir)
    # dist_fp = "../data/drosophila/drosophila_spatial_dist.npy"
    # n_neighbors = [10]
    # for spatial_constraint in [True, False]:
    #     for n_neighbor in n_neighbors:
    #         suffix = "_spatial" if spatial_constraint else "_non_spatial"
    #         feature_fp = os.path.join(feature_dir, "%s%s.tsv" % (args.dataset, suffix))
    #         plot_neibor_graph(feature_fp, args.dataset, dist_fp, n_neighbors=n_neighbor,
    #                         spatial_constrained=spatial_constraint)

    # types = ["weight", "spatial_distance"]
    # pernodes = [True, False]
    # for type in types:
    #     for pernode in pernodes:
    #         for n_neighbor in n_neighbors:
    #             plot_statistics(args.dataset, dist_fp, n_neighbors=n_neighbor, type=type, pernode=pernode)

    #plot_umap(feature_fp, args.dataset, root_idx=300)
    # spatial_cord_fp = os.path.join("../", args.dataset_dir, args.dataset, "spatial_pred.h5ad")
    # plot_spatial_cord_with_pseudo_time_3d(feature_fp, spatial_cord_fp, args.dataset)
    # feature_non_spatial_fp, feature_spatial_fp = os.path.join(feature_dir, "%s_non_spatial.tsv" % args.dataset), os.path.join(feature_dir, "%s_spatial.tsv" % args.dataset)
    # plot_spatial_cord_with_pseudo_time_3d_comparison(feature_non_spatial_fp, feature_spatial_fp, spatial_cord_fp, args.dataset)
    # expr_fp = os.path.join("../", args.dataset_dir, args.dataset, "%s.txt" % args.dataset)
    # get_pseudo_time(expr_fp, args.dataset, VASC=False)
