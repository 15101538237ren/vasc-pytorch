# -*- coding:utf-8 -*-
import os
import anndata
import loompy
import json
import pandas as pd
import squidpy as sq
import scanpy as sc
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.config import get_args
from utils.util import mkdir, get_osmFISH, get_expr_name, SPATIAL_LIBD_DATASETS, SQUIDPY_DATASETS, get_squipy, VISIUM_DATASETS
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import distance, distance_matrix
from sklearn import metrics
import seaborn as sns

def plt_setting(fontsz = 10):
    plt.rc('font', family='Arial', weight='bold')
    plt.rc('xtick', labelsize=fontsz)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=fontsz)  # fontsize of the tick labels

def plotOSMFISH_ground_truth(ax, x, y, ground_truth_clusters, region_colors):
    ax.axis('off')

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
                       color=region_colors[uniq], cmap='spectral', label=label)

    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.set_title("Annotated Domain", fontsize=12, weight='bold')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})

def plot_visium(ax, expr_dir, dataset):
    scale_factor_fp = os.path.join(expr_dir, "spatial", "scalefactors_json.json")
    with open(scale_factor_fp, "r") as json_file:
        data_dict = json.load(json_file)
        scale = data_dict["tissue_lowres_scalef"]
    adata = sc.datasets.visium_sge(dataset)
    spatial_cords = adata.obsm['spatial'].astype(float) * scale
    x, y = spatial_cords[:, 0], spatial_cords[:, 1]
    img = plt.imread(os.path.join(expr_dir, "spatial", "tissue_lowres_image.png"))
    ax.axis('off')
    ax.set_title("H&E image", fontsize=12, weight='bold')
    ax.imshow(img)
    return x, y, img

def plot_spatial_LIBD(ax, expr_dir, dataset_dir, dataset):
    cm = plt.get_cmap('Set1')
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
        ax.scatter(x[ind], y[ind], s=1, color=color, label=cluster)
    ax.axis('off')
    ax.set_title("Ground Truth", fontsize=12, weight='bold')
    return x, y, clusters, img

def plot_figures(args, dataset, deep_method= "DGI", resolution="0.8", clustering_method= "leiden", n_neighbors=50, comparison=False, expr_name = "500_penalty1"):
    dataset_dir = args.dataset_dir
    args.dataset = dataset
    expr_dir = os.path.join(dataset_dir, args.dataset)

    plt_setting()
    cm1 = plt.get_cmap('Set1')
    ncol = 5 if comparison else 3
    fig, axs = plt.subplots(1, ncol, figsize=(ncol * 5, 4))
    hspace = 0.5 if dataset in SPATIAL_LIBD_DATASETS else 0.7
    wspace = 0.25 if dataset in SPATIAL_LIBD_DATASETS else 0.4
    plt.subplots_adjust(wspace=wspace, hspace=hspace, bottom=0.2)

    fig_counter = 0
    if dataset == "osmFISH":
        expr_fp = os.path.join(expr_dir, "osmFISH_SScortex_mouse_all_cells.loom")
        x, y, ground_truth_clusters, region_colors = get_osmFISH(expr_fp)
        plotOSMFISH_ground_truth(axs[fig_counter],x, y, ground_truth_clusters, region_colors)
    elif dataset in SQUIDPY_DATASETS:
        x, y, ground_truth_clusters, cell_type_strs, cell_type_colors, colors = get_squipy(args, dataset)
        plot_squidpy_ground_truth(axs[fig_counter], x, y, ground_truth_clusters, cell_type_strs, cell_type_colors, colors)
    elif dataset in VISIUM_DATASETS:
        x, y, img = plot_visium(axs[fig_counter], expr_dir, dataset)
    else:#spatial_LIBD
        x, y, ground_truth_clusters, img = plot_spatial_LIBD(axs[fig_counter], expr_dir, dataset_dir, dataset)
    spatials = [False, True] if comparison else [True]
    section_title = "Segmentation:"
    titles = ["%s %s SP-" % (section_title, deep_method), "%s%s SP+" % (section_title, deep_method)] if comparison else ["%s%s SP+" % (section_title, deep_method)]
    feature_dir = os.path.join(args.dataset_dir, "features")
    labels_dir = os.path.join(feature_dir, "%s_labels_resolution_%s" % (clustering_method, resolution))

    #Plot cluster
    for sid, spatial in enumerate(spatials):
        fig_counter += 1
        ax = axs[fig_counter]
        ax.axis('off')
        ax.grid(False)
        args.spatial = spatial
        args.expr_name = expr_name if spatial else "default"
        args.arch = deep_method
        name = get_expr_name(args)
        label_fp = os.path.join(labels_dir, "%s_label_nNeighbor_%d.tsv" % (name, n_neighbors))
        if os.path.exists(label_fp):

            clusters = pd.read_csv(label_fp, header=None).values.astype(int).flatten()
            unique_clusters = np.unique(clusters)

            if dataset in VISIUM_DATASETS + SPATIAL_LIBD_DATASETS:
                ax.imshow(img)
                ax.set_title("%s" % titles[sid], fontsize=12, weight='bold')
                if dataset in SPATIAL_LIBD_DATASETS:
                    ari = metrics.adjusted_rand_score(ground_truth_clusters, clusters)
                    ax.set_title("%s\n ARI: %.2f" % (titles[sid], ari), fontsize=12, weight='bold')
            else:
                ari = metrics.adjusted_rand_score(ground_truth_clusters, clusters)
                ax.set_title("%s\n ARI: %.2f" % (titles[sid], ari), fontsize=12, weight='bold')
            for cid, cluster in enumerate(unique_clusters[:-1]):
                color = cm1(1. * cid / (len(unique_clusters) + 1))
                ind = clusters == cluster
                ax.scatter(x[ind], y[ind], s=1, color=color, label=cluster)

            #ax.legend()
            if dataset in SQUIDPY_DATASETS:
                ax.invert_yaxis()

    section_title = "Spatial-pseudotime: "
    titles = ["%s %s SP-" % (section_title, deep_method),
              "%s%s SP+" % (section_title, deep_method)] if comparison else ["%s%s SP+" % (section_title, deep_method)]

    cm2 = plt.get_cmap('gist_rainbow')
    # Plot pseudotime
    for sid, spatial in enumerate(spatials):
        fig_counter += 1
        ax = axs[fig_counter]
        ax.axis('off')
        ax.grid(False)

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
            distances = distance_matrix(adata.X, adata.X)
            adata.uns['iroot'] = np.argmax(distances.sum(axis=1))
            try:
                if dataset in VISIUM_DATASETS + SPATIAL_LIBD_DATASETS:
                    ax.imshow(img)
                sc.tl.diffmap(adata)
                sc.tl.dpt(adata)
                st = ax.scatter(x, y, s=1, c=adata.obs['dpt_pseudotime'], cmap=cm2)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                clb = fig.colorbar(st, cax=cax)
                clb.ax.set_ylabel("pseudotime", labelpad=10, rotation=270, fontsize=8, weight='bold')
                ax.set_title("%s" % titles[sid], fontsize=12, weight='bold')
                if dataset in SQUIDPY_DATASETS:
                   ax.invert_yaxis()
            except ValueError as e:
                print(e)

    fig_dir = os.path.join(args.figure_dir, "%s" % dataset)
    mkdir(fig_dir)
    fig_fp = os.path.join(fig_dir, "%s_%s.pdf" % (dataset, deep_method))
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')

def plot_squidpy_ground_truth(ax, x, y, ground_truth_clusters, cell_type_strs, cell_type_colors, colors):
    ax.axis('off')
    for cid in range(len(cell_type_colors)):
        cit = ground_truth_clusters == cid
        ax.scatter(x[cit], y[cit], s=1, c=colors[cit], label=cell_type_strs[cid])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(cell_type_strs, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})
    ax.invert_yaxis()

def plot_SPATIAL_LIBD_FIG2(args, datasets):
    plt_setting()
    nrow, ncol = len(datasets), 8
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 4, nrow * 4))
    plt.subplots_adjust(wspace=0.2, hspace=0.25, bottom=0.2)
    for did, dataset in enumerate(datasets):
        args.dataset = dataset
        expr_dir = os.path.join(args.dataset_dir, args.dataset)
        library_id = dataset.split("_")[-1]
        cm = plt.get_cmap('Set1')

        ax = axs[did, 0]
        img = plt.imread(os.path.join(args.dataset_dir, dataset, "%s_tissue_lowres_image.png" % library_id))
        ax.imshow(img)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        if did == 0:
            ax.set_title("H&E\nimage", fontsize=14, weight='bold')
        ax.set_ylabel("Sample %s" % library_id, fontsize=14, weight='bold')

        scale = 0.045
        coord_fp = os.path.join(expr_dir, "spatial_coords.csv")
        spatial_cords = pd.read_csv(coord_fp).values.astype(float) * scale
        x, y = spatial_cords[:, 1], spatial_cords[:, 0]

        info_df = pd.read_csv(os.path.join(expr_dir, "spot_info.csv"))
        ground_truth_clusters = info_df["layer_guess_reordered"].values.astype(str)

        ax = axs[did, 1]
        clusters = info_df["SpatialDE_PCA"].values.astype(int)
        ax.imshow(img)
        unique_clusters = np.unique(clusters)
        for cid, cluster in enumerate(unique_clusters):
            color = cm(1. * cid / (len(unique_clusters) + 1))
            ind = clusters == cluster
            ax.scatter(x[ind], y[ind], s=1, color=color, label=cluster)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ari = metrics.adjusted_rand_score(ground_truth_clusters, clusters)
        suff = "" if did else "SpatialDE_PCA\n "
        ax.set_title("%sARI: %.2f" % (suff, ari), fontsize=14, weight='bold')

        ax = axs[did, 2]
        clusters = info_df["HVG_PCA"].values.astype(int)
        ax.imshow(img)
        unique_clusters = np.unique(clusters)
        for cid, cluster in enumerate(unique_clusters):
            color = cm(1. * cid / (len(unique_clusters) + 1))
            ind = clusters == cluster
            ax.scatter(x[ind], y[ind], s=1, color=color, label=cluster)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ari = metrics.adjusted_rand_score(ground_truth_clusters, clusters)
        suff = "" if did else "HVG_PCA\n "
        ax.set_title("%sARI: %.2f" % (suff, ari), fontsize=14, weight='bold')

        methods = ["VASC", "DGI"]
        withwt = ["SP-", "SP+"]
        feature_dir = os.path.join(args.dataset_dir, "features")
        for mid, method in enumerate(methods):
            for wid, spatial in enumerate(withwt):
                ax = axs[did, 3 + mid * len(withwt) + wid]
                expr_name = "p1" if method == "DGI" else "500_penalty1"
                resolution = "0.2" if method == "DGI" else "0.6"
                ax.imshow(img)
                args.spatial = wid
                args.expr_name = expr_name if wid else "default"
                args.arch = method
                name = get_expr_name(args)
                labels_dir = os.path.join(feature_dir, "%s_labels_resolution_%s" % ("leiden", resolution))
                label_fp = os.path.join(labels_dir, "%s_label_nNeighbor_%d.tsv" % (name, 100))
                if os.path.exists(label_fp):
                    clusters = pd.read_csv(label_fp, header=None).values.astype(int).flatten()
                    unique_clusters = np.unique(clusters)
                    for cid, cluster in enumerate(unique_clusters):
                        color = cm(1. * cid / (len(unique_clusters) + 1))
                        ind = clusters == cluster
                        ax.scatter(x[ind], y[ind], s=1, color=color, label=cluster)
                    ax.get_xaxis().set_ticks([])
                    ax.get_yaxis().set_ticks([])
                    ari = metrics.adjusted_rand_score(ground_truth_clusters, clusters)
                    suff = "" if did else "%s %s\n " % (method, spatial)
                    ax.set_title("%sARI: %.2f" % (suff, ari), fontsize=14, weight='bold')
        ax = axs[did, ncol - 1]

        ax.imshow(img)
        unique_clusters = np.unique(ground_truth_clusters)
        for cid, cluster in enumerate(unique_clusters[:-1]):
            color = cm(1. * cid / (len(unique_clusters) + 1))
            ind = ground_truth_clusters == cluster
            ax.scatter(x[ind], y[ind], s=1, color=color, label=cluster)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        suff = "" if did else "Ground Truth\n"
        ax.set_title("%s" % suff, fontsize=14, weight='bold')
    fig_dir = os.path.join(args.figure_dir)
    mkdir(fig_dir)
    fig_fp = os.path.join(fig_dir, "Fig2.pdf")
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')

def plot_PseudoTime_Fig3(args, method = "DGI"):
    datasets = SPATIAL_LIBD_DATASETS + VISIUM_DATASETS
    plt_setting()
    nrow, ncol = 3 , len(datasets)
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 4, nrow * 4))
    plt.subplots_adjust(wspace=0.2, hspace=0.25, bottom=0.2)
    cm = plt.get_cmap('gist_rainbow')
    spatials = [False, True]
    titles = ["%s by %s SP-" % ("Pseudotime", method),
              "%s by %s SP+" % ("Pseudotime", method)]
    for did, dataset in enumerate(datasets):
        args.dataset = dataset
        feature_dir = os.path.join(args.dataset_dir, "features")
        expr_dir = os.path.join(args.dataset_dir, args.dataset)
        ax = axs[0, did]
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        if did == 0:
            ax.set_ylabel("H&E image", fontsize=14, weight='bold')
        if dataset in SPATIAL_LIBD_DATASETS:
            library_id = dataset.split("_")[-1]
            img = plt.imread(os.path.join(args.dataset_dir, dataset, "%s_tissue_lowres_image.png" % library_id))
            ax.imshow(img)
            ax.set_title("SpatialLIBD %s" % library_id, fontsize=14, weight='bold')

            scale = 0.045
            coord_fp = os.path.join(expr_dir, "spatial_coords.csv")
            spatial_cords = pd.read_csv(coord_fp).values.astype(float) * scale
            x, y = spatial_cords[:, 1], spatial_cords[:, 0]
            n_neighbors = 100
        else:
            scale_factor_fp = os.path.join(expr_dir, "spatial", "scalefactors_json.json")
            with open(scale_factor_fp, "r") as json_file:
                data_dict = json.load(json_file)
                scale = data_dict["tissue_lowres_scalef"]
            adata = sc.datasets.visium_sge(dataset)
            img = plt.imread(os.path.join(expr_dir, "spatial", "tissue_lowres_image.png"))
            ax.axis('off')
            ax.set_title(dataset, fontsize=12, weight='bold')
            ax.imshow(img)
            spatial_cords = adata.obsm['spatial'].astype(float) * scale
            x, y = spatial_cords[:, 0], spatial_cords[:, 1]
            n_neighbors = 50
        # Plot pseudotime

        for sid, spatial in enumerate(spatials):
            ax = axs[sid + 1, did]
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

            args.spatial = spatial
            args.expr_name = "p1" if spatial else "default"
            args.arch = method
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
                    if dataset in VISIUM_DATASETS + SPATIAL_LIBD_DATASETS:
                        ax.imshow(img)
                    sc.tl.diffmap(adata)
                    sc.tl.dpt(adata)
                    st = ax.scatter(x, y, s=1, c=adata.obs['dpt_pseudotime'], cmap=cm)
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    clb = fig.colorbar(st, cax=cax)
                    clb.ax.set_ylabel("pseudotime", labelpad=10, rotation=270, fontsize=8, weight='bold')

                except ValueError as e:
                    print(e)
            if did == 0:
                ax.set_ylabel("%s" % titles[sid], fontsize=12, weight='bold')

    fig_dir = os.path.join(args.figure_dir)
    mkdir(fig_dir)
    fig_fp = os.path.join(fig_dir, "Fig3.pdf")
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')

def plot_PseudoTime_Fig4(args, method = "DGI"):
    datasets = ["imc", "seqfish", "slideseqv2", "osmFISH"]
    plt_setting()
    nrow, ncol = 3 , len(datasets)
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 4, nrow * 4))
    plt.subplots_adjust(wspace=0.5, hspace=0.25, bottom=0.2)
    n_neighbors = 50
    cm = plt.get_cmap('gist_rainbow')
    spatials = [False, True]
    titles = ["%s by %s SP-" % ("Pseudotime", method),
              "%s by %s SP+" % ("Pseudotime", method)]

    for did, dataset in enumerate(datasets):
        args.dataset = dataset
        feature_dir = os.path.join(args.dataset_dir, "features")
        expr_dir = os.path.join(args.dataset_dir, args.dataset)
        ax = axs[0, did]
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        if dataset == "osmFISH":
            expr_fp = os.path.join(expr_dir, "osmFISH_SScortex_mouse_all_cells.loom")
            x, y, ground_truth_clusters, region_colors = get_osmFISH(expr_fp)
            plotOSMFISH_ground_truth(ax, x, y, ground_truth_clusters, region_colors)
        else:
            x, y, ground_truth_clusters, cell_type_strs, cell_type_colors, colors = get_squipy(args, dataset)
            plot_squidpy_ground_truth(ax, x, y, ground_truth_clusters, cell_type_strs, cell_type_colors,
                                      colors)
        ax.set_title(dataset, fontsize=14, weight='bold')

        for sid, spatial in enumerate(spatials):
            ax = axs[sid + 1, did]
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

            args.spatial = spatial
            args.expr_name = "50_penalty1" if spatial else "default"
            args.arch = method
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
                    st = ax.scatter(x, y, s=1, c=adata.obs['dpt_pseudotime'], cmap=cm)
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    clb = fig.colorbar(st, cax=cax)
                    clb.ax.set_ylabel("pseudotime", labelpad=10, rotation=270, fontsize=8, weight='bold')

                except ValueError as e:
                    print(e)
            if did == 0:
                ax.set_ylabel("%s" % titles[sid], fontsize=12, weight='bold')
            if dataset in SQUIDPY_DATASETS:
                ax.invert_yaxis()

    fig_dir = os.path.join(args.figure_dir)
    mkdir(fig_dir)
    fig_fp = os.path.join(fig_dir, "Fig4.pdf")
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')

def plot_UMAP_Comparison(args):
    datasets = SPATIAL_LIBD_DATASETS
    plt_setting()
    umap_fig_dir = os.path.join(args.figure_dir, "umap_comparison")
    mkdir(umap_fig_dir)
    feature_dir = os.path.join(args.dataset_dir, "features")
    methods = ["Seurat", "DGI", "DGI_with_spatial"]
    titles = ["Seurat", "DGI", "DGI+SP"]

    nrow, ncol = (1, 3)
    for dataname in datasets:
        args.dataset = dataname
        fig, axs = plt.subplots(nrow, ncol, figsize=(5 * ncol, 5 * nrow))
        cm = plt.get_cmap('gist_rainbow')
        print(f"Plotting UMAP for {dataname}")
        info_df = pd.read_csv(os.path.join(args.dataset_dir, dataname, "spot_info.csv"))
        annotations = info_df["layer_guess"].values.astype(str)
        cluster_names = list(np.unique(annotations))
        for idx, method in enumerate(methods):
            ax = axs[idx]
            if idx == 0:
                seurat_fp = os.path.join(args.dataset_dir, dataname, "Seurat", "seurat.PCs.tsv")
                adata = sc.read_csv(seurat_fp, delimiter="\t")
            else:
                args.spatial = True if idx == 2 else False
                args.expr_name = "p1" if args.spatial else "default"
                args.arch = "DGI"
                name = get_expr_name(args)
                feature_fp = os.path.join(feature_dir, "%s.tsv" % name)
                adata = sc.read_csv(feature_fp, delimiter="\t", first_column_names=None)

            sc.pp.neighbors(adata, n_neighbors=10)
            sc.tl.umap(adata)
            umap_positions = adata.obsm["X_umap"]
            for cid, cluster in enumerate(cluster_names):
                if cluster != "nan":
                    umap_sub = umap_positions[annotations == cluster]
                    color = cm(1. * cid / (len(cluster_names) + 1))
                    ax.scatter(umap_sub[:, 0], umap_sub[:, 1], s=2, color=color, label=cluster)
            ax.set_title(titles[idx], fontsize=12, weight='bold')
            if idx == 0:
                ax.legend()
        fig_fp = os.path.join(umap_fig_dir, f"{dataname}.pdf")
        plt.savefig(fig_fp, dpi=300)
        plt.close('all')


def plot_PAGA_Comparison(args):
    datasets = SPATIAL_LIBD_DATASETS
    plt_setting()
    paga_fig_dir = os.path.join(args.figure_dir, "paga_comparison")
    mkdir(paga_fig_dir)
    feature_dir = os.path.join(args.dataset_dir, "features")
    methods = ["Seurat", "stLearn", "DGI", "DGI_with_spatial"]
    titles = ["Seurat", "stLearn", "DGI", "DGI+SP"]

    nrow, ncol = (1, 4)
    for dataname in datasets:
        args.dataset = dataname
        fig, axs = plt.subplots(nrow, ncol, figsize=(3.75 * ncol, 3.75 * nrow))
        plt.subplots_adjust(wspace=0.25, hspace=0.3, bottom=0.2)
        print(f"Plotting PAGA for {dataname}")
        info_df = pd.read_csv(os.path.join(args.dataset_dir, dataname, "spot_info.csv"))
        annotations = info_df["layer_guess"].values.astype(str)
        annotations_indices = annotations != "nan"
        annotations = annotations[annotations_indices]
        for idx, method in enumerate(methods):
            ax = axs[idx]
            if idx < 2:
                if idx == 0:
                    fp = os.path.join(args.dataset_dir, dataname, "Seurat", "seurat.PCs.tsv")
                else:
                    fp = os.path.join(args.dataset_dir, dataname, "stLearn", "PCs.tsv")
                adata = sc.read_csv(fp, delimiter="\t")
            else:
                args.spatial = True if idx == 3 else False
                args.expr_name = "p1" if args.spatial else "default"
                args.arch = "DGI"
                name = get_expr_name(args)
                feature_fp = os.path.join(feature_dir, "%s.tsv" % name)
                adata = sc.read_csv(feature_fp, delimiter="\t", first_column_names=None)
            adata = adata[annotations_indices, :]
            sc.pp.neighbors(adata, n_neighbors=10)
            sc.tl.umap(adata)
            adata.obs['ground_truth_groups'] = pd.Categorical(annotations)
            sc.tl.paga(adata, groups="ground_truth_groups")
            sc.pl.paga(adata, threshold=0.03, show=False, ax=ax)
            ax.set_title(titles[idx], fontsize=16, weight='bold')
        fig_fp = os.path.join(paga_fig_dir, f"{dataname}.jpg")
        plt.savefig(fig_fp, dpi=300)
        plt.close('all')

def plot_pseudo_time_comparison_with_seurat(args, n_neighbors=50, deep_method="DGI"):
    datasets = SPATIAL_LIBD_DATASETS + VISIUM_DATASETS
    plt_setting()
    cm = plt.get_cmap('gist_rainbow')
    feature_dir = os.path.join(args.dataset_dir, "features")
    fig_dir = os.path.join(args.figure_dir, "pseudotime_comparison_with_seurat")
    mkdir(fig_dir)
    dataset_dir = args.dataset_dir
    for dataset in datasets:
        args.dataset = dataset
        expr_dir = os.path.join(dataset_dir, dataset)
        fig_fp = os.path.join(fig_dir, f"{dataset}.pdf")
        nrow, ncol = (1, 4) if dataset in SPATIAL_LIBD_DATASETS else (1, 3)
        fig, axs = plt.subplots(nrow, ncol, figsize=(5 * ncol, 5 * nrow))
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
            img = plt.imread(os.path.join(dataset_dir, dataset, "%s_tissue_lowres_image.png" % library_id))
            ax.imshow(img)

            unique_clusters = np.unique(clusters)
            for cid, cluster in enumerate(unique_clusters[:-1]):
                color = cm(1. * cid / (len(unique_clusters) + 1))
                ind = clusters == cluster
                ax.scatter(x[ind], y[ind], s=1, color=color, label= cluster)
            ax.set_title("Ground Truth", fontsize=12, weight='bold')
            ax.legend()
        offset = 0
        if dataset in SPATIAL_LIBD_DATASETS:
            offset = 1
            seurat_fp = os.path.join(args.dataset_dir, dataset, "Seurat", "seurat.PCs.tsv")
            adata = sc.read_csv(seurat_fp, delimiter="\t")
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')
            sc.tl.umap(adata)
            sc.tl.leiden(adata, resolution=.8)
            sc.tl.paga(adata)
            distances = distance_matrix(adata.X, adata.X)
            adata.uns['iroot'] = np.argmax(distances.sum(axis=1))
            sc.tl.diffmap(adata)
            sc.tl.dpt(adata)

            ax = axs[1]
            ax.axis('off')
            if dataset not in ["osmFISH"] + SQUIDPY_DATASETS:
                ax.imshow(img)
            ax.grid(False)
            st = ax.scatter(x, y, s=1, c=adata.obs['dpt_pseudotime'], cmap=cm)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            clb = fig.colorbar(st, cax=cax)
            clb.ax.set_ylabel("pseudotime", labelpad=10, rotation=270, fontsize=8, weight='bold')
            ax.set_title("Seurat", fontsize=12, weight='bold')
            if dataset in SQUIDPY_DATASETS:
               ax.invert_yaxis()

        spatials = [False, True]
        titles = ["%s" % deep_method, "%s + SP" % deep_method]

        for sid, spatial in enumerate(spatials):
            args.spatial = spatial
            args.expr_name = "p1" if spatial else "default"
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
                try:
                    ax = axs[sid + 1 + offset]
                    ax.axis('off')
                    ax.grid(False)
                    if dataset not in ["osmFISH"] + SQUIDPY_DATASETS:
                        ax.imshow(img)
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
            else:
                return
        plt.savefig(fig_fp, dpi=300)
        plt.close('all')
        print("Plotted %s" % fig_fp)

def plot_top_n_gene_expression_with_max_corr_to_pseudotime(args, n_neighbors=50, deep_method="DGI", n_top_gene = 6):
    datasets = SPATIAL_LIBD_DATASETS + VISIUM_DATASETS
    plt_setting()
    cm = plt.get_cmap('gist_rainbow')
    cm2 = plt.get_cmap('magma')
    feature_dir = os.path.join(args.dataset_dir, "features")
    fig_dir = os.path.join(args.figure_dir, "gene_expr_with_top_corr_to_pseudotime")
    mkdir(fig_dir)
    dataset_dir = args.dataset_dir
    for dataset in datasets:
        args.dataset = dataset
        expr_dir = os.path.join(dataset_dir, dataset)
        fig_fp = os.path.join(fig_dir, f"{dataset}.pdf")
        nrow, ncol = ((3 + n_top_gene)//3, 3)
        fig, axs = plt.subplots(nrow, ncol, figsize=(5 * ncol, 5 * nrow))
        ax = axs[0][0]

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

            library_id = dataset.split("_")[-1]
            img = plt.imread(os.path.join(dataset_dir, dataset, "%s_tissue_lowres_image.png" % library_id))
            ax.imshow(img)

            clusters = info_df["layer_guess_reordered"].values.astype(str)
            unique_clusters = np.unique(clusters)
            for cid, cluster in enumerate(unique_clusters[:-1]):
                color = cm(1. * cid / (len(unique_clusters) + 1))
                ind = clusters == cluster
                ax.scatter(x[ind], y[ind], s=1, color=color, label=cluster)
            ax.set_title("Ground Truth", fontsize=12, weight='bold')
            ax.legend()

        pseudotimes = None
        spatials = [False, True]
        titles = ["%s" % deep_method, "%s + SP" % deep_method]
        for sid, spatial in enumerate(spatials):
            args.spatial = spatial
            args.expr_name = "p1" if spatial else "default"
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
                pseudotimes = adata.obs['dpt_pseudotime']
                try:
                    ax = axs[0][sid + 1]
                    ax.axis('off')
                    ax.grid(False)
                    if dataset not in ["osmFISH"] + SQUIDPY_DATASETS:
                        ax.imshow(img)
                    st = ax.scatter(x, y, s=1, c=pseudotimes, cmap=cm)
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    clb = fig.colorbar(st, cax=cax)
                    clb.ax.set_ylabel("pseudotime", labelpad=10, rotation=270, fontsize=8, weight='bold')
                    ax.set_title("%s" % titles[sid], fontsize=12, weight='bold')
                    if dataset in SQUIDPY_DATASETS:
                        ax.invert_yaxis()
                except ValueError as e:
                    print(e)
            else:
                return
        if dataset in SPATIAL_LIBD_DATASETS:
            expr_dir = os.path.join(dataset_dir, dataset)
            adata = sc.read_10x_mtx(expr_dir)
        else:
            adata = sc.datasets.visium_sge(dataset, include_hires_tiff=False)
        adata.var_names_make_unique(join="-")
        sc.pp.filter_genes(adata, min_counts=1)  # only consider genes with more than 1 count
        sc.pp.normalize_per_cell(adata, key_n_counts='n_counts_all', min_counts=0)  # normalize with total UMI count per cell
        filter_result = sc.pp.filter_genes_dispersion(adata.X, flavor='cell_ranger', log=False)# select highly-variable genes
        adata = adata[:, filter_result.gene_subset]  # subset the genes
        sc.pp.normalize_per_cell(adata, min_counts=0)  # renormalize after filtering
        sc.pp.log1p(adata)  # log transform: adata.X = log(adata.X + 1)
        genes = adata.var_names
        cells = adata.obs_names
        if type(adata.X).__module__ != np.__name__:
            expr = adata.X.todense()
        else:
            expr = adata.X
        for i in range(expr.shape[0]):
            expr[i, :] = expr[i, :] / np.max(expr[i, :])
        correlations = [np.corrcoef(expr[:, i].flatten(), pseudotimes)[0][1] for i in range(len(genes))]
        sorted_corr_indices = sorted(range(len(genes)), key= lambda x:abs(correlations[x]), reverse=True)

        for idx in range(n_top_gene):
            row, col = (idx // 3) + 1, idx % 3
            ax = axs[row][col]
            gene_idx = sorted_corr_indices[idx]
            gene_name = genes[gene_idx]
            expr_corr = correlations[gene_idx]
            ax.axis('off')
            ax.grid(False)
            if dataset not in ["osmFISH"] + SQUIDPY_DATASETS:
                ax.imshow(img)
            st = ax.scatter(x, y, s=1, c=list(expr[:, gene_idx].flatten()), cmap=cm)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            clb = fig.colorbar(st, cax=cax)
            clb.ax.set_ylabel("expr", labelpad=10, rotation=270, fontsize=8, weight='bold')
            ax.set_title(f"#{idx + 1}:{gene_name}, Corr:%.2f" % expr_corr, fontsize=12, weight='bold')

        plt.savefig(fig_fp, dpi=300)
        plt.close('all')
        print("Plotted %s" % fig_fp)

def plot_pseudotime_ordering_hiearchical_clustering(args, n_neighbors=50, deep_method="DGI"):
    datasets = VISIUM_DATASETS + SPATIAL_LIBD_DATASETS
    plt_setting()
    sns.set_theme(color_codes=True)
    cm = plt.get_cmap('gist_rainbow')
    feature_dir = os.path.join(args.dataset_dir, "features")
    fig_dir = os.path.join(args.figure_dir, "pseudotime_ordering_hiearchical_clustering")
    mkdir(fig_dir)
    dataset_dir = args.dataset_dir
    for dataset in datasets:
        args.dataset = dataset
        fig_fp = os.path.join(fig_dir, f"{dataset}.pdf")
        args.spatial = True
        args.expr_name = "p1"
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
            pseudotimes = adata.obs['dpt_pseudotime']

        if dataset in SPATIAL_LIBD_DATASETS:
            expr_dir = os.path.join(dataset_dir, dataset)
            adata = sc.read_10x_mtx(expr_dir)
        else:
            adata = sc.datasets.visium_sge(dataset, include_hires_tiff=False)
        adata.var_names_make_unique(join="-")
        sc.pp.filter_genes(adata, min_counts=1)  # only consider genes with more than 1 count
        sc.pp.normalize_per_cell(adata, key_n_counts='n_counts_all', min_counts=0)  # normalize with total UMI count per cell
        filter_result = sc.pp.filter_genes_dispersion(adata.X, flavor='cell_ranger', log=False)# select highly-variable genes
        adata = adata[:, filter_result.gene_subset]  # subset the genes
        sc.pp.normalize_per_cell(adata, min_counts=0)  # renormalize after filtering
        sc.pp.log1p(adata)  # log transform: adata.X = log(adata.X + 1)
        genes = adata.var_names
        cells = adata.obs_names
        if type(adata.X).__module__ != np.__name__:
            expr = adata.X.todense()
        else:
            expr = adata.X
        for i in range(expr.shape[0]):
            expr[i, :] = expr[i, :] / np.max(expr[i, :])
        correlations = [np.corrcoef(expr[:, i].flatten(), pseudotimes)[0][1] for i in range(len(genes))]
        sorted_corr_indices = sorted(range(len(genes)), key=lambda x: abs(correlations[x]), reverse=True)
        sorted_cell_indices = sorted(range(len(cells)), key=lambda x: pseudotimes[x])
        cells = cells[sorted_cell_indices]
        n_top_corr_gene = 50
        top_sorted_corr_indices = sorted_corr_indices[:n_top_corr_gene]
        genes = genes[top_sorted_corr_indices]
        print(dataset)
        print(genes)
        expr = expr[:, top_sorted_corr_indices]
        expr = expr[sorted_cell_indices, :].transpose()
        df = pd.DataFrame(expr, index=genes, columns=cells)
        sns.clustermap(df, col_cluster=False, cmap="rocket_r")
        plt.savefig(fig_fp, dpi=150)
        plt.close('all')
        print("Plotted %s" % fig_fp)

if __name__ == "__main__":
    mpl.use('macosx')
    args = get_args()
    # plot_UMAP_Comparison(args)
    plot_PAGA_Comparison(args)
    #plot_pseudotime_ordering_hiearchical_clustering(args)
    #plot_top_n_gene_expression_with_max_corr_to_pseudotime(args)
    # plot_pseudo_time_comparison_with_seurat(args)
    # plot_SPATIAL_LIBD_FIG2(args, SPATIAL_LIBD_DATASETS)
    # plot_PseudoTime_Fig4(args)
    # comparison = True
    # deep_method = "DGI"
    # plot_figures(args, dataset= "osmFISH", deep_method=deep_method, comparison=comparison)
    # plot_figures(args, dataset="seqfish", deep_method=deep_method, resolution="0.6", expr_name="50_penalty1",comparison=comparison)
    # plot_figures(args, dataset="seqfish", deep_method="VASC", resolution="0.6", expr_name="50_penalty1",comparison=comparison)
    # plot_figures(args, dataset="slideseqv2", deep_method="DGI", resolution="0.6", expr_name="50_penalty1",comparison=comparison)
    # plot_figures(args, dataset="slideseqv2", deep_method="VASC", resolution="0.6", expr_name="50_penalty1",comparison=comparison)
    # plot_figures(args, dataset="imc", deep_method=deep_method, resolution="0.6", expr_name="100_penalty1",
    #              comparison=comparison)
    # expr_name = "p1" if deep_method == "DGI" else "500_penalty1"
    # resolution = "0.2" if deep_method == "DGI" else "0.6"
    # plot_figures(args, dataset="V1_Breast_Cancer_Block_A_Section_1", deep_method=deep_method, resolution="0.6", expr_name=expr_name,comparison=comparison)
    # plot_figures(args, dataset="V1_Mouse_Brain_Sagittal_Anterior", deep_method=deep_method, resolution="0.6", expr_name=expr_name,comparison=comparison)
    # plot_figures(args, dataset="V1_Mouse_Brain_Sagittal_Posterior", deep_method=deep_method, resolution="0.6", expr_name=expr_name,comparison=comparison)
    # plot_figures(args, dataset="Spatial_LIBD_151507", deep_method=deep_method, resolution=resolution, expr_name=expr_name, n_neighbors=100, comparison=comparison)
    # plot_figures(args, dataset="Spatial_LIBD_151673", deep_method=deep_method, resolution=resolution, expr_name=expr_name, n_neighbors=100, comparison=comparison)
    # plot_figures(args, dataset="Spatial_LIBD_151671", deep_method=deep_method, resolution=resolution, expr_name=expr_name, n_neighbors=100, comparison=comparison)



