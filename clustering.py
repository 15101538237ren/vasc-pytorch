# -*- coding:utf-8 -*-
import os
import anndata
import pandas as pd
import scanpy as sc
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from utils.config import get_args
from utils.util import mkdir, get_expr_name, SPATIAL_LIBD_DATASETS, VISIUM_DATASETS
from sklearn import metrics

def clustering(args, feature_dir, expr_name, n_neighbors=30, resolution="1.0", clustering_method= "leiden"):
    spatials = [False, True]
    labels_dir = os.path.join(feature_dir, "%s_labels_resolution_%s" % (clustering_method, resolution))
    mkdir(labels_dir)

    for sid, spatial in enumerate(spatials):
        args.spatial = spatial
        args.expr_name = expr_name
        name = get_expr_name(args)
        feature_fp = os.path.join(feature_dir, "%s.tsv" % name)
        if os.path.exists(feature_fp):
            adata = sc.read_csv(feature_fp, delimiter="\t", first_column_names=None)

            # Neighbor Graph
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')
            if clustering_method == "louvain":
                sc.tl.louvain(adata, resolution=float(resolution))
            else:
                sc.tl.leiden(adata, resolution=float(resolution))
            labels = adata.obs[clustering_method].cat.codes
            label_fp = os.path.join(labels_dir, "%s_label.tsv" % name)
            np.savetxt(label_fp, labels, fmt='%d', header='', footer='', comments='')
            print("Saved %s succesful!" % label_fp)

def adjusted_rand_index_for_spatialLIBD(args, expr_name):
    sns.set(style="whitegrid")
    feature_suff = "features"
    datasets = SPATIAL_LIBD_DATASETS
    with_spatials = [False, True]
    dim_reds = ["PCA", "UMAP"]
    fields = ["SpatialDE", "SpatialDE_pool", "HVG", "pseudobulk", "markers"]
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plt.subplots_adjust(wspace=0.4, hspace=0.5, bottom=0.5)
    rows = []
    for field in fields:
        for with_spatial in with_spatials:
             for dim in dim_reds:
                for dataset in datasets:
                    expr_dir = os.path.join(args.dataset_dir, dataset)
                    info_df = pd.read_csv(os.path.join(expr_dir, "spot_info.csv"))
                    ground_truth_clusters = info_df["layer_guess_reordered"].values.flatten().astype(str)
                    whole_field = "%s_%s" % (field, dim)
                    if with_spatial:
                        whole_field += "_spatial"
                    clusters = info_df[whole_field].values.flatten().astype(str)
                    ari = metrics.adjusted_rand_score(ground_truth_clusters, clusters)
                    row = ["%s_%s" % (field, "spatial"), dim, ari]
                    rows.append(row)


    spatials = [False, True]
    titles = ["VASC", "VASC + SP"]
    for sid, spatial in enumerate(spatials):
        for dataset in datasets:
            name = dataset if not spatial else "%s_with_spatial" % dataset
            feature_dir = os.path.join(args.dataset_dir, feature_suff)
            labels_dir = os.path.join(feature_dir, "leiden_labels")
            label_fp = os.path.join(labels_dir, "%s_%s_label.tsv" % (name, expr_name))
            clusters = pd.read_csv(label_fp, header=None).values.flatten().astype(str)

            expr_dir = os.path.join(args.dataset_dir, dataset)
            info_df = pd.read_csv(os.path.join(expr_dir, "spot_info.csv"))
            ground_truth_clusters = info_df["layer_guess_reordered"].values.flatten().astype(str)

            ari = metrics.adjusted_rand_score(ground_truth_clusters, clusters)
            row = ["VASC -/+ SP", dim_reds[sid], ari]
            rows.append(row)
    df = pd.DataFrame(rows, columns=['Method', 'Dim Reduction', 'ARI'])
    sns.violinplot(data=df, x="Method", y="ARI", hue="Dim Reduction",
                   split=True, linewidth=1, palette={"PCA": "b", "UMAP": ".85"}, ax= ax)

    fig_dir = os.path.join(args.figure_dir)
    mkdir(fig_dir)
    fig_fp = os.path.join(fig_dir, "ARI_violin_plot_%s.pdf" % (expr_name))
    plt.xticks(rotation=45)
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')

if __name__ == "__main__":
    args = get_args()
    datasets = SPATIAL_LIBD_DATASETS + VISIUM_DATASETS #SPATIAL_LIBD_DATASETS +
    expr_names = ["500_penalty1", "500_penalty1_-100_penalty2"]#["-100_penalty2", "-500_penalty2", "-1000_penalty2", "200_penalty1", "1000_penalty1",
                 # "500_penalty1_100_penalty2", "500_penalty1_-50_penalty2", "500_penalty1_200_penalty2"]
    resolutions = ["1.0", "0.8"]
    clustering_methods = ["leiden"]#, "louvain"
    for resolution in resolutions:
        for method in clustering_methods:
            for dataset in datasets:
                for expr_name in expr_names:
                    args.dataset = dataset
                    feature_dir = os.path.join(args.dataset_dir, "features")
                    clustering(args, feature_dir, expr_name, resolution=resolution, clustering_method=method)

    # for expr_name in expr_names:
    #     adjusted_rand_index_for_spatialLIBD(args, expr_name)