# -*- coding:utf-8 -*-
import os
import anndata
import pandas as pd
import scanpy as sc
import squidpy as sq
import numpy as np
from utils.config import get_args
from utils.util import mkdir, get_spatial_coords, get_squidpy_data, SPATIAL_LIBD_DATASETS, SPATIAL_N_FEATURE_MAX, SQUIDPY_DATASETS
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def k_means_clustering(args, feature_dir, dataset, k=10, linear=True, n_clusters=8):
    spatials = [False, True]
    labels_dir = os.path.join(feature_dir, "cluster_labels")
    mkdir(labels_dir)

    for sid, spatial in enumerate(spatials):
        name = args.dataset if not spatial else "%s_with_spatial" % dataset
        feature_fp = os.path.join(feature_dir, "%s.tsv" % name)
        features = pd.read_csv(feature_fp, header=None, sep="\t").values.astype(float)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        kmeans = KMeans( init="random", n_clusters=n_clusters, n_init=10, max_iter=300, random_state=42)
        kmeans.fit(scaled_features)
        clusters = kmeans.labels_
        label_fp = os.path.join(labels_dir, "%s_kmeans_label.tsv" % name)
        np.savetxt(label_fp, clusters, fmt='%d', header='', footer='', comments='')
        print("Saved %s succesful!" % label_fp)

def leiden_clustering(args, feature_dir, dataset, linear=True, n_neighbors=10):
    spatials = [False, True]
    labels_dir = os.path.join(feature_dir, "cluster_labels")
    mkdir(labels_dir)

    for sid, spatial in enumerate(spatials):
        name = args.dataset if not spatial else "%s_with_spatial" % dataset
        feature_fp = os.path.join(feature_dir, "%s.tsv" % name)
        adata = sc.read_csv(feature_fp, delimiter="\t", first_column_names=None)

        # Neighbor Graph
        sc.pp.neighbors(adata, n_neighbors=n_neighbors)
        sc.tl.leiden(adata)
        labels = adata.obs["leiden"].cat.codes
        label_fp = os.path.join(labels_dir, "%s_leiden_label.tsv" % name)
        np.savetxt(label_fp, labels, fmt='%d', header='', footer='', comments='')
        print("Saved %s succesful!" % label_fp)
if __name__ == "__main__":
    args = get_args()
    linears = [True, False]
    datasets = SPATIAL_LIBD_DATASETS
    for linear in linears:
        for dataset in datasets:
            args.dataset = dataset
            feature_suff = "features_linear" if linear else "features_switch"
            feature_dir = os.path.join(args.dataset_dir, feature_suff)
            leiden_clustering(args, feature_dir, args.dataset, linear=linear)
