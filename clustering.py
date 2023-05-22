# -*- coding:utf-8 -*-
import os
import anndata
import pandas as pd
import scanpy as sc
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from utils.config import get_args
from utils.util import mkdir, get_expr_name, generate_expr_name, save_features, load_embeddings, SPATIAL_LIBD_DATASETS, VISIUM_DATASETS, SQUIDPY_DATASETS
from sklearn import metrics

data_root = '../SEDR/data/DLPFC'
save_root = '../SEDR/output/DLPFC'

def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path

def clustering(args, dataset, feature_dir, expr_name, n_neighbors=100, clustering_method= "leiden", deep_method="VASC"):
    print(f"Clustering {dataset}")
    if deep_method =="VASC" and expr_name != "default":
        spatials = [True]#, True
    elif expr_name != "default":
        spatials = [True]
    else:
        spatials = [False]
    labels_dir = os.path.join(feature_dir, "%s_labels" % (clustering_method))
    mkdir(labels_dir)

    for sid, spatial in enumerate(spatials):
        args.spatial = spatial
        args.expr_name = expr_name
        args.arch = deep_method
        name = get_expr_name(args)
        feature_fp = os.path.join(feature_dir, "%s.tsv" % (name))
        print(f"Feature at {feature_fp}")
        if os.path.exists(feature_fp):
            adata_feat = sc.read_csv(feature_fp, delimiter="\t", first_column_names=None)
            # expr_dir = os.path.join(args.dataset_dir, dataset)
            # coord_fp = os.path.join(expr_dir, "spatial_coords.csv")
            # coord_df = pd.read_csv(coord_fp).values.astype(float)
            # adata_feat.obsm['spatial'] = coord_df
            # adata_feat.uns['spatial'] = coord_df

            sc.pp.neighbors(adata_feat, n_neighbors=n_neighbors, use_rep='X')
            n_clusters = 5 if dataset in ["Spatial_LIBD_%s" % item for item in ['151669', '151670', '151671', '151672']] else 6
            if clustering_method in ["leiden", "louvain"]:
                resolution = res_search_fixed_clus(adata_feat, n_clusters)
                # Neighbor Graph
                if clustering_method == "louvain":
                    sc.tl.louvain(adata_feat, resolution=float(resolution))
                else:
                    sc.tl.leiden(adata_feat, resolution=float(resolution))
                labels = adata_feat.obs[clustering_method].cat.codes
            else:
                kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(adata_feat.X)
                labels = kmeans.labels_

            data_name = dataset.split("_")[-1]
            expriment_name = deep_method + "_with_spatial" if spatial else deep_method
            args.save_path = f'{save_root}/{data_name}/{expriment_name}'
            mk_dir(args.save_path)
            feature_fp = f'{args.save_path}/features.npz'
            np.savez(feature_fp, adata_feat=adata_feat.X)
            print(f"feature saved at {args.save_path}/features.npz")

            df_meta = pd.read_csv(f'{data_root}/{data_name}/metadata.tsv', sep='\t')
            df_meta[expriment_name] = list(labels)
            df_meta.to_csv(f'{args.save_path}/metadata.tsv', sep='\t', index=False)

            # ---------- Load manually annotation ---------------
            df_meta = df_meta[~pd.isnull(df_meta['layer_guess'])]
            ARI = metrics.adjusted_rand_score(df_meta['layer_guess'], df_meta[expriment_name])
            print('===== Project: {} ARI score: {:.3f}'.format(data_name, ARI))

            label_fp = os.path.join(labels_dir, "%s_label_nNeighbor_%d.tsv" % (name, n_neighbors))
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

def gen_dist_based_embedding():
    args = get_args()
    deep_methods = ["VASC"]
    feature_dir = os.path.join(args.dataset_dir, "features")
    datasets = ["osmFISH"]#SPATIAL_LIBD_DATASETS + VISIUM_DATASETS
    p1_coefs = [int(0), int(100)] #int(0),int(0), int(500),, int(0), int(200), int(500), int(500)
    p2_coefs = [int(0), int(0)] #int(0),int(-500), int(0),, int(-100), int(0), int(-100), int(100)
    for pid, p1_coef in enumerate(p1_coefs):
        p2_coef = p2_coefs[pid]
        args.expr_name = generate_expr_name(p1_coef, p2_coef)
        if p1_coef == 0 and p2_coef == 0:
            args.spatial = False
        else:
            args.spatial = True
        for deep_method in deep_methods:
            for dataset in datasets:
                args.dataset = dataset
                args.arch = deep_method
                X_embeds = []
                for i in range(args.n_consensus):
                    name = get_expr_name(args, idx=i)
                    print("Now loading %s" % name)
                    X_embed = load_embeddings(feature_dir, name, mds=False)
                    X_embeds.append(X_embed)

                n_spot = X_embeds[0].shape[0]
                W_consensus = np.zeros([n_spot, n_spot])
                n_sample = 0
                for i in range(len(X_embeds)):
                    if X_embeds[i].shape[0] == n_spot:
                        n_sample += 1.0
                embeds_weights = np.ones(int(n_sample)) / n_sample

                for i in range(len(X_embeds)):
                    if X_embeds[i].shape[0] == n_spot:
                        W = distance_matrix(X_embeds[i], X_embeds[i])
                        W_consensus += W * embeds_weights[i]
                print("STARTING MDS!")
                mds_model = MDS(n_components=args.n_comps_proj, dissimilarity='precomputed', n_jobs=16,
                                random_state=args.seed)
                X_embed = mds_model.fit_transform(W_consensus)
                print("MDS DONE!")
                name = get_expr_name(args)
                save_features(X_embed, feature_dir, name, mds=True)
                print("FEATURE SAVED!")
def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    '''
        arg1(adata)[AnnData matrix]
        arg2(fixed_clus_count)[int]

        return:
            resolution[int]
    '''
    for res in sorted(list(np.arange(0.02, 2, increment)), reverse=False):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
        print("Try resolution %3f found %d clusters: target %d" % (res, count_unique_leiden, fixed_clus_count))
        if count_unique_leiden == fixed_clus_count:
            print("Found resolution: %.3f" % res)
            return res
        elif count_unique_leiden > fixed_clus_count:
            print("Found resolution: %.3f" % (res - increment))
            return res - increment

if __name__ == "__main__":
    #gen_dist_based_embedding()
    args = get_args()
    datasets = SPATIAL_LIBD_DATASETS#SQUIDPY_DATASETS#["osmFISH"]#VISIUM_DATASETS# + SPATIAL_LIBD_DATASETS#
    expr_names = ["default", "p1"]#, "50_penalty1", "250_penalty1", "500_penalty1", "1000_penalty1"]#["default", "100_penalty1", "500_penalty1"]#["500_penalty1", "500_penalty1_200_penalty2", "500_penalty1_-100_penalty2"]  # ["default",
    # "-500_penalty2"], "-100_penalty2", "-1000_penalty2","200_penalty1", "1000_penalty1", "500_penalty1_100_penalty2", "500_penalty1_-50_penalty2", ]#["default", "p1", "p1_0.4p2"]#
    deep_methods = ['DGI'] #['GAE', 'DGI', 'VGAE']
    clustering_methods = ["kmeans"]
    n_neighbors = [50]

    for deep_method in deep_methods:
        for method in clustering_methods:
            for dataset in datasets:
                for expr_name in expr_names:
                    for n_neighbor in n_neighbors:
                        args.dataset = dataset
                        feature_dir = os.path.join(args.dataset_dir, "features")
                        clustering(args, dataset, feature_dir, expr_name, n_neighbors=n_neighbor, clustering_method=method, deep_method=deep_method)
    #
    # # for expr_name in expr_names:
    # #     adjusted_rand_index_for_spatialLIBD(args, expr_name)