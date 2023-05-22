# -*- coding: utf-8 -*-
import os
import random
import loompy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scanpy as sc
import torch.nn.functional as F
from scipy.spatial import distance
import matplotlib.pyplot as plt
from somde import SomNode
from sklearn.neighbors import kneighbors_graph
from models.alpha import graph_alpha

SPATIAL_N_FEATURE_MAX = 1.0
SPATIAL_THRESHOLD = 0.5
FEATURE_THRESHOLD = 0.5

VISIUM_DATASETS = [
"V1_Breast_Cancer_Block_A_Section_1",
"V1_Mouse_Brain_Sagittal_Anterior",
"V1_Mouse_Brain_Sagittal_Posterior",
"V1_Human_Lymph_Node",
"V1_Adult_Mouse_Brain_Coronal_Section_1",
"Parent_Visium_Human_Cerebellum"
]


VISIUM_DATASETS_DICT = {
    "V1_Mouse_Brain_Sagittal_Anterior" : "Mouse Brain\nSagittal Anterior",
    "V1_Mouse_Brain_Sagittal_Posterior": "Mouse Brain\nSagittal Posterior",
    "V1_Breast_Cancer_Block_A_Section_1": "Human Breast\nCancer Block"
}

# , "V1_Breast_Cancer_Block_A_Section_2",
#         "V1_Human_Heart",  "V1_Mouse_Brain_Sagittal_Posterior",
#         "V1_Mouse_Brain_Sagittal_Posterior_Section_2", "V1_Mouse_Brain_Sagittal_Anterior",
#         "V1_Mouse_Brain_Sagittal_Anterior_Section_2",
#         "Targeted_Visium_Human_Cerebellum_Neuroscience", "Parent_Visium_Human_Cerebellum", "Targeted_Visium_Human_BreastCancer_Immunology","Targeted_Visium_Human_OvarianCancer_Pan_Cancer",
#         "Targeted_Visium_Human_OvarianCancer_Immunology"
# "V1_Human_Brain_Section_2",
#         "V1_Adult_Mouse_Brain_Coronal_Section_1", "V1_Adult_Mouse_Brain_Coronal_Section_2",
#         "Targeted_Visium_Human_SpinalCord_Neuroscience", "Parent_Visium_Human_SpinalCord",
#         "Targeted_Visium_Human_Glioblastoma_Pan_Cancer", "Parent_Visium_Human_Glioblastoma",
#          "Parent_Visium_Human_BreastCancer",
#         "Parent_Visium_Human_OvarianCancer", "Targeted_Visium_Human_ColorectalCancer_GeneSignature",
#         "Parent_Visium_Human_ColorectalCancer", "V1_Mouse_Kidney",

SQUIDPY_DATASETS = ["imc", "seqfish", "slideseqv2"]#]#, "four_i"]#
SPATIAL_LIBD_DATASETS = ["Spatial_LIBD_%s" % item for item in
                         ['151507']]
#
# , '151671', '151673', '151508', '151509', '151510',
#                           '151669', '151670', '151672',
#                           '151674', '151675', '151676'

def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def estimate_cutoff_knn(pts, k=10):
    A_knn = kneighbors_graph(pts, n_neighbors=k, mode='distance')
    est_cut = A_knn.sum() / float(A_knn.count_nonzero())
    return est_cut

def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index

def sparse_mx_to_torch_edge_list(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    edge_list = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    return edge_list

def get_squidpy_data(dataset):
    if dataset == "seqfish":
        adata = sq.datasets.seqfish()
    elif dataset == "slideseqv2":
        adata = sq.datasets.slideseqv2()
    elif dataset == "four_i":
        adata = sq.datasets.four_i()
    else:
        adata = sq.datasets.imc()
    return adata

def get_data(args):
    dataset_dir = args.dataset_dir
    dataset = args.dataset
    graph_A = None
    if dataset == "osmFISH":
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

        print('Final shape is: {} genes and {} cells'.format(df_osmfish.shape[0], df_osmfish.shape[1]))

        # normalize data by total number of molecules per cell and per gene.
        # df_osmfish_totmol = df_osmfish.divide(df_osmfish.sum(axis=1), axis=0) * df_osmfish.shape[
        #     0]  # Corrected for total molecules per gene
        df_osmfish_totmol = df_osmfish.divide(df_osmfish.sum(axis=0), axis=1) * df_osmfish.shape[
            1]  # Corrected for the total per cell
        # Replace NA and Nan with zero:
        df_osmfish_totmol = df_osmfish_totmol.fillna(0)
        non_zero_count_rows = np.where(df_osmfish_totmol.values.T.sum(axis=1) > 0)[0]
        df_osmfish_totmol = df_osmfish_totmol.iloc[:, non_zero_count_rows]

        # Load the cell coordinates into a Pandas Dataframe. Units are pixels
        coordinates = np.stack((ds.ca.X, ds.ca.Y))
        df_coordinates = pd.DataFrame(data=coordinates, index=['X', 'Y'], columns=ds.ca.CellID)
        df_coordinates = df_coordinates.loc[:, include].values
        df_coordinates = df_coordinates[:, non_zero_count_rows]
        expr = np.log(df_osmfish_totmol.values.T + 1.0)
        genes = list(df_osmfish_totmol.index.values)
        cells = list(df_osmfish_totmol.columns.values)
        coords = df_coordinates.T
    else:
        if dataset == "drosophila":
            expr_fp = os.path.join(dataset_dir, dataset, "%s.txt" % dataset)
            expr_df = pd.read_csv(expr_fp, sep="\t", header=0, index_col=0)
            expr = expr_df.values
            cells, genes = expr_df.index.tolist(), list(expr_df.columns.values)
            spatial_dists = torch.from_numpy(np.load(os.path.join(dataset_dir, dataset, "spatial_dist.npy")))
            return expr, genes, cells, spatial_dists, graph_A
        elif dataset in SQUIDPY_DATASETS:
            adata = get_squidpy_data(dataset)
        elif dataset in SPATIAL_LIBD_DATASETS:
            expr_dir = os.path.join(dataset_dir, dataset)
            adata = sc.read_10x_mtx(expr_dir)
            coord_fp = os.path.join(expr_dir,"spatial_coords.csv")
            coord_df = pd.read_csv(coord_fp).values.astype(float)
            adata.obsm['spatial'] = coord_df
        else:
            adata = sc.datasets.visium_sge(dataset, include_hires_tiff=False)
        adata.var_names_make_unique(join="-")
        sc.pp.filter_genes(adata, min_counts=1)  # only consider genes with more than 1 count
        sc.pp.normalize_per_cell(adata, key_n_counts='n_counts_all', min_counts=0)# normalize with total UMI count per cell
        filter_result = sc.pp.filter_genes_dispersion(adata.X, flavor='cell_ranger', log=False)# select highly-variable genes
        adata = adata[:, filter_result.gene_subset]  # subset the genes
        if args.SVGene:
            SVGene_fp = os.path.join(dataset_dir, dataset, "SVGene_somde.csv")
            if os.path.exists(SVGene_fp):
                result = pd.read_csv(SVGene_fp)
            else:
                som = SomNode(adata.obsm['spatial'], 14)
                df = pd.DataFrame(adata.X.toarray().T.astype(float), index=adata.var_names.astype(str), columns=adata.obs_names.astype(str))
                ndf, ninfo = som.mtx(df)
                nres = som.norm()
                result, SVnum = som.run()
                result.to_csv(SVGene_fp)
            adata = adata[:, result[result.qval < 0.05].index]
        sc.pp.normalize_per_cell(adata, min_counts=0)  # renormalize after filtering
        sc.pp.log1p(adata)  # log transform: adata.X = log(adata.X + 1)
        genes = adata.var_names
        cells = adata.obs_names
        if type(adata.X).__module__ != np.__name__:
            expr = adata.X.todense()
        else:
            expr = adata.X
        coords = adata.obsm['spatial']
    if args.scale:
        for i in range(expr.shape[0]):
            expr[i, :] = expr[i, :] / np.max(expr[i, :])

    n_cells = len(cells)
    if n_cells > args.max_cells:
        expr_dir = os.path.join(dataset_dir, dataset)
        mkdir(expr_dir)
        indices_fp = os.path.join(expr_dir, "indices.npy")
        if os.path.exists(indices_fp):
            with open(indices_fp, 'rb') as f:
                indices = np.load(f)
                print("loaded indices successful!")
        else:
            indices = np.random.choice(n_cells, args.max_cells, replace=False)
            with open(indices_fp, 'wb') as f:
                np.save(f, indices)
            print("Saved indices")
        expr = expr[indices, :]
        cells = cells[indices]
        coords = coords[indices, :]
    if args.arch != "VASC":
        cut = estimate_cutoff_knn(coords, k=args.knn_n_neighbors)
        graph_A = graph_alpha(coords, cut=cut, n_layer=args.alpha_n_layer, draw=False)

    spatial_dists = distance.cdist(coords, coords, 'euclidean')
    spatial_dists = (spatial_dists / np.max(spatial_dists)) * SPATIAL_N_FEATURE_MAX
    return expr, genes, cells, spatial_dists, graph_A

def loss_function(recon_x, x, mu, log_var, args):
    n, in_dim = x.shape
    loss = nn.BCELoss(reduction="mean")
    BCE = in_dim * loss(recon_x, x)

    if args.var:
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
    else:
        var_ones = torch.ones(log_var.size()).cuda()
        KLD = -0.5 * torch.sum(1 + 1 - mu.pow(2) - torch.exp(var_ones), dim=-1)
    KLD = torch.mean(KLD)
    VAE_Loss = BCE + KLD
    return VAE_Loss

def train(model, device, X, graph_A, spatial_dists, args,
          torch_seed=None, python_seed=None, numpy_seed=None, p1_coef=0, p2_coef=0):
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    if not torch_seed is None:
        torch.manual_seed(torch_seed)
    if not python_seed is None:
        random.seed(python_seed)
    if not numpy_seed is None:
        np.random.seed(numpy_seed)

    if args.arch == "VASC":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4)
        model.z_mean.register_forward_hook(get_activation('z_mean'))
    elif args.arch == "DGI":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = args.epochs
    min_loss = np.inf
    patience = 0
    data = X.to(device)
    if args.arch != "VASC":
        edge_list = sparse_mx_to_torch_edge_list(graph_A)
        edge_list = edge_list.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        if epoch % 150 == 0 and args.annealing and args.arch == "VASC":
            tau = max(args.tau0 * np.exp(-args.anneal_rate * epoch), args.min_tau)
            print("tau = %.2f" % tau)
        optimizer.zero_grad()

        if args.arch == "VASC":
            recon, z, log_var = model.forward(data, tau)
            loss = loss_function(recon, data, z, log_var, args)
        elif args.arch == "DGI":
            z, neg_z, summary = model(data, edge_list)
            loss = model.loss(z, neg_z, summary)
        else:
            z = model.encode(data, edge_list)
            loss = model.recon_loss(z, edge_list)
            if args.arch == 'VGAE':
                g_kl_loss = (1 / data.shape[0]) * model.kl_loss()
                expr_recon_loss = 0 #model.expr_recon_loss(z, data)
                loss += g_kl_loss# + expr_recon_loss
                print("Total loss: {:.3f}, G kl: {:.3f}, Expr: {:.3f}".format(loss, g_kl_loss, expr_recon_loss))
        if args.spatial:
            n, in_dim = data.shape
            mu = z.to(device)
            f_dists = torch.cdist(mu, mu, p=2)
            f_dists = torch.mul(torch.div(f_dists, torch.max(f_dists)), 1.0).to(device)
            spatial_dists = spatial_dists.to(device)
            penalty_1 = torch.div(torch.sum(torch.mul(1.0 - f_dists, spatial_dists)), n * n).to(device)
            #penalty_2 = torch.div(torch.sum(torch.mul(1.0 - spatial_dists, f_dists)), n * n).to(device)
            loss = loss + p1_coef * penalty_1# + p2_coef * penalty_2
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if train_loss > min_loss:
            patience += 1
        else:
            patience = 0
            min_loss = train_loss
        if epoch % 10 == 1:
            print("Epoch %d/%d" % (epoch + 1, epochs))
            print("Loss:" + str(train_loss))
            # if patience == 0:
            #     torch.save(model.state_dict(), model_fp)
            #     print("Saved model at epoch %d with min_loss: %.0f" % (epoch + 1, min_loss))
        if patience > args.patience and epoch > args.min_stop:
            break

    if args.arch == "VASC":
        model(data, args.min_tau)
        reduced_reprs = activation['z_mean'].detach().cpu().numpy()
        return reduced_reprs
    else:
        edge_list = sparse_mx_to_torch_edge_list(graph_A)
        edge_list = edge_list.to(device)
        if args.arch == "DGI":
            z, _, _ = model(data, edge_list)
        else:
            z = model.encode(data, edge_list)
        return z.cpu().detach().numpy()

def evaluate(model, device, X, graph_A, model_fp, args):
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    model.load_state_dict(torch.load(model_fp))
    print("Load state dict from %s successful!" % model_fp)
    model.eval()
    X = X.cuda()

    if args.arch == "VASC":
        model.z_mean.register_forward_hook(get_activation('z_mean'))
        _ = model(X, args.min_tau)
        reduced_reprs = activation['z_mean'].detach().cpu().numpy()
        return reduced_reprs
    else:
        edge_list = sparse_mx_to_torch_edge_list(graph_A)
        edge_list = edge_list.to(device)
        z = model.encode(X, edge_list)
        return z.cpu().detach().numpy()

def get_osmFISH(expr_fp):
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
    return x, y, ground_truth_clusters, region_colors

def get_squipy(args, dataset):
    adata = get_squidpy_data(dataset)
    coords = adata.obsm['spatial']
    annotation_dict = {"imc": "cell type", "seqfish": "celltype_mapped_refined", "slideseqv2": "cluster"}
    cell_types = adata.obs[annotation_dict[dataset]]
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
    cell_type_colors = list(adata.uns['%s_colors' % annotation_dict[dataset]].astype(str))
    if dataset == "seqfish":
        cm = plt.get_cmap('gist_rainbow')
        colors = np.array([cm(1. * cell_type / (len(cell_type_colors) + 1)) for cell_type in cell_type_ints])
    else:
        colors = np.array([cell_type_colors[item] for item in cell_type_ints])
    return x, y, cell_type_ints, cell_type_strs, cell_type_colors, colors

def generate_expr_name(p1_coef, p2_coef):
    if p1_coef == 0 and p2_coef == 0:
        return "default"
    else:
        if p1_coef == 0:
            return "%d_penalty2" % p2_coef
        elif p2_coef == 0:
            return "%d_penalty1" % p1_coef
        else:
            return "%d_penalty1_%d_panelty2" % (p1_coef, p2_coef)

def get_expr_name(args):
    method_name = "_%s" % args.arch# if args.arch != "VASC" else ""
    if args.spatial:
        name = "%s%s_%s_with_spatial" % (args.dataset, method_name, args.expr_name)
    else:
        if args.expr_name != "default":
            name = "%s%s_%s" % (args.dataset, method_name, args.expr_name)
        else:
            name = "%s%s" % (args.dataset, method_name)
    return name

def load_embeddings(feature_dir, name, mds=False):
    suff = "_MDS_Consensus" if mds else ""
    feature_fp = os.path.join(feature_dir, "%s%s.tsv" % (name, suff))
    X_embed = pd.read_csv(feature_fp, sep="\t", header=None).values
    return X_embed

def save_features(reduced_reprs, feature_dir, name, mds=False):
    suff = "_MDS_Consensus" if mds else ""
    feature_fp = os.path.join(feature_dir, "%s%s.tsv" % (name, suff))
    mkdir(feature_dir)
    np.savetxt(feature_fp, reduced_reprs[:, :], delimiter="\t")
    print("Features saved successful! %s" % feature_fp)

def prepare_cuda(args):
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:%d" % args.gpu if cuda else "cpu")
    if cuda:
        torch.cuda.manual_seed_all(args.seed)
        print("GPU count: %d, using gpu: %d" % (torch.cuda.device_count(), args.gpu))
        torch.cuda.set_device(args.gpu)
    else:
        torch.manual_seed(args.seed)
    return device