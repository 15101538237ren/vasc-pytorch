# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scanpy as sc
import squidpy as sq
from scipy.spatial import distance
from somde import SomNode
from sklearn.neighbors import kneighbors_graph
from models.alpha import graph_alpha

SPATIAL_N_FEATURE_MAX = 1.0
SPATIAL_THRESHOLD = 0.5
FEATURE_THRESHOLD = 0.5
VISIUM_DATASETS = [
    "V1_Mouse_Brain_Sagittal_Posterior",
    "V1_Breast_Cancer_Block_A_Section_1",
        "Targeted_Visium_Human_Cerebellum_Neuroscience", "Parent_Visium_Human_Cerebellum", "Targeted_Visium_Human_BreastCancer_Immunology","Targeted_Visium_Human_OvarianCancer_Pan_Cancer",
        "Targeted_Visium_Human_OvarianCancer_Immunology"
]

#
# , "V1_Breast_Cancer_Block_A_Section_2",
#         "V1_Human_Heart", "V1_Human_Lymph_Node",  "V1_Mouse_Brain_Sagittal_Posterior",
#         "V1_Mouse_Brain_Sagittal_Posterior_Section_2", "V1_Mouse_Brain_Sagittal_Anterior",
#         "V1_Mouse_Brain_Sagittal_Anterior_Section_2",
# "V1_Human_Brain_Section_2",
#         "V1_Adult_Mouse_Brain_Coronal_Section_1", "V1_Adult_Mouse_Brain_Coronal_Section_2",
#         "Targeted_Visium_Human_SpinalCord_Neuroscience", "Parent_Visium_Human_SpinalCord",
#         "Targeted_Visium_Human_Glioblastoma_Pan_Cancer", "Parent_Visium_Human_Glioblastoma",
#          "Parent_Visium_Human_BreastCancer",
#         "Parent_Visium_Human_OvarianCancer", "Targeted_Visium_Human_ColorectalCancer_GeneSignature",
#         "Parent_Visium_Human_ColorectalCancer", "V1_Mouse_Kidney",

SQUIDPY_DATASETS = ["seqfish", "imc"]
SPATIAL_LIBD_DATASETS = ["Spatial_LIBD_%s" % item for item in ["151671", "151673"]]#"151507", "151671", "151673"]]#, "151509", "151510", "151669", "151670", "151671", "151673", "151674", "151675"]]#, "151672", "151676", "151507"]]#["151507"]]#

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

def get_spatial_coords(args):
    dataset_dir = args.dataset_dir
    dataset = args.dataset
    if dataset in ["Kidney", "Liver"]:
        coord_fp = os.path.join(dataset_dir, dataset, "%s.idx" % dataset)
        coords = pd.read_csv(coord_fp, header=False, index_col=0).values[:, 1:]
        return coords
    elif dataset in SQUIDPY_DATASETS:
        adata = get_squidpy_data(dataset)
    elif dataset in SPATIAL_LIBD_DATASETS:
        expr_dir = os.path.join(dataset_dir, dataset)
        adata = sc.read_10x_mtx(expr_dir)
        coord_fp = os.path.join(expr_dir, "spatial_coords.csv")
        coord_df = pd.read_csv(coord_fp).values.astype(float)
        adata.obsm['spatial'] = coord_df
    else:
        adata = sc.datasets.visium_sge(dataset)
    if args.SVGene:
        SVGene_fp = os.path.join(dataset_dir, dataset, "SVGene_somde.csv")
        if os.path.exists(SVGene_fp):
            result = pd.read_csv(SVGene_fp)
        else:
            som = SomNode(adata.obsm['spatial'], 14)
            df = pd.DataFrame(adata.X.toarray().T.astype(float), index=adata.var_names.astype(str),
                              columns=adata.obs_names.astype(str))
            ndf, ninfo = som.mtx(df)
            nres = som.norm()
            result, SVnum = som.run()
            result.to_csv(SVGene_fp)
        adata = adata[:, result[result.qval < 0.05].index]
        sc.pp.normalize_per_cell(adata, key_n_counts='n_counts_all', min_counts=0)  # normalize with total UMI count per cell
    else:
        sc.pp.filter_genes(adata, min_counts=1)  # only consider genes with more than 1 count
        sc.pp.normalize_per_cell(adata, key_n_counts='n_counts_all', min_counts=0)  # normalize with total UMI count per cell
        filter_result = sc.pp.filter_genes_dispersion(adata.X, flavor='cell_ranger', n_top_genes=args.n_top_genes,
                                                      log=False)  # select highly-variable genes
        adata = adata[:, filter_result.gene_subset]  # subset the genes
    sc.pp.log1p(adata)  # log transform: adata.X = log(adata.X + 1)
    coords = adata.obsm['spatial']
    return coords

def get_squidpy_data(dataset):
    if dataset == "seqfish":
        adata = sq.datasets.seqfish()
    else:
        adata = sq.datasets.imc()
    return adata

def get_data(args):
    dataset_dir = args.dataset_dir
    dataset = args.dataset
    graph_A = None
    if dataset in ["Kidney", "Liver"]:
        expr_fp = os.path.join(dataset_dir, dataset, "%s.count.csv" % dataset)
        expr_df = pd.read_csv(expr_fp, header=False, index_col=0)
        expr = expr_df.values.T
        genes, cells = expr_df.index.tolist(), list(expr_df.columns.values)

        coord_fp = os.path.join(dataset_dir, dataset, "%s.idx" % dataset)
        coords = pd.read_csv(coord_fp, header=0, index_col=0).values[:, 1:]
        cut = estimate_cutoff_knn(coords, k=args.knn_n_neighbors)
        graph_A = graph_alpha(coords, cut=cut, n_layer=args.alpha_n_layer, draw=False)
        spatial_dists = distance.cdist(coords, coords, 'euclidean')
        spatial_dists = (spatial_dists/np.max(spatial_dists)) * SPATIAL_N_FEATURE_MAX
        expr[expr < 0] = 0.0
        if args.log:
            expr = np.log2(expr + 1)
        if args.scale:
            for i in range(expr.shape[0]):
                expr[i, :] = expr[i, :] / np.max(expr[i, :])
        return expr, genes, cells, spatial_dists, graph_A
    elif dataset == "drosophila":
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
        adata = sc.datasets.visium_sge(dataset)

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
    genes = adata.var_names
    cells = adata.obs_names
    sc.pp.normalize_per_cell(adata, min_counts=0)  # renormalize after filtering
    sc.pp.log1p(adata)  # log transform: adata.X = log(adata.X + 1)
    if type(adata.X).__module__ != np.__name__:
        expr = adata.X.todense()
    else:
        expr = adata.X
    if args.scale:
        for i in range(expr.shape[0]):
            expr[i, :] = expr[i, :] / np.max(expr[i, :])
    coords = adata.obsm['spatial']
    if args.arch != "VASC":
        cut = estimate_cutoff_knn(coords, k=args.knn_n_neighbors)
        graph_A = graph_alpha(coords, cut=cut, n_layer=args.alpha_n_layer, draw=False)
    spatial_dists = distance.cdist(coords, coords, 'euclidean')
    spatial_dists = (spatial_dists / np.max(spatial_dists)) * SPATIAL_N_FEATURE_MAX
    return expr, genes, cells, spatial_dists, graph_A

def loss_function(recon_x, x, mu, log_var, spatial_distances, args):
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
    if args.spatial:
        # p_nominator = spatial_distances.cuda()#torch.exp(-torch.div(spatial_distances, args.sigma_sq)).cuda()
        # nrow, ncol = spatial_distances.shape
        # mask = torch.eye(nrow, dtype=torch.bool).cuda()
        # p_nominator = p_nominator[~mask].view(-1, ncol - 1)
        # p_dominator = torch.transpose(torch.broadcast_to(torch.sum(p_nominator, 1).cuda(), (nrow - 1, ncol)), 0, 1)
        # pji = torch.div(p_nominator, p_dominator)
        #
        f_dists = torch.cdist(mu, mu, p=2).cuda() # feature distances
        f_dists = torch.mul(torch.div(f_dists, torch.max(f_dists)), SPATIAL_N_FEATURE_MAX)
        # q_nominator = f_dists.cuda()#torch.exp(-f_dists).cuda()
        # q_nominator = q_nominator[~mask].view(-1, ncol - 1)
        # q_dominator = torch.transpose(torch.broadcast_to(torch.sum(q_nominator, 1).cuda(), (nrow - 1, ncol)), 0, 1)
        # qji = torch.div(q_nominator, q_dominator)
        #
        # kl_sf = torch.mul(torch.sum(torch.mul(pji, torch.log(torch.div(pji, qji)))), 0.2)

        dist_penalty_1 = torch.div(torch.sum(torch.mul(SPATIAL_N_FEATURE_MAX - f_dists, spatial_distances)), n * n)
        dist_penalty_2 = torch.div(torch.sum(torch.mul(SPATIAL_N_FEATURE_MAX - spatial_distances, f_dists)), n * n)
        # dist_penalty_3 = torch.div(torch.sum(torch.mul(spatial_distances, f_dists)), n * n)
        #
        # diagnal_mask = torch.eye(f_dists.shape[0], dtype=torch.bool).cuda()
        # spatial_closed_mask = spatial_distances.le(0.25)
        # feature_close_mask = f_dists.le(0.25)

        # spatial_far_mask = spatial_distances.ge(0.7)
        # feature_far_mask = f_dists.ge(0.6)

        # feature_close_spatial_far = torch.logical_and(torch.logical_and(spatial_far_mask, feature_close_mask), ~diagnal_mask)
        # dist_1 = torch.mul(SPATIAL_N_FEATURE_MAX - f_dists, spatial_distances)
        # dist_penalty_11 = torch.div(torch.nansum(dist_1[feature_close_spatial_far]), torch.nansum(feature_close_spatial_far))
        # dist_penalty_11 = 0.0 if torch.isnan(dist_penalty_11) else dist_penalty_11

        # spatial_close_feature_far = torch.logical_and(torch.logical_and(spatial_closed_mask, feature_far_mask), ~diagnal_mask)
        # dist_2 = torch.mul(SPATIAL_N_FEATURE_MAX - spatial_distances, f_dists)
        # dist_penalty_22 = torch.div(torch.nansum(dist_2[spatial_close_feature_far]), torch.nansum(spatial_close_feature_far))
        # dist_penalty_22 = 0.0 if torch.isnan(dist_penalty_22) else dist_penalty_22

        # both_closed_mask = torch.logical_and(torch.logical_and(spatial_closed_mask, feature_close_mask), ~diagnal_mask)
        # dist_3 = torch.mul(spatial_distances, f_dists)
        # dist_penalty_33 = torch.div(torch.nansum(dist_3[both_closed_mask]), torch.nansum(both_closed_mask))
        # dist_penalty_33 = 0.0 if torch.isnan(dist_penalty_33) else dist_penalty_33

        dist_penalty = torch.mul(dist_penalty_1, 500)# + torch.mul(dist_penalty_2, -100)# + + torch.mul(dist_penalty_3, -250) + dist_penalty_22*50 #+ dist_penalty_11*20# + dist_penalty_22 * 10 + dist_penalty_33*-500
        #print("1: %.2f, 2: %.2f, 3: %.2f, 4: %.2f, 5: %.2f, 6: %.2f " % (dist_penalty_1*500, dist_penalty_2*200, dist_penalty_3*-250, dist_penalty_11 * 50, dist_penalty_22 * 20, dist_penalty_33*-1000))
        #print("BCE Loss:%.2f, KLD: %.2f, kl_sf:%.2f" % (BCE, KLD, kl_sf))
        # print("BCE Loss:%.2f, kl_sf:%.2f" % (BCE, kl_sf))
        return VAE_Loss + dist_penalty
    else:
        return VAE_Loss

def train_in_batch(model, device, train_loader, graph_A, spatial_dists, args,
          torch_seed=None, python_seed=None, numpy_seed=None):
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
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
        model.z_mean.register_forward_hook(get_activation('z_mean'))
    elif args.arch == "DGI":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = args.epochs
    min_loss = np.inf
    patience = 0
    if args.arch != "VASC":
        edge_list = sparse_mx_to_torch_edge_list(graph_A)
        edge_list = edge_list.to(device)
    for epoch in range(epochs):
        train_loss = 0
        if epoch % 150 == 0 and args.annealing and args.arch == "VASC":
            tau = max(args.tau0 * np.exp(-args.anneal_rate * epoch), args.min_tau)
            print("tau = %.2f" % tau)
        for batch_idx, data in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            data = data[0].cuda()
            sidx, eidx = args.batch_size * batch_idx, args.batch_size * (batch_idx + 1)
            s_dists = spatial_dists[sidx: eidx, sidx: eidx].cuda()
            if args.arch == "VASC":
                recon, mu, log_var = model.forward(data, tau)
                loss = loss_function(recon, data, mu, log_var, s_dists, args)
            elif args.arch == "DGI":
                pos_z, neg_z, summary = model(data, edge_list)
                loss = model.loss(pos_z, neg_z, summary)
            else:
                z = model.encode(data, edge_list)
                loss = model.recon_loss(z, edge_list)
                if args.arch == 'VGAE':
                    loss = loss + (1 / data.shape[0]) * model.kl_loss()

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        min_loss = min(train_loss, min_loss)
        if train_loss > min_loss:
            patience += 1
        else:
            patience = 0
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
        z = model.encode(data, edge_list)
        return z.cpu().detach().numpy()

def train(model, device, X, graph_A, spatial_dists, args,
          torch_seed=None, python_seed=None, numpy_seed=None):
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
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
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
            recon, mu, log_var = model.forward(data, tau)
            loss = loss_function(recon, data, mu, log_var, spatial_dists, args)
        elif args.arch == "DGI":
            pos_z, neg_z, summary = model(data, edge_list)
            loss = model.loss(pos_z, neg_z, summary)
        else:
            z = model.encode(data, edge_list)
            loss = model.recon_loss(z, edge_list)
            if args.arch == 'VGAE':
                loss = loss + (1 / data.shape[0]) * model.kl_loss()

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

def get_expr_name(args, idx = 0):
    idx_suf = "" if idx == 0 else "_%d" % idx
    method_name = "_%s" % args.arch if args.arch != "VASC" else ""
    if args.spatial:
        name = "%s%s_%s_with_spatial%s" % (args.dataset, method_name, args.expr_name, idx_suf)
    else:
        name = "%s%s%s" % (args.dataset, method_name,idx_suf)
    return name

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