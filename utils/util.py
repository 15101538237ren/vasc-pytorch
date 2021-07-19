# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scanpy as sc
import squidpy as sq
from scipy.spatial import distance
from somde import SomNode

SPATIAL_N_FEATURE_MAX = 1.0
SPATIAL_THRESHOLD = 0.5
FEATURE_THRESHOLD = 0.5
VISIUM_DATASETS = [
    "V1_Breast_Cancer_Block_A_Section_1", "V1_Breast_Cancer_Block_A_Section_2",
        "V1_Human_Heart", "V1_Human_Lymph_Node",  "V1_Mouse_Brain_Sagittal_Posterior",
        "V1_Mouse_Brain_Sagittal_Posterior_Section_2", "V1_Mouse_Brain_Sagittal_Anterior",
        "V1_Mouse_Brain_Sagittal_Anterior_Section_2", "V1_Human_Brain_Section_2",
        "V1_Adult_Mouse_Brain_Coronal_Section_1", "V1_Adult_Mouse_Brain_Coronal_Section_2",
        "Targeted_Visium_Human_SpinalCord_Neuroscience", "Parent_Visium_Human_SpinalCord",
        "Targeted_Visium_Human_Glioblastoma_Pan_Cancer", "Parent_Visium_Human_Glioblastoma",
         "Parent_Visium_Human_BreastCancer",
        "Parent_Visium_Human_OvarianCancer", "Targeted_Visium_Human_ColorectalCancer_GeneSignature",
        "Parent_Visium_Human_ColorectalCancer", "V1_Mouse_Kidney",
        "Targeted_Visium_Human_Cerebellum_Neuroscience", "Parent_Visium_Human_Cerebellum", "Targeted_Visium_Human_BreastCancer_Immunology","Targeted_Visium_Human_OvarianCancer_Pan_Cancer",
        "Targeted_Visium_Human_OvarianCancer_Immunology",
]
SQUIDPY_DATASETS = ["seqfish", "imc"]
SPATIAL_LIBD_DATASETS = ["Spatial_LIBD_%s" % item for item in ["151507", "151509", "151672"]]#, "151508", "151509", "151510", "151669", "151670", "151671", "151672", "151673", "151674", "151675", "151676"]]#
def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

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
    if dataset in ["Kidney", "Liver"]:
        expr_fp = os.path.join(dataset_dir, dataset, "%s.count.csv" % dataset)
        expr_df = pd.read_csv(expr_fp, header=False, index_col=0)
        expr = expr_df.values.T
        genes, cells = expr_df.index.tolist(), list(expr_df.columns.values)

        coord_fp = os.path.join(dataset_dir, dataset, "%s.idx" % dataset)
        coords = pd.read_csv(coord_fp, header=0, index_col=0).values[:, 1:]
        spatial_dists = distance.cdist(coords, coords, 'euclidean')
        spatial_dists = (spatial_dists/np.max(spatial_dists)) * SPATIAL_N_FEATURE_MAX
        expr[expr < 0] = 0.0
        if args.log:
            expr = np.log2(expr + 1)
        if args.scale:
            for i in range(expr.shape[0]):
                expr[i, :] = expr[i, :] / np.max(expr[i, :])
        return expr, genes, cells, spatial_dists
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
    spatial_dists = distance.cdist(coords, coords, 'euclidean')
    spatial_dists = (spatial_dists / np.max(spatial_dists)) * SPATIAL_N_FEATURE_MAX
    return expr, genes, cells, spatial_dists

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
        f_dists = torch.cdist(mu, mu, p=2) # feature distances
        f_dists = torch.mul(torch.div(f_dists, torch.max(f_dists)), SPATIAL_N_FEATURE_MAX)

        dist_penalty_1 = torch.div(torch.sum(torch.mul(SPATIAL_N_FEATURE_MAX - f_dists, spatial_distances)), n * n)
        dist_penalty_2 = torch.div(torch.sum(torch.mul(SPATIAL_N_FEATURE_MAX - spatial_distances, f_dists)), n * n)
        dist_penalty = torch.mul(dist_penalty_1, 500) + torch.mul(dist_penalty_2, 200)
        # dist_penalty_3 = torch.div(torch.sum(torch.mul(SPATIAL_N_FEATURE_MAX - spatial_distances, SPATIAL_N_FEATURE_MAX - f_dists)), n * n)

        # diagnal_mask = torch.eye(f_dists.shape[0], dtype=torch.bool).cuda()
        # spatial_closed_mask = spatial_distances.le(0.1)
        # feature_close_mask = f_dists.le(0.25)
        #
        # spatial_far_mask = spatial_distances.ge(0.25)
        # feature_far_mask = f_dists.ge(0.7)
        #
        # feature_close_spatial_far = torch.logical_and(torch.logical_and(spatial_far_mask, feature_close_mask), ~diagnal_mask)
        # dist_1 = torch.mul(SPATIAL_N_FEATURE_MAX - f_dists, spatial_distances)
        # dist_penalty_11 = torch.div(torch.nansum(dist_1[feature_close_spatial_far]), torch.nansum(feature_close_spatial_far))
        # dist_penalty_11 = 0.0 if torch.isnan(dist_penalty_11) else dist_penalty_11
        #
        # spatial_close_feature_far = torch.logical_and(torch.logical_and(spatial_closed_mask, feature_far_mask), ~diagnal_mask)
        # dist_2 = torch.mul(SPATIAL_N_FEATURE_MAX - spatial_distances, f_dists)
        # dist_penalty_22 = -torch.div(torch.nansum(dist_2[spatial_close_feature_far]), torch.nansum(spatial_close_feature_far))
        # dist_penalty_22 = 0.0 if torch.isnan(dist_penalty_22) else dist_penalty_22
        #
        # both_closed_mask = torch.logical_and(torch.logical_and(spatial_closed_mask, feature_close_mask), ~diagnal_mask)
        # dist_3 = torch.mul(spatial_distances, f_dists)
        # dist_penalty_33 = torch.div(torch.nansum(dist_3[both_closed_mask]), torch.nansum(both_closed_mask))
        # dist_penalty_33 = 0.0 if torch.isnan(dist_penalty_33) else dist_penalty_33

        #attract_term = torch.div(torch.sum(torch.exp(torch.add(SPATIAL_N_FEATURE_MAX - f_dists, SPATIAL_N_FEATURE_MAX - spatial_distances))), n * n)
        #repulse_term = torch.mul(torch.div(torch.sum(torch.exp(torch.add(torch.add(SPATIAL_N_FEATURE_MAX - f_dists, spatial_distances)*5.0, torch.add(SPATIAL_N_FEATURE_MAX - spatial_distances, f_dists)*2.0))), n * n), 1.0)
        #dist_penalty = -torch.log(torch.div(1, repulse_term))
        # dist_penalty = torch.mul(dist_penalty_1, 500) #+ torch.mul(dist_penalty_3, 200)# + torch.mul(dist_penalty_2, 200)# # + dist_penalty_11 * 50 + dist_penalty_11 * 100 + dist_penalty_22 * 20 + dist_penalty_33*2000
        # print("1: %.2f, 2: %.2f, 3: %.2f " % (dist_penalty_1*500, dist_penalty_2*200, dist_penalty_3*200))#, dist_penalty_11 * 100, dist_penalty_22 * 20, dist_penalty_33*2000 ))
        # print("VAE Loss:%.2f, dist_penalty:%.2f" % (VAE_Loss, dist_penalty))
        return VAE_Loss + dist_penalty# # reconstruction error + KL divergence losses
    else:
        return VAE_Loss

def train(vasc, optimizer, train_loader, model_fp, spatial_dists, args):
    vasc.train()
    epochs = args.epochs
    min_loss = np.inf
    patience = 0
    for epoch in range(epochs):
        train_loss = 0
        if epoch % 150 == 0 and args.annealing:
            tau = max(args.tau0 * np.exp(-args.anneal_rate * epoch), args.min_tau)
            print("tau = %.2f" % tau)
        for batch_idx, data in enumerate(train_loader):
            data = data[0].cuda()
            sidx, eidx = args.batch_size * batch_idx, args.batch_size * (batch_idx + 1)
            s_dists = spatial_dists[sidx: eidx, sidx: eidx].cuda()
            optimizer.zero_grad()
            recon_batch, mu, log_var = vasc.forward(data, tau)
            loss = loss_function(recon_batch, data, mu, log_var, s_dists, args)
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
            if patience == 0:
                torch.save(vasc.state_dict(), model_fp)
                print("Saved model at epoch %d with min_loss: %.0f" % (epoch + 1, min_loss))
        if patience > args.patience and epoch > args.min_stop:
            break


def evaluate(vasc, expr, model_fp, args):
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    vasc.z_mean.register_forward_hook(get_activation('z_mean'))
    vasc.load_state_dict(torch.load(model_fp))
    print("Load state dict successful!")
    vasc.eval()
    expr = expr.cuda()
    _ = vasc(expr, args.min_tau)
    reduced_reprs = activation['z_mean'].detach().cpu().numpy()
    return reduced_reprs

def get_expr_name(args):
    if args.spatial:
        name = "%s_%s_with_spatial" % (args.dataset, args.expr_name)
    else:
        name = "%s" % args.dataset
    return name

def save_features(reduced_reprs, feature_dir, name):
    feature_fp = os.path.join(feature_dir, "%s.tsv" % name)
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