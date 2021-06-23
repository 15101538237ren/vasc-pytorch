# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial import distance

SPATIAL_N_FEATURE_MAX = 100.0
SPATIAL_THRESHOLD = 50.0
FEATURE_THRESHOLD = 50.0
def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_data(dataset_dir, dataset, ncells):
    if dataset in ["Kidney", "Liver"]:
        expr_fp = os.path.join(dataset_dir, dataset, "%s.count.csv" % dataset)
        expr_df = pd.read_csv(expr_fp, header=0, index_col=0)
        expr = expr_df.values.T[:ncells,:]
        genes, samples = expr_df.index.tolist(), list(expr_df.columns.values)[:ncells]

        coord_fp = os.path.join(dataset_dir, dataset, "%s.idx" % dataset)
        coords = pd.read_csv(coord_fp, header=0, index_col=0).values[:ncells, 1:]
        spatial_dists = distance.cdist(coords, coords, 'euclidean')
        spatial_dists = (spatial_dists/np.max(spatial_dists)) * SPATIAL_N_FEATURE_MAX
        return expr, genes, samples, spatial_dists
    else:
        return [], [], [], []

def get_labels(dataset_dir, dataset, samples):
    if dataset == "petropoulus":
        stage_arr = []
        for sample in samples:
            ssplit = sample.split(".")
            if ssplit[1] in ["early", "late"]:
                stage = "%s_%s" % (ssplit[0], ssplit[1])
            else:
                stage = ssplit[0]
            stage_arr.append(stage)
    else:
        label_fp = os.path.join(dataset_dir, dataset, "%s_label.txt" % dataset)
        label_df = pd.read_csv(label_fp, sep="\t", header=None, index_col=0)
        sample_names = label_df.index.tolist()
        sample_dict = { sample: sid for sid, sample in enumerate(samples)}
        stages = label_df.values.flatten()
        stage_arr = [stages[sample_dict[sample]] for sample in sample_names]
    return stage_arr

def get_psedo_times(dataset_dir, dataset, samples):
    label_fp = os.path.join(dataset_dir, dataset, "%s_info.tsv" % dataset)
    label_df = pd.read_csv(label_fp, sep="\t", header=0, index_col=0)
    sample_names = label_df.index.tolist()
    sample_dict = {sample: sid for sid, sample in enumerate(samples)}
    pseudo_times = label_df.values[:, 13].flatten()
    pseudo_times_arr = [pseudo_times[sample_dict[sample]] for sample in sample_names]
    return pseudo_times_arr

def preprocess_data(expr, log, scale):
    expr[expr < 0] = 0.0
    if log:
        expr = np.log2(expr + 1)
    if scale:
        for i in range(expr.shape[0]):
            expr[i, :] = expr[i, :] / np.max(expr[i, :])
    return expr

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
    if args.spatial:
        f_dists = torch.cdist(mu, mu, p=2) # feature distances
        f_dists = torch.mul(torch.div(f_dists, torch.max(f_dists)), SPATIAL_N_FEATURE_MAX)
        f_dists_transformed = torch.exp(-(torch.div(f_dists, FEATURE_THRESHOLD)).pow(4))
        s_dists_transformed = torch.ones(spatial_distances.size()).cuda() - torch.exp(-(torch.div(spatial_distances, SPATIAL_THRESHOLD)).pow(4))
        dist_penalty = torch.mul(torch.div(torch.sum(torch.mul(f_dists_transformed, s_dists_transformed)), n*n), 500.0)
        return BCE + KLD + dist_penalty # reconstruction error + KL divergence losses
    else:
        return BCE + KLD

def train(vasc, optimizer, train_loader, model_fp, spatial_dists, args):
    vasc.train()
    epochs = args.epochs
    min_loss = np.inf
    patience = 0
    for epoch in range(epochs):
        train_loss = 0
        if epoch % 100 == 0 and args.annealing:
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

def save_features(reduced_reprs, feature_dir, dataset):
    feature_fp = os.path.join(feature_dir, "%s.tsv" % dataset)
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