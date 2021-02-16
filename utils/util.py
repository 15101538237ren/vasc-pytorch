# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_data(dataset_dir, dataset):
    expr_fp = os.path.join(dataset_dir, dataset, "%s.txt" % dataset)
    expr_df = pd.read_csv(expr_fp, sep="\t", header=0, index_col=0)
    if dataset == "drosophila":
        expr = expr_df.values
        samples, genes = expr_df.index.tolist(), list(expr_df.columns.values)
    else:
        expr = expr_df.values.T
        genes, samples = expr_df.index.tolist(), list(expr_df.columns.values)
    return expr, genes, samples

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

def loss_function(recon_x, x, mu, log_var, var=False):
    in_dim = x.shape[1]
    BCE = in_dim * F.binary_cross_entropy(recon_x, x, reduction='mean')
    if var:
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
    else:
        var_ones = torch.ones(log_var.size()).cuda()
        KLD = -0.5 * torch.sum(1 + 1 - mu.pow(2) - torch.exp(var_ones), dim=-1)
    KLD = torch.mean(KLD)
    return BCE + KLD, KLD # reconstruction error + KL divergence losses

def train(vasc, optimizer, train_loader, model_fp, args):
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
            optimizer.zero_grad()
            recon_batch, mu, log_var = vasc.forward(data, tau)
            loss, kld = loss_function(recon_batch, data, mu, log_var)
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