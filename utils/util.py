# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

def get_data(dataset_dir, dataset):
    expr_fp = os.path.join(dataset_dir, dataset, "%s.txt" % dataset)
    expr_df = pd.read_csv(expr_fp, sep="\t", header=0, index_col=0)
    expr = expr_df.values.T
    return expr

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
    return BCE + torch.mean(KLD) * 50.0 # reconstruction error + KL divergence losses

def train(vasc, optimizer, train_loader, args):
    vasc.train()
    epochs = args.epochs
    losses = []
    prev_loss = np.inf
    for epoch in range(epochs):
        cur_loss = prev_loss
        train_loss = 0
        if epoch % 100 == 0 and args.annealing:
            tau = max(args.tau0 * np.exp(-args.anneal_rate * epoch), args.min_tau)
            print("tau = %.2f" % tau)
        for batch_idx, data in enumerate(train_loader):
            data = data[0].cuda()
            optimizer.zero_grad()

            recon_batch, mu, log_var = vasc.forward(data, tau)
            loss = loss_function(recon_batch, data, mu, log_var)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        cur_loss = min(train_loss, cur_loss)
        losses.append(train_loss)

        if epoch % args.patience == 1:
            print( "Epoch %d/%d"%(epoch + 1, epochs))
            print( "Loss:" + str(train_loss) )
            if abs(cur_loss - prev_loss) < 1 and epoch > args.min_stop:
                break
            prev_loss = train_loss

