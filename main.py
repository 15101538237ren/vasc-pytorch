# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from models.vasc import VASC
from utils.config import get_args
from utils.util import get_data, prepare_cuda, train, train_in_batch, evaluate, save_features, get_expr_name, VISIUM_DATASETS, SPATIAL_LIBD_DATASETS

args = get_args()
device = prepare_cuda(args)
#writer = SummaryWriter(args.out_dir)

datasets = SPATIAL_LIBD_DATASETS + VISIUM_DATASETS#["drosophila"]#VISIUM_DATASETS #SPATIAL_LIBD_DATASETS +SPATIAL_LIBD_DATASETS#

for dataset in datasets:
    args.dataset = dataset
    expr, genes, samples, spatial_dists = get_data(args)
    expr_t = torch.tensor(expr).float()
    spatial_dists = torch.tensor(spatial_dists).float()
    vasc = VASC(x_dim=len(genes), z_dim=args.z_dim, var=args.var, dropout=args.dropout, isTrain=args.train).to(device)
    optimizer = torch.optim.RMSprop(vasc.parameters(), lr=args.lr)
    name = get_expr_name(args)
    model_fp = os.path.join("data", "models", "%s.pt" % name)
    if args.train:
        if args.batch:
            train_loader = DataLoader(dataset=TensorDataset(expr_t), batch_size=args.batch_size, shuffle=False)
            train_in_batch(vasc, optimizer, train_loader, model_fp, spatial_dists, args)
        else:
            train(vasc, optimizer, expr_t, model_fp, spatial_dists, args)
    reduced_reprs = evaluate(vasc, expr_t, model_fp, args)
    feature_dir = os.path.join(args.dataset_dir, "features")
    save_features(reduced_reprs, feature_dir, name)
