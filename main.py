# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from models.vasc import VASC
from utils.config import get_args
from utils.util import get_data, prepare_cuda, preprocess_data, train, evaluate, save_features

args = get_args()
device = prepare_cuda(args)
#writer = SummaryWriter(args.out_dir)
expr, genes, samples, spatial_dists = get_data(args.dataset_dir, args.dataset, args.ncells)
expr = preprocess_data(expr, args.log, args.scale)
expr_t = torch.tensor(expr).float()
spatial_dists = torch.tensor(spatial_dists).float()
train_loader = DataLoader(dataset=TensorDataset(expr_t), batch_size=args.batch_size, shuffle=False)
vasc = VASC(x_dim=len(genes), z_dim=args.z_dim, var=args.var, dropout=args.dropout, isTrain=args.train).to(device)
optimizer = torch.optim.RMSprop(vasc.parameters(), lr=args.lr)

model_fp = os.path.join("data", "models", "%s.pt" % args.dataset)
if args.train:
    train(vasc, optimizer, train_loader, model_fp, spatial_dists, args)
reduced_reprs = evaluate(vasc, expr_t, model_fp, args)
save_features(reduced_reprs, os.path.join(args.dataset_dir, args.feature_dir), args.dataset)
