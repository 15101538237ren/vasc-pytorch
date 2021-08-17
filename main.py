# -*- coding: utf-8 -*-
import os
import torch
from torch_geometric.nn import DeepGraphInfomax, GAE, VGAE
from torch.utils.data import TensorDataset, DataLoader
from models.vasc import VASC
from utils.config import get_args
from utils.util import get_data, prepare_cuda, train, train_in_batch, evaluate, save_features, get_expr_name, VISIUM_DATASETS, SPATIAL_LIBD_DATASETS, corruption
from models.dgi import DGI_Encoder
from models.gae import GAE_Encoder
args = get_args()
device = prepare_cuda(args)

#writer = SummaryWriter(args.out_dir)

datasets = SPATIAL_LIBD_DATASETS + VISIUM_DATASETS

for dataset in datasets:
    args.dataset = dataset
    expr, genes, samples, spatial_dists, graph_A = get_data(args)
    X = torch.tensor(expr).float()
    spatial_dists = torch.tensor(spatial_dists).float()
    if args.arch == "VASC":
        model = VASC(x_dim=len(genes), z_dim=args.z_dim, var=args.var, dropout=args.dropout, isTrain=args.train).to(
            device)
    elif args.arch == "DGI":
        model = DeepGraphInfomax(
            hidden_channels=args.z_dim, encoder=DGI_Encoder(len(genes), args.z_dim),
            summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
            corruption=corruption).to(device)
    elif args.arch == 'VGAE':
        model = VGAE(GAE_Encoder(len(genes), args.z_dim, args.arch)).to(device)
    else:
        model = GAE(GAE_Encoder(len(genes), args.z_dim, args.arch)).to(device)

    name = get_expr_name(args)
    model_fp = os.path.join("data", "models", "%s.pt" % name)
    if args.train:
        if args.batch:
            train_loader = DataLoader(dataset=TensorDataset(X), batch_size=args.batch_size, shuffle=False)
            train_in_batch(model, device, train_loader, graph_A, model_fp, spatial_dists, args)
        else:
            train(model, device, X, graph_A, model_fp, spatial_dists, args)
    reduced_reprs = evaluate(model, device, X, graph_A, model_fp, args)
    feature_dir = os.path.join(args.dataset_dir, "features")
    save_features(reduced_reprs, feature_dir, name)
