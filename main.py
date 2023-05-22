# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from torch_geometric.nn import DeepGraphInfomax, GAE, VGAE
from models.vasc import VASC
from utils.config import get_args
from utils.util import get_data, prepare_cuda, train, generate_expr_name, save_features, get_expr_name, VISIUM_DATASETS, SPATIAL_LIBD_DATASETS, corruption, SQUIDPY_DATASETS
from models.dgi import DGI_Encoder
from models.gae import GAE_Encoder, VGAEsc
args = get_args()


#writer = SummaryWriter(args.out_dir)

p1_coefs = [int(0), int(1)]#, int(50), int(250), int(500), int(1000)int(20), , int(1000) [ int(100), int(500)]#int(0),  , int(0), int(LF),int(0), int(SS), int(LF), int(LF)]
p2_coefs = [int(0), int(0)]#, int(0), int(0), int(0), int(0)int(0), , int(0)[ int(0), int(0)]# int(0),, int(-LF), int(SS),int(-SS//2), int(0), int(-SS//2), int(SS//2)]
device = prepare_cuda(args)
for arch in ['DGI']:
    args.arch = arch
    for pid, p1_coef in enumerate(p1_coefs):
        p2_coef = p2_coefs[pid]
        args.expr_name = generate_expr_name(p1_coef, p2_coef)
        print(args.expr_name)
        if p1_coef == 0 and p2_coef == 0:
            args.spatial = False
        else:
            args.spatial = True

        datasets = SPATIAL_LIBD_DATASETS#["slideseqv2"]#VISIUM_DATASETS["seqfish"] #SQUIDPY_DATASETS#["osmFISH"]#VISIUM_DATASETS# + SPATIAL_LIBD_DATASETS#

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
                model = VGAEsc(encoder=GAE_Encoder(len(genes), args.z_dim, args.arch), in_dim=len(genes), zdim=args.z_dim).to(device)
            else:
                model = GAE(GAE_Encoder(len(genes), args.z_dim, args.arch)).to(device)

            # model_fp = os.path.join("data", "models", "%s.pt" % name)
            feature_dir = os.path.join(args.dataset_dir, "features")
            if args.train:
                np.random.seed(args.seed)
                torch_seeds = np.random.choice(10000, size=args.n_consensus, replace=False)
                np.random.seed(args.seed)
                python_seeds = np.random.choice(10000, size=args.n_consensus, replace=False)
                np.random.seed(args.seed)
                numpy_seeds = np.random.choice(10000, size=args.n_consensus, replace=False)
                name = get_expr_name(args)
                print("Now training %s" % name)
                X_embed = train(model, device, X, graph_A, spatial_dists, args, torch_seed=torch_seeds[0], python_seed=python_seeds[0],
                        numpy_seed=numpy_seeds[0], p1_coef=p1_coef, p2_coef=p2_coef)

                save_features(X_embed, feature_dir, name)
