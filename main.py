# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.manifold import MDS
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

datasets = SPATIAL_LIBD_DATASETS# + VISIUM_DATASETS

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

    # model_fp = os.path.join("data", "models", "%s.pt" % name)
    feature_dir = os.path.join(args.dataset_dir, "features")
    X_embeds = []
    if args.train:
        np.random.seed(args.seed)
        torch_seeds = np.random.choice(10000, size=args.n_consensus, replace=False)
        np.random.seed(args.seed)
        python_seeds = np.random.choice(10000, size=args.n_consensus, replace=False)
        np.random.seed(args.seed)
        numpy_seeds = np.random.choice(10000, size=args.n_consensus, replace=False)

        for i in range(args.n_consensus):
            if args.batch:
                train_loader = DataLoader(dataset=TensorDataset(X), batch_size=args.batch_size, shuffle=False)
                X_embed = train_in_batch(model, device, train_loader, graph_A, spatial_dists, args, torch_seed=torch_seeds[i], python_seed=python_seeds[i],
                    numpy_seed=numpy_seeds[i])
            else:
                X_embed = train(model, device, X, graph_A, spatial_dists, args, torch_seed=torch_seeds[i], python_seed=python_seeds[i],
                    numpy_seed=numpy_seeds[i])
            name = get_expr_name(args, idx=i)
            save_features(X_embed, feature_dir, name)
            X_embeds.append(X_embed)

        embeds_weights = np.ones(len(X_embeds)) / float(len(X_embeds))
        n_spot = len(samples)
        W_consensus = np.zeros([n_spot, n_spot])

        for i in range(len(X_embeds)):
            W = distance_matrix(X_embeds[i], X_embeds[i])
            W_consensus += W * embeds_weights[i]
        print("STARTING MDS!")
        mds_model = MDS(n_components=args.n_comps_proj, dissimilarity='precomputed', n_jobs=64, random_state=args.seed)
        X_embed = mds_model.fit_transform(W_consensus)
        print("MDS DONE!")
        name = get_expr_name(args, idx=0)
        save_features(X_embed, feature_dir, name, mds=True)
        print("FEATURE SAVED!")
