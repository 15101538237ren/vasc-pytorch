# -*- coding: utf-8 -*-
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
from models.vasc import VASC
from utils.config import get_args
from utils.util import get_data,get_labels , preprocess_data, train, evaluate
from utils.visualization import plot_2d_features
args = get_args()
model_fp = os.path.join("data", "models", "%s.pt" % args.dataset)
cuda = torch.cuda.is_available()
device = torch.device("cuda:%d" % args.gpu if cuda else "cpu")

if cuda:
    torch.cuda.manual_seed_all(args.seed)
    print("GPU count: %d, using gpu: %d" % (torch.cuda.device_count(), args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    torch.manual_seed(args.seed)
#writer = SummaryWriter(args.out_dir)

expr, genes, samples = get_data(args.dataset_dir, args.dataset)
stages = get_labels(args.dataset_dir, args.dataset, samples)
expr = preprocess_data(expr, args.log, args.scale)
n_cell, n_gene = expr.shape
expr_t = torch.Tensor(expr)
train_loader = DataLoader(dataset=TensorDataset(expr_t), batch_size=args.batch_size, shuffle=False)

vasc = VASC(x_dim=n_gene, z_dim=args.z_dim, var=args.var, dropout=args.dropout, isTrain=args.train).to(device)
optimizer = torch.optim.RMSprop(vasc.parameters(), lr=args.lr)
if args.train:
    train(vasc, optimizer, train_loader, model_fp, args)
reduced_reprs = evaluate(vasc, expr_t, model_fp, args)
plot_2d_features(reduced_reprs, stages, fig_name=args.dataset)
