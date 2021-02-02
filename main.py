# -*- coding: utf-8 -*-
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
from models.vasc import VASC
from utils.config import get_args
from utils.util import get_data, preprocess_data, train

args = get_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
cuda = torch.cuda.is_available()
device = torch.device("cuda:%d" % args.gpu if cuda else "cpu")

if cuda:
    torch.cuda.manual_seed_all(args.seed)
else:
    torch.manual_seed(args.seed)
writer = SummaryWriter(args.out_dir)

expr = get_data(args.dataset_dir, args.dataset)
expr = preprocess_data(expr, args.log, args.scale)
n_cell, n_gene = expr.shape
train_loader = DataLoader(dataset=TensorDataset(torch.Tensor(expr)), batch_size=args.batch_size, shuffle=True)


vasc = VASC(x_dim=n_gene, z_dim=args.z_dim, var=args.var, dropout=args.dropout).to(device)
optimizer = torch.optim.RMSprop(vasc.parameters(), lr=args.lr)

train(vasc, optimizer, train_loader, args)