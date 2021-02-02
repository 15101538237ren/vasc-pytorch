import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--log', type=bool, default=True)
parser.add_argument('--scale', type=bool, default=True)
parser.add_argument('--var', type=bool, default=True)

parser.add_argument('--arch', type=str, default='vasc', choices=['vasc'])
parser.add_argument('--dataset_dir', type=str, default='data')
parser.add_argument('--dataset', type=str, default='biase',
                    choices=['biase'])

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--test_batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=10000)

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--annealing', type=bool, default=True)
parser.add_argument('--anneal_rate', type=float, default=0.0003)
parser.add_argument('--tau0', type=float, default=1.0)
parser.add_argument('--min_tau', type=float, default=0.5)
parser.add_argument('--z_dim', type=int, default=2)
parser.add_argument('--patience', type=int, default=300)
parser.add_argument('--min_stop', type=int, default=500)
parser.add_argument('--dropout', type=float, default=0.5)

def get_args():
    args = parser.parse_args()

    args.exp_name = "%s_%s_seed%d_bs%d_z%d_lr%e_epochs%d" % \
                    (args.dataset, args.arch, args.seed, args.batch_size,
                       args.z_dim, args.lr,args.epochs)

    args.figs_dir = os.path.join('figs', args.exp_name)
    args.out_dir = os.path.join('result', args.exp_name)
    args.best_model_file = os.path.join('result', args.exp_name, 'best_model.pt')

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if not os.path.exists(args.figs_dir):
        os.makedirs(args.figs_dir)

    return args