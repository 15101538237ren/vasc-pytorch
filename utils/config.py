import os
import argparse

parser = argparse.ArgumentParser()

# params to change
parser.add_argument('--gpu', type=int, default=4)
parser.add_argument('--expr_name', type=str, default='500_penalty1') #'500_penalty1_200_penalty2_-250_penalty3_50p22_spc_0.25_ftf_0.6'
parser.add_argument('--arch', type=str, default='GAE', choices=['VASC', 'GAE', 'DGI', 'VGAE'])
parser.add_argument('--spatial', type=bool, default=True)
parser.add_argument('--sigma_sq', type=float, default=0.1)
parser.add_argument('--batch', type=bool, default=False)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--knn_n_neighbors', type=int, default=10)
parser.add_argument('--alpha_n_layer', type=int, default=1)
parser.add_argument('--patience', type=int, default=30)
parser.add_argument('--min_stop', type=int, default=50)
parser.add_argument('--z_dim', type=int, default=50)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--log', type=bool, default=True)
parser.add_argument('--scale', type=bool, default=False)
parser.add_argument('--var', type=bool, default=True)
parser.add_argument('--SVGene', type=bool, default=False)
parser.add_argument('--dataset_dir', type=str, default='data')
parser.add_argument('--feature_dir', type=str, default='features')
parser.add_argument('--figure_dir', type=str, default='figures')
parser.add_argument('--dataset', type=str, default='drosophila',
                    choices=['drosophila'])

parser.add_argument('--test_batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--anneal_rate', type=float, default=0.0003)
parser.add_argument('--annealing', type=bool, default=True)
parser.add_argument('--tau0', type=float, default=1.50)
parser.add_argument('--min_tau', type=float, default=0.5)
parser.add_argument('--dropout', type=float, default=0.5)

def get_args():
    args = parser.parse_args()

    args.exp_name = "%s_%s_seed%d_bs%d_z%d_lr%e_epochs%d" % \
                    (args.dataset, args.arch, args.seed, args.batch_size,
                       args.z_dim, args.lr,args.epochs)

    args.best_model_file = os.path.join('result', args.exp_name, 'best_model.pt')
    return args