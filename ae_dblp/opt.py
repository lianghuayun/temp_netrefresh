import argparse

parser = argparse.ArgumentParser(description='DFCN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--name', type=str, default='kyberpublic')
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--n_clusters', type=int, default=5)
parser.add_argument('--n_z', type=int, default=60)
parser.add_argument('--n_input', type=int, default=30)
parser.add_argument('--data_path', type=str, default='.txt')
parser.add_argument('--label_path', type=str, default='.txt')
parser.add_argument('--save_path', type=str, default='.txt')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--n_components', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--acc', type=float, default=-1)
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--ae_n_enc_1', type=int, default=32)
parser.add_argument('--ae_n_enc_2', type=int, default=64)
parser.add_argument('--ae_n_dec_1', type=int, default=64)
parser.add_argument('--ae_n_dec_2', type=int, default=32)

args = parser.parse_args()