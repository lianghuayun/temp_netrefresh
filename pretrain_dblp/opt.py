import argparse

parser = argparse.ArgumentParser(description='DFCN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--name', type=str, default='usps')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--k', type=int, default=None)
parser.add_argument('--n_clusters', type=int, default=5)
parser.add_argument('--n_z', type=int, default=60)
parser.add_argument('--n_input', type=int, default=50)
parser.add_argument('--freedom_degree', type=float, default=1.0)
parser.add_argument('--data_path', type=str, default='.txt')
parser.add_argument('--label_path', type=str, default='.txt')
parser.add_argument('--save_path', type=str, default='.txt')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--alpha', type=float, default=0.01)
parser.add_argument('--beta', type=int, default=0.01)
parser.add_argument('--omega', type=float, default=0.01)
parser.add_argument('--n_components', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epoch', type=int, default=60)
parser.add_argument('--acc', type=float, default=-1)
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--ae_n_enc_1', type=int, default=128)
parser.add_argument('--ae_n_enc_2', type=int, default=256)
parser.add_argument('--ae_n_enc_3', type=int, default=512)
parser.add_argument('--ae_n_dec_1', type=int, default=512)
parser.add_argument('--ae_n_dec_2', type=int, default=256)
parser.add_argument('--ae_n_dec_3', type=int, default=128)
parser.add_argument('--gae_n_enc_1', type=int, default=128)
parser.add_argument('--gae_n_enc_2', type=int, default=256)
parser.add_argument('--gae_n_enc_3', type=int, default=60)
parser.add_argument('--gae_n_dec_1', type=int, default=60)
parser.add_argument('--gae_n_dec_2', type=int, default=256)
parser.add_argument('--gae_n_dec_3', type=int, default=128)

args = parser.parse_args()