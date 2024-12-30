import opt
import torch
import numpy as np
from GAE import IGAE
from utils import setup_seed
from train import Pretrain_gae
from sklearn.decomposition import PCA
from load_data import LoadDataset, load_graph

setup_seed(1)

print("use cuda: {}".format(opt.args.cuda))
device = torch.device("cuda" if opt.args.cuda else "cpu")

opt.args.data_path = 'data/{}.txt'.format(opt.args.name)
opt.args.label_path = 'data/{}_label.txt'.format(opt.args.name)
opt.args.graph_k_save_path = 'graph/{}{}_graph.txt'.format(opt.args.name, opt.args.k)
opt.args.graph_save_path = 'graph/{}_graph.txt'.format(opt.args.name)
opt.args.model_save_path = 'model/model_save_gae/{}_gae.pkl'.format(opt.args.name)

print("Data: {}".format(opt.args.data_path))
print("Label: {}".format(opt.args.label_path))

x = np.loadtxt(opt.args.data_path, dtype=float)
y = np.loadtxt(opt.args.label_path, dtype=int)

pca = PCA(n_components=opt.args.n_input)
X_pca = pca.fit_transform(x)

dataset = LoadDataset(X_pca)

adj = load_graph(opt.args.k, opt.args.graph_k_save_path, opt.args.graph_save_path, opt.args.data_path).to(device)
data = torch.Tensor(dataset.x).to(device)
label = y

model_gae = IGAE(
    gae_n_enc_1=opt.args.gae_n_enc_1,
    gae_n_enc_2=opt.args.gae_n_enc_2,
    gae_n_enc_3=opt.args.gae_n_enc_3,
    gae_n_dec_1=opt.args.gae_n_dec_1,
    gae_n_dec_2=opt.args.gae_n_dec_2,
    gae_n_dec_3=opt.args.gae_n_dec_3,
    n_input=opt.args.n_components,
).to(device)

Pretrain_gae(model_gae, data, adj, y, opt.args.gamma_value)
