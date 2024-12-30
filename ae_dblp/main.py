import torch
from AE import AE
import numpy as np
from opt import args
from sklearn.decomposition import PCA
from utils import setup_seed
from torch.utils.data import Dataset, DataLoader
from train import Pretrain_ae

from kymatio import Scattering1D
import matplotlib.pyplot as plt
import pywt


setup_seed(1)

print("use cuda: {}".format(args.cuda))
device = torch.device("cuda" if args.cuda else "cpu")

args.data_path = 'data/{}.txt'.format(args.name)
args.label_path = 'data/{}_label.txt'.format(args.name)
args.model_save_path = 'model/model_save_ae/{}_ae.pkl'.format(args.name)

print("Data: {}".format(args.data_path))
print("Label: {}".format(args.label_path))


class LoadDataset(Dataset):

    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


x = np.loadtxt(args.data_path, dtype=float)
y = np.loadtxt(args.label_path, dtype=int)

scattering = Scattering1D(J=2, shape=x[0].shape)
x = scattering(x)[:,0,:]

wavelet = "cmor0.5-1.5"
coef, freqs = pywt.cwt(x, np.arange(1, 101), wavelet)
coef = np.abs(coef)
xx =coef.transpose(1, 0, 2).reshape(-1,1,100,100)
# log_eps = 1e-6
# scattering = Scattering1D(J=2, shape=x[0].shape)
# X_pca = scattering(x)
# yy = np.where(y == -1)
# plt.figure(1)
# plt.subplot(5,1,1)
# plt.plot(X_pca[5])
# plt.subplot(5,1,2)
# plt.plot(X_pca[1])
# plt.subplot(5,1,3)
# plt.plot(X_pca[2])
# plt.subplot(5,1,4)
# plt.plot(X_pca[3])
# plt.subplot(5,1,5)
# plt.plot(X_pca[4])
#
# plt.figure(2)
# plt.subplot(5,1,1)
# plt.imshow(X_pca[5].reshape(1,-1), cmap='viridis', aspect='auto')
# plt.colorbar()
#
# plt.subplot(5,1,2)
# plt.imshow(X_pca[1].reshape(1,-1), cmap='viridis', aspect='auto')
# plt.colorbar()
#
# plt.subplot(5,1,3)
# plt.imshow(X_pca[2].reshape(1,-1), cmap='viridis', aspect='auto')
# plt.colorbar()
#
# plt.subplot(5,1,4)
# plt.imshow(X_pca[3].reshape(1,-1), cmap='viridis', aspect='auto')
# plt.colorbar()
#
# plt.subplot(5,1,5)
# plt.imshow(X_pca[4].reshape(1,-1), cmap='viridis', aspect='auto')
# plt.colorbar()
#
# plt.show()

# pca = PCA(n_components=args.n_components)
# X_pca = pca.fit_transform(x)

dataset = LoadDataset(xx)
train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)

model = AE(
    ae_n_enc_1=args.ae_n_enc_1,
    ae_n_enc_2=args.ae_n_enc_2,
    ae_n_dec_1=args.ae_n_dec_1,
    ae_n_dec_2=args.ae_n_dec_2,
    n_input=1,
    n_z=args.n_z).to(device)

Pretrain_ae(model, dataset, y, train_loader, device)

