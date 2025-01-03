import torch
from opt import args
from utils import eva
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.cluster import KMeans

acc_reuslt = []
nmi_result = []
ari_result = []
f1_result = []
use_adjust_lr = []


def Pretrain(model, data, adj, label):
    optimizer = Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epoch):
        # if (args.name in use_adjust_lr):
        #     adjust_learning_rate(optimizer, epoch)

        x_hat, z_hat, adj_hat, z_ae, z_igae, z_tilde = model(data, adj)

        loss_1 = F.mse_loss(x_hat, data)
        loss_2 = F.mse_loss(z_hat, torch.spmm(adj, data))
        loss_3 = F.mse_loss(adj_hat, adj.to_dense())
        loss_4 = F.mse_loss(z_ae, z_igae)  # simple aligned

        loss = loss_1 + args.alpha * loss_2 + args.beta \
               * loss_3 + args.omega * loss_4  # you can tune all kinds of hyper-parameters to get better performance.
        print('{} loss: {}'.format(epoch, loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(z_tilde.data.cpu().numpy())

        acc, nmi, ari, f1 = eva(label, kmeans.labels_, epoch)
        acc_reuslt.append(acc)
        nmi_result.append(nmi)
        ari_result.append(ari)
        f1_result.append(f1)

        torch.save(model.state_dict(), args.pre_model_save_path)
