import torch
from opt import args
from utils import eva, target_distribution
from torch.optim import Adam, AdamW
import torch.nn.functional as F
from sklearn.cluster import KMeans

from sklearn.cluster import AgglomerativeClustering

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

acc_reuslt = []
nmi_result = []
ari_result = []
f1_result = []
use_adjust_lr = []


def Pretrain_ae(model, dataset, y, train_loader, device):
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loss_arr = []
    for epoch in range(args.epoch):
        # if (args.name in use_adjust_lr):
        #     adjust_learning_rate(optimizer, epoch)
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            x_hat, _, q, q1 = model(x)
            loss = F.mse_loss(x_hat, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            x = torch.Tensor(dataset.x).to(device).float()
            x_hat, z, q, q1 = model(x)

            tmp_q = q.data
            p = target_distribution(tmp_q)


            loss = F.mse_loss(x_hat, x)
            loss_kl = F.kl_div((q.log() + q1.log()) / 2, p, reduction='batchmean')
            loss = 0.001*loss + 100000 * loss_kl
            loss_arr.append(loss.data.cpu().item())
            print('{} loss: {}'.format(epoch, loss))



            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(z.data.cpu().numpy())
            hierarchical_clustering = AgglomerativeClustering(n_clusters=5, affinity='chebyshev', linkage='single')
            kmeans.labels_ = hierarchical_clustering.fit_predict(z.data.cpu().numpy())

            # ax = plt.figure().add_subplot(projection='3d')
            # pca = PCA(n_components=3)
            # z_visual = pca.fit_transform(z.data.cpu().numpy())
            #
            # df_A_0 = z_visual[y == 0]
            # df_A_1 = z_visual[y == 1]
            # df_A_2 = z_visual[y == 2]
            # df_A_3 = z_visual[y == -1]
            # df_A_4 = z_visual[y == -2]
            #
            # ax.scatter(df_A_0[:, 0], df_A_0[:, 1], df_A_0[:, 2], c='g', marker='*', s=10, label='data0')
            # ax.scatter(df_A_1[:, 0], df_A_1[:, 1], df_A_1[:, 2], c='r', marker='*', s=10, label='data1')
            # ax.scatter(df_A_2[:, 0], df_A_2[:, 1], df_A_2[:, 2], c='b', marker='*', s=10, label='data2')
            # ax.scatter(df_A_3[:, 0], df_A_3[:, 1], df_A_3[:, 2], c='orange', marker='*', s=10, label='data3')
            # ax.scatter(df_A_4[:, 0], df_A_4[:, 1], df_A_4[:, 2], c='purple', marker='*', s=10, label='data4')
            #
            #
            #
            # plt.show()

            acc, nmi, ari, f1 = eva(y, kmeans.labels_, epoch)
            acc_reuslt.append(acc)
            nmi_result.append(nmi)
            ari_result.append(ari)
            f1_result.append(f1)

            torch.save(model.state_dict(), args.model_save_path)
    print(loss_arr)
    print(acc_reuslt)
    print(model)