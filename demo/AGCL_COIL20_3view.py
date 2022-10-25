from __future__ import print_function, division
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam, SGD
from sklearn.preprocessing import StandardScaler
import scipy.io
from idecutils import cluster_acc, rebuild, fit_hnsw_index, indices2feature, graph_error, InstanceLoss
from queue import Queue
from models import AE_3views as AE
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
manual_seed = 0
os.environ['PYTHONHASHSEED'] = str(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)
np.random.seed(manual_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def wmse_loss(input, target):
    ret = (target - input) ** 2
    ret = torch.mean(ret)
    return ret


class MFC(nn.Module):

    def __init__(self,
                 n_stacks,
                 n_input,
                 n_z,
                 n_clusters,
                 v=1,
                 train_path=''):
        super(MFC, self).__init__()
        self.train_path = train_path

        self.ae = AE(
            n_stacks=n_stacks,
            n_input=n_input,
            n_z=n_z)

        self.v = v
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def train(self, path=''):
        if args.train_MFC_flag == 0:
            train_mfc(self.ae)
            print('trained ae finished')
            args.train_MFC_flag = 1
        else:
            self.ae.load_state_dict(torch.load(self.train_path))
            print('load trained ae model from', self.train_path)

    def forward(self, xv0, xv1, xv2):
        _, _, _, z, zv0, zv1, zv2 = self.ae(xv0, xv1, xv2)
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return z, q, zv0, zv1, zv2


def train_mfc(model):
    # print(model)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)

    optimizer = SGD(model.parameters(), lr=args.lr_train, momentum=0.95)
    index_array = np.arange(X0.shape[0])
    np.random.shuffle(index_array)
    loss_q = Queue(maxsize=50)
    for epoch in range(args.AE_epoch):
        g_error0 = graph_init_error0
        g_error1 = graph_init_error1
        g_error2 = graph_init_error2
        if epoch >= args.start_ann and epoch % args.t == 0:
            _, _, _, _, z0, z1, z2 = model(X0, X1, X2)
            p0 = fit_hnsw_index(z0.data.cpu().numpy(), ef=args.knn * 10)
            ann_neighbor_indices0, _ = p0.knn_query(z0.data.cpu().numpy(), args.knn)
            p1 = fit_hnsw_index(z1.data.cpu().numpy(), ef=args.knn * 10)
            ann_neighbor_indices1, _ = p1.knn_query(z1.data.cpu().numpy(), args.knn)
            p2 = fit_hnsw_index(z2.data.cpu().numpy(), ef=args.knn * 10)
            ann_neighbor_indices2, _ = p2.knn_query(z2.data.cpu().numpy(), args.knn)
            g_error0 = graph_error(ann_neighbor_indices0, y)
            g_error1 = graph_error(ann_neighbor_indices1, y)
            g_error2 = graph_error(ann_neighbor_indices2, y)
            X0_knn_n = indices2feature(X0, ann_neighbor_indices0, device)
            X1_knn_n = indices2feature(X1, ann_neighbor_indices1, device)
            X2_knn_n = indices2feature(X2, ann_neighbor_indices2, device)
        total_loss = 0.
        for batch_idx in range(np.int_(np.ceil(X0.shape[0] / args.batch_size))):
            idx = index_array[batch_idx * args.batch_size: min((batch_idx + 1) * args.batch_size, X0.shape[0])]
            x0 = X0[idx].to(device)
            x1 = X1[idx].to(device)
            x2 = X2[idx].to(device)
            if epoch >= args.start_ann:
                x0_knn = X0_knn_n[idx].to(device)
                x1_knn = X1_knn_n[idx].to(device)
                x2_knn = X2_knn_n[idx].to(device)
            else:
                x0_knn = X0_knn[idx].to(device)
                x1_knn = X1_knn[idx].to(device)
                x2_knn = X2_knn[idx].to(device)

            x0_knn = x0_knn.reshape(x0_knn.shape[0] * x0_knn.shape[1], x0_knn.shape[2])
            x1_knn = x1_knn.reshape(x1_knn.shape[0] * x1_knn.shape[1], x1_knn.shape[2])
            x2_knn = x2_knn.reshape(x2_knn.shape[0] * x2_knn.shape[1], x2_knn.shape[2])

            optimizer.zero_grad()
            _, _, _, _, kz0, kz1, kz2 = model(x0_knn, x1_knn, x2_knn)

            x0_bar, x1_bar, x2_bar, hidden, vz0, vz1, vz2 = model(x0, x1, x2)

            # cross-view contrastive loss
            cl_loss_0_all = torch.zeros(1).to(device)
            cl_loss_1_all = torch.zeros(1).to(device)
            cl_loss_2_all = torch.zeros(1).to(device)
            cl_loss_0_all += criterion_instance(vz0, vz0) / args.knn
            cl_loss_1_all += criterion_instance(vz1, vz1) / args.knn
            cl_loss_2_all += criterion_instance(vz2, vz2) / args.knn
            for i in range(args.knn):
                num_i = np.arange(len(idx)) * args.knn + i
                cl_loss_0 = criterion_instance(kz0[num_i], vz0) / args.knn
                cl_loss_1 = criterion_instance(kz1[num_i], vz1) / args.knn
                cl_loss_2 = criterion_instance(kz2[num_i], vz2) / args.knn
                cl_loss_0_all += cl_loss_0
                cl_loss_1_all += cl_loss_1
                cl_loss_2_all += cl_loss_2
            cl_loss = cl_loss_0_all + cl_loss_1_all + cl_loss_2_all

            # view-inter consistency loss
            con_loss_all = torch.zeros(1).to(device)
            for j in range(len(idx)):
                num_j = np.arange(args.knn) + j * args.knn
                con_loss_01 = wmse_loss(kz0[num_j], kz1[num_j])
                con_loss_02 = wmse_loss(kz0[num_j], kz2[num_j])
                con_loss_12 = wmse_loss(kz1[num_j], kz2[num_j])
                con_loss_all += (con_loss_01 + con_loss_02 + con_loss_12)

            # view-specific recons loss
            rec_loss = wmse_loss(x0_bar, x0) + wmse_loss(x1_bar, x1) + wmse_loss(x2_bar, x2)

            fusion_loss = rec_loss + args.lambda1 * cl_loss + args.lambda2 * con_loss_all
            total_loss += fusion_loss.item()
            fusion_loss.backward()
            optimizer.step()

        loss_q.put(total_loss)
        if loss_q.full():
            loss_q.get()
        mean_loss = np.mean(list(loss_q.queue))
        if np.abs(mean_loss - total_loss) <= 0.001 and epoch >= (args.AE_epoch * 0.6):
            print('Training stopped: epoch=%d, loss=%.4f, loss=%.4f' % (
                epoch, total_loss / (batch_idx + 1), mean_loss / (batch_idx + 1)))
            break
        if epoch % args.t == 0:
            _, _, _, hidden, _, _, _ = model(X0, X1, X2)
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20, random_state=20)
            y_pred = kmeans.fit_predict(hidden.data.cpu().numpy())
            acc, f1 = cluster_acc(y, y_pred)
            nmi = nmi_score(y, y_pred)
            ari = ari_score(y, y_pred)
        print("ae_epoch {} loss={:.4f} mean_loss={:.4f}".format(epoch,
                                                                total_loss / (batch_idx + 1),
                                                                mean_loss / (batch_idx + 1)))
        torch.save(model.state_dict(), args.train_MFC_path)
    print("model saved to {}.".format(args.train_MFC_path))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--knn', default=6, type=int, help='number of nodes for subgraph embedding')
    parser.add_argument('--start_ann', default=300, type=int)
    parser.add_argument('--lr_train', default=0.01, type=float)
    parser.add_argument('--lr_cluster', default=0.0001, type=float)
    parser.add_argument('--lambda1', default=0.01, type=float)
    parser.add_argument('--lambda2', default=0.01, type=float)
    parser.add_argument('--t', default=5, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--tol', default=1e-7, type=float)
    parser.add_argument('--CL_temperature', default=1, type=float)
    parser.add_argument('--max_epoch', default=200, type=int)
    parser.add_argument('--AE_epoch', default=1500, type=int)
    # Data
    parser.add_argument('--drop_index', default=1, type=int)
    parser.add_argument('--percent_del', type=int, default=3)
    parser.add_argument('--dataset', type=str, default='COIL20')
    parser.add_argument('--basis_train_path', type=str, default='../save_weight/COIL20/')
    args = parser.parse_args()
    start_time = time.time()

    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    args.train_MFC_flag = 0
    args.train_MFC_path = args.basis_train_path + 'new_manualSeed_' + str(manual_seed) + '_drop_' + str(
        args.drop_index) + '_k_' + str(args.knn) + '_percentDel_0.' + str(args.percent_del) + '_startAnn_' + str(
        args.start_ann) + '_aelr_' + str(args.lr_train) + '_lambda1_' + str(args.lambda1) + '_lambda2_' + str(
        args.lambda2) + '_aeproches_' + str(args.AE_epoch) + '.pkl'

    ####################################################################
    # Load data, label, incomplete_index_matrix, and knn_index_matrix
    ####################################################################

    data = scipy.io.loadmat('../data/' + args.dataset + '/' + args.dataset + '.mat')
    label = data['Y']
    label = label.reshape(-1)
    label = np.array(label, 'float64')
    args.n_clusters = len(np.unique(label))
    y = label
    X = data['X'].T.squeeze()
    args.n_views = X.shape[0]

    dropMatrix = scipy.io.loadmat(
        '../data/' + args.dataset + '/' + args.dataset + '_0.' + str(args.percent_del) + '.mat')
    if args.drop_index is None:
        dM = dropMatrix['IN_Index']
    else:
        dM = dropMatrix['IN_Index'][args.drop_index]

    distanceM = np.load('../data/' + args.dataset + '/' + args.dataset + '_disMat.npy', allow_pickle=True).item()

    ####################################################################
    # Preprocessing
    ####################################################################
    # Obtain the existing knn_feature_matrix
    disMat = {}
    for i in range(len(distanceM)):
        del_index = np.array(np.where(dM[:, i] == 0)).squeeze()
        final_shape = np.delete(distanceM[i], del_index, 0).shape[0]
        disMat[i] = np.zeros((distanceM[i].shape[0], final_shape))
        for ii in range(disMat[i].shape[0]):
            if ii not in del_index:
                disMat[i][ii] = np.delete(distanceM[i][ii], np.where(distanceM[i][ii] == del_index[:, None])[1])
    del label, data, distanceM, dropMatrix, del_index

    # view-specific data 79, 1750 features
    X0 = np.array(X[0], 'float64').reshape((y.shape[0], -1))
    X1 = np.array(X[1], 'float64').reshape((y.shape[0], -1))
    X2 = np.array(X[2], 'float64')
    args.n_input = [X0.shape[1], X1.shape[1], X2.shape[1]]

    # For each view,
    iv = 0
    # obtain drop and exist array
    WEiv = np.copy(dM[:, iv])
    ind_0_complete = np.where(WEiv == 1)
    ind_0_complete = (np.array(ind_0_complete)).reshape(-1)
    ind_0_dropped = np.where(WEiv == 0)
    ind_0_dropped = (np.array(ind_0_dropped)).reshape(-1)
    # obtain the knn_feature of dropping sample
    temp_dis_dict0 = {}
    for i in range(len(y)):
        exist_view = (np.array(np.where(dM[i] == 1))).reshape(-1)
        if iv in exist_view:
            temp_dis_dict0[i] = []
            temp_dis_dict0[i].append(disMat[iv][i][1:100].astype(int))
        else:
            j = 0
            for ii in np.array(exist_view):
                if j == 0:
                    temp_dis_dict0[i] = []
                    temp_dis_dict0[i].append(disMat[ii][i][1:100].astype(int))
                    j += 1
                else:
                    temp_dis_dict0[i].append(disMat[ii][i][1:100].astype(int))
            if len(temp_dis_dict0[i]) == 2:
                temp_dis_dict0[i] = np.concatenate((temp_dis_dict0[i][0], temp_dis_dict0[i][1]))
        temp_dis_dict0[i] = np.setdiff1d(temp_dis_dict0[i], ind_0_dropped, True)
        temp_dis_dict0[i] = temp_dis_dict0[i][:args.knn]
    # normalize
    X0[ind_0_complete, :] = StandardScaler().fit_transform(X0[ind_0_complete, :])
    X0[ind_0_dropped, :] = 0
    X0_re, X0_knn_feature, graph_init_error0 = rebuild(X0, ind_0_dropped, temp_dis_dict0, args.knn, y)

    iv = 1
    # obtain drop and exist array
    WEiv = np.copy(dM[:, iv])
    ind_1_complete = np.where(WEiv == 1)
    ind_1_complete = (np.array(ind_1_complete)).reshape(-1)
    ind_1_dropped = np.where(WEiv == 0)
    ind_1_dropped = (np.array(ind_1_dropped)).reshape(-1)
    # obtain the adj of dropping sample
    temp_dis_dict1 = {}
    for i in range(len(y)):
        exist_view = (np.array(np.where(dM[i] == 1))).reshape(-1)
        if iv in exist_view:
            temp_dis_dict1[i] = []
            temp_dis_dict1[i].append(disMat[iv][i][1:100].astype(int))
        else:
            j = 0
            for ii in np.array(exist_view):
                if j == 0:
                    temp_dis_dict1[i] = []
                    temp_dis_dict1[i].append(disMat[ii][i][1:100].astype(int))
                    j += 1
                else:
                    temp_dis_dict1[i].append(disMat[ii][i][1:100].astype(int))
            if len(temp_dis_dict1[i]) == 2:
                temp_dis_dict1[i] = np.concatenate((temp_dis_dict1[i][0], temp_dis_dict1[i][1]))
        temp_dis_dict1[i] = np.setdiff1d(temp_dis_dict1[i], ind_1_dropped, True)
        temp_dis_dict1[i] = temp_dis_dict1[i][:args.knn]
    # normalize
    X1[ind_1_complete, :] = StandardScaler().fit_transform(X1[ind_1_complete, :])
    X1[ind_1_dropped, :] = 0
    X1_re, X1_knn_feature, graph_init_error1 = rebuild(X1, ind_1_dropped, temp_dis_dict1, args.knn, y)

    iv = 2
    # obtain drop and exist array
    WEiv = np.copy(dM[:, iv])
    ind_2_complete = np.where(WEiv == 1)
    ind_2_complete = (np.array(ind_2_complete)).reshape(-1)
    ind_2_dropped = np.where(WEiv == 0)
    ind_2_dropped = (np.array(ind_2_dropped)).reshape(-1)
    # obtain the adj of dropping sample
    temp_dis_dict2 = {}
    for i in range(len(y)):
        exist_view = (np.array(np.where(dM[i] == 1))).reshape(-1)
        j = 0
        if iv in exist_view:
            temp_dis_dict2[i] = []
            temp_dis_dict2[i].append(disMat[iv][i][1:100].astype(int))
        else:
            for ii in np.array(exist_view):
                if j == 0:
                    temp_dis_dict2[i] = []
                    temp_dis_dict2[i].append(disMat[ii][i][1:100].astype(int))
                    j += 1
                else:
                    temp_dis_dict2[i].append(disMat[ii][i][1:100].astype(int))
            if len(temp_dis_dict2[i]) == 2:
                temp_dis_dict2[i] = np.concatenate((temp_dis_dict2[i][0], temp_dis_dict2[i][1]))
        temp_dis_dict2[i] = np.setdiff1d(temp_dis_dict2[i], ind_2_dropped, True)
        temp_dis_dict2[i] = temp_dis_dict2[i][:args.knn]
    # normalize
    X2[ind_2_complete, :] = StandardScaler().fit_transform(X2[ind_2_complete, :])
    X2[ind_2_dropped, :] = 0
    X2_re, X2_knn_feature, graph_init_error2 = rebuild(X2, ind_2_dropped, temp_dis_dict2, args.knn, y)
    del iv, WEiv, X0, X1, X2
    X0_res = np.nan_to_num(X0_re)
    X1_res = np.nan_to_num(X1_re)
    X2_res = np.nan_to_num(X2_re)

    X0_knn_feature = np.nan_to_num(X0_knn_feature)
    X1_knn_feature = np.nan_to_num(X1_knn_feature)
    X2_knn_feature = np.nan_to_num(X2_knn_feature)

    ##################################################################################
    # TrainProcess-1: View-specific Reconstruction Loss + Contrastive Loss (Shuffled samples)
    ##################################################################################

    X0 = torch.Tensor(X0_res).to(device)
    X1 = torch.Tensor(X1_res).to(device)
    X2 = torch.Tensor(X2_res).to(device)

    X0_knn = torch.Tensor(X0_knn_feature).to(device)
    X1_knn = torch.Tensor(X1_knn_feature).to(device)
    X2_knn = torch.Tensor(X2_knn_feature).to(device)

    model = MFC(
        n_stacks=4,
        n_input=args.n_input,
        n_z=args.n_clusters,
        n_clusters=args.n_clusters,
        train_path=args.train_MFC_path).to(device)

    criterion_instance = InstanceLoss(args.batch_size, args.CL_temperature, 0).to(device)

    model.train()
    #######################################################
    # obtain the k-means clustering assignments based on the train data
    #######################################################
    hidden, _, _, _, _ = model(X0, X1, X2)
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20, random_state=20)
    y_pred = kmeans.fit_predict(hidden.data.cpu().numpy())
    acc, f1 = cluster_acc(y, y_pred)
    nmi = nmi_score(y, y_pred)
    ari = ari_score(y, y_pred)
    print(acc, nmi, ari, f1)
    del hidden, kmeans
    # re-order according to the pre-assignment, requiring the same class together
    X0_train = np.zeros(X0_res.shape)
    X1_train = np.zeros(X1_res.shape)
    X2_train = np.zeros(X2_res.shape)
    label = np.zeros(y.shape)
    basis_index = 0
    for li in range(args.n_clusters):
        index_li = np.where(y_pred == li)
        index_li = (np.array(index_li)).reshape(-1)

        X0_train[np.arange(len(index_li)) + basis_index, :] = np.copy(X0_res[index_li])
        X1_train[np.arange(len(index_li)) + basis_index, :] = np.copy(X1_res[index_li])
        X2_train[np.arange(len(index_li)) + basis_index, :] = np.copy(X2_res[index_li])

        label[np.arange(len(index_li)) + basis_index] = np.copy(y[index_li])
        basis_index = basis_index + len(index_li)

    X0 = np.copy(X0_train)
    X1 = np.copy(X1_train)
    X2 = np.copy(X2_train)

    X0 = torch.Tensor(X0).to(device)
    X1 = torch.Tensor(X1).to(device)
    X2 = torch.Tensor(X2).to(device)
    ######################################################
    # TrainProcess-2: Obtain final clustering assignments using K-means
    ######################################################
    optimizer = Adam(model.parameters(), lr=args.lr_cluster)
    hidden, q, _, _, _ = model(X0, X1, X2)
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20, random_state=20)
    hidden = np.nan_to_num(hidden.data.cpu().numpy())
    y_pred = kmeans.fit_predict(hidden)
    del hidden
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    acc, f1 = cluster_acc(label, y_pred)
    nmi = nmi_score(label, y_pred)
    ari = ari_score(label, y_pred)
    print(':Acc {:.4f}'.format(acc), 'nmi {:.4f}'.format(nmi), 'ari {:.4f}'.format(ari), 'f1 {:.4f}'.format(f1))
    y_pred_last = y_pred

    best_acc2 = 0
    best_nmi2 = 0
    best_ari2 = 0
    best_f12 = 0
    best_epoch = 0
    total_loss = 0

    for epoch in range(int(args.max_epoch)):

        if epoch % 1 == 0:
            _, tmp_q, _, _, _ = model(X0, X1, X2)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

            y_pred = tmp_q.cpu().numpy().argmax(1)
            acc, f1 = cluster_acc(label, y_pred)
            nmi = nmi_score(label, y_pred)
            ari = ari_score(label, y_pred)
            if acc > best_acc2:
                best_acc2 = np.copy(acc)
                best_nmi2 = np.copy(nmi)
                best_ari2 = np.copy(ari)
                best_f12 = np.copy(f1)
                best_epoch = epoch
            print('best_Iter {}'.format(best_epoch), ':best_Acc2 {:.4f}'.format(best_acc2), 'Iter {}'.format(epoch),
                  ':Acc {:.4f}'.format(acc),
                  ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari), 'total_loss {:.4f}'.format(total_loss))
            total_loss = 0
            # check stop criterion
            delta_y = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = np.copy(y_pred)
            if epoch > 80 and delta_y < args.tol:
                print('Training stopped: epoch=%d, delta_label=%.4f, tol=%.4f' % (epoch, delta_y, args.tol))
                break

        y_pred = torch.tensor(y_pred)
        index_array = np.arange(X0.shape[0])

        for batch_idx in range(np.int_(np.ceil(X0.shape[0] / args.batch_size))):
            idx = index_array[batch_idx * args.batch_size: min((batch_idx + 1) * args.batch_size, X0.shape[0])]
            x0 = X0[idx].to(device)
            x1 = X1[idx].to(device)
            x2 = X2[idx].to(device)

            optimizer.zero_grad()
            hidden, q, vz0, vz1, vz2 = model(x0, x1, x2)

            if np.isnan(hidden.data.cpu().numpy()).any():
                break

            kl_loss = F.kl_div(q.log(), p[idx], reduction='batchmean')
            fusion_loss = kl_loss
            total_loss += fusion_loss
            fusion_loss.backward()
            optimizer.step()



