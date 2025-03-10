from __future__ import absolute_import
import scipy.io as scio
import numpy as np
import networkx as nx
import scipy.sparse as sp
from pairs import pair
import argparse
import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='COIL20')
    parser.add_argument('--metrix', type=str, default=None)
    args = parser.parse_args()

    data = scio.loadmat(args.dataset + '/' + args.dataset + '.mat')
    label = data['Y']
    label = label.reshape(-1)
    label = np.array(label, 'float64')
    args.n_clusters = len(np.unique(label))
    X = data['X'].T.squeeze()
    args.n_views = X.shape[0]
    for i in range(X.shape[0]):
        X[i] = X[i].reshape((X[i].shape[0], -1))
    idx_dict = {}
    err = []

    for i in range(len(X)):
        if args.metrix is None:
            me = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
                  'manhattan']
        else:
            me = args.metrix
        idx_all = []
        e_all = []
        for met in me:
            start_time = time.time()
            id, e = pair(X[i], label, metrix=met)
            print(i, met, e, (time.time() - start_time))
            idx_all.append(id)
            e_all.append(e)

        err.append(min(e_all))
        idx_dict[i] = idx_all[e_all.index(min(e_all))]
        print('View-{}: Best metrix-{}: 10-nn error:{}'.format(i + 1, me[e_all.index(min(e_all))], min(e_all)))

    save_all_file = args.dataset + '/' + args.dataset + '_disMat.npy'
    np.save(save_all_file, idx_dict)

    print(time.time() - start_time)
