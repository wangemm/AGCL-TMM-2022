import numpy as np
import scipy.io as scio
import argparse
import math


def del_sq2(n, v, p):
    delM = np.zeros((10, n, v))
    for i in range(10):
        np.random.seed(i)
        d_index = np.arange(n)
        np.random.shuffle(d_index)
        complete_mark = int(math.ceil(n / v))
        drop_mark = int(math.ceil(n * (10-p) / 10))
        if drop_mark < complete_mark:
            return None
        del_matrix = np.zeros(shape=(n, v))
        for ii in range(v):
            c_index = d_index[complete_mark * ii: complete_mark * (ii + 1)]
            del_matrix[c_index, ii] = 1

            o_index = np.setdiff1d(d_index, c_index)
            np.random.shuffle(o_index)
            del_matrix[o_index[:(drop_mark-complete_mark)], ii] = 1
        print(np.shape(np.where(del_matrix[:, 0] == 0)))
        print(np.shape(np.where(del_matrix[:, 1] == 0)))
        delM[i] = del_matrix
    return delM


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='COIL20',
                        help='HW_5view, BDGP_4view, COIL20')
    parser.add_argument('--percentDel', type=int, default=7)
    args = parser.parse_args()
    data_path = './' + args.dataset + '/' + args.dataset + '.mat'
    data = scio.loadmat(data_path)

    label = data['Y'].squeeze()
    views = data['X'].shape[1]

    print('Number of samples:', label.shape[0])
    print('Number of views:', views)

    for j in range(1, args.percentDel):
        print(j)
        del_mat = del_sq2(label.shape[0], views, j)
        zero_mark = 0
        for i in range(len(del_mat)):
            if np.sum(del_mat[i]) < 1:
                zero_mark += 1
        print(zero_mark)
        if zero_mark == 0:
            scio.savemat('./' + args.dataset + '/' + args.dataset + '_0.' + str(j) + '.mat',
                         {'IN_Index': del_mat})
