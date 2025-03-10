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
        paired_mark = int(math.ceil(n * p / 10))
        del_matrix = np.zeros(shape=(n, v))
        del_matrix[d_index[:paired_mark]] = 1
        o_index = d_index[paired_mark:]
        l = math.ceil(o_index.shape[0] / 2)
        del_matrix[o_index[:l], 0] = 1
        del_matrix[o_index[l:], 1] = 1
        delM[i] = del_matrix
        print(np.shape(np.where(del_matrix[:, 0] == 0)))
        print(np.shape(np.where(del_matrix[:, 1] == 0)))
    return delM


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='SampledMNIST, Caltech101-20, BDGP_2view, Animal, MNIST')
    parser.add_argument('--percentDel', type=int, default=8)
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
