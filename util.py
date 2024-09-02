import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, begin=0, days=288, pad_with_last_sample=False):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        self.ind = np.arange(begin, begin + self.size)
        self.days = days

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.ind = self.ind[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                i_i = self.ind[start_ind: end_ind, ...] % self.days
                # xi_i = np.tile(np.arange(x_i.shape[1]), [x_i.shape[0], x_i.shape[2], 1, 1]).transpose(
                #     [0, 3, 1, 2]) + self.ind[start_ind: end_ind, ...].reshape([-1, 1, 1, 1])
                # x_i = np.concatenate([x_i, xi_i % self.days / self.days, np.eye(7)[xi_i // self.days % 7].squeeze(-2)],
                #                      axis=-1)
                yield (x_i, y_i, i_i)
                self.current_ind += 1

        return _wrapper()

class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(adj_filename):
    adj_mx = np.load(adj_filename)
    print('adj_mx: ', adj_mx.shape)
    adj = [asym_adj(adj_mx)]
    return adj

def load_dataset(dataset_dir, batch_size, valid_batch_size, test_batch_size, input_length=12, output_length=12):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data[category + '1'] = cat_data['x'][:, :12, :, 0:2].astype(np.float32)  # B T N F speed flow
        data[category + '2'] = cat_data['y'][:, :12, :, 0:1].astype(np.float32)

        # data[category + '1'] = np.where(np.isfinite(data[category + '1']), data[category + '1'], 0) 
        # data[category + '2'] = np.where(np.isfinite(data[category + '1']), data[category + '1'], 0)  


        #data[category] = np.load(os.path.join(dataset_dir, category + '.npz'))
        #print(data[category].files)
        # data[category][..., 1] = data[category][..., 1]*288
        #print('*'*10, category, data[category].shape, '*'*10)
    scaler = StandardScaler(mean=np.nanmean(data['train1'][..., 0]), std=np.nanstd(data['train1'][..., 0]))
    #scaler = StandardScaler(mean=np.nanmean(data['train1'][..., 0].astype(np.float64)).astype(np.float16), std=np.nanstd(data['train1'][..., 0].astype(np.float64)).astype(np.float16))
    # Data format
    for category in ['train', 'val', 'test']:
        data[category + '1'][..., 0:1] = scaler.transform(data[category + '1'][..., 0:1])
        data[category + '2'][..., 0:1] = scaler.transform(data[category + '2'][..., 0:1])
    data['train_loader'] = DataLoader(data['train1'], data['train2'], batch_size)
    data['val_loader'] = DataLoader(data['val1'], data['val2'], valid_batch_size)
    data['test_loader'] = DataLoader(data['test1'], data['test2'], test_batch_size)
    data['scaler'] = scaler
    return data
