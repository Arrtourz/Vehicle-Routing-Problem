from settings import data_path
import numpy as np
import pandas as pd
import json
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from torch_inputs import *

class DataIterator:
    ''' Iterator class '''
    def __init__(self, dataset, batchsize, device):
        # object reference (a list of tuples of lists)
        self._dataset = dataset
        self._len = len(dataset[0])
        # keep track of current index
        self._index = 0
        # the batch size
        self._batchsize = batchsize
        self._device = device

    def __next__(self):
        ''''Returns the next value from object's lists '''

        n = min(self._len, self._index + self._batchsize)
        if self._index < self._len:
            p = self._dataset[0][self._index:n]
            v = self._dataset[1][self._index:n]
            y = self._dataset[2][self._index:n]
            ## flatten y array in
            y_transformed = [y[i].flatten() for i in range(len(y))]
            self._index = n
            return (Ten(p).to(self._device),
                    Ten(v).to(self._device),
                    Ten(y_transformed).to(self._device))
                   # Ten(y).to(self._device))
        # End of Iteration
        raise StopIteration


class Dataset:
    ''' Class containing the data samples organized as follows:
        {p_i, v_i, y_i}_{i \in [N]}

        where p_i are the samples associated to passengers:

        p_i = <pOL^j, pOT^j, pDL^j, pDT^j>_{j \in R}
        where
            - pOL^j = Origin Latitude of request j
            - pOT^j = Origin Longitude of request j
            - pDL^j = Destination Latitude of request j
            - pDT^j = Destination Longitude of request j
            R = set of requests

        v_i = <vOL^j, vOT^j>_{j \in V}
        where
            - vOL^j = Origin Latitude of vehicle j
            - vOT^j = Origin Longitude of vehicle j
            V = set of vehicles

        y_i \in {0,1}^{|R| x |V|}
            is a 0-1 matrix whose entry:

            y_i[j,k] = 1 <->  request j is associated to vehicle k
    '''
    def __init__(self, params, mode, device):
        assert mode in ['train', 'test', 'valid']

        self._dataset = list()
        self._batchsize = params['batch_size']
        self._pfeat = list()
        self._vfeat = list()
        self._yfeat = None
        self._indices = list()
        self._device = device
        np.random.seed(params['seed'])
        P, V, Y, IDs = load_dataset(params)
        indexes = self._get_indexes(params, P, mode)
        _P = P.values[indexes]
        _V = V.values[indexes]
        _Y = Y[indexes]
        self._dataset = tuple([_P, _V, _Y])
        # self._ids = IDs
        self._ids = {'p': np.array(IDs['p'])[indexes],
                     'v': np.array(IDs['v'])[indexes]}

    @property
    def size_p(self):
        return self._ids['p'].shape[1]

    @property
    def size_v(self):
        return self._ids['v'].shape[1]

    def __iter__(self):
        return DataIterator(self._dataset, self._batchsize, self._device)

    ''' Compute dataset indexes '''
    def _get_indexes(self, params, D, mode):
        indices = np.arange(len(D))
        np.random.shuffle(indices)
        split_size = dict()
        modeidx = {'train': 0, 'test': 1, 'valid': 2}
        for m in ['train', 'test', 'valid']:
            split_size[m] = int(params['split'][modeidx[m]] * len(D))
        if mode == 'train':
            indices = indices[0:split_size['train']]
        elif mode == 'test':
            indices = indices[split_size['train']:split_size['test'] + split_size['train']]
        else:
            indices = indices[split_size['train'] + split_size['test']:-1]
        return indices

def load_dataset(params):
    p_samples = []
    v_samples = []
    y_samples = []
    ids = {'p': [], 'v': []}
    for fnum in range(params['n_samples']):
        with open(data_path + 'requests/'+str(fnum)+'.json', 'r') as rfile:
            p_data = json.load(rfile)
        with open(data_path + 'vehicles/'+str(fnum)+'.json', 'r') as rfile:
            v_data = json.load(rfile)
        with open(data_path + 'formatted-output/'+str(fnum)+'.json', 'r') as rfile:
            o_data = json.load(rfile)

        p_sample = {'p'+str(i)+'_'+s : [] for i in range(len(p_data)) for s in ['o_lat', 'o_lon', 'd_lat', 'd_lon']}
        v_sample = {'v'+str(i)+'_'+s : [] for i in range(len(v_data)) for s in ['o_lat', 'o_lon']}
        p_id, v_id = [], []

        for (i,v) in enumerate(p_data):
            p_id.append(v['passenger_id'])
            #p_sample['p_count'].append(v['passenger_count'])
            for _dir in ['lat', 'lon']:
                p_sample['p'+str(i)+'_o_'+_dir].append(v['origin'][_dir])
                p_sample['p'+str(i)+'_d_'+_dir].append(v['destination'][_dir])

        for (i,v) in enumerate(v_data):
            v_id.append(v['vehicle_id'])
            for _dir in ['lat', 'lon']:
                v_sample['v'+str(i)+'_o_'+_dir].append(v['origin'][_dir])

        y = np.zeros((len(p_id), len(v_id)), dtype=int)
        for (i,vid) in enumerate(v_id):
            for (j, pid) in enumerate(p_id):
                if vid in o_data and str(pid) in o_data[vid]:
                    y[j,i] = 1

        p_samples.append(pd.DataFrame(p_sample))
        v_samples.append(pd.DataFrame(v_sample))
        y_samples.append(y)
        ids['p'].append(p_id)
        ids['v'].append(v_id)

    return (pd.concat(p_samples), pd.concat(v_samples), np.array(y_samples), ids)




if __name__ == '__main__':
    PARAMS = {'seed': 1234,
              'batch_size': 10,
              'n_samples': 1000,
              'split': [0.7, 0.2, 0.1]}
    dataset = Dataset(PARAMS, "train", "cpu")
