import pickle
import numpy as np
import gc
from pathlib import Path
from keras.utils import Sequence
from tqdm import tqdm
from helpers.normalization import normalize


class DataGenerator(Sequence):
    def __init__(self, data, batch_size, profile_inputs, actuators, targets, lookback, predict_deltas):
        self.batch_size = batch_size
        self.data = data
        self.profile_inputs = profile_inputs
        self.actuators = actuators
        self.targets = targets
        self.lookback = lookback
        self.predict_deltas = predict_deltas

    def __len__(self):
        return int(np.ceil(len(self.data['time']) / float(self.batch_size)))

    def __getitem__(self, idx):
        inp = {}
        targ = {}
        for sig in self.profile_inputs:
            inp['input_' + sig] = self.data[sig][idx * self.batch_size:
                                                 (idx+1)*self.batch_size,
                                                 0: self.lookback]
        for sig in self.actuators:
            inp['input_' + sig] = self.data[sig][idx * self.batch_size:
                                                 (idx+1)*self.batch_size]
        for sig in self.targets:
            targ['target_' + sig] = self.data[sig][idx * self.batch_size:
                                                   (idx+1)*self.batch_size, -1]
            if self.predict_deltas:
                targ['target_' + sig] -= self.data[sig][idx * self.batch_size:
                                                        (idx+1)*self.batch_size,
                                                        self.lookback]
        return inp, targ


def process_data(rawdata, sig_names, normalization_method, window_length=1,
                 window_overlap=0, lookback=5, lookahead=3, sample_step=5,
                 uniform_normalization=True, train_frac=0.7, val_frac=0.2, nshots=None):
    """Organize data into correct format for training

    Gathers raw data into bins, group into training sequences, normalize, 
    and split into training and validation sets.

    Args:
        rawdata (dict): Nested dictionary of raw signal data, or path to pickle. 
            Should be of the form rawdata[shot][signal_name] = signal_data.
        sig_names (list): List of signal names as strings.
        normalization_method (str): One of `StandardScaler`, `MinMax`, `MaxAbs`,
            `RobustScaler`, `PowerTransform`.
        window_length (int): Number of samples to average over in each bin/window.
        window_overlap (int): How many timesteps to overlap windows.  
        lookback (int): How many window lengths for lookback.
        lookahead (int): How many window lengths to predict into the future.
        sample_step (int): How much to offset sequential training sequences. 
            Step of 1 means sample[i] and sample[i+1] will be offset by 1, with 
            the rest overlapping.
        uniform_normalization (bool): 'True' uses the same normalization 
            parameters over a whole profile, 'False' normalizes each spatial 
            point separately.
        val_frac (float): Fraction of samples to use for validation.
        nshots (int): How many shots to use. If None, all available will be used.
    Returns:
        traindata (dict): Dictionary of numpy arrays, one entry for each signal.
            Each array has shape [nsamples,lookback+lookahead,signal_shape]
        valdata (dict): Dictionary of numpy arrays, one entry for each signal.
            Each array has shape [nsamples,lookback+lookahead,signal_shape]
        param_dict (dict): Dictionary of parameters used during normalization,
            to be used for denormalizing later. Eg, mean, stddev, method, etc.
    """
    sig_names = list(np.unique(sig_names))
    if type(rawdata) is not dict:
        abs_path = Path(rawdata).resolve()
        if abs_path.exists():
            with open(abs_path, 'rb') as f:
                rawdata = pickle.load(f, encoding='latin1')
        else:
            raise IOError("No such path to data file")
    # find which shots have all the signals needed
    if 'time' not in sig_names:
        # should be there for all shots, used as reference length
        sig_names += ['time']
    print('Signals: ' + ', '.join(sig_names))
    usabledata = []
    for shot in rawdata.keys():
        if set(sig_names).issubset(set(rawdata[shot].keys())) \
           and rawdata[shot]['time'].size > (lookback+lookahead):
            usabledata.append(rawdata[shot])
    usabledata = np.array(usabledata)
    del rawdata
    gc.collect()
    print('Number of useable shots: ', str(len(usabledata)))
    usabledata = usabledata[np.random.permutation(len(usabledata))]
    if nshots is not None:
        usabledata = usabledata[:np.minimum(nshots, len(usabledata))]
    t = 0
    for shot in usabledata:
        t += shot['time'].size
    print('Total number of timesteps: ', str(t))

    def binavg(array, start):
        """averages over bins"""
        return np.mean(array[start:start+window_length], axis=0)

    alldata = {}
    for sig in sig_names:
        alldata[sig] = []  # initalize empty lists
    for shot in tqdm(usabledata, desc='Gathering', ascii=True, dynamic_ncols=True):
        for sig in sig_names:
            temp = shot[sig]
            nbins = int(temp.shape[0]/(window_length-window_overlap))
            shotdata = []
            for i in range(nbins):
                # populate array of binned/windowed data for each shot
                shotdata.append(binavg(temp, i*(window_length-window_overlap)))
            shotdata = np.stack(shotdata)
            for i in range(lookback, nbins-lookahead, sample_step):
                # group into arrays of input/output pairs
                alldata[sig].append(shotdata[i-lookback:i+lookahead])
    del usabledata
    gc.collect()
    for sig in tqdm(sig_names, desc='Stacking', ascii=True, dynamic_ncols=True):
        alldata[sig] = np.stack(alldata[sig])
    alldata, normalization_params = normalize(
        alldata, normalization_method, uniform_normalization)
    nsamples = alldata['time'].shape[0]
    print('Total number of samples: ', str(nsamples))
    inds = np.random.permutation(nsamples)
    traininds = inds[:int(nsamples*train_frac)]
    valinds = inds[int(nsamples*train_frac)
                       :int(nsamples*(val_frac+train_frac))]
    print('Number of training samples: ', str(traininds.size))
    print('Number of validation samples: ', str(valinds.size))

    traindata = {}
    valdata = {}
    for sig in tqdm(sig_names, desc='Splitting', ascii=True, dynamic_ncols=True):
        traindata[sig] = alldata[sig][traininds]
        valdata[sig] = alldata[sig][valinds]
    return traindata, valdata, normalization_params
