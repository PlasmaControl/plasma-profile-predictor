import pickle
import numpy as np
import gc
from pathlib import Path
from keras.utils import Sequence
from keras.callbacks import TensorBoard
from tqdm import tqdm
from helpers.normalization import normalize


class DataGenerator(Sequence):
    def __init__(self, data, batch_size, profile_inputs, actuator_inputs, targets,
                 profile_lookback, actuator_lookback, lookahead, predict_deltas):
        """Make a data generator for training or validation data

        Args:
            data: dict of data arrays to draw from.
            batch_size (int): size of each batch.
            profile_inputs (str): List of names of profile inputs, as strings.
            actuator_inputs (str): List of names of actuator inputs, as strings.        
            targets (str): List of names of profile targets, as strings.
            profile_lookback (int): Number of previous steps for profile data.
            actuator_lookback (int): Number of previous steps for actuator data.
            lookahead (int): How many steps ahead to predict (prediction window)
            predict_deltas (bool): Whether to predict changes or full profiles.
        """

        self.batch_size = batch_size
        self.data = data
        self.profile_inputs = profile_inputs
        self.actuator_inputs = actuator_inputs
        self.targets = targets
        self.profile_lookback = profile_lookback
        self.actuator_lookback = actuator_lookback
        self.lookahead = lookahead
        self.predict_deltas = predict_deltas
        self.cur_shotnum = np.zeros(self.batch_size)
        self.cur_times = np.zeros(self.batch_size)

    def __len__(self):
        return int(np.ceil(len(self.data['time']) / float(self.batch_size)))

    def __getitem__(self, idx):
        inp = {}
        targ = {}
        self.cur_shotnum = self.data['shotnum'][idx * self.batch_size:
                                                (idx+1)*self.batch_size]
        self.cur_times = self.data['time'][idx * self.batch_size:
                                           (idx+1)*self.batch_size]
        for sig in self.profile_inputs:
            inp['input_' + sig] = self.data[sig][idx * self.batch_size:
                                                 (idx+1)*self.batch_size,
                                                 self.actuator_lookback-self.profile_lookback: max(self.profile_lookback, self.actuator_lookback)]
        for sig in self.actuator_inputs:
            inp['input_' + sig] = self.data[sig][idx * self.batch_size:
                                                 (idx+1)*self.batch_size,
                                                 0:self.actuator_lookback+self.lookahead]
        for sig in self.targets:
            if self.predict_deltas:
                baseline = self.data[sig][idx * self.batch_size:(idx+1)*self.batch_size,
                                          max(self.profile_lookback, self.actuator_lookback)-1]
            else:
                baseline = 0
            targ['target_' + sig] = self.data[sig][idx * self.batch_size:
                                                   (idx+1)*self.batch_size,
                                                   max(self.profile_lookback, self.actuator_lookback)+self.lookahead-1] - baseline

        return inp, targ


def process_data(rawdata, sig_names, normalization_method, window_length=1,
                 window_overlap=0, lookback=5, lookahead=3, sample_step=5,
                 uniform_normalization=True, train_frac=0.7, val_frac=0.2,
                 nshots=None, verbose=1):
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
        verbose (int): verbosity level. 0 is no CL output, 1 shows progress.
    Returns:
        traindata (dict): Dictionary of numpy arrays, one entry for each signal.
            Each array has shape [nsamples,lookback+lookahead,signal_shape]
        valdata (dict): Dictionary of numpy arrays, one entry for each signal.
            Each array has shape [nsamples,lookback+lookahead,signal_shape]
        param_dict (dict): Dictionary of parameters used during normalization,
            to be used for denormalizing later. Eg, mean, stddev, method, etc.
    """
    verbose = bool(verbose)
    sig_names = list(np.unique(sig_names))
    if type(rawdata) is not dict:
        if verbose:
            print('Loading')
        abs_path = Path(rawdata).resolve()
        if abs_path.exists():
            with open(abs_path, 'rb') as f:
                rawdata = pickle.load(f, encoding='latin1')
        else:
            raise IOError("No such path to data file")
    if 'time' not in sig_names:
        # should be there for all shots, used as reference length
        sig_names += ['time']
    if 'shotnum' not in sig_names:
        sig_names += ['shotnum']
    if verbose:
        print('Signals: ' + ', '.join(sig_names))
    usabledata = []
    # find which shots have all the signals needed
    for shot in rawdata.keys():
        rawdata[shot]['shotnum'] = np.ones(rawdata[shot]['time'].shape[0])*shot
        if set(sig_names).issubset(set(rawdata[shot].keys())) \
           and rawdata[shot]['time'].size > (lookback+lookahead):
            usabledata.append(rawdata[shot])
    usabledata = np.array(usabledata)
    del rawdata
    gc.collect()
    nshots = np.minimum(nshots, len(usabledata))
    if verbose:
        print('Number of useable shots: ', str(len(usabledata)))
        print('Number of shots used: ', str(nshots))
    usabledata = usabledata[np.random.permutation(len(usabledata))]
    usabledata = usabledata[:nshots]
    if verbose:
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
    for shot in tqdm(usabledata, desc='Gathering', ascii=True, dynamic_ncols=True,
                     disable=not verbose):
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
    for sig in tqdm(sig_names, desc='Stacking', ascii=True, dynamic_ncols=True,
                    disable=not verbose):
        alldata[sig] = np.stack(alldata[sig])
    alldata, normalization_params = normalize(
        alldata, normalization_method, uniform_normalization, verbose)
    nsamples = alldata['time'].shape[0]
    inds = np.random.permutation(nsamples)
    traininds = inds[:int(nsamples*train_frac)]
    valinds = inds[int(nsamples*train_frac):int(nsamples*(val_frac+train_frac))]
    traindata = {}
    valdata = {}
    for sig in tqdm(sig_names, desc='Splitting', ascii=True, dynamic_ncols=True,
                    disable=not verbose):
        traindata[sig] = alldata[sig][traininds]
        valdata[sig] = alldata[sig][valinds]
    if verbose:
        print('Total number of samples: ', str(nsamples))
        print('Number of training samples: ', str(traininds.size))
        print('Number of validation samples: ', str(valinds.size))
    return traindata, valdata, normalization_params
