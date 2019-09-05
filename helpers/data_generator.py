import pickle
import numpy as np
import gc
from pathlib import Path
from keras.utils import Sequence
from keras.callbacks import TensorBoard
from helpers.normalization import normalize
from tqdm import tqdm


class DataGenerator(Sequence):
    def __init__(self, data, batch_size, input_profile_names, actuator_names,
                 target_profile_names, scalar_input_names,
                 lookbacks, lookahead, predict_deltas,
                 profile_downsample, **kwargs):
        """Make a data generator for training or validation data

        Args:
            data: dict of data arrays to draw from.
            batch_size (int): size of each batch.
            input_profile_names (str): List of names of profile inputs, as strings.
            actuator_names (str): List of names of actuator inputs, as strings.
            target_profile_names (str): List of names of profile targets, as strings.
            scalar_input_names (str): List of names of scalar inputs (shape parameters etc).
            lookbacks (dict): Dictionary of lookback values for each input signal name.
            lookahead (int): How many steps ahead to predict (prediction window)
            predict_deltas (bool): Whether to predict changes or full profiles.
            profile_downsample (int): How much to downsample the profile data.
        """
        self.data = data
        self.batch_size = batch_size
        self.profile_inputs = input_profile_names
        self.actuator_inputs = actuator_names
        self.targets = target_profile_names
        self.scalar_inputs = scalar_input_names
        self.lookbacks = lookbacks
        self.lookahead = lookahead
        self.predict_deltas = predict_deltas
        self.profile_downsample = profile_downsample
        self.cur_shotnum = np.zeros(self.batch_size)
        self.cur_times = np.zeros(self.batch_size)
        max_lookback = 0
        for val in lookbacks.values():
            if val > max_lookback:
                max_lookback = val
        self.max_lookback = max_lookback
        self.kwargs = kwargs

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
                                                 0:self.lookbacks[sig],
                                                 ::self.profile_downsample]
        for sig in self.actuator_inputs:
            inp['input_past_' + sig] = self.data[sig][idx * self.batch_size:
                                                      (idx+1)*self.batch_size,
                                                      0:self.lookbacks[sig]]
            inp['input_future_' + sig] = self.data[sig][idx * self.batch_size:
                                                        (idx+1)*self.batch_size,
                                                        self.lookbacks[sig]:
                                                        self.lookbacks[sig]+self.lookahead]
        for sig in self.scalar_inputs:
            inp['input_' + sig] = self.data[sig][idx * self.batch_size:
                                                 (idx+1)*self.batch_size,
                                                 0:self.lookbacks[sig]]

        for sig in self.targets:
            if self.predict_deltas:
                baseline = self.data[sig][idx * self.batch_size:(idx+1)*self.batch_size,
                                          self.lookbacks[sig]-1, ::self.profile_downsample]
            else:
                baseline = 0
            targ['target_' + sig] = self.data[sig][idx * self.batch_size:
                                                   (idx+1)*self.batch_size,
                                                   -1, ::self.profile_downsample] - baseline
            if self.kwargs.get('predict_mean'):
                targ['target_' + sig] = np.mean(targ['target_' + sig], axis=-1)

        return inp, targ

    def get_data_by_shot_time(self, shots, times):
        """Gets input/target pairs for specific times within specified shots

        If no data is present for a given shot, that shot will be ignored. 
        Attempts to find the input data that is closest to the requested time value,
        but the actual time should be verified manually.

        Args:
            shots (list or array): Array of shot numbers, as integers.
            times (list or array): Array of times. Should be the same length as shots array.

        Returns:
            inputs (dict): Dictionary of input arrays, with all data stored as 1 batch.
            targets (dict): Dictionary of target values.
        """

        if type(shots) is not np.ndarray:
            shots = np.array(shots)
        if type(times) is not np.ndarray:
            times = np.array(times)
        inds = np.array([])
        for shot, time in zip(shots, times):
            shot_inds = np.nonzero(self.data['shotnum'][:, 0] == shot)[0]
            if len(shot_inds) < 1:
                continue
            ind = shot_inds[np.argmin(
                np.abs(time-self.data['time'][shot_inds, self.max_lookback-1]))]
            inds = np.append(inds, ind)
        inds = inds.astype(int)

        inp = {}
        targ = {}
        self.cur_shotnum = self.data['shotnum'][inds]
        self.cur_times = self.data['time'][inds]

        for sig in self.profile_inputs:
            inp['input_' + sig] = self.data[sig][inds,
                                                 0:self.lookbacks[sig],
                                                 ::self.profile_downsample]
        for sig in self.actuator_inputs:
            inp['input_past_' + sig] = self.data[sig][inds,
                                                      0:self.lookbacks[sig]]
            inp['input_future_' + sig] = self.data[sig][inds,
                                                        self.lookbacks[sig]:
                                                        self.lookbacks[sig]+self.lookahead]
        for sig in self.scalar_inputs:
            inp['input_' + sig] = self.data[sig][inds,
                                                 0:self.lookbacks[sig]]
        for sig in self.targets:
            if self.predict_deltas:
                baseline = self.data[sig][inds,
                                          self.lookbacks[sig]-1, ::self.profile_downsample]
            else:
                baseline = 0
            targ['target_' + sig] = self.data[sig][inds,
                                                   -1, ::self.profile_downsample] - baseline

            if self.kwargs.get('predict_mean'):
                targ['target_' + sig] = np.mean(targ['target_' + sig], axis=-1)
        return inp, targ


def process_data(rawdata, sig_names, normalization_method, window_length=1,
                 window_overlap=0, lookbacks={}, lookahead=3, sample_step=5,
                 uniform_normalization=True, train_frac=0.7, val_frac=0.2,
                 nshots=None,
                 verbose=1, flattop_only=True, **kwargs):
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
        lookbacks (dict of int): How many window lengths for lookback for each sig.
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
        flattop_only (bool): Whether to only include data from flattop.

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
            print(abs_path)
            raise IOError("No such path to data file")
    sigsplustime = sig_names + ['time', 'shotnum']
    if verbose:
        print('Signals: ' + ', '.join(sig_names))
    usabledata = []
    # find which shots have all the signals needed
    max_lookback = 0
    for val in lookbacks.values():
        if val > max_lookback:
            max_lookback = val

    all_shots = sorted(list(rawdata.keys()))

    for shot in all_shots:
        rawdata[shot]['shotnum'] = np.ones(rawdata[shot]['time'].shape[0])*shot
        if set(sigsplustime).issubset(set(rawdata[shot].keys())) \
           and rawdata[shot]['time'].size > (max_lookback+lookahead):
            usabledata.append(rawdata[shot])
    usabledata = np.array(usabledata)
    del rawdata
    gc.collect()
    nshots = np.minimum(nshots, len(usabledata))
    if verbose:
        print('Number of useable shots: ', str(len(usabledata)))
        print('Number of shots used: ', str(nshots))

    usabledata = usabledata[:nshots]
    if verbose:
        t = 0
        for shot in usabledata:
            t += shot['time'].size
        print('Total number of timesteps: ', str(t))

    def binavg(array, start):
        """averages over bins"""
        return np.mean(array[start:start+window_length], axis=0)

    # check if each sig is not completely nan
    def is_valid(shot):
        for sig in sigsplustime:
            if np.isnan(shot[sig]).all():  # or np.isinf(shot[sig]).any():
                return False
        if (flattop_only):
            if (shot['t_ip_flat'] == None or shot['ip_flat_duration'] == None):
                return False
        return True

    def get_non_nan_inds(arr):
        if len(arr.shape) == 1:
            return np.where(~np.isnan(arr))[0]
        else:
            return np.where(np.any(~np.isnan(arr), axis=1))[0]

    def get_first_index(shot):
        input_max = max([get_non_nan_inds(shot[sig])[0] +
                         lookbacks[sig] for sig in sig_names])
        output_max = max([get_non_nan_inds(shot[sig])[0] -
                          lookahead for sig in sig_names])
        if (flattop_only) and (shot['t_ip_flat'] != None):
            current_max = np.searchsorted(
                shot['time'], shot['t_ip_flat'], side='left')
            return max(input_max, output_max, current_max)
        else:
            return max(input_max, output_max)

    def get_last_index(shot):
        partial_min = min([get_non_nan_inds(shot[sig])[-1]
                           for sig in sig_names])
        full_min = min([get_non_nan_inds(shot[sig])[-1] -
                        lookahead for sig in sig_names])
        if (flattop_only) and (shot['t_ip_flat'] != None) and (shot['ip_flat_duration'] != None):
            current_min = np.searchsorted(
                shot['time'], shot['t_ip_flat']+shot['ip_flat_duration'], side='right')
            return min(full_min, partial_min, current_min)
        else:
            return min(full_min, partial_min)

    alldata = {}
    shots_with_complete_nan = []
    for sig in sigsplustime:
        alldata[sig] = []  # initalize empty lists
    for shot in tqdm(usabledata, desc='Gathering', ascii=True, dynamic_ncols=True,
                     disable=not verbose):
        # check to see if each sig in the shot is not completely nan

        ######################################
        if not is_valid(shot):
            shots_with_complete_nan.append(np.unique(shot["shotnum"]))
            continue

        first = int(np.ceil(get_first_index(shot) /
                            (window_length-window_overlap)))
        last = int(np.floor(((get_last_index(shot)+1) /
                             (window_length-window_overlap)) - 1))
        for sig in sigsplustime:
            temp = shot[sig]
            if np.any(np.isinf(temp)):
                temp[np.isinf(temp)] = np.nan
            nbins = int(np.floor(temp.shape[0]/(window_length-window_overlap)))
            shotdata = []
            for i in range(nbins):
                # populate array of binned/windowed data for each shot
                shotdata.append(binavg(temp, i*(window_length-window_overlap)))
            shotdata = np.stack(shotdata)

            for i in range(first, last, sample_step):
                # group into arrays of input/output pairs
                if sig not in ['time', 'shotnum']:
                    alldata[sig].append(shotdata[i-lookbacks[sig]:i+lookahead])
                else:
                    alldata[sig].append(shotdata[i-max_lookback:i+lookahead])
######################################

    print("Shots with Complete NaN: " + ', '.join(str(e)
                                                  for e in shots_with_complete_nan))
    del usabledata
    gc.collect()
    for sig in tqdm(sigsplustime, desc='Stacking', ascii=True, dynamic_ncols=True,
                    disable=not verbose):
        alldata[sig] = np.stack(alldata[sig])
    alldata, normalization_params = normalize(
        alldata, normalization_method, uniform_normalization, verbose)
    nsamples = alldata['time'].shape[0]

    inds = np.arange(nsamples)

    traininds = inds[:int(nsamples*train_frac)]
    valinds = inds[int(nsamples*train_frac):int(nsamples*(val_frac+train_frac))]
    traindata = {}
    valdata = {}
    for sig in tqdm(sigsplustime, desc='Splitting', ascii=True, dynamic_ncols=True,
                    disable=not verbose):
        traindata[sig] = alldata[sig][traininds]
        valdata[sig] = alldata[sig][valinds]
    if verbose:
        print('Total number of samples: ', str(nsamples))
        print('Number of training samples: ', str(traininds.size))
        print('Number of validation samples: ', str(valinds.size))
    return traindata, valdata, normalization_params
