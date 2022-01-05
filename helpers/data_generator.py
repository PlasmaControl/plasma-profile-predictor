import pickle
import os
import gc
import sys
import time
import copy
import numba
from pathlib import Path
from tqdm import tqdm
import numpy as np
from tensorflow.keras.utils import Sequence
from helpers.normalization import normalize
from helpers.pruning_functions import (
    remove_dudtrip,
    remove_I_coil,
    remove_ECH,
    remove_gas,
    remove_nan,
    remove_non_gas_feedback,
    remove_non_beta_feedback,
    remove_outliers,
)
from helpers import exclude_shots


class DataGenerator(Sequence):
    def __init__(
        self,
        data,
        batch_size,
        input_profile_names,
        actuator_names,
        target_profile_names,
        scalar_input_names,
        lookbacks,
        lookahead,
        predict_deltas,
        profile_downsample,
        shuffle,
        **kwargs
    ):
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
            shuffle (bool): Whether to reorder training samples on epoch end.
            sample_weight (str): how to weight training samples. One of either None or 'std' to weight by standard deviation
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
        self.shuffle = shuffle
        self.kwargs = kwargs
        self.nsamples = self.data["time"].shape[0]
        self.times_called = 0
        if self.shuffle:
            self.inds = np.random.permutation(range(len(self)))
        else:
            self.inds = np.arange(len(self))

    def __len__(self):
        return int(np.ceil(len(self.data["time"]) / float(self.batch_size)))

    def __getitem__(self, idx):
        self.times_called += 1
        idx = self.inds[idx]
        inp = {}
        targ = {}
        uncertainties = {
            sig: {} for sig in set(self.profile_inputs).union(self.targets)
        }

        self.cur_shotnum = self.data["shotnum"][
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        self.cur_times = self.data["time"][
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        sample_weights = np.ones(len(self.cur_shotnum))
        for sig in self.profile_inputs:
            inp["input_" + sig] = self.data[sig][
                idx * self.batch_size : (idx + 1) * self.batch_size,
                0 : self.lookbacks[sig] + 1,
                :: self.profile_downsample,
            ]
            if self.kwargs.get("return_uncertainties"):
                try:
                    uncertainties[sig]["input"] = self.data["error_" + sig][
                        idx * self.batch_size : (idx + 1) * self.batch_size,
                        0 : self.lookbacks[sig] + 1,
                        :: self.profile_downsample,
                    ]
                except:
                    continue

        for sig in self.actuator_inputs:
            inp["input_past_" + sig] = self.data[sig][
                idx * self.batch_size : (idx + 1) * self.batch_size,
                0 : self.lookbacks[sig] + 1,
            ]
            inp["input_future_" + sig] = self.data[sig][
                idx * self.batch_size : (idx + 1) * self.batch_size,
                self.lookbacks[sig] + 1 : self.lookbacks[sig] + 1 + self.lookahead,
            ]
            if self.kwargs.get("sample_weights") == "std":
                sample_weights += np.std(
                    self.data[sig][
                        idx * self.batch_size : (idx + 1) * self.batch_size, :
                    ],
                    axis=1,
                )
        for sig in self.scalar_inputs:
            inp["input_" + sig] = self.data[sig][
                idx * self.batch_size : (idx + 1) * self.batch_size,
                0 : self.lookbacks[sig] + 1,
            ]

        for sig in self.targets:
            if self.predict_deltas:
                baseline = self.data[sig][
                    idx * self.batch_size : (idx + 1) * self.batch_size,
                    self.lookbacks[sig],
                    :: self.profile_downsample,
                ]
            else:
                baseline = 0
            targ["target_" + sig] = (
                self.data[sig][
                    idx * self.batch_size : (idx + 1) * self.batch_size,
                    -1,
                    :: self.profile_downsample,
                ]
                - baseline
            )
            if self.kwargs.get("predict_mean"):
                targ["target_" + sig] = np.mean(targ["target_" + sig], axis=-1)
            if self.kwargs.get("return_uncertainties"):
                try:
                    uncertainties[sig]["target"] = self.data["error_" + sig][
                        idx * self.batch_size : (idx + 1) * self.batch_size,
                        -1,
                        :: self.profile_downsample,
                    ]
                except:
                    continue

        if self.times_called % len(self) == 0 and self.shuffle:
            self.inds = np.random.permutation(range(len(self)))
        sample_weights_dict = {"target_" + sig: sample_weights for sig in self.targets}
        if self.kwargs.get("return_uncertainties"):
            return inp, targ, sample_weights_dict, uncertainties
        else:
            return inp, targ, sample_weights_dict

    def get_data_by_shot_time(self, shots, times=None):
        """Gets input/target pairs for specific times within specified shots

        If no data is present for a given shot, that shot will be ignored.
        Attempts to find the input data that is closest to the requested time value,
        but the actual time should be verified manually.

        Args:
            shots (list or array): Array of shot numbers, as integers.
            times (list or array): Array of times (in ms). Should be the same length as shots array.

        Returns:
            inputs (dict): Dictionary of input arrays, with all data stored as 1 batch.
            targets (dict): Dictionary of target values.
            actual (dict): Dictionary with keys 'shots','times' containing the actual shots and time values returned
        """

        if not isinstance(shots, np.ndarray) and shots is not None:
            shots = np.array(shots)
        if not isinstance(times, np.ndarray) and times is not None:
            times = np.array(times)
        idx_arr = np.empty((self.nsamples, 3))
        idx_arr[:, 0] = self.data["shotnum"][:, self.max_lookback]
        idx_arr[:, 1] = self.data["time"][:, self.max_lookback]
        idx_arr[:, 2] = np.arange(self.nsamples)
        if times is None:
            inds = idx_arr[
                np.where(
                    np.any(np.array([idx_arr[:, 0] == foo for foo in shots]), axis=0)
                )
            ][:, 2].astype(int)
        else:
            inds = []
            for shot, time in zip(shots, times):
                shot_inds = idx_arr[np.where(idx_arr[:, 0] == shot)]
                ind = shot_inds[np.argmin(np.abs(time - shot_inds), axis=0)[1]][2]
                inds.append(ind)
            inds = np.array(inds).astype(int)

        inp = {}
        targ = {}
        uncertainties = {
            sig: {} for sig in set(self.profile_inputs).union(self.targets)
        }
        self.cur_shotnum = self.data["shotnum"][inds]
        self.cur_times = self.data["time"][inds]

        for sig in self.profile_inputs:
            inp["input_" + sig] = self.data[sig][
                inds, 0 : self.lookbacks[sig] + 1, :: self.profile_downsample
            ]
            if self.kwargs.get("return_uncertainties"):
                try:
                    uncertainties[sig]["input"] = self.data["error_" + sig][
                        inds, 0 : self.lookbacks[sig] + 1, :: self.profile_downsample
                    ]
                except:
                    continue

        for sig in self.actuator_inputs:
            inp["input_past_" + sig] = self.data[sig][inds, 0 : self.lookbacks[sig] + 1]
            inp["input_future_" + sig] = self.data[sig][
                inds, self.lookbacks[sig] + 1 : self.lookbacks[sig] + 1 + self.lookahead
            ]
        for sig in self.scalar_inputs:
            inp["input_" + sig] = self.data[sig][inds, 0 : self.lookbacks[sig] + 1]
        for sig in self.targets:
            if self.predict_deltas:
                baseline = self.data[sig][
                    inds, self.lookbacks[sig], :: self.profile_downsample
                ]
            else:
                baseline = 0
            targ["target_" + sig] = (
                self.data[sig][inds, -1, :: self.profile_downsample] - baseline
            )

            if self.kwargs.get("predict_mean"):
                targ["target_" + sig] = np.mean(targ["target_" + sig], axis=-1)
            if self.kwargs.get("return_uncertainties"):
                try:
                    uncertainties[sig]["target"] = self.data["error_" + sig][
                        inds, -1, :: self.profile_downsample
                    ]
                except:
                    continue

        actual_shots_times = {
            "shots": self.data["shotnum"][inds, self.max_lookback],
            "times": self.data["time"][inds, self.max_lookback],
        }
        if self.kwargs.get("return_uncertainties"):
            return inp, targ, actual_shots_times, uncertainties
        else:
            return inp, targ, actual_shots_times


class AutoEncoderDataGenerator(Sequence):
    def __init__(
        self,
        data,
        batch_size,
        profile_names,
        actuator_names,
        scalar_names,
        lookahead,
        profile_downsample,
        state_latent_dim,
        discount_factor=1,
        x_weight=1,
        u_weight=1,
        shuffle=True,
        **kwargs
    ):
        """Make a data generator for training or validation data for autoencoder model

        Args:
            data: dict of data arrays to draw from.
            batch_size (int): size of each batch.
            profile_names (str): List of names of profile inputs, as strings.
            actuator_names (str): List of names of actuator inputs, as strings.
            scalar_names (str): List of names of scalar inputs (shape parameters etc).
            lookahead (int): How many steps ahead to predict (prediction window)
            profile_downsample (int): How much to downsample the profile data.
            state_latent_dim (int): dimension of the state latent space.
            discount_factor (0< float <=1): Geometric decay rate for importance of future predictions
            x_weight (float): weight to apply to x residual during training
            u_weight (float): weight to apply to u residual during training
            shuffle (bool): Whether to reorder training samples on epoch end.

        """
        self.data = data
        self.batch_size = batch_size
        self.profile_inputs = profile_names
        self.actuator_inputs = actuator_names
        self.scalar_inputs = scalar_names
        self.lookahead = lookahead
        self.profile_downsample = profile_downsample
        self.profile_length = int(np.ceil(65 / profile_downsample))
        self.num_actuators = len(actuator_names)
        self.num_profiles = len(profile_names)
        self.num_scalars = len(scalar_names)
        self.state_dim = self.num_profiles * self.profile_length + self.num_scalars
        self.state_latent_dim = state_latent_dim
        self.cur_shotnum = np.zeros(self.batch_size)
        self.cur_times = np.zeros(self.batch_size)
        self.discount_factor = discount_factor
        self.x_weight = x_weight
        self.u_weight = u_weight
        self.shuffle = shuffle
        self.kwargs = kwargs
        self.times_called = 0
        self.sample_weights = kwargs.get("sample_weights", True)
        self.nsamples = self.data["time"].shape[0]
        if self.shuffle:
            self.inds = np.random.permutation(range(len(self)))
        else:
            self.inds = np.arange(len(self))

    def __len__(self):
        return int(np.floor(len(self.data["time"]) / float(self.batch_size)))

    def __getitem__(self, idx):
        self.times_called += 1
        idx = self.inds[idx]
        inp = {}
        self.cur_shotnum = self.data["shotnum"][
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        self.cur_times = self.data["time"][
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        for sig in self.profile_inputs:
            inp["input_" + sig] = self.data[sig][
                idx * self.batch_size : (idx + 1) * self.batch_size,
                :,
                :: self.profile_downsample,
            ]
        sample_weights = np.ones(len(self.cur_shotnum))
        for sig in self.actuator_inputs:
            inp["input_" + sig] = self.data[sig][
                idx * self.batch_size : (idx + 1) * self.batch_size, :, np.newaxis
            ]
            if self.sample_weights == "std":
                sample_weights += np.std(
                    self.data[sig][
                        idx * self.batch_size : (idx + 1) * self.batch_size, :
                    ],
                    axis=1,
                )

        for sig in self.scalar_inputs:
            inp["input_" + sig] = self.data[sig][
                idx * self.batch_size : (idx + 1) * self.batch_size, :, np.newaxis
            ]
        targ = {
            "x_residual": np.zeros(
                (self.batch_size, self.lookahead + 1, self.state_dim)
            ),
            "u_residual": np.zeros(
                (self.batch_size, self.lookahead + 1, self.num_actuators)
            ),
            "linear_system_residual": np.zeros(
                (self.batch_size, self.lookahead, self.state_latent_dim)
            ),
        }
        time_weights = np.array(
            [self.discount_factor ** i for i in range(self.lookahead)]
        ).reshape((1, -1))
        time_ones = np.ones(self.lookahead + 1).reshape((1, -1))
        sample_weights = sample_weights.reshape((-1, 1))
        weights_dict = {
            "x_residual": self.x_weight * sample_weights * time_ones,
            "u_residual": self.u_weight * sample_weights * time_ones,
            "linear_system_residual": time_weights * sample_weights,
        }

        if self.times_called % len(self) == 0 and self.shuffle:
            self.inds = np.random.permutation(range(len(self)))

        if self.sample_weights:
            return inp, targ, weights_dict
        else:
            return inp, targ

    def get_data_by_shot_time(self, shots, times=None):
        """Gets inputs for specific times within specified shots

        If no data is present for a given shot, that shot will be ignored.
        Attempts to find the input data that is closest to the requested time value,
        but the actual time should be verified manually.

        Args:
            shots (list or array): Array of shot numbers, as integers.
            times (list or array): Array of times. Should be the same length as shots array.

        Returns:
            inputs (dict): Dictionary of input arrays, with all data stored as 1 batch.
            targets (dict): Dictionary of target values.
            actual (dict): Dictionary with keys 'shots','times' containing the actual shots and time values returned
        """

        if not isinstance(shots, np.ndarray) and shots is not None:
            shots = np.array(shots)
        if not isinstance(times, np.ndarray) and times is not None:
            times = np.array(times)
        idx_arr = np.empty((self.nsamples, 3))
        idx_arr[:, 0] = self.data["shotnum"][:, 0]
        idx_arr[:, 1] = self.data["time"][:, 0]
        idx_arr[:, 2] = np.arange(self.nsamples)
        if times is None:
            inds = idx_arr[
                np.where(
                    np.any(np.array([idx_arr[:, 0] == foo for foo in shots]), axis=0)
                )
            ][:, 2].astype(int)
        else:
            inds = []
            for shot, time in zip(shots, times):
                shot_inds = idx_arr[np.where(idx_arr[:, 0] == shot)]
                ind = shot_inds[np.argmin(np.abs(time - shot_inds), axis=0)[1]][2]
                inds.append(ind)
            inds = np.array(inds).astype(int)

        inp = {}
        targ = {}
        self.cur_shotnum = self.data["shotnum"][inds]
        self.cur_times = self.data["time"][inds]

        for sig in self.profile_inputs:
            inp["input_" + sig] = self.data[sig][inds, :, :: self.profile_downsample]
        for sig in self.actuator_inputs:
            inp["input_" + sig] = self.data[sig][inds, :]
        for sig in self.scalar_inputs:
            inp["input_" + sig] = self.data[sig][inds, :]
        targ = {
            "x_residual": np.zeros(
                (self.batch_size, self.lookahead + 1, self.state_dim)
            ),
            "u_residual": np.zeros(
                (self.batch_size, self.lookahead, self.num_actuators)
            ),
            "linear_system_residual": np.zeros(
                (self.batch_size, self.lookahead, self.state_latent_dim)
            ),
        }

        actual_shots_times = {
            "shots": self.data["shotnum"][inds, 0],
            "times": self.data["time"][inds, 0],
        }

        return inp, targ, actual_shots_times


def process_data(
    rawdata,
    sig_names,
    normalization_method,
    window_length=1,
    window_overlap=0,
    lookbacks={},
    lookahead=3,
    sample_step=5,
    uniform_normalization=True,
    train_frac=0.7,
    val_frac=0.2,
    nshots=None,
    verbose=1,
    flattop_only=True,
    randomize=True,
    **kwargs
):
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
        verbose (int): verbosity level. 0 is no CL output, 1 shows progress, 2 is abbreviated.
        flattop_only (bool): Whether to only include data from flattop.
        randomize (bool): Whether to randomize the order of data
    Keyword Args:
        pruning_functions (array-like of str or fn handle): Names of pruning functions to use.
            Options are:
                "remove_nan": (on by default): remove all time slices with NaN data
                "remove_ECH": remove time slices during and after ECH turn on
                "remove_gas": remove time slices during and after non H/D/T gas injection
                "remove_I_coil": remove time slices during and after non standard I coil operations
                "remove_dudtrip": remove time slices during and after dudtrip signal (ie disruption, pcs crash etc)
                "remove_non_gas_feedback": remove time slices where density feedback control is not used
                "remove_non_beta_feedback": remove time slices where beta feedback is not used
        excluded_shots (array): List of shot numbers to exclude, or name of standard set based on topology:
            'topology_TOP','topology_SNT','topology_SNB','topology_OUT','topology_MAR','topology_IN',
            'topology_DN','topology_BOT','test_set'
        uncertainties (bool): Return uncertainties for signals that have them available.
        invert_q (bool): Whether to use regular q or 1/q (iota)
        val_idx (int, 1-9): Which set to use for validation. 0 is for testing.
            If present, overrides val_frac & train_frac.

    Returns:
        traindata (dict): Dictionary of numpy arrays, one entry for each signal.
            Each array has shape [nsamples,lookback+lookahead,signal_shape]
        valdata (dict): Dictionary of numpy arrays, one entry for each signal.
            Each array has shape [nsamples,lookback+lookahead,signal_shape]
        param_dict (dict): Dictionary of parameters used during normalization,
            to be used for denormalizing later. Eg, mean, stddev, method, etc.
    """
    ##############################
    # Load data
    ##############################
    if type(rawdata) is not dict:
        if verbose:
            print("Loading")
        abs_path = Path(os.path.expanduser(rawdata)).resolve()
        if abs_path.exists():
            with open(abs_path, "rb") as f:
                rawdata = pickle.load(f, encoding="latin1")
        else:
            print(abs_path)
            raise IOError("No such path to data file")

    ##############################
    # get pruning functions
    ##############################
    IC_coils = [
        "C_coil_139",
        "C_coil_19",
        "C_coil_199",
        "C_coil_259",
        "C_coil_319",
        "C_coil_79",
        "I_coil_150L",
        "I_coil_150U",
        "I_coil_210L",
        "I_coil_210U",
        "I_coil_270L",
        "I_coil_270U",
        "I_coil_30L",
        "I_coil_30U",
        "I_coil_330L",
        "I_coil_330U",
        "I_coil_90L",
        "I_coil_90U",
    ]

    pruning_functions = copy.copy(kwargs.get("pruning_functions", []))
    if "ech" not in sig_names:
        pruning_functions.append("remove_ECH")
    if not {"gasB", "gasC", "gasD", "gasE"}.issubset(set(sig_names)):
        pruning_functions.append("remove_gas")
    if not set(IC_coils).issubset(set(sig_names)):
        pruning_functions.append("remove_I_coil")

    prun_dict = {
        "remove_nan": remove_nan,
        "remove_ECH": remove_ECH,
        "remove_I_coil": remove_I_coil,
        "remove_gas": remove_gas,
        "remove_dudtrip": remove_dudtrip,
        "remove_non_gas_feedback": remove_non_gas_feedback,
        "remove_non_beta_feedback": remove_non_beta_feedback,
        "remove_outliers": remove_outliers,
    }
    for i, elem in enumerate(pruning_functions):
        if isinstance(elem, str):
            pruning_functions[i] = prun_dict[elem]

    ##############################
    # get excluded shots
    ##############################
    excluded_shots = copy.copy(kwargs.get("excluded_shots", []))
    if kwargs.get("val_idx") == 0 and "test_set" in excluded_shots:
        excluded_shots.remove("test_set")
    exclude_dict = {
        "topology_TOP": exclude_shots.topology_TOP,
        "topology_SNT": exclude_shots.topology_SNT,
        "topology_SNB": exclude_shots.topology_SNB,
        "topology_OUT": exclude_shots.topology_OUT,
        "topology_MAR": exclude_shots.topology_MAR,
        "topology_IN": exclude_shots.topology_IN,
        "topology_DN": exclude_shots.topology_DN,
        "topology_BOT": exclude_shots.topology_BOT,
        "year_2010": exclude_shots.year_2010,
        "year_2011": exclude_shots.year_2011,
        "year_2012": exclude_shots.year_2012,
        "year_2013": exclude_shots.year_2013,
        "year_2014": exclude_shots.year_2014,
        "year_2015": exclude_shots.year_2015,
        "year_2016": exclude_shots.year_2016,
        "year_2017": exclude_shots.year_2017,
        "year_2018": exclude_shots.year_2018,
        "year_2019": exclude_shots.year_2019,
        "test_set": exclude_shots.test,
    }

    for i, elem in enumerate(excluded_shots):
        if isinstance(elem, str):
            excluded_shots[i] = exclude_dict[elem]
        # if not isinstance(elem, list):
        #    excluded_shots[i] = [elem]
    excluded_shots = np.unique([item for sublist in excluded_shots for item in sublist])

    ##############################
    # get sig names
    ##############################
    extra_sigs = ["time", "shotnum"]
    sigs_with_errors = ["temp", "dens", "itemp", "idens", "rotation"]
    for sig in sig_names:
        if sig in sigs_with_errors and kwargs.get("uncertainties"):
            extra_sigs += ["error_" + sig]
    if remove_non_gas_feedback in pruning_functions:
        extra_sigs += ["gas_feedback"]
    if remove_non_beta_feedback in pruning_functions:
        extra_sigs += ["beam_feedback_switch", "beam_feedback_power_target_quantity"]
    if remove_dudtrip in pruning_functions:
        extra_sigs += ["dud_trip"]
    if remove_I_coil in pruning_functions:
        extra_sigs += ["bt", "curr", "C_coil_method", "I_coil_method"]
    if remove_gas in pruning_functions:
        extra_sigs += ["gasB", "gasC", "gasD", "gasE", "pfx1", "pfx2"]
    if remove_ECH in pruning_functions:
        extra_sigs += ["ech"]
    sig_names = list(np.unique(sig_names))
    sigsplustime = list(np.unique(sig_names + extra_sigs))
    if verbose:
        print("Signals: " + ", ".join(sig_names))

    ##############################
    # figure out lookbacks
    ##############################
    if isinstance(lookbacks, int):
        max_lookback = lookbacks
        lookbacks = {sig: max_lookback for sig in sigsplustime}
    else:
        max_lookback = 0
        for val in lookbacks.values():
            if val > max_lookback:
                max_lookback = val
        for sig in sigsplustime:
            if sig not in lookbacks.keys() and "error" not in sig:
                lookbacks[sig] = max_lookback
        for sig in sigsplustime:
            if "error" in sig:
                lookbacks[sig] = lookbacks[sig[6:]]

    ##############################
    # find which shots have all the signals needed
    ##############################
    usabledata = []
    shots_without_sigs = []
    shots_too_short = []
    shots_excluded = []
    all_shots = sorted(list(rawdata.keys()))
    # remove unusable shots to reduce the number of iterations we do in vain
    for shot in all_shots:
        # the shotnum entry simply gives the shot number
        rawdata[shot]["shotnum"] = np.ones(rawdata[shot]["time"].shape[0]) * shot
        if (
            set(sigsplustime).issubset(set(rawdata[shot].keys()))
            and rawdata[shot]["time"].size
            > (max_lookback + lookahead) * (window_length - window_overlap)
            and shot not in excluded_shots
        ):
            usabledata.append(rawdata[shot])
        if (
            set(sigsplustime).issubset(set(rawdata[shot].keys()))
            and shot not in excluded_shots
        ):
            # shot is too short
            shots_too_short.append(shot)
        if (
            rawdata[shot]["time"].size
            > (max_lookback + lookahead) * (window_length - window_overlap)
            and shot not in excluded_shots
        ):
            # shot missing sigs
            shots_without_sigs.append(shot)
        if set(sigsplustime).issubset(set(rawdata[shot].keys())) and rawdata[shot][
            "time"
        ].size > (max_lookback + lookahead) * (window_length - window_overlap):
            shots_excluded.append(shot)

    usabledata = np.array(usabledata)
    if len(usabledata) == 0:
        s = "No valid shots \n"
        s += "Num Total: {}\n".format(len(all_shots))
        s += "Num without sigs: {}\n".format(len(shots_without_sigs))
        s += "Num too short: {}\n".format(len(shots_too_short))
        s += "Num excluded: {}\n".format(len(shots_excluded))
        if len(shots_without_sigs) == len(all_shots):
            missing_sigs = []
            for shot in all_shots:
                missing_sigs.append(
                    set(sigsplustime).difference(set(rawdata[shot].keys()))
                )
            missing_sigs = missing_sigs[0].intersection(*missing_sigs[1:])
            s += "Missing sigs: " + str(missing_sigs)
        raise ValueError(s)

    del rawdata
    gc.collect()
    if nshots is not None:
        nshots = np.minimum(nshots, len(usabledata))
        usabledata = usabledata[:nshots]
    else:
        nshots = len(usabledata)
    if verbose:
        print("Number of useable shots: ", str(len(usabledata)))
        print("Number of shots used: ", str(nshots))
        t = 0
        for shot in usabledata:
            t += shot["time"].size
        print("Total number of timesteps: ", str(t))
        sys.stdout.flush()

    ##############################
    # some helper functions
    ##############################
    def moving_average(a, n):
        """moving average of array a with window size n"""
        ret = np.nancumsum(a, axis=0)
        ret[n:] = ret[n:] - ret[:-n]
        return (ret[n - 1 :] / n).astype("float32")

    def is_valid(shot):
        """checks if a shot is completely NaN or if it never reached flattop"""
        for sig in sigsplustime:
            if np.isnan(shot[sig]).all():  # or np.isinf(shot[sig]).any():
                return False
        if flattop_only:
            if shot["t_ip_flat"] == None or shot["ip_flat_duration"] == None:
                return False
        return True

    def get_non_nan_inds(arr):
        """gets indices of array where value is not NaN"""
        if len(arr.shape) == 1:
            return np.where(~np.isnan(arr))[0]
        else:
            return np.where(np.any(~np.isnan(arr), axis=1))[0]

    def get_first_index(shot):
        """gets index of first valid timeslice for a shot"""
        input_max = max(
            [get_non_nan_inds(shot[sig])[0] + lookbacks[sig] for sig in sig_names]
        )
        output_max = max(
            [get_non_nan_inds(shot[sig])[0] - lookahead for sig in sig_names]
        )
        if (flattop_only) and (shot["t_ip_flat"] != None):
            current_max = np.searchsorted(shot["time"], shot["t_ip_flat"], side="left")
            return np.ceil(max(input_max, output_max, current_max)).astype(int)
        else:
            return np.ceil(max(input_max, output_max)).astype(int)

    def get_last_index(shot):
        """gets index of last valid timeslice for a shot"""
        partial_min = min([get_non_nan_inds(shot[sig])[-1] for sig in sig_names])
        full_min = min(
            [get_non_nan_inds(shot[sig])[-1] - lookahead for sig in sig_names]
        )
        if (
            (flattop_only)
            and (shot["t_ip_flat"] != None)
            and (shot["ip_flat_duration"] != None)
        ):
            current_min = np.searchsorted(
                shot["time"], shot["t_ip_flat"] + shot["ip_flat_duration"], side="right"
            )
            return np.floor(min(full_min, partial_min, current_min)).astype(int)
        else:
            return np.floor(min(full_min, partial_min)).astype(int)

    @numba.njit
    def group_data(array, first, last, sample_step, lookback, lookahead):
        """groups shot data into i/o chunks"""
        data = []
        for i in range(first, last, sample_step):
            data.append(array[i - lookback : i + lookahead + 1])
        return data

    ##############################
    # loop through shots and do stuff
    ##############################
    alldata = {}
    shots_with_complete_nan = []
    for sig in sigsplustime:
        alldata[sig] = []  # initalize empty lists
    for shot in tqdm(
        usabledata,
        desc="Gathering",
        ascii=True,
        dynamic_ncols=True,
        disable=not verbose == 1,
    ):
        ##############################
        # take moving average of data and bin it
        ##############################
        binned_shot = {}
        for sig in sigsplustime:
            if np.any(np.isinf(shot[sig])):
                shot[sig][np.isinf(shot[sig])] = np.nan
            binned_shot[sig] = moving_average(shot[sig], window_length)[
                :: window_length - window_overlap
            ]
        binned_shot["t_ip_flat"] = shot["t_ip_flat"].astype("float32")
        binned_shot["ip_flat_duration"] = shot["ip_flat_duration"].astype("float32")
        if not is_valid(binned_shot):
            shots_with_complete_nan.append(np.unique(shot["shotnum"]))
            continue

        ##############################
        # group into arrays of input/output pairs
        ##############################
        first = get_first_index(binned_shot)
        last = get_last_index(binned_shot)
        for sig in sigsplustime:
            alldata[sig] += group_data(
                binned_shot[sig], first, last, sample_step, lookbacks[sig], lookahead
            )

    if verbose:
        print(
            "Shots with Complete NaN: "
            + ", ".join(str(e) for e in shots_with_complete_nan)
        )
    sys.stdout.flush()
    del usabledata
    gc.collect()

    ##############################
    # stack data from all shots together
    ##############################
    for sig in tqdm(
        sigsplustime,
        desc="Stacking",
        ascii=True,
        dynamic_ncols=True,
        disable=not verbose == 1,
    ):
        alldata[sig] = np.stack(alldata[sig]).astype("float32")
        gc.collect()

    if verbose:
        print("{} samples total".format(len(alldata["time"])))
    sys.stdout.flush()
    gc.collect()

    ##############################
    # apply pruning functions
    ##############################
    # call fns in the right order to speed things up
    if remove_ECH in pruning_functions:
        alldata = remove_ECH(alldata, verbose)
    if remove_non_gas_feedback in pruning_functions:
        alldata = remove_non_gas_feedback(alldata, verbose)
    if remove_non_beta_feedback in pruning_functions:
        alldata = remove_non_beta_feedback(alldata, verbose)
    if remove_gas in pruning_functions:
        alldata = remove_gas(alldata, verbose)
    if remove_I_coil in pruning_functions:
        alldata = remove_I_coil(alldata, verbose)
    if remove_nan in pruning_functions:
        alldata = remove_nan(alldata, verbose)
    if remove_dudtrip in pruning_functions:
        alldata = remove_dudtrip(alldata, verbose)
    if remove_outliers in pruning_functions:
        alldata = remove_outliers(alldata, verbose)

    if verbose:
        print("{} samples remaining after pruning".format(len(alldata["time"])))
    sys.stdout.flush()
    gc.collect()

    if kwargs.get("invert_q"):
        qs = ["q", "q_EFIT01", "q_EFIT02", "q_EFITRT1", "q_EFITRT2"]
        for sig in alldata.keys():
            if sig in qs:
                alldata[sig] = 1 / alldata[sig]
                alldata[sig][:, :, -1] = 0

    ##############################
    # normalize data
    ##############################
    alldata, normalization_params = normalize(
        alldata, normalization_method, uniform_normalization, verbose
    )
    gc.collect()

    ##############################
    # split into train and validation sets
    ##############################
    nsamples = alldata["time"].shape[0]
    val_idx = kwargs.get("val_idx", None)
    if val_idx is not None and val_idx in np.arange(10):
        valinds = np.where(alldata["shotnum"][:, 0] % 10 == val_idx)[0]
        traininds = np.where(alldata["shotnum"][:, 0] % 10 != val_idx)[0]
        if randomize:
            valinds = np.random.permutation(valinds)
            traininds = np.random.permutation(traininds)
    else:
        inds = np.random.permutation(nsamples) if randomize else np.arange(nsamples)
        traininds = inds[: int(nsamples * train_frac)]
        valinds = inds[
            int(nsamples * train_frac) : int(nsamples * (val_frac + train_frac))
        ]
    traindata = {}
    valdata = {}
    for sig in tqdm(
        sigsplustime,
        desc="Splitting",
        ascii=True,
        dynamic_ncols=True,
        disable=not verbose == 1,
    ):
        traindata[sig] = alldata[sig][traininds]
        valdata[sig] = alldata[sig][valinds]
    time.sleep(0.1)
    if verbose:
        print("Total number of samples: ", str(nsamples))
        print("Number of training samples: ", str(traininds.size))
        print("Number of validation samples: ", str(valinds.size))
    return traindata, valdata, normalization_params
