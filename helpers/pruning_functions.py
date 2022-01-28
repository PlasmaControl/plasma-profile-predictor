import numpy as np
import numba


def prune_loop(inds, shotnumarr, timearr):
    """Find indices to remove.

    Given a list of indices where an event occurs and corresponding shot
    numbers and times, find the indices during and after the event occurs
    during corresponding shots.

    Args:
        inds (array): Array of indices where an event occurs.
        shotnumarr (array): Array of shot numbers for each training sample
        timearr (array): Array of time indices for each training sample.

    Returns:
        remove_inds (arr): Indices during and after event occured, in each shot.
    """
    remove_inds = set()
    shot_remove = shotnumarr[inds]
    time_remove = timearr[inds]
    for shot in np.unique(shot_remove):
        min_time = np.min(time_remove[shot_remove == shot])
        remove_inds = remove_inds.union(
            set(np.where(np.logical_and(shotnumarr == shot, timearr > min_time))[0])
        )
    return remove_inds


def remove_dudtrip(data, verbose):
    """Remove samples during and after dudtrips (ie, disruption, PCS crash, other misc stuff going wrong)

    Args:
        data (dict): Dictionary of training samples.
        verbose (int or bool): Whether to print info to the screen

    Returns:
        data (dict): Dictionary of training samples with values removed.
    """
    if verbose:
        print("Removing dudtrip")
    dud_trip_inds = np.nonzero(data["dud_trip"])[0]
    if len(dud_trip_inds) == 0:
        if verbose:
            print("Removed 0 samples")
        return data
    remove_inds = prune_loop(dud_trip_inds, data["shotnum"], data["time"])
    if verbose:
        print("Removed {} samples".format(len(remove_inds)))
    keep_inds = set(range(len(data["time"]))).difference(remove_inds)
    if verbose:
        print("{} samples remaining".format(len(keep_inds)))
    for sig in data.keys():
        data[sig] = data[sig][list(keep_inds)]
    return data


def remove_non_gas_feedback(data, verbose):
    """Remove all samples where not all of the timesteps have gas feedback on

    Args:
        data (dict): Dictionary of training samples.
        verbose (int or bool): Whether to print info to the screen

    Returns:
        data (dict): Dictionary of training samples with values removed.
    """
    if verbose:
        print("Removing timesteps WITHOUT gas feedback")
    keep_sample = np.all(data["gas_feedback"] == 1, axis=1)
    if verbose:
        print("Removed {} samples".format(len(data["gas_feedback"]) - sum(keep_sample)))
    if verbose:
        print("{} samples remaining".format(sum(keep_sample)))
    for sig in data.keys():
        data[sig] = data[sig][keep_sample]
    return data


def remove_non_beta_feedback(data, verbose):
    """Remove all samples where not all of the timesteps have betan feedback on

    Args:
        data (dict): Dictionary of training samples.
        verbose (int or bool): Whether to print info to the screen

    Returns:
        data (dict): Dictionary of training samples with values removed.
    """
    if verbose:
        print("Removing timesteps WITHOUT betan feedback")
    # beam_feedback_switch of 2 means there is feedback on power
    # beam_feedback_power_target_quantity of 9 means the power quantity is betan
    keep_sample = np.all(
        np.logical_and(
            data["beam_feedback_switch"] == 2,
            data["beam_feedback_power_target_quantity"] == 9,
        ),
        axis=1,
    )
    if verbose:
        print(
            "Removed {} samples".format(
                len(data["beam_feedback_switch"]) - sum(keep_sample)
            )
        )
    if verbose:
        print("{} samples remaining".format(sum(keep_sample)))
    for sig in data.keys():
        data[sig] = data[sig][keep_sample]
    return data


def remove_I_coil(data, verbose):
    """Remove all samples during and after non-standard coil operation

    Args:
        data (dict): Dictionary of training samples.
        verbose (int or bool): Whether to print info to the screen

    Returns:
        data (dict): Dictionary of training samples with values removed.
    """
    if verbose:
        print("Removing weird I-coils")

    @numba.njit
    def find_Icoil_inds(n, bt, curr, C_coil_method, I_coil_method):
        c_coil = list()
        i_coil = list()
        EFC = list()
        for i in range(n):
            if np.mean(bt[i] * curr[i]) < 0:
                # left-handed
                if not set(np.unique(C_coil_method[i])).issubset({5, 0, -1}):
                    c_coil.append(i)
                if not set(np.unique(I_coil_method[i])).issubset({5, 0, -1}):
                    i_coil.append(i)
                if not np.all(
                    np.logical_xor(C_coil_method[i] == 5, I_coil_method[i] == 5)
                ):
                    EFC.append(i)
            else:
                # right-handed
                if not set(np.unique(C_coil_method[i])).issubset({6, 0, -1}):
                    c_coil.append(i)
                if not set(np.unique(I_coil_method[i])).issubset({7, 0, -1}):
                    i_coil.append(i)
                if not np.any(
                    np.logical_or(
                        np.logical_and(C_coil_method[i] == 6, I_coil_method[i] != 7),
                        np.logical_and(C_coil_method[i] != 6, I_coil_method[i] == 7),
                    )
                ):
                    EFC.append(i)

        coil_inds = c_coil + i_coil + EFC
        return coil_inds

    coil_inds = np.unique(
        find_Icoil_inds(
            len(data["time"]),
            data["bt"],
            data["curr"],
            data["C_coil_method"].astype(int),
            data["I_coil_method"].astype(int),
        )
    )
    if len(coil_inds) == 0:
        if verbose:
            print("Removed 0 samples")
        return data
    remove_inds = prune_loop(coil_inds, data["shotnum"], data["time"])
    if verbose:
        print("Removed {} samples".format(len(remove_inds)))
    keep_inds = set(range(len(data["time"]))).difference(remove_inds)
    if verbose:
        print("{} samples remaining".format(len(keep_inds)))
    for sig in data.keys():
        data[sig] = data[sig][list(keep_inds)]
    return data


def remove_gas(data, verbose):
    """Remove all samples during and after non-standard gas injection

    Args:
        data (dict): Dictionary of training samples.
        verbose (int or bool): Whether to print info to the screen

    Returns:
        data (dict): Dictionary of training samples with values removed.
    """

    if verbose:
        print("Removing weird gas")
    from functools import reduce

    threshold = 2
    gasB_inds = np.nonzero(np.any(data["gasB"] > threshold, axis=1))[0]
    gasC_inds = np.nonzero(np.any(data["gasC"] > threshold, axis=1))[0]
    gasD_inds = np.nonzero(np.any(data["gasD"] > threshold, axis=1))[0]
    gasE_inds = np.nonzero(np.any(data["gasE"] > threshold, axis=1))[0]
    pfx1_inds = np.nonzero(np.any(data["pfx1"] > threshold, axis=1))[0]
    pfx2_inds = np.nonzero(np.any(data["pfx2"] > threshold, axis=1))[0]
    gas_inds = reduce(
        np.union1d, (gasB_inds, gasC_inds, gasD_inds, gasE_inds, pfx1_inds, pfx2_inds)
    )
    if len(gas_inds) == 0:
        if verbose:
            print("Removed 0 samples")
        return data
    remove_inds = prune_loop(gas_inds, data["shotnum"], data["time"])
    if verbose:
        print("Removed {} samples".format(len(remove_inds)))
    keep_inds = set(range(len(data["time"]))).difference(remove_inds)
    if verbose:
        print("{} samples remaining".format(len(keep_inds)))
    for sig in data.keys():
        data[sig] = data[sig][list(keep_inds)]
    return data


def remove_ECH(data, verbose):
    """Remove all samples during and after ECH turn on

    Args:
        data (dict): Dictionary of training samples.
        verbose (int or bool): Whether to print info to the screen

    Returns:
        data (dict): Dictionary of training samples with values removed.
    """
    if verbose:
        print("Removing ECH")
    ech_inds = np.nonzero(np.any(data["ech"] > 0.5, axis=1))[0]
    if len(ech_inds) == 0:
        if verbose:
            print("Removed 0 samples")
        return data
    remove_inds = prune_loop(ech_inds, data["shotnum"], data["time"])
    if verbose:
        print("Removed {} samples".format(len(remove_inds)))
    keep_inds = set(range(len(data["time"]))).difference(remove_inds)
    if verbose:
        print("{} samples remaining".format(len(keep_inds)))
    for sig in data.keys():
        data[sig] = data[sig][list(keep_inds)]
    return data


def remove_nan(data, verbose):
    """Remove all samples that contain NaN values

    Args:
        data (dict): Dictionary of training samples.
        verbose (int or bool): Whether to print info to the screen

    Returns:
        data (dict): Dictionary of training samples with values removed.
    """
    if verbose:
        print("Removing NaN")
    remove_inds = []
    for sig in data.keys():
        if data[sig].ndim == 1:
            remove_inds += np.where(np.isnan(data[sig]))[0].tolist()
        else:
            ax = tuple(np.arange(1, data[sig].ndim).astype(int))
            remove_inds += np.where(np.any(np.isnan(data[sig]), axis=ax))[0].tolist()
    remove_inds = np.unique(remove_inds)
    if verbose:
        print("Removed {} samples".format(len(remove_inds)))
    keep_inds = set(range(len(data["time"]))).difference(remove_inds)
    if verbose:
        print("{} samples remaining".format(len(keep_inds)))
    for sig in data.keys():
        data[sig] = data[sig][list(keep_inds)]
    return data


def remove_outliers(data, verbose):
    """Remove all samples that contain outliers

    Gets rid of jagged or all zero q profiles, negative values of pressure, etc

    Args:
        data (dict): Dictionary of training samples.
        verbose (int or bool): Whether to print info to the screen

    Returns:
        data (dict): Dictionary of training samples with values removed.
    """
    # remove samples where any q==0
    if verbose:
        print("Removing zero q profiles")
    qs = ["q", "q_EFIT01", "q_EFIT02", "q_EFITRT1", "q_EFITRT2"]
    mask = np.ones(len(data["time"]), dtype=bool)
    for sig in data.keys():
        if sig in qs:
            temp_mask = (data[sig] != 0).all(axis=-1).all(axis=-1)
            mask = mask & temp_mask
    for sig in data.keys():
        data[sig] = data[sig][mask]
    if verbose:
        print("Removed {} samples".format(len(np.nonzero(mask == 0)[0])))
        print("{} samples remaining".format(len(np.nonzero(mask)[0])))

    if "q_EFIT02" in data:
        # get rid of jagged profiles
        if verbose:
            print("Removing jagged q profiles")
        remove_inds = np.where(np.std(data["q_EFIT02"][:, :, :50], axis=-1) > 5)[0]
        remove_inds = np.unique(remove_inds)
        if verbose:
            print("Removed {} samples".format(len(remove_inds)))
        keep_inds = set(range(len(data["time"]))).difference(remove_inds)
        if verbose:
            print("{} samples remaining".format(len(keep_inds)))
        for sig in data.keys():
            data[sig] = data[sig][list(keep_inds)]
        # get rid of all zero profiles
        remove_inds = np.where(np.sum(np.abs(data["q_EFIT02"]), axis=-1) < 0.1)[0]
        remove_inds = np.unique(remove_inds)
        if verbose:
            print("Removed {} samples".format(len(remove_inds)))
        keep_inds = set(range(len(data["time"]))).difference(remove_inds)
        if verbose:
            print("{} samples remaining".format(len(keep_inds)))
        for sig in data.keys():
            data[sig] = data[sig][list(keep_inds)]

    if "press_EFIT02" in data:
        # get rid of extreme outliers
        if verbose:
            print("Removing outlier pressure profiles")
        remove_inds = np.where(np.mean(np.abs(data["press_EFIT02"]), axis=-1) > 1e5)[0]
        remove_inds = np.unique(remove_inds)
        if verbose:
            print("Removed {} samples".format(len(remove_inds)))
        keep_inds = set(range(len(data["time"]))).difference(remove_inds)
        if verbose:
            print("{} samples remaining".format(len(keep_inds)))
        for sig in data.keys():
            data[sig] = data[sig][list(keep_inds)]
        # remove negative pressure profiles
        if verbose:
            print("Removing negative pressure profiles")
        remove_inds = np.where(data["press_EFIT02"] < 0)[0]
        remove_inds = np.unique(remove_inds)
        if verbose:
            print("Removed {} samples".format(len(remove_inds)))
        keep_inds = set(range(len(data["time"]))).difference(remove_inds)
        if verbose:
            print("{} samples remaining".format(len(keep_inds)))
        for sig in data.keys():
            data[sig] = data[sig][list(keep_inds)]

    if "press_EFIT01" in data:
        # remove negative pressure profiles
        if verbose:
            print("Removing negative pressure profiles")
        remove_inds = np.where(data["press_EFIT01"] < 0)[0]
        remove_inds = np.unique(remove_inds)
        if verbose:
            print("Removed {} samples".format(len(remove_inds)))
        keep_inds = set(range(len(data["time"]))).difference(remove_inds)
        if verbose:
            print("{} samples remaining".format(len(keep_inds)))
        for sig in data.keys():
            data[sig] = data[sig][list(keep_inds)]

    return data
