import numpy as np
import numba


#removes indices PLUS all later indices in the shot
#e.g. if we turn ECH or weird I coils on in the beginning of the 
#shot the plasma is fundamentally different so we don't want
#to train on those timesteps
@numba.njit
def prune_loop(inds,shotnumarr,timearr):
    remove_inds = set()
    for ind in inds:
        shot = shotnumarr[ind]
        time = timearr[ind]
        i = ind
        while np.any(shotnumarr[i] == shot) and np.any(timearr[i] >= time):
            remove_inds.add(i)
            i += 1
            if i>=len(timearr):
                break
            
    return remove_inds


def remove_dudtrip(data, verbose):
    if verbose:
        print('Removing dudtrip')
    dud_trip_inds = np.nonzero(data['dud_trip'])[0]
    if len(dud_trip_inds)==0:
        return data
    remove_inds = prune_loop(dud_trip_inds,data['shotnum'],data['time'])
    if verbose:
        print("Removed {} samples".format(len(remove_inds)))
    keep_inds = set(range(len(data['time']))).difference(remove_inds)
    if verbose:
        print("{} samples remaining".format(len(keep_inds)))
    for sig in data.keys():
        data[sig] = data[sig][list(keep_inds)]
    return data

# remove all samples where not all of the timesteps have gas feedback on
def remove_non_gas_feedback(data, verbose):
    if verbose:
        print('Removing timesteps WITHOUT gas feedback')
    keep_sample = np.all(data['gas_feedback']==1,axis=1)
    if verbose:
        print("Removed {} samples".format(len(data['gas_feedback'])-sum(keep_sample)))
    if verbose:
        print("{} samples remaining".format(sum(keep_sample)))
    for sig in data.keys():
        data[sig] = data[sig][keep_sample]
    return data

# remove all samples where not all of the timesteps have betan feedback on
def remove_non_beta_feedback(data, verbose):
    if verbose:
        print('Removing timesteps WITHOUT betan feedback')
    # beam_feedback_switch of 2 means there is feedback on power
    # beam_feedback_power_target_quantity of 9 means the power quantity is betan
    keep_sample = np.all(np.logical_and(data['beam_feedback_switch']==2,data['beam_feedback_power_target_quantity']==9),axis=1)
    if verbose:
        print("Removed {} samples".format(len(data['beam_feedback_switch'])-sum(keep_sample)))
    if verbose:
        print("{} samples remaining".format(sum(keep_sample)))
    for sig in data.keys():
        data[sig] = data[sig][keep_sample]
    return data

def remove_I_coil(data, verbose):
    if verbose:
        print('Removing weird I-coils')
    
    @numba.njit
    def find_Icoil_inds(n,bt,curr,C_coil_method,I_coil_method):
        c_coil = list()
        i_coil = list()
        EFC = list()
        for i in range(n):
            if np.mean(bt[i]*curr[i]) < 0:
                # left-handed
                if not set(np.unique(C_coil_method[i])).issubset({5, 0, -1}):
                    c_coil.append(i)
                if not set(np.unique(I_coil_method[i])).issubset({5, 0, -1}):
                    i_coil.append(i)
                if not np.all(np.logical_xor(C_coil_method[i] == 5, I_coil_method[i] == 5)):
                    EFC.append(i)
            else:
                # right-handed
                if not set(np.unique(C_coil_method[i])).issubset({6, 0, -1}):
                    c_coil.append(i)
                if not set(np.unique(I_coil_method[i])).issubset({7, 0, -1}):
                    i_coil.append(i)
                if not np.any(np.logical_or(np.logical_and(C_coil_method[i] == 6, 
                                                           I_coil_method[i] != 7), 
                                            np.logical_and(C_coil_method[i] != 6, 
                                                           I_coil_method[i] == 7))):
                    EFC.append(i)
                    
        coil_inds = c_coil + i_coil + EFC
        return coil_inds

    coil_inds = np.unique(find_Icoil_inds(len(data['time']),
                                          data['bt'],
                                          data['curr'],
                                          data['C_coil_method'].astype(int),
                                          data['I_coil_method'].astype(int)))
    if len(coil_inds)==0:
        return data
    remove_inds = prune_loop(coil_inds,data['shotnum'],data['time'])
    if verbose:
        print("Removed {} samples".format(len(remove_inds)))
    keep_inds = set(range(len(data['time']))).difference(remove_inds)
    if verbose:
        print("{} samples remaining".format(len(keep_inds)))
    for sig in data.keys():
        data[sig] = data[sig][list(keep_inds)]
    return data


def remove_gas(data, verbose):
    if verbose:
        print('Removing weird gas')
    from functools import reduce
    threshold=2
    gasB_inds = np.nonzero(np.any(data['gasB'] > threshold, axis=1))[0]
    gasC_inds = np.nonzero(np.any(data['gasC'] > threshold, axis=1))[0]
    gasD_inds = np.nonzero(np.any(data['gasD'] > threshold, axis=1))[0]
    gasE_inds = np.nonzero(np.any(data['gasE'] > threshold, axis=1))[0]
    pfx1_inds = np.nonzero(np.any(data['pfx1'] > threshold, axis=1))[0]
    pfx2_inds = np.nonzero(np.any(data['pfx2'] > threshold, axis=1))[0]
    gas_inds = reduce(np.union1d, (gasB_inds, gasC_inds,
                                   gasD_inds, gasE_inds, pfx1_inds, pfx2_inds))
    if len(gas_inds)==0:
        return data
    remove_inds = prune_loop(gas_inds,data['shotnum'],data['time'])
    if verbose:
        print("Removed {} samples".format(len(remove_inds)))
    keep_inds = set(range(len(data['time']))).difference(remove_inds)
    if verbose:
        print("{} samples remaining".format(len(keep_inds)))
    for sig in data.keys():
        data[sig] = data[sig][list(keep_inds)]
    return data


def remove_ECH(data, verbose):
    if verbose:
        print('Removing ECH')
    ech_inds = np.nonzero(np.any(data['ech'] > .5, axis=1))[0]
    if len(ech_inds)==0:
        return data
    remove_inds = prune_loop(ech_inds,data['shotnum'],data['time'])
    if verbose:
        print("Removed {} samples".format(len(remove_inds)))
    keep_inds = set(range(len(data['time']))).difference(remove_inds)
    if verbose:
        print("{} samples remaining".format(len(keep_inds)))
    for sig in data.keys():
        data[sig] = data[sig][list(keep_inds)]
    return data


def remove_nan(data, verbose):
    if verbose:
        print('Removing NaN')
    remove_inds = []
    for sig in data.keys():
        if data[sig].ndim==1:
            remove_inds += np.where(np.isnan(data[sig]))[0].tolist()
        else:
            ax = tuple(np.arange(1,data[sig].ndim).astype(int))
            remove_inds += np.where(np.any(np.isnan(data[sig]),axis=ax))[0].tolist()
    remove_inds = np.unique(remove_inds)
    if verbose:
        print("Removed {} samples".format(len(remove_inds)))
    keep_inds = set(range(len(data['time']))).difference(remove_inds)
    if verbose:
        print("{} samples remaining".format(len(keep_inds)))
    for sig in data.keys():
        data[sig] = data[sig][list(keep_inds)]
    return data
