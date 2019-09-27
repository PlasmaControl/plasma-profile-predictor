import numpy as np
from tqdm import tqdm

def remove_dudtrip(data, verbose):
    if verbose==2:
        print('Removing dudtrip')
    dud_trip_inds = np.nonzero(data['dud_trip'])[0]
    remove_inds = set()
    for ind in tqdm(dud_trip_inds, desc='Removing dudtrip', ascii=True, dynamic_ncols=True, disable=not verbose==1):
        shot = data['shotnum'][ind]
        time = data['time'][ind]
        i = ind
        while any(data['shotnum'][i] == shot) and any(data['time'][i] >= time):
                remove_inds.add(i)
                i += 1
                if i>=len(data['time']):
                    break
    if verbose:
        print("Removed {} samples".format(len(remove_inds)))
    keep_inds = set(range(len(data['time']))).difference(remove_inds)
    if verbose:
        print("{} samples remaining".format(len(keep_inds)))
    for sig in data.keys():
        data[sig] = data[sig][list(keep_inds)]
    return data


def remove_I_coil(data, verbose):
    if verbose==2:
        print('Removing weird I-coils')
    c_coil = []
    i_coil = []
    EFC = []
    left = []
    non_standard = []
    non_exist = []


    for i in tqdm(range(len(data['time'])),desc='Finding weird I-coils', ascii=True, dynamic_ncols=True, leave=False, disable=not verbose==1):
        if np.mean(data['bt'][i]*data['curr'][i]) < 0:
            # left-handed
            if not set(np.unique(data['C_coil_method'][i])).issubset({5, 0, -1}):
                c_coil.append(i)
            if not set(np.unique(data['I_coil_method'][i])).issubset({5, 0, -1}):
                i_coil.append(i)
            if not all(np.logical_xor(data['C_coil_method'][i] == 5, data['I_coil_method'][i] == 5)):
                EFC.append(i)
        else:
            left.append(i)
            # right-handed
            if not set(np.unique(data['C_coil_method'][i])).issubset({6, 0, -1}):
                c_coil.append(i)
            if not set(np.unique(data['I_coil_method'][i])).issubset({7, 0, -1}):
                i_coil.append(i)
            if not any(np.logical_or(np.logical_and(data['C_coil_method'][i] == 6, data['I_coil_method'][i] != 7), np.logical_and(data['C_coil_method'][i] != 6, data['I_coil_method'][i] == 7))):
                EFC.append(i)

    coil_inds = [c_coil, i_coil, EFC, non_standard, non_exist]
    coil_inds = list(set().union(*coil_inds))

    remove_inds = set()
    for ind in tqdm(coil_inds, desc='Removing weird I-coils', ascii=True, dynamic_ncols=True, disable=not verbose==1):
        shot = data['shotnum'][ind]
        time = data['time'][ind]
        i = ind
        while any(data['shotnum'][i] == shot) and any(data['time'][i] >= time):
            remove_inds.add(i)
            i += 1
            if i>=len(data['time']):
                break
    if verbose:
        print("Removed {} samples".format(len(remove_inds)))
    keep_inds = set(range(len(data['time']))).difference(remove_inds)
    if verbose:
        print("{} samples remaining".format(len(keep_inds)))
    for sig in data.keys():
        data[sig] = data[sig][list(keep_inds)]
    return data


def remove_gas(data, verbose):
    if verbose==2:
        print('Removing weird gas')
    from functools import reduce
    gasB_inds = np.nonzero(np.any(data['gasB'] > .1, axis=1))[0]
    gasC_inds = np.nonzero(np.any(data['gasC'] > .1, axis=1))[0]
    gasD_inds = np.nonzero(np.any(data['gasD'] > .1, axis=1))[0]
    gasE_inds = np.nonzero(np.any(data['gasE'] > .1, axis=1))[0]
    pfx1_inds = np.nonzero(np.any(data['pfx1'] > .1, axis=1))[0]
    pfx2_inds = np.nonzero(np.any(data['pfx2'] > .1, axis=1))[0]
    gas_inds = reduce(np.union1d, (gasB_inds, gasC_inds,
                                   gasD_inds, gasE_inds, pfx1_inds, pfx2_inds))
    remove_inds = set()
    for ind in tqdm(gas_inds, desc='Removing weird gas', ascii=True, dynamic_ncols=True, disable=not verbose==1):
        shot = data['shotnum'][ind]
        time = data['time'][ind]
        i = ind
        while any(data['shotnum'][i] == shot) and any(data['time'][i] >= time):
            remove_inds.add(i)
            i += 1
            if i>=len(data['time']):
                break
    if verbose:
        print("Removed {} samples".format(len(remove_inds)))
    keep_inds = set(range(len(data['time']))).difference(remove_inds)
    if verbose:
        print("{} samples remaining".format(len(keep_inds)))
    for sig in data.keys():
        data[sig] = data[sig][list(keep_inds)]
    return data


def remove_ECH(data, verbose):
    if verbose==2:
        print('Removing ECH')
    ech_inds = np.nonzero(np.any(data['ech'] > .5, axis=1))[0]
    remove_inds = set()
    for ind in tqdm(ech_inds, desc='Removing ECH', ascii=True, dynamic_ncols=True, disable=not verbose==1):
        shot = data['shotnum'][ind]
        time = data['time'][ind]
        i = ind
        while any(data['shotnum'][i] == shot) and any(data['time'][i] >= time):
            remove_inds.add(i)
            i += 1
            if i>=len(data['time']):
                break
    if verbose:
        print("Removed {} samples".format(len(remove_inds)))
    keep_inds = set(range(len(data['time']))).difference(remove_inds)
    if verbose:
        print("{} samples remaining".format(len(keep_inds)))
    for sig in data.keys():
        data[sig] = data[sig][list(keep_inds)]
    return data

def remove_nan(data, verbose):
    if verbose==2:
        print('Removing NaN')
    remove_inds = set()
    for sig in tqdm(data.keys(), desc='Removing NaN', ascii=True, dynamic_ncols=True, disable=not verbose==1):
        for i, elem in enumerate(data[sig]):
            if np.any(np.isnan(elem)):
                remove_inds.add(i)
    if verbose:
        print("Removed {} samples".format(len(remove_inds)))
    keep_inds = set(range(len(data['time']))).difference(remove_inds)
    if verbose:
        print("{} samples remaining".format(len(keep_inds)))
    for sig in data.keys():
        data[sig] = data[sig][list(keep_inds)]
    return data
