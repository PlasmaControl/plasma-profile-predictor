def remove_dudtrip(data):
    print("Removing samples with dudtrip")
    dud_trip_inds = np.nonzero(data['dud_trip'])[0]
    remove_ind = set()
    for ind in dud_trip_inds:
        shot = data['shotnum'][ind]
        time = data['time'][ind]
        for i in range(len(data['time'])):
            if data['shotnum'][i] == shot and data['time'][i] > time:
                remove_inds.add(i)
    print("Removed {} samples".format(len(remove_inds)))
    for sig in data.keys():
        data[sig] = data[sig][list(remove_inds)]
    return data


def remove_I_coil(data):
    print("Removing weird coil configurations")
    c_coil = []
    i_coil = []
    EFC = []
    left = []
    non_standard = []
    non_exist = []


for i in range(len(data['time'])):
    if np.mean(data['bt'][i]*data['curr'][i]) < 0:
        # left-handed
        if not set(np.unique(data['C_coil_method'][i])).issubset({5, 0, -1}):
            c_coil.append(i)
        if not set(np.unique(data['I_coil_method'][i])).issubset({5, 0, -1}):
            i_coil.append(i)
        if not np.logical_xor(data['C_coil_method'][i] == 5, data['I_coil_method'][i] == 5):
            EFC.append(i)
    else:
        left.append(i)
        # right-handed
        if not set(np.unique(data['C_coil_method'][i])).issubset({6, 0, -1}):
            c_coil.append(i)
        if not set(np.unique(data['I_coil_method'][i])).issubset({7, 0, -1}):
            i_coil.append(i)
        if not np.any(np.logical_or(np.logical_and(data['C_coil_method'][i] == 6, data['I_coil_method'][i] != 7), np.logical_and(data['C_coil_method'][i] != 6, data['I_coil_method'][i] == 7))):
            EFC.append(i)

    errors = [c_coil, i_coil, EFC, non_standard, non_exist]
    remove_inds = list(set(c_coil).union(*errors))
    print("Removed {} samples".format(len(remove_inds)))
    keep_ind = set(range(len(data['time']))).difference(remove_inds)
    print("{} samples remaining".format(len(keep_inds)))
    for sig in data.keys():
        data[sig] = data[sig][list(keep_inds)]
    return data


def remove_gas(data):
    from functools import reduce
    print("Removing samples with weird gas")
    gasB_inds = np.nonzero(np.any(data['gasB'] > .1, axis=1))[0]
    gasC_inds = np.nonzero(np.any(data['gasC'] > .1, axis=1))[0]
    gasD_inds = np.nonzero(np.any(data['gasD'] > .1, axis=1))[0]
    gasE_inds = np.nonzero(np.any(data['gasE'] > .1, axis=1))[0]
    pfx1_inds = np.nonzero(np.any(data['pfx1'] > .1, axis=1))[0]
    pfx2_inds = np.nonzero(np.any(data['pfx2'] > .1, axis=1))[0]
    gas_inds = reduce(np.union, (gasB_inds, gasC_inds,
                                 gasD_inds, gasE_inds, pfx1_inds, pfx2_inds))
    remove_inds = set()
    for ind in gas_inds:
        shot = data['shotnum'][ind]
        time = data['time'][ind]
        for i in range(len(data['time'])):
            if data['shotnum'][i] == shot and data['time'][i] > time:
                remove_inds.add(i)
    print("Removed {} samples".format(len(remove_inds)))
    keep_ind = set(range(len(data['time']))).difference(remove_inds)
    print("{} samples remaining".format(len(keep_inds)))
    for sig in data.keys():
        data[sig] = data[sig][list(keep_inds)]
    return data


def remove_ECH(data):
    from functools import reduce
    print("Removing samples with weird gas")
    ech_inds = np.nonzero(np.any(data['ech'] > .5, axis=1))[0]
    remove_inds = set()
    for ind in ech_inds:
        shot = data['shotnum'][ind]
        time = data['time'][ind]
        for i in range(len(data['time'])):
            if data['shotnum'][i] == shot and data['time'][i] > time:
                remove_inds.add(i)
    print("Removed {} samples".format(len(remove_inds)))
    keep_ind = set(range(len(data['time']))).difference(remove_inds)
    print("{} samples remaining".format(len(keep_inds)))
    for sig in data.keys():
        data[sig] = data[sig][list(keep_inds)]
    return data


def remove_shot(data, shots):
    print("Removing selected shots")
    remove_inds = set()
    for i, elem in enumerate(data['shotnum']):
        if elem in shots:
            remove_inds.add(i)
    print("Removed {} samples".format(len(remove_inds)))
    keep_ind = set(range(len(data['time']))).difference(remove_inds)
    print("{} samples remaining".format(len(keep_inds)))
    for sig in data.keys():
        data[sig] = data[sig][list(keep_inds)]
    return data


def remove_nan(data):
    print("Removing samples with partial NaN")
    remove_inds = set()
    for sig in data.keys():
        for i, elem in enumerate(data[sig]):
            if np.any(np.isnan(elem)):
                remove_inds.add(i)
    print("Removed {} samples".format(len(remove_inds)))
    keep_ind = set(range(len(data['time']))).difference(remove_inds)
    print("{} samples remaining".format(len(keep_inds)))
    for sig in data.keys():
        data[sig] = data[sig][list(keep_inds)]
    return data
