from helpers import helper_functions
import numpy as np

def generator(data, batch_size=128, lookback=30, delay=1, min_index=0, max_index=None, shuffle=False, step=1):
    if max_index is None: 
        max_index = len(separated_data)-1
    border_indices = np.cumsum([len(elem) for elem in separated_data])
    border_indices = np.insert(border_indices, 0, 0, axis=0)
    data=[]
    for elem in data:
        data.extend(elem)
    data=np.asarray(data)
    # i iterates over shots, j iterates over timesteps within shots
    separated_possible_indices = [np.arange(border_indices[i]+lookback,border_indices[i+1]-delay) for i in range(min_index, max_index)]
    possible_indices = []
    for elem in separated_possible_indices:
        possible_indices.extend(elem)
    count = 0
    while 1:
        if shuffle:
            rows = np.random.choice(possible_indices, size=batch_size)
        else:
            if count + batch_size >= len(possible_indices):
                count = 0
            rows=possible_indices[count:min(count + batch_size, len(possible_indices))]
            count += len(rows)
        
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),data.shape[-1]-num_sigs))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay,:-num_sigs]
        yield samples, targets

def get_datasets(batch_size, input_data_file):
    k=5 # 1/k of the data is used for validation
    fold=k-1 # which fold of the data to use for validation

    separated_data=helper_functions.load_obj(input_data_file)
    separated_data=[separated_data[key] for key in sorted(separated_data.keys())]

    num_val_samples= len(separated_data) // k

    train_iter = generator(data=separated_data[:fold * num_val_samples]+train_data[(fold + 1) * num_val_samples:],
                           batch_size=batch_size)

    valid_iter = generator(data=separated_data[fold * num_val_samples: (fold + 1) * num_val_samples],
                           batch_size=batch_size)

    return train_iter, valid_iter
