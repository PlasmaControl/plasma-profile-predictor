import pickle
import numpy as np

def generator(separated_data, lookback, delay=1, min_index=0, max_index=None, shuffle=False, batch_size=128, step=1, num_sigs=3):
    if max_index is None: 
        max_index = len(separated_data)-1
    border_indices = np.cumsum([len(elem) for elem in separated_data])
    border_indices = np.insert(border_indices, 0, 0, axis=0)
    data=[]
    for elem in separated_data:
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

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def get_rho_points(num_rho):
    core_points=np.linspace(0,.9*num_rho,25,dtype=int)
    near_edge_points=np.linspace(.9*num_rho,.95*num_rho,5,dtype=int)
    edge_points=np.linspace(.95*num_rho,num_rho,10,dtype=int)
    rho_points=np.concatenate((core_points,near_edge_points,edge_points))
    rho_points=np.unique(rho_points)
    return(rho_points)

def save_obj(obj, name):
    with open(name+'.pkl','wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name+'.pkl', 'rb') as f:
        return pickle.load(f, encoding='latin1')

def get_train_data(root_dir):
    lstm_train_data=load_obj(root_dir+'train_data') 
    lstm_train_target=load_obj(root_dir+'train_target')
    train_data=[]
    train_target=[]
    for shot in sorted(list(lstm_train_data.keys())):
        train_data.extend(lstm_train_data[shot])
        train_target.extend(lstm_train_target[shot])
    return (train_data, train_target)
