from helpers import helper_functions
import numpy as np
from keras.utils import Sequence

# Ultimately, we'll want to have one file for each shot I think, so we 
# don't have to hold so much data in memory

class RnnDataset(Sequence):

    def __init__(self, batch_size, num_sigs, input_data_file, train_or_val='train', shuffle='False', lookback=30, delay=1):
        self.batch_size = batch_size
        self.num_sigs = num_sigs 
        self.shuffle = shuffle
        self.lookback = lookback
        self.delay = delay

        separated_data = helper_functions.load_obj(input_data_file)
        separated_data = [separated_data[key] for key in sorted(separated_data.keys())]

        k=5 # 1/k of the data is used for validation
        fold=k-1 # which fold of the data to use for validation
        num_val_samples = len(separated_data) // k
        if train_or_val=='train':
            separated_data=separated_data[:fold * num_val_samples]+separated_data[(fold + 1) * num_val_samples:]
        elif train_or_val=='val':
            separated_data=separated_data[fold * num_val_samples: (fold + 1) * num_val_samples]
        else: 
            raise ValueError("Specify either 'train' or 'val' for variable 'train_or_val'")

        border_indices = np.cumsum([len(elem) for elem in separated_data])
        border_indices = np.insert(border_indices, 0, 0, axis=0)
        data=[]
        for elem in separated_data:
            data.extend(elem)
        data=np.asarray(data)

        # i iterates over shots, j iterates over timesteps within shots
        separated_possible_indices = [np.arange(border_indices[i]+self.lookback,border_indices[i+1]-self.delay) for i in range(len(separated_data))]
        possible_indices = []
        for elem in separated_possible_indices:
            possible_indices.extend(elem)

        self.data = data
        self.possible_indices = possible_indices
    
    def __len__(self):
        return int(np.ceil(len(self.possible_indices) / self.batch_size))

    def __getitem__(self, idx):
        return self.__data_generation(idx)

    def __data_generation(self, idx, step=1):
        if (self.shuffle==True):
            rows = np.random.choice(a=self.possible_indices, size=self.batch_size)
        else:
            #if count + self.batch_size >= len(self.possible_indices):
            #    count = 0
            rows = self.possible_indices[idx * self.batch_size : (idx + 1) * self.batch_size]
            #rows = self.possible_indices[count:min(count + self.batch_size, len(self.possible_indices))]
            #count += len(rows)

        samples = np.zeros((len(rows), self.lookback // step, self.data.shape[-1]))
        targets = np.zeros((len(rows),self.data.shape[-1]-self.num_sigs))
        for j, row in enumerate(rows):
            indices = range(rows[j] - self.lookback, rows[j], step)
            samples[j] = self.data[indices]
            targets[j] = self.data[rows[j] + self.delay,:-self.num_sigs]
        return samples, targets

def get_datasets(batch_size, input_data_file, num_sigs):

    train_iter = RnnDataset(batch_size=batch_size,
                            num_sigs=num_sigs,
                            input_data_file=input_data_file,
                            shuffle='True',
                            train_or_val='train')

    valid_iter = RnnDataset(batch_size=batch_size,
                            num_sigs=num_sigs,
                            input_data_file=input_data_file,
                            shuffle='False',
                            train_or_val='val')

    return train_iter, valid_iter
