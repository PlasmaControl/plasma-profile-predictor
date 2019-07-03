import numpy as np
from helpers import helper_functions
from keras.utils import Sequence

class RnnDataset(Sequence):
    def __init__(self, batch_size, input_dir_name, train_or_val='train', 
                 shuffle='False', data_package=None):
        self.batch_size = batch_size
        self.shuffle = shuffle

        if data_package is None:
            self.data = np.array(helper_functions.load_obj(input_dir_name+train_or_val+'_data'))
            self.target = np.array(helper_functions.load_obj(input_dir_name+train_or_val+'_target'))
        else:
            self.data = data_package['{}_data'.format(train_or_val)]
            self.target = data_package['{}_target'.format(train_or_val)]

    def __len__(self):
        return int(len(self.data) / self.batch_size)

    def __getitem__(self, idx):
        return self.__data_generation(idx)

    def __data_generation(self, idx, step=1):
        if (self.shuffle==True):
            inds=np.random.choice(len(self.data), size=self.batch_size)
        else:
            inds=list(range(idx * self.batch_size, (idx + 1) * self.batch_size))
            
        return self.data[inds], self.target[inds]

def get_datasets(batch_size, input_dir_name, preprocess, sigs_0d, sigs_1d, sigs_predict,
                 n_components, avg_window, lookback, delay, noised_signal,
                 train_frac, val_frac, pad_1d_to):

    if (preprocess):
        data_package = helper_functions.preprocess_data(input_dir_name, 
                                  sigs_0d, sigs_1d, sigs_predict,
                                  n_components, avg_window, 
                                  lookback, delay,
                                  train_frac, val_frac, False, pad_1d_to = pad_1d_to)
        print('baseline maes:\n'+str(np.mean(abs(data_package['val_target']),axis=0)))
        print('baseline mae average:\n'+str(np.mean(abs(data_package['val_target']))))

    else:
        data_package = None

    train_iter = RnnDataset(batch_size=batch_size,
                            input_dir_name=input_dir_name,
                            shuffle='True',
                            train_or_val='train',
                            data_package=data_package)

    valid_iter = RnnDataset(batch_size=batch_size,
                            input_dir_name=input_dir_name,
                            shuffle='False',
                            train_or_val='val',
                            data_package=data_package)

    return train_iter, valid_iter

