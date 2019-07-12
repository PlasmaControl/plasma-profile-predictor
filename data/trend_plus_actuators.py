import numpy as np
from helpers.helper_functions import load_obj, save_obj, preprocess_data
from keras.utils import Sequence
import os

class RnnDataset(Sequence):
    def __init__(self, batch_size, processed_data_dirname, train_or_val='train', 
                 shuffle='False', data_package=None):
        self.batch_size = batch_size
        self.shuffle = shuffle

        if data_package is None:
            self.data = load_obj(os.path.join(processed_data_dirname,'{}_data'.format(train_or_val)))
            self.target = load_obj(os.path.join(processed_data_dirname,'{}_target'.format(train_or_val)))
        else:
            self.data = data_package['{}_data'.format(train_or_val)]
            self.target = data_package['{}_target'.format(train_or_val)]

    def __len__(self):
        return int(len(self.target) / self.batch_size)

    def __getitem__(self, idx):
        return self.__data_generation(idx)

    def __data_generation(self, idx, step=1):
        if (self.shuffle==True):
            inds=np.random.choice(len(self.target), size=self.batch_size)
        else:
            inds=list(range(idx * self.batch_size, (idx + 1) * self.batch_size))
            
        return [elem[inds] for elem in self.data], self.target[inds]

def get_datasets(batch_size, input_filename, output_dirname, preprocess, 
                 sigs_0d, sigs_1d, sigs_predict,
                 lookback, delay, 
                 train_frac, val_frac):
    
    if (preprocess): 
        data_package = preprocess_data(input_filename, output_dirname, 
                                       sigs_0d, sigs_1d, sigs_predict,
                                       lookback, delay,
                                       train_frac, val_frac, 
                                       save_data=False)
        # data_package = preprocess_data(input_filname,
        #                                                 output_dirname,
        #                           sigs_0d, sigs_1d, sigs_predict,
        #                           n_components, avg_window, 
        #                           lookback, delay,
        #                           train_frac, val_frac, False, pad_1d_to = pad_1d_to)
        # print('baseline maes:\n'+str(np.mean(abs(data_package['val_target']),axis=0)))
        # print('baseline mae average:\n'+str(np.mean(abs(data_package['val_target']))))

    else:
        data_package = None

    train_iter = RnnDataset(batch_size=batch_size,
                            processed_data_dirname=output_dirname,
                            shuffle='True',
                            train_or_val='train',
                            data_package=data_package)

    valid_iter = RnnDataset(batch_size=batch_size,
                            processed_data_dirname=output_dirname,
                            shuffle='False',
                            train_or_val='val',
                            data_package=data_package)

    return train_iter, valid_iter
