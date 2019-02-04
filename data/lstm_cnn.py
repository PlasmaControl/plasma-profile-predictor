import pickle
import numpy as np
from helpers import helper_functions
from keras.utils import Sequence

class RnnDataset(Sequence):
    def __init__(self, batch_size, input_dir_name, train_or_val='train', shuffle='False'):
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.data = np.array(helper_functions.load_obj(input_dir_name+train_or_val+'_data'))
        self.target = np.array(helper_functions.load_obj(input_dir_name+train_or_val+'_target'))
    
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

def get_datasets(batch_size, input_dir_name):

    train_iter = RnnDataset(batch_size=batch_size,
                            input_dir_name=input_dir_name,
                            shuffle='True',
                            train_or_val='train')

    valid_iter = RnnDataset(batch_size=batch_size,
                            input_dir_name=input_dir_name,
                            shuffle='False',
                            train_or_val='val')

    return train_iter, valid_iter