import numpy as np
from helpers.helper_functions import load_obj, save_obj
from keras.utils import Sequence
import os

class RnnDataset(Sequence):
    def __init__(self, batch_size, processed_data_dirname, train_or_val='train', 
                 shuffle='False', data_package=None):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data = load_obj(os.path.join(processed_data_dirname,'final_data'))

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

def get_datasets(batch_size, input_filename, output_dirname, preprocess, sigs_0d, sigs_1d, sigs_predict,
                 n_components, avg_window, lookback, delay, 
                 train_frac, val_frac,
                 noised_signal=None, pad_1d_to=0):
    
    if (preprocess):
        data_package = preprocess_data(input_filname,
                                                        output_dirname,
                                  sigs_0d, sigs_1d, sigs_predict,
                                  n_components, avg_window, 
                                  lookback, delay,
                                  train_frac, val_frac, False, pad_1d_to = pad_1d_to)
        print('baseline maes:\n'+str(np.mean(abs(data_package['val_target']),axis=0)))
        print('baseline mae average:\n'+str(np.mean(abs(data_package['val_target']))))

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

def preprocess_data(input_filename, output_dirname, sigs_0d, sigs_1d, sigs_predict,
                    n_components=8,
                    avg_window=10, lookback=10,
                    delay=1, train_frac=.05, val_frac=.05,
                    save_data=False, noised_signal = None, sigma = 0.5, 
                    noised_signal_complete = None, sigma_complete = 1):
    
    # n_components 0 means we don't want to include 1d signal in the input at all
    if (n_components==0):
        sigs_1d=[]

    # Gaussian normalization, return 0 if std is 0
    def normalize(obj, mean, std):
        a=obj-mean
        b=std
        return np.divide(a, b, out=np.zeros_like(a), where=b!=0)
    
    def finalize_signal(arr):
        arr[np.isnan(arr)]=0
        arr[np.isinf(arr)]=0
        return arr

    # load in the raw data
    data=load_obj(input_filename) #os.path.join(dirname,'final_data'))

    # extract all shots that are in the raw data so we can iterate over them
    shots = sorted(data.keys())    
    sigs = list(np.unique(sigs_0d+sigs_1d+sigs_predict))

    # first get the indices that contain all the data we need
    # (both train and validation)
    all_shots=[]
    for shot in shots:
       if set(sigs).issubset(data[shot].keys()):
           if all([data[shot][sig].size!=0 for sig in sigs]):
               all_shots.append(shot)
            
    data_all_times={}
    for sig in sigs+['time']:
        data_all_times[sig]=np.array([data[shot][sig] for shot in all_shots])
        data_all_times[sig]=np.concatenate(data_all_times[sig],axis=0)
        data_all_times[sig]=finalize_signal(data_all_times[sig])
    data_all_times['shot']=np.array([[shot]*data[shot][sigs[0]].shape[0] for shot in all_shots])
    data_all_times['shot']=np.concatenate(data_all_times['shot'],axis=0)

    indices={}
    subsets=['train','val']
    train_shots = all_shots[:int(len(all_shots)*train_frac)]    
    val_shots = all_shots[int(len(all_shots)*train_frac):int(len(all_shots)*(train_frac+val_frac))]
    subset_shots={'train':train_shots,'val':val_shots}
    
    def get_first_ind(arr,val):
        return np.searchsorted(arr,val) + lookback
    def get_last_ind(arr,val):
        return np.searchsorted(arr,val,side='right') - delay

    for subset in subsets:
        indices[subset]=[np.arange(get_first_ind(data_all_times['shot'],shot),get_last_ind(data_all_times['shot'],shot)+1) 
                         for shot in subset_shots[subset]]
        indices[subset]=np.concatenate(indices[subset])

    means={}
    stds={}
    for sig in sigs:
        means[sig]=np.mean(data_all_times[sig][indices['train']],axis=0)
        stds[sig]=np.std(data_all_times[sig][indices['train']],axis=0)

    data_all_times_normed={}
    for sig in sigs:
        data_all_times_normed[sig]=normalize(data_all_times[sig],means[sig],stds[sig])

    target={}
    input_data={}
    times={}
    for subset in subsets:
        final_target={}
        for sig in sigs_predict:
            final_target[sig]=data_all_times_normed[sig][indices[subset]+delay]-data_all_times_normed[sig][indices[subset]]
        target[subset]=np.concatenate([final_target[sig] for sig in sigs_predict],axis=1)
        
        final_input={}
        for sig in sigs_0d+sigs_1d:
            final_input[sig]={}
            final_input[sig]=np.stack([data_all_times_normed[sig][indices[subset]+offset] for offset in range(-lookback,delay+1)],axis=1)
        final_input_0d=np.concatenate([final_input[sig][:,:,np.newaxis] for sig in sigs_0d],axis=2)
        final_input_1d=np.concatenate([final_input[sig] for sig in sigs_1d],axis=2)
        final_input_1d[:,-delay:,:]=pad_1d_to
        input_data[subset]=np.concatenate([final_input_0d,final_input_1d],axis=2)
        
    if save_data:
        for subset in subsets:
            save_obj(data_all_times['time'][indices[subset]], os.path.join(output_dirname,'{}_time'.format(subset)))
            save_obj(data_all_times['shot'][indices[subset]], os.path.join(output_dirname,'{}_shot'.format(subset)))
            save_obj(target[subset],os.path.join(output_dirname,'{}_target'.format(subset)))
            save_obj(input_data[subset],os.path.join(output_dirname,'{}_data'.format(subset)))
        save_obj(means,os.path.join(output_dirname,'means'))
        save_obj(stds,os.path.join(output_dirname,'stds'))

    # not yet implemented
    else:
        pass
