import pickle
import numpy as np
import os
import yaml

def load_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f)
    return config

def save_obj(obj, name):
    with open('{}.pkl'.format(name),'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('{}.pkl'.format(name), 'rb') as f:
        return pickle.load(f, encoding='latin1')

def preprocess_data(input_filename, output_dirname, 
                    sigs_0d, sigs_1d, sigs_predict,
                    lookbacks, delay, 
                    train_frac=.7, val_frac=.2,
                    save_data=False,
                    separated=True, pad_1d_to=0,
                    noised_signal = None, sigma = 0.5, 
                    noised_signal_complete = None, sigma_complete = 1):
    
    # Gaussian normalization, return 0 if std is 0
    def normalize(obj, mean, std):
        a=obj-mean
        b=std
        return np.divide(a, b, out=np.zeros_like(a), where=b!=0)
    
    def remove_nans(arr):
        arr[np.isnan(arr)]=0
        arr[np.isinf(arr)]=0
        return arr

    # load in the raw data
    data=load_obj(input_filename) #os.path.join(dirname,'final_data'))
    # extract all shots that are in the raw data so we can iterate over them
    # shots = sorted(data.keys())    
    shots = data.keys()
    sigs = list(np.unique(sigs_0d+sigs_1d+sigs_predict))

    # first get the indices that contain all the data we need
    # (both train and validation)
    all_shots=[]
    for shot in shots:
       if set(sigs).issubset(data[shot].keys()):
           if all([data[shot][sig].size!=0 and ~np.all(np.isnan(data[shot][sig])) for sig in sigs]):
               all_shots.append(shot)

    def get_non_nan_inds(arr):
        if len(arr.shape)==1:
            return np.where(~np.isnan(arr))[0]
        else:
            return np.where(np.any(~np.isnan(arr),axis=1))[0]
            
    def get_first_index(shot):
        input_max=max([get_non_nan_inds(data[shot][sig])[0]+lookbacks[sig] for sig in sigs_0d+sigs_1d])
        output_max=max([get_non_nan_inds(data[shot][sig])[0]-delay for sig in sigs_predict])
        return max(input_max,output_max)

    def get_last_index(shot):
        partial_min=min([get_non_nan_inds(data[shot][sig])[-1] for sig in sigs_1d])
        full_min=min([get_non_nan_inds(data[shot][sig])[-1]-delay for sig in sigs_0d+sigs_predict])
        return min(full_min, partial_min)
        
    data_all_times={}
    for sig in sigs+['time']:
        data_all_times[sig]=np.array([data[shot][sig] for shot in all_shots])
        data_all_times[sig]=np.concatenate(data_all_times[sig],axis=0)
    data_all_times['shot']=np.array([[shot]*data[shot][sigs[0]].shape[0] for shot in all_shots])
    data_all_times['shot']=np.concatenate(data_all_times['shot'],axis=0)

    indices={}
    subsets=['train','val']
    np.random.seed(1)
    all_shots=np.array(all_shots)
    permuted_shot_inds=np.random.permutation(len(all_shots))
    train_shots=all_shots[permuted_shot_inds[0:int(len(all_shots)*train_frac)]]
    val_shots=all_shots[permuted_shot_inds[int(len(all_shots)*train_frac):int(len(all_shots)*(train_frac+val_frac))]]
    subset_shots={'train':train_shots,'val':val_shots}

    for subset in subsets:
        indices[subset]=[np.where(shot == data_all_times['shot'])[0][get_first_index(shot):get_last_index(shot)] for shot in subset_shots[subset]]
        indices[subset]=np.concatenate(indices[subset])

    means,stds={},{}

    print("Normalizing the same across all rho points")
    for sig in sigs:
        means[sig]=np.nanmean(data_all_times[sig][indices['train']],axis=0)
        stds[sig]=np.nanstd(data_all_times[sig][indices['train']],axis=0)
        # For normalizing all by the same amount
        means[sig]=np.nanmean(means[sig]) 
        stds[sig]=np.nanmean(stds[sig])

    for sig in sigs:
        data_all_times[sig]=normalize(data_all_times[sig],means[sig],stds[sig])
        data_all_times[sig]=remove_nans(data_all_times[sig])

    target={}
    input_data={}
    times={} # never used
    for subset in subsets:
        final_target={}
        # we predict deltas rather than the profiles themselves
        for sig in sigs_predict:
            final_target[sig]=data_all_times[sig][indices[subset]+delay]-data_all_times[sig][indices[subset]]
        target[subset] = final_target
#        print("Sample Target shape for {} data: {}".format(subset, target[subset][sigs_predict[0]].shape))

        # alex's changes here
        pre_0d_dict = {}
        post_0d_dict = {}
        pre_1d_dict = {}
        for sig in sigs_0d:
            pre_0d_dict[sig]=np.stack([data_all_times[sig][indices[subset]+offset] for offset in range(-lookbacks[sig],1)],axis=1)
            post_0d_dict[sig]=np.stack([data_all_times[sig][indices[subset]+offset] for offset in range(1,delay+1)],axis=1)
        for sig in sigs_1d:
            if lookbacks[sig]==0:
                pre_1d_dict[sig]=data_all_times[sig][indices[subset]]
            else:
                pre_1d_dict[sig]=np.stack([data_all_times[sig][indices[subset]+offset] for offset in range(-lookbacks[sig],1)],axis=1)
        
        pre_input_0d = np.array([pre_0d_dict[sig] for sig in sigs_0d])

        pre_input_1d = np.array([pre_1d_dict[sig] for sig in sigs_1d])    

        post_input_0d = np.array([post_0d_dict[sig] for sig in sigs_0d])
        
        
#         print("Pre input 1d shape for {} data: {}".format(subset, pre_input_1d.shape))
#         print("Pre input 0d shape for {} data: {}".format(subset, pre_input_0d.shape))
#         print("Post input 0d shape for {} data: {}".format(subset, post_input_0d.shape))

        input_data[subset] = {"previous_actuators": pre_0d_dict, "previous_profiles": pre_1d_dict, "future_actuators": post_0d_dict}
            
    if save_data:
        print("Saving data to {}...".format(output_dirname))
        for subset in subsets:
            save_obj(data_all_times['time'][indices[subset]], os.path.join(output_dirname,'{}_time'.format(subset)))
            save_obj(data_all_times['shot'][indices[subset]], os.path.join(output_dirname,'{}_shot'.format(subset)))
            save_obj(target[subset],os.path.join(output_dirname,'{}_target'.format(subset)))
            save_obj(input_data[subset],os.path.join(output_dirname,'{}_data'.format(subset)))
        save_obj(means,os.path.join(output_dirname,'means'))
        save_obj(stds,os.path.join(output_dirname,'stds'))

    else:
        return {'train_data': input_data['train'],
                'train_target': target['train'],
                'val_data': input_data['val'],
                'val_target': target['val'],
            }
