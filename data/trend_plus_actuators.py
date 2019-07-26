import numpy as np
from helpers.helper_functions import load_obj, save_obj, preprocess_data
from keras.utils import Sequence
import os

class RnnDataset(Sequence):
    def __init__(self, batch_size, processed_data_dirname, sigs_0d, sigs_1d, sigs_predict, train_or_val='train', 
                 shuffle='False', data_package=None):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sigs_0d = sigs_0d
        self.sigs_1d = sigs_1d
        self.sigs_predict = sigs_predict

        if data_package is None:
            self.data = load_obj(os.path.join(processed_data_dirname,'{}_data'.format(train_or_val)))
            self.target = load_obj(os.path.join(processed_data_dirname,'{}_target'.format(train_or_val)))
        else:
            self.data = data_package['{}_data'.format(train_or_val)]
            self.target = data_package['{}_target'.format(train_or_val)]
            
        if train_or_val == "val":
            for sig in sigs_predict:
                baseline = np.mean(np.abs(self.target[sig]))
                #print("baseline mae for {} for each rho point: {}".format(sig, rho_baseline))
                print("baseline mae averaged for {} over all rho points: {}".format(sig, baseline))
        

    def __len__(self):
        return int(len(self.target[self.sigs_predict[0]]) / self.batch_size)
#         return int(self.target.shape[1] / self.batch_size)
            
    
    def __getitem__(self, idx):
        return self.__data_generation(idx)

    def __data_generation(self, idx, step=1):
        if (self.shuffle==True):
            inds=np.random.choice(len(self.target[self.sigs_predict[0]]), size=self.batch_size)
        else:
            inds=list(range(idx * self.batch_size, (idx + 1) * self.batch_size))
        

        generated_train_dict = {}
        generated_target_dict = {}
        
        
        # TRAINING WORK***************************************************************************************************

        # combine the actuators together    
        future_actuators = np.concatenate(([self.data["future_actuators"][sig][inds, :, np.newaxis] for sig in self.sigs_0d]), axis =2)
        previous_actuators = np.concatenate(([self.data["previous_actuators"][sig][inds, :, np.newaxis] for sig in self.sigs_0d]), axis =2)
        
        
        
        
#         # if you want to lump actuators together (Rory's models) and have profile sigs separate. 
#         #########################layer names: ----1d sig names, "all_actuators"
            
#         generated_train_dict["all_actuators"] = np.concatenate((previous_actuators, future_actuators), axis = 1)
#         # get previous profiles data
#         for index, sig in enumerate(self.sigs_1d):
#             generated_train_dict[sig] = self.data["previous_profiles"][sig][inds]
            
#         ##########################
        
        
        
        
        
        
        # if you want to split based on time (ie previous vs future) where you mash together all the sigs 
        # Joe's model#####################################
        # layer names:
        # ---"all_previous_sigs", "future_actuators"
        
        # combine all profiles
        prev_profiles = np.concatenate(([self.data["previous_profiles"][sig][inds] for sig in self.sigs_1d]), axis = 2)
        prev_profiles = np.concatenate((prev_profiles, previous_actuators), axis = 2)
        
        # read data into the output dictionary
        generated_train_dict["all_previous_sigs"] = prev_profiles
        
        generated_train_dict["future_actuators"] = future_actuators
        
        ###################################################
        
        
        
        
        
        
           

        # TARGET WORK ####################################
        for index, sig in enumerate(self.sigs_predict):
            generated_target_dict["target_{}".format(sig)] = self.target[sig][inds]

    
            
        return generated_train_dict, generated_target_dict

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
                            data_package=data_package, sigs_0d = sigs_0d, sigs_1d = sigs_1d, 
                            sigs_predict = sigs_predict)

    valid_iter = RnnDataset(batch_size=batch_size,
                            processed_data_dirname=output_dirname,
                            shuffle='False',
                            train_or_val='val',
                            data_package=data_package, sigs_0d = sigs_0d, sigs_1d = sigs_1d, 
                            sigs_predict = sigs_predict)

    return train_iter, valid_iter
