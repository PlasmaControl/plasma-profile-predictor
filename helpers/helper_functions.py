import pickle
import numpy as np
import copy 
from sklearn import decomposition
import os
import random
from random import sample

def save_obj(obj, name):
    with open('{}.pkl'.format(name),'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('{}.pkl'.format(name), 'rb') as f:
        return pickle.load(f, encoding='latin1')

def preprocess_data(input_filename, output_dirname, sigs_0d, sigs_1d, sigs_predict,
                    n_components=8,
                    avg_window=10, lookback=10,
                    delay=1, train_frac=.05, val_frac=.05,
                    save_data=False, noised_signal = None, sigma = 0.5, 
                    noised_signal_complete = None, sigma_complete = 1, pad_1d_to = 0):
    
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
        input_data[subset]=np.concatenate([final_input_0d,final_input_1d],axis=2)
        
    if save_data:
        for subset in subsets:
            save_obj(data_all_times['time'][indices[subset]], os.path.join(output_dirname,'{}_time'.format(subset)))
            save_obj(data_all_times['shot'][indices[subset]], os.path.join(output_dirname,'{}_shot'.format(subset)))
            save_obj(target[subset],os.path.join(output_dirname,'{}_target'.format(subset)))
            save_obj(input_data[subset],os.path.join(output_dirname,'{}_data'.format(subset)))
        save_obj(means,os.path.join(output_dirname,'means'))
        save_obj(stds,os.path.join(output_dirname,'stds'))

    else:
        pass

    # function for creating data using the raw and the means / stds
#     def make_final_data(my_shots): #start_time,end_time): 
#         import time
#         time_very_beg = time.time()
#         final_data=[]
#         final_target=[]
#         shot_indices=[]
#         times=[]
#         i = 0
#         count = 0
        
#         shot_indices.append(0) #always start the first shot at 0

#         sigs_predict_lengths=[data[my_shots[0]][sig_predict].shape[1] for sig_predict in sigs_predict]
# #        num_timesteps=data[my_shots][sigs[0]]
        
#         time_before=time.time()
#         # time for normalization:
#         data_all_times_normed={}
#         data_all_times_normed['thomson_temp']=normalize(data_all_times['thomson_temp'],means['thomson_temp'],stds['thomson_temp'])
#         print(data_all_times['thomson_temp'].shape)
#         print(data_all_times_normed['thomson_temp'].shape)
#         import pdb; pdb.set_trace()
#         for shot in my_shots:
#             num_timesteps=len(data[shot][sigs[0]])
#             count+=num_timesteps
#              # normalize each sig for each timestep for each shot
#             for sig in sigs:
#                 for cur_time in range(num_timesteps):
#                     data[shot][sig][cur_time] = normalize(data[shot][sig][cur_time], means[sig], stds[sig])
            
            
#             all_timesteps=range(lookback, num_timesteps-delay)
#             shot_indices.append(shot_indices[-1]+len(all_timesteps))
#         print('Normalizing time: {}s'.format(time.time()-time_before))

#         time_before=time.time()
#         # time to do targets
#         final_target=np.full((count,sum(sigs_predict_lengths)),0)
#         count=0
#         for shot in my_shots:
#             num_timesteps=len(data[shot][sigs[0]])
#             all_timesteps=range(lookback, num_timesteps-delay)
#              # first get the targets done before noising things up
#             for end_time in all_timesteps:
#                 for sig in sigs_predict:
#                     final_target[count,:]=data[shot][sig][end_time+delay]-data[shot][sig][end_time]
#                 count+=1
#         print('Target time: {}s'.format(time.time()-time_before))

#         time_before=time.time()
#         # time to do data
#         for shot in my_shots: 
#             # i is a debuger for noise
            
#             count = 0 # count counts how many total timesteps there are
#             # i also gets incremented when padding in prediction mode
#             i = i + 1
#             num_timesteps=len(data[shot][sigs[0]])
#              # normalize each sig for each timestep for each shot
#             # for cur_time in range(num_timesteps):
#             #     for sig in sigs:
#             #         data[shot][sig][cur_time] = normalize(data[shot][sig][cur_time], means[sig], stds[sig])
            
            
#             all_timesteps=range(lookback, num_timesteps-delay)
#             # shot_indices.append(shot_indices[-1]+len(all_timesteps))
            
#             # # first get the targets done before noising things up
#             # for end_time in all_timesteps:
#             #     count += 1
#             #     final_target.append([])
#             #     for sig in sigs_predict:
#             #         final_target[-1].extend((data[shot][sig][end_time+delay])-
#             #                                (data[shot][sig][end_time]))
            
    
#             # noise data along curve
#             if noised_signal is not None:
#                 if i == 1: 
#                     print("noising data along curve for " + noised_signal + " with " + str(sigma))
                
#                 for cur_time in range(num_timesteps):
#                     data[shot][noised_signal][cur_time] = data[shot][noised_signal][cur_time] + np.random.normal(0,sigma, data[shot][noised_signal][cur_time].shape)
#             else:
#                 if i ==1:
#                     pass#print("data is not being noised along curve")
                    
#             # noise data COMPLETELY
#             if noised_signal_complete is not None:
#                 if i == 1: 
#                     print("noising data COMPLETELY " + noised_signal_complete + " with " + str(sigma_complete))
                
#                 for cur_time in range(num_timesteps):
#                     data[shot][noised_signal_complete][cur_time] = np.random.normal(0,sigma_complete, data[shot][noised_signal_complete][cur_time].shape)
#             else:
#                 if i ==1:
#                     pass#print("data is not being noised completely")
                    
                    
                    
            
            
            
#             for end_time in all_timesteps:
#                 times.append(data[shot]['time'][end_time])
                
#                 final_data.append([])
                
                
                
#                     # for MEAN:
#                     #final_target[-1].append(np.mean(normalize(data[shot][sig][end_time+delay], means[sig], stds[sig])))

#                     # for predicting DIFFERENCES


#                     # for predicting MEAN, DIFFERENCES
#                     #final_target[-1].append(np.mean(normalize(data[shot][sig][end_time+delay], means[sig], stds[sig])-
#                     #                       normalize(data[shot][sig][end_time], means[sig], stds[sig])))

#                     # regular:
#                     #final_target[-1].extend(normalize(data[shot][sig][end_time+delay], means[sig], stds[sig]))

#                 # start lookback steps behind, then add the current signal ("end_time"). For 0d, fill in the rest of the values
#                 # up through delay. For 1d, fill in with 0s
#                 for mytime in range(end_time-lookback,end_time+1+delay):
#                     final_data[-1].append([])
#                     for sig in sigs_0d:
#                         new_sig = (data[shot][sig][mytime])
#                         final_data[-1][-1].append(new_sig)
                    
#                     for sig in sigs_1d:
#                         # pad with 0s once we start going into prediction mode
#                         if (mytime>end_time):
#                             if i == 1:
#                                 #print("padding 1d sigs in prediction mode to " + str(pad_1d_to))
#                                 i += 1
#                             new_sig = np.zeros(data[shot][sig][mytime].shape) + pad_1d_to
#                         else:
#                             new_sig = (data[shot][sig][mytime])

#                         # for just MEAN:
#                         #final_data[-1][-1].append(np.mean(new_sig))

#                         # regular:
#                         final_data[-1][-1].extend(new_sig)
        
       
#         print('Data time: {}s'.format(time.time()-time_before))     
#         shot_indices.pop() #we added 0 to beginning, so exclude the last element
#         #print("Number of timesteps in data:\n {}".format(count))
#         print('Total time for make final data: {}s'.format(time.time()-time_very_beg))     

#         return (np.array(final_data), np.array(final_target), np.array(shot_indices), np.array(times))

#     time_before=time.time()
#     train_tuple = make_final_data(train_shots)
#     print('Putting training data into right shape: {}s'.format(time.time()-time_before))

#     train_data = train_tuple[0]
#     train_target = train_tuple[1]
#     train_indices = train_tuple[2]
#     train_time = train_tuple[3]

#     time_before=time.time()
#     val_tuple = make_final_data(val_shots)
#     print('Putting validation data into right shape: {}s'.format(time.time()-time_before))

#     val_data = val_tuple[0]
#     val_target = val_tuple[1]
#     val_indices = val_tuple[2]
#     val_time = val_tuple[3]

    ##############################
    # BEGIN PCA
    ##############################

    # if ((type(n_components) is int) and (len(sigs_1d)>0)):
    #     rho_length=int((train_data.shape[2]-len(sigs_0d))/len(sigs_1d))
    #     train_data_pca = np.zeros((train_data.shape[0],train_data.shape[1],len(sigs_0d)+len(sigs_1d)*n_components))
    #     val_data_pca = np.zeros((val_data.shape[0],val_data.shape[1],len(sigs_0d)+len(sigs_1d)*n_components))
        
    #     if len(sigs_0d)>0:
    #         train_data_pca[:,:,:len(sigs_0d)]=train_data[:,:,:len(sigs_0d)]
    #         val_data_pca[:,:,:len(sigs_0d)]=val_data[:,:,:len(sigs_0d)]
        
    #     for i,sig in enumerate(sigs_1d):
    #         pre_pca_data=train_data[:,-1,len(sigs_0d)+i*rho_length:len(sigs_0d)+(i+1)*rho_length]
    #         pca=decomposition.PCA(n_components=n_components)
    #         pca.fit(pre_pca_data)

    #         for t in range(train_data.shape[1]):
    #             train_data_pca[:,t,len(sigs_0d)+i*n_components:len(sigs_0d)+(i+1)*n_components] = pca.transform(train_data[:,t,len(sigs_0d)+i*rho_length:len(sigs_0d)+(i+1)*rho_length])

    #         for t in range(val_data.shape[1]):
    #             val_data_pca[:,t,len(sigs_0d)+i*n_components:len(sigs_0d)+(i+1)*n_components] = pca.transform(val_data[:,t,len(sigs_0d)+i*rho_length:len(sigs_0d)+(i+1)*rho_length])
            
            
    #     train_data=train_data_pca
    #     val_data=val_data_pca
        
    #     if save_data:
    #         with open(dirname+'pca.pkl', 'wb') as f:
    #             pickle.dump(pca, f)

    ##############################
    # END PCA
    ##############################


    ##############################
    # BEGIN SAVING ALL INFO
    ##############################
    # save train and val inputs and outputs
    # if save_data:
    #     print("saving data to " + dirname + "...")
    #     with open(dirname+'train_data.pkl', 'wb') as f: 
    #         pickle.dump(train_data, f)
    #     with open(dirname+'train_target.pkl', 'wb') as f: 
    #         pickle.dump(train_target, f)
    #     with open(dirname+'val_data.pkl', 'wb') as f: 
    #         pickle.dump(val_data, f)
    #     with open(dirname+'val_target.pkl', 'wb') as f: 
    #         pickle.dump(val_target, f)

    #     # save means and stds
    #     with open(dirname+'means.pkl', 'wb') as f: 
    #         pickle.dump(means, f)
    #     with open(dirname+'stds.pkl', 'wb') as f: 
    #         pickle.dump(stds, f)

    #     shot_indices={'train_shot_indices': train_indices,
    #                  'val_shot_indices': val_indices,
    #                  'train_shot_names': train_shots,
    #                  'val_shot_names': val_shots}
    #     with open(dirname+'shot_indices.pkl', 'wb') as f: 
    #         pickle.dump(shot_indices, f)

    #     with open(dirname+'train_time.pkl', 'wb') as f: 
    #         pickle.dump(train_time, f)
    #     with open(dirname+'val_time.pkl', 'wb') as f: 
    #         pickle.dump(val_time, f)
    #     print("data saved to " + dirname)
    #     ##############################
    #     # END SAVING ALL INFO
    #     ##############################
    # else:
    #     return {'train_data': train_data, 
    #             'train_target': train_target, 
    #             'val_data': val_data, 
    #             'val_target': val_target}
