import numpy as np
import os
import pickle

input_dir='/home/jabbate/processed_data/'
output_dir='/home/jabbate/full_data/'

for which_data in ['train','val']:
    for i,filenum in enumerate([0,1]):
        with open(os.path.join(input_dir,'{}_{}.pkl'.format(which_data,i)),'rb') as f:
            new_dict=pickle.load(f)
        if i==0:
            big_dict=new_dict
        else:
            for sig in big_dict:
                big_dict[sig]=np.concatenate([big_dict[sig],new_dict[sig]],axis=0)
    with open(os.path.join(output_dir,'{}.pkl'.format(which_data)),'wb') as f:
        pickle.dump(big_dict,f)
    
import pdb; pdb.set_trace()
