import numpy as np
import os
import pickle

input_dir='/scratch/gpfs/jabbate/test_data'
output_dir='/scratch/gpfs/jabbate/test_data'

for i,filenum in enumerate(range(20)):
    with open(os.path.join(input_dir,'final_data_batch_{}.pkl'.format(filenum)),'rb') as f:
        new_dict=pickle.load(f,encoding='latin1')
    if i==0:
        big_dict=new_dict
    else:
        big_dict.update(new_dict)
with open(os.path.join(output_dir,'final_data.pkl'),'wb') as f:
    pickle.dump(big_dict,f)
        
