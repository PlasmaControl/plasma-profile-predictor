import numpy as np
import os
import pickle
import time

input_dir='/scratch/gpfs/jabbate/mixed_data'
output_dir='/scratch/gpfs/jabbate/mixed_data'

for i,filenum in enumerate(range(222)):
    begin_time=time.time()
    with open(os.path.join(input_dir,'final_data_batch_{}.pkl'.format(filenum)),'rb') as f:
        new_dict=pickle.load(f,encoding='latin1')
    if i==0:
        big_dict=new_dict
    else:
        big_dict.update(new_dict)
    print('{}: {}'.format(i,time.time()-begin_time))
with open(os.path.join(output_dir,'final_data.pkl'),'wb') as f:
    pickle.dump(big_dict,f)
        
