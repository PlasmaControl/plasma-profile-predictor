import numpy as np
import os
import pickle
import time

input_dir='/scratch/gpfs/jabbate/new_data_EFIT02'
output_dir='/scratch/gpfs/jabbate/new_data_EFIT02'

for i,filenum in enumerate(range(227)):
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
        
