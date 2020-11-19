import numpy as np
import os
import pickle
import time

input_dir='/projects/EKOLEMEN/profile_predictor/DATA/data_with_rt'
output_dir='/projects/EKOLEMEN/profile_predictor/DATA/data_with_rt'

# to get list: ls | grep -o '[0-9][0-9][0-9]' | tr '\n' ','
filenums=[125,134,145,155,189,197,213,225,229,233,243,248,258,259,269,272,273,281,299,303,335] #enumerate(range(227)):
for i,filenum in enumerate(filenums): 
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
        
