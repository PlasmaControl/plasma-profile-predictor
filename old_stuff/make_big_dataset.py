import os
import pickle
import numpy as np
import time

output_dir='/scratch/gpfs/jabbate/mixed_data'

old_dir='/scratch/gpfs/jabbate/old_full_data'
new_dir='/scratch/gpfs/jabbate/full_data'

signals=[]


begin_time=time.time()

with open(os.path.join(old_dir,'final_data.pkl'),'rb') as f:
    olddata=pickle.load(f,encoding='latin1')
print('Loaded old data, took {}s'.format(time.time()-begin_time))

for i in range(222):
    begin_time=time.time()
    with open(os.path.join(new_dir,'final_data_batch_{}.pkl'.format(i)),'rb') as f:
        newdata=pickle.load(f,encoding='latin1')
    for shot in newdata:
        try: 
            for signal in olddata[shot]:
                if signal not in newdata[shot]:
                    newdata[shot][signal]=olddata[shot][signal]
        except:
            print('Error with shot {}'.format(shot))
    with open(os.path.join(output_dir,'final_data_batch_{}.pkl'.format(i)),'wb') as f:
        pickle.dump(newdata,f)
    print('{}: {}s'.format(i,time.time()-begin_time))

