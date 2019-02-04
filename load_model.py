from helpers import helper_functions
import time

import tensorflow as tf
import numpy as np

root_dir='/global/homes/a/abbatej/plasma_profiles_predictor/'
input_dir='/global/homes/a/abbatej/'
train_data = helper_functions.load_obj(input_dir+'train_data')
train_target = helper_functions.load_obj(input_dir+'train_target')
val_data = helper_functions.load_obj(input_dir+'val_data')
val_target = helper_functions.load_obj(input_dir+'val_target')
rho_points = helper_functions.load_obj(input_dir+'rho_standard')


from keras import models
from keras import layers
from keras import optimizers

#num_rho=200
#k=5 # number by which you divided the data into train/validation
#i=0 # choose which fold to use
#num_val_samples=len(train_data) // k
#val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
#val_target = train_target[i * num_val_samples: (i + 1) * num_val_samples]

loaded_model=models.load_model(root_dir+'test_model.h5')
print("Loaded model from disk")

loaded_model.compile(optimizer=optimizers.RMSprop(lr=.001),
                     metrics=['mae'], loss='mse')

import matplotlib.pyplot as plt

font={'size': 10}
plt.rc('font', **font)
def plot_timesteps(arr, train=False):
    if isinstance(arr,int):
        arr=[arr]
    #rho_points=helper_functions.get_rho_points(num_rho)
    #rho_points=[i/max(rho_points) for i in rho_points]
    if(train):
        data=train_data
        target=train_target
    else:
        data=val_data
        target=val_target

    fig,axes = plt.subplots(len(arr),1)
    for i,timestep in enumerate(arr):
        pred=np.ndarray.flatten(loaded_model.predict(np.array(data)[timestep:timestep+1]))
        true=target[timestep]
        prev=data[timestep][-1][-len(true):]
        sigs=data[timestep][-1][:-len(true)] #pinj, tinj, curr
        
        try: 
            ax=axes[i]
        except: 
            ax=axes
        if(i==0):
            ax.set_title('Normalized e_temp Predictions on Novel Data')
        if(i==len(arr)-1):
            ax.set_xlabel('Normalized rho')
        ax.plot(rho_points,pred,label='Predicted')
        ax.plot(rho_points,true,label='True')
        ax.plot(rho_points,prev,label='Previous')
        textstr='\n'.join((
            'Normalized actuator values:',
            'Injected Power=%.2f'%sigs[1],
            'Injected Torque=%.2f'%sigs[2],
            'Target Current=%.2f'%sigs[0]
        ))
        ax.text(0.05,0.05,textstr, transform=ax.transAxes)
        ax.legend(loc='lower right')
    plt.savefig(root_dir+'pic.png')

s="""
Entering interactive mode.
Feel free to play with val_data, val_target, train_data,
train_target, and the loaded_model. Use plot_timestep
with a given timestep 
"""
print(s)
import code
code.interact(local=locals())

