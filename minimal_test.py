# first, module load tensorflow/intel-1.11.0-py36

import yaml

import os
from data import get_datasets
from models import get_model

import keras
from utils.optimizers import get_optimizer
from utils.callbacks import TimingCallback

import numpy as np

output_dir=os.path.join(os.getenv("HOME"),'plasma-profile-predictor')

output_file_name='test_model_small.h5' 
model_name='trend_plus_actuators_joe.yaml'

def load_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f)
    return config

config=load_config('configs/'+model_name)
train_config = config['training']

#if (type(config['data']['n_components']) is int):
#    rho_length_in = config['data']['n_components']
#else:
rho_length_in = config['model']['rho_length_out']

model = get_model(rho_length_in=rho_length_in, 
                  **config['data_and_model'],
                  **config['model'])

rank=0
n_ranks=1

# Configure optimizer
opt = get_optimizer(n_ranks=n_ranks, distributed=False,
                    **config['optimizer'])

# Compile the model
model.compile(loss=train_config['loss'], optimizer=opt,
              metrics=train_config['metrics'])
train_gen, valid_gen = get_datasets(batch_size=train_config['batch_size'],
                                    **config['data_and_model'],**config['data'])

steps_per_epoch = len(train_gen) // n_ranks

# Timing
callbacks = []
timing_callback = TimingCallback()
callbacks.append(timing_callback)
callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss',patience=5))

callbacks.append(keras.callbacks.ModelCheckpoint(filepath=os.path.join(output_dir, output_file_name),
                                                 monitor='val_mean_absolute_error',
                                                 save_best_only=True,
                                                 verbose=1))

history = model.fit_generator(train_gen,
                              epochs=train_config['n_epochs'],
                              steps_per_epoch=steps_per_epoch,
                              validation_data=valid_gen,
                              validation_steps=len(valid_gen),
                              callbacks=callbacks,
                              workers=4, verbose=1)

print('Mean time per epoch: {}s'.format(np.mean(timing_callback.times)))
