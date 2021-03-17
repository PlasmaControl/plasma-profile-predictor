import pickle
import keras
import numpy as np
import random
import os
import sys
import itertools
import copy
from collections import OrderedDict
from time import strftime, localtime

from helpers.data_generator import process_data, AutoEncoderDataGenerator
from helpers.hyperparam_helpers import make_bash_scripts
from helpers.custom_losses import denorm_loss, hinge_mse_loss, percent_baseline_error, baseline_MAE
from helpers.custom_losses import percent_correct_sign, baseline_MAE
from helpers.results_processing import write_autoencoder_results
import models.autoencoder
from helpers.callbacks import CyclicLR, TensorBoardWrapper, TimingCallback
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
from keras import backend as K