import tensorflow as tf
import keras
from keras import backend as K
import numpy as np

class downsample(keras.initializers.Initializer):
    
    def __init__(self, gain=1., k = 0):
        self.gain = gain
        self.k = k
        

    def __call__(self, shape, dtype = None):
        return self.gain * np.eye(shape[0], shape[1], self.k)

    def get_config(self):
        return{
            'gain' : self.gain,
            'k':self.k
            }

#TODO: implement diagonally dominant initializer
'''
class diagdom(keras.initializers.Initializer):
   
    def __init__ (self, gain=1., k=0, mean, std):
        self. 
'''
