import keras
import numpy as np
import tensorflow as tf
from keras import backend as K

class groupLasso(keras.regularizers.Regularizer):
    
    def __init__(self, strength = 0., units=165):
        self.strength = strength['value']
        self.units = units

    def __call__(self, x):
        arr = tf.norm(x, ord=2, axis=0)
        zero = np.zeros(shape = (self.units), dtype=np.float32)
        arr = tf.where(arr<self.strength, zero, arr)
        assert K.ndim(arr) == 1
        return K.sum(arr)
        

    def get_config(self):
        return {'strength':float(self.strength)}
