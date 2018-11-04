"""
Hardware/device configuration
"""

# System
import os

# Externals
import keras
import tensorflow as tf

# Default settings are for Haswell
def configure_session(intra_threads=16, inter_threads=2,
                      blocktime=1, affinity='granularity=fine,compact,1,0'):
    """Sets the thread knobs in the TF backend"""
    os.environ['KMP_BLOCKTIME'] = str(blocktime)
    os.environ['KMP_AFFINITY'] = affinity
    os.environ['OMP_NUM_THREADS'] = str(intra_threads)
    config = tf.ConfigProto(
        inter_op_parallelism_threads=inter_threads,
        intra_op_parallelism_threads=intra_threads
    )
    keras.backend.set_session(tf.Session(config=config))