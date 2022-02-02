from tensorflow import keras
from . import callbacks
from . import custom_activations
from . import custom_constraints
from . import custom_layers
from . import custom_losses
from . import data_generator
from . import schedulers
from . import signal_groups
from . import hyperparam_helpers

custom_objects = {
    "MultiTimeDistributed": custom_layers.MultiTimeDistributed,
    "InverseDense": custom_layers.InverseDense,
    "InverseBatchNormalization": custom_layers.InverseBatchNormalization,
    "RelativeError": custom_layers.RelativeError,
    "ParametricLinearSystem": custom_layers.ParametricLinearSystem,
    "ParaMatrix": custom_layers.ParaMatrix,
    "Orthonormal": custom_constraints.Orthonormal,
    "SoftOrthonormal": custom_constraints.SoftOrthonormal,
    "Invertible": custom_constraints.Invertible,
    "LeakyReLU": keras.layers.LeakyReLU,  # for some reason keras doesn't recognize this, so need to include here
    "InverseLeakyReLU": custom_activations.InverseLeakyReLU,
    "inverse_selu": custom_activations.inverse_selu,
}
