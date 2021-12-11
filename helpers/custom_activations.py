import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils


class InverseLeakyReLU(Layer):
    """Inverse of Leaky version of a Rectified Linear Unit.

    ie, undoes the nonlinear activation so that
        InverseLeakyReLU(LeakyRelu(x)) == x

    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the batch axis)
      when using this layer as the first layer in a model.
    Output shape:
      Same shape as the input.
    Args:
      alpha: Float >= 0. Negative slope coefficient. Default to 0.3.
    """

    def __init__(self, alpha=0.3, **kwargs):
        super(InverseLeakyReLU, self).__init__(**kwargs)
        if alpha is None:
            raise ValueError(
                "The alpha value of an InverseLeakyReLU layer "
                "cannot be None, needs a float. "
                "Got %s" % alpha
            )
        self.supports_masking = True
        self.alpha = backend.cast_to_floatx(alpha)

    def call(self, inputs):
        return tf.where(inputs > 0, inputs, inputs / self.alpha)

    def get_config(self):
        config = {"alpha": float(self.alpha)}
        base_config = super(InverseLeakyReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape
