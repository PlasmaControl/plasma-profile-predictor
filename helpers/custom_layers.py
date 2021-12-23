import numpy as np
import copy
import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import nest
from tensorflow.python.keras.layers.wrappers import Wrapper
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import activations


class ParaMatrix(tf.keras.layers.Layer):
    """Layer that returns a matrix parameterized by inputs.

    Parameters
    ----------
    matrix_shape : tuple
        shape of desired output
    num_layers : int
        number of transformations to apply
    norm : bool
        whether to apply batchnorm to inputs
    activation : str or callable
        activation function to apply after each transformation
    use_bias : bool
        whether to use bias
    """

    def __init__(
        self,
        matrix_shape,
        num_layers=1,
        norm=True,
        activation=None,
        use_bias=True,
        **kwargs
    ):

        assert len(matrix_shape) == 2
        self.matrix_shape = matrix_shape
        self.matrix_size = np.prod(matrix_shape)

        super(ParaMatrix, self).__init__(**kwargs)

        self.num_layers = num_layers
        self.norm = norm
        self.activation = activation
        self.use_bias = use_bias

        self.layers = []
        if self.norm:
            self.norm_layer = tf.keras.layers.BatchNormalization(
                center=False, scale=False, name="norm_" + self.name
            )
        self.cat = tf.keras.layers.Concatenate(axis=-1)
        self.flatten = tf.keras.layers.Flatten()
        for i in range(num_layers):
            a = self.activation
            if i == num_layers - 1:
                a = None
            if i == 0:
                units = self.matrix_size
            else:
                units = self.matrix_shape[1]
            self.layers.append(
                tf.keras.layers.Dense(units, activation=a, use_bias=self.use_bias)
            )
            if i == 0:
                self.layers.append(tf.keras.layers.Reshape(matrix_shape))

    def get_config(self):
        config = super(ParaMatrix, self).get_config()
        config.update(
            {
                "matrix_shape": self.matrix_shape,
                "num_layers": self.num_layers,
                "norm": self.norm,
                "activation": self.activation,
                "use_bias": self.use_bias,
            }
        )
        return config

    def call(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        inputs = [self.flatten(inp) for inp in inputs]
        if len(inputs) > 1:
            inputs = self.cat(inputs)
        else:
            inputs = inputs[0]
        if self.norm:
            inputs = self.norm_layer(inputs)
        for layer in self.layers:
            inputs = layer(inputs)

        return inputs


# copied from tf v2.6.0 to get support for multiple inputs/outputs of time distributed layer
class MultiTimeDistributed(Wrapper):
    """This wrapper allows to apply a layer to every temporal slice of an input.
    Every input should be at least 3D, and the dimension of index one of the
    first input will be considered to be the temporal dimension.
    Consider a batch of 32 video samples, where each sample is a 128x128 RGB image
    with `channels_last` data format, across 10 timesteps.
    The batch input shape is `(32, 10, 128, 128, 3)`.
    You can then use `TimeDistributed` to apply the same `Conv2D` layer to each
    of the 10 timesteps, independently:
    >>> inputs = tf.keras.Input(shape=(10, 128, 128, 3))
    >>> conv_2d_layer = tf.keras.layers.Conv2D(64, (3, 3))
    >>> outputs = tf.keras.layers.TimeDistributed(conv_2d_layer)(inputs)
    >>> outputs.shape
    TensorShape([None, 10, 126, 126, 64])
    Because `TimeDistributed` applies the same instance of `Conv2D` to each of the
    timestamps, the same set of weights are used at each timestamp.
    Args:
      layer: a `tf.keras.layers.Layer` instance.
    Call arguments:
      inputs: Input tensor of shape (batch, time, ...) or nested tensors,
        and each of which has shape (batch, time, ...).
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. This argument is passed to the
        wrapped layer (only if the layer supports this argument).
      mask: Binary tensor of shape `(samples, timesteps)` indicating whether
        a given timestep should be masked. This argument is passed to the
        wrapped layer (only if the layer supports this argument).
    Raises:
      ValueError: If not initialized with a `tf.keras.layers.Layer` instance.
    """

    def __init__(self, layer, **kwargs):
        if not isinstance(layer, Layer):
            raise ValueError(
                "Please initialize `TimeDistributed` layer with a "
                "`tf.keras.layers.Layer` instance. You passed: {input}".format(
                    input=layer
                )
            )
        super(MultiTimeDistributed, self).__init__(layer, **kwargs)
        self.supports_masking = True

        # It is safe to use the fast, reshape-based approach with all of our
        # built-in Layers.
        self._always_use_reshape = layer_utils.is_builtin_layer(layer) and not getattr(
            layer, "stateful", False
        )

    def _get_shape_tuple(self, init_tuple, tensor, start_idx, int_shape=None):
        """Finds non-specific dimensions in the static shapes.
        The static shapes are replaced with the corresponding dynamic shapes of the
        tensor.
        Args:
          init_tuple: a tuple, the first part of the output shape
          tensor: the tensor from which to get the (static and dynamic) shapes
            as the last part of the output shape
          start_idx: int, which indicate the first dimension to take from
            the static shape of the tensor
          int_shape: an alternative static shape to take as the last part
            of the output shape
        Returns:
          The new int_shape with the first part from init_tuple
          and the last part from either `int_shape` (if provided)
          or `tensor.shape`, where every `None` is replaced by
          the corresponding dimension from `tf.shape(tensor)`.
        """
        # replace all None in int_shape by backend.shape
        if int_shape is None:
            int_shape = backend.int_shape(tensor)[start_idx:]
        if isinstance(int_shape, tensor_shape.TensorShape):
            int_shape = int_shape.as_list()
        if not any(not s for s in int_shape):
            return init_tuple + tuple(int_shape)
        shape = backend.shape(tensor)
        int_shape = list(int_shape)
        for i, s in enumerate(int_shape):
            if not s:
                int_shape[i] = shape[start_idx + i]
        return init_tuple + tuple(int_shape)

    def _remove_timesteps(self, dims):
        dims = dims.as_list()
        return tensor_shape.TensorShape([dims[0]] + dims[2:])

    def build(self, input_shape):
        input_shape = tf_utils.convert_shapes(input_shape, to_tuples=False)
        input_dims = nest.flatten(nest.map_structure(lambda x: x.ndims, input_shape))
        if any(dim < 3 for dim in input_dims):
            raise ValueError(
                "`MultiTimeDistributed` Layer should be passed an `input_shape ` "
                "with at least 3 dimensions, received: " + str(input_shape)
            )
        # Don't enforce the batch or time dimension.
        self.input_spec = nest.map_structure(
            lambda x: InputSpec(shape=[None, None] + x.as_list()[2:]), input_shape
        )
        child_input_shape = nest.map_structure(self._remove_timesteps, input_shape)
        child_input_shape = tf_utils.convert_shapes(child_input_shape)
        super(MultiTimeDistributed, self).build(tuple(child_input_shape))
        self.built = True

    def compute_output_shape(self, input_shape):
        input_shape = tf_utils.convert_shapes(input_shape, to_tuples=False)

        child_input_shape = nest.map_structure(self._remove_timesteps, input_shape)
        child_output_shape = self.layer.compute_output_shape(child_input_shape)
        child_output_shape = tf_utils.convert_shapes(
            child_output_shape, to_tuples=False
        )
        timesteps = tf_utils.convert_shapes(input_shape)
        timesteps = nest.flatten(timesteps)[1]

        def insert_timesteps(dims):
            dims = dims.as_list()
            return tensor_shape.TensorShape([dims[0], timesteps] + dims[1:])

        return nest.map_structure(insert_timesteps, child_output_shape)

    def call(self, inputs, training=None, mask=None):
        kwargs = {}
        if generic_utils.has_arg(self.layer.call, "training"):
            kwargs["training"] = training

        input_shape = nest.map_structure(
            lambda x: tensor_shape.TensorShape(backend.int_shape(x)), inputs
        )
        batch_size = tf_utils.convert_shapes(input_shape)
        batch_size = nest.flatten(batch_size)[0]
        if batch_size and not self._always_use_reshape:
            inputs, row_lengths = backend.convert_inputs_if_ragged(inputs)
            is_ragged_input = row_lengths is not None
            input_length = tf_utils.convert_shapes(input_shape)
            input_length = nest.flatten(input_length)[1]

            # batch size matters, use rnn-based implementation
            def step(x, _):
                output = self.layer(x, **kwargs)
                return output, []

            _, outputs, _ = backend.rnn(
                step,
                inputs,
                initial_states=[],
                input_length=row_lengths[0] if is_ragged_input else input_length,
                mask=mask,
                unroll=False,
            )
            # pylint: disable=g-long-lambda
            y = nest.map_structure(
                lambda output: backend.maybe_convert_to_ragged(
                    is_ragged_input, output, row_lengths
                ),
                outputs,
            )
        else:
            # No batch size specified, therefore the layer will be able
            # to process batches of any size.
            # We can go with reshape-based implementation for performance.
            is_ragged_input = nest.map_structure(
                lambda x: isinstance(x, ragged_tensor.RaggedTensor), inputs
            )
            is_ragged_input = nest.flatten(is_ragged_input)
            if all(is_ragged_input):
                input_values = nest.map_structure(lambda x: x.values, inputs)
                input_row_lenghts = nest.map_structure(
                    lambda x: x.nested_row_lengths()[0], inputs
                )
                y = self.layer(input_values, **kwargs)
                y = nest.map_structure(
                    ragged_tensor.RaggedTensor.from_row_lengths, y, input_row_lenghts
                )
            elif any(is_ragged_input):
                raise ValueError(
                    "All inputs has to be either ragged or not, "
                    "but not mixed. You passed: {}".format(inputs)
                )
            else:
                input_length = tf_utils.convert_shapes(input_shape)
                input_length = nest.flatten(input_length)[1]
                if not input_length:
                    input_length = nest.map_structure(
                        lambda x: array_ops.shape(x)[1], inputs
                    )
                    input_length = generic_utils.to_list(nest.flatten(input_length))[0]

                inner_input_shape = nest.map_structure(
                    lambda x: self._get_shape_tuple((-1,), x, 2), inputs
                )
                # Shape: (num_samples * timesteps, ...). And track the
                # transformation in self._input_map.
                inputs = nest.map_structure_up_to(
                    inputs, array_ops.reshape, inputs, inner_input_shape
                )
                # (num_samples * timesteps, ...)
                if generic_utils.has_arg(self.layer.call, "mask") and mask is not None:
                    inner_mask_shape = self._get_shape_tuple((-1,), mask, 2)
                    kwargs["mask"] = backend.reshape(mask, inner_mask_shape)

                y = self.layer(inputs, **kwargs)

                # Shape: (num_samples, timesteps, ...)
                output_shape = self.compute_output_shape(input_shape)
                # pylint: disable=g-long-lambda
                output_shape = nest.map_structure(
                    lambda tensor, int_shape: self._get_shape_tuple(
                        (-1, input_length), tensor, 1, int_shape[2:]
                    ),
                    y,
                    output_shape,
                )
                y = nest.map_structure_up_to(y, array_ops.reshape, y, output_shape)
                if not context.executing_eagerly():
                    # Set the static shape for the result since it might be lost during
                    # array_ops reshape, eg, some `None` dim in the result could be
                    # inferred.
                    nest.map_structure_up_to(
                        y,
                        lambda tensor, shape: tensor.set_shape(shape),
                        y,
                        self.compute_output_shape(input_shape),
                    )

        return y
