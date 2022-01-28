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
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras import backend
from helpers.custom_activations import inverse_activation
from helpers.custom_constraints import Orthonormal, SoftOrthonormal


class InverseDense(tf.keras.layers.Layer):
    """Inverse of a dense layer

    Initialized with the forward layer, this layer computes the
    inverse transformation, or the least squares approximation if
    the layer is not directly invertible

    Parameters
    ----------
    layer : keras layer
        forward dense layer to invert
    """

    def __init__(self, layer, **kwargs):

        self.kernel = layer.kernel
        self.bias = layer.bias if hasattr(layer, "bias") else 0
        self.activation = inverse_activation(layer.activation)
        super(InverseDense, self).__init__(**kwargs)

        self.kernel_initializer = layer.kernel_initializer
        self.bias_initializer = layer.bias_initializer
        self.kernel_regularizer = layer.kernel_regularizer
        self.bias_regularizer = layer.bias_regularizer
        self.kernel_constraint = layer.kernel_constraint
        self.bias_constraint = layer.bias_constraint

    def build(self, input_shape):
        assert (
            input_shape[1] == self.kernel.shape[1]
        ), "shapes don't match input shape={}, kernel shape={}".format(
            input_shape[1], self.kernel_shape[1]
        )
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    # TODO: get_config / from_config for serialization?

    def call(self, inputs):

        sy = self.activation(inputs)
        if self.bias is not None:
            sy = sy - self.bias
        if isinstance(self.kernel_constraint, (Orthonormal, SoftOrthonormal)):
            x = tf.linalg.matmul(self.kernel, sy, transpose_a=False, transpose_b=True)
        elif self.kernel.shape[0] == self.kernel.shape[1]:
            x = tf.linalg.solve(self.kernel, tf.transpose(sy), adjoint=True)
        else:
            x = tf.linalg.lstsq(tf.transpose(self.kernel), tf.transpose(sy), fast=False)
        return tf.transpose(x)


class InverseBatchNormalization(tf.keras.layers.Layer):
    """Inverse of BatchNormalization layer

    Initialized with a forward BatchNormalization layer, this layer undoes the
    normalization to recover the inputs

    Parameters
    ----------
    layer : keras layer
        forward BatchNormalization layer to invert
    """

    def __init__(self, layer, **kwargs):

        super(InverseBatchNormalization, self).__init__(**kwargs)

        self.moving_mean = layer.moving_mean
        self.moving_variance = layer.moving_variance
        self.epsilon = layer.epsilon
        self.gamma = layer.gamma
        self.beta = layer.beta
        self.axis = layer.axis

        self.beta_initializer = layer.beta_initializer
        self.gamma_initializer = layer.gamma_initializer
        self.beta_regularizer = layer.beta_regularizer
        self.gamma_regularizer = layer.gamma_regularizer
        self.beta_constraint = layer.beta_constraint
        self.gamma_constraint = layer.gamma_constraint

    def build(self, input_shape):
        if isinstance(input_shape, int):
            input_shape = (input_shape,)
        if input_shape[0] is None and len(input_shape) > 1:
            input_shape = input_shape[1:]
        assert (
            input_shape == self.moving_mean.shape
        ), "shapes don't match input shape={}, kernel shape={}".format(
            input_shape, self.moving_mean.shape
        )

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        # out = gamma * (inputs - mean) / sqrt(var + epsilon) + beta
        # inputs = (out - beta)/gamma * sqrt(var + epsilon) + mean

        # note - this will not working during training
        out = inputs
        if self.beta is not None:
            out = out - self.beta
        if self.gamma is not None:
            out = out / self.gamma
        out = out * tf.math.sqrt(self.moving_variance + self.epsilon) + self.moving_mean
        return out


class relativeSquaredError(tf.keras.layers.Layer):
    """Layer for computing relative squared error
    for the latent state

    """

    def __init__(self, lookahead, stepwise=False, name=""):
        super(relativeSquaredError, self).__init__(name=name)
        self.lookahead = lookahead
        self.stepwise = stepwise

    # Call with true value as first element, predicted
    # as second
    def call(self, inputs):
        pred_unstack = tf.unstack(inputs[1], num=self.lookahead, axis=1)
        true_unstack = tf.unstack(inputs[0], num=self.lookahead, axis=1)

        # compute relative error normalized by the true encoded state at
        # each time step
        if self.stepwise:
            rel_error = [
                (elem[0] - elem[1]) / (tf.norm(elem[1], axis=1, keepdims=True))
                for elem in zip(pred_unstack, true_unstack)
            ]
            return tf.stack(rel_error, axis=1)
        # scale everything by the first true encoded state and then subtract
        else:
            scaling_factor = tf.norm(true_unstack[0], axis=1, keepdims=True)
            rel_error = [
                (elem[0] - elem[1]) / scaling_factor
                for elem in zip(pred_unstack, true_unstack)
            ]
            return tf.stack(rel_error, axis=1)

    def get_config(self):
        config = super(relativeSquaredError, self).get_config()
        config.update({"stepwise_normalization": self.stepwise})
        return config


class ParametricLinearSystem(tf.keras.layers.Layer):
    """Layer representing a discrete time linear system

    Denoting state x, input u and output y, computes the following:
        x(t+1) = A x(t) + B u(t)
        y(t+1) = C x(t+1) + D u(t)
    with learnable matrices A,B,C,D
    Alternatively, matrices A,B,C,D can be provided at call,
    allowing them to be the output of another layer

    Parameters
    ----------
    state_size : int
        latent state dimension, len(x)
    output_size : int
        output dimension, len(y). If None, defaults to state_size
    learn_A, etc : bool
        whether to learn matrix within this layer, or assume
        it will be supplied elsewhere. Defaults to True for A,B
        False for C,D (in which case C is fixed to the identity and D to zero)
    A_initializer, etc : str or callable
        method to use to initialize matrix, if matrix is learned
    A_regularizer, etc : str or callable
        regularization applied to matrix, if matrix is learned
    A_constraint, etc : str or callable
        constraint applied to matrix, if matrix is learned
    return_sequences: bool (default `False`)
        whether to return the last output in the output sequence, or the full sequence.
    return_state: bool (default `False`)
        whether to return the last state in addition to the output.
    go_backwards: bool (default `False`)
        if True, process the input sequence backwards and return the
        reversed sequence.
    stateful: bool (default `False`)
        if True, the last state for each sample at index i in a batch
        will be used as initial state for the sample of index i in the
        following batch.
    """

    def __init__(
        self,
        state_size=1,
        output_size=None,
        learn_A=True,
        learn_B=True,
        learn_C=False,
        learn_D=False,
        A_initializer="orthogonal",
        B_initializer="glorot_uniform",
        C_initializer="glorot_uniform",
        D_initializer="zeros",
        A_regularizer=None,
        B_regularizer=None,
        C_regularizer=None,
        D_regularizer=None,
        A_constraint=None,
        B_constraint=None,
        C_constraint=None,
        D_constraint=None,
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        **kwargs,
    ):
        super(ParametricLinearSystem, self).__init__(**kwargs)

        self.state_size = state_size
        self.output_size = self.state_size if output_size is None else output_size
        self.learn_A = learn_A
        self.learn_B = learn_B
        self.learn_C = learn_C
        self.learn_D = learn_D
        self.A_initializer = A_initializer
        self.B_initializer = B_initializer
        self.C_initializer = C_initializer
        self.D_initializer = D_initializer
        self.A_regularizer = A_regularizer
        self.B_regularizer = B_regularizer
        self.C_regularizer = C_regularizer
        self.D_regularizer = D_regularizer
        self.A_constraint = A_constraint
        self.B_constraint = B_constraint
        self.C_constraint = C_constraint
        self.D_constraint = D_constraint
        self.stateful = stateful
        self.return_state = return_state
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards

        self._states = None
        self._batch_size = None

    def get_config(self):
        config = super(ParametricLinearSystem, self).get_config()
        config.update(
            {
                "state_size": self.state_size,
                "output_size": self.output_size,
                "learn_A": self.learn_A,
                "learn_B": self.learn_B,
                "learn_C": self.learn_C,
                "learn_D": self.learn_D,
                "A_initializer": self.A_initializer,
                "B_initializer": self.B_initializer,
                "C_initializer": self.C_initializer,
                "D_initializer": self.D_initializer,
                "A_regularizer": self.A_regularizer,
                "B_regularizer": self.B_regularizer,
                "C_regularizer": self.C_regularizer,
                "D_regularizer": self.D_regularizer,
                "A_constraint": self.A_constraint,
                "B_constraint": self.B_constraint,
                "C_constraint": self.C_constraint,
                "D_constraint": self.D_constraint,
                "return_sequences": self.return_sequences,
                "return_state": self.return_state,
                "go_backwards": self.go_backwards,
                "stateful": self.stateful,
            }
        )
        return config

    def build(self, input_shape):
        assert len(input_shape) == 3
        assert all([np.isscalar(i) or (i is None) for i in input_shape])
        batch, time_step, input_size = input_shape
        self._batch_size = batch
        if self.learn_A:
            self.A = self.add_weight(
                name="A",
                shape=(self.state_size, self.state_size),
                dtype=tf.float32,
                initializer=self.A_initializer,
                regularizer=self.A_regularizer,
                constraint=self.A_constraint,
            )
        else:
            self.A = tf.zeros(
                (self.state_size, self.state_size),
                dtype=tf.float32,
                name="A",
            )
        if self.learn_B:
            self.B = self.add_weight(
                name="B",
                shape=(self.state_size, input_size),
                dtype=tf.float32,
                initializer=self.A_initializer,
                regularizer=self.B_regularizer,
                constraint=self.B_constraint,
            )

        else:
            self.B = tf.zeros(
                (self.state_size, input_size),
                dtype=tf.float32,
                name="B",
            )
        if self.learn_C:
            self.C = self.add_weight(
                name="C",
                shape=(self.output_size, self.state_size),
                dtype=tf.float32,
                initializer=self.A_initializer,
                regularizer=self.C_regularizer,
                constraint=self.C_constraint,
            )
        else:
            self.C = tf.eye(
                self.output_size,
                self.state_size,
                dtype=tf.float32,
                name="C",
            )

        if self.learn_D:
            self.D = self.add_weight(
                name="D",
                shape=(self.output_size, input_size),
                dtype=tf.float32,
                initializer=self.A_initializer,
                regularizer=self.D_regularizer,
                constraint=self.D_constraint,
            )
        else:
            self.D = tf.zeros(
                (self.output_size, input_size),
                dtype=tf.float32,
                name="D",
            )

        if self.stateful:
            self.reset_states()
        self.built = True

    @property
    def states(self):
        return self._states

    @states.setter
    def states(self, states):
        self._states = states

    def compute_output_shape(self, input_shape):
        # TODO: simplify this, we know state and output dim at init, just need batch + time
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        input_shape = tf.TensorShape(input_shape)

        batch = input_shape[0]
        time_step = input_shape[1]
        state_size = self.state_size

        def _get_output_shape(flat_output_size):
            output_dim = tf.TensorShape(flat_output_size).as_list()
            if self.return_sequences:
                output_shape = tf.TensorShape([batch, time_step] + output_dim)
            else:
                output_shape = tf.TensorShape([batch] + output_dim)
            return output_shape

        output_shape = tf.nest.flatten(
            tf.nest.map_structure(_get_output_shape, self.output_size)
        )
        output_shape = output_shape[0] if len(output_shape) == 1 else output_shape

        if self.return_state:

            def _get_state_shape(flat_state):
                if self.return_sequences:
                    state_shape = tf.TensorShape(
                        [batch, time_step] + tf.TensorShape(flat_state).as_list()
                    )
                else:
                    state_shape = tf.TensorShape(
                        [batch] + tf.TensorShape(flat_state).as_list()
                    )
                return state_shape

            state_shape = tf.nest.map_structure(_get_state_shape, state_size)
            return generic_utils.to_list(output_shape) + tf.nest.flatten(state_shape)
        else:
            return output_shape

    def get_initial_state(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        dtype = inputs.dtype
        init_state = tf.zeros((batch_size, self.state_size), dtype)
        return init_state

    def call(self, inputs, x0=None, A=None, B=None, C=None, D=None):

        if A is None:
            A = self.A
        if B is None:
            B = self.B
        if C is None:
            C = self.C
        if D is None:
            D = self.D

        inputs, x0 = self._process_inputs(inputs, x0)

        input_shape = backend.int_shape(inputs)
        timesteps = input_shape[1]
        batch = input_shape[0]
        y0 = tf.zeros((batch, self.output_size))
        initial_state = [y0, x0]

        def step(inputs, states):
            u = inputs
            y0, x0 = states
            x1 = tf.linalg.matvec(A, x0) + tf.linalg.matvec(B, u)
            y1 = tf.linalg.matvec(C, x1) + tf.linalg.matvec(D, u)
            return y1, [y1, x1]

        last_output, outputs, states = backend.rnn(
            step,
            inputs,
            initial_state,
            constants=None,
            go_backwards=self.go_backwards,
            mask=None,
            unroll=False,
            input_length=timesteps,
            time_major=False,
            zero_output_for_mask=True,
        )

        # we don't care about y, thats already taken care of in outputs
        states = states[1]
        if self.stateful:
            updates = [
                tf.compat.v1.assign(self_state, tf.cast(state, self_state.dtype))
                for self_state, state in zip(
                    tf.nest.flatten(self.states), tf.nest.flatten(states)
                )
            ]
            self.add_update(updates)

        if self.return_sequences:
            output = outputs
        else:
            output = last_output

        if self.return_state:
            if not isinstance(states, (list, tuple)):
                states = [states]
            else:
                states = list(states)
            return generic_utils.to_list(output) + states
        else:
            return output

    def _process_inputs(self, inputs, initial_state):
        # input shape: `(samples, time (padded with zeros), input_dim)`
        if self.stateful:
            if initial_state is not None:
                # When layer is stateful and initial_state is provided, check if the
                # recorded state is same as the default value (zeros). Use the recorded
                # state if it is not same as the default.
                non_zero_count = tf.add_n(
                    [tf.math.count_nonzero(s) for s in tf.nest.flatten(self.states)]
                )
                # Set strict = True to keep the original structure of the state.
                initial_state = tf.compat.v1.cond(
                    non_zero_count > 0,
                    true_fn=lambda: self.states,
                    false_fn=lambda: initial_state,
                    strict=True,
                )
            else:
                initial_state = self.states
        elif initial_state is None:
            initial_state = self.get_initial_state(inputs)

        return inputs, initial_state

    def reset_states(self, states=None):
        """Reset the recorded states for the stateful RNN layer.
        Can only be used when RNN layer is constructed with `stateful` = `True`.

        Parameters
        ----------
        states : ndarray
            values for the initial state. When the value is None,
            zero filled numpy array will be created based on the cell state size.

        Raises
        ------
        AttributeError: When the RNN layer is not stateful.
        ValueError: When the batch size of the RNN layer is unknown.
        ValueError: When the input numpy array is not compatible with the RNN
            layer state, either size wise or dtype wise.
        """
        if not self.stateful:
            raise AttributeError("Layer must be stateful.")
        batch_size = self._batch_size
        if not batch_size:
            raise ValueError(
                "If a RNN is stateful, it needs to know "
                "its batch size. Specify the batch size "
                "of your input tensors: \n"
                "- If using a Sequential model, "
                "specify the batch size by passing "
                "a `batch_input_shape` "
                "argument to your first layer.\n"
                "- If using the functional API, specify "
                "the batch size by passing a "
                "`batch_shape` argument to your Input layer."
            )
        # initialize state if None
        if self.states is None:
            flat_init_state_values = tf.nest.flatten(
                self.get_initial_state(
                    inputs=tf.zeros(
                        [batch_size, 1, self.state_size], dtype=backend.floatx()
                    )
                )
            )
            flat_states_variables = tf.nest.map_structure(
                backend.variable, flat_init_state_values
            )
            self.states = tf.nest.pack_sequence_as(
                self.state_size, flat_states_variables
            )
        elif states is None:
            backend.set_value(
                self.states,
                np.zeros([batch_size] + tf.TensorShape(self.state_size).as_list()),
            )
        else:
            flat_states = tf.nest.flatten(self.states)
            flat_input_states = tf.nest.flatten(states)
            if len(flat_input_states) != len(flat_states):
                raise ValueError(
                    f"Layer {self.name} expects {len(flat_states)} "
                    f"states, but it received {len(flat_input_states)} "
                    f"state values. States received: {states}"
                )
            set_value_tuples = []
            for i, (value, state) in enumerate(zip(flat_input_states, flat_states)):
                if value.shape != state.shape:
                    raise ValueError(
                        f"State {i} is incompatible with layer {self.name}: "
                        f"expected shape={(batch_size, state)} "
                        f"but found shape={value.shape}"
                    )
                set_value_tuples.append((state, value))
            backend.batch_set_value(set_value_tuples)


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
        **kwargs,
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
            x = self.cat(inputs)
        else:
            x = inputs[0]
        if self.norm:
            x = self.norm_layer()
        for layer in self.layers:
            x = layer(x)

        return x


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
