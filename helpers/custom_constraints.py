import tensorflow as tf


class Orthonormal(tf.keras.constraints.Constraint):
    """Constrains weight tensors to be orthonormal matrices."""

    def __call__(self, w):
        s, u, v = tf.linalg.svd(w, full_matrices=False, compute_uv=True)
        w = tf.linalg.matmul(u, v, transpose_b=True)
        return w

    def get_config(self):
        return {}
