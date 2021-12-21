import tensorflow as tf


class Orthonormal(tf.keras.constraints.Constraint):
    """Constrains weight tensors to be orthonormal matrices."""

    def __call__(self, w):
        s, u, v = tf.linalg.svd(w, full_matrices=False, compute_uv=True)
        w = tf.linalg.matmul(u, v, adjoint_b=True)
        return w

    def get_config(self):
        return {}


class Invertible(tf.keras.constraints.Constraint):
    """Constrains weight tensors to be invertible matrices by bounding singular values."""

    def __init__(self, sigma_min=0.5, sigma_max=2):

        self._sigma_min = sigma_min
        self._sigma_max = sigma_max

    def __call__(self, w):
        s, u, v = tf.linalg.svd(w, full_matrices=False, compute_uv=True)
        s = tf.clip_by_value(s, self._sigma_min, self._sigma_min)
        w = tf.matmul(u, tf.matmul(tf.linalg.diag(s), v, adjoint_b=True))

        return w

    def get_config(self):
        return {"sigma_min": self._sigma_min, "sigma_max": self._sigma_max}
