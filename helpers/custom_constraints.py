import tensorflow as tf


class SoftOrthonormal(tf.keras.constraints.Constraint):
    """Soft constrains weight tensors to be orthonormal matrices.

    Should be about 4x faster than regular Orthonormal constraint, at the expense
    of not being exactly orthonormal, but gets closer during training.
    """

    def __init__(self, step_size=None):
        self.step_size = step_size

    def __call__(self, w):
        if self.step_size is None:
            h = 1 / w.shape[0]
        else:
            h = self.step_size
        w = w - h * w @ (tf.matmul(w, w, transpose_a=True) - tf.eye(w.shape[1]))
        return w

    def get_config(self):
        return {"step_size": self.step_size}


class Orthonormal(tf.keras.constraints.Constraint):
    """Constrains weight tensors to be orthonormal matrices.

    Can be somewhat slow, as it uses the full QR of the weight matrix.
    """

    def __call__(self, w):
        m, n = w.shape
        if m >= n:
            q, r = tf.linalg.qr(w)
            return q
        else:
            q, r = tf.linalg.qr(tf.transpose(w))
            return tf.transpose(q)

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
