import tensorflow as tf
import numpy as np

# DISCLAIMER:
# Parts of this code file are derived from
# https://github.com/tkipf/gae/


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random_uniform(
        shape, minval=-init_range, maxval=init_range, dtype=tf.float32
    )
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """Zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)
