from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from inits import zeros

# flags = tf.app.flags
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

# DISCLAIMER:
# Boilerplate parts of this code file were originally forked from
# https://github.com/tkipf/gae
# which itself was very inspired by the keras package

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=""):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def weight_variable_glorot(input_dim, output_dim, name=""):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    output_dim = int(output_dim)
    input_dim = int(input_dim)
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    print("init_range for weights:", init_range)
    initial = tf.compat.v1.random_uniform(
        [input_dim, output_dim], minval=-init_range, maxval=init_range, dtype=tf.float32
    )
    return tf.Variable(initial, name=name)


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)"""
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.compat.v1.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.compat.v1.sparse_retain(x, dropout_mask)
    return pre_out * (1.0 / keep_prob)


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).
    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {"name", "logging", "model_size"}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, "Invalid keyword argument: " + kwarg
        name = kwargs.get("name")
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + "_" + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get("logging", False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):

        with tf.name_scope(self.name):
            outputs = self._call(inputs)
        return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + "/vars/" + var, self.vars[var])


class GraphConvolution(Layer):
    """Dense layer."""

    def __init__(
        self,
        placeholders,
        input_dim,
        output_dim,
        adj,
        features_nonzero,
        act=tf.nn.relu,
        dropout=0.0,
        bias=True,
        featureless=False,
        sparse_inputs=False,
        **kwargs
    ):
        super(GraphConvolution, self).__init__(**kwargs)

        self.dropout = dropout
        self.adj = adj
        self.act = act
        # self.features=features
        self.featureless = featureless
        self.bias = bias
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.placeholders = placeholders
        self.features_nonzero = features_nonzero
        print("input_dim, output_dim", input_dim, output_dim)
        with tf.compat.v1.variable_scope(self.name + "_vars"):
            self.vars["weights"] = weight_variable_glorot(
                input_dim, output_dim, name="weights"
            )
            if self.bias:
                self.vars["bias"] = zeros([output_dim], name="bias")

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        x = tf.cast(x, tf.float32)
        x = tf.nn.dropout(x, 1 - self.dropout)
        print("x", tf.shape(x), x.get_shape(), x)
        print(
            "self.vars['weights']",
            tf.shape(self.vars["weights"]),
            tf.cast(self.vars["weights"], tf.float32).get_shape(),
        )
        # transform
        # import pdb
        # pdb.set_trace()
        x = tf.matmul(x, tf.cast(self.vars["weights"], tf.float32))
        x = tf.compat.v1.sparse_tensor_dense_matmul(tf.cast(self.adj, tf.float32), x)
        # bias
        if self.bias:
            x += self.vars["bias"]
        outputs = self.act(x)
        return outputs


class GraphConvolutionSparse(Layer):
    """Graph convolution layer for sparse inputs."""

    def __init__(
        self,
        placeholders,
        input_dim,
        output_dim,
        adj,
        features_nonzero,
        dropout=0.0,
        act=tf.nn.relu,
        bias=True,
        **kwargs
    ):
        super(GraphConvolutionSparse, self).__init__(**kwargs)
        self.bias = bias
        with tf.compat.v1.variable_scope(self.name + "_vars"):
            self.vars["weights"] = weight_variable_glorot(
                input_dim, output_dim, name="weights"
            )
            if self.bias:
                self.vars["bias"] = zeros([output_dim], name="bias")
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero
        # self.bias=bias

    def _call(self, inputs):
        x = inputs
        x = dropout_sparse(x, 1 - self.dropout, self.features_nonzero)
        x = tf.compat.v1.sparse_tensor_dense_matmul(x, self.vars["weights"])
        x = tf.compat.v1.sparse_tensor_dense_matmul(self.adj, x)
        if self.bias:
            x += self.vars["bias"]
        outputs = self.act(x)
        return outputs


class InnerProductDecoder(Layer):
    """Decoder model layer for link prediction."""

    def __init__(self, input_dim, dropout=1, act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.act = act
        # self.bias = bias
        self.eps = 1e-7
        self.dropout = dropout
        self.act = act
        # self.P = P

    def _call(self, inputs):

        inputs = tf.compat.v1.cast(tf.nn.dropout(inputs, 1 - self.dropout), tf.float32)
        x = tf.transpose(inputs)
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
        return outputs
