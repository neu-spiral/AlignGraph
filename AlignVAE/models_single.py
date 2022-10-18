from collections import namedtuple
import tensorflow.compat.v1 as tf
import math
import layers as layers

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

# DISCLAIMER:
# Boilerplate parts of this code file were originally forked from
# https://github.com/tkipf/gae/
# which itself was very inspired by the keras package


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {"name", "logging", "model_size"}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, "Invalid keyword argument: " + kwarg
        name = kwargs.get("name")
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get("logging", False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """Wrapper for _build()"""
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class GCNModelAE(Model):
    """A standard multi-layer perceptron"""

    def __init__(self, placeholders, dims, features_nonzero, **kwargs):
        super(GCNModelAE, self).__init__(**kwargs)

        self.dims = dims
        self.input_dim = dims[0]
        self.output_dim = dims[-1]
        self.placeholders = placeholders

        self.inputs = placeholders["features"]
        self.adj = placeholders["adj"]
        self.features_nonzero = features_nonzero
        self.P = placeholders["P"]

        self.build()

    def _build(self):
        self.hidden1 = layers.GraphConvolutionSparse(
            placeholders=self.placeholders,
            input_dim=self.input_dim,
            output_dim=16 * self.dims[1],
            adj=self.adj,
            features_nonzero=self.features_nonzero,
            act=tf.nn.relu,
            dropout=self.placeholders["dropout"],
            logging=self.logging,
        )(self.inputs)

        self.outputs = layers.GraphConvolution(
            placeholders=self.placeholders,
            input_dim=16 * self.dims[1],
            output_dim=8 * self.output_dim,
            adj=self.adj,
            features_nonzero=self.features_nonzero,
            act=tf.nn.relu,
            dropout=self.placeholders["dropout"],
            logging=self.logging,
        )(self.hidden1)

        self.outputs = layers.GraphConvolution(
            placeholders=self.placeholders,
            input_dim=8 * self.output_dim,
            output_dim=8 * self.output_dim / 2,
            adj=self.adj,
            features_nonzero=self.features_nonzero,
            act=tf.nn.relu,
            dropout=self.placeholders["dropout"],
            logging=self.logging,
        )(self.outputs)

        self.outputs = layers.GraphConvolution(
            placeholders=self.placeholders,
            input_dim=8 * self.output_dim / 2,
            output_dim=8 * self.output_dim / 4,
            adj=self.adj,
            features_nonzero=self.features_nonzero,
            act=tf.nn.relu,
            dropout=self.placeholders["dropout"],
            logging=self.logging,
        )(self.outputs)

        self.outputs = layers.GraphConvolution(
            placeholders=self.placeholders,
            input_dim=8 * self.output_dim / 4,
            output_dim=8 * self.output_dim / 8,
            adj=self.adj,
            features_nonzero=self.features_nonzero,
            act=lambda x: x,
            dropout=self.placeholders["dropout"],
            logging=self.logging,
        )(self.outputs)

        self.outputs = tf.nn.l2_normalize(self.outputs, 1)

        self.reconstructions = layers.InnerProductDecoder(
            input_dim=self.output_dim, P=self.P, act=lambda x: x, logging=self.logging
        )(self.outputs)

    def build(self):
        self._build()

        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}
        # self.outputs = tf.nn.l2_normalize(self.outputs, 1)

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCNModelVAE(Model):
    def __init__(self, placeholders, dims, num_nodes, features_nonzero, **kwargs):
        super(GCNModelVAE, self).__init__(**kwargs)

        self.dims = dims
        self.input_dim = dims[0]
        self.output_dim = dims[-1]
        self.placeholders = placeholders
        self.inputs = placeholders["features"]
        self.adj = placeholders["adj"]
        self.P = placeholders["P"]
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.build()

    def _build(self):
        self.hidden1 = layers.GraphConvolutionSparse(
            placeholders=self.placeholders,
            input_dim=self.input_dim,
            output_dim=16 * self.dims[1],
            adj=self.adj,
            features_nonzero=self.features_nonzero,
            act=tf.nn.relu,
            dropout=self.placeholders["dropout"],
            logging=self.logging,
        )(self.inputs)

        self.hidden1 = layers.GraphConvolution(
            placeholders=self.placeholders,
            input_dim=16 * self.dims[1],
            output_dim=8 * self.dims[1],
            adj=self.adj,
            features_nonzero=self.features_nonzero,
            act=tf.nn.relu,
            dropout=self.placeholders["dropout"],
            logging=self.logging,
        )(self.hidden1)

        self.hidden1 = layers.GraphConvolution(
            placeholders=self.placeholders,
            input_dim=8 * self.dims[1],
            output_dim=4 * self.dims[1],
            adj=self.adj,
            features_nonzero=self.features_nonzero,
            act=tf.nn.relu,
            dropout=self.placeholders["dropout"],
            logging=self.logging,
        )(self.hidden1)

        self.hidden1 = layers.GraphConvolution(
            placeholders=self.placeholders,
            input_dim=4 * self.dims[1],
            output_dim=2 * self.dims[1],
            adj=self.adj,
            features_nonzero=self.features_nonzero,
            act=tf.nn.relu,
            dropout=self.placeholders["dropout"],
            logging=self.logging,
        )(self.hidden1)

        self.z_mean = layers.GraphConvolution(
            placeholders=self.placeholders,
            input_dim=2 * self.dims[1],
            output_dim=self.dims[1],
            adj=self.adj,
            features_nonzero=self.features_nonzero,
            act=lambda x: x,
            dropout=self.placeholders["dropout"],
            logging=self.logging,
        )(self.hidden1)

        self.z_log_std = layers.GraphConvolution(
            placeholders=self.placeholders,
            input_dim=2 * self.dims[1],
            output_dim=self.dims[1],
            adj=self.adj,
            features_nonzero=self.features_nonzero,
            act=lambda x: x,
            dropout=self.placeholders["dropout"],
            logging=self.logging,
        )(self.hidden1)

        self.z = self.z_mean + tf.random_normal(
            [self.n_samples, self.dims[1]]
        ) * tf.exp(self.z_log_std)

        # self.outputs =self.z

        self.outputs = self.z = tf.nn.l2_normalize(self.z, 1)

        self.reconstructions = layers.InnerProductDecoder(
            input_dim=self.output_dim,
            P=self.P,
            act=lambda x: x,
            logging=self.logging,
        )(self.z)

        self.new_adj = self.reconstructions

    def build(self):
        self._build()

        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}
