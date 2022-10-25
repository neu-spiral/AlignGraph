import tensorflow as tf
import numpy as np

# DISCLAIMER:
# Boilerplate parts of this code file were originally forked from
# https://github.com/tkipf/gae/

# flags = tf.app.flags
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS


class OptimizerAE(object):
    def __init__(self, preds1, labels1, pos_weight, norm, adj1, preds0, labels0, adj0):
        preds_sub1 = preds1
        preds_sub0 = preds0
        labels_sub1 = labels1
        labels_sub0 = labels0

        m = num_nodes
        adj1 = placeholders["adjj1"]
        adj0 = placeholders["adjj0"]

        mod1_loss1 = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                labels=labels_sub1, logits=preds_sub1, pos_weight=pos_weight
            )
        )

        mod2_loss1 = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                labels=labels_sub0, logits=preds_sub0, pos_weight=pos_weight
            )
        )

        z1 = model1.outputs
        z2 = model0.outputs

        D = []
        for i in range(m):
            for j in range(m):
                D.append(tf.norm(z1[i] - z2[j]))
        D = tf.reshape(D, (m, m))
        mod12_loss = tf.linalg.trace(D)

        self.cost = 500 * (mod1_loss1 + mod2_loss1) + mod12_loss

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate
        )  # Adam Optimizer
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction1 = tf.equal(
            tf.cast(tf.greater_equal(tf.sigmoid(preds_sub1), 0.5), tf.int32),
            tf.cast(labels_sub1, tf.int32),
        )
        self.correct_prediction2 = tf.equal(
            tf.cast(tf.greater_equal(tf.sigmoid(preds_sub0), 0.5), tf.int32),
            tf.cast(labels_sub0, tf.int32),
        )
        self.accuracy = 0.5 * (
            tf.reduce_mean(tf.cast(self.correct_prediction1, tf.float32))
            + tf.reduce_mean(tf.cast(self.correct_prediction2, tf.float32))
        )
        self.accuracy1 = tf.reduce_mean(tf.cast(self.correct_prediction1, tf.float32))
        self.accuracy2 = tf.reduce_mean(tf.cast(self.correct_prediction2, tf.float32))


class OptimizerVAE(object):
    def __init__(
        self,
        placeholders,
        preds1,
        labels1,
        model1,
        num_nodes,
        pos_weight,
        norm,
        preds0,
        labels0,
        model0,
    ):
        preds_sub1 = preds1
        preds_sub0 = preds0
        labels_sub1 = labels1
        labels_sub0 = labels0

        m = num_nodes
        adj1 = placeholders["adjj1"]
        adj0 = placeholders["adjj0"]

        mod1_loss1 = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                labels=labels_sub1, logits=preds_sub1, pos_weight=pos_weight
            )
        )

        mod2_loss1 = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                labels=labels_sub0, logits=preds_sub0, pos_weight=pos_weight
            )
        )

        z1 = model1.outputs
        z2 = model0.outputs

        D = []
        for i in range(m):
            for j in range(m):
                D.append(tf.norm(z1[i] - z2[j]))
        D = tf.reshape(D, (m, m))
        mod12_loss = tf.linalg.trace(D)

        self.cost = 500 * (mod1_loss1 + mod2_loss1) + (1 / 120) * mod12_loss
        self.optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate
        )

        # Latent loss
        self.log_lik = self.cost
        self.kl = (1 / 2) * (
            (0.5 / num_nodes)
            * tf.reduce_mean(
                tf.reduce_sum(
                    1
                    + 2 * model1.z_log_std
                    - tf.square(model1.z_mean)
                    - tf.square(tf.exp(model1.z_log_std)),
                    1,
                )
            )
            + (0.5 / num_nodes)
            * tf.reduce_mean(
                tf.reduce_sum(
                    1
                    + 2 * model0.z_log_std
                    - tf.square(model0.z_mean)
                    - tf.square(tf.exp(model0.z_log_std)),
                    1,
                )
            )
        )
        self.cost -= self.kl
        # Adam Optimizer
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction1 = tf.equal(
            tf.cast(
                tf.greater_equal(tf.sigmoid(preds_sub1), 1 / (1 + np.exp(-0.0))),
                tf.int32,
            ),
            tf.cast(labels_sub1, tf.int32),
        )
        self.correct_prediction2 = tf.equal(
            tf.cast(
                tf.greater_equal(tf.sigmoid(preds_sub0), 1 / (1 + np.exp(-0.0))),
                tf.int32,
            ),
            tf.cast(labels_sub0, tf.int32),
        )

        self.accuracy = 0.5 * (
            tf.reduce_mean(tf.cast(self.correct_prediction1, tf.float32))
            + tf.reduce_mean(tf.cast(self.correct_prediction2, tf.float32))
        )

        self.accuracy1 = tf.reduce_mean(tf.cast(self.correct_prediction1, tf.float32))
        self.accuracy2 = tf.reduce_mean(tf.cast(self.correct_prediction2, tf.float32))
