from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import pickle
import argparse
import cvxpy
import cvxpy as cp
import networkx as nx
import scipy
import pandas as pd
import scipy.sparse as sp
import multiprocessing as mp
from scipy import sparse
import multiprocessing
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib
import math

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import layers as layers
from models_single import GCNModelAE, GCNModelVAE
from Graph_processor import (
    Graph_processor,
    preprocess_graph,
    sparse_to_tuple,
)

from validation import validation_score
from optimizer_single import OptimizerAE, OptimizerVAE
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from numpy import linalg as LA
from scipy.optimize import linear_sum_assignment

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# DISCLAIMER:
# Boilerplate parts of this code file were originally forked from
# https://github.com/tkipf/gae/

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

tf.app.flags.DEFINE_boolean(
    "log_device_placement", False, """Whether to log device placement."""
)
# core params..
flags.DEFINE_string("model", "GCNModelVAE", "model names.")
flags.DEFINE_string("data", "graphs", "graph directory.")
flags.DEFINE_float("learning_rate", 0.000002, "initial learning rate.")
flags.DEFINE_string("model_size", "small", "Can be big or small; model specific def'ns")
flags.DEFINE_string(
    "train_prefix",
    "",
    "name of the object file that stores the training data. must be specified.",
)
flags.DEFINE_string("feat", "TRUE", "Whether using one-hot or given features.")
flags.DEFINE_string(
    "variational", "True", "Whether we have the autoencoder is variational or not."
)
flags.DEFINE_string("training", "True", "Whether we are trainig or testing.")
flags.DEFINE_string(
    "distance_type",
    "G_align",
    "The distance function to compute alignment. It can be either Fermat or G-align.",
)
flags.DEFINE_integer("epochs", 10, "number of epochs to train.")
flags.DEFINE_float("dropout", 1.0, "dropout rate (1 - keep probability).")
flags.DEFINE_integer(
    "dim_1", 16, "Size of output dim (final is 2x this, if using concat)"
)
flags.DEFINE_integer(
    "dim_2", 16, "Size of output dim (final is 2x this, if using concat)"
)
flags.DEFINE_integer("id_gpu", 0, "gpu id")
flags.DEFINE_integer("num_batch", 2, "number of batch of graphs.")
flags.DEFINE_integer("num_graphs", 4, "number of graphs in each group.")
flags.DEFINE_integer(
    "iter_num", 50, "number of iterstions for Frank-Wolfe or alternationg minimization."
)
flags.DEFINE_string(
    "base_log_dir", ".", "base directory for logging and saving embeddings"
)
flags.DEFINE_integer("gpu", 1, "which gpu to use.")
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
tf.compat.v1.disable_eager_execution()
GPU_MEM_FRACTION = 0.8

if FLAGS.id_gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # The GPU id to use
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.id_gpu)


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def save_graph_list(G_list, fname):
    with open(fname, "wb") as f:
        pickle.dump(G_list, f)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def log_dir():
    log_dir_orig = FLAGS.base_log_dir + str(FLAGS.epochs) + "epochs"
    log_dir_orig += "/{model:s}_{model_size:s}_{lr:0.6f}/".format(
        model=FLAGS.model, model_size=FLAGS.model_size, lr=FLAGS.learning_rate
    )

    if not os.path.exists(log_dir_orig):
        os.makedirs(log_dir_orig)
    return log_dir_orig


def Graph_load(G, graph_labels=True):
    """
    load many graphs
    return: node ids
    """

    node_id = {str(k): int(k) for k in G.nodes()}
    return node_id, G


def opt_thr(graph_list, A_first, min_deg, thr):
    """opt_thr finds the best thr
    for binarizing the decoder's output."""

    new_adj = []
    new_adj1 = []
    for x in range(len(A_first)):
        A0_rec = np.zeros((len(A_first[0]), len(A_first[0])))
        q = len(A_first[0])
        A0 = A_first[x]

        for n in range(len(A0)):
            for j in range(len(A0)):
                if A0[n, j] >= thr:
                    A0_rec[n, j] = 1

        new_adj.append(A0_rec)
        new_adj1.append(np.reshape(A0_rec, (len(A0) ** 2, 1)))

    graph_list0 = []
    for l in range(len(new_adj)):
        G = nx.Graph()
        edges = []
        adj_nonzero1 = np.nonzero(new_adj[l])
        for i in range(len(adj_nonzero1[0])):
            edges.append((adj_nonzero1[0][i], adj_nonzero1[1][i]))

        data_tuple = list(map(tuple, edges))
        G.add_edges_from(data_tuple)
        id1 = np.arange(len(A0))
        for i in id1:
            G.add_node(i)
        G.remove_nodes_from(list(nx.isolates(G)))
        graph_list0.append(G)

    test_result = validation_score(graph_list, new_adj1, min_deg, thr)
    return test_result.s1, test_result.s2, thr, graph_list0, new_adj


def construct_placeholders():
    # Define placeholders

    placeholders = {
        "dropout": tf.placeholder_with_default(0.0, shape=(), name="dropout"),
        "features": tf.sparse_placeholder(tf.float32),
        "adj_orig": tf.sparse_placeholder(tf.float32),
        "adj": tf.sparse_placeholder(tf.float32),
        "P": tf.placeholder(tf.float32),
    }
    return placeholders


def train(graph_list, min_deg, graph_ref, t_init, test_data=None):

    train_data = Graph_load(graph_list[0])
    G = train_data[1]
    id1 = list(G.nodes())
    id1.sort()
    m = G.number_of_nodes()

    node_id = train_data[0]

    placeholders = construct_placeholders()

    batch = Graph_processor(
        G,
        node_id,
        placeholders,
    )

    nodes = list(G.nodes())
    features1 = np.eye(len(nodes))[nodes]
    features_nonzero = features1[1].shape[0]

    features = sparse_to_tuple(
        scipy.sparse.csr_matrix(features1).astype(np.float32).tocoo()
    )
    adj = batch.adj

    adj_orig = batch.adj_train
    adj_norm = preprocess_graph(batch.adj_train)

    adj_label = batch.adj_train + sp.eye(batch.adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    # Create model
    if FLAGS.variational == "False":
        model = GCNModelAE(
            placeholders,
            [features_nonzero, FLAGS.dim_1],
            features_nonzero,
            logging=True,
        )
        print("AUTOENCODER")
    elif FLAGS.variational == "True":
        model = GCNModelVAE(
            placeholders,
            [features_nonzero, FLAGS.dim_1],
            len(id1),
            features_nonzero,
            logging=True,
        )
        print("VARIATIONAL AUTOENCODER")

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = (
        adj.shape[0]
        * adj.shape[0]
        / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    )

    # Optimizer
    with tf.name_scope("optimizer"):
        if FLAGS.variational == "False":
            opt = OptimizerAE(
                preds=model.reconstructions,
                labels=tf.reshape(
                    tf.sparse_tensor_to_dense(
                        placeholders["adj_orig"], validate_indices=False
                    ),
                    [-1],
                ),
                pos_weight=pos_weight,
                norm=norm,
                adj=adj,
                P=P,
            )
            print("AUTO-ENCODER")
        elif FLAGS.variational == "True":
            opt = OptimizerVAE(
                placeholders,
                preds=model.reconstructions,
                labels=tf.reshape(
                    tf.sparse_tensor_to_dense(
                        placeholders["adj_orig"], validate_indices=False
                    ),
                    [-1],
                ),
                model=model,
                num_nodes=len(nodes),
                pos_weight=pos_weight,
                norm=norm,
                adj=adj,
            )
            print("VARIATIONAL AUTO-ENCODER")

    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.allow_soft_placement = True

    ## Initialize session
    sess = tf.Session(config=config)
    merged = tf.summary.merge_all()
    log_dir_orig = log_dir()
    summary_writer = tf.summary.FileWriter(log_dir_orig, sess.graph)

    # Init variables
    sess.run(tf.global_variables_initializer())

    for i in range(len(graph_list)):
        train_data = Graph_load(graph_list[i])
        G = train_data[1]
        id1 = list(G.nodes())
        id1.sort()
        m = G.number_of_nodes()

        node_id = train_data[0]

        batch = Graph_processor(
            G,
            node_id,
            placeholders,
        )

        nodes = list(G.nodes())
        features1 = np.eye(len(nodes))[nodes]
        features_nonzero = features1[1].shape[0]

        features = sparse_to_tuple(
            scipy.sparse.csr_matrix(features1).astype(np.float32).tocoo()
        )
        adj = batch.adj

        adj_orig = batch.adj_train
        adj_norm = preprocess_graph(batch.adj_train)

        adj_label = batch.adj_train + sp.eye(batch.adj_train.shape[0])
        adj_label = sparse_to_tuple(adj_label)
        # Train model
        total_steps = 0
        avg_time = 0.0
        epoch_val_costs = []

        acc_val = []
        val_roc_score = []

        def get_roc_score(edges_pos, edges_neg, emb=None):
            if emb is None:
                # feed_dict.update({placeholders["dropout"]: 0})
                emb = sess.run(model.outputs, feed_dict=feed_dict)

            def sigmoid(x):
                return 1 / (1 + np.exp(-x))

            # Predict on test set of edgess
            adj_rec = np.dot(emb, emb.T)
            preds = []
            pos = []
            for e in edges_pos:
                preds.append(sigmoid(adj_rec[e[0], e[1]]))
                pos.append(adj_orig[e[0], e[1]])

            preds_neg = []
            neg = []
            for e in edges_neg:
                preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
                neg.append(adj_orig[e[0], e[1]])

            preds_all = np.hstack([preds, preds_neg])
            labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])

            roc_score = roc_auc_score(labels_all, preds_all)
            ap_score = average_precision_score(labels_all, preds_all)

            def Find_Optimal_Cutoff(labels_all, preds_all):
                """Find the optimal probability cutoff point for a classification model related to event rate
                Parameters
                ----------
                target : Matrix with dependent or target data, where rows are observations
                predicted : Matrix with predicted data, where rows are observations
                Returns
                -------
                list type, with optimal cutoff value
                """
                fpr, tpr, threshold = roc_curve(labels_all, preds_all)
                i = np.arange(len(tpr))
                roc = pd.DataFrame(
                    {
                        "fpr": pd.Series(fpr, index=i),
                        "tpr": pd.Series(tpr, index=i),
                        "1-fpr": pd.Series(1 - fpr, index=i),
                        "tf": pd.Series(tpr - (1 - fpr), index=i),
                        "threshold": pd.Series(threshold, index=i),
                    }
                )
                roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]

                li = []
                for i in range(len(fpr)):
                    li.append((1 - tpr[i]) ** 2 + fpr[i] ** 2)

                x = np.amin(li)
                ind = li.index(x)
                tr = threshold[ind]

                return list(roc_t["threshold"]), tr

            thr, tr = Find_Optimal_Cutoff(labels_all, preds_all)

            return roc_score, ap_score, thr, tr

        val_ap_score = []
        val_roc_score = []
        train_loss = []
        train_acc = []
        iter = 0
        for epoch in range(FLAGS.epochs):
            batch.shuffle()

            epoch_val_costs.append(0)
            feed_dict = dict()
            feed_dict.update({placeholders["dropout"]: FLAGS.dropout})
            feed_dict.update({placeholders["adj"]: adj_norm})
            feed_dict.update({placeholders["adj_orig"]: adj_label})
            feed_dict.update({placeholders["features"]: features})
            t = time.time()
            # Training step
            outs = sess.run(
                [
                    opt.opt_op,
                    opt.cost,
                    opt.accuracy,
                    model.outputs,
                    model.reconstructions,
                ],
                feed_dict=feed_dict,
            )

            # Compute average loss
            avg_cost = outs[1]
            avg_accuracy = outs[2]
            roc_curr, ap_curr, thr, tr = get_roc_score(
                batch.train_edges, batch.edges_false
            )

            print(
                "Epoch:",
                "%04d" % (epoch + 1),
                "train_loss=",
                "{:.05f}".format(avg_cost),
                "train_acc=",
                "{:.5f}".format(avg_accuracy),
                "val_roc=",
                "{:.5f}".format(roc_curr),
                "val_ap=",
                "{:.5f}".format(ap_curr),
                "time=",
                "{:.5f}".format(time.time() - t),
            )
            train_loss.append(avg_cost)
            train_acc.append(avg_accuracy)

            if iter % 20 == 0:
                roc_curr, ap_curr, thr, tr = get_roc_score(
                    batch.train_edges, batch.edges_false
                )
                val_roc_score.append(roc_curr)
                val_ap_score.append(ap_curr)
            iter = iter + 1

    print("total training time", time.time() - t_init)
    t1 = time.time()
    recon_set = []
    for i in range(100):
        output = sess.run([model.outputs, model.reconstructions], feed_dict=feed_dict)
        recon_graph = output[1]
        recon_set.append(recon_graph)

    np.save(
        str(FLAGS.num_batch)
        + str(FLAGS.model_size)
        + str(FLAGS.train_prefix)
        + "_parallel_align",
        recon_set,
    )
    print("total time to generate graphs=", "{:.5f}".format(time.time() - t1))


def opt_val(min_deg):
    """Comute performance scores in testing"""

    with open(FLAGS.data, "rb") as f:

        graph_ref = pickle.load(f, encoding="latin1")

    recon_set = np.load(
        str(FLAGS.num_batch)
        + str(FLAGS.model_size)
        + str(FLAGS.train_prefix)
        + "_parallel_align.npy",
        allow_pickle=True,
    )

    hist, bin_edges = np.histogram(recon_set[58])
    print("hist0, bin_edges0", hist, bin_edges)

    rec0 = recon_set[0:20]

    thr = list(np.linspace(0.0, 0.5, num=20))

    pool_size = mp.cpu_count()
    pool = mp.Pool(
        processes=pool_size,
    )
    score = pool.starmap(
        validation_score, [(graph_ref[0:20], rec0, min_deg, i) for i in thr]
    )
    pool.close()
    pool.join()
    print(
        [score[i].s1 for i in range(len(score))],
        [score[i].s2 for i in range(len(score))],
        thr,
    )

    smallest = min([score[i].s1 for i in range(len(score))])
    index = [
        index
        for index, element in enumerate([score[i].s1 for i in range(len(score))])
        if min([score[i].s1 for i in range(len(score))]) == element
    ]

    thr_opt = thr[index[-1]]

    test_result = validation_score(
        graph_ref[80:100], recon_set[80:100], min_deg, thr_opt
    )
    print("s1 and s2 and s3 scores", test_result.s1, test_result.s2, test_result.s3)


def main(argv=None):
    print("Loading training data..")

    with open(FLAGS.data, "rb") as f:

        graphs = pickle.load(f, encoding="latin1")

    n_train = int(0.8 * len(graphs))
    n_test = int(0.2 * len(graphs))
    graph_train = graphs[0:n_train]

    A_list = []
    all_deg = []
    mindeg = []
    for i in range(len(graph_train)):
        g = graph_train[i]
        deg = [g.degree[i] for i in g.nodes()]
        all_deg = all_deg + deg
        mindeg.append(min(deg))

    deg_list = all_deg
    min_deg = min(mindeg)

    if FLAGS.training == "True":

        t0 = time.time()
        # load the aligned graphs' adjacency matrices
        adj_aligned = np.load("A_align_citeseer.npy")

        graphs = []
        num_nodes = []
        A = []
        for i in range(len(adj_aligned)):
            adj = adj_aligned[i]
            for j in range(len(adj)):
                if adj[j, j] == 1:
                    adj[j, j] == 0
            A.append(adj)
            g = nx.from_numpy_matrix(adj)
            g.remove_nodes_from(list(nx.isolates(g)))
            num_nodes.append(len(adj))
            print("# of nodes", g.number_of_nodes())
            graphs.append(g)

        m = max(num_nodes)
        for i in range(len(graphs)):
            adj_tmp = np.zeros((m, m))
            ln = len(A[i])
            adj_tmp[0:ln, 0:ln] = A[i]
            adj_tmp[ln:, :] = 0.01
            adj_tmp[:, ln:] = 0.01
            A[i] = adj_tmp
            graphs[i] = nx.from_numpy_matrix(adj_tmp)

        print("len(graphs)", len(graphs))
        graph_train = graphs
        t0 = time.time()
        train(graph_train, min_deg, graphs, t0)
    else:
        opt_val(np.min(min_deg))


if __name__ == "__main__":
    tf.app.run()
