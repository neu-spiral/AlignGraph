from __future__ import division
from __future__ import print_function

import numpy as np
import math
import tensorflow as tf
import scipy.sparse as sp
import networkx as nx

# Boilerplate parts of this code file were originally forked from
# https://github.com/tkipf/gae/

np.random.seed(123)
tf.random.set_seed(123)


def preprocess_graph(x):
    adj = sp.coo_matrix(x)
    adj_ = adj
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = (
        adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    )
    return sparse_to_tuple(adj_normalized)


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


class Graph_processor(object):

    """
    This minibatch iterator iterates over nodes for supervised learning.
    G -- networkx graph
    id2idx -- dict mapping node ids to integer values indexing feature tensor
    placeholders -- standard tensorflow placeholders object for feeding
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    """

    def __init__(self, G, id2idx, placeholders, **kwargs):

        self.G = G
        self.nodes = G.nodes()
        self.id2idx = id2idx
        self.placeholders = placeholders
        self.train_nodes = G.nodes()
        print("number of trained nodes:", len(self.train_nodes))

        edges = G.edges()

        self.train_edges = self.edges = np.random.permutation(edges)
        print("len(self.train_edges):", len(self.train_edges))

        (
            self.train_edges,
            self.edges_false,
            self.adj,
            self.adj_train,
            self.deg,
        ) = self.construct_adj_edge()

    def construct_adj_edge(self):

        print("self.id2idx", self.id2idx.values())
        id1 = list(self.id2idx.values())[0]
        adj = nx.adjacency_matrix(self.G).todense()
        for i in range(len(adj)):
            if adj[i, i] == 1:
                adj[i, i] == 0

        edge_train = []
        adj_nonzero = np.nonzero(adj)

        for i in range(len(adj_nonzero[0])):
            edge_train.append((adj_nonzero[0][i], adj_nonzero[1][i]))

        adj_neg = np.ones((len(self.id2idx), len(self.id2idx))) - adj
        adj_neg_non_zero = np.nonzero(adj_neg)
        edge_neg = []
        print("len(adj_neg_non_zero[0])", len(adj_neg_non_zero[0]))
        for i in range(len(adj_neg_non_zero[0])):
            edge_neg.append((adj_neg_non_zero[0][i], adj_neg_non_zero[1][i]))
        edge_neg = np.random.permutation(edge_neg)
        edges_false = edge_neg[: len(edge_train)]

        deg = [
            nx.from_numpy_matrix(adj).degree[i]
            for i in nx.from_numpy_matrix(adj).nodes()
        ]
        adj_train = sp.csr_matrix(adj)
        print("self.adj_train", np.max(adj_train[0]))
        return edge_train, edges_false, adj, adj_train, deg

    def shuffle(self):
        """Re-shuffle the training set."""
        self.train_nodes = np.random.permutation(self.train_nodes)
