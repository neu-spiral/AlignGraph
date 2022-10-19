import numpy as np
import networkx as nx
import argparse
import pickle
import sknetwork
import scipy
from scipy import sparse
from scipy.sparse import csc_matrix
import cvxpy
import cvxpy as cp
import multiprocessing as mp
import multiprocessing
from sklearn.base import ClusterMixin, BaseEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy.sparse import csr_matrix, spdiags
from sklearn.neighbors import kneighbors_graph
from fermat_distance import Fermat_distance, Fermat_Frank_Wolfe
from sknetwork.clustering import Louvain
from sknetwork.data import karate_club
from sknetwork.path import shortest_path
from sknetwork.clustering import KMeans
from sknetwork.embedding import GSVD
from scipy.optimize import linear_sum_assignment
from k_means_constrained import KMeansConstrained
import time

import os

os.environ["MKL_NUM_THREADS"] = "1"


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def P_estimate(A0, A):
    a = A
    m = len(a[0])
    n = len(A)
    P = []
    s = 0
    print("len(A))", len(A))
    for j in range(len(A)):
        P_opt = cp.Variable((m, m))
        # s = s + cp.norm(((self.A[j]@P_opt[0:m , m*j:m +  m*j ])-(P_opt[0:m , m*j:m+  m*j]@ A0)) , 'fro')
        s = cp.norm(((A[j] @ P_opt) - (P_opt @ A0)), "fro") ** 2
        constraints = [P_opt >= 0]
        constraints += [sum(P_opt) == 1]
        constraints += [sum(P_opt.T) == 1]
        prob = cp.Problem(cp.Minimize(s), constraints)
        prob.solve(solver=cp.SCS)
        P.append(P_opt.value)

    return P


def aligned_subgraph(G, features, P_set, cluster_indmatrix, feat_dict):
    sub_graph = []
    B = np.matmul(P_set.T, np.array(cluster_indmatrix))
    cluster_align = [feat_dict[k] for k in B]
    for i in range(len(cluster_align)):
        sub_graph.append(G.subgraph(cluster_align[i]))

    return cluster_align, sub_graph


def align_sparse(A_sparse, sprs_G, max_nodes):
    """Aligning the edges connecting clusteres by building subgraphs and build back the original adjacency matrix of connecting edges from it."""
    num_nodes = [len(sprs_G[i].nodes()) for i in range(len(sprs_G))]
    list_nodes = [list(sprs_G[i].nodes()) for i in range(len(sprs_G))]
    print("num_nodes", num_nodes)
    A = []
    for i in range(len(sprs_G)):
        adj = nx.adjacency_matrix(sprs_G[i]).todense()
        A.append(adj)
        if len(adj) < max(num_nodes):
            adj_tmp = np.zeros((max(num_nodes), max(num_nodes)))
            adj_tmp[0 : len(adj), 0 : len(adj)] = adj
            adj_tmp[len(adj) :, :] = 0.01
            adj_tmp[:, len(adj) :] = 0.01
            A[i] = adj_tmp
    P_Galign_sprs = P_est(A)
    pool_size = mp.cpu_count()
    pool = mp.Pool(
        processes=pool_size,
    )
    PPA_sprs = pool.starmap(
        binarized_P_adj,
        [(A[i], P_Galign_sprs, i) for i in range(len(A))],
    )

    P_WS_sprs = [PPA_sprs[i][0] for i in range(len(PPA_sprs))]
    P_set_sprs = [PPA_sprs[i][1] for i in range(len(PPA_sprs))]
    A_list = [PPA_sprs[i][2] for i in range(len(PPA_sprs))]
    pool.close()
    pool.join()

    A_list_sprs = []
    total_nodes = list(np.arange(max_nodes))
    for i in range(len(P_set_sprs)):
        P_rec = np.zeros((max_nodes, max_nodes))
        a = np.nonzero(P_set_sprs[i])
        node_li = set(total_nodes).difference(list_nodes[i])
        node_li = list(set(node_li).difference(list_nodes[0]))
        print("node_li", node_li)
        if len(sprs_G[i].nodes()) < max(num_nodes):
            n = max(num_nodes) - len(sprs_G[i].nodes())
            print("n for Gi", n, num_nodes[i])
            for s in range(n):
                if s < len(node_li):
                    list_nodes[i].append(node_li[s])
                else:
                    list_nodes[i].append(node_li[-1])
        if len(sprs_G[i].nodes()) < max(num_nodes):
            n = max(num_nodes) - len(sprs_G[0].nodes())
            print("n for G0", n, num_nodes[0])
            for s in range(n):
                if s < len(node_li):
                    list_nodes[0].append(node_li[s])
                else:
                    list_nodes[0].append(node_li[-1])
        for j in range(len(a[0])):
            P_rec[list_nodes[i][list(a[0])[j]], list_nodes[0][list(a[1])[j]]] = 1
        A_tmp = np.matmul(np.matmul(P_rec.T, A_sparse[i]), P_rec)

        pool_size = mp.cpu_count()
        pool = mp.Pool(
            processes=pool_size,
        )
        bin = pool.starmap(
            binarized_vec,
            [(A_tmp[i], 0.5) for i in range(len(A_tmp))],
        )
        pool.close()
        pool.join()
        A_tmp = [bin[i] for i in range(len(A_tmp))]
        A_list_sprs.append(np.asarray(A_tmp))

    return A_list_sprs


def between_cluster_edges(G, cluster_ind):
    """finding edges connecting clusters whithin a graph"""
    edg = []
    n_edges = []
    edge_orig = G.edges()

    for k in range(num_clusters):

        inter_edge = [
            (min(a, b), max(a, b)) for a in cluster_ind[k] for b in cluster_ind[k]
        ]
        inter_edge = inter_edge + [
            (max(a, b), min(a, b)) for a in cluster_ind[k] for b in cluster_ind[k]
        ]
        edge_orig = set(edge_orig).difference(inter_edge)
        n_edges.append(G.subgraph(cluster_ind[k]).size(weight="weight"))

    return list(edge_orig), n_edges


def binarized_P_adj(A_cluster, P_opt, i):
    print("len(P_opt)", len(P_opt))
    m = len(A_cluster)
    P = P_opt[i]
    P_binary = np.zeros((m, m))
    P1 = -1 * P
    row_ind, col_ind = linear_sum_assignment(P1)
    P_binary[row_ind, col_ind] = 1
    adj = np.matmul(np.matmul(P_binary.T, A_cluster), P_binary)
    return P, P_binary, adj


def binarized_P_adjF(A_cluster, P_opt, i):
    m = len(A_cluster)
    P = P_opt[0]
    P_binary = np.zeros((m, m))
    P1 = -1 * P
    row_ind, col_ind = linear_sum_assignment(P1)
    P_binary[row_ind, col_ind] = 1
    adj = np.matmul(np.matmul(P_binary.T, A_cluster), P_binary)
    return P, P_binary, adj


def binarized_weighted_P_adj(A_cluster, P_opt, i):
    m = len(A_cluster)
    P = P_opt[m * i : m + m * i, 0:m]
    P_binary = np.zeros((m, m))
    P1 = -1 * P
    row_ind, col_ind = linear_sum_assignment(P1)
    P_binary[row_ind, col_ind] = 1
    adj = np.matmul(np.matmul(P_binary.T, A_cluster), P_binary)
    return P, P_binary, adj


def binarized_vec(vec, thr):
    tmp = vec
    tmp = [0 if tmp_ < thr else 1 for tmp_ in tmp]
    vec = tmp
    return vec


def center_graph(A, P_opt):
    """Computes the center of a group of graphs."""
    m = len(A[0])
    n = len(A)

    A0 = cp.Variable((m, m))
    constraints = [-A0[0, 0] <= 0]
    constraints += [A0[0, 0] <= 1]
    s = 0
    for j in range(n):
        P = P_opt[m * j : m + m * j, 0:m]
        P_binary = np.zeros((m, m))
        P1 = -1 * P
        row_ind, col_ind = linear_sum_assignment(P1)
        P_binary[row_ind, col_ind] = 1
        s = s + cp.norm(np.matmul(np.matmul(P_binary.T, A[j]), P_binary) - A0, "fro")

    for e in range(m):
        for f in range(m):
            constraints += [A0[e, f] <= 1]
            constraints += [-A0[e, f] <= 0]
    prob = cp.Problem(cp.Minimize(s), constraints)
    prob.solve(solver=cp.SCS)

    counts, bin_edges = np.histogram(np.asarray(A0.value))
    return A0.value


def center_clsgraph(A):
    """Computes the center of a group of graphs."""
    m = len(A[0])
    n = len(A)
    # A0= np.zeros((m, m))
    A0 = cp.Variable((m, m))
    constraints = [-A0[0, 0] <= 0]
    constraints += [A0[0, 0] <= 1]
    s = 0
    for j in range(n):
        s = s + cp.norm(A[j] - A0, "fro")

    for e in range(m):
        for f in range(m):
            constraints += [A0[e, f] <= 1]
            constraints += [-A0[e, f] <= 0]
    prob = cp.Problem(cp.Minimize(s), constraints)
    prob.solve(solver=cp.SCS)

    return A0.value


def center_cls_ave(A):
    """Computes the center of a group of clusters by taking their average."""

    return (1 / (len(A))) * sum(A)


def cluster_graph(num_clusters, intra_edges, cluster_ind, num_edges):
    """build a graph with each node representing a cluster."""
    A_cluster = np.zeros((num_clusters, num_clusters))
    print("cluster_graph starts!!!!")
    print("intra_edges", len(intra_edges))
    for i in range(len(num_edges)):
        A_cluster[i, i] = num_edges[i]
    for k in range(len(intra_edges)):
        for l in range(num_clusters):
            for j in range(num_clusters):
                if j > l:
                    if (
                        intra_edges[k][0] in cluster_ind[l]
                        or intra_edges[k][1] in cluster_ind[l]
                    ):
                        if (
                            intra_edges[0] in cluster_ind[j]
                            or intra_edges[k][1] in cluster_ind[j]
                        ):
                            A_cluster[l, j] = A_cluster[l, j] + 1

    A_cluster = A_cluster + A_cluster.T
    print("cluster_graph done!!!!")
    return A_cluster


def feat_cluster_build(adj, num_clusters):
    """Build feature matrices with number of nodes in each community as a feature and identify node indexes of each cluster."""
    # kmeans = KMeans(n_clusters=num_clusters, embedding_method=GSVD(3))
    # labels= kmeans.fit_transform(sparse.csr_matrix(adj))
    # kmeans = KMeansConstrained(n_clusters=2, size_min=18, size_max=18, random_state=0 )
    labels = kmeans.fit_predict(adj)
    unique_labels, counts = np.unique(labels, return_counts=True)
    features = list(counts)
    cls = []
    for k in range(num_clusters):
        cls.append(np.where(labels == unique_labels[k])[0].tolist())
    cluster_ind = cls
    return labels, features, cluster_ind


def feat_cluster_buildF(adj, num_clusters):

    """Build feature matrices with number of nodes in each community as a feature and identify node indexes of each cluster."""

    labels = kmeans.fit_predict(adj)
    # labels= kmeans.fit_transform(sparse.csr_matrix(adj))
    unique_labels, counts = np.unique(labels, return_counts=True)
    features = list(counts)
    cls = []
    for k in range(num_clusters):
        cls.append(np.where(labels == unique_labels[k])[0].tolist())
    cluster_ind = cls
    return labels, features, cluster_ind


def recover_A_P(
    G,
    A_sprs,
    A_aligned_list,
    cluster_align,
    intra_cluster_edges,
    P_bin_list,
    P_list,
    num_clusters,
    ind,
):
    n = len(G.nodes())
    A_tmp = np.zeros((n, n))
    P_rec = np.zeros((n, n))
    P_Wrec = np.zeros((n, n))
    # A_rec= np.zeros((n, n))
    A_rec = A_sprs

    for i in range(num_clusters):
        a = np.nonzero(P_bin_list[i][ind])
        for j in range(len(a[0])):
            P_rec[
                cluster_align[ind][i][list(a[0])[j]], cluster_align[0][i][list(a[1])[j]]
            ] = 1

    for i in range(num_clusters):
        a = np.nonzero(P_list[i][ind])
        for j in range(len(a[0])):
            P_Wrec[
                cluster_align[ind][i][list(a[0])[j]], cluster_align[0][i][list(a[1])[j]]
            ] = P_list[i][ind][[list(a[0])[j]], [list(a[1])[j]]]

    # ************ NEW ADDED ************
    A_tmp = A_sprs
    print(
        "number of non-zero elements before alignment",
        np.count_nonzero(A_sprs),
        np.count_nonzero(P_Wrec),
    )

    A_sprs_aligned = np.matmul(np.matmul(P_rec.T, A_tmp), P_rec)

    for i in range(len(A_sprs_aligned)):
        tmp = binarized_vec(A_sprs_aligned[i], 0.2)
        A_sprs_aligned[i] = tmp

    print(
        "number of non-zero elements after alignment",
        np.count_nonzero(A_sprs_aligned),
        np.count_nonzero(P_rec),
    )
    # print("A_sprs_aligned", A_sprs_aligned)

    tmp = np.nonzero(A_sprs_aligned)
    A_rec = A_sprs_aligned
    # A_rec= np.zeros((n,n))

    for i in range(num_clusters):
        for k in range(len(A_aligned_list[i][ind])):
            tmp = binarized_vec(np.asarray(A_aligned_list[i][ind])[k], 0.2)
            A_aligned_list[i][ind][k] = tmp
        a = np.nonzero(np.asarray(A_aligned_list[i][ind]))
        for j in range(len(a[0])):
            A_rec[
                cluster_align[0][i][list(a[0])[j]], cluster_align[0][i][list(a[1])[j]]
            ] = 1
            A_rec[
                cluster_align[0][i][list(a[1])[j]], cluster_align[0][i][list(a[0])[j]]
            ] = 1

    return A_rec, P_rec, P_Wrec, A_sprs_aligned


def sparse_adj(graph_train, num_tr, intra_cluster_edges):
    print("len(intra_cluster_edges)", len(intra_cluster_edges))
    A_sparse = []
    for j in range(len(graph_train)):
        n = len(graph_train[j].nodes())
        adjj = nx.adjacency_matrix(graph_train[j]).todense()
        A_tmp = np.zeros((n, n))
        for i in range(len(intra_cluster_edges[j])):
            A_tmp[intra_cluster_edges[j][i][0], intra_cluster_edges[j][i][1]] = adjj[
                intra_cluster_edges[j][i][0], intra_cluster_edges[j][i][1]
            ]
            A_tmp[intra_cluster_edges[j][i][1], intra_cluster_edges[j][i][0]] = adjj[
                intra_cluster_edges[j][i][1], intra_cluster_edges[j][i][0]
            ]
        A_sparse.append(A_tmp)

    return A_sparse


def sparse_adjF(graph_train, num_tr, intra_cluster_edges):

    A_sparse = []
    sprs_G = []
    for j in range(len(graph_train)):
        n = len(graph_train[j].nodes())
        adjj = nx.adjacency_matrix(graph_train[j]).todense()
        sprs_G.append(graph_train[j].edge_subgraph(intra_cluster_edges[j]))
        A_tmp = np.zeros((n, n))
        for i in range(len(intra_cluster_edges[j])):
            A_tmp[intra_cluster_edges[j][i][0], intra_cluster_edges[j][i][1]] = adjj[
                intra_cluster_edges[j][i][0], intra_cluster_edges[j][i][1]
            ]
            A_tmp[intra_cluster_edges[j][i][1], intra_cluster_edges[j][i][0]] = adjj[
                intra_cluster_edges[j][i][1], intra_cluster_edges[j][i][0]
            ]

        A_sparse.append(A_tmp)

    return A_sparse, sprs_G


def WeightedAdj_P(A, F):
    """Computes graph alinmnet matrices via CVXPY"""
    m = len(A[0])
    n = len(A)
    s = 0
    P_opt = cp.Variable((m * n, m * n), symmetric=True)
    for i in range(len(A)):
        for j in range(len(A)):
            s = s + cp.norm(
                (
                    (A[i] @ P_opt[m * i : m + m * i, m * j : m + m * j])
                    - (P_opt[m * i : m + m * i, m * j : m + m * j] @ A[j])
                ),
                "fro",
            )
            D = np.zeros((len(F[i]), len(F[j])))
            for k in range(len(F[i])):
                for l in range(len(F[j])):
                    D[k, l] = np.linalg.norm(F[i][k] - F[j][l])

            s = s + cp.trace(P_opt[m * i : m + m * i, m * j : m + m * j].T @ D)
    constraints = [P_opt >> 0]
    constraints += [cp.diag(P_opt) == 1]
    for i in range(len(A)):
        for j in range(len(A)):
            constraints += [P_opt[m * i : m + m * i, m * j : m + m * j] >= 0]
            constraints += [sum(P_opt[m * i : m + m * i, m * j : m + m * j]) == 1]
            constraints += [sum(P_opt[m * i : m + m * i, m * j : m + m * j].T) == 1]

    prob = cp.Problem(cp.Minimize(0.5 * s), constraints)
    prob.solve()

    return P_opt.value


def WeightedAdj_PF(A, F):
    """Computes graph alinmnet matrices via CVXPY"""
    m = len(A[0])
    n = len(A)
    s = 0
    P_opt = cp.Variable((m * n, m * n), symmetric=True)
    for i in range(len(A)):
        for j in range(len(A)):
            s = s + cp.norm(
                (
                    (A[i] @ P_opt[m * i : m + m * i, m * j : m + m * j])
                    - (P_opt[m * i : m + m * i, m * j : m + m * j] @ A[j])
                ),
                "fro",
            )
            D = np.zeros((len(F[i]), len(F[j])))
            for k in range(len(F[i])):
                for l in range(len(F[j])):
                    D[k, l] = np.linalg.norm(F[i][k] - F[j][l])

            s = s + cp.trace(P_opt[m * i : m + m * i, m * j : m + m * j].T @ D)
    constraints = [P_opt >> 0]
    constraints += [cp.diag(P_opt) == 1]
    for i in range(len(A)):
        for j in range(len(A)):
            constraints += [P_opt[m * i : m + m * i, m * j : m + m * j] >= 0]
            constraints += [sum(P_opt[m * i : m + m * i, m * j : m + m * j]) == 1]
            constraints += [sum(P_opt[m * i : m + m * i, m * j : m + m * j].T) == 1]

    prob = cp.Problem(cp.Minimize(0.5 * s), constraints)
    prob.solve()

    return P_opt.value


def P_Fest(A0, A):
    m = len(A)
    n = len(A)
    P = []
    s = 0

    P_opt = cp.Variable((m, m))
    s = cp.norm(((A @ P_opt) - (P_opt @ A0)), "fro") ** 2
    constraints = [P_opt >= 0]
    constraints += [sum(P_opt) == 1]
    constraints += [sum(P_opt).T == 1]
    prob = cp.Problem(cp.Minimize(s), constraints)
    prob.solve(solver=cp.SCS)
    P.append(P_opt.value)

    return P


class AP_cent(object):
    """
    The Frank Wolfe class
    To solve \sum_{i,j \in [n]} ||A_iP_ij-P_ijA_j||_{2}^{2}+tr(P^{T}D) via Frank-Wolfe.
    """

    def __init__(
        self, max_nodes, num_clusters, num_tr, A_thr, itr, distance_type="CVX_clustered"
    ):

        self.num_clusters = num_clusters
        self.max_nodes = max_nodes
        self.num_tr = num_tr
        self.A_thr = A_thr
        self.itr = itr
        # self.kmeans= kmeans
        self.distance_type = distance_type
        Process = NoDaemonProcess

    def final_graphs(self, A, graph_train):

        # kmeans = KMeans(n_clusters=num_clusters, embedding_method=GSVD(3))
        labels = []
        features = []
        cluster_ind = []

        pool_size = mp.cpu_count()
        pool = mp.Pool(
            processes=pool_size,
        )
        lab_feat_cls = pool.starmap(
            feat_cluster_build,
            [(A[i], self.num_clusters) for i in range(len(A))],
        )

        labels = [lab_feat_cls[i][0] for i in range(len(A))]
        features = [lab_feat_cls[i][1] for i in range(len(A))]
        cluster_ind = [lab_feat_cls[i][2] for i in range(len(A))]
        print("features", features)

        intra_cluster_edges_numedges = pool.starmap(
            between_cluster_edges,
            [(graph_train[i], cluster_ind[i]) for i in range(len(graph_train))],
        )

        intra_cluster_edges = [
            intra_cluster_edges_numedges[i][0]
            for i in range(len(intra_cluster_edges_numedges))
        ]
        num_edges = [
            intra_cluster_edges_numedges[i][1]
            for i in range(len(intra_cluster_edges_numedges))
        ]

        clustered_adj = pool.starmap(
            cluster_graph,
            [
                (
                    self.num_clusters,
                    intra_cluster_edges[i],
                    cluster_ind[i],
                    num_edges[i],
                )
                for i in range(len(intra_cluster_edges))
            ],
        )

        P_opt = WeightedAdj_P(clustered_adj, features)

        PPA = pool.starmap(
            binarized_weighted_P_adj,
            [(clustered_adj[i], P_opt, i) for i in range(len(clustered_adj))],
        )

        P_WS = [PPA[i][0] for i in range(len(PPA))]
        P_set = [PPA[i][1] for i in range(len(PPA))]
        A_list = [PPA[i][2] for i in range(len(PPA))]
        pool.close()
        pool.join()

        feat_dict = {}
        cluster_indmatrix = []
        cnt = 0
        for i in range(len(cluster_ind)):
            tmp = []
            for k in range(len(cluster_ind[i])):
                feat_dict[cnt] = cluster_ind[i][k]
                tmp.append(cnt)
                cnt = cnt + 1
            cluster_indmatrix.append(tmp)

        pool_size = mp.cpu_count()
        pool = mp.Pool(
            processes=pool_size,
        )
        cls_align_sub_graph = pool.starmap(
            aligned_subgraph,
            [
                (graph_train[i], features, P_set[i], cluster_indmatrix[i], feat_dict)
                for i in range(len(features))
            ],
        )

        cluster_align = [
            cls_align_sub_graph[i][0] for i in range(len(cls_align_sub_graph))
        ]
        sub_list = [cls_align_sub_graph[i][1] for i in range(len(cls_align_sub_graph))]

        pool.close()
        pool.join()

        A_subfinal = []
        P_subfinal = []
        A0_subfinal = []
        for k in range(self.num_clusters):
            A1 = [nx.to_numpy_matrix(sub_list[i][k]) for i in range(len(sub_list))]
            fermat = Fermat_distance(self.itr, epsilon=10 ** (-3))

            if self.distance_type == "CVX_clustered":
                P_f, A0 = fermat.AM(A1)
            # if self.distance_type == "FW":
            #     P = np.ones((len(A1[0]) * len(A1), len(A1[0]) * len(A1)))
            #     obj = Frank_Wolfe(P, -1, self.itr)
            #     P_Galign1= obj.iteration(A1)
            A_subfinal.append(A1)
            # P_subfinal.append(P_Galign1)
            A0_subfinal.append(A0)
            P_subfinal.append(P_f)
        P_set1 = []
        A_list1 = []
        A_aligned_list = []
        P_bin_list = []
        P_list = []

        for k in range(len(A_subfinal)):

            pool_size = mp.cpu_count()
            pool = mp.Pool(
                processes=pool_size,
            )
            PPA_sub = pool.starmap(
                binarized_P_adj,
                [
                    (A_subfinal[k][i], P_subfinal[k], i)
                    for i in range(len(A_subfinal[0]))
                ],
            )
            P_W_list = [PPA_sub[i][0] for i in range(len(PPA_sub))]
            P_sub_list = [PPA_sub[i][1] for i in range(len(PPA_sub))]
            A_sub_list = [PPA_sub[i][2] for i in range(len(PPA_sub))]
            pool.close()
            pool.join()
            A_aligned_list.append(A_sub_list)
            P_bin_list.append(P_sub_list)
            P_list.append(P_W_list)

        print(
            "graph_train, self.num_tr, intra_cluster_edges",
            len(graph_train),
            self.num_tr,
            len(intra_cluster_edges),
        )
        sparse_A = sparse_adj(graph_train, self.num_tr, intra_cluster_edges)
        sprs_G = [nx.from_numpy_matrix(sparse_A[i]) for i in range(len(sparse_A))]

        if self.distance_type == "CVX_clustered":
            A_list_sprs = sparse_A
        pool_size = mp.cpu_count()
        pool = mp.Pool(
            processes=pool_size,
        )
        APP_rec = pool.starmap(
            recover_A_P,
            [
                (
                    graph_train[ind],
                    A_list_sprs[ind],
                    A_aligned_list,
                    cluster_align,
                    intra_cluster_edges,
                    P_bin_list,
                    P_list,
                    self.num_clusters,
                    ind,
                )
                for ind in range(self.num_tr)
            ],
        )
        A_total = [APP_rec[i][0] for i in range(len(APP_rec))]
        P_total = [APP_rec[i][1] for i in range(len(APP_rec))]
        P_totalW = [APP_rec[i][2] for i in range(len(APP_rec))]
        A_sprs_aligned = [APP_rec[i][3] for i in range(len(APP_rec))]
        pool.close()
        pool.join()

        pool_size = mp.cpu_count()
        pool = mp.Pool(
            processes=pool_size,
        )
        A_cent = pool.map(
            center_clsgraph,
            [A_aligned_list[i] for i in range(len(A_aligned_list))],
        )
        pool.close()
        pool.join()

        # # A_cent_sprs= center_clsgraph(A_list_sprs)
        # A_cent_sprs= A0_sprs
        A_cent_sprs = center_cls_ave(A_sprs_aligned)
        A_cent_rec = np.zeros((len(A_cent_sprs), len(A_cent_sprs)))

        # thr = list(np.linspace(0.1, 1.0, num=10))
        thr = [0.5]
        for t in thr:
            self.A_thr = t
            pool_size = mp.cpu_count()
            pool = mp.Pool(
                processes=pool_size,
            )
            bin = pool.starmap(
                binarized_vec,
                [(A_cent_sprs[i], self.A_thr) for i in range(len(A_cent_sprs))],
            )
            pool.close()
            pool.join()

            A_cent_sprs = [bin[i] for i in range(len(A_cent_sprs))]
            tmp = np.nonzero(A_cent_sprs)

            for i in range(self.num_clusters):
                pool_size = mp.cpu_count()
                pool = mp.Pool(
                    processes=pool_size,
                )
                tmp_vecs = pool.starmap(
                    binarized_vec,
                    [(A_cent[i][k], self.A_thr) for k in range(len(A_cent[i]))],
                )
                pool.close()
                pool.join()
                A_cent[i] = [tmp_vecs[k] for k in range(len(A_cent[i]))]

                a = np.nonzero(A_cent[i])
                for j in range(len(a[0])):
                    A_cent_rec[
                        cluster_align[0][i][list(a[0])[j]],
                        cluster_align[0][i][list(a[1])[j]],
                    ] = 1
                    A_cent_rec[
                        cluster_align[0][i][list(a[1])[j]],
                        cluster_align[0][i][list(a[0])[j]],
                    ] = 1

            for i in range(len(tmp[0])):
                A_cent_rec[list(tmp[0])[i], list(tmp[1])[i]] = 1

            G_cent = nx.from_numpy_matrix(A_cent_rec)
            G_total = []
            for i in range(len(A_total)):
                g = nx.from_numpy_matrix(A_total[i])
                g.remove_edges_from(list(nx.selfloop_edges(g)))
                G_total.append(g)
                print("G_total[i]", G_total[i].nodes(), i)
            print("len(G_total)", len(G_total), G_total, len(A_total), A_total[0])
            print("G_total[0]", G_total[0].nodes())
        return A_cent_rec, G_cent, A_total, G_total, P_total


class AP_align(object):
    """
    The Frank Wolfe class
    To solve \sum_{i,j \in [n]} ||A_iP_ij-P_ijA_j||_{2}^{2}+tr(P^{T}D) via Frank-Wolfe.
    """

    def __init__(self, A0, num_clusters, num_tr, A_thr, distance_type="CVX_clustered"):

        self.num_clusters = num_clusters
        self.num_tr = num_tr
        self.A_thr = A_thr
        self.A0 = A0
        self.distance_type = distance_type
        Process = NoDaemonProcess

    def final_graphs(self, A, graph_train):

        # kmeans = KMeans(n_clusters=num_clusters, embedding_method=GSVD(3))
        labels = []
        features = []
        cluster_ind = []
        print(
            "# of nodes and edges",
            len(graph_train[0].nodes()),
            len(graph_train[0].edges()),
        )
        print("A", A)
        A = [self.A0] + A

        graph_train = [nx.from_numpy_matrix(self.A0)] + graph_train
        print("len(A), len(graph_train)", len(A), len(graph_train))

        pool_size = mp.cpu_count()
        pool = mp.Pool(
            processes=pool_size,
        )
        lab_feat_cls = pool.starmap(
            feat_cluster_buildF,
            [(A[i], self.num_clusters) for i in range(len(A))],
        )

        labels = [lab_feat_cls[i][0] for i in range(len(A))]
        features = [lab_feat_cls[i][1] for i in range(len(A))]
        cluster_ind = [lab_feat_cls[i][2] for i in range(len(A))]
        # print("features", features)
        # print("LINE 936")
        intra_cluster_edges_numedges = pool.starmap(
            between_cluster_edges,
            [(graph_train[i], cluster_ind[i]) for i in range(len(graph_train))],
        )

        # print("****intra_cluster_edges_numedges", len(intra_cluster_edges_numedges), len(intra_cluster_edges_numedges[0]), len(intra_cluster_edges_numedges[1]))
        intra_cluster_edges = [
            intra_cluster_edges_numedges[i][0]
            for i in range(len(intra_cluster_edges_numedges))
        ]
        num_edges = [
            intra_cluster_edges_numedges[i][1]
            for i in range(len(intra_cluster_edges_numedges))
        ]
        # print("***num_edges", num_edges)

        clustered_adj = pool.starmap(
            cluster_graph,
            [
                (
                    self.num_clusters,
                    intra_cluster_edges[i],
                    cluster_ind[i],
                    num_edges[i],
                )
                for i in range(len(intra_cluster_edges))
            ],
        )

        P_opt = WeightedAdj_PF(clustered_adj, features)

        PPA = pool.starmap(
            binarized_weighted_P_adj,
            [(clustered_adj[i], P_opt, i) for i in range(len(clustered_adj))],
        )

        P_WS = [PPA[i][0] for i in range(len(PPA))]
        P_set = [PPA[i][1] for i in range(len(PPA))]
        A_list = [PPA[i][2] for i in range(len(PPA))]
        pool.close()
        pool.join()

        feat_dict = {}
        cluster_indmatrix = []
        cnt = 0
        for i in range(len(cluster_ind)):
            tmp = []
            for k in range(len(cluster_ind[i])):
                feat_dict[cnt] = cluster_ind[i][k]
                tmp.append(cnt)
                cnt = cnt + 1
            cluster_indmatrix.append(tmp)

        # print("*****first step done!")
        pool_size = mp.cpu_count()
        pool = mp.Pool(
            processes=pool_size,
        )
        cls_align_sub_graph = pool.starmap(
            aligned_subgraph,
            [
                (graph_train[i], features, P_set[i], cluster_indmatrix[i], feat_dict)
                for i in range(len(features))
            ],
        )

        cluster_align = [
            cls_align_sub_graph[i][0] for i in range(len(cls_align_sub_graph))
        ]
        sub_list = [cls_align_sub_graph[i][1] for i in range(len(cls_align_sub_graph))]

        pool.close()
        pool.join()

        A_subfinal = []
        P_subfinal = []
        A0_subfinal = []

        for k in range(self.num_clusters):
            A1 = [nx.to_numpy_matrix(sub_list[i][k]) for i in range(len(sub_list))]

            if self.distance_type == "CVX_clustered":
                # print("len(A1[0]), len(A1[1])", len(A1[0]), len(A1[1]))
                P_f = P_Fest(A1[0], A1[1])
            A_subfinal.append(A1[1])
            # print("A dim", len(A1[0]), A1[0][0])
            A0_subfinal.append(A1[0])
            P_subfinal.append(P_f)
            # print("P_f", P_f)
        P_set1 = []
        A_list1 = []
        A_aligned_list = []
        P_bin_list = []
        P_list = []

        # for k in range(len(A_subfinal)):
        # print("len(A_subfinal, A_subfinal[k], len(A_subfinal[k][0]), len(P_subfinal), len(P_subfinal[k]), len(P_subfinal[k][0])", len(A_subfinal), len(A_subfinal[k][0]), len(A_subfinal[k]), len(P_subfinal), len(P_subfinal[k]),len(P_subfinal[k][0]) )
        pool_size = mp.cpu_count()
        pool = mp.Pool(
            processes=pool_size,
        )
        PPA_sub = pool.starmap(
            binarized_P_adjF,
            [(A_subfinal[k], P_subfinal[k], k) for k in range(len(A_subfinal))],
        )
        P_W_list = [PPA_sub[i][0] for i in range(len(PPA_sub))]
        P_sub_list = [PPA_sub[i][1] for i in range(len(PPA_sub))]
        A_sub_list = [PPA_sub[i][2] for i in range(len(PPA_sub))]
        pool.close()
        pool.join()
        A_aligned_list.append(A_sub_list)
        P_bin_list.append(P_sub_list)
        P_list.append(P_W_list)

        sparse_A, sprs_G = sparse_adjF(graph_train, self.num_tr, intra_cluster_edges)

        if self.distance_type == "CVX_clustered":
            # A_list_sprs= align_sparseF(sparse_A, sprs_G, self.max_nodes)
            A_list_sprs = sparse_A

        n = len(graph_train[1].nodes())
        A_tmp = np.zeros((n, n))
        P_rec = np.zeros((n, n))
        P_Wrec = np.zeros((n, n))
        # A_rec= np.zeros((n, n))
        print("len(A_list_sprs)", len(A_list_sprs))
        print(
            "len(P_bin_list)",
            len(P_bin_list),
            len(P_bin_list[0]),
            len(P_bin_list[0][0]),
        )
        print(
            "len(cluster_align), len(cluster_align[0])",
            len(cluster_align),
            len(cluster_align[0]),
        )
        print("len(P_list), len(P_list[0])", len(P_list), len(P_list[0]))
        A_rec = A_list_sprs[1]

        for i in range(num_clusters):
            # print("P_bin_list[i][0]", P_bin_list[i][0])
            a = np.nonzero(P_bin_list[0][i])
            for j in range(len(a[0])):
                P_rec[
                    cluster_align[1][i][list(a[0])[j]],
                    cluster_align[0][i][list(a[1])[j]],
                ] = 1

        print("***sum P_rec***", np.sum(P_rec))
        for i in range(num_clusters):
            a = np.nonzero(P_list[0][i])
            for j in range(len(a[0])):
                P_Wrec[
                    cluster_align[1][i][list(a[0])[j]],
                    cluster_align[0][i][list(a[1])[j]],
                ] = P_list[0][i][[list(a[0])[j]], [list(a[1])[j]]]

        A_tmp = A_list_sprs[1]

        pool_size = mp.cpu_count()
        pool = mp.Pool(
            processes=pool_size,
        )
        tmp_vecs = pool.starmap(
            binarized_vec,
            [(A_tmp[i], 0.2) for i in range(len(A_tmp))],
        )
        pool.close()
        pool.join()
        A_tmp = np.asarray([tmp_vecs[k] for k in range(len(A_tmp))])

        A_sprs_aligned = np.matmul(np.matmul(P_rec.T, A_tmp), P_rec)

        print(
            "number of non-zero elements after alignment",
            np.count_nonzero(A_sprs_aligned),
            np.count_nonzero(P_rec),
        )

        # tmp= np.nonzero(A_sprs_aligned)
        A_rec = A_sprs_aligned

        for i in range(num_clusters):
            print(
                "len(A_aligned_list), len(A_aligned_list[0][i]), len(A_aligned_list[0][i][1][0])",
                len(A_aligned_list),
                len(A_aligned_list[0][i]),
                len(A_aligned_list[0][i][1][0]),
            )

            pool_size = mp.cpu_count()
            pool = mp.Pool(
                processes=pool_size,
            )
            tmp_vecs = pool.starmap(
                binarized_vec,
                [
                    (np.asarray(A_aligned_list[0][i])[j], 0.2)
                    for j in range(len(A_aligned_list[0][i]))
                ],
            )
            pool.close()
            pool.join()
            bin_adj = np.asarray(
                [tmp_vecs[k] for k in range(len(A_aligned_list[0][i]))]
            )
            print("unique elements:", np.unique(bin_adj))
            # a= np.nonzero(A_aligned_list[i][0])
            a = np.nonzero(bin_adj)
            for j in range(len(a[0])):
                A_rec[
                    cluster_align[0][i][list(a[0])[j]],
                    cluster_align[0][i][list(a[1])[j]],
                ] = 1
                A_rec[
                    cluster_align[0][i][list(a[1])[j]],
                    cluster_align[0][i][list(a[0])[j]],
                ] = 1

        pool_size = mp.cpu_count()
        pool = mp.Pool(
            processes=pool_size,
        )
        tmp_vecs = pool.starmap(
            binarized_vec,
            [(A_rec[i], 0.2) for i in range(len(A_rec))],
        )
        pool.close()
        pool.join()
        A_total = np.asarray([tmp_vecs[k] for k in range(len(A_rec))])

        # A_total= A_rec
        P_total = P_rec
        # A_total, P_total, P_totalW, A_sprs_aligned = recover_A_P(graph_train[1], A_list_sprs[1], A_aligned_list, cluster_align, intra_cluster_edges, P_bin_list, P_list, self.num_clusters, 1)
        print("len(graph_train)", len(graph_train))

        return A_total, P_total


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--center", type=str2bool, default=True)
    parser.add_argument("--data", type=str)
    args = parser.parse_args()

    with open(args.data, "rb") as f:

        graphs = pickle.load(f, encoding="latin1")

    for i in range(len(graphs)):
        print(
            "len(graphs[i].nodes()), len(graphs[i].edges())",
            len(graphs[i].nodes()),
            len(graphs[i].edges()),
        )

    print(
        "len(graphs)",
        len(graphs),
        len(graphs[0].nodes()),
        len(graphs[1].nodes()),
        len(graphs[0].edges()),
        len(graphs[1].edges()),
    )

    itr = 5  # number of iterations in Fermat alternating minimization
    num_clusters = 5
    n = 2  # number of graphs in each small group
    num_tr = 4  # total number of small groups

    graph_train = graphs[0 : 0 + num_tr * n]

    A = []
    nodes_num = []
    for i in range(len(graph_train)):
        A.append(nx.to_numpy_matrix(graph_train[i]))
        print("graph_train nodes", graph_train[i].number_of_nodes())
        nodes_num.append(graph_train[i].number_of_nodes())

    print("len(graph_train), len(A)", len(graph_train), len(A))

    print("nodes_num, max(nodes_num)", nodes_num, max(nodes_num))
    node_ind = np.where(np.array(nodes_num) < max(nodes_num))[0].tolist()
    print("node_ind", node_ind)

    nodes_num.append(130)
    if len(node_ind) > 0:
        for i in range(len(graph_train)):
            adj_tmp = np.zeros((max(nodes_num), max(nodes_num)))
            adj_tmp[0 : len(A[i]), 0 : len(A[i])] = A[i]
            if len(A[i]) < max(nodes_num):
                adj_tmp[len(A[i]) :, :] = 0.01
                adj_tmp[:, len(A[i]) :] = 0.01
            A[i] = adj_tmp
            # print("i, len(adj_tmp)", i, len(adj_tmp))
            graph_train[i] = nx.from_numpy_matrix(adj_tmp)

    graph_train = graph_train[0 : 0 + num_tr * n]

    print("len(graph_train), len(A)", len(graph_train), len(A))

    kmeans = KMeansConstrained(n_clusters=5, size_min=26, size_max=26, random_state=0)
    # kmeans = KMeans(n_clusters=num_clusters, embedding_method=GSVD(3))
    # clf.fit_predict(X)
    # A_cent_rec, A_total, P_total= final_graphs(A, graph_train, num_clusters, num_tr, 0.7, kmeans)

    if args.center:
        # Cluster and parallelization
        print("Cluster and parallelization")
        t0 = time.time()
        thr = 0.5

        obj = AP_cent(
            max(nodes_num),
            num_clusters,
            num_tr,
            thr,
            itr,
            distance_type="CVX_clustered",
        )

        pool_size = mp.cpu_count()
        Process = NoDaemonProcess

        pool = MyPool(
            processes=pool_size,
        )
        AAP = pool.starmap(
            obj.final_graphs,
            [
                (
                    A[0 + i * num_tr : num_tr + i * num_tr],
                    graph_train[0 + i * num_tr : num_tr + i * num_tr],
                )
                for i in range(n)
            ],
        )
        pool.close()
        pool.join()

        A_cent_rec_cls = [AAP[i][0] for i in range(len(AAP))]
        G_cent_rec_cls = [AAP[i][1] for i in range(len(AAP))]
        print("G_cent_rec_cls[0]", G_cent_rec_cls[0].nodes())
        A_total_cls = [AAP[i][2] for i in range(len(AAP))]
        G_total_cls = [AAP[i][3] for i in range(len(AAP))]
        print("len(G_total_cls)", len(G_total_cls), len(G_total_cls[0]))
        for i in range(len(G_total_cls[0])):
            print("G_total_cls", G_total_cls[0][i].nodes())
        P_total = [AAP[i][4] for i in range(len(AAP))]

        num_tr = num_tr + 1
        while len(A_cent_rec_cls) > num_tr:
            print("Further parallelization")
            A_tr = A_cent_rec_cls
            obj = AP_cent(
                max(nodes_num),
                num_clusters,
                num_tr,
                thr,
                itr,
                distance_type="CVX_clustered",
            )
            pool_size = mp.cpu_count()
            Process = NoDaemonProcess

            pool = MyPool(
                processes=pool_size,
            )
            AAP = pool.starmap(
                obj.final_graphs,
                [
                    (
                        A_tr[0 + i * num_tr : num_tr + i * num_tr],
                        G_cent_rec_cls[0 + i * num_tr : num_tr + i * num_tr],
                    )
                    for i in range(int(len(A_cent_rec_cls) / num_tr))
                ],
            )
            pool.close()
            pool.join()

            A_cent_rec_cls = [AAP[i][0] for i in range(len(AAP))]
            G_cent_rec_cls = [AAP[i][1] for i in range(len(AAP))]
            A_total_cls = [AAP[i][2] for i in range(len(AAP))]
            G_total_cls = [AAP[i][3] for i in range(len(AAP))]
            P_total = [AAP[i][4] for i in range(len(AAP))]

        if len(A_cent_rec_cls) > 1:
            obj = AP_cent(
                max(nodes_num),
                num_clusters,
                len(A_cent_rec_cls),
                thr,
                itr,
                distance_type="CVX_clustered",
            )
            print("len(A_cent_rec_cls)", len(A_cent_rec_cls))
            (
                A_cent_rec_cls,
                G_cent_rec_cls,
                A_total_cls,
                G_total_cls,
                P_total,
            ) = obj.final_graphs(A_cent_rec_cls, G_cent_rec_cls)

        print("running time for parallel-cluster", time.time() - t0)

        np.save("A_cent_parall_fermat", A_cent_rec_cls)
        np.save("A_total_parall_fermat", A_total_cls)

        score = 0
        for i in range(len(A_total_cls)):
            print("||A-A0||", np.linalg.norm(A_total_cls[i] - A_cent_rec_cls))

            score = score + np.linalg.norm(A_total_cls[i] - A_cent_rec_cls)

        print(
            "performance score and the average score",
            score,
            np.mean(score) / (len(A_total_cls) * np.linalg.norm(A_cent_rec_cls)),
        )
        print("||A0||, ||A[0]||", np.linalg.norm(A_cent_rec_cls), np.linalg.norm(A[0]))

        print("A_cent_rec_cls", np.count_nonzero(A_cent_rec_cls))
        print("*" * 80)

    else:
        print("Computing alignment")
        t0 = time.time()
        kmeans = KMeansConstrained(
            n_clusters=5, size_min=26, size_max=26, random_state=0
        )
        # kmeans = KMeans(n_clusters=num_clusters, embedding_method=GSVD(3))
        A_cent_rec_cls = np.load("A_cent_parall_fermat.npy")
        print(
            "len(A_cent_rec_cls)",
            len(A_cent_rec_cls),
            len(A_cent_rec_cls[0]),
            A_cent_rec_cls,
        )
        ### Final alignment
        num_tr = 1

        print("len(graph_train), len(A)", len(graph_train), len(A))
        obj = AP_align(A_cent_rec_cls, num_clusters, num_tr, 0.5)

        pool_size = mp.cpu_count()

        print("pool_size", pool_size)
        Process = NoDaemonProcess

        pool = MyPool(
            processes=pool_size,
        )
        AAP = pool.starmap(
            obj.final_graphs,
            [([A[i]], [graph_train[i]]) for i in range(len(graph_train))],
        )
        pool.close()
        pool.join()

        A_final_cls = [AAP[i][0] for i in range(len(AAP))]
        P_final = [AAP[i][1] for i in range(len(AAP))]

        print("total running time", time.time() - t0)
        print("len(A_final_cls), len(P_final)", len(A_final_cls), len(P_final))

        print("total time", time.time() - t0)
        # np.save("A_align_fermat_large_EgoFinal", A_final_cls)
        score = 0
        for i in range(len(A_final_cls)):
            score = score + np.linalg.norm(A_final_cls[i][0] - A_cent_rec_cls)

        print(
            "Final performance score and the average score",
            score,
            np.mean(score) / (len(A) * np.linalg.norm(A_cent_rec_cls)),
        )
        print("||A0||, ||A[0]||", np.linalg.norm(A_cent_rec_cls), np.linalg.norm(A[0]))
