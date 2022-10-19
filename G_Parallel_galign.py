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
from sknetwork.clustering import Louvain
from sknetwork.data import karate_club
from sknetwork.path import shortest_path
from sknetwork.clustering import KMeans
from sknetwork.embedding import GSVD
from scipy.optimize import linear_sum_assignment
from k_means_constrained import KMeansConstrained
import time


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


def save_graph_list(G_list, fname):
    with open(fname, "wb") as f:
        pickle.dump(G_list, f)


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
        # print("num_nodes, k", len(cluster_ind[k]), k)
        edge_orig = set(edge_orig).difference(inter_edge)
        n_edges.append(G.subgraph(cluster_ind[k]).size(weight="weight"))

    return list(edge_orig), n_edges


def binarized_P_adj(A_cluster, P_opt, i):
    m = len(A_cluster)
    P = P_opt[m * i : m + m * i, 0:m]
    P_binary = np.zeros((m, m))
    P1 = -1 * P
    row_ind, col_ind = linear_sum_assignment(P1)
    P_binary[row_ind, col_ind] = 1
    adj = np.matmul(np.matmul(P_binary.T, A_cluster), P_binary)
    return P, P_binary, adj


def binarized_P_adjF(A, P_opt):
    m = len(A)
    P = P_opt
    P_binary = np.zeros((m, m))
    P1 = -1 * P
    row_ind, col_ind = linear_sum_assignment(P1)
    P_binary[row_ind, col_ind] = 1
    adj = np.matmul(np.matmul(P_binary.T, A), P_binary)
    return adj


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
    return A_cluster


def feat_cluster_build(adj):
    """Build feature matrices with number of nodes in each community as a feature and identify node indexes of each cluster."""
    # labels= kmeans.fit_transform(sparse.csr_matrix(adj))
    labels = kmeans.fit_predict(adj)
    unique_labels, counts = np.unique(labels, return_counts=True)
    features = list(counts)
    cls = []
    for k in range(num_clusters):
        cls.append(np.where(labels == unique_labels[k])[0].tolist())
    cluster_ind = cls
    return labels, features, cluster_ind


def P_est(A):
    """Computes graph alinmnet matrices via CVXPY"""

    m = len(A[0])
    n = len(A)
    P_opt = cp.Variable((m * n, m * n), symmetric=True)
    s = 0
    for i in range(len(A)):
        for j in range(len(A)):
            s = s + cp.norm(
                (
                    (A[i] @ P_opt[m * i : m + m * i, m * j : m + m * j])
                    - (P_opt[m * i : m + m * i, m * j : m + m * j] @ A[j])
                ),
                "fro",
            )
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
            # P_Wrec[cluster_align[ind][i][list(a[0])[j]], cluster_align[0][i][list(a[1])[j]]]= 1
    # tmp= np.nonzero(P_rec)
    for i in range(num_clusters):
        a = np.nonzero(P_list[i][ind])
        for j in range(len(a[0])):
            P_Wrec[
                cluster_align[ind][i][list(a[0])[j]], cluster_align[0][i][list(a[1])[j]]
            ] = P_list[i][ind][[list(a[0])[j]], [list(a[1])[j]]]

    A_tmp = A_sprs
    A_sprs_aligned = np.matmul(np.matmul(P_rec.T, A_tmp), P_rec)
    # A_sprs_aligned= sparse.csr_matrix(P_rec.T).multiply(sparse.csr_matrix(A_tmp)).multiply(sparse.csr_matrix(P_rec))
    # A_sprs_aligned= A_sprs_aligned.toarray()

    tmp = np.nonzero(A_sprs_aligned)
    A_rec = A_sprs_aligned

    for i in range(num_clusters):
        a = np.nonzero(A_aligned_list[i][ind])
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
        A_tmp = np.zeros((n, n))
        for i in range(len(intra_cluster_edges[j])):
            A_tmp[intra_cluster_edges[j][i][0], intra_cluster_edges[j][i][1]] = 1
            A_tmp[intra_cluster_edges[j][i][1], intra_cluster_edges[j][i][0]] = 1
        A_sparse.append(A_tmp)

    return A_sparse


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
            # D= np.reshape(D, (len(F[i]), len(F[j])) )
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


def final_graphs(A, graph_train, num_clusters, num_tr, A_thr, kmeans):
    print("len(A),len(graph_train)", len(A), len(graph_train))
    # kmeans = KMeans(n_clusters=num_clusters, embedding_method=GSVD(3))
    labels = []
    features = []
    cluster_ind = []

    pool_size = mp.cpu_count()
    pool = mp.Pool(
        processes=pool_size,
    )
    lab_feat_cls = pool.map(feat_cluster_build, [A[i] for i in range(len(A))])

    labels = [lab_feat_cls[i][0] for i in range(len(A))]
    features = [lab_feat_cls[i][1] for i in range(len(A))]
    cluster_ind = [lab_feat_cls[i][2] for i in range(len(A))]

    print("len(cluster_ind), len(graph_train)", len(cluster_ind), len(graph_train))
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
            (num_clusters, intra_cluster_edges[i], cluster_ind[i], num_edges[i])
            for i in range(len(intra_cluster_edges))
        ],
    )

    P_opt = WeightedAdj_P(clustered_adj, features)

    PPA = pool.starmap(
        binarized_P_adj,
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

    cluster_align = [cls_align_sub_graph[i][0] for i in range(len(cls_align_sub_graph))]
    sub_list = [cls_align_sub_graph[i][1] for i in range(len(cls_align_sub_graph))]

    pool.close()
    pool.join()

    A_subfinal = []
    P_subfinal = []
    for k in range(num_clusters):
        A1 = [nx.to_numpy_matrix(sub_list[i][k]) for i in range(len(sub_list))]
        P_Galign1 = P_est(A1)
        A_subfinal.append(A1)
        P_subfinal.append(P_Galign1)

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
            [(A_subfinal[k][i], P_subfinal[k], i) for i in range(len(A_subfinal[0]))],
        )
        P_W_list = [PPA_sub[i][0] for i in range(len(PPA_sub))]
        P_sub_list = [PPA_sub[i][1] for i in range(len(PPA_sub))]
        A_sub_list = [PPA_sub[i][2] for i in range(len(PPA_sub))]
        pool.close()
        pool.join()
        A_aligned_list.append(A_sub_list)
        P_bin_list.append(P_sub_list)
        P_list.append(P_W_list)

    sparse_A = sparse_adj(graph_train, num_tr, intra_cluster_edges)
    pool_size = mp.cpu_count()
    pool = mp.Pool(
        processes=pool_size,
    )
    P_Galign_sprs = P_est(sparse_A)
    PPA_sprs = pool.starmap(
        binarized_P_adj,
        [(sparse_A, P_Galign_sprs, i) for i in range(len(csparse_A))],
    )

    P_WS_sprs = [PPA_sprs[i][0] for i in range(len(PPA))]
    P_set_sprs = [PPA_sprs[i][1] for i in range(len(PPA))]
    A_list_sprs = [PPA_sprs[i][2] for i in range(len(PPA))]
    pool.close()
    pool.join()

    print(
        "len(A_aligned_list), len(A_aligned_list[0])",
        len(A_aligned_list),
        len(A_aligned_list[0]),
    )
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
                num_clusters,
                ind,
            )
            for ind in range(num_tr)
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

    A_cent_sprs = center_clsgraph(A_list_sprs)

    pool_size = mp.cpu_count()
    pool = mp.Pool(
        processes=pool_size,
    )
    bin = pool.starmap(
        binarized_vec,
        [(A_cent_sprs[i], A_thr) for i in range(len(A_cent_sprs))],
    )
    pool.close()
    pool.join()

    A_cent_sprs = [bin[i] for i in range(len(A_cent_sprs))]
    tmp = np.nonzero(A_cent_sprs)

    A_cent_rec = np.zeros((len(A_cent_sprs), len(A_cent_sprs)))
    # print("len(A_cent), len(A_cent_sprs)", len(A_cent), len(A_cent_sprs), np.linalg.norm(A_cent[0]), np.linalg.norm(A_cent[1]), np.linalg.norm(A_cent[2]))

    for i in range(num_clusters):
        pool_size = mp.cpu_count()
        pool = mp.Pool(
            processes=pool_size,
        )
        tmp_vecs = pool.starmap(
            binarized_vec,
            [(A_cent[i][k], A_thr) for k in range(len(A_cent[i]))],
        )
        pool.close()
        pool.join()
        A_cent[i] = [tmp_vecs[k] for k in range(len(A_cent[i]))]

        a = np.nonzero(A_cent[i])
        for j in range(len(a[0])):
            A_cent_rec[
                cluster_align[0][i][list(a[0])[j]], cluster_align[0][i][list(a[1])[j]]
            ] = 1

    for i in range(len(tmp[0])):
        A_cent_rec[list(tmp[0])[i], list(tmp[1])[i]] = 1

    return A_cent_rec, A_total, P_total


def P_Fest(A0, A):
    m = len(A)
    # P = []
    s = 0
    print("m", m)
    P_opt = cp.Variable((m, m))
    s = cp.norm(((A @ P_opt) - (P_opt @ A0)), "fro") ** 2
    constraints = [P_opt >= 0]
    constraints = [sum(P_opt) == 1]
    constraints += [sum(P_opt).T == 1]
    prob = cp.Problem(cp.Minimize(s), constraints)
    prob.solve(solver=cp.SCS)
    # P.append(P_opt.value)

    return P_opt.value


class AP_align(object):
    """
    The Frank Wolfe class
    To solve \sum_{i,j \in [n]} ||A_iP_ij-P_ijA_j||_{2}^{2}+tr(P^{T}D) via Frank-Wolfe.
    """

    def __init__(self, A0, num_tr, A_thr):

        self.num_tr = num_tr
        self.A_thr = A_thr
        self.A0 = A0
        Process = NoDaemonProcess

    def final_graphs(self, A, graph_train):

        n = len(graph_train[1].nodes())

        print(
            "number of non-zero elements after alignment",
            np.count_nonzero(A_sprs_aligned),
            np.count_nonzero(P_rec),
        )

        # tmp= np.nonzero(A_sprs_aligned)
        A_rec = A_sprs_aligned
        # A_rec= np.zeros((n,n))
        # print("A_aligned_list[i]", A_aligned_list[0])
        # print("A_aligned_list[0][i], A_aligned_list[0][i][0]", np.asarray(A_aligned_list[0][i]), np.asarray(A_aligned_list[0][i])[0])
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
                    (np.asarray(A_aligned_list[0][i])[j], 0.3)
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
                # A_rec[cluster_align[1][i][list(a[0])[j]], cluster_align[1][i][list(a[1])[j]]]= 1
                # A_rec[cluster_align[1][i][list(a[1])[j]], cluster_align[1][i][list(a[0])[j]]]= 1
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
            [(A_rec[i], 0.3) for i in range(len(A_rec))],
        )
        pool.close()
        pool.join()
        A_total = np.asarray([tmp_vecs[k] for k in range(len(A_rec))])

        # A_total= A_rec
        P_total = P_rec
        # A_total, P_total, P_totalW, A_sprs_aligned = recover_A_P(graph_train[1], A_list_sprs[1], A_aligned_list, cluster_align, intra_cluster_edges, P_bin_list, P_list, self.num_clusters, 1)
        print("len(graph_train)", len(graph_train))
        # print("len(A_total) before", A_total)
        # A_total= A_total[1]
        # print("len(A_total) after", len(A_total), len(A_total[0]))

        return A_total, P_total, A_sprs_aligned


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

    print("len(graphs)", len(graphs), len(graphs[1].nodes()), len(graphs[1].edges()))

    num_tr = 4  # number of graphs in each small group
    n = 2  # total number of small groups

    # center = True
    graph_train = graphs[0 : 0 + num_tr * n]
    A = []
    nodes_num = []
    for i in range(len(graph_train)):
        A.append(nx.to_numpy_matrix(graph_train[i]))
        nodes_num.append(graph_train[i].number_of_nodes())

    print("max nodes", nodes_num, max(nodes_num))
    node_ind = np.where(np.array(nodes_num) < max(nodes_num))[0].tolist()

    if len(node_ind) > 0:
        for i in range(len(node_ind)):
            adj_tmp = np.zeros((max(nodes_num), max(nodes_num)))
            adj_tmp[0 : len(A[node_ind[i]]), 0 : len(A[node_ind[i]])] = A[node_ind[i]]
            adj_tmp[len(A[node_ind[i]]) :, :] = 0.01
            adj_tmp[:, len(A[node_ind[i]]) :] = 0.01
            A[node_ind[i]] = adj_tmp
            graph_train[node_ind[i]] = nx.from_numpy_matrix(adj_tmp)

    print("A_orig", np.linalg.norm(A[1]))

    if args.center:
        # Group graphs
        print("Compute center graph")
        t0 = time.time()
        pool_size = mp.cpu_count()
        pool = mp.Pool(
            processes=pool_size,
        )
        P_opt = pool.map(
            P_est, [A[0 + i * num_tr : num_tr + i * num_tr] for i in range(n)]
        )
        A_cent_orig = pool.starmap(
            center_graph,
            [(A[0 + i * num_tr : num_tr + i * num_tr], P_opt[i]) for i in range(n)],
        )
        pool.close()
        pool.join()
        A_cent_orig = [np.asarray(A_cent_orig[i]) for i in range(len(A_cent_orig))]

        tmp = []
        # print("step 1, A_cent_orig", A_cent_orig)
        for k in range(len(A_cent_orig)):
            pool_size = mp.cpu_count()
            pool = mp.Pool(
                processes=pool_size,
            )
            bin = pool.starmap(
                binarized_vec,
                [(A_cent_orig[k][i], 0.5) for i in range(len(A_cent_orig[k]))],
            )
            pool.close()
            pool.join()
            # print("bin", len(bin), len(bin[0]))
            # A_cent_orig[k]= np.asarray([bin[i] for i in range(len(A_cent_orig[k]))])
            tmp.append(np.asarray([bin[i] for i in range(len(A_cent_orig[k]))]))

        A_cent_orig = tmp
        print("large group graph totoal time", time.time() - t0)
        print("len(A_cent_orig)", len(A_cent_orig), A_cent_orig[0])

        num_tr = num_tr + 1
        while len(A_cent_orig) > num_tr + 1:
            pool = mp.Pool(
                processes=pool_size,
            )
            P_opt = pool.map(
                P_est,
                [
                    A_cent_orig[0 + i * num_tr : num_tr + i * num_tr]
                    for i in range(int(len(A_cent_orig) / num_tr))
                ],
            )
            A_first1 = pool.starmap(
                center_graph,
                [
                    (A_cent_orig[0 + i * num_tr : num_tr + i * num_tr], P_opt[i])
                    for i in range(int(len(A_cent_orig) / num_tr))
                ],
            )
            A_cent_orig = [np.asarray(A_first1[i]) for i in range(len(A_first1))]
            print("len(A_cent_orig)", len(A_cent_orig), len(A_cent_orig[0]))
            for k in range(len(A_cent_orig)):
                print("np.asarray(A_cent_orig[k])", np.asarray(A_cent_orig[k]))
                pool_size = mp.cpu_count()
                pool = mp.Pool(
                    processes=pool_size,
                )
                bin = pool.starmap(
                    binarized_vec,
                    [
                        (np.asarray(A_cent_orig[k])[i], 0.5)
                        for i in range(len(A_cent_orig[k]))
                    ],
                )
                pool.close()
                pool.join()
                A_cent_orig[k] = [bin[i] for i in range(len(A_cent_orig[k]))]
                A_cent_orig[k] = np.asarray(A_cent_orig[k])

        print("Center and alignment total time", time.time() - t0)
        print("len(A_cent_orig)", len(A_cent_orig))

        P_opt = P_est(A_cent_orig)
        A_cent_final = center_graph(A_cent_orig, P_opt)

        pool_size = mp.cpu_count()
        pool = mp.Pool(
            processes=pool_size,
        )
        bin = pool.starmap(
            binarized_vec,
            [(A_cent_final[i], 0.5) for i in range(len(A_cent_final))],
        )
        pool.close()
        pool.join()
        A_cent_final = np.array([bin[i] for i in range(len(A_cent_final))])
        print("len(A_cent_final)", len(A_cent_final), A_cent_final)

        np.save("A_cent_smallcomm", A_cent_final)

    else:
        print("Compute alignments")
        A_cent_final = np.load("A_cent_smallcomm.npy")
        print("len(A)", len(A))
        A_cent_final = [A_cent_final] * len(A)
        print("A_cent_final[0]", np.asarray(A_cent_final[0]))

        # num_tr= 1

        t0 = time.time()
        pool_size = mp.cpu_count()
        pool = mp.Pool(
            processes=pool_size,
        )
        P_opt = pool.starmap(
            P_Fest,
            [(np.asarray(A_cent_final[i]), np.asarray(A[i])) for i in range(len(A))],
        )
        pool.close()
        pool.join()
        print("Final alignment step 1")
        print("len(P_opt), len(P_opt[0])", len(P_opt), len(P_opt[0]), len(P_opt[0][0]))
        pool_size = mp.cpu_count()
        pool = mp.Pool(
            processes=pool_size,
        )
        A_align = pool.starmap(
            binarized_P_adjF, [(A[i], np.asarray(P_opt[i])) for i in range(len(A))]
        )
        pool.close()
        pool.join()
        print("Final alignment step 2")
        for k in range(len(A_align)):
            # print("len(A_align[k]), len(A_align[k][0])", len(A_align[k]), len(np.asarray(A_align[k])[0]), A_align[k])
            pool_size = mp.cpu_count()
            pool = mp.Pool(
                processes=pool_size,
            )
            bin = pool.starmap(
                binarized_vec,
                [(np.asarray(A_align[k])[i], 0.5) for i in range(len(A_align[k]))],
            )
            pool.close()
            pool.join()
            A_align[k] = np.asarray([bin[i] for i in range(len(A_align[k]))])

        # print("A_align", A_align)
        G_align = [nx.from_numpy_matrix(A_align[i]) for i in range(len(A_align))]
        for i in range(len(G_align)):
            g = G_align[i]
            g.remove_nodes_from(list(nx.isolates(g)))
            G_align[i] = g
        # G_align= [G_align[i].remove_nodes_from(list(nx.isolates(G_align[i]))) for i in range(len(G_align))]

        # print("G_align", G_align)
        for i in range(len(G_align)):
            print("# of nodes", len(G_align[i].nodes()))

        print("Final alignment: total time", time.time() - t0)
        np.save("A_align_SmallComm_noise5", A_align)
        save_graph_list(G_align, "G_align_Smallcomm_noise5")
