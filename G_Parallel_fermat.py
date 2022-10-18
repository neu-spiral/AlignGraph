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
from fermat_distance import Fermat_distance
from sknetwork.clustering import Louvain
from sknetwork.data import karate_club
from sknetwork.path import shortest_path
from sknetwork.clustering import KMeans
from sknetwork.embedding import GSVD
from scipy.optimize import linear_sum_assignment
from k_means_constrained import KMeansConstrained
from validation_center import validation_center
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


def binarized_P_adjF(A_cluster, P_opt):
    m = len(A_cluster)
    P = P_opt[0]
    P_binary = np.zeros((m, m))
    P1 = -1 * P
    row_ind, col_ind = linear_sum_assignment(P1)
    P_binary[row_ind, col_ind] = 1
    adj = np.matmul(np.matmul(P_binary.T, A_cluster), P_binary)
    return P, P_binary, adj


def opt_thr(graph_list, A_first, min_deg, thr):
    """opt_thr finds the best thr
    for binarizing the decoder's output."""
    print("min_deg", min_deg)

    new_adj = A_first
    graph_list0 = []
    for l in range(len(new_adj)):
        G = nx.from_numpy_matrix(new_adj[l])
        G.remove_nodes_from(list(nx.isolates(G)))
        print("# of nodes", len(G.nodes()))
        graph_list0.append(G)
    test_result = validation_center(graph_list, graph_list0, thr)
    return test_result.s1, test_result.s2, thr, graph_list0, new_adj


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


def binarized_vec(vec, thr):
    tmp = vec
    tmp = [0 if tmp_ < thr else 1 for tmp_ in tmp]
    vec = tmp
    return vec


def center_cls_ave(A):
    # Computes the center of a group of clusters by taking their average.

    return (1 / (len(A))) * sum(A)


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
    args = parser.parse_args()

    with open("test_grid_graphs", "rb") as f:

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

    n = 2  # total number of small groups
    num_tr = 4  # number of graphs in each small group
    iter_num = 5  # number of iterations in Fermat alternation minimization

    graph_train = graphs[0 : 0 + num_tr * n]
    # graph_train= graphs

    A = []
    nodes_num = []
    min_degs = []
    for i in range(len(graph_train)):
        A.append(nx.to_numpy_matrix(graph_train[i]))
        print("graph_train nodes", graph_train[i].number_of_nodes())
        nodes_num.append(graph_train[i].number_of_nodes())
        g = graph_train[i]
        deg = [g.degree[j] for j in g.nodes()]
        min_degs.append(min(deg))

    min_deg = min(min_degs)
    print("len(graph_train), len(A)", len(graph_train), len(A))

    print("nodes_num, max(nodes_num)", nodes_num, max(nodes_num))
    node_ind = np.where(np.array(nodes_num) < max(nodes_num))[0].tolist()
    print("node_ind", node_ind)

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

    if args.center:
        # Group graphs
        print("Compute center graph")
        fermat = Fermat_distance(iter_num=iter_num, epsilon=10 ** (-10))
        t0 = time.time()

        pool_size = mp.cpu_count()
        pool = mp.Pool(
            processes=pool_size,
        )
        P = pool.map(
            fermat.AM, [A[0 + i * num_tr : num_tr + i * num_tr] for i in range(n)]
        )
        pool.close()
        pool.join()
        print("parallel time", time.time() - t0)
        P_first = [P[i][0] for i in range(len(P))]
        A_first = [P[i][1] for i in range(len(P))]

        np.save("A_first", A_first)
        print("first step time", time.time() - t0)

        new_adj = A_first
        num_tr = num_tr + 1
        # Compute graph distance for FLAGS.num_batch/n  whith n graphs in each batch via F-W and optimal threshold for central graphs.
        while len(new_adj) > num_tr:

            pool = MyPool(
                processes=pool_size,
            )
            P_opt = pool.map(
                fermat.AM,
                [
                    new_adj[0 + i * num_tr : num_tr + i * num_tr]
                    for i in range(int(len(new_adj) / num_tr))
                ],
            )
            P_first = [P_opt[i][0] for i in range(len(P_opt))]
            A_first = [P_opt[i][1] for i in range(len(P_opt))]
            pool_size = mp.cpu_count()
            pool = MyPool(
                processes=pool_size,
            )

            new_adj = A_first

        P_fermat, A0 = Fermat_distance(iter_num=iter_num, epsilon=10 ** (-3)).AM(
            new_adj
        )
        A_cent_final = A0
        np.save("A_cent_final", A_cent_final)

    else:
        print("Compute alignments")
        t0 = time.time()
        A_cent_final = np.load("A_cent_final.npy")
        print("len(A)", len(A))
        A_cent_final = [A_cent_final] * len(A)
        print("A_cent_final[0]", np.asarray(A_cent_final[0]))

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
        adj_fin = [A_align[i][2] for i in range(len(A_align))]
        print("Final alignment step 2")
        print("A_align", len(A_align), np.asarray(A_align[0]))
        for k in range(len(A_align)):
            # print("len(A_align[k]), len(A_align[k][0])", len(A_align[k]), len(np.asarray(A_align[k])[0]), A_align[k])
            pool_size = mp.cpu_count()
            pool = mp.Pool(
                processes=pool_size,
            )
            bin = pool.starmap(
                binarized_vec,
                [(np.asarray(adj_fin[k])[i], 0.5) for i in range(len(adj_fin[k]))],
            )
            pool.close()
            pool.join()
            adj_fin[k] = np.asarray([bin[i] for i in range(len(adj_fin[k]))])
        A_align = adj_fin
        # print("A_align", A_align)
        G_align = [nx.from_numpy_matrix(A_align[i]) for i in range(len(A_align))]
        for i in range(len(G_align)):
            g = G_align[i]
            g.remove_nodes_from(list(nx.isolates(g)))
            G_align[i] = g

        for i in range(len(G_align)):
            print("# of nodes", len(G_align[i].nodes()))

        print("Final alignment: total time", time.time() - t0)
        np.save("A_align_fermat_SmallGrid", A_align)
        # save_graph_list(G_align, "G_align_fermat_SmallGrid")
        score = 0
        for i in range(len(A_align)):
            score = score + np.linalg.norm(A_align[i] - A_cent_final)

        print(
            "Final performance score and the average score",
            score,
            np.mean(score) / (len(A) * np.linalg.norm(A_cent_final)),
        )
        print("||A0||, ||A[0]||", np.linalg.norm(A_cent_final), np.linalg.norm(A[0]))
