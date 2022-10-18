import numpy as np
import networkx as nx
import time
import scipy
import cvxpy as cp
import argparse
import pickle
from scipy.optimize import check_grad
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import multiprocessing
import multiprocessing as mp
from multiprocessing import Pool
from multiprocessing import Process, Queue
import multiprocessing.pool


def V_dot(v):
    """
    The F-W algorithm to solve:
    min <S, V>
    subject to S>=0, S1=1, 1^{T}S=1^{T}
    """
    row_ind, col_ind = linear_sum_assignment(v)
    b = np.zeros(v.shape)
    b[row_ind, col_ind] = 1
    return b


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def binarized_vec(vec, thr):
    tmp = vec
    tmp = [0 if tmp_ < thr else 1 for tmp_ in tmp]
    vec = tmp
    return vec


class Fermat_distance(object):
    """
    Solve \sum_{i,j \in [n]} ||A_iP_i-P_iA_0||_{2}^{2}+tr(P^{T}D) using cvxpy
    """

    def __init__(self, iter_num, epsilon):
        # self.A = A
        self.iter_num = iter_num
        self.epsilon = epsilon
        # self.A0, self.P_opt=self.AM()
        Process = NoDaemonProcess

    def P_est(self, A0, A):
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

    def center_graph(self, P_opt, A):
        a = A
        m = len(a[0])
        n = len(A)
        A0 = cp.Variable((m, m))
        constraints = [-A0[0, 0] <= 0]
        constraints += [A0[0, 0] <= 1]
        s = 0
        for j in range(len(A)):
            s = s + cp.norm(np.matmul(A[j], P_opt[j]) - (P_opt[j] @ A0), "fro") ** 2

        for e in range(m):
            for f in range(m):
                constraints += [A0[e, f] <= 1]
                constraints += [-A0[e, f] <= 0]
        prob = cp.Problem(cp.Minimize(s), constraints)
        prob.solve(solver=cp.SCS)

        return A0.value

    def center_ave(self, P_opt, A, thr):
        """Computes the center of a group of clusters by taking their average."""

        for j in range(len(A)):
            m = len(A[j])
            P = P_opt[j]
            P_binary = np.zeros((m, m))
            P1 = -1 * P
            row_ind, col_ind = linear_sum_assignment(P1)
            P_binary[row_ind, col_ind] = 1
            A[j] = np.matmul(np.matmul(P_binary.T, A[j]), P_binary)

        A_cent = (1 / (len(A))) * sum(A)
        print("unique elements:", np.unique(A_cent))
        # thr= 0.5
        A_cent = np.asarray(
            [binarized_vec(np.asarray(A_cent)[i], thr) for i in range(len(A_cent))]
        )
        s = 0
        for i in range(len(A)):
            s = s + np.linalg.norm(A[i] - A_cent)

        print("center accuracy", s, s / (len(A) * np.linalg.norm(A_cent)))

        return (1 / (len(A))) * sum(A)

    def AM(self, A):
        start_time = time.time()
        m = len(A[0])
        n = len(A)
        P_opt = self.P_est(A[0], A)
        cost_old = -100
        itr = 0
        while True:
            # A0 = self.center_graph(P_opt, A)
            A0 = self.center_ave(P_opt, A, 0.5)
            P_opt = self.P_est(A0, A)
            cost = 0
            for j in range(len(A)):
                cost = cost + np.linalg.norm(
                    (np.matmul(A[j], P_opt[j]) - np.matmul(P_opt[j], A0)), "fro"
                )
            print("abs(cost-cost_old", abs(cost - cost_old))
            if abs(cost - cost_old) <= self.epsilon:
                break
            else:
                pass
            if itr <= self.iter_num:
                pass
            else:
                break
            itr = itr + 1
            cost_old = cost
            print("cost", cost)
            print("itr", itr)
        print("time  %s seconds " % (time.time() - start_time))
        return P_opt, A0


class Fermat_Frank_Wolfe(object):
    """
    The Frank Wolfe class
    To solve \sum_{i,j \in [n]} ||A_iP_ij-P_ijA_j||_{2}^{2}+tr(P^{T}D) via Frank-Wolfe.
    """

    def __init__(self, P, epsilon, itr_num):
        self.P = P
        self.epsilon = epsilon
        self.itr_num = itr_num
        Process = NoDaemonProcess
        print("HI")

    def s_score(self, A, A0):
        s = 0
        n = len(A)

        for j in range(len(A)):
            b = np.matmul(self.P[j], A0)
            s = (
                s
                + np.linalg.norm(
                    (np.matmul(A[j], self.P[j]) - np.matmul(self.P[j], A0)), "fro"
                )
                ** 2
            )

        return s

    def center_graph(self, A):
        a = A
        m = len(a[0])
        n = len(A)
        A0 = cp.Variable((m, m))
        constraints = [-A0[0, 0] <= 0]
        constraints += [A0[0, 0] <= 1]
        s = 0
        for j in range(len(A)):
            s = s + cp.norm(np.matmul(A[j], self.P[j]) - (self.P[j] @ A0), "fro") ** 2
            # s = s + cp.norm(np.matmul(A[j],P_bin[j])-(P_bin[j]@A0) , 'fro')**2
        for e in range(m):
            for f in range(m):
                constraints += [A0[e, f] <= 1]
                constraints += [-A0[e, f] <= 0]
        prob = cp.Problem(cp.Minimize(s), constraints)
        prob.solve(solver=cp.SCS)

        return A0.value

    def P_opt(self, A, A0, nabla_P):

        n = len(A0)
        # print("A0 length", n)
        S_optimal = cp.Variable((n, n))
        constraints = [-S_optimal[0, 0] <= 0]
        constraints += [S_optimal[0, 0] <= 1]
        for i in range(len(A0)):
            for j in range(len(A0)):
                constraints += [S_optimal[i, j] <= 1]
                constraints += [-S_optimal[i, j] <= 0]
            constraints += [sum(S_optimal[i]) == 1]
            constraints += [sum(S_optimal.T[i]) == 1]
        prob = cp.Problem(cp.Minimize(cp.trace(S_optimal.T @ nabla_P)), constraints)
        prob.solve(solver=cp.SCS, verbose=True)
        return S_optimal.value

    def Derivative(self, A, A0):
        n = len(A[0])
        # print("n",n)
        grad_P = []
        for i in range(len(A)):

            L = np.matmul(A[i], self.P[i]) - np.matmul(self.P[i], A0)

            s_P = (
                np.linalg.norm((np.matmul(A[i], self.P[i]) - np.matmul(self.P[i], A0)))
                ** 2
            )
            grad_Pi = 2 * (np.matmul(A[i].T, L) - np.matmul(L, A0.T))

            grad_P.append(np.asarray(grad_Pi))
        return grad_P

    def find_gamma(self, A, A0, k, S_optimal, gamma):
        self.P[k] = (1 - gamma) * self.P[k] + gamma * S_optimal[k]
        s = (
            np.linalg.norm(
                (np.matmul(A[k], self.P[k]) - np.matmul(self.P[k], A0)), "fro"
            )
            ** 2
        )
        return s

    def iteration(self, A):
        itr = 0
        n = len(A)
        A0 = self.center_graph(A)
        # print(A[0])
        s_old = self.s_score(A, A0) + 100
        print("s_old", s_old)
        while True:
            start_time = time.time()
            nabla_P = self.Derivative(A, A0)
            a0 = [A0] * len(A)
            pool_size = mp.cpu_count()
            pool = mp.Pool(
                processes=pool_size,
            )
            S_optimal = pool.starmap(
                self.P_opt, [(A[j], a0[j], nabla_P[j]) for j in range(n)]
            )
            pool.close()
            pool.join()
            delta_P = [(S_optimal[j] - self.P[j]) for j in range(n)]

            for k in range(len(A)):
                gamma = list(np.linspace(0.01, 1.0, num=50))
                pool_size = mp.cpu_count()
                pool = mp.Pool(
                    processes=pool_size,
                )
                S_list = pool.starmap(
                    self.find_gamma, [(A, A0, k, S_optimal, i) for i in gamma]
                )
                pool.close()
                pool.join()
                # print("S_list", S_list)
                index = [
                    index
                    for index, element in enumerate(
                        [S_list[i] for i in range(len(gamma))]
                    )
                    if min([S_list[i] for i in range(len(gamma))]) == element
                ]

                print("index[0], gamma[index[0]]", index[0], gamma[index[0]])

                self.P[k] = (1 - gamma[index[0]]) * self.P[k] + gamma[
                    index[0]
                ] * S_optimal[k]
            s = self.s_score(A, A0)
            itr += 1
            if itr % 20 == 0:
                A0 = self.center_graph(A)

            dual = np.mean([-np.sum(nabla_P[i] * delta_P[i]) for i in range(len(A))])
            s_old = s
            print(
                "Iter:",
                itr,
                "time  %s seconds " % (time.time() - start_time),
                " s=",
                s,
                "dual=",
                dual,
            )

            if itr >= self.itr_num:
                break
            else:
                pass
            if abs(dual) <= self.epsilon:

                break
            else:
                pass

        return self.P, A0
