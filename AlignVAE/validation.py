import numpy as np
import networkx as nx
import time
import scipy
import cvxpy as cp
import argparse
import pickle
from scipy.optimize import check_grad
from scipy.optimize import minimize
import tkinter
import matplotlib

import matplotlib.pyplot as plt
from pylab import *
import random
import argparse
import numpy as np
import os
import re
from random import shuffle
from scipy.stats import wasserstein_distance
import pickle as pkl
import subprocess as sp
import time
from scipy.linalg import toeplitz
import pyemd
import math
from sklearn.metrics.pairwise import polynomial_kernel

# DISCLAIMER:
# Boilerplate parts of this code file were originally forked from
# https://github.com/JiaxuanYou/graph-generation/


class validation_score:
    def __init__(self, graph_ref_list, graph_list, min_deg, thr):
        self.min_deg = min_deg
        self.graph_ref_list = graph_ref_list
        self.graph_list = graph_list
        self.thr = thr
        # print("min_deg", min_deg)
        self.graph_pred_list, self.all_deg_pred = self.build_graph()
        self.all_deg_ref, all_deg_pred = self.ref_graph()
        self.mmd_degree = self.degree_stats()
        self.mmd_clustering, self.cc_ref, self.cc_pred = self.clustering_stats()
        (
            self.mmd_assortavity,
            self.assort_ref,
            self.assort_pred,
        ) = self.assortavity_stats()
        self.mmd_tri, self.tri_ref, self.tri_pred = self.tri_count()
        self.mmd_claw, self.claw_ref, self.claw_pred = self.Claw_stats()
        self.mmd_wedge, self.wedge_ref, self.wedge_pred = self.Wedge_stats()
        self.s1, self.s2, self.s3 = self.score()

    def build_graph(self):

        deg_list = []
        all_deg = []
        min_deg = []
        max_deg = []
        graphs1 = []

        for k in range(len(self.graph_list)):
            A0 = self.graph_list[k]

            q = int(np.sqrt(len(A0)))

            A1 = np.reshape(A0, (q, q))

            A_rec = np.zeros((len(A1), len(A1)))

            A = sorted(A0, reverse=True)[0:400]

            m = self.min_deg

            for i in range(q):
                tmp1 = sorted(
                    A1[
                        i,
                    ],
                    reverse=True,
                )[0 : int(m) + 0]
                for j in range(q):
                    if A1[i, j] in tmp1:
                        A_rec[i, j] = 1
                        A_rec[j, i] = 1

            for i in range(len(A1)):
                for j in range(len(A1)):
                    if A1[i, j] >= self.thr:
                        A_rec[i, j] = 1
                        A_rec[j, i] = 1

            deg = np.zeros(len(A_rec))
            for i in range(len(A_rec)):
                deg[i] = sum(
                    A_rec[
                        i,
                    ]
                )
                all_deg.append(deg[i])

            max_deg.append(max(deg))
            min_deg.append(min(deg))
            G = nx.from_numpy_matrix(A_rec)
            # G = nx.Graph()
            # edges = []
            #
            # adj_nonzero1 = np.nonzero(A_rec)
            #
            # # adj_nonzero1 = np.nonzero(A0)
            # for i in range(len(adj_nonzero1[0])):
            #     edges.append((adj_nonzero1[0][i], adj_nonzero1[1][i]))
            #
            #     data_tuple = list(map(tuple, edges))
            #     G.add_edges_from(data_tuple)
            # id1 = np.arange(len(A0))
            # for i in id1:
            #     G.add_node(i)
            G.remove_nodes_from(list(nx.isolates(G)))
            graphs1.append(G)

        graph_pred_list = graphs1
        return graph_pred_list, all_deg

    def ref_graph(self):

        all_deg_ref = []
        for i in range(len(self.graph_ref_list)):
            """
            adj = np.zeros(
                (
                    len(self.graph_ref_list[i].nodes()),
                    len(self.graph_ref_list[i].nodes()),
                )
            )
            for t, j in self.graph_ref_list[i].edges():
                adj[t, j] = 1
                adj[j, t] = 1
            """
            adj = nx.adjacency_matrix(self.graph_ref_list[i]).todense()
            deg = np.zeros(len(adj))
            for l in range(len(adj)):
                deg[l] = sum(
                    adj[
                        l,
                    ]
                )
                all_deg_ref.append(deg[l])

        all_deg_pred = []

        for i in range(len(self.graph_pred_list)):
            """
            adj = np.zeros(
                (
                    len(self.graph_pred_list[i].nodes()),
                    len(self.graph_pred_list[i].nodes()),
                )
            )
            for t, j in self.graph_pred_list[i].edges():
                adj[t, j] = 1
                adj[j, t] = 1
            """
            adj = nx.adjacency_matrix(self.graph_pred_list[i]).todense()
            deg = np.zeros(len(adj))
            for l in range(len(adj)):
                deg[l] = sum(
                    adj[
                        l,
                    ]
                )
                all_deg_pred.append(deg[l])
                # print("min and average deg", np.min(all_deg_pred), np.mean(all_deg_pred) )

        return all_deg_ref, all_deg_pred

    def gaussian_emd(self, x, y, sigma=1, distance_scaling=1.0):
        """Gaussian kernel with squared distance in exponential term replaced by EMD
        Args:
        x, y: 1D pmf of two distributions with the same support
        sigma: standard deviation
        """
        support_size = max(len(x), len(y))
        d_mat = toeplitz(range(support_size)).astype(np.float)

        distance_mat = d_mat / distance_scaling

        x = x.astype(np.float)
        y = y.astype(np.float)
        if len(x) < len(y):
            x = np.hstack((x, [0.0] * (support_size - len(x))))
        elif len(y) < len(x):
            y = np.hstack((y, [0.0] * (support_size - len(y))))

        emd = pyemd.emd(x, y, distance_mat)

        return np.exp(-emd * emd / (2 * sigma * sigma))

    def gaussian_wasserstein(self, x, y, sigma=1):
        """Gaussian kernel with squared distance in exponential term replaced by EMD
        Args:
        x, y: 1D pmf of two distributions with the same support
        sigma: standard deviation
        """
        emd = wasserstein_distance(x, y)
        return np.exp(-emd * emd / (2 * sigma * sigma))

    def gaussian_emdsamples(self, x, y, sigma=1, bins=10):
        """Gaussian kernel with squared distance in exponential term replaced by EMD
        Args:
        x, y: 1D pmf of two distributions with the same support
        sigma: standard deviation
        """
        emd = pyemd.emd_samples(x, y, bins=10, range=(0, 1))

        return np.exp(-emd * emd / (2 * sigma * sigma))

    def gaussian(self, x, y, sigma=1.0):
        dist = np.linalg.norm(x - y, 2)
        return np.exp(-dist * dist / (2 * sigma * sigma))

    def disc(self, samples1, samples2, kernel, is_parallel=False, *args, **kwargs):
        """Discrepancy between 2 samples"""
        d = 0
        if not is_parallel:
            for s1 in samples1:
                for s2 in samples2:
                    d += kernel(s1, s2, *args, **kwargs)

        d /= len(samples1) * len(samples2)
        return d

    def compute_emd(self, samples1, samples2, kernel, is_hist=True, *args, **kwargs):
        """EMD between average of two samples"""
        # normalize histograms into pmf
        samples1 = [np.mean(samples1)]
        samples2 = [np.mean(samples2)]
        return self.disc(samples1, samples2, kernel, *args, **kwargs), [
            samples1[0],
            samples2[0],
        ]

    def compute_mmd(
        self, samples1, samples2, kernel, histogram="True", *args, **kwargs
    ):
        """MMD between two samples"""
        if histogram == "True":
            samples1 = [s1 / np.sum(s1) for s1 in samples1]
            samples2 = [s2 / np.sum(s2) for s2 in samples2]

        return (
            self.disc(samples1, samples1, kernel, *args, **kwargs)
            + self.disc(samples2, samples2, kernel, *args, **kwargs)
            - 2 * self.disc(samples1, samples2, kernel, *args, **kwargs)
        )

    def degree_stats(self, bins=100):
        """Compute the distance between the degree distributions of two unordered sets of graphs.
        Args:
        graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
        """
        sample_ref = []
        sample_pred = []
        deg_ref = []
        deg_pred = []
        # in case an empty graph is generated
        graph_pred_list_remove_empty = [
            G for G in self.graph_pred_list if not G.number_of_nodes() == 0
        ]

        for i in range(len(self.graph_ref_list)):
            degree_temp = np.array(nx.degree_histogram(self.graph_ref_list[i]))
            sample_ref.append(degree_temp)

        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(nx.degree_histogram(graph_pred_list_remove_empty[i]))
            sample_pred.append(degree_temp)

        mmd_dist = self.compute_mmd(
            sample_ref,
            sample_pred,
            kernel=self.gaussian_emd,
            sigma=np.std(self.all_deg_ref) + 10 ** (-6),
        )
        return mmd_dist

    def clustering_stats(self, bins=10):
        sample_ref = []
        sample_pred = []
        graph_pred_list_remove_empty = [
            G for G in self.graph_pred_list if not G.number_of_nodes() == 0
        ]
        coef_ref_all = []
        coef_pred_all = []

        coef_ref_all1 = []
        coef_pred_all1 = []

        for i in range(len(self.graph_ref_list)):

            clustering_coeffs_list = list(
                nx.clustering(self.graph_ref_list[i]).values()
            )
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
            )

            sample_ref.append(hist)
            coef_ref_all.append(clustering_coeffs_list)
            coef_ref_all1.append(nx.average_clustering(self.graph_ref_list[i]))

        for i in range(len(graph_pred_list_remove_empty)):

            clustering_coeffs_list = list(
                nx.clustering(self.graph_pred_list[i]).values()
            )
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
            )

            sample_pred.append(hist)
            coef_pred_all.append(clustering_coeffs_list)
            coef_pred_all1.append(nx.average_clustering(self.graph_pred_list[i]))

        cc_ref = coef_ref_all[0]
        cc_pred = coef_pred_all[0]
        for i in range(len(coef_ref_all) - 1):
            cc_ref = cc_ref + coef_ref_all[i + 1]
        for j in range(len(coef_pred_all) - 1):
            cc_pred = cc_pred + coef_pred_all[j + 1]

        mmd_cc = self.compute_mmd(
            sample_ref,
            sample_pred,
            kernel=self.gaussian_emd,
            sigma=np.std(cc_ref) + 10 ** (-6),
            distance_scaling=bins,
        )

        return mmd_cc, cc_ref, cc_pred

    def tri_count(self):
        tri_ref = []
        for i in range(len(self.graph_ref_list)):
            cnt = sum(list(nx.triangles(self.graph_ref_list[i]).values()))
            tri_ref.append(cnt)

        tri_pred = []
        for j in range(len(self.graph_pred_list)):
            cnt = sum(list(nx.triangles(self.graph_pred_list[j]).values()))
            tri_pred.append(cnt)

        tri_ref_cnt = {}
        tri_pred_cnt = {}
        for i in tri_ref:
            if i not in tri_ref_cnt:
                tri_ref_cnt[i] = 0
                tri_ref_cnt[i] += 1
            else:
                tri_ref_cnt[i] += 1

        for j in tri_pred:
            if j not in tri_pred_cnt:
                tri_pred_cnt[j] = 0
                tri_pred_cnt[j] += 1
            else:
                tri_pred_cnt[j] += 1
        ref_array = np.zeros(max(tri_ref_cnt.keys()) + 1)
        pred_array = np.zeros(max(tri_pred_cnt.keys()) + 1)
        for i in tri_ref_cnt.keys():
            ref_array[i] = tri_ref_cnt[i]
        for j in tri_pred_cnt.keys():
            pred_array[j] = tri_pred_cnt[j]

        m1 = min(min(tri_ref_cnt.keys()), min(tri_pred_cnt.keys()))
        m2 = max(max(tri_ref_cnt.keys()), max(tri_pred_cnt.keys()))

        sample_ref = []
        sample_pred = []

        hist, bins = np.histogram(tri_ref, range=(m1, m2))
        sample_ref.append(hist)

        hist1, bins1 = np.histogram(tri_pred, range=(m1, m2))
        sample_pred.append(hist1)

        mmd_tri = self.compute_mmd(
            sample_ref,
            sample_pred,
            kernel=self.gaussian_emd,
            sigma=np.std(tri_ref) + 10 ** (-6),
        )

        return mmd_tri, tri_ref, tri_pred

    def assortavity_stats(self, bins=10):

        sample_ref = []
        sample_pred = []
        assort_ref = []
        assort_pred = []

        for i in range(len(self.graph_ref_list)):
            asref = nx.degree_assortativity_coefficient(self.graph_ref_list[i])
            assort_ref.append(asref)
            hist, _ = np.histogram(asref, bins=bins)
            sample_ref.append(hist)

        for j in range(len(self.graph_pred_list)):
            aspred = nx.degree_assortativity_coefficient(self.graph_pred_list[j])
            assort_pred.append(aspred)
            hist1, _ = np.histogram(aspred, bins=bins)
            sample_pred.append(hist1)

        mmd_assort = self.compute_mmd(
            sample_ref,
            sample_pred,
            kernel=self.gaussian_emd,
            sigma=np.std(assort_ref) + 10 ** (-6),
            distance_scaling=bins,
        )

        return mmd_assort, assort_ref, assort_pred

    def Wedge_stats(self):

        wedge_pred = []
        wedge_ref = []
        sample_ref = []
        sample_pred = []

        for i in range(len(self.graph_pred_list)):
            m = len(self.graph_pred_list[i].nodes())
            wedge_pred.append(
                float(
                    np.sum(
                        np.array(
                            [
                                0.5 * x * (x - 1)
                                for x in self.all_deg_pred[m * i : m * i + m]
                            ]
                        )
                    )
                )
            )

        for j in range(len(self.graph_ref_list)):
            n = len(self.graph_ref_list[j].nodes())
            wedge_ref.append(
                float(
                    np.sum(
                        np.array(
                            [
                                0.5 * x * (x - 1)
                                for x in self.all_deg_ref[n * j : n * j + n]
                            ]
                        )
                    )
                )
            )

        m1 = min(min(wedge_ref), min(wedge_pred))
        m2 = max(max(wedge_ref), max(wedge_pred))

        hist, bins = np.histogram(wedge_ref, range=(m1, m2))
        sample_ref.append(hist)

        hist1, bins1 = np.histogram(wedge_pred, range=(m1, m2))
        sample_pred.append(hist1)

        mmd_wedge = self.compute_mmd(
            sample_ref,
            sample_pred,
            kernel=self.gaussian_emd,
            sigma=np.std(wedge_ref) + 10 ** (-6),
        )

        return mmd_wedge, wedge_ref, wedge_pred

    def Claw_stats(self, bins=100):

        claw_pred = []
        claw_ref = []
        sample_ref = []
        sample_pred = []
        for i in range(len(self.graph_pred_list)):
            m = len(self.graph_pred_list[i].nodes())
            claw_pred.append(
                float(
                    np.sum(
                        np.array(
                            [
                                1 / 6.0 * x * (x - 1) * (x - 2)
                                for x in self.all_deg_pred[m * i : m * i + m]
                            ]
                        )
                    )
                )
            )

        for j in range(len(self.graph_ref_list)):
            n = len(self.graph_ref_list[j].nodes())
            claw_ref.append(
                float(
                    np.sum(
                        np.array(
                            [
                                1 / 6.0 * x * (x - 1) * (x - 2)
                                for x in self.all_deg_ref[n * j : n * j + n]
                            ]
                        )
                    )
                )
            )

        m1 = min(min(claw_ref), min(claw_pred))
        m2 = max(max(claw_ref), max(claw_pred))

        hist, bins = np.histogram(claw_ref)

        hist, bins = np.histogram(claw_ref, range=(m1, m2))
        sample_ref.append(hist)

        hist1, bins1 = np.histogram(claw_pred, range=(m1, m2))
        sample_pred.append(hist1)

        mmd_claw = self.compute_mmd(
            sample_ref,
            sample_pred,
            kernel=self.gaussian_emd,
            sigma=np.std(claw_ref) + 10 ** (-6),
        )

        return mmd_claw, claw_ref, claw_pred

    def score(self):

        s1 = (1 / 12) * (
            self.mmd_degree
            + self.mmd_clustering
            + self.mmd_assortavity
            + self.mmd_tri
            + self.mmd_claw
            + self.mmd_wedge
        )

        s2 = (1 / 6) * (
            (np.mean(self.all_deg_ref) - np.mean(self.all_deg_pred)) ** 2
            / np.var(self.all_deg_ref)
            + (np.mean(self.cc_ref) - np.mean(self.cc_pred)) ** 2 / np.var(self.cc_ref)
            + (np.mean(self.tri_ref) - np.mean(self.tri_pred)) ** 2
            / np.var(self.tri_ref)
            + (np.mean(self.assort_ref) - np.mean(self.assort_pred)) ** 2
            / np.var(self.assort_ref)
            + (np.mean(self.claw_ref) - np.mean(self.claw_pred)) ** 2
            / np.var(self.claw_ref)
            + (np.mean(self.wedge_ref) - np.mean(self.wedge_pred)) ** 2
            / np.var(self.wedge_ref)
        )

        s3 = (1 / 6) * (
            (np.mean(self.all_deg_ref) - np.mean(self.all_deg_pred)) ** 2
            / np.std(self.all_deg_ref)
            + (np.mean(self.cc_ref) - np.mean(self.cc_pred)) ** 2 / np.std(self.cc_ref)
            + (np.mean(self.tri_ref) - np.mean(self.tri_pred)) ** 2
            / np.std(self.tri_ref)
            + (np.mean(self.assort_ref) - np.mean(self.assort_pred)) ** 2
            / np.std(self.assort_ref)
            + (np.mean(self.claw_ref) - np.mean(self.claw_pred)) ** 2
            / np.std(self.claw_ref)
            + (np.mean(self.wedge_ref) - np.mean(self.wedge_pred)) ** 2
            / np.std(self.wedge_ref)
        )

        # print("np.mean(self.all_deg_ref), np.mean(self.all_deg_pred)", np.mean(self.all_deg_ref), np.mean(self.all_deg_pred))
        # print("np.mean(self.cc_ref), np.mean(self.cc_pred)", np.mean(self.cc_ref), np.mean(self.cc_pred))
        # print("np.mean(self.tri_ref), np.mean(self.tri_pred)", np.mean(self.tri_ref), np.mean(self.tri_pred))
        # print("np.mean(self.assort_ref), np.mean(self.assort_pred)", np.mean(self.assort_ref), np.mean(self.assort_pred))
        # print("np.mean(self.claw_ref), np.mean(self.claw_pred)", np.mean(self.claw_ref), np.mean(self.claw_pred))
        # print("np.mean(self.wedge_ref), np.mean(self.wedge_pred)", np.mean(self.wedge_ref), np.mean(self.wedge_pred))
        # print("np.var(self.all_deg_pred), np.var(self.cc_ref), np.var(self.tri_ref), np.var(self.assort_ref), np.var(self.claw_ref), np.var(self.wedge_ref)", np.var(self.all_deg_pred), np.var(self.cc_ref), np.var(self.tri_ref), np.var(self.assort_ref), np.var(self.claw_ref), np.var(self.wedge_ref))
        # print("self.mmd_degree, self.mmd_clustering, self.mmd_assortavity,self.mmd_tri, self.mmd_claw, self.mmd_wedge", self.mmd_degree, self.mmd_clustering, self.mmd_assortavity, self.mmd_tri, self.mmd_claw, self.mmd_wedge)
        # print("*"*50)
        """
        s1= (1/10)*(self.mmd_degree+self.mmd_clustering+self.mmd_assortavity+self.mmd_claw+self.mmd_wedge)

        s2= (1/5)*((np.mean(self.all_deg_ref)-np.mean(self.all_deg_pred))**2/np.var(self.all_deg_pred)+(np.mean(self.cc_ref)-np.mean(self.cc_pred))**2/np.var(self.cc_ref)+(np.mean(self.assort_ref)-np.mean(self.assort_pred))**2/np.var(self.assort_ref)+(np.mean(self.claw_ref)-np.mean(self.claw_pred))**2/np.var(self.claw_ref)+(np.mean(self.wedge_ref)-np.mean(self.wedge_pred))**2/np.var(self.wedge_ref))

        s3 = (1 / 5) * (
            (np.mean(self.all_deg_ref) - np.mean(self.all_deg_pred)) ** 2
            / np.std(self.all_deg_ref)
            + (np.mean(self.cc_ref) - np.mean(self.cc_pred)) ** 2 / np.std(self.cc_ref)
            + (np.mean(self.assort_ref) - np.mean(self.assort_pred)) ** 2
            / np.std(self.assort_ref)
            + (np.mean(self.claw_ref) - np.mean(self.claw_pred)) ** 2
            / np.std(self.claw_ref)
            + (np.mean(self.wedge_ref) - np.mean(self.wedge_pred)) ** 2
            / np.std(self.wedge_ref)
        )
        """
        return s1, s2, s3
