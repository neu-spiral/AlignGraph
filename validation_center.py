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

# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from pylab import *
import random
import argparse
import numpy as np
import os
import re
from random import shuffle

# import eval.stats
# import utils
import pickle as pkl
import subprocess as sp
import time
from scipy.linalg import toeplitz
import pyemd

# import powerlaw
from sklearn.metrics.pairwise import polynomial_kernel


class validation_center:
    def __init__(self, graph_ref_list, graph_list, thr):
        self.graph_ref_list = graph_ref_list
        self.graph_pred_list = graph_list
        # self.graph_list= graph_list
        self.thr = thr
        self.all_deg_ref, self.all_deg_pred = self.ref_graph()
        # self.graph_pred_list, self.all_deg_pred= self.build_graph()
        """
        self.graph_pred_list= self.graph_list
        self.all_deg_ref, self.all_deg_pred= self.ref_graph()
        """
        # self.all_deg_ref= self.ref_graph()
        self.mmd_degree = self.degree_stats()
        self.mmd_clustering, self.cc_ref, self.cc_pred = self.clustering_stats()
        (
            self.mmd_assortavity,
            self.assort_ref,
            self.assort_pred,
        ) = self.assortavity_stats()
        # print("self.mmd_assortavity, self.assort_ref, self.assort_pred", self.mmd_assortavity, self.assort_ref, self.assort_pred)
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
        # deg_list=np.load("degree_list_diverse_0.003_63.npy", allow_pickle=True)
        deg_list = self.all_deg_ref

        for k in range(len(self.graph_list)):
            A0 = self.graph_list[k]
            # print(min(A0), max(A0))
            # q= int(np.sqrt(len(A0)))
            q = int(len(self.graph_ref_list[0].nodes()))
            # print("q",  q)
            A1 = np.reshape(A0, (q, q))
            # A1=A1-np.eye(q)

            A_rec = np.zeros((len(A1), len(A1)))
            # m=min([ k for (k,v) in deg_list[0]])
            m = int(min(deg_list))
            # A=sorted(A0, reverse=True)[0:400]

            for i in range(q):
                tmp1 = sorted(
                    A1[
                        i,
                    ],
                    reverse=True,
                )[0:m]
                # print("first 5 elements",tmp1)
                for j in range(q):
                    if A1[i, j] in tmp1:
                        A_rec[i, j] = 1
                        A_rec[j, i] = 1

            for i in range(len(A1)):
                for j in range(len(A1)):
                    # if abs(A1[i,j])/sum(abs(A1[i,]))>=1.05/len(A1):
                    if abs(A1[i, j]) >= self.thr:
                        A_rec[i, j] = 1
                        A_rec[j, i] = 1
            # print(np.linalg.norm(A0_rec[i,]-A0[i,]))

            """
            for i in range(q):
                tmp1=sorted(A1[i,], reverse=True)[0:m]
                #print("first 5 elements",tmp1)
                for j in range(q):
                    if A1[i,j] in tmp1:
                    #if (A1[i,j])>=0:
                        A_rec[i,j]=1
                        A_rec[j,i]=1
            """

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
            # deg_list.append(deg.tolist())
            # print("min_deg, max_deg",min_deg, max_deg)

            G = nx.Graph()
            edges = []

            adj_nonzero1 = np.nonzero(A_rec)
            for i in range(len(adj_nonzero1[0])):
                edges.append((adj_nonzero1[0][i], adj_nonzero1[1][i]))

                # print("len(edges)",len(edges))
                data_tuple = list(map(tuple, edges))
                G.add_edges_from(data_tuple)
            id1 = np.arange(len(A0))
            for i in id1:
                G.add_node(i)
            G.remove_nodes_from(list(nx.isolates(G)))
            # print("number of edges in generated graphs", len(G.edges()))
            graphs1.append(G)

            graph_pred_list = graphs1
            # print("np.mean(all_deg_pred)", np.mean(all_deg))
        return graph_pred_list, all_deg

    def ref_graph(self):

        all_deg_ref = []
        for i in range(len(self.graph_ref_list)):
            adj = np.zeros(
                (
                    len(self.graph_ref_list[i].nodes()),
                    len(self.graph_ref_list[i].nodes()),
                )
            )
            for t, j in self.graph_ref_list[i].edges():
                adj[t, j] = 1
                adj[j, t] = 1
            deg = np.zeros(len(adj))
            for l in range(len(adj)):
                deg[l] = sum(
                    adj[
                        l,
                    ]
                )
                all_deg_ref.append(deg[l])

        all_deg_pred = []
        # print("len(self.graph_pred_list)", len(self.graph_pred_list), self.graph_pred_list[0])

        for i in range(len(self.graph_pred_list)):
            adj = np.zeros(
                (
                    len(self.graph_pred_list[i].nodes()),
                    len(self.graph_pred_list[i].nodes()),
                )
            )
            for t, j in self.graph_pred_list[i].edges():
                adj[t, j] = 1
                adj[j, t] = 1
            deg = np.zeros(len(adj))
            for l in range(len(adj)):
                deg[l] = sum(
                    adj[
                        l,
                    ]
                )
                all_deg_pred.append(deg[l])

            # print("np.mean(all_deg_ref)", np.mean(all_deg_ref))
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

        # convert histogram values x and y to float, and make them equal len
        x = x.astype(np.float)
        y = y.astype(np.float)
        if len(x) < len(y):
            x = np.hstack((x, [0.0] * (support_size - len(x))))
        elif len(y) < len(x):
            y = np.hstack((y, [0.0] * (support_size - len(y))))

        emd = pyemd.emd(x, y, distance_mat)
        return np.exp(-emd * emd / (2 * sigma * sigma))

    def gaussian(self, x, y, sigma=1.0):
        dist = np.linalg.norm(x - y, 2)
        return np.exp(-dist * dist / (2 * sigma * sigma))

    def disc(self, samples1, samples2, kernel, is_parallel=False, *args, **kwargs):
        """Discrepancy between 2 samples"""
        # print("samples1, samples2", samples1, samples2)
        d = 0
        if not is_parallel:
            for s1 in samples1:
                for s2 in samples2:
                    d += kernel(s1, s2, *args, **kwargs)
        # print("len(samples1), len(samples2)", len(samples1), len(samples2))
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

    def compute_mmd(self, samples1, samples2, kernel, *args, **kwargs):
        """MMD between two samples"""
        samples1 = [s1 / np.sum(s1) for s1 in samples1]
        samples2 = [s2 / np.sum(s2) for s2 in samples2]

        return (
            self.disc(samples1, samples1, kernel, *args, **kwargs)
            + self.disc(samples2, samples2, kernel, *args, **kwargs)
            - 2 * self.disc(samples1, samples2, kernel, *args, **kwargs)
        )

    def degree_stats(self):
        """Compute the distance between the degree distributions of two unordered sets of graphs.
        Args:
        graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
        """
        sample_ref = []
        sample_pred = []
        # in case an empty graph is generated
        graph_pred_list_remove_empty = [
            G for G in self.graph_pred_list if not G.number_of_nodes() == 0
        ]

        for i in range(len(self.graph_ref_list)):
            degree_temp = np.array(nx.degree_histogram(self.graph_ref_list[i]))
            # print("degree_temp", degree_temp)
            sample_ref.append(degree_temp)
        # print("np.std(self.all_deg_ref)", np.std(self.all_deg_ref))
        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(nx.degree_histogram(graph_pred_list_remove_empty[i]))
            sample_pred.append(degree_temp)
        # print(sample_ref, sample_pred, len(sample_ref),len(sample_pred))
        mmd_dist = self.compute_mmd(
            sample_ref,
            sample_pred,
            kernel=self.gaussian_emd,
            sigma=np.std(self.all_deg_ref),
        )

        # mmd_dist = self.compute_mmd(sample_ref, sample_pred, kernel= self.gaussian_emd, sigma= 1)

        # print("np.std(self.all_deg_ref", np.std(self.all_deg_ref))
        # mmd_dist1=compute_mmd(sample_ref, sample_pred, kernel=poly)
        # print("mmd_dist", mmd_dist)
        return mmd_dist

    def clustering_stats(self, bins=100):
        sample_ref = []
        sample_pred = []
        graph_pred_list_remove_empty = [
            G for G in self.graph_pred_list if not G.number_of_nodes() == 0
        ]
        coef_ref_all = []
        coef_pred_all = []

        for i in range(len(self.graph_ref_list)):
            clustering_coeffs_list = list(
                nx.clustering(self.graph_ref_list[i]).values()
            )
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
            )
            sample_ref.append(hist)
            coef_ref_all.append(list(nx.clustering(self.graph_ref_list[i]).values()))
        for i in range(len(graph_pred_list_remove_empty)):
            clustering_coeffs_list1 = list(
                nx.clustering(graph_pred_list_remove_empty[i]).values()
            )
            hist, _ = np.histogram(
                clustering_coeffs_list1, bins=bins, range=(0.0, 1.0), density=False
            )
            sample_pred.append(hist)
            coef_pred_all.append(clustering_coeffs_list1)
        # print("clustering coef std for ref and pred", np.std(clustering_coeffs_list), np.std(clustering_coeffs_list1), np.mean(clustering_coeffs_list), np.mean(clustering_coeffs_list1)  )

        # print(coef_ref_all,coef_pred_all)
        cc_ref = coef_ref_all[0]
        cc_pred = coef_pred_all[0]
        for i in range(len(coef_ref_all) - 1):
            cc_ref = cc_ref + coef_ref_all[i + 1]
        for j in range(len(coef_pred_all) - 1):
            cc_pred = cc_pred + coef_pred_all[j + 1]

        # print("cc_ref, cc_pred", len(cc_ref), len(cc_pred))
        # print("cls coef all std and mean for ref and pred", np.std(cc_ref), np.std(cc_pred), np.mean(cc_ref), np.mean(cc_pred))

        # print("sample ref and sample pred for c.c", sample_ref, sample_pred)
        mmd_cc = self.compute_mmd(
            sample_ref,
            sample_pred,
            kernel=self.gaussian_emd,
            sigma=np.std(cc_ref),
            distance_scaling=bins,
        )

        # mmd_cc = self.compute_mmd(sample_ref, sample_pred, kernel=self.gaussian_emd,
        #                           sigma=1, distance_scaling=bins)

        # print("np.std(cc_ref)", np.std(cc_ref))
        # print("mmd_cc", mmd_cc)
        return mmd_cc, cc_ref, cc_pred
        """

    def clustering_stats(self, bins=100):
        sample_ref = []
        sample_pred = []
        graph_pred_list_remove_empty = [G for G in self.graph_pred_list if not G.number_of_nodes() == 0]


        # check non-zero elements in hist
        #total = 0
        #for i in range(len(sample_pred)):
        #    nz = np.nonzero(sample_pred[i])[0].shape[0]
        #    total += nz
        #print(total)
        for i in range(len(self.graph_ref_list)):
            clustering_coeffs_list = list(nx.clustering(self.graph_ref_list[i]).values())
            hist, _ = np.histogram(
                    clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_ref.append(hist)

        for i in range(len(graph_pred_list_remove_empty)):
            clustering_coeffs_list = list(nx.clustering(graph_pred_list_remove_empty[i]).values())
            hist, _ = np.histogram(
                    clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_pred.append(hist)

        mmd_cc = self.compute_mmd(sample_ref, sample_pred, kernel=self.gaussian_emd,
                               sigma=0.1, distance_scaling=bins)

        return mmd_cc, sample_ref, sample_pred
        """

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
        # print("hist, hist1",hist, hist1, bins, bins1)

        # print("std and mean ref and gen", np.std(tri_ref), np.std(tri_pred), np.mean(tri_ref), np.mean(tri_pred))

        """
        sample_ref=[]
        sample_pred=[]
        sample_ref.append(ref_array)
        sample_pred.append(pred_array)
        """
        # print("tri_ref_cnt, tri_pred_cnt", tri_ref_cnt.keys(), tri_pred_cnt.keys(), ref_array, pred_array, len(ref_array),len(pred_array))
        mmd_tri = self.compute_mmd(
            sample_ref, sample_pred, kernel=self.gaussian_emd, sigma=np.std(tri_ref)
        )
        # mmd_tri = self.compute_mmd(sample_ref, sample_pred, kernel=self.gaussian_emd, sigma=1)
        # print("triangle tri", mmd_tri)
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

        # print("min and max assortavity reference", min(assort_ref), max(assort_ref))

        for j in range(len(self.graph_pred_list)):
            aspred = nx.degree_assortativity_coefficient(self.graph_pred_list[j])
            assort_pred.append(aspred)
            #   print("j, min(aspred), max(aspred)", j, aspred)
            hist1, _ = np.histogram(aspred, bins=bins)
            sample_pred.append(hist1)

        # print("min and max assortavity prediction", min(assort_pred), max(assort_pred))

        # print("average ref and pred assortavity", np.mean(assort_ref), np.mean(assort_pred))

        mmd_assort = self.compute_mmd(
            sample_ref, sample_pred, kernel=self.gaussian_emd, sigma=np.std(assort_ref)
        )

        # mmd_assort = self.compute_mmd(sample_ref, sample_pred, kernel=self.gaussian_emd, sigma=1)

        # print("mmd_assort", mmd_assort)
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
        # print("hist, hist1",hist, hist1, bins, bins1)
        # print("average ref and pred for wedge count", np.mean(wedge_ref), np.mean(wedge_pred))
        # print("wedge_ref", wedge_ref)
        mmd_wedge = self.compute_mmd(
            sample_ref, sample_pred, kernel=self.gaussian_emd, sigma=np.std(wedge_ref)
        )
        # mmd_wedge = self.compute_mmd(sample_ref, sample_pred, kernel=self.gaussian_emd, sigma=1)
        # print("mmd_wedge", mmd_wedge)
        return mmd_wedge, wedge_ref, wedge_pred

    def Claw_stats(self):

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

        hist, bins = np.histogram(claw_ref, range=(m1, m2))
        sample_ref.append(hist)

        hist1, bins1 = np.histogram(claw_pred, range=(m1, m2))
        sample_pred.append(hist1)
        # print("hist, hist1",hist, hist1, bins, bins1)
        # print("claw_ref", claw_ref)
        # print("average ref and pred for claw count", np.mean(claw_ref), np.mean(claw_pred))
        mmd_claw = self.compute_mmd(
            sample_ref, sample_pred, kernel=self.gaussian_emd, sigma=np.std(claw_ref)
        )

        # print("mmd_claw", mmd_claw)
        return mmd_claw, claw_ref, claw_pred

    def score(self):

        """
        mmd_degree= self.degree_stats()
        mmd_clustering, cc_ref, cc_pred = self.clustering_stats()
        mmd_assortavity, assort_ref, assort_pred= self.assortavity_stats()
        mmd_tri, tri_ref, tri_pred= self.tri_count()
        mmd_claw, claw_ref, claw_pred= self.Claw_stats()
        mmd_wedge, wedge_ref, wedge_pred= self.Wedge_stats()
        """
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
            / np.var(self.all_deg_pred)
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
            / np.std(self.all_deg_pred)
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

        # print("(self.mmd_degree,self.mmd_clustering,self.mmd_assortavity,self.mmd_tri,self.mmd_claw,self.mmd_wedge)", self.mmd_degree,self.mmd_clustering,self.mmd_tri,self.mmd_claw,self.mmd_wedge)
        #
        # print("np.var(self.all_deg_pred), np.var(self.cc_ref), np.var(self.tri_ref), np.var(self.assort_ref), np.var(self.claw_ref), np.var(self.wedge_ref)", np.var(self.all_deg_pred), np.var(self.cc_ref), np.var(self.tri_ref), np.var(self.claw_ref), np.var(self.wedge_ref))

        return s1, s2, s3
