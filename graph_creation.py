import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle as pkl
import scipy.sparse as sp
import argparse
from sklearn.decomposition import PCA
import pickle
import re
import operator
from operator import itemgetter

# DISCLAIMER:
# Boilerplate parts of this code file were originally forked from
# https://github.com/JiaxuanYou/graph-generation/

# load cora, citeseer and pubmed dataset
def Graph_load(data="cora"):
    """
    Load a single graph dataset
    :param dataset: dataset name
    :return:
    """
    names = ["x", "tx", "allx", "graph"]
    objects = []
    for i in range(len(names)):
        load = pkl.load(
            open("data/ind.{}.{}".format(data, names[i]), "rb"), encoding="latin1"
        )
        # print('loaded')
        objects.append(load)
        # print(load)
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(data))
    test_idx_range = np.sort(test_idx_reorder)

    if data == "citeseer":
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    G = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(G)
    return adj, features, G


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def Prot_Enz(name):
    G = nx.Graph()
    data_adj = np.loadtxt(name + "_A.txt", delimiter=",").astype(int)
    data_node_att = np.loadtxt(name + "_node_attributes.txt", delimiter=",")
    data_node_label = np.loadtxt(name + "_node_labels.txt", delimiter=",").astype(int)
    data_graph_indicator = np.loadtxt(
        name + "_graph_indicator.txt", delimiter=","
    ).astype(int)
    data_graph_labels = np.loadtxt(name + "_graph_labels.txt", delimiter=",").astype(
        int
    )
    data_tuple = list(map(tuple, data_adj))

    # add edges
    G.add_edges_from(data_tuple)
    # add node attributes
    for i in range(data_node_att.shape[0]):
        G.add_node(i, feature=data_node_att[i])
        G.add_node(i, label=data_node_label[i])
    G.remove_nodes_from(list(nx.isolates(G)))

    # split into graphs

    node_list = np.arange(data_graph_indicator.shape[0])
    graphs = []
    node_num_list = []

    nodes = node_list[0:701]
    G_sub = G.subgraph(nodes)
    G_sub.graph["label"] = data_graph_labels[0]
    node_num_list.append(G_sub.number_of_nodes())
    node_num_list = np.array(node_num_list)
    print("selected", len(node_num_list[node_num_list > 10]))

    keys = tuple(G_sub.nodes())
    dictionary = nx.get_node_attributes(G_sub, "feature")

    features = np.zeros((len(dictionary), list(dictionary.values())[0].shape[0]))
    for i in range(len(dictionary)):
        features[i, :] = list(dictionary.values())[i]
    return G_sub


def connected_component_subgraphs(G):

    # (G.subgraph(c) for c in nx.connected_components(G))
    for c in nx.connected_components(G):
        yield G.subgraph(c)


def n_community(c_sizes, p_inter=0.01):
    graphs = [nx.gnp_random_graph(c_sizes[i], 0.6, seed=i) for i in range(len(c_sizes))]
    G = nx.disjoint_union_all(graphs)
    communities = list(connected_component_subgraphs(G))
    for i in range(len(communities)):
        subG1 = communities[i]
        nodes1 = list(subG1.nodes())
        for j in range(i + 1, len(communities)):
            subG2 = communities[j]
            nodes2 = list(subG2.nodes())
            has_inter_edge = False
            for n1 in nodes1:
                for n2 in nodes2:
                    if np.random.rand() < p_inter:
                        G.add_edge(n1, n2)
                        has_inter_edge = True
            if not has_inter_edge:
                G.add_edge(nodes1[0], nodes2[0])
    return G


def Graph_load_batch(
    min_num_nodes=20,
    max_num_nodes=1000,
    name="DD",
    node_attributes=True,
    graph_labels=True,
):
    """
    load many graphs, e.g. protein
    :return: a list of graphs
    """
    print("Loading graph dataset: " + str(name))
    G = nx.Graph()
    # load data
    # path = 'dataset/'+name+'/'
    data_adj = np.loadtxt(name + "_A.txt", delimiter=",").astype(int)
    if node_attributes:
        data_node_att = np.loadtxt(name + "_node_attributes.txt", delimiter=",")
    data_node_label = np.loadtxt(name + "_node_labels.txt", delimiter=",").astype(int)
    data_graph_indicator = np.loadtxt(
        name + "_graph_indicator.txt", delimiter=","
    ).astype(int)
    if graph_labels:
        data_graph_labels = np.loadtxt(
            name + "_graph_labels.txt", delimiter=","
        ).astype(int)

    data_tuple = list(map(tuple, data_adj))

    G.add_edges_from(data_tuple)
    # add node attributes
    for i in range(data_node_label.shape[0]):
        if node_attributes:
            G.add_node(i, feature=data_node_att[i])
        G.add_node(i, label=data_node_label[i])
    G.remove_nodes_from(list(nx.isolates(G)))

    # split into graphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0])
    graphs = []
    max_nodes = 1000
    for i in range(graph_num):
        # find the nodes for each graph
        nodes = node_list[data_graph_indicator == i]
        G_sub = G.subgraph(nodes)
        G.remove_nodes_from(list(nx.isolates(G_sub)))
        if graph_labels:
            G_sub.graph["label"] = data_graph_labels[i]

        if (
            G_sub.number_of_nodes() >= min_num_nodes
            and G_sub.number_of_nodes() <= max_num_nodes
        ):
            graphs.append(G_sub)
            if G_sub.number_of_nodes() > max_nodes:
                max_nodes = G_sub.number_of_nodes()

    return graphs


def perm(g, one_noise=0.99, zero_noise=0.0001):
    G = nx.Graph()
    id1 = list(g.nodes())
    # print("len(id1)", len(id1))
    id1.sort()
    id1 = np.arange(len(id1))
    # adj = np.zeros((len(id1), len(id1)))
    # for i, j in g.edges():
    #     adj[i, j] = 1

    adj = nx.to_numpy_matrix(g)
    # print("adj.shape[0], adj.shape[1]", adj.shape[0], adj.shape[1])
    id2 = np.random.permutation(id1)
    zipped = zip(id1, id2)
    mapping = dict(zipped)

    P = np.zeros((adj.shape[0], adj.shape[1]))
    for i, t in list(enumerate(id1)):
        ind = np.where(np.asarray(id1) == mapping[t])
        P[i, ind[0][0]] = 1

    new_adj = np.zeros((adj.shape[0], adj.shape[1]))
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i, j] == 1:
                new_adj[i, j] = np.random.binomial(1, one_noise)
            else:
                new_adj[i, j] = np.random.binomial(1, zero_noise)

    adj1 = np.matmul(np.matmul(P, new_adj), P.T)
    G = nx.from_numpy_matrix(adj1)
    G.remove_nodes_from(list(nx.isolates(G)))
    # print("len(G.edges())", len(G.edges()))
    return G, P, adj1


def draw_graph_list(
    G_list,
    row,
    col,
    fname="figures/test",
    layout="spring",
    is_single=False,
    k=1,
    node_size=55,
    alpha=1,
    width=1.3,
):

    plt.switch_backend("agg")
    for i, G in enumerate(G_list):
        plt.subplot(row, col, i + 1)

        plt.axis("off")
        if layout == "spring":
            pos = nx.spring_layout(
                G, k=k / np.sqrt(G.number_of_nodes()), iterations=100
            )

        elif layout == "spectral":
            pos = nx.spectral_layout(G)

        if is_single:
            nx.draw_networkx_nodes(
                G,
                pos,
                node_size=node_size,
                node_color="#336699",
                alpha=1,
                linewidths=0,
                font_size=0,
            )
            nx.draw_networkx_edges(G, pos, alpha=alpha, width=width)
        else:
            nx.draw_networkx_nodes(
                G,
                pos,
                node_size=1.5,
                node_color="#336699",
                alpha=1,
                linewidths=0.2,
                font_size=1.5,
            )
            nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.2)

    plt.tight_layout()
    plt.savefig(fname + ".png", dpi=600)
    plt.close()


def save_graph_list(G_list, fname):
    with open(fname, "wb") as f:
        pickle.dump(G_list, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph", type=str, default="PROTEINS_full", help="graph structure"
    )
    args = parser.parse_args()

    if args.graph == "PROTEINS_full":
        graphs = Graph_load_batch(
            min_num_nodes=50,
            max_num_nodes=100,
            name="data/DD",
            node_attributes=True,
            graph_labels=True,
        )
        G_sub = graphs[0]
        graphs = []
        graphs.append(G_sub)
        P_orig = []
        P_orig.append(np.eye(G_sub.number_of_nodes()))

        for i in range(199):
            G, P, adj = perm(G_sub)
            graphs.append(G)
            P_orig.append(P)
        save_graph_list(graphs, "PROTEINS_full_graphs")
        np.save("P_orig_" + args.graph + ".npy", P_orig)
        for i in range(0, 16, 16):
            draw_graph_list(
                graphs[i : i + 16], 4, 4, fname="figures/test/protein_" + str(i)
            )

    if args.graph == "Communities":
        graphs = []
        num_communities = int(2)
        print("Creating dataset with ", num_communities, "communities")

        c_sizes = [40, 50, 60]
        graphs.append(n_community(c_sizes, p_inter=0.005))
        print("nodes", graphs[0].number_of_nodes())
        print("edges", graphs[0].number_of_edges())
        max_prev_node = 80

        G_sub = graphs[0]
        P_orig = []
        A = []
        adj = np.zeros((len(G_sub), len(G_sub)))
        for i, j in G_sub.edges():
            adj[i, j] = 1
        A.append(adj)
        print(
            "# of nodes and edges orig",
            G_sub.number_of_nodes(),
            G_sub.number_of_edges(),
        )
        P_orig.append(np.eye(G_sub.number_of_nodes()))
        for i in range(199):
            G, P, adj = perm(G_sub, one_noise=0.95, zero_noise=0.005)
            graphs.append(G)
            print("# of nodes and edges", G.number_of_nodes(), G.number_of_edges())
            P_orig.append(P)
            A.append(adj)
        save_graph_list(graphs, "Community_graphs")

        np.save(args.graph, P_orig)
        np.save(args.graph, A)

    if args.graph == "ego":
        n = 950
        m = 5
        cnt = 0
        g_orig = []
        for i in range(100):
            G = nx.generators.barabasi_albert_graph(n, m)
            # find node with largest degree
            node_and_degree = G.degree()
            list1 = [(k, v) for k, v in node_and_degree.items()]
            node_and_degree = list1
            (largest_hub, degree) = sorted(node_and_degree, key=itemgetter(1))[-1]

            # Create ego graph of main hub
            hub_ego = nx.ego_graph(G, largest_hub)
            if hub_ego.number_of_nodes() >= 100 and hub_ego.number_of_nodes() <= 150:
                cnt = cnt + 1
                g_orig.append(hub_ego)
        save_graph_list(g_orig, "Ego_graphs")

    if args.graph == "grid":
        n = 6
        graphs = []
        P_orig = []
        G1 = nx.Graph()
        G0 = nx.grid_2d_graph(n, n)
        A = []
        adj = nx.adjacency_matrix(G0).todense()
        edges = []
        adj_nonzero1 = np.nonzero(adj)

        for i in range(len(adj_nonzero1[0])):
            edges.append((adj_nonzero1[0][i], adj_nonzero1[1][i]))

        print("len(edges)", len(edges))
        data_tuple = list(map(tuple, edges))
        G1.add_edges_from(data_tuple)
        for i in range(n):
            G1.add_node(i)
        G1.remove_nodes_from(list(nx.isolates(G1)))
        graphs.append(G1)

        for i in range(199):
            G, P, adj = perm(G1, one_noise=0.995, zero_noise=0.0005)
            graphs.append(G)
            P_orig.append(P)
            A.append(adj)
        save_graph_list(graphs, "grid_graphs")
        np.save("P_test_" + args.graph, P_orig)
        np.save("A_test" + args.graph, A)

    if args.graph == "DD":
        graphs = Graph_load_batch(
            min_num_nodes=100,
            max_num_nodes=130,
            name="data/DD/DD",
            node_attributes=False,
            graph_labels=True,
        )
        # save_graph_list(graphs, "DD_graphs_orig")
        args.max_prev_node = 230
        print("len(graphs)", len(graphs))
        node_num = []
        edge_num = []
        for i in range(len(graphs)):
            node_num.append(graphs[i].number_of_nodes())
            edge_num.append(graphs[i].number_of_edges())
            print("||A||", np.linalg.norm(nx.adjacency_matrix(graphs[i]).todense()))

        print(
            "min, max and mean of node_num",
            min(node_num),
            max(node_num),
            np.mean(node_num),
        )
        print(
            "min, max and mean of edge_num",
            min(edge_num),
            max(edge_num),
            np.mean(edge_num),
        )
        A = []
        graphs_list = []
        for i in range(100):
            G_sub = graphs[i]
            graphs_list.append(G_sub)
            P_orig = []
            P_orig.append(np.eye(G_sub.number_of_nodes()))
            adj = nx.to_numpy_matrix(G_sub)
            A.append(adj)
            print("number of edges", G_sub.number_of_edges())
        print("len(graphs_list)", len(graphs_list))
        save_graph_list(graphs_list, "DD_graphs")

    if args.graph == "citeseer_small":
        _, _, G = Graph_load(data="citeseer")
        # G = max(nx.connected_component_subgraphs(G), key=len)
        G = max(connected_component_subgraphs(G), key=len)
        G = nx.convert_node_labels_to_integers(G)
        graphs = []
        adj_list = []
        for i in range(G.number_of_nodes()):
            G_ego = nx.ego_graph(G, i, radius=3)
            if (G_ego.number_of_nodes() >= 30) and (G_ego.number_of_nodes() <= 40):
                graphs.append(G_ego)
                adj_list.append(nx.adjacency_matrix(G_ego).todense())
        np.save("citeseer_adj_orig", adj_list)
        print("len(graphs)", len(graphs))
        save_graph_list(graphs, "citseer_graphs_orig")
        graphs = graphs[0:200]
        args.max_prev_node = 15
