import time
import cvxpy
import cvxpy as cp
from validation import validation_score
import multiprocessing
from multiprocessing import Pool
from multiprocessing import Process, Queue
import multiprocessing.pool
import multiprocessing as mp
from scipy.optimize import linear_sum_assignment
from train_double import *

# DISCLAIMER:
# Boilerplate parts of this code file were originally forked from
# https://github.com/JiaxuanYou/graph-generation/

# # if FLAGS.id_gpu >= 0:
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# # The GPU id to use
# os.environ["CUDA_VISIBLE_DEVICES"] = str(1)


def save_graph_list(G_list, fname):
    with open(fname, "wb") as f:
        pickle.dump(G_list, f)


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


if __name__ == "__main__":
    # All necessary arguments are defined in args.py

    args = Args()
    print("args", args.note)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    print("CUDA", args.cuda)
    print("File name prefix", args.fname)

    with open("citeseer_graphs_orig", "rb") as f:
        graphs = pickle.load(f, encoding="latin1")

    A_list = []
    all_deg = []
    mindeg = []
    num_nodes = []
    for i in range(len(graphs)):
        g = graphs[i]
        id1 = list(g.nodes())
        id1.sort()
        adj = nx.adjacency_matrix(g).todense()
        num_nodes.append(len(adj))
        deg = [g.degree[i] for i in g.nodes()]
        all_deg.append(deg)
        A_list.append(adj)
        mindeg.append(min(deg))

    deg_list = all_deg

    min_deg = min(mindeg)
    m = np.max(num_nodes)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # # The GPU id to use
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    # check if necessary directories exist

    if not os.path.isdir(args.model_save_path):
        os.makedirs(args.model_save_path)
    if not os.path.isdir(args.graph_save_path):
        os.makedirs(args.graph_save_path)
    if not os.path.isdir(args.figure_save_path):
        os.makedirs(args.figure_save_path)
    if not os.path.isdir(args.timing_save_path):
        os.makedirs(args.timing_save_path)
    if not os.path.isdir(args.figure_prediction_save_path):
        os.makedirs(args.figure_prediction_save_path)
    if not os.path.isdir(args.nll_save_path):
        os.makedirs(args.nll_save_path)

    # time = time.time()
    # logging.basicConfig(filename='logs/train' + time + '.log', level=logging.DEBUG)

    if args.clean_tensorboard:
        if os.path.isdir("tensorboard"):
            shutil.rmtree("tensorboard")
    configure("tensorboard/run" + str(time), flush_secs=5)

    # upload adjacency matrices of aligned graphs
    adj_align = np.load("A_align_citeseer.npy")

    graphs_train = []
    num_nodes = []
    for i in range(len(adj_align)):
        adj = adj_align[i]
        for j in range(len(adj)):
            if adj[j, j] == 1:
                adj[j, j] == 0
        g = nx.from_numpy_matrix(adj)
        g.remove_nodes_from(list(nx.isolates(g)))
        num_nodes.append(len(adj))
        print("# of nodes", g.number_of_nodes())
        graphs_train.append(g)

    m = max(num_nodes)
    # check if necessary directories exist

    if not os.path.isdir(args.model_save_path):
        os.makedirs(args.model_save_path)
    if not os.path.isdir(args.graph_save_path):
        os.makedirs(args.graph_save_path)
    if not os.path.isdir(args.figure_save_path):
        os.makedirs(args.figure_save_path)
    if not os.path.isdir(args.timing_save_path):
        os.makedirs(args.timing_save_path)
    if not os.path.isdir(args.figure_prediction_save_path):
        os.makedirs(args.figure_prediction_save_path)
    if not os.path.isdir(args.nll_save_path):
        os.makedirs(args.nll_save_path)

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # # The GPU id to use
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    # split datasets
    random.seed(123)
    shuffle(graphs)
    graphs_len = len(graphs)
    graphs_validate = graphs_train[0 : int(0.2 * graphs_len)]
    graphs_test = graphs[int(0.8 * graphs_len) :]

    print("graphs_train", len(graphs_train))

    # Upload adjacency matrix of center graph
    adj_cent = np.load("A0_fermat_third_Egoorig.npy")
    for k in range(len(adj_cent)):
        if adj_cent[k, k] == 1:
            adj_cent[k, k] == 0
    G_cent_rec_cls = nx.from_numpy_matrix(adj_cent)
    G_cent_rec_cls.remove_nodes_from(list(nx.isolates(G_cent_rec_cls)))
    g0 = [G_cent_rec_cls] * len(graphs_train)

    print("len(g0)", len(g0), len(g0[0].nodes()))

    graph_validate_len = 0
    for graph in graphs_validate:
        graph_validate_len += graph.number_of_nodes()
    graph_validate_len /= len(graphs_validate)
    print("graph_validate_len", graph_validate_len)

    graph_test_len = 0
    for graph in graphs_test:
        graph_test_len += graph.number_of_nodes()
    graph_test_len /= len(graphs_test)
    print("graph_test_len", graph_test_len)

    args.max_num_node = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
    max_num_edge = max([graphs[i].number_of_edges() for i in range(len(graphs))])
    min_num_edge = min([graphs[i].number_of_edges() for i in range(len(graphs))])

    # args.max_num_node = 2000
    # show graphs statistics
    print(
        "total graph num: {}, training set: {}".format(len(graphs), len(graphs_train))
    )
    print("max number node: {}".format(args.max_num_node))
    print("max/min number edge: {}; {}".format(max_num_edge, min_num_edge))
    print("max previous node: {}".format(args.max_prev_node))

    # save ground truth graphs
    # To get train and test set, after loading you need to manually slice
    save_graph_list(
        graphs, args.graph_save_path + args.fname_train + "_double" + "0.dat"
    )
    save_graph_list(
        graphs, args.graph_save_path + args.fname_test + "_double" + "0.dat"
    )
    print(
        "train and test graphs saved at: ",
        args.graph_save_path + args.fname_test + "0.dat",
    )

    ### dataset initialization
    if "nobfs" in args.bfs:
        print("nobfs")
        dataset = Graph_sequence_sampler_pytorch_nobfs(
            graphs_train, max_num_node=args.max_num_node
        )
        dataset0 = Graph_sequence_sampler_pytorch_nobfs(
            g0, max_num_node=args.max_num_node
        )
        args.max_prev_node = args.max_num_node - 1

    if "barabasi_noise" in args.bfs:
        print("barabasi_noise")
        dataset = Graph_sequence_sampler_pytorch_canonical(
            graphs_train, max_prev_node=args.max_prev_node
        )
        args.max_prev_node = args.max_num_node - 1
    else:
        dataset = Graph_sequence_sampler_pytorch(
            graphs_train,
            max_prev_node=args.max_prev_node,
            max_num_node=args.max_num_node,
        )

    sample_strategy = torch.utils.data.sampler.WeightedRandomSampler(
        [1.0 / len(dataset) for i in range(len(dataset))],
        num_samples=args.batch_size * args.batch_ratio,
        replacement=True,
    )

    sample_strategy0 = torch.utils.data.sampler.WeightedRandomSampler(
        [1.0 / len(dataset0) for i in range(len(dataset0))],
        num_samples=args.batch_size * args.batch_ratio,
        replacement=True,
    )

    dataset_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=sample_strategy,
    )

    dataset_loader0 = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=sample_strategy,
    )

    # print("len(dataset_loader), len(dataset_loader0)", len(dataset_loader), len(dataset_loader0))

    ### model initialization
    ## Graph RNN VAE model
    # lstm = LSTM_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_lstm,
    #                   hidden_size=args.hidden_size, num_layers=args.num_layers).cuda()

    if "GraphRNN_VAE_conditional" in args.note:
        rnn = GRU_plain(
            input_size=args.max_prev_node,
            embedding_size=args.embedding_size_rnn,
            hidden_size=args.hidden_size_rnn,
            num_layers=args.num_layers,
            has_input=True,
            has_output=False,
        ).cuda()
        output = MLP_VAE_conditional_plain(
            h_size=args.hidden_size_rnn,
            embedding_size=args.embedding_size_output,
            y_size=args.max_prev_node,
        ).cuda()
    elif "GraphRNN_MLP" in args.note:
        rnn = GRU_plain(
            input_size=args.max_prev_node,
            embedding_size=args.embedding_size_rnn,
            hidden_size=args.hidden_size_rnn,
            num_layers=args.num_layers,
            has_input=True,
            has_output=False,
        ).cuda()
        output = MLP_plain(
            h_size=args.hidden_size_rnn,
            embedding_size=args.embedding_size_output,
            y_size=args.max_prev_node,
        ).cuda()
    elif "GraphRNN_RNN" in args.note:

        rnn = GRU_plain(
            input_size=args.max_prev_node,
            embedding_size=args.embedding_size_rnn,
            hidden_size=args.hidden_size_rnn,
            num_layers=args.num_layers,
            has_input=True,
            has_output=True,
            output_size=args.hidden_size_rnn_output,
        ).cuda()
        output = GRU_plain(
            input_size=1,
            embedding_size=args.embedding_size_rnn_output,
            hidden_size=args.hidden_size_rnn_output,
            num_layers=args.num_layers,
            has_input=True,
            has_output=True,
            output_size=1,
        ).cuda()

        rnn0 = GRU_plain(
            input_size=args.max_prev_node,
            embedding_size=args.embedding_size_rnn,
            hidden_size=args.hidden_size_rnn,
            num_layers=args.num_layers,
            has_input=True,
            has_output=True,
            output_size=args.hidden_size_rnn_output,
        ).cuda()
        output0 = GRU_plain(
            input_size=1,
            embedding_size=args.embedding_size_rnn_output,
            hidden_size=args.hidden_size_rnn_output,
            num_layers=args.num_layers,
            has_input=True,
            has_output=True,
            output_size=1,
        ).cuda()

    t0 = time.time()
    G_test, G_pred, G_test0, G_pred0 = train_double(
        args, dataset_loader, dataset_loader0, rnn, output, rnn0, output0
    )
    print("running time:", time.time() - t0)
    save_graph_list(G_test, "G_double")

    print("len(G_test), len(G_test0)", len(G_test), len(G_test0))

    s_total = validation_score(graphs_test[0:20], G_test0[0:20], 0.5)
    print("Test s1 and s2 scores", s_total.s1, s_total.s2)

    ### graph completion
    # train_graph_completion(args,dataset_loader,rnn,output)

    ### nll evaluation
    # train_nll(args, dataset_loader, dataset_loader, rnn, output, max_iter = 200, graph_validate_len=graph_validate_len,graph_test_len=graph_test_len)
