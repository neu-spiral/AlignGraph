import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.decomposition import PCA
import logging
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from time import gmtime, strftime
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from random import shuffle
import pickle
from tensorboard_logger import configure, log_value
import scipy.misc
import time as tm

from utils import *
from model import *
from data import *
from args import Args


# DISCLAIMER:
# Boilerplate parts of this code file were originally forked from
# https://github.com/JiaxuanYou/graph-generation/


def train_vae_epoch(
    epoch,
    args,
    rnn,
    output,
    data_loader,
    optimizer_rnn,
    optimizer_output,
    scheduler_rnn,
    scheduler_output,
):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data["x"].float()
        y_unsorted = data["y"].float()
        y_len_unsorted = data["len"]
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

        # sort input
        y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted, 0, sort_index)
        y = torch.index_select(y_unsorted, 0, sort_index)
        x = Variable(x).cuda()
        y = Variable(y).cuda()

        # if using ground truth to train
        h = rnn(x, pack=True, input_len=y_len)
        y_pred, z_mu, z_lsgms = output(h)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        z_mu = pack_padded_sequence(z_mu, y_len, batch_first=True)
        z_mu = pad_packed_sequence(z_mu, batch_first=True)[0]
        z_lsgms = pack_padded_sequence(z_lsgms, y_len, batch_first=True)
        z_lsgms = pad_packed_sequence(z_lsgms, batch_first=True)[0]
        # use cross entropy loss
        loss_bce = binary_cross_entropy_weight(y_pred, y)
        loss_kl = -0.5 * torch.sum(1 + z_lsgms - z_mu.pow(2) - z_lsgms.exp())
        loss_kl /= y.size(0) * y.size(1) * sum(y_len)  # normalize
        loss = loss_bce + loss_kl
        loss.backward()
        # update deterministic and lstm
        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()

        z_mu_mean = torch.mean(z_mu.data)
        z_sgm_mean = torch.mean(z_lsgms.mul(0.5).exp_().data)
        z_mu_min = torch.min(z_mu.data)
        z_sgm_min = torch.min(z_lsgms.mul(0.5).exp_().data)
        z_mu_max = torch.max(z_mu.data)
        z_sgm_max = torch.max(z_lsgms.mul(0.5).exp_().data)

        if (
            epoch % args.epochs_log == 0 and batch_idx == 0
        ):  # only output first batch's statistics
            print(
                "Epoch: {}/{}, train bce loss: {:.6f}, train kl loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}".format(
                    epoch,
                    args.epochs,
                    loss_bce.data[0],
                    loss_kl.data[0],
                    args.graph_type,
                    args.num_layers,
                    args.hidden_size_rnn,
                )
            )
            print(
                "z_mu_mean",
                z_mu_mean,
                "z_mu_min",
                z_mu_min,
                "z_mu_max",
                z_mu_max,
                "z_sgm_mean",
                z_sgm_mean,
                "z_sgm_min",
                z_sgm_min,
                "z_sgm_max",
                z_sgm_max,
            )

        # logging
        log_value(
            "bce_loss_" + args.fname,
            loss_bce.data[0],
            epoch * args.batch_ratio + batch_idx,
        )
        log_value(
            "kl_loss_" + args.fname,
            loss_kl.data[0],
            epoch * args.batch_ratio + batch_idx,
        )
        log_value(
            "z_mu_mean_" + args.fname, z_mu_mean, epoch * args.batch_ratio + batch_idx
        )
        log_value(
            "z_mu_min_" + args.fname, z_mu_min, epoch * args.batch_ratio + batch_idx
        )
        log_value(
            "z_mu_max_" + args.fname, z_mu_max, epoch * args.batch_ratio + batch_idx
        )
        log_value(
            "z_sgm_mean_" + args.fname, z_sgm_mean, epoch * args.batch_ratio + batch_idx
        )
        log_value(
            "z_sgm_min_" + args.fname, z_sgm_min, epoch * args.batch_ratio + batch_idx
        )
        log_value(
            "z_sgm_max_" + args.fname, z_sgm_max, epoch * args.batch_ratio + batch_idx
        )

        loss_sum += loss.data[0]
    return loss_sum / (batch_idx + 1)


def test_vae_epoch(
    epoch, args, rnn, output, test_batch_size=16, save_histogram=False, sample_time=1
):
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()

    # generate graphs
    max_num_node = int(args.max_num_node)
    y_pred = Variable(
        torch.zeros(test_batch_size, max_num_node, args.max_prev_node)
    ).cuda()  # normalized prediction score
    y_pred_long = Variable(
        torch.zeros(test_batch_size, max_num_node, args.max_prev_node)
    ).cuda()  # discrete prediction
    x_step = Variable(torch.ones(test_batch_size, 1, args.max_prev_node)).cuda()
    for i in range(max_num_node):
        h = rnn(x_step)
        y_pred_step, _, _ = output(h)
        y_pred[:, i : i + 1, :] = F.sigmoid(y_pred_step)
        x_step = sample_sigmoid(y_pred_step, sample=True, sample_time=sample_time)
        y_pred_long[:, i : i + 1, :] = x_step
        rnn.hidden = Variable(rnn.hidden.data).cuda()
    y_pred_data = y_pred.data
    y_pred_long_data = y_pred_long.data.long()

    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        G_pred = get_graph(adj_pred)  # get a graph from zero-padded adj
        G_pred_list.append(G_pred)

    # save prediction histograms, plot histogram over each time step
    # if save_histogram:
    #     save_prediction_histogram(y_pred_data.cpu().numpy(),
    #                           fname_pred=args.figure_prediction_save_path+args.fname_pred+str(epoch)+'.jpg',
    #                           max_num_node=max_num_node)

    return G_pred_list


def test_vae_partial_epoch(
    epoch, args, rnn, output, data_loader, save_histogram=False, sample_time=1
):
    rnn.eval()
    output.eval()
    G_pred_list = []
    for batch_idx, data in enumerate(data_loader):
        x = data["x"].float()
        y = data["y"].float()
        y_len = data["len"]
        test_batch_size = x.size(0)
        rnn.hidden = rnn.init_hidden(test_batch_size)
        # generate graphs
        max_num_node = int(args.max_num_node)
        y_pred = Variable(
            torch.zeros(test_batch_size, max_num_node, args.max_prev_node)
        ).cuda()  # normalized prediction score
        y_pred_long = Variable(
            torch.zeros(test_batch_size, max_num_node, args.max_prev_node)
        ).cuda()  # discrete prediction
        x_step = Variable(torch.ones(test_batch_size, 1, args.max_prev_node)).cuda()
        for i in range(max_num_node):
            print("finish node", i)
            h = rnn(x_step)
            y_pred_step, _, _ = output(h)
            y_pred[:, i : i + 1, :] = F.sigmoid(y_pred_step)
            x_step = sample_sigmoid_supervised(
                y_pred_step,
                y[:, i : i + 1, :].cuda(),
                current=i,
                y_len=y_len,
                sample_time=sample_time,
            )

            y_pred_long[:, i : i + 1, :] = x_step
            rnn.hidden = Variable(rnn.hidden.data).cuda()
        y_pred_data = y_pred.data
        y_pred_long_data = y_pred_long.data.long()

        # save graphs as pickle
        for i in range(test_batch_size):
            adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            G_pred = get_graph(adj_pred)  # get a graph from zero-padded adj
            G_pred_list.append(G_pred)
    return G_pred_list


def train_mlp_epoch(
    epoch,
    args,
    rnn,
    output,
    data_loader,
    optimizer_rnn,
    optimizer_output,
    scheduler_rnn,
    scheduler_output,
):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data["x"].float()
        y_unsorted = data["y"].float()
        y_len_unsorted = data["len"]
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

        # sort input
        y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted, 0, sort_index)
        y = torch.index_select(y_unsorted, 0, sort_index)
        x = Variable(x).cuda()
        y = Variable(y).cuda()

        h = rnn(x, pack=True, input_len=y_len)
        y_pred = output(h)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        # use cross entropy loss
        loss = binary_cross_entropy_weight(y_pred, y)
        loss.backward()
        # update deterministic and lstm
        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()

        if (
            epoch % args.epochs_log == 0 and batch_idx == 0
        ):  # only output first batch's statistics
            print(
                "Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}".format(
                    epoch,
                    args.epochs,
                    loss.data[0],
                    args.graph_type,
                    args.num_layers,
                    args.hidden_size_rnn,
                )
            )

        # logging
        log_value(
            "loss_" + args.fname, loss.data[0], epoch * args.batch_ratio + batch_idx
        )

        loss_sum += loss.data[0]
    return loss_sum / (batch_idx + 1)


def test_mlp_epoch(
    epoch, args, rnn, output, test_batch_size=16, save_histogram=False, sample_time=1
):
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()

    # generate graphs
    max_num_node = int(args.max_num_node)
    y_pred = Variable(
        torch.zeros(test_batch_size, max_num_node, args.max_prev_node)
    ).cuda()  # normalized prediction score
    y_pred_long = Variable(
        torch.zeros(test_batch_size, max_num_node, args.max_prev_node)
    ).cuda()  # discrete prediction
    x_step = Variable(torch.ones(test_batch_size, 1, args.max_prev_node)).cuda()
    for i in range(max_num_node):
        h = rnn(x_step)
        y_pred_step = output(h)
        y_pred[:, i : i + 1, :] = F.sigmoid(y_pred_step)
        x_step = sample_sigmoid(y_pred_step, sample=True, sample_time=sample_time)
        y_pred_long[:, i : i + 1, :] = x_step
        rnn.hidden = Variable(rnn.hidden.data).cuda()
    y_pred_data = y_pred.data
    y_pred_long_data = y_pred_long.data.long()

    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        G_pred = get_graph(adj_pred)  # get a graph from zero-padded adj
        G_pred_list.append(G_pred)

    return G_pred_list


def test_mlp_partial_epoch(
    epoch, args, rnn, output, data_loader, save_histogram=False, sample_time=1
):
    rnn.eval()
    output.eval()
    G_pred_list = []
    for batch_idx, data in enumerate(data_loader):
        x = data["x"].float()
        y = data["y"].float()
        y_len = data["len"]
        test_batch_size = x.size(0)
        rnn.hidden = rnn.init_hidden(test_batch_size)
        # generate graphs
        max_num_node = int(args.max_num_node)
        y_pred = Variable(
            torch.zeros(test_batch_size, max_num_node, args.max_prev_node)
        ).cuda()  # normalized prediction score
        y_pred_long = Variable(
            torch.zeros(test_batch_size, max_num_node, args.max_prev_node)
        ).cuda()  # discrete prediction
        x_step = Variable(torch.ones(test_batch_size, 1, args.max_prev_node)).cuda()
        for i in range(max_num_node):
            print("finish node", i)
            h = rnn(x_step)
            y_pred_step = output(h)
            y_pred[:, i : i + 1, :] = F.sigmoid(y_pred_step)
            x_step = sample_sigmoid_supervised(
                y_pred_step,
                y[:, i : i + 1, :].cuda(),
                current=i,
                y_len=y_len,
                sample_time=sample_time,
            )

            y_pred_long[:, i : i + 1, :] = x_step
            rnn.hidden = Variable(rnn.hidden.data).cuda()
        y_pred_data = y_pred.data
        y_pred_long_data = y_pred_long.data.long()

        # save graphs as pickle
        for i in range(test_batch_size):
            adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            G_pred = get_graph(adj_pred)  # get a graph from zero-padded adj
            G_pred_list.append(G_pred)
    return G_pred_list


def test_mlp_partial_simple_epoch(
    epoch, args, rnn, output, data_loader, save_histogram=False, sample_time=1
):
    rnn.eval()
    output.eval()
    G_pred_list = []
    for batch_idx, data in enumerate(data_loader):
        x = data["x"].float()
        y = data["y"].float()
        y_len = data["len"]
        test_batch_size = x.size(0)
        rnn.hidden = rnn.init_hidden(test_batch_size)
        # generate graphs
        max_num_node = int(args.max_num_node)
        y_pred = Variable(
            torch.zeros(test_batch_size, max_num_node, args.max_prev_node)
        ).cuda()  # normalized prediction score
        y_pred_long = Variable(
            torch.zeros(test_batch_size, max_num_node, args.max_prev_node)
        ).cuda()  # discrete prediction
        x_step = Variable(torch.ones(test_batch_size, 1, args.max_prev_node)).cuda()
        for i in range(max_num_node):
            print("finish node", i)
            h = rnn(x_step)
            y_pred_step = output(h)
            y_pred[:, i : i + 1, :] = F.sigmoid(y_pred_step)
            x_step = sample_sigmoid_supervised_simple(
                y_pred_step,
                y[:, i : i + 1, :].cuda(),
                current=i,
                y_len=y_len,
                sample_time=sample_time,
            )

            y_pred_long[:, i : i + 1, :] = x_step
            rnn.hidden = Variable(rnn.hidden.data).cuda()
        y_pred_data = y_pred.data
        y_pred_long_data = y_pred_long.data.long()

        # save graphs as pickle
        for i in range(test_batch_size):
            adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            G_pred = get_graph(adj_pred)  # get a graph from zero-padded adj
            G_pred_list.append(G_pred)
    return G_pred_list


def train_mlp_forward_epoch(epoch, args, rnn, output, data_loader):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data["x"].float()
        y_unsorted = data["y"].float()
        y_len_unsorted = data["len"]
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

        # sort input
        y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted, 0, sort_index)
        y = torch.index_select(y_unsorted, 0, sort_index)
        x = Variable(x).cuda()
        y = Variable(y).cuda()

        h = rnn(x, pack=True, input_len=y_len)
        y_pred = output(h)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        # use cross entropy loss

        loss = 0
        for j in range(y.size(1)):
            # print('y_pred',y_pred[0,j,:],'y',y[0,j,:])
            end_idx = min(j + 1, y.size(2))
            loss += (
                binary_cross_entropy_weight(y_pred[:, j, 0:end_idx], y[:, j, 0:end_idx])
                * end_idx
            )

        if (
            epoch % args.epochs_log == 0 and batch_idx == 0
        ):  # only output first batch's statistics
            print(
                "Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}".format(
                    epoch,
                    args.epochs,
                    loss.data[0],
                    args.graph_type,
                    args.num_layers,
                    args.hidden_size_rnn,
                )
            )

        # logging
        log_value(
            "loss_" + args.fname, loss.data[0], epoch * args.batch_ratio + batch_idx
        )

        loss_sum += loss.data[0]
    return loss_sum / (batch_idx + 1)


def train_rnn_double(
    epoch,
    args,
    rnn,
    output,
    rnn0,
    output0,
    data_loader,
    data_loader0,
    optimizer_rnn,
    optimizer_output,
    optimizer_rnn0,
    optimizer_output0,
    scheduler_rnn,
    scheduler_output,
    scheduler_rnn0,
    scheduler_output0,
):

    rnn.train()
    output.train()

    rnn0.train()
    output0.train()

    loss_sum = 0

    for batch_idx, data in enumerate(zip(data_loader, data_loader0)):

        # for batch_idx, data in enumerate(data_loader0):
        # print("data_loader0", data )
        # for batch_idx, data in enumerate(data_loader):
        # for batch_idx0, data0 in enumerate(data_loader0):
        # print("data", len(data))

        rnn.zero_grad()
        output.zero_grad()

        rnn0.zero_grad()
        output0.zero_grad()
        """"
        x_unsorted0 = data['x'].float()
        y_unsorted0 = data['y'].float()
        y_len_unsorted0 = data['len']
        """
        x_unsorted = data[0]["x"].float()
        y_unsorted = data[0]["y"].float()
        y_len_unsorted = data[0]["len"]

        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))
        # print("rnn.hidden", len(rnn.hidden), len(rnn.hidden[0][0]))
        # output.hidden = output.init_hidden(batch_size=x_unsorted.size(0)*x_unsorted.size(1))

        x_unsorted0 = data[1]["x"].float()
        y_unsorted0 = data[1]["y"].float()
        y_len_unsorted0 = data[1]["len"]

        y_len_max0 = max(y_len_unsorted0)
        y_len_max0 = max(y_len_unsorted0)
        x_unsorted0 = x_unsorted0[:, 0:y_len_max0, :]
        y_unsorted0 = y_unsorted0[:, 0:y_len_max0, :]
        # initialize lstm hidden state according to batch size
        rnn0.hidden = rnn0.init_hidden(batch_size=x_unsorted0.size(0))

        # sort input

        y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted, 0, sort_index)
        y = torch.index_select(y_unsorted, 0, sort_index)

        y_len0, sort_index0 = torch.sort(y_len_unsorted0, 0, descending=True)
        y_len0 = y_len0.numpy().tolist()
        x0 = torch.index_select(x_unsorted0, 0, sort_index0)
        y0 = torch.index_select(y_unsorted0, 0, sort_index0)

        # input, output for output rnn module
        # a smart use of pytorch builtin function: pack variable--b1_l1,b2_l1,...,b1_l2,b2_l2,...

        y_reshape = pack_padded_sequence(y, y_len, batch_first=True).data
        # reverse y_reshape, so that their lengths are sorted, add dimension
        idx = [i for i in range(y_reshape.size(0) - 1, -1, -1)]
        idx = torch.LongTensor(idx)
        y_reshape = y_reshape.index_select(0, idx)
        y_reshape = y_reshape.view(y_reshape.size(0), y_reshape.size(1), 1)

        y_reshape0 = pack_padded_sequence(y0, y_len0, batch_first=True).data
        # reverse y_reshape, so that their lengths are sorted, add dimension
        idx0 = [i for i in range(y_reshape0.size(0) - 1, -1, -1)]
        idx0 = torch.LongTensor(idx0)
        y_reshape0 = y_reshape0.index_select(0, idx0)
        y_reshape0 = y_reshape0.view(y_reshape0.size(0), y_reshape0.size(1), 1)

        output_x = torch.cat(
            (torch.ones(y_reshape.size(0), 1, 1), y_reshape[:, 0:-1, 0:1]), dim=1
        )
        output_y = y_reshape

        output_x0 = torch.cat(
            (torch.ones(y_reshape0.size(0), 1, 1), y_reshape0[:, 0:-1, 0:1]), dim=1
        )
        output_y0 = y_reshape0

        # batch size for output module: sum(y_len)

        output_y_len = []
        output_y_len_bin = np.bincount(np.array(y_len))
        for i in range(len(output_y_len_bin) - 1, 0, -1):
            count_temp = np.sum(output_y_len_bin[i:])  # count how many y_len is above i
            output_y_len.extend(
                [min(i, y.size(2))] * count_temp
            )  # put them in output_y_len; max value should not exceed y.size(2)

        output_y_len0 = []
        output_y_len_bin0 = np.bincount(np.array(y_len0))
        for i in range(len(output_y_len_bin0) - 1, 0, -1):
            count_temp0 = np.sum(
                output_y_len_bin0[i:]
            )  # count how many y_len is above i
            output_y_len0.extend(
                [min(i, y0.size(2))] * count_temp0
            )  # put them in output_y_len; max value should not exceed y.size(2)

        # pack into variable

        # print("output_y_len_bin, output_y_len_bin0", len(output_y_len_bin), len(output_y_len_bin0))
        # pack into variable

        x = Variable(x).cuda()
        y = Variable(y).cuda()
        # print("output_x, output_y", len(output_x), len(output_y))
        output_x = Variable(output_x).cuda()
        output_y = Variable(output_y).cuda()

        x0 = Variable(x0).cuda()
        y0 = Variable(y0).cuda()
        # print("output_x, output_y", len(output_x), len(output_y))
        output_x0 = Variable(output_x0).cuda()
        output_y0 = Variable(output_y0).cuda()

        # if using ground truth to train

        h = rnn(x, pack=True, input_len=y_len)
        h_loss = h

        h0 = rnn0(x0, pack=True, input_len=y_len0)
        h_loss0 = h0

        # print("graph level representation y_len, h", len(y_len), len(node_hidden))#### graph level hidden state
        h = pack_padded_sequence(
            h, y_len, batch_first=True
        ).data  # get packed hidden vector

        h0 = pack_padded_sequence(h0, y_len0, batch_first=True).data

        # reverse h

        idx = [i for i in range(h.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx)).cuda()
        h = h.index_select(0, idx)
        hidden_null = Variable(
            torch.zeros(args.num_layers - 1, h.size(0), h.size(1))
        ).cuda()
        output.hidden = torch.cat(
            (h.view(1, h.size(0), h.size(1)), hidden_null), dim=0
        )  # num_layers, batch_size, hidden_size

        idx0 = [i for i in range(h0.size(0) - 1, -1, -1)]
        idx0 = Variable(torch.LongTensor(idx0)).cuda()
        h0 = h0.index_select(0, idx0)
        hidden_null0 = Variable(
            torch.zeros(args.num_layers - 1, h0.size(0), h0.size(1))
        ).cuda()
        output0.hidden = torch.cat(
            (h0.view(1, h0.size(0), h0.size(1)), hidden_null0), dim=0
        )

        y_pred = output(output_x, pack=True, input_len=output_y_len)
        y_loss = y_pred
        # print("edge level representation y_pred", len(edge_hidden)) #edge output hidden
        y_pred = F.sigmoid(y_pred)

        y_pred0 = output0(output_x0, pack=True, input_len=output_y_len0)
        y_loss0 = y_pred0
        y_pred0 = F.sigmoid(y_pred0)

        # clean

        y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        output_y = pack_padded_sequence(output_y, output_y_len, batch_first=True)
        output_y = pad_packed_sequence(output_y, batch_first=True)[0]

        y_pred0 = pack_padded_sequence(y_pred0, output_y_len0, batch_first=True)
        y_pred0 = pad_packed_sequence(y_pred0, batch_first=True)[0]
        output_y0 = pack_padded_sequence(output_y0, output_y_len0, batch_first=True)
        output_y0 = pad_packed_sequence(output_y0, batch_first=True)[0]

        m = min(len(h_loss.data.cpu().numpy()[0]), len(h_loss0.data.cpu().numpy()[0]))

        D = []
        for i in range(m):
            for j in range(m):
                D.append(
                    np.linalg.norm(
                        h_loss.data.cpu().numpy()[0][i]
                        - h_loss0.data.cpu().numpy()[0][j]
                    )
                )

        D = np.reshape(D, (m, m))

        loss = 50 * (
            binary_cross_entropy_weight(y_pred, output_y)
            + binary_cross_entropy_weight(y_pred0, output_y0)
        ) + (1 / 1000) * np.trace(D)
        loss.backward()
        # update deterministic and lstm

        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()

        optimizer_output0.step()
        optimizer_rnn0.step()
        scheduler_output0.step()
        scheduler_rnn0.step()

        if (
            epoch % args.epochs_log == 0 and batch_idx == 0
        ):  # only output first batch's statistics
            print(
                "Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}".format(
                    epoch,
                    args.epochs,
                    loss.data,
                    args.graph_type,
                    args.num_layers,
                    args.hidden_size_rnn,
                )
            )

        # logging
        # log_value('loss_'+args.fname, loss.data, epoch*args.batch_ratio+batch_idx)

        feature_dim = y.size(1) * y.size(2)

        # feature_dim = y0.size(1)*y0.size(2)
        loss_sum += loss.data * feature_dim
        # print("loss_sum", loss_sum/(batch_idx+1))
    return loss_sum / (batch_idx + 1)


def test_rnn_double(epoch, args, rnn, output, rnn0, output0, test_batch_size=16):

    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()

    rnn0.hidden = rnn0.init_hidden(test_batch_size)
    rnn0.eval()
    output0.eval()

    # generate graphs
    max_num_node = int(args.max_num_node)

    y_pred_long = Variable(
        torch.zeros(test_batch_size, max_num_node, args.max_prev_node)
    ).cuda()  # discrete prediction
    x_step = Variable(torch.ones(test_batch_size, 1, args.max_prev_node)).cuda()

    y_pred_long0 = Variable(
        torch.zeros(test_batch_size, max_num_node, args.max_prev_node)
    ).cuda()  # discrete prediction
    x_step0 = Variable(torch.ones(test_batch_size, 1, args.max_prev_node)).cuda()

    for i in range(max_num_node):

        h = rnn(x_step)

        h0 = rnn0(x_step0)

        # output.hidden = h.permute(1,0,2)

        hidden_null = Variable(
            torch.zeros(args.num_layers - 1, h.size(0), h.size(2))
        ).cuda()
        output.hidden = torch.cat(
            (h.permute(1, 0, 2), hidden_null), dim=0
        )  # num_layers, batch_size, hidden_size

        hidden_null0 = Variable(
            torch.zeros(args.num_layers - 1, h0.size(0), h0.size(2))
        ).cuda()
        output0.hidden = torch.cat(
            (h0.permute(1, 0, 2), hidden_null0), dim=0
        )  # num_layers, batch_size, hidden_size

        x_step = Variable(torch.zeros(test_batch_size, 1, args.max_prev_node)).cuda()
        output_x_step = Variable(torch.ones(test_batch_size, 1, 1)).cuda()

        x_step0 = Variable(torch.zeros(test_batch_size, 1, args.max_prev_node)).cuda()
        output_x_step0 = Variable(torch.ones(test_batch_size, 1, 1)).cuda()

        for j in range(min(args.max_prev_node, i + 1)):

            output_y_pred_step = output(output_x_step)
            output_x_step = sample_sigmoid(
                output_y_pred_step, sample=True, sample_time=1
            )
            x_step[:, :, j : j + 1] = output_x_step
            output.hidden = Variable(output.hidden.data).cuda()

            output_y_pred_step0 = output0(output_x_step0)
            output_x_step0 = sample_sigmoid(
                output_y_pred_step0, sample=True, sample_time=1
            )
            x_step0[:, :, j : j + 1] = output_x_step0
            output0.hidden = Variable(output0.hidden.data).cuda()

        y_pred_long[:, i : i + 1, :] = x_step
        rnn.hidden = Variable(rnn.hidden.data).cuda()

        y_pred_long0[:, i : i + 1, :] = x_step0
        rnn0.hidden = Variable(rnn0.hidden.data).cuda()

    y_pred_long_data = y_pred_long.data.long()

    y_pred_long_data0 = y_pred_long0.data.long()

    # save graphs as pickle
    G_pred_list = []

    G_pred_list0 = []

    for i in range(test_batch_size):

        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        G_pred = get_graph(adj_pred)  # get a graph from zero-padded adj
        G_pred_list.append(G_pred)

        adj_pred0 = decode_adj(y_pred_long_data0[i].cpu().numpy())
        G_pred0 = get_graph(adj_pred0)  # get a graph from zero-padded adj
        G_pred_list0.append(G_pred0)

    return G_pred_list, G_pred_list0
    # return G_pred_list0


def train_rnn_forward_epoch(epoch, args, rnn, output, data_loader):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data["x"].float()
        y_unsorted = data["y"].float()
        y_len_unsorted = data["len"]
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))
        # output.hidden = output.init_hidden(batch_size=x_unsorted.size(0)*x_unsorted.size(1))

        # sort input
        y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted, 0, sort_index)
        y = torch.index_select(y_unsorted, 0, sort_index)

        # input, output for output rnn module
        # a smart use of pytorch builtin function: pack variable--b1_l1,b2_l1,...,b1_l2,b2_l2,...
        y_reshape = pack_padded_sequence(y, y_len, batch_first=True).data
        # reverse y_reshape, so that their lengths are sorted, add dimension
        idx = [i for i in range(y_reshape.size(0) - 1, -1, -1)]
        idx = torch.LongTensor(idx)
        y_reshape = y_reshape.index_select(0, idx)
        y_reshape = y_reshape.view(y_reshape.size(0), y_reshape.size(1), 1)

        output_x = torch.cat(
            (torch.ones(y_reshape.size(0), 1, 1), y_reshape[:, 0:-1, 0:1]), dim=1
        )
        output_y = y_reshape
        # batch size for output module: sum(y_len)
        output_y_len = []
        output_y_len_bin = np.bincount(np.array(y_len))
        for i in range(len(output_y_len_bin) - 1, 0, -1):
            count_temp = np.sum(output_y_len_bin[i:])  # count how many y_len is above i
            output_y_len.extend(
                [min(i, y.size(2))] * count_temp
            )  # put them in output_y_len; max value should not exceed y.size(2)
        # pack into variable
        x = Variable(x).cuda()
        y = Variable(y).cuda()
        output_x = Variable(output_x).cuda()
        output_y = Variable(output_y).cuda()
        # print(output_y_len)
        # print('len',len(output_y_len))
        # print('y',y.size())
        # print('output_y',output_y.size())

        # if using ground truth to train
        h = rnn(x, pack=True, input_len=y_len)
        h = pack_padded_sequence(
            h, y_len, batch_first=True
        ).data  # get packed hidden vector
        # reverse h
        idx = [i for i in range(h.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx)).cuda()
        h = h.index_select(0, idx)
        hidden_null = Variable(
            torch.zeros(args.num_layers - 1, h.size(0), h.size(1))
        ).cuda()
        output.hidden = torch.cat(
            (h.view(1, h.size(0), h.size(1)), hidden_null), dim=0
        )  # num_layers, batch_size, hidden_size
        y_pred = output(output_x, pack=True, input_len=output_y_len)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        output_y = pack_padded_sequence(output_y, output_y_len, batch_first=True)
        output_y = pad_packed_sequence(output_y, batch_first=True)[0]
        # use cross entropy loss
        loss = binary_cross_entropy_weight(y_pred, output_y)

        if (
            epoch % args.epochs_log == 0 and batch_idx == 0
        ):  # only output first batch's statistics
            print(
                "Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}".format(
                    epoch,
                    args.epochs,
                    loss.data[0],
                    args.graph_type,
                    args.num_layers,
                    args.hidden_size_rnn,
                )
            )

        # logging
        log_value(
            "loss_" + args.fname, loss.data[0], epoch * args.batch_ratio + batch_idx
        )
        # print(y_pred.size())
        feature_dim = y_pred.size(0) * y_pred.size(1)
        loss_sum += loss.data[0] * feature_dim / y.size(0)
    return loss_sum / (batch_idx + 1)


########### train function for LSTM + VAE


def train_double(args, dataset_train, dataset_train0, rnn, output, rnn0, output0):

    # check if load existing model
    if args.load:
        print("*" * 50)
        print("args.load", args.load)
        fname = (
            args.model_save_path
            + args.fname
            + "_double"
            + "lstm_"
            + str(args.load_epoch)
            + ".dat"
        )
        print("fname", fname)
        rnn.load_state_dict(torch.load(fname))
        fname = (
            args.model_save_path
            + args.fname
            + "_double"
            + "output_"
            + str(args.load_epoch)
            + ".dat"
        )
        output.load_state_dict(torch.load(fname))

        args.lr = 0.00001
        epoch = args.load_epoch
        print("model loaded!, lr: {}".format(args.lr))
    else:
        epoch = 1

    # epoch = 1
    # initialize optimizer

    optimizer_rnn = optim.Adam(list(rnn.parameters()), lr=args.lr)
    optimizer_output = optim.Adam(list(output.parameters()), lr=args.lr)

    optimizer_rnn0 = optim.Adam(list(rnn0.parameters()), lr=args.lr)
    optimizer_output0 = optim.Adam(list(output0.parameters()), lr=args.lr)

    scheduler_rnn = MultiStepLR(
        optimizer_rnn, milestones=args.milestones, gamma=args.lr_rate
    )
    scheduler_output = MultiStepLR(
        optimizer_output, milestones=args.milestones, gamma=args.lr_rate
    )

    scheduler_rnn0 = MultiStepLR(
        optimizer_rnn0, milestones=args.milestones, gamma=args.lr_rate
    )
    scheduler_output0 = MultiStepLR(
        optimizer_output0, milestones=args.milestones, gamma=args.lr_rate
    )

    # start main loop
    time_all = np.zeros(args.epochs)
    while epoch <= args.epochs:
        time_start = tm.time()
        # train
        if "GraphRNN_VAE" in args.note:
            train_vae_epoch(
                epoch,
                args,
                rnn,
                output,
                dataset_train,
                optimizer_rnn,
                optimizer_output,
                scheduler_rnn,
                scheduler_output,
            )
        elif "GraphRNN_MLP" in args.note:
            train_mlp_epoch(
                epoch,
                args,
                rnn,
                output,
                dataset_train,
                optimizer_rnn,
                optimizer_output,
                scheduler_rnn,
                scheduler_output,
            )
        elif "GraphRNN_RNN" in args.note:

            train_rnn_double(
                epoch,
                args,
                rnn,
                output,
                rnn0,
                output0,
                dataset_train,
                dataset_train0,
                optimizer_rnn,
                optimizer_output,
                optimizer_rnn0,
                optimizer_output0,
                scheduler_rnn,
                scheduler_output,
                scheduler_rnn0,
                scheduler_output0,
            )

        time_end = tm.time()
        time_all[epoch - 1] = time_end - time_start
        # test
        if epoch % args.epochs_test == 0 and epoch >= args.epochs_test_start:
            for sample_time in range(1, 4):
                G_pred = []
                G_pred0 = []
                while len(G_pred) < args.test_total_size:
                    if "GraphRNN_VAE" in args.note:
                        G_pred_step = test_vae_epoch(
                            epoch,
                            args,
                            rnn,
                            output,
                            test_batch_size=args.test_batch_size,
                            sample_time=sample_time,
                        )
                    elif "GraphRNN_MLP" in args.note:
                        G_pred_step = test_mlp_epoch(
                            epoch,
                            args,
                            rnn,
                            output,
                            test_batch_size=args.test_batch_size,
                            sample_time=sample_time,
                        )
                    elif "GraphRNN_RNN" in args.note:
                        # G_pred_step0  = test_rnn_double(epoch, args, rnn0, output0, test_batch_size=args.test_batch_size)

                        G_pred_step, G_pred_step0 = test_rnn_double(
                            epoch,
                            args,
                            rnn,
                            output,
                            rnn0,
                            output0,
                            test_batch_size=args.test_batch_size,
                        )

                        G_pred0.extend(G_pred_step0)

                    G_pred.extend(G_pred_step)

                # save graphs
                fname = (
                    args.graph_save_path
                    + args.fname_pred
                    + "_double"
                    + str(epoch)
                    + "_"
                    + str(sample_time)
                    + ".dat"
                )
                # save_graph_list(G_pred, fname)
                if "GraphRNN_RNN" in args.note:
                    break
            print("test done, graphs saved")

        # save model checkpoint
        #     if args.save:
        #         if epoch % args.epochs_save == 0:
        #             fname = args.model_save_path + args.fname + 'lstm_' + str(epoch) + '.dat'
        #             torch.save(rnn.state_dict(), fname)
        #             fname = args.model_save_path + args.fname + 'output_' + str(epoch) + '.dat'
        #             torch.save(output.state_dict(), fname)
        epoch += 1
    # np.save(args.timing_save_path+args.fname,time_all)

    G_test = []

    G_test0 = []

    # print("args.test_total_size and total training time", args.test_total_size, tm.time()-t)
    while len(G_test) < args.test_total_size:

        G_pred_step, G_pred_step0 = test_rnn_double(
            epoch,
            args,
            rnn,
            output,
            rnn0,
            output0,
            test_batch_size=args.test_batch_size,
        )

        # G_pred_step0 = test_rnn_double(epoch, args, rnn0, output0, test_batch_size=args.test_batch_size)
        print("len(G_pred_step)", len(G_pred_step0))

        G_test.extend(G_pred_step)

        G_test0.extend(G_pred_step0)

    return G_test, G_pred, G_test0, G_pred0


########### for graph completion task
def train_graph_completion(args, dataset_test, rnn, output):
    fname = (
        args.model_save_path
        + args.fname
        + "_double"
        + "lstm_"
        + str(args.load_epoch)
        + ".dat"
    )
    rnn.load_state_dict(torch.load(fname))
    fname = (
        args.model_save_path
        + args.fname
        + "_double"
        + "output_"
        + str(args.load_epoch)
        + ".dat"
    )
    output.load_state_dict(torch.load(fname))

    epoch = args.load_epoch
    print("model loaded!, epoch: {}".format(args.load_epoch))

    for sample_time in range(1, 4):
        if "GraphRNN_MLP" in args.note:
            G_pred = test_mlp_partial_simple_epoch(
                epoch, args, rnn, output, dataset_test, sample_time=sample_time
            )
        if "GraphRNN_VAE" in args.note:
            G_pred = test_vae_partial_epoch(
                epoch, args, rnn, output, dataset_test, sample_time=sample_time
            )
        # save graphs
        fname = (
            args.graph_save_path
            + args.fname_pred
            + "_double"
            + str(epoch)
            + "_"
            + str(sample_time)
            + "graph_completion.dat"
        )
        save_graph_list(G_pred, fname)
    print("graph completion done, graphs saved")


########### for NLL evaluation
def train_nll(
    args,
    dataset_train,
    dataset_test,
    rnn,
    output,
    graph_validate_len,
    graph_test_len,
    max_iter=1000,
):
    fname = args.model_save_path + args.fname + "lstm_" + str(args.load_epoch) + ".dat"
    rnn.load_state_dict(torch.load(fname))
    fname = (
        args.model_save_path + args.fname + "output_" + str(args.load_epoch) + ".dat"
    )
    output.load_state_dict(torch.load(fname))

    epoch = args.load_epoch
    print("model loaded!, epoch: {}".format(args.load_epoch))
    fname_output = args.nll_save_path + args.note + "_" + args.graph_type + ".csv"
    with open(fname_output, "w+") as f:
        f.write(str(graph_validate_len) + "," + str(graph_test_len) + "\n")
        f.write("train,test\n")
        for iter in range(max_iter):
            if "GraphRNN_MLP" in args.note:
                nll_train = train_mlp_forward_epoch(
                    epoch, args, rnn, output, dataset_train
                )
                nll_test = train_mlp_forward_epoch(
                    epoch, args, rnn, output, dataset_test
                )
            if "GraphRNN_RNN" in args.note:
                nll_train = train_rnn_forward_epoch(
                    epoch, args, rnn, output, dataset_train
                )
                nll_test = train_rnn_forward_epoch(
                    epoch, args, rnn, output, dataset_test
                )
            print("train", nll_train, "test", nll_test)
            f.write(str(nll_train) + "," + str(nll_test) + "\n")

    print("NLL evaluation done")
