import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

EPS = np.finfo(np.float32).eps

__all__ = ["GRANMixtureBernoulli"]

# DISCLAIMER:
# Boilerplate parts of this code file were originally forked from
# https://github.com/lrjconan/GRAN


class GNN(nn.Module):
    def __init__(
        self,
        msg_dim,
        node_state_dim,
        edge_feat_dim,
        num_prop=1,
        num_layer=1,
        has_attention=True,
        att_hidden_dim=128,
        has_residual=False,
        has_graph_output=False,
        output_hidden_dim=128,
        graph_output_dim=None,
    ):
        super(GNN, self).__init__()
        self.msg_dim = msg_dim
        self.node_state_dim = node_state_dim
        self.edge_feat_dim = edge_feat_dim
        self.num_prop = num_prop
        self.num_layer = num_layer
        self.has_attention = has_attention
        self.has_residual = has_residual
        self.att_hidden_dim = att_hidden_dim
        self.has_graph_output = has_graph_output
        self.output_hidden_dim = output_hidden_dim
        self.graph_output_dim = graph_output_dim

        self.update_func = nn.ModuleList(
            [
                nn.GRUCell(input_size=self.msg_dim, hidden_size=self.node_state_dim)
                for _ in range(self.num_layer)
            ]
        )

        self.msg_func = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        nn.Linear(
                            self.node_state_dim + self.edge_feat_dim, self.msg_dim
                        ),
                        nn.ReLU(),
                        nn.Linear(self.msg_dim, self.msg_dim),
                    ]
                )
                for _ in range(self.num_layer)
            ]
        )

        if self.has_attention:
            self.att_head = nn.ModuleList(
                [
                    nn.Sequential(
                        *[
                            nn.Linear(
                                self.node_state_dim + self.edge_feat_dim,
                                self.att_hidden_dim,
                            ),
                            nn.ReLU(),
                            nn.Linear(self.att_hidden_dim, self.msg_dim),
                            nn.Sigmoid(),
                        ]
                    )
                    for _ in range(self.num_layer)
                ]
            )

        if self.has_graph_output:
            self.graph_output_head_att = nn.Sequential(
                *[
                    nn.Linear(self.node_state_dim, self.output_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.output_hidden_dim, 1),
                    nn.Sigmoid(),
                ]
            )

            self.graph_output_head = nn.Sequential(
                *[nn.Linear(self.node_state_dim, self.graph_output_dim)]
            )

    def _prop(self, state, edge, edge_feat, layer_idx=0):
        ### compute message
        state_diff = state[edge[:, 0], :] - state[edge[:, 1], :]
        if self.edge_feat_dim > 0:
            edge_input = torch.cat([state_diff, edge_feat], dim=1)
        else:
            edge_input = state_diff

        msg = self.msg_func[layer_idx](edge_input)
        # msg0 = self.msg_func[layer_idx0](edge_input)
        ### attention on messages
        if self.has_attention:
            att_weight = self.att_head[layer_idx](edge_input)
            msg = msg * att_weight

        ### aggregate message by sum
        state_msg = torch.zeros(state.shape[0], msg.shape[1]).to(state.device)
        scatter_idx = edge[:, [1]].expand(-1, msg.shape[1])
        state_msg = state_msg.scatter_add(0, scatter_idx, msg)

        ### state update
        state = self.update_func[layer_idx](state_msg, state)
        return state

    def forward(self, node_feat, edge, edge_feat, graph_idx=None):
        """
        N.B.: merge a batch of graphs as a single graph

        node_feat: N X D, node feature
        edge: M X 2, edge indices
        edge_feat: M X D', edge feature
        graph_idx: N X 1, graph indices
        """

        state = node_feat
        prev_state = state
        for ii in range(self.num_layer):
            if ii > 0:
                state = F.relu(state)

            for jj in range(self.num_prop):
                state = self._prop(state, edge, edge_feat=edge_feat, layer_idx=ii)

        if self.has_residual:
            state = state + prev_state

        if self.has_graph_output:
            num_graph = graph_idx.max() + 1
            node_att_weight = self.graph_output_head_att(state)
            node_output = self.graph_output_head(state)

            # weighted average
            reduce_output = torch.zeros(num_graph, node_output.shape[1]).to(
                node_feat.device
            )
            reduce_output = reduce_output.scatter_add(
                0,
                graph_idx.unsqueeze(1).expand(-1, node_output.shape[1]),
                node_output * node_att_weight,
            )

            const = torch.zeros(num_graph).to(node_feat.device)
            const = const.scatter_add(
                0, graph_idx, torch.ones(node_output.shape[0]).to(node_feat.device)
            )

            reduce_output = reduce_output / const.view(-1, 1)

            return reduce_output
        else:
            return state


class GRANMixtureBernoulli_double(nn.Module):
    """Graph Recurrent Attention Networks"""

    def __init__(self, config):
        super(GRANMixtureBernoulli_double, self).__init__()
        self.config = config
        self.device = config.device
        self.max_num_nodes = config.model.max_num_nodes
        self.hidden_dim = config.model.hidden_dim
        self.is_sym = config.model.is_sym
        self.block_size = config.model.block_size
        self.sample_stride = config.model.sample_stride
        self.num_GNN_prop = config.model.num_GNN_prop
        self.num_GNN_layers = config.model.num_GNN_layers
        self.edge_weight = (
            config.model.edge_weight if hasattr(config.model, "edge_weight") else 1.0
        )
        self.dimension_reduce = config.model.dimension_reduce
        self.has_attention = config.model.has_attention
        self.num_canonical_order = config.model.num_canonical_order
        self.output_dim = 1
        self.num_mix_component = config.model.num_mix_component
        self.has_rand_feat = False  # use random feature instead of 1-of-K encoding
        self.att_edge_dim = 64

        self.output_theta = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.output_dim * self.num_mix_component),
        )

        self.output_alpha = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.num_mix_component),
        )

        self.output_theta0 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.output_dim * self.num_mix_component),
        )

        self.output_alpha0 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.num_mix_component),
        )

        if self.dimension_reduce:
            self.embedding_dim = config.model.embedding_dim
            self.decoder_input = nn.Sequential(
                nn.Linear(self.max_num_nodes, self.embedding_dim)
            )
        else:
            self.embedding_dim = self.max_num_nodes

        self.decoder = GNN(
            msg_dim=self.hidden_dim,
            node_state_dim=self.hidden_dim,
            edge_feat_dim=2 * self.att_edge_dim,
            num_prop=self.num_GNN_prop,
            num_layer=self.num_GNN_layers,
            has_attention=self.has_attention,
        )

        self.decoder0 = GNN(
            msg_dim=self.hidden_dim,
            node_state_dim=self.hidden_dim,
            edge_feat_dim=2 * self.att_edge_dim,
            num_prop=self.num_GNN_prop,
            num_layer=self.num_GNN_layers,
            has_attention=self.has_attention,
        )

        ### Loss functions
        pos_weight = torch.ones([1]) * self.edge_weight
        self.adj_loss_func = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight, reduction="none"
        )

        self.adj_loss_func0 = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight, reduction="none"
        )

    def _inference(
        self,
        A_pad=None,
        A_pad0=None,
        edges=None,
        edges0=None,
        node_idx_gnn=None,
        node_idx_gnn0=None,
        node_idx_feat=None,
        node_idx_feat0=None,
        att_idx=None,
        att_idx0=None,
    ):
        """generate adj in row-wise auto-regressive fashion"""

        B, C, N_max, _ = A_pad.shape
        H = self.hidden_dim
        K = self.block_size
        A_pad = A_pad.view(B * C * N_max, -1)

        B0, C0, N_max0, _ = A_pad0.shape
        H0 = self.hidden_dim
        K0 = self.block_size
        A_pad0 = A_pad0.view(B0 * C0 * N_max0, -1)

        if self.dimension_reduce:
            node_feat = self.decoder_input(A_pad)  # BCN_max X H
            node_feat0 = self.decoder_input(A_pad0)
        else:
            node_feat = A_pad  # BCN_max X N_max
            node_feat0 = A_pad0

        ### GNN inference
        # pad zero as node feature for newly generated nodes (1st row)
        node_feat = F.pad(
            node_feat, (0, 0, 1, 0), "constant", value=0.0
        )  # (BCN_max + 1) X N_max

        node_feat0 = F.pad(node_feat0, (0, 0, 1, 0), "constant", value=0.0)
        # create symmetry-breaking edge feature for the newly generated nodes
        att_idx = att_idx.view(-1, 1)

        att_idx0 = att_idx.view(-1, 1)

        if self.has_rand_feat:
            # create random feature
            att_edge_feat = torch.zeros(edges.shape[0], 2 * self.att_edge_dim).to(
                node_feat.device
            )
            idx_new_node = (att_idx[[edges[:, 0]]] > 0).long() + (
                att_idx[[edges[:, 1]]] > 0
            ).long()
            idx_new_node = idx_new_node.byte().squeeze()
            att_edge_feat[idx_new_node, :] = torch.randn(
                idx_new_node.long().sum(), att_edge_feat.shape[1]
            ).to(node_feat.device)

            att_edge_feat0 = torch.zeros(edges0.shape[0], 2 * self.att_edge_dim).to(
                node_feat0.device
            )
            idx_new_node0 = (att_idx0[[edges0[:, 0]]] > 0).long() + (
                att_idx0[[edges0[:, 1]]] > 0
            ).long()
            idx_new_node0 = idx_new_node0.byte().squeeze()
            att_edge_feat0[idx_new_node0, :] = torch.randn(
                idx_new_node0.long().sum(), att_edge_feat0.shape[1]
            ).to(node_feat0.device)

        else:
            # create one-hot feature
            att_edge_feat = torch.zeros(edges.shape[0], 2 * self.att_edge_dim).to(
                node_feat.device
            )
            # scatter with empty index seems to cause problem on CPU but not on GPU
            att_edge_feat = att_edge_feat.scatter(1, att_idx[[edges[:, 0]]], 1)
            att_edge_feat = att_edge_feat.scatter(
                1, att_idx[[edges[:, 1]]] + self.att_edge_dim, 1
            )

            att_edge_feat0 = torch.zeros(edges0.shape[0], 2 * self.att_edge_dim).to(
                node_feat0.device
            )
            # scatter with empty index seems to cause problem on CPU but not on GPU
            att_edge_feat0 = att_edge_feat0.scatter(1, att_idx0[[edges0[:, 0]]], 1)
            att_edge_feat0 = att_edge_feat0.scatter(
                1, att_idx0[[edges0[:, 1]]] + self.att_edge_dim, 1
            )

        # GNN inference
        # N.B.: node_feat is shared by multiple subgraphs within the same batch
        self.node_state = self.decoder(
            node_feat[node_idx_feat], edges, edge_feat=att_edge_feat
        )

        self.node_state0 = self.decoder(
            node_feat0[node_idx_feat0], edges0, edge_feat=att_edge_feat0
        )
        ### Pairwise predict edges
        diff = (
            self.node_state[node_idx_gnn[:, 0], :]
            - self.node_state[node_idx_gnn[:, 1], :]
        )

        diff0 = (
            self.node_state0[node_idx_gnn0[:, 0], :]
            - self.node_state0[node_idx_gnn0[:, 1], :]
        )

        log_theta = self.output_theta(diff)  # B X (tt+K)K
        log_alpha = self.output_alpha(diff)  # B X (tt+K)K
        log_theta = log_theta.view(-1, self.num_mix_component)  # B X CN(N-1)/2 X K
        log_alpha = log_alpha.view(-1, self.num_mix_component)  # B X CN(N-1)/2 X K

        log_theta0 = self.output_theta0(diff0)
        log_alpha0 = self.output_alpha0(diff0)
        log_theta0 = log_theta0.view(-1, self.num_mix_component)
        log_alpha0 = log_alpha0.view(-1, self.num_mix_component)

        # return log_theta, log_alpha, log_theta0, log_alpha0
        return log_theta, log_alpha, log_theta0, log_alpha0

    def _sampling(self, B):
        """generate adj in row-wise auto-regressive fashion"""
        with torch.no_grad():

            K = self.block_size
            S = self.sample_stride
            H = self.hidden_dim
            N = self.max_num_nodes
            mod_val = (N - K) % S
            if mod_val > 0:
                N_pad = N - K - mod_val + int(np.ceil((K + mod_val) / S)) * S
            else:
                N_pad = N

            A = torch.zeros(B, N_pad, N_pad).to(self.device)
            dim_input = (
                self.embedding_dim if self.dimension_reduce else self.max_num_nodes
            )

            A0 = torch.zeros(B, N_pad, N_pad).to(self.device)

            ### cache node state for speed up
            node_state = torch.zeros(B, N_pad, dim_input).to(self.device)

            node_state0 = torch.zeros(B, N_pad, dim_input).to(self.device)

            for ii in range(0, N_pad, S):
                # for ii in range(0, 3530, S):
                jj = ii + K
                if jj > N_pad:
                    break

                # reset to discard overlap generation
                A[:, ii:, :] = 0.0
                A = torch.tril(A, diagonal=-1)

                A0[:, ii:, :] = 0.0
                A0 = torch.tril(A0, diagonal=-1)

                if ii >= K:
                    if self.dimension_reduce:
                        node_state[:, ii - K : ii, :] = self.decoder_input(
                            A[:, ii - K : ii, :N]
                        )

                        node_state0[:, ii - K : ii, :] = self.decoder_input(
                            A0[:, ii - K : ii, :N]
                        )

                    else:
                        node_state[:, ii - K : ii, :] = A[:, ii - S : ii, :N]

                        node_state0[:, ii - K : ii, :] = A0[:, ii - S : ii, :N]
                else:
                    if self.dimension_reduce:
                        node_state[:, :ii, :] = self.decoder_input(A[:, :ii, :N])

                        node_state0[:, :ii, :] = self.decoder_input(A0[:, :ii, :N])
                    else:
                        node_state[:, :ii, :] = A[:, ii - S : ii, :N]

                        node_state0[:, :ii, :] = A0[:, ii - S : ii, :N]

                node_state_in = F.pad(
                    node_state[:, :ii, :], (0, 0, 0, K), "constant", value=0.0
                )

                node_state_in0 = F.pad(
                    node_state0[:, :ii, :], (0, 0, 0, K), "constant", value=0.0
                )

                ### GNN propagation
                adj = F.pad(
                    A[:, :ii, :ii], (0, K, 0, K), "constant", value=1.0
                )  # B X jj X jj
                adj = torch.tril(adj, diagonal=-1)
                adj = adj + adj.transpose(1, 2)

                adj0 = F.pad(
                    A0[:, :ii, :ii], (0, K, 0, K), "constant", value=1.0
                )  # B X jj X jj
                adj0 = torch.tril(adj0, diagonal=-1)
                adj0 = adj0 + adj0.transpose(1, 2)

                edges = [
                    adj[bb].to_sparse().coalesce().indices() + bb * adj.shape[1]
                    for bb in range(B)
                ]
                edges = torch.cat(edges, dim=1).t()

                edges0 = [
                    adj0[bb].to_sparse().coalesce().indices() + bb * adj0.shape[1]
                    for bb in range(B)
                ]
                edges0 = torch.cat(edges0, dim=1).t()

                att_idx = torch.cat(
                    [torch.zeros(ii).long(), torch.arange(1, K + 1)]
                ).to(self.device)
                att_idx = att_idx.view(1, -1).expand(B, -1).contiguous().view(-1, 1)

                att_idx0 = torch.cat(
                    [torch.zeros(ii).long(), torch.arange(1, K + 1)]
                ).to(self.device)
                att_idx0 = att_idx0.view(1, -1).expand(B, -1).contiguous().view(-1, 1)

                if self.has_rand_feat:
                    # create random feature
                    att_edge_feat = torch.zeros(
                        edges.shape[0], 2 * self.att_edge_dim
                    ).to(self.device)
                    idx_new_node = (att_idx[[edges[:, 0]]] > 0).long() + (
                        att_idx[[edges[:, 1]]] > 0
                    ).long()
                    idx_new_node = idx_new_node.byte().squeeze()
                    att_edge_feat[idx_new_node, :] = torch.randn(
                        idx_new_node.long().sum(), att_edge_feat.shape[1]
                    ).to(self.device)

                    att_edge_feat0 = torch.zeros(
                        edges0.shape[0], 2 * self.att_edge_dim
                    ).to(self.device)
                    idx_new_node0 = (att_idx0[[edges0[:, 0]]] > 0).long() + (
                        att_idx0[[edges0[:, 1]]] > 0
                    ).long()
                    idx_new_node0 = idx_new_node0.byte().squeeze()
                    att_edge_feat0[idx_new_node0, :] = torch.randn(
                        idx_new_node0.long().sum(), att_edge_feat0.shape[1]
                    ).to(self.device)

                else:
                    # create one-hot feature
                    att_edge_feat = torch.zeros(
                        edges.shape[0], 2 * self.att_edge_dim
                    ).to(self.device)
                    att_edge_feat = att_edge_feat.scatter(1, att_idx[[edges[:, 0]]], 1)
                    att_edge_feat = att_edge_feat.scatter(
                        1, att_idx[[edges[:, 1]]] + self.att_edge_dim, 1
                    )

                    att_edge_feat0 = torch.zeros(
                        edges0.shape[0], 2 * self.att_edge_dim
                    ).to(self.device)
                    att_edge_feat0 = att_edge_feat0.scatter(
                        1, att_idx0[[edges0[:, 0]]], 1
                    )
                    att_edge_feat0 = att_edge_feat0.scatter(
                        1, att_idx0[[edges0[:, 1]]] + self.att_edge_dim, 1
                    )

                node_state_out = self.decoder(
                    node_state_in.view(-1, H), edges, edge_feat=att_edge_feat
                )
                node_state_out = node_state_out.view(B, jj, -1)

                node_state_out0 = self.decoder(
                    node_state_in0.view(-1, H), edges0, edge_feat=att_edge_feat0
                )
                node_state_out0 = node_state_out0.view(B, jj, -1)

                idx_row, idx_col = np.meshgrid(np.arange(ii, jj), np.arange(jj))
                idx_row = torch.from_numpy(idx_row.reshape(-1)).long().to(self.device)
                idx_col = torch.from_numpy(idx_col.reshape(-1)).long().to(self.device)

                idx_row0, idx_col0 = np.meshgrid(np.arange(ii, jj), np.arange(jj))
                idx_row0 = torch.from_numpy(idx_row0.reshape(-1)).long().to(self.device)
                idx_col0 = torch.from_numpy(idx_col0.reshape(-1)).long().to(self.device)

                diff = (
                    node_state_out[:, idx_row, :] - node_state_out[:, idx_col, :]
                )  # B X (ii+K)K X H
                diff = diff.view(-1, node_state.shape[2])
                log_theta = self.output_theta(diff)
                log_alpha = self.output_alpha(diff)

                diff0 = (
                    node_state_out0[:, idx_row0, :] - node_state_out0[:, idx_col0, :]
                )  # B X (ii+K)K X H
                diff0 = diff0.view(-1, node_state0.shape[2])
                log_theta0 = self.output_theta0(diff0)
                log_alpha0 = self.output_alpha0(diff0)

                log_theta = log_theta.view(
                    B, -1, K, self.num_mix_component
                )  # B X K X (ii+K) X L
                log_theta = log_theta.transpose(1, 2)  # B X (ii+K) X K X L

                log_theta0 = log_theta0.view(
                    B, -1, K, self.num_mix_component
                )  # B X K X (ii+K) X L
                log_theta0 = log_theta0.transpose(1, 2)

                log_alpha = log_alpha.view(
                    B, -1, self.num_mix_component
                )  # B X K X (ii+K)
                prob_alpha = F.softmax(log_alpha.mean(dim=1), -1)
                alpha = torch.multinomial(prob_alpha, 1).squeeze(dim=1).long()

                log_alpha0 = log_alpha0.view(
                    B, -1, self.num_mix_component
                )  # B X K X (ii+K)
                prob_alpha0 = F.softmax(log_alpha0.mean(dim=1), -1)
                alpha0 = torch.multinomial(prob_alpha0, 1).squeeze(dim=1).long()

                prob = []
                for bb in range(B):
                    prob += [torch.sigmoid(log_theta[bb, :, :, alpha[bb]])]

                prob = torch.stack(prob, dim=0)
                A[:, ii:jj, :jj] = torch.bernoulli(prob[:, : jj - ii, :])

                prob0 = []
                for bb in range(B):
                    prob0 += [torch.sigmoid(log_theta0[bb, :, :, alpha0[bb]])]

                prob0 = torch.stack(prob0, dim=0)
                A0[:, ii:jj, :jj] = torch.bernoulli(prob0[:, : jj - ii, :])

            ### make it symmetric
            if self.is_sym:
                A = torch.tril(A, diagonal=-1)
                A = A + A.transpose(1, 2)

                A0 = torch.tril(A0, diagonal=-1)
                A0 = A0 + A0.transpose(1, 2)

            return A, A0

    def forward(self, input_dict):
        """
        B: batch size
        N: number of rows/columns in mini-batch
        N_max: number of max number of rows/columns
        M: number of augmented edges in mini-batch
        H: input dimension of GNN
        K: block size
        E: number of edges in mini-batch
        S: stride
        C: number of canonical orderings
        D: number of mixture Bernoulli

        Args:
          A_pad: B X C X N_max X N_max, padded adjacency matrix
          node_idx_gnn: M X 2, node indices of augmented edges
          node_idx_feat: N X 1, node indices of subgraphs for indexing from feature
                        (0 indicates indexing from 0-th row of feature which is
                          always zero and corresponds to newly generated nodes)
          att_idx: N X 1, one-hot encoding of newly generated nodes
                        (0 indicates existing nodes, 1-D indicates new nodes in
                          the to-be-generated block)
          subgraph_idx: E X 1, indices corresponding to augmented edges
                        (representing which subgraph in mini-batch the augmented
                        edge belongs to)
          edges: E X 2, edge as [incoming node index, outgoing node index]
          label: E X 1, binary label of augmented edges
          num_nodes_pmf: N_max, empirical probability mass function of number of nodes

        Returns:
          loss                        if training
          list of adjacency matrices  else
        """
        is_sampling = (
            input_dict["is_sampling"] if "is_sampling" in input_dict else False
        )
        batch_size = input_dict["batch_size"] if "batch_size" in input_dict else None

        A_pad = input_dict["adj"] if "adj" in input_dict else None

        A_pad0 = input_dict["adj0"] if "adj0" in input_dict else None

        node_idx_gnn = (
            input_dict["node_idx_gnn"] if "node_idx_gnn" in input_dict else None
        )

        node_idx_gnn0 = (
            input_dict["node_idx_gnn0"] if "node_idx_gnn0" in input_dict else None
        )

        node_idx_feat = (
            input_dict["node_idx_feat"] if "node_idx_feat" in input_dict else None
        )

        node_idx_feat0 = (
            input_dict["node_idx_feat0"] if "node_idx_feat0" in input_dict else None
        )

        att_idx = input_dict["att_idx"] if "att_idx" in input_dict else None

        att_idx0 = input_dict["att_idx0"] if "att_idx0" in input_dict else None

        subgraph_idx = (
            input_dict["subgraph_idx"] if "subgraph_idx" in input_dict else None
        )

        subgraph_idx0 = (
            input_dict["subgraph_idx0"] if "subgraph_idx0" in input_dict else None
        )

        edges = input_dict["edges"] if "edges" in input_dict else None

        edges0 = input_dict["edges0"] if "edges0" in input_dict else None

        label = input_dict["label"] if "label" in input_dict else None

        label0 = input_dict["label0"] if "label0" in input_dict else None

        num_nodes_pmf = (
            input_dict["num_nodes_pmf"] if "num_nodes_pmf" in input_dict else None
        )

        num_nodes_pmf0 = (
            input_dict["num_nodes_pmf0"] if "num_nodes_pmf0" in input_dict else None
        )

        subgraph_idx_base = (
            input_dict["subgraph_idx_base"]
            if "subgraph_idx_base" in input_dict
            else None
        )

        subgraph_idx_base0 = (
            input_dict["subgraph_idx_base0"]
            if "subgraph_idx_base0" in input_dict
            else None
        )

        N_max = self.max_num_nodes

        if not is_sampling:
            B, _, N, _ = A_pad.shape

            ### compute adj loss
            log_theta, log_alpha, log_theta0, log_alpha0 = self._inference(
                A_pad=A_pad,
                A_pad0=A_pad0,
                edges=edges,
                edges0=edges0,
                node_idx_gnn=node_idx_gnn,
                node_idx_gnn0=node_idx_gnn0,
                node_idx_feat=node_idx_feat,
                node_idx_feat0=node_idx_feat0,
                att_idx=att_idx,
                att_idx0=att_idx0,
            )

            num_edges = log_theta.shape[0]
            num_edges0 = log_theta0.shape[0]

            adj_loss = mixture_bernoulli_loss(
                self,
                label,
                label0,
                log_theta,
                log_theta0,
                log_alpha,
                log_alpha0,
                self.adj_loss_func,
                self.adj_loss_func0,
                subgraph_idx,
                subgraph_idx0,
                subgraph_idx_base,
                subgraph_idx_base0,
                self.num_canonical_order,
            )

            return adj_loss
        else:
            A, A0 = self._sampling(batch_size)

            ### sample number of nodes
            num_nodes_pmf = torch.from_numpy(num_nodes_pmf).to(self.device)

            num_nodes_pmf0 = torch.from_numpy(num_nodes_pmf0).to(self.device)

            num_nodes = (
                torch.multinomial(num_nodes_pmf, batch_size, replacement=True) + 1
            )  # shape B X 1

            num_nodes0 = (
                torch.multinomial(num_nodes_pmf0, batch_size, replacement=True) + 1
            )  # shape B X 1

            A_list = [
                A[ii, : num_nodes[ii], : num_nodes[ii]] for ii in range(batch_size)
            ]

            A_list0 = [
                A0[ii, : num_nodes0[ii], : num_nodes0[ii]] for ii in range(batch_size)
            ]

            return A_list, A_list0


def mixture_bernoulli_loss(
    self,
    label,
    label0,
    log_theta,
    log_theta0,
    log_alpha,
    log_alpha0,
    adj_loss_func,
    adj_loss_func0,
    subgraph_idx,
    subgraph_idx0,
    subgraph_idx_base,
    subgraph_idx_base0,
    num_canonical_order,
    sum_order_log_prob=False,
    return_neg_log_prob=False,
    reduction="mean",
):
    """
    Compute likelihood for mixture of Bernoulli model

    Args:
      label: E X 1, see comments above
      label0: E X 1, see comments above
      log_theta: E X D, see comments above
      log_alpha: E X D, see comments above
      log_theta0: E X D, see comments above
      log_alpha0: E X D, see comments above
      adj_loss_func: BCE loss
      adj_loss_func0: BCE loss
      subgraph_idx: E X 1, see comments above
      subgraph_idx0: E X 1, see comments above
      subgraph_idx_base: B+1, cumulative # of edges in the subgraphs associated with each batch
      subgraph_idx_base0: B+1, cumulative # of edges in the subgraphs associated with each batch
      num_canonical_order: int, number of node orderings considered
      sum_order_log_prob: boolean, if True sum the log prob of orderings instead of taking logsumexp
        i.e. log p(G, pi_1) + log p(G, pi_2) instead of log [p(G, pi_1) + p(G, pi_2)]
        This is equivalent to the original GRAN loss.
      return_neg_log_prob: boolean, if True also return neg log prob
      reduction: string, type of reduction on batch dimension ("mean", "sum", "none")

    Returns:
      loss (and potentially neg log prob)
    """

    num_subgraph = subgraph_idx_base[-1]  # == subgraph_idx.max() + 1

    num_subgraph0 = subgraph_idx_base0[-1]

    B = subgraph_idx_base.shape[0] - 1
    C = num_canonical_order
    E = log_theta.shape[0]
    K = log_theta.shape[1]
    assert E % C == 0
    adj_loss = torch.stack(
        [adj_loss_func(log_theta[:, kk], label) for kk in range(K)], dim=1
    )

    adj_loss0 = torch.stack(
        [adj_loss_func0(log_theta0[:, kk], label0) for kk in range(K)], dim=1
    )

    const = torch.zeros(num_subgraph).to(label.device)  # S

    const0 = torch.zeros(num_subgraph0).to(label0.device)

    const = const.scatter_add(0, subgraph_idx, torch.ones_like(subgraph_idx).float())

    const0 = const0.scatter_add(
        0, subgraph_idx0, torch.ones_like(subgraph_idx0).float()
    )

    reduce_adj_loss = torch.zeros(num_subgraph, K).to(label.device)

    reduce_adj_loss0 = torch.zeros(num_subgraph0, K).to(label0.device)

    reduce_adj_loss = reduce_adj_loss.scatter_add(
        0, subgraph_idx.unsqueeze(1).expand(-1, K), adj_loss
    )

    reduce_adj_loss0 = reduce_adj_loss0.scatter_add(
        0, subgraph_idx0.unsqueeze(1).expand(-1, K), adj_loss0
    )

    reduce_log_alpha = torch.zeros(num_subgraph, K).to(label.device)
    reduce_log_alpha = reduce_log_alpha.scatter_add(
        0, subgraph_idx.unsqueeze(1).expand(-1, K), log_alpha
    )
    reduce_log_alpha = reduce_log_alpha / const.view(-1, 1)
    reduce_log_alpha = F.log_softmax(reduce_log_alpha, -1)

    reduce_log_alpha0 = torch.zeros(num_subgraph0, K).to(label0.device)
    reduce_log_alpha0 = reduce_log_alpha0.scatter_add(
        0, subgraph_idx0.unsqueeze(1).expand(-1, K), log_alpha0
    )
    reduce_log_alpha0 = reduce_log_alpha0 / const.view(-1, 1)
    reduce_log_alpha0 = F.log_softmax(reduce_log_alpha0, -1)

    log_prob = -reduce_adj_loss + reduce_log_alpha
    log_prob = torch.logsumexp(log_prob, dim=1)  # S, K

    log_prob0 = -reduce_adj_loss0 + reduce_log_alpha0
    log_prob0 = torch.logsumexp(log_prob0, dim=1)

    bc_log_prob = torch.zeros([B * C]).to(label.device)  # B*C
    bc_log_prob0 = torch.zeros([B * C]).to(label0.device)
    bc_idx = torch.arange(B * C).to(label.device)  # B*C
    bc_idx0 = torch.arange(B * C).to(label0.device)
    bc_const = torch.zeros(B * C).to(label.device)
    bc_const0 = torch.zeros(B * C).to(label0.device)
    bc_size = (subgraph_idx_base[1:] - subgraph_idx_base[:-1]) // C  # B
    bc_size0 = (subgraph_idx_base0[1:] - subgraph_idx_base0[:-1]) // C
    bc_size = torch.repeat_interleave(bc_size, C)  # B*C
    bc_size0 = torch.repeat_interleave(bc_size0, C)
    bc_idx = torch.repeat_interleave(bc_idx, bc_size)  # S
    bc_idx0 = torch.repeat_interleave(bc_idx0, bc_size0)
    bc_log_prob = bc_log_prob.scatter_add(0, bc_idx, log_prob)
    bc_log_prob0 = bc_log_prob0.scatter_add(0, bc_idx0, log_prob0)
    # loss must be normalized for numerical stability
    bc_const = bc_const.scatter_add(0, bc_idx, const)
    bc_loss = bc_log_prob / bc_const

    bc_const0 = bc_const0.scatter_add(0, bc_idx0, const0)
    bc_loss0 = bc_log_prob0 / bc_const0

    bc_log_prob = bc_log_prob.reshape(B, C)
    bc_loss = bc_loss.reshape(B, C)

    bc_log_prob0 = bc_log_prob0.reshape(B, C)
    bc_loss0 = bc_loss0.reshape(B, C)

    m = len(self.node_state.cpu().detach().numpy())

    D = []
    for i in range(m):
        for j in range(m):
            D.append(
                np.linalg.norm(
                    self.node_state.cpu().detach().numpy()[i]
                    - self.node_state0.cpu().detach().numpy()[j]
                )
            )

    D = np.reshape(D, (m, m))

    if sum_order_log_prob:
        b_log_prob = torch.sum(bc_log_prob, dim=1)
        b_loss = torch.sum(bc_loss, dim=1)

        b_log_prob0 = torch.sum(bc_log_prob0, dim=1)
        b_loss0 = torch.sum(bc_loss0, dim=1)

    else:
        b_log_prob = torch.logsumexp(bc_log_prob, dim=1)
        b_loss = torch.logsumexp(bc_loss, dim=1)

        b_log_prob0 = torch.logsumexp(bc_log_prob0, dim=1)
        b_loss0 = torch.logsumexp(bc_loss0, dim=1)
    # probability calculation was for lower-triangular edges
    # must be squared to get probability for entire graph
    b_neg_log_prob = -2 * b_log_prob
    b_loss = -b_loss

    b_neg_log_prob0 = -2 * b_log_prob0
    b_loss0 = -b_loss0

    if reduction == "mean":
        neg_log_prob = b_neg_log_prob.mean()
        loss = b_loss.mean()

        neg_log_prob0 = b_neg_log_prob0.mean()
        loss0 = b_loss0.mean()

    elif reduction == "sum":
        neg_log_prob = b_neg_log_prob.sum()
        loss = b_loss.sum()

        neg_log_prob0 = b_neg_log_prob0.sum()
        loss0 = b_loss0.sum()

    else:
        assert reduction == "none"
        neg_log_prob = b_neg_log_prob
        loss = b_loss

        neg_log_prob0 = b_neg_log_prob0
        loss0 = b_loss0

    if return_neg_log_prob:
        return loss + loss0 + 0.1 * np.trace(D), neg_log_prob + neg_log_prob0
    else:
        return 10 * (loss + loss0) + np.trace(D)
