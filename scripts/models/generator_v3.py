import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as t_func
import torch_geometric as tg
# Import my modules
from scripts.layers import Edger, MLP
from scripts.utils import Utils, timeit
from scripts.utils import convert_adj_to_edges


class GeneratorNetV3(nn.Module):
    def __init__(self, z_dim=16, n_nodes=7,
                 hidden_dim=256, tau=1,
                 dataset='nas101', sub_skip=False):
        super(GeneratorNetV3, self).__init__()
        # Dimensionality of random input
        self.z_dim = z_dim
        # Maximum number of nodes
        self.n_nodes = n_nodes
        # Define input and output embeddings
        operations = Utils().get_class_name_dict(bench=dataset, sub_skip=sub_skip)
        self.n_ops = len(list(operations.keys()))
        # hidden state size of each vertex
        self.hidden_dim = hidden_dim

        self.tau = tau

        self.gru = nn.GRUCell(self.n_ops, hidden_dim)  # decoder GRU
        self.fc3 = nn.Linear(z_dim, hidden_dim)  # from latent z to initial hidden state h0
        self.node_adder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, self.n_ops - 1),
            nn.Sigmoid(),
            nn.BatchNorm1d(self.n_ops - 1)
        )  # which type of new vertex to add
        self.edge_adder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, 2),
            nn.Sigmoid(),
            nn.BatchNorm1d(2)
        )  # whether to add edge between v_i and v_new

        self.graph_conv = tg.nn.GCNConv(self.hidden_dim, self.hidden_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def get_device(self):
        return next(self.parameters()).device

    def forward(self, z):
        # Generate graph from random vectors z
        # Get number of graphs
        n_graphs = z.shape[0]
        # print('n_graphs: {}'.format(n_graphs))
        # Get encoding from random vector
        h = self.tanh(self.fc3(z))  # or relu activation, similar performance
        # Define input and output embeddings
        input_embedding = torch.zeros((self.n_ops,), device=self.get_device())
        input_embedding[0] = 1
        output_embedding = torch.zeros((self.n_ops,), device=self.get_device())
        output_embedding[-1] = 1
        # Define nodes of graphs and empty adjacency matrix
        nodes = torch.stack([input_embedding for _ in range(n_graphs)])
        adj_mat = torch.zeros((n_graphs, n_graphs), device=self.get_device(), dtype=torch.int64)
        # Define indices where input and output nodes are
        input_indices_list = [i for i in range(n_graphs)]
        output_indices_list = [i for i in range(n_graphs)]
        # self._update_v(G, 0, H0)  # h = self.graph_conv(h, edge_index)
        finished = [False for _ in range(n_graphs)]
        for idx in range(1, self.n_nodes):
            # decide the type of the next added vertex
            if idx == self.n_nodes - 1:  # force the last node to be output_embedding
                new_types = [output_embedding for _ in range(n_graphs)]
            else:
                # Get type scores from the graph embedding
                # type_scores = torch.cat((torch.zeros((self.node_adder(h).shape[0], 1), device=self.get_device()), self.node_adder(h)), dim=-1)
                type_scores = self.node_adder(h)
                # print('type_scores.shape: {}'.format(type_scores.shape))
                # print('type_scores: {}'.format(type_scores))
                new_types = t_func.gumbel_softmax(type_scores,
                                                  tau=self.tau,
                                                  hard=True,
                                                  dim=-1)
                new_types = torch.cat((torch.zeros((new_types.shape[0], 1), device=self.get_device()), new_types), dim=-1)
                # print('new_types: {}'.format(new_types))
            for index in range(n_graphs):
                if not finished[index]:
                    # Add node to the node matrix
                    # print('new_types: {}'.format(new_types))
                    # print('new_types[index]: {}'.format(new_types[index]))
                    # print('new_types[index].unsqueeze(dim=0): {}'.format(new_types[index].unsqueeze(dim=0)))
                    nodes = torch.cat((nodes[:output_indices_list[index]+1, :],
                                      new_types[index].unsqueeze(dim=0),
                                      nodes[output_indices_list[index]+1:, :]), dim=0)
                    # Add node to the adjacency matrix
                    column = torch.zeros((adj_mat.shape[0], 1), dtype=torch.int64, device=self.get_device())
                    # print('adj_mat: {}'.format(adj_mat))
                    adj_mat = torch.cat((adj_mat[:, :output_indices_list[index]+1],
                                        column,
                                        adj_mat[:, output_indices_list[index]+1:]), dim=1)
                    row = torch.zeros((1, adj_mat.shape[1]), dtype=torch.int64, device=self.get_device())
                    adj_mat = torch.cat((adj_mat[:output_indices_list[index]+1, :],
                                        row,
                                        adj_mat[output_indices_list[index]+1:, :]), dim=0)
                    # Update input and output indices
                    for i in range(index + 1, len(input_indices_list)):
                        input_indices_list[i] += 1
                    for i in range(index, len(output_indices_list)):
                        output_indices_list[i] += 1
            #     print('input indices are: {}'.format(input_indices_list))
            #     print('output indices are: {}'.format(output_indices_list))
            # print('nodes after appending new operations: {}'.format(nodes))
            h = self.graph_conv(self.gru(nodes), convert_adj_to_edges(adj_mat))

            # decide connections
            for vi in range(idx - 1, -1, -1):
                indices = torch.zeros((nodes.shape[0],), dtype=torch.int64, device=self.get_device())
                for ind in range(n_graphs):
                    if idx + ind < output_indices_list[ind]:
                        indices[idx+ind] = 1
                h_n1 = h[indices, :]
                indices = torch.zeros((nodes.shape[0],), dtype=torch.int64, device=self.get_device())
                for ind in range(n_graphs):
                    if vi + ind < output_indices_list[ind]:
                        indices[vi + ind] = 1
                h_n2 = h[indices, :]
                ei_score = self.sigmoid(self.edge_adder(torch.cat([h_n1, h_n2], -1)))
                # decisions = ei_score > 0.5
                # print('decisions: {}'.format(decisions))
                decisions = t_func.gumbel_softmax(ei_score,
                                                  tau=self.tau,
                                                  hard=True,
                                                  dim=-1)[:, 1].bool().unsqueeze(dim=-1)
                # print('decisions: {}'.format(decisions))
                for i in range(n_graphs):
                    # print('Working with {}-th graph:'.format(i))
                    # print('input index is: {}'.format(input_indices_list[i]))
                    # print('output index is: {}'.format(output_indices_list[i]))
                    if finished[i]:
                        continue
                    if torch.equal(new_types[i], output_embedding):
                        # if new node is output_embedding, connect it to all loose-end vertices (out_degree==0)
                        end_vertices = set([index for index in range(input_indices_list[i], output_indices_list[i])
                                            if torch.all(adj_mat[index, :] == 0)])
                        for v in end_vertices:
                            adj_mat[v, output_indices_list[i]] = 1
                            # print('start edge: {}'.format(v))
                            # print('end edge: {}'.format(output_indices_list[i]))
                        # connect all nodes without input (in_degree==0) to first node
                        end_vertices = set([index for index in range(input_indices_list[i] + 1, output_indices_list[i])
                                            if torch.all(adj_mat[:, index] == 0)])
                        for v in end_vertices:
                            adj_mat[input_indices_list[i], v] = 1
                            # print('start edge: {}'.format(v))
                            # print('end edge: {}'.format(output_indices_list[i]))
                        finished[i] = True
                        continue
                    if decisions[i, 0]:
                        adj_mat[vi+input_indices_list[i], output_indices_list[i]] = 1
                    #     print('vi: {}'.format(vi))
                    #     print('start edge: {}'.format(vi+input_indices_list[i]))
                    #     print('end edge: {}'.format(output_indices_list[i]))
                    # print('adj_mat: {}'.format(adj_mat))
                # Update graph embeddings
                # print('edges: {}'.format(convert_adj_to_edges(adj_mat)))
                # print('h.shape: {}'.format(h.shape))
                h = self.graph_conv(self.gru(nodes), convert_adj_to_edges(adj_mat))

        # Redefine batch tensor
        batch = []
        for index in range(n_graphs):
            n_nodes_in_single_graph = output_indices_list[index] - input_indices_list[index] + 1
            batch += [index for _ in range(n_nodes_in_single_graph)]
        batch = torch.tensor(batch, device=self.get_device())
        # Convert to tg data format
        generated_graph = tg.data.Data(x=nodes,
                                       edge_index=convert_adj_to_edges(adj_mat),
                                       batch=batch).to(self.get_device())
        # print('Time taken by moving data in TG format: {}'.format(time.time() - st))
        # print('Generated graph: {}'.format(generated_graph))
        return generated_graph
