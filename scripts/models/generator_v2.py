import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as t_func
import torch_geometric as tg
# Import my modules
from scripts.layers import Edger, MLP
from scripts.utils import Utils, timeit
from scripts.utils import GraphsTorch as Graphs


class GeneratorNetV2(nn.Module):
    def __init__(self, z_dim=16, n_nodes=7,
                 hidden_dims=[128, 256, 512],
                 hidden_dim_ops=256,
                 n_ops=5, tau=1,
                 refine=True, dataset='nas101', sub_skip=False):
        super(GeneratorNetV2, self).__init__()
        self.hidden_dims = hidden_dims
        self.hidden_dim_ops = hidden_dim_ops
        self.n_nodes = n_nodes
        self.edges_ops = 2
        self.n_ops = n_ops
        self.tau = tau
        self.refine = refine
        self.dataset = dataset
        self.sub_skip = sub_skip
        # Define MLP for generating edges
        self.activation_f = torch.nn.Tanh()
        self.mlp = MLP(z_dim, hidden_dims, self.activation_f)
        # Define layer for generating edges
        self.edges_layer = nn.Linear(hidden_dims[-1], 2 * n_nodes * n_nodes)
        # Define layer for generating nodes
        self.nodes_layer = nn.Linear(hidden_dims[-1], n_nodes * self.hidden_dim_ops)
        self.dropoout = nn.Dropout(p=0.2)
        # Define layer for mapping embedding to operation
        self.mapper = tg.nn.GCNConv(self.hidden_dim_ops, self.n_ops)
        # # Define one hot encoding of input and output operation
        # operations = Utils().get_class_name_dict(bench=self.dataset, sub_skip=self.sub_skip)
        # n_dataset_ops = len(list(operations.keys()))
        # self.input_embedding = torch.zeros((n_dataset_ops,))
        # self.input_embedding[0] = 1
        # self.output_embedding = torch.zeros((n_dataset_ops,))
        # self.output_embedding[-1] = 1

    # @timeit
    def forward(self, z):
        # print('z shape: {}'.format(z.shape))
        # Propagate random vector to a learned hidden state
        common_hidden_state = self.mlp(z)
        # print('common_hidden_state shape: {}'.format(common_hidden_state.shape))
        # Generate edges from the common hidden state
        adj_scores = self.edges_layer(common_hidden_state).view(-1, self.edges_ops, self.n_nodes, self.n_nodes)
        adj_scores = (adj_scores + adj_scores.permute(0, 1, 3, 2)) / 2
        adj_scores = self.dropoout(adj_scores.permute(0, 2, 3, 1))
        # print('adj_scores shape: {}'.format(adj_scores.shape))
        # Generate nodes from the common hidden state
        nodes_hidden = self.nodes_layer(common_hidden_state)
        nodes_hidden = self.dropoout(nodes_hidden.view(-1, self.n_nodes, self.hidden_dim_ops))
        # print('nodes_hidden shape: {}'.format(nodes_hidden.shape))
        # Refine edges
        nodes_scores, edges_indices, batch = self.sample_and_get_edges(adj_scores, nodes_hidden)
        # print('nodes_scores: {}'.format(nodes_scores))
        # print('edges_indices: {}'.format(edges_indices))
        # Apply last convolutional layer to get the operation for each node
        op_embedding = self.mapper(nodes_scores, edges_indices)
        # Get operation embeddings using gumbel softmax
        op_indices = t_func.gumbel_softmax(op_embedding, tau=self.tau, hard=True, dim=-1)
        # Refine graphs if requested
        if self.refine:
            op_indices, edges_indices, batch = self.refine_batched_graphs(nodes=op_indices,
                                                                          edges=edges_indices,
                                                                          batch=batch)
        # Convert to tg data format
        generated_graph = tg.data.Data(x=op_indices,
                                       edge_index=edges_indices,
                                       batch=batch).to(
            torch.device('cuda' if next(self.parameters()).is_cuda else 'cpu'))
        # print('Time taken by moving data in TG format: {}'.format(time.time() - st))
        # print('Generated graph: {}'.format(generated_graph))
        return generated_graph

    def sample_and_get_edges(self, adj_scores, nodes_scores):
        # Get device where model is stored
        self.device = torch.device('cuda' if next(self.parameters()).is_cuda else 'cpu')
        # Convert edges into adjacency matrix
        n_nodes_in_single_graph = int(self.n_nodes)
        batch_size = int(nodes_scores.shape[0])
        n_nodes_in_batch = int(nodes_scores.shape[1])
        # print('n_nodes_in_single_graph: {}'.format(n_nodes_in_single_graph))
        # print('n_nodes_in_batch: {}'.format(n_nodes_in_batch))
        # print('batch_size: {}'.format(batch_size))
        # Pass edges through gumbel softmax to get the sampled adjacency matrix
        adj_mat = self.sample_edges(adj_scores)
        # For each sample in the batch refine the adjacency matrix and convert it into edges list
        refined_nodes_scores, refined_edges = self.remove_unused_edges(nodes_scores[0, :, :], adj_mat[0, :, :])
        last_input_node_index = refined_nodes_scores.shape[0]
        refined_batch = torch.zeros([refined_nodes_scores.shape[0], ], device=self.device, dtype=torch.int64)
        for batch_index in range(1, batch_size):
            tmp_nodes_scores, tmp_edges = self.remove_unused_edges(nodes_scores[batch_index, :, :],
                                                                   adj_mat[batch_index, :, :])
            refined_nodes_scores = torch.cat([refined_nodes_scores, tmp_nodes_scores], dim=0)
            refined_edges = torch.cat([refined_edges, tmp_edges + last_input_node_index], dim=1)
            last_input_node_index = refined_nodes_scores.shape[0]
            tmp_batch = batch_index * torch.ones([tmp_nodes_scores.shape[0], ], device=self.device, dtype=torch.int64)
            refined_batch = torch.cat([refined_batch, tmp_batch], dim=0)
        # print('refined_nodes_scores shape: {}'.format(refined_nodes_scores.shape))
        # print('refined_edges shape: {}'.format(refined_edges.shape))
        # print('refined_batch shape: {}'.format(refined_batch.shape))
        return refined_nodes_scores, refined_edges, refined_batch

    def sample_edges(self, adj_scores):
        # Method to remove unused edges
        # Sample edges using gumbel softmax
        adj_scores = t_func.gumbel_softmax(adj_scores,
                                           tau=self.tau,
                                           hard=True,
                                           dim=-1)
        adj_mat = adj_scores[:, :, :, -1]
        return adj_mat

    def remove_unused_edges(self, nodes_scores, adj_mat):
        # print('adj_mat shape: {}'.format(adj_mat.shape))
        # Make sure graph is dag using torch triu
        adj_mat = torch.triu(adj_mat, diagonal=1)
        # Remove unconnected nodes from the adjacency matrix and the list of nodes
        index = 1
        while index < adj_mat.shape[1] - 1:
            # Check if column is all zeros
            if torch.all(adj_mat[:, index] == 0) and torch.all(adj_mat[index, :] == 0):
                # Remove row and column from adjacency matrix
                adj_mat = torch.cat([adj_mat[:, :index], adj_mat[:, index + 1:]], dim=1)
                adj_mat = torch.cat([adj_mat[:index, :], adj_mat[index + 1:, :]], dim=0)
                # Remove node from matrix of nodes embeddings
                nodes_scores = torch.cat([nodes_scores[:index, :], nodes_scores[index + 1:, :]], dim=0)
            else:
                # Update index only if column and row aren't removed
                index += 1
        # print('adj_mat: {}'.format(adj_mat))
        # If a single graph has remained with no edges, add a single edge between input and output
        if adj_mat.shape[0] == 2:
            adj_mat[0, 1] = 1
        # Recompute edges from adjacency matrix
        edges_indices = (adj_mat == 1).nonzero(as_tuple=True)
        edges_indices = list(zip(edges_indices[0].cpu().numpy(), edges_indices[1].cpu().numpy()))
        edges_indices = torch.tensor(edges_indices, device=self.device)
        # print('edges_indices: {}'.format(edges_indices))
        edges_indices = edges_indices.permute(-1, 0)
        return nodes_scores, edges_indices

    def refine_batched_graphs(self, nodes, edges, batch):
        # Convert edges into adjacency matrix
        batch_size = torch.max(batch).item() + 1
        n_nodes = int(nodes.shape[0])
        adj_mat = torch.zeros([n_nodes, n_nodes], device=self.device, dtype=torch.int64)
        adj_mat_scores = torch.zeros([n_nodes, n_nodes], device=self.device, dtype=torch.int64)
        edges_list = list(zip(edges[0], edges[1]))
        for index, edge in enumerate(edges_list):
            adj_mat[edge[0], edge[1]] = 1
        # Refine operations embedding if input and outputs are not predicted
        operations = Utils().get_class_name_dict(bench=self.dataset, sub_skip=self.sub_skip)
        n_dataset_ops = len(list(operations.keys()))
        if self.n_ops != n_dataset_ops:
            # print('new_nodes before appending: {}'.format(nodes))
            nodes = torch.cat((nodes, torch.zeros((nodes.shape[0], 1), device=self.device)), dim=1)
            nodes = torch.cat((torch.zeros((nodes.shape[0], 1), device=self.device), nodes), dim=1)
            # print('new_nodes after appending: {}'.format(nodes))
        # Set input and output nodes for each graph in the batch
        input_embedding = torch.zeros((n_dataset_ops,), device=self.device)
        input_embedding[0] = 1
        output_embedding = torch.zeros((n_dataset_ops,), device=self.device)
        output_embedding[-1] = 1
        # Define indices where input and output nodes are
        input_indices_list = [(batch == i).nonzero(as_tuple=True)[0][0] for i in range(batch_size)]
        output_indices_list = [(batch == i).nonzero(as_tuple=True)[0][-1] for i in range(batch_size)]
        # Set first and last node of each batch to have input and output operation
        nodes[input_indices_list, :] = input_embedding
        nodes[output_indices_list, :] = output_embedding
        # Remove nodes not having inputs
        index = 0
        while index < adj_mat.shape[1]:
            if index not in input_indices_list and index not in output_indices_list:
                # Check if column is all zeros
                if torch.all(adj_mat[:, index] == 0):
                    # Remove row and column from adjacency matrix
                    adj_mat = torch.cat([adj_mat[:, :index], adj_mat[:, index + 1:]], dim=1)
                    adj_mat = torch.cat([adj_mat[:index, :], adj_mat[index + 1:, :]], dim=0)
                    # Remove node from matrix of nodes embeddings
                    nodes = torch.cat([nodes[:index, :], nodes[index + 1:, :]], dim=0)
                    # Update input and output indices
                    input_indices_list = [i - 1 if i > index else i for i in input_indices_list]
                    output_indices_list = [i - 1 if i > index else i for i in output_indices_list]
                else:
                    # Update index only if column and row aren't removed
                    index += 1
            else:
                # Update index only if column and row aren't removed
                index += 1
        # Remove nodes not having outputs
        index = adj_mat.shape[0] - 1
        while index > 0:
            if index not in input_indices_list and index not in output_indices_list:
                # Check if row is all zeros
                if torch.all(adj_mat[index, :] == 0):
                    # Remove row and column from adjacency matrix
                    adj_mat = torch.cat([adj_mat[:, :index], adj_mat[:, index + 1:]], dim=1)
                    adj_mat = torch.cat([adj_mat[:index, :], adj_mat[index + 1:, :]], dim=0)
                    # Remove node from matrix of nodes embeddings
                    nodes = torch.cat([nodes[:index, :], nodes[index + 1:, :]], dim=0)
                    # Update input and output indices
                    input_indices_list = [i - 1 if i > index else i for i in input_indices_list]
                    output_indices_list = [i - 1 if i > index else i for i in output_indices_list]
                else:
                    # Update index only if column and row aren't removed
                    index -= 1
            else:
                # Update index only if column and row aren't removed
                index -= 1
        # If a single graph has remained with no edges, add a single edge between input and output
        for index in range(batch_size):
            if input_indices_list[index] == output_indices_list[index] - 1:
                adj_mat[input_indices_list[index], output_indices_list[index]] = 1
        # Recompute edges from adjacency matrix
        edges_indices = (adj_mat == 1).nonzero(as_tuple=True)
        edges_indices = list(zip(edges_indices[0].cpu().numpy(), edges_indices[1].cpu().numpy()))
        edges_indices = torch.tensor(edges_indices, device=self.device)
        edges_indices = edges_indices.permute(-1, 0)
        # Redefine batch tensor
        batch = []
        for index in range(batch_size):
            n_nodes_in_single_graph = output_indices_list[index] - input_indices_list[index] + 1
            batch += [index for _ in range(n_nodes_in_single_graph)]
        batch = torch.tensor(batch, device=self.device)
        return nodes, edges_indices, batch

    # def set_in_out_graph(self, nodes_ops):
    #     # Get number of operations predicted
    #     n_predicted_ops = nodes_ops.shape[-1]
    #     # Refine operations embedding if input and outputs are not predicted
    #     operations = Utils().get_class_name_dict(bench=self.dataset, sub_skip=self.sub_skip)
    #     n_dataset_ops = len(list(operations.keys()))
    #     if n_predicted_ops is not None and n_predicted_ops != n_dataset_ops:
    #         # print('new_nodes before appending: {}'.format(nodes))
    #         nodes = torch.cat((nodes_ops, torch.zeros((nodes_ops.shape[0], 1), device=self.device)), dim=1)
    #         nodes = torch.cat((torch.zeros((nodes.shape[0], 1), device=self.device), nodes), dim=1)
    #         # print('new_nodes after appending: {}'.format(nodes))
    #     # Set input and output nodes for each graph in the batch
    #     input_embedding = torch.zeros((n_dataset_ops,), device=self.device)
    #     input_embedding[0] = 1
    #     output_embedding = torch.zeros((n_dataset_ops,), device=self.device)
    #     output_embedding[-1] = 1
    #     # Define indices where input and output nodes are
    #     input_indices_list = [i * n_nodes_in_single_graph for i in range(batch_size)]
    #     output_indices_list = [(i + 1) * n_nodes_in_single_graph - 1 for i in range(batch_size)]
    #     # Set first and last node of each batch to have input and output operation
    #     nodes[input_indices_list, :] = input_embedding
    #     nodes[output_indices_list, :] = output_embedding
