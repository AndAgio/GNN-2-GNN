import torch
import torch.nn.functional as t_func
import networkx as nx
from scripts.utils import Utils
from .decorators import timeit


class GraphsTorch():
    def __init__(self, sample_pre=False, tau=None, device='cpu', dataset='nas101', sub_skip=False):
        self.sample_pre = sample_pre
        self.tau = tau
        self.device = device
        self.dataset = dataset
        self.sub_skip = sub_skip

        self.utils = Utils()
        self.epsilon = 0.001

    def convert_tg_to_nx(self, data):
        # Method for converting pytorch geometric graph into networkx graph
        graph = nx.DiGraph()
        for index, node in enumerate(data.x):
            graph.add_nodes_from([(index, {'ops': torch.argmax(node, dim=-1).item()})])
        for index in range(data.edge_index.shape[1]):
            src = data.edge_index[0, index]
            dst = data.edge_index[1, index]
            graph.add_edge(src.item(), dst.item())
        return graph

    def refine_batched_graphs(self, nodes, edges, edges_scores, batch, n_predicted_ops=None):
        # Convert edges into adjacency matrix
        batch_size = torch.max(batch).item() + 1
        n_nodes = int(nodes.shape[0])
        n_nodes_in_single_graph = int(nodes.shape[0]/batch_size)
        adj_mat = torch.zeros([n_nodes, n_nodes], device=self.device, dtype=torch.int64)
        adj_mat_scores = torch.zeros([n_nodes, n_nodes], device=self.device, dtype=torch.int64)
        edges_list = list(zip(edges[0], edges[1]))
        for index, edge in enumerate(edges_list):
            adj_mat[edge[0], edge[1]] = 1
            adj_mat_scores[edge[0], edge[1]] = edges_scores[index]
        # Remove edges which didn't survive from the graph
        adj_mat = self.remove_unused_edges(adj_mat, adj_mat_scores)
        # Make sure graph is dag using torch triu
        adj_mat = torch.triu(adj_mat, diagonal=1)
        # Refine operations embedding if input and outputs are not predicted
        operations = Utils().get_class_name_dict(bench=self.dataset, sub_skip=self.sub_skip)
        n_dataset_ops = len(list(operations.keys()))
        if n_predicted_ops is not None and n_predicted_ops != n_dataset_ops:
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
        input_indices_list = [i * n_nodes_in_single_graph for i in range(batch_size)]
        output_indices_list = [(i+1) * n_nodes_in_single_graph - 1 for i in range(batch_size)]
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
                    adj_mat = torch.cat([adj_mat[:, :index], adj_mat[:, index+1:]], dim=1)
                    adj_mat = torch.cat([adj_mat[:index, :], adj_mat[index+1:, :]], dim=0)
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

    def remove_unused_edges_from_batched_graphs(self, nodes, edges, edges_scores, batch, n_predicted_ops=None):
        # Convert edges into adjacency matrix
        batch_size = torch.max(batch).item() + 1
        n_nodes = int(nodes.shape[0])
        n_nodes_in_single_graph = int(nodes.shape[0]/batch_size)
        adj_mat = torch.zeros([n_nodes, n_nodes], device=self.device, dtype=torch.int64)
        adj_mat_scores = torch.zeros([n_nodes, n_nodes], device=self.device, dtype=torch.int64)
        edges_list = list(zip(edges[0], edges[1]))
        for index, edge in enumerate(edges_list):
            adj_mat[edge[0], edge[1]] = 1
            adj_mat_scores[edge[0], edge[1]] = edges_scores[index]
        # Remove edges which didn't survive from the graph
        adj_mat = self.remove_unused_edges(adj_mat, adj_mat_scores)
        # Make sure graph is dag using torch triu
        adj_mat = torch.triu(adj_mat, diagonal=1)
        # Refine operations embedding if input and outputs are not predicted
        operations = Utils().get_class_name_dict(bench=self.dataset, sub_skip=self.sub_skip)
        n_dataset_ops = len(list(operations.keys()))
        if n_predicted_ops is not None and n_predicted_ops != n_dataset_ops:
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
        input_indices_list = [i * n_nodes_in_single_graph for i in range(batch_size)]
        output_indices_list = [(i+1) * n_nodes_in_single_graph - 1 for i in range(batch_size)]
        # Set first and last node of each batch to have input and output operation
        nodes[input_indices_list, :] = input_embedding
        nodes[output_indices_list, :] = output_embedding
        # If a single graph has remained with no edges, add a single edge between input and output
        for index in range(batch_size):
            if input_indices_list[index] == output_indices_list[index] - 1:
                adj_mat[input_indices_list[index], output_indices_list[index]] = 1
        # Recompute edges from adjacency matrix
        edges_indices = (adj_mat == 1).nonzero(as_tuple=True)
        edges_indices = list(zip(edges_indices[0].cpu().numpy(), edges_indices[1].cpu().numpy()))
        edges_indices = torch.tensor(edges_indices, device=self.device)
        edges_indices = edges_indices.permute(-1, 0)
        return nodes, edges_indices, batch

    # @timeit
    def remove_unused_edges(self, adj, adj_scores):
        # Method to remove unused edges
        # Sample edges using gumbel softmax
        if self.sample_pre:
            adj_scores = adj_scores
        else:
            adj_scores = t_func.gumbel_softmax(adj_scores,
                                                 tau=self.tau,
                                                 hard=True,
                                                 dim=-1)
            adj_scores = torch.argmax(adj_scores, dim=-1)
        return adj_scores

