import numpy as np
import torch
import torch_geometric as tg
# My modules
from scripts.utils import timeit, convert_edges_to_adj, hash_model
from .validity import is_model_valid


class Novelty():
    def __init__(self, train_models_hashes):
        self.train_models_hashes = train_models_hashes

    # @timeit
    def compute(self, data):
        if isinstance(data, tg.data.Data):
            x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        else:
            x, edge_index, batch = data
        # Convert edges to adj_mat for faster computation
        adj_mat = convert_edges_to_adj(x, edge_index)
        # Get splits for single graphs
        batch_size = torch.max(data.batch).item() + 1
        input_indices_list = [(batch == i).nonzero(as_tuple=True)[0][0] for i in range(batch_size)]
        output_indices_list = [(batch == i).nonzero(as_tuple=True)[0][-1] for i in range(batch_size)]
        # Iterate over batch and compute validity of single graph
        valid_graphs_count = 0
        for batch_index in range(batch_size):
            start_node_index = input_indices_list[batch_index]
            end_node_index = output_indices_list[batch_index]
            model_nodes = x[start_node_index:end_node_index+1, :]
            model_adj = adj_mat[start_node_index:end_node_index+1, start_node_index:end_node_index+1]
            # Compute validity for single model
            if self.is_model_novel(model_nodes, model_adj):
                valid_graphs_count += 1
        score = float(valid_graphs_count) / float(batch_size)
        return score

    def is_model_novel(self, model_nodes, model_adj):
        reply = is_model_valid(model_nodes, model_adj) \
                and not self.is_model_in_train_models(model_nodes, model_adj)
        # print('reply: {}'.format(reply))
        return reply

    def is_model_in_train_models(self, model_nodes, model_adj):
        # st = time.time()
        in_train = hash_model(model_nodes, model_adj) in self.train_models_hashes
        # print('Time taken to compute single model novelty is {:.5f} s'.format(time.time() - st))
        # print('in_train: {}'.format(in_train))
        return in_train


# class Novelty():
#     def __init__(self, dataset):
#         assert dataset is not None
#         self.dataset = dataset
#         # Get train models from the dataset
#         self.train_models = self.dataset.get_train_models()
#         # Apply hash function to each model in novel models list
#         self.train_models_hashes = [hash((model['x'].numpy().tobytes(), model['edge_index'].numpy().tobytes())) for
#                                     model in self.train_models]
#         # # Get top-k models from the dataset
#         # self.novel_models = self.dataset.get_test_models()
#         # # Apply hash function to each model in novel models list
#         # self.novel_models_hashes = [hash((model['x'].numpy().tobytes(), model['edge_index'].numpy().tobytes())) for
#         #                             model in self.novel_models]
#
#     # @timeit
#     def compute(self, data):
#         if isinstance(data, tg.data.Data):
#             x, edge_index, batch = data.x.float(), data.edge_index, data.batch
#         else:
#             x, edge_index, batch = data
#         # Get batch size from batch of data
#         batch_size = torch.max(batch).item() + 1
#         # Set to zero the number of valid generated graphs
#         novel_graphs_count = 0
#         novel_graphs_indices = []
#         # Iterate over the number of generated graphs
#         for batch_index in range(batch_size):
#             subgraph_nodes_indices = (batch == batch_index).nonzero(as_tuple=True)[0]
#             # print('subgraph_nodes_indices: {}'.format(subgraph_nodes_indices))
#             # Get the subgraph containing the NN graph at ith batch
#             subgraph_nodes = x[subgraph_nodes_indices]
#             graph_edges_src = edge_index[0, :]
#             graph_edges_dst = edge_index[1, :]
#             src_indices_to_keep = [index for index, value in enumerate(graph_edges_src) if
#                                    value in subgraph_nodes_indices]
#             dst_indices_to_keep = [index for index, value in enumerate(graph_edges_dst) if
#                                    value in subgraph_nodes_indices]
#             edges_indices_to_keep = src_indices_to_keep + list(set(dst_indices_to_keep) - set(src_indices_to_keep))
#             subgraph_edges = edge_index[:, edges_indices_to_keep]
#             # print('subgraph_nodes: {}'.format(subgraph_nodes))
#             # print('subgraph_edges: {}'.format(subgraph_edges))
#             # Check if graph is valid or not
#             if self.is_novel(subgraph_nodes, subgraph_edges):
#                 novel_graphs_count += 1
#                 novel_graphs_indices.append(batch_index)
#         # Compute score as the percentage of valid graphs
#         score = float(novel_graphs_count) / float(batch_size)
#         # print('Novelty score: {}'.format(score))
#         # print('Novelty graphs: {}'.format(novel_graphs_indices))
#         return score
#
#     def is_novel(self, graph_nodes, graph_edges):
#         # Convert edges into an adjacency matrix
#         first_node = torch.min(graph_edges).item()
#         graph_edges = graph_edges - first_node
#         # print('graph_edges: {}'.format(graph_edges))
#         # print('graph is_valid: {}'.format(is_model_valid(graph_nodes, graph_edges)))
#         reply = is_model_valid(graph_nodes, graph_edges) and not self.is_model_in_train_models(graph_nodes, graph_edges)
#         # print('reply: {}'.format(reply))
#         return reply
#
#     def is_model_in_train_models(self, model_nodes, model_edges):
#         # st = time.time()
#         in_train = hash((model_nodes.detach().cpu().numpy().tobytes(),
#                          model_edges.detach().cpu().numpy().tobytes())) in self.train_models_hashes
#         # print('Time taken to compute single model novelty is {:.5f} s'.format(time.time() - st))
#         # print('in_train: {}'.format(in_train))
#         return in_train
#
#     # def is_novel(self, graph_nodes, graph_edges):
#     #     # Convert edges into an adjacency matrix
#     #     first_node = torch.min(graph_edges).item()
#     #     graph_edges = graph_edges - first_node
#     #     return self.is_model_in_test_models(graph_nodes, graph_edges)
#
#     # def is_model_in_test_models(self, model_nodes, model_edges):
#     #     # st = time.time()
#     #     reply = hash((model_nodes.detach().cpu().numpy().tobytes(),
#     #                   model_edges.detach().cpu().numpy().tobytes())) in self.novel_models_hashes
#     #     # print('Time taken to compute single model novelty is {:.5f} s'.format(time.time() - st))
#     #     return reply
#
#
# def is_model_valid(model_nodes, model_edges):
#     # Check that no input nor output operations are in the middle nodes
#     input_embedding = np.argmax(model_nodes[0, :].detach().cpu().numpy())
#     output_embedding = np.argmax(model_nodes[-1, :].detach().cpu().numpy())
#     non_input_output_nodes = model_nodes.detach().cpu().numpy()[1:-1, :]
#     for non_input_output_node in non_input_output_nodes:
#         if np.argmax(non_input_output_node) == input_embedding or np.argmax(non_input_output_node) == output_embedding:
#             return False
#     # Check if graph is a DAG
#     edges = list(zip(model_edges[0], model_edges[1]))
#     for edge in edges:
#         if edge[0] >= edge[1]:
#             return False
#     edge_starters = model_edges[0].unique()
#     n_nodes = model_nodes.shape[0]
#     for node_index in range(0, n_nodes-1):
#         if node_index not in edge_starters:
#             return False
#     edge_enders = model_edges[1].unique()
#     for node_index in range(1, n_nodes):
#         if node_index not in edge_enders:
#             return False
#     return True
