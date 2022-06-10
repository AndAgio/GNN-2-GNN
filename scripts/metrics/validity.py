import time
import numpy as np
import torch
import torch_geometric as tg
# My modules
from scripts.utils import timeit, convert_edges_to_adj


class Validity():
    def __init__(self):
        pass

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
            if is_model_valid(model_nodes, model_adj):
                valid_graphs_count += 1
        score = float(valid_graphs_count) / float(batch_size)
        return score


def is_model_valid(model_nodes, model_adj):
    # Check that no input nor output operations are in the middle nodes
    non_input_output_nodes = model_nodes[1:-1, :]
    for non_input_output_node in non_input_output_nodes:
        if non_input_output_node[0] == 1 or non_input_output_node[-1] == 1:
            return False
    # Check if graph is a DAG
    triu_adj = torch.triu(model_adj, diagonal=1)
    if not torch.equal(triu_adj, model_adj):
        return False
    # Check that no node is appended
    for index in range(1, model_adj.shape[0] - 1):
        # Check if column is all zeros
        if torch.all(model_adj[:, index] == 0):
            return False
        # Check if row is all zeros
        if torch.all(model_adj[index, :] == 0):
            return False
    return True


# class Validity():
#     def __init__(self, dataset):
#         assert dataset is not None
#         self.dataset = dataset
#         # Get all valid models from the dataset
#         # st = time.time()
#         self.dataset_models = self.dataset.get_all_valid_models()
#         # print('Time taken to import all models from dataset is {:.5f} s'.format(time.time() - st))
#         # st = time.time()
#         # for i, model in enumerate(self.valid_models):
#         #     print('model[\'x\'].numpy(): {}'.format(model['x'].numpy()))
#         #     print('model[\'edge_index\'].numpy(): {}'.format(model['edge_index'].numpy()))
#         #     if i >= 0:
#         #         break
#         self.dataset_models_hashes = [hash((model['x'].numpy().tobytes(), model['edge_index'].numpy().tobytes())) for
#                                     model in self.dataset_models]
#         # print('Time taken to apply hash to all models is {:.5f} s'.format(time.time() - st))
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
#         valid_graphs_count = 0
#         valid_graphs_indices = []
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
#             # print('subgraph_nodes.type(): {}'.format(subgraph_nodes.type()))
#             # print('subgraph_edges.type(): {}'.format(subgraph_edges.type()))
#             # Check if graph is valid or not
#             # start_single = time.time()
#             if self.is_valid(subgraph_nodes, subgraph_edges):
#                 valid_graphs_count += 1
#                 valid_graphs_indices.append(batch_index)
#             # print('Single validity check took: {}'.format(time.time()-start_single))
#         # Compute score as the percentage of valid graphs
#         score = float(valid_graphs_count) / float(batch_size)
#         # print('Validity score: {}'.format(score))
#         # print('Valid graphs: {}'.format(valid_graphs_indices))
#         # print('All validity check took: {}'.format(time.time() - start_compute))
#         return score
#
#     def is_valid(self, graph_nodes, graph_edges):
#         # Convert edges into an adjacency matrix
#         # print('Graph edges: {}'.format(graph_edges))
#         first_node = torch.min(graph_edges).item()
#         # print('First node is: {}'.format(first_node))
#         # print('Graph edges: {}'.format(graph_edges))
#         graph_edges = graph_edges - first_node
#         return self.is_model_in_valid_models(graph_nodes, graph_edges)
#
#     def is_model_in_valid_models(self, model_nodes, model_edges):
#         # Check that no input nor output operations are in the middle nodes
#         input_embedding = np.argmax(model_nodes[0, :].detach().cpu().numpy())
#         output_embedding = np.argmax(model_nodes[-1, :].detach().cpu().numpy())
#         non_input_output_nodes = model_nodes.detach().cpu().numpy()[1:-1, :]
#         for non_input_output_node in non_input_output_nodes:
#             if np.argmax(non_input_output_node) == input_embedding or np.argmax(non_input_output_node) == output_embedding:
#                 return False
#         # Check if graph is a DAG
#         edges = list(zip(model_edges[0], model_edges[1]))
#         for edge in edges:
#             if edge[0] >= edge[1]:
#                 return False
#         return True
#
#     def is_model_in_dataset_models(self, model_nodes, model_edges):
#         # st = time.time()
#         # print('model_nodes.numpy(): {}'.format(model_nodes.numpy()))
#         # print('model_edges.numpy(): {}'.format(model_edges.numpy()))
#         reply = hash((model_nodes.detach().cpu().numpy().tobytes(),
#                       model_edges.detach().cpu().numpy().tobytes())) in self.dataset_models_hashes
#         # print('Time taken to compute single model novelty is {:.5f} s'.format(time.time() - st))
#         return reply
#
#     def get_accuracies(self, data):
#         if isinstance(data, tg.data.Data):
#             x, edge_index, batch = data.x.float(), data.edge_index, data.batch
#         else:
#             x, edge_index, batch = data
#         # Get batch size from batch of data
#         batch_size = torch.max(batch).item() + 1
#         # Set to zero the number of valid generated graphs
#         graphs_accuracies = []
#         # Iterate over the number of generated graphs
#         for batch_index in range(batch_size):
#             subgraph_nodes_indices = (batch == batch_index).nonzero(as_tuple=True)[0]
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
#             # Get accuracy of single graph
#             acc = self.get_accuracy(subgraph_nodes, subgraph_edges)
#             graphs_accuracies.append(acc)
#         return graphs_accuracies
#
#     def get_accuracy(self, graph_nodes, graph_edges):
#         st = time.time()
#         # Convert edges into an adjacency matrix
#         first_node = torch.min(graph_edges).item()
#         graph_edges = graph_edges - first_node
#         if not self.is_model_in_dataset_models(graph_nodes, graph_edges):
#             return 0
#         else:
#             # Define hash for queried model
#             queried_model_hash = hash((graph_nodes.detach().cpu().numpy().tobytes(),
#                                        graph_edges.detach().cpu().numpy().tobytes()))
#             # Find index corresponding to the queried model in the list of valid models
#             queried_model_index = self.dataset_models_hashes.index(queried_model_hash)
#             # Pick the queried model from the list of valid models
#             queried_model = self.dataset_models[queried_model_index]
#             # print('Time taken to get accuracy of queried model from hashes list of valid models is: {:.5f} s'.format(time.time()-st))
#             # Return the queried model accuracy over the test set
#             if queried_model['test_accuracy'] >= 90:
#                 value = 1
#             else:
#                 value = 0
#             # value = queried_model['test_accuracy']/float(100)
#             return value
#
#     # def get_accuracy(self, graph_nodes, graph_edges):
#     #     # Convert edges into an adjacency matrix
#     #     first_node = torch.min(graph_edges).item()
#     #     graph_edges = graph_edges - first_node
#     #     n_nodes = torch.max(graph_edges).item() + 1
#     #     graph_edges = graph_edges.cpu().numpy()
#     #     edges_weights = [1 for _ in range(len(graph_edges[0, :]))]
#     #     matrix = sparse.coo_matrix((edges_weights,
#     #                                 (graph_edges[0, :],
#     #                                  graph_edges[1, :])),
#     #                                shape=(n_nodes, n_nodes)).todense()
#     #     graph_nodes = graph_nodes.detach().cpu().numpy()
#     #     class_name_dict = Utils().get_class_name_dict()
#     #     nodes_labels = [class_name_dict[np.argmax(graph_nodes[index, :])] for index in range(graph_nodes.shape[0])]
#     #     # Convert matrix to numpy
#     #     matrix = np.array(matrix)
#     #     infos = self.data.get_infos(matrix, nodes_labels)
#     #     accuracy = infos['test_accuracy']
#     #     return accuracy
