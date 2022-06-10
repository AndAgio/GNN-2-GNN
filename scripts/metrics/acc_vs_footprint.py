import time
import torch
import torch_geometric as tg
# My modules
from scripts.utils import timeit, convert_edges_to_adj, hash_model


class AccVsFootprint():
    def __init__(self, dataset_metrics):
        self.dataset_metrics = dataset_metrics

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
        # Define empty list of acc_vs_footprint for generated graphs
        acc_vs_footprint_list = []
        for batch_index in range(batch_size):
            start_node_index = input_indices_list[batch_index]
            end_node_index = output_indices_list[batch_index]
            model_nodes = x[start_node_index:end_node_index+1, :]
            model_adj = adj_mat[start_node_index:end_node_index+1, start_node_index:end_node_index+1]
            # Compute acc_vs_footprint for single model
            accuracy, footprint = self.get_model_acc_and_params(model_nodes, model_adj)
            acc_vs_footprint_list.append([accuracy, footprint])
        return acc_vs_footprint_list

    def get_model_acc_and_params(self, model_nodes, model_adj):
        # Compute hash for single model
        model_hash = hash_model(model_nodes, model_adj)
        # Try to get model metrics from the dictionary of metrics of the dataset
        try:
            model_metrics = self.dataset_metrics[model_hash]
        except KeyError:
            return None, None
        accuracy = model_metrics['test_accuracy']
        footprint = model_metrics['footprint']
        return accuracy, footprint

    def get_all_acc_vs_footprint(self):
        # Define empty list of acc_vs_footprint for generated graphs
        acc_vs_footprint_list = []
        for model_hash, model_metrics in self.dataset_metrics.items():
            accuracy = model_metrics['test_accuracy']
            footprint = model_metrics['footprint']
            acc_vs_footprint_list.append([accuracy, footprint])
        return acc_vs_footprint_list


# class AccVsFootprint():
#     def __init__(self, dataset, dataset_type='nas101'):
#         assert dataset is not None
#         self.dataset = dataset
#         assert dataset_type in ['nas101', 'nats']
#         self.dataset_type = dataset_type
#         # Get all models from the dataset
#         self.all_models = self.dataset.get_all_valid_models()
#         # Apply hash function to each model dataset
#         self.all_models_hashes = [hash((model['x'].numpy().tobytes(), model['edge_index'].numpy().tobytes())) for
#                                   model in self.all_models]
#
#     # @timeit
#     def compute(self, data):
#         if isinstance(data, tg.data.Data):
#             x, edge_index, batch = data.x.float(), data.edge_index, data.batch
#         else:
#             x, edge_index, batch = data
#         # Get batch size from batch of data
#         batch_size = torch.max(batch).item() + 1
#         # Define empty list of acc_vs_footprint for generated graphs
#         acc_vs_footprint_list = []
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
#             # start_single = time.time()
#             accuracy, footprint = self.get_acc_and_params(subgraph_nodes, subgraph_edges)
#             acc_vs_footprint_list.append([accuracy, footprint])
#             # print('Single in-top-k check took: {}'.format(time.time() - start_single))
#         return acc_vs_footprint_list
#
#     def get_acc_and_params(self, graph_nodes, graph_edges):
#         # Convert edges into an adjacency matrix
#         first_node = torch.min(graph_edges).item()
#         graph_edges = graph_edges - first_node
#         return self.get_model_acc_and_params(graph_nodes, graph_edges)
#
#     def get_model_acc_and_params(self, model_nodes, model_edges):
#         # print('TOP K: {}'.format(self.top_k))
#         # # Iterate over all top-n models and check if
#         # st = time.time()
#         # reply = False
#         # for top_model in self.top_k_models:
#         #     if torch.equal(top_model['x'], model_nodes) and torch.equal(top_model['edge_index'], model_edges):
#         #         reply = True
#         #         break
#         # print('Time taken with for is {:.5f} s and reply is: {}'.format(time.time() - st, reply))
#         # Check using hashes
#         # st = time.time()
#         model_hash = hash((model_nodes.detach().cpu().numpy().tobytes(),
#                            model_edges.detach().cpu().numpy().tobytes()))
#         try:
#             index = self.all_models_hashes.index(model_hash)
#         except ValueError:
#             return None, None
#         model_in_dataset = self.all_models[index]
#         accuracy = model_in_dataset['test_accuracy']
#         if self.dataset_type == 'nas101':
#             footprint = model_in_dataset['trainable_parameters']
#         elif self.dataset_type == 'nats':
#             params = model_in_dataset['params']
#             flops = model_in_dataset['flops']
#             footprint = [params, flops]
#         # print('Time taken with in is {:.5f} s and reply is: {}'.format(time.time() - st, reply))
#         # print('Hash of first model in top-k models is: {}'.format(self.top_k_models_hashes[0]))
#         # print('Reconstructed hash of first model in top-k models is: {}'.format(hash((self.top_k_models[0][
#         #                                                                                   'x'].detach().numpy().tobytes(),
#         #                                                                               self.top_k_models[0][
#         #                                                                                   'edge_index'].detach().numpy().tobytes()))))
#         # print('Frst model in top-k models is: {}'.format(self.top_k_models[0]['x'].numpy().dtype))
#         # print('Frst model in top-k models is: {}'.format(self.top_k_models[0]['edge_index'].numpy().dtype))
#         # nodes = np.array([[1., 0., 0., 0., 0.],
#         #                   [0., 1., 0., 0., 0.],
#         #                   [0., 0., 0., 1., 0.],
#         #                   [0., 0., 0., 1., 0.],
#         #                   [0., 0., 0., 1., 0.],
#         #                   [0., 0., 0., 0., 1.]], dtype=np.float32)
#         # edges = np.array([[0, 1, 1, 1, 1, 2, 3, 3, 4],
#         #                   [1, 2, 3, 4, 5, 3, 4, 5, 5]], dtype=np.int64)
#         # print('nodes type is: {}'.format(nodes.dtype))
#         # print('edges type is: {}'.format(edges.dtype))
#         # print('Reconstructed hash of first model in top-k models is: {}'.format(hash((nodes.tobytes(), edges.tobytes()))))
#         return accuracy, footprint
#
#     def get_all_acc_vs_footprint(self):
#         # Define empty list of acc_vs_footprint for generated graphs
#         acc_vs_footprint_list = []
#         for model_in_dataset in self.all_models:
#             accuracy = model_in_dataset['test_accuracy']
#             if self.dataset_type == 'nas101':
#                 footprint = model_in_dataset['trainable_parameters']
#             elif self.dataset_type == 'nats':
#                 params = model_in_dataset['params']
#                 flops = model_in_dataset['flops']
#                 footprint = [params, flops]
#             acc_vs_footprint_list.append([accuracy, footprint])
#         return acc_vs_footprint_list
