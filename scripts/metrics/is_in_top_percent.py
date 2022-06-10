import time
import torch
import torch_geometric as tg
# My modules
from scripts.utils import timeit, convert_edges_to_adj, hash_model


class InTopKPercent():
    def __init__(self, dataset_metrics, top_k=10):
        assert 0 < top_k <= 100
        self.dataset_metrics = dataset_metrics
        # print('self.dataset_metrics: {}'.format(self.dataset_metrics))
        self.top_k = top_k

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
            if self.is_model_in_top_n_percent(model_nodes, model_adj):
                valid_graphs_count += 1
        score = float(valid_graphs_count) / float(batch_size)
        return score

    def is_model_in_top_n_percent(self, model_nodes, model_adj):
        # Compute model hash
        model_hash = hash_model(model_nodes, model_adj)
        # Try to get model top level from the dictionary of metrics of the dataset
        try:
            top_n = self.dataset_metrics[model_hash]['top']
            reply = top_n <= self.top_k
        except KeyError:
            reply = False
        return reply

# class InTopKPercent():
#     def __init__(self, dataset, top_k=10):
#         assert dataset is not None
#         assert 0 < top_k <= 100
#         self.dataset = dataset
#         self.top_k = top_k
#         # Get top-k models from the dataset
#         self.top_k_models = self.dataset.get_top_n_models(top_n=top_k)
#         # Apply hash function to each model in top-k models list
#         self.top_k_models_hashes = [hash((model['x'].numpy().tobytes(), model['edge_index'].numpy().tobytes())) for
#                                     model in self.top_k_models]
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
#         top_graphs_count = 0
#         top_graphs_indices = []
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
#             if self.is_in_top_percent(subgraph_nodes, subgraph_edges):
#                 top_graphs_count += 1
#                 top_graphs_indices.append(batch_index)
#             # print('Single in-top-k check took: {}'.format(time.time() - start_single))
#         # Compute score as the percentage of valid graphs
#         score = float(top_graphs_count) / float(batch_size)
#         # print('Topty score: {}'.format(score))
#         # print('Topty graphs: {}'.format(top_graphs_indices))
#         # print('All in-top-k check took: {}'.format(time.time() - start_compute))
#         return score
#
#     def is_in_top_percent(self, graph_nodes, graph_edges):
#         # Convert edges into an adjacency matrix
#         first_node = torch.min(graph_edges).item()
#         graph_edges = graph_edges - first_node
#         # print('self.is_model_in_top_n_percent(graph_nodes, graph_edges): {}'.format(
#         #     self.is_model_in_top_n_percent(self.top_k_models[0]['x'], self.top_k_models[0]['edge_index'])))
#         # print('self.is_model_in_top_n_percent(graph_nodes, graph_edges): {}'.format(
#         #     self.is_model_in_top_n_percent(self.dataset.get(0)['x'], self.dataset.get(0)['edge_index'])))
#         # print('self.is_model_in_top_n_percent(graph_nodes, graph_edges): {}'.format(
#         #     self.is_model_in_top_n_percent(graph_nodes, graph_edges)))
#         # raise ValueError('Not implemented error')
#         return self.is_model_in_top_n_percent(graph_nodes, graph_edges)
#
#     def is_model_in_top_n_percent(self, model_nodes, model_edges):
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
#         reply = hash((model_nodes.detach().cpu().numpy().tobytes(),
#                       model_edges.detach().cpu().numpy().tobytes())) in self.top_k_models_hashes
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
#         return reply
