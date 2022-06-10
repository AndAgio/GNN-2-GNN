import time
import numpy as np
import torch
import torch_geometric as tg
# My modules
from scripts.utils import timeit, convert_edges_to_adj, hash_model
from .validity import is_model_valid


class Uniqueness():
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
        valid_hashes = []
        for batch_index in range(batch_size):
            start_node_index = input_indices_list[batch_index]
            end_node_index = output_indices_list[batch_index]
            model_nodes = x[start_node_index:end_node_index+1, :]
            model_adj = adj_mat[start_node_index:end_node_index+1, start_node_index:end_node_index+1]
            # Compute validity for single model
            if is_model_valid(model_nodes, model_adj):
                valid_hashes.append(hash_model(model_nodes, model_adj))
        # Get unique hash values from the list of valid hashes
        values, counts = np.unique(valid_hashes, return_counts=True)
        # print('values: {}'.format(values))
        # print('counts: {}'.format(counts))
        # print('counts==1: {}'.format(counts==1))
        # print('values[counts==1]: {}'.format(values[counts==1]))
        if len(valid_hashes) > 0:
            score = float(len(values[counts==1])) / float(len(valid_hashes))
        else:
            score = 0.0
        return score