import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as t_func
import torch_geometric as tg
# Import my modules
from scripts.layers import Edger
from scripts.utils import Utils, timeit
from scripts.utils import GraphsTorch as Graphs


class GeneratorNet(nn.Module):
    def __init__(self, n_nodes=7, hidden_dim=10, n_ops=5, mu=2,
                 sample_pre=False, tau=1, dataset='nas101', sub_skip=False, refine=False):
        super(GeneratorNet, self).__init__()
        self.n_nodes = n_nodes
        self.n_edges_fc = np.sum([i for i in range(self.n_nodes + 1)])
        self.hidden_dim = hidden_dim
        self.n_ops = n_ops
        self.mu = mu
        self.sample_pre = sample_pre
        self.tau = tau
        self.dataset = dataset
        self.sub_skip = False if dataset == 'nas101' else sub_skip
        self.refine = refine
        # Define layers for embedding
        self.convs = nn.ModuleList()
        for _ in range(self.mu):
            self.convs.append(tg.nn.GCNConv(self.hidden_dim, self.hidden_dim))
        # Define layers for edge scoring
        self.edge_scorer = Edger(self.hidden_dim, sample_pre=self.sample_pre, tau=self.tau)
        # Define layer for mapping embedding to operation
        self.mapper = tg.nn.GCNConv(self.hidden_dim, self.n_ops)
        # Define one hot encoding of input and output operation
        operations = Utils().get_class_name_dict(bench=self.dataset, sub_skip=self.sub_skip)
        n_dataset_ops = len(list(operations.keys()))
        self.input_embedding = torch.zeros((n_dataset_ops,))
        self.input_embedding[0] = 1
        self.output_embedding = torch.zeros((n_dataset_ops,))
        self.output_embedding[-1] = 1

    # @timeit
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch
        batch_size = torch.max(batch).item() + 1
        # Apply Graph convolutions to obtain nodes embedding
        # print('self.convs[0].device: {}'.format(next(self.convs[0].parameters()).device))
        # print('X device: {}'.format(x.device))
        # print('edge_index device: {}'.format(edge_index.device))
        embedding = t_func.relu(self.convs[0](x, edge_index))
        for index in range(1, self.mu):
            embedding = t_func.relu(self.convs[index](embedding, edge_index))
        # Get scores of edges using the linear layer
        embedding, surviving_edges, batch, edges_scores = self.edge_scorer(embedding, edge_index, batch)
        if not self.sample_pre:
            avg_edges_scores = (edges_scores[:, 0] + edges_scores[:, 1]) / 2
            # print('avg_edges_scores: {}'.format(avg_edges_scores))
            # Apply last convolutional layer to get the operation for each node
            op_embedding = self.mapper(embedding, surviving_edges, avg_edges_scores)
        else:
            # print('edges_scores: {}'.format(edges_scores))
            # Apply last convolutional layer to get the operation for each node
            op_embedding = self.mapper(embedding, surviving_edges, edges_scores)
        # Get operation embeddings using gumbel softmax
        op_indices = t_func.gumbel_softmax(op_embedding, tau=self.tau, hard=True, dim=-1)
        # # Set first and last node of each batch to have input and output operation
        # for i in range(batch_size):
        #     # print('self.n_nodes = {}'.format(self.n_nodes))
        #     op_indices[i * self.n_nodes, :] = self.input_embedding
        #     op_indices[(i + 1) * self.n_nodes - 1, :] = self.output_embedding
        # Refine generated graphs
        if self.refine:
            st = time.time()
            op_indices, surviving_edges, batch = Graphs(sample_pre=self.sample_pre,
                                                        device='cuda' if next(self.parameters()).is_cuda else 'cpu',
                                                        dataset=self.dataset,
                                                        sub_skip=self.sub_skip).refine_batched_graphs(nodes=op_indices,
                                                                                                      edges=surviving_edges,
                                                                                                      edges_scores=edges_scores,
                                                                                                      batch=batch,
                                                                                                      n_predicted_ops=self.n_ops)
            # print('Time taken by refine operation is: {:.5f} s  '.format(time.time()-st))
        else:
            op_indices, surviving_edges, batch = Graphs(sample_pre=self.sample_pre,
                                device='cuda' if next(self.parameters()).is_cuda else 'cpu',
                                dataset=self.dataset,
                                sub_skip=self.sub_skip).remove_unused_edges_from_batched_graphs(nodes=op_indices,
                                                                                                edges=surviving_edges,
                                                                                                edges_scores=edges_scores,
                                                                                                batch=batch,
                                                                                                n_predicted_ops=self.n_ops)
        # st = time.time()
        generated_graph = tg.data.Data(x=op_indices,
                                       edge_index=surviving_edges,
                                       batch=batch).to(
            torch.device('cuda' if next(self.parameters()).is_cuda else 'cpu'))
        # print('Time taken by moving data in TG format: {}'.format(time.time() - st))
        # print('Generated graph: {}'.format(generated_graph))
        return generated_graph
