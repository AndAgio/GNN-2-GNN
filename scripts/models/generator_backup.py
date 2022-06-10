import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as t_func
import torch_geometric as tg
# Import my modules
from scripts.layers import Edger
from scripts.utils import Graphs, timeit


class GeneratorNet(nn.Module):
    def __init__(self, n_nodes=7, hidden_dim=10, n_ops=5):
        super(GeneratorNet, self).__init__()
        self.n_nodes = n_nodes
        self.n_edges_fc = np.sum([i for i in range(self.n_nodes + 1)])
        # print('# of edges: {}'.format(self.n_edges_fc))
        self.hidden_dim = hidden_dim
        self.n_ops = n_ops
        # Define layers for embedding
        self.conv = tg.nn.GCNConv(self.hidden_dim, self.hidden_dim)
        # Define layers for edge scoring
        self.layer = True
        # if self.layer:
        self.edge_scorer = Edger(self.hidden_dim)
        # else:
        #     self.edge_scorer = nn.Linear(self.hidden_dim * 2, 1)
        # Define layer for mapping embedding to operation
        self.mapper = tg.nn.GCNConv(self.hidden_dim, self.n_ops)
        # Define the learnable threshold for link selection
        # self.learnable_threshold = torch.autograd.Variable(nn.Parameter(torch.zeros(self.n_edges_fc, ),
        #                                                                 requires_grad=True))
        # Define one hot encoding of input and output operation
        self.input_embedding = torch.zeros((self.n_ops,))
        self.input_embedding[0] = 1
        self.output_embedding = torch.zeros((self.n_ops,))
        self.output_embedding[-1] = 1

    # @timeit
    def forward(self, data):  # x, edge_index, batch):
        # data = iter(self.generate_fc_graphs(batch_size=batch_size)).next()
        # print('Data in input: {}'.format(data))
        x, edge_index = data.x, data.edge_index
        batch = data.batch
        # print('Batch: {}'.format(batch))
        batch_size = torch.max(batch).item() + 1
        # print('X shape: {}'.format(x.shape))
        # print('Edge index: {}'.format(edge_index.shape))
        # Apply Graph convolutions to obtain nodes embedding
        embedding = t_func.relu(self.conv(x, edge_index))
        # Get scores of edges using the linear layer
        edges_src = embedding[edge_index[0, :]]
        edges_dst = embedding[edge_index[1, :]]
        edges_src_dst_concat = torch.cat([edges_src, edges_dst], dim=1)
        # print('Edges concatenation: {}'.format(edges_src_dst_concat.requires_grad))
        # if self.layer:
        # print('x: {}'.format(embedding))
        embedding, surviving_edges, batch, edges_scores = self.edge_scorer(embedding, edge_index, batch)
        avg_edges_scores = (edges_scores[:, 0] + edges_scores[:, 1]) / 2
        # else:
        #     edges_scores = self.edge_scorer(edges_src_dst_concat)
        #     # print('Edges scores before softmax: {}'.format(edges_scores.requires_grad))
        #     edges_scores = t_func.softmax(edges_scores, dim=0).squeeze()
        #     # edges_scores = edges_scores.squeeze()
        #     # print('Edges scores after softmax: {}'.format(edges_scores.requires_grad))
        #     # print('Edges scores shape: {}'.format(edges_scores.shape))
        #     # Threshold the edges scores to keep only the links with score higher than threshold
        #     # surviving_edges_indices = edges_scores.squeeze() > threshold
        #     # threshold = self.learnable_threshold.repeat(batch_size)
        #     threshold = torch.tensor(0)  # 1 / self.n_edges_fc)
        #     # print('Learnable threshold: {}'.format(self.learnable_threshold))
        #     # print('Threshold: {}'.format(threshold.requires_grad))
        #     indices = torch.where(edges_scores > threshold,
        #                           torch.tensor(True),
        #                           torch.tensor(False))
        #     # edges_scores = torch.autograd.Variable(torch.ones((28*batch_size,), requires_grad=True))
        #     # print('Edges scores shape: {}'.format(edges_scores.shape))
        #     # mapped_edges_scores = torch.index_select(edges_scores,
        #     #                                          dim=0,
        #     #                                          index=torch.autograd.Variable(indices))
        #     # print('Mapped edges scores shape: {}'.format(mapped_edges_scores.shape))
        #     surviving_edges = edge_index[:, indices]
        #     # print('Surviving edges: {}'.format(surviving_edges))
        # Apply last convolutional layer to get the operation for each node
        op_embedding = self.mapper(embedding, surviving_edges, avg_edges_scores)  # edge_index, mapped_edges_scores)
        # print('Operations embedding: {}'.format(op_embedding.requires_grad))
        op_indices = t_func.gumbel_softmax(op_embedding, tau=1, hard=True, dim=-1)
        # Set first and last node of each batch to have input and output operation
        for i in range(batch_size):
            op_indices[i * self.n_nodes, :] = self.input_embedding
            op_indices[(i + 1) * self.n_nodes - 1, :] = self.output_embedding
        # print('Op indices using gumbel: {}'.format(op_indices.requires_grad))
        # op_indices_dummy = torch.argmax(op_embedding, dim=-1)
        # op_indices_dummy = t_func.one_hot(op_indices_dummy.squeeze().to(torch.int64),
        #                                   num_classes=self.n_ops).float()
        # print('Op indices using argmax: {}'.format(op_indices_dummy))
        op_indices, surviving_edges, batch = Graphs(
            device='cuda' if next(self.parameters()).is_cuda else 'cpu').refine_batched_graphs(nodes=op_indices,
                                                                                               edges=surviving_edges,
                                                                                               edges_scores=edges_scores,
                                                                                               batch=batch)

        generated_graph = tg.data.Data(x=op_indices,
                                       edge_index=surviving_edges,
                                       batch=batch).to(
            torch.device('cuda' if next(self.parameters()).is_cuda else 'cpu'))
        return generated_graph  # op_indices, edge_index, batch
