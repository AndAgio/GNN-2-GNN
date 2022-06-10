from collections import namedtuple
import random

import torch
import torch.nn.functional as t_func
from torch_scatter import scatter_add
from torch_sparse import coalesce
from torch_geometric.utils import softmax


class Edger(torch.nn.Module):
    r"""The edge pooling operator from the `"Towards Graph Pooling by Edge
    Contraction" <https://graphreason.github.io/papers/17.pdf>`_ and
    `"Edge Contraction Pooling for Graph Neural Networks"
    <https://arxiv.org/abs/1905.10990>`_ papers.

    In short, a score is computed for each edge.
    Edges are contracted iteratively according to that score unless one of
    their nodes has already been part of a contracted edge.

    To duplicate the configuration from the "Towards Graph Pooling by Edge
    Contraction" paper, use either
    :func:`EdgePooling.compute_edge_score_softmax`
    or :func:`EdgePooling.compute_edge_score_tanh`, and set
    :obj:`add_to_edge_score` to :obj:`0`.

    To duplicate the configuration from the "Edge Contraction Pooling for
    Graph Neural Networks" paper, set :obj:`dropout` to :obj:`0.2`.

    Args:
        in_channels (int): Size of each input sample.
        edge_score_method (function, optional): The function to apply
            to compute the edge score from raw edge scores. By default,
            this is the softmax over all incoming edges for each node.
            This function takes in a :obj:`raw_edge_score` tensor of shape
            :obj:`[num_nodes]`, an :obj:`edge_index` tensor and the number of
            nodes :obj:`num_nodes`, and produces a new tensor of the same size
            as :obj:`raw_edge_score` describing normalized edge scores.
            Included functions are
            :func:`EdgePooling.compute_edge_score_softmax`,
            :func:`EdgePooling.compute_edge_score_tanh`, and
            :func:`EdgePooling.compute_edge_score_sigmoid`.
            (default: :func:`EdgePooling.compute_edge_score_softmax`)
        dropout (float, optional): The probability with
            which to drop edge scores during training. (default: :obj:`0`)
        add_to_edge_score (float, optional): This is added to each
            computed edge score. Adding this greatly helps with unpool
            stability. (default: :obj:`0.5`)
    """

    def __init__(self, in_channels, edge_score_method=None, dropout=0,
                 add_to_edge_score=0.5):
        super(Edger, self).__init__()
        self.in_channels = in_channels
        if edge_score_method is None:
            edge_score_method = self.compute_edge_score_softmax
        self.compute_edge_score = edge_score_method
        self.add_to_edge_score = add_to_edge_score
        self.dropout = dropout
        self.threshold = 0.8
        # self.threshold = torch.nn.Parameter(torch.rand(1, ),
        #                                     requires_grad=True)

        # self.lin = torch.nn.Linear(2 * in_channels, 1)
        self.lin = torch.nn.Linear(2 * in_channels, 2)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    @staticmethod
    def compute_edge_score_softmax(raw_edge_score, edge_index, num_nodes):
        return softmax(raw_edge_score, edge_index[1], num_nodes=num_nodes)

    @staticmethod
    def compute_edge_score_tanh(raw_edge_score, edge_index, num_nodes):
        return torch.tanh(raw_edge_score)

    @staticmethod
    def compute_edge_score_sigmoid(raw_edge_score, edge_index, num_nodes):
        return torch.sigmoid(raw_edge_score)

    def forward(self, x, edge_index, batch):
        r"""Forward computation which computes the raw edge score, normalizes
        it, and merges the edges.

        Args:
            x (Tensor): The node features.
            edge_index (LongTensor): The edge indices.
            batch (LongTensor): Batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific example.

        Return types:
            * **x** *(Tensor)* - The pooled node features.
            * **edge_index** *(LongTensor)* - The coarsened edge indices.
            * **batch** *(LongTensor)* - The coarsened batch vector.
            * **unpool_info** *(unpool_description)* - Information that is
              consumed by :func:`EdgePooling.unpool` for unpooling.
        """
        e = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        e = self.lin(e)#.squeeze()
        e = t_func.dropout(e, p=self.dropout, training=self.training)
        e = self.compute_edge_score(e, edge_index, x.size(0))
        edges_scores = e + self.add_to_edge_score

        # edges_scores = t_func.softmax(edges_scores)

        # edges_scores = t_func.gumbel_softmax(edges_scores,
        #                                      tau=1,
        #                                      hard=True,
        #                                      dim=-1)
        # edges_scores = torch.argmax(edges_scores, dim=-1).float()

        # print('Edges scores shape: {}'.format(edges_scores))
        # print('Surviving edges shape: {}'.format(surviving_edges.shape))
        # print('edge_index[:, surviving_edges].shape: {}'.format(edge_index[:, surviving_edges.bool()].shape))
        # print('Surviving edges: {}'.format(surviving_edges))
        # edge_index = edge_index[:, surviving_edges.bool()]

        # x, edge_index, batch = self._merge_edges(x, edge_index, batch, e)

        # print('self.lin: {}'.format(self.lin.weight))

        return x, edge_index, batch, edges_scores

    def __merge_edges__(self, x, edge_index, batch, edge_score):
        # print('Start!')
        # print('X: {}'.format(x.shape))
        # print('edge_index: {}'.format(edge_index.shape))
        # print('batch: {}'.format(batch.shape))
        # print('edge_score: {}'.format(edge_score.shape))

        nodes_remaining = set(range(x.size(0)))

        cluster = torch.empty_like(batch, device=torch.device('cpu'))
        edge_argsort = torch.argsort(edge_score, descending=True)

        # Iterate through all edges, selecting it if it is not incident to
        # another already chosen edge.
        i = 0
        new_edge_indices = []
        edge_index_cpu = edge_index.cpu()
        for edge_idx in edge_argsort.tolist():
            source = edge_index_cpu[0, edge_idx].item()
            if source not in nodes_remaining:
                continue

            target = edge_index_cpu[1, edge_idx].item()
            if target not in nodes_remaining:
                continue

            new_edge_indices.append(edge_idx)

            cluster[source] = i
            nodes_remaining.remove(source)

            if source != target:
                cluster[target] = i
                nodes_remaining.remove(target)

            i += 1

        #     print('cluster: {}'.format(cluster))
        #     print('nodes_remaining: {}'.format(nodes_remaining))
        #     print('i: {}'.format(i))
        #
        # print('nodes_remaining: {}'.format(nodes_remaining))
        # The remaining nodes are simply kept.
        for node_idx in nodes_remaining:
            cluster[node_idx] = i
            i += 1
        cluster = cluster.to(x.device)

        # print('i: {}'.format(i))
        # We compute the new features as an addition of the old ones.
        new_x = scatter_add(x, cluster, dim=0, dim_size=i)
        # print('x: {}'.format(x))
        # print('cluster: {}'.format(cluster))
        # print('new_x = x + cluster: {}'.format(new_x.shape))
        new_edge_score = edge_score[new_edge_indices]
        # print('edge_score: {}'.format(edge_score))
        # print('new_edge_indices: {}'.format(new_edge_indices))
        # print('new_edge_score = edge_score[new_edge_indices]: {}'.format(new_edge_score.shape))
        if len(nodes_remaining) > 0:
            remaining_score = x.new_ones((new_x.size(0) - len(new_edge_indices),))
            new_edge_score = torch.cat([new_edge_score, remaining_score])
        new_x = new_x * new_edge_score.view(-1, 1)
        # print('new_edge_score: {}'.format(new_edge_score.shape))
        # print('new_x = new_x * new_edge_score.view(-1, 1): {}'.format(new_x.shape))

        N = new_x.size(0)
        new_edge_index, _ = coalesce(cluster[edge_index], None, N, N)
        # print('cluster: {}'.format(cluster.shape))
        # print('edge_index: {}'.format(edge_index.shape))
        # print('cluster[edge_index]: {}'.format(cluster[edge_index].shape))
        # print('new_edge_index, _ = coalesce(cluster[edge_index], None, N, N): {}'.format(new_edge_index.shape))

        new_batch = x.new_empty(new_x.size(0), dtype=torch.long)
        # print('new_batch: {}'.format(new_batch))
        new_batch = new_batch.scatter_(0, cluster, batch)
        # print('new_batch: {}'.format(new_batch))

        print('Finish!')
        print('new_x: {}'.format(x.shape))
        print('new_edge_index: {}'.format(new_edge_index.shape))
        print('new_batch: {}'.format(new_batch.shape))
        print('self.lin: {}'.format(self.lin.weight))

        return new_x, new_edge_index, new_batch

    def _merge_edges(self, x, edge_index, batch, edge_score):
        print('Start!')
        print('X: {}'.format(x.shape))
        print('edge_index: {}'.format(edge_index.shape))
        print('batch: {}'.format(batch.shape))
        # print('edge_score: {}'.format(edge_score))

        cluster = torch.empty_like(batch, device=torch.device('cpu'))
        edge_argsort = torch.argsort(edge_score, descending=True)

        # Iterate through all edges, selecting it if it is not incident to
        # another already chosen edge.
        # i = 0
        new_edge_indices = []
        # edge_index_cpu = edge_index.cpu()
        for edge_idx in edge_argsort.tolist():
            # print('edge_idx: {}'.format(edge_idx))
            # source = edge_index[0, edge_idx].item()
            # target = edge_index_cpu[1, edge_idx].item()
            # if edge_score[edge_idx] > self.threshold:
            if random.uniform(0, 1) > 0.5:
                new_edge_indices.append(edge_idx)
            # else:
            #     break

        n_nodes = list(x.shape)[0]
        for node_index in range(n_nodes):
            cluster[node_index] = node_index

        # print('new_edge_indices: {}'.format(new_edge_indices))
        # print('cluster: {}'.format(cluster))

        # We compute the new features as an addition of the old ones.
        new_x = scatter_add(x, cluster, dim=0, dim_size=n_nodes)
        # print('x: {}'.format(x))
        # print('new_x: {}'.format(new_x))
        # new_edge_score = edge_score[new_edge_indices]
        # print('edge_score: {}'.format(edge_score))
        # print('new_edge_indices: {}'.format(new_edge_indices))
        # print('new_edge_score = edge_score[new_edge_indices]: {}'.format(new_edge_score.shape))
        # new_x = new_x * new_edge_score.view(-1, 1)
        # print('new_edge_score: {}'.format(new_edge_score.shape))
        # print('new_x = new_x * new_edge_score.view(-1, 1): {}'.format(new_x.shape))

        # print('edge_index: {}'.format(edge_index))
        # print('edge_index[:, new_edge_indices]: {}'.format(edge_index[:, new_edge_indices]))
        new_edge_index, _ = coalesce(cluster[edge_index[:, new_edge_indices]],
                                     None,
                                     n_nodes,
                                     n_nodes)
        # print('cluster: {}'.format(cluster.shape))
        # print('edge_index: {}'.format(edge_index.shape))
        # print('cluster[edge_index]: {}'.format(cluster[edge_index].shape))
        # print('new_edge_index, _ = coalesce(cluster[edge_index], None, N, N): {}'.format(new_edge_index.shape))

        new_batch = x.new_empty(n_nodes, dtype=torch.long)
        # print('new_batch: {}'.format(new_batch))
        new_batch = new_batch.scatter_(0, cluster, batch)
        # print('new_batch: {}'.format(new_batch))
        #
        print('Finish!')
        print('new_x: {}'.format(new_x.shape))
        print('new_edge_index: {}'.format(new_edge_index.shape))
        print('new_batch: {}'.format(new_batch.shape))
        print('self.threshold: {}'.format(self.threshold))
        print('self.lin: {}'.format(self.lin.weight))

        # print('new_edge_score: {}'.format(new_edge_score))

        return new_x, new_edge_index, new_batch
