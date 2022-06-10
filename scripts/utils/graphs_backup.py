import torch
import torch.nn.functional as t_func
import networkx as nx


class Graphs():
    def __init__(self, device='cpu'):
        self.device = device

    def convert_tg_to_nx(self, data):
        graph = nx.DiGraph()
        for index, node in enumerate(data.x):
            # # print('Node: {}'.format(node))
            graph.add_nodes_from([(index, {'ops': torch.argmax(node, dim=-1).item()})])
        for index in range(data.edge_index.shape[1]):
            src = data.edge_index[0, index]
            dst = data.edge_index[1, index]
            graph.add_edge(src.item(), dst.item())
        # # print('TG graph edges: {}'.format(data.edge_index))
        # # print('NX graph edges: {}'.format(graph.edges))
        return graph

    def refine_batched_graphs(self, nodes, edges, edges_scores, batch):
        edges = self.remove_unused_edges(edges, edges_scores)
        edges = self.remove_self_loops(edges)
        new_nodes_embeddings = []
        new_edges = []
        new_batch = []
        batch_size = torch.max(batch).item() + 1
        n_nodes = int(list(nodes.size())[0] / batch_size)
        # print('n_nodes: {}'.format(n_nodes))
        edges_src = edges[0, :]
        edges_dst = edges[1, :]
        for batch_index in range(batch_size):
            nodes_graph = nodes[batch_index * n_nodes:(batch_index + 1) * n_nodes, :]
            # print('nodes_graph: {}'.format(nodes_graph))
            nodes_indices = [i for i in range(batch_index * n_nodes, (batch_index + 1) * n_nodes)]
            # print('Nodes indices: {}'.format(nodes_indices))
            indices_to_keep_src = [i for i in range(len(edges_src)) if edges_src[i] in nodes_indices]
            indices_to_keep_dst = [i for i in range(len(edges_dst)) if edges_dst[i] in nodes_indices]
            indices_to_keep = indices_to_keep_src + list(set(indices_to_keep_dst) - set(indices_to_keep_src))
            edges_graph = edges[:, indices_to_keep]
            # print('Edges graph: {}'.format(edges_graph))
            nodes_graph, edges_graph, batch_graph = self.refine_single_graph(nodes_graph,
                                                                             nodes_indices,
                                                                             edges_graph,
                                                                             batch_index)
            new_nodes_embeddings.append(nodes_graph)
            new_edges.append(edges_graph)
            new_batch.append(batch_graph)

        # print('Nodes before refining: {}'.format(nodes))
        # print('Edges before refining: {}'.format(edges))
        # print('Batch before refining: {}'.format(batch))
        # print('New edges: {}'.format(new_edges))
        new_nodes = torch.cat(new_nodes_embeddings, dim=0)
        new_edges = torch.cat(new_edges, dim=1)
        new_batch = torch.cat(new_batch, dim=0)

        # Refine edges indices since node 55 may be mapped to node 34 23 or 47 depending on the number of dropped nodes
        new_edges_src = list(new_edges[0, :].cpu().numpy())
        new_edges_dst = list(new_edges[1, :].cpu().numpy())
        # print('new_edges_src: {}'.format(new_edges_src))
        # print('new_edges_dst: {}'.format(new_edges_dst))
        nodes_indices_in_new_edges = list(set(new_edges_src)) + list(set(new_edges_dst) - set(new_edges_src))
        nodes_indices_in_new_edges.sort()
        # print('nodes_indices_in_new_edges: {}'.format(nodes_indices_in_new_edges))
        old_to_new_nodes_mapping = {nodes_indices_in_new_edges[i]: i for i in range(len(nodes_indices_in_new_edges))}
        # print('old_to_new_nodes_mapping: {}'.format(old_to_new_nodes_mapping))
        n_columns = list(new_edges.size())[1]
        for edge_index in range(n_columns):
            old_edge = new_edges[:, edge_index].cpu().numpy()
            old_edge_src = old_edge[0]
            old_edge_dst = old_edge[1]
            new_edge_src = old_to_new_nodes_mapping[old_edge_src]
            new_edge_dst = old_to_new_nodes_mapping[old_edge_dst]
            new_edge = torch.tensor([[new_edge_src], [new_edge_dst]], dtype=torch.int64, device=self.device).squeeze()
            # print('new_edge: {}'.format(new_edge.shape))
            # print('new_edges[:, edge_index]: {}'.format(new_edges[:, edge_index].shape))
            new_edges[:, edge_index] = new_edge
            # new_edges[0, edge_index] = torch.tensor([new_edge_src], dtype=torch.int64)
            # new_edges[1, edge_index] = torch.tensor([new_edge_dst], dtype=torch.int64)
        # print('Nodes after refining: {}'.format(new_nodes))
        # print('Edges after refining: {}'.format(new_edges))
        # print('Batch after refining: {}'.format(new_batch))

        # print('\n\n')
        # print('Nodes shape before refining: {}'.format(nodes.shape))
        # print('Edges shape before refining: {}'.format(edges.shape))
        # print('Batch shape before refining: {}'.format(batch.shape))
        # print('Nodes shape after refining: {}'.format(new_nodes.shape))
        # print('Edges shape after refining: {}'.format(new_edges.shape))
        # print('Batch shape after refining: {}'.format(new_batch.shape))
        # print('\n\n')
        return new_nodes, new_edges, new_batch

    def refine_single_graph(self, nodes_embedding, nodes_indices, edges, batch_index):
        nodes_embedding_mapping = {nodes_indices[i]: nodes_embedding[i, :].unsqueeze(dim=0) for i in
                                   range(len(nodes_indices))}
        # print('nodes_embedding_mapping: {}'.format(nodes_embedding_mapping))
        refined, indices_of_nodes_to_drop = self.check_graph_is_refined(nodes_indices, edges)
        while not refined:
            nodes_indices, edges = self.refine_step(nodes_indices, edges, indices_of_nodes_to_drop)
            refined, indices_of_nodes_to_drop = self.check_graph_is_refined(nodes_indices, edges)
        new_nodes_embedding = [nodes_embedding_mapping[node_index] for node_index in nodes_indices]
        # print('new_nodes_embedding: {}'.format(new_nodes_embedding))
        new_nodes_embedding = torch.cat(new_nodes_embedding, dim=0)
        # print('new_nodes_embedding: {}'.format(new_nodes_embedding))
        # print('new edges: {}'.format(edges))
        batch_graph = torch.ones((len(nodes_indices),), dtype=torch.int64) * batch_index
        # print('batch_graph: {}'.format(batch_graph))
        return new_nodes_embedding, edges, batch_graph

    def check_graph_is_refined(self, nodes_indices, edges):
        edges = self.remove_self_loops(edges)
        first_node_index = nodes_indices[0]
        last_node_index = nodes_indices[-1]
        edges_src = edges[0, :]
        edges_dst = edges[1, :]
        nodes_without_input = [node_index for node_index in nodes_indices if node_index not in edges_src]
        nodes_without_output = [node_index for node_index in nodes_indices if node_index not in edges_dst]
        indices_of_nodes_to_drop = nodes_without_input + list(set(nodes_without_output) - set(nodes_without_input))
        if first_node_index in indices_of_nodes_to_drop:
            indices_of_nodes_to_drop.remove(first_node_index)
        if last_node_index in indices_of_nodes_to_drop:
            indices_of_nodes_to_drop.remove(last_node_index)
        # print('indices_of_nodes_to_drop: {}'.format(indices_of_nodes_to_drop))
        if indices_of_nodes_to_drop == []:
            return True, None
        else:
            return False, indices_of_nodes_to_drop

    def refine_step(self, nodes_indices, edges, indices_of_nodes_to_drop):
        # print('Nodes indices before refine step: {}'.format(nodes_indices))
        # print('Edges before refine step: {}'.format(edges))
        # print('indices_of_nodes_to_drop before refine step: {}'.format(indices_of_nodes_to_drop))
        kept_nodes_indices = [node_index for node_index in nodes_indices if node_index not in indices_of_nodes_to_drop]
        # print('kept_nodes_indices after refine step: {}'.format(kept_nodes_indices))
        kept_edges = []
        n_columns = list(edges.size())[1]
        for edge_index in range(n_columns):
            edge = edges[:, edge_index]
            edge_src = edge[0]
            edge_dst = edge[1]
            if edge_src not in indices_of_nodes_to_drop and edge_dst not in indices_of_nodes_to_drop:
                edge = edge.unsqueeze(dim=-1)
                kept_edges.append(edge)
                # print('Edge: {}'.format(edge))
        kept_edges_without_self_loops = self.remove_self_loops(kept_edges)
        # print('kept_edges_without_self_loops: {}'.format(kept_edges_without_self_loops))
        if kept_edges_without_self_loops.nelement() == 0:  # == torch.empty((2, 0), dtype=torch.int64):
            kept_edges.append(torch.tensor([[kept_nodes_indices[0]], [kept_nodes_indices[-1]]], dtype=torch.int64,
                                           device=self.device))
        kept_edges = torch.cat(kept_edges, dim=-1)
        # if kept_edges == []:
        #     kept_edges.append(torch.tensor([[kept_nodes_indices[0]], [kept_nodes_indices[-1]]], dtype=torch.int64))
        # kept_edges = torch.cat(kept_edges, dim=-1)
        # print('kept_edges after refine step: {}'.format(kept_edges))
        return kept_nodes_indices, kept_edges

    def remove_unused_edges(self, edges, edges_scores):
        edges_scores = t_func.gumbel_softmax(edges_scores,
                                             tau=1,
                                             hard=True,
                                             dim=-1)
        edges_scores = torch.argmax(edges_scores, dim=-1)
        surviving_edges_indices = edges_scores.bool()
        # print('1/list(edges.shape)[1]: {}'.format(1/list(edges.shape)[1]))
        # surviving_edges_indices = edges_scores > 1/list(edges.shape)[1]
        # print('surviving_edges_indices: {}'.format(surviving_edges_indices))
        edges = edges[:, surviving_edges_indices]
        # print('Edges after shape: {}'.format(edges.shape))
        return edges

    def remove_self_loops(self, edges):
        # print('Edges before removing self loops: {}'.format(edges))
        if type(edges) == list:
            if not edges == []:
                edges = torch.cat(edges, dim=-1)
            else:
                edges = torch.empty((2, 0), dtype=torch.int64)
        # print('Edges before removing self loops converted: {}'.format(edges))
        kept_edges = []
        n_columns = list(edges.size())[1]
        for edge_index in range(n_columns):
            edge = edges[:, edge_index]
            edge_src = edge[0]
            edge_dst = edge[1]
            if edge_src != edge_dst:
                edge = edge.unsqueeze(dim=-1)
                kept_edges.append(edge)
                # print('Edge: {}'.format(edge))
        if not kept_edges == []:
            kept_edges = torch.cat(kept_edges, dim=-1)
        else:
            kept_edges = torch.empty((2, 0), dtype=torch.int64)
        # print('kept_edges after removal of self loops: {}'.format(kept_edges))
        return kept_edges
