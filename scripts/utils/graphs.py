import torch
import torch.nn.functional as t_func
import networkx as nx
from scripts.utils import Utils
from .decorators import timeit


class Graphs():
    def __init__(self, sample_pre=False, device='cpu', dataset='nas101', sub_skip=False):
        self.sample_pre = sample_pre
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
        # Method for refining graphs that are stored in a single batch
        edges = self.remove_unused_edges(edges, edges_scores)
        edges = self.remove_self_loops(edges)
        # Identify splits between graphs in a single batch
        new_nodes_embeddings = []
        new_edges = []
        new_batch = []
        batch_size = torch.max(batch).item() + 1
        n_nodes = int(list(nodes.size())[0] / batch_size)
        edges_src = edges[0, :]
        edges_dst = edges[1, :]
        # Refine operations embedding if input and outputs are not predicted
        operations = Utils().get_class_name_dict(bench=self.dataset, sub_skip=self.sub_skip)
        n_dataset_ops = len(list(operations.keys()))
        input_embedding = torch.zeros((n_dataset_ops,))
        input_embedding[0] = 1
        output_embedding = torch.zeros((n_dataset_ops,))
        output_embedding[-1] = 1
        if n_predicted_ops is not None and n_predicted_ops != n_dataset_ops:
            # print('new_nodes before appending: {}'.format(nodes))
            nodes = torch.cat((nodes, torch.zeros((nodes.shape[0], 1))), dim=1)
            nodes = torch.cat((torch.zeros((nodes.shape[0], 1)), nodes), dim=1)
            # print('new_nodes after appending: {}'.format(nodes))
        # For each graph in the batch ...
        for batch_index in range(batch_size):
            # Set first and last node of each batch to have input and output operation
            nodes[batch_index * n_nodes, :] = input_embedding
            nodes[(batch_index + 1) * n_nodes - 1, :] = output_embedding
            # Identify nodes in a single graph
            nodes_graph = nodes[batch_index * n_nodes:(batch_index + 1) * n_nodes, :]
            nodes_indices = [i for i in range(batch_index * n_nodes, (batch_index + 1) * n_nodes)]
            # Identify edges in a single graph
            indices_to_keep_src = [i for i in range(len(edges_src)) if edges_src[i] in nodes_indices]
            indices_to_keep_dst = [i for i in range(len(edges_dst)) if edges_dst[i] in nodes_indices]
            indices_to_keep = indices_to_keep_src + list(set(indices_to_keep_dst) - set(indices_to_keep_src))
            edges_graph = edges[:, indices_to_keep]
            # Refine single graph appending input and output nodes, and removing cycles
            nodes_graph, edges_graph, batch_graph = self.refine_single_graph(nodes_graph,
                                                                             nodes_indices,
                                                                             edges_graph,
                                                                             batch_index)
            new_nodes_embeddings.append(nodes_graph)
            new_edges.append(edges_graph)
            new_batch.append(batch_graph)

        # Define new batch with all refined graphs
        new_nodes = torch.cat(new_nodes_embeddings, dim=0)
        new_edges = torch.cat(new_edges, dim=1)
        new_batch = torch.cat(new_batch, dim=0)
        # Refine edges indices since node 55 may be mapped to node 34, 23, 47, or whatever, depending on the number of dropped nodes
        new_edges_src = list(new_edges[0, :].cpu().numpy())
        new_edges_dst = list(new_edges[1, :].cpu().numpy())
        # Map old nodes to new nodes
        nodes_indices_in_new_edges = list(set(new_edges_src)) + list(set(new_edges_dst) - set(new_edges_src))
        nodes_indices_in_new_edges.sort()
        old_to_new_nodes_mapping = {nodes_indices_in_new_edges[i]: i for i in range(len(nodes_indices_in_new_edges))}
        # Map old edges to new edges
        n_columns = list(new_edges.size())[1]
        for edge_index in range(n_columns):
            old_edge = new_edges[:, edge_index].cpu().numpy()
            old_edge_src = old_edge[0]
            old_edge_dst = old_edge[1]
            new_edge_src = old_to_new_nodes_mapping[old_edge_src]
            new_edge_dst = old_to_new_nodes_mapping[old_edge_dst]
            new_edge = torch.tensor([[new_edge_src], [new_edge_dst]], dtype=torch.int64, device=self.device).squeeze()
            new_edges[:, edge_index] = new_edge
        # Return the refined batch
        return new_nodes, new_edges, new_batch

    @timeit
    def refine_single_graph(self, nodes_embedding, nodes_indices, edges, batch_index):
        # Method to refine a single graph
        # print('nodes_embedding: {}'.format(nodes_embedding))
        # print('nodes_indices: {}'.format(nodes_indices))
        # print('edges: {}'.format(edges))
        # print('batch_index: {}'.format(batch_index))
        nodes_embedding_mapping = {nodes_indices[i]: nodes_embedding[i, :].unsqueeze(dim=0) for i in
                                   range(len(nodes_indices))}
        # print('nodes_embedding_mapping: {}'.format(nodes_embedding_mapping))
        refined, indices_of_nodes_to_drop = self.check_graph_is_refined(nodes_indices, edges)
        while not refined:
            nodes_indices, edges = self.refine_step(nodes_indices, edges, indices_of_nodes_to_drop)
            refined, indices_of_nodes_to_drop = self.check_graph_is_refined(nodes_indices, edges)
        # Remove skip connections if dealing with NATS dataset.
        # if self.dataset == 'nats' and self.sub_skip:
        #     nodes_indices, edges = self.remove_skips(nodes_indices, nodes_embedding_mapping, edges)
        # Compute back node embeddings from stored dictionary
        new_nodes_embedding = [nodes_embedding_mapping[node_index] for node_index in nodes_indices]
        # print('new_nodes_embedding: {}'.format(new_nodes_embedding))
        new_nodes_embedding = torch.cat(new_nodes_embedding, dim=0)
        # print('new_nodes_embedding: {}'.format(new_nodes_embedding))
        # print('new edges: {}'.format(edges))
        batch_graph = torch.ones((len(nodes_indices),), dtype=torch.int64) * batch_index
        # print('batch_graph: {}'.format(batch_graph))
        return new_nodes_embedding, edges, batch_graph

    @timeit
    def check_graph_is_refined(self, nodes_indices, edges):
        # Method to check if a single graph is refined
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

    @timeit
    def refine_step(self, nodes_indices, edges, indices_of_nodes_to_drop):
        # Method defining the single refinement step that should be repeated in order to obtain the refined graph
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

    @timeit
    def remove_skips(self, nodes_indices, nodes_embedding_mapping, edges):
        # Check if skip connections need to be removed or not
        name_class_dict = self.utils.get_name_class_dict(bench='nats', sub_skip=self.sub_skip)
        print('name_class_dict: {}'.format(name_class_dict))
        # Identify nodes corresponding to skip connections
        skip_indices = []
        for node_index in nodes_indices:
            operation = nodes_embedding_mapping[node_index]
            # print('operation: {}'.format(operation))
            # print('torch.argmax(operation, dim=1).numpy()[0]: {}'.format(torch.argmax(operation, dim=1).numpy()[0]))
            if torch.argmax(operation, dim=1).numpy()[0] == name_class_dict['skip_connect']:
                skip_indices.append(node_index)
        # print('skip_indices: {}'.format(skip_indices))
        # For each node corresponding to a skip connection...
        # print('edges: {}'.format(edges))
        while len(skip_indices) > 0:
            node_index = skip_indices[-1]
            # print('node_index: {}'.format(node_index))
            # Identify starting and ending point of the skip connection
            starting_node_pos = list(torch.where(edges[1, :] == node_index)[0])
            starting_nodes = edges[0, starting_node_pos]
            ending_node_pos = list(torch.where(edges[0, :] == node_index)[-1])
            ending_nodes = edges[1, ending_node_pos]
            # print('starting_node: {}'.format(starting_nodes))
            # print('ending_node: {}'.format(ending_nodes))
            # Create link between starting and ending nodes
            for st_node in starting_nodes:
                for end_node in ending_nodes:
                    # print('st_node: {}'.format(st_node))
                    # print('end_node: {}'.format(end_node))
                    edge = torch.tensor([[st_node], [end_node]])
                    edges = torch.cat((edges, edge), 1)
            # print('edges: {}'.format(edges))
            # Remove node corresponding to skip connections from adjacency matrix and list of nodes
            columns_to_remove_from_edges = starting_node_pos + ending_node_pos
            columns_to_remove_from_edges = [col.numpy() for col in columns_to_remove_from_edges]
            # print('columns_to_remove_from_edges: {}'.format(columns_to_remove_from_edges))
            for node_pos in reversed(columns_to_remove_from_edges):
                # print('node_pos: {}'.format(node_pos))
                edges = torch.cat((edges[:, 0:node_pos], edges[:, node_pos + 1:]), 1)
            # Sort edges
            # print('edges: {}'.format(edges))
            # print('torch.sort(edges, 0)[1]: {}'.format(torch.sort(edges[0, :])[1]))
            edges = edges[:, torch.sort(edges[0, :])[1]]
            # sorted(edges, key=lambda x: x[0, :])
            # print('edges: {}'.format(edges))
            # Remove duplicates from edges
            edges = torch.unique(edges, dim=1)
            # print('edges: {}'.format(edges))
            # print('nodes_indices before pop: {}'.format(nodes_indices))
            nodes_indices.remove(node_index)
            # print('nodes_indices after pop: {}'.format(nodes_indices))
            # Remove treated node from list of indices to remove
            skip_indices = skip_indices[:-1]
        # for node_index in nodes_indices:
        #     operation = nodes_embedding_mapping[node_index]
        #     print('operation: {}'.format(operation))
        #     print('torch.argmax(operation, dim=1).numpy()[0]: {}'.format(torch.argmax(operation, dim=1).numpy()[0]))
        # Return refined nodes and adjacency matrix
        return nodes_indices, edges

    @timeit
    def remove_unused_edges(self, edges, edges_scores):
        # Method to remove unused edges
        # Sample edges using gumbel softmax
        if self.sample_pre:
            edges_scores = edges_scores - self.epsilon
        else:
            edges_scores = t_func.gumbel_softmax(edges_scores,
                                                 tau=1,
                                                 hard=True,
                                                 dim=-1)
            edges_scores = torch.argmax(edges_scores, dim=-1)
        surviving_edges_indices = edges_scores.bool()
        # Refine edges
        edges = edges[:, surviving_edges_indices]
        return edges

    @timeit
    def remove_self_loops(self, edges):
        # Method to remove self loops from a graph
        # print('Edges before removing self loops: {}'.format(edges))
        if type(edges) == list:
            if not edges == []:
                edges = torch.cat(edges, dim=-1)
            else:
                edges = torch.empty((2, 0), dtype=torch.int64, device=self.device)
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
            kept_edges = torch.empty((2, 0), dtype=torch.int64, device=self.device)
        # print('kept_edges after removal of self loops: {}'.format(kept_edges))
        return kept_edges

    def remove_unused_edges_from_batched_graphs(self, nodes, edges, edges_scores, batch):
        # Method for refining graphs that are stored in a single batch
        edges = self.remove_unused_edges(edges, edges_scores)
        # Refine operations embedding if input and outputs are not predicted
        batch_size = torch.max(batch).item() + 1
        n_nodes = int(list(nodes.size())[0] / batch_size)
        operations = Utils().get_class_name_dict(bench=self.dataset, sub_skip=self.sub_skip)
        n_dataset_ops = len(list(operations.keys()))
        input_embedding = torch.zeros((n_dataset_ops,))
        input_embedding[0] = 1
        output_embedding = torch.zeros((n_dataset_ops,))
        output_embedding[-1] = 1
        n_predicted_ops = nodes.shape[1]
        if n_predicted_ops != n_dataset_ops:
            # print('new_nodes before appending: {}'.format(nodes))
            nodes = torch.cat((nodes, torch.zeros((nodes.shape[0], 1))), dim=1)
            nodes = torch.cat((torch.zeros((nodes.shape[0], 1)), nodes), dim=1)
            # print('new_nodes after appending: {}'.format(nodes))
        # For each graph in the batch ...
        for batch_index in range(batch_size):
            # Set first and last node of each batch to have input and output operation
            nodes[batch_index * n_nodes, :] = input_embedding
            nodes[(batch_index + 1) * n_nodes - 1, :] = output_embedding
        # Return the refined batch
        return edges

    # def refine_batched_graphs(self, nodes, edges, edges_scores, batch, n_predicted_ops=None):
    #     # Method for refining graphs that are stored in a single batch
    #     edges = self.remove_unused_edges(edges, edges_scores)
    #     edges = self.remove_self_loops(edges)
    #     # Identify splits between graphs in a single batch
    #     new_nodes_embeddings = []
    #     new_edges = []
    #     new_batch = []
    #     batch_size = torch.max(batch).item() + 1
    #     n_nodes = int(list(nodes.size())[0] / batch_size)
    #     edges_src = edges[0, :]
    #     edges_dst = edges[1, :]
    #     # Refine operations embedding if input and outputs are not predicted
    #     operations = Utils().get_class_name_dict(bench=self.dataset, sub_skip=self.sub_skip)
    #     n_dataset_ops = len(list(operations.keys()))
    #     input_embedding = torch.zeros((n_dataset_ops,))
    #     input_embedding[0] = 1
    #     output_embedding = torch.zeros((n_dataset_ops,))
    #     output_embedding[-1] = 1
    #     if n_predicted_ops is not None and n_predicted_ops != n_dataset_ops:
    #         # print('new_nodes before appending: {}'.format(nodes))
    #         nodes = torch.cat((nodes, torch.zeros((nodes.shape[0], 1))), dim=1)
    #         nodes = torch.cat((torch.zeros((nodes.shape[0], 1)), nodes), dim=1)
    #         # print('new_nodes after appending: {}'.format(nodes))
    #     # For each graph in the batch ...
    #     for batch_index in range(batch_size):
    #         # Set first and last node of each batch to have input and output operation
    #         nodes[batch_index * n_nodes, :] = input_embedding
    #         nodes[(batch_index + 1) * n_nodes - 1, :] = output_embedding
    #         # Identify nodes in a single graph
    #         nodes_graph = nodes[batch_index * n_nodes:(batch_index + 1) * n_nodes, :]
    #         nodes_indices = [i for i in range(batch_index * n_nodes, (batch_index + 1) * n_nodes)]
    #         # Identify edges in a single graph
    #         indices_to_keep_src = [i for i in range(len(edges_src)) if edges_src[i] in nodes_indices]
    #         indices_to_keep_dst = [i for i in range(len(edges_dst)) if edges_dst[i] in nodes_indices]
    #         indices_to_keep = indices_to_keep_src + list(set(indices_to_keep_dst) - set(indices_to_keep_src))
    #         edges_graph = edges[:, indices_to_keep]
    #         # Refine single graph appending input and output nodes, and removing cycles
    #         nodes_graph, edges_graph, batch_graph = self.refine_single_graph(nodes_graph,
    #                                                                          nodes_indices,
    #                                                                          edges_graph,
    #                                                                          batch_index)
    #         new_nodes_embeddings.append(nodes_graph)
    #         new_edges.append(edges_graph)
    #         new_batch.append(batch_graph)
    #
    #     # Define new batch with all refined graphs
    #     new_nodes = torch.cat(new_nodes_embeddings, dim=0)
    #     new_edges = torch.cat(new_edges, dim=1)
    #     new_batch = torch.cat(new_batch, dim=0)
    #     # Refine edges indices since node 55 may be mapped to node 34, 23, 47, or whatever, depending on the number of dropped nodes
    #     new_edges_src = new_edges[0, :]
    #     new_edges_dst = new_edges[1, :]
    #     # Map old nodes to new nodes
    #     nodes_indices_in_new_edges = torch.unique(new_edges.flatten(), sorted=True)
    #     # print('nodes_indices_in_new_edges: {}'.format(nodes_indices_in_new_edges))
    #     old_indices = torch.arange(0, nodes_indices_in_new_edges.shape[0], dtype=torch.int64, device=self.device).unsqueeze(dim=-1)
    #     old_to_new_nodes_mapping = torch.cat([nodes_indices_in_new_edges.unsqueeze(dim=-1), old_indices], dim=-1)
    #     # print('old_to_new_nodes_mapping: {}'.format(old_to_new_nodes_mapping))
    #     # Map old edges to new edges
    #     n_columns = list(new_edges.size())[1]
    #     # print('n_columns: {}'.format(n_columns))
    #     for edge_index in range(n_columns):
    #         old_edge = new_edges[:, edge_index]
    #         # print('old_edge: {}'.format(old_edge))
    #         old_edge_src = old_edge[0]
    #         # print('old_edge_src: {}'.format(old_edge_src))
    #         old_edge_dst = old_edge[1]
    #         # print('old_edge_dst: {}'.format(old_edge_dst))
    #         pos_src = torch.where(old_to_new_nodes_mapping[:, 0] == old_edge_src)
    #         new_edge_src = old_to_new_nodes_mapping[pos_src, 1]
    #         # print('new_edge_src: {}'.format(new_edge_src))
    #         pos_dst = torch.where(old_to_new_nodes_mapping[:, 0] == old_edge_dst)
    #         new_edge_dst = old_to_new_nodes_mapping[pos_dst, 1]
    #         # print('new_edge_dst: {}'.format(new_edge_dst))
    #         new_edge = torch.tensor([[new_edge_src], [new_edge_dst]], dtype=torch.int64, device=self.device).squeeze()
    #         new_edges[:, edge_index] = new_edge
    #     # Return the refined batch
    #     return new_nodes, new_edges, new_batch

