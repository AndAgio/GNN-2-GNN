import torch
from scripts.utils import Utils

class Refiner():
    def __init__(self, dataset, sub_skip=False):
        self.dataset = dataset
        self.sub_skip = sub_skip

    def refine_single_graph(self, node_indices, node_embeddings, edges):
        # Method to refine a single graph
        nodes_embedding_mapping = {node_indices[i]: node_embeddings[i, :].unsqueeze(dim=0) for i in
                                   range(len(node_indices))}
        # print('nodes_embedding_mapping: {}'.format(nodes_embedding_mapping))
        refined, indices_of_nodes_to_drop = self.check_graph_is_refined(node_indices, edges)
        while not refined:
            node_indices, edges = self.refine_step(node_indices, edges, indices_of_nodes_to_drop)
            refined, indices_of_nodes_to_drop = self.check_graph_is_refined(node_indices, edges)
        # print('node_indices before removing skips: {}'.format(node_indices))
        # print('edges before removing skips: {}'.format(edges))
        # # Remove skip connections if dealing with NATS dataset.
        # if self.dataset == 'nats' and self.sub_skip:
        #     node_indices, edges = self.remove_skips(node_indices, nodes_embedding_mapping, edges)
        # print('node_indices after removing skips: {}'.format(node_indices))
        # print('edges after removing skips: {}'.format(edges))
        # Compute back node embeddings from stored dictionary
        new_nodes_embedding = [nodes_embedding_mapping[node_index] for node_index in node_indices]
        # print('new_nodes_embedding: {}'.format(new_nodes_embedding))
        new_nodes_embedding = torch.cat(new_nodes_embedding, dim=0)
        # print('new_nodes_embedding: {}'.format(new_nodes_embedding))
        # print('new edges: {}'.format(edges))

        # Refine edges indices since node 55 may be mapped to node 34, 23, 47, or whatever, depending on the number of dropped nodes
        new_edges_src = list(edges[0, :].numpy())
        new_edges_dst = list(edges[1, :].numpy())
        # Map old nodes to new nodes
        nodes_indices_in_new_edges = list(set(new_edges_src)) + list(set(new_edges_dst) - set(new_edges_src))
        nodes_indices_in_new_edges.sort()
        old_to_new_nodes_mapping = {nodes_indices_in_new_edges[i]: i for i in range(len(nodes_indices_in_new_edges))}
        # Map old edges to new edges
        n_columns = list(edges.size())[1]
        for edge_index in range(n_columns):
            old_edge = edges[:, edge_index].numpy()
            old_edge_src = old_edge[0]
            old_edge_dst = old_edge[1]
            new_edge_src = old_to_new_nodes_mapping[old_edge_src]
            new_edge_dst = old_to_new_nodes_mapping[old_edge_dst]
            new_edge = torch.tensor([[new_edge_src], [new_edge_dst]], dtype=torch.int64).squeeze()
            edges[:, edge_index] = new_edge

        return new_nodes_embedding, edges

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
            kept_edges.append(torch.tensor([[kept_nodes_indices[0]], [kept_nodes_indices[-1]]], dtype=torch.int64))
        # print('kept_edge: {}'.format(kept_edges))
        kept_edges = torch.cat(kept_edges, dim=-1)
        # if kept_edges == []:
        #     kept_edges.append(torch.tensor([[kept_nodes_indices[0]], [kept_nodes_indices[-1]]], dtype=torch.int64))
        # kept_edges = torch.cat(kept_edges, dim=-1)
        # print('kept_edges after refine step: {}'.format(kept_edges))
        return kept_nodes_indices, kept_edges

    def remove_skips(self, nodes_indices, nodes_embedding_mapping, edges):
        # Check if skip connections need to be removed or not
        skip_conn_index = Utils().get_skip_conn_index(bench='nats')
        # print('name_class_dict: {}'.format(name_class_dict))
        # Identify nodes corresponding to skip connections
        skip_indices = []
        for node_index in nodes_indices:
            operation = nodes_embedding_mapping[node_index]
            print('operation: {}'.format(operation))
            print('torch.argmax(operation, dim=1).numpy()[0]: {}'.format(torch.argmax(operation, dim=1).numpy()[0]))
            if torch.argmax(operation, dim=1).numpy()[0] == skip_conn_index:
                skip_indices.append(node_index)
        print('skip_indices: {}'.format(skip_indices))
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

    def remove_self_loops(self, edges):
        # Method to remove self loops from a graph
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
