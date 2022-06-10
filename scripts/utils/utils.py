import hashlib
import os
import torch
import torch_geometric as tg
import matplotlib.pyplot as plt
import networkx as nx

class Utils():
    def __init__(self):
        pass

    def get_color_map(self, bench='nas101', sub_skip=False):
        if bench == 'nas101':
            color_map = {'In': 'tomato',
                         'Out': 'tomato',
                         'C3': 'forestgreen',
                         'C1': 'forestgreen',
                         'P3': 'lightskyblue'}
        elif bench == 'nats':
            if sub_skip:
                color_map = {'In': 'tomato',
                             'Out': 'tomato',
                             'C3': 'forestgreen',
                             'C1': 'forestgreen',
                             'P3': 'lightskyblue'}
            else:
                color_map = {'In': 'tomato',
                             'Out': 'tomato',
                             'C3': 'forestgreen',
                             'C1': 'forestgreen',
                             'P3': 'lightskyblue',
                             'S': 'darkgray'}
        else:
            raise ValueError('The benchmark {} doesn\'t have a color map yet!'.format(bench))
        return color_map

    def get_class_name_dict(self, bench='nas101', sub_skip=False):
        if bench == 'nas101':
            class_dict = {0: 'In',
                          1: 'C1',
                          2: 'C3',
                          3: 'P3',
                          4: 'Out'}
        elif bench == 'nats':
            if sub_skip:
                class_dict = {0: 'In',
                              1: 'C1',
                              2: 'C3',
                              3: 'P3',
                              4: 'Out'}
            else:
                class_dict = {0: 'In',
                              1: 'C1',
                              2: 'C3',
                              3: 'P3',
                              4: 'S',
                              5: 'Out'}
        else:
            raise ValueError('The benchmark {} doesn\'t have a color map yet!'.format(bench))
        return class_dict

    def get_name_class_dict(self, bench='nas101', sub_skip=False):
        if bench == 'nas101':
            class_dict = {'In': 0,
                          'C1': 1,
                          'C3': 2,
                          'P3': 3,
                          'Out': 4}
        elif bench == 'nats':
            if sub_skip:
                class_dict = {'In': 0,
                              'C1': 1,
                              'C3': 2,
                              'P3': 3,
                              4: 'Out'}
            else:
                class_dict = {'In': 0,
                              'C1': 1,
                              'C3': 2,
                              'P3': 3,
                              'S': 4,
                              'Out': 5}
        else:
            raise ValueError('The benchmark {} doesn\'t have a color map yet!'.format(bench))
        return class_dict

    def get_skip_conn_index(self, bench='nas101'):
        if bench == 'nas101':
            raise ValueError('The NAS101 dataset doesn\'t have a skip connection operation!')
        elif bench == 'nats':
            return 4
        else:
            raise ValueError('The benchmark {} doesn\'t have a color map yet!'.format(bench))

    def plot_tg_data(self, data, bench='nas101', sub_skip=False):
        figure = plt.figure()
        figure = self.get_plot_data(figure, data, bench=bench, sub_skip=sub_skip)
        plt.show()

    # def get_plot_data(self, figure, data, bench='nas101', sub_skip=False):
    #     # Plot example if plot is required
    #     nx_graph = tg.utils.convert.to_networkx(data)
    #     # if nx.is_directed_acyclic_graph(nx_graph):
    #     #     print('The graph is a DAG!')
    #     # else:
    #     #     print('The graph is NOT a DAG!')
    #     x = data['x']
    #     nodes_operations = torch.argmax(x, dim=-1).numpy()
    #     nodes_ops = [self.get_class_name_dict(bench=bench, sub_skip=sub_skip)[node_op] for node_op in nodes_operations]
    #     nodes_labels = {node: nodes_ops[node] for node in nx_graph.nodes}
    #     color_map = self.get_color_map(bench=bench, sub_skip=sub_skip)
    #     nodes_colors_dict = {node: color_map[nodes_ops[node]] for node in nx_graph.nodes}
    #     nodes_colors = list(nodes_colors_dict.values())
    #     nx.draw(nx_graph, labels=nodes_labels, node_color=nodes_colors, with_labels=True, node_size=450,
    #             edge_color='darkgray', arrowsize=35, arrowstyle='simple')
    #
    #     return figure

    def plot_tg_data_to_store(self, data, store_path, bench='nas101', sub_skip=False, title=None):
        plt.rcParams["figure.figsize"] = (12, 7)
        figure = plt.figure()
        if title:
            plt.title(title)
        figure = self.get_plot_data(figure, data, bench=bench, sub_skip=sub_skip)
        folder = os.path.join(*store_path.split('/')[:-1])
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(store_path)
        plt.close()

    def get_plot_data(self, figure, data, bench='nas101', sub_skip=False):
        # Plot example if plot is required
        nx_graph = tg.utils.convert.to_networkx(data)
        # if nx.is_directed_acyclic_graph(nx_graph):
        #     print('The graph is a DAG!')
        # else:
        #     print('The graph is NOT a DAG!')
        x = data['x']
        nodes_operations = torch.argmax(x, dim=-1).numpy()
        nodes_ops = [self.get_class_name_dict(bench=bench, sub_skip=sub_skip)[node_op] for node_op in nodes_operations]
        nodes_labels = {node: nodes_ops[node] for node in nx_graph.nodes}
        color_map = self.get_color_map(bench=bench, sub_skip=sub_skip)
        nodes_colors_dict = {node: color_map[nodes_ops[node]] for node in nx_graph.nodes}
        nodes_colors = list(nodes_colors_dict.values())


        nodes_pos = {node: [index, 0] for index, node in enumerate(nx_graph.nodes)}
        nx.draw_networkx_nodes(nx_graph, nodes_pos, node_color=nodes_colors, node_shape='8', node_size=750, alpha=1)
        nx.draw_networkx_labels(nx_graph, nodes_pos, nodes_labels)
        nx.draw_networkx_edges(nx_graph, nodes_pos, arrowstyle="-|>", arrowsize=20, alpha=0.5,
                               connectionstyle="arc3,rad=-0.3")
        plt.axis('off')

        return figure


def convert_edges_to_adj(nodes, edges):
    n_nodes = int(nodes.shape[0])
    adj_mat = torch.zeros([n_nodes, n_nodes], dtype=torch.int64)
    edges_list = list(zip(edges[0], edges[1]))
    for index, edge in enumerate(edges_list):
        adj_mat[edge[0], edge[1]] = 1
    return adj_mat

def convert_adj_to_edges(adj):
    # Recompute edges from adjacency matrix
    edges_indices = (adj == 1).nonzero(as_tuple=True)
    edges_indices = list(zip(edges_indices[0].cpu().numpy(), edges_indices[1].cpu().numpy()))
    edges_indices = torch.tensor(edges_indices)
    if edges_indices.nelement() == 0:
        return torch.zeros((2, 1), dtype=torch.int64)
    edges_indices = edges_indices.permute(-1, 0)
    return edges_indices

def hash_model(nodes, edges):
    hasher = hashlib.sha256()
    hasher.update(nodes.detach().cpu().numpy().tobytes())
    hasher.update(edges.detach().cpu().numpy().tobytes())
    model_hash = hasher.digest()
    return model_hash

def hash_model_from_tg(graph):
    nodes, edges = graph.x, graph.edge_index
    adj = convert_edges_to_adj(nodes, edges)
    return hash_model(nodes, adj)