import os
import torch
import torch_geometric as tg
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import shutil
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
# Import my modules


class SyntheticDataset(InMemoryDataset):
    def __init__(self, n_nodes, hidden_dim, root='datasets/synthetic', transform=None,
                 pre_transform=None, sh=True):
        self.n_nodes = n_nodes
        print('Synthethic dataset n_nodes: {}'.format(self.n_nodes))
        self.hidden_dim = hidden_dim
        root = os.path.join(os.getcwd(), root)
        raw_path = os.path.join(root, 'raw')
        processed_path = os.path.join(root, 'processed')
        if sh and os.path.exists(raw_path) and os.path.isdir(raw_path):
            print('Deleting previous raw data folder...')
            shutil.rmtree(raw_path)
        if sh and os.path.exists(processed_path) and os.path.isdir(processed_path):
            print('Deleting previous processed data folder...')
            shutil.rmtree(processed_path)
        super(SyntheticDataset, self).__init__(root, transform, pre_transform)

        path = self.processed_paths[0]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['generated.pt']

    @property
    def processed_file_names(self):
        return ['generated.pt']

    def download(self):
        pass

    def generate_fc_graph(self, plot=False):
        # Generate random nodes
        x = torch.rand((self.n_nodes, self.hidden_dim), dtype=torch.float)  # torch.normal(0, 1, size=(self.n_nodes, self.hidden_dim), dtype=torch.float)
        # Connect all nodes in a DAG
        adj_mat = np.triu(np.ones((self.n_nodes, self.n_nodes), dtype=np.int64))
        # Remove self loops
        for i in range(self.n_nodes):
            adj_mat[i, i] = 0
        # Get indices of nodes having links
        links_indices = np.where(adj_mat == 1)
        links_indices = list(zip(links_indices[0], links_indices[1]))
        # Convert links into COO matrix for torch geometric
        edge_index = torch.tensor(links_indices, dtype=torch.long)
        # Get data format
        data = tg.data.Data(x=x,
                            edge_index=edge_index.t().contiguous())
        # Convert the generated tg data into nx graph and plot it if necessary
        if plot:
            nx_graph = tg.utils.convert.to_networkx(data)
            nodes_pos = {node: [index, 0] for index, node in enumerate(nx_graph.nodes)}
            plt.figure()
            nx.draw_networkx_nodes(nx_graph, nodes_pos, alpha=1)
            nx.draw_networkx_labels(nx_graph, nodes_pos)
            nx.draw_networkx_edges(nx_graph, nodes_pos, arrowstyle="-|>", arrowsize=20, alpha=0.5,
                                   connectionstyle="arc3,rad=-0.3")
            plt.axis('off')
            plt.show()
            # plt.figure()
            # nx.draw(nx_graph, node_size=450, with_labels=True,
            #         edge_color='gray', arrowsize=35, arrowstyle='simple')
            # plt.show()
        return data

    def generate_fc_graphs(self, batch_size=32):
        # print('Generating random graphs...')
        graphs = []
        for index in range(batch_size):
            # print('index: {}'.format(index))
            single_graph = self.generate_fc_graph()
            graphs.append(single_graph)
        # dataset = SyntheticDataset(graphs)
        # data_loader = dataset.get_data_loader(batch_size=batch_size)
        return graphs  # data_loader

    def process(self, data_list=None, batch_size=32):
        # Read data from `Data` list and collate them.
        index = 0
        if data_list is not None:
            data, slices = self.collate(self.data_list)
        else:
            # print('Generating random graphs...')
            # print('batch_size in process: {}'.format(batch_size))
            # print('self.generate_fc_graphs(batch_size=batch_size): {}'.format(len(self.generate_fc_graphs(batch_size=batch_size))))
            data, slices = self.collate(self.generate_fc_graphs(batch_size=batch_size))
        torch.save((data, slices), self.processed_paths[index])
        self.data, self.slices = torch.load(self.processed_paths[index])

    def get_data_loader(self, batch_size=32, shuffle=False):
        # print('batch_size: {}'.format(batch_size))
        loader = DataLoader(self, batch_size=batch_size, shuffle=shuffle)
        return loader