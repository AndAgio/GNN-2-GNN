import numpy as np
import torch
import sklearn.manifold as sk_man
import matplotlib.pyplot as plt


class TSNE():
    def __init__(self, dataset, chosen_dataset='nas101', nats_data='cifar10'):
        assert dataset is not None
        self.dataset = dataset
        assert chosen_dataset in ['nas101', 'nats']
        self.chosen_dataset = chosen_dataset
        self.nats_data = nats_data
        # Get all models from the dataset
        self.all_models = self.dataset.get_all_valid_models()
        # Get max number of nodes and encoding dimension from dataset
        self.n_nodes = self.dataset.get_num_nodes()
        self.encodings_dimension = self.dataset.num_features

    def run(self):
        encodings, top_list = self.convert_all_models()
        tsne_points = sk_man.TSNE(n_components=2, learning_rate='auto',
                                  init='random').fit_transform(encodings)
        self.plot_tsne(tsne_points, top_list)

    def convert_all_models(self):
        # Define empty list of encodings for generated graphs
        encodings_list = []
        top_list = []
        for model_in_dataset in self.all_models:
            x = model_in_dataset['x']
            edges = model_in_dataset['edge_index']
            top = model_in_dataset['top']
            top_list.append(top)
            # Convert edges to adjacency matrix
            adj_mat = np.zeros((self.n_nodes, self.n_nodes), dtype=np.int64)
            edges = list(zip(edges[0], edges[1]))
            for edge in edges:
                adj_mat[edge[0], edge[1]] = 1
            # Expand x and edges if necessary
            if x.shape[0] != self.n_nodes:
                # Fill up x
                n_missing_nodes = self.n_nodes - x.shape[0]
                x = torch.cat([x, torch.zeros((n_missing_nodes, self.encodings_dimension))], dim=0)
                # # Fill up edges
                # edges = torch.cat([edges, torch.zeros((n_missing_nodes, x.shape[1]))], dim=0)
                # edges = torch.cat([edges, torch.zeros((x.shape[0], n_missing_nodes))], dim=1)
            # Convert encoding in mono dimensional vector
            encoding = x.numpy().flatten()
            encoding = np.concatenate((encoding, adj_mat.flatten()))
            encodings_list.append([encoding])
        # Convert encodings list into encoding matrix
        encodings = np.squeeze(np.array(encodings_list))
        print('encodings shape: {}'.format(encodings.shape))
        return encodings, top_list

    def plot_tsne(self, tsne_points, top_list):
        # Find index of top models
        top_10_indices = [i for i, v in enumerate(top_list) if v < 10]
        top_10_20_indices = [i for i, v in enumerate(top_list) if 10 <= v < 20]
        top_20_50_indices = [i for i, v in enumerate(top_list) if 20 <= v < 50]
        top_50_100_indices = [i for i, v in enumerate(top_list) if 50 <= v < 100]
        # Group tsne points depending on top value
        top_10_tsne_point = tsne_points[top_10_indices]
        top_10_20_tsne_point = tsne_points[top_10_20_indices]
        top_20_50_tsne_point = tsne_points[top_20_50_indices]
        top_50_100_tsne_point = tsne_points[top_50_100_indices]
        groups_tsne_points = [top_10_tsne_point, top_10_20_tsne_point, top_20_50_tsne_point, top_50_100_tsne_point]
        # Scatter plot groups
        fig = plt.figure()
        axis = fig.add_subplot(111)
        labels = ['top 10%', 'top 20%', 'top 50%', 'all']
        colors = ['r', 'y', 'g', 'b']
        for i in range(4):
            points = groups_tsne_points[i]
            label = labels[i]
            axis.scatter(points[:, 0], points[:, 1], s=10, c=colors[i], label=label.upper(), zorder=4-i)
        plt.legend(loc='lower right')
        if self.chosen_dataset == 'nats':
            if 'cifar' in self.nats_data:
                nats_data_string = self.nats_data.upper()
            else:
                nats_data_string = self.nats_data
            title = '{} benchmark trained on {}'.format(self.chosen_dataset.upper(), nats_data_string)
        else:
            title = '{} benchmark trained on CIFAR10'.format(self.chosen_dataset.upper())
        plt.title(title)
        plt.show()


