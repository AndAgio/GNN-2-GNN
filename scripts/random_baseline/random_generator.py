import os
import random
import numpy as np
import torch
import torch.nn.functional as t_func
import torch_geometric as tg
from torch_geometric.loader import DataLoader
# Import my modules
from .refiner import Refiner
from scripts.utils import Utils


class RandomGenerator():
    def __init__(self, n_nodes, dataset,
                 sub_skip=False,
                 cio=True,
                 random_model='erdos',
                 probability=0.5,
                 out_path='outputs'):
        self.n_nodes = n_nodes
        self.random_model = random_model
        self.probability = probability
        assert dataset in ['nas101', 'nats']
        self.dataset = dataset
        self.sub_skip = sub_skip
        self.cio = cio
        # Setup operations depending on the dataset
        self.operations = Utils().get_class_name_dict(bench=self.dataset, sub_skip=self.sub_skip)
        print('self.operations: {}'.format(self.operations))
        self.out_path = os.path.join(out_path, 'randomly_generated_{}'.format(self.random_model))

    def generate_graph(self, refine=False, plot=False):
        # Pick operations keys
        operations_keys = list(self.operations.keys())
        # Randomly define operations at nodes and set input and output nodes
        available_operations_min = 0 if self.cio else 1
        available_operations_max = len(operations_keys) - 1 if self.cio else len(operations_keys) - 2
        nodes_ops_int = [operations_keys[random.randint(available_operations_min, available_operations_max)] for _ in range(self.n_nodes - 2)]
        nodes_ops_int.append(operations_keys[-1])
        nodes_ops_int.insert(0, operations_keys[0])
        nodes_ops_str = [self.operations[node] for node in nodes_ops_int]
        # Get nodes to torch geometric form
        x = torch.tensor(nodes_ops_int, dtype=torch.float)
        x = t_func.one_hot(x.squeeze().to(torch.int64), num_classes=len(operations_keys)).float()
        # Debugging
        # print('nodes_ops_int: {}'.format(nodes_ops_int))
        # print('nodes_ops_str: {}'.format(nodes_ops_str))
        # Define fully connected DAG edges
        adj_mat = np.triu(np.ones((self.n_nodes, self.n_nodes), dtype=np.int64))
        # print('adj_mat: {}'.format(adj_mat))
        # Get indices of nodes having links
        links_indices = np.where(adj_mat == 1)
        links_indices = list(zip(links_indices[0], links_indices[1]))
        # Debugging
        # print('links_indices: {}'.format(links_indices))
        # Iterate over all links indices
        for link_index in links_indices:
            # print('link_index: {}'.format(link_index))
            # Edge will survive with probability self.probability
            survival = random.random() < self.probability
            if not survival or link_index[0] == link_index[1]:
                adj_mat[link_index[0], link_index[1]] = 0
        # print('adj_mat: {}'.format(adj_mat))
        # Get back new link indices
        links_indices = np.where(adj_mat == 1)
        # print('links_indices after np.where: {}'.format(links_indices))
        links_indices = list(zip(links_indices[0], links_indices[1]))
        # Convert links into COO matrix for torch geometric
        edge_index = torch.tensor(links_indices, dtype=torch.long)
        # Get data format
        data = tg.data.Data(x=x,
                            edge_index=edge_index.t().contiguous())

        if refine:
            # Refine generated graphs
            # Get back new link indices
            links_indices = np.where(adj_mat == 1)
            # Convert links into COO matrix for torch geometric
            edge_index = torch.tensor(links_indices, dtype=torch.long)
            x, edge_index = Refiner(dataset=self.dataset,
                                    sub_skip=self.sub_skip).refine_single_graph(
                node_indices=[i for i in range(self.n_nodes)],
                node_embeddings=x,
                edges=edge_index)
            # print('new_node_embeddings: {}'.format(new_node_embeddings))
            # print('new_edges: {}'.format(new_edges))
            data = tg.data.Data(x=x,
                                edge_index=edge_index.contiguous())

        if plot:
            # Convert the generated tg data into nx graph and plot it if necessary
            Utils().plot_tg_data(data, bench=self.dataset, sub_skip=self.sub_skip)

        return data, DataLoader([data])

    def generate_graphs(self, n_graphs=10, refine=True):
        # Generating random DAGs...
        graphs = []
        for index in range(n_graphs):
            single_graph = self.generate_graph(refine=refine)
            graphs.append(single_graph)
        return graphs, DataLoader(graphs)

    def get_single_graph(self, refine=False, plot=False):
        return self.generate_graph(refine=refine, plot=plot)

    def run(self, metrics, n_graphs=1000, refine=False):
        # Define running scores
        running_scores = {met_name: 0.0 if met_name != 'acc_vs_foot' else [] for met_name in metrics.keys()}
        avg_scores = {met_name: 0.0 for met_name in metrics.keys() if met_name != 'acc_vs_foot'}
        # Iterate over the number of total graphs to generate
        for index in range(n_graphs):
            # Generate single graph
            _, gen_graph_loader = self.generate_graph(refine=refine, plot=False)
            graph = iter(gen_graph_loader).next()
            # Compute score over single graph and add it to running scores
            for metric_name, metric_object in metrics.items():
                running_scores[metric_name] += metric_object.compute(graph)
                if metric_name != 'acc_vs_foot':
                    avg_scores[metric_name] = running_scores[metric_name] / float(index + 1)
            self.print_message(index=index,
                               max=n_graphs,
                               scores=avg_scores)
        print('')
        # Divide scores sums by number of graphs generated
        for metric_name, metric_object in metrics.items():
            if metric_name != 'acc_vs_foot':
                running_scores[metric_name] = running_scores[metric_name] / float(n_graphs) * 100
        # Print metrics
        print('Final Metrics:')
        # print('running_scores[\'acc_vs_foot\']: {}'.format(running_scores['acc_vs_foot']))
        for metric_name, metric_object in metrics.items():
            if metric_name != 'acc_vs_foot':
                print('{} score: {:.3f}%'.format(metric_name, running_scores[metric_name]))
        # Return metrics for scores
        return running_scores

    def print_message(self, index, max, scores):
        bar_length = 20
        progress = float(index) / float(max)
        if progress >= 1.:
            progress = 1
        block = int(round(bar_length * progress))
        message = '[{}]'.format('=' * block + ' ' * (bar_length - block))
        if scores is not None:
            metrics_message = ' | '
            for metric_name, metric_value in scores.items():
                metrics_message += '{}={:.3f}% '.format(metric_name, metric_value * 100)
            message += metrics_message
        message += '|'
        print(message, end='\r')