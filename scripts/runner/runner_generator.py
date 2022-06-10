import os
import random
import numpy as np
import torch
import torch.nn.functional as t_func
import torch_geometric as tg
from torch_geometric.loader import DataLoader
# Import my modules
from scripts.utils import Utils, hash_model
from scripts.data import SyntheticDataset
from scripts.models import GeneratorNet, GeneratorNetV2, GeneratorNetV3
from scripts.utils.utils import hash_model_from_tg


class GeneratorRunner():
    def __init__(self, generator, n_nodes, dataset,
                 generator_name='nas101_to_nas101',
                 sub_skip=False,
                 out_path='outputs'):
        self.generator = generator
        self.generator.eval()
        self.n_nodes = n_nodes
        assert dataset in ['nas101', 'nats']
        self.dataset = dataset
        self.sub_skip = sub_skip
        # Setup operations depending on the dataset
        self.operations = Utils().get_class_name_dict(bench=self.dataset, sub_skip=self.sub_skip)
        print('self.operations: {}'.format(self.operations))
        self.out_path = os.path.join(out_path, 'gnn2gnn_generated', generator_name)
        # Setup synthetic dataset for running generator model
        if isinstance(self.generator, GeneratorNet):
            self.graph_gen_utils = SyntheticDataset(n_nodes=n_nodes,
                                                    hidden_dim=generator.hidden_dim)

    def generate_graph(self):
        # Depending on the generator available, produce a graph
        if isinstance(self.generator, GeneratorNet):
            # Pick random graph and pass them through the generator
            self.graph_gen_utils.process(batch_size=1)
            random_fc_graph = iter(self.graph_gen_utils.get_data_loader(batch_size=1)).next()
            gen_graph = self.generator(random_fc_graph)
        elif isinstance(self.generator, GeneratorNetV2):
            z = np.random.normal(0, 1, size=(1, 256))
            z = torch.from_numpy(z).float()
            gen_graph = self.generator(z)
        elif isinstance(self.generator, GeneratorNetV3):
            z = np.random.normal(0, 1, size=(1, 256))
            z = torch.from_numpy(z).float()
            gen_graph = self.generator(z)
        return gen_graph, DataLoader([gen_graph])

    def generate_graphs(self, n_graphs=10):
        # Generating DAGs...
        graphs = []
        for index in range(n_graphs):
            single_graph, _ = self.generate_graph()
            graphs.append(single_graph)
        return graphs, DataLoader(graphs)

    def get_single_graph(self):
        return self.generate_graph()

    def run(self, metrics, n_graphs=10, plot=False):
        # Define running scores
        running_scores = {met_name: 0.0 if met_name != 'acc_vs_foot' else [] for met_name in metrics.keys()}
        avg_scores = {met_name: 0.0 for met_name in metrics.keys() if met_name != 'acc_vs_foot'}
        last_scores = {met_name: 0.0 for met_name in metrics.keys() if met_name != 'acc_vs_foot'}
        hashes_list = []
        # Iterate over the number of total graphs to generate
        for index in range(n_graphs):
            # Generate single graph
            _, gen_graph_loader = self.generate_graph()
            graph = iter(gen_graph_loader).next()
            hashes_list.append(hash_model_from_tg(graph))
            # Compute score over single graph and add it to running scores
            for metric_name, metric_object in metrics.items():
                score = metric_object.compute(graph)
                last_scores[metric_name] = score
                running_scores[metric_name] += metric_object.compute(graph)
                if metric_name != 'acc_vs_foot':
                    avg_scores[metric_name] = running_scores[metric_name] / float(index + 1)
            self.print_message(index=index,
                               max=n_graphs,
                               scores=avg_scores)
            # If plotting is needed, plot and store the generated graph
            if plot:
                title = define_title_from_metrics(validity=last_scores['val'],
                                                  top10ty=last_scores['top_10'])
                # Convert the generated tg data into nx graph and plot it if necessary
                store_path = os.path.join(self.out_path, '{}.pdf'.format(index))
                Utils().plot_tg_data_to_store(graph,
                                              store_path=store_path,
                                              title=title,
                                              bench=self.dataset,
                                              sub_skip=self.sub_skip)
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
        return running_scores, hashes_list

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


def define_title_from_metrics(validity, top10ty):
    # Check validity of the graph
    is_valid = True if validity == 1. else False
    title = 'Valid' if is_valid else 'NOT valid'
    # Get accuracy of the generated graph
    if is_valid:
        # Check if the model is in top 10 percentage of best models
        percentage = 10
        is_performing = True if top10ty == 1. else False
        if is_performing:
            title += ' & TOP {}% of NNs'.format(percentage)
    return title
