import hashlib
import time
import numpy as np
import torch
import torch_geometric as tg
import os
import pickle
from progress.bar import IncrementalBar
# My modules
from scripts.utils import timeit, convert_edges_to_adj, hash_model


class GeneratorMetrics():
    def __init__(self, dataset):
        assert dataset is not None
        self.dataset = dataset
        # Get all models from the dataset
        self.dataset_models = self.dataset.get_all_dataset_models()
        self.dataset_metrics = {}
        # Get all models from the training dataset
        self.train_models = self.dataset.get_train_models()
        self.train_models_hashes = [hash_model(model['x'], convert_edges_to_adj(model['x'], model['edge_index']))
                                          for model in self.train_models]
        self.setup_dataset_metrics()

    @timeit
    def setup_dataset_metrics(self):
        # Check if pickle file containing dataset metrics is available or not
        if self.dataset.name == 'nas101':
            filename = 'dataset_metrics_file_epoch_{}.pk'.format(self.dataset.epoch)
        elif self.dataset.name == 'nats':
            filename = 'dataset_metrics_file_data_{}_setting_{}_sub_skip_{}.pk'.format(self.dataset.chosen_data,
                                                                                       self.dataset.setting,
                                                                                       self.dataset.sub_skip)
        else:
            raise ValueError('Not a valid dataset!')
        dataset_metrics_file = os.path.join(self.dataset.root, filename)
        if os.path.exists(dataset_metrics_file):
            print('Loading file containing dataset metrics...')
            self.dataset_metrics = pickle.load(open(dataset_metrics_file, 'rb'))
            print('Loaded!')
        else:
            print('Couldn\'t find file containing dataset metrics!')
            bar = IncrementalBar('Getting the metrics of models in dataset. This may take a while...',
                                 max=len(self.dataset_models))
            for model in self.dataset_models:
                x = model['x']
                adj_mat = convert_edges_to_adj(x, model['edge_index'])
                # print('x: {}'.format(x))
                # print('model[\'edge_index\']: {}'.format(model['edge_index']))
                # print('adj_mat: {}'.format(adj_mat))
                model_hash = hash_model(x, adj_mat)
                # print('model_hash: {}'.format(model_hash))
                dictionary = {'val': 1,
                              'nov': 0 if model_hash in self.train_models_hashes else 1,
                              'top': model['top'],
                              'test_accuracy': model['test_accuracy'],
                              'footprint': model['footprint'],
                              }
                self.dataset_metrics[model_hash] = dictionary
                bar.next()
            bar.finish()
            # Store dataset metrics into pickle file for later use
            print('Saving pickle file containing dataset metrics for future use...')
            pickle.dump(self.dataset_metrics, open(dataset_metrics_file, 'wb'))


    def get_dataset_metrics(self):
        return self.dataset_metrics

    def get_train_models_hashes(self):
        return self.train_models_hashes

    def print_models_ranks(self):
        for index, model in enumerate(reversed(self.dataset_models)):
            print('Index: {} -> Acc: {:.3f}'.format(index, model['test_accuracy']))