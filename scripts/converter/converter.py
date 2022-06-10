# From the tensorflow dataset covert it to a torch_geometric dataset
from progress.bar import IncrementalBar
import os
import random
import pickle
import torch
import torch.nn.functional as t_func
import torch_geometric as tg
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
# Import my modules
from .extractor import Extractor
from .storer import Storer
from scripts.utils import Utils


class Converter():
    def __init__(self,
                 bench_folder='nas_benchmark_datasets/NAS101',
                 dataset_name='nasbench_full',
                 tg_dataset_folder='gnn2gnn_datasets/NAS101',
                 epochs=108):
        self.bench_folder = bench_folder
        self.dataset_name = dataset_name
        self.tg_dataset_folder = tg_dataset_folder
        self.epochs = epochs
        # Define constants
        self.define_constants()
        # Setup storer class
        self.storer = Storer(tg_dataset_folder)

    def define_constants(self):
        # Useful constants
        self.input = 'input'
        self.output = 'output'
        self.conv3 = 'conv3x3-bn-relu'
        self.conv1 = 'conv1x1-bn-relu'
        self.maxpool3 = 'maxpool3x3'
        self.nodes_emb_dict = {self.input: 0, self.conv1: 1, self.conv3: 2, self.maxpool3: 3, self.output: 4}
        self.nodes_emb_dict_inv = {0: self.input, 1: self.conv1, 2: self.conv3, 3: self.maxpool3, 4: self.output}
        self.n_vertices = 7
        self.max_edges = 9
        self.edge_spots = self.n_vertices * (self.n_vertices - 1) / 2  # Upper triangular matrix
        self.op_spots = self.n_vertices - 2  # Input/output vertices are fixed
        self.allowed_ops = [self.conv3, self.conv1, self.maxpool3]
        self.allowed_edges = [0, 1]  # Binary adjacency matrix
        # # Get iterator over hashes to query models directly from hashes
        # self.hash_keys = list(self.api.hash_iterator())
        # self.n_models = len(self.hash_keys)
        # self.hash_iterator = iter(self.hash_keys)
        # print('Number of models in NAS101: {}'.format(len(self.hash_keys)))
        # print('Hash iterator is: {}'.format(self.hash_iterator))

    def api_from_extractor(self):
        # Run extractor setup
        extractor = Extractor(bench_folder=self.bench_folder,
                              dataset_name=self.dataset_name)
        # Get api from extractor
        self.api = extractor.get_api()

    def check_api(self):
        try:
            self.api
        except AttributeError:
            self.api_from_extractor()

    def run(self):
        # Define list of epochs to convert
        epochs_list = [4, 12, 36, 108]
        # Iterate over each epoch
        for index, epochs in enumerate(epochs_list):
            print('[{}/{}] Extracting dataset for NNs trained over {} epochs...'.format(index+1,
                                                                                        len(epochs_list),
                                                                                        epochs))
            # Query models and store dictionary containing all information in the tfrecord file
            print('Converting tf_record to list of dictionaries...')
            datas_dict_form = self.convert_tfrecord_to_dictionary(epochs)
            # Convert models in dictionary to torch-geometric file and store it
            print('Converting list of dictionaries into torch geometric raw data...')
            self.convert_dictionary_to_tg_raw_and_store_it(datas_dict_form, epochs)

    def convert_tfrecord_to_dictionary(self, epochs):
        # Check if pickled ordered dictionary exists
        pickled_ordered_dictionary_name = '{}_{}_ordered_dict.pkl'.format(self.dataset_name,
                                                                          epochs)
        pickled_ordered_dictionary_path = os.path.join(self.bench_folder, pickled_ordered_dictionary_name)
        if os.path.exists(pickled_ordered_dictionary_path):
            print('Loading pickled ordered dictionary...')
            datas = pickle.load(open(pickled_ordered_dictionary_path, 'rb'))
            print('Loaded!')
        else:
            # Check if pickled unordered dictionary exists
            pickled_unordered_dictionary_name = '{}_{}_unordered_dict.pkl'.format(self.dataset_name,
                                                                                  epochs)
            pickled_unordered_dictionary_path = os.path.join(self.bench_folder, pickled_unordered_dictionary_name)
            if os.path.exists(pickled_unordered_dictionary_path):
                print('Loading pickled unordered dictionary...')
                datas = pickle.load(open(pickled_unordered_dictionary_path, 'rb'))
                print('Loaded!')
            else:
                print('Couldn\'t find unordered dictionary!')
                # Iterate over all hashes available in the api...
                datas = []
                self.check_api()
                n_models = len(list(self.api.hash_iterator()))
                bar = IncrementalBar('Querying all hash keys...', max=n_models)
                for hash_key in list(self.api.hash_iterator()):
                    # Convert sample into dictionary and append it to data list
                    datas.append(self.query(hash_key, epochs))
                    bar.next()
                bar.finish()
                print('Saving pickle file containing unordered dictionary for future use...')
                pickle.dump(datas, open(pickled_unordered_dictionary_path, 'wb'))
            # Order the NNs by performance and add top n feature to data dictionary
            # print('Non ordered first and last datas:\n{}\n{}'.format(datas[0], datas[-1]))
            datas = self.order_nns_by_performance(datas)
            # print('Ordered first and last datas:\n{}\n{}'.format(datas[0], datas[-1]))
            print('Saving pickle file containing ordered dictionary for future use...')
            pickle.dump(datas, open(pickled_ordered_dictionary_path, 'wb'))
        # Return variable containing data in dictionary form
        return datas

    def order_nns_by_performance(self, datas):
        # Order dictionary by performance value
        print('Sorting NNs by performance. This may take a while...')
        datas = list(sorted(datas, key=lambda x: x['test_accuracy'], reverse=False))
        # Add new field to the dictionary corresponding to the top percentage of the model
        self.check_api()
        n_models = len(list(self.api.hash_iterator()))
        datas = [dict(item, **{'top': 100 - (index / n_models * 100)}) for index, item in enumerate(datas)]
        # bar = IncrementalBar('Adding top n performance...', max=n_models)
        # for index, _ in enumerate(datas):
        #     datas[index]['top'] = (index + 1) / n_models * 100
        #     bar.next()
        # bar.finish()
        return datas

    def convert_dictionary_to_tg_raw_and_store_it(self, datas, epochs):
        # Iterate over all data in the dictionary
        tg_datas = []
        bar = IncrementalBar('Converting all data in dictionary form into tg.data form...', max=len(datas))
        for data in datas:
            # Transform data in tg_data
            tg_data = self.info_to_tg_data(data)
            # Append it
            tg_datas.append(tg_data)
            bar.next()
        bar.finish()
        # Store it as raw files
        self.storer.store_tg_data_raw(tg_datas,
                                      name='data_epochs_{}'.format(epochs))

    def info_to_tg_data(self, info, one_hot=True, verbose=False, plot=False):
        # Get relevant information from the info dictionary
        adj_mat = info['module_adjacency']
        nodes_ops = info['module_operations']
        # Convert nodes operations into their encoding and into torch tensor
        node_ops_embedding = [[self.nodes_emb_dict[node_op]] for node_op in nodes_ops]
        x = torch.tensor(node_ops_embedding, dtype=torch.float)
        if one_hot:
            n_ops = len(list(self.nodes_emb_dict.keys()))
            x = t_func.one_hot(x.squeeze().to(torch.int64), num_classes=n_ops).float()
        # Get indeces of nodes having links
        links_indeces = np.where(adj_mat == 1)
        links_indeces = list(zip(links_indeces[0], links_indeces[1]))
        # Convert links into COO matrix for torch geometric
        edge_index = torch.tensor(links_indeces, dtype=torch.long)
        # Get data format
        data = tg.data.Data(x=x,
                            edge_index=edge_index.t().contiguous(),
                            trainable_parameters=info['trainable_parameters'],
                            footprint=info['trainable_parameters'],
                            training_time=info['training_time'],
                            train_accuracy=info['train_accuracy'],
                            validation_accuracy=info['validation_accuracy'],
                            test_accuracy=info['test_accuracy'],
                            top=info['top'],
                            y_class=1 if info['test_accuracy'] > 0.5 else 0)
        # Print everything if verbose is True
        if verbose:
            print('Operations embeddings: {}'.format(node_ops_embedding))
            print('Links indices: {}'.format(links_indeces))
            print('Edge index: {}'.format(edge_index))
            print('TG data is: {}'.format(data))
        # Plot example if plot is required
        if plot:
            nx_graph = tg.utils.convert.to_networkx(data)
            nodes_labels = {node: nodes_ops[node] for node in nx_graph.nodes}
            color_map = Utils().get_color_map(bench='101')
            nodes_colors_dict = {node: color_map[nodes_ops[node]] for node in nx_graph.nodes}
            nodes_colors = list(nodes_colors_dict.values())
            plt.figure()
            nx.draw(nx_graph, labels=nodes_labels, node_color=nodes_colors, with_labels=True, node_size=450,
                    edge_color='gray', arrowsize=35, arrowstyle='simple')
            plt.show()
        return data

    def query(self, hash_key, epochs):
        self.check_api()
        fixed_stat, computed_stat = self.api.get_metrics_from_hash(hash_key)
        sampled_index = 0  # random.randint(0, 2)
        computed_stat = computed_stat[epochs][sampled_index]
        data = {'module_adjacency': fixed_stat['module_adjacency'],
                'module_operations': fixed_stat['module_operations'],
                'trainable_parameters': fixed_stat['trainable_parameters'],
                'training_time': computed_stat['final_training_time'],
                'train_accuracy': computed_stat['final_train_accuracy'],
                'validation_accuracy': computed_stat['final_validation_accuracy'],
                'test_accuracy': computed_stat['final_test_accuracy'], }
        return data
