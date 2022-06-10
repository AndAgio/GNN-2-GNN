import os
import sys
import time
import random
import pickle
import torch
import torch.nn.functional as t_func
import itertools
import numpy as np
import networkx as nx
import torch_geometric as tg
import matplotlib.pyplot as plt
from progress.bar import IncrementalBar
from nasbench import api
# Import my modules
from scripts.utils import Utils


class NAS101Bench():
    def __init__(self, pickled_api_file='pickled_api.pkl', bench_folder='nas_benchmark_datasets/NAS101', datasets_folder='datasets/NAS101'):
        self.bench_folder = bench_folder
        # Setup APIs
        pickled_api_file_path = os.path.join(self.bench_folder, pickled_api_file)
        print('Pickled api file: {}'.format(pickled_api_file))
        if os.path.exists(pickled_api_file_path):
            print('Found pickled dataset!')
            bf = time.time()
            self.api = pickle.load(open(pickled_api_file_path, 'rb'))
            af = time.time()
            print('Time taken to load API from pickle: {:5f} s'.format(af-bf))
        else:
            path = os.path.join(self.bench_folder, 'nasbench_only108.tfrecord')
            self.api = api.NASBench(path)
            pickle.dump(self.api, open(pickled_api_file_path, 'wb'))
        # Setup useful constants for the api
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
        self.edge_spots = self.n_vertices * (self.n_vertices - 1) / 2   # Upper triangular matrix
        self.op_spots = self.n_vertices - 2   # Input/output vertices are fixed
        self.allowed_ops = [self.conv3, self.conv1, self.maxpool3]
        self.allowed_edges = [0, 1]   # Binary adjacency matrix
        # Get iterator over hashes to query models directly from hashes
        self.hash_keys = list(self.api.hash_iterator())
        self.n_models = len(self.hash_keys)
        self.hash_iterator = iter(self.hash_keys)
        # print('Number of models in NAS101: {}'.format(len(self.hash_keys)))
        # print('Hash iterator is: {}'.format(self.hash_iterator))
        # Setup output folder
        out_datasets_path = os.path.join(os.getcwd(), datasets_folder)
        if not os.path.exists(out_datasets_path):
            os.makedirs(out_datasets_path)
        self.out_datasets_path = out_datasets_path

    def get_random_model(self):
        rand_hash = self.hash_keys[random.randint(0,self.n_models)]
        #print('Random regressor hash is: {}'.format(rand_hash))
        info = self.query(rand_hash)
        #print(info)
        return info

    def get_iterator_over_hashes(self):
        return self.hash_iterator

    def get_next_model(self):
        return self.query(self.get_next_hash())

    def get_next_hash(self):
        next_hash = next(self.hash_iterator)
        #print('Model hash is: {}'.format(next_hash))
        return next_hash

    def query(self, hash):
        epochs=108
        fixed_stat, computed_stat = self.api.get_metrics_from_hash(hash)
        sampled_index = random.randint(0, 2)
        computed_stat = computed_stat[epochs][sampled_index]
        data = {}
        data['module_adjacency'] = fixed_stat['module_adjacency']
        data['module_operations'] = fixed_stat['module_operations']
        data['trainable_parameters'] = fixed_stat['trainable_parameters']
        data['training_time'] = computed_stat['final_training_time']
        data['train_accuracy'] = computed_stat['final_train_accuracy']
        data['validation_accuracy'] = computed_stat['final_validation_accuracy']
        data['test_accuracy'] = computed_stat['final_test_accuracy']
        return data

    def get_model_from_random_matrix(self):
        index = 1
        while True:
            # print('Iteration number {}'.format(index))
            # Define a random adjacency matrix
            matrix = np.random.choice(self.allowed_edges, size=(self.n_vertices, self.n_vertices))
            matrix = np.triu(matrix, 1)
            # print('Matrix: {}'.format(matrix))
            # Define random set of operations for middle nodes and append input and output ops
            ops = np.random.choice(self.allowed_ops, size=(self.n_vertices)).tolist()
            ops[0] = self.input
            ops[-1] = self.output
            # print('Ops: {}'.format(ops))
            # Define spec for the random regressor
            spec = api.ModelSpec(matrix=matrix, ops=ops)
            # print('spec: {}'.format(spec))
            # Check if the regressor is valid, if so returns it
            if self.api.is_valid(spec):
                info = self.api.query(spec)
                # print('info: {}'.format(info))
                return info
                #break
            index += 1

    def check_validity(self, matrix, ops):
        # Define spec for the given NN architecture
        spec = api.ModelSpec(matrix=matrix, ops=ops)
        # Check if the NN structure is valid
        if self.api.is_valid(spec):
            return True
        else:
            return False

    def get_infos(self, matrix, ops):
        # print('\nChecking availability of api.query(model_spec)...')
        # self.get_model_from_random_matrix()
        # print()
        # print('Matrix: {}'.format(matrix))
        # print('Ops: {}'.format(ops))
        # in_edges = np.sum(matrix, axis=0).tolist()
        # out_edges = np.sum(matrix, axis=1).tolist()
        # labeling = [-1] + [ops.index(op) for op in self.ops[1:-1]] + [-2]
        # print('in_edges: {}'.format(in_edges))
        # print('out_edges: {}'.format(out_edges))
        # print('labeling: {}'.format(labeling))
        # Define spec for the given NN architecture
        spec = api.ModelSpec(matrix=matrix, ops=ops)
        # print('Spec: {}'.format(spec))
        # Check if the NN structure is valid and return its info through API
        if self.api.is_valid(spec):
            info = self.api.query(spec)
            return info

    def info_to_tg_data(self, info, one_hot=True, verbose=False, plot=False):
        # Get relevant informations from the info dictionary
        adj_mat = info['module_adjacency']
        nodes_ops = info['module_operations']
        acc = info['test_accuracy']
        # Convert nodes operations into their encoding and into torch tensor
        node_ops_embedding = [[self.nodes_emb_dict[node_op]] for node_op in nodes_ops]
        x = torch.tensor(node_ops_embedding, dtype=torch.float)
        if one_hot:
            n_ops = len(list(self.nodes_emb_dict.keys()))
            x = t_func.one_hot(x.squeeze().to(torch.int64), num_classes=n_ops).float()
        # If accuracy is greater than 0.5 get class label as 1
        y_class = 1 if acc>0.5 else 0
        # Get indeces of nodes having links
        links_indeces = np.where(adj_mat==1)
        links_indeces = list(zip(links_indeces[0], links_indeces[1]))
        # Convert links into COO matrix for torch geometric
        edge_index = torch.tensor(links_indeces, dtype=torch.long)
        # Get data format
        data = tg.data.Data(x=x,
                            edge_index=edge_index.t().contiguous(),
                            y=acc,
                            y_class=y_class)
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
            nx.draw(nx_graph, labels=nodes_labels, node_color=nodes_colors, with_labels=True, node_size=450, edge_color='gray', arrowsize=35, arrowstyle='simple')
            plt.show()
        return data

    def get_data_list_from_hashes(self, hashes_list):
        # Define empty list of data to be returned
        datas = []
        for hash in hashes_list:
            info = self.query(hash)
            datas.append(self.info_to_tg_data(info, verbose=False, plot=False))
        return datas

    def get_best_models(self, perc=10):
        # Load best models dictionary from pickle file
        file_path = os.path.join(os.getcwd(), self.bench_folder,
                                 'top_{}_models_dict.pkl'.format(perc))
        if os.path.exists(file_path):
            best_models_dict = pickle.load(open(file_path, 'rb'))
            return best_models_dict
        # Otherwise compute best models and store them
        n_best_models = int(float(perc)/float(100)*self.n_models)
        # print('Looking for the best {} models'.format(n_best_models))
        hash_perf_dict = {}
        bar = IncrementalBar('Getting the best {} models (TOP {})...'.format(n_best_models, perc), max=self.n_models)
        for hash in self.hash_iterator:
            acc = self.query(hash)['test_accuracy']
            hash_perf_dict[hash] = acc
            bar.next()
        bar.finish()
        # print('hash_perf_dict: {}'.format(hash_perf_dict))
        hash_perf_ordered_dict = dict(sorted(hash_perf_dict.items(), key=lambda item: item[1]))
        # print('hash_perf_ordered_dict: {}'.format(hash_perf_ordered_dict))
        best_models_dict = list(hash_perf_ordered_dict.items())[-n_best_models:]
        best_models_dict = {item[0]: item[1] for item in best_models_dict}
        # print('best_models_dict: {}'.format(best_models_dict))
        # Store best models dictionary in pickle file
        file_path = os.path.join(os.getcwd(), self.bench_folder, 'top_{}_models_dict.pkl'.format(int(perc)))
        pickle.dump(best_models_dict, open(file_path, 'wb'))
        return best_models_dict

    def define_splits(self, complexity=1):
        my_random = random.Random(12345)
        # Get best models from the dataset
        best_models_dict = self.get_best_models()
        best_models_hashes = list(best_models_dict.keys())
        n_best_models = len(best_models_hashes)
        # Get number of best models that will end up in the test set depending on the complexity parameter
        n_best_models_test = int(complexity*n_best_models)
        n_best_models_train = int(0.75*(n_best_models-n_best_models_test))
        n_best_models_val = int(0.25*(n_best_models-n_best_models_test))
        # Split randomly best models into train, val and test depending on complexity parameter
        # print('Shuffling and distributing best models...')
        my_random.shuffle(best_models_hashes)
        best_models_hashes_train = best_models_hashes[0:n_best_models_train]
        best_models_hashes_val = best_models_hashes[n_best_models_train:n_best_models_train+n_best_models_val]
        best_models_hashes_test = best_models_hashes[n_best_models_train+n_best_models_val:]
        # Fill the lists of train, val and test hashes with remaining models
        # print('Getting hashes of non best models...')
        remaining_hashes = list(set(self.hash_keys) - set(best_models_hashes)) #[hash for hash in self.hash_keys if hash not in best_models_hashes]
        n_remaining_models = len(remaining_hashes)
        n_remaining_models_test = int(0.25*n_remaining_models)
        n_remaining_models_train = int(0.75*0.75*n_remaining_models)
        n_remaining_models_val = int(0.75*0.25*n_remaining_models)
        my_random.shuffle(remaining_hashes)
        # print('Shuffling and distributing non best models...')
        remaining_models_hashes_train = remaining_hashes[0:n_remaining_models_train]
        remaining_models_hashes_val = remaining_hashes[n_remaining_models_train:n_remaining_models_train+n_remaining_models_val]
        remaining_models_hashes_test = remaining_hashes[n_remaining_models_train+n_remaining_models_val:]
        # Merge lists of hashes
        # print('Merging lists...')
        models_hashes_train = best_models_hashes_train + remaining_models_hashes_train
        my_random.shuffle(models_hashes_train)
        models_hashes_val = best_models_hashes_val + remaining_models_hashes_val
        my_random.shuffle(models_hashes_val)
        models_hashes_test = best_models_hashes_test + remaining_models_hashes_test
        my_random.shuffle(models_hashes_test)
        # print('Overall number of models: {}'.format(self.n_models))
        # print('Number of models for train: {}'.format(len(models_hashes_train)))
        # print('Number of models for validation: {}'.format(len(models_hashes_val)))
        # print('Number of models for test: {}'.format(len(models_hashes_test)))
        # Store test hashes in pickle file
        file_path = os.path.join(self.out_datasets_path, 'test_hashes.pkl')
        pickle.dump(models_hashes_test, open(file_path, 'wb'))
        # Return hashes splits
        return models_hashes_train, models_hashes_val, models_hashes_test

    def get_dataloaders(self, complexity=1, batch_size=32):
        # Get hashes of the splits
        print('Defining splits...')
        models_hashes_train, models_hashes_val, models_hashes_test = self.define_splits(complexity=complexity)
        # Get train dataset and return train dataloader
        print('Getting train loader...')
        train_data_list = self.get_data_list_from_hashes(models_hashes_train)
        train_loader = tg.data.DataLoader(train_data_list, batch_size=batch_size)
        # Get validation dataset and return validation dataloader
        print('Getting validation loader...')
        val_data_list = self.get_data_list_from_hashes(models_hashes_val)
        val_loader = tg.data.DataLoader(val_data_list, batch_size=batch_size)
        # Get test dataset and return test dataloader
        print('Getting test loader...')
        test_data_list = self.get_data_list_from_hashes(models_hashes_test)
        test_loader = tg.data.DataLoader(test_data_list, batch_size=batch_size)
        return train_loader, val_loader, test_loader

    def store_data_in_folders(self, complexity=1):
        # Get hashes of the splits
        print('Defining splits...')
        splits = self.define_splits(complexity=complexity)
        models_hashes_train, models_hashes_val, models_hashes_test = splits
        # Store train data
        print('Storing train data to folder...')
        train_data_list = self.get_data_list_from_hashes(models_hashes_train)
        torch.save(train_data_list, os.path.join(self.out_datasets_path,
                                                 'train_data_compl_{}.pt'.format(complexity)))
        # Store validation data
        print('Storing validation data to folder...')
        val_data_list = self.get_data_list_from_hashes(models_hashes_val)
        torch.save(val_data_list, os.path.join(self.out_datasets_path,
                                                'val_data_compl_{}.pt'.format(complexity)))
        # Store test data
        print('Storing test data to folder...')
        test_data_list = self.get_data_list_from_hashes(models_hashes_test)
        torch.save(test_data_list, os.path.join(self.out_datasets_path,
                                                 'test_data_compl_{}.pt'.format(complexity)))

    def get_test_hashes(self):
        # Load test hashes in pickle file
        file_path = os.path.join(self.out_datasets_path, 'test_hashes.pkl')
        if os.path.exists(file_path):
            models_hashes_test = pickle.load(open(file_path, 'rb'))
        else:
            models_hashes_test = self.define_splits()
        return models_hashes_test