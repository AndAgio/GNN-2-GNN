import os
import sys
import random
import pickle
import torch
import numpy as np
import networkx as nx
import torch.nn.functional as t_func
import torch_geometric as tg
import matplotlib.pyplot as plt
from progress.bar import IncrementalBar
from nats_bench import create
# Import my modules
from scripts.utils import Utils


class NATSBench():
    def __init__(self, bench_folder='nas_benchmark_datasets/NATS',
                 datasets_folder='gnn2gnn_datasets/NATS', chosen_data='cifar10',
                 setting='topo', sub_skip=False):
        self.bench_folder = bench_folder
        self.chosen_data = chosen_data
        assert setting in ['topo', 'size']
        self.setting = setting
        self.sub_skip = sub_skip
        # Setup output folder
        self.out_datasets_path = os.path.join(os.getcwd(), datasets_folder)
        if not os.path.exists(self.out_datasets_path):
            os.makedirs(self.out_datasets_path)
        # Setup path for files
        path = os.path.join(os.getcwd(), self.bench_folder, self.setting)
        if self.setting == 'topo':
            # Setup API
            self.api = create(path, 'tss', fast_mode=True, verbose=False)
            # Setup number of models to remember them
            self.n_models = 15625
        else:
            # Setup API
            self.api = create(path, 'sss', fast_mode=True, verbose=False)
            # Setup number of models to remember them
            self.n_models = 32768
        # Setup lists of hashes
        self.hash_list = [i for i in range(self.n_models)]
        # Setup iterators over size and topo indices
        self.hash_iterator = iter(self.hash_list)
        # Useful constants
        self.input = 'input'
        self.output = 'output'
        self.conv3 = 'nor_conv_3x3'
        self.conv1 = 'nor_conv_1x1'
        self.avgpool3 = 'avg_pool_3x3'
        self.skipcon = 'skip_connect'
        if self.sub_skip:
            self.nodes_emb_dict = {self.input: 0, self.conv1: 1,
                                   self.conv3: 2, self.avgpool3: 3,
                                   self.output: 4}
            self.nodes_emb_dict_inv = {0: self.input, 1: self.conv1,
                                       2: self.conv3, 3: self.avgpool3,
                                       4: self.output}
        else:
            self.nodes_emb_dict = {self.input: 0, self.conv1: 1,
                                   self.conv3: 2, self.avgpool3: 3,
                                   self.skipcon: 4, self.output: 5}
            self.nodes_emb_dict_inv = {0: self.input, 1: self.conv1,
                                       2: self.conv3, 3: self.avgpool3,
                                       4: self.skipcon, 5: self.output}
        # Setup output folder
        out_datasets_path = os.path.join(os.getcwd(), datasets_folder)
        if not os.path.exists(out_datasets_path):
            os.makedirs(out_datasets_path)
        self.out_datasets_path = out_datasets_path

    def get_random_model(self):
        rand_hash = random.randint(0, self.n_models)
        info = self.query(rand_hash)
        print('Info: {}'.format(info))
        return info

    def get_next_model(self):
        info = self.query(self.get_next_hash())
        print('Info: {}'.format(info))
        return info

    def get_next_hash(self):
        next_hash = next(self.hash_iterator)
        # print('Model hash is: {}'.format(next_hash))
        return next_hash

    def query(self, hash):
        config = self.api.get_net_config(hash, self.chosen_data)
        perf_info = self.api.get_more_info(hash, self.chosen_data)
        cost_info = self.api.get_cost_info(hash, self.chosen_data)
        # print('Config: {}'.format(config))
        # print('Perf info: {}'.format(perf_info))
        # print('Cost info: {}'.format(cost_info))
        if self.setting == 'size':
            # Combine perf info and net configurations into info
            info = {'arch': config['genotype'],
                    'filters': config['channels'],
                    'train_time': perf_info['train-all-time'],
                    'train_accuracy': perf_info['train-accuracy'],
                    'test_accuracy': perf_info['test-accuracy'],
                    'params': cost_info['params'],
                    'flops': cost_info['flops'], }
        elif self.setting == 'topo':
            # Combine perf info and net configurations into info
            info = {'arch': config['arch_str'],
                    'filters': config['C'],
                    'train_time': perf_info['train-all-time'],
                    'train_accuracy': perf_info['train-accuracy'],
                    'test_accuracy': perf_info['test-accuracy'],
                    'params': cost_info['params'],
                    'flops': cost_info['flops'], }
        else:
            raise ValueError('The setting {} is not in the NATSBench dataset'.format(self.setting))
        # print('Config: {}'.format(config))
        # print('Perf info: {}'.format(perf_info))
        return info

    def check_graph_consistency(self, adj_mat):
        shape = adj_mat.shape
        if shape[0] != shape[1]:
            raise ValueError('Something went wrong with adjacency matrix!')
        rows_sums = np.sum(adj_mat, axis=0)
        cols_sums = np.sum(adj_mat, axis=1).T
        # print('Rows sums: {}'.format(rows_sums))
        # print('Cols sums: {}'.format(cols_sums))
        for index in range(1, shape[0] - 1):
            sum = rows_sums[0, index] + cols_sums[0, index]
            if sum < 2:
                # print('Index to remove is {}'.format(index))
                return False, index
        return True, None

    def substitute_skip_connections(self, nodes_ops, adj_mat):
        # Identify nodes corresponding to skip connections
        skip_indices = []
        for ind, operation in enumerate(nodes_ops):
            if operation == self.skipcon:
                skip_indices.append(ind)
        # print('skip_indices: {}'.format(skip_indices))
        # For each node corresponding to a skip connection...
        while len(skip_indices) > 0:
            node_index = skip_indices[-1]
            # print('node_index: {}'.format(node_index))
            # Identify starting and ending point of the skip connection
            starting_node = list(np.where(adj_mat[:, node_index] == 1)[0])
            ending_node = list(np.where(adj_mat[node_index, :] == 1)[-1])
            # print('starting_node: {}'.format(starting_node))
            # print('ending_node: {}'.format(ending_node))
            # Create link between starting and ending nodes
            adj_mat[starting_node, ending_node] = 1
            # Remove node corresponding to skip connections from adjacency matrix and list of nodes
            adj_mat = np.delete(adj_mat, node_index, axis=0)
            adj_mat = np.delete(adj_mat, node_index, axis=1)
            nodes_ops.pop(node_index)
            # Remove treated node from list of indices to remove
            skip_indices = skip_indices[:-1]
        # Return refined nodes and adjacency matrix
        return nodes_ops, adj_mat

    def get_nodes_adj_from_nats_arch(self, arch):
        nodes_ops_dict = {i: None for i in range(8)}
        nodes_ops_dict[0] = self.input
        adj_mat = np.matrix([[0, 1, 1, 0, 1, 0, 0, 0],
                             [0, 0, 0, 1, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 0, 0, 1],
                             [0, 0, 0, 0, 0, 0, 0, 1],
                             [0, 0, 0, 0, 0, 0, 0, 1],
                             [0, 0, 0, 0, 0, 0, 0, 0]])
        # Split arch into levels
        # print('arch is: {}'.format(arch))
        levels = arch.split('+')
        # For each level get the operations in it
        op_index = 1
        for index, level in enumerate(levels):
            # operations_dict[index] = {}
            ops = level.split('|')
            ops = [op for op in ops if op != '']
            for sub_index, operation in enumerate(ops):
                refined_operation = operation.split('~')[0]
                # operations_dict[index][sub_index] = refined_operation
                nodes_ops_dict[op_index] = refined_operation
                op_index += 1
        # Refine adjacency matrix
        nodes_ops_dict[7] = self.output
        indices_to_remove = []
        for ind, operation in nodes_ops_dict.items():
            if operation == 'none':
                indices_to_remove.append(ind)
        indices_to_remove = indices_to_remove[::-1]
        for ind in indices_to_remove:
            adj_mat = np.delete(adj_mat, ind, axis=0)
            adj_mat = np.delete(adj_mat, ind, axis=1)
        nodes_ops = [op for op in nodes_ops_dict.values() if op != 'none']
        # Check if there exist node not linked with input/output
        consistency, index_to_remove = self.check_graph_consistency(adj_mat)
        # print('Nodes operations: {}'.format(nodes_ops))
        # print('Adjacency matrix:\n{}'.format(adj_mat))
        # print('Consistency: {}'.format(consistency))
        while not consistency:
            adj_mat = np.delete(adj_mat, index_to_remove, axis=0)
            adj_mat = np.delete(adj_mat, index_to_remove, axis=1)
            del nodes_ops[index_to_remove]
            consistency, index_to_remove = self.check_graph_consistency(adj_mat)
        # print('Nodes operations: {}'.format(nodes_ops))
        # print('Adjacency matrix:\n{}'.format(adj_mat))
        # print('Consistency: {}'.format(consistency))
        # Remove nodes containing skip-connections and substitute them with direct link
        # print('nodes_ops: {}'.format(nodes_ops))
        # print('adj_mat: {}'.format(adj_mat))
        # print('self.sub_skip: {}'.format(self.sub_skip))
        if self.sub_skip:
            nodes_ops, adj_mat = self.substitute_skip_connections(nodes_ops, adj_mat)
        # print('nodes_ops: {}'.format(nodes_ops))
        # print('adj_mat: {}'.format(adj_mat))
        return nodes_ops, adj_mat

    def info_to_tg_data(self, info, one_hot=True, verbose=False, plot=True):
        # Get relevant information from the info dictionary
        arch = info['arch']
        acc = info['test_accuracy']
        # print('info are: {}'.format(info))
        nodes_ops, adj_mat = self.get_nodes_adj_from_nats_arch(arch)  # edges_list
        # Convert nodes operations into their encoding and into torch tensor
        node_ops_embedding = [[self.nodes_emb_dict[node_op]] for node_op in nodes_ops]
        x = torch.tensor(node_ops_embedding, dtype=torch.float)
        if one_hot:
            n_ops = len(list(self.nodes_emb_dict.keys()))
            x = t_func.one_hot(x.squeeze().to(torch.int64), num_classes=n_ops).float()
        # # Get indices of nodes having links
        links_indices = np.where(adj_mat == 1)
        links_indices = list(zip(links_indices[0], links_indices[1]))
        # Convert links into COO matrix for torch geometric
        edge_index = torch.tensor(links_indices, dtype=torch.long)
        # Get data format
        data = tg.data.Data(x=x,
                            edge_index=edge_index.t().contiguous(),
                            training_time=info['train_time'],
                            train_accuracy=info['train_accuracy'],
                            test_accuracy=info['test_accuracy'],
                            y_class=1 if info['test_accuracy'] > 0.5 else 0,
                            top=info['top'],
                            footprint=info['params'],
                            params=info['params'],
                            flops=info['flops'])
        # Print everything if verbose is True
        if verbose:
            print('Operations embeddings: {}'.format(node_ops_embedding))
            print('Links indices: {}'.format(links_indices))
            print('Edge index: {}'.format(edge_index))
            print('TG data is: {}'.format(data))
        # Plot example if plot is required
        if plot:
            nx_graph = tg.utils.convert.to_networkx(data)
            nodes_labels = {node: nodes_ops[node] for node in nx_graph.nodes}
            color_map = Utils().get_color_map(bench='nats', sub_skip=self.sub_skip)
            nodes_colors_dict = {node: color_map[nodes_ops[node]] for node in nx_graph.nodes}
            nodes_colors = list(nodes_colors_dict.values())
            plt.figure()
            nx.draw(nx_graph, labels=nodes_labels, node_color=nodes_colors, with_labels=True, node_size=450,
                    edge_color='gray', arrowsize=35, arrowstyle='simple')
            plt.show()
        return data

    def get_data_list_from_hashes(self, hashes_list):
        # Define empty list of data to be returned
        datas = []
        for hash in hashes_list:
            info = self.query(hash)
            data = self.info_to_tg_data(info, verbose=False, plot=False)
            # print('Data: {}'.format(data))
            datas.append(data)
        return datas

    def get_best_models(self, perc=0.1):
        hash_perf_dict = {}
        n_best_models = int(perc * self.n_models)
        # print('Looking for the best {} models'.format(n_best_models))
        bar = IncrementalBar('Getting the best {} models...'.format(n_best_models), max=self.n_models)
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
        return best_models_dict

    def define_splits(self, complexity=1):
        my_random = random.Random(12345)
        # Get best models from the dataset
        best_models_dict = self.get_best_models()
        best_models_hashes = list(best_models_dict.keys())
        n_best_models = len(best_models_hashes)
        # Get number of best models that will end up in the test set depending on the complexity parameter
        n_best_models_test = int(complexity * n_best_models)
        n_best_models_train = int(0.75 * (n_best_models - n_best_models_test))
        n_best_models_val = int(0.25 * (n_best_models - n_best_models_test))
        # Split randomly best models into train, val and test depending on complexity parameter
        # print('Shuffling and distributing best models...')
        my_random.shuffle(best_models_hashes)
        best_models_hashes_train = best_models_hashes[0:n_best_models_train]
        best_models_hashes_val = best_models_hashes[n_best_models_train:n_best_models_train + n_best_models_val]
        best_models_hashes_test = best_models_hashes[n_best_models_train + n_best_models_val:]
        # Fill the lists of train, val and test hashes with remaining models
        # print('Getting hashes of non best models...')
        remaining_hashes = list(set(self.hash_list) - set(best_models_hashes))
        n_remaining_models = len(remaining_hashes)
        n_remaining_models_test = int(0.25 * n_remaining_models)
        n_remaining_models_train = int(0.75 * 0.75 * n_remaining_models)
        n_remaining_models_val = int(0.75 * 0.25 * n_remaining_models)
        my_random.shuffle(remaining_hashes)
        # print('Shuffling and distributing non best models...')
        remaining_models_hashes_train = remaining_hashes[0:n_remaining_models_train]
        remaining_models_hashes_val = remaining_hashes[
                                      n_remaining_models_train:n_remaining_models_train + n_remaining_models_val]
        remaining_models_hashes_test = remaining_hashes[n_remaining_models_train + n_remaining_models_val:]
        # Merge lists of hashes
        # print('Merging lists...')
        models_hashes_train = best_models_hashes_train + remaining_models_hashes_train
        my_random.shuffle(models_hashes_train)
        models_hashes_val = best_models_hashes_val + remaining_models_hashes_val
        my_random.shuffle(models_hashes_val)
        models_hashes_test = best_models_hashes_test + remaining_models_hashes_test
        my_random.shuffle(models_hashes_test)
        print('Overall number of models: {}'.format(self.n_models))
        print('Number of models for train: {}'.format(len(models_hashes_train)))
        print('Number of models for validation: {}'.format(len(models_hashes_val)))
        print('Number of models for test: {}'.format(len(models_hashes_test)))
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
                                                 'train_data_{}_setting_{}_compl_{}.pt'.format(self.chosen_data,
                                                                                               self.setting,
                                                                                               complexity)))
        # Store validation data
        print('Storing validation data to folder...')
        val_data_list = self.get_data_list_from_hashes(models_hashes_val)
        torch.save(val_data_list, os.path.join(self.out_datasets_path,
                                               'val_data_{}_setting_{}_compl_{}.pt'.format(self.chosen_data,
                                                                                           self.setting, complexity)))
        # Store test data
        print('Storing test data to folder...')
        test_data_list = self.get_data_list_from_hashes(models_hashes_test)
        torch.save(test_data_list, os.path.join(self.out_datasets_path,
                                                'test_data_{}_setting_{}_compl_{}.pt'.format(self.chosen_data,
                                                                                             self.setting, complexity)))

    def run(self):
        # Query models and store dictionary containing all information in the tfrecord file
        print('Converting online format to list of dictionaries...')
        datas_dict_form = self.convert_models_info_to_dictionary()
        # Convert models in dictionary to torch-geometric file and store it
        print('Converting list of dictionaries into torch geometric raw data...')
        self.convert_dictionary_to_tg_raw_and_store_it(datas_dict_form)

    def convert_models_info_to_dictionary(self):
        # Check if pickled ordered dictionary exists
        pickled_ordered_dictionary_name = 'nats_data_{}_setting_{}_sub_skip_{}_ordered_dict.pkl'.format(self.chosen_data,
                                                                                                        self.setting,
                                                                                                        self.sub_skip)
        pickled_ordered_dictionary_path = os.path.join(self.bench_folder, pickled_ordered_dictionary_name)
        if os.path.exists(pickled_ordered_dictionary_path):
            print('Loading pickled ordered dictionary...')
            datas = pickle.load(open(pickled_ordered_dictionary_path, 'rb'))
            print('Loaded!')
        else:
            # Check if pickled unordered dictionary exists
            pickled_unordered_dictionary_name = 'nats_data_{}_setting_{}_sub_skip_{}_unordered_dict.pkl'.format(
                self.chosen_data,
                self.setting,
                self.sub_skip)
            pickled_unordered_dictionary_path = os.path.join(self.bench_folder, pickled_unordered_dictionary_name)
            if os.path.exists(pickled_unordered_dictionary_path):
                print('Loading pickled unordered dictionary...')
                datas = pickle.load(open(pickled_unordered_dictionary_path, 'rb'))
                print('Loaded!')
            else:
                print('Couldn\'t find unordered dictionary!')
                # Iterate over all hashes available in the api...
                datas = []
                bar = IncrementalBar('Querying all hash keys...', max=self.n_models)
                for hash_key in self.hash_iterator:
                    # Convert sample into dictionary and append it to data list
                    datas.append(self.query(hash_key))
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
        datas = [dict(item, **{'top': 100 - (index / self.n_models * 100)}) for index, item in enumerate(datas)]
        return datas

    def convert_dictionary_to_tg_raw_and_store_it(self, datas):
        # Iterate over all data in the dictionary
        tg_datas = []
        my_hashes_list = []
        bar = IncrementalBar('Converting all data in dictionary form into tg.data form...', max=len(datas))
        for index, data in enumerate(datas):
            # Transform data in tg_data
            if index >= len(datas) - 50:
                plot = True
            else:
                plot = False
            tg_data = self.info_to_tg_data(data, plot=False)
            if not tg_data.edge_index.nelement() == 0:
                model_hash = hash((tg_data['x'].numpy().tobytes(), tg_data['edge_index'].numpy().tobytes()))
                if model_hash not in my_hashes_list:
                    # Append it
                    # print('Appending graph to torch geometric dataset...')
                    tg_datas.append(tg_data)
                    my_hashes_list.append(model_hash)
            bar.next()
        bar.finish()
        # Recompute top-n value, since duplicates models have been removed
        max = len(tg_datas)
        bar = IncrementalBar('Recomputing top-n values, since duplicates have been removed...', max=max)
        for index in range(max):
            tg_datas[index]['top'] = 100 - (index / max * 100)
            if index > max:  # max-50:
                print('TG DATA: x: {}\ntrain-acc={} test-acc={}\ntop-n={}'.format(tg_datas[index]['x'],
                                                                                  tg_datas[index]['train_accuracy'],
                                                                                  tg_datas[index]['train_accuracy'],
                                                                                  tg_datas[index]['top']))
                self.plot_tg_data(tg_datas[index])
            bar.next()
        bar.finish()
        # Store it as raw files
        if not os.path.exists(os.path.join(self.out_datasets_path, 'raw')):
            os.makedirs(os.path.join(self.out_datasets_path, 'raw'))
        torch.save(tg_datas, os.path.join(self.out_datasets_path,
                                          'raw',
                                          'data_{}_setting_{}_sub_skip_{}.pt'.format(self.chosen_data, self.setting,
                                                                                     self.sub_skip)))

    def plot_tg_data(self, data):
        # Plot example if plot is required
        nx_graph = tg.utils.convert.to_networkx(data)
        x = data['x']
        nodes_operations = torch.argmax(x, dim=-1).numpy()
        nodes_ops = [self.nodes_emb_dict_inv[node_op] for node_op in nodes_operations]
        nodes_labels = {node: nodes_ops[node] for node in nx_graph.nodes}
        color_map = Utils().get_color_map(bench='nats', sub_skip=self.sub_skip)
        nodes_colors_dict = {node: color_map[nodes_ops[node]] for node in nx_graph.nodes}
        nodes_colors = list(nodes_colors_dict.values())
        plt.figure()
        nx.draw(nx_graph, labels=nodes_labels, node_color=nodes_colors, with_labels=True, node_size=450,
                edge_color='gray', arrowsize=35, arrowstyle='simple')
        plt.show()
