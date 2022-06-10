import os
import random
import time
import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.loader import DataLoader
from scripts.benchmarks import NATSBench


class NATSDatasetSkeleton(InMemoryDataset):
    def __init__(self,
                 root='gnn2gnn_datasets/NATS',
                 bench_folder='nas_benchmark_datasets/NATS',
                 transform=None,
                 pre_transform=None,
                 split='train',
                 chosen_data='cifar10',
                 setting='topo',
                 sub_skip=False,
                 top_n=10,
                 complexity=1):
        self.name='nats'
        self.bench_folder = bench_folder
        assert chosen_data in ['cifar10', 'cifar100', 'ImageNet16-120']
        self.chosen_data = chosen_data
        assert setting in ['topo', 'size']
        self.setting = setting
        self.sub_skip = sub_skip
        self.top_n = top_n
        self.complexity = complexity
        root = os.path.join(os.getcwd(), root)
        super(NATSDatasetSkeleton, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ['data_{}_setting_{}_sub_skip_{}.pt'.format(self.chosen_data, self.setting, self.sub_skip)]

    @property
    def processed_file_names(self):
        raise ('Not implemented error!')

    def download(self):
        # Download tfrecord file if not found...
        print('Downloading dataset file, this may take a while...')
        if self.setting == 'topo':
            dataset_url = None
        elif self.setting == 'size':
            dataset_url = None
        else:
            raise ValueError('Setting {} is not available for NATS dataset...'.format(self.setting))
        # download_url(dataset_url, self.bench_folder)
        # Convert dataset files into tg samples
        print('Converting dataset files to torch geometric raw data, this may take a while...')
        converter = NATSBench(bench_folder=self.bench_folder,
                              datasets_folder=self.root,
                              chosen_data=self.chosen_data,
                              setting=self.setting,
                              sub_skip=self.sub_skip)
        converter.run()

    def _preprocess(self):
        # Import stored dictionary of data
        file_name = self.raw_file_names[0]
        try:
            tg_datas = torch.load(os.path.join(self.raw_dir, file_name))
        except FileNotFoundError:
            self.download()
            tg_datas = torch.load(os.path.join(self.raw_dir, file_name))
        # Iterate over all data and keep only models outside top-n percentage
        data_list = [data for data in tg_datas if data['top'] > self.top_n]
        self.top_n_models = [data for data in tg_datas if data['top'] <= self.top_n]
        # Shuffle data list and top n models
        my_random = random.Random(12345)
        my_random.shuffle(data_list)
        my_random.shuffle(self.top_n_models)
        return data_list, self.top_n_models

    def store_processed_data(self, data_list, name):
        data, slices = self.collate(data_list)
        torch.save((data, slices), name)

    def get_data_loader(self, batch_size=32, shuffle=False):
        loader = DataLoader(self, batch_size=batch_size, shuffle=shuffle)
        print(self)
        return loader

    def get_top_n_models(self, top_n):
        if top_n == self.top_n:
            try:
                return self.top_n_models
            except AttributeError:
                # Check if processed file containing top n models is available
                if os.path.exists(os.path.join(self.raw_dir, self.processed_file_names[-1])):
                    self.top_n_models = torch.load(os.path.join(self.raw_dir, self.processed_file_names[-1]))
                else:
                    tg_datas = torch.load(os.path.join(self.raw_dir, self.raw_file_names[0]))
                    self.top_n_models = [data for data in tg_datas if data['top'] <= self.top_n]
                return self.top_n_models
        else:
            # Import stored dictionary of data
            file_name = self.raw_file_names[0]
            tg_datas = torch.load(os.path.join(self.raw_dir, file_name))
            # Iterate over all data and keep only models outside top-n percentage
            top_n_models = [data for data in tg_datas if data['top'] <= top_n]
            return top_n_models

    def get_all_dataset_models(self):
        # Import stored dictionary of data
        file_name = self.raw_file_names[0]
        tg_datas = torch.load(os.path.join(self.raw_dir, file_name))
        # Iterate over all data and keep only models outside top-n percentage
        data_list = [data for data in tg_datas]
        return data_list

    def get_num_nodes(self):
        # Import stored dictionary of data
        file_name = self.raw_file_names[0]
        tg_datas = torch.load(os.path.join(self.raw_dir, file_name))
        # Iterate over all data and get number of nodes
        n_nodes = 0
        for data in tg_datas:
            if data.x.shape[0] > n_nodes:
                n_nodes = data.x.shape[0]
        return n_nodes