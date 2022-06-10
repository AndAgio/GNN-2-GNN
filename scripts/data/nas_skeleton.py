import os
import random
import time
import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.loader import DataLoader


class NASDatasetSkeleton(InMemoryDataset):
    def __init__(self,
                 root='gnn2gnn_datasets/NAS101',
                 bench_folder='nas_benchmark_datasets/NAS101',
                 transform=None,
                 pre_transform=None,
                 split='train',
                 epoch=108,
                 top_n=10,
                 complexity=1):
        self.name = 'nas101'
        self.bench_folder = bench_folder
        self.epoch = epoch
        self.top_n = top_n
        self.complexity = complexity
        root = os.path.join(os.getcwd(), root)
        super(NASDatasetSkeleton, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ['data_epochs_{}.pt'.format(self.epoch)]

    @property
    def processed_file_names(self):
        raise ('Not implemented error!')

    def download(self):
        # Download tfrecord file if not found...
        print('Downloading tfrecord dataset file, this will take a while...')
        full_tf_record_url = 'https://storage.googleapis.com/nasbench/nasbench_full.tfrecord'
        download_url(full_tf_record_url, self.bench_folder)
        # Convert tfrecord file into tg samples
        from scripts.converter import Converter
        converter = Converter(bench_folder=self.bench_folder,
                              dataset_name='nasbench_full',
                              tg_dataset_folder=self.root)
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
                if os.path.exists(os.path.join(self.processed_dir, self.processed_file_names[-1])):
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
