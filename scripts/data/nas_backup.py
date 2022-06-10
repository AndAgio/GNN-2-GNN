import os
import random
import time
import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.loader import DataLoader


class NASDataset(InMemoryDataset):
    def __init__(self,
                 root='gnn2gnn_datasets/NAS101',
                 bench_folder='nas_benchmark_datasets/NAS101',
                 transform=None,
                 pre_transform=None,
                 split='train',
                 epoch=108,
                 top_n=10,
                 complexity=1):
        self.bench_folder = bench_folder
        self.epoch = epoch
        self.top_n = top_n
        self.complexity = complexity
        root = os.path.join(os.getcwd(), root)
        super(NASDataset, self).__init__(root, transform, pre_transform)
        if split == 'train':
            path = self.processed_paths[0]
        elif split == 'test':
            path = self.processed_paths[1]
        # elif split == 'top-n':
        #     path = self.processed_paths[-1]
        else:
            raise ValueError(('Split {} not available. Expected either '
                              'train, test or top-n'.format(split)))
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['data_epochs_{}.pt'.format(self.epoch)]

    @property
    def processed_file_names(self):
        return ['train_data_top_{}_epochs_{}.pt'.format(self.top_n, self.epoch),
                'test_data_top_{}_epochs_{}.pt'.format(self.top_n, self.epoch),
                'top_{}_epochs_{}.pt'.format(self.top_n, self.epoch), ]

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

    def process(self):
        print('Download done, handling processing...')
        # Import stored dictionary of data
        file_name = self.raw_file_names[0]
        tg_datas = torch.load(os.path.join(self.raw_dir, file_name))
        # Iterate over all data and keep only models outside top-n percentage
        data_list = [data for data in tg_datas if data['top'] > self.top_n]
        self.top_n_models = [data for data in tg_datas if data['top'] <= self.top_n]
        # Shuffle data list and top n models
        my_random = random.Random(12345)
        my_random.shuffle(data_list)
        my_random.shuffle(self.top_n_models)
        # Compute number of best models to add to train dataset
        top_n_models_in_train = int((1 - self.complexity) * len(self.top_n_models))
        # Add 75% of non best models into train data
        train_data_list = data_list[0:int(0.75 * len(data_list))]
        # Add also some top_n_models to the train dataset depending on the complexity
        train_data_list += self.top_n_models[0: top_n_models_in_train]
        # Remaining 25% of non best models is test set
        test_data_list = data_list[int(0.75 * len(data_list)):]
        # Store train, test and top-n models
        datas = [train_data_list, test_data_list]
        for index, split in enumerate(datas):
            self.store_processed_data(split, self.processed_paths[index])
        # Store top-n models
        torch.save(self.top_n_models, self.processed_paths[-1])

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
            print('IMPORTING TOP MODELS WITH TOP-K={}'.format(top_n))
            tg_datas = torch.load(os.path.join(self.raw_dir, file_name))
            # Iterate over all data and keep only models outside top-n percentage
            top_n_models = [data for data in tg_datas if data['top'] <= top_n]
            return top_n_models

    def get_test_models(self):
        # Import stored dictionary of data
        file_name = self.raw_file_names[0]
        tg_datas = torch.load(os.path.join(self.raw_dir, file_name))
        # Iterate over all data and keep only models outside top-n percentage
        data_list = [data for data in tg_datas if data['top'] > self.top_n]
        # Shuffle data list and top n models
        my_random = random.Random(12345)
        my_random.shuffle(data_list)
        # Last 25% of non best models is test set
        test_data_list = data_list[int(0.75 * len(data_list)):]
        return test_data_list

    def get_all_valid_models(self):
        # Import stored dictionary of data
        file_name = self.raw_file_names[0]
        tg_datas = torch.load(os.path.join(self.raw_dir, file_name))
        # Iterate over all data and keep only models outside top-n percentage
        data_list = [data for data in tg_datas]
        return data_list
