import os
import random
import time
import torch
# Import my modules
from scripts.data import NASDatasetSkeleton
from scripts.utils import timeit


class NASDataset3Splits(NASDatasetSkeleton):
    def __init__(self,
                 root='gnn2gnn_datasets/NAS101',
                 bench_folder='nas_benchmark_datasets/NAS101',
                 transform=None,
                 pre_transform=None,
                 split='train',
                 epoch=108,
                 top_n=10,
                 complexity=1):
        root = os.path.join(os.getcwd(), root)
        super(NASDataset3Splits, self).__init__(root, bench_folder, transform, pre_transform,
                                                split, epoch, top_n, complexity)
        if split == 'train':
            path = self.processed_paths[0]
        elif split == 'val':
            path = self.processed_paths[1]
        elif split == 'test':
            path = self.processed_paths[2]
        else:
            raise ValueError(('Split {} not available. Expected either '
                              'train, val or test'.format(split)))
        self.data, self.slices = torch.load(path)

    @property
    def processed_file_names(self):
        return ['train_data_top_{}_complexity_{}_epochs_{}.pt'.format(self.top_n, self.complexity, self.epoch),
                'val_data_top_{}_complexity_{}_epochs_{}.pt'.format(self.top_n, self.complexity, self.epoch),
                'test_data_top_{}_complexity_{}_epochs_{}.pt'.format(self.top_n, self.complexity, self.epoch),
                'top_{}_complexity_{}_epochs_{}.pt'.format(self.top_n, self.complexity, self.epoch), ]

    @timeit
    def process(self):
        data_list, self.top_n_models = self._preprocess()
        # Compute number of best models to add to train dataset
        top_n_models_in_train = int((1 - self.complexity) * len(self.top_n_models))
        # Define splits sizes
        train_size = int(len(data_list) * 0.75 * 0.75)
        val_size = int(len(data_list) * 0.75 * 0.25)
        # Add non best models into train data
        self.train_data_list = data_list[0:train_size]
        # Add also some top_n_models to the train dataset depending on the complexity
        self.train_data_list += self.top_n_models[0: top_n_models_in_train]
        # Add non best models to validation data
        self.val_data_list = data_list[train_size:train_size + val_size]
        # Last part of non best models is test set
        self.test_data_list = data_list[train_size + val_size:]
        # top_n_models not in the train dataset are stored in the test set
        self.test_data_list += self.top_n_models[top_n_models_in_train:]
        # Store train, test and top-n models
        datas = [self.train_data_list, self.val_data_list, self.test_data_list]
        for index, split in enumerate(datas):
            self.store_processed_data(split, self.processed_paths[index])
        # Store top-n models
        torch.save(self.top_n_models, self.processed_paths[-1])

    def get_train_models(self):
        try:
            return self.train_data_list
        except AttributeError:
            data_list, top_n_models = self._preprocess()
            # Compute number of best models to add to train dataset
            top_n_models_in_train = int((1 - self.complexity) * len(self.top_n_models))
            # Define splits sizes
            train_size = int(len(data_list) * 0.75 * 0.75)
            val_size = int(len(data_list) * 0.75 * 0.25)
            # Add non best models into train data
            self.train_data_list = data_list[0:train_size]
            # Add also some top_n_models to the train dataset depending on the complexity
            self.train_data_list += self.top_n_models[0: top_n_models_in_train]
            return self.train_data_list

    def get_validation_models(self):
        try:
            return self.val_data_list
        except AttributeError:
            data_list, self.top_n_models = self._preprocess()
            # Compute number of best models to add to train dataset
            top_n_models_in_train = int((1 - self.complexity) * len(self.top_n_models))
            # Define splits sizes
            train_size = int(len(data_list) * 0.75 * 0.75)
            val_size = int(len(data_list) * 0.75 * 0.25)
            # Add non best models into train data
            self.train_data_list = data_list[0:train_size]
            # Add also some top_n_models to the train dataset depending on the complexity
            self.train_data_list += self.top_n_models[0: top_n_models_in_train]
            # Add non best models to validation data
            self.val_data_list = data_list[train_size:train_size + val_size]
            # Last part of non best models is test set
            self.test_data_list = data_list[train_size + val_size:]
            # top_n_models not in the train dataset are stored in the test set
            self.test_data_list += self.top_n_models[top_n_models_in_train:]
            return self.val_data_list

    def get_val_and_test_models(self):
        try:
            return self.test_data_list
        except AttributeError:
            data_list, self.top_n_models = self._preprocess()
            # Compute number of best models to add to train dataset
            top_n_models_in_train = int((1 - self.complexity) * len(self.top_n_models))
            # Define splits sizes
            train_size = int(len(data_list) * 0.75 * 0.75)
            val_size = int(len(data_list) * 0.75 * 0.25)
            # Last part of non best models is test set
            self.test_data_list = data_list[train_size + val_size:]
            # top_n_models not in the train dataset are stored in the test set
            self.test_data_list += self.top_n_models[top_n_models_in_train:]
            return self.test_data_list
