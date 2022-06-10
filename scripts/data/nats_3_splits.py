import os
import random
import time
import torch
# Import my modules
from scripts.data import NATSDatasetSkeleton
from scripts.utils import timeit


class NATSDataset3Splits(NATSDatasetSkeleton):
    def __init__(self,
                 root='gnn2gnn_datasets/NATS',
                 bench_folder='nas_benchmark_datasets/NATS',
                 transform=None,
                 pre_transform=None,
                 split='train',
                 chosen_data='cifar10',
                 setting=108,
                 sub_skip=False,
                 top_n=10,
                 complexity=1):
        root = os.path.join(os.getcwd(), root)
        super(NATSDataset3Splits, self).__init__(root=root,
                                                 bench_folder=bench_folder,
                                                 transform=transform,
                                                 pre_transform=pre_transform,
                                                 split=split,
                                                 chosen_data=chosen_data,
                                                 setting=setting,
                                                 sub_skip=sub_skip,
                                                 top_n=top_n,
                                                 complexity=complexity)
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
        return ['train_data_{}_setting_{}_sub_skip_{}_top_{}_complexity_{}.pt'.format(self.chosen_data,
                                                                                      self.setting,
                                                                                      self.sub_skip,
                                                                                      self.top_n,
                                                                                      self.complexity),
                'val_data_{}_setting_{}_sub_skip_{}_top_{}_complexity_{}.pt'.format(self.chosen_data,
                                                                                    self.setting,
                                                                                    self.sub_skip,
                                                                                    self.top_n,
                                                                                    self.complexity),
                'test_data_{}_setting_{}_sub_skip_{}_top_{}_complexity_{}.pt'.format(self.chosen_data,
                                                                                     self.setting,
                                                                                     self.sub_skip,
                                                                                     self.top_n,
                                                                                     self.complexity),
                'data_{}_setting_{}_sub_skip_{}_top_{}_complexity_{}.pt'.format(self.chosen_data,
                                                                                self.setting,
                                                                                self.sub_skip,
                                                                                self.top_n,
                                                                                self.complexity), ]

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

    def get_test_models(self):
        # Import stored dictionary of data
        file_name = self.raw_file_names[0]
        tg_datas = torch.load(os.path.join(self.raw_dir, file_name))
        # Iterate over all data and keep only models outside top-n percentage
        data_list = [data for data in tg_datas if data['top'] > self.top_n]
        # Shuffle data list and top n models
        my_random = random.Random(12345)
        my_random.shuffle(data_list)
        # Define splits sizes
        train_size = int(len(data_list) * 0.75 * 0.75)
        val_size = int(len(data_list) * 0.75 * 0.25)
        # Last part of non best models is test set
        test_data_list = data_list[train_size + val_size:]
        return test_data_list
