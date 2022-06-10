import os
import random
import time
import torch
# Import my modules
from scripts.data import NATSDatasetSkeleton
from scripts.utils import timeit


class NATSDataset2Splits(NATSDatasetSkeleton):
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
        root = os.path.join(os.getcwd(), root)
        super(NATSDataset2Splits, self).__init__(root=root,
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
        elif split == 'test':
            path = self.processed_paths[1]
        else:
            raise ValueError(('Split {} not available. Expected either '
                              'train, test or top-n'.format(split)))
        self.data, self.slices = torch.load(path)

    @property
    def processed_file_names(self):
        return ['train_data_{}_setting_{}_sub_skip_{}_top_{}_complexity_{}.pt'.format(self.chosen_data,
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
        # Add 75% of non best models into train data
        self.train_data_list = data_list[0:int(0.75 * len(data_list))]
        # Add also some top_n_models to the train dataset depending on the complexity
        self.train_data_list += self.top_n_models[0: top_n_models_in_train]
        # Remaining 25% of non best models is test set
        self.test_data_list = data_list[int(0.75 * len(data_list)):]
        # Remaining of top_n models are in test set
        self.test_data_list += self.top_n_models[top_n_models_in_train:]
        # Store train, test and top-n models
        datas = [self.train_data_list, self.test_data_list]
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
            # Add 75% of non best models into train data
            self.train_data_list = data_list[0:int(0.75 * len(data_list))]
            # Add also some top_n_models to the train dataset depending on the complexity
            self.train_data_list += self.top_n_models[0: top_n_models_in_train]
            return self.train_data_list

    def get_test_models(self):
        try:
            return self.test_data_list
        except AttributeError:
            data_list, top_n_models = self._preprocess()
            # Compute number of best models to add to train dataset
            top_n_models_in_train = int((1 - self.complexity) * len(top_n_models))
            # Remaining 25% of non best models is test set
            self.test_data_list = data_list[int(0.75 * len(data_list)):]
            # Remaining of top_n models are in test set
            self.test_data_list += self.top_n_models[top_n_models_in_train:]
            return self.test_data_list
