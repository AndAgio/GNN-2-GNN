# Store the torch_geometric dataset and the auxiliary dictionaries
import os
import torch


class Storer():
    def __init__(self, dataset_folder='gnn2gnn_datasets/NAS101'):
        # Setup output folder
        self.out_datasets_path = os.path.join(os.getcwd(), dataset_folder)
        if not os.path.exists(self.out_datasets_path):
            os.makedirs(self.out_datasets_path)

    def store_tg_data_raw(self, raw_data, name='raw_dataset'):
        if not os.path.exists(os.path.join(self.out_datasets_path, 'raw')):
            os.makedirs(os.path.join(self.out_datasets_path, 'raw'))
        torch.save(raw_data, os.path.join(self.out_datasets_path,
                                          'raw',
                                          '{}.pt'.format(name)))
