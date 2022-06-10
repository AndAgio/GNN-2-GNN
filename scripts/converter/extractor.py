# Import tensorflow dataset and extracts it

import os
import time
import pickle
from nasbench import api


class Extractor():
    def __init__(self, bench_folder='nas_benchmark_datasets/NAS101', dataset_name='nasbench_only108'):
        # Setup APIs
        pickled_api_file = '{}_api.pkl'.format(dataset_name)
        tfrecord_file = '{}.tfrecord'.format(dataset_name)
        pickled_api_file_path = os.path.join(bench_folder, pickled_api_file)
        print('Loading API...')
        if os.path.exists(pickled_api_file_path):
            print('Found pickled API!')
            bf = time.time()
            self.api = pickle.load(open(pickled_api_file_path, 'rb'))
            af = time.time()
            print('Time taken to load API from pickle: {:5f} s'.format(af - bf))
        else:
            print('Couldn\'t find the pickled API!')
            print('Loading API from tfrecord...')
            path = os.path.join(bench_folder, tfrecord_file)
            self.api = api.NASBench(path)
            print('Dumping API to pickle for future use...')
            pickle.dump(self.api, open(pickled_api_file_path, 'wb'))
        print('API extraction completed!')

    def get_api(self):
        return self.api

