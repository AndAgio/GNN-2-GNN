import os
import argparse
# from scripts.data import NASDatasetV2
from scripts.data import NATSDatasetSkeleton as NATSDataset
from scripts.converter import Converter

def gather_settings():
    parser = argparse.ArgumentParser(description='Settings for NN generation using GNN.')
    parser.add_argument('--dataset_folder', default='gnn2gnn_datasets',
                        help='folder where to store dataset once obtained')
    parser.add_argument('--dataset_name', default='nasbench_full',
                        help='name of file containing dataset api')
    parser.add_argument('--benchmarks_folder', default='nas_benchmark_datasets',
                        help='folder where benchmarks api data can be found')
    args = parser.parse_args()
    return args

def main(args):
    # dataset = NASDatasetV2(root=os.path.join(args.dataset_folder, 'NAS101'))

    nas_bench_folder = os.path.join(args.benchmarks_folder, 'NAS101')
    nas_out_folder = os.path.join(args.dataset_folder, 'NAS101/raw')
    converter = Converter(bench_folder=nas_bench_folder,
                         dataset_name=args.dataset_name)
    converter.run()

    # dataset = NATSDataset(root=os.path.join(args.dataset_folder, 'NATS'))



# def main(args):
#     print('Preprocessing NAS-101 benchmark to store it in the datasets folder...')
#     nas_bench_folder = os.path.join(args.benchmarks_folder, 'NAS101')
#     nas_out_folder = os.path.join(args.dataset_folder, 'NAS101/raw')
#     nasbench = NAS101Bench(bench_folder=nas_bench_folder,
#                            datasets_folder=nas_out_folder)
#     nasbench.store_data_in_folders()
#     # print('Preprocessing NATS dataset to store it in the datasets folder...')
#     # nats_bench_folder = os.path.join(args.benchmarks_folder, 'NATS')
#     # nats_out_folder = os.path.join(args.dataset_folder, 'NATS')
#     # natsbench = NATSBench(bench_folder=nats_bench_folder,
#     #                       datasets_folder=nats_out_folder)
#     # natsbench.store_data_in_folders(split='size')

if __name__=='__main__':
    args = gather_settings()
    main(args)
